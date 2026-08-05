//! The Kimi-K3 **latent MoE block** — `KimiSparseMoeBlock` in
//! `modeling_kimi_linear.py`, layers 1..92.
//!
//! ```text
//!                       hidden [B, T, 7168]
//!                         |            \
//!            routed_expert_down_proj    \  (identity)
//!                         |              \
//!                  latent [T, 3584]       \
//!                         |                \
//!         moe_infer: Σ_k w_k · expert_{i_k} \
//!                         |                  \
//!                routed_expert_norm           shared_experts (KimiMLP,
//!                         |                    2 experts fused: 2·3072 = 6144)
//!             routed_expert_up_proj             |
//!                         |                    /
//!                    [B, T, 7168]  <---- + ---/
//! ```
//!
//! Five things about this block are not obvious from the shape diagram, and
//! each of them is a place a port goes wrong silently. All five were settled
//! against the whole-layer oracle (`k3_moe_gate`), not read off the source:
//!
//! 1. **The router's combining weight comes from the UNBIASED sigmoid score.**
//!    `e_score_correction_bias` is added to `scores` to make
//!    `scores_for_choice`, which decides *which* experts fire; the weight is
//!    then gathered from `scores`, not from `scores_for_choice`. Using the
//!    biased score is a plausible mis-port that changes every weight.
//! 2. **The bias is added to the sigmoid SCORE, not to the logit.** Adding it
//!    pre-sigmoid selects a different expert set — on the oracle's 32 tokens
//!    it changes the set for *all 32*.
//! 3. **`routed_expert_norm` runs AFTER the top-16 combination** and before
//!    the up-projection. Its input is the weighted sum of expert outputs, which
//!    has a per-token RMS spread of 0.107..3.81 — it is emphatically not
//!    already normalised.
//! 4. **The shared experts are added in the ORIGINAL hidden space**, from the
//!    block's own input (not from the latent), after the up-projection.
//! 5. **`w1` is the gate half and `w3` is the up half** of `situ`'s
//!    concatenated input, in that order.
//!
//! # Arithmetic, and why [`ActRound`] exists
//!
//! The shipped module pins several reductions to fp32 whatever the parameter
//! dtype is (`KimiRMSNorm` upcasts, `SituAndMul` upcasts, `KimiMoEGate` runs
//! the whole router in fp32, and `moe_infer`'s top-k combination inherits fp32
//! from `topk_weight`). Everything else runs in the model dtype, which for
//! this checkpoint is **bfloat16**: every `nn.Linear` output, every activation
//! output and the final residual add are bf16 values.
//!
//! This port computes in f32 throughout — that is the shipped fp32-island
//! arithmetic exactly — and reproduces the bf16 storage lane by *rounding at
//! the points the reference rounds*, which is what [`ActRound::Bf16`] does.
//! With that rounding in place the port is **bit-identical** to the shipped
//! bf16 run for the projections, the activation and the norms, and differs
//! only where fp32 accumulation order flips a single bf16 rounding decision.
//! Without it (`ActRound::None`) the port reproduces the shipped **f32** run.
//!
//! [`ActRound::Bf16`] round-trips through the host. That is right for a gate
//! and wrong for a production bf16 lane, which would hold bf16 tensors and let
//! the backend round natively; the rounding *points* established here are what
//! such a lane would need, and they are the part that is hard to get right.
//!
//! # What this module does NOT do
//!
//! * Top-k selection runs on the host. Correct, deterministic, and the wrong
//!   shape for a GPU decode loop — a device top-k is a separate change.
//! * Expert weights are supplied one at a time by a caller-provided closure,
//!   so the block never assumes all 896 experts are resident. It calls the
//!   closure exactly once per expert the router actually selected, in
//!   ascending expert id — mirroring `moe_infer`'s loop, which `continue`s
//!   over experts with no tokens.
//! * The grouped top-k path (`num_expert_group > topk_group`) is **refused**,
//!   not implemented: this checkpoint has `num_expert_group = 1`, so there is
//!   nothing here to check an implementation of it against. See
//!   [`MoeDims::from_text_config`].

use burn::prelude::*;
use burn::tensor::IndexingUpdateOp;

use super::situ::Situ;

/// Where the block rounds intermediate activations. Re-exported from
/// [`super::ops`], which is where the layer-wide rounding policy now lives —
/// the MLA block, the KDA block and the two per-layer norms all need the same
/// one, and four copies of it is how a rounding fix goes missing from three of
/// them.
pub use super::ops::ActRound;

/// The subset of `text_config` the MoE block reads, with every field this port
/// does *not* model turned into a refusal at construction time.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeDims {
    /// Residual-stream width (7168).
    pub hidden_size: usize,
    /// Width the routed experts operate in — `routed_expert_hidden_size`
    /// (3584). The latent bottleneck.
    pub moe_hidden_size: usize,
    /// FFN width of ONE routed expert (3072).
    pub moe_intermediate_size: usize,
    /// Combined FFN width of the fused shared MLP
    /// (`moe_intermediate_size · num_shared_experts` = 6144), or `None` when
    /// the block has no shared experts.
    pub shared_intermediate_size: Option<usize>,
    /// Number of routed experts (896).
    pub num_experts: usize,
    /// Routed experts per token (16).
    pub top_k: usize,
    /// Whether the top-k weights are renormalised to sum to one.
    pub moe_renormalize: bool,
    /// Multiplier applied to the top-k weights after renormalisation. **1.0 in
    /// this checkpoint**, which means no captured vector can distinguish a port
    /// that applies it from one that ignores it — see `k3_moe_gate`'s
    /// `I3` check, which exercises it synthetically instead.
    pub routed_scaling_factor: f64,
    /// Whether the latent carries an RMSNorm between the combination and the
    /// up-projection.
    pub latent_moe_use_norm: bool,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// The `situ` activation, already parameterised.
    pub situ: Situ,
}

impl MoeDims {
    /// The Kimi-K3 settings, as literals. Kept next to
    /// [`Self::from_text_config`] so a config-driven build and a hard-coded one
    /// can be compared against each other (the gate does exactly that).
    pub fn k3() -> Self {
        Self {
            hidden_size: 7168,
            moe_hidden_size: 3584,
            moe_intermediate_size: 3072,
            shared_intermediate_size: Some(6144),
            num_experts: 896,
            top_k: 16,
            moe_renormalize: true,
            routed_scaling_factor: 1.0,
            latent_moe_use_norm: true,
            rms_norm_eps: 1e-5,
            situ: Situ::k3(),
        }
    }
}

#[cfg(feature = "k3")]
impl MoeDims {
    /// Read the block's dimensions out of a parsed `config.json`.
    ///
    /// Every `Err` here is a shape of MoE block this module does not implement.
    /// Refusing is the point: a config with a softmax router or a grouped top-k
    /// would otherwise run to completion and produce a plausible wrong answer,
    /// and there is nothing in this checkpoint to check such a path against.
    pub fn from_text_config(cfg: &crate::models::k3::K3TextConfig) -> Result<Self, String> {
        if cfg.hidden_act != "situ" {
            return Err(format!(
                "hidden_act is {:?}; this port implements only `situ`",
                cfg.hidden_act
            ));
        }
        if cfg.moe_router_activation_func != "sigmoid" {
            return Err(format!(
                "moe_router_activation_func is {:?}; this port implements only `sigmoid`",
                cfg.moe_router_activation_func
            ));
        }
        if cfg.topk_method != "noaux_tc" {
            return Err(format!(
                "topk_method is {:?}; this port implements only `noaux_tc`",
                cfg.topk_method
            ));
        }
        // The shipped gate takes the grouped branch exactly when this holds.
        if cfg.num_expert_group > 1 && cfg.num_expert_group > cfg.topk_group {
            return Err(format!(
                "grouped top-k (num_expert_group {} > topk_group {}) is not modelled; \
                 this checkpoint has num_expert_group = 1 so there is no vector to check it against",
                cfg.num_expert_group, cfg.topk_group
            ));
        }
        let moe_hidden_size = cfg
            .routed_expert_hidden_size
            .ok_or_else(|| "routed_expert_hidden_size is absent; this port is the LATENT MoE block \
                            (routed_expert_down_proj / _up_proj) and has no non-latent path"
                .to_string())?;
        Ok(Self {
            hidden_size: cfg.hidden_size,
            moe_hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            shared_intermediate_size: cfg
                .num_shared_experts
                .map(|n| cfg.moe_intermediate_size * n),
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_token,
            moe_renormalize: cfg.moe_renormalize,
            routed_scaling_factor: cfg.routed_scaling_factor,
            latent_moe_use_norm: cfg.latent_moe_use_norm,
            rms_norm_eps: cfg.rms_norm_eps,
            situ: Situ::new(
                cfg.activation_situ_beta,
                Some(cfg.activation_situ_linear_beta),
            ),
        })
    }
}

/// `KimiMoEGate`'s two parameters.
#[derive(Clone, Debug)]
pub struct RouterWeights<B: Backend> {
    /// `gate.weight`, `[num_experts, hidden_size]`.
    pub weight: Tensor<B, 2>,
    /// `gate.e_score_correction_bias`, `[num_experts]`. Added to the sigmoid
    /// **score**, and only for selection.
    pub bias: Tensor<B, 1>,
}

/// One routed expert (`KimiBlockSparseMLP`), decoded to f32.
#[derive(Clone, Debug)]
pub struct ExpertWeights<B: Backend> {
    /// `w1` — the **gate** projection, `[moe_intermediate_size, moe_hidden_size]`.
    pub w1: Tensor<B, 2>,
    /// `w2` — the **down** projection, `[moe_hidden_size, moe_intermediate_size]`.
    pub w2: Tensor<B, 2>,
    /// `w3` — the **up** projection, `[moe_intermediate_size, moe_hidden_size]`.
    pub w3: Tensor<B, 2>,
}

/// The fused shared-expert MLP (`KimiMLP` with `intermediate_size = 6144`).
#[derive(Clone, Debug)]
pub struct SharedExpertWeights<B: Backend> {
    /// `[shared_intermediate_size, hidden_size]`.
    pub gate_proj: Tensor<B, 2>,
    /// `[shared_intermediate_size, hidden_size]`.
    pub up_proj: Tensor<B, 2>,
    /// `[hidden_size, shared_intermediate_size]`.
    pub down_proj: Tensor<B, 2>,
}

/// Everything in the block except the 896 routed experts, which arrive one at
/// a time through the caller's closure.
#[derive(Clone, Debug)]
pub struct LatentMoeWeights<B: Backend> {
    /// `routed_expert_down_proj.weight`, `[moe_hidden_size, hidden_size]`.
    pub down_proj: Tensor<B, 2>,
    /// `routed_expert_up_proj.weight`, `[hidden_size, moe_hidden_size]`.
    pub up_proj: Tensor<B, 2>,
    /// `routed_expert_norm.weight`, `[moe_hidden_size]`. `None` iff
    /// `latent_moe_use_norm` is false.
    pub norm: Option<Tensor<B, 1>>,
    /// The router.
    pub router: RouterWeights<B>,
    /// The shared experts. `None` iff `num_shared_experts` is absent.
    pub shared: Option<SharedExpertWeights<B>>,
}

/// Everything `KimiMoEGate.forward` computes, including the intermediates it
/// does not return — so a gate can check the parts and not only the pair.
#[derive(Clone, Debug)]
pub struct Routing<B: Backend> {
    /// `F.linear(h.float(), W.float())`, `[tokens, num_experts]`.
    pub logits: Tensor<B, 2>,
    /// `sigmoid(logits)`.
    pub scores: Tensor<B, 2>,
    /// `scores + e_score_correction_bias` — selection only.
    pub scores_for_choice: Tensor<B, 2>,
    /// Selected expert ids, `tokens · top_k`, row-major, **descending by
    /// `scores_for_choice`**. The shipped gate calls `torch.topk(...,
    /// sorted=False)` and its order is unspecified — measured on the oracle it
    /// is neither ascending by id nor descending by score — so a port must be
    /// compared against it **as a set per token**, never elementwise.
    pub topk_idx: Vec<usize>,
    /// `scores.gather(topk_idx)`, before the renormalisation. Aligned with
    /// [`Self::topk_idx`].
    pub topk_weight_prerenorm: Vec<f32>,
    /// The combining weight: `topk_weight_prerenorm / (Σ + 1e-20) ·
    /// routed_scaling_factor`. Aligned with [`Self::topk_idx`].
    pub topk_weight: Vec<f32>,
    /// Number of tokens.
    pub tokens: usize,
    /// `top_k`.
    pub top_k: usize,
    /// Smallest gap, over all tokens, between the k-th and (k+1)-th
    /// `scores_for_choice`. A port and the reference can only disagree about
    /// *which* experts fire through a tie; measuring the margin turns "assume
    /// no ties" into a number.
    pub min_topk_margin: f32,
}

impl<B: Backend> Routing<B> {
    /// The distinct experts the router selected, ascending — the exact set
    /// `moe_infer` will call, and the set a streaming loader must fetch.
    pub fn touched_experts(&self) -> Vec<usize> {
        let mut v = self.topk_idx.clone();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// `(token, weight)` pairs for expert `id`, in ascending token order —
    /// the shipped `moe_infer`'s per-expert token block.
    pub fn tokens_for(&self, id: usize) -> (Vec<usize>, Vec<f32>) {
        let mut toks = Vec::new();
        let mut ws = Vec::new();
        for t in 0..self.tokens {
            for k in 0..self.top_k {
                if self.topk_idx[t * self.top_k + k] == id {
                    toks.push(t);
                    ws.push(self.topk_weight[t * self.top_k + k]);
                }
            }
        }
        (toks, ws)
    }
}

/// The intermediates of one routed expert's forward pass.
#[derive(Clone, Debug)]
pub struct ExpertTrace<B: Backend> {
    /// `w1(x)` — the gate half.
    pub w1_out: Tensor<B, 2>,
    /// `w3(x)` — the up half.
    pub w3_out: Tensor<B, 2>,
    /// `cat([w1(x), w3(x)], -1)` — what `SituAndMul` actually receives.
    pub situ_in: Tensor<B, 2>,
    /// `situ(situ_in)`.
    pub situ_out: Tensor<B, 2>,
    /// `w2(situ_out)` — the expert's output.
    pub out: Tensor<B, 2>,
}

/// The intermediates of the fused shared-expert MLP.
#[derive(Clone, Debug)]
pub struct SharedTrace<B: Backend> {
    /// `gate_proj(x)`.
    pub gate_out: Tensor<B, 2>,
    /// `up_proj(x)`.
    pub up_out: Tensor<B, 2>,
    /// `cat([gate_proj(x), up_proj(x)], -1)`.
    pub situ_in: Tensor<B, 2>,
    /// `situ(situ_in)`.
    pub situ_out: Tensor<B, 2>,
    /// `down_proj(situ(...))`.
    pub out: Tensor<B, 2>,
}

/// The intermediates of the whole block, in execution order.
#[derive(Clone, Debug)]
pub struct BlockTrace<B: Backend> {
    /// The router's full output.
    pub routing: Routing<B>,
    /// `routed_expert_down_proj(hidden)` — the latent, `[tokens, moe_hidden]`.
    pub latent_down_out: Tensor<B, 2>,
    /// `moe_infer(...)` — the top-16 combination, `[tokens, moe_hidden]`.
    /// Also `routed_expert_norm`'s input.
    pub combined: Tensor<B, 2>,
    /// `routed_expert_norm(combined)`, or `combined` when the block has no norm.
    pub normed: Tensor<B, 2>,
    /// `routed_expert_up_proj(normed)`, `[tokens, hidden]`.
    pub latent_up_out: Tensor<B, 2>,
    /// The fused shared MLP's output, `[tokens, hidden]`.
    pub shared_out: Option<Tensor<B, 2>>,
    /// `latent_up_out + shared_out`, `[tokens, hidden]`.
    pub out: Tensor<B, 2>,
}

/// The latent MoE block.
#[derive(Clone, Debug)]
pub struct LatentMoe {
    /// Dimensions and the switches this port models.
    pub dims: MoeDims,
    /// Storage precision of the intermediate activations.
    pub round: ActRound,
}

impl LatentMoe {
    /// A block in the shipped bf16 storage lane.
    pub fn new(dims: MoeDims) -> Self {
        Self {
            dims,
            round: ActRound::Bf16,
        }
    }

    /// A block whose intermediates stay f32 — the `dtype=torch.float32` lane.
    pub fn new_f32(dims: MoeDims) -> Self {
        Self {
            dims,
            round: ActRound::None,
        }
    }

    /// `nn.Linear(bias=False)`, at this block's rounding policy.
    ///
    /// One line, because the operation itself lives in [`super::ops::linear`]:
    /// it is shared with the KDA block and the two per-layer norms, and the
    /// place where the rounding goes is exactly the kind of subtlety that must
    /// not exist in four copies.
    pub fn linear<B: Backend>(&self, x: Tensor<B, 2>, w: Tensor<B, 2>) -> Tensor<B, 2> {
        super::ops::linear(x, &w, self.round)
    }

    /// `KimiRMSNorm` at this block's epsilon and rounding policy.
    ///
    /// The two load-bearing subtleties — where the cast goes, and why this must
    /// be a division and never `sqrt().recip()` — are documented at
    /// [`super::ops::rms_norm`], with the code they describe.
    pub fn rms_norm<B: Backend>(&self, x: Tensor<B, 2>, weight: Tensor<B, 1>) -> Tensor<B, 2> {
        super::ops::rms_norm(x, &weight, self.dims.rms_norm_eps, self.round)
    }

    /// `KimiMoEGate.forward`, plus the intermediates it discards.
    ///
    /// Runs entirely in f32 — the shipped gate does
    /// `hidden_states.type(torch.float32)` and `weight.type(torch.float32)`
    /// regardless of the model dtype, so there is no bf16 rounding anywhere in
    /// here and [`ActRound`] deliberately does not touch it.
    pub fn route<B: Backend>(&self, hidden: Tensor<B, 2>, w: &RouterWeights<B>) -> Routing<B> {
        let [tokens, h] = hidden.dims();
        let [e, hw] = w.weight.dims();
        assert_eq!(h, self.dims.hidden_size, "router input width");
        assert_eq!(hw, self.dims.hidden_size, "router weight width");
        assert_eq!(e, self.dims.num_experts, "router weight rows");
        assert_eq!(w.bias.dims()[0], self.dims.num_experts, "router bias length");

        let logits = hidden.matmul(w.weight.clone().transpose());
        let scores = burn::tensor::activation::sigmoid(logits.clone());
        let scores_for_choice = scores.clone() + w.bias.clone().unsqueeze::<2>();

        let sfc: Vec<f32> = scores_for_choice
            .clone()
            .into_data()
            .to_vec()
            .expect("scores_for_choice f32");
        let sc: Vec<f32> = scores.clone().into_data().to_vec().expect("scores f32");

        let k = self.dims.top_k;
        let mut topk_idx = Vec::with_capacity(tokens * k);
        let mut prerenorm = Vec::with_capacity(tokens * k);
        let mut weight = Vec::with_capacity(tokens * k);
        let mut min_margin = f32::INFINITY;

        for t in 0..tokens {
            let row = &sfc[t * e..(t + 1) * e];
            // Deterministic total order: score descending, then expert id
            // ascending. torch's `sorted=False` top-k leaves the order
            // unspecified, so nothing here may depend on reproducing it; what
            // must be reproduced is the SET, and the set is unambiguous as long
            // as the k-th and (k+1)-th scores differ (`min_topk_margin`).
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_unstable_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .expect("router scores are finite")
                    .then(a.cmp(&b))
            });
            min_margin = min_margin.min(row[order[k - 1]] - row[order[k]]);

            // f32, because the shipped `topk_weight.sum(dim=-1)` is f32. Its
            // *order* is `torch.topk(sorted=False)`'s, which is unspecified, so
            // the last ulp of `topk_weight` is not reproducible by anyone —
            // that limit is measured rather than assumed (`R8` in the gate).
            let mut sum = 0f32;
            for &id in &order[..k] {
                let wv = sc[t * e + id];
                topk_idx.push(id);
                prerenorm.push(wv);
                sum += wv;
            }
            // `denominator = topk_weight.sum(-1, keepdim=True) + 1e-20`.
            let denom = if self.dims.moe_renormalize && k > 1 {
                sum + 1e-20f32
            } else {
                1.0
            };
            let scale = self.dims.routed_scaling_factor as f32;
            for j in 0..k {
                let wv = prerenorm[t * k + j];
                weight.push((wv / denom) * scale);
            }
        }

        Routing {
            logits,
            scores,
            scores_for_choice,
            topk_idx,
            topk_weight_prerenorm: prerenorm,
            topk_weight: weight,
            tokens,
            top_k: k,
            min_topk_margin: min_margin,
        }
    }

    /// One routed expert, with its intermediates.
    pub fn expert_traced<B: Backend>(
        &self,
        x: Tensor<B, 2>,
        w: &ExpertWeights<B>,
    ) -> ExpertTrace<B> {
        let w1_out = self.linear(x.clone(), w.w1.clone());
        let w3_out = self.linear(x, w.w3.clone());
        let situ_in = Tensor::cat(vec![w1_out.clone(), w3_out.clone()], 1);
        let situ_out = self.round.apply(self.dims.situ.forward(situ_in.clone()));
        let out = self.linear(situ_out.clone(), w.w2.clone());
        ExpertTrace {
            w1_out,
            w3_out,
            situ_in,
            situ_out,
            out,
        }
    }

    /// The fused shared-expert MLP, with its intermediates.
    pub fn shared_traced<B: Backend>(
        &self,
        x: Tensor<B, 2>,
        w: &SharedExpertWeights<B>,
    ) -> SharedTrace<B> {
        let gate_out = self.linear(x.clone(), w.gate_proj.clone());
        let up_out = self.linear(x, w.up_proj.clone());
        let situ_in = Tensor::cat(vec![gate_out.clone(), up_out.clone()], 1);
        let situ_out = self.round.apply(self.dims.situ.forward(situ_in.clone()));
        let out = self.linear(situ_out.clone(), w.down_proj.clone());
        SharedTrace {
            gate_out,
            up_out,
            situ_in,
            situ_out,
            out,
        }
    }

    /// `moe_infer`: run each selected expert on its own token block and sum the
    /// blocks back with the router's weights.
    ///
    /// `expert` is called **once per distinct selected expert, in ascending
    /// id** — never for an expert with no tokens. That is the shipped loop's
    /// contract and it is what lets a 2.78 T-parameter model touch 172 of 896
    /// experts for a 32-token batch.
    ///
    /// The weighted sum is accumulated in f32 and rounded **once** at the end,
    /// because the shipped `moe_infer` casts to `topk_weight.dtype` (fp32 from
    /// the gate) before multiplying and back to the activation dtype only after
    /// the `.sum(dim=1)`.
    pub fn moe_infer<B: Backend, F>(
        &self,
        latent: Tensor<B, 2>,
        routing: &Routing<B>,
        mut expert: F,
    ) -> Tensor<B, 2>
    where
        F: FnMut(usize) -> ExpertWeights<B>,
    {
        let [tokens, hm] = latent.dims();
        assert_eq!(tokens, routing.tokens, "moe_infer: token count");
        assert_eq!(hm, self.dims.moe_hidden_size, "moe_infer: latent width");
        let device = latent.device();
        let mut acc: Tensor<B, 2> = Tensor::zeros([tokens, hm], &device);

        for id in routing.touched_experts() {
            let (toks, ws) = routing.tokens_for(id);
            debug_assert!(!toks.is_empty(), "touched_experts yielded an empty expert");
            let sel = Tensor::<B, 1, Int>::from_data(
                TensorData::new(toks.iter().map(|&t| t as i64).collect::<Vec<_>>(), [toks.len()]),
                &device,
            );
            let block = latent.clone().select(0, sel.clone());
            let out = self.expert_traced(block, &expert(id)).out;
            let wt = Tensor::<B, 2>::from_data(TensorData::new(ws, [toks.len(), 1]), &device);
            acc = acc.select_assign(0, sel, out * wt, IndexingUpdateOp::Add);
        }
        self.round.apply(acc)
    }

    /// The whole block: `KimiSparseMoeBlock.forward`, with its intermediates.
    ///
    /// `hidden` is `[tokens, hidden_size]` — already flattened, because
    /// everything downstream of the router is per-token and the shipped module
    /// flattens on its first line.
    pub fn forward_traced<B: Backend, F>(
        &self,
        hidden: Tensor<B, 2>,
        w: &LatentMoeWeights<B>,
        expert: F,
    ) -> BlockTrace<B>
    where
        F: FnMut(usize) -> ExpertWeights<B>,
    {
        let routing = self.route(hidden.clone(), &w.router);
        let latent_down_out = self.linear(hidden.clone(), w.down_proj.clone());
        let combined = self.moe_infer(latent_down_out.clone(), &routing, expert);
        let normed = match (&w.norm, self.dims.latent_moe_use_norm) {
            (Some(nw), true) => self.rms_norm(combined.clone(), nw.clone()),
            (None, false) => combined.clone(),
            (Some(_), false) => panic!("latent_moe_use_norm is false but a norm weight was given"),
            (None, true) => panic!("latent_moe_use_norm is true but no norm weight was given"),
        };
        let latent_up_out = self.linear(normed.clone(), w.up_proj.clone());
        let shared_out = w
            .shared
            .as_ref()
            .map(|sw| self.shared_traced(hidden, sw).out);
        let out = match &shared_out {
            Some(s) => self.combine_with_shared(latent_up_out.clone(), s.clone()),
            None => latent_up_out.clone(),
        };
        BlockTrace {
            routing,
            latent_down_out,
            combined,
            normed,
            latent_up_out,
            shared_out,
            out,
        }
    }

    /// The block's last line: `y + shared_experts(identity)`, in the ORIGINAL
    /// hidden space, rounded once.
    ///
    /// A separate method because the rounding here is a claim a gate has to be
    /// able to test on its own: driven from two captured tensors it must
    /// reproduce the shipped `moe_out` **bit-for-bit**, which an unrounded add
    /// does not. Measured on the oracle: `moe_out ==
    /// round_bf16(latent_up_out + shared_out)` on all 229,376 elements.
    pub fn combine_with_shared<B: Backend>(
        &self,
        latent_up_out: Tensor<B, 2>,
        shared_out: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.round.apply(latent_up_out + shared_out)
    }

    /// [`Self::forward_traced`], keeping only the output.
    pub fn forward<B: Backend, F>(
        &self,
        hidden: Tensor<B, 2>,
        w: &LatentMoeWeights<B>,
        expert: F,
    ) -> Tensor<B, 2>
    where
        F: FnMut(usize) -> ExpertWeights<B>,
    {
        self.forward_traced(hidden, w, expert).out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `half::bf16::from_f32` is load-bearing: [`ActRound::Bf16`] is what makes
    /// this port comparable to the shipped bf16 run bit-for-bit, and it is a
    /// third-party rounding. Check it against the round-to-nearest-EVEN bit
    /// trick directly rather than trusting the crate's docs — a
    /// round-half-away-from-zero implementation would agree on almost every
    /// value and disagree on exactly the ties.
    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        fn rne(x: f32) -> f32 {
            let u = x.to_bits();
            if x.is_nan() {
                return x;
            }
            let r = ((u >> 16) & 1) + 0x7fff;
            f32::from_bits((u.wrapping_add(r)) & 0xffff_0000)
        }
        // Exact ties (mantissa low half == 0x8000) are where the two rules
        // differ, so walk them explicitly as well as a broad sweep.
        let mut cases: Vec<f32> = Vec::new();
        for hi in 0u32..512 {
            cases.push(f32::from_bits((0x3f00_0000 + (hi << 16)) | 0x8000));
            cases.push(f32::from_bits((0x3f00_0000 + (hi << 16)) | 0x0000));
            cases.push(f32::from_bits((0xbf00_0000 + (hi << 16)) | 0x8000));
        }
        let mut x = 1e-30f32;
        while x < 1e30 {
            cases.push(x);
            cases.push(-x);
            x *= 1.37;
        }
        for &c in &cases {
            assert_eq!(
                half::bf16::from_f32(c).to_f32().to_bits(),
                rne(c).to_bits(),
                "bf16 rounding of {c:e} (bits {:#010x})",
                c.to_bits()
            );
        }
    }
}
