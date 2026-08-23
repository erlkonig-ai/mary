//! Kimi K3's MoE router: the `noaux_tc` sigmoid gate with a trained
//! `e_score_correction_bias`, ported from `KimiMoEGate.forward`.
//!
//! # The one thing that is easy to get wrong
//!
//! The trained per-expert bias steers **which** experts are chosen and must not
//! reach the **combining weight**:
//!
//! ```text
//! scores            = sigmoid(h @ Wᵀ)              in f32
//! scores_for_choice = scores + e_score_correction_bias      <- SELECTION only
//! idx               = top_k(scores_for_choice)
//! w                 = scores.gather(idx)                    <- UNBIASED
//! w                 = w / (Σw + 1e-20) * routed_scaling_factor
//! ```
//!
//! A port that gathers from `scores_for_choice` still routes to exactly the
//! right experts and still produces a plausible, normalised, sums-to-one weight
//! vector. On this checkpoint's 12 gated layers the resulting weights are wrong
//! by up to 1.03e-01 absolute on a max weight of ~0.36 — a ~30% error that
//! *nothing* in an aggregate statistic reveals: the expert histogram, the load
//! balance, and the weight sum are all identical.
//!
//! So the two halves are not both `Vec<f32>` here. [`Scores`] and
//! [`ScoresForChoice`] are distinct types, [`Router::select`] takes only the
//! latter, and [`Router::combine_weights`] takes only the former. The mistake
//! is a compile error rather than a silent 30%.
//!
//! # `noaux_tc` on this config
//!
//! `config.json` says `topk_method: "noaux_tc"`, but the shipped
//! `KimiMoEGate.forward` never reads that field. It branches on
//! `num_expert_group > 1 and num_expert_group > topk_group`, and Kimi K3 ships
//! `num_expert_group = 1`, `topk_group = 1` — so the DeepSeek-V3 group-limited
//! routing the name refers to is **dead code on this checkpoint**. What
//! survives of `noaux_tc` is exactly the bias-corrected selection above.
//!
//! Rather than carry an untestable transcription of the grouped branch, this
//! port refuses it: [`RouterConfig::validate`] rejects any config that would
//! take it. Not-ported fails loudly; it does not silently route as if grouping
//! were absent.
//!
//! # `e_score_correction_bias` is FLOAT32 on disk — and that changes routing
//!
//! MEASURED, header scan of all 96 shards: of 2,628 non-MXFP4 tensors, exactly
//! 506 are F32, and 92 of them are `block_sparse_moe.gate.e_score_correction_bias`
//! — one per MoE layer. (The other 414 are the six KDA families.) Everything
//! else in the gate, including `gate.weight`, is bf16.
//!
//! `config.json` says `dtype: bfloat16`, so a plain `from_pretrained` **rounds
//! the bias down to bf16**, and the whole-layer oracle was captured that way.
//! The rounding is up to 4.8e-04, and the margin between the 16th and 17th
//! `scores_for_choice` gets down to 3.09e-06 — so it is not cosmetic: on the
//! oracle's 32 real tokens it changes the chosen expert set for **6 of 384**
//! token/layer selections (1.6%), at layers 1, 3 and 12.
//!
//! Nothing available here decides which one production serves. [`Router`] takes
//! `bias: Vec<f32>` and does not choose: the caller decides whether to load the
//! f32 the checkpoint stores or the bf16 a `dtype=bfloat16` load would hold.
//! The gate binary runs the bf16 lane (to match the oracle) and *measures* the
//! f32 lane's divergence against a pinned count, so the question stays visible
//! instead of being quietly settled by a default.
//!
//! # Precision
//!
//! The gate runs in f32 even when the model is bf16 (`hidden_states.type(f32)`,
//! `weight.type(f32)`), so [`Router`] holds f32 and returns f32 throughout.
//! The accumulation order of the `h @ Wᵀ` reduction is *not* incidental: the
//! margin between the 16th and 17th `scores_for_choice` on real tokens gets as
//! small as 3.09e-06, so a sloppier reduction reroutes tokens. [`Accum`] makes
//! that choice explicit instead of implicit in whatever BLAS is linked.

/// How the `h @ Wᵀ` reduction accumulates. The product and the final result are
/// f32 either way; this is the accumulator only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accum {
    /// f32 accumulator — what a fused kernel does, and what `F.linear` on f32
    /// inputs does up to its own blocking.
    F32,
    /// f64 accumulator, rounded to f32 at the end. The reference lane: it
    /// removes reduction order from the comparison so that a selection
    /// disagreement means a *semantic* disagreement.
    F64,
}

/// The router's activation. Kimi K3 ships `sigmoid`; `softmax` exists in the
/// shipped gate and is carried so a config that selects it cannot be read as
/// sigmoid by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterActivation {
    Sigmoid,
    Softmax,
}

/// Everything `KimiMoEGate` reads out of the config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouterConfig {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub activation: RouterActivation,
    pub renormalize: bool,
    pub routed_scaling_factor: f32,
    pub num_expert_group: usize,
    pub topk_group: usize,
}

impl RouterConfig {
    /// Kimi K3's shipping values (`config.json`, `text_config`). Every one of
    /// these is re-read from the checkpoint by the gate binary rather than
    /// trusted here.
    pub fn k3() -> Self {
        Self {
            hidden_size: 7168,
            num_experts: 896,
            top_k: 16,
            activation: RouterActivation::Sigmoid,
            renormalize: true,
            routed_scaling_factor: 1.0,
            num_expert_group: 1,
            topk_group: 1,
        }
    }

    /// Reject configs this port does not implement, loudly.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_experts == 0 || self.hidden_size == 0 {
            return Err("router: zero-sized config".into());
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(format!(
                "router: top_k {} out of range for {} experts",
                self.top_k, self.num_experts
            ));
        }
        if self.num_expert_group > 1 && self.num_expert_group > self.topk_group {
            return Err(format!(
                "router: group-limited noaux_tc routing (num_expert_group {} > topk_group {}) \
                 is NOT PORTED — Kimi K3 ships num_expert_group=1 so the shipped gate never \
                 takes that branch, and an ungated transcription of it would be a green check \
                 over code no oracle has ever executed",
                self.num_expert_group, self.topk_group
            ));
        }
        Ok(())
    }
}

/// `sigmoid(h @ Wᵀ)` — the **unbiased** per-expert scores. The combining
/// weights come from here and from nowhere else.
#[derive(Debug, Clone)]
pub struct Scores {
    v: Vec<f32>,
    tokens: usize,
    experts: usize,
}

/// [`Scores`] plus the trained correction bias — used to **choose** experts and
/// for nothing else. Deliberately not the same type as [`Scores`]: this is the
/// value that must never reach a combining weight.
#[derive(Debug, Clone)]
pub struct ScoresForChoice {
    v: Vec<f32>,
    tokens: usize,
    experts: usize,
}

impl Scores {
    /// Wrap values that did not come from [`Router::scores`] — an oracle array,
    /// or a deliberately-wrong construction in a gate. Named to be conspicuous
    /// at the call site; nothing in the model path should call it.
    pub fn from_raw(v: Vec<f32>, tokens: usize, experts: usize) -> Self {
        assert_eq!(v.len(), tokens * experts, "Scores::from_raw shape");
        assert!(tokens > 0 && experts > 0, "Scores::from_raw is empty");
        Self { v, tokens, experts }
    }
    pub fn row(&self, t: usize) -> &[f32] {
        &self.v[t * self.experts..(t + 1) * self.experts]
    }
    pub fn as_slice(&self) -> &[f32] {
        &self.v
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn experts(&self) -> usize {
        self.experts
    }
}

impl ScoresForChoice {
    /// See [`Scores::from_raw`].
    pub fn from_raw(v: Vec<f32>, tokens: usize, experts: usize) -> Self {
        assert_eq!(v.len(), tokens * experts, "ScoresForChoice::from_raw shape");
        assert!(
            tokens > 0 && experts > 0,
            "ScoresForChoice::from_raw is empty"
        );
        Self { v, tokens, experts }
    }
    pub fn row(&self, t: usize) -> &[f32] {
        &self.v[t * self.experts..(t + 1) * self.experts]
    }
    pub fn as_slice(&self) -> &[f32] {
        &self.v
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn experts(&self) -> usize {
        self.experts
    }
}

/// One token's routing decision: which experts, and with what combining weight.
#[derive(Debug, Clone)]
pub struct Routing {
    pub tokens: usize,
    pub top_k: usize,
    /// `tokens * top_k` expert indices.
    ///
    /// **Order.** `torch.topk(..., sorted=False)` returns an implementation-
    /// defined order, so the shipped `topk_idx` is *not* sorted and downstream
    /// code must not depend on its order (`moe_infer` scatters by index). This
    /// port emits a deterministic order — descending by `scores_for_choice`,
    /// ties broken by ascending expert index — so that a port-vs-port diff is
    /// meaningful. Compare against the shipped tensor as a *set*, pairing the
    /// weights through the index.
    ///
    /// That a fixed order is *safe* is READ from `moe_infer`, not executed
    /// here: it argsorts the flat `topk_ids`, runs each expert over its own
    /// tokens, scatters back to the original slot, and finishes with
    /// `mul_(topk_weight.unsqueeze(-1)).sum(dim=1)` — a sum over slots, hence
    /// invariant under a *joint* permutation of `(idx, weight)`. A consumer
    /// must therefore carry the two together; permuting one alone is a bug this
    /// port cannot detect.
    pub idx: Vec<u32>,
    /// `tokens * top_k` combining weights, aligned with `idx`, after
    /// renormalisation and the routed scaling factor.
    pub weight: Vec<f32>,
    /// The same weights *before* `/Σ` and the scaling factor — the raw gathered
    /// `scores`. Kept because it is the value that isolates the bias mistake.
    pub weight_prerenorm: Vec<f32>,
}

impl Routing {
    pub fn idx_row(&self, t: usize) -> &[u32] {
        &self.idx[t * self.top_k..(t + 1) * self.top_k]
    }
    pub fn weight_row(&self, t: usize) -> &[f32] {
        &self.weight[t * self.top_k..(t + 1) * self.top_k]
    }
}

/// The router: the gate projection, the correction bias, and the config.
#[derive(Debug, Clone)]
pub struct Router {
    cfg: RouterConfig,
    /// `[num_experts * hidden_size]`, row-major — expert `e`'s row is
    /// `weight[e*hidden .. (e+1)*hidden]`, the layout `nn.Linear` stores.
    weight: Vec<f32>,
    /// `[num_experts]`.
    bias: Vec<f32>,
}

impl Router {
    pub fn new(cfg: RouterConfig, weight: Vec<f32>, bias: Vec<f32>) -> Result<Self, String> {
        cfg.validate()?;
        if weight.len() != cfg.num_experts * cfg.hidden_size {
            return Err(format!(
                "router weight has {} elements, expected {}x{} = {}",
                weight.len(),
                cfg.num_experts,
                cfg.hidden_size,
                cfg.num_experts * cfg.hidden_size
            ));
        }
        if bias.len() != cfg.num_experts {
            return Err(format!(
                "router bias has {} elements, expected {}",
                bias.len(),
                cfg.num_experts
            ));
        }
        Ok(Self { cfg, weight, bias })
    }

    pub fn config(&self) -> &RouterConfig {
        &self.cfg
    }

    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// `F.linear(h.float(), W.float())` — `[tokens, num_experts]`, row-major.
    ///
    /// `h` is `[tokens * hidden_size]`; the caller has already flattened
    /// `[batch, seq, hidden]`, exactly as the shipped `view(-1, h)` does.
    ///
    /// This is a reference reduction, not a tuned kernel: three nested loops,
    /// no blocking, no SIMD intrinsics. It exists to make the accumulation
    /// order explicit and inspectable, because that order is what decides the
    /// 16th/17th boundary. A serving path should replace it with a gemm and
    /// then re-run this gate to confirm the selection did not move.
    pub fn logits(&self, h: &[f32], tokens: usize, accum: Accum) -> Vec<f32> {
        assert!(tokens > 0, "logits() on zero tokens");
        assert_eq!(
            h.len(),
            tokens * self.cfg.hidden_size,
            "hidden states are {} elements, expected {} tokens x {}",
            h.len(),
            tokens,
            self.cfg.hidden_size
        );
        let hid = self.cfg.hidden_size;
        let mut out = vec![0f32; tokens * self.cfg.num_experts];
        for t in 0..tokens {
            let x = &h[t * hid..(t + 1) * hid];
            for e in 0..self.cfg.num_experts {
                let w = &self.weight[e * hid..(e + 1) * hid];
                out[t * self.cfg.num_experts + e] = match accum {
                    Accum::F32 => x.iter().zip(w).fold(0f32, |s, (&a, &b)| s + a * b),
                    Accum::F64 => x
                        .iter()
                        .zip(w)
                        .fold(0f64, |s, (&a, &b)| s + a as f64 * b as f64)
                        as f32,
                };
            }
        }
        out
    }

    /// The activation. Computed in f64 and rounded to f32: the shipped call is
    /// `logits.sigmoid()` on an f32 tensor, whose per-element result is the
    /// correctly-rounded f32 of the exact value, and a naive f32
    /// `1/(1+exp(-x))` is not.
    pub fn scores(&self, logits: &[f32], tokens: usize) -> Scores {
        assert!(tokens > 0, "scores() on zero tokens");
        assert_eq!(logits.len(), tokens * self.cfg.num_experts);
        let v: Vec<f32> = match self.cfg.activation {
            RouterActivation::Sigmoid => logits
                .iter()
                .map(|&x| (1.0f64 / (1.0 + (-(x as f64)).exp())) as f32)
                .collect(),
            RouterActivation::Softmax => {
                let n = self.cfg.num_experts;
                let mut v = vec![0f32; logits.len()];
                for t in 0..tokens {
                    let row = &logits[t * n..(t + 1) * n];
                    let m = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));
                    let mut s = 0f64;
                    for (i, &x) in row.iter().enumerate() {
                        let e = ((x as f64) - m).exp();
                        v[t * n + i] = e as f32;
                        s += e;
                    }
                    for i in 0..n {
                        v[t * n + i] = ((v[t * n + i] as f64) / s) as f32;
                    }
                }
                v
            }
        };
        Scores {
            v,
            tokens,
            experts: self.cfg.num_experts,
        }
    }

    /// `scores + e_score_correction_bias`. SELECTION ONLY.
    pub fn scores_for_choice(&self, scores: &Scores) -> ScoresForChoice {
        assert_eq!(scores.experts, self.cfg.num_experts);
        let n = self.cfg.num_experts;
        let mut v = vec![0f32; scores.v.len()];
        for t in 0..scores.tokens {
            for e in 0..n {
                v[t * n + e] = scores.v[t * n + e] + self.bias[e];
            }
        }
        ScoresForChoice {
            v,
            tokens: scores.tokens,
            experts: n,
        }
    }

    /// The top-`top_k` experts per token, by [`ScoresForChoice`].
    ///
    /// Takes `ScoresForChoice` and nothing else, so selection cannot silently
    /// be run on the unbiased scores.
    pub fn select(&self, sfc: &ScoresForChoice) -> Vec<u32> {
        assert_eq!(sfc.experts, self.cfg.num_experts);
        assert!(sfc.tokens > 0, "select() on zero tokens");
        let k = self.cfg.top_k;
        let n = self.cfg.num_experts;
        let mut idx = vec![0u32; sfc.tokens * k];
        let mut order: Vec<u32> = Vec::with_capacity(n);
        for t in 0..sfc.tokens {
            let row = sfc.row(t);
            order.clear();
            order.extend(0..n as u32);
            // descending by score, ties by ascending index — total order, so
            // the result does not depend on the sort's stability
            order.sort_unstable_by(|&a, &b| {
                let (sa, sb) = (row[a as usize], row[b as usize]);
                sb.partial_cmp(&sa)
                    .unwrap_or_else(|| {
                        panic!("router: non-finite score at token {t}, experts {a}/{b}")
                    })
                    .then(a.cmp(&b))
            });
            idx[t * k..(t + 1) * k].copy_from_slice(&order[..k]);
        }
        idx
    }

    /// `scores.gather(1, idx)` — the raw combining weights, **before**
    /// normalisation.
    ///
    /// Takes [`Scores`], the unbiased ones. This signature is the whole point
    /// of the newtype: `scores_for_choice` does not typecheck here.
    pub fn combine_weights(&self, scores: &Scores, idx: &[u32]) -> Vec<f32> {
        let k = self.cfg.top_k;
        assert_eq!(scores.experts, self.cfg.num_experts);
        assert_eq!(idx.len(), scores.tokens * k, "idx length vs tokens*top_k");
        let n = self.cfg.num_experts;
        let mut out = vec![0f32; idx.len()];
        for t in 0..scores.tokens {
            for j in 0..k {
                let e = idx[t * k + j] as usize;
                assert!(e < n, "expert index {e} out of range");
                out[t * k + j] = scores.v[t * n + e];
            }
        }
        out
    }

    /// `w / (Σw + 1e-20) * routed_scaling_factor`, the shipped normalisation.
    /// The `+ 1e-20` is a no-op at f32 for any realistic sum and is kept
    /// because the shipped expression has it.
    pub fn normalize(&self, prerenorm: &[f32], tokens: usize) -> Vec<f32> {
        let k = self.cfg.top_k;
        assert_eq!(prerenorm.len(), tokens * k);
        let mut out = prerenorm.to_vec();
        if k > 1 && self.cfg.renormalize {
            for t in 0..tokens {
                let row = &mut out[t * k..(t + 1) * k];
                let mut s = 0f32;
                for &x in row.iter() {
                    s += x;
                }
                let d = s + 1e-20f32;
                for x in row.iter_mut() {
                    *x /= d;
                }
            }
        }
        if self.cfg.routed_scaling_factor != 1.0 {
            for x in out.iter_mut() {
                *x *= self.cfg.routed_scaling_factor;
            }
        }
        out
    }

    /// The whole gate: flattened hidden states in, routing out.
    pub fn route(&self, h: &[f32], tokens: usize, accum: Accum) -> Routing {
        let logits = self.logits(h, tokens, accum);
        let scores = self.scores(&logits, tokens);
        let sfc = self.scores_for_choice(&scores);
        let idx = self.select(&sfc);
        let prerenorm = self.combine_weights(&scores, &idx);
        let weight = self.normalize(&prerenorm, tokens);
        Routing {
            tokens,
            top_k: self.cfg.top_k,
            idx,
            weight,
            weight_prerenorm: prerenorm,
        }
    }
}

/// Widen a bfloat16 bit pattern to f32: the 16 bits are the *top* half of the
/// f32, so this is a shift, not a numeric cast.
pub fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> RouterConfig {
        RouterConfig {
            hidden_size: 2,
            num_experts: 4,
            top_k: 2,
            activation: RouterActivation::Sigmoid,
            renormalize: true,
            routed_scaling_factor: 1.0,
            num_expert_group: 1,
            topk_group: 1,
        }
    }

    /// A bias big enough to reorder the experts must reorder the *selection*
    /// and leave the *weights* alone.
    #[test]
    fn bias_moves_selection_not_weights() {
        let cfg = tiny();
        // expert rows: e0 strongest, e3 weakest, on the token [1, 0]
        let w = vec![
            2.0, 0.0, // e0
            1.0, 0.0, // e1
            0.5, 0.0, // e2
            0.0, 0.0, // e3
        ];
        let h = vec![1.0f32, 0.0];

        let unbiased = Router::new(cfg, w.clone(), vec![0.0; 4]).unwrap();
        let r0 = unbiased.route(&h, 1, Accum::F64);
        assert_eq!(r0.idx_row(0), &[0, 1]);

        // push e3 to the front by selection only
        let biased = Router::new(cfg, w, vec![0.0, 0.0, 0.0, 1.0]).unwrap();
        let r1 = biased.route(&h, 1, Accum::F64);
        assert_eq!(r1.idx_row(0), &[3, 0]);

        // the weight for e0 must be the SAME unbiased score in both routings,
        // up to the different renormalising partner
        let s = biased.scores(&biased.logits(&h, 1, Accum::F64), 1);
        assert_eq!(r1.weight_prerenorm[0], s.row(0)[3]);
        assert_eq!(r1.weight_prerenorm[1], s.row(0)[0]);
        // and the bias must NOT be in them
        assert!(r1.weight_prerenorm[0] < 1.0, "bias leaked into the weight");
    }

    #[test]
    fn weights_renormalise_to_one() {
        let cfg = tiny();
        let w = vec![2.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0];
        let r = Router::new(cfg, w, vec![0.0; 4]).unwrap();
        let out = r.route(&[1.0, 0.0], 1, Accum::F64);
        let s: f32 = out.weight_row(0).iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "weights sum to {s}");
    }

    #[test]
    fn grouped_routing_is_refused_not_faked() {
        let mut cfg = tiny();
        cfg.num_expert_group = 2;
        cfg.topk_group = 1;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("NOT PORTED"), "{err}");
    }

    #[test]
    fn bf16_widening_is_a_shift() {
        // 0x3F80 is bf16 1.0; a numeric cast of the integer would give 16256.0
        assert_eq!(bf16_bits_to_f32(&[0x3F80]), vec![1.0f32]);
        assert_eq!(bf16_bits_to_f32(&[0xBF80]), vec![-1.0f32]);
    }
}
