//! Multi-head Latent Attention as Kimi K3 ships it — **without any positional
//! encoding at all**.
//!
//! # The trap this module exists to not fall into
//!
//! K3's MLA is DeepSeek-V3's MLA with the rotary embedding *removed but not
//! deleted*. `config.mla_use_nope` is `true`, `KimiMLAAttention.__init__` ends
//! with `assert self.use_nope` and sets `self.rotary_emb = None`, and the
//! forward never calls a rotation. The **tensor shapes were kept**: the query
//! projection still emits `qk_nope_head_dim + qk_rope_head_dim = 128 + 64 =
//! 192` per head, and `kv_a_proj_with_mqa` still emits `kv_lora_rank +
//! qk_rope_head_dim = 512 + 64 = 576`. So a 64-wide slice *named for RoPE*
//! flows through the block and is never rotated.
//!
//! A port that reads the name and helpfully applies RoPE compiles, runs,
//! produces plausible activations, and is wrong in every one of the 24 full
//! attention layers — and, because MLA is the *only* place K3 could carry
//! absolute position, wrong in a way that corrupts the model's sense of order
//! rather than merely its accuracy. This module therefore calls the slice
//! **carried**, never "rope", and asserts the no-rotation property four ways:
//!
//! 1. **Nothing to call.** There is no rotation function, no rotary table, no
//!    `Option<RopeTable>` field, and no cos/sin cache anywhere in this module
//!    or reachable from [`MlaBlock`]. Applying RoPE is not a bug you can make
//!    here; it is code you would have to add.
//! 2. **Nothing to configure.** [`MlaConfig::from_text_config`] *refuses* a
//!    config with `mla_use_nope: false`. There is no rotated variant of this
//!    block to select, so the port cannot be pointed at one by a config edit.
//! 3. **Verbatim, checked bit-for-bit.** [`MlaTrace::assert_carried_verbatim`]
//!    asserts that the 64 carried lanes of the assembled `query_states` and
//!    `key_states` are *bit-identical* to the projection outputs they were
//!    sliced from. Any rotation — or scale, or reorder, or sign flip — between
//!    projection and assembly makes that comparison fail on the first element.
//!    [`forward`] runs it on every call.
//! 4. **Measured consequence.** `k3_mla_gate` carries a positive control that
//!    applies a real RoPE to the carried lanes and measures how far the block
//!    output moves. It moves ~50% of the output's dynamic range, so the gate is
//!    demonstrably sensitive to exactly the mistake this module is guarding
//!    against.
//!
//! What is *not* asserted, because it is not true: the **split point** between
//! the 128 passed dims and the 64 carried dims does not affect the attention
//! output. With nothing rotated, `q·k` is a plain sum over all 192 dims and is
//! invariant under any joint permutation of q's and k's last axis. What matters
//! is where each dimension *comes from* — `key_states[..., 0:128]` from
//! `kv_b_proj`'s first half, `[..., 128:192]` from `kv_a_proj_with_mqa`'s tail,
//! broadcast across all 96 heads — and that is what points 3 above pins.
//!
//! # Shape of the block
//!
//! ```text
//! hidden [B,T,7168]
//!   ├─ q_a_proj    → [B,T,1536]  ─ q_a_layernorm ─ q_b_proj → [B,T,96*192]
//!   │                                                        → q [B,96,T,192]
//!   │                                    = cat(q_pass[128], q_carried[64])
//!   ├─ kv_a_proj_with_mqa → [B,T,576] = cat(latent[512], k_carried[64])
//!   │        latent ─ kv_a_layernorm ─ kv_b_proj → [B,T,96*256]
//!   │                                            → cat(k_pass[128], v[128])
//!   │        k_carried ─ broadcast over 96 heads ─┐
//!   │                             k [B,96,T,192] = cat(k_pass, k_carried)
//!   ├─ attention: softmax(q·kᵀ · 192^-0.5 + mask) · v  → [B,T,96,128]
//!   └─ g_proj → sigmoid → elementwise gate → o_proj → [B,T,7168]
//! ```
//!
//! # Three details that are easy to get wrong and are pinned here
//!
//! * **The softmax scale is `q_head_dim^-0.5 = 192^-0.5`, not `128^-0.5`.** The
//!   never-rotated 64 dims still participate in the dot product, so they count
//!   in the scale. `self.scaling = self.q_head_dim ** (-0.5)` in the shipped
//!   `__init__`.
//! * **`q_a_layernorm` and `kv_a_layernorm` use eps `1e-6`, not
//!   `config.rms_norm_eps`.** The shipped constructions are bare
//!   `KimiRMSNorm(self.q_lora_rank)` / `KimiRMSNorm(self.kv_lora_rank)` — no
//!   `eps=` argument — so they take `KimiRMSNorm.__init__`'s default of `1e-6`,
//!   while every norm in `KimiDecoderLayer` is built with
//!   `eps=config.rms_norm_eps` (`1e-5` for K3). Two different epsilons live in
//!   the same layer. See [`LORA_NORM_EPS`].
//! * **RMSNorm casts back to the activation dtype *before* the weight
//!   multiply**: `return self.weight * x.to(dtype)`. In a bf16 run the scale
//!   and the weight multiply therefore round separately. See [`Precision`].
//!
//! # Precision
//!
//! The shipped module pins two reductions to fp32 whatever the parameter dtype
//! is: `KimiRMSNorm` computes in `.float()`, and `eager_attention_forward` does
//! `F.softmax(..., dtype=torch.float32)` and then casts the probabilities
//! *back*. Everything else runs in the storage dtype. [`Precision::Bf16`]
//! reproduces a bfloat16 run inside an f32 backend by rounding to bfloat16 at
//! exactly the points where torch materialises a bfloat16 tensor, leaving the
//! two islands wide; [`Precision::Exact`] runs everything in the backend's own
//! element type, which is what an f32 or f64 reference run does.
//!
//! Gated against a whole-layer oracle captured from the shipped
//! `KimiMLAAttention` on real checkpoint weights — see `src/bin/k3_mla_gate.rs`.

use burn::prelude::*;
use burn::tensor::TensorData;
use half::bf16;

use super::config::K3TextConfig;

/// Epsilon of the two RMSNorms *inside* MLA (`q_a_layernorm`, `kv_a_layernorm`).
///
/// **Not** `config.rms_norm_eps`. The shipped code builds these two norms
/// without an `eps=` argument, so they take `KimiRMSNorm.__init__`'s default of
/// `1e-6`, while `input_layernorm`, `post_attention_layernorm` and both AttnRes
/// norms in the same decoder layer are built with `eps=config.rms_norm_eps`,
/// which K3's `config.json` sets to `1e-5`. A port that routes
/// `config.rms_norm_eps` into MLA's norms is off by a factor of ten in the
/// epsilon — invisible in bfloat16, ~4e-6 relative in float64.
pub const LORA_NORM_EPS: f64 = 1e-6;

/// The additive value the shipped causal mask uses for a disallowed position:
/// `torch.finfo(torch.bfloat16).min`.
///
/// It is the bfloat16 minimum even when the model runs in float32 or float64,
/// because the mask is built once at the model's storage dtype and reused. It
/// is finite, not `-inf`, so `scores + mask` never produces a NaN from
/// `inf - inf`.
pub fn mask_neg() -> f64 {
    bf16::MIN.to_f64()
}

/// Where a run rounds.
///
/// The shipped module's arithmetic is *not* uniformly one dtype: two reductions
/// are pinned to fp32 regardless of storage. Modelling that faithfully is the
/// difference between matching a bfloat16 reference to one ULP and matching it
/// to 4e-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Every operation runs in the backend's element type and nothing is
    /// rounded in between. This is what a float32 or float64 reference run of
    /// the shipped module does — the fp32 islands are already at (or wider
    /// than) the storage precision, so they are no-ops.
    Exact,
    /// Reproduce a bfloat16 run inside a wider backend: round to bfloat16 at
    /// every point where the shipped module materialises a bfloat16 tensor, and
    /// leave the RMSNorm reciprocal-sqrt and the attention softmax at the
    /// backend's precision (they are the fp32 islands).
    Bf16,
}

impl Precision {
    /// The same choice, as the crate-wide [`ActRound`](crate::models::k3::ops::ActRound).
    ///
    /// The two enums are the same distinction named twice — this one predates
    /// the shared one. Kept as a conversion rather than a rename so the MLA
    /// gate's vocabulary does not move under it.
    pub fn act_round(self) -> crate::models::k3::ops::ActRound {
        match self {
            Precision::Exact => crate::models::k3::ops::ActRound::None,
            Precision::Bf16 => crate::models::k3::ops::ActRound::Bf16,
        }
    }

    /// Round to bfloat16 if this is the bfloat16 lane; otherwise identity.
    ///
    /// Round-to-nearest-even, matching torch's f32→bf16 cast. Going through
    /// f64 is exact for f32 inputs, so there is no double rounding.
    pub fn round<B: Backend, const D: usize>(self, t: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Precision::Exact => t,
            Precision::Bf16 => {
                let device = t.device();
                let dims = t.dims();
                let v: Vec<f64> = t
                    .into_data()
                    .convert::<f64>()
                    .into_vec()
                    .expect("host readback for bfloat16 rounding");
                let v: Vec<f64> = v.into_iter().map(|x| bf16::from_f64(x).to_f64()).collect();
                Tensor::from_data(TensorData::new(v, dims).convert::<B::FloatElem>(), &device)
            }
        }
    }
}

/// The MLA hyper-parameters, all of them read from the checkpoint's config.
///
/// There is deliberately no `use_nope` field: this block *is* the NoPE block,
/// and [`MlaConfig::from_text_config`] refuses to build one from a config that
/// says otherwise. Carrying the flag would imply a rotated variant exists.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaConfig {
    /// Residual-stream width. 7168.
    pub hidden_size: usize,
    /// Query heads. 96. K3 sets `num_key_value_heads` equal to this, so the
    /// `repeat_kv` in the shipped eager path is the identity.
    pub num_heads: usize,
    /// Rank of the query down-projection, or `None` for an undecomposed
    /// `q_proj`. K3 sets 1536, so the MLA layers carry `q_a_proj`/`q_b_proj`
    /// with an RMSNorm between them.
    pub q_lora_rank: Option<usize>,
    /// Width of the latent KV cache. 512.
    pub kv_lora_rank: usize,
    /// Per-head width of the part of q/k that comes through `kv_b_proj`. 128.
    pub qk_nope_head_dim: usize,
    /// Per-head width of the slice DeepSeek-V3 rotates and K3 does not. 64.
    ///
    /// Named `qk_rope_head_dim` in the checkpoint config; deliberately renamed
    /// here, because in this model it is a *carried* lane and nothing else.
    pub qk_carried_head_dim: usize,
    /// Per-head width of the value. 128.
    pub v_head_dim: usize,
    /// Whether the block has the sigmoid output gate (`g_proj`). True for K3.
    pub use_output_gate: bool,
}

impl MlaConfig {
    /// Full per-head query/key width: passed dims plus carried dims. 192.
    ///
    /// This — not [`Self::qk_nope_head_dim`] — is what the softmax scale is
    /// derived from, because the carried dims are in the dot product too.
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_carried_head_dim
    }

    /// `q_head_dim ** -0.5`, the shipped `self.scaling`.
    pub fn scaling(&self) -> f64 {
        (self.q_head_dim() as f64).powf(-0.5)
    }

    /// Width of one head's `kv_b_proj` output: the passed key dims followed by
    /// the value dims, in that order.
    pub fn kv_b_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.v_head_dim
    }

    /// Build from a parsed checkpoint config.
    ///
    /// Refuses a config that does not say NoPE. This is the structural half of
    /// the no-rotation claim: there is no rotated block for a config edit to
    /// select, so `mla_use_nope: false` is a load-time error rather than a
    /// silently different model.
    pub fn from_text_config(cfg: &K3TextConfig) -> Result<Self, String> {
        if !cfg.mla_use_nope {
            return Err(
                "mla_use_nope is false: this port implements Kimi K3's NoPE MLA only. \
                 There is no rotated variant here — a rotated MLA is a different block \
                 and would need its own implementation and its own oracle."
                    .to_string(),
            );
        }
        Ok(Self {
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_attention_heads,
            q_lora_rank: cfg.q_lora_rank,
            kv_lora_rank: cfg.kv_lora_rank,
            qk_nope_head_dim: cfg.qk_nope_head_dim,
            qk_carried_head_dim: cfg.qk_rope_head_dim,
            v_head_dim: cfg.v_head_dim,
            use_output_gate: cfg.mla_use_output_gate,
        })
    }
}

/// The eight weights of one MLA block, in the checkpoint's own `[out, in]`
/// layout (torch `nn.Linear.weight`), so that a transposition mistake is a
/// shape error rather than a plausible wrong answer.
#[derive(Debug, Clone)]
pub struct MlaWeights<B: Backend> {
    /// `[q_lora_rank, hidden]`.
    pub q_a_proj: Tensor<B, 2>,
    /// `[q_lora_rank]`.
    pub q_a_layernorm: Tensor<B, 1>,
    /// `[num_heads * q_head_dim, q_lora_rank]`.
    pub q_b_proj: Tensor<B, 2>,
    /// `[kv_lora_rank + qk_carried_head_dim, hidden]`.
    pub kv_a_proj_with_mqa: Tensor<B, 2>,
    /// `[kv_lora_rank]`.
    pub kv_a_layernorm: Tensor<B, 1>,
    /// `[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]`.
    pub kv_b_proj: Tensor<B, 2>,
    /// `[hidden, num_heads * v_head_dim]`.
    pub o_proj: Tensor<B, 2>,
    /// `[num_heads * v_head_dim, hidden]`, present iff the output gate is on.
    pub g_proj: Option<Tensor<B, 2>>,
}

/// The latent-expanded key/value cache of one MLA layer.
///
/// K3's shipped `KimiDynamicCache` stores the **expanded** `key_states`
/// `[B, H, S, 192]` and `value_states` `[B, H, S, 128]`, not the 512-wide
/// latent — `past_key_values.update(key_states, value_states, layer_idx)` is
/// called after `kv_b_proj` has already run. That is 6× the bytes a latent
/// cache would need, and it is what the oracle captured, so it is what this
/// reproduces.
#[derive(Debug, Clone, Default)]
pub struct MlaKvCache<B: Backend> {
    key: Option<Tensor<B, 4>>,
    value: Option<Tensor<B, 4>>,
}

impl<B: Backend> MlaKvCache<B> {
    pub fn new() -> Self {
        Self { key: None, value: None }
    }

    /// Tokens already cached.
    pub fn len(&self) -> usize {
        self.key.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append this step's keys and values and return the full history.
    ///
    /// New tokens go **after** the cached ones; the returned tensors are what
    /// attention runs over.
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k = match self.key.take() {
            Some(prev) => Tensor::cat(vec![prev, k], 2),
            None => k,
        };
        let v = match self.value.take() {
            Some(prev) => Tensor::cat(vec![prev, v], 2),
            None => v,
        };
        self.key = Some(k.clone());
        self.value = Some(v.clone());
        (k, v)
    }

    /// The cached keys `[B, H, S, q_head_dim]`, if any.
    pub fn key(&self) -> Option<Tensor<B, 4>> {
        self.key.clone()
    }

    /// The cached values `[B, H, S, v_head_dim]`, if any.
    pub fn value(&self) -> Option<Tensor<B, 4>> {
        self.value.clone()
    }
}

/// Every intermediate the shipped block materialises, in the shipped layout.
///
/// Returned by [`MlaBlock::forward`] so a gate can compare sub-block by
/// sub-block instead of only end to end — an end-to-end match can hide two
/// compensating errors, and a single number cannot say *which* projection is
/// mis-loaded. Burn tensors are handles, so carrying them costs a refcount.
#[derive(Debug, Clone)]
pub struct MlaTrace<B: Backend> {
    /// `[B, T, q_lora_rank]`.
    pub q_a_proj_out: Tensor<B, 3>,
    /// `[B, T, q_lora_rank]`.
    pub q_a_layernorm_out: Tensor<B, 3>,
    /// `[B, T, num_heads * q_head_dim]`.
    pub q_b_proj_out: Tensor<B, 3>,
    /// `[B, T, kv_lora_rank + qk_carried_head_dim]`.
    pub kv_a_proj_out: Tensor<B, 3>,
    /// `[B, T, kv_lora_rank]` — the slice that feeds `kv_a_layernorm`.
    pub kv_a_layernorm_in: Tensor<B, 3>,
    /// `[B, T, kv_lora_rank]`.
    pub kv_a_layernorm_out: Tensor<B, 3>,
    /// `[B, T, num_heads * (qk_nope_head_dim + v_head_dim)]`.
    pub kv_b_proj_out: Tensor<B, 3>,
    /// `[B, H, T_kv, q_head_dim]` — including anything the cache replayed.
    pub key_states: Tensor<B, 4>,
    /// `[B, H, T_q, q_head_dim]`.
    pub query_states: Tensor<B, 4>,
    /// `[B, H, T_kv, v_head_dim]`.
    pub value_states: Tensor<B, 4>,
    /// `[B, H, T_q, T_kv]` — `q·kᵀ · scaling + mask`, before the softmax.
    pub scores: Tensor<B, 4>,
    /// `[B, H, T_q, T_kv]` — the softmax, at the softmax's own precision.
    pub probs_precast: Tensor<B, 4>,
    /// `[B, H, T_q, T_kv]` — the softmax cast back to the activation dtype.
    pub probs: Tensor<B, 4>,
    /// `[B, T_q, H, v_head_dim]` — `probs · v`, heads still separate.
    pub attn_out_heads: Tensor<B, 4>,
    /// `[B, T_q, H * v_head_dim]` — `g_proj(hidden)`, before the sigmoid.
    pub g_proj_out: Option<Tensor<B, 3>>,
    /// `[B, T_q, H * v_head_dim]` — what `o_proj` is applied to.
    pub o_proj_in: Tensor<B, 3>,
    /// `[B, T_q, hidden]` — the block output.
    pub out: Tensor<B, 3>,

    /// `[B, H, T_q, qk_carried_head_dim]` — the carried lane as `q_b_proj`
    /// produced it, before assembly. Kept for
    /// [`Self::assert_carried_verbatim`].
    pub q_carried_source: Tensor<B, 4>,
    /// `[B, H, T_kv, qk_carried_head_dim]` — the carried lane as
    /// `kv_a_proj_with_mqa` produced it and broadcast over heads, before
    /// assembly.
    pub k_carried_source: Tensor<B, 4>,
    /// Width of the carried lane, so the assertion can slice without the config.
    pub carried: usize,
}

impl<B: Backend> MlaTrace<B> {
    /// Assert that the carried lanes came through **verbatim**.
    ///
    /// This is the no-rotation property stated as an executable claim rather
    /// than a comment: the last `carried` dims of the assembled `query_states`
    /// and `key_states` must be *bit-identical* to the projection outputs they
    /// were sliced from. A rotation, a scaling, a reordering, a sign flip or a
    /// stale clone between projection and assembly all fail here, on the first
    /// differing element.
    ///
    /// Returns the number of elements compared, so a caller can report that it
    /// compared something rather than nothing.
    pub fn assert_carried_verbatim(&self) -> usize {
        let qd = self.query_states.dims();
        let kd = self.key_states.dims();
        // With a cache, `key_states` is the whole history while the source is
        // only this step's tokens, so the comparison takes the tail. That is
        // deliberately the assembled tensor attention actually reads, not the
        // pre-cache one: it asserts the carried lane is verbatim *where it is
        // used*, cache round trip included.
        let t_new = self.k_carried_source.dims()[2];
        assert!(
            t_new > 0 && t_new <= kd[2],
            "carried source has {} tokens, key_states has {}",
            t_new,
            kd[2]
        );
        let q_asm = self
            .query_states
            .clone()
            .slice([0..qd[0], 0..qd[1], 0..qd[2], (qd[3] - self.carried)..qd[3]]);
        let k_asm = self
            .key_states
            .clone()
            .slice([0..kd[0], 0..kd[1], (kd[2] - t_new)..kd[2], (kd[3] - self.carried)..kd[3]]);
        let n = bit_identical(&q_asm, &self.q_carried_source, "query_states carried lane")
            + bit_identical(&k_asm, &self.k_carried_source, "key_states carried lane");
        assert!(n > 0, "carried-lane assertion compared zero elements");
        n
    }

    /// Assert the carried lane of `key_states` is identical across all heads,
    /// and that the passed lane is *not* — the broadcast is real, and the two
    /// halves are genuinely different tensors.
    ///
    /// The first half is the shipped `k_rot.expand(...)`: one 64-wide vector
    /// per token, shared by all 96 heads (MLA's multi-query part). The second
    /// half is the non-degeneracy check that keeps the first from being
    /// vacuous — if every head were identical everywhere, "identical across
    /// heads" would prove nothing.
    pub fn assert_carried_is_broadcast(&self) -> usize {
        let [b, h, t, d] = self.key_states.dims();
        assert!(h > 1, "broadcast check needs more than one head, got {}", h);
        let head0 = |from: usize, to: usize| {
            self.key_states
                .clone()
                .slice([0..b, 0..1, 0..t, from..to])
                .repeat_dim(1, h)
        };
        let all = |from: usize, to: usize| self.key_states.clone().slice([0..b, 0..h, 0..t, from..to]);

        let carried_from = d - self.carried;
        let n = bit_identical(
            &all(carried_from, d),
            &head0(carried_from, d),
            "key_states carried lane across heads",
        );
        let passed = to_f64(&all(0, carried_from));
        let passed0 = to_f64(&head0(0, carried_from));
        assert!(
            passed.iter().zip(&passed0).any(|(a, b)| a != b),
            "key_states PASSED lane is identical across all {} heads — kv_b_proj is \
             producing one head's worth of key and broadcasting it, which is wrong, \
             and it would make the carried-lane broadcast check vacuous",
            h
        );
        n
    }
}

/// What [`MlaBlock::attend`] produces: the attention core's four intermediates.
#[derive(Debug, Clone)]
pub struct AttnParts<B: Backend> {
    /// `[B, H, T_q, T_kv]` — `q·kᵀ · scaling + mask`, before the softmax.
    pub scores: Tensor<B, 4>,
    /// `[B, H, T_q, T_kv]` — the softmax, at the softmax's own precision.
    pub probs_precast: Tensor<B, 4>,
    /// `[B, H, T_q, T_kv]` — the softmax cast back to the activation dtype.
    pub probs: Tensor<B, 4>,
    /// `[B, T_q, H, v_head_dim]` — `probs · v`, heads still separate.
    pub attn_out_heads: Tensor<B, 4>,
}

/// One MLA block: the eight weights, the shapes, and nothing else.
///
/// Note what is absent: no rotary table, no position offset used for anything
/// but slicing the mask, no `Option<Rope>`. The block cannot rotate because
/// there is nothing in it to rotate with.
#[derive(Debug, Clone)]
pub struct MlaBlock<B: Backend> {
    pub cfg: MlaConfig,
    pub w: MlaWeights<B>,
    /// Where this block rounds. See [`Precision`].
    pub precision: Precision,
}

impl<B: Backend> MlaBlock<B> {
    pub fn new(cfg: MlaConfig, w: MlaWeights<B>, precision: Precision) -> Self {
        assert_eq!(
            w.g_proj.is_some(),
            cfg.use_output_gate,
            "g_proj presence ({}) disagrees with config.use_output_gate ({})",
            w.g_proj.is_some(),
            cfg.use_output_gate
        );
        Self { cfg, w, precision }
    }

    // ---- the block, one shipped operation per method ---------------------
    //
    // `forward` below is exactly the composition of these. They are public and
    // individually callable so a gate can drive each one from the *oracle's*
    // captured input for that boundary, instead of only from the previous
    // method's output. That difference is not cosmetic: bfloat16 rounding
    // compounds, so a cascade run diverges from a reference bfloat16 run by a
    // few ULP by the end even when every single step is right, and a cascade-
    // only comparison therefore cannot be tight enough to see a real one-ULP
    // mistake. Driven per operation from the reference's own inputs, each step
    // is comparable to one ULP. Splitting them out is what makes that possible;
    // `forward` calling them is what keeps the two from drifting apart.

    /// `q_a_proj`: hidden → the query latent.
    pub fn q_a_proj(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.precision.round(linear(hidden, &self.w.q_a_proj))
    }

    /// `q_a_layernorm`, at MLA's own epsilon. See [`LORA_NORM_EPS`].
    pub fn q_a_norm(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        rms_norm(x, &self.w.q_a_layernorm, LORA_NORM_EPS, self.precision)
    }

    /// `q_b_proj`: the query latent → all heads, `q_head_dim` wide each.
    pub fn q_b_proj(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.precision.round(linear(x, &self.w.q_b_proj))
    }

    /// `kv_a_proj_with_mqa`: hidden → `[latent | carried]`, one tensor.
    pub fn kv_a_proj(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.precision.round(linear(hidden, &self.w.kv_a_proj_with_mqa))
    }

    /// The first `kv_lora_rank` dims of `kv_a_proj_with_mqa`'s output — the
    /// part that goes through `kv_a_layernorm` and `kv_b_proj`.
    pub fn kv_latent(&self, kv_a_out: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, w] = kv_a_out.dims();
        assert_eq!(w, self.cfg.kv_lora_rank + self.cfg.qk_carried_head_dim);
        kv_a_out.slice([0..b, 0..t, 0..self.cfg.kv_lora_rank])
    }

    /// The trailing `qk_carried_head_dim` dims — the lane that is carried
    /// through unrotated and shared by every head.
    pub fn kv_carried(&self, kv_a_out: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, w] = kv_a_out.dims();
        assert_eq!(w, self.cfg.kv_lora_rank + self.cfg.qk_carried_head_dim);
        kv_a_out.slice([0..b, 0..t, self.cfg.kv_lora_rank..w])
    }

    /// `kv_a_layernorm`, at MLA's own epsilon. See [`LORA_NORM_EPS`].
    pub fn kv_a_norm(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        rms_norm(x, &self.w.kv_a_layernorm, LORA_NORM_EPS, self.precision)
    }

    /// `kv_b_proj`: the KV latent → all heads, `qk_nope_head_dim + v_head_dim`
    /// wide each, key first and value second.
    pub fn kv_b_proj(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.precision.round(linear(x, &self.w.kv_b_proj))
    }

    /// `q_b_proj`'s output → `query_states [B,H,T,q_head_dim]`, and the carried
    /// lane it was built from.
    ///
    /// The concatenation is the only thing that happens to the carried lane.
    pub fn assemble_query(&self, q_b_out: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let c = &self.cfg;
        let (h, qh, nope) = (c.num_heads, c.q_head_dim(), c.qk_nope_head_dim);
        let [b, t, w] = q_b_out.dims();
        assert_eq!(w, h * qh, "q_b_proj output width {} != {} heads x {}", w, h, qh);
        let q = q_b_out.reshape([b, t, h, qh]).swap_dims(1, 2);
        let pass = q.clone().slice([0..b, 0..h, 0..t, 0..nope]);
        let carried = q.slice([0..b, 0..h, 0..t, nope..qh]);
        (Tensor::cat(vec![pass, carried.clone()], 3), carried)
    }

    /// `kv_b_proj`'s output and the carried lane → `key_states`,
    /// `value_states`, and the head-broadcast carried lane.
    ///
    /// `kv_b_proj`'s per-head output is key-then-value, in that order; the two
    /// halves are not interchangeable. The carried lane is one vector per
    /// token, shared by every head — MLA's multi-query part — and is broadcast,
    /// never re-projected.
    pub fn assemble_kv(
        &self,
        kv_b_out: Tensor<B, 3>,
        kv_carried: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let c = &self.cfg;
        let (h, nope, dv, carried) =
            (c.num_heads, c.qk_nope_head_dim, c.v_head_dim, c.qk_carried_head_dim);
        let kvh = c.kv_b_head_dim();
        let [b, t, w] = kv_b_out.dims();
        assert_eq!(w, h * kvh, "kv_b_proj output width {} != {} heads x {}", w, h, kvh);
        assert_eq!(kv_carried.dims(), [b, t, carried], "carried lane shape");
        let kv4 = kv_b_out.reshape([b, t, h, kvh]).swap_dims(1, 2);
        let k_pass = kv4.clone().slice([0..b, 0..h, 0..t, 0..nope]);
        let value = kv4.slice([0..b, 0..h, 0..t, nope..kvh]);
        let k_carried = kv_carried.reshape([b, 1, t, carried]).repeat_dim(1, h);
        let key = Tensor::cat(vec![k_pass, k_carried.clone()], 3);
        assert_eq!(value.dims()[3], dv);
        (key, value, k_carried)
    }

    /// `q·kᵀ · scaling + mask` — the pre-softmax scores, `[B,H,T_q,T_kv]`.
    ///
    /// The scale is `q_head_dim^-0.5` over the **full** 192-wide head, carried
    /// lane included, not `qk_nope_head_dim^-0.5`.
    pub fn attn_scores(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let p = self.precision;
        let [b, h, t_q, qh] = q.dims();
        let [_, _, t_kv, _] = k.dims();
        assert_eq!(qh, self.cfg.q_head_dim(), "query head width");
        assert_eq!(k.dims()[3], qh, "key head width != query head width");
        assert!(b > 0 && h > 0 && t_q > 0 && t_kv > 0, "empty attention");

        let scores = p.round(q.matmul(k.swap_dims(2, 3)));
        let scores = p.round(scores.mul_scalar(self.cfg.scaling()));
        match &mask {
            Some(m) => {
                let md = m.dims();
                assert_eq!(
                    [md[0], md[2], md[3]],
                    [b, t_q, t_kv],
                    "mask shape {:?} does not match [B={}, .., T_q={}, T_kv={}]",
                    md,
                    b,
                    t_q,
                    t_kv
                );
                p.round(scores + m.clone().repeat_dim(1, h))
            }
            None => scores,
        }
    }

    /// The fp32 island: the softmax runs at the backend's precision whatever
    /// the activation dtype is, and only then is cast back. Returns
    /// `(before the cast, after it)`.
    pub fn attn_probs(&self, scores: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let precast = softmax_dim(scores, 3);
        let cast = self.precision.round(precast.clone());
        (precast, cast)
    }

    /// `probs · v`, transposed back to `[B, T_q, H, v_head_dim]`.
    pub fn attn_apply(&self, probs: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
        assert_eq!(probs.dims()[3], v.dims()[2], "probs width != value length");
        self.precision.round(probs.matmul(v)).swap_dims(1, 2)
    }

    /// The attention core: `softmax(q·kᵀ · scaling + mask) · v`.
    ///
    /// The composition of [`Self::attn_scores`], [`Self::attn_probs`] and
    /// [`Self::attn_apply`], split so each can be driven from a reference's own
    /// captured tensor rather than from the previous step's output.
    ///
    /// `q` is `[B,H,T_q,q_head_dim]`, `k` `[B,H,T_kv,q_head_dim]`, `v`
    /// `[B,H,T_kv,v_head_dim]`, `mask` `[B,1,T_q,T_kv]`.
    pub fn attend(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<Tensor<B, 4>>,
    ) -> AttnParts<B> {
        assert_eq!(v.dims()[2], k.dims()[2], "value length != key length");
        let scores = self.attn_scores(q, k, mask);
        let (probs_precast, probs) = self.attn_probs(scores.clone());
        let attn_out_heads = self.attn_apply(probs.clone(), v);
        AttnParts { scores, probs_precast, probs, attn_out_heads }
    }

    /// `g_proj`: hidden → the pre-sigmoid output gate. `None` when the block
    /// has no output gate.
    pub fn g_proj(&self, hidden: Tensor<B, 3>) -> Option<Tensor<B, 3>> {
        self.w
            .g_proj
            .as_ref()
            .map(|g| self.precision.round(linear(hidden, g)))
    }

    /// Flatten the per-head attention output and apply the sigmoid gate to it,
    /// producing what `o_proj` is called on.
    ///
    /// The gate multiplies the attention output **before** `o_proj`, in the
    /// `num_heads * v_head_dim` space, not the residual-stream space.
    pub fn apply_output_gate(
        &self,
        attn_out_heads: Tensor<B, 4>,
        g_proj_out: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let p = self.precision;
        let [b, t, h, dv] = attn_out_heads.dims();
        let flat = attn_out_heads.reshape([b, t, h * dv]);
        match (g_proj_out, self.cfg.use_output_gate) {
            (Some(g), true) => {
                assert_eq!(g.dims(), [b, t, h * dv], "g_proj output shape");
                p.round(flat * p.round(sigmoid(g.clone())))
            }
            (None, false) => flat,
            (g, gate) => panic!("output gate present: {} but config says {}", g.is_some(), gate),
        }
    }

    /// `o_proj`: back to the residual stream.
    pub fn o_proj(&self, o_in: Tensor<B, 3>) -> Tensor<B, 3> {
        self.precision.round(linear(o_in, &self.w.o_proj))
    }

    /// Run the block.
    ///
    /// `hidden` is `[B, T_new, hidden_size]` — the **new** tokens only when a
    /// cache is supplied. `mask` is the additive attention mask
    /// `[B, 1, T_new, T_kv]`, broadcast over heads, with [`mask_neg`] at
    /// disallowed positions. Returns the full [`MlaTrace`]; the block output is
    /// `trace.out`.
    ///
    /// This is the composition of the methods above and nothing else, so a gate
    /// that drives them individually is driving this.
    ///
    /// The carried-lane assertions run on every call: they cost one comparison
    /// over `B·H·T·64` elements against six large GEMMs, and they are the whole
    /// point of the module.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        mask: Option<Tensor<B, 4>>,
        cache: Option<&mut MlaKvCache<B>>,
    ) -> MlaTrace<B> {
        let c = &self.cfg;
        let [b, t, dh] = hidden.dims();
        assert_eq!(dh, c.hidden_size, "hidden width {} != config {}", dh, c.hidden_size);
        assert!(b > 0 && t > 0, "empty input: [{}, {}, {}]", b, t, dh);

        let q_a_proj_out = self.q_a_proj(hidden.clone());
        let q_a_layernorm_out = self.q_a_norm(q_a_proj_out.clone());
        let q_b_proj_out = self.q_b_proj(q_a_layernorm_out.clone());
        let (query_states, q_carried_source) = self.assemble_query(q_b_proj_out.clone());

        let kv_a_proj_out = self.kv_a_proj(hidden.clone());
        let kv_a_layernorm_in = self.kv_latent(kv_a_proj_out.clone());
        let kv_carried_flat = self.kv_carried(kv_a_proj_out.clone());
        let kv_a_layernorm_out = self.kv_a_norm(kv_a_layernorm_in.clone());
        let kv_b_proj_out = self.kv_b_proj(kv_a_layernorm_out.clone());
        let (key_new, value_new, k_carried_source) =
            self.assemble_kv(kv_b_proj_out.clone(), kv_carried_flat);

        let (key_states, value_states) = match cache {
            Some(kv) => kv.update(key_new, value_new),
            None => (key_new, value_new),
        };
        assert_eq!(
            value_states.dims()[2],
            key_states.dims()[2],
            "key/value cache lengths disagree"
        );

        let a = self.attend(
            query_states.clone(),
            key_states.clone(),
            value_states.clone(),
            mask,
        );
        let g_proj_out = self.g_proj(hidden);
        let o_proj_in = self.apply_output_gate(a.attn_out_heads.clone(), g_proj_out.as_ref());
        let out = self.o_proj(o_proj_in.clone());

        let trace = MlaTrace {
            q_a_proj_out,
            q_a_layernorm_out,
            q_b_proj_out,
            kv_a_proj_out,
            kv_a_layernorm_in,
            kv_a_layernorm_out,
            kv_b_proj_out,
            key_states,
            query_states,
            value_states,
            scores: a.scores,
            probs_precast: a.probs_precast,
            probs: a.probs,
            attn_out_heads: a.attn_out_heads,
            g_proj_out,
            o_proj_in,
            out,
            q_carried_source,
            k_carried_source,
            carried: c.qk_carried_head_dim,
        };
        // Point 3 of the module docs. Not optional, not behind a flag.
        trace.assert_carried_verbatim();
        trace
    }

    /// The additive causal mask the shipped model builds:
    /// `0` where a query at position `offset + i` may attend to key `j`
    /// (`j <= offset + i`), [`mask_neg`] elsewhere. `[B, 1, T_q, T_kv]`.
    pub fn causal_mask(
        batch: usize,
        t_q: usize,
        t_kv: usize,
        offset: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        assert!(batch > 0 && t_q > 0 && t_kv > 0, "empty mask requested");
        assert!(
            offset + t_q <= t_kv,
            "offset {} + T_q {} exceeds T_kv {}",
            offset,
            t_q,
            t_kv
        );
        let neg = mask_neg();
        let mut v = Vec::with_capacity(batch * t_q * t_kv);
        for _ in 0..batch {
            for i in 0..t_q {
                for j in 0..t_kv {
                    v.push(if j <= offset + i { 0.0f64 } else { neg });
                }
            }
        }
        Tensor::from_data(
            TensorData::new(v, [batch, 1, t_q, t_kv]).convert::<B::FloatElem>(),
            device,
        )
    }
}

/// `y = x Wᵀ` for a `[out, in]` weight, over a `[B, T, in]` activation.
fn linear<B: Backend>(x: Tensor<B, 3>, w: &Tensor<B, 2>) -> Tensor<B, 3> {
    let [b, t, din] = x.dims();
    let [dout, win] = w.dims();
    assert_eq!(
        din, win,
        "linear: activation width {} != weight input width {} (weight is [out, in])",
        din, win
    );
    x.reshape([b * t, din])
        .matmul(w.clone().transpose())
        .reshape([b, t, dout])
}

/// `KimiRMSNorm`: normalise in the wide precision, cast back, *then* scale.
///
/// The cast-back-before-scale is load bearing in a bfloat16 run — `self.weight
/// * x.to(dtype)` rounds twice, once after the reciprocal-sqrt and once after
/// the weight multiply. Doing the multiply first and rounding once is a
/// different (and wrong) answer.
fn rms_norm<B: Backend>(
    x: Tensor<B, 3>,
    weight: &Tensor<B, 1>,
    eps: f64,
    p: Precision,
) -> Tensor<B, 3> {
    let [_b, _t, d] = x.dims();
    assert_eq!(weight.dims()[0], d, "rms_norm: weight width != activation width");
    let [b, t, _] = x.dims();
    // Delegated to the one shared transcription rather than repeated here.
    //
    // This function used to read `x * (ms + eps).sqrt().recip()`, and that is a
    // MEASURED accuracy bug, not a style point: `Tensor::recip` on the ndarray
    // backend dispatches to a SIMD *approximate* reciprocal, ~1.4e-3 relative
    // on aarch64 — ten bits, not twenty-four. The error lands on a scale shared
    // by every one of the row's 1536 (or 512) dimensions, so it survives the
    // reduction intact and re-rounds a fifth of the row. Measured against the
    // shipped bf16 `q_a_layernorm`, the `recip` form reproduced 81.3% of the
    // bits and the division reproduces 99.99%. `attn_res.rs`, `moe.rs` and
    // `ops.rs` each carry a comment warning about exactly this; this module had
    // the bug. That is what four copies of a two-line function costs, and it is
    // why there is now one.
    crate::models::k3::ops::rms_norm_with(x.reshape([b * t, d]), weight, eps, |v| p.round(v))
        .reshape([b, t, d])
}

/// `F.softmax(x, dim)`: subtract the max along `dim`, exponentiate, divide by
/// the sum. Written out rather than taken from `burn::tensor::activation` so
/// the arithmetic that has to match `eager_attention_forward` is visible here
/// and cannot drift with a backend's choice of formulation.
///
/// The max subtraction is what keeps the shipped mask's `finfo(bf16).min`
/// finite: `exp(-3.4e38 - max)` underflows to exactly 0 instead of producing a
/// NaN, in every float width this runs in.
fn softmax_dim<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let max = x.clone().max_dim(dim);
    let e = (x - max).exp();
    let s = e.clone().sum_dim(dim);
    e / s
}

/// `torch.sigmoid`: `1 / (1 + exp(-x))`. Explicit for the same reason as
/// [`softmax_dim`]. Saturates to 1 and 0 rather than overflowing.
fn sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // A DIVISION, not `recip()` — see `rms_norm` above. The gate measured the
    // difference here too: with `recip()` the output gate reproduced 72.2% of
    // the shipped bf16 bits, with a division 99.9%.
    let d = x.neg().exp() + 1.0;
    d.clone().ones_like() / d
}

/// Host-readback of a tensor's elements as f64, in row-major order.
fn to_f64<B: Backend, const D: usize>(t: &Tensor<B, D>) -> Vec<f64> {
    t.clone()
        .into_data()
        .convert::<f64>()
        .into_vec()
        .expect("host readback")
}

/// Assert two tensors hold bit-identical values, returning the count compared.
fn bit_identical<B: Backend, const D: usize>(
    a: &Tensor<B, D>,
    b: &Tensor<B, D>,
    what: &str,
) -> usize {
    assert_eq!(a.dims(), b.dims(), "{}: shape mismatch", what);
    let av = to_f64(a);
    let bv = to_f64(b);
    assert!(!av.is_empty(), "{}: compared zero elements", what);
    for (i, (x, y)) in av.iter().zip(&bv).enumerate() {
        // Bitwise, not approximate: `!=` on floats is false for equal values
        // and true for any difference including a NaN on either side.
        assert!(
            x.to_bits() == y.to_bits(),
            "{}: element {} differs — assembled {:?} vs source {:?}. \
             Something was applied to the carried lane between projection and \
             assembly. Kimi K3's MLA is NoPE: nothing may be.",
            what,
            i,
            x,
            y
        );
    }
    av.len()
}

#[cfg(all(test, feature = "k3-mla"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TB = NdArray<f64>;

    fn tiny_cfg() -> MlaConfig {
        MlaConfig {
            hidden_size: 16,
            num_heads: 3,
            q_lora_rank: Some(8),
            kv_lora_rank: 6,
            qk_nope_head_dim: 4,
            qk_carried_head_dim: 2,
            v_head_dim: 4,
            use_output_gate: true,
        }
    }

    /// Deterministic pseudo-random fill — no rand dep, and the same weights
    /// every run so a failure is reproducible.
    fn fill(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    fn t2(r: usize, c: usize, seed: u64, dev: &burn_ndarray::NdArrayDevice) -> Tensor<TB, 2> {
        Tensor::from_data(TensorData::new(fill(r * c, seed), [r, c]), dev)
    }

    fn block(cfg: &MlaConfig, dev: &burn_ndarray::NdArrayDevice) -> MlaBlock<TB> {
        let h = cfg.num_heads;
        let w = MlaWeights::<TB> {
            q_a_proj: t2(cfg.q_lora_rank.unwrap(), cfg.hidden_size, 1, dev),
            q_a_layernorm: Tensor::from_data(
                TensorData::new(fill(cfg.q_lora_rank.unwrap(), 2), [cfg.q_lora_rank.unwrap()]),
                dev,
            ),
            q_b_proj: t2(h * cfg.q_head_dim(), cfg.q_lora_rank.unwrap(), 3, dev),
            kv_a_proj_with_mqa: t2(
                cfg.kv_lora_rank + cfg.qk_carried_head_dim,
                cfg.hidden_size,
                4,
                dev,
            ),
            kv_a_layernorm: Tensor::from_data(
                TensorData::new(fill(cfg.kv_lora_rank, 5), [cfg.kv_lora_rank]),
                dev,
            ),
            kv_b_proj: t2(h * cfg.kv_b_head_dim(), cfg.kv_lora_rank, 6, dev),
            o_proj: t2(cfg.hidden_size, h * cfg.v_head_dim, 7, dev),
            g_proj: Some(t2(h * cfg.v_head_dim, cfg.hidden_size, 8, dev)),
        };
        MlaBlock::new(cfg.clone(), w, Precision::Exact)
    }

    /// Point 3 of the module docs, on random weights and without a checkpoint:
    /// the carried lanes arrive at assembly bit-identical to their sources.
    #[test]
    fn carried_lanes_are_verbatim() {
        let dev = Default::default();
        let cfg = tiny_cfg();
        let blk = block(&cfg, &dev);
        let x: Tensor<TB, 3> =
            Tensor::from_data(TensorData::new(fill(2 * 5 * 16, 99), [2, 5, 16]), &dev);
        let mask = MlaBlock::<TB>::causal_mask(2, 5, 5, 0, &dev);
        let tr = blk.forward(x, Some(mask), None);
        let n = tr.assert_carried_verbatim();
        assert_eq!(n, 2 * 3 * 5 * 2 * 2, "expected both carried lanes compared");
        assert!(tr.assert_carried_is_broadcast() > 0);
    }

    /// The carried lane of `key_states` really is one vector per token shared by
    /// every head, and it is exactly `kv_a_proj_with_mqa`'s tail.
    #[test]
    fn carried_key_is_the_projection_tail_broadcast() {
        let dev = Default::default();
        let cfg = tiny_cfg();
        let blk = block(&cfg, &dev);
        let x: Tensor<TB, 3> =
            Tensor::from_data(TensorData::new(fill(1 * 4 * 16, 7), [1, 4, 16]), &dev);
        let tr = blk.forward(x, None, None);
        let tail = to_f64(&tr.kv_a_proj_out.clone().slice([0..1, 0..4, 6..8]));
        let [b, h, t, d] = tr.key_states.dims();
        let carried = to_f64(&tr.key_states.clone().slice([0..b, 0..h, 0..t, (d - 2)..d]));
        for head in 0..h {
            for tok in 0..t {
                for i in 0..2 {
                    assert_eq!(
                        carried[head * t * 2 + tok * 2 + i].to_bits(),
                        tail[tok * 2 + i].to_bits(),
                        "head {} token {} dim {}",
                        head,
                        tok,
                        i
                    );
                }
            }
        }
    }

    /// Prefill-then-continue over a cache reproduces a single full pass. The
    /// cache is a KV *history*, so this is an identity, not an approximation —
    /// in exact arithmetic the two differ only by GEMM shape.
    #[test]
    fn cache_continuation_matches_full_pass() {
        let dev = Default::default();
        let cfg = tiny_cfg();
        let blk = block(&cfg, &dev);
        let x: Tensor<TB, 3> =
            Tensor::from_data(TensorData::new(fill(1 * 6 * 16, 21), [1, 6, 16]), &dev);
        let full = blk.forward(
            x.clone(),
            Some(MlaBlock::<TB>::causal_mask(1, 6, 6, 0, &dev)),
            None,
        );
        let mut cache = MlaKvCache::new();
        let a = blk.forward(
            x.clone().slice([0..1, 0..4, 0..16]),
            Some(MlaBlock::<TB>::causal_mask(1, 4, 4, 0, &dev)),
            Some(&mut cache),
        );
        assert_eq!(cache.len(), 4);
        let b = blk.forward(
            x.slice([0..1, 4..6, 0..16]),
            Some(MlaBlock::<TB>::causal_mask(1, 2, 6, 4, &dev)),
            Some(&mut cache),
        );
        assert_eq!(cache.len(), 6);
        let joined = Tensor::cat(vec![a.out, b.out], 1);
        let d = to_f64(&(joined - full.out).abs().max());
        assert!(d[0] < 1e-12, "cache continuation differs by {}", d[0]);
    }
}
