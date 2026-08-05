//! Kimi Delta Attention — the decay gate, the gated delta-rule recurrence, and
//! the kernel-4 short convolution that feeds it.
//!
//! ## What KDA is in this model
//!
//! Kimi K3 is 93 layers, of which **69 are KDA** and 24 are full MLA attention
//! at every fourth position. The MLA layers carry no positional encoding at all
//! (`mla_use_nope: true`, `rotary_emb is None`), so *every* piece of position
//! information in the model arrives through the KDA recurrence and the short
//! convolutions in front of it. That makes this primitive load-bearing in a way
//! a linear-attention layer usually is not: get the decay wrong and the model
//! does not merely lose accuracy, it loses its only sense of order.
//!
//! ## The gate, and what `gate_lower_bound = -5.0` does to it
//!
//! `linear_attn_config.gate_lower_bound` is `-5.0`, which selects fla's bounded
//! branch:
//!
//! ```text
//! g = -5 · sigmoid(exp(A_log) · (g_raw + dt_bias))          # Kimi K3
//! g = -exp(A_log) · softplus(g_raw + dt_bias)               # the other branch
//! ```
//!
//! Two consequences worth stating out loud, because both are easy to carry a
//! wrong intuition about:
//!
//! 1. **Retention is floored.** The per-channel retention is `exp(g)`, and with
//!    `g ∈ (-5, 0)` that is `exp(g) ∈ (e⁻⁵, 1) = (0.006738…, 1)`. A channel can
//!    never be fully erased in a single step — the strongest possible one-token
//!    forget keeps 0.67% of the state, and driving a channel to zero takes
//!    several tokens of sustained maximum decay. The floor is *structural*, not
//!    a clamp: it falls out of the sigmoid's range, so nothing ever tests for
//!    it. (Under the unbounded softplus branch `g` runs to -∞ and one token
//!    genuinely can wipe a channel.)
//! 2. **`A_log` stops being a decay rate.** In the Mamba-style softplus form,
//!    `exp(A_log)` multiplies the *result* and so sets how fast a channel
//!    forgets. Here it multiplies the sigmoid's *argument*, so it sets how
//!    sharply the gate switches between "retain almost everything" and "forget
//!    as hard as the floor allows" as `g_raw` crosses `-dt_bias`. It is a
//!    per-head **switching temperature**, not a rate: a large `exp(A_log)` head
//!    is nearly a hard binary gate, a small one interpolates smoothly. Both
//!    heads bottom out at exactly the same retention.
//!
//! Kimi's `use_full_rank_gate: true` refers to the *output* gate ([`rms_norm_gated`],
//! a single `g_proj`), not to this one. The decay gate stays low-rank: `g_raw`
//! comes out of an `f_a_proj`/`f_b_proj` pair with inner rank 128.
//!
//! ## The recurrence
//!
//! Per token, per value head, with state `S: [K, V]` (fla's `[N, HV, K, V]`):
//!
//! ```text
//! S  = S · exp(g)[:, None]        # per-channel (diagonal) decay along K, FIRST
//! d  = v − kᵀS                    # the delta rule's prediction error
//! d  = d · sigmoid(beta)
//! S  = S + k ⊗ d
//! o  = qᵀS                        # the POST-update state, same token
//! ```
//!
//! `q`/`k` are L2-normalized as `x / sqrt(Σx² + 1e-6)` — a plain sum with the
//! epsilon *inside* the sqrt, **not** the usual `rsqrt(mean(x²) + eps)` RMS
//! form. Confusing the two is a silent `sqrt(K)` ≈ 11.3× error at K=128. The
//! `K^-0.5` scale is applied to `q` only, and *after* the normalization.
//!
//! `beta` arrives as raw logits and is sigmoided here
//! (`use_beta_sigmoid_in_kernel=True`); Kimi does not set `allow_neg_eigval`,
//! so there is no `2·sigmoid` doubling.
//!
//! ## O(1) in sequence length
//!
//! [`KdaState`] holds one `Box<[E]>` of exactly `HV·K·V` elements and
//! [`ShortConvState`] one of exactly `D·W`, both sized from [`KdaConfig`] at
//! construction. The claim is carried by the *type*, not by a comment: a boxed
//! slice has no `push`, `extend` or `resize`, so no code path in this module —
//! or in any caller — can make the state grow with `T`. [`Kda::step`] consumes
//! one token, writes into a caller-provided output slice, and allocates
//! nothing (its working buffers live in a separately-owned [`KdaScratch`], also
//! fixed-size). The `state_is_o1_in_sequence_length` test below drives 1, 16
//! and 128 tokens and asserts the state's byte size is identical.
//!
//! ## Precision
//!
//! Everything is generic over [`Elem`] so the same code runs in f32 (what
//! inference uses) and f64 (what the parity gate uses). That separation is the
//! point: an algebraic slip in the decay term shows as ~1e-3 against a float64
//! oracle but hides inside fp32 rounding noise, and an f64 run agreeing with
//! `flash-linear-attention`'s own float64 output to machine epsilon is a much
//! sharper statement than an f32 run agreeing to 1e-7. See the
//! `kimi_kda_gate` binary.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// The float widths this port runs in.
///
/// Deliberately tiny — only the operations the recurrence, the gate and the
/// convolution actually perform, so that `f32` and `f64` execute *the same*
/// arithmetic in the same order and any difference between them is rounding
/// alone.
pub trait Elem:
    Copy
    + std::fmt::Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn exp(self) -> Self;
    fn ln_1p(self) -> Self;
    fn sqrt(self) -> Self;
}

macro_rules! impl_elem {
    ($t:ty) => {
        impl Elem for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            #[inline]
            fn from_f64(x: f64) -> Self {
                x as Self
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            #[inline]
            fn ln_1p(self) -> Self {
                <$t>::ln_1p(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
        }
    };
}

impl_elem!(f32);
impl_elem!(f64);

/// Logistic sigmoid, branched so neither tail overflows: the naive
/// `1/(1+exp(-x))` returns `inf` inside the denominator for `x ≲ -88` in f32,
/// and the gate sweep runs the argument out to ±3.2e4.
#[inline]
pub fn sigmoid<E: Elem>(x: E) -> E {
    if x >= E::ZERO {
        E::ONE / (E::ONE + (-x).exp())
    } else {
        let e = x.exp();
        e / (E::ONE + e)
    }
}

/// `log(1 + exp(x))`, linearized above 20.
///
/// The threshold is not a free choice: it is torch's `F.softplus` default
/// (`beta=1, threshold=20`), which is what fla's `naive_kda_gate` calls, and it
/// is also what fla's Triton kernel does — its generated PTX opens with
/// `setp.gt.f32 p, $in, 20.` (`fla/ops/utils/softplus.py`). Both
/// implementations that actually run this model agree on 20, so the port does
/// too. (This matters only for the unbounded branch, which Kimi K3 does not
/// take; see the `kimi_kda_gate` binary for a float64 oracle array that was
/// transcribed with 30 instead and the 5.14e-08 that costs.)
#[inline]
pub fn softplus<E: Elem>(x: E) -> E {
    if x > E::from_f64(20.0) {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `x · sigmoid(x)` — the short convolution's activation.
#[inline]
pub fn silu<E: Elem>(x: E) -> E {
    x * sigmoid(x)
}

/// The KDA decay gate, in log space: the `g` that the recurrence exponentiates.
///
/// `lower_bound` selects the branch — `Some(-5.0)` is Kimi K3's bounded gate
/// (see the module docs for what the bound does), `None` the unbounded
/// Mamba-style softplus form. Both are here because the difference between
/// them is enormous and silent: at a strongly-negative pre-activation the
/// bounded branch saturates at `g = -5` while the softplus branch keeps
/// falling, so a port that takes the wrong branch produces a state that decays
/// to zero instead of holding a floor, with outputs that still look plausible.
#[inline]
pub fn decay_gate<E: Elem>(a_log: E, g_raw: E, dt_bias: E, lower_bound: Option<E>) -> E {
    let z = g_raw + dt_bias;
    match lower_bound {
        Some(lb) => lb * sigmoid(a_log.exp() * z),
        None => -(a_log.exp() * softplus(z)),
    }
}

/// `x / sqrt(Σx² + eps)` over one head's channels — the fused q/k
/// normalization. Note the plain sum and the epsilon inside the sqrt; this is
/// not RMS norm (see the module docs).
#[inline]
pub fn l2_normalize<E: Elem>(x: &[E], eps: E, out: &mut [E]) {
    debug_assert_eq!(x.len(), out.len());
    let mut ss = E::ZERO;
    for &xi in x {
        ss += xi * xi;
    }
    let inv = E::ONE / (ss + eps).sqrt();
    for (o, &xi) in out.iter_mut().zip(x) {
        *o = xi * inv;
    }
}

/// Shape and flags of one KDA layer. `H == HV` in Kimi K3 (no grouped-query
/// sharing inside the linear-attention layers), which this port requires: fla
/// supports `HV % H == 0` by repeat-interleaving the keys, but that path is not
/// exercised by this model and is not gated by any oracle vector, so it is
/// rejected rather than written blind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KdaConfig {
    /// Number of heads, `H == HV`.
    pub num_heads: usize,
    /// Key/query channels per head.
    pub head_k_dim: usize,
    /// Value channels per head.
    pub head_v_dim: usize,
    /// Short-convolution kernel width.
    pub conv_kernel: usize,
    /// `Some(-5.0)` for Kimi K3; `None` selects the unbounded softplus gate.
    pub gate_lower_bound: Option<f64>,
    /// Epsilon *inside* the q/k L2-norm's sqrt.
    pub l2norm_eps: f64,
}

impl KdaConfig {
    /// Kimi K3's `linear_attn_config`: 96 heads, `head_dim` 128,
    /// `short_conv_kernel_size` 4, `gate_lower_bound` -5.0.
    pub fn kimi_k3() -> Self {
        Self {
            num_heads: 96,
            head_k_dim: 128,
            head_v_dim: 128,
            conv_kernel: 4,
            gate_lower_bound: Some(-5.0),
            l2norm_eps: 1e-6,
        }
    }

    /// `K^-0.5`, applied to `q` *after* the L2 normalization.
    pub fn q_scale(&self) -> f64 {
        (self.head_k_dim as f64).powf(-0.5)
    }

    /// Elements in one sequence's recurrent state: `HV·K·V`. Note the absence
    /// of any `T`.
    pub fn state_elems(&self) -> usize {
        self.num_heads * self.head_k_dim * self.head_v_dim
    }
}

/// The learned per-head/per-channel gate parameters.
#[derive(Debug, Clone)]
pub struct KdaParams<E> {
    /// `[HV]` — exponentiated into the sigmoid's slope, not into a rate.
    pub a_log: Box<[E]>,
    /// `[HV·K]` — the flat `[HV*K]` checkpoint parameter viewed as `[HV, K]`,
    /// so the bias is per (head, channel).
    pub dt_bias: Box<[E]>,
}

impl<E: Elem> KdaParams<E> {
    pub fn new(cfg: &KdaConfig, a_log: &[E], dt_bias: &[E]) -> Self {
        assert_eq!(a_log.len(), cfg.num_heads, "A_log must be [HV]");
        assert_eq!(
            dt_bias.len(),
            cfg.num_heads * cfg.head_k_dim,
            "dt_bias must be [HV*K]"
        );
        Self {
            a_log: a_log.to_vec().into_boxed_slice(),
            dt_bias: dt_bias.to_vec().into_boxed_slice(),
        }
    }
}

/// One sequence's recurrent state, `[HV][K][V]` in C order — fla's default
/// `[N, HV, K, V]` layout.
///
/// The backing store is a `Box<[E]>`: fixed at construction from
/// [`KdaConfig::state_elems`], with no API that could grow it. That is the
/// O(1)-in-`T` guarantee, expressed in the type.
#[derive(Debug, Clone)]
pub struct KdaState<E> {
    s: Box<[E]>,
}

impl<E: Elem> KdaState<E> {
    /// A fresh sequence: `S = 0`.
    pub fn zeros(cfg: &KdaConfig) -> Self {
        Self {
            s: vec![E::ZERO; cfg.state_elems()].into_boxed_slice(),
        }
    }

    /// Adopt a cached state in the `[HV][K][V]` layout.
    pub fn from_kv(cfg: &KdaConfig, kv: &[E]) -> Self {
        assert_eq!(kv.len(), cfg.state_elems(), "initial state must be [HV,K,V]");
        Self {
            s: kv.to_vec().into_boxed_slice(),
        }
    }

    /// Adopt a cached state in the **transposed** `[HV][V][K]` layout.
    ///
    /// Kimi passes `transpose_state_layout=True`, which in fla 0.5.2 is a
    /// deprecated alias for `state_v_first` — and that flag governs the layout
    /// of the *input* `initial_state` as well as the returned one, which fla's
    /// own docstring does not say. Feeding a `[K,V]` state to a v-first call
    /// changes the output by 56% relative, silently. Any cache crossing this
    /// boundary must come through here, not through [`from_kv`](Self::from_kv).
    pub fn from_vk(cfg: &KdaConfig, vk: &[E]) -> Self {
        assert_eq!(vk.len(), cfg.state_elems(), "initial state must be [HV,V,K]");
        let (k, v) = (cfg.head_k_dim, cfg.head_v_dim);
        let mut s = vec![E::ZERO; cfg.state_elems()].into_boxed_slice();
        for h in 0..cfg.num_heads {
            let base = h * k * v;
            for vv in 0..v {
                for kk in 0..k {
                    s[base + kk * v + vv] = vk[base + vv * k + kk];
                }
            }
        }
        Self { s }
    }

    /// The state in the `[HV][V][K]` layout a `state_v_first` consumer expects.
    pub fn to_vk(&self, cfg: &KdaConfig) -> Vec<E> {
        let (k, v) = (cfg.head_k_dim, cfg.head_v_dim);
        let mut out = vec![E::ZERO; self.s.len()];
        for h in 0..cfg.num_heads {
            let base = h * k * v;
            for kk in 0..k {
                for vv in 0..v {
                    out[base + vv * k + kk] = self.s[base + kk * v + vv];
                }
            }
        }
        out
    }

    /// The state in the `[HV][K][V]` layout, as stored.
    pub fn as_kv(&self) -> &[E] {
        &self.s
    }

    /// Element count — constant for the life of the sequence.
    pub fn elems(&self) -> usize {
        self.s.len()
    }

    /// Bytes occupied by the state proper (what the O(1) claim is about).
    pub fn byte_len(&self) -> usize {
        std::mem::size_of_val(&*self.s)
    }
}

/// Fixed-size working buffers for [`Kda::step`], owned by the caller so a
/// decode loop allocates exactly once. Not state: nothing here survives a step.
#[derive(Debug, Clone)]
pub struct KdaScratch<E> {
    qn: Box<[E]>,
    kn: Box<[E]>,
    d: Box<[E]>,
}

impl<E: Elem> KdaScratch<E> {
    pub fn new(cfg: &KdaConfig) -> Self {
        Self {
            qn: vec![E::ZERO; cfg.head_k_dim].into_boxed_slice(),
            kn: vec![E::ZERO; cfg.head_k_dim].into_boxed_slice(),
            d: vec![E::ZERO; cfg.head_v_dim].into_boxed_slice(),
        }
    }
}

/// One token's KDA inputs for one sequence, as they come off the projections —
/// *before* the L2 norm, the gate and the beta sigmoid, all three of which the
/// fla kernel fuses and this port therefore performs inside [`Kda::step`].
#[derive(Debug, Clone, Copy)]
pub struct KdaToken<'a, E> {
    /// `[H·K]`, un-normalized.
    pub q_raw: &'a [E],
    /// `[H·K]`, un-normalized.
    pub k_raw: &'a [E],
    /// `[HV·V]`.
    pub v: &'a [E],
    /// `[HV·K]`, pre-gate and pre-`dt_bias`.
    pub g_raw: &'a [E],
    /// `[HV]`, raw logits.
    pub beta_raw: &'a [E],
}

/// A KDA layer's recurrence: shape plus the learned gate parameters.
#[derive(Debug, Clone)]
pub struct Kda<E> {
    pub cfg: KdaConfig,
    pub params: KdaParams<E>,
}

impl<E: Elem> Kda<E> {
    pub fn new(cfg: KdaConfig, params: KdaParams<E>) -> Self {
        assert!(cfg.num_heads > 0 && cfg.head_k_dim > 0 && cfg.head_v_dim > 0);
        Self { cfg, params }
    }

    /// Advance one token. `out` receives `[HV·V]`.
    ///
    /// Two passes over the `K·V` state per head: the first decays it and
    /// accumulates the delta-rule error `d = v − kᵀS` against the already-decayed
    /// rows; the second applies the rank-1 update and reads the output off the
    /// updated rows. Fusing them this way is exact — `d` only ever needs the
    /// post-decay state and `o` only the post-update state — and halves the
    /// traffic over what a literal transcription of the five-line recurrence
    /// would do.
    pub fn step(
        &self,
        st: &mut KdaState<E>,
        scr: &mut KdaScratch<E>,
        tok: KdaToken<'_, E>,
        out: &mut [E],
    ) {
        let cfg = &self.cfg;
        let (h_n, k_n, v_n) = (cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim);
        debug_assert_eq!(st.s.len(), cfg.state_elems());
        assert_eq!(tok.q_raw.len(), h_n * k_n, "q must be [H,K]");
        assert_eq!(tok.k_raw.len(), h_n * k_n, "k must be [H,K]");
        assert_eq!(tok.v.len(), h_n * v_n, "v must be [HV,V]");
        assert_eq!(tok.g_raw.len(), h_n * k_n, "g_raw must be [HV,K]");
        assert_eq!(tok.beta_raw.len(), h_n, "beta must be [HV]");
        assert_eq!(out.len(), h_n * v_n, "out must be [HV,V]");

        let eps = E::from_f64(cfg.l2norm_eps);
        let scale = E::from_f64(cfg.q_scale());
        let lb = cfg.gate_lower_bound.map(E::from_f64);

        for h in 0..h_n {
            let kr = h * k_n;
            let vr = h * v_n;

            l2_normalize(&tok.q_raw[kr..kr + k_n], eps, &mut scr.qn);
            for q in scr.qn.iter_mut() {
                *q = *q * scale;
            }
            l2_normalize(&tok.k_raw[kr..kr + k_n], eps, &mut scr.kn);
            let beta = sigmoid(tok.beta_raw[h]);
            let a_log = self.params.a_log[h];

            let s = &mut st.s[h * k_n * v_n..(h + 1) * k_n * v_n];
            scr.d.copy_from_slice(&tok.v[vr..vr + v_n]);

            // Pass 1: per-channel decay, then d = v − kᵀS on the decayed state.
            for kk in 0..k_n {
                let g = decay_gate(a_log, tok.g_raw[kr + kk], self.params.dt_bias[kr + kk], lb);
                let retain = g.exp();
                let kv = scr.kn[kk];
                let row = &mut s[kk * v_n..(kk + 1) * v_n];
                for (r, d) in row.iter_mut().zip(scr.d.iter_mut()) {
                    *r = *r * retain;
                    *d = *d - kv * *r;
                }
            }

            for d in scr.d.iter_mut() {
                *d = *d * beta;
            }

            // Pass 2: S += k ⊗ d, and o = qᵀS off the updated rows.
            out[vr..vr + v_n].fill(E::ZERO);
            for kk in 0..k_n {
                let kv = scr.kn[kk];
                let qv = scr.qn[kk];
                let row = &mut s[kk * v_n..(kk + 1) * v_n];
                for ((r, &d), o) in row
                    .iter_mut()
                    .zip(scr.d.iter())
                    .zip(out[vr..vr + v_n].iter_mut())
                {
                    *r += kv * d;
                    *o += qv * *r;
                }
            }
        }
    }

    /// Drive a whole sequence for one batch element. `q_raw`/`k_raw`/`g_raw`
    /// are `[T][H·K]`, `v` is `[T][HV·V]`, `beta_raw` is `[T][HV]`, `out` is
    /// `[T][HV·V]`.
    ///
    /// A convenience over [`step`](Self::step), and nothing more: the loop
    /// keeps no per-token history, so prefill and decode are the same code and
    /// the state cost does not move.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        st: &mut KdaState<E>,
        scr: &mut KdaScratch<E>,
        t_len: usize,
        q_raw: &[E],
        k_raw: &[E],
        v: &[E],
        g_raw: &[E],
        beta_raw: &[E],
        out: &mut [E],
    ) {
        let (hk, hv) = (
            self.cfg.num_heads * self.cfg.head_k_dim,
            self.cfg.num_heads * self.cfg.head_v_dim,
        );
        for t in 0..t_len {
            self.step(
                st,
                scr,
                KdaToken {
                    q_raw: &q_raw[t * hk..(t + 1) * hk],
                    k_raw: &k_raw[t * hk..(t + 1) * hk],
                    v: &v[t * hv..(t + 1) * hv],
                    g_raw: &g_raw[t * hk..(t + 1) * hk],
                    beta_raw: &beta_raw[t * self.cfg.num_heads..(t + 1) * self.cfg.num_heads],
                },
                &mut out[t * hv..(t + 1) * hv],
            );
        }
    }
}

/// Depthwise causal convolution of width `W`, with a SiLU activation — the
/// short conv in front of KDA's q/k/v projections.
///
/// With the MLA layers running NoPE, these four-tap windows plus the recurrence
/// are the model's entire positional mechanism, which is why the port carries
/// the streaming form (a rolling window) rather than only the whole-sequence
/// one.
#[derive(Debug, Clone)]
pub struct ShortConv<E> {
    /// Channels.
    pub dim: usize,
    /// Kernel width.
    pub width: usize,
    /// `[D·W]` — the `nn.Conv1d` depthwise weight `[D, 1, W]`, flattened.
    pub weight: Box<[E]>,
}

/// The convolution's rolling window: `[D][W]`, **most recent LAST**.
///
/// The ordering was settled by measurement against fla's own cache, not by
/// reading: `|state − most_recent_LAST| = 1.19e-07` against `3.87` for the
/// reverse. A fresh state of zeros is exactly the causal zero-padding at the
/// start of a sequence.
///
/// Fixed at `D·W` elements, in a `Box<[E]>`, for the same reason as
/// [`KdaState`].
#[derive(Debug, Clone)]
pub struct ShortConvState<E> {
    buf: Box<[E]>,
}

impl<E: Elem> ShortConvState<E> {
    pub fn zeros(conv: &ShortConv<E>) -> Self {
        Self {
            buf: vec![E::ZERO; conv.dim * conv.width].into_boxed_slice(),
        }
    }

    /// The window in fla's `[D, W]` cache layout, most-recent-last.
    pub fn as_slice(&self) -> &[E] {
        &self.buf
    }

    pub fn byte_len(&self) -> usize {
        std::mem::size_of_val(&*self.buf)
    }
}

impl<E: Elem> ShortConv<E> {
    pub fn new(dim: usize, width: usize, weight: &[E]) -> Self {
        assert_eq!(weight.len(), dim * width, "conv weight must be [D, 1, W]");
        Self {
            dim,
            width,
            weight: weight.to_vec().into_boxed_slice(),
        }
    }

    /// Advance one token: shift the window, admit `x`, emit
    /// `silu(Σᵢ w[d,i]·window[d,i])`.
    pub fn step(&self, st: &mut ShortConvState<E>, x: &[E], out: &mut [E]) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(out.len(), self.dim);
        let w = self.width;
        for d in 0..self.dim {
            let win = &mut st.buf[d * w..(d + 1) * w];
            win.copy_within(1.., 0);
            win[w - 1] = x[d];
            let mut acc = E::ZERO;
            for i in 0..w {
                acc += self.weight[d * w + i] * win[i];
            }
            out[d] = silu(acc);
        }
    }

    /// Drive `t_len` tokens of `[T][D]` input into `[T][D]` output.
    pub fn forward(&self, st: &mut ShortConvState<E>, t_len: usize, x: &[E], out: &mut [E]) {
        for t in 0..t_len {
            self.step(
                st,
                &x[t * self.dim..(t + 1) * self.dim],
                &mut out[t * self.dim..(t + 1) * self.dim],
            );
        }
    }
}

/// KDA's output gate: `rmsnorm(x)·weight·sigmoid(g)` over one head's `D`
/// channels.
///
/// The norm is over the **ungated** `x` and the sigmoid multiplies afterwards.
/// The alternative ordering — normalizing `x·sigmoid(g)` — was measured against
/// the executed `FusedRMSNormGated` and missed by 3.93 against this one's
/// 8.96e-07, so it is not a matter of taste. Note this *is* an RMS norm
/// (`mean`, eps outside), unlike the q/k [`l2_normalize`] above; the two live
/// three lines apart in the same layer and are not the same operation.
pub fn rms_norm_gated<E: Elem>(x: &[E], g: &[E], weight: &[E], eps: f64, out: &mut [E]) {
    let n = x.len();
    assert_eq!(g.len(), n);
    assert_eq!(weight.len(), n);
    assert_eq!(out.len(), n);
    let mut ss = E::ZERO;
    for &xi in x {
        ss += xi * xi;
    }
    let inv = E::ONE / (ss / E::from_f64(n as f64) + E::from_f64(eps)).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv * weight[i] * sigmoid(g[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> KdaConfig {
        KdaConfig {
            num_heads: 2,
            head_k_dim: 8,
            head_v_dim: 8,
            conv_kernel: 4,
            gate_lower_bound: Some(-5.0),
            l2norm_eps: 1e-6,
        }
    }

    /// The point of the whole design: the per-layer state does not know how
    /// long the sequence is. Driving 1, 16 and 128 tokens must leave the state
    /// byte-for-byte the same size.
    #[test]
    fn state_is_o1_in_sequence_length() {
        let cfg = tiny_cfg();
        let params = KdaParams::new(
            &cfg,
            &vec![0.3f64; cfg.num_heads],
            &vec![0.1f64; cfg.num_heads * cfg.head_k_dim],
        );
        let kda = Kda::new(cfg, params);
        let hk = cfg.num_heads * cfg.head_k_dim;
        let conv = ShortConv::new(hk, cfg.conv_kernel, &vec![0.25f64; hk * cfg.conv_kernel]);

        let hv = cfg.num_heads * cfg.head_v_dim;
        let mut sizes = Vec::new();
        for &t_len in &[1usize, 16, 128] {
            let mut st = KdaState::<f64>::zeros(&cfg);
            let mut cst = ShortConvState::zeros(&conv);
            let mut scr = KdaScratch::new(&cfg);
            let mut out = vec![0.0f64; hv];
            let mut conv_out = vec![0.0f64; hk];
            for t in 0..t_len {
                let x: Vec<f64> = (0..hk).map(|i| ((i + t) as f64 * 0.017).sin()).collect();
                conv.step(&mut cst, &x, &mut conv_out);
                kda.step(
                    &mut st,
                    &mut scr,
                    KdaToken {
                        q_raw: &conv_out,
                        k_raw: &conv_out,
                        v: &conv_out[..hv],
                        g_raw: &x,
                        beta_raw: &vec![0.5f64; cfg.num_heads],
                    },
                    &mut out,
                );
            }
            sizes.push((st.byte_len(), cst.byte_len()));
            assert!(out.iter().all(|o| o.is_finite()), "T={} produced non-finite output", t_len);
        }
        assert_eq!(sizes[0], sizes[1], "state grew between T=1 and T=16");
        assert_eq!(sizes[1], sizes[2], "state grew between T=16 and T=128");
        assert_eq!(sizes[0].0, cfg.state_elems() * 8);
    }

    /// The bounded gate's floor is structural, and the two branches genuinely
    /// disagree — a port that took the softplus branch would decay to zero
    /// where this one holds e⁻⁵.
    ///
    /// Mathematically `g ∈ (-5, 0)` is open at both ends, but in floating point
    /// the sigmoid saturates to exactly 1.0 (and to exactly 0.0), so `g = -5`
    /// and `g = 0` are both attained. That does not weaken the claim — the
    /// floor is still `e⁻⁵ = 0.0067…` and never below — but a port that
    /// asserted the open interval would fail on its own model's inputs, which
    /// routinely drive the sigmoid past f64 saturation.
    #[test]
    fn bounded_gate_floors_retention_at_e_minus_5() {
        let lb = Some(-5.0f64);
        let floor = (-5.0f64).exp();
        for &a_log in &[0.0f64, 1.0, 3.0] {
            let mut saturated = false;
            for i in -60..=60 {
                let z = i as f64;
                let g = decay_gate(a_log, z, 0.0, lb);
                assert!(g >= -5.0 && g <= 0.0, "g={} out of [-5, 0]", g);
                assert!(g.exp() >= floor, "retention {} below e^-5", g.exp());
                saturated |= g == -5.0;
            }
            assert!(saturated, "the sweep never reached the floor");
            // A large POSITIVE pre-activation is maximum forgetting (the
            // sigmoid saturates high and `g → lower_bound`); a large negative
            // one is full retention. Deep in the forget direction the two
            // branches diverge without bound — the bounded gate stops at -5
            // while the softplus gate keeps falling, which is the whole
            // consequence this port turns on.
            let bounded_hi = decay_gate(a_log, 50.0, 0.0, lb);
            let unbounded_hi = decay_gate(a_log, 50.0, 0.0, None);
            assert!((bounded_hi - (-5.0)).abs() < 1e-12, "bounded gate saturates at -5");
            assert!(unbounded_hi < -49.0 * a_log.exp(), "softplus(50) ≈ 50");
            let bounded_lo = decay_gate(a_log, -50.0, 0.0, lb);
            let unbounded_lo = decay_gate(a_log, -50.0, 0.0, None);
            assert!(bounded_lo.abs() < 1e-12 && bounded_lo <= 0.0, "retain: g ≈ 0");
            assert!(unbounded_lo.abs() < 1e-12 && unbounded_lo <= 0.0, "retain: g ≈ 0");
        }
    }

    /// `[K,V]` → `[V,K]` → `[K,V]` is the identity, and the transpose is not.
    #[test]
    fn state_layout_roundtrip() {
        let cfg = tiny_cfg();
        let n = cfg.state_elems();
        let kv: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let st = KdaState::from_kv(&cfg, &kv);
        let vk = st.to_vk(&cfg);
        assert_ne!(vk, kv, "transpose is not a no-op for K == V with distinct values");
        assert_eq!(KdaState::from_vk(&cfg, &vk).as_kv(), &kv[..]);
    }
}
