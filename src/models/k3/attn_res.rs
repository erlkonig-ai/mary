//! AttnRes — attention along the **depth** axis.
//!
//! Kimi K3 does not have a plain residual stream. Every sublayer's input is a
//! per-token softmax mixture over a *bank* of depth checkpoints plus the
//! running accumulator, with one learned query direction per site. The residual
//! add is still there — the accumulator is `x + attn` then `+ mlp` — but what
//! the next `RMSNorm` sees is not the accumulator, it is a convex combination
//! of the accumulator with snapshots taken every `attn_res_block_size` layers.
//!
//! Shipped definition (`modeling_kimi_linear.py`, `_apply_attn_res`), verbatim:
//!
//! ```text
//! v          = cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)  # [T, S+1, H]
//! v_float    = v.float()
//! variance   = v_float.pow(2).mean(-1, keepdim=True)
//! k          = v_float * rsqrt(variance + norm.variance_epsilon)
//! score_wght = norm.weight.float() * proj.weight.squeeze(0).float()   # [H]
//! scores     = (k * score_wght).sum(-1)                               # [T, S+1]
//! probs      = scores.softmax(-1).unsqueeze(1)                        # [T, 1, S+1]
//! out        = matmul(probs, v_float).squeeze(1).to(v.dtype)          # [T, H]
//! ```
//!
//! Three things in there are easy to get wrong and are settled by measurement
//! against a capture of the shipped module (see `src/bin/k3_attn_res_gate.rs`):
//!
//! * The accumulator is concatenated **last**, after the bank. The mixture
//!   *output* does not care — softmax is permutation-equivariant and the
//!   matmul sums over the slots — but the **bank does**, because the next
//!   snapshot is appended after the existing entries and every later layer
//!   sees that ordering.
//! * The mixture is over the **raw** `v`. Only the *scores* see the normalised
//!   `k`. Mixing the normalised copy is a different function entirely.
//! * The score weight is the elementwise product of an RMSNorm gain `[H]` and
//!   a projection row `[1, H]`. The product is commutative, so the *product*
//!   cannot tell those two apart — the shapes can, which is why
//!   [`AttnResParams::new`] takes them at their checkpoint ranks and asserts
//!   both.
//!
//! ## The depth state machine, and its boundary
//!
//! [`DepthMixer`] is the part with a memory. Per layer, in order:
//!
//! ```text
//! accumulator := layer_in
//! if bank is non-empty:  to_attention = mix(bank ++ [accumulator], sa_params)
//! else:                  to_attention = layer_in          # no mix call at all
//! if layer is a checkpoint layer:
//!     bank.push(layer_in)          # the RAW layer input, appended LAST
//!     accumulator := None          # ← the reset
//! ...attention runs on norm(to_attention)...
//! accumulator := match accumulator { Some(a) => a + attn_out, None => attn_out }
//! to_mlp = mix(bank ++ [accumulator], mlp_params)
//! ...mlp runs on norm(to_mlp)...
//! accumulator := accumulator + mlp_out
//! layer_out = accumulator
//! ```
//!
//! The reset is the whole point of the primitive and the thing an off-by-one
//! destroys silently: with `attn_res_block_size = 12` over 93 layers, nothing
//! accumulates across more than 12 layers, and the bank ends up holding 8
//! snapshots (layers 0, 12, 24, …, 84). Snapshot one layer early or late and
//! the model still runs, still produces finite activations, and is wrong.
//!
//! This type therefore **does not own the boundary rule**. It is constructed
//! from a schedule — one `bool` per layer — and
//! [`K3TextConfig::is_attn_res_checkpoint`](super::config::K3TextConfig::is_attn_res_checkpoint)
//! is the single place that rule is written down. [`DepthMixer::from_config`]
//! is the only constructor a model should use; a mutation to the config
//! predicate propagates here rather than being masked by a second copy.
//!
//! ## Precision
//!
//! The shipped module keeps hidden states in bfloat16 and does *all* of the
//! AttnRes arithmetic in f32, rounding only on the way out. So do we: tensors
//! are f32 (that is the arithmetic), and [`round_bf16`] is applied at exactly
//! the points torch's dtype does it — each mixture output, each accumulator
//! add. Skipping those rounds is not a harmless accuracy win: it changes which
//! bits reach the next layer, and the depth axis carries them 93 layers deep.

use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::config::K3TextConfig;

/// Round to bfloat16 and back, round-to-nearest-even, in pure f32 arithmetic.
///
/// This is Dekker's splitting constant `2^16 + 1`: `c = C·x; hi = c − (c − x)`
/// leaves `hi` holding the top `24 − 16 = 8` significand bits of `x`, correctly
/// rounded. It is what `tensor.to(torch.bfloat16).to(torch.float32)` does, and
/// it needs no bit manipulation, no host round trip and no backend support for
/// a half element type — three multiplies-and-subtracts that every backend has.
///
/// **Domain: the normal f32 range, both ends.** Exhaustively checked against a
/// bit-level round-to-nearest-even over every f32 bit pattern in the two
/// suspect regions:
///
/// * Above `2^128 / 65537 ≈ 5.19e33` the product `C·x` overflows and this
///   returns NaN where the true rounding is finite.
/// * Every **subnormal** f32 (`|x| < 2^-126 ≈ 1.1755e-38`) is returned
///   unchanged instead of being rounded to the nearest bfloat16, because the
///   split's intermediate loses the bits that decide it. `2^-126` itself, and
///   everything above it, is exact.
///
/// Both ends are decades outside anything a hidden state visits — across the
/// gated 13-layer prefix the AttnRes activations span `7.2e-16 .. 25.0` — but
/// this is a documented domain, not a universal `f32 -> bf16`, and the gate
/// *measures* both breakdown points and the margin to the real data rather
/// than trusting this paragraph. Do not lift it into a general utility.
pub fn round_bf16<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    /// `2^16 + 1` — splits f32's 24 significand bits into 8 + 16.
    const DEKKER_C: f32 = 65537.0;
    let c = x.clone().mul_scalar(DEKKER_C);
    c.clone() - (c - x)
}

/// Stack the mixture's candidates: the bank in order, then the accumulator
/// **last**.
///
/// This is `torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)`, and
/// it is a free function rather than a private method because the ordering is
/// the load-bearing convention of the whole primitive and a gate has to be
/// able to assert the stack itself, not only what the mixture made of it.
///
/// The *mixture output* is insensitive to this order — softmax is
/// permutation-equivariant and the matmul sums over the slot axis — so no
/// comparison of outputs can catch it being wrong. What it changes is the
/// bank: the next snapshot is appended after the existing entries, and every
/// later layer sees that ordering.
pub fn stack_candidates<B: Backend>(bank: &[Tensor<B, 2>], accumulator: Tensor<B, 2>) -> Tensor<B, 3> {
    let [tokens, hidden] = accumulator.dims();
    let mut slots: Vec<Tensor<B, 3>> = Vec::with_capacity(bank.len() + 1);
    for entry in bank {
        assert_eq!(
            entry.dims(),
            [tokens, hidden],
            "bank entry shape differs from the accumulator's"
        );
        slots.push(entry.clone().reshape([tokens, 1, hidden]));
    }
    slots.push(accumulator.reshape([tokens, 1, hidden]));
    Tensor::cat(slots, 1)
}

/// Sum over the last axis with a pairwise (binary-tree) reduction.
///
/// **Not an optimisation — an accuracy requirement.** The shipped module's two
/// reductions each run over all 7168 hidden dimensions, and the score one has
/// heavy cancellation: the sum is far smaller than the sum of the magnitudes.
/// A left-to-right f32 accumulation carries `O(n)` rounding into that, and
/// measured against a float64 transcription on the real activations it lands
/// **2.4e-3 relative** — while torch, whose reduction is a tree, lands
/// **2.4e-7**. That is not a rounding detail hiding under a tolerance: the
/// scores go straight into a softmax over the depth candidates, and 2.4e-3 of
/// score error moved the mixing probabilities by 6.6e-4. The port would be
/// making measurably different depth-attention decisions from the model it is
/// a port of.
///
/// Pairwise summation restores `O(log n)` error, which is what torch's
/// reduction has, so the two implementations differ by their own last ulp
/// rather than by a visible amount. The cost is the same `n - 1` additions in
/// `ceil(log2 n)` passes instead of one.
///
/// An odd length carries its last element forward unpaired, so this is exact
/// for every width, not only powers of two.
fn tree_sum_last<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [_, _, width] = x.dims();
    assert!(width > 0, "tree_sum_last over a zero-width axis");
    let mut cur = x;
    let mut n = width;
    while n > 1 {
        let half = n / 2;
        let lo = cur.clone().narrow(2, 0, half);
        let hi = cur.clone().narrow(2, half, half);
        let mut next = lo + hi;
        if n % 2 == 1 {
            next = Tensor::cat(vec![next, cur.narrow(2, n - 1, 1)], 2);
            n = half + 1;
        } else {
            n = half;
        }
        cur = next;
    }
    cur
}

/// The two learned parameters of one AttnRes call site, pre-multiplied.
///
/// One site is one `(RMSNorm gain, projection row)` pair: `mlp_res_*`,
/// `self_attention_res_*`, or the model-level `output_attn_res_*`.
#[derive(Debug, Clone)]
pub struct AttnResParams<B: Backend> {
    score_weight: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> AttnResParams<B> {
    /// Build from the two checkpoint tensors, **at their checkpoint ranks**.
    ///
    /// `norm_weight` is `*_res_norm.weight`, shape `[hidden]`; `proj_weight` is
    /// `*_res_proj.weight`, shape `[1, hidden]` — a single query direction, not
    /// a bank of them. Taking them at different ranks is deliberate: the score
    /// weight is their elementwise product, and multiplication is commutative,
    /// so nothing downstream can ever detect the two being swapped. The rank
    /// assertion is the only place that error can be caught, so it is made
    /// here and not left to a caller's discipline.
    pub fn new(norm_weight: Tensor<B, 1>, proj_weight: Tensor<B, 2>, eps: f64) -> Self {
        let [hidden] = norm_weight.dims();
        let [rows, cols] = proj_weight.dims();
        assert_eq!(
            rows, 1,
            "AttnRes projection has {rows} rows; the shipped `nn.Linear(hidden, 1)` \
             stores exactly one — more than one row means this is not the projection"
        );
        assert_eq!(
            cols, hidden,
            "AttnRes projection is {cols} wide but the norm gain is {hidden} — \
             they must both be the hidden size"
        );
        assert!(eps > 0.0, "AttnRes RMSNorm epsilon must be positive, got {eps}");
        Self {
            score_weight: norm_weight * proj_weight.reshape([hidden]),
            eps,
        }
    }

    /// `norm.weight * proj.weight.squeeze(0)`, in f32 — what the scores dot
    /// the normalised candidates with.
    pub fn score_weight(&self) -> Tensor<B, 1> {
        self.score_weight.clone()
    }

    /// The RMSNorm epsilon (`rms_norm_eps`) used inside the score path.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Hidden size this site operates on.
    pub fn hidden(&self) -> usize {
        self.score_weight.dims()[0]
    }

    /// The mixture itself: `v` is `[tokens, slots, hidden]` and already carries
    /// the accumulator in its **last** slot.
    ///
    /// Returns the intermediates as well as the output. They are not debug
    /// output — `scores` and `probs` are the only place the normalised copy is
    /// used, and a gate that only ever sees the composed result cannot tell a
    /// wrong normalisation from a wrong mixture.
    pub fn mix(&self, v: Tensor<B, 3>) -> AttnResMix<B> {
        let [tokens, slots, hidden] = v.dims();
        assert!(tokens > 0, "AttnRes mix over zero tokens");
        assert!(
            slots > 0,
            "AttnRes mix over zero slots: the accumulator is always a slot, so an \
             empty `v` means the caller built the concatenation wrong"
        );
        assert_eq!(
            hidden,
            self.hidden(),
            "AttnRes mix width {hidden} does not match this site's parameters"
        );

        // The score path — and only the score path — sees the normalised copy.
        // Both reductions are pairwise: see `tree_sum_last` for why a flat sum
        // is not an acceptable substitute here.
        //
        // The normalisation is a DIVISION, not a multiply by `.recip()`. The
        // shipped module writes `rsqrt`, and the transcription is tempting —
        // but `recip()` is an *approximate* reciprocal on at least one backend
        // this port runs on: measured against float64, burn-ndarray's is
        // 2.85e-3 relative, twenty-four thousand f32 ulp. Fed into an RMSNorm
        // scale that is the same for all 7168 dimensions of a row, that error
        // survives the reduction intact and lands whole on the score, moving
        // the depth-attention probabilities by 6.6e-4. Its `sqrt` and its
        // division are both exact to within an ulp; only the reciprocal is
        // not. The gate measures all three every run, so this comment cannot
        // quietly go stale.
        let variance = tree_sum_last(v.clone().powf_scalar(2.0)).div_scalar(hidden as f64);
        let k = v.clone() / variance.add_scalar(self.eps).sqrt();
        let scores = tree_sum_last(k * self.score_weight.clone().reshape([1, 1, hidden]))
            .reshape([tokens, slots]);
        let probs = softmax(scores.clone(), 1);

        // The mixture is over the RAW v — and it is a weighted sum, written as
        // a weighted sum.
        //
        // The shipped module writes `matmul(probs.unsqueeze(1), v)`, and that
        // is the natural transcription. It is also, on this project's GPU
        // backend, a **reduced-precision** operation: burn's CUDA matmul goes
        // through the tensor cores, whose f32 path keeps a ~10-bit significand.
        // Measured, that moved the mixture output by 5.4e-3 relative — 1.4
        // bfloat16 steps, changing 4.9% of the output elements — while the CPU
        // lane, computing the identical formula, sat at 1.0e-3. Nothing about
        // the mixture needs a matmul: the contracted axis is the *slot* axis,
        // at most nine long. A broadcast multiply and a nine-term sum is the
        // same arithmetic in full precision, and cheaper than a GEMM launch
        // whose inner dimension is 3.
        let out = (probs.clone().reshape([tokens, slots, 1]) * v)
            .sum_dim(1)
            .reshape([tokens, hidden]);

        AttnResMix { scores, probs, out: round_bf16(out) }
    }
}

/// One AttnRes call site's output, with the two intermediates that decide it.
#[derive(Debug, Clone)]
pub struct AttnResMix<B: Backend> {
    /// `[tokens, slots]` — one logit per depth candidate, from the normalised
    /// candidate dotted with this site's query direction.
    pub scores: Tensor<B, 2>,
    /// `[tokens, slots]` — `softmax(scores)`, the per-token depth attention.
    pub probs: Tensor<B, 2>,
    /// `[tokens, hidden]` — the convex combination of the **raw** candidates,
    /// rounded to bfloat16 as the shipped module's dtype does.
    pub out: Tensor<B, 2>,
}

/// Which sublayer of a decoder layer a mixture feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Next call must be [`DepthMixer::enter_layer`].
    Entry,
    /// Next call must be [`DepthMixer::after_attention`].
    Attention,
    /// Next call must be [`DepthMixer::after_mlp`].
    Mlp,
}

/// The depth-axis state: the snapshot bank and the running accumulator.
///
/// Drive it one layer at a time, in order — [`enter_layer`](Self::enter_layer),
/// [`after_attention`](Self::after_attention), [`after_mlp`](Self::after_mlp) —
/// and [`finish`](Self::finish) once at the end. The call order is checked, not
/// assumed: a decoder that mixes the sequence up gets a panic rather than a
/// plausible-looking bank.
#[derive(Debug, Clone)]
pub struct DepthMixer<B: Backend> {
    /// One flag per layer: does this layer snapshot the accumulator into the
    /// bank on entry? Supplied by the config; never recomputed here.
    schedule: Vec<bool>,
    bank: Vec<Tensor<B, 2>>,
    accumulator: Option<Tensor<B, 2>>,
    layer: usize,
    stage: Stage,
}

impl<B: Backend> DepthMixer<B> {
    /// Build from an explicit per-layer checkpoint schedule.
    ///
    /// Prefer [`from_config`](Self::from_config). This exists so the schedule
    /// is *data* — the boundary rule lives in exactly one place (the config)
    /// and this type consumes it.
    pub fn new(schedule: Vec<bool>) -> Self {
        assert!(!schedule.is_empty(), "AttnRes schedule covers zero layers");
        assert!(
            schedule[0],
            "layer 0 is not a checkpoint layer in this schedule; the shipped \
             predicate is `layer_idx % block_size == 0`, and 0 satisfies it for \
             every block size — a schedule that misses it starts the first \
             mixture with an empty bank forever"
        );
        Self { schedule, bank: Vec::new(), accumulator: None, layer: 0, stage: Stage::Entry }
    }

    /// Resume a mixer at layer `layer` with a bank someone else built.
    ///
    /// The 93-layer state machine is a chain: layer *n*'s bank is everything
    /// layers 0..n snapshotted, so gating layer 4 in isolation means either
    /// running layers 0..3 first or supplying the bank. This is the second
    /// option, and it is not free — the resumed bank is *evidence from
    /// elsewhere*, and a gate that uses it is checking one layer against a
    /// reference, not thirteen layers against themselves. Callers must say so.
    ///
    /// What is still checked here is the one thing a caller cannot get wrong
    /// quietly: the bank's **length** must equal the number of checkpoint
    /// layers strictly before `layer`, because that count is a function of the
    /// schedule alone. A bank of the wrong depth is the failure mode this
    /// constructor exists to make impossible, and it is exactly the failure a
    /// hand-built `bank` invites.
    ///
    /// The stage is [`Entry`](Self::enter_layer): a resumed mixer is always
    /// about to enter a layer, never in the middle of one.
    pub fn resume(schedule: Vec<bool>, bank: Vec<Tensor<B, 2>>, layer: usize) -> Self {
        assert!(!schedule.is_empty(), "AttnRes schedule covers zero layers");
        assert!(
            schedule[0],
            "layer 0 is not a checkpoint layer in this schedule"
        );
        assert!(
            layer < schedule.len(),
            "resuming at layer {layer} but the schedule covers {} layers",
            schedule.len()
        );
        let want = schedule[..layer].iter().filter(|&&b| b).count();
        assert_eq!(
            bank.len(),
            want,
            "resuming at layer {layer} with a bank of {} snapshots; the schedule \
             takes {want} before it",
            bank.len()
        );
        if let Some(first) = bank.first() {
            let d = first.dims();
            assert!(d[0] > 0 && d[1] > 0, "resumed bank entry is empty: {d:?}");
            for (i, e) in bank.iter().enumerate() {
                assert_eq!(e.dims(), d, "resumed bank entry {i} has a different shape");
            }
        }
        Self { schedule, bank, accumulator: None, layer, stage: Stage::Entry }
    }

    /// Build the schedule from the model config. `None` when the config has no
    /// `attn_res_block_size`, i.e. the model is a plain-residual one.
    pub fn from_config(cfg: &K3TextConfig) -> Option<Self> {
        cfg.attn_res_block_size?;
        Some(Self::new(
            (0..cfg.num_hidden_layers).map(|l| cfg.is_attn_res_checkpoint(l)).collect(),
        ))
    }

    /// The per-layer checkpoint schedule this mixer was built with.
    pub fn schedule(&self) -> &[bool] {
        &self.schedule
    }

    /// Number of layers the schedule covers.
    pub fn num_layers(&self) -> usize {
        self.schedule.len()
    }

    /// Snapshots taken so far.
    pub fn bank_len(&self) -> usize {
        self.bank.len()
    }

    /// The snapshot bank, oldest first. Each entry is `[tokens, hidden]`.
    pub fn bank(&self) -> &[Tensor<B, 2>] {
        &self.bank
    }

    /// The running accumulator, or `None` between a snapshot and the attention
    /// output that restarts it.
    pub fn accumulator(&self) -> Option<&Tensor<B, 2>> {
        self.accumulator.as_ref()
    }

    /// Index of the layer the mixer is currently inside (or about to enter).
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Layer entry: mix the bank with the incoming residual, then take the
    /// snapshot if this is a checkpoint layer.
    ///
    /// Returns what `input_layernorm` should be given. When the bank is empty
    /// there is **no mixture at all** — the shipped code guards the call on
    /// `block_residual.shape[1] > 0` — so `mix` is `None` and the layer input
    /// passes through untouched. That happens exactly once, at layer 0.
    pub fn enter_layer(&mut self, layer_in: Tensor<B, 2>, sa: &AttnResParams<B>) -> LayerEntry<B> {
        assert_eq!(self.stage, Stage::Entry, "enter_layer out of order at layer {}", self.layer);
        assert!(
            self.layer < self.schedule.len(),
            "entering layer {} but the schedule covers {} layers",
            self.layer,
            self.schedule.len()
        );

        let mix = if self.bank.is_empty() {
            None
        } else {
            Some(sa.mix(stack_candidates(&self.bank, layer_in.clone())))
        };
        let to_attention = match &mix {
            Some(m) => m.out.clone(),
            None => layer_in.clone(),
        };

        if self.schedule[self.layer] {
            // The snapshot is the RAW layer input — not the mixture, not the
            // normalised copy — appended after everything already in the bank.
            self.bank.push(layer_in);
            self.accumulator = None;
        } else {
            self.accumulator = Some(layer_in);
        }

        self.stage = Stage::Attention;
        LayerEntry { to_attention, mix }
    }

    /// Fold the attention output into the accumulator and mix for the MLP.
    ///
    /// On a checkpoint layer the accumulator was just reset, so this *replaces*
    /// it with the attention output instead of adding to it. That single
    /// branch is the reset: it is why nothing accumulates across a boundary.
    pub fn after_attention(&mut self, attn_out: Tensor<B, 2>, mlp: &AttnResParams<B>) -> AttnResMix<B> {
        assert_eq!(
            self.stage,
            Stage::Attention,
            "after_attention out of order at layer {}",
            self.layer
        );
        let accumulator = match self.accumulator.take() {
            Some(prefix) => round_bf16(prefix + attn_out),
            None => attn_out,
        };
        self.accumulator = Some(accumulator.clone());
        self.stage = Stage::Mlp;
        // Unconditional: layer 0 is always a checkpoint layer (`0 % n == 0`),
        // so the bank is never empty by the time control reaches here.
        assert!(!self.bank.is_empty(), "MLP mixture with an empty bank at layer {}", self.layer);
        mlp.mix(stack_candidates(&self.bank, accumulator))
    }

    /// Fold the MLP output in and close the layer. Returns the layer output —
    /// the accumulator, which is what the next layer receives as `layer_in`.
    pub fn after_mlp(&mut self, mlp_out: Tensor<B, 2>) -> Tensor<B, 2> {
        assert_eq!(self.stage, Stage::Mlp, "after_mlp out of order at layer {}", self.layer);
        let accumulator = round_bf16(
            self.accumulator.take().expect("accumulator missing after attention") + mlp_out,
        );
        self.accumulator = Some(accumulator.clone());
        self.layer += 1;
        self.stage = Stage::Entry;
        accumulator
    }

    /// The model-level AttnRes, applied to the final hidden state **before**
    /// `model.norm`. Its bank is every snapshot the run took; its accumulator
    /// slot is the last layer's output.
    ///
    /// Not a method that mutates: the model ends here.
    pub fn finish(&self, hidden: Tensor<B, 2>, out: &AttnResParams<B>) -> AttnResMix<B> {
        assert_eq!(
            self.layer,
            self.schedule.len(),
            "finish() after {} of {} layers",
            self.layer,
            self.schedule.len()
        );
        assert_eq!(self.stage, Stage::Entry, "finish() inside a layer");
        out.mix(stack_candidates(&self.bank, hidden))
    }
}

/// What [`DepthMixer::enter_layer`] produced.
pub struct LayerEntry<B: Backend> {
    /// The tensor `input_layernorm` should see.
    pub to_attention: Tensor<B, 2>,
    /// The mixture that produced it, or `None` when the bank was empty and the
    /// layer input passed straight through.
    pub mix: Option<AttnResMix<B>>,
}
