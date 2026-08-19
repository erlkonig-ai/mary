//! Inkling's device lane — the whole arithmetic of a decoder layer, in Burn.
//!
//! There is no other lane. This file used to be one of a pair: a scalar f32
//! host lane in [`crate::models::inkling::mlp`] was the reference, this was
//! "the Burn lane beside it", and `inkling_forward` picked between them per
//! stage with `INK_ATTN`/`INK_DENSE`/`INK_HEAD`. A host reference you can
//! select at run time is a host reference you can accidentally run, and it was
//! costing 401 s a forward when it was.
//!
//! The switches went first and the CODE stayed, which turned out to be the
//! wrong half of the fix: an unreachable host MoE is not run by accident, it is
//! WORKED ON by accident, because it is the version of the algorithm a reader
//! can follow. That is why `mlp` now holds only the dense MLP the MTP heads
//! need, and why the routed algorithm is spelled out in
//! `inkling_forward::routed_experts_fp4` rather than in a function nothing
//! calls.
//!
//! Scope is attention with its two short convolutions and its KV cache, the
//! shared experts, the dense MLP and RMSNorm. The routed experts are NOT here:
//! they go through `mma.sync...kind::mxf4nvf4` in
//! [`crate::models::inkling::fp4gemm`] as the NVFP4 they are stored as. The
//! `dequant_nvfp4`/`expert_weight_from_packed` chain that used to live at the
//! bottom of this file decoded them into f32 first, which is the widening this
//! model is quantized precisely to avoid.
//!
//! Everything here is f32 because the weights it touches ARE f32 in the
//! checkpoint or are BF16 whose only consumer is an f32 op; the accumulator of
//! an FP4 MMA is f32 too, and that is the instruction's own output type, not a
//! widening.

use burn::prelude::*;
use burn::tensor::{Int, Tensor, TensorData};

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::seam::{client_of, handle_of, tensor_of, Bk};

/// `x * sigmoid(x)`, elementwise.
pub fn silu<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let s = burn::tensor::activation::sigmoid(x.clone());
    x * s
}

/// `nn.Linear(bias=False)`: `x @ Wᵀ` for a `[out, in]` weight.
///
/// The weight keeps its checkpoint orientation, so a transposition mistake is
/// a shape error rather than a plausible wrong answer.
pub fn linear<B: Backend>(x: Tensor<B, 2>, w: Tensor<B, 2>) -> Tensor<B, 2> {
    let [_, k] = x.dims();
    let [_, kw] = w.dims();
    assert_eq!(k, kw, "linear: x is [_, {k}] but the weight is [_, {kw}]");
    x.matmul(w.transpose())
}

/// The same product against a weight ALREADY in `[in, out]`, transposed once.
///
/// [`linear`] transposes on every call, which is a kernel and a full copy of
/// the weight per call for a permutation whose result never changes. A weight
/// the run multiplies by every token can be stored in the orientation the
/// matmul wants, and then this is the matmul with nothing in front of it.
///
/// The two are NOT interchangeable at the call site and are not meant to be:
/// this one asserts on `w`'s FIRST dimension where [`linear`] asserts on its
/// second, so handing either the other's weight is a shape error at the
/// assertion rather than a plausible wrong answer further down.
pub fn linear_pre_t<B: Backend>(x: Tensor<B, 2>, w: Tensor<B, 2>) -> Tensor<B, 2> {
    let [_, k] = x.dims();
    let [kw, _] = w.dims();
    assert_eq!(k, kw, "linear_pre_t: x is [_, {k}] but the weight is [{kw}, _]");
    x.matmul(w)
}

/// The same product, against the BF16 the checkpoint actually stores.
///
/// [`linear`] takes a `Tensor<B, 2>`, and on this backend that means f32 — so
/// every BF16 weight reaching it had to be widened first, at twice the bytes
/// the pile holds and twice the memory traffic to multiply by. This is the
/// [`crate::models::inkling::bf16gemm`] lane instead: the stored bytes go into
/// `mma.sync…bf16` as they lie, the activation is cast to BF16 on the device
/// by the hardware's round-to-nearest-even, and the f32 that comes back is the
/// MMA's own accumulator type rather than a widened weight.
///
/// Concrete on [`Bk`] for the reason [`short_conv_step`] is: [`Bf16W`] is a raw
/// cubecl handle and the seam that produces one is not generic.
///
/// The M padding is the instruction's, not the caller's: a decode step feeds
/// one row against a sixteen-row tile, and the fifteen padding rows are written
/// as zeros by the cast that was visiting every element anyway, then sliced off
/// here.
pub fn linear_bf16(x: Tensor<Bk, 2>, w: &Bf16W) -> Tensor<Bk, 2> {
    use crate::models::inkling::bf16gemm::rows_for;
    let [m, k] = x.dims();
    assert_eq!(k, w.k, "linear_bf16: x is [_, {k}] but the weight is [_, {}]", w.k);
    // Only the hand lane pads, and only because its grid is its tiling; ask
    // rather than assume, or a decode step slices fifteen rows that were never
    // computed.
    let rows = rows_for(w.align, m);
    let client = client_of(&x);
    let dev = x.device();
    let out = crate::models::inkling::bf16gemm::linear_bf16(&client, &handle_of(x), w, m);
    tensor_of(client, dev, out, rows, w.n).slice([0..m, 0..w.n])
}

/// RMS normalization with a per-feature gain.
///
/// Divides by `sqrt(var + eps)` rather than multiplying by its reciprocal: on
/// some backends `recip` dispatches to an approximate SIMD reciprocal, which
/// cost K3 about fourteen bits of accuracy before it was caught. Same hazard
/// here, same avoidance.
pub fn rms_norm<B: Backend>(x: Tensor<B, 2>, gain: Tensor<B, 1>, eps: f64) -> Tensor<B, 2> {
    let [_, w] = x.dims();
    assert_eq!(gain.dims()[0], w, "rms_norm: gain is {} wide, input {w}", gain.dims()[0]);
    let mean_sq = x.clone().powf_scalar(2.0).mean_dim(1);
    let denom = mean_sq.add_scalar(eps).sqrt();
    let normed = x / denom;
    normed * gain.unsqueeze::<2>()
}

/// The last `kernel - 1` rows of `x`, left-padded with zeros when `x` is short.
///
/// This is the short convolution's whole memory: the taps reach `kernel - 1`
/// positions back, and a sequence that has not got that far yet is padded with
/// the same zeros [`short_conv`] assumes for `x[<0]`. Seeding the history from a
/// prefill shorter than the kernel and *not* padding would silently shift every
/// subsequent tap by one position.
pub fn conv_history<B: Backend>(x: Tensor<B, 2>, kernel: usize) -> Tensor<B, 2> {
    let [tokens, dim] = x.dims();
    let want = kernel - 1;
    if tokens >= want {
        x.slice([tokens - want..tokens, 0..dim])
    } else {
        let dev = x.device();
        let pad: Tensor<B, 2> = Tensor::zeros([want - tokens, dim], &dev);
        Tensor::cat(vec![pad, x], 0)
    }
}

/// One position of the short convolution, given the `kernel - 1` inputs before
/// it. Returns the output row and the history to carry to the next position.
///
/// `hist` is oldest-first, so the window this convolves is
/// `[hist[0], …, hist[kernel-2], x]` — exactly the `x[pos - (kernel - 1) ..=
/// pos]` that [`short_conv`]'s last row reads.
///
/// This used to BE that call: concatenate, convolve every position of the
/// window, keep the last row. It was nineteen launches to produce one row, four
/// times a layer, and `nsys` charged 1520 of a decode step's 4720 kernels to
/// it — a third of them, for 1.5 ms of the 84 ms the GPU was busy. It is now
/// [`crate::models::inkling::sconv`], one kernel, one thread a channel,
/// accumulating the taps in the same ascending order the slice lane added them.
///
/// Concrete on [`Bk`] rather than generic over `B: Backend`, because the kernel
/// takes a `cubecl::server::Handle` and the seam that produces one is concrete.
/// That costs nothing: this file's own tests already say "the only backend
/// there is", and [`short_conv`] — the prefill form, which runs once — stays
/// generic.
pub fn short_conv_step(
    hist: Tensor<Bk, 2>,
    x: Tensor<Bk, 2>,
    weight: Tensor<Bk, 2>,
) -> (Tensor<Bk, 2>, Tensor<Bk, 2>) {
    let [rows, dim] = x.dims();
    assert_eq!(rows, 1, "a decode step convolves exactly one position");
    let [wdim, kernel] = weight.dims();
    assert_eq!(dim, wdim, "short_conv_step: x is [_, {dim}] but the weight is [{wdim}, _]");
    assert_eq!(
        hist.dims(),
        [kernel - 1, dim],
        "the history must be the {} rows before this one",
        kernel - 1
    );
    let client = client_of(&x);
    let dev = x.device();
    let (h_hist, h_x, h_w) = (handle_of(hist), handle_of(x), handle_of(weight));
    let (out, next) =
        crate::models::inkling::sconv::short_conv_decode(&client, &h_hist, &h_x, &h_w, dim, kernel);
    (
        tensor_of(client.clone(), dev.clone(), out, 1, dim),
        tensor_of(client, dev, next, kernel - 1, dim),
    )
}

/// Depthwise causal short convolution **plus its internal residual**, on device.
///
/// The device twin of [`crate::models::inkling::block::short_conv`]:
///
/// ```text
/// conv[t] = sum_{j=0}^{k-1} w[j] * x[t + j - (k - 1)]        x[<0] = 0
/// out[t]  = x[t] + conv[t]
/// ```
///
/// Written as `k` shifted slices of a front-zero-padded input rather than as a
/// convolution kernel, because `k` is 4 and the shift is exactly what the
/// formula says. Returning only `conv` — dropping the module's own residual —
/// is the mistake this shape makes hard to hide.
pub fn short_conv<B: Backend>(x: Tensor<B, 2>, weight: Tensor<B, 2>) -> Tensor<B, 2> {
    let [tokens, dim] = x.dims();
    let [wdim, kernel] = weight.dims();
    assert_eq!(dim, wdim, "short_conv: x is [_, {dim}] but the weight is [{wdim}, _]");
    assert!(kernel > 0, "a short convolution needs a kernel");
    let dev = x.device();
    let pad: Tensor<B, 2> = Tensor::zeros([kernel - 1, dim], &dev);
    let padded = Tensor::cat(vec![pad, x.clone()], 0);

    let mut conv: Option<Tensor<B, 2>> = None;
    for j in 0..kernel {
        // t + j - (kernel - 1) in x is t + j in the padded tensor.
        let seg = padded.clone().slice([j..j + tokens, 0..dim]);
        let wj = weight.clone().slice([0..dim, j..j + 1]).reshape([1, dim]);
        let term = seg * wj;
        conv = Some(match conv {
            None => term,
            Some(c) => c + term,
        });
    }
    x + conv.expect("kernel > 0")
}

/// RMS-normalize each head slice of `[tokens, heads * head_dim]`.
fn head_rms_norm<B: Backend>(
    v: Tensor<B, 2>,
    gain: Tensor<B, 1>,
    heads: usize,
    head_dim: usize,
    eps: f64,
) -> Tensor<B, 2> {
    let [tokens, width] = v.dims();
    assert_eq!(width, heads * head_dim, "{width} is not {heads} x {head_dim}");
    rms_norm(v.reshape([tokens * heads, head_dim]), gain, eps).reshape([tokens, width])
}

/// Every weight one attention layer needs, already on the device.
///
/// Orientations are the checkpoint's: `w*` are `[out, in]` the way `nn.Linear`
/// stores them, the short convolutions are `[dim, kernel]`, and `rel_proj` is
/// `[d_rel, rel_extent]`.
///
/// ## Why the five projections are [`Bf16W`] and the rest are not
///
/// The projections are the weight. On a 20-layer node they were 3.29 GiB of
/// device f32 holding 1.64 GiB of stored BF16 — the same widening the dense and
/// shared MLPs already stopped doing, left in place here because the change was
/// once refused for not being bit-exact against the f32 lane. It does not need
/// to be bit-exact against the f32 lane; it needs to be inside budget against
/// the reference, and `inkling_attn_bf16_gate` measures that.
///
/// What stays f32 is what is not worth the plumbing and would not pay: the two
/// short convolutions are `[dim, 4]`, the two norm gains are `[head_dim]`, and
/// `rel_proj` is `[16, rel_extent]` — 70 KB a layer between them, against
/// 88 MB of projections, and none of them is a `[out, in]` matmul the MMA lane
/// takes.
///
/// Not generic over `B` any more: [`Bf16W`] is a raw cubecl handle, so this
/// struct and the three functions that read it are concrete on [`Bk`], exactly
/// as [`short_conv_step`] is and for the same reason.
pub struct AttnWeightsDev {
    pub wq: Bf16W,
    pub wk: Bf16W,
    pub wv: Bf16W,
    pub wr: Bf16W,
    pub wo: Bf16W,
    pub k_sconv: Tensor<Bk, 2>,
    pub v_sconv: Tensor<Bk, 2>,
    pub q_norm: Tensor<Bk, 1>,
    pub k_norm: Tensor<Bk, 1>,
    pub rel_proj: Tensor<Bk, 2>,
}

/// Everything one attention layer must retain between generated tokens.
///
/// Two kinds of state, and forgetting the second is the bug this type exists to
/// make hard: the keys and values themselves, and the `kernel - 1`
/// **pre-convolution** K and V projections that the *next* position's short
/// convolution reaches back into. Caching only the post-convolution K/V reads
/// as complete and silently truncates every short convolution at the prefill
/// boundary — the taps see zeros where three real positions should be.
///
/// `k` is post-convolution **and** post-QK-norm, `v` post-convolution; both are
/// functions of the prefix alone, which is the property that makes them
/// cacheable. Row 0 is absolute position [`AttnCache::base`], not 0: a local
/// layer drops keys that have left its window, so the row index is not the
/// position and every distance must be computed through `base`.
#[derive(Clone)]
pub struct AttnCache<B: Backend> {
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    k_pre: Tensor<B, 2>,
    v_pre: Tensor<B, 2>,
    base: usize,
    /// Set by [`attention_steps`] and cleared by [`AttnCache::commit`]: the
    /// rows a SPECULATIVE batch appended, which may turn out not to have
    /// happened.
    pending: Option<Pending<B>>,
}

/// What a speculative batch must be able to undo.
///
/// Truncating K and V is a slice; restoring the short convolution's memory is
/// not, because the history the next position reads is a function of the last
/// KEPT row and the rows before it, and those pre-convolution projections are
/// gone once the batch is over. So the batch keeps the whole
/// `kernel - 1 + rows` window it built, and any accepted prefix is a slice of
/// it. Rolling back to `keep` rows is then exactly as cheap as rolling back to
/// all of them, which is the property that lets a verifier decide LATE.
#[derive(Clone)]
struct Pending<B: Backend> {
    k_pre: Tensor<B, 2>,
    v_pre: Tensor<B, 2>,
    rows: usize,
}

impl<B: Backend> AttnCache<B> {
    /// Keys retained — *not* the sequence length, because a windowed layer
    /// forgets.
    pub fn len(&self) -> usize {
        self.k.dims()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Absolute position of row 0.
    pub fn base(&self) -> usize {
        self.base
    }

    /// Keep the first `keep` of the rows the last [`attention_steps`] appended
    /// and discard the rest.
    ///
    /// This is the whole of speculative rollback. A verifier accepts a PREFIX
    /// of a drafted batch, and the positions past it were computed against
    /// tokens the model did not choose; leaving their K and V behind does not
    /// error, it shows up months later as an acceptance rate that drifts down.
    ///
    /// The window trim is deferred to here rather than done inside
    /// [`attention_steps`], and that is not tidiness: trimming to the last
    /// `window` keys of a batch that is then rolled BACK would have dropped
    /// keys the shorter sequence still needs. A speculative batch may not
    /// forget until it knows how long it was.
    ///
    /// Idempotent and safe with no batch outstanding — it still trims, which is
    /// what makes it correct to call after every verify pass.
    pub fn commit(&mut self, keep: usize, window: Option<usize>) {
        if let Some(p) = self.pending.take() {
            assert!(keep <= p.rows, "kept {keep} of a {}-row batch", p.rows);
            let drop = p.rows - keep;
            if drop > 0 {
                let [len, dim] = self.k.dims();
                self.k = self.k.clone().slice([0..len - drop, 0..dim]);
                let [vlen, vdim] = self.v.dims();
                self.v = self.v.clone().slice([0..vlen - drop, 0..vdim]);
            }
            // `k_pre` holds `kernel - 1` history rows followed by the batch's
            // own pre-convolution projections, so the history ending at the
            // last kept row is the window starting at `keep`.
            let hist = p.k_pre.dims()[0] - p.rows;
            let dim = p.k_pre.dims()[1];
            self.k_pre = p.k_pre.slice([keep..keep + hist, 0..dim]);
            let vdim = p.v_pre.dims()[1];
            self.v_pre = p.v_pre.slice([keep..keep + hist, 0..vdim]);
        }
        trim(self, window);
    }
}

/// Drop the keys no future query can reach.
///
/// A query at `pos` sees `pos - p < window`, and every later query sees a
/// strictly older cut-off, so the last `window` rows are exactly enough and
/// never more. Without this a local layer's cache grows without bound over a
/// long generation while the extra rows are masked to `-inf` on every step —
/// correct, and quadratic in a layer that was chosen to be linear.
fn trim<B: Backend>(c: &mut AttnCache<B>, window: Option<usize>) {
    let Some(w) = window else { return };
    let [len, dim] = c.k.dims();
    if len <= w {
        return;
    }
    let drop = len - w;
    c.k = c.k.clone().slice([drop..len, 0..dim]);
    c.v = c.v.clone().slice([drop..len, 0..dim]);
    c.base += drop;
}

/// One attention layer over a whole sequence, on the device, no cache.
///
/// The device twin of [`crate::models::inkling::attn::attention`] and gated
/// against the same `transformers` capture, not against it: matching the slice
/// lane would only prove the two agree, and they were written by the same hand.
///
/// `mask_window` is the sliding window a local layer masks with, and `None` on a
/// global layer. It is a predicate on `q - k` rather than an additive
/// `[tokens, tokens]` tensor: the epilogue kernel recomputes visibility per
/// element, so nothing quadratic is materialised to hold it.
///
/// Two things are folded together here that a careless reading separates:
/// log scaling multiplies the query **and** the relative-position bias, and only
/// on global layers; and the bias is zero outside `0 <= q - k < rel_extent`,
/// while causality is the `k <= q` half of the same predicate.
pub fn attention(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    mask_window: Option<usize>,
) -> Tensor<Bk, 2> {
    attention_prefill(x, w, d, log_scaling, mask_window, None).0
}

/// The same layer, keeping what a decode step will need.
///
/// Identical arithmetic to [`attention`] — that function is this one with the
/// cache dropped, so the `transformers` gate covers both and there is no second
/// transcription of the layer to drift.
///
/// `window` is the sliding window on a local layer and `None` on a global one,
/// the same distinction [`crate::models::inkling::attn::causal_mask`] takes;
/// it decides how much of the cache survives, and passing `None` for a local
/// layer would grow the cache past the window rather than give a wrong answer.
pub fn attention_prefill(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    mask_window: Option<usize>,
    window: Option<usize>,
) -> (Tensor<Bk, 2>, AttnCache<Bk>) {
    attention_prefill_lane(x, w, d, log_scaling, mask_window, window, true, None)
}

/// The same layer with the banded lane REFUSED, so a test can hold the two
/// implementations side by side.
///
/// Exists because the only check that catches a band which disagrees with the
/// dense triangle is one that runs both on the same weights: a banded kernel
/// checked against a banded reference proves the two share an author, not that
/// either is right.
#[cfg(test)]
pub(crate) fn attention_prefill_dense(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    mask_window: Option<usize>,
    window: Option<usize>,
    block: Option<usize>,
) -> (Tensor<Bk, 2>, AttnCache<Bk>) {
    attention_prefill_lane(x, w, d, log_scaling, mask_window, window, false, block)
}

#[allow(clippy::too_many_arguments)]
fn attention_prefill_lane(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    mask_window: Option<usize>,
    window: Option<usize>,
    banded_ok: bool,
    block: Option<usize>,
) -> (Tensor<Bk, 2>, AttnCache<Bk>) {
    use crate::models::inkling::config::AttnKind;

    let [tokens, hidden] = x.dims();
    assert_eq!(hidden, d.hidden, "x is [_, {hidden}] but the config says {}", d.hidden);
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(groups * kv_heads, heads, "{heads} heads do not divide into {kv_heads} kv heads");

    // K and V pass through their short convolutions; Q does not. The
    // pre-convolution projections are kept: they are the convolution's memory,
    // and a decode step cannot reconstruct them from the cached K and V.
    let q = linear_bf16(x.clone(), &w.wq);
    let k_pre = linear_bf16(x.clone(), &w.wk);
    let v_pre = linear_bf16(x.clone(), &w.wv);
    let k = short_conv(k_pre.clone(), w.k_sconv.clone());
    let v = short_conv(v_pre.clone(), w.v_sconv.clone());
    let r = linear_bf16(x, &w.wr);

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k = head_rms_norm(k, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    // Log scaling: the same vector the slice lane builds, from the same method.
    let taus: Vec<f32> = (0..tokens)
        .map(|t| match (d.kind, log_scaling) {
            (AttnKind::Global, Some(ls)) => ls.tau(t),
            _ => 1.0,
        })
        .collect();
    let tau: Tensor<Bk, 1> = Tensor::from_data(TensorData::new(taus, [tokens]), &dev);
    let q = q * tau.clone().reshape([tokens, 1]);

    // Only distances that can occur are worth projecting: a distance is at most
    // `tokens - 1` and the table stops at `rel_extent`.
    let eff = d.rel_extent.min(tokens);

    // A local layer is a BAND, and the band is the whole of its attention:
    // `[tokens, heads * head_dim]` out, one kernel, nothing quadratic written
    // down at any point. Thirty-five of this model's forty-two attention layers
    // take this arm. The dense arm below -- the GQA expansion, the two
    // transposes and the `[heads, n, n]` scores -- is built only for the seven
    // that are global, because a global layer really does read every key.
    let out: Tensor<Bk, 2> = match mask_window {
        // `win` and not `w`: the weights are `w` in this function, and a match
        // binding that shadowed them would silently reach the window wherever a
        // projection was meant.
        Some(win)
            if banded_ok
                && crate::models::inkling::banded::applies(heads, kv_heads, head_dim, win) =>
        {
            use crate::models::inkling::banded::banded_attention_launch;
            use crate::models::inkling::seam::{client_of, handle_of, tensor_of};
            // `[tokens, heads, eff]`, and eff <= 1024 -- linear in the
            // sequence. The band reads it whole because it runs the whole
            // sequence in one launch; the dense arm below builds it a query
            // block at a time, for which see the loop there.
            //
            // Left at `[tokens, heads, eff]` rather than swapped to
            // `[heads, tokens, eff]`: the swap is a permuted VIEW, and every
            // later reshape of a permuted view is a copy. The kernel takes the
            // head stride as an argument instead, which costs an index multiply
            // and moves no bytes.
            let rel = r
                .reshape([tokens * heads, d.d_rel])
                .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
                .reshape([tokens, heads, eff])
                * tau.reshape([tokens, 1, 1]);
            let client = client_of(&q);
            // Q, K, V and the relative table exactly as the projections left
            // them: `[tokens, heads * head_dim]` and `[tokens, kv_heads *
            // head_dim]`. The kernel indexes `h / groups` for the KV head, so
            // the repeat that used to materialise two more
            // `heads * tokens * head_dim` tensors does not happen here.
            let q_h = handle_of(q);
            // K DIMENSION-MAJOR, `[kv_heads, head_dim, tokens]`: the band's
            // score phase has one unit per key walking every dimension, and in
            // the `[tokens, kv_heads * head_dim]` layout that is 32 memory
            // transactions per warp instruction instead of one. `handle_of`
            // makes the permuted view contiguous, which is the transpose --
            // one linear pass per layer against a quadratic saving.
            let k_h = handle_of(
                k.clone().reshape([tokens, kv_heads, head_dim]).swap_dims(0, 1).swap_dims(1, 2),
            );
            let v_h = handle_of(v.clone());
            let rel_h = handle_of(rel);
            let o = banded_attention_launch(
                &client,
                &q_h,
                &k_h,
                &v_h,
                &rel_h,
                tokens,
                heads,
                kv_heads,
                head_dim,
                eff,
                win,
                d.scaling(),
            );
            tensor_of(client, dev.clone(), o, tokens, heads * head_dim)
        }
        _ => {
            // A global layer reads every key, so there is a square to compute.
            // It does not have to be MATERIALISED square: the queries come in
            // blocks of `rows`, and one block holds `[heads, rows, tokens]`
            // instead of `[heads, tokens, tokens]`. Same arithmetic, same
            // answer, same number of multiplies -- the only thing that changes
            // is that the largest allocation grows linearly in the sequence
            // rather than quadratically, and the per-buffer cap this device
            // enforces stops being what decides how long an input may be.
            use crate::models::inkling::budget::query_block;
            use crate::models::inkling::scorebias::score_epilogue_launch;
            use crate::models::inkling::seam::{
                client_of, handle_of, strided_of3, tensor_of3, tensor_strided3,
            };

            // [heads, tokens, head_dim]; the KV heads are repeated in place, so
            // head h reads kv head h / groups exactly as the slice lane indexes
            // it.
            let expand = |t: Tensor<Bk, 2>| -> Tensor<Bk, 3> {
                t.reshape([tokens, kv_heads, head_dim])
                    .swap_dims(0, 1)
                    .reshape([kv_heads, 1, tokens, head_dim])
                    .repeat_dim(1, groups)
                    .reshape([heads, tokens, head_dim])
            };
            let qv = q.reshape([tokens, heads, head_dim]).swap_dims(0, 1);
            let client = client_of(&qv);

            // Q, K^T and V made contiguous ONCE, ahead of the loop. Every query
            // block reads all of K and V, and a permuted view handed to the
            // matmul is made contiguous BY the matmul -- which is the whole of
            // it, per block, instead of once per layer.
            let qh = {
                let h = handle_of(qv);
                tensor_of3(client.clone(), dev.clone(), h, heads, tokens, head_dim)
            };
            let kt = {
                let h = handle_of(expand(k.clone()).swap_dims(1, 2));
                tensor_of3(client.clone(), dev.clone(), h, heads, head_dim, tokens)
            };
            let vh = {
                let h = handle_of(expand(v.clone()));
                tensor_of3(client.clone(), dev.clone(), h, heads, tokens, head_dim)
            };

            let rel_proj = w.rel_proj.clone().slice([0..d.d_rel, 0..eff]);
            // A parameter, not just `query_block`, because the only bug this
            // change can introduce is a block that reads its query position
            // LOCALLY, and that bug is invisible in block zero. A test needs to
            // force several blocks at a shape small enough to check, and a
            // process-global env var cannot do that while other tests run.
            let block = block.unwrap_or_else(|| query_block(heads, tokens)).clamp(1, tokens);
            let mut parts: Vec<Tensor<Bk, 2>> = Vec::with_capacity(tokens.div_ceil(block));
            for lo in (0..tokens).step_by(block) {
                let hi = (lo + block).min(tokens);
                let rows = hi - lo;

                // This block's relative table, `[rows, heads, eff]`, built here
                // rather than sliced out of a whole-sequence one. The
                // whole-sequence version is `tokens * heads * eff` floats --
                // 13.2 GiB at 100k tokens, LARGER than the score block it feeds,
                // and it would have become the ceiling the moment the scores
                // stopped being it.
                let rel = r
                    .clone()
                    .slice([lo..hi, 0..heads * d.d_rel])
                    .reshape([rows * heads, d.d_rel])
                    .matmul(rel_proj.clone())
                    .reshape([rows, heads, eff])
                    * tau.clone().slice([lo..hi]).reshape([rows, 1, 1]);

                // `q_block @ k^T` raw, then scaled, biased and masked IN PLACE
                // by one kernel. `handle_of` consumes the tensor, so after that
                // line there is exactly one name for the buffer and writing
                // through it is not aliasing.
                //
                // `lo` goes to the kernel because the causal predicate, the
                // window and the relative distance are all functions of the
                // ABSOLUTE query position. A block that used its local row
                // index would attend to the wrong keys everywhere except block
                // zero -- which is exactly the block a small test exercises.
                let raw = qh.clone().slice([0..heads, lo..hi, 0..head_dim]).matmul(kt.clone());
                let rel_h = handle_of(rel);
                let (s_h, st) = strided_of3(raw);
                score_epilogue_launch(
                    &client,
                    &s_h,
                    &rel_h,
                    heads,
                    rows,
                    lo,
                    tokens,
                    eff,
                    st,
                    d.scaling(),
                    mask_window,
                );
                let scores =
                    tensor_strided3(client.clone(), dev.clone(), s_h, [heads, rows, tokens], st);
                let probs = burn::tensor::activation::softmax(scores, 2);
                parts.push(
                    probs.matmul(vh.clone()).swap_dims(0, 1).reshape([rows, heads * head_dim]),
                );
            }
            // One block is the whole sequence at short context, and `cat` of a
            // single tensor is a copy of it.
            match parts.len() {
                1 => parts.pop().expect("one block"),
                _ => Tensor::cat(parts, 0),
            }
        }
    };

    let mut cache = AttnCache {
        k,
        v,
        k_pre: conv_history(k_pre, d.kernel),
        v_pre: conv_history(v_pre, d.kernel),
        base: 0,
        pending: None,
    };
    trim(&mut cache, window);
    (linear_bf16(out, &w.wo), cache)
}

/// How far a decode step rounds its context length up, from `INK_KV_PAD`.
///
/// 64 by default, which is what turns a 40-step generation from 81 in-loop
/// kernel compilations into a handful; `1` is the unpadded arm, which is
/// exactly what [`attention_step`] did before. Read once per process: this sits
/// inside the token loop, once per attention layer per step.
fn kv_pad_bucket() -> usize {
    static BUCKET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUCKET.get_or_init(|| {
        std::env::var("INK_KV_PAD")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n >= 1)
            .unwrap_or(64)
    })
}

/// One generated token through one attention layer, reading the cache.
///
/// The whole point of the cache: `x` is the single new position, `pos` is its
/// absolute index in the sequence, and the prefix is never recomputed. The
/// cache is advanced in place — the new K and V are appended, the short
/// convolution histories roll forward, and a windowed layer forgets its oldest
/// key.
///
/// `pos` is a parameter and not `cache.len()` because those are different
/// numbers the moment a window drops a key, and log scaling and the relative
/// bias are both functions of the **absolute** position. Deriving one from the
/// other works for exactly as long as the sequence is shorter than the window.
///
/// No `mask` argument: causality is structural here — every cached key precedes
/// `pos` — and the window is applied against the retained distances directly.
///
/// Concrete on [`Bk`] for the same reason [`short_conv_step`] is: the two
/// convolutions it runs on K and V are that function, and that function is a
/// raw cubecl kernel now.
pub fn attention_step(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    pos: usize,
    window: Option<usize>,
    cache: &mut AttnCache<Bk>,
) -> Tensor<Bk, 2> {
    use crate::models::inkling::config::AttnKind;

    let [rows, hidden] = x.dims();
    assert_eq!(rows, 1, "a decode step feeds exactly one token, got {rows}");
    assert_eq!(hidden, d.hidden, "x is [_, {hidden}] but the config says {}", d.hidden);
    assert!(pos >= cache.base + cache.len(), "position {pos} is already cached");
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(groups * kv_heads, heads, "{heads} heads do not divide into {kv_heads} kv heads");

    let q = linear_bf16(x.clone(), &w.wq);
    let (k_new, k_hist) =
        short_conv_step(cache.k_pre.clone(), linear_bf16(x.clone(), &w.wk), w.k_sconv.clone());
    let (v_new, v_hist) =
        short_conv_step(cache.v_pre.clone(), linear_bf16(x.clone(), &w.wv), w.v_sconv.clone());
    cache.k_pre = k_hist;
    cache.v_pre = v_hist;
    let r = linear_bf16(x, &w.wr);

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k_new = head_rms_norm(k_new, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    cache.k = Tensor::cat(vec![cache.k.clone(), k_new], 0);
    cache.v = Tensor::cat(vec![cache.v.clone(), v_new], 0);
    trim(cache, window);
    let len = cache.len();
    let base = cache.base;
    // The context length ROUNDED UP to a bucket, and the length every shape in
    // this function is built at.
    //
    // Every step of a cached generation attends over one more key than the last
    // one, and cubecl keys its compiled kernels on the shapes it is handed: the
    // score matmul gets a fresh tiling blueprint, and every elementwise and
    // reduce kernel over the `[heads, 1, len]` score row gets a fresh line size
    // as `len` walks through the residues mod 4 and the layouts flip between
    // plain and strided. Measured on an 8-layer node with `INK_KV=1`: 81 kernel
    // compilations inside the token loop over 40 steps, and the correlation
    // with the step cost is exact -- every step that compiled nothing took
    // 46.1-47.6 ms and every step that compiled anything took 54-296 ms, about
    // 1.2 s of extra latency at the head of the generation.
    //
    // Rounding to a multiple of `INK_KV_PAD` collapses that: 64 lengths become
    // one shape, and the padded keys are masked to `-inf` so the softmax gives
    // them exactly zero weight. `INK_KV_PAD=1` is the unpadded arm, which is
    // what this function did before.
    let bucket = kv_pad_bucket();
    let padded = len.next_multiple_of(bucket);

    let tau = match (d.kind, log_scaling) {
        (AttnKind::Global, Some(ls)) => ls.tau(pos),
        _ => 1.0,
    };
    let q = q.mul_scalar(tau);

    // One row of what the full lane builds as a [tokens, tokens] table: the
    // backward distance to each retained key, whether the relative table
    // reaches that far, and whether the window admits it at all. Built on the
    // host because `len` is the context length, not a matrix.
    //
    // `padded`, not `len`: the tail beyond the real keys carries index 0 (in
    // range for the gather, and multiplied out by `valid = 0`) and `-inf` in
    // the mask, so it contributes nothing to the softmax and nothing to the
    // value average. There is always at least one real key, so no row is
    // entirely `-inf`.
    let mut idx = vec![0i32; padded];
    let mut valid = vec![0f32; padded];
    let mut wmask = vec![0f32; padded];
    let mut max_dist = 0usize;
    for j in 0..len {
        // Every retained key is at or before `pos`, so this cannot go negative.
        let dist = pos - (base + j);
        if dist < d.rel_extent {
            idx[j] = dist as i32;
            valid[j] = 1.0;
        }
        if window.is_some_and(|wnd| dist >= wnd) {
            wmask[j] = f32::NEG_INFINITY;
        }
        max_dist = max_dist.max(dist);
    }
    for slot in wmask.iter_mut().take(padded).skip(len) {
        *slot = f32::NEG_INFINITY;
    }
    // Bucketed for the same reason: `eff` grows one per step until it saturates
    // at `rel_extent`, and it is the width of the relative-projection matmul
    // and of the gather it feeds. Rounding up only ever admits COLUMNS the
    // gather does not index, because every `idx` is `< max_dist + 1 <= eff`.
    let eff = d.rel_extent.min(max_dist + 1).next_multiple_of(bucket).min(d.rel_extent);
    let idx: Tensor<Bk, 3, Int> =
        Tensor::from_data(TensorData::new(idx, [1, 1, padded]), &dev).repeat_dim(0, heads);
    let valid: Tensor<Bk, 3> = Tensor::from_data(TensorData::new(valid, [1, 1, padded]), &dev);
    let wmask: Tensor<Bk, 3> = Tensor::from_data(TensorData::new(wmask, [1, 1, padded]), &dev);

    let rel = r
        .reshape([heads, d.d_rel])
        .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
        .reshape([heads, 1, eff])
        .mul_scalar(tau);
    let bias = rel.gather(2, idx) * valid;

    let qh = q.reshape([1, heads, head_dim]).swap_dims(0, 1);
    let expand = |t: Tensor<Bk, 2>| -> Tensor<Bk, 3> {
        t.reshape([padded, kv_heads, head_dim])
            .swap_dims(0, 1)
            .reshape([kv_heads, 1, padded, head_dim])
            .repeat_dim(1, groups)
            .reshape([heads, padded, head_dim])
    };
    // Zeros, not uninitialized rows: the padded keys score 0 against any query
    // (harmless, the mask removes them) but the padded VALUES are multiplied by
    // a probability of exactly zero, and `0 * NaN` is NaN.
    let pad_rows = |t: Tensor<Bk, 2>| -> Tensor<Bk, 2> {
        if padded == len {
            return t;
        }
        let dim = t.dims()[1];
        Tensor::cat(vec![t, Tensor::zeros([padded - len, dim], &dev)], 0)
    };
    let kh = expand(pad_rows(cache.k.clone()));
    let vh = expand(pad_rows(cache.v.clone()));

    let scores = qh.matmul(kh.swap_dims(1, 2)).mul_scalar(d.scaling()) + bias + wmask;
    let probs = burn::tensor::activation::softmax(scores, 2);
    let out = probs.matmul(vh).swap_dims(0, 1).reshape([1, heads * head_dim]);
    linear_bf16(out, &w.wo)
}

/// SEVERAL generated positions through one attention layer, reading the cache.
///
/// [`attention_step`] with `rows > 1`, which is the shape speculative decoding
/// verifies in: the accepted token followed by `k` drafts, all attending to a
/// prefix the cache already holds and to each other causally. It is a separate
/// function rather than a relaxed assertion on that one because everything
/// per-position becomes per-ROW — the log-scaling factor, the relative
/// distance, the visibility of every other new row — and a one-row function
/// that happens to work for two is a function whose mask nobody checked.
///
/// `pos0` is the ABSOLUTE position of row 0; row `i` sits at `pos0 + i`.
///
/// The batch is left PENDING: nothing is trimmed and nothing is final until
/// [`AttnCache::commit`] says how many of these rows the verifier kept. A
/// caller that forgets to commit gets a cache that grows past its window, which
/// the next call's `pending` assertion turns into a failure rather than a slow
/// leak.
pub fn attention_steps(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    pos0: usize,
    window: Option<usize>,
    cache: &mut AttnCache<Bk>,
) -> Tensor<Bk, 2> {
    use crate::models::inkling::config::AttnKind;

    let [rows, hidden] = x.dims();
    assert!(rows >= 1, "a batched step feeds at least one token");
    assert_eq!(hidden, d.hidden, "x is [_, {hidden}] but the config says {}", d.hidden);
    assert!(pos0 >= cache.base + cache.len(), "position {pos0} is already cached");
    assert!(cache.pending.is_none(), "a speculative batch is still uncommitted");
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(groups * kv_heads, heads, "{heads} heads do not divide into {kv_heads} kv heads");

    let q = linear_bf16(x.clone(), &w.wq);
    // The convolution over the batch, taps and all: the `kernel - 1` history
    // rows the cache carries, then this batch's own projections. Rows
    // `kernel - 1 ..` of that see a full window of real inputs, which is
    // exactly the rows this batch is asking for — the front-padding
    // [`short_conv`] applies is never reached.
    let k_all = Tensor::cat(vec![cache.k_pre.clone(), linear_bf16(x.clone(), &w.wk)], 0);
    let v_all = Tensor::cat(vec![cache.v_pre.clone(), linear_bf16(x.clone(), &w.wv)], 0);
    let hist = d.kernel - 1;
    let kdim = k_all.dims()[1];
    let vdim = v_all.dims()[1];
    let k_new = short_conv(k_all.clone(), w.k_sconv.clone()).slice([hist..hist + rows, 0..kdim]);
    let v_new = short_conv(v_all.clone(), w.v_sconv.clone()).slice([hist..hist + rows, 0..vdim]);
    let r = linear_bf16(x, &w.wr);

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k_new = head_rms_norm(k_new, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    cache.k = Tensor::cat(vec![cache.k.clone(), k_new], 0);
    cache.v = Tensor::cat(vec![cache.v.clone(), v_new], 0);
    cache.k_pre = k_all.clone().slice([rows..rows + hist, 0..kdim]);
    cache.v_pre = v_all.clone().slice([rows..rows + hist, 0..vdim]);
    cache.pending = Some(Pending { k_pre: k_all, v_pre: v_all, rows });

    let len = cache.len();
    let base = cache.base;
    let bucket = kv_pad_bucket();
    let padded = len.next_multiple_of(bucket);

    let taus: Vec<f32> = (0..rows)
        .map(|i| match (d.kind, log_scaling) {
            (AttnKind::Global, Some(ls)) => ls.tau(pos0 + i),
            _ => 1.0,
        })
        .collect();
    let tau: Tensor<Bk, 1> = Tensor::from_data(TensorData::new(taus, [rows]), &dev);
    let q = q * tau.clone().reshape([rows, 1]);

    // One row of [`attention_step`]'s tables per new position. Three things at
    // once, and all three are per (row, key): whether the relative table
    // reaches that far, whether the window admits it, and — new here, because a
    // one-row step could not need it — whether the key is in the row's FUTURE.
    // Causality inside the batch is not structural the way it is for a single
    // position, and the drafts are exactly the keys a wrong sign would leak.
    let mut idx = vec![0i32; rows * padded];
    let mut valid = vec![0f32; rows * padded];
    let mut wmask = vec![0f32; rows * padded];
    let mut max_dist = 0usize;
    for i in 0..rows {
        let pos = pos0 + i;
        for j in 0..padded {
            let cell = i * padded + j;
            if j >= len {
                wmask[cell] = f32::NEG_INFINITY;
                continue;
            }
            let abs = base + j;
            if abs > pos {
                wmask[cell] = f32::NEG_INFINITY;
                continue;
            }
            let dist = pos - abs;
            if dist < d.rel_extent {
                idx[cell] = dist as i32;
                valid[cell] = 1.0;
            }
            if window.is_some_and(|wnd| dist >= wnd) {
                wmask[cell] = f32::NEG_INFINITY;
            }
            max_dist = max_dist.max(dist);
        }
    }
    let eff = d.rel_extent.min(max_dist + 1).next_multiple_of(bucket).min(d.rel_extent);
    let idx: Tensor<Bk, 3, Int> =
        Tensor::from_data(TensorData::new(idx, [1, rows, padded]), &dev).repeat_dim(0, heads);
    let valid: Tensor<Bk, 3> =
        Tensor::from_data(TensorData::new(valid, [1, rows, padded]), &dev);
    let wmask: Tensor<Bk, 3> =
        Tensor::from_data(TensorData::new(wmask, [1, rows, padded]), &dev);

    let rel = (r
        .reshape([rows * heads, d.d_rel])
        .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
        .reshape([rows, heads, eff])
        .swap_dims(0, 1))
        * tau.reshape([1, rows, 1]);
    let bias = rel.gather(2, idx) * valid;

    let qh = q.reshape([rows, heads, head_dim]).swap_dims(0, 1);
    let expand = |t: Tensor<Bk, 2>| -> Tensor<Bk, 3> {
        t.reshape([padded, kv_heads, head_dim])
            .swap_dims(0, 1)
            .reshape([kv_heads, 1, padded, head_dim])
            .repeat_dim(1, groups)
            .reshape([heads, padded, head_dim])
    };
    let pad_rows = |t: Tensor<Bk, 2>| -> Tensor<Bk, 2> {
        if padded == len {
            return t;
        }
        let dim = t.dims()[1];
        Tensor::cat(vec![t, Tensor::zeros([padded - len, dim], &dev)], 0)
    };
    let kh = expand(pad_rows(cache.k.clone()));
    let vh = expand(pad_rows(cache.v.clone()));

    let scores = qh.matmul(kh.swap_dims(1, 2)).mul_scalar(d.scaling()) + bias + wmask;
    let probs = burn::tensor::activation::softmax(scores, 2);
    let out = probs.matmul(vh).swap_dims(0, 1).reshape([rows, heads * head_dim]);
    linear_bf16(out, &w.wo)
}

// `dense_mlp`, `shared_experts` and `shared_experts_dev` were here and are
// gone. They took `Tensor<B, 2>` weights, which on this backend means f32,
// which means every BF16 leaf on the way to them was doubled -- 4.88 GiB of
// device f32 on the 20-layer head to hold 2.44 GiB of stored weight. Their
// replacements multiply the stored BF16 through `mma.sync...bf16`
// (`inkling_forward::dense_mlp_bf16` / `shared_experts_bf16`), and they live
// beside the caller because `Bf16W` is a raw cubecl handle and this file is
// generic over `B: Backend`.
//
// Not kept as a control. The header of this file already says why: a slower
// implementation you can still call is one you will call by accident, and this
// pair had no caller left the moment the BF16 path landed.

/// The KV cache against the lane it is supposed to be an optimization of.
///
/// The oracle here is [`attention`] itself, which `inkling_real_gate` holds to
/// a Python-generated bundle. So these tests do not re-litigate whether the
/// layer is right; they check the only thing a cache can get wrong, which is
/// whether feeding one token at a time reproduces feeding all of them.
///
/// Deliberately at dimensions the real checkpoint never reaches: a window of 5
/// against 11 tokens, so the cache actually forgets, and a log-scaling floor of
/// 4 so `tau` varies per position. On the real config the window is 512 and the
/// floor 128000, which means an end-to-end run of a dozen tokens exercises
/// neither path — a passing generation would prove nothing about either.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inkling::attn::{AttnDims, LogScaling};
    use crate::models::inkling::config::AttnKind;

    // The only backend there is. These tests compare a cached lane against an
    // uncached one on the SAME device, so what they need from a backend is
    // that it exists — and after the feature collapse exactly one does.
    type B = burn::backend::Cuda<f32>;

    /// How far the cached lane may differ from the uncached one on a GLOBAL
    /// layer, where both sides build their scores with the same Burn matmul and
    /// the only difference is the order of the additions.
    const CACHE_TOLERANCE_GLOBAL: f32 = 2e-5;

    /// The same for a LOCAL layer, where the two sides no longer share an
    /// implementation.
    ///
    /// Prefill goes through [`crate::models::inkling::banded`], which
    /// accumulates `q . k` in f32 on the CUDA cores. A decode step goes through
    /// Burn's matmul, which on this runtime is **TF32** -- ten mantissa bits, not
    /// twenty-three. [`f32_matmul_is_tf32_on_this_runtime`] measures it at 9.3e-4
    /// relative to the largest term of a 128-deep product, and through a softmax
    /// on these deliberately small synthetic weights that reaches 2.2e-2 of the
    /// output.
    ///
    /// The band is the ACCURATE side, not the tolerant one. That is not an
    /// assumption: `banded::device_tests::the_cached_tests_shape_against_f64`
    /// holds the kernel to 2e-5 against an f64 host reference at exactly these
    /// shapes, and it is the dense triangle that drifts from f64 by 2.2e-2.
    ///
    /// 5e-2 is set against the SABOTAGE floor rather than against the noise:
    /// [`dropping_the_conv_history_is_caught`] moves the answer by 1.156, so
    /// what these tests exist to catch still has more than twenty times the
    /// margin it needs.
    const CACHE_TOLERANCE_LOCAL: f32 = 5e-2;

    /// Deterministic filler. A fixed pattern rather than a seeded RNG so a
    /// failure is reproducible from the source alone.
    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.7919 + seed).sin() * 0.5 + (i as f32 * 0.1237).cos() * 0.25)
            .collect()
    }

    /// Small, but no longer arbitrary.
    ///
    /// These were `hidden 8, head_dim 2, d_rel 3`, which is the smallest thing
    /// that exercises grouped KV heads. It is also a shape `m16n8k16` cannot
    /// tile — `k` must be a multiple of 16 and `n` of 8 — and since the five
    /// projections became [`Bf16W`] the weights here have to be shapes the
    /// instruction can actually multiply. Doubling `hidden` and `head_dim` and
    /// taking `d_rel` to 4 does that while keeping every structural property
    /// the tests rely on: 4 heads over 2 KV heads is still `groups = 2`, so the
    /// repeat-in-place indexing is still under test.
    ///
    ///     wq [16, 16]   wk/wv [8, 16]   wr [16, 16]   wo [16, 16]
    fn dims(kind: AttnKind, rel_extent: usize) -> AttnDims {
        AttnDims {
            hidden: 16,
            heads: 4,
            kv_heads: 2,
            head_dim: 4,
            d_rel: 4,
            rel_extent,
            kernel: 4,
            rms_eps: 1e-6,
            kind,
        }
    }

    fn weights(d: &AttnDims, dev: &burn::backend::cuda::CudaDevice) -> AttnWeightsDev {
        let m = |rows: usize, cols: usize, seed: f32| -> Tensor<B, 2> {
            Tensor::from_data(TensorData::new(fill(rows * cols, seed), [rows, cols]), dev)
        };
        let v = |n: usize, seed: f32| -> Tensor<B, 1> {
            Tensor::from_data(TensorData::new(fill(n, seed), [n]), dev)
        };
        // The same filler, rounded to the BF16 the projections now multiply as.
        // `bf16::from_f32` is round-to-nearest-even, which is what the device
        // cast and `torch.Tensor.to(torch.bfloat16)` both do, so both lanes
        // under comparison see identical operand bits.
        let client = client_of(&m(1, 1, 0.0));
        let w16 = |rows: usize, cols: usize, seed: f32| -> Bf16W {
            assert!(Bf16W::tileable(rows, cols), "test weight {rows}x{cols} does not tile");
            let mut bytes = Vec::with_capacity(rows * cols * 2);
            for x in fill(rows * cols, seed) {
                bytes.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes());
            }
            Bf16W { h: client.create_from_slice(&bytes), n: rows, k: cols, align: 16 }
        };
        let (q_w, kv_w) = (d.heads * d.head_dim, d.kv_heads * d.head_dim);
        AttnWeightsDev {
            wq: w16(q_w, d.hidden, 0.1),
            wk: w16(kv_w, d.hidden, 0.2),
            wv: w16(kv_w, d.hidden, 0.3),
            wr: w16(d.heads * d.d_rel, d.hidden, 0.4),
            wo: w16(d.hidden, q_w, 0.5),
            k_sconv: m(kv_w, d.kernel, 0.6),
            v_sconv: m(kv_w, d.kernel, 0.7),
            q_norm: v(d.head_dim, 0.8),
            k_norm: v(d.head_dim, 0.9),
            rel_proj: m(d.d_rel, d.rel_extent, 1.0),
        }
    }

    /// Run `tokens` positions two ways and return the largest absolute
    /// disagreement over the decoded rows, plus the decoded rows themselves.
    fn compare(
        kind: AttnKind,
        rel_extent: usize,
        window: Option<usize>,
        ls: Option<LogScaling>,
        tokens: usize,
        prefill: usize,
        sabotage_conv_history: bool,
    ) -> f32 {
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = dims(kind, rel_extent);
        let w = weights(&d, &dev);
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(tokens * d.hidden, 2.5), [tokens, d.hidden]),
            &dev,
        );

        let full = attention(xs.clone(), &w, &d, ls, window);

        let head = xs.clone().slice([0..prefill, 0..d.hidden]);
        let (_, mut cache) = attention_prefill(head, &w, &d, ls, window, window);
        if sabotage_conv_history {
            // The mutation this cache exists to avoid: keep K and V, forget
            // what the short convolution still needs to see.
            cache.k_pre = cache.k_pre.clone().zeros_like();
            cache.v_pre = cache.v_pre.clone().zeros_like();
        }

        let mut worst = 0f32;
        for pos in prefill..tokens {
            let step = attention_step(
                xs.clone().slice([pos..pos + 1, 0..d.hidden]),
                &w,
                &d,
                ls,
                pos,
                window,
                &mut cache,
            );
            let want = full.clone().slice([pos..pos + 1, 0..d.hidden]);
            let diff = (step - want).abs().max().into_scalar();
            worst = worst.max(diff);
        }
        worst
    }

    /// Burn's f32 matmul on this runtime is NOT f32, and this is the tripwire.
    ///
    /// It compares `Tensor::matmul` against the same product accumulated in f64
    /// on the host. TF32 carries ten mantissa bits and lands near 1e-3; true f32
    /// carries twenty-three and lands near 1e-7. Measured here: 9.3e-4.
    ///
    /// The assertion is deliberately the "still imprecise" direction. It is what
    /// [`CACHE_TOLERANCE_LOCAL`] is sized for, and if this test ever FAILS the
    /// runtime has moved to a real f32 product and that tolerance can go back to
    /// 2e-5. A failure here is good news, not a regression.
    #[test]
    fn f32_matmul_is_tf32_on_this_runtime() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let (m, k, n) = (32usize, 128usize, 32usize);
        let a = fill(m * k, 1.3);
        let b = fill(k * n, 2.1);
        let at: Tensor<B, 2> = Tensor::from_data(TensorData::new(a.clone(), [m, k]), &dev);
        let bt: Tensor<B, 2> = Tensor::from_data(TensorData::new(b.clone(), [k, n]), &dev);
        let got: Vec<f32> = at.matmul(bt).into_data().to_vec().unwrap();
        let mut worst = 0f64;
        let mut scale = 0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for t in 0..k {
                    acc += a[i * k + t] as f64 * b[t * n + j] as f64;
                }
                worst = worst.max((got[i * n + j] as f64 - acc).abs());
                scale = scale.max(acc.abs());
            }
        }
        let rel = worst / scale;
        println!("f32 matmul worst absolute error {worst:e} against a largest term of {scale:e} -> {rel:e}");
        assert!(
            rel > 1e-5,
            "Burn's f32 matmul now agrees with f64 to {rel:e}: it is a real f32 product, and \
             CACHE_TOLERANCE_LOCAL can go back to CACHE_TOLERANCE_GLOBAL"
        );
    }

    /// The band and the dense triangle, on the same weights, over the shapes
    /// the cached tests use.
    ///
    /// This is the check that was missing when the band landed: the kernel has
    /// its own f64 reference in `banded.rs`, and a kernel agreeing with a
    /// reference written beside it says nothing about whether it agrees with the
    /// lane it REPLACES. It did not -- by up to 2.2e-2 -- and running this is how
    /// the TF32 matmul was found. Prints the disagreement per configuration and
    /// per row, so a failure names which one moved.
    ///
    /// The bound is [`CACHE_TOLERANCE_LOCAL`], and it is a bound on TF32, not on
    /// the band: the tight gate on the band is in `banded.rs`, against f64.
    /// A blocked dense lane must answer what an unblocked one does.
    ///
    /// The one thing query blocking can get wrong is the query POSITION: the
    /// causal predicate, the sliding window and the relative distance are all
    /// functions of the absolute position, and a block that used its local row
    /// index instead would attend to the wrong keys in every block but the
    /// first. That failure is invisible at any shape small enough to fit in one
    /// block, which is every other test in this module, so the block size is a
    /// parameter here and the sizes below are chosen NOT to divide the
    /// sequence -- an off-by-one in the last, short block reads the same as a
    /// correct one when the blocks are even.
    ///
    /// Both arms are the same kernels on the same weights, so the tolerance is
    /// the f64 one and not the cross-implementation `CACHE_TOLERANCE_LOCAL`:
    /// only the matmul tiling differs, and TF32 accumulation order with it.
    #[test]
    fn query_blocks_agree_with_one_block() {
        let dev = burn::backend::cuda::CudaDevice::default();
        for (kind, rel_extent, win) in [
            (AttnKind::Global, 16usize, None),
            (AttnKind::Local, 8, Some(6usize)),
        ] {
            let d = dims(kind, rel_extent);
            let w = weights(&d, &dev);
            let ls = match kind {
                AttnKind::Global => Some(LogScaling { n_floor: 8.0, alpha: 0.5 }),
                AttnKind::Local => None,
            };
            for tokens in [37usize, 64, 91] {
                let xs: Tensor<B, 2> = Tensor::from_data(
                    TensorData::new(fill(tokens * d.hidden, 0.05), [tokens, d.hidden]),
                    &dev,
                );
                let whole =
                    attention_prefill_dense(xs.clone(), &w, &d, ls, win, win, Some(tokens))
                        .0
                        .into_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .expect("f32 rows");
                for block in [1usize, 5, 8, 13, 32] {
                    if block >= tokens {
                        continue;
                    }
                    let parts =
                        attention_prefill_dense(xs.clone(), &w, &d, ls, win, win, Some(block))
                            .0
                            .into_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .expect("f32 rows");
                    let worst = whole
                        .iter()
                        .zip(&parts)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        worst < 2e-5,
                        "{kind:?}, {tokens} tokens, block {block}: worst {worst:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn banded_and_dense_agree_to_tf32() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let mut worst_of_all = 0f32;
        for (rel_extent, win, tokens) in [
            (16usize, 16usize, 11usize), // window past the sequence: a full triangle
            (16, 16, 4),
            (5, 16, 11),  // triangle, table shorter than the sequence
            (16, 5, 11),  // clipped band, table longer than the window
            (5, 5, 11),
            (5, 3, 11),
            (5, 5, 4),
            (5, 5, 2),
            (8, 6, 20),
        ] {
            let d = dims(AttnKind::Local, rel_extent);
            let w = weights(&d, &dev);
            let xs: Tensor<B, 2> = Tensor::from_data(
                TensorData::new(fill(tokens * d.hidden, 2.5), [tokens, d.hidden]),
                &dev,
            );
            let band = attention_prefill(xs.clone(), &w, &d, None, Some(win), Some(win)).0;
            let dense = attention_prefill_dense(xs, &w, &d, None, Some(win), Some(win), None).0;
            let per_row: Vec<f32> = (0..tokens)
                .map(|r| {
                    (band.clone().slice([r..r + 1, 0..d.hidden])
                        - dense.clone().slice([r..r + 1, 0..d.hidden]))
                    .abs()
                    .max()
                    .into_scalar()
                })
                .collect();
            let diff = per_row.iter().cloned().fold(0f32, f32::max);
            println!("rel_extent={rel_extent} window={win} tokens={tokens} -> {diff}  rows {per_row:?}");
            worst_of_all = worst_of_all.max(diff);
        }
        assert!(
            worst_of_all < CACHE_TOLERANCE_LOCAL,
            "the band disagrees with the triangle by {worst_of_all}, which is more than TF32 \
             explains"
        );
    }

    /// A global layer with log scaling that actually varies, and a relative
    /// table that runs out before the sequence does — so distances past
    /// `rel_extent` must contribute a zero bias rather than a gathered one.
    #[test]
    fn cached_global_matches_full() {
        let ls = Some(LogScaling { n_floor: 4.0, alpha: 0.5 });
        let worst = compare(AttnKind::Global, 5, None, ls, 11, 4, false);
        assert!(worst < CACHE_TOLERANCE_GLOBAL, "cached global attention drifts by {worst}");
    }

    /// A local layer whose window is shorter than the sequence, so the cache
    /// must forget: 11 tokens through a window of 5 drops six keys.
    #[test]
    fn cached_local_matches_full_across_the_window() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 4, false);
        assert!(worst < CACHE_TOLERANCE_LOCAL, "cached windowed attention drifts by {worst}");
    }

    /// The cache must survive a prefill shorter than the convolution kernel,
    /// where the history is mostly the zero padding `short_conv` assumes.
    #[test]
    fn cached_matches_full_from_a_two_token_prefill() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 2, false);
        assert!(
            worst < CACHE_TOLERANCE_LOCAL,
            "cached attention from a short prefill drifts by {worst}"
        );
    }

    /// The batched cached step against the uncached lane, in batches of
    /// `batch`, optionally rolling back `reject` rows of every batch and
    /// re-running them — which is what a rejected draft does.
    fn compare_batched(
        kind: AttnKind,
        rel_extent: usize,
        window: Option<usize>,
        ls: Option<LogScaling>,
        tokens: usize,
        prefill: usize,
        batch: usize,
        reject: usize,
    ) -> f32 {
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = dims(kind, rel_extent);
        let w = weights(&d, &dev);
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(tokens * d.hidden, 2.5), [tokens, d.hidden]),
            &dev,
        );
        let full = attention(xs.clone(), &w, &d, ls, window);
        let (_, mut cache) = attention_prefill(
            xs.clone().slice([0..prefill, 0..d.hidden]),
            &w,
            &d,
            ls,
            window,
            window,
        );

        let mut worst = 0f32;
        let mut pos = prefill;
        while pos < tokens {
            let rows = batch.min(tokens - pos);
            // The rejection arm: run the batch, keep only what a verifier
            // would have kept, then run the SAME positions again. A cache that
            // rolled back wrongly answers differently the second time, and the
            // second answer is the one compared.
            if reject > 0 && rows > reject {
                let _ = attention_steps(
                    xs.clone().slice([pos..pos + rows, 0..d.hidden]),
                    &w,
                    &d,
                    ls,
                    pos,
                    window,
                    &mut cache,
                );
                cache.commit(rows - reject, window);
                let redo = rows - reject;
                let got = attention_steps(
                    xs.clone().slice([pos + redo..pos + rows, 0..d.hidden]),
                    &w,
                    &d,
                    ls,
                    pos + redo,
                    window,
                    &mut cache,
                );
                cache.commit(rows - redo, window);
                let want = full.clone().slice([pos + redo..pos + rows, 0..d.hidden]);
                worst = worst.max((got - want).abs().max().into_scalar());
            } else {
                let got = attention_steps(
                    xs.clone().slice([pos..pos + rows, 0..d.hidden]),
                    &w,
                    &d,
                    ls,
                    pos,
                    window,
                    &mut cache,
                );
                cache.commit(rows, window);
                let want = full.clone().slice([pos..pos + rows, 0..d.hidden]);
                worst = worst.max((got - want).abs().max().into_scalar());
            }
            pos += rows;
        }
        worst
    }

    /// Three positions at a time against the uncached lane, on a global layer
    /// with log scaling that varies per row — so a batch that used one `tau`
    /// for all of its rows would show up here.
    #[test]
    fn batched_global_matches_full() {
        let ls = Some(LogScaling { n_floor: 4.0, alpha: 0.5 });
        let worst = compare_batched(AttnKind::Global, 5, None, ls, 11, 4, 3, 0);
        assert!(worst < CACHE_TOLERANCE_GLOBAL, "batched global attention drifts by {worst}");
    }

    /// The same on a local layer whose window is shorter than the sequence, so
    /// the batch must not forget a key until it knows how long it was.
    #[test]
    fn batched_local_matches_full_across_the_window() {
        let worst = compare_batched(AttnKind::Local, 5, Some(5), None, 11, 4, 3, 0);
        assert!(worst < CACHE_TOLERANCE_LOCAL, "batched windowed attention drifts by {worst}");
    }

    /// Rejection: run three, keep one, re-run the two that were rejected. This
    /// is the speculative path, and it passes only if `commit` restored the
    /// short convolution's memory as well as truncating K and V.
    #[test]
    fn rejected_rows_leave_no_trace() {
        let ls = Some(LogScaling { n_floor: 4.0, alpha: 0.5 });
        let worst = compare_batched(AttnKind::Global, 5, None, ls, 11, 4, 3, 2);
        assert!(worst < CACHE_TOLERANCE_GLOBAL, "rolled-back batch drifts by {worst}");
    }

    /// The same against a window, where rollback and the window's own
    /// forgetting interact.
    #[test]
    fn rejected_rows_leave_no_trace_windowed() {
        let worst = compare_batched(AttnKind::Local, 5, Some(5), None, 11, 4, 3, 2);
        assert!(worst < CACHE_TOLERANCE_LOCAL, "rolled-back windowed batch drifts by {worst}");
    }

    /// A batch of one must agree with [`attention_step`], which is the claim
    /// that makes the two functions one algorithm rather than two.
    #[test]
    fn a_batch_of_one_matches_the_single_step() {
        let ls = Some(LogScaling { n_floor: 4.0, alpha: 0.5 });
        let worst = compare_batched(AttnKind::Global, 5, None, ls, 11, 4, 1, 0);
        assert!(worst < CACHE_TOLERANCE_GLOBAL, "a one-row batch drifts by {worst}");
    }

    /// A gate that cannot fail and a gate that has never failed look identical
    /// from outside. Forgetting the pre-convolution history is the plausible
    /// version of this bug — it leaves K and V cached and correct-looking — so
    /// prove the comparison above would have caught it.
    #[test]
    fn dropping_the_conv_history_is_caught() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 4, true);
        println!("sabotaged conv history moves the answer by {worst}");
        // Against the TOLERANCE, not against a constant: what has to hold is
        // that the bug class these tests exist for is far outside the band the
        // tolerance admits. Measured at 1.156 against a 5e-2 tolerance.
        assert!(
            worst > 20.0 * CACHE_TOLERANCE_LOCAL,
            "sabotaged conv history only moved the answer by {worst}, which is inside the \
             tolerance the windowed tests use"
        );
    }
}
