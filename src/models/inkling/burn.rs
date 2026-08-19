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
pub struct AttnCache<B: Backend> {
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    k_pre: Tensor<B, 2>,
    v_pre: Tensor<B, 2>,
    base: usize,
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
/// `mask` is the additive `[tokens, tokens]` mask — zero where a key is visible
/// and `-inf` where it is not — because a local layer's mask carries the sliding
/// window and a global layer's does not.
///
/// Two things are folded together here that a careless reading separates:
/// log scaling multiplies the query **and** the relative-position bias, and only
/// on global layers; and the bias is zero outside `0 <= q - k < rel_extent`,
/// while causality lives in the mask.
pub fn attention(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    mask: Tensor<Bk, 2>,
) -> Tensor<Bk, 2> {
    attention_prefill(x, w, d, log_scaling, mask, None).0
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
    mask: Tensor<Bk, 2>,
    window: Option<usize>,
) -> (Tensor<Bk, 2>, AttnCache<Bk>) {
    use crate::models::inkling::config::AttnKind;

    let [tokens, hidden] = x.dims();
    assert_eq!(hidden, d.hidden, "x is [_, {hidden}] but the config says {}", d.hidden);
    assert_eq!(mask.dims(), [tokens, tokens], "the mask must be [tokens, tokens]");
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
    let mut idx = vec![0i32; tokens * tokens];
    let mut valid = vec![0f32; tokens * tokens];
    for qi in 0..tokens {
        for ki in 0..tokens {
            let dist = qi as isize - ki as isize;
            if dist >= 0 && (dist as usize) < d.rel_extent {
                idx[qi * tokens + ki] = dist as i32;
                valid[qi * tokens + ki] = 1.0;
            }
        }
    }
    let idx: Tensor<Bk, 3, Int> =
        Tensor::from_data(TensorData::new(idx, [1, tokens, tokens]), &dev).repeat_dim(0, heads);
    let valid: Tensor<Bk, 3> = Tensor::from_data(TensorData::new(valid, [1, tokens, tokens]), &dev);

    let rel = r
        .reshape([tokens * heads, d.d_rel])
        .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
        .reshape([tokens, heads, eff])
        .swap_dims(0, 1)
        * tau.reshape([1, tokens, 1]);
    let bias = rel.gather(2, idx) * valid;

    // [heads, tokens, head_dim]; the KV heads are repeated in place, so head h
    // reads kv head h / groups exactly as the slice lane indexes it.
    let qh = q.reshape([tokens, heads, head_dim]).swap_dims(0, 1);
    let expand = |t: Tensor<Bk, 2>| -> Tensor<Bk, 3> {
        t.reshape([tokens, kv_heads, head_dim])
            .swap_dims(0, 1)
            .reshape([kv_heads, 1, tokens, head_dim])
            .repeat_dim(1, groups)
            .reshape([heads, tokens, head_dim])
    };
    let kh = expand(k.clone());
    let vh = expand(v.clone());

    let scores = qh.matmul(kh.swap_dims(1, 2)).mul_scalar(d.scaling()) + bias
        + mask.reshape([1, tokens, tokens]);
    let probs = burn::tensor::activation::softmax(scores, 2);
    let out = probs
        .matmul(vh)
        .swap_dims(0, 1)
        .reshape([tokens, heads * head_dim]);

    let mut cache = AttnCache {
        k,
        v,
        k_pre: conv_history(k_pre, d.kernel),
        v_pre: conv_history(v_pre, d.kernel),
        base: 0,
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
    use crate::models::inkling::attn::{causal_mask, AttnDims, LogScaling};
    use crate::models::inkling::config::AttnKind;

    // The only backend there is. These tests compare a cached lane against an
    // uncached one on the SAME device, so what they need from a backend is
    // that it exists — and after the feature collapse exactly one does.
    type B = burn::backend::Cuda<f32>;

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

        let full = attention(
            xs.clone(),
            &w,
            &d,
            ls,
            Tensor::from_data(
                TensorData::new(causal_mask(tokens, window), [tokens, tokens]),
                &dev,
            ),
        );

        let head = xs.clone().slice([0..prefill, 0..d.hidden]);
        let (_, mut cache) = attention_prefill(
            head,
            &w,
            &d,
            ls,
            Tensor::from_data(
                TensorData::new(causal_mask(prefill, window), [prefill, prefill]),
                &dev,
            ),
            window,
        );
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

    /// A global layer with log scaling that actually varies, and a relative
    /// table that runs out before the sequence does — so distances past
    /// `rel_extent` must contribute a zero bias rather than a gathered one.
    #[test]
    fn cached_global_matches_full() {
        let ls = Some(LogScaling { n_floor: 4.0, alpha: 0.5 });
        let worst = compare(AttnKind::Global, 5, None, ls, 11, 4, false);
        assert!(worst < 2e-5, "cached global attention drifts by {worst}");
    }

    /// A local layer whose window is shorter than the sequence, so the cache
    /// must forget: 11 tokens through a window of 5 drops six keys.
    #[test]
    fn cached_local_matches_full_across_the_window() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 4, false);
        assert!(worst < 2e-5, "cached windowed attention drifts by {worst}");
    }

    /// The cache must survive a prefill shorter than the convolution kernel,
    /// where the history is mostly the zero padding `short_conv` assumes.
    #[test]
    fn cached_matches_full_from_a_two_token_prefill() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 2, false);
        assert!(worst < 2e-5, "cached attention from a short prefill drifts by {worst}");
    }

    /// A gate that cannot fail and a gate that has never failed look identical
    /// from outside. Forgetting the pre-convolution history is the plausible
    /// version of this bug — it leaves K and V cached and correct-looking — so
    /// prove the comparison above would have caught it.
    #[test]
    fn dropping_the_conv_history_is_caught() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 4, true);
        assert!(worst > 1e-2, "sabotaged conv history only moved the answer by {worst}");
    }
}
