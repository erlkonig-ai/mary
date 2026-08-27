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

use cubecl::server::Handle;

use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::seam::{Bk, client_of, handle_of, tensor_of, tensor_of3};

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
    assert_eq!(
        k, kw,
        "linear_pre_t: x is [_, {k}] but the weight is [{kw}, _]"
    );
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
    assert_eq!(
        k, w.k,
        "linear_bf16: x is [_, {k}] but the weight is [_, {}]",
        w.k
    );
    // Only the hand lane pads, and only because its grid is its tiling; ask
    // rather than assume, or a decode step slices fifteen rows that were never
    // computed.
    let rows = rows_for(w.align, m);
    let client = client_of(&x);
    let dev = x.device();
    // The activation may arrive f32 or BF16 -- the narrow lane's normed
    // residual stream is BF16 by the time it reaches a projection -- and the
    // difference is a whole `[m, k]` f32 buffer that would exist only to be
    // cast back. `handle_of_any` reports which it is instead of asserting one
    // of them.
    let (xh, xdt) = crate::models::inkling::seam::handle_of_any(x);
    let out = match xdt {
        burn::tensor::DType::BF16 => {
            crate::models::inkling::bf16gemm::linear_bf16_narrow(&client, &xh, w, m)
        }
        burn::tensor::DType::F32 => {
            crate::models::inkling::bf16gemm::linear_bf16(&client, &xh, w, m)
        }
        other => panic!("linear_bf16: no lane for a {other:?} activation"),
    };
    tensor_of(client, dev, out, rows, w.n).slice([0..m, 0..w.n])
}

/// An NVFP4 weight: E2M1 codes, one E4M3 scale per [`fp4quant::GROUP`], and the
/// single f32 that scales the whole tensor.
///
/// The twin of [`Bf16W`], and it exists for the same reason that one does. The
/// published checkpoint quantises `mlp.experts.w13_weight` and `w2_weight` and
/// NOTHING else -- 39 layers of routed experts carry `.scale` tensors, and the
/// attention projections and the shared experts do not. So the shared experts
/// are 100.7 MB of BF16 a layer against the six ROUTED experts' 84.9 MB of
/// NVFP4: two tensors that cost more bandwidth than the six that do the
/// routing. A decode step on this part is bound on streaming weights, and
/// `scale2` is `1.0` for anything quantised here rather than by the publisher,
/// because [`fp4quant::quantize_nvfp4_bf16`] folds the whole range into the
/// per-16 E4M3 scales the way the activation quantiser already does.
pub struct PackedW {
    pub codes: Handle,
    pub scales: Handle,
    /// Output rows.
    pub n: usize,
    /// Input columns.
    pub k: usize,
    /// The tensor-wide scale. `1.0` unless the checkpoint supplied one.
    pub scale2: f32,
    /// Whether `codes` and `scales` are written in `m16n8k16` MMA-fragment
    /// order rather than row-major `[n, k/8]` / `[n, k/16]`.
    ///
    /// Truth about the BYTES, set only where the permutation actually ran --
    /// never a request for it. A flag that can disagree with the bytes is how
    /// a kernel comes to read the wrong layout and produce NUMBERS instead of
    /// an error, so every consumer branches on this and
    /// [`linear_fp4`] refuses a weight carrying it outright: the k16
    /// permutation is not `fp4gemm`'s k64 one and the two are not
    /// interchangeable.
    pub swizzled: bool,
}

/// One projection's weight, in whichever precision it is held.
///
/// The dispatch is on the WEIGHT and not on the call site, so that moving a
/// tensor between lanes is a change at the bind and nowhere else. That is the
/// property that matters here: the same move is wanted for the five attention
/// projections, and a second copy of the shared-expert plumbing is exactly what
/// this avoids.
pub enum ProjW {
    Bf16(Bf16W),
    Fp4(PackedW),
    /// The same NVFP4 bytes as [`Self::Fp4`], multiplied against a BF16
    /// activation instead of a quantised one.
    ///
    /// Not a third weight format -- the payload is the identical [`PackedW`],
    /// and moving a tensor between these two variants changes nothing on the
    /// device. What it changes is whose numerics the ACTIVATION follows. The
    /// checkpoint asks for a 4-bit activation on the routed experts and says
    /// nothing about anything else, so `Fp4` is right there and this is right
    /// everywhere the publisher left BF16: the sink experts, the attention
    /// projections, and the unembedding.
    W4a16(PackedW),
}

impl ProjW {
    /// Output rows.
    pub fn n(&self) -> usize {
        match self {
            Self::Bf16(w) => w.n,
            Self::Fp4(w) | Self::W4a16(w) => w.n,
        }
    }

    /// Input columns.
    pub fn k(&self) -> usize {
        match self {
            Self::Bf16(w) => w.k,
            Self::Fp4(w) | Self::W4a16(w) => w.k,
        }
    }
}

/// `x @ w^T` for a weight in whichever lane it is bound to.
pub fn linear_w(x: Tensor<Bk, 2>, w: &ProjW) -> Tensor<Bk, 2> {
    match w {
        ProjW::Bf16(b) => linear_bf16(x, b),
        ProjW::Fp4(p) => linear_fp4(x, p),
        ProjW::W4a16(p) => linear_w4a16(x, p),
    }
}

/// `x @ w^T` against an NVFP4 weight, quantising the activation on the device.
///
/// The activation is padded to [`fp4gemm::MTILE`] and sliced back afterwards.
/// That padding is fifteen rows of zeros on a decode step and it is NOT the
/// hazard [`crate::models::inkling::bf16gemm::rows_for`] documents for the hand
/// lane: this lane is bound on the WEIGHT stream, which a wider `m` reads
/// exactly once either way, and the routed experts already run this shape at
/// 198 GB/s of a measured 242.9 GB/s bus.
pub fn linear_fp4(x: Tensor<Bk, 2>, w: &PackedW) -> Tensor<Bk, 2> {
    use crate::models::inkling::fp4gemm::{MTILE, fp4_linear_launch};
    use crate::models::inkling::fp4quant::{quantize_nvfp4, quantize_nvfp4_bf16};
    let [m, k] = x.dims();
    assert_eq!(
        k, w.k,
        "linear_fp4: x is [_, {k}] but the weight is [_, {}]",
        w.k
    );
    assert!(
        !w.swizzled,
        "linear_fp4 was handed a weight in m16n8k16 fragment order. fp4_linear is m16n8k64 \
         and would read those bytes as if they were row-major -- silently, and as numbers. \
         The k16 permutation belongs to the W4A16 lane."
    );
    let m_pad = m.div_ceil(MTILE) * MTILE;
    let client = client_of(&x);
    let dev = x.device();
    let x = if m_pad == m {
        x
    } else {
        Tensor::cat(vec![x, Tensor::zeros([m_pad - m, k], &dev)], 0)
    };
    // Same two arrivals the BF16 lane handles: the narrow lane's normed
    // residual stream reaches a projection as BF16, the wide one as f32.
    // Quantising from BF16 costs nothing the destination was not already
    // paying -- four bits with one E4M3 scale per sixteen.
    let (xh, xdt) = crate::models::inkling::seam::handle_of_any(x);
    let (a, asc) = match xdt {
        burn::tensor::DType::BF16 => quantize_nvfp4_bf16(&client, &xh, m_pad, k),
        burn::tensor::DType::F32 => quantize_nvfp4(&client, &xh, m_pad, k),
        other => panic!("linear_fp4: no lane for a {other:?} activation"),
    };
    let out = fp4_linear_launch(
        &client, &a, &asc, &w.codes, &w.scales, m_pad, k, w.n, w.scale2,
    );
    tensor_of(client, dev, out, m_pad, w.n).slice([0..m, 0..w.n])
}

/// `x @ w^T` against an NVFP4 weight with the activation left BF16.
///
/// [`linear_fp4`] minus the activation quantiser. The weight is the same
/// [`PackedW`] and reaches the MMA as the same four bits; what does not happen
/// is the `quantize_nvfp4*` launch and the `input_quantizer` numerics it
/// implies, which the publisher only calibrated for the routed experts.
///
/// The activation arrives f32 (the wide residual stream) or BF16 (the narrow
/// one). Either way the M padding is done by the same kernel that does the
/// cast, so the padding rows never exist as a separate buffer -- the hazard
/// [`linear_fp4`] pays with a `Tensor::cat`.
pub fn linear_w4a16(x: Tensor<Bk, 2>, w: &PackedW) -> Tensor<Bk, 2> {
    use crate::models::inkling::bf16gemm::{pad_bf16_launch, to_bf16_launch};
    use crate::models::inkling::w4a16gemm::{MTILE, w4a16_linear_launch};
    let [m, k] = x.dims();
    assert_eq!(
        k, w.k,
        "linear_w4a16: x is [_, {k}] but the weight is [_, {}]",
        w.k
    );
    let m_pad = m.div_ceil(MTILE) * MTILE;
    let client = client_of(&x);
    let dev = x.device();
    let (xh, xdt) = crate::models::inkling::seam::handle_of_any(x);
    let a = match xdt {
        burn::tensor::DType::BF16 if m_pad == m => xh,
        burn::tensor::DType::BF16 => pad_bf16_launch(&client, &xh, m * k, m_pad * k),
        burn::tensor::DType::F32 => to_bf16_launch(&client, &xh, m * k, m_pad * k),
        other => panic!("linear_w4a16: no lane for a {other:?} activation"),
    };
    // The permutation is a change of LAYOUT, so the branch is on the bytes and
    // not on a setting: `w.swizzled` is set where the permutation ran, and a
    // weight that was never permuted takes the row-major lane here however the
    // knob is set.
    let out = if w.swizzled {
        crate::models::inkling::w4a16gemm::w4a16_linear_swz_launch(
            &client, &a, &w.codes, &w.scales, m_pad, k, w.n, true, w.scale2,
        )
    } else {
        w4a16_linear_launch(&client, &a, &w.codes, &w.scales, m_pad, k, w.n, w.scale2)
    };
    tensor_of(client, dev, out, m_pad, w.n).slice([0..m, 0..w.n])
}

/// RMS normalization with a per-feature gain.
///
/// Divides by `sqrt(var + eps)` rather than multiplying by its reciprocal: on
/// some backends `recip` dispatches to an approximate SIMD reciprocal, which
/// cost K3 about fourteen bits of accuracy before it was caught. Same hazard
/// here, same avoidance.
pub fn rms_norm<B: Backend>(x: Tensor<B, 2>, gain: Tensor<B, 1>, eps: f64) -> Tensor<B, 2> {
    let [_, w] = x.dims();
    assert_eq!(
        gain.dims()[0],
        w,
        "rms_norm: gain is {} wide, input {w}",
        gain.dims()[0]
    );
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

/// [`conv_history`] after a TREE verify pass: the `kernel - 1` window rows the
/// next position must carry, given the batch rows the verifier kept.
///
/// The block's own two convolutions (`attn_sconv`, `mlp_sconv`) roll back the
/// same way the attention's do, and for the same reason: their memory is a
/// function of the last KEPT row and the rows before it ALONG THE ACCEPTED
/// PATH. On a chain that path is a prefix and this is
/// `conv_history(all.slice([0..hist + keep]), kernel)`, the slice the loop
/// takes today. On a tree it is a gather, because the accepted rows are not
/// contiguous.
///
/// `all` is the `kernel - 1 + rows` window `short_conv_steps` handed back.
pub fn conv_history_rows<B: Backend>(
    all: Tensor<B, 2>,
    kernel: usize,
    kept: &[usize],
) -> Tensor<B, 2> {
    let [len, _] = all.dims();
    let take = crate::models::inkling::spectree::conv_next_history(kernel, kept);
    assert!(
        take.iter().all(|&r| r < len),
        "a {len}-row window cannot supply history {take:?}"
    );
    let dev = all.device();
    let idx: Tensor<B, 1, Int> = Tensor::from_data(
        TensorData::new(
            take.iter().map(|&r| r as i32).collect::<Vec<_>>(),
            [take.len()],
        ),
        &dev,
    );
    all.select(0, idx)
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
///
/// # This is NOT commutable with a partial sum
///
/// A property of the architecture, recorded here because the place it bites is
/// a long way from here and the branch that found it may never ship.
///
/// Any scheme that computes the residual stream in PIECES and adds them up —
/// tensor parallelism across nodes, a split across devices, a hand-written
/// accumulation over head groups — must complete that addition BEFORE calling
/// this. The convolution mixes `x` with `hist`, which carries state from
/// previous tokens and is already whole. So for partials `a` and `b`:
///
/// ```text
///   conv(a, hist) + conv(b, hist)  !=  conv(a + b, hist)
/// ```
///
/// The two sides differ by exactly one extra application of the history term,
/// and they coincide only when `hist` is zero — i.e. on the very first token,
/// which is why a naive implementation looks correct in a one-token test and
/// drifts from the second token on.
///
/// The dangerous part is where it reads naturally. In a decoder layer the
/// obvious place to combine partials is beside the residual add, one line
/// BELOW this call:
///
/// ```text
///   WRONG:  attention(partial) -> conv -> combine -> residual
///   RIGHT:  attention(partial) -> combine -> conv -> residual
/// ```
///
/// The wrong order does not crash, does not produce NaN, and does not fail a
/// shape check. It returns a finite hidden state that goes on to generate
/// fluent, wrong text. Same rule for the MLP half, whose own short convolution
/// sits in the same position relative to the MoE and dense outputs.
///
/// Found while wiring TP2 (`tp2-within-layer-split`, parked); the reasoning
/// applies to any within-layer split, not to that branch.
pub fn short_conv_step(
    hist: Tensor<Bk, 2>,
    x: Tensor<Bk, 2>,
    weight: Tensor<Bk, 2>,
) -> (Tensor<Bk, 2>, Tensor<Bk, 2>) {
    let [rows, dim] = x.dims();
    assert_eq!(rows, 1, "a decode step convolves exactly one position");
    let [wdim, kernel] = weight.dims();
    assert_eq!(
        dim, wdim,
        "short_conv_step: x is [_, {dim}] but the weight is [{wdim}, _]"
    );
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
    // Where the carried history ENDS UP. See `sconv::carry_in_place`: a
    // captured region records addresses, so a history that lands in a fresh
    // buffer every step is a history a replayed step can never advance.
    //
    // The returned tensor is built from the handle actually WRITTEN, not from
    // the caller's `hist` object. Those are the same buffer whenever `hist` was
    // contiguous, which is the case this is for -- but `handle_of` silently
    // makes a non-contiguous tensor contiguous into a NEW buffer, and returning
    // the caller's object then would hand back the one that was not written.
    // Correctness must not depend on that accident; only pointer STABILITY
    // does, and stability is what `INK_GRAPH_DIFF` measures.
    let carried = match crate::models::inkling::sconv::carry_in_place() {
        true => {
            crate::models::inkling::sconv::carry_into(&client, &h_hist, &next, (kernel - 1) * dim);
            h_hist
        }
        false => next,
    };
    (
        tensor_of(client.clone(), dev.clone(), out, 1, dim),
        tensor_of(client, dev, carried, kernel - 1, dim),
    )
}

/// SEVERAL positions of the short convolution at once, given the `kernel - 1`
/// inputs before them.
///
/// The batched twin of [`short_conv_step`], and the shape a speculative verify
/// pass convolves in: the accepted token followed by `k` drafts. It returns the
/// output rows and the WHOLE `kernel - 1 + rows` window it convolved, because
/// that window is what a rollback slices — the history ending at the last kept
/// row is `all[keep .. keep + kernel - 1]`, and the pre-convolution projections
/// it is made of are gone once the batch is over. Same reasoning as
/// [`AttnCache`]'s `Pending`, and for the same reason: a verifier decides late.
///
/// ONE kernel, like [`short_conv_step`] and for the same reason.
///
/// This was slice-built — [`short_conv`] over the concatenation, sliced — on
/// the reasoning that it runs four times a layer per VERIFY pass rather than
/// per position, so the launch count this module exists to remove would not
/// matter here. Measured on the two-node pipe it dominates: the MLP's single
/// convolution costs 1.4 ms a pass at one row and **32.7 ms at two**, a step
/// that is 60% of the whole one-row-to-two-row penalty and is paid again by
/// the two convolutions inside [`attention_steps`]. A widened pass is not four
/// calls, it is four calls times twenty-one layers, and the shifted-slice form
/// describes each of them with a `cat`, `kernel` slices, `kernel` broadcast
/// multiplies and `kernel` adds.
///
/// The window it returns is unchanged and is still the point: a rollback is a
/// slice of it, and the pre-convolution projections it is made of are gone once
/// the batch is over. Same reasoning as [`AttnCache`]'s `Pending`.
///
/// The taps now contract to `fma.rn.f32` exactly as [`short_conv_step`]'s do,
/// so `rows == 1` through here is BIT-IDENTICAL to the one-row lane. Under the
/// slice form it was not, and that difference was one more thing separating a
/// widened pass's arithmetic from a narrow one's.
/// Carries [`short_conv_step`]'s partial-sum rule: a residual computed in
/// PIECES must be summed BEFORE this call, never after it. See that function.
pub fn short_conv_steps(
    hist: Tensor<Bk, 2>,
    x: Tensor<Bk, 2>,
    weight: Tensor<Bk, 2>,
) -> (Tensor<Bk, 2>, Tensor<Bk, 2>) {
    let [rows, dim] = x.dims();
    let [wdim, kernel] = weight.dims();
    assert_eq!(
        dim, wdim,
        "short_conv_steps: x is [_, {dim}] but the weight is [{wdim}, _]"
    );
    assert_eq!(
        hist.dims(),
        [kernel - 1, dim],
        "the history must be the {} rows before this batch",
        kernel - 1
    );
    let all = Tensor::cat(vec![hist, x], 0);
    (short_conv_window(all.clone(), weight, rows), all)
}

/// The same convolution over a window the caller already holds.
///
/// [`short_conv_steps`] concatenates a history onto a batch and then convolves
/// it; [`attention_steps`] has built that concatenation for K and V already,
/// and re-splitting it into (history, batch) so this function could re-join
/// them would be two copies to describe an identity.
///
/// `all` is `[kernel - 1 + rows, dim]` and the output is its LAST `rows` rows —
/// the front `kernel - 1` are history and every output row reads a full window
/// of real input, so [`short_conv`]'s front zero-padding is never reached here.
pub fn short_conv_window(all: Tensor<Bk, 2>, weight: Tensor<Bk, 2>, rows: usize) -> Tensor<Bk, 2> {
    let [len, dim] = all.dims();
    let [wdim, kernel] = weight.dims();
    assert_eq!(
        dim, wdim,
        "short_conv_window: the window is [_, {dim}] but the weight is [{wdim}, _]"
    );
    assert_eq!(
        len,
        kernel - 1 + rows,
        "a {rows}-row convolution wants {} window rows, got {len}",
        kernel - 1 + rows
    );
    let client = client_of(&all);
    let dev = all.device();
    let h_all = handle_of(all);
    let h_w = handle_of(weight);
    let out =
        crate::models::inkling::sconv::short_conv_batch(&client, &h_all, &h_w, dim, rows, kernel);
    tensor_of(client, dev, out, rows, dim)
}

/// [`short_conv_window`] for a batch whose rows are a TREE rather than
/// consecutive positions.
///
/// The batched kernel reads `all[i ..= i + kernel - 1]` — the rows physically
/// preceding row `i`. For a chain those are row `i`'s ancestors, which is why
/// nothing has ever had to say so. For a tree they are whatever the layout put
/// there, and the smallest tree there is (`b = 2` at depth 1) already
/// convolves the second candidate out of the FIRST candidate's projections.
/// Masking attention does not reach this: it is a different operator, and the
/// contamination arrives already mixed into K, V or the residual.
///
/// `taps[i][t]` names the window row that tap `t` of output row `i` must read
/// — computed once, in `spectree::tree_attn`, along each row's own ancestry.
/// The arithmetic is otherwise the kernel's, residual included:
/// `out[i] = all[taps[i][kernel-1]] + sum_t w[:, t] * all[taps[i][t]]`.
///
/// Written with `select` rather than a fourth cubecl kernel because a verify
/// batch is three or four rows: this is `kernel` gathers of a `[rows, dim]`
/// tensor, and a kernel would be a kernel to maintain. Take the contiguous
/// path ([`crate::models::inkling::spectree::TreeAttn::is_linear`]) whenever
/// the batch is a chain, so a non-tree run is untouched.
pub fn short_conv_tree(
    all: Tensor<Bk, 2>,
    weight: Tensor<Bk, 2>,
    taps: &[Vec<usize>],
) -> Tensor<Bk, 2> {
    let [len, dim] = all.dims();
    let [wdim, kernel] = weight.dims();
    assert_eq!(
        dim, wdim,
        "short_conv_tree: the window is [_, {dim}] but the weight is [{wdim}, _]"
    );
    let rows = taps.len();
    assert_eq!(
        len,
        kernel - 1 + rows,
        "a {rows}-row convolution wants {} window rows, got {len}",
        kernel - 1 + rows
    );
    let dev = all.device();
    let gather = |t: usize| -> Tensor<Bk, 2> {
        let idx: Vec<i32> = taps
            .iter()
            .map(|r| {
                assert!(r[t] < len, "tap {} is outside a {len}-row window", r[t]);
                r[t] as i32
            })
            .collect();
        let sel: Tensor<Bk, 1, Int> = Tensor::from_data(TensorData::new(idx, [rows]), &dev);
        all.clone().select(0, sel)
    };
    // The residual the kernel adds is the row's OWN value, which is its last
    // tap by construction.
    let mut acc = gather(kernel - 1);
    for t in 0..kernel {
        let w = weight.clone().slice([0..dim, t..t + 1]).reshape([1, dim]);
        acc = acc + gather(t) * w;
    }
    acc
}

/// [`short_conv_steps`] for a tree batch: the same concatenation, the tree's
/// taps, and the same window handed back for the rollback to gather out of.
pub fn short_conv_tree_steps(
    hist: Tensor<Bk, 2>,
    x: Tensor<Bk, 2>,
    weight: Tensor<Bk, 2>,
    taps: &[Vec<usize>],
) -> (Tensor<Bk, 2>, Tensor<Bk, 2>) {
    let all = Tensor::cat(vec![hist, x], 0);
    (short_conv_tree(all.clone(), weight, taps), all)
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
/// Carries [`short_conv_step`]'s partial-sum rule: a residual computed in
/// PIECES must be summed BEFORE this call, never after it. See that function.
pub fn short_conv<B: Backend>(x: Tensor<B, 2>, weight: Tensor<B, 2>) -> Tensor<B, 2> {
    let [tokens, dim] = x.dims();
    let [wdim, kernel] = weight.dims();
    assert_eq!(
        dim, wdim,
        "short_conv: x is [_, {dim}] but the weight is [{wdim}, _]"
    );
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
    assert_eq!(
        width,
        heads * head_dim,
        "{width} is not {heads} x {head_dim}"
    );
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
    /// `wq|wk|wv|wr` concatenated along the OUTPUT axis, when `INK_FUSE_QKVR`
    /// bound it. See [`project_qkvr`].
    pub wqkvr: Option<Bf16W>,
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
    k: super::kvpages::KvStore<B>,
    v: super::kvpages::KvStore<B>,
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
    /// The rows this batch APPENDED, post-normalisation, kept so a TREE
    /// rollback can put a scattered subset of them back.
    ///
    /// A linear speculation never needs them: it keeps a prefix, and a prefix
    /// is a truncation of the store. A tree keeps the root and one PATH, whose
    /// rows are not contiguous in the batch, and there is no slice of the store
    /// that is those rows — so the store is truncated whole and the kept rows
    /// re-appended. `rows * kv_width` floats, which at width 3 is noise beside
    /// the window this struct already carries.
    k_new: Tensor<B, 2>,
    v_new: Tensor<B, 2>,
    rows: usize,
}

impl AttnCache<Bk> {
    /// Move this cache's KV onto PRE-ALLOCATED pages, if the run asked for
    /// them. **Off unless `INK_KV_PREALLOC` is set.**
    ///
    /// Called once, at the seam between the prefill and the first decode step,
    /// and after the window trim so a local layer's reservation is its window
    /// rather than the whole prompt. From here on the store never allocates,
    /// never frees and never moves a page: the addresses the first decode step
    /// hands the attention kernel are the addresses the ten-thousandth hands
    /// it.
    ///
    /// ## What it is for, in the order the value actually falls
    ///
    /// 1. **Capture.** Cross-step CUDA graph replay is built and works, and the
    ///    thing that ends a replay run is not a value -- `q0`, the mask bounds
    ///    and the KV write row are already patched as scalars -- it is the KV
    ///    page STRUCTURE changing, which is an address problem. A reservation
    ///    removes the address problem outright and leaves one tunable epoch
    ///    ([`super::kvpages::kv_epoch`]) where there used to be a hard 128-row
    ///    page boundary.
    /// 2. **A recurring cost removed.** A capture is 62-101 ms; taking one
    ///    every 128 steps is 0.5-0.8 ms a step amortised, and every epoch
    ///    multiple takes that straight down.
    /// 3. **Control that is only possible on fixed addresses.** L2 residency
    ///    hints, and anything else that names a buffer, cannot be asked of a
    ///    page that may move.
    ///
    /// It is NOT a speed change and should not be sold as one. The host-side
    /// allocator work it deletes is of order 0.06 ms a step (1901 reservations
    /// at 591-918 us a pass, of which ~7% of removed host time reaches the
    /// step, measured on GB10), and the device is busy 81% of a decode step.
    ///
    /// ## The boundary, said plainly
    ///
    /// Single-sequence NVFP4 decode only. A dense store would have to COPY to
    /// cut its reservation down to the live rows -- Burn's `slice` allocates --
    /// where the NVFP4 read is a smaller scalar row count against the same
    /// handle and costs nothing, so this refuses anything but an FP4 cache. The
    /// batched slot lane ([`SlotCache`]) is a different type on a different
    /// path at f32/BF16 and is not touched by this at all.
    pub fn reserve_kv(&mut self, window: Option<usize>, dev: &burn::backend::cuda::CudaDevice) {
        let Some(plan) = super::kvpages::KvPlan::from_env(window.unwrap_or(0)) else {
            return;
        };
        // The NVFP4 arm only. See the boundary above -- and refusing loudly
        // here is better than a dense store quietly paying a `slice` per layer
        // per step for a reservation nobody asked it to hold.
        if !self.kv_is_fp4() {
            return;
        }
        let rows = plan.rows_for(window);
        let held = self.k.len();
        if held > rows {
            // A prompt longer than the reservation. Leave the cache on the
            // grow-on-demand arm rather than truncating a context: the
            // admission gate is what should have refused this, and it says so
            // with the numbers.
            return;
        }
        let placeholder = || super::kvpages::KvStore::wide(1);
        let k = std::mem::replace(&mut self.k, placeholder());
        self.k = k.into_reserved(rows, plan.epoch, dev);
        let v = std::mem::replace(&mut self.v, placeholder());
        self.v = v.into_reserved(rows, plan.epoch, dev);
    }

    /// DEBUG: host-side absolute sums of everything this cache carries to the
    /// next decode step, in the order (K pages, V pages, k_pre, v_pre).
    ///
    /// Exists to name WHICH carried buffer a repeated graph replay moves. It
    /// syncs and reads back, so it belongs behind a flag and nowhere near a
    /// timed path.
    pub fn debug_carry_sums(&self, dev: &burn::backend::cuda::CudaDevice) -> [f64; 4] {
        fn s(t: Tensor<Bk, 2>) -> f64 {
            t.abs()
                .sum()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .expect("device readback")[0] as f64
        }
        [
            s(self.k.materialize(dev)),
            s(self.v.materialize(dev)),
            s(self.k_pre.clone()),
            s(self.v_pre.clone()),
        ]
    }
}

impl<B: Backend> AttnCache<B> {
    /// Keys retained — *not* the sequence length, because a windowed layer
    /// forgets.
    pub fn len(&self) -> usize {
        self.k.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Absolute position of row 0.
    pub fn base(&self) -> usize {
        self.base
    }

    /// Can a REPLAYED graph have done this step's whole device write?
    ///
    /// A captured region records the append's destination row and the window's
    /// dropped prefix as baked-in values, so a replay reproduces exactly the
    /// step it was captured for, shifted only by whatever the caller patches.
    /// What it cannot reproduce is a step that changes the page STRUCTURE --
    /// pushing a new page, releasing an old one, cutting page 0, merging the
    /// settled ones. Each of those moves a buffer a graph node points at, and
    /// none of them is a parameter.
    ///
    /// So the lane asks this first and runs the step eagerly when the answer
    /// is no. Asked before the step and pure, because a lane that discovers
    /// mid-write that it chose wrong has already written.
    pub fn step_is_replayable(&self, n: usize, window: Option<usize>) -> bool {
        if !self.k.append_is_in_place(n) || !self.v.append_is_in_place(n) {
            return false;
        }
        let drop = match window {
            Some(w) if self.k.len() + n > w => self.k.len() + n - w,
            _ => 0,
        };
        // Asked of the PRE-append store, which is conservative in the only
        // direction that matters: on a single-page store `rows_at(0)` is
        // `fill`, so the post-append answer can only be more permissive.
        self.k.drop_is_bookkeeping_only(drop) && self.v.drop_is_bookkeeping_only(drop)
    }

    /// Advance the bookkeeping for a step a replay already performed.
    ///
    /// This is [`attention_step`]'s host half with the device half removed --
    /// the append's row counters and the window's advance -- and it exists
    /// because a replayed step runs no host code inside the region at all. The
    /// short-convolution histories are deliberately NOT touched: with
    /// `INK_GRAPH_CARRY=1` the new history lands back in the buffer it was read
    /// from, so the host rebinding the eager path does is already a no-op, and
    /// with the carry off a replayed region would be reading step k's history
    /// forever and no bookkeeping could fix that.
    pub fn note_replayed_step(&mut self, n: usize, window: Option<usize>) {
        self.k.note_appended(n);
        self.v.note_appended(n);
        if let Some(w) = window {
            let len = self.k.len();
            if len > w {
                let drop = len - w;
                self.k.note_dropped(drop);
                self.v.note_dropped(drop);
                self.base += drop;
            }
        }
    }

    /// Whether this cache is holding its keys and values as NVFP4.
    ///
    /// The arm is chosen inside [`super::kvpages::KvStore::new`] from a switch
    /// and a width, and neither is visible from here — so without this a test
    /// asserting "the FP4 lane still agrees" can pass while never having built
    /// an FP4 store at all. It is the difference between checking the claim and
    /// checking something adjacent to it.
    pub fn kv_is_fp4(&self) -> bool {
        self.k.is_fp4() && self.v.is_fp4()
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
                let keep_k = self.k.len() - drop;
                self.k.truncate(keep_k);
                let keep_v = self.v.len() - drop;
                self.v.truncate(keep_v);
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

/// The tree half of speculative rollback, on the concrete backend.
///
/// Separate from the generic `impl` above only because a gather has to put
/// rows BACK into the store, and putting rows back means going through
/// `as_kv` — the narrowing that decides whether this cache holds BF16 — which
/// is a fact about this backend and not about `B: Backend`.
impl AttnCache<Bk> {
    /// [`AttnCache::commit`] for a TREE verify pass: keep the batch rows named
    /// by `kept`, in order, and discard the rest.
    ///
    /// `kept` is `spectree::TreeAccept::kept_rows` — the root followed by the
    /// accepted path. Ascending, but NOT contiguous, which is the entire
    /// difference from [`AttnCache::commit`]: a linear speculation accepts a
    /// PREFIX of its batch and rolls back with a truncation, while a tree
    /// accepts a path through it and rolls back with a GATHER. The store is
    /// therefore truncated whole and the kept rows re-appended, out of the
    /// copy [`Pending`] holds.
    ///
    /// `kept = 0..keep` reduces to `commit(keep, window)` — see the equality
    /// test — so a chain run through this function is the same cache it was.
    pub fn commit_rows(&mut self, kept: &[usize], window: Option<usize>) {
        // A contiguous accepted set is a PREFIX, and a prefix is what
        // [`AttnCache::commit`] already does -- by TRUNCATION, leaving the kept
        // rows exactly where they were written. Delegating is not just an
        // optimisation: re-appending rows the store already holds sends them
        // through `as_kv` and the store's own packing a second time, and an
        // NVFP4 store computes its scales over the block it is handed, so three
        // rows re-appended as one can quantise differently from the same row
        // appended as one of three. Identical only on a WIDE cache -- which is
        // precisely what the unit test pins, so the unit test could not have
        // seen it.
        //
        // Rank 0 and total rejection are both contiguous, so the common tree
        // pass now takes the same path a chain does, bit for bit.
        if kept.iter().copied().eq(0..kept.len()) {
            self.commit(kept.len(), window);
            return;
        }
        if let Some(p) = self.pending.take() {
            assert!(
                kept.windows(2).all(|w| w[0] < w[1]),
                "an accepted path is ascending"
            );
            assert!(
                kept.last().is_none_or(|&r| r < p.rows),
                "row {:?} is past the {}-row batch",
                kept.last(),
                p.rows
            );
            let dev = p.k_new.device();
            self.k.truncate(self.k.len() - p.rows);
            self.v.truncate(self.v.len() - p.rows);
            if !kept.is_empty() {
                let idx: Tensor<Bk, 1, Int> = Tensor::from_data(
                    TensorData::new(
                        kept.iter().map(|&r| r as i32).collect::<Vec<_>>(),
                        [kept.len()],
                    ),
                    &dev,
                );
                self.k.append(as_kv(p.k_new.clone().select(0, idx.clone())));
                self.v.append(as_kv(p.v_new.clone().select(0, idx)));
            }
            // The next position's convolution memory: the tail of
            // `history ++ accepted path`, gathered out of the window this
            // batch kept for exactly this purpose.
            let hist = p.k_pre.dims()[0] - p.rows;
            let take = crate::models::inkling::spectree::conv_next_history(hist + 1, kept);
            let sel: Tensor<Bk, 1, Int> = Tensor::from_data(
                TensorData::new(
                    take.iter().map(|&r| r as i32).collect::<Vec<_>>(),
                    [take.len()],
                ),
                &dev,
            );
            self.k_pre = p.k_pre.select(0, sel.clone());
            self.v_pre = p.v_pre.select(0, sel);
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
    let len = c.k.len();
    if len <= w {
        return;
    }
    let drop = len - w;
    c.k.drop_front(drop);
    c.v.drop_front(drop);
    c.base += drop;
}

/// `wq`, `wk`, `wv` and `wr` against the same activation — as ONE matmul when
/// the fused weight is bound.
///
/// All four read the residual stream and all four have `k = hidden`, so they
/// differ only in output width: 4096, 1024, 1024 and `heads * d_rel` = 512.
/// Issued separately that is four grids of 512, 128, 128 and 64 cubes, and
/// [`AttnWeightsDev`]'s own doc already records what that costs — "256 cubes of
/// one warp cannot cover DRAM latency on this part ... the unembed, same kernel
/// same instruction, 25128 cubes, ran at 175". The shared experts were fused
/// along `n` for exactly this reason and reach 195 GB/s of a measured 242.9;
/// these four were left split and reach 59.
///
/// Concatenating along the output axis is a SCHEDULING change and not a
/// numerical one: every output element is still the same `k`-loop over the same
/// row. What it can change is which lane [`bf16_gemm`] picks, since that is
/// chosen by shape — so this is a switch, and the arms are compared end to end
/// rather than assumed equal.
///
/// [`bf16_gemm`]: crate::models::inkling::bf16gemm::bf16_gemm
fn project_qkvr(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
) -> (Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 2>) {
    let Some(fused) = w.wqkvr.as_ref() else {
        return (
            linear_bf16(x.clone(), &w.wq),
            linear_bf16(x.clone(), &w.wk),
            linear_bf16(x.clone(), &w.wv),
            linear_bf16(x, &w.wr),
        );
    };
    let rows = x.dims()[0];
    let y = linear_bf16(x, fused);
    let (a, b, c) = (w.wq.n, w.wq.n + w.wk.n, w.wq.n + w.wk.n + w.wv.n);
    (
        y.clone().slice([0..rows, 0..a]),
        y.clone().slice([0..rows, a..b]),
        y.clone().slice([0..rows, b..c]),
        y.slice([0..rows, c..c + w.wr.n]),
    )
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

/// The same layer with the FUSED lanes REFUSED, so a test can hold the two
/// implementations side by side.
///
/// Exists because the only check that catches a fused kernel which disagrees
/// with the dense triangle is one that runs both on the same weights: a banded
/// kernel checked against a banded reference proves the two share an author,
/// not that either is right. It refuses BOTH fused lanes — the band on a local
/// layer and [`super::flash`] on a global one — because both are the same kind
/// of claim about the same dense arm.
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
    fused_ok: bool,
    block: Option<usize>,
) -> (Tensor<Bk, 2>, AttnCache<Bk>) {
    use crate::models::inkling::config::AttnKind;

    let [tokens, hidden] = x.dims();
    assert_eq!(
        hidden, d.hidden,
        "x is [_, {hidden}] but the config says {}",
        d.hidden
    );
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(
        groups * kv_heads,
        heads,
        "{heads} heads do not divide into {kv_heads} kv heads"
    );

    // K and V pass through their short convolutions; Q does not. The
    // pre-convolution projections are kept: they are the convolution's memory,
    // and a decode step cannot reconstruct them from the cached K and V.
    let (q, k_pre, v_pre, r) = project_qkvr(x, w);
    let k = short_conv(k_pre.clone(), w.k_sconv.clone());
    let v = short_conv(v_pre.clone(), w.v_sconv.clone());

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k = head_rms_norm(k, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    // Log scaling: the same vector the slice lane builds, from the same method.
    let taus: Vec<f32> = (0..tokens)
        .map(|t| match (d.kind, log_scaling) {
            (AttnKind::Global, Some(ls)) => ls.tau(t),
            _ => 1.0,
        })
        .collect();
    // AND ONLY WHEN IT SCALES ANYTHING. On a LOCAL layer -- thirty-five of this
    // model's forty-two -- the match above returns 1.0 for every token, and so
    // does a global layer the caller handed no `LogScaling`. Building the vector
    // anyway cost a host->device upload and a broadcast multiply per layer per
    // step to compute `x * 1.0`: ~42 launches a node a step, ~0.8 ms of pure
    // host enqueue, on a pass this file's own brackets show is host-enqueue-
    // bound. In a captured-graph world it is 42 fewer nodes to record and patch.
    //
    // The skip is BIT-IDENTICAL rather than approximately so, which is why it is
    // a guard and not a fast path: multiplying by exactly 1.0 is the identity on
    // every finite value, on both signed zeros, on the infinities and on NaN, so
    // an arm that skips it cannot differ from one that does not. The comparison
    // is against exactly 1.0 for the same reason -- a tau that is 1.0 to six
    // places is not 1.0, and it takes the multiply.
    // `None` when every tau is exactly 1.0, and then NOTHING below multiplies.
    let tau: Option<Tensor<Bk, 1>> = taus
        .iter()
        .any(|&t| t != 1.0)
        .then(|| Tensor::from_data(TensorData::new(taus, [tokens]), &dev));
    let q = match &tau {
        Some(t) => q * t.clone().reshape([tokens, 1]),
        None => q,
    };

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
            if fused_ok
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
                .reshape([tokens, heads, eff]);
            let rel = match &tau {
                Some(t) => rel * t.clone().reshape([tokens, 1, 1]),
                None => rel,
            };
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
                k.clone()
                    .reshape([tokens, kv_heads, head_dim])
                    .swap_dims(0, 1)
                    .swap_dims(1, 2),
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
        // THE OTHER SEVEN. A global layer reads every key, so there is no band
        // to exploit -- but there is no need to write the square down either.
        // [`super::flash`] tiles the key axis and carries the softmax's running
        // max and sum across the tiles, so the peak working set of a global
        // layer stops being `[heads, rows, tokens]` and becomes the
        // relative-bias table for one query block. The GQA expansion goes with
        // it: a cube holds all `groups` query heads of one KV head, so K and V
        // are read once for the four heads that share them rather than once
        // each off a materialised copy.
        _ if fused_ok
            && flash_lane()
            && crate::models::inkling::flash::applies(
                heads,
                kv_heads,
                head_dim,
                crate::models::inkling::flash::prefill_rows(groups),
            ) =>
        {
            use crate::models::inkling::flash::{
                self, KeyRun, KvElem, flash_attention_launch, query_block as flash_block,
            };
            use crate::models::inkling::seam::{client_of, handle_of, handle_of_any, tensor_of};

            let rows_tile = flash::prefill_rows(groups);
            let client = client_of(&q);
            // K and V exactly as the convolutions left them --
            // `[tokens, kv_heads * head_dim]`, token-major, narrowed to the
            // operand dtype. No transpose (the kernel stages the key tile
            // through shared memory instead, which is what lets the decode lane
            // read the same buffers straight out of the cache) and no repeat
            // (the kernel indexes `h / groups`).
            let (k_h, k_dt) = handle_of_any(as_act(k.clone()));
            let (v_h, _) = handle_of_any(as_act(v.clone()));
            let kv_elem = match k_dt {
                burn::tensor::DType::BF16 => KvElem::Bf16,
                _ => KvElem::F32,
            };
            let rel_proj = w.rel_proj.clone().slice([0..d.d_rel, 0..eff]);
            // The query block is sized by the RELATIVE TABLE now, not by the
            // score block, because the score block no longer exists. That is
            // the whole change in one line: what used to be
            // `rows * heads * tokens` is `rows * heads * eff`, and `eff` is at
            // most `rel_extent` however long the sequence is.
            let block = block
                .unwrap_or_else(|| flash_block(heads, eff, head_dim, tokens))
                .clamp(1, tokens);
            let mut parts: Vec<Tensor<Bk, 2>> = Vec::with_capacity(tokens.div_ceil(block));
            for lo in (0..tokens).step_by(block) {
                let hi = (lo + block).min(tokens);
                let rows = hi - lo;
                let rel = r
                    .clone()
                    .slice([lo..hi, 0..heads * d.d_rel])
                    .reshape([rows * heads, d.d_rel])
                    .matmul(rel_proj.clone())
                    .reshape([rows, heads, eff]);
                let rel = match &tau {
                    Some(t) => rel * t.clone().slice([lo..hi]).reshape([rows, 1, 1]),
                    None => rel,
                };
                let rel_h = handle_of(rel);
                let q_h = handle_of(q.clone().slice([lo..hi, 0..heads * head_dim]));
                let o = flash_attention_launch(
                    &client,
                    &q_h,
                    &[KeyRun {
                        k: &k_h,
                        v: &v_h,
                        rows: tokens,
                        base: 0,
                        lo: 0,
                        hi: tokens,
                    }],
                    &rel_h,
                    kv_elem,
                    rows,
                    lo,
                    heads,
                    kv_heads,
                    head_dim,
                    eff,
                    mask_window,
                    d.scaling(),
                    rows_tile,
                );
                parts.push(tensor_of(
                    client.clone(),
                    dev.clone(),
                    o,
                    rows,
                    heads * head_dim,
                ));
            }
            match parts.len() {
                1 => parts.pop().expect("one block"),
                _ => Tensor::cat(parts, 0),
            }
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
                client_of, contiguous, handle_of, strided_of3, tensor_strided3,
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
            let client = client_of(&q);

            // Q, K^T and V made contiguous ONCE, ahead of the loop. Every query
            // block reads all of K and V, and a permuted view handed to the
            // matmul is made contiguous BY the matmul -- which is the whole of
            // it, per block, instead of once per layer.
            //
            // Narrowed BEFORE the expansion, not after: `expand` repeats each
            // KV head `groups` times and that repeat is a materialised write of
            // the whole `[heads, tokens, head_dim]`. Casting first halves what
            // that write moves as well as what it leaves behind. These three
            // are 16 KiB a token each on this model at f32, they are linear in
            // the sequence, and `INK_QBLOCK` -- which bounds the score block --
            // does not reach them.
            //
            // `contiguous` rather than the `handle_of`/`tensor_of3` round trip
            // that used to be here: that pair asserts f32 on the way through,
            // for the good reason that the kernels behind it index bytes, and
            // nothing here is going to a kernel. It is the same
            // `into_contiguous` either way.
            let qh = contiguous(as_act(q.reshape([tokens, heads, head_dim]).swap_dims(0, 1)));
            let kt = contiguous(expand(as_act(k.clone())).swap_dims(1, 2));
            let vh = contiguous(expand(as_act(v.clone())));

            let rel_proj = w.rel_proj.clone().slice([0..d.d_rel, 0..eff]);
            // A parameter, not just `query_block`, because the only bug this
            // change can introduce is a block that reads its query position
            // LOCALLY, and that bug is invisible in block zero. A test needs to
            // force several blocks at a shape small enough to check, and a
            // process-global env var cannot do that while other tests run.
            let block = block
                .unwrap_or_else(|| query_block(heads, tokens))
                .clamp(1, tokens);
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
                    .reshape([rows, heads, eff]);
                let rel = match &tau {
                    Some(t) => rel * t.clone().slice([lo..hi]).reshape([rows, 1, 1]),
                    None => rel,
                };

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
                // Wide again for the epilogue: the scale, the bias, the mask
                // and the softmax below are the arithmetic the reference also
                // keeps in f32, and the narrow lane's saving is in the
                // OPERANDS, which are already spent by this line.
                let raw = from_act(
                    qh.clone()
                        .slice([0..heads, lo..hi, 0..head_dim])
                        .matmul(kt.clone()),
                );
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
                // The probabilities go back down to the operand dtype for
                // `p @ v`, which is what the reference does at this point too.
                parts.push(
                    from_act(as_act(probs).matmul(vh.clone()))
                        .swap_dims(0, 1)
                        .reshape([rows, heads * head_dim]),
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

    // The prefill's whole K and V go in as one append, so the store cuts them
    // into pages once rather than growing a page at a time.
    let (mut ks, mut vs) = {
        let kk = as_kv(k);
        let vv = as_kv(v);
        // The dtype is asked of the tensor rather than of `narrow_now()`: an
        // NVFP4 store promises to hand back what it was given, and the only
        // authority on what that is is the buffer itself.
        let mut ks = super::kvpages::KvStore::new(kk.dims()[1], super::seam::dtype_of(&kk));
        let mut vs = super::kvpages::KvStore::new(vv.dims()[1], super::seam::dtype_of(&vv));
        ks.append(kk);
        vs.append(vv);
        (ks, vs)
    };
    let _ = (&mut ks, &mut vs);
    let mut cache = AttnCache {
        k: ks,
        v: vs,
        k_pre: conv_history(k_pre, d.kernel),
        v_pre: conv_history(v_pre, d.kernel),
        base: 0,
        pending: None,
    };
    trim(&mut cache, window);
    // AFTER the trim, so a windowed layer's reservation is sized by its window
    // and not by the prompt. See `AttnCache::reserve_kv`.
    cache.reserve_kv(window, &dev);
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

/// Whether a KV cache HOLDS its keys and values as BF16, and multiplies them
/// there.
///
/// **On by default; `INK_ATTN_BF16=0` is the wide arm.** It is a switch and not
/// an unconditional change for one reason -- what it trades is precision, and
/// the only honest way to price precision is to run both arms of the same
/// binary against the same harness. It is on by default because it won that
/// comparison, and because a default is what the unit tests below exercise.
///
/// Both caches, and only the CACHED reads. A prefill computes its scores from
/// the keys it has just projected and never round-trips them through memory, so
/// it has nothing to save and is left alone; every decode step reads its whole
/// retained context back, and that read is what this narrows.
///
/// The argument is that the four bytes were never buying twenty-four mantissa
/// bits. Burn's f32 matmul on this runtime is TF32
/// ([`f32_matmul_is_tf32_on_this_runtime`], 9.3e-4 relative on a 128-deep
/// product), so the PRODUCT already carries about ten; the four-byte load pays
/// for twenty-three that the tensor cores then round away.
///
/// **A closer float is not a more correct one, and that is why there is no
/// tolerance here.** The weights are 4-bit NVFP4; an f64 reference is not what
/// anyone is trying to match, and a gate on numerical distance from one would
/// measure a theorem rather than an outcome. What decides this lane is
/// `golden/paired/`: the same prompts through the BF16 reference and through
/// each arm of this runtime, reported as AGREEMENT and as LOSS with McNemar on
/// the discordant pairs. A layer-RMS ladder is still worth a look as a smoke
/// check -- a lane that has gone wrong emits garbage, not a moved fourth
/// decimal -- but it is not the acceptance criterion.
///
/// The switch is a process-global `OnceLock`, so a test binary would get ONE
/// lane and never exercise the other -- the wrong shape for a change whose
/// whole subject is a comparison. [`CacheLane`] is the per-thread override that
/// fixes it: the cached-lane tests below take `CacheLane::wide()` explicitly,
/// because their tolerances are statements about two implementations of the
/// same arithmetic and a narrow cache is not that, and
/// `narrow_slots_are_still_independent` takes `CacheLane::narrow()` and asserts
/// the one thing that holds exactly at any dtype.
///
/// What the two bytes buy is the term itself. The attention matmuls read the
/// KV cache and almost nothing else: 6.0 GB a pass at 32 slots against 5.4 GB
/// of slot KV, at 152 GB/s of a measured 236 GB/s bus, which is a
/// bandwidth-bound lane and not an arithmetic one. Storing the cache narrow --
/// rather than casting it per pass, which would ADD a full-width read -- halves
/// what the lane moves AND halves what the run holds, and on this part the
/// second of those is the ceiling: the slot KV is 15.04 GiB of the head at 96
/// slots, on a node whose `MemAvailable` bottoms at 1.38 GiB there.
pub fn attn_bf16() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_ATTN_BF16")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

thread_local! {
    /// A per-thread override of [`attn_bf16`], for tests only.
    ///
    /// The env switch is a process-global `OnceLock`, which means a test binary
    /// gets ONE lane and the other is never exercised. That is the wrong shape
    /// for a change whose whole subject is a comparison between two lanes, and
    /// it bites immediately: the cached-lane tests below assert that feeding one
    /// token at a time reproduces feeding all of them, to a tolerance sized for
    /// two implementations of the SAME arithmetic. A narrow cache is not the
    /// same arithmetic -- it stores less -- so those tolerances are statements
    /// about the wide lane and have to be able to say so.
    ///
    /// Rust runs each test on its own thread, so a thread-local is exactly the
    /// scope: [`with_narrow`] wraps a test body and nothing outside it moves.
    /// Production never sets it.
    static NARROW: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether THIS thread's caches are narrow: the override if one is set, the
/// process default otherwise.
fn narrow_now() -> bool {
    NARROW.with(|c| c.get()).unwrap_or_else(attn_bf16)
}

/// Force a cache dtype for as long as this value lives. Tests only.
///
/// A guard rather than a closure because it is a one-line addition at the top
/// of a function body, where wrapping a body would reindent it and hide the
/// change in the diff.
#[cfg(test)]
pub(crate) struct CacheLane(Option<bool>);

#[cfg(test)]
impl CacheLane {
    /// The wide cache: what every tolerance in this module's cached tests was
    /// written against.
    pub(crate) fn wide() -> Self {
        CacheLane(NARROW.with(|c| c.replace(Some(false))))
    }

    /// The narrow one.
    pub(crate) fn narrow() -> Self {
        CacheLane(NARROW.with(|c| c.replace(Some(true))))
    }
}

#[cfg(test)]
impl Drop for CacheLane {
    fn drop(&mut self) {
        NARROW.with(|c| c.set(self.0));
    }
}

/// `t` in the dtype the KV cache is held in. The identity on the wide lane.
fn as_kv<const D: usize>(t: Tensor<Bk, D>) -> Tensor<Bk, D> {
    if narrow_now() {
        t.cast(burn::tensor::FloatDType::BF16)
    } else {
        t
    }
}

/// Back to f32, for the softmax and the residual stream. The identity on the
/// wide lane.
fn from_kv<const D: usize>(t: Tensor<Bk, D>) -> Tensor<Bk, D> {
    if narrow_now() {
        t.cast(burn::tensor::FloatDType::F32)
    } else {
        t
    }
}

/// A cached decode read, as the chunks the two attention products consume.
///
/// ## What this replaces, and why
///
/// [`KvStore::materialize`] built one contiguous tensor out of the pages on
/// every layer of every step. That is two costs, not one. The obvious one is
/// the copy — a full read and a full write of the retained K and V per layer,
/// per step, at EVERY context length, and on the NVFP4 arm a dequantization of
/// the whole context beside it. The other one is launches: `Tensor::cat` on
/// this backend is one `slice_assign` kernel per input
/// (`burn-backend`'s `cat_with_slice_assign`), so a long context was already
/// paying a launch per page just to rebuild a tensor it rebuilt last step.
///
/// Reading the pages directly removes both, because attention is a sum over key
/// positions and so both products decompose: the scores are `q @ k^T` per chunk
/// concatenated along the key axis, and the output is `p @ v` per chunk summed.
/// The softmax in between still sees the whole key axis, so this is the
/// materialized read's arithmetic exactly, not an approximation of it — the
/// `paged_read_matches_the_materialized_read` test is that claim.
///
/// ## The two kinds of column that are not keys
///
/// A chunk is a page read WHOLE, so the read carries `head` rows the sliding
/// window has already dropped, and the last chunk is padded up to the shape
/// bucket. Both are masked to `-inf` by the caller. Neither can produce a NaN:
/// the dropped rows hold real keys and the pad rows hold zeros, so every score
/// is finite before the mask and every value is finite after it.
///
/// Both exist to keep SHAPES still. Slicing the head off would make chunk 0
/// walk `1..=PAGE` rows as a window advances; padding the tail collapses the
/// last chunk to one of a couple of sizes. cubecl keys compiled kernels on
/// shapes, and the comment on `bucket` in [`attention_step`] records what a
/// shape that moves every step costs.
struct PagedKv {
    /// `[kv_heads, chunk_rows, head_dim]`, in key order.
    k: Vec<Tensor<Bk, 3>>,
    v: Vec<Tensor<Bk, 3>>,
    /// Dead rows at the front of chunk 0 — keys `drop_front` discarded.
    head: usize,
    /// Columns on the key axis: `head + len + tail padding`.
    slots: usize,
}

impl PagedKv {
    /// Read `cache`'s pages, padding the last chunk to a multiple of `bucket`.
    fn read(
        cache: &AttnCache<Bk>,
        dev: &burn::backend::cuda::CudaDevice,
        dims: (usize, usize),
        bucket: usize,
    ) -> Self {
        let (kv_heads, head_dim) = dims;
        let head = cache.k.head();
        debug_assert_eq!(head, cache.v.head(), "the two stores drifted apart");
        let mut k = cache.k.parts(dev);
        let mut v = cache.v.parts(dev);
        assert!(
            !k.is_empty(),
            "a cached read wants at least one page; the step appends before it reads"
        );
        // PHYSICAL rows, which is what the key axis is built at. It is `head +
        // len` plus whatever of the page being written is not yet real: a page
        // is allocated at its full capacity and filled in place, so the last
        // chunk carries dead rows at its BACK exactly as chunk 0 carries them
        // at its front. Both are masked below, over `slots` rather than over
        // `len`, which is why nothing here has to know which is which.
        let stored: usize = k.iter().map(|p| p.dims()[0]).sum();
        debug_assert!(
            stored >= head + cache.len(),
            "the pages lost rows: {stored} stored against {} live",
            head + cache.len()
        );
        let tail = k.last().expect("checked").dims()[0];
        let pad = tail.next_multiple_of(bucket) - tail;
        if pad > 0 {
            // Zeros, not uninitialized rows: a padded key scores 0 against any
            // query (harmless, the mask removes it) but a padded VALUE is
            // multiplied by a probability of exactly zero, and `0 * NaN` is NaN.
            let extend = |ps: &mut Vec<Tensor<Bk, 2>>| {
                let last = ps.pop().expect("checked");
                let dim = last.dims()[1];
                ps.push(Tensor::cat(
                    vec![last, as_kv(Tensor::zeros([pad, dim], dev))],
                    0,
                ));
            };
            extend(&mut k);
            extend(&mut v);
        }
        let headwise = |t: Tensor<Bk, 2>| -> Tensor<Bk, 3> {
            let rows = t.dims()[0];
            t.reshape([rows, kv_heads, head_dim]).swap_dims(0, 1)
        };
        Self {
            k: k.into_iter().map(headwise).collect(),
            v: v.into_iter().map(headwise).collect(),
            head,
            slots: stored + pad,
        }
    }

    /// `q @ k^T` over every chunk, joined on the key axis.
    ///
    /// `qg` is `[kv_heads, qrows, head_dim]` — the GQA regrouping, queries
    /// batched by the KV head they read — and the result is
    /// `[kv_heads, qrows, slots]`. The join is a `cat` of SCORES, which is the
    /// narrow tensor in this step: one f32 per (head, query row, key) against
    /// the cache's `head_dim` values per key.
    fn scores(&self, qg: Tensor<Bk, 3>) -> Tensor<Bk, 3> {
        let q = as_kv(qg);
        let parts: Vec<Tensor<Bk, 3>> = self
            .k
            .iter()
            .map(|k| q.clone().matmul(k.clone().swap_dims(1, 2)))
            .collect();
        from_kv(if parts.len() == 1 {
            parts.into_iter().next().expect("one chunk")
        } else {
            Tensor::cat(parts, 2)
        })
    }

    /// `p @ v` over every chunk, summed.
    ///
    /// `probs` is `[kv_heads, qrows, slots]` and the result is
    /// `[kv_heads, qrows, head_dim]`. Summing is not an approximation: the
    /// output for one query is `sum_j p_j v_j`, and cutting that sum at page
    /// boundaries is the associativity of addition.
    fn weighted_values(&self, probs: Tensor<Bk, 3>) -> Tensor<Bk, 3> {
        let [kv_heads, qrows, _] = probs.dims();
        let p = as_kv(probs);
        let mut off = 0usize;
        let mut acc: Option<Tensor<Bk, 3>> = None;
        for v in &self.v {
            let rows = v.dims()[1];
            let part = p
                .clone()
                .slice([0..kv_heads, 0..qrows, off..off + rows])
                .matmul(v.clone());
            acc = Some(match acc {
                None => part,
                Some(a) => a + part,
            });
            off += rows;
        }
        debug_assert_eq!(off, self.slots, "the chunks did not cover the key axis");
        from_kv(acc.expect("at least one chunk"))
    }
}

/// Whether a PREFILL holds its `[heads, tokens, head_dim]` attention operands
/// as BF16, and multiplies them there.
///
/// **On by default; `INK_ACT_BF16=0` is the wide arm.** A switch for the same
/// reason [`attn_bf16`] is one: what it trades is precision, and the only
/// honest way to price precision is to run both arms of the same binary against
/// the same harness.
///
/// [`attn_bf16`] narrows the CACHE, which is a decode-step term and does
/// nothing for a prefill -- a prefill computes its scores from keys it has just
/// projected. This narrows what a prefill actually holds. On a global layer
/// that is Q, the GQA-expanded K transposed, and the GQA-expanded V: three
/// `[32, n, 128]` buffers, `16 KiB` a token EACH on this model, and they are
/// linear in the sequence with no knob on them. `INK_QBLOCK` bounds the score
/// block and cannot touch these.
///
/// ## What the reference does here
///
/// BF16, at every one of these points. The upstream implementation keeps its
/// whole residual stream, its projections, its attention operands and its
/// attention output in BF16 and reserves f32 for accumulation: the RMSNorm
/// variance, the short convolution's four taps, the softmax running max and
/// sum, and the two matmul accumulators. The probabilities are cast BACK to the
/// query dtype before `p @ v` there, which is exactly what this does.
///
/// So the bias, the mask and the softmax stay f32 here -- those are the
/// accumulations the reference also keeps wide -- and the operands do not.
///
/// ## What it costs that the reference does not pay
///
/// A fused attention kernel keeps the score in an f32 accumulator from the MMA
/// straight into the softmax and never writes it out. This lane materialises
/// the score block, so a BF16 `q @ k^T` rounds the f32 accumulator to BF16 on
/// the store and the epilogue reads it back. That rounding is real and it is
/// not one the reference takes. It IS the one [`attention_step`] has taken on
/// the cached lane since the BF16 cache landed, priced there against
/// `golden/paired/` rather than against a tolerance, and this is the same
/// arithmetic on the other lane.
/// Whether the GLOBAL attention layers take the fused lane. **On by default;
/// `INK_FLASH=0` is the old dense-blocked arm.**
///
/// A switch and not a silent replacement, for the reason [`act_bf16`] is one:
/// the two arms differ in numerics as well as in memory — the fused lane keeps
/// Q and the scores in f32 where the dense lane narrows them to BF16 for the
/// matmul — and the only honest way to price that is to run both arms of the
/// same binary against the same harness. It is also the only way to measure
/// them at all on this box, where a cross-worktree comparison is meaningless
/// because cubecl's autotune cache is per working directory.
///
/// It is ON rather than OFF because what it changes is not a trade: the dense
/// arm cannot express a long context at all. It GQA-expands K and V to
/// `[heads, tokens, head_dim]` before it starts, which is two buffers linear in
/// the sequence with no knob on them, and then holds `[heads, rows, tokens]`
/// of scores on top.
pub fn flash_lane() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INK_FLASH").map(|v| v != "0").unwrap_or(true))
}

pub fn act_bf16() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_ACT_BF16")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Whether `wq|wk|wv|wr` are bound as ONE `[6656, hidden]` weight.
/// `INK_FUSE_QKVR=0` is the ablation.
///
/// # Why this is the default arm
///
/// A SCHEDULING change and nothing else: every output element is the same
/// k-loop over the same row of the same weight, in the same order, and
/// [`project_qkvr`] slices the one output back into the four the caller
/// expects. Its output was measured **BIT-IDENTICAL** to the four-launch arm,
/// which is the bar this project's default-on policy names -- so it is on, and
/// the switch survives only so the two can be priced against each other.
///
/// What it buys is grid occupancy. The `m16n8k16` kernel gives one warp each
/// `(m_tile, n_tile)`, so with `m` padded to one tile the grid IS `n / 8`:
/// four launches of 512, 128, 128 and 64 cubes become one of 832. Four grids
/// that small cannot cover DRAM latency on this part -- the same argument the
/// shared experts' `gate_up` concatenation rests on, measured there at 79 GB/s
/// for 256 cubes against 175 for 25128.
///
/// # What it costs, and who charges it
///
/// A concatenation is not a span of the pile's mapping, so `bind_bf16` cannot
/// alias it and this is a REAL 54.5 MB (52 MiB) a layer of device memory. That
/// is why this function lives here and not in `inkling_forward`: it is a
/// process-global lane switch that changes admission arithmetic, and
/// [`super::budget::AdmissionPolicy::runtime`] reads it so the gate and the
/// binding lane cannot disagree about it -- the same one-reader rule
/// [`super::budget::dense_weights`] documents, and for the same reason.
/// [`super::pile::device_weight_bytes`] charges those bytes in BOTH placement
/// arms, because the copy happens in both.
///
/// At the production split it clears with room: `INK_LAYERS=21:42` admits at
/// 95.09 GiB of a 119.63 GiB node with 19.68 GiB of headroom, and 21 layers of
/// this is 1.07 GiB -- 5.4% of that headroom.
pub fn fuse_qkvr() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_FUSE_QKVR")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// `t` in the dtype a prefill holds its attention operands in. The identity on
/// the wide lane.
pub fn as_act<const D: usize>(t: Tensor<Bk, D>) -> Tensor<Bk, D> {
    if act_bf16() {
        t.cast(burn::tensor::FloatDType::BF16)
    } else {
        t
    }
}

/// Back to f32, for the epilogue and the residual stream. The identity on the
/// wide lane.
pub fn from_act<const D: usize>(t: Tensor<Bk, D>) -> Tensor<Bk, D> {
    if act_bf16() {
        t.cast(burn::tensor::FloatDType::F32)
    } else {
        t
    }
}

/// Whether a layer's cached lanes take [`super::flash`].
///
/// One predicate rather than three copies of the same three arguments, because
/// the three cached entry points ([`attention_step`], [`attention_steps`] and
/// the prefill's own arm) must not be able to disagree about it: a batch of one
/// that took a different lane from a step of one would show up as drift in
/// [`drift_table_at_real_width`] and be blamed on the cache.
fn flash_cached_applies(d: &crate::models::inkling::attn::AttnDims) -> bool {
    flash_lane()
        && crate::models::inkling::flash::applies(
            d.heads,
            d.kv_heads,
            d.head_dim,
            crate::models::inkling::flash::decode_rows(d.groups()),
        )
}

/// `nq` query rows at absolute positions `q0 ..` against the cache's PAGES.
///
/// `q` is `[nq, heads * head_dim]` with log scaling already applied, and `rel`
/// is `[nq, heads, eff]` likewise. The pages go in as pages, at the dtype the
/// store holds them in — a dense store hands over slices of its own buffers, an
/// NVFP4 store dequantises one page at a time — so nothing here rebuilds a
/// contiguous cache, and nothing pads the key axis to a shape bucket: a fused
/// kernel takes its lengths as scalars and cubecl does not recompile on a
/// scalar.
fn flash_cached(
    q: Tensor<Bk, 2>,
    rel: Tensor<Bk, 3>,
    cache: &AttnCache<Bk>,
    dev: &burn::backend::cuda::CudaDevice,
    d: &crate::models::inkling::attn::AttnDims,
    nq: usize,
    q0: usize,
    eff: usize,
    window: Option<usize>,
) -> Tensor<Bk, 2> {
    use crate::models::inkling::flash::{self, KeyRun, KvElem, flash_attention_launch};
    use crate::models::inkling::seam::{client_of, handle_of, handle_of_any, tensor_of};

    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let (len, base) = (cache.len(), cache.base);
    let head = cache.k.head();
    let kparts = cache.k.parts(dev);
    let vparts = cache.v.parts(dev);
    debug_assert_eq!(kparts.len(), vparts.len(), "the two stores drifted apart");
    let client = client_of(&q);
    let q_h = handle_of(q);
    let rel_h = handle_of(rel);
    let mut held: Vec<(cubecl::server::Handle, cubecl::server::Handle, usize)> =
        Vec::with_capacity(kparts.len());
    let mut kv_dt = burn::tensor::DType::F32;
    for (kp, vp) in kparts.into_iter().zip(vparts) {
        let rows = kp.dims()[0];
        let (kh, dt) = handle_of_any(kp);
        let (vh, _) = handle_of_any(vp);
        kv_dt = dt;
        held.push((kh, vh, rows));
    }
    // Stored row `off + i` of page `p` is logical key `off + i - head`, at
    // absolute position `base + off + i - head`. `head` is the prefix a window
    // has dropped but the page still carries, and it never exceeds `base` —
    // every dropped row was counted into `base` first — so the subtraction
    // cannot go under.
    let mut off = 0usize;
    let mut runs: Vec<KeyRun<'_>> = Vec::with_capacity(held.len());
    for (kh, vh, rows) in &held {
        runs.push(KeyRun {
            k: kh,
            v: vh,
            rows: *rows,
            base: base + off - head,
            lo: head.saturating_sub(off).min(*rows),
            hi: (head + len).saturating_sub(off).min(*rows),
        });
        off += rows;
    }
    let out = flash_attention_launch(
        &client,
        &q_h,
        &runs,
        &rel_h,
        match kv_dt {
            burn::tensor::DType::BF16 => KvElem::Bf16,
            _ => KvElem::F32,
        },
        nq,
        q0,
        heads,
        kv_heads,
        head_dim,
        eff,
        window,
        d.scaling(),
        flash::decode_rows(d.groups()),
    );
    tensor_of(client, dev.clone(), out, nq, heads * head_dim)
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
    assert_eq!(
        hidden, d.hidden,
        "x is [_, {hidden}] but the config says {}",
        d.hidden
    );
    assert!(
        pos >= cache.base + cache.len(),
        "position {pos} is already cached"
    );
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(
        groups * kv_heads,
        heads,
        "{heads} heads do not divide into {kv_heads} kv heads"
    );

    let (q, k_proj, v_proj, r) = project_qkvr(x, w);
    let (k_new, k_hist) = short_conv_step(cache.k_pre.clone(), k_proj, w.k_sconv.clone());
    let (v_new, v_hist) = short_conv_step(cache.v_pre.clone(), v_proj, w.v_sconv.clone());
    cache.k_pre = k_hist;
    cache.v_pre = v_hist;

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k_new = head_rms_norm(k_new, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    cache.k.append(as_kv(k_new.clone()));
    cache.v.append(as_kv(v_new.clone()));
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
    // THE FUSED LANE, on the layers that have one.
    //
    // Everything below this block builds a `[heads, 1, slots]` score row and
    // walks it five more times — the epilogue's bias and mask, the softmax's
    // max, its sum, its divide — before `p @ v` reads it a sixth. At 1M tokens
    // of context that row is 128 MB per global layer. The fused kernel never
    // writes it: it reads each page once, keeps the softmax's running max and
    // sum in registers, and accumulates `p @ v` as it goes. Measured on a GB10
    // at the release's global shape, one layer, one step, NVFP4 KV: 2.5 ms
    // against 4.2 at 16k of context, 7.8 against 33.2 at 64k, and 28.5 against
    // 143.8 at 256k.
    if flash_cached_applies(d) {
        let tau = match (d.kind, log_scaling) {
            (AttnKind::Global, Some(ls)) => ls.tau(pos),
            _ => 1.0,
        };
        let q = q.mul_scalar(tau);
        // Only the distances that can occur are worth projecting: the oldest
        // retained key is `pos - base` back and the table stops at
        // `rel_extent`. Rounded up to the pad bucket for the same reason the
        // lane below rounds — this is the width of a matmul, and a width that
        // moves every step is a kernel compiled every step.
        let eff = d
            .rel_extent
            .min(pos - base + 1)
            .next_multiple_of(kv_pad_bucket())
            .min(d.rel_extent);
        let rel = r
            .reshape([heads, d.d_rel])
            .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
            .reshape([1, heads, eff])
            .mul_scalar(tau);
        let out = flash_cached(q, rel, cache, &dev, d, 1, pos, eff, window);
        return linear_bf16(out, &w.wo);
    }

    let bucket = kv_pad_bucket();
    // THE PAGED READ. `slots` is the key axis this function builds every shape
    // at, and it is `head + len + tail padding` rather than `len` rounded up:
    // the chunks are pages read whole, so both ends carry columns that are not
    // live keys and both are masked below. See [`PagedKv`].
    let kv = PagedKv::read(cache, &dev, (kv_heads, head_dim), bucket);
    let (head, slots) = (kv.head, kv.slots);

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
    // Over `slots`, not `len`: slot `s` is logical key `s - head`, and the
    // slots outside `head .. head + len` are the dropped prefix and the tail
    // pad. They carry index 0 (in range for the gather, and multiplied out by
    // `valid = 0`) and `-inf` in the mask, so they contribute nothing to the
    // softmax and nothing to the value average. There is always at least one
    // real key, so no row is entirely `-inf`.
    let mut idx = vec![0i32; slots];
    let mut valid = vec![0f32; slots];
    let mut wmask = vec![0f32; slots];
    let mut max_dist = 0usize;
    for (s, cell) in wmask.iter_mut().enumerate() {
        if s < head || s >= head + len {
            *cell = f32::NEG_INFINITY;
            continue;
        }
        let j = s - head;
        // Every retained key is at or before `pos`, so this cannot go negative.
        let dist = pos - (base + j);
        if dist < d.rel_extent {
            idx[s] = dist as i32;
            valid[s] = 1.0;
        }
        if window.is_some_and(|wnd| dist >= wnd) {
            *cell = f32::NEG_INFINITY;
        }
        max_dist = max_dist.max(dist);
    }
    // Bucketed for the same reason: `eff` grows one per step until it saturates
    // at `rel_extent`, and it is the width of the relative-projection matmul
    // and of the gather it feeds. Rounding up only ever admits COLUMNS the
    // gather does not index, because every `idx` is `< max_dist + 1 <= eff`.
    let eff = d
        .rel_extent
        .min(max_dist + 1)
        .next_multiple_of(bucket)
        .min(d.rel_extent);
    let idx: Tensor<Bk, 3, Int> =
        Tensor::from_data(TensorData::new(idx, [1, 1, slots]), &dev).repeat_dim(0, heads);
    let valid: Tensor<Bk, 3> = Tensor::from_data(TensorData::new(valid, [1, 1, slots]), &dev);
    let wmask: Tensor<Bk, 3> = Tensor::from_data(TensorData::new(wmask, [1, 1, slots]), &dev);

    let rel = r
        .reshape([heads, d.d_rel])
        .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
        .reshape([heads, 1, eff])
        .mul_scalar(tau);
    let bias = rel.gather(2, idx) * valid;

    // GQA WITHOUT MATERIALISING THE EXPANSION.
    //
    // Head `i` reads KV head `i / groups`, so the old `repeat_dim(1, groups)`
    // built a `[heads, padded, head_dim]` copy of a `[kv_heads, ..]` cache --
    // on this model 32 heads from 8, a 4x materialisation of the largest
    // tensor in the step, every layer, every token. The same product is a
    // batched matmul over `kv_heads` if the QUERIES are grouped instead: the
    // `groups` queries that share a KV head become that batch entry's rows.
    //
    // `q` is `[1, heads * head_dim]` with heads contiguous, and the ordering
    // the repeat produced was exactly kv-head-major -- head `k * groups + g`
    // read KV head `k` -- so the reshape below is the same correspondence read
    // the other way round, not a new convention.
    let qg = q.reshape([kv_heads, groups, head_dim]);

    // Narrow on both sides of each product and wide again immediately after, so
    // the bias, the mask and the softmax are the same arithmetic the f32 lane
    // runs. The scores are `[heads, 1, slots]`; the cache is the megabytes.
    //
    // The grouped product is `[kv_heads, groups, head_dim] @ [kv_heads,
    // head_dim, slots]` -> `[kv_heads, groups, slots]`, and that reshapes to
    // `[heads, 1, slots]` because `kv_heads * groups == heads` in this order.
    // Everything after the reshape is the arithmetic that was here before,
    // element for element -- the bias, the mask and the softmax never saw the
    // expansion, and [`PagedKv`] never breaks the key axis they run over.
    let scores = kv
        .scores(qg)
        .reshape([heads, 1, slots])
        .mul_scalar(d.scaling())
        + bias
        + wmask;
    let probs = burn::tensor::activation::softmax(scores, 2);
    let out = kv
        .weighted_values(probs.reshape([kv_heads, groups, slots]))
        .reshape([1, heads * head_dim]);
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
    attention_steps_tree(x, w, d, log_scaling, pos0, window, cache, None)
}

/// [`attention_steps`] over a TOKEN TREE instead of a chain.
///
/// `tree` describes what stops being structural the moment the batch's rows
/// are not consecutive positions of one sequence, and there are three such
/// things rather than the one the word "mask" suggests:
///
/// * **visibility.** Row `i` may see the cached prefix and its own ANCESTORS,
///   and must not see a sibling. A chain's rule — every earlier row — is the
///   special case where every earlier row is an ancestor.
/// * **position.** Row `i` sits at `pos0 + depth[i]`, so siblings share a
///   position. Both readers of position have to be told: the relative-bias
///   table, whose `dist` is `pos - abs` and would otherwise place a sibling
///   one key apart, and the log-scaling `tau`.
/// * **the convolutions.** `k_sconv` and `v_sconv` read the rows physically
///   preceding row `i`, which for a tree is the wrong branch. See
///   [`short_conv_tree`], and note that this function is only two of the four
///   short convolutions a widened pass runs — the block's `attn_sconv` and
///   `mlp_sconv` are the caller's and need the same taps.
///
/// `tree: None`, or a tree that [`crate::models::inkling::spectree::TreeAttn::is_linear`]
/// accepts, runs the arithmetic [`attention_steps`] always ran, including the
/// fused lane. A real tree takes the paged lane: the fused kernel derives
/// causality from positions, which is exactly the assumption a tree breaks.
#[allow(clippy::too_many_arguments)]
pub fn attention_steps_tree(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    pos0: usize,
    window: Option<usize>,
    cache: &mut AttnCache<Bk>,
    tree: Option<&crate::models::inkling::spectree::TreeAttn>,
) -> Tensor<Bk, 2> {
    use crate::models::inkling::config::AttnKind;

    // A descriptor that asks for nothing new is not a tree, and saying so here
    // once keeps every branch below from having to ask twice.
    let tree = tree.filter(|t| !t.is_linear());

    let [rows, hidden] = x.dims();
    assert!(rows >= 1, "a batched step feeds at least one token");
    assert_eq!(
        hidden, d.hidden,
        "x is [_, {hidden}] but the config says {}",
        d.hidden
    );
    assert!(
        pos0 >= cache.base + cache.len(),
        "position {pos0} is already cached"
    );
    assert!(
        cache.pending.is_none(),
        "a speculative batch is still uncommitted"
    );
    if let Some(t) = tree {
        assert_eq!(
            t.rows, rows,
            "the tree describes {} rows and the batch has {rows}",
            t.rows
        );
        assert_eq!(
            t.kernel, d.kernel,
            "the tree's taps were built for kernel {} and the layer's is {}",
            t.kernel, d.kernel
        );
    }
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(
        groups * kv_heads,
        heads,
        "{heads} heads do not divide into {kv_heads} kv heads"
    );

    let (q, k_proj, v_proj, r) = project_qkvr(x, w);
    // The convolution over the batch, taps and all: the `kernel - 1` history
    // rows the cache carries, then this batch's own projections. Rows
    // `kernel - 1 ..` of that see a full window of real inputs, which is
    // exactly the rows this batch is asking for — the front-padding
    // [`short_conv`] applies is never reached.
    let k_all = Tensor::cat(vec![cache.k_pre.clone(), k_proj], 0);
    let v_all = Tensor::cat(vec![cache.v_pre.clone(), v_proj], 0);
    let hist = d.kernel - 1;
    let kdim = k_all.dims()[1];
    let vdim = v_all.dims()[1];
    // The batched kernel, not [`short_conv`] over the concatenation: this is
    // the second and third of the four convolutions a widened pass runs per
    // layer, and the shifted-slice form is what made a two-row pass cost 1.6x
    // a one-row one.
    let (k_new, v_new) = match tree {
        None => (
            short_conv_window(k_all.clone(), w.k_sconv.clone(), rows),
            short_conv_window(v_all.clone(), w.v_sconv.clone(), rows),
        ),
        Some(t) => (
            short_conv_tree(k_all.clone(), w.k_sconv.clone(), &t.taps),
            short_conv_tree(v_all.clone(), w.v_sconv.clone(), &t.taps),
        ),
    };

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k_new = head_rms_norm(k_new, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    // Cloned rather than moved: `Pending` keeps these rows so a TREE rollback
    // can put a scattered subset of them back. A tensor clone is a handle.
    cache.k.append(as_kv(k_new.clone()));
    cache.v.append(as_kv(v_new.clone()));
    cache.k_pre = k_all.clone().slice([rows..rows + hist, 0..kdim]);
    cache.v_pre = v_all.clone().slice([rows..rows + hist, 0..vdim]);
    cache.pending = Some(Pending {
        k_pre: k_all,
        v_pre: v_all,
        k_new,
        v_new,
        rows,
    });

    let len = cache.len();
    let base = cache.base;

    // The fused lane, the `rows > 1` twin of [`attention_step`]'s. It is the
    // SAME kernel with `nq = rows`: a draft row is a query at its own absolute
    // position, and the rows of the batch are keys the later rows can see
    // because they were appended above. Causality inside the batch is
    // therefore the same predicate as causality against the prefix, which is
    // what makes one kernel enough.
    //
    // Wired here and not only on the single step because a batch of one and a
    // step of one must be the same arithmetic. While this function was on the
    // dense lane and [`attention_step`] was fused, they were not, and
    // [`drift_table_at_real_width`]'s batch=1 column — which had been exactly
    // zero — moved to 4e-3.
    if tree.is_none() && flash_cached_applies(d) {
        let eff = d
            .rel_extent
            .min(pos0 + rows - base)
            .next_multiple_of(kv_pad_bucket())
            .min(d.rel_extent);
        let taus: Vec<f32> = (0..rows)
            .map(|i| match (d.kind, log_scaling) {
                (AttnKind::Global, Some(ls)) => ls.tau(pos0 + i),
                _ => 1.0,
            })
            .collect();
        let tau: Tensor<Bk, 1> = Tensor::from_data(TensorData::new(taus, [rows]), &dev);
        let q = q * tau.clone().reshape([rows, 1]);
        let rel = r
            .reshape([rows * heads, d.d_rel])
            .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
            .reshape([rows, heads, eff])
            * tau.reshape([rows, 1, 1]);
        let out = flash_cached(q, rel, cache, &dev, d, rows, pos0, eff, window);
        return linear_bf16(out, &w.wo);
    }

    let bucket = kv_pad_bucket();
    // The paged read, the `rows > 1` twin of [`attention_step`]'s. See
    // [`PagedKv`] for what `head` and the tail padding are and why the mask
    // below covers them rather than the pages being sliced.
    let kv = PagedKv::read(cache, &dev, (kv_heads, head_dim), bucket);
    let (head, slots) = (kv.head, kv.slots);

    // Row `i`'s ABSOLUTE position. `pos0 + i` for a chain; `pos0 + depth[i]`
    // for a tree, where siblings share one. The cache SLOT is still `i`
    // either way — the two indices coincide for a chain and that coincidence
    // is what a tree removes.
    let pos_row: Vec<usize> = match tree {
        None => (0..rows).map(|i| pos0 + i).collect(),
        Some(t) => t.positions(pos0),
    };
    let first_row = len - rows;

    let taus: Vec<f32> = pos_row
        .iter()
        .map(|&pos| match (d.kind, log_scaling) {
            (AttnKind::Global, Some(ls)) => ls.tau(pos),
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
    let mut idx = vec![0i32; rows * slots];
    let mut valid = vec![0f32; rows * slots];
    let mut wmask = vec![0f32; rows * slots];
    let mut max_dist = 0usize;
    for i in 0..rows {
        let pos = pos_row[i];
        for s in 0..slots {
            let cell = i * slots + s;
            if s < head || s >= head + len {
                wmask[cell] = f32::NEG_INFINITY;
                continue;
            }
            let j = s - head;
            // A key that is one of THIS batch's rows is a tree node, and its
            // position is its depth's, not its slot's. A key in the committed
            // prefix is where it always was.
            let (abs, admits) = match tree {
                Some(t) if j >= first_row => {
                    let r = j - first_row;
                    (pos_row[r], t.visible[i][r])
                }
                _ => (base + j, base + j <= pos),
            };
            if !admits {
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
    let eff = d
        .rel_extent
        .min(max_dist + 1)
        .next_multiple_of(bucket)
        .min(d.rel_extent);
    let idx: Tensor<Bk, 3, Int> =
        Tensor::from_data(TensorData::new(idx, [1, rows, slots]), &dev).repeat_dim(0, heads);
    let valid: Tensor<Bk, 3> = Tensor::from_data(TensorData::new(valid, [1, rows, slots]), &dev);
    let wmask: Tensor<Bk, 3> = Tensor::from_data(TensorData::new(wmask, [1, rows, slots]), &dev);

    let rel = (r
        .reshape([rows * heads, d.d_rel])
        .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
        .reshape([rows, heads, eff])
        .swap_dims(0, 1))
        * tau.reshape([1, rows, 1]);
    let bias = rel.gather(2, idx) * valid;

    // GQA without materialising the expansion, the `rows > 1` twin of the same
    // change in [`attention_step`]. Here it matters more: this is the shape a
    // speculative VERIFY runs in, so the expanded copy was `rows` queries
    // against a 4x copy of the cache.
    //
    // `q` is `[rows, heads * head_dim]` and head `k * groups + g` reads KV head
    // `k`, so the permutation below gathers the (group, row) pairs that share a
    // KV head into that batch entry's rows. It permutes `q`, which is
    // `rows * heads * head_dim`; the expansion permuted the CACHE.
    let qg = q
        .reshape([rows, kv_heads, groups, head_dim])
        .swap_dims(0, 1)
        .swap_dims(1, 2)
        .reshape([kv_heads, groups * rows, head_dim]);

    // `[kv_heads, groups * rows, slots]` back to `[heads, rows, slots]`:
    // `kv_heads` and `groups` collapse to `heads` in that order, which is the
    // correspondence the old repeat produced.
    let scores = kv
        .scores(qg)
        .reshape([heads, rows, slots])
        .mul_scalar(d.scaling())
        + bias
        + wmask;
    let probs = burn::tensor::activation::softmax(scores, 2);
    let out = kv
        .weighted_values(probs.reshape([kv_heads, groups * rows, slots]))
        .reshape([kv_heads, groups, rows, head_dim])
        .swap_dims(1, 2)
        .swap_dims(0, 1)
        .reshape([rows, heads * head_dim]);
    linear_bf16(out, &w.wo)
}

/// One position of the short convolution for `slots` INDEPENDENT sequences.
///
/// [`short_conv_step`] with a slot dimension. `hist` is
/// `[slots, kernel - 1, dim]` and `x` is `[slots, dim]`; the returned history
/// is the one to carry into the next position of each slot.
///
/// Not [`short_conv_steps`] with `rows = slots`: that function's rows are
/// CONSECUTIVE positions of one sequence and overlap by `kernel - 1`, so it
/// would convolve slot `s`'s output out of slot `s - 1`'s inputs. The result
/// still reads as fluent text, which is exactly why the two shapes need
/// different functions rather than one function and a comment.
/// Carries [`short_conv_step`]'s partial-sum rule: a residual computed in
/// PIECES must be summed BEFORE this call, never after it. See that function.
pub fn short_conv_slot_step(
    hist: Tensor<Bk, 3>,
    x: Tensor<Bk, 2>,
    weight: Tensor<Bk, 2>,
) -> (Tensor<Bk, 2>, Tensor<Bk, 3>) {
    let [slots, dim] = x.dims();
    let [wdim, kernel] = weight.dims();
    assert_eq!(
        dim, wdim,
        "short_conv_slot_step: x is [_, {dim}] but the weight is [{wdim}, _]"
    );
    assert_eq!(
        hist.dims(),
        [slots, kernel - 1, dim],
        "each of the {slots} slots needs its own {} history rows",
        kernel - 1
    );
    let client = client_of(&x);
    let dev = x.device();
    let (h_hist, h_x, h_w) = (handle_of(hist), handle_of(x), handle_of(weight));
    let (out, next) = crate::models::inkling::sconv::short_conv_slots(
        &client, &h_hist, &h_x, &h_w, dim, slots, kernel,
    );
    (
        tensor_of(client.clone(), dev.clone(), out, slots, dim),
        tensor_of3(client, dev, next, slots, kernel - 1, dim),
    )
}

/// The attention state of `b` INDEPENDENT decode slots, advancing in lockstep.
///
/// [`AttnCache`] is one sequence's keys and values; this is `b` of them, and
/// the difference from [`attention_steps`] — which also takes several rows — is
/// the whole point. That function's rows are consecutive positions of ONE
/// sequence: they share a cache, row `i` sits at `pos0 + i`, and its mask
/// admits every earlier row of the batch. These rows are `b` sequences that
/// share nothing but the weights.
///
/// ## The block-diagonal mask is the batch dimension
///
/// Batched decode is usually described as wanting "a block-diagonal mask
/// instead of a causal one", and that is true of a layout that concatenates the
/// slots along the KEY axis. This one does not: K and V are held
/// `[slots * kv_heads, cap, head_dim]` and the scores are a BATCHED matmul over
/// that leading axis, so slot `s`'s query multiplies slot `s`'s keys and no
/// others. The separation is structural — there is no mask element whose sign
/// could be wrong, and no way to write one that leaks — and what is left for
/// the mask to say is exactly what a single-row step's mask says: how far the
/// relative table reaches, and what the sliding window admits.
///
/// ## Head-major, and why the layout is not [`AttnCache`]'s
///
/// [`AttnCache`] holds K as `[len, kv_heads * head_dim]` and expands it to
/// `[heads, len, head_dim]` per step, repeating each KV head `groups` times so
/// the score matmul can be one batched GEMM over `heads`. At one row that
/// materialises 63 MB a layer at a 3.8k context; at eight slots it would be
/// 500 MB a layer, twice (K and V), on every layer of every step — about 12 GB
/// of pure copy per pass, which would have priced the b-th cache read at
/// something that was never about the cache.
///
/// So the slot cache is head-major from the start, `[slots * kv_heads, cap,
/// head_dim]`, and the GQA repetition moves to the QUERY side where it is free:
/// the `groups` queries that share a KV head become the `m` rows of that head's
/// GEMM instead of `groups` copies of its keys. `q` reshapes to
/// `[slots * kv_heads, groups, head_dim]` with no transpose at all, because
/// head `h` is `kv_h * groups + g` and that is exactly the order a `[slots,
/// heads * head_dim]` projection is already in.
///
/// ## `cap` against `len`
///
/// The tensors are allocated to `cap`, a multiple of the KV pad bucket, and
/// only `len` of those rows are real. Appending is then a `slice_assign` of one
/// row rather than a `cat` that copies the whole cache, and the bucket-sized
/// growth keeps cubecl's compiled-kernel cache keyed on a handful of shapes for
/// the same reason [`attention_step`] rounds its context length up. The rows
/// between `len` and `cap` are ZERO, not uninitialised: they score nothing
/// against any query, but they are also multiplied by a probability of exactly
/// zero, and `0 * NaN` is NaN.
///
/// ## What is deliberately not here
///
/// A slot scheduler. Every slot holds the same number of keys and stands at the
/// same absolute position, because they were prefilled with prompts of the same
/// length and advance one token per pass together. Admission, eviction and
/// ragged lengths are a layer above this and would want a per-slot `len` and a
/// key-axis mask to go with it; nothing here would have to change shape for
/// that, which is why it is worth building the rectangular case first.
pub struct SlotCache<B: Backend> {
    /// `[slots * kv_heads, kcap, head_dim]`, post-convolution and
    /// post-QK-norm. The first `frozen` rows are real; the rest are the pad the
    /// prefill was widened by and are masked.
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    /// `[slots * kv_heads, recent_rows(), head_dim]`: the rows written since
    /// the last merge. The first `recent` are real and the rest are zero.
    kr: Tensor<B, 3>,
    vr: Tensor<B, 3>,
    /// `[slots, kernel - 1, kv_heads * head_dim]` — the short convolution's
    /// memory, per slot, PRE-convolution. Same reason [`AttnCache`] keeps it:
    /// the next position's taps reach back into projections the cached K and V
    /// cannot reconstruct.
    k_pre: Tensor<B, 3>,
    v_pre: Tensor<B, 3>,
    slots: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Real rows at the front of `k`/`v`.
    frozen: usize,
    /// Rows allocated in `k`/`v`. Equal to `frozen` after any merge; larger
    /// only between [`SlotCache::from_prefills`] and the first one.
    kcap: usize,
    /// Real rows in `kr`/`vr`, at most [`recent_rows`].
    recent: usize,
    /// Absolute position of frozen row 0.
    base: usize,
}

/// Rows the recent half holds before it is merged into the frozen half.
///
/// Two KV pad buckets. It is the amortisation constant of the whole layout: a
/// step rewrites one of these rows and a merge rewrites the context, so the
/// copy per step is `L / recent_rows()` rows rather than `L`.
fn recent_rows() -> usize {
    2 * kv_pad_bucket()
}

impl<B: Backend> SlotCache<B> {
    /// Keys retained per slot — *not* the sequence length, because a windowed
    /// layer forgets.
    pub fn len(&self) -> usize {
        self.frozen + self.recent
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Absolute position of row 0.
    pub fn base(&self) -> usize {
        self.base
    }

    pub fn slots(&self) -> usize {
        self.slots
    }
}

/// Move a tensor out of a field so the op that consumes it can do so IN PLACE.
///
/// `t.clone().slice_assign(..)` leaves two live references to the same buffer,
/// and Burn answers that by copying the whole tensor before writing one row of
/// it. The placeholder is `Tensor::empty`, an allocation and no kernel, and it
/// is overwritten before anything can read it.
fn take3<B: Backend>(slot: &mut Tensor<B, 3>, dev: &B::Device) -> Tensor<B, 3> {
    std::mem::replace(slot, Tensor::empty([1, 1, 1], dev))
}

/// Write one slot's row into a `[slots, ..]` batch, in place.
///
/// `take3` first, because `dst.clone().slice_assign(..)` leaves two live
/// references to the same buffer and Burn answers that by copying the whole
/// batch to write one row of it.
pub fn seat_row3(dst: &mut Tensor<Bk, 3>, s: usize, src: Tensor<Bk, 3>) {
    let [slots, a, b] = dst.dims();
    assert!(s < slots, "slot {s} of a {slots}-slot batch");
    assert_eq!(
        src.dims(),
        [1, a, b],
        "a seated row is one slot of [{a}, {b}]"
    );
    let dev = src.device();
    let d = take3(dst, &dev);
    *dst = d.slice_assign([s..s + 1, 0..a, 0..b], src);
}

/// A `[slots, ..]` batch with slot 0's row in it and the rest zero.
pub fn seat_first3(slots: usize, src: Tensor<Bk, 3>) -> Tensor<Bk, 3> {
    let [one, a, b] = src.dims();
    assert_eq!(one, 1, "a seated row is one slot");
    let mut dst: Tensor<Bk, 3> = Tensor::zeros([slots, a, b], &src.device());
    seat_row3(&mut dst, 0, src);
    dst
}

impl SlotCache<Bk> {
    /// An empty `slots`-wide batch shaped by the FIRST prefilled slot, with
    /// that slot already seated.
    ///
    /// The prefills run one at a time — a prefill is compute-bound and gains
    /// nothing from a batch — and this is the seam where they become a batch.
    /// Every slot must hold the same number of keys and start at the same
    /// absolute position: that is what makes the rectangular layout legal, and
    /// it is guaranteed by prompts of equal length rather than assumed, one
    /// slot at a time in [`SlotCache::seat`].
    ///
    /// # Why the batch is built now and not when the b-th prefill lands
    ///
    /// It used to take all `b` finished caches at once, and holding them was
    /// the most expensive thing this lane did. A prefilled `AttnCache` is not
    /// the `keep * kv_width * 4` bytes its contents come to — measured on the
    /// 21-layer head at eight slots and a 3732-token prompt, each one held
    /// **3.59 GiB**, against 0.16 GiB of keys and values, and the `cat` in here
    /// is what collapsed it. Eight of those is 28.7 GiB on a node with 25.8 GiB
    /// of headroom, so the sixth prefill ran the machine out of memory: the
    /// first five took 6.7 s each and the sixth through eighth took 118 s,
    /// 244 s and 57 s, the box swapped, and the decode that followed spent
    /// dozens of passes climbing back out.
    ///
    /// Seating each slot the moment it is prefilled makes the batch the ONLY
    /// long-lived allocation, and it is made once, from slot 0, before any of
    /// the churn. Every later prefill leaves nothing behind.
    pub fn seeded(
        slots: usize,
        first: AttnCache<Bk>,
        kv_heads: usize,
        head_dim: usize,
    ) -> SlotCache<Bk> {
        assert!(slots >= 1, "a slot batch has at least one slot");
        let len = first.len();
        let base = first.base();
        let kernel_hist = first.k_pre.dims()[0];
        let kv_width = kv_heads * head_dim;
        // `k_pre` rather than `k`: same cache, same device, and it is still a
        // plain tensor now that K and V are paged.
        let dev = first.k_pre.device();
        let rows = slots * kv_heads;
        let rec = recent_rows();
        let kcap = (len + 1).next_multiple_of(kv_pad_bucket());
        let mut c = SlotCache {
            // K and V are the bytes this lane moves and the bytes it holds;
            // `k_pre`/`v_pre` are `kernel - 1` rows and stay f32, because they
            // feed a convolution whose output is normed and would pay the
            // rounding twice.
            k: as_kv(Tensor::zeros([rows, kcap, head_dim], &dev)),
            v: as_kv(Tensor::zeros([rows, kcap, head_dim], &dev)),
            kr: as_kv(Tensor::zeros([rows, rec, head_dim], &dev)),
            vr: as_kv(Tensor::zeros([rows, rec, head_dim], &dev)),
            k_pre: Tensor::zeros([slots, kernel_hist, kv_width], &dev),
            v_pre: Tensor::zeros([slots, kernel_hist, kv_width], &dev),
            slots,
            kv_heads,
            head_dim,
            frozen: len,
            kcap,
            recent: 0,
            base,
        };
        c.seat(0, first);
        c
    }

    /// Seat one prefilled slot.
    ///
    /// The shape agreement the rectangular layout needs, checked against the
    /// slot that defined it rather than across a batch that is all present at
    /// once.
    pub fn seat(&mut self, s: usize, c: AttnCache<Bk>) {
        assert!(s < self.slots, "slot {s} of a {}-slot batch", self.slots);
        assert_eq!(
            c.len(),
            self.frozen,
            "slot {s} holds {} keys against slot 0's {}",
            c.len(),
            self.frozen
        );
        assert_eq!(
            c.base(),
            self.base,
            "slot {s} starts at {} against slot 0's {}",
            c.base(),
            self.base
        );
        let (kv_heads, head_dim, kcap, len) =
            (self.kv_heads, self.head_dim, self.kcap, self.frozen);
        let kv_width = kv_heads * head_dim;
        assert_eq!(
            c.k.width(),
            kv_width,
            "slot {s} was built at a different layer shape"
        );
        let dev = c.k_pre.device();
        let kernel_hist = c.k_pre.dims()[0];

        // `[len, kv_heads * head_dim]` -> `[kv_heads, kcap, head_dim]`. Every
        // later step writes and reads head-major and transposes nothing.
        //
        // The pad is not optional and not for the mask's benefit. `swap_dims`
        // returns a permuted VIEW, and this runtime fails to `slice_assign` one
        // — CUDA_ERROR_INVALID_VALUE and no other symptom. Concatenating the
        // pad along dim 1 materialises the permutation, so what reaches the
        // batch is a real layout. The tell was that only prefill lengths which
        // were a multiple of the KV pad bucket died — exactly the ones with
        // nothing to pad.
        let headwise = |t: Tensor<Bk, 2>| -> Tensor<Bk, 3> {
            // The pad is narrowed with the body rather than after it: `cat`
            // over two dtypes is a promotion nobody asked for, and the seam
            // between a BF16 cache and an f32 zero block is exactly where one
            // would happen unnoticed.
            let pad: Tensor<Bk, 3> = as_kv(Tensor::zeros([kv_heads, kcap - len, head_dim], &dev));
            let body = as_kv(t.reshape([len, kv_heads, head_dim]).swap_dims(0, 1));
            Tensor::cat(vec![body, pad], 1)
        };
        let r0 = s * kv_heads;
        let k = take3(&mut self.k, &dev);
        self.k = k.slice_assign(
            [r0..r0 + kv_heads, 0..kcap, 0..head_dim],
            headwise(c.k.materialize(&dev)),
        );
        let v = take3(&mut self.v, &dev);
        self.v = v.slice_assign(
            [r0..r0 + kv_heads, 0..kcap, 0..head_dim],
            headwise(c.v.materialize(&dev)),
        );
        seat_row3(
            &mut self.k_pre,
            s,
            c.k_pre.reshape([1, kernel_hist, kv_width]),
        );
        seat_row3(
            &mut self.v_pre,
            s,
            c.v_pre.reshape([1, kernel_hist, kv_width]),
        );
    }

    /// `b` prefilled single-sequence caches, stacked into one slot batch.
    ///
    /// What [`SlotCache::seeded`] and [`SlotCache::seat`] do over the b prefill
    /// passes, done at once. Only the tests use it: it is the shortest way to
    /// say "these b caches, as a batch" when all b are already in hand, and the
    /// run cannot afford to hold all b.
    pub fn from_prefills(
        caches: Vec<AttnCache<Bk>>,
        kv_heads: usize,
        head_dim: usize,
    ) -> SlotCache<Bk> {
        assert!(!caches.is_empty(), "a slot batch has at least one slot");
        let slots = caches.len();
        let mut it = caches.into_iter();
        let first = it.next().expect("a slot batch has at least one slot");
        let mut batch = SlotCache::seeded(slots, first, kv_heads, head_dim);
        for (s, c) in it.enumerate() {
            batch.seat(s + 1, c);
        }
        batch
    }

    /// Append one key and value per slot.
    ///
    /// Into the recent half, which is [`recent_rows`] wide however long the
    /// context is. That is the point of the split: `slice_assign` copies the
    /// tensor it writes into whenever anything else still holds a reference to
    /// it, and at eight slots and a 3.8k context the whole cache is 126 MB a
    /// tensor — 5.3 GB of copy a pass over 21 layers, and 20 GiB of allocator
    /// pages to hold it, on a node with 24 GiB of headroom. Written this way a
    /// step touches 4 MB and the context is copied once every
    /// [`recent_rows`] steps.
    fn push(&mut self, k_new: Tensor<Bk, 2>, v_new: Tensor<Bk, 2>) {
        let rows = self.slots * self.kv_heads;
        let hd = self.head_dim;
        let rec = recent_rows();
        let dev = k_new.device();
        if self.recent == rec {
            self.merge();
        }
        let r = self.recent;
        let kn = as_kv(k_new.reshape([rows, 1, hd]));
        let vn = as_kv(v_new.reshape([rows, 1, hd]));
        let kr = take3(&mut self.kr, &dev);
        let vr = take3(&mut self.vr, &dev);
        self.kr = kr.slice_assign([0..rows, r..r + 1, 0..hd], kn);
        self.vr = vr.slice_assign([0..rows, r..r + 1, 0..hd], vn);
        self.recent += 1;
    }

    /// Fold the recent half into the frozen one.
    ///
    /// The one place the whole context is copied, and it happens once every
    /// [`recent_rows`] steps. The recent half is FULL here — a partial merge
    /// would put a zero row inside the frozen half, where no mask covers it.
    fn merge(&mut self) {
        let rec = recent_rows();
        assert_eq!(
            self.recent, rec,
            "a partial recent half has zero rows in it"
        );
        let rows = self.slots * self.kv_heads;
        let hd = self.head_dim;
        let dev = self.k.device();
        let frozen = self.frozen;
        let k = take3(&mut self.k, &dev);
        let v = take3(&mut self.v, &dev);
        let kr = take3(&mut self.kr, &dev);
        let vr = take3(&mut self.vr, &dev);
        // The pad the prefill was widened by, and anything a trim left behind,
        // go here: the frozen half is sliced to its real rows on the way in, so
        // after a merge `kcap` is `frozen` and every column is a key.
        let real = |t: Tensor<Bk, 3>| t.slice([0..rows, 0..frozen, 0..hd]);
        self.k = Tensor::cat(vec![real(k), kr], 1);
        self.v = Tensor::cat(vec![real(v), vr], 1);
        self.kr = as_kv(Tensor::zeros([rows, rec, hd], &dev));
        self.vr = as_kv(Tensor::zeros([rows, rec, hd], &dev));
        self.frozen = frozen + rec;
        self.kcap = self.frozen;
        self.recent = 0;
    }

    /// Drop the keys no future query can reach — in whole [`recent_rows`]
    /// chunks.
    ///
    /// [`trim`] drops down to exactly `window` rows every step, which is a copy
    /// of the whole cache per layer per step; here the extra rows past the
    /// window are masked to `-inf` anyway, so the trim waits until it has a
    /// chunk's worth to move and rides the same amortisation as the merge. The
    /// cost of the delay is that `len` may exceed `window`, which is why the
    /// window predicate below is on the DISTANCE and not on the row count.
    fn trim(&mut self, window: Option<usize>) {
        let Some(w) = window else { return };
        let len = self.len();
        if len <= w {
            return;
        }
        let rec = recent_rows();
        let drop = ((len - w) / rec * rec).min(self.frozen);
        if drop == 0 {
            return;
        }
        let rows = self.slots * self.kv_heads;
        let (kcap, hd) = (self.kcap, self.head_dim);
        let dev = self.k.device();
        let k = take3(&mut self.k, &dev);
        let v = take3(&mut self.v, &dev);
        self.k = k.slice([0..rows, drop..kcap, 0..hd]);
        self.v = v.slice([0..rows, drop..kcap, 0..hd]);
        self.frozen -= drop;
        self.kcap -= drop;
        self.base += drop;
    }
}

/// `b` INDEPENDENT sequences, one generated position each, through one
/// attention layer.
///
/// The batched-decode twin of [`attention_step`], and everything that separates
/// it from [`attention_steps`] is in [`SlotCache`]'s header: these rows do not
/// see each other, at all, because they are not in each other's key axis.
///
/// `pos` is the absolute position of the row every slot is about to write, and
/// it is one number rather than `b` of them because the slots advance in
/// lockstep from prompts of equal length — see [`SlotCache`]. Log scaling and
/// the relative bias are functions of that position, so they are the same
/// scalar for every slot, which is the only place the uniformity is used.
///
/// ## The softmax is split because the cache is
///
/// The keys arrive in two tensors, the frozen context and the last `RECENT`
/// rows, and joining them to run one softmax would copy the context every step
/// — which is the copy the split exists to remove. So the softmax is taken
/// across both halves without joining them: one max, one denominator, and the
/// two value products added. That is the same arithmetic a single softmax
/// performs, in the same stable form, and it is the reason the split costs a
/// second matmul rather than a second pass over the context.
pub fn attention_slots(
    x: Tensor<Bk, 2>,
    w: &AttnWeightsDev,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    pos: usize,
    window: Option<usize>,
    cache: &mut SlotCache<Bk>,
) -> Tensor<Bk, 2> {
    use crate::models::inkling::config::AttnKind;

    let [slots, hidden] = x.dims();
    assert_eq!(
        slots, cache.slots,
        "x has {slots} rows against a {}-slot cache",
        cache.slots
    );
    assert_eq!(
        hidden, d.hidden,
        "x is [_, {hidden}] but the config says {}",
        d.hidden
    );
    assert!(
        pos >= cache.base + cache.len(),
        "position {pos} is already cached"
    );
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(
        groups * kv_heads,
        heads,
        "{heads} heads do not divide into {kv_heads} kv heads"
    );
    assert_eq!(
        kv_heads, cache.kv_heads,
        "this cache was built at a different layer shape"
    );

    let (q, k_proj, v_proj, r) = project_qkvr(x, w);
    let (k_new, k_hist) = short_conv_slot_step(cache.k_pre.clone(), k_proj, w.k_sconv.clone());
    let (v_new, v_hist) = short_conv_slot_step(cache.v_pre.clone(), v_proj, w.v_sconv.clone());
    cache.k_pre = k_hist;
    cache.v_pre = v_hist;

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k_new = head_rms_norm(k_new, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    cache.trim(window);
    cache.push(k_new, v_new);
    let (frozen, kcap, base) = (cache.frozen, cache.kcap, cache.base);
    let rec = recent_rows();
    let cap = kcap + rec;
    let rows = slots * kv_heads;

    let tau = match (d.kind, log_scaling) {
        (AttnKind::Global, Some(ls)) => ls.tau(pos),
        _ => 1.0,
    };

    // One row of what the prefill lane builds as a [tokens, tokens] table, and
    // it is one row for every slot: the backward distance to each retained key,
    // whether the relative table reaches that far, and whether the window
    // admits it. The slots share it because they share `pos` and `base`.
    //
    // Built over `cap` — the frozen rows followed by the WHOLE recent half —
    // because column `j` of the concatenation is at absolute position
    // `base + j` for every j below `len`, and the rows above it are the recent
    // half's unwritten tail. Those carry index 0 (in range for the gather,
    // multiplied out by `valid = 0`) and `-inf` in the mask.
    let mut idx = vec![0i32; cap];
    let mut valid = vec![0f32; cap];
    let mut wmask = vec![0f32; cap];
    let mut max_dist = 0usize;
    for (j, cell) in wmask.iter_mut().enumerate() {
        // Column `j` of the frozen half is at `base + j` while `j < frozen`;
        // column `kcap + i` of the recent half is at `base + frozen + i`. The
        // gap between them is the prefill's pad and is not a key.
        let at = if j < kcap {
            if j >= frozen {
                *cell = f32::NEG_INFINITY;
                continue;
            }
            j
        } else if j - kcap < cache.recent {
            frozen + (j - kcap)
        } else {
            *cell = f32::NEG_INFINITY;
            continue;
        };
        let dist = pos - (base + at);
        // Rows the chunked trim has not got around to dropping yet. They are
        // masked exactly as a key that left the window mid-batch would be, and
        // they are excluded from `max_dist` so the relative table stays the
        // width the single-row lane would have built.
        if window.is_some_and(|wnd| dist >= wnd) {
            *cell = f32::NEG_INFINITY;
            continue;
        }
        if dist < d.rel_extent {
            idx[j] = dist as i32;
            valid[j] = 1.0;
        }
        max_dist = max_dist.max(dist);
    }
    let bucket = kv_pad_bucket();
    let eff = d
        .rel_extent
        .min(max_dist + 1)
        .next_multiple_of(bucket)
        .min(d.rel_extent);

    let rel = r
        .reshape([slots * heads, d.d_rel])
        .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
        .reshape([rows, groups, eff])
        .mul_scalar(tau);

    // The two halves of every per-key table, so neither half of the scores has
    // to be joined to the other.
    let table = |from: usize, to: usize| -> (Tensor<Bk, 3>, Tensor<Bk, 3>) {
        let n = to - from;
        let i: Tensor<Bk, 3, Int> =
            Tensor::from_data(TensorData::new(idx[from..to].to_vec(), [1, 1, n]), &dev)
                .repeat_dim(0, rows)
                .repeat_dim(1, groups);
        let vt: Tensor<Bk, 3> =
            Tensor::from_data(TensorData::new(valid[from..to].to_vec(), [1, 1, n]), &dev);
        let m: Tensor<Bk, 3> =
            Tensor::from_data(TensorData::new(wmask[from..to].to_vec(), [1, 1, n]), &dev);
        (rel.clone().gather(2, i) * vt, m)
    };

    // The GQA repetition, on the query side: the `groups` heads that share KV
    // head `kv_h` are the `m` rows of its GEMM. `[slots, heads * head_dim]` is
    // already `[slots][kv_h][g][head_dim]` in memory, so this is a reshape and
    // not a permutation.
    // Narrowed on the QUERY side too, so both operands of the score matmul are
    // the same dtype: a mixed-dtype product is a supported thing on this fork
    // and is also a second lane to reason about, and `q` is `[rows, groups,
    // head_dim]` -- kilobytes against the cache's megabytes -- so nothing is
    // saved by leaving it wide.
    let qh = as_kv(q.mul_scalar(tau).reshape([rows, groups, head_dim]));
    let scores = |keys: Tensor<Bk, 3>, from: usize, to: usize| -> Tensor<Bk, 3> {
        let (bias, m) = table(from, to);
        // Back to f32 BEFORE the bias, the mask and the softmax. The scores are
        // `[slots * kv_heads, groups, cap]` -- 16 MB at 32 slots and a 3.8k
        // context, against 5.4 GB of cache -- so keeping the reduction's OUTPUT
        // wide costs nothing and keeps every op downstream of it, the softmax
        // included, exactly the arithmetic the f32 lane runs.
        from_kv(qh.clone().matmul(keys.swap_dims(1, 2))).mul_scalar(d.scaling()) + bias + m
    };
    let sf = scores(cache.k.clone(), 0, kcap);
    let sr = scores(cache.kr.clone(), kcap, cap);

    // The SCORES are joined and the keys are not, and that asymmetry is the
    // whole trade. `[slots * kv_heads, groups, cap]` is 4 MB at eight slots and
    // a 3.8k context, against 126 MB for one of K or V; joining the small thing
    // keeps ONE softmax over the row — the same op, in the same order, that
    // every other lane in this file uses — and joining the large one is the
    // copy the split exists to remove.
    //
    // A hand-written split softmax was tried here first and is not what this
    // is: taking the max, the exponentials and the denominator across two
    // tensors is the same arithmetic on paper and measured 2.1e-2 away from
    // the uncached lane where a joined softmax is BIT-IDENTICAL to it.
    let probs = burn::tensor::activation::softmax(Tensor::cat(vec![sf, sr], 2), 2);
    let pf = probs.clone().slice([0..rows, 0..groups, 0..kcap]);
    let pr = probs.slice([0..rows, 0..groups, kcap..cap]);
    // The probabilities are in [0, 1] and sum to one across the two halves, so
    // narrowing them costs a relative 2^-9 on each weight and nothing on their
    // sum; the values they weight are the other half of the cache and are the
    // bytes this is here for.
    let out =
        from_kv(as_kv(pf).matmul(cache.v.clone())) + from_kv(as_kv(pr).matmul(cache.vr.clone()));
    linear_bf16(out.reshape([slots, heads * head_dim]), &w.wo)
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
    /// layer.
    ///
    /// This was 2e-5 until 2026-08-25, on the premise that "both sides build
    /// their scores with the same Burn matmul and the only difference is the
    /// order of the additions". That premise is false, and nothing enforces
    /// it. Burn's matmul is AUTOTUNED PER SHAPE, and the two lanes are not the
    /// same shape: the cached side multiplies `[kv_heads, groups, head_dim]`
    /// against the retained keys, the uncached side `[heads, tokens,
    /// head_dim]` over the whole square. Different shape, different autotune
    /// entry, different winning kernel, different accumulation order. 2e-5
    /// held only while the two entries happened to name kernels whose TF32
    /// error cancelled -- and WHICH kernel wins is decided by a timing
    /// measurement cached under `target/autotune/`, which is per worktree and
    /// different in each. That is why these tests passed in one checkout and
    /// failed in another with identical source: 58 of the 64 matmul shapes two
    /// such caches shared named a different winner.
    ///
    /// The size is not a guess, and it is not the drift. It is what the
    /// UNCACHED lane -- the side this comparison treats as ground truth -- is
    /// itself worth. [`the_cache_algorithm_is_right_on_the_host`] measures both
    /// device lanes against the host transcription in
    /// [`crate::models::inkling::attn`], which shares no kernel, no matmul and
    /// no accumulation order with either: the uncached lane is 1.15e-2 away
    /// from it and the cached lane 1.19e-2. Neither is the accurate one, and
    /// their mutual disagreement of 1.43e-2 is two comparably-wrong numbers
    /// landing on opposite sides. Measured 2026-08-25 on the GB10, `cargo test
    /// --release --lib --features inkling-cuda`, global layer, 11 tokens,
    /// prefill 4, `rel_extent` 5, log scaling on -- identical to the last digit
    /// over five independent from-empty autotune states.
    ///
    /// So this is now [`CACHE_TOLERANCE`]'s number for
    /// [`CACHE_TOLERANCE`]'s reason -- TF32 through a softmax on these
    /// deliberately small synthetic weights -- and the global/local distinction
    /// it used to encode does not survive contact with an autotuned matmul.
    ///
    /// What these tests exist to catch is untouched:
    /// [`dropping_the_conv_history_is_caught`] moves the answer by 1.156,
    /// twenty-three times this bound. And the exact equivalence the 2e-5 was
    /// reaching for is not given up, it moved somewhere it can actually be
    /// asserted -- [`the_cache_algorithm_is_right_on_the_host`], where it holds
    /// to the last bit and no kernel choice can reach it.
    /// There was a second, looser constant here for LOCAL layers, on the
    /// grounds that only there do the two sides "no longer share an
    /// implementation" -- prefill through
    /// [`crate::models::inkling::banded`], which accumulates `q . k` in f32 on
    /// the CUDA cores, against a decode step through Burn's TF32 matmul. That
    /// distinction is gone because the paragraph above dissolved it: on a
    /// global layer the two sides do not share an implementation either, they
    /// share a FUNCTION whose implementation autotune picks per shape. One
    /// bound, one reason.
    ///
    /// The band remains the ACCURATE side, which is not an assumption:
    /// `banded::device_tests` checks the band's defining property -- a key
    /// outside the window cannot move the answer and a key inside it must --
    /// and `golden/paired/` runs thirty-five banded layers on every item it
    /// scores.
    ///
    /// 5e-2, and the ceiling is not taste: it is
    /// [`dropping_the_conv_history_is_caught`], which requires the sabotage it
    /// performs to move the answer by more than TWENTY times this number. The
    /// sabotage moves it by 1.156, so anything above 5.78e-2 disarms the one
    /// test that proves this comparison can still catch the bug it exists for.
    /// Widening past that is not a loosening, it is a deletion, and an attempt
    /// at 1e-1 was refused by exactly that assertion.
    ///
    /// Worth writing down because the margin is nearly gone: the widest
    /// cached-lane disagreement measured across autotune states is 3.9e-2,
    /// against a ceiling of 5.78e-2. There is about a third of an octave left
    /// between the arithmetic noise on this runtime and the point where this
    /// suite stops being able to tell the two apart. The next kernel that
    /// widens the spread does not fail this bound, it dissolves it -- and the
    /// answer then is not a bigger number here, it is more of the equivalence
    /// moved onto [`the_cache_algorithm_is_right_on_the_host`], where the
    /// margin is not finite.
    ///
    /// Being loose costs nothing it used to buy. The equivalence claim these
    /// tests were really making is now asserted exactly, and on a lane no
    /// kernel choice can reach, by
    /// [`the_cache_algorithm_is_right_on_the_host`]. What is left here is a
    /// sanity bound on the DEVICE, and a sanity bound is all a number sampled
    /// from a timing measurement can honestly be.
    const CACHE_TOLERANCE: f32 = 5e-2;

    /// The band against the dense triangle, which is NOT the comparison
    /// [`CACHE_TOLERANCE`] describes and does not belong under it.
    ///
    /// Everything [`CACHE_TOLERANCE`] bounds is one algorithm run two ways,
    /// where the disagreement is which kernel autotune picked for each shape.
    /// This is two DIFFERENT kernels -- [`crate::models::inkling::banded`]
    /// accumulating `q . k` in f32 on the CUDA cores, against the dense
    /// triangle's TF32 matmul -- so it carries the kernel-choice spread AND
    /// the difference between the two implementations, and it is larger for a
    /// reason rather than by concession.
    ///
    /// It was sharing [`CACHE_TOLERANCE`] at 5e-2 and sitting 1.2% above the
    /// worst value that had ever been seen, which is not a bound, it is a
    /// coincidence: [`banded_and_dense_agree_to_tf32`] measures 5.06e-2 in one
    /// autotune state and passes in another, same binary, same minute. 8e-2 is
    /// 1.6x the worst measured across states. Measured 2026-08-25 on the GB10,
    /// `cargo test --release --lib --features inkling-cuda`, worst over the
    /// nine configurations that test sweeps, in four autotune states.
    ///
    /// No sabotage floor constrains this one from above the way it constrains
    /// [`CACHE_TOLERANCE`], which is precisely why it should not be widened
    /// casually -- there is nothing here that would notice.
    const BAND_TOLERANCE: f32 = 8e-2;

    /// Deterministic filler. A fixed pattern rather than a seeded RNG so a
    /// failure is reproducible from the source alone.
    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.7919 + seed).sin() * 0.5 + (i as f32 * 0.1237).cos() * 0.25)
            .collect()
    }

    /// The 42-layer release's GLOBAL attention shape, from its own
    /// `config.json` — the one [`super::super::budget`]'s `small()` parses.
    ///
    /// The tests above run a four-head toy because what they check is
    /// structural. The two benchmarks below run THIS, because what they check
    /// is a cost, and a cost at a shape the model does not have is a number
    /// about nothing.
    fn real_global_dims() -> AttnDims {
        AttnDims {
            hidden: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            d_rel: 16,
            rel_extent: 1024,
            kernel: 4,
            rms_eps: 1e-6,
            kind: AttnKind::Global,
        }
    }

    /// The device barrier the timings below close on.
    ///
    /// A four-byte read rather than the output: it is ordered behind the
    /// launches on this thread's stream, which is what makes it a barrier,
    /// and it charges the measurement nothing. Without it these functions time
    /// the ENQUEUE. Lifted from [`super::super::bf16gemm`]'s bench, which says
    /// the same thing at more length.
    fn barrier(client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>) {
        let _ = client.read_one(client.empty(4));
    }

    /// One global PREFILL layer, timed at lengths, on whichever arm the binary
    /// was started with.
    ///
    /// **Framing rule.** Every number this prints is milliseconds for ONE
    /// attention layer over the whole prefill, at the 42-layer release's global
    /// shape (hidden 4096, 32 heads over 8 KV heads, head_dim 128, d_rel 16,
    /// rel_extent 1024), on a GB10, with the projections and the two short
    /// convolutions INCLUDED — it times `attention_prefill`, not the kernel.
    /// The model has seven such layers. `reserved` is what the cubecl pool is
    /// holding afterwards, which is the closest thing to a peak working set
    /// this seam exposes.
    ///
    /// Both arms must be run from THIS directory, because cubecl's autotune
    /// cache lives at `$CWD/target/autotune` and stores which kernel won a
    /// timing race — two worktrees name different winners for most shapes, so a
    /// cross-worktree comparison measures the cache and not the change:
    ///
    /// ```text
    /// INK_FLASH=1 cargo test --release --features inkling-cuda -- \
    ///     --ignored --nocapture flash_prefill_cost
    /// INK_FLASH=0 cargo test --release --features inkling-cuda -- \
    ///     --ignored --nocapture flash_prefill_cost
    /// ```
    #[test]
    #[ignore = "a benchmark, not a check"]
    fn flash_prefill_cost_at_length() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = real_global_dims();
        let w = weights(&d, &dev);
        let ls = Some(LogScaling {
            n_floor: 128000.0,
            alpha: 0.1,
        });
        let client = client_of(&Tensor::<B, 2>::zeros([1, 1], &dev));
        println!(
            "\nprefill, one global layer, GB10, INK_FLASH={}",
            if flash_lane() { 1 } else { 0 }
        );
        println!("  tokens        ms   reserved MiB");
        for tokens in [1024usize, 4096, 8192, 16_384, 32_768] {
            let xs: Tensor<B, 2> = Tensor::random(
                [tokens, d.hidden],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &dev,
            );
            // Two warm runs: the first call at a shape compiles kernels and
            // resolves autotune, and the first call at ANY shape also pays for
            // the pool growing under it. Neither is what this measures.
            for _ in 0..2 {
                let (o, c) = attention_prefill(xs.clone(), &w, &d, ls, None, None);
                drop((o, c));
            }
            barrier(&client);
            let t0 = std::time::Instant::now();
            let (out, c) = attention_prefill(xs, &w, &d, ls, None, None);
            core::hint::black_box(&out);
            barrier(&client);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let reserved = crate::models::inkling::seam::pool_reserved(&client);
            drop((out, c));
            println!(
                "  {tokens:>6}  {ms:>8.1}   {:>12.0}",
                reserved as f64 / (1 << 20) as f64
            );
        }
    }

    /// One global DECODE step, timed against the context behind it.
    ///
    /// **Framing rule.** Milliseconds for ONE `attention_step` — one generated
    /// token through ONE global attention layer, projections and short
    /// convolutions included — at the release's global shape on a GB10, against
    /// a synthetic NVFP4 KV cache of `context` keys. The model has seven such
    /// layers, so multiply by seven for the per-step attention cost of a decode
    /// and compare it against the ~46 ms the rest of a step takes. The cache is
    /// built by appending random rows rather than by prefilling, because a
    /// prefill at 262,144 would allocate 4.3 GB of activations to produce a
    /// cache this test only wants the SIZE of.
    ///
    /// Same two-run protocol as [`flash_prefill_cost_at_length`], same reason.
    #[test]
    #[ignore = "a benchmark, not a check"]
    fn flash_decode_cost_at_context() {
        use crate::models::inkling::kvpages::KvStore;
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = real_global_dims();
        let w = weights(&d, &dev);
        let ls = Some(LogScaling {
            n_floor: 128000.0,
            alpha: 0.1,
        });
        let kv_w = d.kv_heads * d.head_dim;
        let client = client_of(&Tensor::<B, 2>::zeros([1, 1], &dev));
        println!(
            "\ndecode, one global layer, GB10, INK_FLASH={}, NVFP4 KV",
            if flash_lane() { 1 } else { 0 }
        );
        println!("  context   window       ms");
        for (context, window) in [
            (4096usize, None),
            (16_384, None),
            (65_536, None),
            (262_144, None),
            // A LOCAL layer, which is thirty-five of the forty-two: the window
            // trims the cache to its last 512 keys on the first step, so what
            // this row measures is a 512-key read whatever `context` says.
            (16_384, Some(512usize)),
        ] {
            let fill_store = || {
                let mut st = KvStore::<Bk>::new(kv_w, burn::tensor::DType::BF16);
                let mut done = 0usize;
                while done < context {
                    let n = (context - done).min(8192);
                    st.append(as_kv(Tensor::<B, 2>::random(
                        [n, kv_w],
                        burn::tensor::Distribution::Uniform(-1.0, 1.0),
                        &dev,
                    )));
                    done += n;
                }
                st
            };
            let mut cache = AttnCache {
                k: fill_store(),
                v: fill_store(),
                k_pre: Tensor::zeros([d.kernel - 1, kv_w], &dev),
                v_pre: Tensor::zeros([d.kernel - 1, kv_w], &dev),
                base: 0,
                pending: None,
            };
            let x: Tensor<B, 2> = Tensor::random(
                [1, d.hidden],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &dev,
            );
            // Warm, then time, then WARM AND TIME AGAIN, and report the
            // second. Two warm steps at the head are not enough: the first
            // context in the loop pays for the pool growing under it as well
            // as for compilation, and it showed — 4,096 measured 3.6 ms as the
            // first row and 1.1 ms as the last, on both arms. Timing the same
            // context twice inside one run is what distinguishes "this length
            // is slow" from "this row was first".
            let mut ms = 0f64;
            let mut at = context;
            for _ in 0..2 {
                for _ in 0..2 {
                    let o = attention_step(x.clone(), &w, &d, ls, at, window, &mut cache);
                    core::hint::black_box(&o);
                    at += 1;
                }
                barrier(&client);
                let iters = 5usize;
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let o = attention_step(x.clone(), &w, &d, ls, at, window, &mut cache);
                    core::hint::black_box(&o);
                    at += 1;
                }
                barrier(&client);
                ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            }
            let wl = match window {
                Some(n) => n.to_string(),
                None => "-".to_string(),
            };
            println!("  {context:>7}  {wl:>7}  {ms:>7.2}");
        }
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
            assert!(
                Bf16W::tileable(rows, cols),
                "test weight {rows}x{cols} does not tile"
            );
            let mut bytes = Vec::with_capacity(rows * cols * 2);
            for x in fill(rows * cols, seed) {
                bytes.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes());
            }
            Bf16W {
                h: client.create_from_slice(&bytes),
                n: rows,
                k: cols,
                align: 16,
            }
        };
        let (q_w, kv_w) = (d.heads * d.head_dim, d.kv_heads * d.head_dim);
        AttnWeightsDev {
            wq: w16(q_w, d.hidden, 0.1),
            wqkvr: None,
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
        // The WIDE cache, explicitly. What follows compares the cached lane
        // against recomputing, at a tolerance sized for two implementations of
        // the SAME arithmetic -- which they are only while the cache is f32. A
        // narrow cache stores less; holding it to this bar would be asking the
        // wrong question, and `golden/paired/` is where it is asked instead.
        let _lane = CacheLane::wide();
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

    /// `slots` independent sequences through [`attention_slots`], each compared
    /// against its OWN uncached whole-sequence run.
    ///
    /// This is the contamination test and the equivalence test at once, and it
    /// is one function because they are one question. Every slot carries
    /// different filler, so if slot `s`'s query reached any key of slot `s'`
    /// the softmax would mix in values from a sequence that has nothing to do
    /// with it — and the oracle each slot is measured against is the whole-
    /// sequence lane run on that slot's tokens ALONE, which cannot contain the
    /// contamination by construction.
    ///
    /// Returns `(worst disagreement, smallest gap between two slots)`. The
    /// second number is what keeps the first honest: if the slots' outputs were
    /// all nearly equal the first assertion would pass on a batch that had
    /// collapsed to one sequence.
    fn slot_compare(
        kind: AttnKind,
        rel_extent: usize,
        window: Option<usize>,
        ls: Option<LogScaling>,
        tokens: usize,
        prefill: usize,
        slots: usize,
    ) -> (f32, f32) {
        // The WIDE cache, explicitly. What follows compares the cached lane
        // against recomputing, at a tolerance sized for two implementations of
        // the SAME arithmetic -- which they are only while the cache is f32. A
        // narrow cache stores less; holding it to this bar would be asking the
        // wrong question, and `golden/paired/` is where it is asked instead.
        let _lane = CacheLane::wide();
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = dims(kind, rel_extent);
        let w = weights(&d, &dev);
        let xs: Vec<Tensor<B, 2>> = (0..slots)
            .map(|s| {
                Tensor::from_data(
                    TensorData::new(
                        fill(tokens * d.hidden, 2.5 + s as f32 * 7.0),
                        [tokens, d.hidden],
                    ),
                    &dev,
                )
            })
            .collect();
        let full: Vec<Tensor<B, 2>> = xs
            .iter()
            .map(|x| attention(x.clone(), &w, &d, ls, window))
            .collect();
        let prefills: Vec<AttnCache<Bk>> = xs
            .iter()
            .map(|x| {
                attention_prefill(
                    x.clone().slice([0..prefill, 0..d.hidden]),
                    &w,
                    &d,
                    ls,
                    window,
                    window,
                )
                .1
            })
            .collect();
        let mut cache = SlotCache::from_prefills(prefills, d.kv_heads, d.head_dim);

        let (mut worst, mut closest) = (0f32, f32::INFINITY);
        for pos in prefill..tokens {
            let rows = Tensor::cat(
                xs.iter()
                    .map(|x| x.clone().slice([pos..pos + 1, 0..d.hidden]))
                    .collect(),
                0,
            );
            let got = attention_slots(rows, &w, &d, ls, pos, window, &mut cache);
            for s in 0..slots {
                let mine = got.clone().slice([s..s + 1, 0..d.hidden]);
                let want = full[s].clone().slice([pos..pos + 1, 0..d.hidden]);
                worst = worst.max((mine.clone() - want).abs().max().into_scalar());
                for other in (s + 1)..slots {
                    let theirs = got.clone().slice([other..other + 1, 0..d.hidden]);
                    closest = closest.min((mine.clone() - theirs).abs().max().into_scalar());
                }
            }
        }
        (worst, closest)
    }

    /// The NARROW cache, checked the only way a narrower float can be checked
    /// without an adjudicator: structurally.
    ///
    /// The tests above hold the cached lane to recomputing, at a tolerance that
    /// is a statement about two implementations of the same arithmetic. A BF16
    /// cache stores less and cannot meet that bar, and loosening it until it
    /// could would be inventing a tolerance to fit the answer. Whether the
    /// narrow lane is good ENOUGH is a capability question and is settled in
    /// `golden/paired/`.
    ///
    /// What can still be asserted exactly is the property batching can actually
    /// get wrong, and it holds at any dtype: slot 0's answer is a function of
    /// slot 0's tokens and of nothing else. So run the same slot beside three
    /// DIFFERENT sequences and beside three copies of ITSELF, and require the
    /// two to be bit-identical -- not close, identical, because contamination
    /// is fluent output and no error at all.
    ///
    /// The second assertion is what keeps the first honest: in the
    /// heterogeneous batch the four slots must DISAGREE, or a batch that had
    /// collapsed into one sequence would pass.
    #[test]
    fn narrow_slots_are_still_independent() {
        let _lane = CacheLane::narrow();
        let dev = burn::backend::cuda::CudaDevice::default();
        let (kind, rel_extent, window) = (AttnKind::Global, 8usize, None);
        let d = dims(kind, rel_extent);
        let w = weights(&d, &dev);
        let (tokens, prefill, slots) = (11usize, 6usize, 4usize);

        // `seeds[0]` is slot 0 in both batches; the rest differ between them.
        let run = |seeds: Vec<f32>| -> Vec<Tensor<B, 2>> {
            let xs: Vec<Tensor<B, 2>> = seeds
                .iter()
                .map(|&sd| {
                    Tensor::from_data(
                        TensorData::new(fill(tokens * d.hidden, sd), [tokens, d.hidden]),
                        &dev,
                    )
                })
                .collect();
            let prefills = xs
                .iter()
                .map(|x| {
                    attention_prefill(
                        x.clone().slice([0..prefill, 0..d.hidden]),
                        &w,
                        &d,
                        None,
                        window,
                        window,
                    )
                    .1
                })
                .collect();
            let mut cache = SlotCache::from_prefills(prefills, d.kv_heads, d.head_dim);
            let mut out = Vec::new();
            for pos in prefill..tokens {
                let rows = Tensor::cat(
                    xs.iter()
                        .map(|x| x.clone().slice([pos..pos + 1, 0..d.hidden]))
                        .collect(),
                    0,
                );
                out.push(attention_slots(rows, &w, &d, None, pos, window, &mut cache));
            }
            out
        };

        let het = run(vec![2.5, 9.5, 16.5, 23.5]);
        let hom = run(vec![2.5; slots]);
        assert_eq!(het.len(), hom.len());

        let (mut moved, mut closest) = (0f32, f32::INFINITY);
        for (a, b) in het.iter().zip(hom.iter()) {
            let mine = a.clone().slice([0..1, 0..d.hidden]);
            let same = b.clone().slice([0..1, 0..d.hidden]);
            moved = moved.max((mine.clone() - same).abs().max().into_scalar());
            for other in 1..slots {
                let theirs = a.clone().slice([other..other + 1, 0..d.hidden]);
                closest = closest.min((mine.clone() - theirs).abs().max().into_scalar());
            }
        }
        println!(
            "narrow slots: slot 0 moved {moved:e} between neighbours, closest pair {closest:e}"
        );
        assert_eq!(
            moved, 0.0,
            "slot 0's answer moved by {moved:e} when its NEIGHBOURS changed: the narrow batch is              reading keys that are not its own"
        );
        assert!(
            closest > 1e-3,
            "the four slots produced nearly the same answer ({closest:e}): the batch collapsed              into one sequence and the assertion above proves nothing"
        );
    }

    /// Eight independent slots on a GLOBAL layer, none of them the same text.
    #[test]
    fn slots_stay_independent_on_a_global_layer() {
        let (worst, closest) = slot_compare(AttnKind::Global, 8, None, None, 11, 6, 8);
        println!("8 slots, global: worst {worst:e}, closest pair {closest:e}");
        assert!(
            worst < CACHE_TOLERANCE,
            "a slot disagreed with its own uncached run by {worst:e}"
        );
        assert!(
            closest > 1e-3,
            "the slots produced nearly the same answer ({closest:e}): the batch is not carrying \
             eight different sequences and the assertion above proves nothing"
        );
    }

    /// The same on a LOCAL layer, where the window drops keys and the cache
    /// trims — so the slots forget together as well as remember together.
    #[test]
    fn slots_stay_independent_on_a_local_layer() {
        let (worst, closest) = slot_compare(AttnKind::Local, 5, Some(5), None, 11, 6, 8);
        println!("8 slots, local: worst {worst:e}, closest pair {closest:e}");
        assert!(
            worst < CACHE_TOLERANCE,
            "a slot disagreed with its own uncached run by {worst:e}"
        );
        assert!(
            closest > 1e-3,
            "the slots produced nearly the same answer ({closest:e})"
        );
    }

    /// Log scaling is a function of the absolute position, which every slot
    /// shares — so it is the one place the batch is allowed to be uniform, and
    /// the one place a wrong position would be invisible at short context.
    #[test]
    fn slots_carry_log_scaling() {
        let ls = Some(LogScaling {
            n_floor: 4.0,
            alpha: 0.1,
        });
        let (worst, closest) = slot_compare(AttnKind::Global, 8, None, ls, 11, 6, 4);
        println!("4 slots, global + log scaling: worst {worst:e}, closest pair {closest:e}");
        assert!(worst < CACHE_TOLERANCE, "worst {worst:e}");
        assert!(closest > 1e-3, "closest {closest:e}");
    }

    /// One slot through the batched lane against [`attention_step`] itself.
    ///
    /// The tests above hold the slot lane to the UNCACHED lane, which is the
    /// right oracle for "did the cache work" and a loose one — the two build
    /// their scores differently. This one holds it to the lane it is a batch
    /// of, where the only difference is a leading dimension of size one.
    ///
    /// That used to be read as "so the bar is rounding rather than tolerance",
    /// and the bar was 1e-5. A leading dimension of size one is not nothing: it
    /// is the `m` of the score matmul, so the two lanes key DIFFERENT autotune
    /// entries and can win different kernels, which is the same mechanism
    /// [`CACHE_TOLERANCE`] documents. Measured 1.97e-3 in one autotune state
    /// and under 1e-5 in three others, same binary.
    ///
    /// 1e-2 is five times the worst of those and still five times tighter than
    /// [`CACHE_TOLERANCE`], which is the point of keeping it separate: both
    /// lanes here are cached, over the same retained keys, so this comparison
    /// really is stricter than cached-against-uncached and should stay able to
    /// say so.
    #[test]
    fn one_slot_is_the_one_row_lane() {
        // The WIDE cache, explicitly. What follows compares the cached lane
        // against recomputing, at a tolerance sized for two implementations of
        // the SAME arithmetic -- which they are only while the cache is f32. A
        // narrow cache stores less; holding it to this bar would be asking the
        // wrong question, and `golden/paired/` is where it is asked instead.
        let _lane = CacheLane::wide();
        let dev = burn::backend::cuda::CudaDevice::default();
        let (kind, rel_extent, window) = (AttnKind::Global, 8usize, None);
        let d = dims(kind, rel_extent);
        let w = weights(&d, &dev);
        let (tokens, prefill) = (11usize, 6usize);
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(tokens * d.hidden, 2.5), [tokens, d.hidden]),
            &dev,
        );
        let head = xs.clone().slice([0..prefill, 0..d.hidden]);
        let mut one = attention_prefill(head.clone(), &w, &d, None, window, window).1;
        let mut batch = SlotCache::from_prefills(
            vec![attention_prefill(head, &w, &d, None, window, window).1],
            d.kv_heads,
            d.head_dim,
        );
        let mut worst = 0f32;
        for pos in prefill..tokens {
            let row = xs.clone().slice([pos..pos + 1, 0..d.hidden]);
            let a = attention_step(row.clone(), &w, &d, None, pos, window, &mut one);
            let b = attention_slots(row, &w, &d, None, pos, window, &mut batch);
            worst = worst.max((a - b).abs().max().into_scalar());
        }
        println!("one slot against the one-row lane: worst {worst:e}");
        assert!(
            worst < 1e-2,
            "a one-slot batch is not the one-row lane: {worst:e}"
        );
    }

    /// Burn's f32 matmul on this runtime is NOT f32, and this is the tripwire.
    ///
    /// It compares `Tensor::matmul` against the same product accumulated in f64
    /// on the host. TF32 carries ten mantissa bits and lands near 1e-3; true f32
    /// carries twenty-three and lands near 1e-7. Measured here: 9.3e-4.
    ///
    /// This is the one f64 comparison left in this tree and it is deliberate:
    /// its operands are thirty-two rows of synthetic sine filler, it computes no
    /// model math at all, and what it establishes is a property of the RUNTIME.
    /// It is also the fact that stops the next reader treating the f32 lane as
    /// ground truth, which is the mistake the f64 references that used to live
    /// in `banded.rs` and `sconv.rs` kept inviting. Do not read it as licence to
    /// gate model arithmetic on an f64 transcription: a closer float is not a
    /// more correct one for a model whose weights are four bits, and what
    /// decides those questions is `golden/paired/`.
    ///
    /// The assertion is deliberately the "still imprecise" direction. It is what
    /// [`CACHE_TOLERANCE`] is sized for, and if this test ever FAILS the runtime
    /// has moved to a real f32 product. That would not on its own earn a tight
    /// bound back -- the kernel-choice spread documented on [`CACHE_TOLERANCE`]
    /// would survive it -- but it would remove the largest term in it. A failure
    /// here is good news, not a regression.
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
        println!(
            "f32 matmul worst absolute error {worst:e} against a largest term of {scale:e} -> {rel:e}"
        );
        assert!(
            rel > 1e-5,
            "Burn's f32 matmul now agrees with f64 to {rel:e}: it is a real f32 product, and \
             the largest term in CACHE_TOLERANCE is gone"
        );
    }

    /// The band and the dense triangle, on the same weights, over the shapes
    /// the cached tests use.
    ///
    /// This is the check that was missing when the band landed: a kernel
    /// checked only against its own module's tests says nothing about whether
    /// it agrees with the lane it REPLACES. It did not -- by up to 2.2e-2 --
    /// and running this is how the TF32 matmul was found. Prints the
    /// disagreement per configuration and per row, so a failure names which one
    /// moved.
    ///
    /// The bound is [`CACHE_TOLERANCE`], and it is a bound on TF32 rather
    /// than on the band. A blocked dense lane must answer what an unblocked one
    /// does; whether either of them is the RIGHT answer is a capability
    /// question and is settled in `golden/paired/`.
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
    /// Both arms are the same weights through the same lane, and the tolerance
    /// is still [`CACHE_TOLERANCE`] rather than something tighter,
    /// because "only the matmul tiling differs" is not the small statement it
    /// reads as. The query block IS the `m` of the score matmul, so each block
    /// size draws its own autotune entry and can win a different kernel from
    /// the unblocked arm; TF32 accumulation order goes with it. This test held
    /// at 2e-5 in one worktree and failed at 3.9e-2 in another, on the same
    /// binary and the same second -- the whole of that difference was the two
    /// `target/autotune/` caches disagreeing about which kernel is fastest.
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
                AttnKind::Global => Some(LogScaling {
                    n_floor: 8.0,
                    alpha: 0.5,
                }),
                AttnKind::Local => None,
            };
            for tokens in [37usize, 64, 91] {
                let xs: Tensor<B, 2> = Tensor::from_data(
                    TensorData::new(fill(tokens * d.hidden, 0.05), [tokens, d.hidden]),
                    &dev,
                );
                let whole = attention_prefill_dense(xs.clone(), &w, &d, ls, win, win, Some(tokens))
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
                        worst < CACHE_TOLERANCE,
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
            (5, 16, 11), // triangle, table shorter than the sequence
            (16, 5, 11), // clipped band, table longer than the window
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
            println!(
                "rel_extent={rel_extent} window={win} tokens={tokens} -> {diff}  rows {per_row:?}"
            );
            worst_of_all = worst_of_all.max(diff);
        }
        assert!(
            worst_of_all < BAND_TOLERANCE,
            "the band disagrees with the triangle by {worst_of_all}, which is more than TF32 \
             explains"
        );
    }

    /// A global layer with log scaling that actually varies, and a relative
    /// table that runs out before the sequence does — so distances past
    /// `rel_extent` must contribute a zero bias rather than a gathered one.
    #[test]
    fn cached_global_matches_full() {
        let ls = Some(LogScaling {
            n_floor: 4.0,
            alpha: 0.5,
        });
        let worst = compare(AttnKind::Global, 5, None, ls, 11, 4, false);
        assert!(
            worst < CACHE_TOLERANCE,
            "cached global attention drifts by {worst}"
        );
    }

    /// The one oracle that can tell an ALGORITHM defect from ARITHMETIC: the
    /// host transcription in
    /// [`crate::models::inkling::attn`], which shares no kernel, no matmul and
    /// no accumulation order with either device lane.
    ///
    /// Three numbers per decoded position, all against the host uncached run:
    ///
    /// * `C = |host cached - host uncached|` — the cache's ALGORITHM, in plain
    ///   f32 on the CPU. If the paged read, the GQA regrouping, the
    ///   `k_pre`/`v_pre` history, `tau` or the window were wrong, the same
    ///   mistake is transcribed here and this number is large. If this is at
    ///   f32 rounding, the cached lane is a correct algorithm and nothing the
    ///   device does can be blamed on it.
    /// * `A = |device uncached - host uncached|` — what the ORACLE side of the
    ///   comparison is itself worth on this runtime.
    /// * `B = |device cached - host uncached|` — the same for the lane under
    ///   test.
    ///
    /// `A` is the number the tolerance forgot. If `A` is already the size of
    /// the drift the tests fail by, then the two device lanes are each about
    /// that far from the truth and neither is wrong — the 2e-5 is a bound on a
    /// difference that no longer cancels.
    #[test]
    fn the_cache_algorithm_is_right_on_the_host() {
        use crate::models::inkling::attn as cpu;
        let _lane = CacheLane::wide();
        let dev = burn::backend::cuda::CudaDevice::default();
        let (kind, rel_extent, window) = (AttnKind::Global, 5usize, None::<usize>);
        let ls = Some(LogScaling {
            n_floor: 4.0,
            alpha: 0.5,
        });
        let d = dims(kind, rel_extent);
        let (tokens, prefill) = (11usize, 4usize);
        let hid = d.hidden;

        // The SAME operand bits the device sees: the five projections are BF16
        // there, so the host copy is those values put through the identical
        // round-to-nearest-even. Everything else is f32 on both sides.
        let bf = |n: usize, seed: f32| -> Vec<f32> {
            fill(n, seed)
                .into_iter()
                .map(|x| half::bf16::from_f32(x).to_f32())
                .collect()
        };
        let (q_w, kv_w) = (d.heads * d.head_dim, d.kv_heads * d.head_dim);
        let wq = bf(q_w * hid, 0.1);
        let wk = bf(kv_w * hid, 0.2);
        let wv = bf(kv_w * hid, 0.3);
        let wr = bf(d.heads * d.d_rel * hid, 0.4);
        let wo = bf(hid * q_w, 0.5);
        let ks = fill(kv_w * d.kernel, 0.6);
        let vs = fill(kv_w * d.kernel, 0.7);
        let qn = fill(d.head_dim, 0.8);
        let kn = fill(d.head_dim, 0.9);
        let rp = fill(d.d_rel * d.rel_extent, 1.0);
        let hw = cpu::AttnWeights {
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wr: &wr,
            wo: &wo,
            k_sconv: &ks,
            v_sconv: &vs,
            q_norm: &qn,
            k_norm: &kn,
            rel_proj: &rp,
        };

        let xs_v = fill(tokens * hid, 2.5);
        let xs: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(xs_v.clone(), [tokens, hid]), &dev);
        let w = weights(&d, &dev);

        let host_full = cpu::attention(
            &xs_v,
            &hw,
            &d,
            ls,
            &cpu::causal_mask(tokens, window),
            tokens,
        );
        let (_, mut hcache) = cpu::attention_prefill(
            &xs_v[..prefill * hid],
            &hw,
            &d,
            ls,
            &cpu::causal_mask(prefill, window),
            prefill,
            window,
        );

        let dev_full = attention(xs.clone(), &w, &d, ls, window);
        let (_, mut dcache) = attention_prefill(
            xs.clone().slice([0..prefill, 0..hid]),
            &w,
            &d,
            ls,
            window,
            window,
        );

        let row = |t: &Tensor<B, 2>, r: usize| -> Vec<f32> {
            t.clone()
                .slice([r..r + 1, 0..hid])
                .into_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("f32 row")
        };
        let worst = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };

        let (mut wa, mut wb, mut wc) = (0f32, 0f32, 0f32);
        for pos in prefill..tokens {
            let hstep = cpu::attention_step(
                &xs_v[pos * hid..(pos + 1) * hid],
                &hw,
                &d,
                ls,
                pos,
                window,
                &mut hcache,
            );
            let dstep = attention_step(
                xs.clone().slice([pos..pos + 1, 0..hid]),
                &w,
                &d,
                ls,
                pos,
                window,
                &mut dcache,
            );
            let truth = &host_full[pos * hid..(pos + 1) * hid];
            let c = worst(&hstep, truth);
            let a = worst(&row(&dev_full, pos), truth);
            let b = worst(&row(&dstep, 0), truth);
            wc = wc.max(c);
            wa = wa.max(a);
            wb = wb.max(b);
            println!(
                "  pos {pos:>2}: C(host cached)={c:e}  A(dev uncached)={a:e}  B(dev cached)={b:e}"
            );
        }
        println!(
            "WORST  C(host cached, the ALGORITHM)={wc:e}  A(dev uncached)={wa:e}  B(dev cached)={wb:e}"
        );
        // The ONLY assertion here, and the only one in this module that an
        // autotune cache cannot reach. `A` and `B` are printed rather than
        // gated on purpose: they are properties of the runtime, they are the
        // evidence behind [`CACHE_TOLERANCE`], and gating on them would
        // re-create exactly the bound that has been failing.
        assert!(
            wc < 1e-5,
            "the cached lane's ALGORITHM disagrees with the uncached one by {wc:e} in plain host \
             f32, where no kernel choice and no autotune cache can explain it. This is the paged \
             read, the GQA regrouping, the k_pre/v_pre history, tau or the window being WRONG -- \
             not rounding, and not a tolerance that needs widening."
        );
    }

    /// A local layer whose window is shorter than the sequence, so the cache
    /// must forget: 11 tokens through a window of 5 drops six keys.
    #[test]
    fn cached_local_matches_full_across_the_window() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 4, false);
        assert!(
            worst < CACHE_TOLERANCE,
            "cached windowed attention drifts by {worst}"
        );
    }

    /// The cache must survive a prefill shorter than the convolution kernel,
    /// where the history is mostly the zero padding `short_conv` assumes.
    #[test]
    fn cached_matches_full_from_a_two_token_prefill() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 2, false);
        assert!(
            worst < CACHE_TOLERANCE,
            "cached attention from a short prefill drifts by {worst}"
        );
    }

    /// The same equivalence over a context long enough to be SEVERAL PAGES.
    ///
    /// Every test above it runs eleven tokens, which is one page and one chunk,
    /// so the whole of [`PagedKv`] — the per-chunk score product, the `cat` on
    /// the key axis, the per-chunk value product and its sum — was covered by
    /// the degenerate case where there is exactly one chunk. The failure that
    /// hides there is a chunk read in the wrong order or at the wrong offset,
    /// which cannot happen with one of them.
    ///
    /// The oracle is the uncached whole-sequence lane, which never touches the
    /// KV store at all, so it cannot share a paging mistake with the thing it
    /// is checking.
    ///
    /// ## This test is RED, and not because of paging
    ///
    /// [`cached_global_matches_full`] — eleven tokens, one chunk, the same
    /// comparison — already fails on `main` at 1.4341354e-2 against a
    /// [`CACHE_TOLERANCE`] of 2e-5, and has nothing to do with this
    /// file's read path. This is the same failure at a longer context, so it
    /// inherits it: 2.353859e-2 over 421 tokens.
    ///
    /// The number is the evidence that it is ONLY that, and it is a PAIRED
    /// figure — which is the only kind that means anything here. Measured
    /// 2026-08-25 on the GB10, `cargo test --release --lib --features
    /// inkling-cuda`, this test against the identical comparison run on the
    /// materialized read it replaces: 2.3562431e-2 both ways, three runs each,
    /// every run identical to the last digit. Four chunks summed give exactly
    /// what one contiguous buffer gave.
    ///
    /// It is paired because the ABSOLUTE value is not stable across autotune
    /// states: an earlier pass of the same two arms, on a GB10 whose autotune
    /// cache had been filled while another job held the GPU, reported
    /// 2.353859e-2 — again both ways, again identical. The two arms move
    /// TOGETHER when the winning matmul changes, which is itself the claim.
    /// So compare this against a same-session run of the other arm, never
    /// against the digits written here.
    #[test]
    fn cached_global_matches_full_across_pages() {
        let tokens = 3 * super::super::kvpages::PAGE + 37;
        let worst = compare(AttnKind::Global, 5, None, None, tokens, 4, false);
        println!("global, {tokens} tokens, {worst:e}");
        assert!(
            worst < CACHE_TOLERANCE,
            "cached global attention over {tokens} tokens drifts by {worst}"
        );
    }

    /// Several pages AND a window that forgets, so `drop_front` keeps a head
    /// that walks across page boundaries.
    ///
    /// The head is the reason a chunk is read whole rather than sliced: the
    /// dead rows at the front of chunk 0 are real keys the window has dropped,
    /// and they are removed by the mask rather than by a copy. A sign error
    /// there does not crash — it attends to keys the window forgot, which is
    /// exactly what this comparison sees.
    #[test]
    fn cached_local_matches_full_with_a_head_across_pages() {
        let page = super::super::kvpages::PAGE;
        let worst = compare(
            AttnKind::Local,
            5,
            Some(page + 19),
            None,
            3 * page,
            4,
            false,
        );
        // Green, and at the same number the materialized read gave:
        // 1.1367798e-2 both ways, 384 tokens through a 147-key window, GB10,
        // 2026-08-25. Paired, like the global twin above — read its note before
        // comparing anything to these digits.
        println!("local, 3 pages with a head, {worst:e}");
        assert!(
            worst < CACHE_TOLERANCE,
            "cached windowed attention over 3 pages drifts by {worst}"
        );
    }

    /// A gathered convolution with LINEAR taps is the contiguous kernel.
    ///
    /// The cheapest possible statement that [`short_conv_tree`] did not invent
    /// a second convolution: hand it the taps a chain produces and it must
    /// return what [`short_conv_window`] returns, bit for bit modulo the
    /// reassociation of a four-term sum.
    #[test]
    fn a_linear_tap_table_is_the_contiguous_convolution() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let (dim, kernel, rows) = (16usize, 4usize, 5usize);
        let all: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(
                fill((kernel - 1 + rows) * dim, 1.3),
                [kernel - 1 + rows, dim],
            ),
            &dev,
        );
        let wt: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(dim * kernel, 0.4), [dim, kernel]),
            &dev,
        );
        let want = short_conv_window(all.clone(), wt.clone(), rows);
        let taps = crate::models::inkling::spectree::TreeAttn::linear(rows, kernel).taps;
        let got = short_conv_tree(all, wt, &taps);
        let diff = (got - want).abs().max().into_scalar();
        assert!(diff < 1e-5, "the gathered convolution drifts by {diff}");
    }

    /// The claim a tree verify pass makes, stated as an equality it can fail.
    ///
    /// Row `i` of a TREE batch must equal the row it would have been in a
    /// LINEAR batch containing only its own path. That is one sentence and it
    /// covers all three of the things a tree changes at once — the mask (a
    /// sibling's key would show up in the softmax), the position (a sibling
    /// shares one, so the relative bias would be gathered at the wrong
    /// distance) and the convolutions (the second candidate would be
    /// convolved out of the first one's projections). Any one of the three
    /// left un-fixed moves this number, and none of them crashes.
    ///
    /// The branches are deliberately given very different filler, because a
    /// leak between two similar rows is a leak this test would not see.
    fn tree_branch_gap(kind: AttnKind, window: Option<usize>, prefill: usize) -> (f32, f32) {
        let _lane = CacheLane::wide();
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = dims(kind, 5);
        let w = weights(&d, &dev);
        let rows = prefill + 3;
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(rows * d.hidden, 2.5), [rows, d.hidden]),
            &dev,
        );
        // The two candidates share a position and must share nothing else.
        let alt: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(fill(d.hidden, -11.0), [1, d.hidden]), &dev);
        let row = |i: usize| xs.clone().slice([i..i + 1, 0..d.hidden]);
        let (_, base) = attention_prefill(
            xs.clone().slice([0..prefill, 0..d.hidden]),
            &w,
            &d,
            None,
            window,
            window,
        );

        let tree = crate::models::inkling::spectree::TreeSpec::breadth(2).unwrap();
        let attn = crate::models::inkling::spectree::tree_attn(&tree, d.kernel);
        let batch = Tensor::cat(vec![row(prefill), row(prefill + 1), alt.clone()], 0);
        let mut c_tree = base.clone();
        let got = attention_steps_tree(
            batch,
            &w,
            &d,
            None,
            prefill,
            window,
            &mut c_tree,
            Some(&attn),
        );

        // Each candidate's own linear world: the confirmed token followed by
        // that candidate alone, which is a chain and takes the ordinary path.
        let mut worst = 0f32;
        for (r, x) in [(1usize, row(prefill + 1)), (2, alt)] {
            let mut c = base.clone();
            let want = attention_steps(
                Tensor::cat(vec![row(prefill), x], 0),
                &w,
                &d,
                None,
                prefill,
                window,
                &mut c,
            );
            let diff = (got.clone().slice([r..r + 1, 0..d.hidden])
                - want.slice([1..2, 0..d.hidden]))
            .abs()
            .max()
            .into_scalar();
            worst = worst.max(diff);
        }
        // ...and the two candidates must not have collapsed onto each other,
        // which would make the equality above pass for the wrong reason.
        let gap = (got.clone().slice([1..2, 0..d.hidden]) - got.slice([2..3, 0..d.hidden]))
            .abs()
            .max()
            .into_scalar();
        (worst, gap)
    }

    #[test]
    fn a_tree_row_is_its_own_branch_global() {
        let (worst, gap) = tree_branch_gap(AttnKind::Global, None, 9);
        println!("tree/global: worst {worst:e}, branch gap {gap:e}");
        assert!(gap > 1e-3, "the two branches collapsed, gap {gap}");
        assert!(worst < CACHE_TOLERANCE, "a tree row drifts by {worst}");
    }

    #[test]
    fn a_tree_row_is_its_own_branch_local() {
        let (worst, gap) = tree_branch_gap(AttnKind::Local, Some(5), 9);
        println!("tree/local: worst {worst:e}, branch gap {gap:e}");
        assert!(gap > 1e-3, "the two branches collapsed, gap {gap}");
        assert!(worst < CACHE_TOLERANCE, "a tree row drifts by {worst}");
    }

    /// The block convolutions' rollback, gathered, is the slice it replaces.
    ///
    /// Same equality as `commit_rows` versus `commit`, one operator down: on a
    /// contiguous accepted set `conv_history_rows` must return exactly what
    /// `conv_history(all.slice([0..hist + keep]))` returns, which is what the
    /// decode loop takes today for `attn_sconv` and `mlp_sconv`.
    #[test]
    fn gathered_conv_history_on_a_prefix_is_the_slice() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let (dim, kernel, rows) = (16usize, 4usize, 4usize);
        let hist = kernel - 1;
        let all: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill((hist + rows) * dim, 0.9), [hist + rows, dim]),
            &dev,
        );
        for keep in 0..=rows {
            let want = conv_history(all.clone().slice([0..hist + keep, 0..dim]), kernel);
            let got = conv_history_rows(all.clone(), kernel, &(0..keep).collect::<Vec<_>>());
            assert_eq!(got.dims(), want.dims(), "keep={keep}");
            let diff = (got - want).abs().max().into_scalar();
            assert!(diff == 0.0, "keep={keep} differs by {diff}");
        }
        // ...and a scattered path takes the rows the path actually named.
        let got = conv_history_rows(all.clone(), kernel, &[0, 2]);
        let want = Tensor::cat(
            vec![
                all.clone().slice([2..3, 0..dim]),
                all.clone().slice([3..4, 0..dim]),
                all.slice([5..6, 0..dim]),
            ],
            0,
        );
        assert!((got - want).abs().max().into_scalar() == 0.0);
    }

    /// [`AttnCache::commit_rows`] on a contiguous set IS [`AttnCache::commit`].
    ///
    /// Compared through a following step rather than by reading the store,
    /// because the thing that must match is not the bytes but what the next
    /// position sees: K, V and the convolution memory together.
    #[test]
    fn commit_rows_on_a_prefix_is_commit() {
        let _lane = CacheLane::wide();
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = dims(AttnKind::Global, 5);
        let w = weights(&d, &dev);
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(16 * d.hidden, 2.5), [16, d.hidden]),
            &dev,
        );
        let row = |i: usize| xs.clone().slice([i..i + 1, 0..d.hidden]);
        let (_, base) = attention_prefill(
            xs.clone().slice([0..9, 0..d.hidden]),
            &w,
            &d,
            None,
            None,
            None,
        );
        for keep in 0..=3usize {
            let batch = Tensor::cat(vec![row(9), row(10), row(11)], 0);
            let mut a = base.clone();
            let _ = attention_steps(batch.clone(), &w, &d, None, 9, None, &mut a);
            a.commit(keep, None);
            let mut b = base.clone();
            let _ = attention_steps(batch, &w, &d, None, 9, None, &mut b);
            b.commit_rows(&(0..keep).collect::<Vec<_>>(), None);
            assert_eq!(a.len(), b.len(), "keep={keep}");
            let next =
                |c: &mut AttnCache<Bk>| attention_step(row(12), &w, &d, None, 9 + keep, None, c);
            let diff = (next(&mut a) - next(&mut b)).abs().max().into_scalar();
            assert!(diff < 1e-5, "keep={keep} diverges by {diff}");
        }
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
        // The WIDE cache, explicitly. What follows compares the cached lane
        // against recomputing, at a tolerance sized for two implementations of
        // the SAME arithmetic -- which they are only while the cache is f32. A
        // narrow cache stores less; holding it to this bar would be asking the
        // wrong question, and `golden/paired/` is where it is asked instead.
        let _lane = CacheLane::wide();
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

    /// The GEMM lane itself: row 0 of an `m == 3` product against the same row
    /// computed alone. Nothing cached, nothing speculative — just
    /// [`linear_bf16`] at two widths on identical operands, at the real
    /// checkpoint's `[4096, 4096]`.
    /// [`linear_fp4`] against [`linear_bf16`] on the SAME weight bytes.
    ///
    /// The question this answers is "is my lane wired right", not "is NVFP4
    /// accurate enough" -- a layout or `scale2` mistake moves the result by
    /// order one, and honest four-bit quantisation moves it by a few percent.
    /// So the bound is deliberately loose: it is a bug detector, and the
    /// quality question is `golden/paired/`'s to answer.
    #[test]
    fn linear_fp4_tracks_linear_bf16_on_the_same_weight() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let (n, k) = (512usize, 256usize);
        let m = 3usize;
        let probe: Tensor<B, 2> = Tensor::from_data(TensorData::new(vec![0f32], [1, 1]), &dev);
        let client = client_of(&probe);

        // One weight, two bindings of the same bytes.
        let wf: Vec<f32> = fill(n * k, 0.17).into_iter().map(|x| x * 0.05).collect();
        let mut bytes = Vec::with_capacity(n * k * 2);
        for x in &wf {
            bytes.extend_from_slice(&half::bf16::from_f32(*x).to_le_bytes());
        }
        let bw = Bf16W {
            h: client.create_from_slice(&bytes),
            n,
            k,
            align: 16,
        };
        let src = client.create_from_slice(&bytes);
        let (codes, scales) =
            crate::models::inkling::fp4quant::quantize_nvfp4_bf16(&client, &src, n, k);
        let pw = PackedW {
            codes,
            scales,
            n,
            k,
            scale2: 1.0,
            swizzled: false,
        };

        let xv: Vec<f32> = fill(m * k, 0.31);
        let x: Tensor<B, 2> = Tensor::from_data(TensorData::new(xv, [m, k]), &dev);

        let a = linear_bf16(x.clone(), &bw).into_data().convert::<f32>();
        let b = linear_fp4(x, &pw).into_data().convert::<f32>();
        let (av, bv) = (a.as_slice::<f32>().unwrap(), b.as_slice::<f32>().unwrap());
        assert_eq!(av.len(), m * n);

        let num: f64 = av
            .iter()
            .zip(bv)
            .map(|(p, q)| ((p - q) as f64).powi(2))
            .sum();
        let den: f64 = av.iter().map(|p| (*p as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        // PRINTED, not just asserted. The bound is loose on purpose (it is a
        // wiring detector), so the assert throws away the one number the test
        // actually measured -- and that number is the only per-lane fidelity
        // figure this tree has. Measured 2026-08-25 on the Spark: 0.0155 for
        // the W4A4 lane, 0.0091 for W4A16, on THIS synthetic 512x256 weight
        // with structured inputs. It is not a claim about the real logits.
        println!("MEASURED rel RMS = {rel:.4}");
        assert!(
            rel < 0.15,
            "relative RMS {rel:.4} between the BF16 and NVFP4 lanes on the same weight \
             is a wiring fault, not quantisation error"
        );
        // And it must not be suspiciously EXACT either -- that would mean the
        // FP4 lane never ran and something handed back the BF16 result.
        assert!(
            rel > 1e-6,
            "the two lanes agree exactly; the FP4 lane did not run"
        );
    }

    /// [`bf16gemm::bf16_gemv_rows`] against the tiled lane on the same bytes.
    ///
    /// The multi-row GEMV accumulates the way a GEMV does — a per-row f32
    /// vector carried across the k segments, summed within the vector and then
    /// across the plane — and the tiled lanes accumulate 16 k at a time in a
    /// tensor-core accumulator. Two orders over the same BF16 products, so they
    /// disagree in the last bits and NOT by more than that. There is no
    /// bit-exactness requirement between BF16 lanes here; what this detects is
    /// wiring — a wrong plane index, a wrong k stride, a row read from the
    /// wrong place — every one of which is an order-one error.
    ///
    /// It sweeps the whole band rather than one width, because the failure this
    /// lane could plausibly have is per-row: m = 1 is not on the band at all,
    /// and a row-indexing bug that is invisible at m = 2 is not invisible at
    /// m = 5.
    #[test]
    fn gemv_rows_tracks_the_tiled_lane() {
        use crate::models::inkling::bf16gemm::{Lane, try_bf16_linear_cubek_launch};
        use cubecl::CubeElement;
        let dev = burn::backend::cuda::CudaDevice::default();
        // k must divide the plane stride (32 units x 8 BF16) and n need not
        // divide anything — 520 is deliberately not a multiple of the eight
        // planes a cube carries, which is the case the column bounds-check is
        // there for.
        let (n, k) = (520usize, 512usize);
        let probe: Tensor<B, 2> = Tensor::from_data(TensorData::new(vec![0f32], [1, 1]), &dev);
        let client = client_of(&probe);

        let wf: Vec<f32> = fill(n * k, 0.17).into_iter().map(|x| x * 0.05).collect();
        let mut wb = Vec::with_capacity(n * k * 2);
        for x in &wf {
            wb.extend_from_slice(&half::bf16::from_f32(*x).to_le_bytes());
        }
        let w = client.create_from_slice(&wb);

        for m in 2..=5usize {
            let xf: Vec<f32> = fill(m * k, 0.31);
            let mut xb = Vec::with_capacity(m * k * 2);
            for x in &xf {
                xb.extend_from_slice(&half::bf16::from_f32(*x).to_le_bytes());
            }
            let a = client.create_from_slice(&xb);

            let hg = try_bf16_linear_cubek_launch(&client, &a, &w, m, k, n, Lane::GemvRows)
                .expect("the gemv-rows lane declined a shape it is meant to take");
            let ht = try_bf16_linear_cubek_launch(&client, &a, &w, m, k, n, Lane::DoubleCyclicMma)
                .expect("the tiled reference declined the shape");

            let g: Vec<f32> =
                f32::from_bytes(&client.read_one(hg).expect("read the gemv lane")).to_vec();
            let t: Vec<f32> =
                f32::from_bytes(&client.read_one(ht).expect("read the tiled lane")).to_vec();
            assert_eq!(g.len(), m * n);
            assert_eq!(t.len(), m * n);

            let num: f64 = g
                .iter()
                .zip(&t)
                .map(|(p, q)| ((p - q) as f64).powi(2))
                .sum();
            let den: f64 = t.iter().map(|p| (*p as f64).powi(2)).sum();
            let rel = (num / den).sqrt();
            println!("gemv rows vs double cyclic mma, m = {m}: relative L2 {rel:.3e}");
            // Two f32 accumulation orders over identical BF16 products. 1e-3 is
            // three orders above what that costs and three orders below what a
            // wiring mistake costs, which is the whole span a detector needs.
            assert!(
                rel < 1e-3,
                "m = {m}: the gemv-rows lane is {rel:.3e} from the tiled lane, \
                 which is a wiring difference and not an accumulation one"
            );
            assert!(
                den > 0.0,
                "m = {m}: the reference produced all zeros, so nothing was compared"
            );
        }
    }

    /// [`linear_w4a16`] against [`linear_bf16`] on the SAME weight bytes.
    ///
    /// The twin of the test above and the same question: is my lane wired
    /// right. A nibble-order, scale-index or fragment-layout mistake moves the
    /// result by order one; honest four-bit WEIGHT quantisation moves it by a
    /// few percent, and less than the W4A4 lane does because the activation
    /// keeps its eight mantissa bits. The bound is the same loose one for the
    /// same reason -- it is a bug detector, not a quality gate.
    #[test]
    fn linear_w4a16_tracks_linear_bf16_on_the_same_weight() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let (n, k) = (512usize, 256usize);
        let m = 3usize;
        let probe: Tensor<B, 2> = Tensor::from_data(TensorData::new(vec![0f32], [1, 1]), &dev);
        let client = client_of(&probe);

        let wf: Vec<f32> = fill(n * k, 0.17).into_iter().map(|x| x * 0.05).collect();
        let mut bytes = Vec::with_capacity(n * k * 2);
        for x in &wf {
            bytes.extend_from_slice(&half::bf16::from_f32(*x).to_le_bytes());
        }
        let bw = Bf16W {
            h: client.create_from_slice(&bytes),
            n,
            k,
            align: 16,
        };
        let src = client.create_from_slice(&bytes);
        let (codes, scales) =
            crate::models::inkling::fp4quant::quantize_nvfp4_bf16(&client, &src, n, k);
        let pw = PackedW {
            codes,
            scales,
            n,
            k,
            scale2: 1.0,
            swizzled: false,
        };

        let xv: Vec<f32> = fill(m * k, 0.31);
        let x: Tensor<B, 2> = Tensor::from_data(TensorData::new(xv, [m, k]), &dev);

        let a = linear_bf16(x.clone(), &bw).into_data().convert::<f32>();
        let b = linear_w4a16(x, &pw).into_data().convert::<f32>();
        let (av, bv) = (a.as_slice::<f32>().unwrap(), b.as_slice::<f32>().unwrap());
        assert_eq!(av.len(), m * n);

        let num: f64 = av
            .iter()
            .zip(bv)
            .map(|(p, q)| ((p - q) as f64).powi(2))
            .sum();
        let den: f64 = av.iter().map(|p| (*p as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        // PRINTED, not just asserted. The bound is loose on purpose (it is a
        // wiring detector), so the assert throws away the one number the test
        // actually measured -- and that number is the only per-lane fidelity
        // figure this tree has. Measured 2026-08-25 on the Spark: 0.0155 for
        // the W4A4 lane, 0.0091 for W4A16, on THIS synthetic 512x256 weight
        // with structured inputs. It is not a claim about the real logits.
        println!("MEASURED rel RMS = {rel:.4}");
        assert!(
            rel < 0.15,
            "relative RMS {rel:.4} between the BF16 and W4A16 lanes on the same weight \
             is a wiring fault, not quantisation error"
        );
        assert!(
            rel > 1e-6,
            "the two lanes agree exactly; the W4A16 lane did not run"
        );
    }

    #[test]
    fn linear_bf16_row0_matches_across_width() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let hidden = 4096usize;
        let probe: Tensor<B, 2> = Tensor::from_data(TensorData::new(vec![0f32], [1, 1]), &dev);
        let client = client_of(&probe);
        let mut bytes = Vec::with_capacity(hidden * hidden * 2);
        for x in fill(hidden * hidden, 0.31) {
            bytes.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes());
        }
        let big = Bf16W {
            h: client.create_from_slice(&bytes),
            n: hidden,
            k: hidden,
            align: 16,
        };
        let rows = 3usize;
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(rows * hidden, 1.25), [rows, hidden]),
            &dev,
        );
        let one = linear_bf16(xs.clone().slice([0..1, 0..hidden]), &big);
        let many = linear_bf16(xs.clone(), &big).slice([0..1, 0..hidden]);
        let scale = one.clone().abs().max().into_scalar().max(1e-6);
        let worst = (many.clone() - one.clone()).abs().max().into_scalar() / scale;

        // Which of the two is RIGHT is not asked here. It used to be, against
        // the same product accumulated in f64 on the host from the same BF16
        // bits, and the answer it printed (1.2e-7 for the gemv lane against
        // 1.36e-5 for the narrow tile) was true and was not the question: an
        // f64 sum is a more expensive computation of a four-bit model, not its
        // ground truth. What decides which lane a run should take is
        // `golden/paired/`. What this test still answers is narrower and is a
        // property of the LANE rather than of the arithmetic -- whether row 0's
        // answer depends on how many other rows were in the batch with it.
        // 1.4e-5 as measured: the two lanes reduce the same 4096-long dot
        // product in different orders and BF16 operands round to 8 mantissa
        // bits. This bound is a REGRESSION guard on that number, not a
        // correctness claim -- what it exists to catch is the day the gap
        // becomes structural rather than arithmetic.
        assert!(
            worst < 1e-3,
            "linear_bf16 row 0 moves with the batch width by {worst}"
        );
    }

    /// Diagnostic: per-position relative drift of the batched lane against the
    /// single-row lane at real width, for batch sizes 1 and 2.
    #[test]
    #[ignore = "a diagnostic, not a gate: run it with --ignored --nocapture to see the per-position table"]
    fn drift_table_at_real_width() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = AttnDims {
            hidden: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            d_rel: 16,
            rel_extent: 1024,
            kernel: 4,
            rms_eps: 1e-6,
            kind: AttnKind::Local,
        };
        let w = weights(&d, &dev);
        let window = Some(512usize);
        let (prefill, tokens) = (5usize, 25usize);
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(tokens * d.hidden, 2.5), [tokens, d.hidden]),
            &dev,
        );
        let (_, base_cache) = attention_prefill(
            xs.clone().slice([0..prefill, 0..d.hidden]),
            &w,
            &d,
            None,
            window,
            window,
        );
        let mut c1 = base_cache.clone();
        let mut ones: Vec<Tensor<B, 2>> = Vec::new();
        for pos in prefill..tokens {
            ones.push(attention_step(
                xs.clone().slice([pos..pos + 1, 0..d.hidden]),
                &w,
                &d,
                None,
                pos,
                window,
                &mut c1,
            ));
        }
        for batch in [1usize, 2] {
            let mut c2 = base_cache.clone();
            let mut pos = prefill;
            let mut i = 0usize;
            let mut line = String::new();
            while pos < tokens {
                let rows = batch.min(tokens - pos);
                let got = attention_steps(
                    xs.clone().slice([pos..pos + rows, 0..d.hidden]),
                    &w,
                    &d,
                    None,
                    pos,
                    window,
                    &mut c2,
                );
                c2.commit(rows, window);
                for r in 0..rows {
                    let want = ones[i + r].clone();
                    let g = got.clone().slice([r..r + 1, 0..d.hidden]);
                    let scale = want.clone().abs().max().into_scalar().max(1e-6);
                    let rel = (g - want).abs().max().into_scalar() / scale;
                    line.push_str(&format!(" {}:{:.2e}", pos + r, rel));
                }
                i += rows;
                pos += rows;
            }
            println!("DRIFT batch={batch}{line}");
            // Where the disagreement lives: in what the batch WROTE (K, V and
            // the pre-convolution projections) or in what it READ them into.
            let rel2 = |a: Tensor<B, 2>, b: Tensor<B, 2>| -> f32 {
                let scale = b.clone().abs().max().into_scalar().max(1e-6);
                (a - b).abs().max().into_scalar() / scale
            };
            println!(
                "  cache k {:.2e}  v {:.2e}  k_pre {:.2e}  v_pre {:.2e}",
                rel2(
                    c2.k.materialize(&c2.k_pre.device()),
                    c1.k.materialize(&c1.k_pre.device())
                ),
                rel2(
                    c2.v.materialize(&c2.v_pre.device()),
                    c1.v.materialize(&c1.v_pre.device())
                ),
                rel2(c2.k_pre.clone(), c1.k_pre.clone()),
                rel2(c2.v_pre.clone(), c1.v_pre.clone()),
            );
        }
    }

    /// That the NVFP4 KV path RUNS, end to end, at the real checkpoint's
    /// attention width — and what it costs on a synthetic input.
    ///
    /// ## Why the engagement assertion is the load-bearing one
    ///
    /// Every other cached test in this module builds an 8-wide KV row
    /// (`kv_heads: 2, head_dim: 4`), and `KvStore::new` sends anything that is
    /// not a multiple of 64 to the DENSE arm whatever the switch says. So
    /// running the whole module under `INK_FP4_KV=1` moved not one of them:
    /// same failures, same drift to the last digit, every test "passing on the
    /// FP4 lane" without an FP4 store ever being constructed. This is the only
    /// place the four-bit path runs at all, so it says which arm it got out
    /// loud, before it looks at a single number.
    ///
    /// ## Why the drift is PRINTED and not asserted
    ///
    /// The codec's own contract is gated where it belongs, in
    /// `kvpages::a_real_width_bf16_row_round_trips_within_the_nvfp4_block_bound`,
    /// and it holds at exactly the theoretical worst case: `amax / 6`, half the
    /// gap between the two widest E2M1 magnitudes. There is nothing left for a
    /// bound here to catch that is a property of this code.
    ///
    /// What a bound here WOULD encode is a claim about how far four-bit keys
    /// and values move an attention output, and this test cannot make that
    /// claim honestly. The input is two summed sinusoids and the weights are
    /// more of the same; the softmax over sixteen positions of a synthetic
    /// sequence is not the softmax over a real one, and a perturbed logit
    /// reshuffles it by an amount that is a property of THAT distribution.
    /// `golden/paired/` is where the question is asked, exactly as
    /// [`attn_bf16`] says for the same reason. So this prints, with its
    /// framing, and asserts only what it can see: that the lane produced
    /// finite, non-degenerate output.
    ///
    /// ## What it prints, and the control that makes it readable
    ///
    /// One local layer, window 512, 5-token prefill, 16 decode steps, synthetic
    /// sinusoidal input and weights, `[1, 4096]` output per step; worst over
    /// the sixteen steps, against the BF16 dense cache:
    ///
    /// ```text
    /// NVFP4 cache : max-abs 4.9e-1 of the dense max-abs, RMS 9.1e-1 of the dense RMS
    /// f32   cache : max-abs 6.2e-3                     , RMS 1.0e-2
    /// ```
    ///
    /// The second row is why the first is worth printing. This probe cancels
    /// heavily in `wo`, so it amplifies ANY cache perturbation, and the FP4
    /// figure alone cannot tell an amplifying probe from a ruinous codec. The
    /// control is the trade [`attn_bf16`] already ships, measured the identical
    /// way on the identical input — 1% where NVFP4 is 91%. So the probe
    /// discriminates, and the ~88x is a real difference in kind and not an
    /// artifact of the harness.
    ///
    /// Neither figure is a statement about the model. Both are one synthetic
    /// layer, and the max-abs one is a single worst ELEMENT against the largest
    /// element rather than a typical one — which is exactly why the RMS is
    /// printed beside it and neither travels alone.
    #[test]
    fn the_fp4_cache_engages_at_real_width() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = AttnDims {
            hidden: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            d_rel: 16,
            rel_extent: 1024,
            kernel: 4,
            rms_eps: 1e-6,
            kind: AttnKind::Local,
        };
        let w = weights(&d, &dev);
        let window = Some(512usize);
        let (prefill, tokens) = (5usize, 21usize);
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(tokens * d.hidden, 2.5), [tokens, d.hidden]),
            &dev,
        );

        let run = |fp4: bool, wide: bool| -> (bool, Vec<Tensor<B, 2>>, AttnCache<Bk>) {
            let _lane = if fp4 {
                super::super::kvpages::Fp4Lane::on()
            } else {
                super::super::kvpages::Fp4Lane::off()
            };
            let _dt = if wide {
                CacheLane::wide()
            } else {
                CacheLane::narrow()
            };
            let (_, mut cache) = attention_prefill(
                xs.clone().slice([0..prefill, 0..d.hidden]),
                &w,
                &d,
                None,
                window,
                window,
            );
            let on = cache.kv_is_fp4();
            let mut outs = Vec::new();
            for pos in prefill..tokens {
                outs.push(attention_step(
                    xs.clone().slice([pos..pos + 1, 0..d.hidden]),
                    &w,
                    &d,
                    None,
                    pos,
                    window,
                    &mut cache,
                ));
            }
            (on, outs, cache)
        };

        let (dense_on, dense, dc) = run(false, false);
        let (fp4_on, narrow, nc) = run(true, false);
        // The control. Without it the FP4 number below has no yardstick: this
        // probe cancels heavily in `wo`, so it amplifies ANY cache
        // perturbation, and a reader shown only the FP4 figure cannot tell an
        // amplifying probe from a ruinous codec. This is the SHIPPED trade --
        // f32 cache against the BF16 one `attn_bf16` turns on by default --
        // measured the identical way, on the identical input.
        let (_, f32_kv, _) = run(false, true);
        assert!(!dense_on, "the dense arm built an NVFP4 store");
        assert!(
            fp4_on,
            "the FP4 arm was forced on at a 1024-wide KV row and the cache is \
             still dense - this test measures nothing"
        );

        // FIRST, the only comparison in this test whose bound means anything:
        // the KV the two lanes actually END UP HOLDING, after a prefill,
        // sixteen single-row appends into a growing page, and a materialize.
        // If a page were reordered, or a scale read against the wrong block,
        // this is where it shows -- and it is a statement about THIS code,
        // unlike anything measured past the softmax.
        for (name, a, b) in [
            ("k", dc.k.materialize(&dev), nc.k.materialize(&dev)),
            ("v", dc.v.materialize(&dev), nc.v.materialize(&dev)),
        ] {
            let a: Vec<f32> = a
                .cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec()
                .unwrap();
            let b: Vec<f32> = b
                .cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec()
                .unwrap();
            assert_eq!(
                a.len(),
                b.len(),
                "{name}: the two lanes hold different sizes"
            );
            let mut w = 0f32;
            for blk in 0..a.len() / 16 {
                let lo = blk * 16;
                let amax = a[lo..lo + 16].iter().fold(0.0f32, |m, x| m.max(x.abs()));
                for i in lo..lo + 16 {
                    w = w.max((a[i] - b[i]).abs() / amax.max(1e-6));
                }
            }
            println!("cached {name} through the real path: worst {w:e} of block amax");
            assert!(
                w < 1.0 / 3.0 + 1.0 / 16.0,
                "cached {name} lost {w} of its block amax - that is not the codec"
            );
        }

        let spread = |base: &[Tensor<B, 2>], other: &[Tensor<B, 2>]| -> (f32, f32) {
            let (mut worst, mut worst_rms) = (0f32, 0f32);
            for (a, b) in base.iter().zip(other.iter()) {
                let scale = a.clone().abs().max().into_scalar().max(1e-6);
                let diff = b.clone() - a.clone();
                worst = worst.max(diff.clone().abs().max().into_scalar() / scale);
                let rms = diff.powf_scalar(2.0).mean().sqrt().into_scalar();
                let arms = a.clone().powf_scalar(2.0).mean().sqrt().into_scalar();
                worst_rms = worst_rms.max(rms / arms.max(1e-6));
            }
            (worst, worst_rms)
        };
        let (worst, worst_rms) = spread(&dense, &narrow);
        let (cw, crms) = spread(&dense, &f32_kv);
        println!(
            "one local layer, window 512, {prefill}-token prefill, {} decode steps, \
             synthetic sinusoidal input, against the BF16 dense cache:\n  \
             NVFP4 cache : worst max-abs {worst:e}, worst RMS {worst_rms:e}\n  \
             f32 cache   : worst max-abs {cw:e}, worst RMS {crms:e}   (the SHIPPED \
             bf16-vs-f32 trade, same probe)",
            tokens - prefill
        );

        // What this test can actually see: the lane ran and produced usable
        // numbers. A dequant that read the wrong scale, or a page the store
        // reordered, does not come out slightly off — it comes out as NaN, as
        // zeros, or as something orders of magnitude adrift.
        for (i, b) in narrow.iter().enumerate() {
            let m = b.clone().abs().max().into_scalar();
            assert!(m.is_finite() && m > 0.0, "fp4 step {i} produced {m}");
            let a = dense[i].clone().abs().max().into_scalar();
            assert!(
                m > a / 4.0 && m < a * 4.0,
                "fp4 step {i} is {m} against a dense {a} — not a quantization gap"
            );
        }
    }

    /// The BATCHED cached lane against the SINGLE-ROW cached lane, at the real
    /// checkpoint's attention width.
    ///
    /// Not the same question as [`compare_batched`], which holds both against
    /// the uncached lane at hidden 16. This one asks what a speculative loop
    /// actually depends on: that stepping positions in batches of two writes
    /// the same K and V as stepping them one at a time. At hidden 16 with f32
    /// weights the two agree to 2e-5; the real layer multiplies BF16 through
    /// `mma.sync`, and `m == 1` and `m > 1` take DIFFERENT GEMM lanes there
    /// (`GemvPlaneParallel` against the narrow tile), so this is where a
    /// difference between them would be visible — and it compounds, because
    /// every row it writes is read by every position after it.
    #[test]
    fn batched_matches_single_step_at_real_width() {
        let dev = burn::backend::cuda::CudaDevice::default();
        let d = AttnDims {
            hidden: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            d_rel: 16,
            rel_extent: 1024,
            kernel: 4,
            rms_eps: 1e-6,
            kind: AttnKind::Local,
        };
        let w = weights(&d, &dev);
        let window = Some(512usize);
        let (prefill, tokens) = (5usize, 45usize);
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(tokens * d.hidden, 2.5), [tokens, d.hidden]),
            &dev,
        );
        let (_, base_cache) = attention_prefill(
            xs.clone().slice([0..prefill, 0..d.hidden]),
            &w,
            &d,
            None,
            window,
            window,
        );

        // One at a time.
        let mut c1 = base_cache.clone();
        let mut ones: Vec<Tensor<B, 2>> = Vec::new();
        for pos in prefill..tokens {
            ones.push(attention_step(
                xs.clone().slice([pos..pos + 1, 0..d.hidden]),
                &w,
                &d,
                None,
                pos,
                window,
                &mut c1,
            ));
        }

        // Two at a time, committed whole -- no rejection, so the only thing
        // under test is the WIDTH.
        let mut c2 = base_cache.clone();
        let mut worst = 0f32;
        let mut i = 0usize;
        let mut pos = prefill;
        while pos < tokens {
            let rows = 2.min(tokens - pos);
            let got = attention_steps(
                xs.clone().slice([pos..pos + rows, 0..d.hidden]),
                &w,
                &d,
                None,
                pos,
                window,
                &mut c2,
            );
            c2.commit(rows, window);
            for r in 0..rows {
                let want = ones[i + r].clone();
                let g = got.clone().slice([r..r + 1, 0..d.hidden]);
                let scale = want.clone().abs().max().into_scalar().max(1e-6);
                worst = worst.max((g - want).abs().max().into_scalar() / scale);
            }
            i += rows;
            pos += rows;
        }
        // Not bit-equality: the two lanes reduce in different orders, so some
        // drift is the hardware and not a bug. What would NOT be the hardware
        // is drift that grows with position, which is what a wrong K or V row
        // does once every later query reads it.
        // NOT bit-equality, and the gap is entirely the GEMM lane: pin one
        // with `INK_GEMM=double cyclic mma` and this is 0.00e0 at every
        // position. Left free-running here because that is how the runtime
        // ships -- `m == 1` takes `gemv plane par` and `m > 1` cannot, so a
        // speculative loop compares two lanes whether or not it wants to.
        assert!(
            worst < 5e-2,
            "batched attention drifts from the single step by {worst} relative"
        );
    }

    /// [`short_conv_steps`] against the whole-sequence [`short_conv`], in
    /// batches, rolling back `reject` rows of every batch and re-running those
    /// positions — which is exactly what a rejected draft does to the
    /// convolution's memory.
    ///
    /// A separate test from the attention ones because the convolution is the
    /// half of a speculative rollback that is NOT a truncation: the taps the
    /// next position reads are a function of the last KEPT row and the rows
    /// before it, and those pre-convolution inputs are gone once the batch is
    /// over unless the batch keeps its whole window.
    fn compare_conv_batched(tokens: usize, prefill: usize, batch: usize, reject: usize) -> f32 {
        // The WIDE cache, explicitly. What follows compares the cached lane
        // against recomputing, at a tolerance sized for two implementations of
        // the SAME arithmetic -- which they are only while the cache is f32. A
        // narrow cache stores less; holding it to this bar would be asking the
        // wrong question, and `golden/paired/` is where it is asked instead.
        let _lane = CacheLane::wide();
        let dev = burn::backend::cuda::CudaDevice::default();
        let (dim, kernel) = (16usize, 4usize);
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(tokens * dim, 1.5), [tokens, dim]),
            &dev,
        );
        let w: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(dim * kernel, 0.25), [dim, kernel]),
            &dev,
        );
        let full = short_conv(xs.clone(), w.clone());
        let mut hist = conv_history(xs.clone().slice([0..prefill, 0..dim]), kernel);

        let mut worst = 0f32;
        let mut pos = prefill;
        while pos < tokens {
            let rows = batch.min(tokens - pos);
            if reject > 0 && rows > reject {
                let keep = rows - reject;
                let (_, all) = short_conv_steps(
                    hist.clone(),
                    xs.clone().slice([pos..pos + rows, 0..dim]),
                    w.clone(),
                );
                // The rollback: the history ending at the last KEPT row.
                hist = conv_history(all.slice([0..kernel - 1 + keep, 0..dim]), kernel);
                let (got, all) = short_conv_steps(
                    hist.clone(),
                    xs.clone().slice([pos + keep..pos + rows, 0..dim]),
                    w.clone(),
                );
                hist = conv_history(all.slice([0..kernel - 1 + reject, 0..dim]), kernel);
                let want = full.clone().slice([pos + keep..pos + rows, 0..dim]);
                worst = worst.max((got - want).abs().max().into_scalar());
            } else {
                let (got, all) = short_conv_steps(
                    hist.clone(),
                    xs.clone().slice([pos..pos + rows, 0..dim]),
                    w.clone(),
                );
                hist = conv_history(all.slice([0..kernel - 1 + rows, 0..dim]), kernel);
                let want = full.clone().slice([pos..pos + rows, 0..dim]);
                worst = worst.max((got - want).abs().max().into_scalar());
            }
            pos += rows;
        }
        worst
    }

    /// Three positions at a time, from a prefill SHORTER than the kernel — so
    /// the batch's first rows read the zero padding the whole-sequence lane
    /// assumes, and a window built from the wrong end would show up here.
    #[test]
    fn batched_short_conv_matches_full() {
        let worst = compare_conv_batched(11, 2, 3, 0);
        assert!(worst < 1e-6, "batched short convolution drifts by {worst}");
    }

    /// The same, rejecting two rows of every three-row batch and re-running
    /// them. The re-run's answer is the one compared, so a history restored
    /// from the wrong offset is a failure and not a rounding difference.
    #[test]
    fn batched_short_conv_survives_rejection() {
        let worst = compare_conv_batched(13, 4, 3, 2);
        assert!(worst < 1e-6, "short convolution rollback drifts by {worst}");
    }

    /// Three positions at a time against the uncached lane, on a global layer
    /// with log scaling that varies per row — so a batch that used one `tau`
    /// for all of its rows would show up here.
    #[test]
    fn batched_global_matches_full() {
        let ls = Some(LogScaling {
            n_floor: 4.0,
            alpha: 0.5,
        });
        let worst = compare_batched(AttnKind::Global, 5, None, ls, 11, 4, 3, 0);
        assert!(
            worst < CACHE_TOLERANCE,
            "batched global attention drifts by {worst}"
        );
    }

    /// The same on a local layer whose window is shorter than the sequence, so
    /// the batch must not forget a key until it knows how long it was.
    #[test]
    fn batched_local_matches_full_across_the_window() {
        let worst = compare_batched(AttnKind::Local, 5, Some(5), None, 11, 4, 3, 0);
        assert!(
            worst < CACHE_TOLERANCE,
            "batched windowed attention drifts by {worst}"
        );
    }

    /// Rejection: run three, keep one, re-run the two that were rejected. This
    /// is the speculative path, and it passes only if `commit` restored the
    /// short convolution's memory as well as truncating K and V.
    #[test]
    fn rejected_rows_leave_no_trace() {
        let ls = Some(LogScaling {
            n_floor: 4.0,
            alpha: 0.5,
        });
        let worst = compare_batched(AttnKind::Global, 5, None, ls, 11, 4, 3, 2);
        assert!(
            worst < CACHE_TOLERANCE,
            "rolled-back batch drifts by {worst}"
        );
    }

    /// The same against a window, where rollback and the window's own
    /// forgetting interact.
    #[test]
    fn rejected_rows_leave_no_trace_windowed() {
        let worst = compare_batched(AttnKind::Local, 5, Some(5), None, 11, 4, 3, 2);
        assert!(
            worst < CACHE_TOLERANCE,
            "rolled-back windowed batch drifts by {worst}"
        );
    }

    /// A batch of one must agree with [`attention_step`], which is the claim
    /// that makes the two functions one algorithm rather than two.
    #[test]
    fn a_batch_of_one_matches_the_single_step() {
        let ls = Some(LogScaling {
            n_floor: 4.0,
            alpha: 0.5,
        });
        let worst = compare_batched(AttnKind::Global, 5, None, ls, 11, 4, 1, 0);
        assert!(worst < CACHE_TOLERANCE, "a one-row batch drifts by {worst}");
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
            worst > 20.0 * CACHE_TOLERANCE,
            "sabotaged conv history only moved the answer by {worst}, which is inside the \
             tolerance the windowed tests use"
        );
    }
}

/// `x @ w^T` for ONE row, by approximate maximum-inner-product search.
///
/// The exact twin is [`linear_w4a16`] and the contract is deliberately the
/// same: a `[1, w.n]` f32 tensor, no `-inf` anywhere, ready for the host argmax
/// and the top-5 report that already read it. What differs is that only the
/// shortlist carries an exact score; see [`crate::models::inkling::annhead`].
///
/// `m > 1` is a caller error rather than a fallback here, because the caller is
/// the one that knows whether the exact lane is available and this function
/// silently doing something else at verify width is exactly the class of
/// mistake this repo keeps finding.
pub fn linear_ann(
    x: Tensor<Bk, 2>,
    w: &PackedW,
    sketch: &crate::models::inkling::annhead::Sketch,
    budget: usize,
    range: f32,
) -> (Tensor<Bk, 2>, crate::models::inkling::annhead::AnnStat) {
    use crate::models::inkling::annhead::ann_logits;
    let [m, k] = x.dims();
    assert_eq!(m, 1, "linear_ann scans one query row, not {m}");
    // `ann_logits` reads codes/scales as row-major [n, k/8]. A fragment-order
    // weight read that way produces PLAUSIBLE LOGITS, not an error -- which is
    // exactly the failure the m>1 assert above exists to prevent, in the other
    // dimension. Both siblings already guard it: `linear_fp4` asserts the same
    // thing and `linear_w4a16` branches on it. This one did not, and the gap
    // between two independently-correct branches broke greedy decode on main:
    // the head emitted 164247, 60701, 9927 ... where the exact lane emits
    // 15009, 24075, 314 ..., and k=1 collapsed into one token repeated 46 times.
    assert!(
        !w.swizzled,
        "linear_ann reads row-major [n, k/8] but this weight is in MMA-fragment \
         order; permuting the head is incompatible with the approximate lane"
    );
    assert_eq!(
        k, w.k,
        "linear_ann: x is [_, {k}] but the weight is [_, {}]",
        w.k
    );
    assert_eq!(
        (sketch.n, sketch.k),
        (w.n, w.k),
        "the sketch is [{}, {}] and the weight is [{}, {}]",
        sketch.n,
        sketch.k,
        w.n,
        w.k
    );
    let client = client_of(&x);
    let dev = x.device();
    let (xh, xdt) = crate::models::inkling::seam::handle_of_any(x);
    let (out, stat) = match xdt {
        burn::tensor::DType::BF16 => ann_logits::<_, half::bf16>(
            &client, sketch, &w.codes, &w.scales, &xh, w.scale2, budget, range,
        ),
        burn::tensor::DType::F32 => ann_logits::<_, f32>(
            &client, sketch, &w.codes, &w.scales, &xh, w.scale2, budget, range,
        ),
        other => panic!("linear_ann: no lane for a {other:?} activation"),
    };
    (tensor_of(client, dev, out, 1, w.n), stat)
}
