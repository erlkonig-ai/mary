//! The residual stream, and the two kernels that let it be BF16.
//!
//! # What the residual stream costs
//!
//! At f32, `[n, hidden]` is 16 KiB a token on this model, and a layer holds
//! three of them at once: the stream itself, the normed copy every stage reads,
//! and the sum the residual add produces. Forty-two layers carry the same
//! buffers, so the old lane kept 48 KiB a token live from the first layer to
//! the last. `INK_QBLOCK` and `INK_ACT_BF16` do not reach this storage;
//! [`resid_bf16`] does, halving each buffer to 8 KiB a token.
//!
//! The reference carries all three in BF16. [`resid_bf16`] does too, and these
//! are the two kernels that make it possible without widening back at every
//! seam.
//!
//! # The ordering, which is the whole reason this is a kernel
//!
//! Inkling's own fused kernel rounds the residual SUM to BF16 and then norms
//! **the rounded value**. That is deliberately unlike vLLM's generic fused
//! add-norm, which keeps the sum in an f32 register and norms the unrounded
//! one; the two disagree in the last bits of every layer and the model was
//! trained against the first.
//!
//! So [`add_resid`] writes BF16 — one rounding of `x + a`, in f32, stored
//! narrow — and [`rms_norm`] reads that BF16 back. The rounding happens
//! between the two kernels because that is where the reference puts it, and
//! keeping the two as separate launches is what makes the ordering legible
//! rather than a comment about a fused kernel's internals.
//!
//! # What stays f32, and why each one does
//!
//! * **The variance.** `sum(x^2)` over `hidden = 4096` accumulated in BF16 has
//!   eight bits of mantissa against four thousand terms; the reference computes
//!   it in f32 and so does [`rms_norm_kernel`], which reads BF16 and widens
//!   every element before it squares it. The reduction itself is a shared-memory
//!   tree in f32.
//! * **The gain.** `[hidden]` f32, read as f32, applied after the divide, in
//!   f32, exactly as [`super::burn::rms_norm`] does.
//! * **The divide.** `x / sqrt(var + eps)`, not `x * recip(...)`: an approximate
//!   SIMD reciprocal cost K3 about fourteen bits once, and the Burn lane avoids
//!   it for that reason. Same hazard, same avoidance, so that the wide and
//!   narrow lanes differ in STORAGE and not in which operations they perform.
//! * **The stage output.** `a` in `x + a` arrives f32 and is read f32. The
//!   reference's `a` is BF16 because its GEMMs emit BF16; ours is an f32
//!   accumulator, and rounding it on the way in would be a rounding the
//!   reference does not take either. One rounding, on the sum.
//!
//! # The reference, read rather than recalled
//!
//! `transformers`' `modeling_inkling.py` puts both of the above in one place,
//! and it is not the RMSNorm:
//!
//! ```text
//! class InklingShortConvolution.forward:
//!     # Keep the computation in fp32
//!     input_dtype = hidden_states.dtype
//!     hidden_states = hidden_states.float()
//!     residual = hidden_states
//!     ...
//!     hidden_states = (hidden_states + residual).to(dtype=input_dtype)
//!
//! class InklingRMSNorm.forward:
//!     input_dtype = hidden_states.dtype
//!     hidden_states = hidden_states.to(torch.float32)
//!     ...
//!     return self.weight * hidden_states.to(input_dtype)
//! ```
//!
//! So the rounding of the residual sum happens in the CONVOLUTION, on its way
//! out, and the norm is simply handed a value that is already narrow. "Round
//! the sum, then norm the rounded value" is not a property of the norm at all --
//! it is what you get when the thing before it returns `input_dtype`. That is
//! why [`add_resid`] and [`rms_norm`] are two launches here: the same ordering,
//! expressed the same way round.
//!
//! It also confirms the two f32 islands above from the source rather than from
//! memory: the convolution's whole computation and its residual are f32 with a
//! single rounding at the end, and the norm's variance is f32 with the cast back
//! before the gain.
//!
//! ## One place this rounds LESS than the reference, on purpose
//!
//! `self.weight * hidden_states.to(input_dtype)` casts the normed value first
//! and multiplies by the gain in the narrow dtype -- two roundings. This kernel
//! divides and applies the gain in f32 and rounds once, on the store, because
//! the gain here is an f32 tensor widened from the checkpoint and because that
//! is what [`super::burn::rms_norm`] has always done on the wide lane. Making
//! the narrow lane round twice to match would be matching a reference's
//! arithmetic against our own wide arm's, and the wide arm is the control.
//!
//! # Why the wide arm does not use these
//!
//! [`rms_norm`] dispatches to [`super::burn::rms_norm`] when the stream is f32,
//! rather than launching this kernel at `I = O = f32`. A tree reduction and
//! Burn's `mean_dim` sum the same 4096 terms in different orders, so a kernel
//! that served both lanes would make every wide-arm number differ from the
//! binary before this change for a reason that has nothing to do with the
//! change. The wide arm is left exactly as it was so that it can go on being
//! the control.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::seam::{client_of, dtype_of, handle_of_any, tensor_of_dt, Bk};
use burn::tensor::{DType, Tensor};

/// Threads per cube in [`rms_norm_kernel`]. One cube per token; each unit walks
/// its own strided share of the row, so the reads are coalesced.
const NORM_UNITS: u32 = 256;

/// Threads per cube in [`add_resid_kernel`]. One thread per element.
const ADD_UNITS: u32 = 256;

/// Whether the residual stream, its normed copy and the residual sum are held
/// in BF16.
///
/// **Defaults to [`super::burn::act_bf16`]** — the narrow lane is one lane —
/// and `INK_RESID_BF16=0` keeps the residual wide while the activations narrow.
/// That override lets the two halves be priced apart: `INK_ACT_BF16=1
/// INK_RESID_BF16=0` is exactly the lane that shipped before this, and the
/// difference between it and the default is this file. The opposite mix is not
/// a lane: a narrow residual feeding deliberately wide activations is rejected
/// at startup by `inkling_forward`.
///
/// ## What the reference does here
///
/// BF16, for all three buffers. The upstream implementation's residual stream
/// is BF16 from the embedding to the final norm, and every f32 in a layer is an
/// accumulator: the RMSNorm variance, the short convolution's four taps, the
/// softmax running max and sum, the GEMM accumulators, the router logits and
/// the top-k weights, `tau`. This module keeps every one of those f32 and
/// narrows only what is STORED between two kernels.
pub fn resid_bf16() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_RESID_BF16")
            .map(|v| v != "0")
            .unwrap_or_else(|_| super::burn::act_bf16())
    })
}

/// The dtype the residual stream is held in.
pub fn resid_dtype() -> DType {
    if resid_bf16() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// `t` in the dtype the residual stream is held in. The identity on the wide
/// lane, and the identity again when `t` is already narrow.
pub fn as_resid<const D: usize>(t: Tensor<Bk, D>) -> Tensor<Bk, D> {
    if resid_bf16() {
        t.cast(burn::tensor::FloatDType::BF16)
    } else {
        t
    }
}

/// Back to f32, for a host readback, the wire, or a reader that has no narrow
/// path. The identity on the wide lane.
pub fn from_resid<const D: usize>(t: Tensor<Bk, D>) -> Tensor<Bk, D> {
    if resid_bf16() {
        t.cast(burn::tensor::FloatDType::F32)
    } else {
        t
    }
}

// ---------------------------------------------------------------------------
// The residual add
// ---------------------------------------------------------------------------

/// `out = x + a`, one rounding, stored at `O`.
///
/// `x` is the residual stream and `a` is the stage output. Both are widened to
/// f32, added there, and the sum is rounded ONCE on the store — which is the
/// rounding Inkling's fused kernel performs and the value its norm then reads.
#[cube(launch_unchecked)]
fn add_resid_kernel<X: Scalar + Cast, A: Scalar + Cast, O: Scalar + Cast>(
    x: &Array<X>,
    a: &Array<A>,
    out: &mut Array<O>,
    total: usize,
) {
    let p = ABSOLUTE_POS as usize;
    if p < total {
        out[p] = O::cast_from(f32::cast_from(x[p]) + f32::cast_from(a[p]));
    }
}

/// Launch [`add_resid_kernel`] over `total` elements.
fn add_resid_as<X: Scalar + Cast, A: Scalar + Cast, O: Scalar + Cast, R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    a: &Handle,
    total: usize,
) -> Handle {
    let out = client.empty(total * core::mem::size_of::<O>());
    let cubes = total.div_ceil(ADD_UNITS as usize) as u32;
    unsafe {
        add_resid_kernel::launch_unchecked::<X, A, O, R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(ADD_UNITS),
            ArrayArg::from_raw_parts(x.clone(), total),
            ArrayArg::from_raw_parts(a.clone(), total),
            ArrayArg::from_raw_parts(out.clone(), total),
            total,
        );
    }
    out
}

/// `xd + a`, where `xd` is the residual stream and `a` is a stage output.
///
/// On the wide lane this is Burn's `+` and nothing else. On the narrow lane it
/// is one kernel that reads the BF16 stream and the f32 stage output, adds in
/// f32, and stores BF16 — replacing an f32 temporary that was 16 KiB a token
/// and live across the add.
///
/// `a` may be either dtype: the attention lane hands back f32, and a caller
/// that has already narrowed its output hands back BF16. Both are read through
/// the same widening cast.
pub fn add_resid(xd: Tensor<Bk, 2>, a: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
    let xdt = dtype_of(&xd);
    let adt = dtype_of(&a);
    if xdt == DType::F32 && adt == DType::F32 {
        return xd + a;
    }
    let [rows, cols] = xd.dims();
    assert_eq!(
        a.dims(),
        [rows, cols],
        "add_resid: {:?} against [{rows}, {cols}]",
        a.dims()
    );
    let client = client_of(&xd);
    let dev = xd.device();
    let out_dt = xdt;
    let (xh, _) = handle_of_any(xd);
    let (ah, _) = handle_of_any(a);
    let total = rows * cols;
    let out = match (xdt, adt, out_dt) {
        (DType::BF16, DType::F32, DType::BF16) => {
            add_resid_as::<half::bf16, f32, half::bf16, _>(&client, &xh, &ah, total)
        }
        (DType::BF16, DType::BF16, DType::BF16) => {
            add_resid_as::<half::bf16, half::bf16, half::bf16, _>(&client, &xh, &ah, total)
        }
        (DType::F32, DType::BF16, DType::F32) => {
            add_resid_as::<f32, half::bf16, f32, _>(&client, &xh, &ah, total)
        }
        _ => panic!("add_resid: no lane for {xdt:?} + {adt:?} -> {out_dt:?}"),
    };
    tensor_of_dt(client, dev, out, rows, cols, out_dt)
}

// ---------------------------------------------------------------------------
// RMS normalization
// ---------------------------------------------------------------------------

/// One cube per token. `x` is `[tokens, h]`, `gain` is `[h]`, `out` is
/// `[tokens, h]`.
///
/// Every element is widened to f32 on the read, so the sum of squares, the
/// division and the gain are f32 whatever `I` and `O` are. The reduction is a
/// shared-memory tree over [`NORM_UNITS`] partials — a fixed order, so the
/// result does not depend on how the units are scheduled.
#[cube(launch_unchecked)]
fn rms_norm_kernel<I: Scalar + Cast, O: Scalar + Cast>(
    x: &Array<I>,
    gain: &Array<f32>,
    out: &mut Array<O>,
    eps: f32,
    width: f32,
    h: u32,
    units: u32,
) {
    let t = CUBE_POS_X;
    let u = UNIT_POS_X;
    let base = (t * h) as usize;

    // Sized by the module constant rather than by `units`, because the tree
    // below assigns to its stride and a comptime value cannot be assigned to
    // inside a runtime loop. `units` is only ever `NORM_UNITS` or less.
    let mut red = SharedMemory::<f32>::new(comptime!(NORM_UNITS as usize));

    // The unit's strided share of the row. Consecutive units read consecutive
    // elements, so each pass over the row is a coalesced line per warp.
    let mut acc = f32::new(0.0);
    let mut i = u;
    while i < h {
        let v = f32::cast_from(x[base + i as usize]);
        acc += v * v;
        i += units;
    }
    red[u as usize] = acc;
    sync_cube();

    let mut step = units / 2;
    while step > 0 {
        if u < step {
            red[u as usize] += red[(u + step) as usize];
        }
        sync_cube();
        step /= 2u32;
    }

    // `mean` then `sqrt(mean + eps)` then a DIVIDE, in that order, because that
    // is the order the Burn lane performs them in and the wide arm has to stay
    // comparable to this one.
    let denom = Sqrt::sqrt(red[0] / width + eps);

    let mut j = u;
    while j < h {
        let v = f32::cast_from(x[base + j as usize]);
        out[base + j as usize] = O::cast_from(v / denom * gain[j as usize]);
        j += units;
    }
}

/// Launch [`rms_norm_kernel`] over a whole `[tokens, h]` stream.
fn rms_norm_as<I: Scalar + Cast, O: Scalar + Cast, R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    gain: &Handle,
    tokens: usize,
    h: usize,
    eps: f32,
) -> Handle {
    let total = tokens * h;
    let out = client.empty(total * core::mem::size_of::<O>());
    let units = NORM_UNITS.min(h.next_power_of_two() as u32);
    assert!(
        units.is_power_of_two(),
        "the reduction tree halves {units} to one"
    );
    unsafe {
        rms_norm_kernel::launch_unchecked::<I, O, R>(
            client,
            CubeCount::new_1d(tokens as u32),
            CubeDim::new_1d(units),
            ArrayArg::from_raw_parts(x.clone(), total),
            ArrayArg::from_raw_parts(gain.clone(), h),
            ArrayArg::from_raw_parts(out.clone(), total),
            eps,
            h as f32,
            h as u32,
            units,
        );
    }
    out
}

/// RMS normalization of the residual stream, at whatever dtype it is held in.
///
/// f32 in, f32 out goes to [`super::burn::rms_norm`] unchanged — see the module
/// doc for why the wide arm does not share this kernel. BF16 in gives BF16 out:
/// every consumer of the normed stream either takes a BF16 activation
/// ([`super::bf16gemm`], through [`super::burn::linear_bf16`]) or gathers out of
/// it into BF16 (the routed lane), so widening here would be a buffer allocated
/// to be narrowed again two kernels later.
pub fn rms_norm(x: Tensor<Bk, 2>, gain: Tensor<Bk, 1>, eps: f64) -> Tensor<Bk, 2> {
    let dt = dtype_of(&x);
    if dt == DType::F32 {
        return super::burn::rms_norm(x, gain, eps);
    }
    let [tokens, h] = x.dims();
    assert_eq!(
        gain.dims()[0],
        h,
        "rms_norm: gain is {} wide, input {h}",
        gain.dims()[0]
    );
    let client = client_of(&x);
    let dev = x.device();
    let (xh, _) = handle_of_any(x);
    let (gh, gdt) = handle_of_any(gain);
    assert_eq!(gdt, DType::F32, "rms_norm: the gain is f32");
    let out = match dt {
        DType::BF16 => {
            rms_norm_as::<half::bf16, half::bf16, _>(&client, &xh, &gh, tokens, h, eps as f32)
        }
        _ => panic!("rms_norm: no lane for a {dt:?} residual stream"),
    };
    tensor_of_dt(client, dev, out, tokens, h, dt)
}
