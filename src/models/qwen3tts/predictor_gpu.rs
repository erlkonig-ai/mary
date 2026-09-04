//! The code predictor on the GPU — hand-fused cubecl kernels for the
//! sub-talker's 16-position frame, replacing the Accelerate CPU path in
//! [`super::predictor`].
//!
//! ## Why
//!
//! The predictor is the voice lane's last host-side stage. Per 80 ms frame it
//! runs 16 strictly sequential positions through a 5×1024 Qwen3 decoder —
//! 1.26 G weight reads, ~5 GB at f32 — and on the CPU that measured ~50
//! ms/frame against a talker that only needs ~28. Synthesis ran below
//! realtime and rebuffered mid-word.
//!
//! The shape is the same one [`super::megakernel`] already solved for the
//! talker, so the kernels here are its siblings rather than a new idea. The
//! two differences that matter:
//!
//!   * **Weights are f16, not f32.** The talker megakernel aliases Burn's f32
//!     buffers; here nothing is aliased and the stage is *bandwidth*-bound
//!     (16 positions re-stream all 78.6 M weights), so halving the bytes
//!     halves the frame. Accumulation stays f32.
//!   * **The q/k-norm + RoPE chain is its own dispatch**, where the
//!     megakernel folds it into a widened qkv matmul. That fold duplicates
//!     the qk columns: `wide_out` would be 7168 against a plain qkv's 4096,
//!     i.e. +20% on the whole layer's weight traffic to save one ~6 µs
//!     dispatch out of 34 per position. At the talker's f32-aliased shapes
//!     the trade pays; at these it does not.
//!
//! ## Layout
//!
//! Weights stay in the checkpoint's row-major `[out, in]` order (no
//! pre-transpose) and ride the thread shape [`crate::nn::q4`] tuned for
//! single-token matvecs: one 32-lane group per output row, 8 rows per cube,
//! `vec4` loads on both sides, shared-memory tree reduce. Three foldings
//! happen once at load, so no dispatch exists purely to normalize:
//!
//!   * `input_layernorm` folds into the qkv columns, `post_attention_layernorm`
//!     into gate‖up, the final `model.norm` into all 15 `lm_head`s — leaving
//!     each kernel a *weightless* rms it computes cube-cooperatively.
//!   * gate and up rows interleave (even = gate_j, odd = up_j) so SwiGLU is
//!     the matvec's epilogue, per [`crate::nn::q4::Q4Linear::forward_swiglu`].
//!   * `1/√d` folds into the RoPE chain's q output.
//!
//! Embedding tables stay **f32**: they are read 15 rows per frame (123 KB —
//! nothing), and their sum is what the talker consumes as its next input, so
//! this is the one place where rounding would leak into the backbone.
//!
//! ## The autoregressive chain never leaves the device
//!
//! Each of the 15 steps samples a token that indexes the next step's input
//! embedding. Reading that token back would cost 15 syncs per frame — more
//! than the whole budget. So sampling runs on the GPU
//! ([`gumbel_argmax_kernel`]) into a device slot that [`embed_gather_kernel`]
//! consumes directly, and a frame syncs **once**, on a single `[2048 + 15]`
//! buffer carrying the embedding sum and the codes.
//!
//! Sampling had to be handled, not avoided: production runs
//! `subtalker_do_sample: true`. Gumbel noise is drawn **on the host** from the
//! caller's `rand::Rng`, in the CPU path's exact order and count (15 × 2048
//! `gen_range(1e-12..1.0)`), and uploaded as one 122 KB buffer. That keeps
//! two properties a device-side RNG would break: the shared `rng` advances
//! identically, so the talker's own draws are unchanged, and the two paths
//! can be compared frame-for-frame on the *same* noise.

use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

use super::config::*;
use crate::nn::q4::{Client, GROUP, Rt};

/// Threads per cube. Fixed; [`PredictorEngine::lanes`] splits it into rows.
const CUBE: u32 = 256;
/// Lanes per output row, the swept default (see [`PredictorEngine::lanes`]).
const DEFAULT_LANES: u32 = 32;

const D: u32 = PRED_HEAD_DIM as u32; // 128
const HALF: u32 = D / 2; // 64
const HID: u32 = PRED_HIDDEN as u32; // 1024
const HEADS: u32 = PRED_HEADS as u32; // 16
const KV_HEADS: u32 = PRED_KV_HEADS as u32; // 8
const Q_DIM: u32 = HEADS * D; // 2048
const KV_DIM: u32 = KV_HEADS * D; // 1024
const QKV_OUT: u32 = Q_DIM + 2 * KV_DIM; // 4096
const INTER: u32 = 3072;
const VOCAB: u32 = PRED_VOCAB as u32; // 2048
/// 2 prefill positions + 14 single-token steps.
const MAX_POS: u32 = NUM_CODE_GROUPS as u32; // 16
const EPS: f32 = 1e-6; // TALKER_EPS
/// Steps that sample (codebooks 1..15).
const STEPS: usize = NUM_CODE_GROUPS - 1; // 15

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

/// `y = W · x`, f16 row-major `[out, in]` weights, f32 accumulation — the
/// workhorse. Comptime epilogues cover every projection in the stack:
///
/// * `pre_rms` — weightless `x / rms(x)` applied to the activation before the
///   dot product (the layernorm weight is already folded into `W`'s columns).
///   Computed redundantly per cube: `in` is at most 12 KB and lands in L1.
/// * `swiglu_pairs` — rows interleaved gate/up, lane 0 of each even row
///   writes `silu(g)·u`, halving `y`.
/// * `residual` — accumulate into `y` instead of overwriting (o_proj, down).
/// * `use_bias` — add `bias[row]` (small_to_mtp_projection is the one biased
///   weight in the predictor).
#[cube(launch_unchecked)]
#[allow(clippy::manual_is_multiple_of)] // `%` is the cube-kernel primitive
#[allow(clippy::too_many_arguments)]
pub(crate) fn matvec_kernel(
    x: &Array<Vector<f32, Const<4>>>,
    w: &Array<Vector<f16, Const<4>>>,
    bias: &Array<f32>,
    y: &mut Array<f32>,
    eps: f32,
    #[comptime] in_dim: u32,
    #[comptime] rows_per_cube: u32,
    #[comptime] threads_per_row: u32,
    #[comptime] pre_rms: bool,
    #[comptime] swiglu_pairs: bool,
    #[comptime] residual: bool,
    #[comptime] use_bias: bool,
) {
    let lane = UNIT_POS_X % threads_per_row;
    let row = CUBE_POS_X * rows_per_cube + UNIT_POS_X / threads_per_row;
    let octs = in_dim / 8;
    let cube_dim = rows_per_cube * threads_per_row;

    let mut red = SharedMemory::<f32>::new(comptime!((rows_per_cube * threads_per_row) as usize));

    // Weightless rms over the whole activation, cube-cooperative. Every cube
    // repeats it; the alternative is a separate dispatch per norm (10 per
    // position, 160 per frame) for a reduction over 4 KB.
    let mut rms_s = f32::new(1.0);
    if comptime![pre_rms] {
        let mut acc = f32::new(0.0);
        let mut k = UNIT_POS_X;
        while k < in_dim / 4 {
            let v = x[k as usize];
            acc += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
            k += cube_dim;
        }
        red[UNIT_POS_X as usize] = acc;
        sync_cube();
        let mut stride = comptime!(rows_per_cube * threads_per_row / 2).runtime();
        while stride > 0 {
            if UNIT_POS_X < stride {
                red[UNIT_POS_X as usize] =
                    red[UNIT_POS_X as usize] + red[(UNIT_POS_X + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        rms_s = 1.0 / (red[0] / (in_dim as f32) + eps).sqrt();
        sync_cube(); // red is reused for the row reduction below
    }

    // 8 weights per lane-strided iteration, vec4 on both sides.
    let mut acc = f32::new(0.0);
    let mut o = lane;
    while o < octs {
        let wb = row * (in_dim / 4) + o * 2;
        let xb = o * 2;
        let w0 = w[wb as usize];
        let w1 = w[(wb + 1) as usize];
        let x0 = x[xb as usize];
        let x1 = x[(xb + 1) as usize];
        acc += f32::cast_from(w0[0]) * x0[0];
        acc += f32::cast_from(w0[1]) * x0[1];
        acc += f32::cast_from(w0[2]) * x0[2];
        acc += f32::cast_from(w0[3]) * x0[3];
        acc += f32::cast_from(w1[0]) * x1[0];
        acc += f32::cast_from(w1[1]) * x1[1];
        acc += f32::cast_from(w1[2]) * x1[2];
        acc += f32::cast_from(w1[3]) * x1[3];
        o += threads_per_row;
    }
    if comptime![pre_rms] {
        acc *= rms_s;
    }

    red[UNIT_POS_X as usize] = acc;
    sync_cube();
    let mut stride = comptime!(threads_per_row / 2).runtime();
    while stride > 0 {
        if lane < stride {
            red[UNIT_POS_X as usize] =
                red[UNIT_POS_X as usize] + red[(UNIT_POS_X + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if comptime![swiglu_pairs] {
        let local_row = UNIT_POS_X / threads_per_row;
        if lane == 0 && local_row % 2 == 0 {
            let g = red[UNIT_POS_X as usize];
            let u = red[(UNIT_POS_X + threads_per_row) as usize];
            y[(row / 2) as usize] = g / (1.0 + (-g).exp()) * u;
        }
    } else if lane == 0 {
        let mut v = red[UNIT_POS_X as usize];
        if comptime![use_bias] {
            v += bias[row as usize];
        }
        if comptime![residual] {
            y[row as usize] = y[row as usize] + v;
        } else {
            y[row as usize] = v;
        }
    }
}

/// q8 twin of [`matvec_kernel`]: same thread shape, same epilogues, weights
/// read as GGUF-Q8_0-style groups instead of raw f16.
///
/// The stack re-streams all 78.6 M weights once per position, sixteen times a
/// frame, so bytes-per-weight *is* the frame time here. f16 costs 2 B; this
/// costs 1.0625 (one byte plus an f16 scale per 32), and on an M4 that is the
/// whole difference — there are no FP4 units to make a narrower format pay,
/// and a nibble format would spend the saving back on unpack ALU (measured
/// elsewhere in this tree: q4 and q8 land within noise of each other at these
/// shapes while q4 gives up an order of magnitude of fidelity).
///
/// Codes are stored **offset by 128** (`q = round(w/d) + 128`, `d = max|w|/127`)
/// so the unpack is a constant shift, a mask and one subtraction — no sign
/// extension and no branch on the hot path.
#[cube(launch_unchecked)]
#[allow(clippy::manual_is_multiple_of)] // `%` is the cube-kernel primitive
#[allow(clippy::too_many_arguments)]
fn matvec_q8_kernel(
    x: &Array<Vector<f32, Const<4>>>,
    wq: &Array<u32>,
    scales: &Array<f16>,
    bias: &Array<f32>,
    y: &mut Array<f32>,
    eps: f32,
    #[comptime] in_dim: u32,
    #[comptime] rows_per_cube: u32,
    #[comptime] threads_per_row: u32,
    #[comptime] pre_rms: bool,
    #[comptime] swiglu_pairs: bool,
    #[comptime] residual: bool,
    #[comptime] use_bias: bool,
) {
    let lane = UNIT_POS_X % threads_per_row;
    let row = CUBE_POS_X * rows_per_cube + UNIT_POS_X / threads_per_row;
    let words_per_row = in_dim / 4;
    let groups = in_dim / 32;
    let cube_dim = rows_per_cube * threads_per_row;

    let mut red = SharedMemory::<f32>::new(comptime!((rows_per_cube * threads_per_row) as usize));

    let mut rms_s = f32::new(1.0);
    if comptime![pre_rms] {
        let mut acc = f32::new(0.0);
        let mut k = UNIT_POS_X;
        while k < in_dim / 4 {
            let v = x[k as usize];
            acc += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
            k += cube_dim;
        }
        red[UNIT_POS_X as usize] = acc;
        sync_cube();
        let mut stride = comptime!(rows_per_cube * threads_per_row / 2).runtime();
        while stride > 0 {
            if UNIT_POS_X < stride {
                red[UNIT_POS_X as usize] =
                    red[UNIT_POS_X as usize] + red[(UNIT_POS_X + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        rms_s = 1.0 / (red[0] / (in_dim as f32) + eps).sqrt();
        sync_cube();
    }

    // 8 weights (2 packed words) per lane-strided iteration, two independent
    // partial sums so the FMA chain is not serialized on one accumulator —
    // the shape the q4 lane converged on. A pair never straddles a scale
    // group (8 words per group), so one scale fetch per pair is exact.
    let mut acc0 = f32::new(0.0);
    let mut acc1 = f32::new(0.0);
    let pairs = words_per_row / 2;
    let mut p = lane;
    while p < pairs {
        let w = p * 2;
        let word0 = wq[(row * words_per_row + w) as usize];
        let word1 = wq[(row * words_per_row + w + 1) as usize];
        let d = f32::cast_from(scales[(row * groups + w / 8) as usize]);
        let x0 = x[w as usize];
        let x1 = x[(w + 1) as usize];
        let mut s0 = (f32::cast_from(word0 & 255) - 128.0) * x0[0];
        let mut s1 = (f32::cast_from(word1 & 255) - 128.0) * x1[0];
        s0 += (f32::cast_from((word0 >> 8) & 255) - 128.0) * x0[1];
        s1 += (f32::cast_from((word1 >> 8) & 255) - 128.0) * x1[1];
        s0 += (f32::cast_from((word0 >> 16) & 255) - 128.0) * x0[2];
        s1 += (f32::cast_from((word1 >> 16) & 255) - 128.0) * x1[2];
        s0 += (f32::cast_from(word0 >> 24) - 128.0) * x0[3];
        s1 += (f32::cast_from(word1 >> 24) - 128.0) * x1[3];
        acc0 += d * s0;
        acc1 += d * s1;
        p += threads_per_row;
    }
    let mut acc = acc0 + acc1;
    if comptime![pre_rms] {
        acc *= rms_s;
    }

    red[UNIT_POS_X as usize] = acc;
    sync_cube();
    let mut stride = comptime!(threads_per_row / 2).runtime();
    while stride > 0 {
        if lane < stride {
            red[UNIT_POS_X as usize] =
                red[UNIT_POS_X as usize] + red[(UNIT_POS_X + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if comptime![swiglu_pairs] {
        let local_row = UNIT_POS_X / threads_per_row;
        if lane == 0 && local_row % 2 == 0 {
            let g = red[UNIT_POS_X as usize];
            let u = red[(UNIT_POS_X + threads_per_row) as usize];
            y[(row / 2) as usize] = g / (1.0 + (-g).exp()) * u;
        }
    } else if lane == 0 {
        let mut v = red[UNIT_POS_X as usize];
        if comptime![use_bias] {
            v += bias[row as usize];
        }
        if comptime![residual] {
            y[row as usize] = y[row as usize] + v;
        } else {
            y[row as usize] = v;
        }
    }
}

/// Per-head q/k RMSNorm + RoPE + KV-cache append, one cube per q or k head.
///
/// Cube `h < heads` normalizes and rotates query head `h` into `q_out`
/// (scaled by `1/√d`, folded here so attention needs no scale); cubes
/// `heads <= h < heads + kv_heads` do the same for key head `h - heads` and
/// write it to `kcache[pos]`; the remaining `kv_heads` cubes copy value heads
/// into `vcache[pos]`. `rope` is `[pos][cos(half) ‖ sin(half)]` — the
/// half-split rotate the reference uses, not the duplicated full-width form.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn qk_norm_rope_kernel(
    qkv: &Array<f32>,
    qn: &Array<f32>,
    kn: &Array<f32>,
    rope: &Array<f32>,
    q_out: &mut Array<f32>,
    kcache: &mut Array<f32>,
    vcache: &mut Array<f32>,
    pos: u32,
    scale: f32,
    eps: f32,
    #[comptime] heads: u32,
    #[comptime] kv_heads: u32,
    #[comptime] d: u32,
) {
    let i = UNIT_POS_X;
    let h = CUBE_POS_X;
    let half = d / 2;
    let q_dim = heads * d;
    let kv_dim = kv_heads * d;

    // value heads: a straight copy, no norm and no rotation
    if h >= heads + kv_heads {
        let vh = h - heads - kv_heads;
        vcache[(pos * kv_dim + vh * d + i) as usize] = qkv[(q_dim + kv_dim + vh * d + i) as usize];
    } else {
        let src = if h < heads {
            h * d + i
        } else {
            q_dim + (h - heads) * d + i
        };
        let v = qkv[src as usize];

        let mut red = SharedMemory::<f32>::new(comptime!(d as usize));
        red[i as usize] = v * v;
        sync_cube();
        let mut stride = comptime!(d / 2).runtime();
        while stride > 0 {
            if i < stride {
                red[i as usize] = red[i as usize] + red[(i + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        let s = 1.0 / (red[0] / (d as f32) + eps).sqrt();
        sync_cube();

        // normalized value at this lane, and at its rotate partner
        let w = if h < heads {
            qn[i as usize]
        } else {
            kn[i as usize]
        };
        red[i as usize] = v * s * w;
        sync_cube();
        let lo = i % half; // index into the cos/sin tables
        let upper = i >= half;
        let a = red[lo as usize];
        let b = red[(lo + half) as usize];
        let c = rope[(pos * d + lo) as usize];
        let sn = rope[(pos * d + half + lo) as usize];
        let out = if upper {
            b * c + a * sn
        } else {
            a * c - b * sn
        };

        if h < heads {
            q_out[(h * d + i) as usize] = out * scale;
        } else {
            kcache[(pos * kv_dim + (h - heads) * d + i) as usize] = out;
        }
    }
}

/// Single-token causal attention over the frame's cache, one cube per query
/// head. GQA is pure indexing; `1/√d` is already in `q`. `len` counts the
/// cache including the current position (at most [`MAX_POS`] = 16, so the
/// score buffer is 64 B).
#[cube(launch_unchecked)]
fn attn_kernel(
    q: &Array<f32>,
    kcache: &Array<f32>,
    vcache: &Array<f32>,
    out: &mut Array<f32>,
    len: u32,
    #[comptime] kv_heads: u32,
    #[comptime] groups: u32,
    #[comptime] d: u32,
    #[comptime] max_pos: u32,
) {
    let i = UNIT_POS_X;
    let h = CUBE_POS_X;
    let kvh = h / groups;
    let kv_dim = kv_heads * d;

    let mut qsh = SharedMemory::<f32>::new(comptime!(d as usize));
    let mut scores = SharedMemory::<f32>::new(comptime!(max_pos as usize));

    qsh[i as usize] = q[(h * d + i) as usize];
    sync_cube();

    if i < len {
        let base = (i * kv_dim + kvh * d) as usize;
        let mut s = f32::new(0.0);
        for dd in 0..d {
            s += qsh[dd as usize] * kcache[base + dd as usize];
        }
        scores[i as usize] = s;
    }
    sync_cube();

    // max / exp / sum over at most 16 entries — one lane does it, then
    // broadcasts through shared memory. A tree reduce over 128 threads for 16
    // values costs more barriers than the serial scan costs arithmetic.
    if i == 0 {
        let mut m = f32::new(-3.40282e38);
        for t in 0..len {
            let sv = scores[t as usize];
            if sv > m {
                m = sv;
            }
        }
        let mut sum = f32::new(0.0);
        for t in 0..len {
            let p = (scores[t as usize] - m).exp();
            scores[t as usize] = p;
            sum += p;
        }
        for t in 0..len {
            scores[t as usize] = scores[t as usize] / sum;
        }
    }
    sync_cube();

    let mut acc = f32::new(0.0);
    for t in 0..len {
        acc += scores[t as usize] * vcache[(t * kv_dim + kvh * d + i) as usize];
    }
    out[(h * d + i) as usize] = acc;
}

/// Full-vocab gumbel-max sampling in one cube, writing the winning id to a
/// **device** slot so the chain never round-trips through the host.
///
/// `z = logits[i]/temperature + noise[base + i]`, argmax with the lowest index
/// winning ties — the CPU path's strict `>` scan. Greedy decode is the same
/// kernel with a zeroed `noise` and `temperature = 1`, which reduces it to
/// argmax(logits) exactly.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gumbel_argmax_kernel(
    logits: &Array<f32>,
    noise: &Array<f32>,
    code_dev: &mut Array<f32>,
    out: &mut Array<f32>,
    noise_base: u32,
    step: u32,
    out_slot: u32,
    temperature: f32,
    #[comptime] vocab: u32,
    #[comptime] cube_dim: u32,
) {
    let i = UNIT_POS_X;
    let mut best = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    let mut bidx = SharedMemory::<u32>::new(comptime!(cube_dim as usize));

    let mut bv = f32::new(-3.40282e38);
    let mut bi = u32::new(0);
    let mut k = i;
    while k < vocab {
        let z = logits[k as usize] / temperature + noise[(noise_base + k) as usize];
        if z > bv {
            bv = z;
            bi = k;
        }
        k += cube_dim;
    }
    best[i as usize] = bv;
    bidx[i as usize] = bi;
    sync_cube();
    let mut stride = comptime!(cube_dim / 2).runtime();
    while stride > 0 {
        if i < stride {
            let ov = best[(i + stride) as usize];
            let oi = bidx[(i + stride) as usize];
            let cv = best[i as usize];
            let ci = bidx[i as usize];
            if ov > cv || (ov == cv && oi < ci) {
                best[i as usize] = ov;
                bidx[i as usize] = oi;
            }
        }
        sync_cube();
        stride /= 2;
    }
    if i == 0 {
        let code = f32::cast_from(bidx[0]);
        code_dev[step as usize] = code;
        out[out_slot as usize] = code;
    }
}

/// Consume the sampled id from its device slot: accumulate the step's
/// talker-width embedding row into `embed_sum` (the predictor's share of the
/// talker's next input) and stage the same row as the next position's input.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn embed_gather_kernel(
    table: &Array<f32>,
    code_dev: &Array<f32>,
    embed_sum: &mut Array<f32>,
    next_in: &mut Array<f32>,
    step: u32,
    #[comptime] width: u32,
) {
    let j = ABSOLUTE_POS as u32;
    if j < width {
        let code = u32::cast_from(code_dev[step as usize]);
        let v = table[(code * width + j) as usize];
        embed_sum[j as usize] = embed_sum[j as usize] + v;
        next_in[j as usize] = v;
    }
}

/// `dst[j] = src[j]` — the 0.6B checkpoint's identity projection, where the
/// talker is already predictor-width and the reference uses `nn.Identity()`.
#[cube(launch_unchecked)]
fn copy_kernel(src: &Array<f32>, dst: &mut Array<f32>, n: u32) {
    let j = ABSOLUTE_POS as u32;
    if j < n {
        dst[j as usize] = src[j as usize];
    }
}

// ---------------------------------------------------------------------------
// host engine
// ---------------------------------------------------------------------------

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Storage format for the streamed weights. The stack is re-read once per
/// position, so this choice *is* the frame time; everything else here is
/// noise beside it.
/// Measured against the CPU f32 oracle over a real utterance
/// (`MARY_PRED_GATE=1`, per-codebook token agreement):
///
/// | format | B/weight | ms/frame | token agreement |
/// |---|---|---|---|
/// | f16 | 2.0 | 17.8 | **1275/1275 = 100.00%**, max \|Δembed_sum\| = 0 |
/// | q8  | 1.0625 | 9.8 | 1148/1350 = 85.04% |
///
/// f16 is the default, and the 8 ms is deliberately left on the table. The
/// talker ahead of this stage costs ~28 ms/frame against an 80 ms budget, so
/// the faster format buys throughput nobody is short of, while the agreement
/// column is the whole reason to trust the port at all — an exact-to-the-token
/// replacement of a CPU stage is a different object from a close one.
///
/// The q8 result is also the useful negative: the lesson carried in from the
/// depth port (q8 costs nothing on an M4 because the path is dispatch-bound)
/// does **not** transfer here. This stage re-streams its whole 78.6 M-weight
/// stack once per position, sixteen times a frame, so it is squarely
/// bandwidth-bound and narrower weights really do buy time — they just buy it
/// with fidelity, because the acoustic codebooks' logits sit close enough
/// together that 8-bit weights reorder the top of a 2048-way argmax one time
/// in seven.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightMode {
    /// 2 B/weight, exact to the token. The default.
    F16,
    /// 1.0625 B/weight: ~1.8× the frame rate for ~15% token divergence.
    /// Opt in with `MARY_PRED_W=q8` when throughput is worth more than
    /// agreement.
    Q8,
}

impl WeightMode {
    /// `MARY_PRED_W=q8` opts into the narrow format.
    pub fn from_env() -> Self {
        match std::env::var("MARY_PRED_W").as_deref() {
            Ok("q8") => Self::Q8,
            _ => Self::F16,
        }
    }

    fn bytes_per_weight(self) -> f64 {
        match self {
            Self::F16 => 2.0,
            Self::Q8 => 1.0 + 2.0 / GROUP as f64,
        }
    }
}

/// A streamed weight on the device, in whichever format the engine was built
/// with.
enum W {
    F16(Handle),
    Q8 { wq: Handle, scales: Handle },
}

/// Quantize a row-major `[out, in]` f32 weight to (packed bytes, f16 scales),
/// 32 weights along the input dim per scale.
///
/// `d = max|w| / 127` per group and `q = round(w/d) + 128`, so the largest
/// magnitude in a group lands on ±127 and the kernel's unpack is a subtract.
/// The scale is rounded through f16 *before* quantizing, so the stored codes
/// are optimal for the scale that will actually be used — the same discipline
/// as [`crate::nn::q4::quantize_q4`].
fn quantize_q8(w: &[f32], out_dim: usize, in_dim: usize) -> (Vec<u32>, Vec<f16>) {
    assert_eq!(in_dim % GROUP, 0);
    let groups = in_dim / GROUP;
    let mut wq = vec![0u32; out_dim * in_dim / 4];
    let mut scales = vec![f16::ZERO; out_dim * groups];
    for r in 0..out_dim {
        for g in 0..groups {
            let src = &w[r * in_dim + g * GROUP..][..GROUP];
            let amax = src.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let d = f16::from_f32(if amax == 0.0 { 1.0 } else { amax / 127.0 });
            scales[r * groups + g] = d;
            let inv = 1.0 / d.to_f32();
            for (i, &v) in src.iter().enumerate() {
                let q = ((v * inv).round() as i32 + 128).clamp(0, 255) as u32;
                let k = g * GROUP + i;
                wq[(r * in_dim + k) / 4] |= q << (8 * (k % 4));
            }
        }
    }
    (wq, scales)
}

/// Encode a row-major `[out, in]` f32 weight in `mode` and upload it.
fn upload(client: &Client, mode: WeightMode, w: &[f32], out_dim: usize, in_dim: usize) -> W {
    assert_eq!(w.len(), out_dim * in_dim);
    match mode {
        WeightMode::F16 => {
            let h: Vec<f16> = w.iter().map(|&v| f16::from_f32(v)).collect();
            W::F16(client.create_from_slice(as_bytes(&h)))
        }
        WeightMode::Q8 => {
            let (wq, scales) = quantize_q8(w, out_dim, in_dim);
            W::Q8 {
                wq: client.create_from_slice(as_bytes(&wq)),
                scales: client.create_from_slice(as_bytes(&scales)),
            }
        }
    }
}

/// A weight with the layernorm that precedes it folded into its columns:
/// `W'[j, k] = W[j, k] · norm[k]`, so the kernel's `pre_rms` epilogue is all
/// that remains of the normalization.
fn fold_norm(w: &[f32], out_dim: usize, in_dim: usize, norm: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(norm.len(), in_dim);
    let mut o = vec![0f32; w.len()];
    for r in 0..out_dim {
        for k in 0..in_dim {
            o[r * in_dim + k] = w[r * in_dim + k] * norm[k];
        }
    }
    o
}

struct LayerBufs {
    qkv: W,         // [4096, 1024], input_layernorm folded into the columns
    q_norm: Handle, // [128] f32
    k_norm: Handle, // [128] f32
    o: W,           // [1024, 2048]
    gate_up: W,     // [6144, 1024], interleaved gate/up, post_norm folded
    down: W,        // [1024, 3072]
    kcache: Handle, // [MAX_POS, 1024] f32, position-major
    vcache: Handle,
}

/// How a frame's ~530 launches reach the device. The first frame runs
/// eagerly, because the first launch of a shape compiles it and a compile
/// must not happen inside a capture; the second frame is captured as one CUDA
/// graph over the persistent buffers; every frame after that is one replay.
/// `MARY_PRED_EAGER=1` holds the eager path (the A/B), as does a backend that
/// cannot capture or a capture the driver invalidated.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Lane {
    Cold,
    Warm,
    Graph(u64),
    Eager,
}

/// The code predictor as a device-resident engine. One
/// [`predict_frame`](Self::predict_frame) call encodes ~530 dispatches and
/// syncs exactly once, mirroring
/// [`super::predictor::CodePredictor::predict_frame`] position for position.
pub struct PredictorEngine {
    client: Client,
    layers: Vec<LayerBufs>,
    /// 15 talker-width embedding tables `[2048, talker_width]`, f32 — the one
    /// place rounding would leak into the talker's own input stream.
    embeddings: Vec<Handle>,
    /// 15 `lm_head`s `[2048, 1024]`, `model.norm` folded into the columns.
    lm_heads: Vec<W>,
    /// small_to_mtp_projection `[1024, talker_width]` + its f32 bias, absent
    /// on the 0.6B checkpoint whose talker is already predictor-width.
    proj: Option<(W, Handle)>,
    rope: Handle,
    dummy: Handle,
    // activations, all persistent
    x: Handle,        // [1024] residual stream
    xin: Handle,      // [talker_width] next position's input
    qkv: Handle,      // [4096]
    q: Handle,        // [2048]
    attn: Handle,     // [2048]
    act: Handle,      // [3072]
    logits: Handle,   // [2048]
    code_dev: Handle, // [15] the sampled ids, device-side, never read back
    // The frame's inputs and its one output, persistent so the launch
    // sequence sees the same pointers every frame and can be captured once.
    noise_buf: Handle, // [15 × 2048] the frame's gumbel noise
    h0: Handle,        // [talker_width] the talker's last hidden state
    h1: Handle,        // [talker_width] codebook-0's embedding row
    out: Handle,       // [talker_width + 15] embedding sum ‖ sampled ids
    zeros: Handle,     // [talker_width + 15] to clear `out` each frame
    /// Where the frame's launches go: eager, or one captured graph replayed.
    lane: std::cell::Cell<Lane>,
    /// The sampling temperature the captured graph bakes in (a by-value
    /// kernel argument); a session keeps one, and replay asserts it.
    captured_temp: std::cell::Cell<f32>,
    talker_width: usize,
    /// The format the streamed stack is stored in.
    pub mode: WeightMode,
    /// Lanes cooperating on one output row, and rows per cube (their product
    /// is the 256-thread cube). The q4 lane's 32 is tuned for Moshi's
    /// 4096-wide rows; the predictor's are 1024–3072, where 32 lanes leave
    /// each lane only 4 loop iterations and no latency to hide behind.
    pub lanes: u32,
    /// Bytes of weight this engine streams per frame — the quantity the
    /// format choice moves, and the one to divide by frame time for an
    /// effective-bandwidth number.
    pub bytes_per_frame: usize,
    /// Dispatches encoded per frame — the count this design trades against
    /// the CPU path's 5 GB of strictly sequential weight traffic.
    pub dispatches: usize,
}

impl PredictorEngine {
    /// Build from a loaded CPU predictor: fold the three layernorms, round to
    /// f16, interleave gate‖up, upload. ~130 MB of device memory for the f16
    /// stack plus ~250 MB for the f32 embedding tables.
    pub fn new(client: Client, p: &super::predictor::CodePredictor) -> Self {
        Self::with_mode(client, p, WeightMode::from_env())
    }

    /// [`new`](Self::new) at an explicit storage format — the seam the
    /// f16-vs-q8 parity comparison runs through.
    pub fn with_mode(
        client: Client,
        p: &super::predictor::CodePredictor,
        mode: WeightMode,
    ) -> Self {
        let lanes = std::env::var("MARY_PRED_LANES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_LANES);
        Self::with_shape(client, p, mode, lanes)
    }

    /// [`with_mode`](Self::with_mode) at an explicit row-sweep width.
    pub fn with_shape(
        client: Client,
        p: &super::predictor::CodePredictor,
        mode: WeightMode,
        lanes: u32,
    ) -> Self {
        assert!(CUBE == lanes * (CUBE / lanes), "lanes must divide the cube");
        let tw = p.talker_width();
        let h = HID as usize;
        let layers = p
            .layer_weights()
            .map(|l| {
                // gate‖up arrives concatenated (3072 gate rows, then 3072 up
                // rows); the fused SwiGLU epilogue wants them interleaved.
                let folded = fold_norm(l.gate_up, 2 * INTER as usize, h, l.post_norm);
                let mut inter = vec![0f32; folded.len()];
                for j in 0..INTER as usize {
                    inter[2 * j * h..(2 * j + 1) * h].copy_from_slice(&folded[j * h..(j + 1) * h]);
                    inter[(2 * j + 1) * h..(2 * j + 2) * h]
                        .copy_from_slice(&folded[(INTER as usize + j) * h..][..h]);
                }
                LayerBufs {
                    qkv: upload(
                        &client,
                        mode,
                        &fold_norm(l.qkv, QKV_OUT as usize, h, l.in_norm),
                        QKV_OUT as usize,
                        h,
                    ),
                    q_norm: client.create_from_slice(as_bytes(l.q_norm)),
                    k_norm: client.create_from_slice(as_bytes(l.k_norm)),
                    o: upload(&client, mode, l.o, h, Q_DIM as usize),
                    gate_up: upload(&client, mode, &inter, 2 * INTER as usize, h),
                    down: upload(&client, mode, l.down, h, INTER as usize),
                    kcache: client.empty(MAX_POS as usize * KV_DIM as usize * 4),
                    vcache: client.empty(MAX_POS as usize * KV_DIM as usize * 4),
                }
            })
            .collect();

        // half-split rotate_half table: [pos][cos(64) ‖ sin(64)]
        let mut rope = vec![0f32; MAX_POS as usize * D as usize];
        for pos in 0..MAX_POS as usize {
            for i in 0..HALF as usize {
                let r = pos as f64 * TALKER_ROPE_THETA.powf(-2.0 * i as f64 / D as f64);
                rope[pos * D as usize + i] = r.cos() as f32;
                rope[pos * D as usize + HALF as usize + i] = r.sin() as f32;
            }
        }

        let lm_heads = p
            .lm_head_weights()
            .map(|w| {
                upload(
                    &client,
                    mode,
                    &fold_norm(w, VOCAB as usize, h, p.norm_weight()),
                    VOCAB as usize,
                    h,
                )
            })
            .collect();
        let embeddings = p
            .embedding_tables()
            .map(|e| client.create_from_slice(as_bytes(e)))
            .collect();
        let proj = p.proj_weights().map(|(w, b)| {
            (
                upload(&client, mode, w, h, tw),
                client.create_from_slice(as_bytes(b)),
            )
        });

        Self {
            rope: client.create_from_slice(as_bytes(&rope)),
            dummy: client.create_from_slice(as_bytes(&[0f32; 4])),
            x: client.empty(h * 4),
            xin: client.empty(tw * 4),
            qkv: client.empty(QKV_OUT as usize * 4),
            q: client.empty(Q_DIM as usize * 4),
            attn: client.empty(Q_DIM as usize * 4),
            act: client.empty(INTER as usize * 4),
            logits: client.empty(VOCAB as usize * 4),
            code_dev: client.empty(STEPS * 4),
            noise_buf: client.empty(STEPS * VOCAB as usize * 4),
            h0: client.empty(tw * 4),
            h1: client.empty(tw * 4),
            out: client.empty((tw + STEPS) * 4),
            zeros: client.create_from_slice(as_bytes(&vec![0f32; tw + STEPS])),
            lane: std::cell::Cell::new(Lane::Cold),
            captured_temp: std::cell::Cell::new(0.0),
            layers,
            embeddings,
            lm_heads,
            proj,
            talker_width: tw,
            mode,
            lanes,
            // 16 positions × the 5-layer stack, plus 16 projections and 15
            // lm_heads — the whole reason the format matters.
            bytes_per_frame: ((MAX_POS as usize
                * (PRED_LAYERS
                    * (QKV_OUT as usize * h
                        + h * Q_DIM as usize
                        + 2 * INTER as usize * h
                        + h * INTER as usize)
                    + h * tw)
                + STEPS * VOCAB as usize * h) as f64
                * mode.bytes_per_weight()) as usize,
            // per position: proj + 5×(qkv, rope, attn, o, gate_up, down);
            // per step: lm_head + sample + gather.
            dispatches: MAX_POS as usize * (1 + PRED_LAYERS * 6) + STEPS * 3,
            client,
        }
    }

    /// One matvec dispatch. `out_dim` counts weight *rows*; the SwiGLU
    /// epilogue halves what lands in `y`.
    #[allow(clippy::too_many_arguments)]
    fn matvec(
        &self,
        x: &Handle,
        w: &W,
        bias: &Handle,
        y: &Handle,
        in_dim: u32,
        out_dim: u32,
        pre_rms: bool,
        swiglu: bool,
        residual: bool,
        use_bias: bool,
    ) {
        let y_len = if swiglu { out_dim / 2 } else { out_dim };
        let rows_per_cube = CUBE / self.lanes;
        let count = CubeCount::new_1d(out_dim / rows_per_cube);
        let dim = CubeDim::new_1d(CUBE);
        unsafe {
            match w {
                W::F16(h) => matvec_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    count,
                    dim,
                    ArrayArg::from_raw_parts(x.clone(), (in_dim / 4) as usize),
                    ArrayArg::from_raw_parts(h.clone(), (out_dim * in_dim / 4) as usize),
                    ArrayArg::from_raw_parts(bias.clone(), out_dim as usize),
                    ArrayArg::from_raw_parts(y.clone(), y_len as usize),
                    EPS,
                    in_dim,
                    rows_per_cube,
                    self.lanes,
                    pre_rms,
                    swiglu,
                    residual,
                    use_bias,
                ),
                W::Q8 { wq, scales } => matvec_q8_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    count,
                    dim,
                    ArrayArg::from_raw_parts(x.clone(), (in_dim / 4) as usize),
                    ArrayArg::from_raw_parts(wq.clone(), (out_dim * in_dim / 4) as usize),
                    ArrayArg::from_raw_parts(
                        scales.clone(),
                        (out_dim * in_dim / GROUP as u32) as usize,
                    ),
                    ArrayArg::from_raw_parts(bias.clone(), out_dim as usize),
                    ArrayArg::from_raw_parts(y.clone(), y_len as usize),
                    EPS,
                    in_dim,
                    rows_per_cube,
                    self.lanes,
                    pre_rms,
                    swiglu,
                    residual,
                    use_bias,
                ),
            }
        }
    }

    /// `x = proj(src)` — small_to_mtp_projection, or the 0.6B's identity.
    fn project(&self, src: &Handle) {
        match &self.proj {
            Some((w, b)) => self.matvec(
                src,
                w,
                b,
                &self.x,
                self.talker_width as u32,
                HID,
                false,
                false,
                false,
                true,
            ),
            // identity: talker_width == PRED_HIDDEN, so a plain copy
            None => unsafe {
                copy_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d(HID.div_ceil(256)),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(src.clone(), HID as usize),
                    ArrayArg::from_raw_parts(self.x.clone(), HID as usize),
                    HID,
                );
            },
        }
    }

    /// One position through the 5-layer stack, in place on `self.x`.
    fn forward_pos(&self, pos: u32) {
        let scale = ((D as f64).powf(-0.5)) as f32;
        for l in &self.layers {
            self.matvec(
                &self.x,
                &l.qkv,
                &self.dummy,
                &self.qkv,
                HID,
                QKV_OUT,
                true,
                false,
                false,
                false,
            );
            unsafe {
                qk_norm_rope_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d(HEADS + 2 * KV_HEADS),
                    CubeDim::new_1d(D),
                    ArrayArg::from_raw_parts(self.qkv.clone(), QKV_OUT as usize),
                    ArrayArg::from_raw_parts(l.q_norm.clone(), D as usize),
                    ArrayArg::from_raw_parts(l.k_norm.clone(), D as usize),
                    ArrayArg::from_raw_parts(self.rope.clone(), (MAX_POS * D) as usize),
                    ArrayArg::from_raw_parts(self.q.clone(), Q_DIM as usize),
                    ArrayArg::from_raw_parts(l.kcache.clone(), (MAX_POS * KV_DIM) as usize),
                    ArrayArg::from_raw_parts(l.vcache.clone(), (MAX_POS * KV_DIM) as usize),
                    pos,
                    scale,
                    EPS,
                    HEADS,
                    KV_HEADS,
                    D,
                );
                attn_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d(HEADS),
                    CubeDim::new_1d(D),
                    ArrayArg::from_raw_parts(self.q.clone(), Q_DIM as usize),
                    ArrayArg::from_raw_parts(l.kcache.clone(), (MAX_POS * KV_DIM) as usize),
                    ArrayArg::from_raw_parts(l.vcache.clone(), (MAX_POS * KV_DIM) as usize),
                    ArrayArg::from_raw_parts(self.attn.clone(), Q_DIM as usize),
                    pos + 1,
                    KV_HEADS,
                    HEADS / KV_HEADS,
                    D,
                    MAX_POS,
                );
            }
            self.matvec(
                &self.attn,
                &l.o,
                &self.dummy,
                &self.x,
                Q_DIM,
                HID,
                false,
                false,
                true,
                false,
            );
            self.matvec(
                &self.x,
                &l.gate_up,
                &self.dummy,
                &self.act,
                HID,
                2 * INTER,
                true,
                true,
                false,
                false,
            );
            self.matvec(
                &self.act,
                &l.down,
                &self.dummy,
                &self.x,
                INTER,
                HID,
                false,
                false,
                true,
                false,
            );
        }
    }

    /// Predict codebooks 1..15 for one frame — the device twin of
    /// [`super::predictor::CodePredictor::predict_frame`], same signature,
    /// same `rng` consumption, one sync.
    ///
    /// `talker_hidden` and `code0_embed` are talker-width slices; returns the
    /// 15 codes and Σ of their talker-width embedding rows.
    pub fn predict_frame(
        &self,
        talker_hidden: &[f32],
        code0_embed: &[f32],
        do_sample: bool,
        temperature: f64,
        rng: &mut impl rand::Rng,
    ) -> ([u32; STEPS], Vec<f32>) {
        let tw = self.talker_width;
        assert_eq!(talker_hidden.len(), tw);
        assert_eq!(code0_embed.len(), tw);

        // The frame's gumbel noise, drawn in the CPU path's exact order and
        // count so `rng` advances identically whichever engine is running.
        let temp = if do_sample { temperature as f32 } else { 1.0 };
        let noise: Vec<f32> = if do_sample {
            (0..STEPS * VOCAB as usize)
                .map(|_| {
                    let u: f64 = rng.gen_range(1e-12..1.0);
                    (-(-u.ln()).ln()) as f32
                })
                .collect()
        } else {
            vec![0f32; STEPS * VOCAB as usize]
        };

        // Inputs into their persistent buffers — one upload and one copy
        // each, outside any graph — and `out` cleared.
        self.refresh(&self.noise_buf, &noise);
        self.refresh(&self.h0, talker_hidden);
        self.refresh(&self.h1, code0_embed);
        self.copy(&self.zeros, &self.out, (tw + STEPS) as u32);

        match self.lane.get() {
            Lane::Eager => self.frame_launches(temp),
            Lane::Cold => {
                let capturable = std::env::var("MARY_PRED_EAGER").is_err()
                    && self.client.graph_capture_supported();
                if capturable {
                    // The warm frame runs with the capture arena open, so the
                    // runtime learns the region's allocation sequence (launch
                    // metadata) at addresses a capture may then bake in.
                    self.client.graph_arena_begin();
                    self.frame_launches(temp);
                    self.client.graph_arena_end();
                    self.lane.set(Lane::Warm);
                } else {
                    self.frame_launches(temp);
                    self.lane.set(Lane::Eager);
                }
            }
            Lane::Warm => {
                // Drain the drop queue before the capture opens: inside it the
                // flush is suppressed, so it must not be due. The arena stays
                // reserved after it closes, which is what keeps every address
                // the graph recorded valid for as long as it is replayed.
                self.client.flush();
                self.client.graph_arena_begin();
                self.client.graph_capture_begin();
                self.frame_launches(temp);
                let intact = self.client.graph_capture_status() == 1;
                let g = self.client.graph_capture_end();
                self.client.graph_arena_end();
                if intact {
                    eprintln!(
                        "[predictor] captured the frame: {} launches → {} graph nodes",
                        self.client.graph_launch_count(g),
                        self.client.graph_node_count(g)
                    );
                    self.captured_temp.set(temp);
                    self.lane.set(Lane::Graph(g));
                    // captured work never executed: this frame runs as a replay
                    self.client.graph_replay(g);
                } else {
                    eprintln!("[predictor] the driver invalidated the capture; staying eager");
                    self.client.graph_destroy(g);
                    self.lane.set(Lane::Eager);
                    self.frame_launches(temp);
                }
            }
            Lane::Graph(g) => {
                assert!(
                    (self.captured_temp.get() - temp).abs() < 1e-9,
                    "the captured frame bakes in the sampling temperature; a session keeps one"
                );
                self.client.graph_replay(g);
            }
        }

        // the frame's one sync
        let bytes = self.client.read_one(self.out.clone()).expect("readback");
        let vals = f32::from_bytes(&bytes);
        let mut codes = [0u32; STEPS];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = vals[tw + i] as u32;
        }
        (codes, vals[..tw].to_vec())
    }

    /// Whether frames are currently replayed from a captured graph.
    pub fn replaying(&self) -> bool {
        matches!(self.lane.get(), Lane::Graph(_))
    }

    /// Upload `host` and copy it into the persistent `dst`.
    fn refresh(&self, dst: &Handle, host: &[f32]) {
        let src = self.client.create_from_slice(as_bytes(host));
        self.copy(&src, dst, host.len() as u32);
    }

    fn copy(&self, src: &Handle, dst: &Handle, n: u32) {
        unsafe {
            copy_kernel::launch_unchecked::<Rt>(
                &self.client,
                CubeCount::new_1d(n.div_ceil(256)),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(src.clone(), n as usize),
                ArrayArg::from_raw_parts(dst.clone(), n as usize),
                n,
            );
        }
    }

    /// The frame's launch sequence over the persistent buffers: two
    /// projected positions, then 15 steps of head → sample → gather →
    /// position. Every pointer and scalar here is the same each frame.
    fn frame_launches(&self, temp: f32) {
        let tw = self.talker_width;
        self.project(&self.h0);
        self.forward_pos(0);
        self.project(&self.h1);
        self.forward_pos(1);

        for step in 0..STEPS {
            self.matvec(
                &self.x,
                &self.lm_heads[step],
                &self.dummy,
                &self.logits,
                HID,
                VOCAB,
                true,
                false,
                false,
                false,
            );
            unsafe {
                gumbel_argmax_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(self.logits.clone(), VOCAB as usize),
                    ArrayArg::from_raw_parts(self.noise_buf.clone(), STEPS * VOCAB as usize),
                    ArrayArg::from_raw_parts(self.code_dev.clone(), STEPS),
                    ArrayArg::from_raw_parts(self.out.clone(), tw + STEPS),
                    step as u32 * VOCAB,
                    step as u32,
                    (tw + step) as u32,
                    temp,
                    VOCAB,
                    256,
                );
                embed_gather_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d((tw as u32).div_ceil(256)),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(self.embeddings[step].clone(), VOCAB as usize * tw),
                    ArrayArg::from_raw_parts(self.code_dev.clone(), STEPS),
                    ArrayArg::from_raw_parts(self.out.clone(), tw + STEPS),
                    ArrayArg::from_raw_parts(self.xin.clone(), tw),
                    step as u32,
                    tw as u32,
                );
            }
            if step + 1 < STEPS {
                self.project(&self.xin);
                self.forward_pos(step as u32 + 2);
            }
        }
    }
}
