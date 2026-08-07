//! PersonaPlex-7B temporal transformer — the **Metal quantized realtime
//! build**.
//!
//! The CPU-f32 [`super::temporal`] stack is the parity oracle (cos = 1.0 vs
//! the moshi goldens, ~3.5 s/step); this module is the same math rebuilt for
//! the 80 ms/frame budget: hand-launched cubecl kernels on the raw (non-fused)
//! wgpu/Metal device — burn-fusion 0.21 miscompiles FusedReduceLaunch strides
//! on the d4096 L=1 rms `mean_dim`, so the Moshi graph never touches a fused
//! backend — with the 7 per-layer matvec weights in a load-time
//! [`WeightFmt`]: **q4** ([`crate::nn::q4`], GGUF-Q4_0-style, 4.5 bit/w —
//! 17.2 GB/step of f16 weight traffic becomes ~3.7 GB), **q8** (Q8_0-style,
//! ~7.0 GB — the fidelity/bandwidth recommendation), or **f16** (the
//! exactness ablation).
//!
//! ## What is exact vs approximate (numerics honesty)
//!
//! Quantization is a REAL numerics change and this build does NOT claim
//! token-exactness vs the f32 oracle — `personaplex_rt_probe gate` runs the
//! 113-step golden stream and reports per-step text-logits cosine, argmax
//! agreement and the first divergence step, per weight format:
//!
//! - **q4** (GGUF-Q4_0, 4.5 bit/w): measured per-matvec rel RMS on the real
//!   checkpoint is **~8.7e-2** — the ANALYTIC q4_0 class for Gaussian-ish
//!   weights (step = max/8 ≈ 0.275σ ⇒ err ≈ 0.08σ; the uniform-weights
//!   figure 6.3e-2 and the spike's "~3e-2" were optimistic for real
//!   weights) — and it COMPOUNDS through 64 residual adds. q4 buys the
//!   lowest bandwidth, at real fidelity cost (see the gate numbers in
//!   PORT_NOTES).
//! - **q8** (GGUF-Q8_0-style biased bytes, 8.5 bit/w): per-matvec ~6e-3 —
//!   an order of magnitude tighter, at 1.9× the q4 bytes; the
//!   fidelity/bandwidth sweet spot for the 80 ms budget.
//! - **f16**: near-exact (~2e-4/matvec) — the ablation format that separates
//!   "kernel/pipeline bugs" from "quantization noise"; too many bytes for
//!   the realtime budget at full context.
//!
//! The KV cache is f16 in all formats (~5e-4-relative on k/v, below even the
//! q8 noise floor). Everything else — embeddings, norms, RoPE tables,
//! activations, accumulation — is f32.
//!
//! ## Convention inheritance (all parity-proven in `temporal.rs`/`layers.rs`)
//!
//! - **Interleaved RoPE** → the per-head de-interleave permutation is applied
//!   to the q/k weight ROWS at load, *before* quantization; the kernels then
//!   run plain split-half rotation. Same trick, same order as the CPU path.
//! - **Norm alphas ride as f32 arrays into the norm kernels** (weighted rms)
//!   instead of being folded into the following matmul's columns like the CPU
//!   parity path does. Measured (`personaplex_rt_probe quantcheck`): the fold
//!   is NOT hostile to q4 (raw 8.7e-2 vs folded 8.8e-2 per-matvec rel RMS —
//!   same class, even with layer 0's 8×10⁴ in-group alpha range); explicit
//!   alphas are simply the cleaner shape for quantized weights (weights
//!   quantize as shipped, no host fold pass) and cost zero extra dispatches
//!   (every norm fuses into `add_rms_kernel`).
//! - **1/√d** folded into q inside the RoPE kernel.
//!
//! ## The logit head: f16 (measured decision)
//!
//! The 32000×4096 `text_linear` head is kept in **f16**, not q4. Measured on
//! the f16-stack golden run — which isolates the head from stack noise: the
//! q4 head ALONE drops text argmax from 113/113 to 108/113 (95.6%, first
//! flip at step 61) and logits cos to min 0.980 / mean 0.9947, to save
//! ~190 MB/step (≈0.6 ms at ~300 GB/s). The text stream teacher-forces the
//! depformer's prev-token chain, so head fidelity is worth 0.6 ms of the
//! 80 ms budget. Both variants stay loaded ([`Head`]) and the probe
//! re-measures the A/B on every gate run.
//!
//! ## KV cache: static slots, no cat
//!
//! Per layer, K and V are preallocated at the full 3000-frame context and
//! written in place at the stream offset:
//! - `kcache` is **dim-major** `[4096, 3000]` f16 — during score computation
//!   the 128 threads of a head-cube sweep positions at a fixed dim, reading
//!   consecutive f16s (coalesced);
//! - `vcache` is **position-major** `[3000, 4096]` f16 — during weighted-V
//!   the threads sweep dims at a fixed position (coalesced).
//!
//! Ring/wrap semantics are NOT implemented: [`TemporalMetal::step_submit`]
//! asserts `len < 3000` (a session under 4 min never gets there). The
//! eventual ring port must also apply moshi's effective-2999 window (the
//! `RingKVCache.complete()` wrap quirk — see depth.rs notes).
//!
//! ## Dispatch shape
//!
//! 9 dispatches/layer — the three attention projections run as ONE fused
//! `[3·4096, 4096]` q‖k‖v matvec (the same rows moshi's `in_proj_weight`
//! ships; quantization groups run along the input dim, so the fusion is
//! bit-identical to the split launches), gate+up+SwiGLU run as ONE fused
//! `[2·11264, 4096]` matvec with interleaved rows and an in-kernel silu·u
//! epilogue, and each residual add fuses with the CONSUMING weighted rms
//! (`add_rms_kernel`): the attention add feeds norm2, the MLP add feeds
//! the next layer's norm1, and the last layer's MLP add feeds `out_norm`
//! directly; attention is two dispatches (split-K partial + combine).
//! Per layer: qkv + rope/cache + attn partial + combine + o + add_rms +
//! gateup-swiglu + down + add_rms = 9 — ×32 layers + the initial norm +
//! head = **290 dispatches/step** (was 418 with split projections and a
//! standalone swiglu), one blocking readback for hidden + logits together.

use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

use crate::nn::q4::Rt;

use super::config as cfg;
use crate::nn::q4::{self, f16_matvec, quantize_q4, Q4Linear};
use crate::nn::weight_loader::{HostF32, WeightLoader};

const DIM: usize = cfg::DIM; // 4096
const HEADS: u32 = cfg::NUM_HEADS as u32; // 32
const HEAD_DIM: u32 = cfg::HEAD_DIM as u32; // 128
const HALF: u32 = HEAD_DIM / 2; // 64
const FFN: usize = cfg::FFN_HIDDEN; // 11264
const EPS: f32 = 1e-8; // cfg::RMS_EPS — moshi's unusual rms eps

/// Static KV capacity = the model's context. A hard stop, not a ring (see
/// module docs).
pub const MAX_SEQ: usize = cfg::CONTEXT; // 3000

/// Split-K attention chunks per head (16 × 32 heads = 512 partial cubes).
const N_CHUNKS: u32 = 16;
/// Shared-memory cap for one chunk's scores: ceil(MAX_SEQ / N_CHUNKS) = 188.
const CHUNK_CAP: u32 = 192;

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

/// Residual add fused with the NEXT weighted RMS: `x += delta;
/// y = x · alpha · rsqrt(mean(x²) + eps)` — single cube (each thread owns
/// its element subset for both the add and the reduction, so no cross-thread
/// hazard). `alpha` is the CONSUMING norm's weight: `norm2` before the MLP,
/// the next layer's `norm1` after it, `out_norm` after the last layer (the
/// final residual add and `out_norm` fuse into one dispatch).
#[cube(launch_unchecked)]
fn add_rms_kernel(
    x: &mut Array<f32>,
    delta: &Array<f32>,
    alpha: &Array<f32>,
    y: &mut Array<f32>,
    eps: f32,
    #[comptime] hidden: u32,
    #[comptime] cube_dim: u32,
) {
    let i = UNIT_POS_X;
    let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    let mut acc = f32::new(0.0);
    let mut k = i;
    while k < hidden {
        let v = x[k as usize] + delta[k as usize];
        x[k as usize] = v;
        acc += v * v;
        k += cube_dim;
    }
    red[i as usize] = acc;
    sync_cube();
    let mut stride = u32::new((cube_dim / 2) as i64);
    while stride > 0 {
        if i < stride {
            red[i as usize] = red[i as usize] + red[(i + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let s = 1.0 / (red[0] / (hidden as f32) + eps).sqrt();
    let mut k = i;
    while k < hidden {
        y[k as usize] = x[k as usize] * s * alpha[k as usize];
        k += cube_dim;
    }
}

/// Weighted RMSNorm: `y = x · alpha · rsqrt(mean(x²) + eps)` — the step's
/// initial norm (layer 0's `norm1`); every later norm fuses into
/// [`add_rms_kernel`].
#[cube(launch_unchecked)]
fn weighted_rms_kernel(
    x: &Array<f32>,
    alpha: &Array<f32>,
    y: &mut Array<f32>,
    eps: f32,
    #[comptime] hidden: u32,
    #[comptime] cube_dim: u32,
) {
    let i = UNIT_POS_X;
    let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    let mut acc = f32::new(0.0);
    let mut k = i;
    while k < hidden {
        let v = x[k as usize];
        acc += v * v;
        k += cube_dim;
    }
    red[i as usize] = acc;
    sync_cube();
    let mut stride = u32::new((cube_dim / 2) as i64);
    while stride > 0 {
        if i < stride {
            red[i as usize] = red[i as usize] + red[(i + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let s = 1.0 / (red[0] / (hidden as f32) + eps).sqrt();
    let mut k = i;
    while k < hidden {
        y[k as usize] = x[k as usize] * s * alpha[k as usize];
        k += cube_dim;
    }
}

/// Split-half RoPE on q (in place, 1/√d folded) and k (rotated into the
/// static `kcache` slot at `pos`, dim-major, f16), plus the V copy into
/// `vcache` (position-major, f16). One dispatch, `hidden` threads, reading
/// the FUSED qkv matvec output (`qkv[0..hidden]` = q, `[hidden..2·hidden]` =
/// k, `[2·hidden..3·hidden]` = v — the single-buffer sibling of the old
/// three-buffer signature; same math, one binding).
///
/// Thread `t < hidden/2` owns q rotation pair `t` (head `t/half`, offset
/// `t%half`); thread `t >= hidden/2` owns k pair `t - hidden/2`. Every thread
/// additionally copies v element `t`. In-place q is race-free: each pair's
/// two elements are read and written by exactly one thread.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn rope_cache_kernel(
    qkv: &mut Array<f32>,
    cos: &Array<f32>,
    sin: &Array<f32>,
    kcache: &mut Array<f16>,
    vcache: &mut Array<f16>,
    pos: u32,
    q_scale: f32,
    #[comptime] hidden: u32,
    #[comptime] half: u32,
    #[comptime] max_seq: u32,
) {
    let t = ABSOLUTE_POS as u32;
    let d = 2 * half;
    let pairs = hidden / 2;

    // v copy (position-major)
    vcache[(pos * hidden + t) as usize] = f16::cast_from(qkv[(2 * hidden + t) as usize]);

    let pair = t % pairs;
    let head = pair / half;
    let j = pair % half;
    let re_i = head * d + j;
    let im_i = head * d + half + j;
    let c = cos[(pos * half + j) as usize];
    let s = sin[(pos * half + j) as usize];

    if t < pairs {
        // q: rotate in place, fold 1/√d
        let re = qkv[re_i as usize];
        let im = qkv[im_i as usize];
        qkv[re_i as usize] = (re * c - im * s) * q_scale;
        qkv[im_i as usize] = (im * c + re * s) * q_scale;
    } else {
        // k: rotate into the dim-major cache slot
        let re = qkv[(hidden + re_i) as usize];
        let im = qkv[(hidden + im_i) as usize];
        kcache[(re_i * max_seq + pos) as usize] = f16::cast_from(re * c - im * s);
        kcache[(im_i * max_seq + pos) as usize] = f16::cast_from(im * c + re * s);
    }
}

/// Split-K decode attention, pass 1 of 2 (flash-decoding shape): cube
/// `(head h, chunk c)` owns positions `[c·chunk_len, min((c+1)·chunk_len,
/// len))` and produces an UNNORMALIZED partial — local max `m`, local
/// exp-sum `s`, local weighted-V `o[d]` (all relative to `m`). One cube per
/// head (the first build) left 31/40 GPU cores idle and latency-bound the
/// serial V sweep: measured ~51 GB/s effective on the KV read, +61 ms/step
/// at fill 3000. 16 chunks × 32 heads = 512 cubes restores occupancy.
///
/// `len` counts the valid prefix INCLUDING the current position; 1/√d is
/// already folded into q. Empty chunks (possible when `len < 16·1`) emit
/// `m = -3.4e38, s = 0, o = 0`, which the combine pass weights to zero.
#[cube(launch_unchecked)]
fn attn_partial_kernel(
    q: &Array<f32>,
    kcache: &Array<f16>,
    vcache: &Array<f16>,
    part_m: &mut Array<f32>,
    part_s: &mut Array<f32>,
    part_o: &mut Array<f32>,
    len: u32,
    #[comptime] d: u32,
    #[comptime] hidden: u32,
    #[comptime] max_seq: u32,
    #[comptime] n_chunks: u32,
    #[comptime] chunk_cap: u32,
) {
    let i = UNIT_POS_X;
    let h = CUBE_POS_X / n_chunks;
    let c = CUBE_POS_X % n_chunks;
    let chunk_len = (len + n_chunks - 1) / n_chunks;
    let start = c * chunk_len;
    let mut end = start + chunk_len;
    if end > len {
        end = len;
    }

    let mut qsh = SharedMemory::<f32>::new(comptime!(d as usize));
    let mut scores = SharedMemory::<f32>::new(comptime!(chunk_cap as usize));

    qsh[i as usize] = q[(h * d + i) as usize];
    sync_cube();

    // scores: thread i sweeps chunk positions start+i, start+i+d, … —
    // dim-major K means the cube's threads read consecutive positions at
    // each dd (coalesced). The dd chain is hand-unrolled 4× with the loads
    // hoisted ahead of the FMAs (4 loads in flight per thread instead of a
    // serial load→add chain); the accumulation ORDER is unchanged (dd
    // ascending, single accumulator), so scores are bit-identical.
    let mut t = start + i;
    while t < end {
        let mut s = f32::new(0.0);
        let mut dd = u32::new(0);
        while dd < d {
            let kb = (h * d + dd) * max_seq + t;
            let k0 = f32::cast_from(kcache[kb as usize]);
            let k1 = f32::cast_from(kcache[(kb + max_seq) as usize]);
            let k2 = f32::cast_from(kcache[(kb + 2 * max_seq) as usize]);
            let k3 = f32::cast_from(kcache[(kb + 3 * max_seq) as usize]);
            s += qsh[dd as usize] * k0;
            s += qsh[(dd + 1) as usize] * k1;
            s += qsh[(dd + 2) as usize] * k2;
            s += qsh[(dd + 3) as usize] * k3;
            dd += 4;
        }
        scores[(t - start) as usize] = s;
        t += d;
    }
    sync_cube();

    // local max
    let mut m = f32::new(-3.40282e38);
    let mut t = start + i;
    while t < end {
        let sv = scores[(t - start) as usize];
        if sv > m {
            m = sv;
        }
        t += d;
    }
    qsh[i as usize] = m;
    sync_cube();
    let mut stride = u32::new((d / 2) as i64);
    while stride > 0 {
        if i < stride {
            let other = qsh[(i + stride) as usize];
            if other > qsh[i as usize] {
                qsh[i as usize] = other;
            }
        }
        sync_cube();
        stride /= 2;
    }
    let mx = qsh[0];
    sync_cube(); // qsh is reused for the sum

    // exp + local sum (relative to the LOCAL max — combine rescales)
    let mut sum = f32::new(0.0);
    let mut t = start + i;
    while t < end {
        let p = (scores[(t - start) as usize] - mx).exp();
        scores[(t - start) as usize] = p;
        sum += p;
        t += d;
    }
    qsh[i as usize] = sum;
    sync_cube();
    let mut stride = u32::new((d / 2) as i64);
    while stride > 0 {
        if i < stride {
            qsh[i as usize] = qsh[i as usize] + qsh[(i + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let total = qsh[0];

    // unnormalized weighted V — thread i owns output dim i; position-major V
    // means the cube's threads read consecutive dims at each t (coalesced).
    // 4× unrolled with hoisted loads, then a strict-order remainder; the
    // per-dim accumulation ORDER over t is unchanged (bit-identical).
    let mut acc = f32::new(0.0);
    let mut t = start;
    while t + 4 <= end {
        let vb = t * hidden + h * d + i;
        let v0 = f32::cast_from(vcache[vb as usize]);
        let v1 = f32::cast_from(vcache[(vb + hidden) as usize]);
        let v2 = f32::cast_from(vcache[(vb + 2 * hidden) as usize]);
        let v3 = f32::cast_from(vcache[(vb + 3 * hidden) as usize]);
        acc += scores[(t - start) as usize] * v0;
        acc += scores[(t - start + 1) as usize] * v1;
        acc += scores[(t - start + 2) as usize] * v2;
        acc += scores[(t - start + 3) as usize] * v3;
        t += 4;
    }
    while t < end {
        acc += scores[(t - start) as usize]
            * f32::cast_from(vcache[(t * hidden + h * d + i) as usize]);
        t += 1;
    }
    let pc = h * n_chunks + c;
    part_o[(pc * d + i) as usize] = acc;
    if i == 0 {
        part_m[pc as usize] = mx;
        part_s[pc as usize] = total;
    }
}

/// Split-K decode attention, pass 2: one cube per head combines its
/// `n_chunks` partials under the global max — `out[d] = Σ_c e^{m_c−M}·o_c[d]
/// / Σ_c e^{m_c−M}·s_c`. Every thread redundantly folds the 16 scalars
/// (cheap); thread i owns output dim i.
#[cube(launch_unchecked)]
fn attn_combine_kernel(
    part_m: &Array<f32>,
    part_s: &Array<f32>,
    part_o: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] d: u32,
    #[comptime] n_chunks: u32,
) {
    let i = UNIT_POS_X;
    let h = CUBE_POS_X;

    let mut gm = f32::new(-3.40282e38);
    for c in 0..n_chunks {
        let mc = part_m[(h * n_chunks + c) as usize];
        if mc > gm {
            gm = mc;
        }
    }
    let mut denom = f32::new(0.0);
    let mut numer = f32::new(0.0);
    for c in 0..n_chunks {
        let pc = h * n_chunks + c;
        let w = (part_m[pc as usize] - gm).exp();
        denom += w * part_s[pc as usize];
        numer += w * part_o[(pc * d + i) as usize];
    }
    out[(h * d + i) as usize] = numer / denom;
}

/// q8_0 matvec — the same skeleton as [`crate::nn::q4::q4_matvec_kernel`]
/// (8 row-groups × 32 lanes per cube, two independent accumulator chains,
/// shared-memory tree reduce) reading BIASED byte weights: nibble words
/// become byte words (`u32` = 4 weights, stored as `q + 128` so the unpack is
/// `cast(byte) - 128` with no sign-extension), one f16 scale per 32 weights.
/// A word pair = 8 weights; groups are 8 words, so pairs never straddle a
/// scale group.
/// `swiglu_pairs`: the fused gate‖up epilogue (see `q4::q4_matvec_kernel`).
#[cube(launch_unchecked)]
#[allow(clippy::manual_is_multiple_of)] // `%` is the cube-kernel primitive
fn q8_matvec_kernel(
    x: &Array<Vector<f32, Const<4>>>,
    wq: &Array<u32>,
    scales: &Array<f16>,
    y: &mut Array<f32>,
    #[comptime] in_dim: u32,
    #[comptime] rows_per_cube: u32,
    #[comptime] threads_per_row: u32,
    #[comptime] swiglu_pairs: bool,
) {
    let lane = UNIT_POS_X % threads_per_row;
    let row = CUBE_POS_X * rows_per_cube + UNIT_POS_X / threads_per_row;
    let groups = in_dim / 32;
    let words_per_row = in_dim / 4;

    // The 8 activations per iteration arrive as two vec4<f32> loads (was 8
    // scalar loads; the unpack chain is ALU-issue-bound, so load-issue width
    // is the cheap win); components are consumed in the original order, so
    // the FMA sequence — and every result bit — is unchanged.
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
    let acc = acc0 + acc1;

    let mut red = SharedMemory::<f32>::new(comptime!((rows_per_cube * threads_per_row) as usize));
    red[UNIT_POS_X as usize] = acc;
    sync_cube();
    let mut stride = u32::new((threads_per_row / 2) as i64);
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
        y[row as usize] = red[UNIT_POS_X as usize];
    }
}

// ---------------------------------------------------------------------------
// host-side load helpers
// ---------------------------------------------------------------------------

/// Per-head de-interleave permutation on the ROWS of a `[DIM, cols]` q or k
/// projection block — the interleaved→split-half RoPE conversion, identical
/// to `temporal::deinterleave_rows` (duplicated here so the parity module
/// stays untouched; the gate cross-checks it against the goldens).
fn deinterleave_rows(w: &[f32], cols: usize) -> Vec<f32> {
    let (h, d) = (cfg::NUM_HEADS, cfg::HEAD_DIM);
    let half = d / 2;
    assert_eq!(w.len(), h * d * cols);
    let mut out = vec![0f32; w.len()];
    for head in 0..h {
        for j in 0..half {
            let dst_re = (head * d + j) * cols;
            let src_re = (head * d + 2 * j) * cols;
            out[dst_re..dst_re + cols].copy_from_slice(&w[src_re..src_re + cols]);
            let dst_im = (head * d + half + j) * cols;
            let src_im = (head * d + 2 * j + 1) * cols;
            out[dst_im..dst_im + cols].copy_from_slice(&w[src_im..src_im + cols]);
        }
    }
    out
}

/// Squeeze a `[1, 1, D]` moshi norm alpha into `[D]`.
pub(crate) fn load_alpha(loader: &WeightLoader, name: &str) -> Vec<f32> {
    let (a, s) = loader.load_f32(name);
    assert_eq!(s, vec![1, 1, DIM], "{name} shape");
    a
}

/// The 4 per-layer matvec weight names, in launch order — the leaf-name
/// suffixes of the derived sibling pile (`t.{layer}.{name}`).
pub(crate) const MAT_NAMES: [&str; 4] = ["qkv", "o", "gateup", "down"];

/// Fetch temporal layer `i`'s 4 runtime matvec weights as row-major f32 in
/// the FUSED kernel layouts, in [`MAT_NAMES`] order with their `[out, in]`
/// dims:
///
/// - `qkv` `[3·DIM, DIM]`: q‖k‖v rows concatenated (the moshi
///   `in_proj_weight` block order), q/k rows de-interleaved (RoPE
///   convention) — one dispatch feeds the fused qkv matvec.
/// - `gateup` `[2·FFN, DIM]`: gate/up rows pair-INTERLEAVED (even = gate_j,
///   odd = up_j) so each 8-row cube owns whole pairs — enables the
///   in-kernel SwiGLU epilogue.
///
/// Row concatenation/permutation is quantization-transparent (groups run
/// along the input dim of each row). SHARED by the quantize-at-load path
/// ([`TemporalMetal::load`]) and the sibling derive
/// (`qpile::derive_temporal_pile`) — one code path is what guarantees the
/// persisted packed bytes are IDENTICAL to the ones quantized at load.
pub(crate) fn layer_mats_f32(
    loader: &WeightLoader,
    i: usize,
) -> Vec<(&'static str, Vec<f32>, usize, usize)> {
    let (d, fh) = (DIM, FFN);
    let src = format!("transformer.layers.{i}");

    let (in_proj, s) = loader.load_f32(&format!("{src}.self_attn.in_proj_weight"));
    assert_eq!(s, vec![3 * d, d], "{src}: in_proj_weight shape");
    let (o_w, s) = loader.load_f32(&format!("{src}.self_attn.out_proj.weight"));
    assert_eq!(s, vec![d, d], "{src}: out_proj shape");
    let (gu, s) = loader.load_f32(&format!("{src}.gating.linear_in.weight"));
    assert_eq!(
        s,
        vec![cfg::FFN_FUSED_IN, d],
        "{src}: gating.linear_in shape"
    );
    let (down_w, s) = loader.load_f32(&format!("{src}.gating.linear_out.weight"));
    assert_eq!(s, vec![d, fh], "{src}: gating.linear_out shape");

    let d2 = d * d;
    // Fused q‖k‖v: de-interleave q/k rows (RoPE convention) in place of the
    // shipped in_proj row blocks. Norm alphas stay explicit (module docs).
    let mut qkv_w = deinterleave_rows(&in_proj[..d2], d);
    qkv_w.extend_from_slice(&deinterleave_rows(&in_proj[d2..2 * d2], d));
    qkv_w.extend_from_slice(&in_proj[2 * d2..]);
    drop(in_proj);
    // Fused gate/up with INTERLEAVED rows (even = gate_j, odd = up_j).
    let mut gateup_w = vec![0f32; 2 * fh * d];
    for j in 0..fh {
        gateup_w[2 * j * d..(2 * j + 1) * d].copy_from_slice(&gu[j * d..(j + 1) * d]);
        gateup_w[(2 * j + 1) * d..(2 * j + 2) * d]
            .copy_from_slice(&gu[(fh + j) * d..(fh + j + 1) * d]);
    }
    drop(gu);

    vec![
        ("qkv", qkv_w, 3 * d, d),
        ("o", o_w, d, d),
        ("gateup", gateup_w, 2 * fh, d),
        ("down", down_w, d, fh),
    ]
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Quantize a row-major `[out, in]` f32 weight to q8_0 (biased bytes + f16
/// scales): per 32-weight group `d = max|w| / 127` (f16-rounded before
/// quantizing), byte = `clamp(round(w/d), -127, 127) + 128` packed 4/word.
pub fn quantize_q8(w: &[f32], out_dim: usize, in_dim: usize) -> (Vec<u32>, Vec<f16>) {
    const GROUP: usize = 32;
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(in_dim % GROUP, 0);
    let words_per_row = in_dim / 4;
    let groups_per_row = in_dim / GROUP;
    let mut wq = vec![0u32; out_dim * words_per_row];
    let mut scales = vec![f16::ZERO; out_dim * groups_per_row];
    for j in 0..out_dim {
        for g in 0..groups_per_row {
            let base = j * in_dim + g * GROUP;
            let grp = &w[base..base + GROUP];
            let amax = grp.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let ds = f16::from_f32(amax / 127.0);
            let d = ds.to_f32();
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            scales[j * groups_per_row + g] = ds;
            for (i, &v) in grp.iter().enumerate() {
                let q = ((v * id).round() as i32).clamp(-127, 127) + 128;
                let k = g * GROUP + i;
                wq[j * words_per_row + k / 4] |= (q as u32) << (8 * (k % 4));
            }
        }
    }
    (wq, scales)
}

/// The weight format the whole stack loads in — the fidelity/bandwidth axis
/// (see module docs). ~3.7 / 7.0 / 13.2 GB of per-step weight traffic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightFmt {
    Q4,
    Q8,
    F16,
}

impl WeightFmt {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "q4" => Some(Self::Q4),
            "q8" => Some(Self::Q8),
            "f16" => Some(Self::F16),
            _ => None,
        }
    }
}

/// One matvec weight in the chosen format, dispatching to the matching
/// kernel (q4/f16 from `nn::q4`, q8 above).
enum QLinear {
    Q4(Q4Linear),
    Q8 {
        wq: Handle,
        scales: Handle,
        out_dim: usize,
        in_dim: usize,
    },
    F16 {
        w: Handle,
        out_dim: usize,
        in_dim: usize,
    },
}

/// Host-encoded weight payload (produced on worker threads, uploaded on the
/// main thread).
pub(crate) enum Encoded {
    Packed(Vec<u32>, Vec<f16>),
    Half(Vec<f16>),
}

fn encode(w: &[f32], out_dim: usize, in_dim: usize, fmt: WeightFmt) -> Encoded {
    match fmt {
        WeightFmt::Q4 => {
            let (wq, sc) = quantize_q4(w, out_dim, in_dim);
            Encoded::Packed(wq, sc)
        }
        WeightFmt::Q8 => {
            let (wq, sc) = quantize_q8(w, out_dim, in_dim);
            Encoded::Packed(wq, sc)
        }
        WeightFmt::F16 => Encoded::Half(w.iter().map(|&x| f16::from_f32(x)).collect()),
    }
}

impl QLinear {
    fn upload(
        client: &q4::Client,
        enc: &Encoded,
        out_dim: usize,
        in_dim: usize,
        fmt: WeightFmt,
    ) -> Self {
        match (fmt, enc) {
            (WeightFmt::Q4, Encoded::Packed(wq, sc)) => {
                Self::Q4(Q4Linear::from_packed(client, wq, sc, out_dim, in_dim))
            }
            (WeightFmt::Q8, Encoded::Packed(wq, sc)) => Self::Q8 {
                wq: client.create_from_slice(as_bytes(wq)),
                scales: client.create_from_slice(as_bytes(sc)),
                out_dim,
                in_dim,
            },
            (WeightFmt::F16, Encoded::Half(h)) => Self::F16 {
                w: client.create_from_slice(as_bytes(h)),
                out_dim,
                in_dim,
            },
            _ => unreachable!("encode/upload format mismatch"),
        }
    }

    fn forward(&self, client: &q4::Client, x: &Handle, y: &Handle) {
        self.launch(client, x, y, false);
    }

    /// Fused gate‖up + SwiGLU (interleaved rows, `y` = `out_dim/2` f32) —
    /// one dispatch replaces gate + up + swiglu (same arithmetic).
    fn forward_swiglu(&self, client: &q4::Client, x: &Handle, y: &Handle) {
        self.launch(client, x, y, true);
    }

    fn launch(&self, client: &q4::Client, x: &Handle, y: &Handle, swiglu_pairs: bool) {
        match self {
            Self::Q4(l) => {
                if swiglu_pairs {
                    l.forward_swiglu(client, x, y)
                } else {
                    l.forward(client, x, y)
                }
            }
            Self::Q8 {
                wq,
                scales,
                out_dim,
                in_dim,
            } => {
                assert_eq!(*out_dim % 8, 0);
                assert_eq!(*in_dim % 32, 0);
                let y_len = if swiglu_pairs { *out_dim / 2 } else { *out_dim };
                unsafe {
                    q8_matvec_kernel::launch_unchecked::<Rt>(
                        client,
                        CubeCount::new_1d(*out_dim as u32 / 8),
                        CubeDim::new_1d(8 * 32),
                        ArrayArg::from_raw_parts(x.clone(), *in_dim / 4),
                        ArrayArg::from_raw_parts(wq.clone(), *out_dim * *in_dim / 4),
                        ArrayArg::from_raw_parts(scales.clone(), *out_dim * *in_dim / 32),
                        ArrayArg::from_raw_parts(y.clone(), y_len),
                        *in_dim as u32,
                        8,
                        32,
                        swiglu_pairs,
                    );
                }
            }
            Self::F16 { w, out_dim, in_dim } => {
                if swiglu_pairs {
                    q4::f16_matvec_swiglu(client, x, w, y, *out_dim, *in_dim)
                } else {
                    f16_matvec(client, x, w, y, *out_dim, *in_dim)
                }
            }
        }
    }
}

/// Encode a batch of row-major f32 weights in parallel (the layer load is
/// pile-read-bound otherwise). Jobs split into row chunks so the wide fused
/// matrices (q‖k‖v, gate/up) don't serialize the encode — every encoding is
/// per-row, so row-major chunks concatenate bit-exactly. `pub(crate)` for
/// the sibling derive (`qpile`), which persists the SAME encoded bytes.
pub(crate) fn encode_batch(jobs: Vec<(Vec<f32>, usize, usize)>, fmt: WeightFmt) -> Vec<Encoded> {
    const CHUNK_ROWS: usize = 4096;
    std::thread::scope(|sc| {
        let handles: Vec<Vec<_>> = jobs
            .iter()
            .map(|(w, o, i)| {
                (0..*o)
                    .step_by(CHUNK_ROWS)
                    .map(|r0| {
                        let rows = CHUNK_ROWS.min(*o - r0);
                        let slice = &w[r0 * *i..(r0 + rows) * *i];
                        sc.spawn(move || encode(slice, rows, *i, fmt))
                    })
                    .collect()
            })
            .collect();
        handles
            .into_iter()
            .map(|parts| {
                let parts: Vec<Encoded> = parts.into_iter().map(|h| h.join().unwrap()).collect();
                match fmt {
                    WeightFmt::F16 => Encoded::Half(
                        parts
                            .into_iter()
                            .flat_map(|p| match p {
                                Encoded::Half(v) => v,
                                Encoded::Packed(..) => unreachable!(),
                            })
                            .collect(),
                    ),
                    WeightFmt::Q4 | WeightFmt::Q8 => {
                        let (mut wq, mut scales) = (Vec::new(), Vec::new());
                        for p in parts {
                            match p {
                                Encoded::Packed(a, b) => {
                                    wq.extend(a);
                                    scales.extend(b);
                                }
                                Encoded::Half(_) => unreachable!(),
                            }
                        }
                        Encoded::Packed(wq, scales)
                    }
                }
            })
            .collect()
    })
}

// ---------------------------------------------------------------------------
// the engine
// ---------------------------------------------------------------------------

/// Which logit head to launch. F16 is the production choice (see module
/// docs); Q4 stays loaded for the probe's A/B.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Head {
    F16,
    Q4,
}

struct MetalLayer {
    /// Fused `[3·DIM, DIM]` q‖k‖v (rows concatenated exactly as the moshi
    /// `in_proj_weight` ships, q/k de-interleaved) — one dispatch.
    qkv: QLinear,
    o: QLinear,
    /// Fused `[2·FFN, DIM]` gate/up with INTERLEAVED rows (even = gate_j,
    /// odd = up_j) so each 8-row cube owns whole pairs and the SwiGLU
    /// epilogue runs in-kernel — one dispatch replaces gate + up + swiglu.
    gateup: QLinear,
    down: QLinear,
    norm1: Handle,  // [DIM] f32 — alphas applied in the norm kernels, NOT
    norm2: Handle,  // folded into the quantized weights (see module docs)
    kcache: Handle, // [DIM, MAX_SEQ] f16, dim-major
    vcache: Handle, // [MAX_SEQ, DIM] f16, position-major
}

/// The 7B temporal transformer on the raw wgpu/Metal device: quantized
/// weights ([`WeightFmt`]), static-slot f16 KV caches, host-side f32
/// embeddings, one readback per step. See module docs for the numerics
/// contract.
pub struct TemporalMetal {
    client: q4::Client,
    layers: Vec<MetalLayer>,
    cos: Handle, // [MAX_SEQ, 64] f32 half-tables
    sin: Handle,
    out_norm_w: Handle, // [DIM] f32
    head_f16: Handle,   // [32000, 4096] f16 rows
    head_q4: Q4Linear,
    /// `text_emb [32001, 4096]` host f32 (row 32000 = text BOS-of-stream).
    /// Owned by the quantize-at-load path; a zero-copy mmap view of the
    /// exact-f32 pile leaf on the sibling path (same bytes either way).
    text_emb: HostF32,
    /// `emb.{0..15} [2049, 4096]` host f32 (row 2048 = audio initial token).
    audio_emb: Vec<HostF32>,
    // activation buffers (f32); the residual-stream buffer `x` itself is
    // created per step from the uploaded input embedding
    xn: Handle,
    /// Fused q‖k‖v matvec output `[3·DIM]` (q rotated in place by the RoPE
    /// kernel; the attention kernels read the q third).
    qkvb: Handle,
    attn: Handle,
    // split-K attention partials: [32 heads × 16 chunks] m/s + ×128 o
    part_m: Handle,
    part_s: Handle,
    part_o: Handle,
    delta: Handle, // o_proj / down_proj output feeding the residual add
    act: Handle,
    hidden: Handle, // post-out_norm [DIM] — transformer_out
    logits: Handle, // [32000]
    len: usize,
}

impl TemporalMetal {
    /// Load + encode the temporal side from the shelf pile (same tensor
    /// names/splits as `temporal.rs`) in the chosen [`WeightFmt`], upload to
    /// the default wgpu/Metal device. Weights ~3.7 GB (q4) / 7.0 GB (q8) /
    /// 13.2 GB (f16) + 336 MB heads + 1.6 GB KV + ~1 GB host embeddings.
    pub fn load(loader: &WeightLoader, fmt: WeightFmt) -> Self {
        let client = q4::client_for_default_device();
        let d = DIM;

        let mut layers = Vec::with_capacity(cfg::NUM_LAYERS);
        for i in 0..cfg::NUM_LAYERS {
            let src = format!("transformer.layers.{i}");
            let a1 = load_alpha(loader, &format!("{src}.norm1.alpha"));
            let a2 = load_alpha(loader, &format!("{src}.norm2.alpha"));

            let mats = layer_mats_f32(loader, i);
            let shapes: Vec<(usize, usize)> = mats.iter().map(|&(_, _, o, i)| (o, i)).collect();
            let encoded = encode_batch(
                mats.into_iter().map(|(_, w, o, i)| (w, o, i)).collect(),
                fmt,
            );
            let mut it = encoded.iter().zip(shapes);
            let mut up = || {
                let (enc, (o, i)) = it.next().unwrap();
                QLinear::upload(&client, enc, o, i, fmt)
            };
            layers.push(MetalLayer {
                qkv: up(),
                o: up(),
                gateup: up(),
                down: up(),
                norm1: client.create_from_slice(as_bytes(&a1)),
                norm2: client.create_from_slice(as_bytes(&a2)),
                kcache: client.empty(d * MAX_SEQ * 2),
                vcache: client.empty(MAX_SEQ * d * 2),
            });
            eprint!(
                "\r  encoded layer {:2}/{} ({fmt:?})",
                i + 1,
                cfg::NUM_LAYERS
            );
        }
        eprintln!();

        // globals
        let out_norm = load_alpha(loader, "out_norm.alpha");
        let (head, s) = loader.load_f32("text_linear.weight");
        assert_eq!(s, vec![cfg::TEXT_LOGITS, d], "text_linear shape");
        let head_half: Vec<f16> = head.iter().map(|&x| f16::from_f32(x)).collect();
        let (hq, hs) = quantize_q4(&head, cfg::TEXT_LOGITS, d);

        let (text_emb, s) = loader.load_f32("text_emb.weight");
        assert_eq!(s, vec![cfg::TEXT_VOCAB, d], "text_emb shape");
        let text_emb = HostF32::Owned(text_emb);
        let audio_emb = (0..cfg::N_Q)
            .map(|cb| {
                let (w, s) = loader.load_f32(&format!("emb.{cb}.weight"));
                assert_eq!(s, vec![cfg::AUDIO_VOCAB, d], "emb.{cb} shape");
                HostF32::Owned(w)
            })
            .collect();

        let out_norm_w = client.create_from_slice(as_bytes(&out_norm));
        let head_f16 = client.create_from_slice(as_bytes(&head_half));
        let head_q4 = Q4Linear::from_packed(&client, &hq, &hs, cfg::TEXT_LOGITS, d);
        Self::assemble(
            client, layers, out_norm_w, head_f16, head_q4, text_emb, audio_emb,
        )
    }

    /// Shared tail of both load paths: the RoPE half-tables (computed, not
    /// loaded), the fixed activation scratch buffers, and the struct wiring.
    fn assemble(
        client: q4::Client,
        layers: Vec<MetalLayer>,
        out_norm_w: Handle,
        head_f16: Handle,
        head_q4: Q4Linear,
        text_emb: HostF32,
        audio_emb: Vec<HostF32>,
    ) -> Self {
        let (d, fh) = (DIM, FFN);
        // RoPE half-tables [MAX_SEQ, 64], split-half convention (same
        // frequencies as layers::RopeTable).
        let half = cfg::HEAD_DIM / 2;
        let mut cos = vec![0f32; MAX_SEQ * half];
        let mut sin = vec![0f32; MAX_SEQ * half];
        for p in 0..MAX_SEQ {
            for j in 0..half {
                let r = p as f64 * cfg::ROPE_THETA.powf(-2.0 * j as f64 / cfg::HEAD_DIM as f64);
                cos[p * half + j] = r.cos() as f32;
                sin[p * half + j] = r.sin() as f32;
            }
        }

        Self {
            cos: client.create_from_slice(as_bytes(&cos)),
            sin: client.create_from_slice(as_bytes(&sin)),
            out_norm_w,
            head_f16,
            head_q4,
            text_emb,
            audio_emb,
            xn: client.empty(d * 4),
            qkvb: client.empty(3 * d * 4),
            attn: client.empty(d * 4),
            part_m: client.empty((HEADS * N_CHUNKS) as usize * 4),
            part_s: client.empty((HEADS * N_CHUNKS) as usize * 4),
            part_o: client.empty((HEADS * N_CHUNKS * HEAD_DIM) as usize * 4),
            delta: client.empty(d * 4),
            act: client.empty(fh * 4),
            hidden: client.empty(d * 4),
            logits: client.empty(cfg::TEXT_LOGITS * 4),
            layers,
            client,
            len: 0,
        }
    }

    /// ZERO-COPY load from a derived sibling pile (`qpile`): every GPU weight
    /// buffer — the packed q4/q8 words + f16 scales (or raw f16 rows), the
    /// norm alphas, both logit heads — ALIASES its mmap'd pile blob via
    /// `register_external_aliased` (no read, no quantize pass, no upload
    /// copy; pages fault in on first kernel touch). The host embeddings map
    /// the canonical pile's exact-f32 leaves the same way. Byte-identical to
    /// [`Self::load`] with the same `fmt` — the sibling stores exactly the
    /// bytes the quantize-at-load pass produces (`derive_temporal_pile`
    /// shares `layer_mats_f32` + the quantizers with `load`), so the gates
    /// must agree BIT-EXACTLY; any difference is a derive bug.
    ///
    /// Errors (marker mismatch, missing leaf, non-mmap blob) are the
    /// caller's cue to fall back to [`Self::load`].
    #[cfg(target_os = "macos")]
    pub fn load_zero_copy(
        sib: &super::qpile::QPile,
        loader: &WeightLoader,
        fmt: WeightFmt,
    ) -> anyhow::Result<Self> {
        use super::qpile;
        let want = qpile::temporal_marker(fmt);
        anyhow::ensure!(
            sib.marker == Some(want),
            "format marker mismatch: sibling has {:?}, this build wants {want:?} ({fmt:?}) — \
             re-derive with personaplex_persist --derive-fmt",
            sib.marker
        );
        let client = q4::client_for_default_device();
        let alias = |b: &anybytes::Bytes| -> anyhow::Result<Handle> {
            q4::alias_pile_blob(&client, b)
                .ok_or_else(|| anyhow::anyhow!("pile blob not mmap-backed — cannot alias"))
        };

        let d = DIM;
        let mut layers = Vec::with_capacity(cfg::NUM_LAYERS);
        for i in 0..cfg::NUM_LAYERS {
            let mut mats = Vec::with_capacity(MAT_NAMES.len());
            for name in MAT_NAMES {
                let key = format!("t.{i}.{name}");
                let q = match fmt {
                    WeightFmt::Q4 => {
                        let (wq, sc, shape) = sib.bytes_q4(&key)?;
                        QLinear::Q4(Q4Linear {
                            wq: alias(&wq)?,
                            scales: alias(&sc)?,
                            out_dim: shape[0],
                            in_dim: shape[1],
                        })
                    }
                    WeightFmt::Q8 => {
                        let (wq, sc, shape) = sib.bytes_q8(&key)?;
                        QLinear::Q8 {
                            wq: alias(&wq)?,
                            scales: alias(&sc)?,
                            out_dim: shape[0],
                            in_dim: shape[1],
                        }
                    }
                    WeightFmt::F16 => {
                        let (w, shape) = sib.bytes_f16(&key)?;
                        anyhow::ensure!(
                            w.len() == shape[0] * shape[1] * 2,
                            "{key}: f16 byte count vs shape"
                        );
                        QLinear::F16 {
                            w: alias(&w)?,
                            out_dim: shape[0],
                            in_dim: shape[1],
                        }
                    }
                };
                mats.push(q);
            }
            let (a1, s) = sib.bytes_f32(&format!("t.{i}.norm1"))?;
            anyhow::ensure!(s == vec![d], "t.{i}.norm1 shape");
            let (a2, s) = sib.bytes_f32(&format!("t.{i}.norm2"))?;
            anyhow::ensure!(s == vec![d], "t.{i}.norm2 shape");
            let mut it = mats.into_iter();
            let mut next = || it.next().unwrap();
            layers.push(MetalLayer {
                qkv: next(),
                o: next(),
                gateup: next(),
                down: next(),
                norm1: alias(&a1)?,
                norm2: alias(&a2)?,
                kcache: client.empty(d * MAX_SEQ * 2),
                vcache: client.empty(MAX_SEQ * d * 2),
            });
        }

        let (onw, s) = sib.bytes_f32("t.out_norm")?;
        anyhow::ensure!(s == vec![d], "t.out_norm shape");
        let out_norm_w = alias(&onw)?;
        let (hf, s) = sib.bytes_f16("t.head_f16")?;
        anyhow::ensure!(s == vec![cfg::TEXT_LOGITS, d], "t.head_f16 shape");
        let head_f16 = alias(&hf)?;
        let (hwq, hsc, s) = sib.bytes_q4("t.head_q4")?;
        anyhow::ensure!(s == vec![cfg::TEXT_LOGITS, d], "t.head_q4 shape");
        let head_q4 = Q4Linear {
            wq: alias(&hwq)?,
            scales: alias(&hsc)?,
            out_dim: cfg::TEXT_LOGITS,
            in_dim: d,
        };

        // Host embeddings: zero-copy mmap views of the CANONICAL pile's
        // exact-f32 leaves (they are consumed row-wise on the CPU as-is).
        let (text_emb, s) = loader.load_host_f32("text_emb.weight");
        assert_eq!(s, vec![cfg::TEXT_VOCAB, d], "text_emb shape");
        let audio_emb = (0..cfg::N_Q)
            .map(|cb| {
                let (w, s) = loader.load_host_f32(&format!("emb.{cb}.weight"));
                assert_eq!(s, vec![cfg::AUDIO_VOCAB, d], "emb.{cb} shape");
                w
            })
            .collect();

        Ok(Self::assemble(
            client, layers, out_norm_w, head_f16, head_q4, text_emb, audio_emb,
        ))
    }

    /// moshi `embed_codes` on the host: one step's 17 tokens (delays already
    /// applied) → the summed f32 input embedding `[4096]`. Sum order mirrors
    /// the oracle (audio codebooks 0..15 first, text LAST); token `-1`
    /// contributes zero.
    pub fn embed_codes(&self, tokens: &[i64]) -> Vec<f32> {
        assert_eq!(tokens.len(), cfg::NUM_STREAMS, "expected 17 stream tokens");
        let mut acc = vec![0f32; DIM];
        for cb in 0..cfg::N_Q {
            let t = tokens[1 + cb];
            if t >= 0 {
                assert!(
                    (t as usize) < cfg::AUDIO_VOCAB,
                    "audio token {t} out of range"
                );
                let row = &self.audio_emb[cb][t as usize * DIM..(t as usize + 1) * DIM];
                for (a, &r) in acc.iter_mut().zip(row) {
                    *a += r;
                }
            }
        }
        let t = tokens[0];
        if t >= 0 {
            assert!(
                (t as usize) < cfg::TEXT_VOCAB,
                "text token {t} out of range"
            );
            let row = &self.text_emb[t as usize * DIM..(t as usize + 1) * DIM];
            for (a, &r) in acc.iter_mut().zip(row) {
                *a += r;
            }
        }
        acc
    }

    /// Encode + submit one decode step (290 dispatches), no readback. The
    /// input is one pre-summed embedding `[4096]` (from [`Self::embed_codes`]
    /// or a voice-prompt row).
    pub fn step_submit(&mut self, x_host: &[f32], head: Head) {
        assert_eq!(x_host.len(), DIM);
        assert!(
            self.len < MAX_SEQ,
            "static KV cache full ({MAX_SEQ} frames = 4 min) — ring wrap not implemented"
        );
        let pos = self.len as u32;
        let d = DIM as u32;
        let c = &self.client;
        let x = c.create_from_slice(as_bytes(x_host));
        let arr = |h: &Handle, n: usize| unsafe { ArrayArg::from_raw_parts(h.clone(), n) };

        unsafe {
            weighted_rms_kernel::launch_unchecked::<Rt>(
                c,
                CubeCount::new_single(),
                CubeDim::new_1d(256),
                arr(&x, DIM),
                arr(&self.layers[0].norm1, DIM),
                arr(&self.xn, DIM),
                EPS,
                d,
                256,
            );
        }
        for i in 0..self.layers.len() {
            let l = &self.layers[i];
            l.qkv.forward(c, &self.xn, &self.qkvb);
            unsafe {
                rope_cache_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_1d(d / 256),
                    CubeDim::new_1d(256),
                    arr(&self.qkvb, 3 * DIM),
                    arr(&self.cos, MAX_SEQ * HALF as usize),
                    arr(&self.sin, MAX_SEQ * HALF as usize),
                    arr(&l.kcache, DIM * MAX_SEQ),
                    arr(&l.vcache, MAX_SEQ * DIM),
                    pos,
                    (cfg::HEAD_DIM as f32).powf(-0.5),
                    d,
                    HALF,
                    MAX_SEQ as u32,
                );
                attn_partial_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_1d(HEADS * N_CHUNKS),
                    CubeDim::new_1d(HEAD_DIM),
                    arr(&self.qkvb, DIM),
                    arr(&l.kcache, DIM * MAX_SEQ),
                    arr(&l.vcache, MAX_SEQ * DIM),
                    arr(&self.part_m, (HEADS * N_CHUNKS) as usize),
                    arr(&self.part_s, (HEADS * N_CHUNKS) as usize),
                    arr(&self.part_o, (HEADS * N_CHUNKS * HEAD_DIM) as usize),
                    pos + 1,
                    HEAD_DIM,
                    d,
                    MAX_SEQ as u32,
                    N_CHUNKS,
                    CHUNK_CAP,
                );
                attn_combine_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_1d(HEADS),
                    CubeDim::new_1d(HEAD_DIM),
                    arr(&self.part_m, (HEADS * N_CHUNKS) as usize),
                    arr(&self.part_s, (HEADS * N_CHUNKS) as usize),
                    arr(&self.part_o, (HEADS * N_CHUNKS * HEAD_DIM) as usize),
                    arr(&self.attn, DIM),
                    HEAD_DIM,
                    N_CHUNKS,
                );
            }
            l.o.forward(c, &self.attn, &self.delta);
            unsafe {
                add_rms_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_single(),
                    CubeDim::new_1d(256),
                    arr(&x, DIM),
                    arr(&self.delta, DIM),
                    arr(&l.norm2, DIM),
                    arr(&self.xn, DIM),
                    EPS,
                    d,
                    256,
                );
            }
            l.gateup.forward_swiglu(c, &self.xn, &self.act);
            l.down.forward(c, &self.act, &self.delta);
            // the trailing residual add fuses with the CONSUMING norm: the
            // next layer's norm1 (into xn), or out_norm (into hidden) after
            // the last layer.
            let (alpha, y) = match self.layers.get(i + 1) {
                Some(next) => (&next.norm1, &self.xn),
                None => (&self.out_norm_w, &self.hidden),
            };
            unsafe {
                add_rms_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_single(),
                    CubeDim::new_1d(256),
                    arr(&x, DIM),
                    arr(&self.delta, DIM),
                    arr(alpha, DIM),
                    arr(y, DIM),
                    EPS,
                    d,
                    256,
                );
            }
        }
        self.submit_head(head);
        self.len += 1;
    }

    /// Launch the logit head over the CURRENT `hidden` buffer (the probe uses
    /// this to A/B both heads against one step's hidden state).
    pub fn submit_head(&self, head: Head) {
        match head {
            Head::F16 => f16_matvec(
                &self.client,
                &self.hidden,
                &self.head_f16,
                &self.logits,
                cfg::TEXT_LOGITS,
                DIM,
            ),
            Head::Q4 => self
                .head_q4
                .forward(&self.client, &self.hidden, &self.logits),
        }
    }

    /// Blocking readback of `transformer_out` `[4096]` (the depformer's
    /// conditioning input).
    pub fn read_hidden(&self) -> Vec<f32> {
        read_f32(&self.client, &self.hidden, DIM)
    }

    /// Blocking readback of the text logits `[32000]`.
    pub fn read_logits(&self) -> Vec<f32> {
        read_f32(&self.client, &self.logits, cfg::TEXT_LOGITS)
    }

    /// Blocking readback of BOTH step outputs in one drain — `transformer_out`
    /// `[4096]` and the text logits `[32000]`. One staging flush + one poll
    /// instead of two sequential blocking reads (pure host-side scheduling;
    /// the values are identical to the two single reads).
    pub fn read_hidden_logits(&self) -> (Vec<f32>, Vec<f32>) {
        use cubecl::CubeElement;
        let mut out = self
            .client
            .read(vec![self.hidden.clone(), self.logits.clone()]);
        let lg = out.pop().expect("logits readback");
        let hd = out.pop().expect("hidden readback");
        let mut h = vec![0f32; DIM];
        h.copy_from_slice(&f32::from_bytes(&hd)[..DIM]);
        let mut l = vec![0f32; cfg::TEXT_LOGITS];
        l.copy_from_slice(&f32::from_bytes(&lg)[..cfg::TEXT_LOGITS]);
        (h, l)
    }

    /// One full step: submit + read (hidden, logits).
    pub fn step(&mut self, x_host: &[f32], head: Head) -> (Vec<f32>, Vec<f32>) {
        self.step_submit(x_host, head);
        self.read_hidden_logits()
    }

    /// Steps consumed (the RoPE position of the next step).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// BENCH-ONLY: pin the cache-fill cursor so a timing step attends over
    /// exactly `len` positions (wgpu zero-fills the untouched cache slots —
    /// timing is value-independent, numerics are meaningless afterwards).
    pub fn force_len(&mut self, len: usize) {
        assert!(len < MAX_SEQ);
        self.len = len;
    }

    /// Reset the streaming state for a new session (the cache slots are
    /// overwritten in place as the new session fills).
    pub fn reset(&mut self) {
        self.len = 0;
    }
}

fn read_f32(client: &q4::Client, h: &Handle, n: usize) -> Vec<f32> {
    use cubecl::CubeElement;
    let bytes = client.read_one(h.clone()).expect("readback");
    let mut v = vec![0f32; n];
    v.copy_from_slice(&f32::from_bytes(&bytes)[..n]);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dequantize_q8(wq: &[u32], scales: &[f16], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let words_per_row = in_dim / 4;
        let groups_per_row = in_dim / 32;
        let mut w = vec![0f32; out_dim * in_dim];
        for j in 0..out_dim {
            for k in 0..in_dim {
                let word = wq[j * words_per_row + k / 4];
                let b = (word >> (8 * (k % 4))) & 255;
                let d = scales[j * groups_per_row + k / 32].to_f32();
                w[j * in_dim + k] = (b as f32 - 128.0) * d;
            }
        }
        w
    }

    #[test]
    fn q8_roundtrip_error_is_q8_class() {
        let (out, inn) = (16, 256);
        let mut s = 7u64;
        let w: Vec<f32> = (0..out * inn)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32;
                (u * 2.0 - 1.0) * 0.05
            })
            .collect();
        let (wq, scales) = quantize_q8(&w, out, inn);
        let wd = dequantize_q8(&wq, &scales, out, inn);
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in w.iter().zip(&wd) {
            num += ((a - b) as f64).powi(2);
            den += (*a as f64).powi(2);
        }
        let rel = (num / den).sqrt();
        // analytic q8_0 on uniform[-a,a]: step ≈ a/127, RMS err step/√12,
        // signal RMS a/√3 => rel ≈ (√3/√12)/127 ≈ 3.9e-3
        assert!(
            rel < 6e-3,
            "q8 round-trip rel RMS {rel} out of class (expect ~4e-3)"
        );
    }
}
