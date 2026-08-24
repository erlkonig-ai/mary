//! Mimi **encoder** on cubecl — the GPU twin of [`super::encoder`].
//!
//! The CPU encoder (Accelerate im2col sgemm) is correct and is kept as the
//! parity oracle, but it is a *host* stage on a hard-realtime path: every
//! millisecond it spends is a millisecond of the 80 ms frame budget taken from
//! the depformer, and every core it takes from Accelerate's thread pool is a
//! core the rest of the pipeline is not running on. This module rebuilds the
//! whole encode — SEANet, the 8-layer transformer bottleneck, the learned
//! downsample and the split-RVQ argmin — as hand-launched cubecl kernels, so
//! the stage costs host time only for one 7.7 kB upload and one 32-byte
//! readback.
//!
//! cubecl (not Metal) because the same kernels have to run on CUDA: the
//! realtime lane is meant to hold on aarch64/Blackwell as well as on Apple
//! silicon, and a Metal-only port would forfeit that for nothing.
//!
//! ## Numerics honesty
//!
//! This is **not** a bit-exact port and does not try to be. Three deliberate
//! differences from [`super::encoder`]:
//!
//! - reduction order — every dot product is a cube-cooperative tree reduce
//!   rather than Accelerate's blocked sgemm order;
//! - LayerNorm statistics accumulate in f32 (two-pass mean/variance), where
//!   the CPU path accumulates in f64;
//! - GELU uses an inline Abramowitz–Stegun 7.1.26 `erf` (max abs error
//!   1.5e-7) instead of `libm::erf`.
//!
//! Everything else — layouts, causal padding, the sliding window, the
//! quantizer's `embedding_sum / clamp(cluster_usage, 1e-5)` derivation — is
//! the same arithmetic. The consequence is that the RVQ argmin can pick a
//! different codebook row when two rows sit within reduction noise of each
//! other, so parity is judged as a **token agreement rate** plus the relative
//! error of the continuous latent that feeds the argmin (`mimi_gpu_probe`),
//! never as integer equality.
//!
//! ## Layout: position-major, one arena
//!
//! The CPU path keeps convolution activations channel-major `[C, L]` and
//! transposes twice around the transformer. Here every activation is
//! **position-major `[L, C]`**, which makes the channel index the contiguous
//! one: consecutive lanes of a reduction group read consecutive channels
//! (coalesced) for both activations and weights, and the transformer needs no
//! transpose at all. Convolution weights are transposed once at load from the
//! shipped `[out, in, k]` into `[out, k, in]` to match.
//!
//! All activations live in ONE arena buffer. Each streaming stage owns a
//! region of `slack + L` rows, where `slack` is the largest causal history any
//! of its consumers needs; producers write at row `slack`, consumers read from
//! row `slack - (k - stride)`, so the concatenation of history and current
//! chunk that the CPU path builds per call is just an index range here. One
//! `shift_kernel` dispatch at the end of a frame slides each region's tail
//! into its own slack — the entire streaming state update, in one dispatch.
//!
//! ## Dispatch shape (100 per 80 ms frame)
//!
//! 1 upload + 14 convolutions + 65 transformer + 1 downsample + 2 quantizer
//! projections + 16 RVQ (8 scans + 8 picks) + 1 history shift. One
//! `conv_kernel` covers both the convolutions and every matmul (a k=1,
//! stride=1 causal convolution *is* a `[N, K]` matmul), which is why the kernel
//! count is small: the variants differ only in comptime epilogue flags.
//!
//! The transformer is 65 of those 100 dispatches and about a fifth of the time,
//! because at T=2 rows nothing in it is compute-bound; its LayerScale residual
//! adds are therefore fused into the LayerNorm that consumes them (`norm2`
//! after attention, the NEXT layer's `norm1` after the MLP) and the MLP's GELU
//! into the matmul that produces it, which is what keeps it at 8 dispatches per
//! layer instead of 12.
//!
//! Measured stage split on M4 Max (GPU-side, 8 frames per drain, real
//! checkpoint): SEANet + upload + shift 1.1 ms, transformer 1.2 ms, downsample
//! + RVQ 0.67 ms.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::config::*;
use crate::nn::q4::{client_for_default_device, Client, Rt};
use crate::nn::weight_loader::WeightLoader;

/// Threads per cube for every kernel that reduces cooperatively.
const CUBE: u32 = 256;

/// K/V ring capacity for the encoder transformer.
///
/// The causal window is 250 ([`TR_WINDOW`]), but both of a frame's two
/// positions are written into the ring *before* attention reads it, so a
/// 250-slot ring would have the newer of the two alias `q - 249` — exactly the
/// oldest key the first query still needs. 256 is the next power of two above
/// the 252 live positions, and makes the slot map a mask.
const RING: u32 = 256;

/// Shared-memory score capacity for one attention cube; the window caps the
/// live key count at [`TR_WINDOW`] = 250.
const SCORE_CAP: u32 = 256;

/// Codebook chunks per RVQ step — 8 cubes of [`CUBE`] threads over 2048 rows
/// is exactly one row per thread.
const RVQ_CHUNKS: u32 = 8;

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

/// Exact (erf-based) GELU, matching torch's default. `erf` is Abramowitz &
/// Stegun 7.1.26 — max absolute error 1.5e-7, an order below f32 resolution
/// at the magnitudes the MLP produces.
#[cube]
fn gelu_scalar(x: f32) -> f32 {
    let z = x * 0.7071067811865476;
    let mut sign = f32::new(1.0);
    let mut a = z;
    if z < 0.0 {
        sign = f32::new(-1.0);
        a = -z;
    }
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign * (1.0 - poly * (-(a * a)).exp());
    0.5 * x * (1.0 + erf)
}

/// Copy `n` floats from a freshly uploaded staging array into the arena.
///
/// The stem's causal history lives in the arena like every other stage's, so
/// the incoming frame has to land *inside* the arena rather than in a buffer of
/// its own; this is the one dispatch that costs.
#[cube(launch_unchecked)]
fn upload_kernel(acts: &mut Array<f32>, src: &Array<f32>, dst_off: u32, n: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        acts[(dst_off + i) as usize] = src[i as usize];
    }
}

/// Slide every streaming region's causal tail into its own leading slack —
/// the whole per-frame state update, one dispatch. `desc` holds
/// `[off, c, l, slack]` per region and cube `r` owns region `r`. Source rows
/// `[l, l+slack)` and destination rows `[0, slack)` never overlap because
/// every region has `l >= slack`.
#[cube(launch_unchecked)]
fn shift_kernel(acts: &mut Array<f32>, desc: &Array<u32>, #[comptime] cube_dim: u32) {
    let r = CUBE_POS_X;
    let off = desc[(r * 4) as usize];
    let c = desc[(r * 4 + 1) as usize];
    let l = desc[(r * 4 + 2) as usize];
    let slack = desc[(r * 4 + 3) as usize];
    let n = slack * c;
    let mut i = UNIT_POS_X;
    while i < n {
        acts[(off + i) as usize] = acts[(off + l * c + i) as usize];
        i += cube_dim;
    }
}

/// Causal streaming convolution — and, at `k = 1, stride = 1`, every matmul in
/// the encoder.
///
/// `y[t, o] = bias[o] + Σ_{j<k} Σ_{c<cin} w[o, j, c] · pre(xcat[t·stride + j, c])`
///
/// where `xcat` is the region view starting at `x_off`, whose first `k−stride`
/// rows are the previous chunk's tail. `lanes` threads cooperate on one output
/// element and tree-reduce in shared memory; it is a runtime argument (not
/// comptime) so that one compiled kernel per epilogue-flag combination covers
/// all 14 convolutions and all 5 matmul shapes, whose arithmetic intensities
/// span three orders of magnitude.
///
/// Comptime epilogue/prologue flags:
/// - `pre_elu` — the SEANet ELU, applied at the *read*, so the activation it
///   transforms never has to be materialized (and its causal history is stored
///   untransformed, which is equivalent because ELU is elementwise);
/// - `has_add` — the residual-unit sum, likewise fused into the read: the
///   downsample convolution consumes `elu(x + res2(x))` without either the sum
///   or its history ever existing as a tensor;
/// - `replicate` — moshi's `pad_mode="replicate"` on the learned downsample:
///   on the first chunk of a stream the history rows read as the first current
///   row instead of zero;
/// - `post_gelu` — the transformer MLP's activation.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn conv_kernel(
    acts: &mut Array<f32>,
    w: &Array<f32>,
    b: &Array<f32>,
    x_off: u32,
    add_off: u32,
    y_off: u32,
    cin: u32,
    cout: u32,
    k: u32,
    stride: u32,
    t: u32,
    lanes: u32,
    first: u32,
    #[comptime] cube_dim: u32,
    #[comptime] pre_elu: bool,
    #[comptime] has_add: bool,
    #[comptime] has_bias: bool,
    #[comptime] replicate: bool,
    #[comptime] post_gelu: bool,
) {
    let lane = UNIT_POS_X % lanes;
    let idx = CUBE_POS_X * (cube_dim / lanes) + UNIT_POS_X / lanes;
    let n = t * cout;
    let hist = k - stride;

    let mut acc = f32::new(0.0);
    if idx < n {
        let ti = idx / cout;
        let o = idx % cout;
        let mut j = u32::new(0);
        while j < k {
            let mut p = ti * stride + j;
            if replicate {
                if first != 0 {
                    if p < hist {
                        p = hist;
                    }
                }
            }
            let xr = x_off + p * cin;
            let ar = add_off + p * cin;
            let wr = (o * k + j) * cin;
            let mut c = lane;
            while c < cin {
                let mut v = acts[(xr + c) as usize];
                if has_add {
                    v += acts[(ar + c) as usize];
                }
                if pre_elu {
                    if v < 0.0 {
                        v = v.exp() - 1.0;
                    }
                }
                acc += w[(wr + c) as usize] * v;
                c += lanes;
            }
            j += 1;
        }
    }

    let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    red[UNIT_POS_X as usize] = acc;
    sync_cube();
    let mut s = lanes / 2;
    while s > 0 {
        if lane < s {
            red[UNIT_POS_X as usize] = red[UNIT_POS_X as usize] + red[(UNIT_POS_X + s) as usize];
        }
        sync_cube();
        s /= 2;
    }
    if lane == 0 {
        if idx < n {
            let o = idx % cout;
            let mut y = red[UNIT_POS_X as usize];
            if has_bias {
                y += b[o as usize];
            }
            if post_gelu {
                y = gelu_scalar(y);
            }
            acts[(y_off + idx) as usize] = y;
        }
    }
}

/// LayerScale residual add fused with the LayerNorm that consumes it.
///
/// `h[r, i] += proj[r, i] · ls[i]`, then (when `do_norm`)
/// `y[r, i] = (h[r, i] − μ_r) · rsqrt(σ²_r + eps) · lnw[i] + lnb[i]`.
/// One cube per row. The variance is a genuine two-pass computation, not
/// `E[x²] − E[x]²`: the residual stream carries a non-zero mean and the
/// one-pass form cancels catastrophically in f32.
///
/// `do_add = false` is the stream's very first norm (layer 0's `norm1`);
/// `do_norm = false` is the last layer's post-MLP add, which has no consumer
/// inside the transformer.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn add_ln_kernel(
    acts: &mut Array<f32>,
    ls: &Array<f32>,
    lnw: &Array<f32>,
    lnb: &Array<f32>,
    h_off: u32,
    proj_off: u32,
    y_off: u32,
    d: u32,
    eps: f32,
    #[comptime] cube_dim: u32,
    #[comptime] do_add: bool,
    #[comptime] do_norm: bool,
) {
    let row = CUBE_POS_X;
    let hb = h_off + row * d;
    let pb = proj_off + row * d;
    let i = UNIT_POS_X;
    let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));

    if do_add {
        let mut c = i;
        while c < d {
            acts[(hb + c) as usize] =
                acts[(hb + c) as usize] + acts[(pb + c) as usize] * ls[c as usize];
            c += cube_dim;
        }
        sync_cube();
    }

    if do_norm {
        let mut sum = f32::new(0.0);
        let mut c = i;
        while c < d {
            sum += acts[(hb + c) as usize];
            c += cube_dim;
        }
        red[i as usize] = sum;
        sync_cube();
        let mut st = u32::new((cube_dim / 2) as i64);
        while st > 0 {
            if i < st {
                red[i as usize] = red[i as usize] + red[(i + st) as usize];
            }
            sync_cube();
            st /= 2;
        }
        let mean = red[0] / (d as f32);
        sync_cube();

        let mut var = f32::new(0.0);
        let mut c2 = i;
        while c2 < d {
            let dv = acts[(hb + c2) as usize] - mean;
            var += dv * dv;
            c2 += cube_dim;
        }
        red[i as usize] = var;
        sync_cube();
        let mut st2 = u32::new((cube_dim / 2) as i64);
        while st2 > 0 {
            if i < st2 {
                red[i as usize] = red[i as usize] + red[(i + st2) as usize];
            }
            sync_cube();
            st2 /= 2;
        }
        let inv = 1.0 / (red[0] / (d as f32) + eps).sqrt();
        sync_cube();

        let mut c3 = i;
        while c3 < d {
            acts[(y_off + row * d + c3) as usize] =
                (acts[(hb + c3) as usize] - mean) * inv * lnw[c3 as usize] + lnb[c3 as usize];
            c3 += cube_dim;
        }
    }
}

/// Interleaved RoPE on q and k, plus the K/V ring write — one dispatch, one
/// cube per position, `hidden/2` threads.
///
/// moshi's RoPE is interleaved (`interleave=True`): pair `(x[2i], x[2i+1])`
/// within a head, NOT the split-half `(x[i], x[i+half])`. Thread `p` owns pair
/// `p` of q AND of k, and copies the two v elements `2p, 2p+1`; every element
/// is touched by exactly one thread, so the in-place q rotation is race-free.
/// `1/√d` is folded into q here rather than scaling scores in the attention
/// kernel.
///
/// `cs` carries this frame's two positions' cos/sin, computed on the host in
/// f64 exactly as the CPU path does — the absolute position grows without
/// bound over a session and an f32 `sin` of ~10⁵ radians is worthless.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn rope_kv_kernel(
    acts: &mut Array<f32>,
    cs: &Array<f32>,
    kring: &mut Array<f32>,
    vring: &mut Array<f32>,
    qkv_off: u32,
    ring_off: u32,
    base: u32,
    q_scale: f32,
    #[comptime] hidden: u32,
    #[comptime] half: u32,
    #[comptime] head_dim: u32,
    #[comptime] ring: u32,
) {
    let row = CUBE_POS_X;
    let pair = UNIT_POS_X;
    let head = pair / half;
    let j = pair % half;
    let e0 = head * head_dim + 2 * j;
    let e1 = e0 + 1;
    let base_off = qkv_off + row * 3 * hidden;
    let c = cs[(row * 2 * half + j) as usize];
    let s = cs[(row * 2 * half + half + j) as usize];

    let qa = acts[(base_off + e0) as usize];
    let qb = acts[(base_off + e1) as usize];
    acts[(base_off + e0) as usize] = (qa * c - qb * s) * q_scale;
    acts[(base_off + e1) as usize] = (qa * s + qb * c) * q_scale;

    let slot = ring_off + ((base + row) % ring) * hidden;
    let ka = acts[(base_off + hidden + e0) as usize];
    let kb = acts[(base_off + hidden + e1) as usize];
    kring[(slot + e0) as usize] = ka * c - kb * s;
    kring[(slot + e1) as usize] = ka * s + kb * c;

    vring[(slot + 2 * pair) as usize] = acts[(base_off + 2 * hidden + 2 * pair) as usize];
    vring[(slot + 2 * pair + 1) as usize] = acts[(base_off + 2 * hidden + 2 * pair + 1) as usize];
}

/// Causal sliding-window attention over the ring — one cube per
/// `(position, head)`, `head_dim` threads.
///
/// Because both of the frame's positions are already in the ring, the current
/// keys need no special case the way the CPU streaming path gives them: the
/// key range is simply `[max(0, q+1−window) ..= q]` in absolute positions,
/// visited in ascending (chronological) order, which is the same summation
/// order the CPU path uses.
///
/// The cube is `cube_dim / head_dim` key-groups × `head_dim` dims. In the score
/// phase thread `i` owns key `i` (the window caps the count at 250 ≤ 256), a
/// serial 64-wide dot; the cube then softmaxes in shared memory; in the
/// weighted-V phase thread `(g, d)` sweeps keys `g, g+ksplit, …` at fixed
/// output dim `d` — consecutive threads read consecutive `vring` dims, the
/// coalesced direction — and the `ksplit` partials are summed at the end.
///
/// The key-split matters: a `head_dim`-wide cube leaves the V sweep 250
/// dependent iterations long on 64 threads, and with only 16 cubes in the
/// dispatch there is no other work to hide it behind.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn attn_kernel(
    acts: &mut Array<f32>,
    kring: &Array<f32>,
    vring: &Array<f32>,
    qkv_off: u32,
    attn_off: u32,
    ring_off: u32,
    base: u32,
    #[comptime] hidden: u32,
    #[comptime] head_dim: u32,
    #[comptime] heads: u32,
    #[comptime] window: u32,
    #[comptime] ring: u32,
    #[comptime] cap: u32,
    #[comptime] cube_dim: u32,
) {
    let row = CUBE_POS_X / heads;
    let h = CUBE_POS_X % heads;
    let i = UNIT_POS_X;
    let ksplit = cube_dim / head_dim;
    let qabs = base + row;
    let mut lo = u32::new(0);
    if qabs + 1 > window {
        lo = qabs + 1 - window;
    }
    let n = qabs - lo + 1;
    let qb = qkv_off + row * 3 * hidden + h * head_dim;

    let mut sc = SharedMemory::<f32>::new(comptime!(cap as usize));
    let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));

    let mut m = i;
    while m < n {
        let kb = ring_off + ((lo + m) % ring) * hidden + h * head_dim;
        let mut dot = f32::new(0.0);
        let mut dd = u32::new(0);
        while dd < head_dim {
            dot += acts[(qb + dd) as usize] * kring[(kb + dd) as usize];
            dd += 1;
        }
        sc[m as usize] = dot;
        m += cube_dim;
    }
    sync_cube();

    let mut lmax = f32::new(-3.4e38);
    let mut m2 = i;
    while m2 < n {
        if sc[m2 as usize] > lmax {
            lmax = sc[m2 as usize];
        }
        m2 += cube_dim;
    }
    red[i as usize] = lmax;
    sync_cube();
    let mut st = u32::new((cube_dim / 2) as i64);
    while st > 0 {
        if i < st {
            if red[(i + st) as usize] > red[i as usize] {
                red[i as usize] = red[(i + st) as usize];
            }
        }
        sync_cube();
        st /= 2;
    }
    let mx = red[0];
    sync_cube();

    let mut lsum = f32::new(0.0);
    let mut m3 = i;
    while m3 < n {
        let e = (sc[m3 as usize] - mx).exp();
        sc[m3 as usize] = e;
        lsum += e;
        m3 += cube_dim;
    }
    red[i as usize] = lsum;
    sync_cube();
    let mut st2 = u32::new((cube_dim / 2) as i64);
    while st2 > 0 {
        if i < st2 {
            red[i as usize] = red[i as usize] + red[(i + st2) as usize];
        }
        sync_cube();
        st2 /= 2;
    }
    let total = red[0];
    sync_cube();

    let g = i / head_dim;
    let d = i % head_dim;
    let mut acc = f32::new(0.0);
    let mut m4 = g;
    while m4 < n {
        let vb = ring_off + ((lo + m4) % ring) * hidden + h * head_dim;
        acc += sc[m4 as usize] * vring[(vb + d) as usize];
        m4 += ksplit;
    }
    red[i as usize] = acc;
    sync_cube();
    if i < head_dim {
        let mut sum = f32::new(0.0);
        let mut gg = u32::new(0);
        while gg < ksplit {
            sum += red[(gg * head_dim + i) as usize];
            gg += 1;
        }
        acts[(attn_off + row * hidden + h * head_dim + i) as usize] = sum / total;
    }
}

/// One residual-VQ step, pass 1 of 2: cube `c` scans codebook rows
/// `[c·per, (c+1)·per)` against the running residual and emits the best
/// `(distance, index)` it saw.
///
/// `‖r − e‖² = ‖e‖² − 2⟨r, e⟩ + ‖r‖²` and the last term is constant across
/// rows, so the kernel minimizes `norms[row] − 2·dot`, exactly as the CPU path
/// does. Ties break to the LOWER index, matching the CPU's ascending scan.
///
/// Split because the residual chain is serial across the eight quantizers and
/// each step reads a 2 MB codebook: as a single cube that is 2 MB pulled
/// through one core, and it measured as the encoder's single most expensive
/// stage (1.79 ms/frame for downsample+RVQ, against 1.1 for the whole SEANet).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn rvq_scan_kernel(
    acts: &Array<f32>,
    cb: &Array<f32>,
    norms: &Array<f32>,
    part_d: &mut Array<f32>,
    part_i: &mut Array<u32>,
    res_off: u32,
    cb_off: u32,
    norm_off: u32,
    rows: u32,
    per: u32,
    #[comptime] dim: u32,
    #[comptime] cube_dim: u32,
) {
    let c = CUBE_POS_X;
    let i = UNIT_POS_X;
    let start = c * per;
    let mut end = start + per;
    if end > rows {
        end = rows;
    }
    let mut bd = f32::new(3.4e38);
    let mut bi = u32::new(0);
    let mut r = start + i;
    while r < end {
        let rb = cb_off + r * dim;
        let mut dot = f32::new(0.0);
        let mut d = u32::new(0);
        while d < dim {
            dot += cb[(rb + d) as usize] * acts[(res_off + d) as usize];
            d += 1;
        }
        let dist = norms[(norm_off + r) as usize] - 2.0 * dot;
        if dist < bd {
            bd = dist;
            bi = r;
        }
        r += cube_dim;
    }

    let mut rd = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    let mut ri = SharedMemory::<u32>::new(comptime!(cube_dim as usize));
    rd[i as usize] = bd;
    ri[i as usize] = bi;
    sync_cube();
    let mut st = u32::new((cube_dim / 2) as i64);
    while st > 0 {
        if i < st {
            let a = rd[i as usize];
            let b = rd[(i + st) as usize];
            let ai = ri[i as usize];
            let bj = ri[(i + st) as usize];
            let mut take = b < a;
            if b == a {
                if bj < ai {
                    take = true;
                }
            }
            if take {
                rd[i as usize] = b;
                ri[i as usize] = bj;
            }
        }
        sync_cube();
        st /= 2;
    }
    if i == 0 {
        part_d[c as usize] = rd[0];
        part_i[c as usize] = ri[0];
    }
}

/// Pass 2 of 2: reduce the per-chunk winners, emit the code, and subtract the
/// chosen entry from the residual so the next quantizer sees the remainder.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn rvq_pick_kernel(
    acts: &mut Array<f32>,
    cb: &Array<f32>,
    part_d: &Array<f32>,
    part_i: &Array<u32>,
    codes: &mut Array<u32>,
    res_off: u32,
    cb_off: u32,
    q: u32,
    #[comptime] dim: u32,
    #[comptime] chunks: u32,
    #[comptime] cube_dim: u32,
) {
    let i = UNIT_POS_X;
    let mut rd = SharedMemory::<f32>::new(comptime!(chunks as usize));
    let mut ri = SharedMemory::<u32>::new(comptime!(chunks as usize));
    if i < chunks {
        rd[i as usize] = part_d[i as usize];
        ri[i as usize] = part_i[i as usize];
    }
    sync_cube();
    let mut st = u32::new((chunks / 2) as i64);
    while st > 0 {
        if i < st {
            let a = rd[i as usize];
            let b = rd[(i + st) as usize];
            let ai = ri[i as usize];
            let bj = ri[(i + st) as usize];
            let mut take = b < a;
            if b == a {
                if bj < ai {
                    take = true;
                }
            }
            if take {
                rd[i as usize] = b;
                ri[i as usize] = bj;
            }
        }
        sync_cube();
        st /= 2;
    }
    let best = ri[0];
    if i == 0 {
        codes[q as usize] = best;
    }
    let mut d = i;
    while d < dim {
        acts[(res_off + d) as usize] =
            acts[(res_off + d) as usize] - cb[(cb_off + best * dim + d) as usize];
        d += cube_dim;
    }
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------

/// One `slack + l` × `c` position-major slice of the activation arena.
#[derive(Clone, Copy, Debug)]
struct Region {
    off: u32,
    c: u32,
    l: u32,
    slack: u32,
}

impl Region {
    /// Element offset of the row a consumer with causal history `hist` must
    /// start reading from.
    fn view(&self, hist: u32) -> u32 {
        debug_assert!(hist <= self.slack);
        self.off + (self.slack - hist) * self.c
    }
    /// Element offset a producer writes its `l` fresh rows at.
    fn write(&self) -> u32 {
        self.off + self.slack * self.c
    }
}

struct Arena {
    regions: Vec<Region>,
    len: u32,
}

impl Arena {
    fn new() -> Self {
        Self {
            regions: Vec::new(),
            len: 0,
        }
    }
    fn push(&mut self, c: usize, l: usize, slack: usize) -> usize {
        let r = Region {
            off: self.len,
            c: c as u32,
            l: l as u32,
            slack: slack as u32,
        };
        self.len += r.c * (r.l + r.slack);
        self.regions.push(r);
        self.regions.len() - 1
    }
}

/// Indices of every arena region, in the order [`build_arena`] lays them out.
struct Layout {
    r_in: usize,
    r_stem: usize,
    /// per SEANet block: `[res1 out, res2 out, downsample out]`
    r_block: Vec<[usize; 3]>,
    r_h: usize,
    r_ds: usize,
    r_xn: usize,
    r_qkv: usize,
    r_attn: usize,
    r_proj: usize,
    r_ff: usize,
    r_res: usize,
}

/// Lay out the activation arena.
///
/// A stage's `slack` is the largest causal history any of its consumers needs;
/// a consumer with history `h` then reads from row `slack − h`. The block input
/// is the interesting case: it feeds both the residual unit's k3 convolution
/// (history 2) and the downsample's `k = 2r, stride = r` (history `r`), so its
/// slack is `max(2, r)` and the two consumers start at different rows of the
/// same region.
///
/// Weight-free by construction, so the invariants it has to satisfy — every
/// consumer's history fits its producer's slack, and every shifted region has
/// `l >= slack` so the tail slide never overlaps itself — are testable without
/// a checkpoint or a device.
fn build_arena() -> (Arena, Layout) {
    let mut a = Arena::new();
    let r_in = a.push(1, SAMPLES_PER_FRAME, 6);
    let r_stem = a.push(64, SAMPLES_PER_FRAME, ENC_RATIOS[0].max(2));
    let mut r_block = Vec::new();
    let mut l = SAMPLES_PER_FRAME;
    for (i, &r) in ENC_RATIOS.iter().enumerate() {
        let dim = 64usize << i;
        let res1 = a.push(dim / 2, l, 0);
        let res2 = a.push(dim, l, r);
        l /= r;
        let next_slack = if i + 1 < ENC_RATIOS.len() {
            ENC_RATIOS[i + 1].max(2)
        } else {
            2 // final_conv, k3 stride 1
        };
        let out = a.push(2 * dim, l, next_slack);
        r_block.push([res1, res2, out]);
    }
    // The final convolution's output IS the transformer's in-place buffer, and
    // also the learned downsample's input (k4 stride 2 → 2 rows of history), so
    // the transformer needs no buffer of its own and no copy on either side.
    let r_h = a.push(HIDDEN, l, 2);
    let r_ds = a.push(HIDDEN, 1, 0);
    // Scratch: no consumer across frames, hence no slack.
    let r_xn = a.push(HIDDEN, l, 0);
    let r_qkv = a.push(3 * HIDDEN, l, 0);
    let r_attn = a.push(HIDDEN, l, 0);
    let r_proj = a.push(HIDDEN, l, 0);
    let r_ff = a.push(TR_INTER, l, 0);
    let r_res = a.push(CODE_DIM, 1, 0);
    (
        a,
        Layout {
            r_in,
            r_stem,
            r_block,
            r_h,
            r_ds,
            r_xn,
            r_qkv,
            r_attn,
            r_proj,
            r_ff,
            r_res,
        },
    )
}

struct GpuConv {
    w: Handle,
    b: Option<Handle>,
    cin: u32,
    cout: u32,
    k: u32,
    stride: u32,
}

impl GpuConv {
    /// Load one causal convolution, transposing the shipped `[out, in, k]`
    /// weight into the `[out, k, in]` order the kernel reads coalesced.
    fn load(
        client: &Client,
        loader: &WeightLoader,
        prefix: &str,
        stride: usize,
        bias: bool,
    ) -> Self {
        let (w, shape) = loader.load_host_f32(&format!("{prefix}.weight"));
        let (out, inc, k) = (shape[0], shape[1], shape[2]);
        let mut t = vec![0f32; out * inc * k];
        for o in 0..out {
            for c in 0..inc {
                for j in 0..k {
                    t[(o * k + j) * inc + c] = w[(o * inc + c) * k + j];
                }
            }
        }
        Self {
            w: client.create_from_slice(as_bytes(&t)),
            b: bias.then(|| {
                let (b, _) = loader.load_host_f32(&format!("{prefix}.bias"));
                client.create_from_slice(as_bytes(&b[..]))
            }),
            cin: inc as u32,
            cout: out as u32,
            k: k as u32,
            stride: stride as u32,
        }
    }

    /// A `[N, K]` matmul dressed as a k=1 stride=1 convolution.
    fn matmul(client: &Client, loader: &WeightLoader, name: &str) -> Self {
        let (w, shape) = loader.load_host_f32(name);
        Self {
            w: client.create_from_slice(as_bytes(&w[..])),
            b: None,
            cin: shape[1] as u32,
            cout: shape[0] as u32,
            k: 1,
            stride: 1,
        }
    }
}

struct GpuTr {
    ln1_w: Handle,
    ln1_b: Handle,
    ln2_w: Handle,
    ln2_b: Handle,
    in_proj: GpuConv,
    out_proj: GpuConv,
    fc1: GpuConv,
    fc2: GpuConv,
    ls1: Handle,
    ls2: Handle,
}

struct GpuRvq {
    input_proj: GpuConv,
    /// All quantizers' codebooks concatenated: `[n_q, rows, CODE_DIM]`.
    cb: Handle,
    /// Matching squared row norms: `[n_q, rows]`.
    norms: Handle,
    rows: u32,
    n_q: u32,
}

impl GpuRvq {
    fn load(client: &Client, loader: &WeightLoader, prefix: &str, n_q: usize) -> Self {
        let mut cb_all = Vec::new();
        let mut norms_all = Vec::new();
        let mut rows = 0usize;
        for i in 0..n_q {
            let (sum, _) =
                loader.load_f32(&format!("{prefix}.vq.layers.{i}._codebook.embedding_sum"));
            let (usage, _) =
                loader.load_f32(&format!("{prefix}.vq.layers.{i}._codebook.cluster_usage"));
            let mut cb = sum;
            for (r, &u) in usage.iter().enumerate() {
                let d = u.max(1e-5);
                for v in &mut cb[r * CODE_DIM..(r + 1) * CODE_DIM] {
                    *v /= d;
                }
            }
            rows = usage.len();
            norms_all.extend((0..rows).map(|r| {
                cb[r * CODE_DIM..(r + 1) * CODE_DIM]
                    .iter()
                    .map(|&v| v * v)
                    .sum::<f32>()
            }));
            cb_all.extend_from_slice(&cb);
        }
        Self {
            input_proj: GpuConv::matmul(client, loader, &format!("{prefix}.input_proj.weight")),
            cb: client.create_from_slice(as_bytes(&cb_all)),
            norms: client.create_from_slice(as_bytes(&norms_all)),
            rows: rows as u32,
            n_q: n_q as u32,
        }
    }
}

/// The GPU Mimi encoder: weights, activation arena and streaming state for
/// **one** stream. Cheap to reset ([`Self::reset`]), not shareable between
/// concurrent streams — the arena and the K/V ring are the stream's causal
/// state, not scratch.
pub struct MimiEncoderGpu {
    client: Client,
    acts: Handle,
    kring: Handle,
    vring: Handle,
    codes: Handle,
    part_d: Handle,
    part_i: Handle,
    desc: Handle,
    n_shift: u32,
    dummy: Handle,
    arena_len: u32,

    stem: GpuConv,
    blocks: Vec<[GpuConv; 3]>,
    final_conv: GpuConv,
    layers: Vec<GpuTr>,
    downsample: GpuConv,
    rvq_first: GpuRvq,
    rvq_rest: GpuRvq,

    // arena region indices
    r_in: usize,
    r_stem: usize,
    /// per block: [res1, res2, down_out]
    r_block: Vec<[usize; 3]>,
    r_h: usize,
    r_ds: usize,
    r_xn: usize,
    r_qkv: usize,
    r_attn: usize,
    r_proj: usize,
    r_ff: usize,
    r_res: usize,
    regions: Vec<Region>,

    pos: u32,
    frames: u64,
    tail: [f32; 6],
}

impl MimiEncoderGpu {
    pub fn load(loader: &WeightLoader) -> Self {
        Self::load_on(client_for_default_device(), loader)
    }

    pub fn load_on(client: Client, loader: &WeightLoader) -> Self {
        let p = "encoder.model";
        let stem = GpuConv::load(&client, loader, &format!("{p}.0.conv.conv"), 1, true);
        let blocks: Vec<[GpuConv; 3]> = ENC_RATIOS
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                [
                    GpuConv::load(
                        &client,
                        loader,
                        &format!("{p}.{}.block.1.conv.conv", 3 * i + 1),
                        1,
                        true,
                    ),
                    GpuConv::load(
                        &client,
                        loader,
                        &format!("{p}.{}.block.3.conv.conv", 3 * i + 1),
                        1,
                        true,
                    ),
                    GpuConv::load(
                        &client,
                        loader,
                        &format!("{p}.{}.conv.conv", 3 * i + 3),
                        r,
                        true,
                    ),
                ]
            })
            .collect();
        let final_conv = GpuConv::load(&client, loader, &format!("{p}.14.conv.conv"), 1, true);
        let t = "encoder_transformer.transformer.layers";
        let layers = (0..TR_LAYERS)
            .map(|i| GpuTr {
                ln1_w: vecs(&client, loader, &format!("{t}.{i}.norm1.weight")),
                ln1_b: vecs(&client, loader, &format!("{t}.{i}.norm1.bias")),
                ln2_w: vecs(&client, loader, &format!("{t}.{i}.norm2.weight")),
                ln2_b: vecs(&client, loader, &format!("{t}.{i}.norm2.bias")),
                in_proj: GpuConv::matmul(
                    &client,
                    loader,
                    &format!("{t}.{i}.self_attn.in_proj_weight"),
                ),
                out_proj: GpuConv::matmul(
                    &client,
                    loader,
                    &format!("{t}.{i}.self_attn.out_proj.weight"),
                ),
                fc1: GpuConv::matmul(&client, loader, &format!("{t}.{i}.linear1.weight")),
                fc2: GpuConv::matmul(&client, loader, &format!("{t}.{i}.linear2.weight")),
                ls1: vecs(&client, loader, &format!("{t}.{i}.layer_scale_1.scale")),
                ls2: vecs(&client, loader, &format!("{t}.{i}.layer_scale_2.scale")),
            })
            .collect();
        let downsample = GpuConv::load(&client, loader, "downsample.conv.conv.conv", 2, false);
        let rvq_first = GpuRvq::load(&client, loader, "quantizer.rvq_first", 1);
        let rvq_rest = GpuRvq::load(&client, loader, "quantizer.rvq_rest", N_ACOUSTIC);

        let (a, lay) = build_arena();

        let regions = a.regions.clone();
        let mut desc: Vec<u32> = Vec::new();
        for r in &regions {
            if r.slack > 0 && r.off != regions[lay.r_in].off {
                desc.extend_from_slice(&[r.off, r.c, r.l, r.slack]);
            }
        }
        let n_shift = (desc.len() / 4) as u32;

        let acts = client.create_from_slice(as_bytes(&vec![0f32; a.len as usize]));
        // ONE RING PER LAYER. The eight layers cache different keys for the
        // same positions; a shared ring silently returns the last layer's keys
        // to every earlier layer, which is invisible on the first frame of a
        // stream (every layer reads back only what it just wrote) and wrong on
        // every frame after it.
        let ring_elems = TR_LAYERS * (RING as usize) * HIDDEN;
        let kring = client.create_from_slice(as_bytes(&vec![0f32; ring_elems]));
        let vring = client.create_from_slice(as_bytes(&vec![0f32; ring_elems]));
        let codes = client.create_from_slice(as_bytes(&vec![0u32; NUM_CODEBOOKS]));
        let part_d = client.create_from_slice(as_bytes(&vec![0f32; RVQ_CHUNKS as usize]));
        let part_i = client.create_from_slice(as_bytes(&vec![0u32; RVQ_CHUNKS as usize]));
        let desc = client.create_from_slice(as_bytes(&desc));
        let dummy = client.create_from_slice(as_bytes(&[0f32]));

        Self {
            client,
            acts,
            kring,
            vring,
            codes,
            part_d,
            part_i,
            desc,
            n_shift,
            dummy,
            arena_len: a.len,
            stem,
            blocks,
            final_conv,
            layers,
            downsample,
            rvq_first,
            rvq_rest,
            r_in: lay.r_in,
            r_stem: lay.r_stem,
            r_block: lay.r_block,
            r_h: lay.r_h,
            r_ds: lay.r_ds,
            r_xn: lay.r_xn,
            r_qkv: lay.r_qkv,
            r_attn: lay.r_attn,
            r_proj: lay.r_proj,
            r_ff: lay.r_ff,
            r_res: lay.r_res,
            regions,
            pos: 0,
            frames: 0,
            tail: [0.0; 6],
        }
    }

    /// Return to the exact beginning-of-stream condition. The ring needs no
    /// clearing: with the position reset, no query's window ever reaches a slot
    /// this session has not written.
    pub fn reset(&mut self) {
        self.acts = self
            .client
            .create_from_slice(as_bytes(&vec![0f32; self.arena_len as usize]));
        self.pos = 0;
        self.frames = 0;
        self.tail = [0.0; 6];
    }

    /// Threads cooperating on one output element. Enough parallelism to fill
    /// the device without splitting reductions that are already short: the
    /// SEANet stem has 123 k output elements and 7 MACs each, the last
    /// downsample has 2 k elements and 8 k MACs each.
    fn lanes_for(outputs: u32) -> u32 {
        let mut l = 1u32;
        while l < 32 && outputs * l < 32768 {
            l *= 2;
        }
        l
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        c: &GpuConv,
        x_off: u32,
        add_off: u32,
        y_off: u32,
        t: u32,
        first: u32,
        pre_elu: bool,
        has_add: bool,
        replicate: bool,
        post_gelu: bool,
    ) {
        let outputs = t * c.cout;
        let lanes = Self::lanes_for(outputs);
        let per_cube = CUBE / lanes;
        let cubes = outputs.div_ceil(per_cube);
        let cl = &self.client;
        let arr = |h: &Handle, n: usize| unsafe { ArrayArg::from_raw_parts(h.clone(), n) };
        let bias = c.b.as_ref().unwrap_or(&self.dummy);
        macro_rules! go {
            ($pe:expr, $ha:expr, $hb:expr, $rp:expr, $pg:expr) => {
                unsafe {
                    conv_kernel::launch_unchecked::<Rt>(
                        cl,
                        CubeCount::new_1d(cubes),
                        CubeDim::new_1d(CUBE),
                        arr(&self.acts, self.arena_len as usize),
                        arr(&c.w, (c.cout * c.cin * c.k) as usize),
                        arr(bias, c.cout as usize),
                        x_off,
                        add_off,
                        y_off,
                        c.cin,
                        c.cout,
                        c.k,
                        c.stride,
                        t,
                        lanes,
                        first,
                        CUBE,
                        $pe,
                        $ha,
                        $hb,
                        $rp,
                        $pg,
                    );
                }
            };
        }
        let hb = c.b.is_some();
        match (pre_elu, has_add, hb, replicate, post_gelu) {
            (false, false, true, false, false) => go!(false, false, true, false, false),
            (true, false, true, false, false) => go!(true, false, true, false, false),
            (true, true, true, false, false) => go!(true, true, true, false, false),
            (false, false, false, true, false) => go!(false, false, false, true, false),
            (false, false, false, false, false) => go!(false, false, false, false, false),
            (false, false, false, false, true) => go!(false, false, false, false, true),
            other => panic!("unwired conv epilogue combination {other:?}"),
        }
    }

    /// Encode exactly one 80 ms, 24 kHz mono frame. Semantically identical to
    /// [`super::MimiEncoder::encode_stream_frame`] up to the numerics noted in
    /// the module docs.
    pub fn encode_frame(&mut self, samples: &[f32; SAMPLES_PER_FRAME]) -> [u32; NUM_CODEBOOKS] {
        self.submit_frame(samples);
        self.read_codes()
    }

    /// Submit one frame's 100 dispatches without draining. Split out from
    /// [`Self::encode_frame`] so a caller can overlap the host work it does
    /// between submission and needing the codes.
    pub fn submit_frame(&mut self, samples: &[f32; SAMPLES_PER_FRAME]) {
        let cl = self.client.clone();
        // Two closures, not one: each is monomorphic in the element type its
        // first use infers (f32 activations, u32 codes/descriptors).
        let arr = |h: &Handle, n: usize| unsafe { ArrayArg::from_raw_parts(h.clone(), n) };
        let arru = |h: &Handle, n: usize| unsafe { ArrayArg::from_raw_parts(h.clone(), n) };
        let reg = |i: usize| self.regions[i];

        // ── 1. the incoming frame, with the stem's six-sample causal tail
        // prepended on the host (the only stage whose history is cheaper to
        // carry outside the arena, because it arrives from outside anyway).
        let mut staging = Vec::with_capacity(6 + SAMPLES_PER_FRAME);
        staging.extend_from_slice(&self.tail);
        staging.extend_from_slice(samples);
        self.tail.copy_from_slice(&samples[SAMPLES_PER_FRAME - 6..]);
        let stage = cl.create_from_slice(as_bytes(&staging));
        let n = staging.len() as u32;
        unsafe {
            upload_kernel::launch_unchecked::<Rt>(
                &cl,
                CubeCount::new_1d(n.div_ceil(CUBE)),
                CubeDim::new_1d(CUBE),
                arr(&self.acts, self.arena_len as usize),
                arr(&stage, n as usize),
                reg(self.r_in).off,
                n,
            );
        }

        // ── 2. SEANet ──
        let r_in = reg(self.r_in);
        let r_stem = reg(self.r_stem);
        let r_h = reg(self.r_h);
        self.conv(
            &self.stem,
            r_in.view(6),
            0,
            r_stem.write(),
            SAMPLES_PER_FRAME as u32,
            0,
            false,
            false,
            false,
            false,
        );
        let mut x = self.r_stem;
        for (bi, convs) in self.blocks.iter().enumerate() {
            let [c1, c2, cd] = convs;
            let rx = reg(x);
            let r1 = reg(self.r_block[bi][0]);
            let r2 = reg(self.r_block[bi][1]);
            let ro = reg(self.r_block[bi][2]);
            // residual unit: elu fused into each conv's read, so neither
            // `elu(x)` nor `elu(res1)` is ever materialized.
            self.conv(
                c1,
                rx.view(2),
                0,
                r1.write(),
                rx.l,
                0,
                true,
                false,
                false,
                false,
            );
            self.conv(
                c2,
                r1.view(0),
                0,
                r2.write(),
                r1.l,
                0,
                true,
                false,
                false,
                false,
            );
            // downsample reads elu(x + res2) directly out of the two regions.
            let hist = cd.k - cd.stride;
            self.conv(
                cd,
                rx.view(hist),
                r2.view(hist),
                ro.write(),
                ro.l,
                0,
                true,
                true,
                false,
                false,
            );
            x = self.r_block[bi][2];
        }
        let rx = reg(x);
        self.conv(
            &self.final_conv,
            rx.view(2),
            0,
            r_h.write(),
            r_h.l,
            0,
            true,
            false,
            false,
            false,
        );

        // ── 3. transformer bottleneck (in place on the r_h rows) ──
        let t = r_h.l;
        let h_off = r_h.write();
        let xn = reg(self.r_xn).off;
        let qkv = reg(self.r_qkv).off;
        let attn = reg(self.r_attn).off;
        let proj = reg(self.r_proj).off;
        let ff = reg(self.r_ff).off;
        let half = (TR_HEAD_DIM / 2) as u32;

        // Absolute-position RoPE tables for this frame's t positions, in f64
        // on the host exactly as the CPU path computes them.
        let mut cs = vec![0f32; (t as usize) * 2 * half as usize];
        for row in 0..t as usize {
            let abs = self.pos as usize + row;
            for j in 0..half as usize {
                let r = abs as f64 * TR_ROPE_THETA.powf(-2.0 * j as f64 / TR_HEAD_DIM as f64);
                cs[row * 2 * half as usize + j] = r.cos() as f32;
                cs[row * 2 * half as usize + half as usize + j] = r.sin() as f32;
            }
        }
        let cs_h = cl.create_from_slice(as_bytes(&cs));
        let q_scale = ((TR_HEAD_DIM as f64).powf(-0.5)) as f32;

        for (li, layer) in self.layers.iter().enumerate() {
            if li == 0 {
                unsafe {
                    add_ln_kernel::launch_unchecked::<Rt>(
                        &cl,
                        CubeCount::new_1d(t),
                        CubeDim::new_1d(CUBE),
                        arr(&self.acts, self.arena_len as usize),
                        arr(&self.dummy, 1),
                        arr(&layer.ln1_w, HIDDEN),
                        arr(&layer.ln1_b, HIDDEN),
                        h_off,
                        0,
                        xn,
                        HIDDEN as u32,
                        TR_EPS as f32,
                        CUBE,
                        false,
                        true,
                    );
                }
            }
            let ring_off = (li as u32) * RING * HIDDEN as u32;
            self.conv(&layer.in_proj, xn, 0, qkv, t, 0, false, false, false, false);
            unsafe {
                rope_kv_kernel::launch_unchecked::<Rt>(
                    &cl,
                    CubeCount::new_1d(t),
                    CubeDim::new_1d(HIDDEN as u32 / 2),
                    arr(&self.acts, self.arena_len as usize),
                    arr(&cs_h, cs.len()),
                    arr(&self.kring, TR_LAYERS * RING as usize * HIDDEN),
                    arr(&self.vring, TR_LAYERS * RING as usize * HIDDEN),
                    qkv,
                    ring_off,
                    self.pos,
                    q_scale,
                    HIDDEN as u32,
                    half,
                    TR_HEAD_DIM as u32,
                    RING,
                );
                attn_kernel::launch_unchecked::<Rt>(
                    &cl,
                    CubeCount::new_1d(t * TR_HEADS as u32),
                    CubeDim::new_1d(CUBE),
                    arr(&self.acts, self.arena_len as usize),
                    arr(&self.kring, TR_LAYERS * RING as usize * HIDDEN),
                    arr(&self.vring, TR_LAYERS * RING as usize * HIDDEN),
                    qkv,
                    attn,
                    ring_off,
                    self.pos,
                    HIDDEN as u32,
                    TR_HEAD_DIM as u32,
                    TR_HEADS as u32,
                    TR_WINDOW as u32,
                    RING,
                    SCORE_CAP,
                    CUBE,
                );
            }
            self.conv(
                &layer.out_proj,
                attn,
                0,
                proj,
                t,
                0,
                false,
                false,
                false,
                false,
            );
            unsafe {
                add_ln_kernel::launch_unchecked::<Rt>(
                    &cl,
                    CubeCount::new_1d(t),
                    CubeDim::new_1d(CUBE),
                    arr(&self.acts, self.arena_len as usize),
                    arr(&layer.ls1, HIDDEN),
                    arr(&layer.ln2_w, HIDDEN),
                    arr(&layer.ln2_b, HIDDEN),
                    h_off,
                    proj,
                    xn,
                    HIDDEN as u32,
                    TR_EPS as f32,
                    CUBE,
                    true,
                    true,
                );
            }
            self.conv(&layer.fc1, xn, 0, ff, t, 0, false, false, false, true);
            self.conv(&layer.fc2, ff, 0, proj, t, 0, false, false, false, false);
            // The post-MLP LayerScale add fuses with the NEXT layer's norm1;
            // the last layer's has no consumer, so it only adds.
            let last = li + 1 == self.layers.len();
            let (nw, nb) = if last {
                (&self.dummy, &self.dummy)
            } else {
                (&self.layers[li + 1].ln1_w, &self.layers[li + 1].ln1_b)
            };
            unsafe {
                if last {
                    add_ln_kernel::launch_unchecked::<Rt>(
                        &cl,
                        CubeCount::new_1d(t),
                        CubeDim::new_1d(CUBE),
                        arr(&self.acts, self.arena_len as usize),
                        arr(&layer.ls2, HIDDEN),
                        arr(nw, 1),
                        arr(nb, 1),
                        h_off,
                        proj,
                        xn,
                        HIDDEN as u32,
                        TR_EPS as f32,
                        CUBE,
                        true,
                        false,
                    );
                } else {
                    add_ln_kernel::launch_unchecked::<Rt>(
                        &cl,
                        CubeCount::new_1d(t),
                        CubeDim::new_1d(CUBE),
                        arr(&self.acts, self.arena_len as usize),
                        arr(&layer.ls2, HIDDEN),
                        arr(nw, HIDDEN),
                        arr(nb, HIDDEN),
                        h_off,
                        proj,
                        xn,
                        HIDDEN as u32,
                        TR_EPS as f32,
                        CUBE,
                        true,
                        true,
                    );
                }
            }
        }
        self.pos += t;

        // ── 4. learned downsample → 12.5 Hz ──
        let r_ds = reg(self.r_ds);
        let first = u32::from(self.frames == 0);
        self.conv(
            &self.downsample,
            r_h.view(2),
            0,
            r_ds.write(),
            r_ds.l,
            first,
            false,
            false,
            true,
            false,
        );

        // ── 5. split-RVQ encode ──
        let res = reg(self.r_res).off;
        let mut q = 0u32;
        for bank in [&self.rvq_first, &self.rvq_rest] {
            self.conv(
                &bank.input_proj,
                r_ds.write(),
                0,
                res,
                1,
                0,
                false,
                false,
                false,
                false,
            );
            for qi in 0..bank.n_q {
                let cb_off = qi * bank.rows * CODE_DIM as u32;
                unsafe {
                    rvq_scan_kernel::launch_unchecked::<Rt>(
                        &cl,
                        CubeCount::new_1d(RVQ_CHUNKS),
                        CubeDim::new_1d(CUBE),
                        arr(&self.acts, self.arena_len as usize),
                        arr(&bank.cb, (bank.n_q * bank.rows) as usize * CODE_DIM),
                        arr(&bank.norms, (bank.n_q * bank.rows) as usize),
                        arr(&self.part_d, RVQ_CHUNKS as usize),
                        arru(&self.part_i, RVQ_CHUNKS as usize),
                        res,
                        cb_off,
                        qi * bank.rows,
                        bank.rows,
                        bank.rows.div_ceil(RVQ_CHUNKS),
                        CODE_DIM as u32,
                        CUBE,
                    );
                    rvq_pick_kernel::launch_unchecked::<Rt>(
                        &cl,
                        CubeCount::new_single(),
                        CubeDim::new_1d(CUBE),
                        arr(&self.acts, self.arena_len as usize),
                        arr(&bank.cb, (bank.n_q * bank.rows) as usize * CODE_DIM),
                        arr(&self.part_d, RVQ_CHUNKS as usize),
                        arru(&self.part_i, RVQ_CHUNKS as usize),
                        arru(&self.codes, NUM_CODEBOOKS),
                        res,
                        cb_off,
                        q,
                        CODE_DIM as u32,
                        RVQ_CHUNKS,
                        CUBE,
                    );
                }
                q += 1;
            }
        }

        // ── 6. slide every causal history forward, one dispatch ──
        unsafe {
            shift_kernel::launch_unchecked::<Rt>(
                &cl,
                CubeCount::new_1d(self.n_shift),
                CubeDim::new_1d(CUBE),
                arr(&self.acts, self.arena_len as usize),
                arru(&self.desc, self.n_shift as usize * 4),
                CUBE,
            );
        }
        self.frames += 1;
    }

    /// Blocking readback of the frame's eight codes.
    pub fn read_codes(&self) -> [u32; NUM_CODEBOOKS] {
        use cubecl::CubeElement;
        let bytes = self.client.read_one(self.codes.clone()).expect("readback");
        let v = u32::from_bytes(&bytes);
        let mut out = [0u32; NUM_CODEBOOKS];
        out.copy_from_slice(&v[..NUM_CODEBOOKS]);
        out
    }

    /// Blocking readback of the 12.5 Hz pre-quantizer latent `[512]` — the
    /// continuous stage the codes are an argmin over, and therefore the honest
    /// place to measure agreement with the CPU path.
    pub fn read_latent(&self) -> Vec<f32> {
        use cubecl::CubeElement;
        let bytes = self.client.read_one(self.acts.clone()).expect("readback");
        let v = f32::from_bytes(&bytes);
        let off = self.regions[self.r_ds].write() as usize;
        v[off..off + HIDDEN].to_vec()
    }
}

fn vecs(client: &Client, loader: &WeightLoader, name: &str) -> Handle {
    let (v, _) = loader.load_host_f32(name);
    client.create_from_slice(as_bytes(&v[..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The causal histories each stage of the encoder actually needs, as
    /// `kernel - stride` per consumer. Written out independently of
    /// [`build_arena`] so the test is a check and not a restatement.
    #[test]
    fn every_consumer_history_fits_its_producer_slack() {
        let (a, lay) = build_arena();
        let reg = |i: usize| a.regions[i];

        // stem: k7 stride 1 over the raw input
        assert!(reg(lay.r_in).slack >= 6);

        let mut x = lay.r_stem;
        for (i, &r) in ENC_RATIOS.iter().enumerate() {
            let r = r as u32;
            let rx = reg(x);
            // residual unit's k3 stride 1
            assert!(rx.slack >= 2, "block {i} input slack {} < 2", rx.slack);
            // downsample k=2r stride r
            assert!(rx.slack >= r, "block {i} input slack {} < {r}", rx.slack);
            // the residual branch is added inside the downsample's read, so it
            // needs the same history the downsample does
            assert!(reg(lay.r_block[i][1]).slack >= r);
            // the k1 conv between them keeps no history at all
            assert_eq!(reg(lay.r_block[i][0]).slack, 0);
            x = lay.r_block[i][2];
        }
        // final conv k3 stride 1, then the learned downsample k4 stride 2
        assert!(reg(x).slack >= 2);
        assert!(reg(lay.r_h).slack >= 2);
    }

    /// `shift_kernel` slides rows `[l, l+slack)` down to `[0, slack)` inside
    /// one region; that is only safe while the two ranges are disjoint.
    #[test]
    fn shifted_regions_never_overlap_themselves() {
        let (a, _) = build_arena();
        for r in &a.regions {
            if r.slack > 0 {
                assert!(
                    r.l >= r.slack,
                    "region at {} has l {} < slack {}",
                    r.off,
                    r.l,
                    r.slack
                );
            }
        }
    }

    /// Regions must tile the arena without overlapping — every kernel writes
    /// through one binding, so an overlap would be silent corruption.
    #[test]
    fn regions_tile_the_arena() {
        let (a, _) = build_arena();
        let mut next = 0u32;
        for r in &a.regions {
            assert_eq!(r.off, next, "region gap or overlap at {}", r.off);
            next += r.c * (r.l + r.slack);
        }
        assert_eq!(next, a.len);
    }

    /// SEANet frame lengths: 1920 samples → 2 latents at 25 Hz through the
    /// ratios [4, 5, 6, 8], and one 12.5 Hz code frame after the downsample.
    #[test]
    fn frame_lengths_reach_two_latents() {
        let (a, lay) = build_arena();
        let mut l = SAMPLES_PER_FRAME;
        for (i, &r) in ENC_RATIOS.iter().enumerate() {
            assert_eq!(a.regions[lay.r_block[i][0]].l as usize, l);
            let _ = r;
            l /= r;
            assert_eq!(a.regions[lay.r_block[i][2]].l as usize, l);
        }
        assert_eq!(l, 2);
        assert_eq!(a.regions[lay.r_h].l, 2);
        assert_eq!(a.regions[lay.r_ds].l, 1);
    }

    #[test]
    fn lanes_are_powers_of_two_that_divide_a_cube() {
        for outputs in [1u32, 7, 1024, 2048, 8192, 61440, 122880] {
            let l = MimiEncoderGpu::lanes_for(outputs);
            assert!(
                l.is_power_of_two() && (1..=32).contains(&l),
                "{outputs} -> {l}"
            );
            assert_eq!(CUBE % l, 0);
        }
        // A tall-and-thin convolution splits its reduction; a wide one does not.
        assert_eq!(MimiEncoderGpu::lanes_for(122880), 1);
        assert_eq!(MimiEncoderGpu::lanes_for(2048), 16);
    }

    /// The K/V ring must outlast the window by the two positions a frame writes
    /// before it reads, or the newer of the two aliases the oldest key the
    /// first query still needs.
    #[test]
    fn ring_clears_the_window_plus_one_frame() {
        assert!(RING as usize >= TR_WINDOW + 2);
        assert!(RING.is_power_of_two());
        assert!(SCORE_CAP as usize >= TR_WINDOW);
    }
}
