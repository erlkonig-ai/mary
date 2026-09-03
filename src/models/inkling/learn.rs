//! The resident backward: online learning of the routed experts, on the GPU,
//! against the packed NVFP4 planes the serving path already reads.
//!
//! # What this is
//!
//! The serving path keeps every routed expert as E2M1 codes with one E4M3
//! scale per sixteen elements, in MMA-fragment order, in one arena the GPU
//! reads in place (see `pile::PileSource::copy_share` and
//! `fp4gemm::swizzle_b_codes`). Learning from a served turn means changing
//! THOSE codes, and this module does it without ever holding a gradient
//! tensor: one kernel streams a packed plane once and, for every element on
//! the way past, decodes it, forms its gradient as a dot product over the
//! turn's rows, takes the step, re-encodes against the frozen block scale
//! with unbiased stochastic rounding, and writes the code back. The gradient
//! of a weight exists for the length of one dot product.
//!
//! The off-line prototype (`train_online.rs`) did the same arithmetic on the
//! host with an f32 copy of each expert; it is the reference this module is
//! checked against, not a path anything serves from.
//!
//! # The chain, for the last layer (K = 1)
//!
//! [`learn_last_layer`] runs after a scored pass has produced its per-row
//! scores, while that pass's residual top, the last layer's expert input and
//! its row plan are still live:
//!
//! 1. Head: `g_logits = (softmax - onehot) / n_scored` per scored row, then
//!    `g_hs = g_logits @ U` through a packed TRANSPOSED copy of the unembed
//!    bound once at load ([`Learner::bind`]), so the existing W4A16 GEMM does
//!    the contraction over the vocabulary; then the muP width division and
//!    the rms-norm backward, all as tensor ops on the device.
//! 2. The MLP short convolution's backward: the transposed causal taps over
//!    this pass's rows only (a later turn's rows do not exist yet).
//! 3. The routed experts: the weighted scatter's backward is a gather with
//!    the routing weight ([`expand_rows`]); the plan's slots are grouped by
//!    expert on the device ([`group_by_expert`]) so one cube owns one
//!    expert's rows; [`expert_gin`] on `w2` gives the activation gradient;
//!    the SiLU backward ([`silu_backward`], with `gate_up` recomputed by the
//!    forward's own kernels) gives `w13`'s output gradient; [`expert_update`]
//!    then steps `w2` and `w13` in place.
//!
//! # Two kernels, one tiling
//!
//! For an expert plane `W [n, k]` (output rows, input columns) and the rows
//! `r` routed to that expert, with `g_out[r, o]` the gradient at the plane's
//! output and `x[r, i]` the plane's input:
//!
//! - [`expert_update`]: `W[o, i] -= lr * sum_r g_out[r, o] * x[r, i]`, in
//!   place, on the codes. Registers only.
//! - [`expert_gin`]: `g_in[r, i] = sum_o g_out[r, o] * W[o, i]`, the input
//!   gradient, which contracts over the plane's OUTPUT axis — the direction
//!   no packed GEMM in the tree has; a chunk of rows lives in shared memory.
//!
//! Both walk the plane in its stored (swizzled) order: a 256-byte block per
//! `(n_tile, k_tile)` holding 8 rows x 64 columns as 64 words of 8 codes, and
//! a 32-byte scale block beside it. One cube is one plane of 32 lanes owning
//! four k tiles: lane `L` reads word `L % 8` of k tile `L / 8`, walking every
//! row. So a lane owns 8 input columns for the whole plane, which is what
//! lets [`expert_gin`] accumulate without an atomic or a shuffle: no two lanes
//! ever share an output column.
//!
//! # Rounding
//!
//! A step is almost always far smaller than half an E2M1 gap (the prototype
//! measured ~1% of codes moving per step at lr 1.0, and its nearest-rounding
//! control learned almost nothing), so the update is stochastic rounding with
//! probability proportional to the distance to each neighbour: unbiased in
//! expectation. The coin is a counter hash of the element's arena position
//! and the turn's seed — reproducible, and independent per element.
//!
//! # The arena contract
//!
//! The arena is registered as an immutable alias (`fp4gemm::Aliases`). These
//! kernels are the ONE writer: they run in stream order with everything that
//! reads the arena, and the host never touches the bytes after `copy_share`.
//! Host-immutable, device-mutated in stream order.

use anyhow::Result;
use burn::tensor::{Int, Tensor, TensorData};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::assembly::{BT, Bk, DevRoute, T2, dev_lane};
use super::devplan::{DevRowPlan, ExpertTable};
use super::fp4gemm::{KTILE, NTILE};
use super::fp4quant::e2m1_bits;
use super::moegroup::BlockPlanDev;
use super::seam::{handle_of, handle_of_any, tensor_of};

type Client = ComputeClient<cubecl::cuda::CudaRuntime>;
type Dev = burn::backend::cuda::CudaDevice;

/// Input columns one plane of 32 lanes owns: four k tiles.
pub const LANE_K: usize = 4 * KTILE;
/// Rows of one expert's stack that [`expert_gin`] accumulates per launch.
pub const GIN_ROWS: usize = 32;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/// A 32-bit integer hash (`lowbias32`) to a unit float with 24 random bits.
/// Counter-based: the same input always gives the same draw.
#[cube]
fn hash_unit(x: u32) -> f32 {
    let mut h = x;
    h ^= h >> 16;
    h *= 0x7feb352du32;
    h ^= h >> 15;
    h *= 0x846ca68bu32;
    h ^= h >> 16;
    f32::cast_from(h >> 8) / 16777216.0f32
}

/// One E2M1 code for the code-space value `q` (already divided by the block
/// scale and the expert constant). `u` is the coin in `[0, 1)`; with
/// `stochastic` off it is unused and the nearest code wins.
///
/// Grid magnitudes by code: `0 .5 1 1.5 2 3 4 6`. Beyond 6 the value clips
/// to 6 — the one biased region, which the host reference clips the same way.
#[cube]
fn e2m1_code(q: f32, u: f32, #[comptime] stochastic: bool) -> u32 {
    let a = min(q.abs(), 6.0f32);
    let mut lo = 0u32;
    let mut glo = 0.0f32;
    let mut gap = 0.5f32;
    if a >= 0.5 {
        lo = 1u32;
        glo = 0.5f32;
        gap = 0.5f32;
    }
    if a >= 1.0 {
        lo = 2u32;
        glo = 1.0f32;
        gap = 0.5f32;
    }
    if a >= 1.5 {
        lo = 3u32;
        glo = 1.5f32;
        gap = 0.5f32;
    }
    if a >= 2.0 {
        lo = 4u32;
        glo = 2.0f32;
        gap = 1.0f32;
    }
    if a >= 3.0 {
        lo = 5u32;
        glo = 3.0f32;
        gap = 1.0f32;
    }
    if a >= 4.0 {
        lo = 6u32;
        glo = 4.0f32;
        gap = 2.0f32;
    }
    let mut code = lo;
    if a >= 6.0 {
        code = 7u32;
    } else {
        let p = (a - glo) / gap;
        if comptime![stochastic] {
            if u < p {
                code = lo + 1;
            }
        } else if p > 0.5 {
            code = lo + 1;
        }
    }
    if q < 0.0 && code != 0 {
        code |= 8u32;
    }
    code
}

/// E4M3 byte to f32. Subnormal: `2^-6 * m/8`; normal: `2^(e-7) * (1 + m/8)`.
/// The exponent-all-ones NaN pattern never appears in a block scale the
/// quantiser wrote (it caps at 448), so it is not special-cased.
#[cube]
fn e4m3_bits_to_f32(b: u32) -> f32 {
    let sign = (b >> 7) & 1u32;
    let e = (b >> 3) & 0xfu32;
    let m = b & 7u32;
    let mut mag = f32::cast_from(m) / 512.0f32;
    if e != 0 {
        let scale = f32::reinterpret((120u32 + e) << 23);
        mag = scale * (1.0f32 + f32::cast_from(m) / 8.0f32);
    }
    if sign != 0 {
        mag = -mag;
    }
    mag
}

/// Word index of `(row c, word w)` inside a swizzled 256-byte block; the
/// device twin of `fp4gemm::swz_word`.
#[cube]
fn swz_word(c: usize, w: usize) -> usize {
    (w / 4) * 32 + c * 4 + (w % 4)
}

// ---------------------------------------------------------------------------
// The two streaming kernels
// ---------------------------------------------------------------------------

/// In-place SGD step on one packed plane per expert, on the codes.
///
/// Cube = one plane of 32 lanes; `CUBE_POS_X` picks its four k tiles,
/// `CUBE_POS_Y` the expert. `w` is the arena as `u32` words; `off` holds each
/// expert's `[codes, scales]` byte offsets into it and `scale2` its constant
/// (the layer's table, indexed by expert id). The expert's rows are
/// `rows[start[e] .. start[e] + cnt[e]]`, indices into the `[m_total, _]`
/// stacks `g_out` and `x`; `x` is read eight columns at a time.
#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
pub fn expert_update<G: Scalar + Cast, X: Scalar + Cast>(
    w: &mut Array<u32>,
    off: &Array<u64>,
    scale2: &Array<f32>,
    grp_start: &Array<u32>,
    grp_cnt: &Array<u32>,
    rows: &Array<u32>,
    g_out: &Array<G>,
    x: &Array<Vector<X, Const<8>>>,
    lr: f32,
    seed: u32,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] stochastic: bool,
) {
    let lane = UNIT_POS as usize;
    let e = CUBE_POS_Y as usize;
    let cnt = grp_cnt[e] as usize;
    if cnt == 0 {
        terminate!();
    }
    let start = grp_start[e] as usize;
    let t = (CUBE_POS_X as usize) * 4 + lane / 8;
    let wd = lane % 8;
    let k_tiles = comptime!(size_k / KTILE);
    let n_tiles = comptime!(size_n / NTILE);

    let b_base = usize::cast_from(off[2 * e]);
    let bsc_base = usize::cast_from(off[2 * e + 1]);
    let sc2 = scale2[e];
    let col8 = (t * KTILE + wd * 8) / 8;
    let k8 = comptime!(size_k / 8);

    for nt in 0..n_tiles {
        let blk_word = ((nt * k_tiles) + t) * 64;
        let blk_sbyte = ((nt * k_tiles) + t) * 32;
        for c in 0..NTILE {
            let o = nt * NTILE + c;
            let widx = (b_base / 4) + blk_word + swz_word(c, wd);
            let sbyte = bsc_base + blk_sbyte + c * 4 + wd / 2;
            let sword = w[sbyte / 4];
            let sbits = (sword >> u32::cast_from((sbyte % 4) * 8)) & 0xffu32;
            let s = e4m3_bits_to_f32(sbits) * sc2;
            let word = w[widx];

            let mut grad = Array::<f32>::new(8usize);
            #[unroll]
            for j in 0..8usize {
                grad[j] = 0.0f32;
            }
            for i in 0..cnt {
                let r = rows[start + i] as usize;
                let g = f32::cast_from(g_out[r * size_n + o]);
                let xv = Vector::<f32, Const<8>>::cast_from(x[r * k8 + col8]);
                #[unroll]
                for j in 0..8usize {
                    grad[j] += g * xv[j];
                }
            }

            let mut out = 0u32;
            if s > 0.0 {
                let inv = 1.0f32 / s;
                #[unroll]
                for j in 0..8usize {
                    let code = (word >> (4 * j as u32)) & 0xfu32;
                    let v = e2m1_bits(code) * s;
                    let q = (v - lr * grad[j]) * inv;
                    let u = hash_unit(u32::cast_from(widx) * 8u32 + u32::cast_from(j) + seed);
                    out |= e2m1_code(q, u, stochastic) << (4 * j as u32);
                }
            }
            w[widx] = out;
        }
    }
}

/// The input gradient of one packed plane per expert:
/// `g_in[r, i] = sum_o g_out[r, o] * W[o, i]` for a chunk of [`GIN_ROWS`] of
/// the expert's rows, `CUBE_POS_Z` choosing the chunk. Cube = one plane; the
/// chunk's partial sums live in shared memory, one lane owning each column.
#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
pub fn expert_gin<G: Scalar + Cast>(
    w: &Array<u32>,
    off: &Array<u64>,
    scale2: &Array<f32>,
    grp_start: &Array<u32>,
    grp_cnt: &Array<u32>,
    rows: &Array<u32>,
    g_out: &Array<G>,
    g_in: &mut Array<f32>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
) {
    let lane = UNIT_POS as usize;
    let e = CUBE_POS_Y as usize;
    let chunk = CUBE_POS_Z as usize;
    let cnt = grp_cnt[e] as usize;
    let first = chunk * GIN_ROWS;
    if first >= cnt {
        terminate!();
    }
    let mut n_rows = GIN_ROWS;
    if first + n_rows > cnt {
        n_rows = cnt - first;
    }
    let start = grp_start[e] as usize + first;
    let t = (CUBE_POS_X as usize) * 4 + lane / 8;
    let wd = lane % 8;
    let k_tiles = comptime!(size_k / KTILE);
    let n_tiles = comptime!(size_n / NTILE);

    let b_base = usize::cast_from(off[2 * e]);
    let bsc_base = usize::cast_from(off[2 * e + 1]);
    let sc2 = scale2[e];

    // `[GIN_ROWS][LANE_K]`, this lane's eight columns at `lane * 8`. Nothing
    // else ever touches them.
    let mut acc = SharedMemory::<f32>::new(comptime!(GIN_ROWS * LANE_K));
    for i in 0..GIN_ROWS {
        #[unroll]
        for j in 0..8usize {
            acc[i * LANE_K + lane * 8 + j] = 0.0f32;
        }
    }

    for nt in 0..n_tiles {
        let blk_word = ((nt * k_tiles) + t) * 64;
        let blk_sbyte = ((nt * k_tiles) + t) * 32;
        for c in 0..NTILE {
            let o = nt * NTILE + c;
            let widx = (b_base / 4) + blk_word + swz_word(c, wd);
            let sbyte = bsc_base + blk_sbyte + c * 4 + wd / 2;
            let sword = w[sbyte / 4];
            let sbits = (sword >> u32::cast_from((sbyte % 4) * 8)) & 0xffu32;
            let s = e4m3_bits_to_f32(sbits) * sc2;
            let word = w[widx];
            let mut v = Array::<f32>::new(8usize);
            #[unroll]
            for j in 0..8usize {
                v[j] = e2m1_bits((word >> (4 * j as u32)) & 0xfu32) * s;
            }
            for i in 0..n_rows {
                let r = rows[start + i] as usize;
                let g = f32::cast_from(g_out[r * size_n + o]);
                #[unroll]
                for j in 0..8usize {
                    acc[i * LANE_K + lane * 8 + j] += g * v[j];
                }
            }
        }
    }

    let col = t * KTILE + wd * 8;
    for i in 0..n_rows {
        let r = rows[start + i] as usize;
        #[unroll]
        for j in 0..8usize {
            g_in[r * size_k + col + j] = acc[i * LANE_K + lane * 8 + j];
        }
    }
}

// ---------------------------------------------------------------------------
// The small kernels around them
// ---------------------------------------------------------------------------

/// Group the plan's slots by expert: one cube of `n_routed` units, unit `e`
/// counting its slots, taking its start as the sum of the counts before it,
/// and writing its slots in plan order. Three passes over `ids` per unit and
/// no atomics; `ids` is a few thousand entries at most.
///
/// `rows[start[e] + i]` is the STACK ROW of the slot — `slot * tile` — which
/// is where the plan puts the one real row of a slot's tile.
#[cube(launch)]
pub fn group_by_expert(
    ids: &Array<u32>,
    grp_start: &mut Array<u32>,
    grp_cnt: &mut Array<u32>,
    rows: &mut Array<u32>,
    slots: usize,
    tile: usize,
    #[comptime] n_routed: usize,
) {
    let e = UNIT_POS as usize;
    let mut cnt = 0u32;
    for s in 0..slots {
        if ids[s] as usize == e {
            cnt += 1;
        }
    }
    let mut counts = SharedMemory::<u32>::new(comptime!(n_routed));
    counts[e] = cnt;
    sync_cube();
    let mut start = 0u32;
    for f in 0..e {
        start += counts[f];
    }
    grp_start[e] = start;
    grp_cnt[e] = cnt;
    let mut at = start as usize;
    for s in 0..slots {
        if ids[s] as usize == e {
            rows[at] = u32::cast_from(s * tile);
            at += 1;
        }
    }
}

/// The weighted scatter's backward: the row of a slot's tile gets its
/// token's gradient times its routing weight; every other stacked row is
/// zero. `g_tok` is `[n, h]`, the output `[m_total, h]`.
#[cube(launch)]
pub fn expand_rows(
    g_tok: &Array<f32>,
    row_tok: &Array<i32>,
    row_wgt: &Array<f32>,
    out: &mut Array<f32>,
    h: usize,
    total: usize,
) {
    let p = ABSOLUTE_POS as usize;
    if p < total {
        let r = p / h;
        let c = p % h;
        let tok = row_tok[r];
        let mut v = 0.0f32;
        if tok >= 0 {
            v = g_tok[(tok as usize) * h + c] * row_wgt[r];
        }
        out[p] = v;
    }
}

/// The SiLU gate's backward and its forward again: from the interleaved
/// gate-and-up `[m, 2*inter]` and the activation gradient `[m, inter]`, the
/// interleaved gradient `g_both` and the activation `act = silu(gate) * up`
/// (which the `w2` step needs as its input). One unit per `(row, i)`.
#[cube(launch)]
pub fn silu_backward<E: Scalar + Cast, O: Scalar + Cast>(
    both: &Array<E>,
    g_act: &Array<f32>,
    g_both: &mut Array<O>,
    act: &mut Array<O>,
    #[comptime] inter: usize,
    total: usize,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx < total {
        let r = idx / inter;
        let i = idx % inter;
        let g = f32::cast_from(both[r * 2 * inter + 2 * i]);
        let u = f32::cast_from(both[r * 2 * inter + 2 * i + 1]);
        let sig = 1.0f32 / (1.0f32 + Exp::exp(-g));
        let silu = g * sig;
        let dsilu = sig * (1.0f32 + g * (1.0f32 - sig));
        let ga = g_act[idx];
        g_both[r * 2 * inter + 2 * i] = O::cast_from(ga * u * dsilu);
        g_both[r * 2 * inter + 2 * i + 1] = O::cast_from(ga * silu);
        act[idx] = O::cast_from(silu * u);
    }
}

/// `dst[c, r] = src[r, c]` over `[rows, cols]` elements of any scalar type.
#[cube(launch)]
pub fn transpose_2d<E: Scalar>(src: &Array<E>, dst: &mut Array<E>, rows: usize, cols: usize) {
    let p = ABSOLUTE_POS as usize;
    if p < rows * cols {
        let r = p / cols;
        let c = p % cols;
        dst[c * rows + r] = src[p];
    }
}

const CUBE: u32 = 256;

fn cubes_for(n: usize) -> CubeCount {
    CubeCount::new_1d(n.div_ceil(CUBE as usize) as u32)
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

/// Everything the learning pass needs from one layer's forward, kept alive
/// by `moe_layer` when learning is armed for that layer.
pub struct LearnKeep {
    pub layer: usize,
    /// The expert input, `[n, h]`: the MLP norm's output.
    pub hn: T2,
    /// The plan of that pass, in plan order.
    pub dp: DevRowPlan,
    /// The MLP short convolution's taps, `[h, kernel]`, filled in by the
    /// session (the layer's device state is not `moe_layer`'s to keep).
    pub sconv: Option<T2>,
}

/// What a session holds to learn: the step and the transposed head.
pub struct Learner {
    pub lr: f32,
    pub stochastic: bool,
    /// The unembedding, transposed to `[h, vocab_pad]` and packed, so that
    /// `g_logits @ U` is one W4A16 GEMM with the vocabulary as its reduction.
    pub unembed_t: dev_lane::ProjW,
    /// Turns learned from on this session, and the seed of the next coin.
    pub steps: u32,
}

impl Learner {
    /// `INK_LEARN_LR=<f32>` arms learning of the last layer's routed experts
    /// at that step; unset or zero leaves the session as it was.
    /// `INK_LEARN_RN=1` swaps stochastic rounding for nearest (the control).
    pub fn from_env() -> Option<(f32, bool)> {
        let lr: f32 = std::env::var("INK_LEARN_LR").ok()?.parse().ok()?;
        (lr > 0.0).then(|| {
            let rn = std::env::var("INK_LEARN_RN").map(|v| v == "1").unwrap_or(false);
            (lr, !rn)
        })
    }

    /// Bind the transposed head from the stored BF16 unembedding
    /// `[vocab_pad, h]`: transposed on the device, quantised on the device.
    pub fn bind(client: &Client, unembed_bf16: &[u8], vocab_pad: usize, h: usize, lr: f32, stochastic: bool) -> Self {
        use super::assembly::w4a16_bind;
        assert_eq!(unembed_bf16.len(), vocab_pad * h * 2);
        let src = client.create_from_slice(unembed_bf16);
        let dst = client.empty(unembed_bf16.len());
        unsafe {
            transpose_2d::launch::<half::bf16, cubecl::cuda::CudaRuntime>(
                client,
                cubes_for(vocab_pad * h),
                CubeDim::new_1d(CUBE),
                ArrayArg::from_raw_parts(src.clone(), vocab_pad * h),
                ArrayArg::from_raw_parts(dst.clone(), vocab_pad * h),
                vocab_pad,
                h,
            )
        };
        let (codes, scales) = super::fp4quant::quantize_nvfp4_bf16(client, &dst, h, vocab_pad);
        let packed = dev_lane::PackedW {
            codes,
            scales,
            n: h,
            k: vocab_pad,
            scale2: 1.0,
            swizzled: false,
        };
        Self {
            lr,
            stochastic,
            unembed_t: w4a16_bind(client, packed, true),
            steps: 0,
        }
    }
}

/// What one learning pass did.
#[derive(Debug, Clone, Copy)]
pub struct LearnReport {
    pub layer: usize,
    pub rows: usize,
    pub slots: usize,
    pub secs: f64,
}

/// The rms-norm backward: `y = x * r * gain`, `r = (mean(x^2) + eps)^-1/2`.
/// `g_x = r * u - x * r^3 * mean(u * x)` with `u = g_y * gain`.
fn rms_norm_backward(x: T2, gain: BT<Bk, 1>, g_y: T2, eps: f64) -> T2 {
    let u = g_y * gain.unsqueeze::<2>();
    let r = x.clone().powf_scalar(2.0).mean_dim(1).add_scalar(eps).sqrt().recip();
    let dot = (u.clone() * x.clone()).mean_dim(1);
    u * r.clone() - x * r.clone().powf_scalar(3.0) * dot
}

/// The short convolution's backward over this pass's rows: with the forward
/// `out[t] = x[t] + sum_j w[:, j] * x[t - (kernel - 1) + j]`, the input
/// gradient is `g_x[t] = g_out[t] + sum_j w[:, j] * g_out[t + (kernel - 1 - j)]`
/// for the rows that exist.
fn short_conv_backward(g_out: T2, weight: T2) -> T2 {
    let [rows, dim] = g_out.dims();
    let [_, kernel] = weight.dims();
    let dev = g_out.device();
    let mut g_x = g_out.clone();
    for j in 0..kernel {
        let s = kernel - 1 - j;
        let shifted = if s == 0 {
            g_out.clone()
        } else if s >= rows {
            continue;
        } else {
            Tensor::cat(
                vec![g_out.clone().slice([s..rows, 0..dim]), Tensor::zeros([s, dim], &dev)],
                0,
            )
        };
        let tap = weight.clone().slice([0..dim, j..j + 1]).reshape([1, dim]);
        g_x = g_x + shifted * tap;
    }
    g_x
}

/// Learn the last layer's routed experts from one scored pass.
///
/// `xd` is the pass's residual top `[n, h]`; `targets[r]` the id row `r` was
/// scored against (rows past `targets.len()` were not scored and carry no
/// gradient); `head` computes the head's logits `[rows, vocab_eff]` from a
/// slice of `xd` exactly as the scored pass did.
#[allow(clippy::too_many_arguments)]
pub fn learn_last_layer(
    client: &Client,
    dev: &Dev,
    learner: &mut Learner,
    keep: LearnKeep,
    route: &DevRoute,
    tab: &ExpertTable,
    swz: bool,
    xd: &T2,
    final_norm: &BT<Bk, 1>,
    head: &dyn Fn(T2) -> T2,
    targets: &[usize],
    mup: f32,
    eps: f64,
    vocab_eff: usize,
    vocab_pad: usize,
    n_routed: usize,
    inter: usize,
    slice_rows: usize,
) -> Result<LearnReport> {
    use super::fp4quant::quantize_nvfp4_bf16;
    use super::moegroup::{fp4_linear_grouped_bf16_launch, gather_grouped_bf16_from_bf16};
    let t0 = std::time::Instant::now();
    let [n, h] = xd.dims();
    let scored = targets.len();
    anyhow::ensure!(scored > 0 && scored <= n, "a learning pass wants scored rows");
    let sconv = keep.sconv.as_ref().expect("the session fills in the last layer's taps");

    // ---- 1. head backward, in row slices ----------------------------------
    let mut g_hs: T2 = Tensor::zeros([n, h], dev);
    let inv = 1.0 / scored as f32;
    let mut lo = 0;
    while lo < scored {
        let hi = (lo + slice_rows).min(scored);
        let rows = hi - lo;
        let logits = head(xd.clone().slice([lo..hi, 0..h]));
        let idx: Vec<i64> = targets[lo..hi].iter().map(|&t| t as i64).collect();
        let idx: Tensor<Bk, 2, Int> = Tensor::from_data(TensorData::new(idx, [rows, 1]), dev);
        let minus: T2 = Tensor::full([rows, 1], -1.0, dev);
        let g = burn::tensor::activation::softmax(logits, 1)
            .scatter(1, idx, minus)
            .mul_scalar(inv);
        let g_full: T2 = Tensor::zeros([rows, vocab_pad], dev).slice_assign([0..rows, 0..vocab_eff], g);
        let g_slice = dev_lane::linear_w(g_full, &learner.unembed_t);
        g_hs = g_hs.slice_assign([lo..hi, 0..h], g_slice);
        lo = hi;
    }
    // hs = rms_norm(xd) / mup, so d/d(rms) = g_hs / mup.
    let g_rms = g_hs.div_scalar(mup);
    let x = xd.clone().cast(burn::tensor::DType::F32);
    let g_xd = rms_norm_backward(x, final_norm.clone(), g_rms, eps);

    // ---- 2. the MLP short convolution's backward ---------------------------
    let g_moe = short_conv_backward(g_xd, sconv.clone());

    // ---- 3. the routed experts ---------------------------------------------
    let slots = route.k;
    let m_total = route.m_total;
    let tile = super::fp4gemm::MTILE;
    let g_tok = handle_of(g_moe);
    let g_rows = client.empty(m_total * h * 4);
    unsafe {
        expand_rows::launch::<cubecl::cuda::CudaRuntime>(
            client,
            cubes_for(m_total * h),
            CubeDim::new_1d(CUBE),
            ArrayArg::from_raw_parts(g_tok.clone(), n * h),
            ArrayArg::from_raw_parts(route.row_tok.clone(), m_total),
            ArrayArg::from_raw_parts(keep.dp.row_wgt.clone(), m_total),
            ArrayArg::from_raw_parts(g_rows.clone(), m_total * h),
            h,
            m_total * h,
        )
    };

    let grp_start = client.empty(n_routed * 4);
    let grp_cnt = client.empty(n_routed * 4);
    let rows = client.empty(slots.max(1) * 4);
    unsafe {
        group_by_expert::launch::<cubecl::cuda::CudaRuntime>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(n_routed as u32),
            ArrayArg::from_raw_parts(keep.dp.ids.clone(), slots),
            ArrayArg::from_raw_parts(grp_start.clone(), n_routed),
            ArrayArg::from_raw_parts(grp_cnt.clone(), n_routed),
            ArrayArg::from_raw_parts(rows.clone(), slots.max(1)),
            slots,
            tile,
            n_routed,
        )
    };
    let groups = Groups {
        start: &grp_start,
        cnt: &grp_cnt,
        rows: &rows,
        n_routed,
        max_rows: slots,
    };

    // w2: [h, inter]. Its input gradient first, against the pre-step codes.
    let w2 = PlaneRef {
        wmap: &tab.wmap,
        wmap_bytes: tab.wmap_bytes,
        off: &tab.off2,
        scale2: &tab.sc2,
    };
    let g_act = handle_of(Tensor::<Bk, 2>::zeros([m_total, inter], dev));
    expert_gin_launch::<f32>(client, &w2, &groups, &g_rows, &g_act, m_total, h, inter);

    // The forward's own kernels, again, for gate_up: gather, quantise, w13.
    let (hn_h, _hn_dt) = handle_of_any(keep.hn.clone());
    let x_h = gather_grouped_bf16_from_bf16(client, &hn_h, &route.row_tok, n, m_total, h);
    let (a, asc) = quantize_nvfp4_bf16(client, &x_h, m_total, h);
    let blk = BlockPlanDev {
        slot: route.blk_slot.clone(),
        tile0: route.blk_tile0.clone(),
        cnt: route.blk_cnt.clone(),
        blocks: route.k,
        planes: route.planes,
        rows_real: route.k,
    };
    let both = fp4_linear_grouped_bf16_launch(
        client, &a, &asc, &tab.wmap, tab.wmap_bytes, &blk, &keep.dp.off13, &keep.dp.sc13, slots,
        m_total, h, 2 * inter, swz,
    );
    let g_both = client.empty(m_total * 2 * inter * 2);
    let act = client.empty(m_total * inter * 2);
    unsafe {
        silu_backward::launch::<half::bf16, half::bf16, cubecl::cuda::CudaRuntime>(
            client,
            cubes_for(m_total * inter),
            CubeDim::new_1d(CUBE),
            ArrayArg::from_raw_parts(both.clone(), m_total * 2 * inter),
            ArrayArg::from_raw_parts(g_act.clone(), m_total * inter),
            ArrayArg::from_raw_parts(g_both.clone(), m_total * 2 * inter),
            ArrayArg::from_raw_parts(act.clone(), m_total * inter),
            inter,
            m_total * inter,
        )
    };

    // The steps. w2 sees g_rows (f32) and act (bf16); w13 sees g_both (bf16)
    // and the gathered expert input (bf16).
    learner.steps += 1;
    let seed = learner.steps.wrapping_mul(0x9E37_79B1);
    expert_update_launch::<f32, half::bf16>(
        client, &w2, &groups, &g_rows, &act, m_total, h, inter, learner.lr, seed, learner.stochastic,
    );
    let w13 = PlaneRef {
        wmap: &tab.wmap,
        wmap_bytes: tab.wmap_bytes,
        off: &tab.off13,
        scale2: &tab.sc13,
    };
    expert_update_launch::<half::bf16, half::bf16>(
        client, &w13, &groups, &g_both, &x_h, m_total, 2 * inter, h, learner.lr, seed ^ 0x5555_5555,
        learner.stochastic,
    );
    // The learning pass is enqueued; its kernels run in stream order before
    // the next pass reads the arena. Nothing here waits.
    let _ = tensor_of;
    Ok(LearnReport {
        layer: keep.layer,
        rows: scored,
        slots,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// One packed plane per expert, named by the layer's table.
pub struct PlaneRef<'a> {
    pub wmap: &'a Handle,
    pub wmap_bytes: usize,
    /// `[n_routed * 2]` u64: `[codes, scales]` byte offsets per expert.
    pub off: &'a Handle,
    /// `[n_routed]` f32.
    pub scale2: &'a Handle,
}

/// The plan's slots grouped by expert, from [`group_by_expert`].
pub struct Groups<'a> {
    pub start: &'a Handle,
    pub cnt: &'a Handle,
    pub rows: &'a Handle,
    pub n_routed: usize,
    /// An upper bound on any expert's row count (the slot count).
    pub max_rows: usize,
}

/// Launch [`expert_update`] over every expert: `[n, k]` planes, `g_out
/// [m_total, n]` at `G` and `x [m_total, k]` at `X`.
#[allow(clippy::too_many_arguments)]
pub fn expert_update_launch<G: Scalar + Cast + CubeElement, X: Scalar + Cast + CubeElement>(
    client: &Client,
    plane: &PlaneRef<'_>,
    groups: &Groups<'_>,
    g_out: &Handle,
    x: &Handle,
    m_total: usize,
    n: usize,
    k: usize,
    lr: f32,
    seed: u32,
    stochastic: bool,
) {
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % LANE_K, 0, "k {k} is not a multiple of {LANE_K}");
    let words = plane.wmap_bytes / 4;
    unsafe {
        expert_update::launch::<G, X, cubecl::cuda::CudaRuntime>(
            client,
            CubeCount::Static((k / LANE_K) as u32, groups.n_routed as u32, 1),
            CubeDim::new_1d(32),
            AddressType::U64,
            ArrayArg::from_raw_parts(plane.wmap.clone(), words),
            ArrayArg::from_raw_parts(plane.off.clone(), 2 * groups.n_routed),
            ArrayArg::from_raw_parts(plane.scale2.clone(), groups.n_routed),
            ArrayArg::from_raw_parts(groups.start.clone(), groups.n_routed),
            ArrayArg::from_raw_parts(groups.cnt.clone(), groups.n_routed),
            ArrayArg::from_raw_parts(groups.rows.clone(), groups.max_rows.max(1)),
            ArrayArg::from_raw_parts(g_out.clone(), m_total * n),
            ArrayArg::from_raw_parts(x.clone(), m_total * k / 8),
            lr,
            seed,
            k,
            n,
            stochastic,
        )
    };
}

/// Launch [`expert_gin`] over every expert into `g_in` (f32 `[m_total, k]`,
/// which the caller zeroed: rows no expert owns are never written).
#[allow(clippy::too_many_arguments)]
pub fn expert_gin_launch<G: Scalar + Cast + CubeElement>(
    client: &Client,
    plane: &PlaneRef<'_>,
    groups: &Groups<'_>,
    g_out: &Handle,
    g_in: &Handle,
    m_total: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % LANE_K, 0, "k {k} is not a multiple of {LANE_K}");
    let words = plane.wmap_bytes / 4;
    let chunks = groups.max_rows.div_ceil(GIN_ROWS).max(1);
    unsafe {
        expert_gin::launch::<G, cubecl::cuda::CudaRuntime>(
            client,
            CubeCount::Static((k / LANE_K) as u32, groups.n_routed as u32, chunks as u32),
            CubeDim::new_1d(32),
            AddressType::U64,
            ArrayArg::from_raw_parts(plane.wmap.clone(), words),
            ArrayArg::from_raw_parts(plane.off.clone(), 2 * groups.n_routed),
            ArrayArg::from_raw_parts(plane.scale2.clone(), groups.n_routed),
            ArrayArg::from_raw_parts(groups.start.clone(), groups.n_routed),
            ArrayArg::from_raw_parts(groups.cnt.clone(), groups.n_routed),
            ArrayArg::from_raw_parts(groups.rows.clone(), groups.max_rows.max(1)),
            ArrayArg::from_raw_parts(g_out.clone(), m_total * n),
            ArrayArg::from_raw_parts(g_in.clone(), m_total * k),
            k,
            n,
        )
    };
}
