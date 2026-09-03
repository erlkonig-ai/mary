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
//! # Two kernels, one tiling
//!
//! For an expert plane `W [n, k]` (output rows, input columns) and the rows
//! `r` the plan stacked for that expert, with `g_out[r, o]` the gradient at
//! the plane's output and `x[r, i]` the plane's input:
//!
//! - [`expert_update`]: `W[o, i] -= lr * sum_r g_out[r, o] * x[r, i]`, in
//!   place, on the codes. Registers only.
//! - [`expert_gin`]: `g_in[r, i] = sum_o g_out[r, o] * W[o, i]`, the input
//!   gradient, which the next plane down (and the SiLU between `w2` and `w13`)
//!   needs. It contracts over the plane's OUTPUT axis, which no packed GEMM in
//!   the tree does; the row chunk it accumulates lives in shared memory.
//!
//! Both walk the plane in its stored (swizzled) order: a 256-byte block per
//! `(n_tile, k_tile)` holding 8 rows x 64 columns as 64 words of 8 codes, and
//! a 32-byte scale block beside it. One plane (32 lanes) owns four k tiles:
//! lane `L` reads word `w = L % 8` of k tile `L / 8`, walking every row of the
//! plane. So a lane owns 8 consecutive input columns for the whole plane,
//! which is what lets [`expert_gin`] accumulate without a single atomic or
//! shuffle: no two lanes ever share an output column.
//!
//! # Rounding
//!
//! A step is almost always far smaller than half an E2M1 gap (`train_online`
//! measured ~1% of codes moving per step at lr 1.0), so round-to-nearest
//! would learn nothing; the update is stochastic rounding with probability
//! proportional to the distance to each neighbour, which is unbiased in
//! expectation. The coin is a counter hash of the element's arena position
//! and the caller's seed, so a step is reproducible given its seed and every
//! element draws independently. `stochastic = false` is the nearest-rounding
//! control.
//!
//! # The arena contract
//!
//! The arena is registered as an immutable alias (`fp4gemm::Aliases`). These
//! kernels are the ONE writer: they run in stream order with everything that
//! reads the arena, and the host never touches the bytes after `copy_share`.
//! That is the contract as it now stands — host-immutable, device-mutated in
//! stream order — and it is written at the registration site too.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::fp4gemm::{GROUP, KTILE, NTILE};
use super::fp4quant::e2m1_bits;
use super::moegroup::BlockPlanDev;

/// Input columns one plane of 32 lanes owns: four k tiles.
pub const LANE_K: usize = 4 * KTILE;
/// Rows of one expert's stack that [`expert_gin`] accumulates per launch.
pub const GIN_ROWS: usize = 32;

/// A 32-bit integer hash (Chris Wellons' `lowbias32`) to a unit float with 24
/// random bits. Counter-based: the same input always gives the same draw.
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
/// The grid magnitudes by code are `0 .5 1 1.5 2 3 4 6`; beyond 6 the value
/// clips to 6, the one biased region, which the host reference clips the same
/// way and counts.
#[cube]
fn e2m1_code(q: f32, u: f32, #[comptime] stochastic: bool) -> u32 {
    let a = min(q.abs(), 6.0f32);
    // Neighbours on the grid: the code below `a` and its magnitude, and the
    // gap to the code above. Seven comparisons, like the encoder's ladder.
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

/// Word index of `(row c, word w)` inside a swizzled 256-byte block; the
/// device twin of `fp4gemm::swz_word`.
#[cube]
fn swz_word(c: usize, w: usize) -> usize {
    (w / 4) * 32 + c * 4 + (w % 4)
}

/// In-place SGD step on one packed plane per expert run, on the codes.
///
/// Cube = one plane of 32 lanes. `CUBE_POS_X` picks the lane group's four k
/// tiles, `CUBE_POS_Y` the expert run. `w` is the arena as `u32` words; `off`
/// holds each slot's `[codes, scales]` byte offsets into it; `scale2` the
/// per-slot constant. `g_out` is `[m_total, n]`, `x` is `[m_total, k]` read
/// eight columns at a time; both BF16 (or f32) with padded rows zero.
#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
pub fn expert_update<E: Scalar + Cast>(
    w: &mut Array<u32>,
    off: &Array<u64>,
    scale2: &Array<f32>,
    blk_slot: &Array<u32>,
    blk_tile0: &Array<u32>,
    blk_cnt: &Array<u32>,
    g_out: &Array<E>,
    x: &Array<Vector<E, Const<8>>>,
    lr: f32,
    seed: u32,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] stochastic: bool,
) {
    let lane = UNIT_POS as usize;
    let blk = CUBE_POS_Y as usize;
    let t = (CUBE_POS_X as usize) * 4 + lane / 8;
    let wd = lane % 8;
    let k_tiles = comptime!(size_k / KTILE);
    let n_tiles = comptime!(size_n / NTILE);

    let slot = blk_slot[blk] as usize;
    let b_base = usize::cast_from(off[2 * slot]);
    let bsc_base = usize::cast_from(off[2 * slot + 1]);
    let sc2 = scale2[slot];
    let m0 = (blk_tile0[blk] as usize) * 16;
    let m1 = m0 + (blk_cnt[blk] as usize) * 16;
    // This lane's eight input columns, as an index into `x`'s 8-wide vectors.
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
            let s = f32::cast_from(e4m3_bits_to_f32(sbits)) * sc2;
            let word = w[widx];

            // The gradient of each of the eight elements: a dot product over
            // the expert's rows, `g_out[r, o] * x[r, col..col+8]`.
            let mut grad = Array::<f32>::new(8usize);
            #[unroll]
            for j in 0..8usize {
                grad[j] = 0.0f32;
            }
            for r in m0..m1 {
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

/// The input gradient of one packed plane per expert run:
/// `g_in[r, i] = sum_o g_out[r, o] * W[o, i]` for a chunk of [`GIN_ROWS`]
/// rows of the run, `CUBE_POS_Z` choosing the chunk. Cube = one plane; the
/// chunk's partial sums live in shared memory, one lane owning each column.
#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
pub fn expert_gin<E: Scalar + Cast>(
    w: &Array<u32>,
    off: &Array<u64>,
    scale2: &Array<f32>,
    blk_slot: &Array<u32>,
    blk_tile0: &Array<u32>,
    blk_cnt: &Array<u32>,
    g_out: &Array<E>,
    g_in: &mut Array<f32>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
) {
    let lane = UNIT_POS as usize;
    let blk = CUBE_POS_Y as usize;
    let chunk = CUBE_POS_Z as usize;
    let t = (CUBE_POS_X as usize) * 4 + lane / 8;
    let wd = lane % 8;
    let k_tiles = comptime!(size_k / KTILE);
    let n_tiles = comptime!(size_n / NTILE);

    let m0 = (blk_tile0[blk] as usize) * 16 + chunk * GIN_ROWS;
    let m_end = (blk_tile0[blk] as usize) * 16 + (blk_cnt[blk] as usize) * 16;
    if m0 >= m_end {
        terminate!();
    }
    let mut m1 = m0 + GIN_ROWS;
    if m1 > m_end {
        m1 = m_end;
    }
    let rows = m1 - m0;

    let slot = blk_slot[blk] as usize;
    let b_base = usize::cast_from(off[2 * slot]);
    let bsc_base = usize::cast_from(off[2 * slot + 1]);
    let sc2 = scale2[slot];

    // `[GIN_ROWS][LANE_K]`: row-major, this lane's eight columns at
    // `lane * 8`. Nothing else ever touches them.
    let mut acc = SharedMemory::<f32>::new(comptime!(GIN_ROWS * LANE_K));
    for r in 0..GIN_ROWS {
        #[unroll]
        for j in 0..8usize {
            acc[r * LANE_K + lane * 8 + j] = 0.0f32;
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
            let s = f32::cast_from(e4m3_bits_to_f32(sbits)) * sc2;
            let word = w[widx];
            let mut v = Array::<f32>::new(8usize);
            #[unroll]
            for j in 0..8usize {
                v[j] = e2m1_bits((word >> (4 * j as u32)) & 0xfu32) * s;
            }
            for r in 0..rows {
                let g = f32::cast_from(g_out[(m0 + r) * size_n + o]);
                #[unroll]
                for j in 0..8usize {
                    acc[r * LANE_K + lane * 8 + j] += g * v[j];
                }
            }
        }
    }

    let col = t * KTILE + wd * 8;
    for r in 0..rows {
        #[unroll]
        for j in 0..8usize {
            g_in[(m0 + r) * size_k + col + j] = acc[r * LANE_K + lane * 8 + j];
        }
    }
}

/// E4M3 byte to f32: `sign * 2^(e-7) * (1 + m/8)`, subnormals at `2^-6 * m/8`.
/// The exponent-all-ones NaN encoding of E4M3 never appears in a block scale
/// the quantiser wrote (it caps at 448), so it decodes as the finite value it
/// would otherwise be rather than being special-cased.
#[cube]
fn e4m3_bits_to_f32(b: u32) -> f32 {
    let sign = (b >> 7) & 1u32;
    let e = (b >> 3) & 0xfu32;
    let m = b & 7u32;
    // Subnormal: `2^-6 * m/8 = m / 512`. Normal: `2^(e-7) * (1 + m/8)`, the
    // power of two built as bits (exponent field `127 + e - 7`).
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

/// Everything a launch needs to name one plane per slot: the arena, the
/// per-slot offsets into it and the per-slot constants, and the plan's runs.
pub struct PlaneRef<'a> {
    pub wmap: &'a Handle,
    pub wmap_bytes: usize,
    pub off: &'a Handle,
    pub scale2: &'a Handle,
    pub slots: usize,
    pub blk: &'a BlockPlanDev,
}

/// Launch [`expert_update`] over every run of the plan: `[n, k]` planes,
/// `g_out [m_total, n]` and `x [m_total, k]` at element type `E`.
#[allow(clippy::too_many_arguments)]
pub fn expert_update_launch<E: Scalar + Cast + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    plane: &PlaneRef<'_>,
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
    assert_eq!(m_total % 16, 0, "m_total {m_total} is not a multiple of 16");
    let words = plane.wmap_bytes / 4;
    let blk = plane.blk;
    unsafe {
        expert_update::launch::<E, R>(
            client,
            CubeCount::Static((k / LANE_K) as u32, blk.blocks as u32, 1),
            CubeDim::new_1d(32),
            AddressType::U64,
            ArrayArg::from_raw_parts(plane.wmap.clone(), words),
            ArrayArg::from_raw_parts(plane.off.clone(), 2 * plane.slots),
            ArrayArg::from_raw_parts(plane.scale2.clone(), plane.slots),
            ArrayArg::from_raw_parts(blk.slot.clone(), blk.blocks),
            ArrayArg::from_raw_parts(blk.tile0.clone(), blk.blocks),
            ArrayArg::from_raw_parts(blk.cnt.clone(), blk.blocks),
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

/// Launch [`expert_gin`] over every run of the plan, returning the f32
/// `[m_total, k]` input gradient (zero on rows no run covers).
pub fn expert_gin_launch<E: Scalar + Cast + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    plane: &PlaneRef<'_>,
    g_out: &Handle,
    m_total: usize,
    n: usize,
    k: usize,
    max_run_rows: usize,
) -> Handle {
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % LANE_K, 0, "k {k} is not a multiple of {LANE_K}");
    let words = plane.wmap_bytes / 4;
    let blk = plane.blk;
    let g_in = client.empty(m_total * k * 4);
    let chunks = max_run_rows.div_ceil(GIN_ROWS).max(1);
    unsafe {
        expert_gin::launch::<E, R>(
            client,
            CubeCount::Static((k / LANE_K) as u32, blk.blocks as u32, chunks as u32),
            CubeDim::new_1d(32),
            AddressType::U64,
            ArrayArg::from_raw_parts(plane.wmap.clone(), words),
            ArrayArg::from_raw_parts(plane.off.clone(), 2 * plane.slots),
            ArrayArg::from_raw_parts(plane.scale2.clone(), plane.slots),
            ArrayArg::from_raw_parts(blk.slot.clone(), blk.blocks),
            ArrayArg::from_raw_parts(blk.tile0.clone(), blk.blocks),
            ArrayArg::from_raw_parts(blk.cnt.clone(), blk.blocks),
            ArrayArg::from_raw_parts(g_out.clone(), m_total * n),
            ArrayArg::from_raw_parts(g_in.clone(), m_total * k),
            k,
            n,
        )
    };
    g_in
}

#[allow(dead_code)]
const _GROUP_IS_SIXTEEN: () = assert!(GROUP == 16);
