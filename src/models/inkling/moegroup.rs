//! The routed-expert lane as ONE launch per stage per LAYER, not per expert.
//!
//! # What this replaces, and why the count was the cost
//!
//! [`super::fp4gemm`] computes one expert. The lane above it looped over the
//! layer's active experts and issued a whole sequence inside each iteration —
//! gather, quantise, GEMM, gate/silu, quantise, GEMM, scatter — so a layer cost
//! `experts x 7` launches plus two host->device index uploads per expert. The
//! per-pass report says what that came to: at 164 expert-loads the host spent
//! 30.8 ms issuing 6.8 ms of device work. The lane was not weight-bound (0.3 ms
//! of slicing, 100% aliased, zero disk) and it was not arithmetic-bound. It was
//! bound by the CPU walking a `BTreeMap` and calling into cubecl.
//!
//! So the loop moves into the grid. Every stage below takes the whole layer at
//! once: the activations of all active experts stacked into one `[M, hidden]`
//! buffer, `M = sum of each expert's M-padded row count`, and the block index
//! selecting which expert a 16-row tile belongs to.
//!
//! # How a kernel reaches 27 different weights in one launch
//!
//! It does not take 27 pointers. A pile is ONE mapping and the zero-copy seam
//! registers it ONCE ([`super::fp4gemm::Aliases`]), so every expert slab in the
//! model is already a byte offset into a single registered buffer. The grouped
//! GEMM binds that buffer whole and takes a small `[2 * slots]` table of
//! offsets; the tile's expert selects its row of the table. Nothing is copied
//! and nothing is re-registered — the offsets are the same arithmetic
//! `Aliases::slice` was doing per expert, hoisted into device memory.
//!
//! That buffer is 38 GiB on this node's share, which is past what a u32 index
//! can address, so these kernels are launched with `AddressType::U64`. It is
//! the one price of the design and it is paid in the index arithmetic, not in
//! traffic.
//!
//! # The accumulation order is the same order, deliberately
//!
//! The per-expert lane scattered with `select_assign(Add)` once per expert, in
//! `BTreeMap` order, into a zeroed accumulator. A token's contributions were
//! therefore summed smallest-expert-first, and each `select_assign` was its own
//! kernel so the order was the enqueue order.
//!
//! An atomic scatter would break exactly that and would break it
//! nondeterministically, so [`scatter_weighted`] does not use one. It gives one
//! thread each `(token, column)` and has it walk that token's contributing rows
//! in ASCENDING row order — and rows are laid out in `BTreeMap` order, so
//! ascending row order IS ascending expert order.
//!
//! # The weight multiply stays INSIDE the sum
//!
//! `sum += y[r] * wgt[r]` contracts to `fma.rn.f32` — NVRTC compiles with
//! `--fmad=true` — which rounds ONCE per term. The per-expert lane materialises
//! `y * wgt` as a Burn tensor and only then adds it: `mul.rn.f32` followed by
//! `add.rn.f32`, TWICE per term. So the two lanes do NOT agree bit for bit
//! here, and the fused one is the one with strictly fewer roundings.
//!
//! This module briefly did the opposite: a `scale_rows` launch whose only
//! purpose was to force the second rounding, so that the accumulator would
//! match the per-expert lane's bits. That is the same mistake `91f81b4` had
//! already found and removed from the short convolution, and its message states
//! the principle — *bit equality against a previous implementation cannot say
//! which of two lanes is right, only which one was written first.* It bought a
//! worse number and an extra launch, in a lane whose entire problem was launch
//! count.
//!
//! There is deliberately no new numerical gate for this. That one rounding
//! beats two is arithmetic, not an open question; and the magnitude is beneath
//! the floor that matters here, because these operands come out of FOUR-BIT
//! weights. `e27384e` measured this runtime disagreeing with ITSELF on
//! 43/503 = 8.55% of argmax positions between two runs of the same binary,
//! median |Δ top-1 logit| 0.34. `6854e9b` is why that is the yardstick: the
//! numerical delta is not the gate, capability is.
//!
//! Everything else in this lane IS bit-exact against the per-expert one, and
//! there bit equality is exactly the right gate — the routing, the expert
//! selection, the gather, the GEMM operand order and the accumulation ORDER are
//! not approximations of anything, so a difference in any of them would be a
//! defect. Only WHERE the two lanes round is exempt, and only by argument.

use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::{e2m1x2, e4m3};

use super::bf16gemm::KTILE as BF16_KTILE;
use super::fp4gemm::{GROUP, KTILE, MTILE, NTILE};

/// Threads per cube for the elementwise kernels here.
const CUBE_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// Gather
// ---------------------------------------------------------------------------

/// `out[r, :] = src[idx[r], :]`, or zeros where `idx[r] < 0`.
///
/// The whole layer's A operand in one launch. `idx` is `[M]`, one entry per row
/// of the stacked buffer, holding the residual-stream row this row copies or
/// `-1` for a row that exists only because the MMA tiles M by 16. The
/// per-expert kernel it replaces took `m` and `m_pad` and derived the same
/// thing from a comparison; a sentinel is what lets one launch serve experts
/// with different row counts.
#[cube(launch_unchecked)]
fn gather_grouped_kernel(
    src: &Array<f32>,
    idx: &Array<i32>,
    out: &mut Array<f32>,
    h: usize,
    total: usize,
) {
    let p = ABSOLUTE_POS as usize;
    if p < total {
        let r = p / h;
        let t = idx[r];
        let mut v = f32::new(0.0f32);
        if t >= 0i32 {
            v = src[u32::cast_from(t) as usize * h + p % h];
        }
        out[p] = v;
    }
}

/// Launch [`gather_grouped_kernel`], returning the `[m_total, h]` buffer.
pub fn gather_grouped<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle,
    idx: &Handle,
    src_rows: usize,
    m_total: usize,
    h: usize,
) -> Handle {
    let total = m_total * h;
    let out = client.empty(total * core::mem::size_of::<f32>());
    let cubes = total.div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        gather_grouped_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(src.clone(), src_rows * h),
            ArrayArg::from_raw_parts(idx.clone(), m_total),
            ArrayArg::from_raw_parts(out.clone(), total),
            h,
            total,
        );
    }
    out
}

// ---------------------------------------------------------------------------
// The grouped GEMM
// ---------------------------------------------------------------------------

/// `out = (a @ b_slot^T) * scale_slot`, where the slot is chosen per M tile.
///
/// Line for line the kernel in [`super::fp4gemm::fp4_linear`], with three
/// changes and no others:
///
/// * `b` and `b_sc` are the WHOLE registered mapping rather than one expert's
///   planes, and the tile's expert contributes a byte offset into each. The
///   offsets are `[codes, scales]` per slot, so `off[2 * slot]` and
///   `off[2 * slot + 1]`.
/// * `scale` — the expert's second-level F32 quantisation constant — comes from
///   a `[slots]` tensor instead of a scalar argument, for the same reason.
/// * the M tile is an index into the stacked buffer, so `CUBE_POS_Y` runs over
///   `M / 16` tiles of the whole layer instead of one expert's `m_pad / 16`.
///
/// Everything inside the K loop — the operand loads, the block scales, the
/// `execute_scaled`, the accumulator — is unchanged, which is what makes the
/// result bit-identical rather than merely close.
#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear_grouped<AB: Scalar, S: Scalar, NA: Size, NC: Size>(
    a: &Tensor<Vector<AB, NA>>,
    a_sc: &Tensor<S>,
    b: &Tensor<Vector<AB, NA>>,
    b_sc: &Tensor<S>,
    tile_slot: &Tensor<u32>,
    off: &Tensor<u64>,
    scale2: &Tensor<f32>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let n_tile = CUBE_POS_X as usize;
    let m_tile = CUBE_POS_Y as usize;
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;

    // Which expert this tile's sixteen rows were routed to, and where its two
    // planes start in the mapping. Both offsets are in ELEMENTS of the plane's
    // own type, which for E2M1 packed pairs and for E4M3 scales is bytes.
    let slot = tile_slot[m_tile] as usize;
    let b_base = usize::cast_from(off[2 * slot]);
    let bsc_base = usize::cast_from(off[2 * slot + 1]);
    let scale = scale2[slot];

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();
    let sia = def.scales_index(lane, MatrixIdent::A) as usize;
    let sib = def.scales_index(lane, MatrixIdent::B) as usize;
    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE);

    for t in 0..k_tiles {
        let kbase = t * KTILE;
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
            let gr = row as usize + m_base;
            let gc = col as usize + kbase;
            reg_a[i] = a[(gr * size_k / 2 + gc / 2) / a.vector_size()];
        }
        #[unroll]
        for i in 0..vc_b {
            let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
            let gr = col as usize + n_base;
            let gc = row as usize + kbase;
            reg_b[i] = b[(b_base + gr * size_k / 2 + gc / 2) / b.vector_size()];
        }

        let mut sa = Vector::<S, NS>::empty();
        let mut sb = Vector::<S, NS>::empty();
        #[unroll]
        for i in 0..scales_count {
            sa[i] = a_sc[(sia + m_base) * spr + t * 4 + i];
            sb[i] = b_sc[bsc_base + (sib + n_base) * spr + t * 4 + i];
        }

        let d = def.execute_scaled(&reg_a, &reg_b, &acc, sa, sb);
        #[unroll]
        for i in 0..vc_c {
            acc[i] = d[i];
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let gr = row as usize + m_base;
        let gc = col as usize + n_base;
        out[(gr * size_n + gc) / out.vector_size()] =
            acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// Launch [`fp4_linear_grouped`] over a whole layer's stacked `[m_total, k]` A.
///
/// `wmap` is the registered mapping, `wmap_bytes` its length; the per-slot
/// offsets in `off` are byte offsets into it.
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear_grouped_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    a_sc: &Handle,
    wmap: &Handle,
    wmap_bytes: usize,
    tile_slot: &Handle,
    off: &Handle,
    scale2: &Handle,
    slots: usize,
    m_total: usize,
    k: usize,
    n: usize,
) -> Handle {
    assert_eq!(m_total % MTILE, 0, "m_total {m_total} is not a multiple of {MTILE}");
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");

    let out = client.empty(m_total * n * core::mem::size_of::<f32>());
    let vs = 32 / e2m1x2::cube_type().size_bits();
    let spr = k / GROUP;
    let tiles = m_total / MTILE;
    // The mapping is bound as a flat plane of packed bytes; the kernel indexes
    // it in `vs`-wide vectors, so the declared length has to be a whole number
    // of them.
    let flat = wmap_bytes - wmap_bytes % vs;

    unsafe {
        fp4_linear_grouped::launch::<e2m1x2, e4m3, R>(
            client,
            CubeCount::Static((n / NTILE) as u32, tiles as u32, 1),
            CubeDim::new_1d(32),
            AddressType::U64,
            vs,
            2,
            TensorArg::from_raw_parts(a.clone(), [k / 2, 1].into(), [m_total, k / 2].into()),
            TensorArg::from_raw_parts(a_sc.clone(), [spr, 1].into(), [m_total, spr].into()),
            TensorArg::from_raw_parts(wmap.clone(), [1].into(), [flat].into()),
            TensorArg::from_raw_parts(wmap.clone(), [1].into(), [wmap_bytes].into()),
            TensorArg::from_raw_parts(tile_slot.clone(), [1].into(), [tiles].into()),
            TensorArg::from_raw_parts(off.clone(), [1].into(), [2 * slots].into()),
            TensorArg::from_raw_parts(scale2.clone(), [1].into(), [slots].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_total, n].into()),
            k,
            n,
        )
    };
    out
}

// ---------------------------------------------------------------------------
// The grouped GEMM, unscaled — layer 2
// ---------------------------------------------------------------------------

/// `out = a @ b_slot^T`, BF16 on both operands, slot chosen per M tile.
///
/// [`super::bf16gemm::bf16_linear`] with the same two additions
/// [`fp4_linear_grouped`] makes to its packed sibling: the tile picks an expert,
/// and the expert contributes a base offset into the one bound mapping. There
/// is only one plane to offset here and no second-level scale to look up,
/// because nothing quantised this layer.
///
/// The offset is in BF16 ELEMENTS, not bytes — that is the unit `b` is indexed
/// in, and converting at the call site keeps the kernel reading like the one it
/// mirrors.
#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
pub fn bf16_linear_grouped<AB: Scalar, NA: Size, NC: Size>(
    a: &Tensor<Vector<AB, NA>>,
    b: &Tensor<Vector<AB, NA>>,
    tile_slot: &Tensor<u32>,
    off: &Tensor<u64>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, BF16_KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let n_tile = CUBE_POS_X as usize;
    let m_tile = CUBE_POS_Y as usize;
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;

    let slot = tile_slot[m_tile] as usize;
    let b_base = usize::cast_from(off[slot]);

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let k_tiles = comptime!(size_k / BF16_KTILE);

    for t in 0..k_tiles {
        let kbase = t * BF16_KTILE;
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
            let gr = row as usize + m_base;
            let gc = col as usize + kbase;
            reg_a[i] = a[(gr * size_k + gc) / a.vector_size()];
        }
        #[unroll]
        for i in 0..vc_b {
            let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
            let gr = col as usize + n_base;
            let gc = row as usize + kbase;
            reg_b[i] = b[(b_base + gr * size_k + gc) / b.vector_size()];
        }

        let d = def.execute(&reg_a, &reg_b, &acc);
        #[unroll]
        for i in 0..vc_c {
            acc[i] = d[i];
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let gr = row as usize + m_base;
        let gc = col as usize + n_base;
        out[(gr * size_n + gc) / out.vector_size()] = acc[i];
    }
}

/// Launch [`bf16_linear_grouped`] over a whole layer's stacked `[m_total, k]` A.
///
/// `off` holds one BF16-ELEMENT offset per slot; `wmap_bytes` is the mapping's
/// length in bytes, halved here because that is the unit the tensor is declared
/// in.
#[allow(clippy::too_many_arguments)]
pub fn bf16_linear_grouped_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    wmap: &Handle,
    wmap_bytes: usize,
    tile_slot: &Handle,
    off: &Handle,
    slots: usize,
    m_total: usize,
    k: usize,
    n: usize,
) -> Handle {
    assert_eq!(m_total % MTILE, 0, "m_total {m_total} is not a multiple of {MTILE}");
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % BF16_KTILE, 0, "k {k} is not a multiple of {BF16_KTILE}");

    let out = client.empty(m_total * n * core::mem::size_of::<f32>());
    let vs = 32 / half::bf16::cube_type().size_bits();
    let tiles = m_total / MTILE;
    let elems = wmap_bytes / 2;
    let flat = elems - elems % vs;

    unsafe {
        bf16_linear_grouped::launch::<half::bf16, R>(
            client,
            CubeCount::Static((n / NTILE) as u32, tiles as u32, 1),
            CubeDim::new_1d(32),
            AddressType::U64,
            vs,
            2,
            TensorArg::from_raw_parts(a.clone(), [k, 1].into(), [m_total, k].into()),
            TensorArg::from_raw_parts(wmap.clone(), [1].into(), [flat].into()),
            TensorArg::from_raw_parts(tile_slot.clone(), [1].into(), [tiles].into()),
            TensorArg::from_raw_parts(off.clone(), [1].into(), [slots].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_total, n].into()),
            k,
            n,
        )
    };
    out
}

// ---------------------------------------------------------------------------
// Scatter
// ---------------------------------------------------------------------------

/// `out[t, :] = sum over t's rows of y[r, :] * wgt[r]`, in ascending `r`.
///
/// One thread per output element, walking that token's contributing rows. NOT
/// an atomic scatter: the sum has to be the same sum in the same ORDER the
/// per-expert lane's chain of `select_assign(Add)` made, and an atomic add
/// gives neither. `tok_rows` is `[n, kmax]` row-major with `tok_cnt[t]` valid
/// entries in each row, ascending — which is `BTreeMap` expert order, because
/// that is the order the rows were laid out in.
///
/// The multiply is inside the accumulation, so it contracts to `fma.rn.f32`:
/// one rounding per term where the lane this replaces takes two. That is the
/// module doc's subject and it is not an accident of codegen — a `scale_rows`
/// launch that forced the second rounding was written, measured, and reverted.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn scatter_weighted_kernel(
    y: &Array<f32>,
    wgt: &Array<f32>,
    tok_rows: &Array<u32>,
    tok_cnt: &Array<u32>,
    out: &mut Array<f32>,
    #[comptime] kmax: usize,
    h: usize,
    total: usize,
) {
    let p = ABSOLUTE_POS as usize;
    if p < total {
        let t = p / h;
        let c = p % h;
        let cnt = tok_cnt[t] as usize;
        let mut sum = f32::new(0.0f32);
        for j in 0..kmax {
            if j < cnt {
                let r = tok_rows[t * kmax + j] as usize;
                sum += y[r * h + c] * wgt[r];
            }
        }
        out[p] = sum;
    }
}

/// Launch [`scatter_weighted_kernel`], returning the `[n, h]` accumulator.
#[allow(clippy::too_many_arguments)]
pub fn scatter_weighted<R: Runtime>(
    client: &ComputeClient<R>,
    y: &Handle,
    wgt: &Handle,
    tok_rows: &Handle,
    tok_cnt: &Handle,
    m_total: usize,
    n: usize,
    h: usize,
    kmax: usize,
) -> Handle {
    let total = n * h;
    let out = client.empty(total * core::mem::size_of::<f32>());
    let cubes = total.div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        scatter_weighted_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(y.clone(), m_total * h),
            ArrayArg::from_raw_parts(wgt.clone(), m_total),
            ArrayArg::from_raw_parts(tok_rows.clone(), n * kmax),
            ArrayArg::from_raw_parts(tok_cnt.clone(), n),
            ArrayArg::from_raw_parts(out.clone(), total),
            kmax,
            h,
            total,
        );
    }
    out
}

// ---------------------------------------------------------------------------
// The layer's row plan, on the host
// ---------------------------------------------------------------------------

/// How one layer's active experts are stacked into the grouped buffers.
///
/// Built once per layer from the router's `by_expert` decision. Every field is
/// a small `Vec` that goes straight to the device — the largest is `[M]` — and
/// together they are the only host->device traffic the grouped lane has, once
/// per layer instead of twice per expert.
pub struct RowPlan {
    /// `[M]`: the residual-stream row each stacked row copies, `-1` for pad.
    pub row_tok: Vec<i32>,
    /// `[M]`: the routing weight each stacked row's output is multiplied by.
    pub row_wgt: Vec<f32>,
    /// `[M / MTILE]`: which expert slot each MMA row tile belongs to.
    pub tile_slot: Vec<u32>,
    /// `[n, kmax]`: each token's contributing rows, ascending.
    pub tok_rows: Vec<u32>,
    /// `[n]`: how many of each row of `tok_rows` are valid.
    pub tok_cnt: Vec<u32>,
    /// Row of `tok_rows`; the largest number of experts any token routed to.
    pub kmax: usize,
}

impl RowPlan {
    /// Stack the layer, one expert after another in the iteration order given.
    ///
    /// `experts` is `(expert, [(token row, routing weight)])` in the order the
    /// accumulation must happen in — which for the lane this replaces is
    /// `BTreeMap` order over the expert index.
    pub fn build<'a>(
        experts: impl Iterator<Item = &'a Vec<(usize, f32)>>,
        n: usize,
    ) -> RowPlan {
        let mut row_tok: Vec<i32> = Vec::new();
        let mut row_wgt: Vec<f32> = Vec::new();
        let mut tile_slot: Vec<u32> = Vec::new();
        let mut per_tok: Vec<Vec<u32>> = vec![Vec::new(); n];

        for (slot, toks) in experts.enumerate() {
            let m = toks.len();
            let m_pad = m.div_ceil(MTILE) * MTILE;
            for &(ti, wgt) in toks.iter() {
                per_tok[ti].push(row_tok.len() as u32);
                row_tok.push(ti as i32);
                row_wgt.push(wgt);
            }
            for _ in m..m_pad {
                row_tok.push(-1);
                row_wgt.push(0.0);
            }
            for _ in 0..(m_pad / MTILE) {
                tile_slot.push(slot as u32);
            }
        }

        let kmax = per_tok.iter().map(|v| v.len()).max().unwrap_or(0).max(1);
        let mut tok_rows = vec![0u32; n * kmax];
        let mut tok_cnt = vec![0u32; n];
        for (t, rows) in per_tok.iter().enumerate() {
            tok_cnt[t] = rows.len() as u32;
            tok_rows[t * kmax..t * kmax + rows.len()].copy_from_slice(rows);
        }

        RowPlan { row_tok, row_wgt, tile_slot, tok_rows, tok_cnt, kmax }
    }

    /// Rows in the stacked buffer, `M`.
    pub fn m_total(&self) -> usize {
        self.row_tok.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the bitwise gate rests on: a token's rows come out in the
    /// order its experts were iterated, and the pad rows carry no token.
    #[test]
    fn rows_are_stacked_in_expert_order_with_padding() {
        // Three experts, in the order a BTreeMap would hand them over.
        let e0 = vec![(1usize, 0.5f32)];
        let e1 = vec![(0usize, 0.25f32), (1usize, 0.125f32)];
        let e2 = vec![(0usize, 0.75f32)];
        let plan = RowPlan::build([&e0, &e1, &e2].into_iter(), 2);

        // Each expert is padded to a whole MMA row tile.
        assert_eq!(plan.m_total(), 3 * MTILE);
        assert_eq!(plan.tile_slot, vec![0, 1, 2]);

        // Row 0 is expert 0's only token; MTILE..MTILE+2 are expert 1's two.
        assert_eq!(plan.row_tok[0], 1);
        assert_eq!(plan.row_tok[1], -1);
        assert_eq!(plan.row_tok[MTILE], 0);
        assert_eq!(plan.row_tok[MTILE + 1], 1);
        assert_eq!(plan.row_tok[2 * MTILE], 0);

        // Token 0 is served by experts 1 and 2, token 1 by experts 0 and 1 --
        // and both lists are ASCENDING in row, i.e. in expert order.
        assert_eq!(plan.kmax, 2);
        assert_eq!(plan.tok_cnt, vec![2, 2]);
        assert_eq!(
            &plan.tok_rows[0..2],
            &[MTILE as u32, 2 * MTILE as u32]
        );
        assert_eq!(&plan.tok_rows[2..4], &[0u32, MTILE as u32 + 1]);

        // Pad rows weigh nothing, and every real row carries its expert's weight.
        assert_eq!(plan.row_wgt[0], 0.5);
        assert_eq!(plan.row_wgt[1], 0.0);
        assert_eq!(plan.row_wgt[MTILE], 0.25);
        assert_eq!(plan.row_wgt[MTILE + 1], 0.125);
        assert_eq!(plan.row_wgt[2 * MTILE], 0.75);
    }
}
