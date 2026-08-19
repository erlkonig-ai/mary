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
//! # How the scatter sums
//!
//! [`scatter_weighted`] gives one thread each `(token, column)` and has it walk
//! that token's contributing rows in ascending row order, accumulating in a
//! register. Rows are laid out in `BTreeMap` expert order, so ascending row
//! order is ascending expert order — the same order the per-expert lane's chain
//! of `select_assign(Add)` produced, which is why the two lanes agreed exactly
//! when this replaced that one.
//!
//! # Why a cube is several planes wide
//!
//! Both GEMMs below started as one warp per 16x8 output tile, reading A and B
//! out of global before every `mma` — the decode schedule, where the lane is
//! weight-streaming-bound and the tile size does not matter. At a 512-token
//! prefill it is the wrong shape, and the reason is B.
//!
//! A cube reads its expert's whole `[n, k]` plane once for every 16 rows it
//! serves, so the layer reads B `m_pad_e / 16` times per expert. With ~65
//! tokens on each of ~47 active experts that is five times, and it is DRAM
//! every time: cubes that share an m tile differ in `CUBE_POS_X` and run
//! together (L2 serves A), cubes that share an n tile are `n / 8` apart in
//! launch order and do not. At `[3456, 4096] x [4096, 4096]` that is 7.25 GB
//! of B against 273 GB/s of memory, i.e. 26.6 ms, and the kernel measured 29.5.
//! The arithmetic was never the limit.
//!
//! So the cube gets `MPLANES` planes and each plane takes one of `MPLANES`
//! CONSECUTIVE m tiles. They read the same B addresses in the same cycle, out
//! of one L1, so B crosses the memory bus once per cube instead of once per
//! tile. Nothing about the fragments changes — one A register array, one B
//! register array, one accumulator, exactly as before — which is why this is a
//! grid change and not a rewrite.
//!
//! The m tiles a plane may share a cube with have to belong to the SAME expert,
//! or they would want different B. That is what [`RowPlan`]'s block plan is:
//! each expert's tiles are cut into runs of at most `MPLANES`, and a run is a
//! cube. Padding stays at 16 rows — padding to `16 * MPLANES` would nearly
//! double the arithmetic on a 65-token expert — so the last run of an expert is
//! short and its spare planes sit out the launch.
//!
//! `INK_MOE_PLANES` sets `MPLANES`; it is a host-side plan parameter and the
//! kernels read the run length out of the plan, so it is tunable without a
//! recompile.
//!
//! # How the scatter sums
//!
//! (continued) That agreement was a convenient property of the port, not a
//! requirement on it. **This is not a prohibition on an atomic scatter.** An atomic float add
//! reassociates the sum and gives up run-to-run reproducibility, and neither of
//! those is a gate here (see the yardstick two sections down: this runtime
//! already disagrees with itself on 8.55% of argmax positions between two runs
//! of the same binary). The one-thread-per-output-element walk is where this
//! started because it needs no atomics and no extra pass; whether an atomic
//! scatter beats it is a measurement, and `INK_SCATTER=atomic` is the arm that
//! takes it.
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

/// E4M3 block scales per vector load in the FP4 GEMM.
///
/// Not a tuning knob: it is `MmaDefinition::scales_vector_size()`, which is the
/// MMA register width over the scale width, 32/8. The instruction takes its
/// scales as one 32-bit register, so this is the width the memory has to be read
/// in for the load to be one instruction.
const SCALE_VEC: usize = 4;

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
pub fn fp4_linear_grouped<AB: Scalar, S: Scalar, NA: Size, NC: Size, NS: Size>(
    a: &Tensor<Vector<AB, NA>>,
    a_sc: &Tensor<Vector<S, NS>>,
    b: &Tensor<Vector<AB, NA>>,
    b_sc: &Tensor<Vector<S, NS>>,
    blk_slot: &Tensor<u32>,
    blk_tile0: &Tensor<u32>,
    blk_cnt: &Tensor<u32>,
    off: &Tensor<u64>,
    scale2: &Tensor<f32>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] nrep: usize,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let n_tile = CUBE_POS_X as usize;
    let blk = CUBE_POS_Y as usize;
    let plane = PLANE_POS;
    // A short run leaves spare planes with nothing to do. They take the branch
    // and exit; there is no barrier in this kernel for them to miss.
    if plane >= blk_cnt[blk] {
        terminate!();
    }
    let m_tile = blk_tile0[blk] as usize + plane as usize;
    let n_base = n_tile * NTILE * nrep;
    let m_base = m_tile * MTILE;

    // Which expert this run of tiles was routed to, and where its two planes
    // start in the mapping. Both offsets are in ELEMENTS of the plane's own
    // type, which for E2M1 packed pairs and for E4M3 scales is bytes.
    let slot = blk_slot[blk] as usize;
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
    // `nrep` B fragments and `nrep` accumulators, live across the whole k
    // loop, so one A fragment and one A scale vector feed `nrep` products.
    // A `Sequence` because `execute_scaled` takes a WHOLE `Array` -- one wider
    // array cannot be sliced into an operand.
    let mut regs_b = Sequence::<Array<Vector<AB, NA>>>::new();
    let mut accs = Sequence::<Array<Vector<f32, NC>>>::new();
    #[unroll]
    for _ in 0..nrep {
        regs_b.push(Array::<Vector<AB, NA>>::new(vc_b));
        let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
        #[unroll]
        for i in 0..vc_c {
            acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
        }
        accs.push(acc);
    }

    // The instruction wants FOUR E4M3 block scales per operand per k tile, and
    // they sit at four CONSECUTIVE addresses. Read as four `Tensor<S>` elements
    // that is four one-byte loads; read as one `Vector<S, 4>` it is one 32-bit
    // load, and 8 of the 14 loads this kernel issues per `mma` were those bytes.
    // `scales_vector_size` is `register_size_bits / 8` = 4 here, which is the
    // same 4 -- the vector the instruction takes IS the vector the memory holds,
    // so there is nothing to assemble.
    //
    // The group of four starts at `(index) * spr + t * 4`, and `spr = k / 16` is
    // a multiple of four at every k this model has, so the group never straddles
    // a vector. The B side adds `bsc_base`, which the caller has already refused
    // unless it is a multiple of four -- the same rule the packed planes carry.
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
        for j in 0..nrep {
            let rb = regs_b.index_mut(j);
            #[unroll]
            for i in 0..vc_b {
                let (row, col) =
                    def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
                let gr = col as usize + n_base + j * NTILE;
                let gc = row as usize + kbase;
                rb[i] = b[(b_base + gr * size_k / 2 + gc / 2) / b.vector_size()];
            }
        }

        // One 32-bit load each, then into a MUTABLE local: the MMA intrinsic
        // takes its scale registers by non-const reference, so a value that
        // came straight out of a load and is never written cannot be handed to
        // it -- NVRTC rejects the generated cast. The four moves below are
        // register traffic, not memory. The A side is loaded ONCE for all
        // `nrep` products; only the B side moves with the N tile.
        let va = a_sc[((sia + m_base) * spr + t * 4) / a_sc.vector_size()];
        let mut sa = Vector::<S, NS>::empty();
        #[unroll]
        for i in 0..SCALE_VEC {
            sa[i] = va[i];
        }

        #[unroll]
        for j in 0..nrep {
            let vb = b_sc
                [(bsc_base + (sib + n_base + j * NTILE) * spr + t * 4) / b_sc.vector_size()];
            let mut sb = Vector::<S, NS>::empty();
            #[unroll]
            for i in 0..SCALE_VEC {
                sb[i] = vb[i];
            }
            let d = def.execute_scaled(&reg_a, regs_b.index(j), accs.index(j), sa, sb);
            let ac = accs.index_mut(j);
            #[unroll]
            for i in 0..vc_c {
                ac[i] = d[i];
            }
        }
    }

    #[unroll]
    for j in 0..nrep {
        let ac = accs.index(j);
        #[unroll]
        for i in 0..vc_c {
            let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
            let gr = row as usize + m_base;
            let gc = col as usize + n_base + j * NTILE;
            out[(gr * size_n + gc) / out.vector_size()] =
                ac[i] * Vector::<f32, NC>::cast_from(scale);
        }
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
    blk: &BlockPlanDev,
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
    assert_eq!(
        (k / GROUP) % SCALE_VEC,
        0,
        "the scale row {} is not a whole number of {SCALE_VEC}-wide vectors",
        k / GROUP
    );

    // How many N tiles one plane keeps in registers; see [`grouped_nrep`].
    let nrep = grouped_nrep(n, m_total);
    let n_cubes = n / (NTILE * nrep);

    let out = client.empty(m_total * n * core::mem::size_of::<f32>());
    let vs = 32 / e2m1x2::cube_type().size_bits();
    let spr = k / GROUP;
    // The mapping is bound as a flat plane of packed bytes; the kernel indexes
    // it in `vs`-wide vectors, so the declared length has to be a whole number
    // of them.
    let flat = wmap_bytes - wmap_bytes % vs;
    // The scale planes are read four bytes at a time, so the bound length has
    // to be a whole number of them too.
    let flat_sc = wmap_bytes - wmap_bytes % SCALE_VEC;

    unsafe {
        fp4_linear_grouped::launch::<e2m1x2, e4m3, R>(
            client,
            CubeCount::Static(n_cubes as u32, blk.blocks as u32, 1),
            CubeDim::new_1d(32 * blk.planes as u32),
            AddressType::U64,
            vs,
            2,
            SCALE_VEC,
            TensorArg::from_raw_parts(a.clone(), [k / 2, 1].into(), [m_total, k / 2].into()),
            TensorArg::from_raw_parts(a_sc.clone(), [spr, 1].into(), [m_total, spr].into()),
            TensorArg::from_raw_parts(wmap.clone(), [1].into(), [flat].into()),
            TensorArg::from_raw_parts(wmap.clone(), [1].into(), [flat_sc].into()),
            TensorArg::from_raw_parts(blk.slot.clone(), [1].into(), [blk.blocks].into()),
            TensorArg::from_raw_parts(blk.tile0.clone(), [1].into(), [blk.blocks].into()),
            TensorArg::from_raw_parts(blk.cnt.clone(), [1].into(), [blk.blocks].into()),
            TensorArg::from_raw_parts(off.clone(), [1].into(), [2 * slots].into()),
            TensorArg::from_raw_parts(scale2.clone(), [1].into(), [slots].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_total, n].into()),
            k,
            n,
            nrep,
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
    blk_slot: &Tensor<u32>,
    blk_tile0: &Tensor<u32>,
    blk_cnt: &Tensor<u32>,
    off: &Tensor<u64>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] nrep: usize,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, BF16_KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let n_tile = CUBE_POS_X as usize;
    let blk = CUBE_POS_Y as usize;
    let plane = PLANE_POS;
    if plane >= blk_cnt[blk] {
        terminate!();
    }
    let m_tile = blk_tile0[blk] as usize + plane as usize;
    let n_base = n_tile * NTILE * nrep;
    let m_base = m_tile * MTILE;

    let slot = blk_slot[blk] as usize;
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
    // `nrep` B fragments and `nrep` accumulators, held live across the whole k
    // loop. A `Sequence` because `MmaDefinition::execute` takes a WHOLE
    // `Array` -- one wider array cannot be sliced into an operand -- so the
    // repetition has to live in the comptime container, not in the indexing.
    let mut regs_b = Sequence::<Array<Vector<AB, NA>>>::new();
    let mut accs = Sequence::<Array<Vector<f32, NC>>>::new();
    #[unroll]
    for _ in 0..nrep {
        regs_b.push(Array::<Vector<AB, NA>>::new(vc_b));
        let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
        #[unroll]
        for i in 0..vc_c {
            acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
        }
        accs.push(acc);
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
        for j in 0..nrep {
            let rb = regs_b.index_mut(j);
            #[unroll]
            for i in 0..vc_b {
                let (row, col) =
                    def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
                let gr = col as usize + n_base + j * NTILE;
                let gc = row as usize + kbase;
                rb[i] = b[(b_base + gr * size_k + gc) / b.vector_size()];
            }
        }

        // The whole point of the repetition: ONE A fragment feeds `nrep` MMAs,
        // so the A loads above are amortised over `nrep` products rather than
        // being reissued for each of them. A is the larger half of this
        // kernel global traffic and the half L2 serves, because cubes that
        // share an m tile differ only in `CUBE_POS_X` and run together.
        #[unroll]
        for j in 0..nrep {
            let d = def.execute(&reg_a, regs_b.index(j), accs.index(j));
            let ac = accs.index_mut(j);
            #[unroll]
            for i in 0..vc_c {
                ac[i] = d[i];
            }
        }
    }

    #[unroll]
    for j in 0..nrep {
        let ac = accs.index(j);
        #[unroll]
        for i in 0..vc_c {
            let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
            let gr = row as usize + m_base;
            let gc = col as usize + n_base + j * NTILE;
            out[(gr * size_n + gc) / out.vector_size()] = ac[i];
        }
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
    blk: &BlockPlanDev,
    off: &Handle,
    slots: usize,
    m_total: usize,
    k: usize,
    n: usize,
) -> Handle {
    assert_eq!(m_total % MTILE, 0, "m_total {m_total} is not a multiple of {MTILE}");
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % BF16_KTILE, 0, "k {k} is not a multiple of {BF16_KTILE}");
    // How many N tiles one plane keeps in registers; see [`grouped_nrep`].
    let nrep = grouped_nrep(n, m_total);
    let n_cubes = n / (NTILE * nrep);

    let out = client.empty(m_total * n * core::mem::size_of::<f32>());
    let vs = 32 / half::bf16::cube_type().size_bits();
    let elems = wmap_bytes / 2;
    let flat = elems - elems % vs;

    unsafe {
        bf16_linear_grouped::launch::<half::bf16, R>(
            client,
            CubeCount::Static(n_cubes as u32, blk.blocks as u32, 1),
            CubeDim::new_1d(32 * blk.planes as u32),
            AddressType::U64,
            vs,
            2,
            TensorArg::from_raw_parts(a.clone(), [k, 1].into(), [m_total, k].into()),
            TensorArg::from_raw_parts(wmap.clone(), [1].into(), [flat].into()),
            TensorArg::from_raw_parts(blk.slot.clone(), [1].into(), [blk.blocks].into()),
            TensorArg::from_raw_parts(blk.tile0.clone(), [1].into(), [blk.blocks].into()),
            TensorArg::from_raw_parts(blk.cnt.clone(), [1].into(), [blk.blocks].into()),
            TensorArg::from_raw_parts(off.clone(), [1].into(), [slots].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_total, n].into()),
            k,
            n,
            nrep,
        )
    };
    out
}

/// How many N tiles one plane of a grouped GEMM keeps in registers.
///
/// A plane holding `nrep` B fragments issues `nrep` MMAs off ONE A fragment
/// and one A scale vector, so the A side of the inner loop is divided by
/// `nrep`. That is a PREFILL schedule, exactly as the plane count is, and for
/// the same reason: it is a trade of registers for loads, and it only pays
/// when there is enough m behind it to pay with.
///
/// Measured on this part, layers 0:8, `INK_REPEAT=1`, p50 over 24 warm passes,
/// two interleaved rounds at each length, `nrep` 1 against 8:
///
/// ```text
///   prompt   m tiles/layer     nrep 1      nrep 8
///     128         83           86.1 ms    97.9-106.8 ms   <- LOSES
///     256        135          113.0 ms     103.5 ms
///     384        176          159.2 ms     140.9 ms
///     512        227          205.9 ms     174.5 ms       <- -31.4 ms
///   decode        ~8           37.6 ms      38.3 ms       <- loses
/// ```
///
/// So the crossover is BRACKETED between 83 and 135 tiles and is not localised
/// further; [`NREP_TILES`] puts the step inside that bracket, at "sixteen m
/// tiles behind each of the `nrep` products". A step there is harmless because
/// the two arms are within a few ms of each other on either side of it -- the
/// 128-token loss and the 256-token win are both about 10 ms, against a 31 ms
/// win at 512 where the schedule is unambiguous.
///
/// The tried mechanism that is NOT the explanation: plane fill. At 128 tokens
/// most experts contribute one m tile, so three of a four-plane cube's planes
/// terminate, and "the register tile costs occupancy and occupancy only
/// matters when the planes are empty" predicts `INK_MOE_PLANES=1` should
/// rescue it. It does the opposite -- 128 tokens at one plane and `nrep` 8 is
/// 178 ms against 105 at four planes -- so the cost is not plane fill and the
/// threshold below is empirical, not derived.
///
/// `INK_MOE_NREP` overrides the width. It has to DIVIDE the N tile count, so a
/// width that does not admit the requested repetition takes the largest one
/// that does -- a kernel that refuses a shape the model issues is not a
/// fallback, it is a crash.
pub fn grouped_nrep(n: usize, m_total: usize) -> usize {
    use std::sync::OnceLock;
    static R: OnceLock<usize> = OnceLock::new();
    let want = *R.get_or_init(|| {
        std::env::var("INK_MOE_NREP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1 && v <= 64)
            .unwrap_or(8)
    });
    let mut r = want;
    if m_total / MTILE < NREP_TILES * want {
        r = 1;
    }
    while r > 1 && (n / NTILE) % r != 0 {
        r -= 1;
    }
    r
}

/// M tiles a layer needs per repeated N tile before the wide register tile is
/// worth its registers. See [`grouped_nrep`] for the measurement that placed
/// it; the bracket it sits in is 83 to 135 tiles at `nrep` 8.
pub const NREP_TILES: usize = 16;

// ---------------------------------------------------------------------------
// Scatter
// ---------------------------------------------------------------------------

/// `out[t, :] = sum over t's rows of y[r, :] * wgt[r]`, in ascending `r`.
///
/// One thread per output element, walking that token's contributing rows and
/// accumulating in a register: no atomics, no second pass, and one write per
/// output element. `tok_rows` is `[n, kmax]` row-major with `tok_cnt[t]` valid
/// entries in each row, ascending — which is `BTreeMap` expert order, because
/// that is the order the rows were laid out in.
///
/// The gather side of this is a `k`-deep strided read per output element, so
/// the alternative shape — one thread per INPUT row, atomically adding into the
/// output — is a real candidate and not a forbidden one; see
/// [`scatter_weighted_atomic_kernel`] and the measurement in its doc.
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
    row_tok: &Handle,
    m_total: usize,
    n: usize,
    h: usize,
    kmax: usize,
) -> Handle {
    if std::env::var("INK_SCATTER").map(|v| v == "atomic").unwrap_or(false) {
        return scatter_weighted_atomic(client, y, wgt, row_tok, m_total, n, h);
    }
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

/// Zero an `[n]` f32 buffer. The atomic scatter accumulates INTO its output, so
/// the output has to start at zero, and `client.empty` does not promise that.
#[cube(launch_unchecked)]
fn zero_f32_kernel(out: &mut Array<f32>, total: usize) {
    let p = ABSOLUTE_POS as usize;
    if p < total {
        out[p] = f32::new(0.0f32);
    }
}

/// The same product as [`scatter_weighted_kernel`], shaped the other way round:
/// one thread per INPUT element, adding its weighted value into the output
/// atomically.
///
/// The two differ in what they make contiguous. The gather kernel writes each
/// output element once and reads `cnt` rows `h` apart to do it; this one reads
/// its input element once, coalesced with its whole warp, and pays an atomic
/// per term instead. Which wins is a property of `kmax`, `h` and how many rows
/// collide on one output element, i.e. of the routing — so it is measured, not
/// argued. `INK_SCATTER=atomic` selects it.
///
/// **Measured on this box, and it does not pay.** Layers 0:8, `INK_ALIGN_COPY=1`,
/// p50 of `pass_ms` over 24 warm passes:
///
///   512-token prefill    gather 339.8 ms   atomic 339.7 ms
///   decode, INK_KV=1     gather  46.4 ms   atomic  46.9 ms
///
/// Both differences are far inside the 2-3 ms run-to-run drift of one binary.
/// The arm stays because a negative with a number is worth keeping and because
/// the routing that makes it negative is not a constant of the model — a
/// heavier `kmax` or a narrower `h` moves the balance — but the gather kernel
/// stays the default.
///
/// An atomic float add reassociates the sum, so the two arms produce different
/// argmaxes on some positions. That is allowed and it is not what decides
/// between them: see the module doc's yardstick — this runtime already
/// disagrees with itself on 8.55% of argmax positions between two runs of the
/// same binary.
#[cube(launch_unchecked)]
fn scatter_weighted_atomic_kernel(
    y: &Array<f32>,
    wgt: &Array<f32>,
    row_tok: &Array<i32>,
    out: &mut Array<Atomic<f32>>,
    h: usize,
    total: usize,
) {
    let p = ABSOLUTE_POS as usize;
    if p < total {
        let r = p / h;
        let c = p % h;
        let t = row_tok[r];
        // `-1` is an MMA row-tile pad: a row that belongs to no token and whose
        // output is arithmetic on zeros.
        if t >= 0 {
            out[t as usize * h + c].fetch_add(y[p] * wgt[r]);
        }
    }
}

/// Launch [`scatter_weighted_atomic_kernel`], returning the `[n, h]` accumulator.
#[allow(clippy::too_many_arguments)]
pub fn scatter_weighted_atomic<R: Runtime>(
    client: &ComputeClient<R>,
    y: &Handle,
    wgt: &Handle,
    row_tok: &Handle,
    m_total: usize,
    n: usize,
    h: usize,
) -> Handle {
    let total = n * h;
    let out = client.empty(total * core::mem::size_of::<f32>());
    unsafe {
        zero_f32_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(total.div_ceil(CUBE_SIZE as usize) as u32),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(out.clone(), total),
            total,
        );
        let src = m_total * h;
        scatter_weighted_atomic_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(src.div_ceil(CUBE_SIZE as usize) as u32),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(y.clone(), src),
            ArrayArg::from_raw_parts(wgt.clone(), m_total),
            ArrayArg::from_raw_parts(row_tok.clone(), m_total),
            ArrayArg::from_raw_parts(out.clone(), total),
            h,
            src,
        );
    }
    out
}

/// The block plan, uploaded: three `[blocks]` arrays and the launch geometry
/// they imply.
///
/// One struct rather than five arguments because the three arrays, the block
/// count and the plane count are one decision — a launch that took `blocks`
/// from one plan and `slot` from another would read past the end of an array
/// and the kernels are launched unchecked.
pub struct BlockPlanDev {
    /// `[blocks]` expert slots.
    pub slot: Handle,
    /// `[blocks]` first m tiles.
    pub tile0: Handle,
    /// `[blocks]` run lengths.
    pub cnt: Handle,
    /// Cubes in the y dimension.
    pub blocks: usize,
    /// Planes per cube.
    pub planes: usize,
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
    /// `[blocks]`: which expert slot each cube serves.
    pub blk_slot: Vec<u32>,
    /// `[blocks]`: the first of the cube's run of m tiles.
    pub blk_tile0: Vec<u32>,
    /// `[blocks]`: how many m tiles the run holds, `1..=planes`.
    pub blk_cnt: Vec<u32>,
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
        planes: usize,
    ) -> RowPlan {
        assert!(planes >= 1, "a cube needs at least one plane");
        let mut row_tok: Vec<i32> = Vec::new();
        let mut row_wgt: Vec<f32> = Vec::new();
        let mut blk_slot: Vec<u32> = Vec::new();
        let mut blk_tile0: Vec<u32> = Vec::new();
        let mut blk_cnt: Vec<u32> = Vec::new();
        let mut per_tok: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut tile = 0usize;

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
            // The expert's tiles, cut into runs of at most `planes`. A run is a
            // cube, and it never straddles two experts, because the planes of a
            // cube share a B plane and that is the whole point of the shape.
            let tiles = m_pad / MTILE;
            let mut done = 0usize;
            while done < tiles {
                let cnt = (tiles - done).min(planes);
                blk_slot.push(slot as u32);
                blk_tile0.push((tile + done) as u32);
                blk_cnt.push(cnt as u32);
                done += cnt;
            }
            tile += tiles;
        }

        let kmax = per_tok.iter().map(|v| v.len()).max().unwrap_or(0).max(1);
        let mut tok_rows = vec![0u32; n * kmax];
        let mut tok_cnt = vec![0u32; n];
        for (t, rows) in per_tok.iter().enumerate() {
            tok_cnt[t] = rows.len() as u32;
            tok_rows[t * kmax..t * kmax + rows.len()].copy_from_slice(rows);
        }

        RowPlan { row_tok, row_wgt, blk_slot, blk_tile0, blk_cnt, tok_rows, tok_cnt, kmax }
    }

    /// How many planes a cube should have, from `INK_MOE_PLANES`.
    ///
    /// Read once and cached: it is a launch parameter, and a `getenv` per layer
    /// per pass in a lane whose subject is per-pass cost would be a joke at its
    /// own expense.
    pub fn planes() -> usize {
        use std::sync::OnceLock;
        static P: OnceLock<usize> = OnceLock::new();
        *P.get_or_init(|| {
            std::env::var("INK_MOE_PLANES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v >= 1 && v <= 32)
                .unwrap_or(4)
        })
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
        let plan = RowPlan::build([&e0, &e1, &e2].into_iter(), 2, 4);

        // Each expert is padded to a whole MMA row tile.
        assert_eq!(plan.m_total(), 3 * MTILE);
        // One tile each, so one block each even at four planes -- a run never
        // crosses an expert.
        assert_eq!(plan.blk_slot, vec![0, 1, 2]);
        assert_eq!(plan.blk_tile0, vec![0, 1, 2]);
        assert_eq!(plan.blk_cnt, vec![1, 1, 1]);

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

    /// A run stops at the expert boundary and at `planes`, whichever comes
    /// first -- the property the shared B plane rests on.
    #[test]
    fn runs_never_cross_an_expert_and_never_exceed_the_plane_count() {
        // 5 tiles, then 1, then 9: 65, 3 and 130 tokens.
        let big: Vec<(usize, f32)> = (0..(4 * MTILE + 1)).map(|i| (i, 1.0f32)).collect();
        let small: Vec<(usize, f32)> = (0..3).map(|i| (i, 1.0f32)).collect();
        let huge: Vec<(usize, f32)> = (0..(8 * MTILE + 2)).map(|i| (i, 1.0f32)).collect();
        let n = 8 * MTILE + 2;
        let plan = RowPlan::build([&big, &small, &huge].into_iter(), n, 4);

        assert_eq!(plan.blk_slot, vec![0, 0, 1, 2, 2, 2]);
        assert_eq!(plan.blk_cnt, vec![4, 1, 1, 4, 4, 1]);
        assert_eq!(plan.blk_tile0, vec![0, 4, 5, 6, 10, 14]);
        // Every tile of the layer is served exactly once.
        let served: usize = plan.blk_cnt.iter().map(|&c| c as usize).sum();
        println!("examined {} blocks covering {served} tiles", plan.blk_slot.len());
        assert_eq!(served, plan.m_total() / MTILE);
        assert_eq!(served, 5 + 1 + 9);
    }
}
