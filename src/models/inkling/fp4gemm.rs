//! Routed-expert FFN on the native NVFP4 tensor-core path.
//!
//! The existing device lane decodes each expert's packed E2M1/E4M3 blocks into
//! a full f32 matrix on the device and then multiplies that in f32. For one
//! expert that materialises 67.1 MB (w13) + 33.6 MB (w2) of f32 that is read
//! back once and thrown away. This module skips the decode: the packed bytes go
//! straight into `mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X…ue4m3`,
//! the instruction `nvfp4_mma_probe` proved CubeCL reaches on sm_121a.
//!
//! ## Why the activations are also 4-bit
//!
//! The MMA takes E2M1 for BOTH operands — there is no mixed f32xE2M1 form at
//! `kind::mxf4nvf4` — so the activation has to be quantised too. That is not a
//! liberty taken for speed: the checkpoint's own `hf_quant_config.json`
//! specifies
//!
//!     "*input_quantizer": num_bits [2,1], block_sizes {-1: 16,
//!                         type: "dynamic", scale_bits: [4,3]}, enable: true
//!
//! i.e. E2M1 activations in dynamic per-16 blocks with E4M3 scales — exactly
//! what this path feeds the instruction. The f32-activation lane it replaces is
//! the one deviating from the checkpoint's intended numerics, not this one.
//!
//! ## What this is NOT
//!
//! It is not a fast GEMM. Each plane owns one 16x8 output tile and streams its
//! own eight weight rows from global memory, so the weights are read exactly
//! once and the activation (a few KB) is re-read per plane out of L2. For the
//! shape this lane actually runs — M is the handful of tokens that routed to
//! one expert, against N=4096, K=4096 — the arithmetic intensity is so low that
//! the kernel is bound by streaming the weights in. `inkling_forward`'s own
//! per-pass report breaks the lane into slice / bind+enqueue+sync / remainder,
//! which is where that number comes from now that the lane-comparison bench is
//! gone with the lanes it compared.
//!
//! ### "and a fancier tiling would not change that" — measured, and it is wrong
//!
//! Bound by streaming is right. Bound AT THE BUS is not, and the two are the
//! same sentence only if you never put a number on it.
//!
//! `nsys -t cuda` over the DECODE lane alone — spark-zt (GB10), commit
//! 56c1ebbcdff6, `INK_KV=1 INK_LAYERS=0:16`, the 3732-token cover, 39 decode
//! passes inside the capture window and no prefill in it — attributes, PER
//! DECODE PASS:
//!
//! ```text
//!   kernel                          launches   ms/pass   weight bytes   GB/s
//!   fp4_linear_grouped (13 layers)      26.6     7.052       1.083 GB    154
//!   bf16_linear_grouped (layer 2)        2.0     1.897       0.302 GB    159
//!   matmul_entry, dense gate/up          4.0     2.176       0.537 GB    247
//!   matmul_entry, shared gate_up        14.3     4.128       0.958 GB    232
//! ```
//!
//! The last two rows are the SAME PASS on the SAME BUS, so they are the
//! control: 247 GB/s is reachable here, and a microbenchmark independently puts
//! achievable streaming read at 242.9 GB/s. This lane gets 154-159 — and it
//! gets the same 154-159 for w13 (56.6 MB a layer) as for w2 (28.3 MB), and
//! again for the BF16 sibling, which is what rules out latency and dtype and
//! leaves the schedule.
//!
//! That reading is **confirmed**, and the competing one — that a packed
//! four-bit read simply cannot go faster on this part, so 171 GB/s was already
//! the ceiling — is **refuted**. `inkling_membw` with `INK_BW_AXES=1` puts both
//! candidate ceilings and this kernel in ONE process on ONE 1 GiB handle:
//! f32, BF16 and packed NVFP4 all read 247-259 GB/s, the E4M3 scale plane
//! alongside them is free, and `fp4_linear` at `m_pad = 16` on that same table
//! reads 105.8. The gap is the `m16n8k64` B fragment's access pattern — a
//! plane's B load spans eight weight rows `k/2` bytes apart, eight sector
//! requests an instruction against a coalesced stream's four — and NOT the
//! dtype. `moegroup`'s header carries the full table and the retraction of the
//! 170.4 GB/s figure that supported the other reading.
//!
//! So of the 8.95 ms a decode pass spends in the grouped expert GEMM, 5.7 ms is
//! bytes / bus and **3.2 ms is not** — 5.5% of a 58.3 ms decode step, sitting in
//! the kernel rather than in the router or the host. A fancier tiling is
//! exactly where it would have to come from.
//!
//! What it is NOT, checked before writing this: plane fill. At decode every
//! expert has one 16-row tile, so [`super::moegroup::RowPlan`] gives every cube
//! `blk_cnt = 1` and three of a four-plane cube's planes `terminate!()` at the
//! first branch. `INK_MOE_PLANES=1` is therefore the obvious fix and it is not
//! one: measured against base in the same interleaved run it is no better, and
//! `grouped_nrep`'s header records the same negative at 128 tokens. The idle
//! planes are not the 3.2 ms.
//!
//! ## What the grid order is worth, at the shapes where M is not 1
//!
//! "The weights are read exactly once" was true only of the ONE m-tile decode
//! runs. Every extra m-tile wants the same weight rows, and whether it gets
//! them out of L2 or out of DRAM is decided by the launch order alone. With N
//! in grid x — which is what this was until the axes were swapped — the
//! consumers of one weight row sat `n / 8` cubes apart, so time scaled exactly
//! linearly with `m_pad`: no reuse whatever.
//!
//! `fp4_lane_dump` at the head shape (`k = 4096`, `n = 201024`, 0.431 GiB of
//! codes + scales, `INK_SKIP_BF16=1`, min of four warm launches, launch + sync
//! with no host readback, DGX Spark GB10 / sm_121a, GPU otherwise idle). These
//! are KERNEL times, not stage times, and they include the launcher's own
//! output `client.empty` — which the same harness reports separately at 0.01 to
//! 0.3 ms, i.e. beneath the differences below:
//!
//! ```text
//!   m_pad   N in x     M in x     ratio
//!      16   4.55 ms    4.53 ms    1.00   one m-tile: nothing to share, as expected
//!      32   8.95       4.54       1.97   the second m-tile becomes FREE
//!      64  18.57       7.71       2.41
//!     128  34.97      15.31       2.28
//! ```
//!
//! The linearity is broken but not gone: from `m_pad` 32 up the cost still
//! roughly doubles per doubling, so a wave's worth of m-tiles is being served
//! and the rest is not. What sets that ceiling is NOT settled here — the
//! candidates (resident-cube count, L2 capacity against the concurrent n-tile
//! span, the output write, which grows from 12.9 MB to 103 MB across this
//! sweep) are not separable with a launch-and-sync timer.
//!
//! At `n <= 16384` the whole weight table is L2-sized already and the sweep
//! shows no reliable difference; below ~1 ms the harness is measuring host
//! jitter, not the kernel. The win is a LARGE-N property.

use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::{e2m1x2, e4m3};

/// Rows of one MMA tile — the M granularity everything here is padded to.
pub const MTILE: usize = 16;
/// Columns of one MMA tile.
pub const NTILE: usize = 8;
/// K covered by one `m16n8k64` instruction.
pub const KTILE: usize = 64;
/// Logical elements per E4M3 block scale.
pub const GROUP: usize = 16;
/// E4M3 block scales per vector load, i.e. per `mma` operand.
///
/// Not a tuning knob: it is `MmaDefinition::scales_vector_size()`, which
/// `cubecl-core`'s `frontend/cmma.rs` defines as the MMA register width over
/// the scale element width, 32/8. It is also `KTILE / GROUP`, the number of
/// block scales one instruction consumes per operand row
/// (`MmaDefinition::scales_count()`), so the vector the instruction takes IS
/// the vector the memory holds and there is nothing to pad or assemble.
pub const SCALE_VEC: usize = KTILE / GROUP;

/// `out = (a @ b^T) * scale`, with `a` and `b` both NVFP4.
///
/// `a` is `[m_pad, k/2]` packed bytes and `a_sc` `[m_pad, k/16]` E4M3 scales;
/// `b` is `[n, k/2]` / `[n, k/16]`, i.e. the checkpoint's own `[out, in]`
/// orientation, which is already the column-major B the instruction wants.
/// `out` is `[m_pad, n]` f32.
///
/// One plane per `(m_tile, n_tile)`; the K loop accumulates in the MMA's own
/// f32 accumulator, which measured closer to an f64 sum than a sequential f32
/// host lane over the same products.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear<AB: Scalar, S: Scalar, NA: Size, NC: Size>(
    a: &Array<Vector<AB, NA>>,
    a_sc: &Array<S>,
    b: &Array<Vector<AB, NA>>,
    b_sc: &Array<S>,
    out: &mut Array<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    // M in x, N in y. Grid x varies fastest, so the cubes that share a weight
    // row — the whole n-tile's `[8, k]` of codes and scales — are the ones
    // launched adjacently, and one DRAM read of that row serves all
    // `m_pad / 16` of them out of L2. The other order puts consumers of the
    // same row `n / 8` cubes apart, which at the head shape is the whole
    // weight table between them, so every m-tile re-reads from DRAM.
    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize;
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;

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
            reg_b[i] = b[(gr * size_k / 2 + gc / 2) / b.vector_size()];
        }

        let mut sa = Vector::<S, NS>::empty();
        let mut sb = Vector::<S, NS>::empty();
        #[unroll]
        for i in 0..scales_count {
            sa[i] = a_sc[(sia + m_base) * spr + t * 4 + i];
            sb[i] = b_sc[(sib + n_base) * spr + t * 4 + i];
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
        out[(gr * size_n + gc) / out.vector_size()] = acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// [`fp4_linear`] with the B operand staged through shared memory.
///
/// The dense sibling of
/// [`super::moegroup::fp4_linear_grouped_smem`], and the same argument: the
/// `m16n8k64` B fragment is 8 columns of `[n, k]`, so a plane's B load spans
/// eight weight rows `k / 2` bytes apart and spends eight sector requests on
/// 128 useful bytes where a coalesced 32-bit stream spends four. The global
/// read becomes a per-row contiguous stream, the plane fills shared memory, and
/// only the smem read keeps the fragment's shape. Nothing about the k order,
/// the operands or the accumulator changes, so this is bit-identical to
/// [`fp4_linear`].
///
/// A cube here is ONE plane ([`fp4_linear_launch`] launches `CubeDim(32)`), so
/// the fill is not cooperative across planes and there is no barrier to pay:
/// the warp stages its own eight rows and reads them back. `sync_plane` is
/// still needed — shared memory written by one lane and read by another is not
/// ordered by the lockstep the fragment layout otherwise relies on.
///
/// `stage_sc` extends the staging to the block scales, and it is a separate
/// knob because the two are separate defects. This kernel reads its four E4M3
/// scales as four INDIVIDUAL `Array<S>` bytes — four instructions, each
/// spanning the same eight rows, i.e. 32 sector requests a k tile against the
/// codes' 16 — where the grouped kernel already reads them as one 32-bit
/// vector. Staging them collapses that; leaving it off measures the code
/// staging alone against an unchanged baseline.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear_smem<AB: Scalar, S: Scalar, NA: Size, NC: Size>(
    a: &Array<Vector<AB, NA>>,
    a_sc: &Array<S>,
    b: &Array<Vector<AB, NA>>,
    b_sc: &Array<S>,
    out: &mut Array<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] kc: usize,
    #[comptime] pad: usize,
    #[comptime] stage_sc: bool,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize;
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let cs = comptime!(kc * 8 + pad);
    let ss = comptime!(kc * 4 + pad);
    let chunks = comptime!(size_k / KTILE / kc);
    let words = comptime!(NTILE * kc * 8);
    let words_s = comptime!(NTILE * kc * 4);
    let per_c = comptime!(words.div_ceil(32));
    let per_s = comptime!(words_s.div_ceil(32));

    let mut sm = SharedMemory::<Vector<AB, NA>>::new(comptime!(NTILE * cs));
    let mut sm_sc = SharedMemory::<S>::new(comptime!(if stage_sc { NTILE * ss } else { 1usize }));

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let size!(NS) = def.scales_vector_size();
    let scales_count = def.scales_count();
    let sia = def.scales_index(lane, MatrixIdent::A) as usize;
    let sib = def.scales_index(lane, MatrixIdent::B) as usize;
    let spr = comptime!(size_k / GROUP);
    let u = lane as usize;

    for c in 0..chunks {
        // The fill. A warp covers `kc * 32` consecutive bytes of ONE weight row
        // per pass, which at `kc >= 4` is whole 128-byte lines fully used.
        #[unroll]
        for j in 0..per_c {
            let f = u + j * 32;
            if f < words {
                let r = f / comptime!(kc * 8);
                let o = f % comptime!(kc * 8);
                let gi = ((n_base + r) * size_k / 2 + c * comptime!(kc * KTILE / 2))
                    / b.vector_size()
                    + o;
                sm[r * cs + o] = b[gi];
            }
        }
        if comptime![stage_sc] {
            #[unroll]
            for j in 0..per_s {
                let f = u + j * 32;
                if f < words_s {
                    let r = f / comptime!(kc * 4);
                    let o = f % comptime!(kc * 4);
                    sm_sc[r * ss + o] = b_sc[(n_base + r) * spr + c * comptime!(kc * 4) + o];
                }
            }
        }
        sync_plane();

        #[unroll]
        for tl in 0..kc {
            let t = c * kc + tl;
            let kbase = t * KTILE;
            #[unroll]
            for i in 0..vc_a {
                let (row, col) =
                    def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
                let gr = row as usize + m_base;
                let gc = col as usize + kbase;
                reg_a[i] = a[(gr * size_k / 2 + gc / 2) / a.vector_size()];
            }
            #[unroll]
            for i in 0..vc_b {
                let (row, col) =
                    def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
                // `col` picks the weight row inside the staged tile, `row` the
                // k element; `row` is a multiple of 8 for every fragment
                // element, so the word inside the chunk is exact.
                reg_b[i] = sm[col as usize * cs + row as usize / 8 + tl * 8];
            }

            let mut sa = Vector::<S, NS>::empty();
            let mut sb = Vector::<S, NS>::empty();
            #[unroll]
            for i in 0..scales_count {
                sa[i] = a_sc[(sia + m_base) * spr + t * 4 + i];
                if comptime![stage_sc] {
                    sb[i] = sm_sc[sib * ss + tl * 4 + i];
                } else {
                    sb[i] = b_sc[(sib + n_base) * spr + t * 4 + i];
                }
            }

            let d = def.execute_scaled(&reg_a, &reg_b, &acc, sa, sb);
            #[unroll]
            for i in 0..vc_c {
                acc[i] = d[i];
            }
        }
        sync_plane();
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let gr = row as usize + m_base;
        let gc = col as usize + n_base;
        out[(gr * size_n + gc) / out.vector_size()] = acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// Launch [`fp4_linear_smem`]; the arguments of [`fp4_linear_launch`] plus the
/// staging parameters.
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear_smem_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    a_sc: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    scale: f32,
    kc: usize,
    pad: usize,
    stage_sc: bool,
) -> Handle {
    assert_eq!(m_pad % MTILE, 0);
    assert_eq!(n % NTILE, 0);
    assert_eq!(k % KTILE, 0);
    assert_eq!(
        (k / KTILE) % kc,
        0,
        "k tiles are not a whole number of chunks"
    );
    assert!(n / NTILE <= 65535);

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let vs = 32 / e2m1x2::cube_type().size_bits();
    let spr = k / GROUP;

    unsafe {
        fp4_linear_smem::launch::<e2m1x2, e4m3, R>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            ArrayArg::from_raw_parts(a.clone(), m_pad * (k / 2)),
            ArrayArg::from_raw_parts(a_sc.clone(), m_pad * spr),
            ArrayArg::from_raw_parts(b.clone(), n * (k / 2)),
            ArrayArg::from_raw_parts(b_sc.clone(), n * spr),
            ArrayArg::from_raw_parts(out.clone(), m_pad * n),
            k,
            n,
            kc,
            pad,
            stage_sc,
            scale,
        )
    };
    out
}

/// De-interleave the fused gate/up result and apply the gate, in one pass.
///
/// The checkpoint stores w13's output rows alternating `g0, u0, g1, u1, …`, so
/// after `out = x @ w13^T` column `2i` is the gate and `2i + 1` the up. Doing
/// the de-interleave here, on the `[m, 2*inter]` result, moves it off the
/// `[2*inter, hidden]` weight — 16x2048 elements touched instead of 4096x4096.
#[cube(launch)]
pub fn gate_up_silu<I: Scalar + Cast, O: Scalar + Cast>(
    both: &Array<I>,
    act: &mut Array<O>,
    #[comptime] inter: usize,
    #[comptime] halved: bool,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx < act.len() {
        let r = idx / inter;
        let i = idx % inter;
        // Two readings of w13's output axis are live in this tree: INTERLEAVED
        // (g0,u0,g1,u1,...) and HALVED (all gates, then all ups). They are
        // shape-identical and numerically different, which is exactly the kind
        // of thing that passes every parity gate built on the same assumption.
        // `halved` exists so the question can be settled by running it.
        let (g, u) = if comptime!(halved) {
            (
                f32::cast_from(both[r * 2 * inter + i]),
                f32::cast_from(both[r * 2 * inter + inter + i]),
            )
        } else {
            (
                f32::cast_from(both[r * 2 * inter + 2 * i]),
                f32::cast_from(both[r * 2 * inter + 2 * i + 1]),
            )
        };
        // The output type is the NEXT matmul's operand type: f32 for the NVFP4
        // lane, whose second GEMM re-quantises from f32 anyway, and bf16 for
        // the layer-2 lane, whose second GEMM takes bf16 directly. ONE
        // implementation of the interleave, because the INTERLEAVED/HALVED
        // question above is exactly the kind that a second transcription gets
        // silently wrong. `O::cast_from` is the identity when `O` is f32.
        act[idx] = O::cast_from((g / (1.0f32 + Exp::exp(-g))) * u);
    }
}

/// Launch [`fp4_linear`] for a `[m_pad, k] x [n, k]^T` product.
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    a_sc: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Handle {
    assert_eq!(
        m_pad % MTILE,
        0,
        "m_pad {m_pad} is not a multiple of {MTILE}"
    );
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");
    // N rides grid y, which CUDA caps at 65535 (x is 2^31-1). The largest N in
    // the model is the unembedding's 201024 = 25128 tiles, well inside it, but
    // the cap is silent if it is ever exceeded so it is checked here.
    assert!(
        n / NTILE <= 65535,
        "{} n-tiles exceed the 65535 grid-y limit",
        n / NTILE
    );

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let vs = 32 / e2m1x2::cube_type().size_bits();
    let spr = k / GROUP;

    unsafe {
        fp4_linear::launch::<e2m1x2, e4m3, R>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            ArrayArg::from_raw_parts(a.clone(), m_pad * (k / 2)),
            ArrayArg::from_raw_parts(a_sc.clone(), m_pad * spr),
            ArrayArg::from_raw_parts(b.clone(), n * (k / 2)),
            ArrayArg::from_raw_parts(b_sc.clone(), n * spr),
            ArrayArg::from_raw_parts(out.clone(), m_pad * n),
            k,
            n,
            scale,
        )
    };
    out
}

/// Launch [`gate_up_silu`] over an `[m_pad, 2 * inter]` fused result, f32 out.
pub fn gate_up_silu_launch<R: Runtime>(
    client: &ComputeClient<R>,
    both: &Handle,
    m_pad: usize,
    inter: usize,
) -> Handle {
    gate_up_silu_launch_as::<f32, f32, R>(client, both, m_pad, inter)
}

/// The same, BF16 on BOTH sides: a narrow gate-and-up in, a narrow activation
/// out. The gate and the SiLU are computed in f32 regardless.
pub fn gate_up_silu_narrow_launch<R: Runtime>(
    client: &ComputeClient<R>,
    both: &Handle,
    m_pad: usize,
    inter: usize,
) -> Handle {
    gate_up_silu_launch_as::<half::bf16, half::bf16, R>(client, both, m_pad, inter)
}

/// The same, BF16 out — what the layer-2 lane feeds straight back into the MMA.
///
/// A separate entry point rather than a turbofish at the call site so the two
/// lanes read the same, and so nothing but the element type differs between
/// them.
pub fn gate_up_silu_bf16_launch<R: Runtime>(
    client: &ComputeClient<R>,
    both: &Handle,
    m_pad: usize,
    inter: usize,
) -> Handle {
    gate_up_silu_launch_as::<f32, half::bf16, R>(client, both, m_pad, inter)
}

fn gate_up_silu_launch_as<I: Scalar + Cast, O: Scalar + Cast + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    both: &Handle,
    m_pad: usize,
    inter: usize,
) -> Handle {
    // INK_W13_HALVED=1 selects the contiguous reading, for the A/B.
    let halved = std::env::var("INK_W13_HALVED")
        .map(|v| v == "1")
        .unwrap_or(false);
    let n = m_pad * inter;
    let act = client.empty(n * core::mem::size_of::<O>());
    let threads = 256u32;
    let blocks = n.div_ceil(threads as usize) as u32;
    unsafe {
        gate_up_silu::launch::<I, O, R>(
            client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(both.clone(), m_pad * 2 * inter),
            ArrayArg::from_raw_parts(act.clone(), m_pad * inter),
            inter,
            halved,
        )
    };
    act
}

// ---------------------------------------------------------------------------
// Activation quantisation
// ---------------------------------------------------------------------------

/// Quantise activations to NVFP4: E2M1 codes, one E4M3 scale per 16.
///
/// One unit per 16-element block. `x` is `[rows, k]` f32 flattened; `codes` is
/// `[rows, k/8]` u32 with element `i` of a block at bits `4*(i%8)` of word
/// `i/8` (low nibble first, so the bytes match the checkpoint's own packing and
/// the same buffer can be bound as `e2m1x2`); `scales` is `[rows, k/16]` E4M3.
///
/// The recipe is the checkpoint's: `scale = amax/6` rounded to E4M3, then each
/// element rounded to the nearest E2M1 code of `x/scale`. Rounding is
/// round-to-nearest with exact midpoints going AWAY from zero (a midpoint lands
/// on the `<` boundary and falls through to the larger code). An all-zero block
/// yields a zero scale byte and zero codes.
#[cube(launch)]
pub fn quantize_act(x: &Array<f32>, codes: &mut Array<u32>, scales: &mut Array<e4m3>) {
    let blk = ABSOLUTE_POS as usize;
    if blk < scales.len() {
        let base = blk * GROUP;

        let mut amax = 0.0f32;
        #[unroll]
        for i in 0..GROUP {
            let v = Abs::abs(x[base + i]);
            if v > amax {
                amax = v;
            }
        }

        // Round the block scale through E4M3 and read back what it became: the
        // codes have to be computed against the scale the MMA will actually
        // apply, not the exact amax/6 the host imagined.
        let sq = e4m3::cast_from(amax / 6.0f32);
        let s = f32::cast_from(sq);
        scales[blk] = sq;

        let mut w0 = 0u32;
        let mut w1 = 0u32;
        if s > 0.0f32 {
            let inv = 1.0f32 / s;
            #[unroll]
            for i in 0..GROUP {
                let q = x[base + i] * inv;
                let a = Abs::abs(q);
                // magnitude grid 0, .5, 1, 1.5, 2, 3, 4, 6 -> midpoints below
                let mut m = 7u32;
                if a < 0.25f32 {
                    m = 0u32;
                } else if a < 0.75f32 {
                    m = 1u32;
                } else if a < 1.25f32 {
                    m = 2u32;
                } else if a < 1.75f32 {
                    m = 3u32;
                } else if a < 2.5f32 {
                    m = 4u32;
                } else if a < 3.5f32 {
                    m = 5u32;
                } else if a < 5.0f32 {
                    m = 6u32;
                }
                let c = if q < 0.0f32 { m + 8u32 } else { m };
                if i < 8 {
                    w0 |= c << (4 * i as u32);
                } else {
                    w1 |= c << (4 * (i - 8) as u32);
                }
            }
        }
        codes[2 * blk] = w0;
        codes[2 * blk + 1] = w1;
    }
}

/// Host-side twin of [`quantize_act`], for gates and for the CPU lane.
///
/// Returns `(packed_bytes, scale_bytes)` in exactly the layout the device
/// kernel writes, so a gate can compare them bitwise.
pub fn quantize_act_host(x: &[f32], k: usize) -> (Vec<u8>, Vec<u8>) {
    use crate::models::inkling::nvfp4::e4m3_to_f32;
    assert_eq!(x.len() % k, 0, "x is not a whole number of rows of {k}");
    assert_eq!(k % GROUP, 0, "{k} is not a multiple of {GROUP}");
    let nblocks = x.len() / GROUP;
    let mut codes = vec![0u8; x.len() / 2];
    let mut scales = vec![0u8; nblocks];
    for b in 0..nblocks {
        let base = b * GROUP;
        let amax = (0..GROUP).map(|i| x[base + i].abs()).fold(0.0f32, f32::max);
        let sb = f32_to_e4m3(amax / 6.0);
        scales[b] = sb;
        let s = e4m3_to_f32(sb);
        if !(s > 0.0) {
            continue;
        }
        for i in 0..GROUP {
            let q = x[base + i] / s;
            let a = q.abs();
            let m: u8 = if a < 0.25 {
                0
            } else if a < 0.75 {
                1
            } else if a < 1.25 {
                2
            } else if a < 1.75 {
                3
            } else if a < 2.5 {
                4
            } else if a < 3.5 {
                5
            } else if a < 5.0 {
                6
            } else {
                7
            };
            let c = if q < 0.0 { m + 8 } else { m };
            let j = base + i;
            if j % 2 == 0 {
                codes[j / 2] |= c;
            } else {
                codes[j / 2] |= c << 4;
            }
        }
    }
    (codes, scales)
}

/// Round a non-negative f32 to the nearest E4M3 (bias 7, 3 mantissa bits) byte.
///
/// Exhaustive rather than clever: E4M3FN has 256 patterns and the finite
/// non-negative ones are 128, so scanning them is both obviously correct and
/// fast enough for the few thousand block scales an expert needs. A hand-rolled
/// bit twiddle is what got the subnormal branch of `e4m3_to_f32` wrong once
/// already.
pub fn f32_to_e4m3(v: f32) -> u8 {
    use crate::models::inkling::nvfp4::e4m3_to_f32;
    if !(v > 0.0) {
        return 0;
    }
    let mut best = 0u8;
    let mut bestd = f32::INFINITY;
    for b in 0u16..128 {
        let d = e4m3_to_f32(b as u8);
        if !d.is_finite() {
            continue;
        }
        let e = (d - v).abs();
        if e < bestd {
            bestd = e;
            best = b as u8;
        }
    }
    best
}

/// Quantise an `[rows, k]` f32 host slice and upload it, ready for
/// [`fp4_linear_launch`]. `rows` is padded up to a multiple of [`MTILE`].
pub fn upload_quantized_act<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    rows: usize,
    k: usize,
) -> (Handle, Handle, usize) {
    let m_pad = rows.div_ceil(MTILE) * MTILE;
    let mut padded = vec![0f32; m_pad * k];
    padded[..rows * k].copy_from_slice(&x[..rows * k]);
    let (codes, scales) = quantize_act_host(&padded, k);
    (
        client.create_from_slice(&codes),
        client.create_from_slice(&scales),
        m_pad,
    )
}

// ---------------------------------------------------------------------------
// Zero copy
// ---------------------------------------------------------------------------

/// Every host mapping a weight source reads through, registered with the GPU
/// **once**.
///
/// The obvious way to alias — call
/// [`ComputeClient::register_external_aliased`] per expert slab — is a trap on
/// this backend, and measurably so: it cost 2.6 s more per forward than simply
/// copying. `create_from_slice` posts its work with `submit`, which returns
/// immediately, but `register_external_aliased` has to hand back a `Handle` the
/// server constructs, so it uses `submit_blocking` — a synchronous round trip
/// to the device thread. At four slabs per expert and ~9950 expert-loads that
/// is ~40 000 blocking hops, and they cost far more than the copies they save.
///
/// Registering is per *mapping*: nine round trips for a sharded checkpoint, ONE
/// for a pile. Everything after that is [`Handle::offset_start`], pure
/// arithmetic on the client side.
///
/// A slab is located by POINTER CONTAINMENT rather than by re-deriving its file
/// offset from a tensor name, an expert index and a shape. That is what lets one
/// implementation serve both sources — the caller already holds the borrowed
/// bytes, and where they live is a fact about the pointer, not something to
/// recompute and get subtly wrong.
pub struct Aliases {
    /// `(base address, length, registered handle)` per mapping.
    maps: Vec<(usize, usize, Handle)>,
    /// What the binds actually did. See [`BindStats`].
    stats: BindCounters,
}

/// Why one bind did or did not become a zero-copy alias.
///
/// The distinction between the two copy causes is the whole value of counting:
/// an unaligned copy is a fact about how the SOURCE lays its bytes out and is
/// fixable by changing the source, while an unmapped one means the registration
/// never happened and no amount of alignment will help. A single "copied"
/// counter conflates a data-layout problem with a setup problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    /// Aliased in place — the GPU reads the source's own pages.
    Alias,
    /// Copied because the slab's ADDRESS is not 4-byte aligned. Carries the
    /// residue, because WHICH residue says where the misalignment came from: a
    /// safetensors shard packs tensors back to back with no padding, so a
    /// residue of 2 is an odd number of BF16 elements sitting upstream.
    CopyUnaligned(usize),
    /// Copied because the slab lives in no registered mapping — a `Vec` the
    /// caller built, or a source whose mappings were never registered.
    CopyUnmapped,
    /// Nothing to bind.
    Empty,
}

/// Interior-mutable counters. `slice_or_copy` takes `&self` because every
/// caller holds a shared reference, so the accounting has to be atomic rather
/// than `&mut`.
#[derive(Default)]
struct BindCounters {
    alias_calls: core::sync::atomic::AtomicU64,
    alias_bytes: core::sync::atomic::AtomicU64,
    copy_calls: core::sync::atomic::AtomicU64,
    copy_bytes: core::sync::atomic::AtomicU64,
    copy_nanos: core::sync::atomic::AtomicU64,
    /// Unaligned copies by residue mod 4; index 0 is unused.
    unaligned: [core::sync::atomic::AtomicU64; 4],
    unmapped: core::sync::atomic::AtomicU64,
}

/// What the binds of one run cost, split by whether they aliased.
///
/// Not a profiler. It answers the one question the zero-copy seam exists to
/// answer — how much of the weight traffic actually avoided a copy — and, when
/// the answer is "not all of it", which of the two reasons was to blame.
#[derive(Default, Clone, Copy, Debug)]
pub struct BindStats {
    pub alias_calls: u64,
    pub alias_bytes: u64,
    pub copy_calls: u64,
    pub copy_bytes: u64,
    /// HOST time inside `create_from_slice`. The copy itself is posted
    /// asynchronously, so this is the staging and enqueue, not the DMA.
    pub copy_nanos: u64,
    /// Unaligned copies by residue mod 4; index 0 is unused.
    pub unaligned: [u64; 4],
    pub unmapped: u64,
}

impl BindStats {
    pub fn calls(&self) -> u64 {
        self.alias_calls + self.copy_calls
    }

    /// Fraction of BINDS that aliased. `None` when nothing was bound, which is
    /// the honest answer — a rate over zero calls is not 0% or 100%.
    pub fn alias_fraction(&self) -> Option<f64> {
        match self.calls() {
            0 => None,
            n => Some(self.alias_calls as f64 / n as f64),
        }
    }

    /// Fraction of BYTES that aliased. Different from the call fraction
    /// whenever the two classes are different sizes, which they are here: a
    /// code plane is eight times a scale plane.
    pub fn alias_byte_fraction(&self) -> Option<f64> {
        match self.alias_bytes + self.copy_bytes {
            0 => None,
            n => Some(self.alias_bytes as f64 / n as f64),
        }
    }

    pub fn report(&self) -> String {
        let mb = |b: u64| b as f64 / (1u64 << 20) as f64;
        let mut s = String::new();
        s.push_str(&format!(
            "    bind ALIAS  {:8} calls  {:10.0} MiB   0.000 s\n",
            self.alias_calls,
            mb(self.alias_bytes)
        ));
        s.push_str(&format!(
            "    bind COPY   {:8} calls  {:10.0} MiB  {:6.3} s\n",
            self.copy_calls,
            mb(self.copy_bytes),
            self.copy_nanos as f64 / 1e9
        ));
        match (self.alias_fraction(), self.alias_byte_fraction()) {
            (Some(c), Some(b)) => s.push_str(&format!(
                "    aliased     {:8.1}% of binds, {:.1}% of bytes\n",
                c * 100.0,
                b * 100.0
            )),
            _ => s.push_str("    aliased     (nothing was bound)\n"),
        }
        // Only printed when there is something to explain. A line of zeroes
        // reads as a finding.
        if self.copy_calls > 0 {
            let residues: Vec<String> = (1..4)
                .filter(|i| self.unaligned[*i] > 0)
                .map(|i| format!("{} at addr%4=={i}", self.unaligned[i]))
                .collect();
            if !residues.is_empty() {
                s.push_str(&format!(
                    "    copied because UNALIGNED: {}\n",
                    residues.join(", ")
                ));
            }
            if self.unmapped > 0 {
                s.push_str(&format!(
                    "    copied because UNMAPPED : {} (outside every registered mapping)\n",
                    self.unmapped
                ));
            }
        }
        s
    }
}

impl Aliases {
    /// Register every mapping of a source. `None` if the device cannot address
    /// host memory directly.
    pub fn register<R: Runtime>(
        client: &ComputeClient<R>,
        mappings: Vec<(
            usize,
            usize,
            std::sync::Arc<dyn core::any::Any + Send + Sync>,
        )>,
    ) -> Option<Self> {
        if !cubecl::cuda::supports_zero_copy_host(0) {
            return None;
        }
        let mut maps = Vec::with_capacity(mappings.len());
        for (base, len, keep) in mappings {
            // SAFETY: the mapping is read-only and `keep` holds it for as long
            // as the handle lives; cubecl pins external handles immutable.
            let h = unsafe {
                client.register_external_aliased(
                    base as *mut core::ffi::c_void,
                    len as u64,
                    0,
                    len as u64,
                    keep,
                )
            };
            maps.push((base, len, h));
        }
        Some(Aliases {
            maps,
            stats: BindCounters::default(),
        })
    }

    /// An `Aliases` that aliases NOTHING, so the copying lane is still counted.
    ///
    /// Without this, `INK_ZEROCOPY=0` and "this device cannot alias" are both
    /// spelled `None` at the call site and neither reports what it moved — the
    /// A/B has a measured side and an unmeasured one, which is not an A/B.
    pub fn disabled() -> Self {
        Aliases {
            maps: Vec::new(),
            stats: BindCounters::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.maps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.maps.is_empty()
    }

    /// The binds so far.
    pub fn stats(&self) -> BindStats {
        use core::sync::atomic::Ordering::Relaxed;
        BindStats {
            alias_calls: self.stats.alias_calls.load(Relaxed),
            alias_bytes: self.stats.alias_bytes.load(Relaxed),
            copy_calls: self.stats.copy_calls.load(Relaxed),
            copy_bytes: self.stats.copy_bytes.load(Relaxed),
            copy_nanos: self.stats.copy_nanos.load(Relaxed),
            unaligned: [
                0,
                self.stats.unaligned[1].load(Relaxed),
                self.stats.unaligned[2].load(Relaxed),
                self.stats.unaligned[3].load(Relaxed),
            ],
            unmapped: self.stats.unmapped.load(Relaxed),
        }
    }

    /// Zero the counters, so a per-token figure is a per-token figure.
    pub fn stats_reset(&self) {
        use core::sync::atomic::Ordering::Relaxed;
        self.stats.alias_calls.store(0, Relaxed);
        self.stats.alias_bytes.store(0, Relaxed);
        self.stats.copy_calls.store(0, Relaxed);
        self.stats.copy_bytes.store(0, Relaxed);
        self.stats.copy_nanos.store(0, Relaxed);
        for u in &self.stats.unaligned {
            u.store(0, Relaxed);
        }
        self.stats.unmapped.store(0, Relaxed);
    }

    /// What [`Aliases::slice`] would decide, and WHY — without binding anything.
    ///
    /// Split out from `slice` so the decision can be audited over a whole model
    /// without a GPU and without a `Handle` per leaf. Both `slice` and the audit
    /// call this, so a check that says "every leaf aliases" is reading the same
    /// predicate the runtime does rather than a second transcription of it.
    pub fn classify(&self, data: &[u8]) -> Bind {
        if data.is_empty() {
            return Bind::Empty;
        }
        let p = data.as_ptr() as usize;
        if p % 4 != 0 {
            return Bind::CopyUnaligned(p % 4);
        }
        match self
            .maps
            .iter()
            .find(|(b, l, _)| p >= *b && p + data.len() <= b + l)
        {
            Some(_) => Bind::Alias,
            None => Bind::CopyUnmapped,
        }
    }

    /// A borrowed slice as a zero-copy offset view of the mapping it lives in.
    ///
    /// `None` when the slice is not 4-byte aligned — the expert GEMM issues
    /// 4-byte vector loads, so that is the real bound and 16 would be a
    /// superstition. It matters that this is the true bound and not a guess:
    /// safetensors packs tensors back to back with no padding, and the
    /// checkpoint puts `w13_weight` and `w2_weight` at offsets congruent to 4
    /// mod 16, so a 16-byte test would refuse every weight slab in the model and
    /// fall back to copying forever while looking like it worked.
    ///
    /// `None` also when the slice belongs to no registered mapping, which is the
    /// honest answer for a `Vec` the caller built.
    pub fn slice(&self, data: &[u8]) -> Option<Handle> {
        if !matches!(self.classify(data), Bind::Alias) {
            return None;
        }
        let p = data.as_ptr() as usize;
        let (base, len, h) = self
            .maps
            .iter()
            .find(|(b, l, _)| p >= *b && p + data.len() <= b + l)?;
        let off = (p - base) as u64;
        Some(
            h.clone()
                .offset_start(off)
                .offset_end(*len as u64 - off - data.len() as u64),
        )
    }

    /// WHERE a borrowed slice lives: `(mapping index, byte offset)`.
    ///
    /// The same pointer-containment lookup [`Aliases::slice`] does, stopping
    /// one step short of building a `Handle`. The grouped routed-expert lane
    /// ([`super::moegroup`]) needs the offsets and not the handles: it binds
    /// the mapping ONCE for the whole layer and lets the kernel pick an
    /// expert's planes out of it, so a per-expert `Handle` would be twenty-odd
    /// clones of the same pointer for the privilege of throwing them away.
    ///
    /// `None` for the same two reasons `slice` returns `None` — unaligned, or
    /// in no registered mapping — and deliberately WITHOUT counting a bind,
    /// because the caller is still deciding whether it can take this lane at
    /// all. It counts with [`Aliases::note_alias`] once it has committed.
    pub fn locate(&self, data: &[u8]) -> Option<(usize, u64)> {
        if !matches!(self.classify(data), Bind::Alias) {
            return None;
        }
        let p = data.as_ptr() as usize;
        let (i, (base, _, _)) = self
            .maps
            .iter()
            .enumerate()
            .find(|(_, (b, l, _))| p >= *b && p + data.len() <= b + l)?;
        Some((i, (p - base) as u64))
    }

    /// The registered handle for a whole mapping, and its length in bytes.
    ///
    /// Not a slice of it: this is the buffer the grouped GEMM binds, with the
    /// per-expert offsets travelling separately as device data.
    pub fn map(&self, i: usize) -> Option<(Handle, usize)> {
        self.maps.get(i).map(|(_, l, h)| (h.clone(), *l))
    }

    /// Charge `bytes` to the alias counters for a bind that went through
    /// [`Aliases::locate`] rather than [`Aliases::slice_or_copy`].
    ///
    /// The seam moved but the accounting must not: the report's "100% of binds
    /// aliased" line is only worth reading if every weight the device sees is
    /// still counted somewhere, and a lane that quietly stopped reporting would
    /// look like a lane that stopped moving bytes.
    pub fn note_alias(&self, bytes: usize) {
        use core::sync::atomic::Ordering::Relaxed;
        self.stats.alias_calls.fetch_add(1, Relaxed);
        self.stats.alias_bytes.fetch_add(bytes as u64, Relaxed);
    }

    /// [`Aliases::slice`], falling back to an ordinary copy — and COUNTING
    /// which of the two happened.
    ///
    /// The counting is here rather than at the call sites because this is the
    /// seam the question is about: every weight the expert lane hands the GPU
    /// passes through exactly this function, so a total taken here is a total
    /// over the whole lane by construction and cannot miss a path someone
    /// added later.
    pub fn slice_or_copy<R: Runtime>(&self, client: &ComputeClient<R>, data: &[u8]) -> Handle {
        use core::sync::atomic::Ordering::Relaxed;
        let kind = self.classify(data);
        match kind {
            Bind::Alias => {
                self.stats.alias_calls.fetch_add(1, Relaxed);
                self.stats.alias_bytes.fetch_add(data.len() as u64, Relaxed);
                self.slice(data).expect("classified as aliasable")
            }
            _ => {
                match kind {
                    Bind::CopyUnaligned(r) => {
                        self.stats.unaligned[r].fetch_add(1, Relaxed);
                    }
                    Bind::CopyUnmapped => {
                        self.stats.unmapped.fetch_add(1, Relaxed);
                    }
                    _ => {}
                }
                let t = std::time::Instant::now();
                let h = client.create_from_slice(data);
                self.stats.copy_calls.fetch_add(1, Relaxed);
                self.stats.copy_bytes.fetch_add(data.len() as u64, Relaxed);
                self.stats
                    .copy_nanos
                    .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                h
            }
        }
    }
}

#[cfg(test)]
mod bind_tests {
    use super::*;

    /// The predicate the runtime binds on, checked without a GPU.
    ///
    /// `classify` is the only place the 4-byte rule lives now, so this covers
    /// `slice`, `slice_or_copy` and the offline audit at once.
    #[test]
    fn classify_names_the_reason_not_just_the_verdict() {
        let al = Aliases::disabled();
        // Nothing is registered, so an aligned slice is UNMAPPED, not aliasable
        // — and saying so is the point: it is a different repair.
        let v = vec![0u8; 64];
        let base = v.as_ptr() as usize;
        let pad = (4 - base % 4) % 4;
        assert_eq!(al.classify(&v[pad..pad + 16]), Bind::CopyUnmapped);
        // and a deliberately odd offset reports its residue
        assert_eq!(al.classify(&v[pad + 1..pad + 17]), Bind::CopyUnaligned(1));
        assert_eq!(al.classify(&v[pad + 2..pad + 18]), Bind::CopyUnaligned(2));
        assert_eq!(al.classify(&[]), Bind::Empty);
    }

    /// A rate over zero calls is neither 0% nor 100%, and reporting either
    /// would be a green check over an empty measurement.
    #[test]
    fn an_empty_run_has_no_alias_rate() {
        assert_eq!(BindStats::default().alias_fraction(), None);
        let s = BindStats {
            alias_calls: 3,
            copy_calls: 1,
            ..Default::default()
        };
        assert_eq!(s.alias_fraction(), Some(0.75));
    }
}

// ---------------------------------------------------------------------------
// Pre-permuted ("swizzled") B layout
// ---------------------------------------------------------------------------

/// Where lane `l`'s `i`-th 32-bit B load lands in a swizzled block — and the
/// reason this file's whole staging apparatus may be unnecessary.
///
/// ## The layout, derived rather than drawn
///
/// [`fp4_linear`] gets its B addresses from
/// `def.position_of_nth(lane, i * vs_b * pack, MatrixIdent::B)`, which on
/// sm_121a for `m16n8k64` returns, for `i` in `0..2`:
///
/// ```text
///   col = lane >> 2                     the n column, 0..8
///   row = (lane & 3) * 8 + i * 32       the k element, 0..64
/// ```
///
/// `fp4_lane_map` dumps that table off the device so it is a measurement and
/// not a diagram. Feed it through the kernel's own index arithmetic
/// (`byte = row_n * k/2 + k_elem / 2`) and one MMA's B operand is:
///
/// * eight weight rows, `n_base .. n_base + 8`;
/// * 32 contiguous bytes of each — the k tile — so 256 bytes in all;
/// * the rows `k / 2` bytes apart, which at `k = 4096` is 2048.
///
/// Within one row's 32 bytes, lane `4c + s` takes word `s` on load 0 and word
/// `s + 4` on load 1. So load `i` across the warp touches 32 four-byte pieces
/// scattered over EIGHT 128-byte lines, using 16 bytes of each. That is the 2x:
/// eight sector requests where a coalesced warp issues four.
///
/// The permutation that fixes it is therefore forced, not chosen — write the
/// bytes down in the order the loads want them:
///
/// ```text
///   dst_word(c, w) = (w / 4) * 32 + c * 4 + (w % 4)
/// ```
///
/// for weight row `c` of the tile (`0..8`) and 32-bit word `w` of that row's k
/// tile (`0..8`). Substituting `c = lane >> 2`, `w = (lane & 3) + 4 * i` gives
/// `dst_word = 32 * i + lane` exactly: load `i` is 128 CONTIGUOUS bytes, in
/// lane order, which is the fully-coalesced case. No staging, no shared
/// memory, no barrier — the only cost is that the bytes were written down
/// differently, once, and the weights are static.
///
/// The block is 256 bytes and blocks run `(n_tile, k_tile)` row-major, so a
/// k loop still walks forward through memory and consecutive n tiles stay
/// `k / 2` bytes apart, i.e. the outer locality is unchanged.
#[inline]
fn swz_word(c: usize, w: usize) -> usize {
    // 4 = words a row contributes to ONE load (8 words / 2 loads); 32 = the
    // words one load takes across the warp (NTILE rows x 4).
    (w / 4) * 32 + c * 4 + (w % 4)
}

/// Permute `[n, k/2]` packed E2M1 codes into MMA-fragment order.
///
/// Output is the same length and the same bytes; only their order changes.
/// Block `(n_tile, k_tile)` occupies 256 consecutive bytes at
/// `((n_tile * k/64) + k_tile) * 256`, and inside it word `32 * i + lane` is
/// exactly what lane `lane`'s load `i` reads. See [`swz_word`].
pub fn swizzle_b_codes(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    swizzle_b_codes_into(src, &mut dst, n, k);
    dst
}

/// [`swizzle_b_codes`] writing into a caller-owned destination.
///
/// The form the LOAD PATH uses. `PileSource::copy_share` already memcpys every
/// expert plane out of the pile mapping into the anonymous arena, so permuting
/// there is a change of destination index inside a copy that already happens —
/// the alternative, `swizzle_b_codes` followed by `copy_from_slice`, would
/// allocate and touch the bytes a second time for nothing. `src` and `dst` must
/// not overlap.
pub fn swizzle_b_codes_into(src: &[u8], dst: &mut [u8], n: usize, k: usize) {
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");
    assert_eq!(src.len(), n * k / 2, "codes are not [n, k/2]");
    assert_eq!(
        dst.len(),
        src.len(),
        "destination is not the source's length"
    );
    let kt = k / KTILE;
    let row_w = k / 8; // 32-bit words in one weight row
    for nt in 0..n / NTILE {
        for t in 0..kt {
            let blk = (nt * kt + t) * 256;
            for c in 0..NTILE {
                for w in 0..8 {
                    let s = ((nt * NTILE + c) * row_w + t * 8 + w) * 4;
                    let d = blk + swz_word(c, w) * 4;
                    dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
                }
            }
        }
    }
}

/// Permute `[n, k/16]` E4M3 block scales to match [`swizzle_b_codes`].
///
/// One MMA consumes four scale bytes per weight row — `scales_count() = 4`,
/// one per 16 k elements of the 64 the instruction covers — so a fragment's
/// scales are 8 rows x 4 bytes = 32 bytes, and unpermuted those are eight
/// separate sectors for 32 useful bytes: the same defect as the codes, at an
/// eighth the volume and the SAME sector count. Blocked as `[n_tile][k_tile][8
/// rows][4]` they are 32 contiguous bytes, i.e. one sector for the whole warp.
///
/// The scale grouping is unchanged — still one E4M3 per 16 elements along k,
/// still the same byte for the same 16 elements. Only where it is written
/// moves, which is why the output is bit-identical rather than close.
pub fn swizzle_b_scales(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    swizzle_b_scales_into(src, &mut dst, n, k);
    dst
}

/// [`swizzle_b_scales`] writing into a caller-owned destination; see
/// [`swizzle_b_codes_into`] for why the load path wants this form.
pub fn swizzle_b_scales_into(src: &[u8], dst: &mut [u8], n: usize, k: usize) {
    assert_eq!(n % NTILE, 0);
    assert_eq!(k % KTILE, 0);
    assert_eq!(src.len(), n * (k / GROUP), "scales are not [n, k/16]");
    assert_eq!(
        dst.len(),
        src.len(),
        "destination is not the source's length"
    );
    let kt = k / KTILE;
    let spr = k / GROUP;
    // E4M3 scales one MMA consumes per weight row: `scales_count()`, which for
    // this instruction is `KTILE / GROUP`.
    let per = KTILE / GROUP;
    for nt in 0..n / NTILE {
        for t in 0..kt {
            let blk = (nt * kt + t) * NTILE * per;
            for c in 0..NTILE {
                let s = (nt * NTILE + c) * spr + t * per;
                dst[blk + c * per..blk + c * per + per].copy_from_slice(&src[s..s + per]);
            }
        }
    }
}

/// The inverse of [`swizzle_b_codes_into`]: fragment order back to row-major
/// `[n, k/2]`. The learner writes the arena in fragment order; a checkpoint
/// wants the row-major plane the quantiser produced, and this is the only
/// way the learned codes get back into one.
pub fn unswizzle_b_codes_into(src: &[u8], dst: &mut [u8], n: usize, k: usize) {
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");
    assert_eq!(src.len(), n * k / 2, "codes are not [n, k/2]");
    assert_eq!(dst.len(), src.len(), "destination is not the source's length");
    let kt = k / KTILE;
    let row_w = k / 8;
    for nt in 0..n / NTILE {
        for t in 0..kt {
            let blk = (nt * kt + t) * 256;
            for c in 0..NTILE {
                for w in 0..8 {
                    let d = ((nt * NTILE + c) * row_w + t * 8 + w) * 4;
                    let s = blk + swz_word(c, w) * 4;
                    dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
                }
            }
        }
    }
}

/// The inverse of [`swizzle_b_scales_into`].
pub fn unswizzle_b_scales_into(src: &[u8], dst: &mut [u8], n: usize, k: usize) {
    assert_eq!(n % NTILE, 0);
    assert_eq!(k % KTILE, 0);
    assert_eq!(src.len(), n * (k / GROUP), "scales are not [n, k/16]");
    assert_eq!(dst.len(), src.len(), "destination is not the source's length");
    let kt = k / KTILE;
    let spr = k / GROUP;
    let per = KTILE / GROUP;
    for nt in 0..n / NTILE {
        for t in 0..kt {
            let blk = (nt * kt + t) * NTILE * per;
            for c in 0..NTILE {
                let d = (nt * NTILE + c) * spr + t * per;
                dst[d..d + per].copy_from_slice(&src[blk + c * per..blk + c * per + per]);
            }
        }
    }
}

#[cfg(test)]
mod swizzle_tests {
    /// Swizzle then unswizzle is the identity on both planes, so a learned
    /// arena plane goes back into a checkpoint as exactly its bytes.
    #[test]
    fn unswizzle_inverts_swizzle_on_both_planes() {
        let (n, k) = (64, 256);
        let codes: Vec<u8> = (0..n * k / 2).map(|i| (i * 7 + 3) as u8).collect();
        let scales: Vec<u8> = (0..n * k / 16).map(|i| (i * 13 + 5) as u8).collect();
        let sw = swizzle_b_codes(&codes, n, k);
        let mut back = vec![0u8; sw.len()];
        unswizzle_b_codes_into(&sw, &mut back, n, k);
        assert_eq!(back, codes);
        let sw = swizzle_b_scales(&scales, n, k);
        let mut back = vec![0u8; sw.len()];
        unswizzle_b_scales_into(&sw, &mut back, n, k);
        assert_eq!(back, scales);
    }

    use super::*;

    /// Every source byte lands somewhere, exactly once.
    ///
    /// The permutation's whole claim is that it MOVES bytes and drops none, so
    /// the failure it has to exclude is a destination formula that collides —
    /// which loses one byte and duplicates another, and which no bit-identity
    /// check against a same-shaped output would call an error. Tested on a
    /// non-square shape so a transposed `n`/`k` cannot pass.
    #[test]
    fn the_permutation_is_a_bijection_on_both_planes() {
        let (n, k) = (16usize, 128usize);
        let codes: Vec<u8> = (0..n * k / 2).map(|i| (i % 251) as u8).collect();
        let mut seen = vec![false; codes.len()];
        // Re-derive the destination of every source byte and mark it.
        let kt = k / KTILE;
        for nt in 0..n / NTILE {
            for t in 0..kt {
                for c in 0..NTILE {
                    for w in 0..8 {
                        let d = (nt * kt + t) * 256 + swz_word(c, w) * 4;
                        for b in 0..4 {
                            assert!(!seen[d + b], "destination {} written twice", d + b);
                            seen[d + b] = true;
                        }
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|v| *v),
            "some destination was never written"
        );
        // and the same byte multiset comes out
        let mut a = swizzle_b_codes(&codes, n, k);
        let mut b = codes.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);

        let scales: Vec<u8> = (0..n * (k / GROUP)).map(|i| (i % 241) as u8).collect();
        let mut a = swizzle_b_scales(&scales, n, k);
        let mut b = scales.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    /// The load path's form agrees with the allocating one, byte for byte.
    ///
    /// `PileSource::copy_share` uses `*_into` against a slice of the arena, so
    /// the two must not be able to drift; this is the only place they meet.
    #[test]
    fn the_in_place_form_matches_the_allocating_one() {
        let (n, k) = (24usize, 192usize);
        let codes: Vec<u8> = (0..n * k / 2).map(|i| (i % 253) as u8).collect();
        let scales: Vec<u8> = (0..n * (k / GROUP)).map(|i| (i % 239) as u8).collect();
        let mut dc = vec![0u8; codes.len()];
        let mut ds = vec![0u8; scales.len()];
        swizzle_b_codes_into(&codes, &mut dc, n, k);
        swizzle_b_scales_into(&scales, &mut ds, n, k);
        assert_eq!(dc, swizzle_b_codes(&codes, n, k));
        assert_eq!(ds, swizzle_b_scales(&scales, n, k));
    }

    /// The shapes the routed lane actually runs, and the ones it cannot.
    #[test]
    fn swizzleable_answers_for_the_shapes_that_tile() {
        assert!(swizzleable(4096, 4096));
        assert!(!swizzleable(4, 4096), "4 rows is half an n tile");
        assert!(!swizzleable(4096, 32), "32 k elements is half a k tile");
    }
}

/// Whether an `[n, k]` NVFP4 weight can be written down in fragment order.
///
/// The permutation is a re-indexing of whole `(n_tile, k_tile)` blocks, so it
/// exists exactly when the matrix tiles: `NTILE` rows and `KTILE` k elements.
/// Every shape the grouped lane accepts already satisfies this — the launcher
/// asserts it — but the LOAD path runs before any launcher and has to answer
/// for weights the run may never multiply, so it asks rather than assumes.
pub fn swizzleable(n: usize, k: usize) -> bool {
    n % NTILE == 0 && k % KTILE == 0
}

/// [`fp4_linear`] reading a B operand that is ALREADY in fragment order.
///
/// Line for line [`fp4_linear`] with one expression changed — the B index — so
/// the k order, the operands, the accumulator and the output store are
/// identical and so is the output, bit for bit, given
/// [`swizzle_b_codes`]/[`swizzle_b_scales`] of the same weights.
///
/// The point of contrast is [`fp4_linear_smem`], which recovers the same
/// bandwidth by STAGING the scattered read through shared memory at runtime.
/// This kernel needs no shared memory, no `sync_plane`, no row padding to dodge
/// bank conflicts and no chunk-size knob — the load is coalesced by
/// construction, so nothing about it depends on how many planes are resident.
/// That last part is not a bandwidth claim and is worth checking separately:
/// the staged arm wants `INK_MOE_PLANES=1` at decode and 4 at prefill, and a
/// kernel with no smem has no such preference to have.
///
/// `swz_sc` says whether the scale plane was permuted too. Separate knob for
/// the same reason [`fp4_linear_smem`]'s `stage_sc` is separate: they are two
/// defects, and leaving it off measures the code permutation against an
/// otherwise unchanged baseline. At the head shape it is worth ~2%, so the
/// codes are nearly all of it.
///
/// ## Measured, against the staged arm, in one process
///
/// `inkling_membw INK_BW_AXES=1`, DGX Spark GB10 / sm_121a, one 1.208 GB handle
/// (`n = 524224`, `k = 4096`, codes + scales), `m_pad = 16`, the three arms
/// INTERLEAVED over 9 rounds and min of each — interleaved because two runs of
/// the block-measured table minutes apart put the unstaged arm at 102.6 and
/// 94.5 GB/s with the coalesced control at 247.7 and 237.4, i.e. the part
/// drifts and whatever is measured last is handicapped. GB/s is the B-operand
/// bytes over the whole kernel time, which includes the `[m_pad, n]` f32 output
/// allocation and store; it is a per-KERNEL-LAUNCH figure, not per token or per
/// layer.
///
/// ```text
///   A  unstaged  fp4_linear            96.9 GB/s   12.461 ms   1.00x
///   B  STAGED    fp4_linear_smem      193.2 GB/s    6.253 ms   1.99x
///   C  PRE-PERM  fp4_linear_swz       193.6 GB/s    6.239 ms   2.00x   C/B 1.002x
/// ```
///
/// So on the DENSE head the two are a tie, and the premise that a pre-permuted
/// layout should reach the coalesced ceiling where staging reaches 87% of it is
/// **refuted**: the coalesced control in that same run reads 222.5 GB/s and BOTH
/// arms land at 87% of it. Whatever the remaining 13% is, it is not the B access
/// pattern — a one-plane 32-thread cube, the MMA's own dependency chain and the
/// output store are all still there in both.
///
/// The grouped lane is where they separate, because there the staged arm has a
/// schedule to get right and this one does not; see
/// [`super::moegroup::fp4_linear_grouped`]'s `swz` flag.
///
/// ## Wired, 2026-08-25 — and what that bought END TO END
///
/// The permutation is applied by [`super::pile::PileSource::copy_share`] (see
/// [`swizzle_b_codes_into`]) and every routed-lane consumer reads it; this
/// kernel is the per-expert FALLBACK lane's reader. So the 2x above is no
/// longer a benchmark's property. What it is worth to a decode STEP, which
/// nobody had measured, is a different question and the answer is: **at the
/// single-row decode this repo benches, nothing.**
///
/// `scripts/bench-decode.sh -n 3 --layers 0:16 --gen 12`, DGX Spark GB10 /
/// sm_121a, three arms in ONE worktree (`mary-swzload`) differing only by
/// environment, interleaved, first two passes discarded, median over reps.
/// PER DECODE STEP of a 16-layer range, ONE row wide:
///
/// ```text
///   prompt          rowmajor   PRE-PERM   staged(smem,1 plane)   spread
///   p256 (268 ctx)    48.2       47.8            47.4           0.6-1.0%
///   cover (3744 ctx)  47.7       48.3            48.0           0.6-18%
/// ```
///
/// +0.84% and -1.24%: both under the spread, i.e. not results. The reason is
/// in the same runs' own breakdown and it is not a defect in the kernel — at
/// this shape **the pass is HOST-ENQUEUE-BOUND**: 34.3 ms in the layer loop
/// plus 4.7 ms after the sync against **3.9 ms blocked on the device**, of a
/// 46 ms pass.
///
/// State that as a bound, because it is the useful form: **exposed device time
/// is 3.9 ms, so the headroom for ANY device-side change at this configuration
/// is at most 8% of the step**, whatever the change is and however large the
/// kernel win. The 2.7-4 ms a step this work was scoped against sits at the
/// very top of that bound — and it is not reachable here, because the exposed
/// 3.9 ms is the tail nothing can overlap (the head's own 1.53 GiB unembed
/// read runs last), not the routed lane, which is enqueued early and finishes
/// under the host.
///
/// ## Where it DOES show: the WIDE pass
///
/// Same worktree, same three arms, `INK_SLOTS=32` on the 3732-token file,
/// `INK_LAYERS=0:16`, `--gen 8`, gated (0% util, 208-227 MHz idle clocks, no
/// compute process at the pre-run gate). That run has two regimes in one
/// series and they must not be medianed together — 31 SLOT-PREFILL passes
/// (one 116-token chunk each) and then 6 BATCHED-DECODE passes (32 rows).
/// Per-rep medians, three reps:
///
/// ```text
///                        rowmajor            PRE-PERM            staged(1)
///   slot-prefill pass  282.4 281.9 280.4  275.7 273.0 272.8  291.4 289.3 292.9
///   batched-decode     152.4 156.3 153.7  153.3 155.1 156.2  154.3 160.1 154.3
/// ```
///
/// On the WIDE pass the arms do not overlap: **281.9 -> 273.0 ms, -3.2% a
/// pass**, against a within-arm spread of 0.7% and 1.1%, and staging is
/// consistently WORSE at +3.4% — which is the prefill column of this module's
/// probe table showing up end to end, where `staged` at one plane collapses.
/// On the batched-decode pass all three overlap, and the same breakdown says
/// why: `DEVICE, one sync` is **0.5-1.4 ms of a 153 ms pass** there, with 93.7
/// ms of the host's mlp half spent enqueueing.
///
/// So the honest end-to-end summary is one line: **the permutation is worth
/// ~3% on a pass wide enough to have device work exposed, and nothing on a
/// pass this runtime spends enqueueing.** Token output was identical in every
/// arm of both runs — 9 reps at 32 slots share one digest over every emitted
/// token of every slot, and every printed top-5 logit agrees to the printed
/// two decimals. That is an observation, not a gate; nothing in this lane
/// tests for it.
///
/// That does not refute the 2x; it locates it. The routed lane is not grid-
/// starved at decode either — one stage launches `n / NTILE = 512` cubes over
/// `blocks` = the active-expert count with `nrep = 1`, i.e. ~3072 working
/// warps, well past the ~1150-cube knee where this part's achieved rate is
/// still climbing. The regime the routed GEMM dominates is the BATCHED one
/// this module's header measures — `INK_SLOTS=32`, where `fp4_linear_grouped`
/// is 52.7% of GPU time and the head's GPU is busy 93% of the time it is not
/// blocked. A single-row decode of sixteen layers reads 1.31 GiB of expert
/// weight a pass, which is ~13 ms of device time hidden under 39 ms of host
/// enqueue.
///
/// ## What is NOT permuted, and why it is not an oversight
///
/// The head. `head_lane()` is `W4a16` with no switch, so the unembedding is
/// multiplied by [`super::w4a16gemm::w4a16_linear`] — `MTILE 16, NTILE 8,
/// KTILE 16`, `MmaDefinition::new`, i.e. **`m16n8k16` with a BF16 activation**,
/// against this file's `m16n8k64` `new_scaled`. [`swz_word`] was derived from
/// `position_of_nth(.., MatrixIdent::B)` for the second of those and describes
/// only it; the W4A16 B fragment is a different map and there is no
/// `w4a16_linear_swz` to read one. So permuting `quantized_bf16`'s output
/// would need its own derivation and its own device check against
/// `fp4_frag_b_map`'s W4A16 twin, which does not exist yet. That is a separate
/// piece of work, not a line of wiring — and it is where the remaining win is,
/// since the head's 25128-cube launch is the one shape already at both its
/// grid ceiling and its access-pattern ceiling.
///
/// The output is identical: 18 reps across three arms and two prompts emitted
/// the SAME 13 tokens each, and `gemm_grid_parity` reproduces pristine main's
/// digests in all 20 cells.
///
/// ## What binding the scale planes 4 wide was worth, measured
///
/// `ptx_fp4_probe`, 6 rounds per arm alternating with >=60 s between runs, one
/// otherwise-idle GB10 (sm_121a), the hand-written PTX arm as a fixed yardstick
/// because this commit does not touch it:
///
/// ```text
/// cubecl / PTX   BEFORE (1-wide)  1.159 1.162 1.173 1.180 1.182 1.191   median 1.1765
/// cubecl / PTX   AFTER  (4-wide)  1.124 1.129 1.131 1.136 1.143 1.156   median 1.1335
/// ```
///
/// FRAMING RULE: that ratio is PER LAUNCH of ONE [16, 4096] x [4096, 4096]^T
/// routed-expert product -- the decode case, one 16-row tile -- p50 of 9 warm
/// rounds of 20 pipelined launches over 30 rotating tables, arms interleaved
/// and reversed on odd rounds. It is NOT a step figure and NOT a two-node
/// figure, and it is a RATIO of two arms inside one process, which is the
/// 0.23%-sd measurement rather than the 1.2-1.4% between-process one.
///
/// The arms separate COMPLETELY: the slowest BEFORE round (1.159) is still
/// faster-ratioed than the fastest AFTER round (1.156). Exact permutation test
/// on the difference of medians, p = 0.0065 one-tailed. The gap to hand-written
/// PTX goes 17.65% -> 13.35%, i.e. 24.4% of it closed, for a 1.038x cubecl arm.
///
/// Output was bit-identical in all 12 runs (0 of 65536 elements differ, max abs
/// diff 0e0), which is the predicted result rather than a lucky one: this
/// changes the WIDTH OF A LOAD and nothing about the arithmetic.
///
/// The remaining 13.35% is the OTHER cause SASS named -- the one that is a
/// redesign and not a binding -- and is not addressed here.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear_swz<AB: Scalar, S: Scalar, NA: Size, NC: Size, NS: Size>(
    a: &Array<Vector<AB, NA>>,
    a_sc: &Array<Vector<S, NS>>,
    b: &Array<Vector<AB, NA>>,
    b_sc: &Array<Vector<S, NS>>,
    out: &mut Array<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] swz_sc: bool,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize;
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;

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

    // The instruction wants FOUR E4M3 block scales per operand per k tile, and
    // they sit at four CONSECUTIVE addresses. Read as four `Array<S>` elements
    // that is four one-byte loads; read as one `Vector<S, 4>` it is one 32-bit
    // load, and 8 of the 14 loads this kernel issued per `mma` were those bytes
    // (measured on a GB10: 8x `LDG.E.U8.CONSTANT` + 6x `LDG.E.CONSTANT`, against
    // the hand-PTX arm's 8x `LDG.E.CONSTANT` for exactly the same operands).
    // `scales_vector_size` is `register_size_bits / 8` = 4 here, which is the
    // same 4 as `scales_count`, so the vector the instruction takes IS the
    // vector the memory holds -- nothing is padded and nothing is assembled.
    // This is the LOAD width only; the emitted `mma` string is unchanged, which
    // is why the result stays bit-identical rather than merely close.
    //
    // Alignment: the row-major group of four starts at
    // `index * spr + t * SCALE_VEC`, and `spr = k / GROUP` is a multiple of
    // four for every `k % KTILE == 0` this kernel accepts, so a group never
    // straddles a vector. The swizzled group starts at
    // `(...) * NTILE * SCALE_VEC + sib * SCALE_VEC`, a multiple of four by
    // construction. `ptxgemm`'s emitter test already asserts `% 4 == 0` on
    // both of these addresses, for four shapes, both scale layouts, nine tile
    // positions and all 32 lanes.
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
        // The B block for this `(n_tile, t)` is 256 contiguous bytes and word
        // `32 * i + lane` of it is this lane's `i`-th load, so the warp's load
        // is 128 consecutive bytes in lane order. Written out of
        // `position_of_nth` rather than assumed, so it tracks the target's own
        // fragment layout: `row` is the k element, `col` the n column, and
        // `swz_word`'s `(c, w)` are `col` and `row / 8`.
        #[unroll]
        for i in 0..vc_b {
            let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
            let w = row as usize / 8;
            let blk = (n_tile * k_tiles + t) * 256;
            let off = (w / 4) * 32 + col as usize * 4 + (w % 4);
            reg_b[i] = b[(blk + off * 4) / b.vector_size()];
        }

        // One 32-bit load each, then into a MUTABLE local: the MMA intrinsic
        // takes its scale registers by non-const reference, so a value that
        // came straight out of a load and is never written cannot be handed to
        // it -- NVRTC rejects the generated cast. The moves below are register
        // traffic, not memory.
        let sbyte = if comptime![swz_sc] {
            ((n_tile * k_tiles + t) * NTILE + sib) * SCALE_VEC
        } else {
            (sib + n_base) * spr + t * SCALE_VEC
        };
        let va = a_sc[((sia + m_base) * spr + t * SCALE_VEC) / a_sc.vector_size()];
        let vb = b_sc[sbyte / b_sc.vector_size()];
        let mut sa = Vector::<S, NS>::empty();
        let mut sb = Vector::<S, NS>::empty();
        #[unroll]
        for i in 0..SCALE_VEC {
            sa[i] = va[i];
            sb[i] = vb[i];
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
        out[(gr * size_n + gc) / out.vector_size()] = acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// Launch [`fp4_linear_swz`]; [`fp4_linear_launch`]'s arguments, with `b` (and
/// `b_sc`, if `swz_sc`) already permuted.
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear_swz_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    a_sc: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    scale: f32,
    swz_sc: bool,
) -> Handle {
    assert_eq!(m_pad % MTILE, 0);
    assert_eq!(n % NTILE, 0);
    assert_eq!(k % KTILE, 0);
    assert!(n / NTILE <= 65535);

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let vs = 32 / e2m1x2::cube_type().size_bits();
    let spr = k / GROUP;
    // The scale planes are bound `SCALE_VEC` wide, so a row has to be a whole
    // number of vectors or a lane's group of four would straddle one. Implied
    // by `k % KTILE == 0` above -- stated anyway, because it is the precondition
    // of the BINDING and not of the tiling, and the two could drift apart.
    assert_eq!(
        spr % SCALE_VEC,
        0,
        "the scale row {spr} is not a whole number of {SCALE_VEC}-wide vectors"
    );

    unsafe {
        fp4_linear_swz::launch::<e2m1x2, e4m3, R>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            SCALE_VEC,
            ArrayArg::from_raw_parts(a.clone(), m_pad * (k / 2)),
            ArrayArg::from_raw_parts(a_sc.clone(), m_pad * spr),
            ArrayArg::from_raw_parts(b.clone(), n * (k / 2)),
            ArrayArg::from_raw_parts(b_sc.clone(), n * spr),
            ArrayArg::from_raw_parts(out.clone(), m_pad * n),
            k,
            n,
            swz_sc,
            scale,
        )
    };
    out
}

/// Dump the MMA fragment map off the device, so the permutation above is a
/// measurement rather than a diagram.
///
/// One plane. Lane `l` writes `out[(l * 4 + i) * 2 + {0,1}] = {row, col}` of
/// `def.position_of_nth(l, i * vs * pack, MatrixIdent::B)` — i.e. the same call
/// [`fp4_linear`] uses to build its B address, asked for its answer instead of
/// for an address. `out` must hold `32 * 4 * 2` u32; entries past `vc_b` stay
/// at whatever the caller put there and should be ignored (`vc_b` is printed
/// alongside as `out[256]`).
#[cube(launch)]
pub fn fp4_frag_b_map<AB: Scalar, S: Scalar>(out: &mut Array<u32>) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    #[unroll]
    for i in 0..vc_b {
        let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
        out[(lane as usize * 4 + i) * 2] = row;
        out[(lane as usize * 4 + i) * 2 + 1] = col;
    }
    if lane == 0 {
        out[256] = vc_b as u32;
        out[257] = vs_b as u32;
        out[258] = ec_b as u32;
        out[259] = def.scales_count() as u32;
        out[260] = def.scales_index(lane, MatrixIdent::B);
    }
    // Every lane's scale row, so the `[n_tile][k_tile][8][4]` claim about the
    // scale plane is checkable too and not inferred from lane 0 alone.
    out[261 + lane as usize] = def.scales_index(lane, MatrixIdent::B);
}

/// Launch [`fp4_frag_b_map`] and return the raw `u32` dump.
pub fn fp4_frag_b_map_launch<R: Runtime>(client: &ComputeClient<R>) -> Handle {
    let out = client.empty(300 * 4);
    unsafe {
        fp4_frag_b_map::launch::<e2m1x2, e4m3, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            ArrayArg::from_raw_parts(out.clone(), 300),
        )
    };
    out
}
