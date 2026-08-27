//! What does the W4A16 head lane's ACCESS PATTERN cost, with the arithmetic removed?
//!
//! ## Why this exists
//!
//! `w4a16gemm`'s ceiling block establishes 242 GB/s as what this part will move
//! over the head's 0.431 GiB weight table, prices the shipped swizzled depth-4
//! lane at 66% of it, and says a third of the traffic is unclaimed. That control
//! (`w4a16_swz_probe`'s `stream ceiling` / `fp4_lane_dump`'s `stream_packed`) is
//! a PURE COALESCED STREAM: 256-thread blocks, 128-bit loads, no arithmetic, no
//! MMA, and enough warps per block to fill every warp slot on the SM. The GEMM
//! lane is none of those things. So "66% of 242" answers "how much DRAM headroom
//! is there on this part", which is a real question, and does NOT answer "how
//! much faster could THIS kernel be", which is the question anyone reading the
//! 66% asks next.
//!
//! This harness puts the missing rungs on the ladder. Every arm reads the SAME
//! two planes out of the SAME buffers in one process, arms round-robined, and
//! every arm's DRAM traffic is byte-identical (`lts__d_sectors_fill_sysmem.sum`
//! within 0.1% across all of them) except the codes-only arm, which is charged
//! only for the codes:
//!
//! ```text
//!   coalesced      256-thread blocks, 128-bit loads, no arithmetic  <- the 242 control
//!   geom width     the lane's 25128 one-warp streams, flat + wide   <- what per-warp streaming costs
//!   geom, no A ..  the lane's LOAD STREAM, fragment-shaped, no math <- what the FRAGMENT MAP costs
//!   geom p1 / pN   the same, 1 vs 8 warps per cube                  <- what OCCUPANCY buys
//!   real           the shipped w4a16_linear_swz, depth 4, mask on   <- what the MATH costs
//! ```
//!
//! FRAMING RULE, and it is the ceiling block's own so the numbers compose: GB/s
//! over the weight bytes THAT ARM ACTUALLY READS (0.431 GiB of codes + E4M3
//! scales for every arm except the codes-only one, which is 0.383 GiB) of ONE
//! `[201024, 4096]` head operand, PER LAUNCH, one launch and one sync each,
//! `m_pad = 16` with one live row, GB10 (sm_121a), `scripts/gb10-lock.sh` held
//! and the box verified idle. Not a step figure and not a two-node figure.
//!
//! ## What it measured (spark2, 2026-08-27, box locked and idle)
//!
//! Min of 10 warm reps of 12, median of three back-to-back processes; the p50
//! column of the same runs tells the same story 2-4% lower with the coalesced
//! arm drifting, which is why the min is quoted. `sect/req` is
//! `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` and
//! `L2 req` is `lts__t_requests.sum`, both from a separate `ncu` pass on the
//! same binary:
//!
//! ```text
//!   arm                         ms     GB/s   % ceil   sect/req    L2 req
//!   coalesced (the 242)      1.831    252.9   100.0          16    3.624M
//!   geom width 16 B/lane     1.853    249.9    98.8          12    3.624M
//!   geom width 4 B/lane      2.021    229.2    90.6           3    4.524M
//!   geom, no A no scales     2.337    176.2    69.7           1    9.495M   (codes only)
//!   geom, no A               2.766    167.4    66.2           1   10.717M
//!   geom, 1 warp/cube        2.934    157.9    62.4           1   11.089M
//!   geom, 8 warps/cube       3.233    143.2    56.6           1   12.676M
//!   real w4a16_linear_swz    2.854    162.3    64.2           1   13.462M
//! ```
//!
//! Four back-to-back processes, and the two BANDS are what to quote rather than
//! any single row: the flat/wide arms land at 90-104% of the coalesced control
//! and the fragment-shaped ones at 56-68%, and the bands do not overlap in any
//! process. Individual width rows move 5-10% between processes because they are
//! close enough to the ceiling to feel the same drift the coalesced arm feels.
//!
//! THE ARITHMETIC IS FREE. The last two rows differ by the ENTIRE E2M1 ladder,
//! the scale multiply, the BF16 cast and the `m16n8k16` — 525 million
//! instructions a launch, 60% of the whole stream, `smsp__inst_executed.sum`
//! 870.6M against 345.8M. On mins the real lane is 2.7% FASTER than the control
//! that does none of it; on p50 it is 0.4% slower. Either way it is inside the
//! ~1.1% paired resolution of this harness. Anything that makes this lane's
//! dequantise cheaper — a branchless E2M1, a narrower cast, fewer ALU ops — is
//! buying something the memory shadow was already paying for.
//!
//! OCCUPANCY IS WORTH LESS THAN NOTHING. `geom, 8 warps/cube` lifts achieved
//! occupancy from 49.3% to 91.0% (one warp per cube against this part's
//! 24-block-per-SM cap is 24 of 48 warp slots, structurally, whatever the
//! register file does) at BYTE-IDENTICAL L1 requests, and it is 10% SLOWER. The
//! request path is already saturated at 24 warps an SM; adding warps adds
//! contention, not throughput.
//!
//! WHERE THE THIRD ACTUALLY GOES: TRANSACTIONS, NOT BYTES. Every arm above
//! moves the same 463 MB from DRAM. What differs is how many transactions carry
//! them, and the m16n8k16 B fragment map is what multiplies those: it puts FOUR
//! LANES ON THE SAME 32-BIT WORD, so a warp-wide load instruction fetches
//! exactly one 32-byte sector — `sect/req` is 1 for every fragment-shaped arm
//! and 16 for the coalesced control. The same 25128 per-warp contiguous streams,
//! read flat instead of through the fragment map, reach 90-99% of the ceiling.
//! So the geometry is not the problem, the stream count is not the problem, the
//! bytes are not the problem and the arithmetic is not the problem: the lane
//! spends 11.1M L2 requests where the control spends 3.6M for the same bytes.
//!
//! WHAT THIS DOES NOT SAY. `geom width` is not a proposed kernel. It reads the
//! right bytes in the right order and hands them to the WRONG LANES — putting
//! them where `m16n8k16` needs them costs shuffles or a shared-memory stage that
//! nothing here measures. Read the width rows as an upper bound on what a
//! cooperative stage could reach for the LOAD half, which is the direction
//! `w4a16gemm`'s own header pointed at ("coalescing needs a cooperative stage
//! through shared memory, which is exactly what one warp per output tile
//! forecloses") and which this prices for the first time.
//!
//! Timing only, deliberately: every arm reads the same bytes in the same order
//! as the lane it stands in for, so an uninitialised table exercises the
//! identical access pattern. Numerics are `w4a16_swz_probe`'s job.
//!
//! `INK_GC_N` / `INK_GC_K` set the shape (default the head's `[201024, 4096]`),
//! `INK_GC_M` the padded rows (default 16), `INK_GC_MLIVE` the live ones
//! (default 1), `INK_GC_REPS` the rep count (default 8, first two discarded),
//! `INK_GC_PLANES` the wide arm's warps per cube (default 8).

use std::time::Instant;

use cubecl::e4m3;
use cubecl::future;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::bf16;
use mary::models::inkling::w4a16gemm::{
    CODES_PER_WORD, GROUP, KTILE, MTILE, NTILE, SWZ_BLOCK_CODES, w4a16_linear_swz_launch,
};

type Rt = cubecl::cuda::CudaRuntime;

/// 128-bit loads per thread in the coalesced control, matching `stream_planes`.
const PER: usize = 8;
const BLOCK: u32 = 256;

/// The coalesced control: the same two planes, 128-bit loads, no arithmetic.
///
/// Byte-for-byte the kernel behind the recorded 242 GB/s — copied rather than
/// imported so this harness's four arms are read out of one file.
#[cube(launch)]
pub fn stream_planes<NW: Size>(
    w: &Tensor<Vector<u32, NW>>,
    sc: &Tensor<Vector<u32, NW>>,
    out: &mut Tensor<u32>,
    #[comptime] threads: usize,
    #[comptime] per: usize,
) {
    let t = ABSOLUTE_POS as usize;
    let mut acc = u32::new(0i64);
    #[unroll]
    for i in 0..per {
        let v = w[t + i * threads];
        acc += v[0];
    }
    let s = sc[t % sc.len()];
    acc += s[0];
    if acc == u32::new(0x5AFE_5AFEi64) {
        out[t % out.len()] = acc;
    }
}

/// `w4a16_linear_swz`'s LOAD half, with the dequantise and the MMA removed.
///
/// Every global read is the one the real lane issues, at the same address, in
/// the same order, under the same `kunroll` grouping: the masked A fragment
/// rows, the swizzled B block, and the one E4M3 scale per k-tile. What is gone
/// is the E2M1 ladder, the scale multiply, the BF16 cast and the `m16n8k16`.
/// Each loaded value is added into one accumulator so that nothing can be
/// dead-code-eliminated, and the store is predicated on a value the accumulator
/// cannot take so no write traffic is charged to a read figure.
///
/// `planes` is warps per cube. At 1 this is the shipped geometry; above 1 the
/// cube covers `planes` n-tiles and the 24-block-per-SM cap stops being the
/// occupancy limit.
///
/// `hi_dead` is hard-coded rather than passed: this control is only run at
/// `m_live <= MTILE / 2`, which the launch asserts, and that is the case where
/// the odd fragment rows are a comptime zero.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn swz_geom_stream<AB: Scalar + Cast, S: Scalar, NA: Size>(
    a: &Tensor<Vector<AB, NA>>,
    b: &Tensor<u32>,
    b_sc: &Tensor<S>,
    out: &mut Tensor<u32>,
    #[comptime] size_k: usize,
    #[comptime] kunroll: usize,
    #[comptime] planes: usize,
    #[comptime] read_a: bool,
    #[comptime] read_sc: bool,
    m_live: u32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize * comptime!(planes) + UNIT_POS_Y as usize;
    let m_base = m_tile * MTILE;

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);

    let k_tiles = comptime!(size_k / KTILE);
    let wpb = comptime!(SWZ_BLOCK_CODES / 4);
    let groups = comptime!(k_tiles / kunroll);

    let mut w_buf = Array::<u32>::new(comptime!(kunroll * vc_b));
    let mut s_buf = Array::<f32>::new(kunroll);
    let mut a_buf = Array::<Vector<AB, NA>>::new(comptime!(kunroll * vc_a));

    let mut acc = u32::new(0i64);

    for g in 0..groups {
        #[unroll]
        for u in 0..kunroll {
            let t = g * kunroll + u;
            let kbase = t * KTILE;
            #[unroll]
            for i in 0..vc_a {
                let (row, col) =
                    def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
                let gr = row as usize + m_base;
                let gc = col as usize + kbase;
                if comptime!(!read_a || (i & 1) == 1) {
                    a_buf[u * vc_a + i] = Vector::<AB, NA>::cast_from(0.0f32);
                } else {
                    let mut v = Vector::<AB, NA>::cast_from(0.0f32);
                    if gr < m_live as usize {
                        v = a[(gr * size_k + gc) / a.vector_size()];
                    }
                    a_buf[u * vc_a + i] = v;
                }
            }
            #[unroll]
            for i in 0..vc_b {
                let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
                let w = row as usize / CODES_PER_WORD;
                let blk = (n_tile * k_tiles + t) * wpb;
                w_buf[u * vc_b + i] = b[blk + w * NTILE + col as usize];
            }
            let (_r0, c0) = def.position_of_nth(lane, 0u32, MatrixIdent::B);
            s_buf[u] = if comptime!(read_sc) {
                f32::cast_from(b_sc[(n_tile * k_tiles + t) * NTILE + c0 as usize])
            } else {
                f32::new(0.0f32)
            };
        }

        // The consume phase, reduced to one add per loaded value. This is the
        // whole difference from `w4a16_linear_swz`.
        #[unroll]
        for u in 0..kunroll {
            #[unroll]
            for i in 0..vc_b {
                acc += w_buf[u * vc_b + i];
            }
            acc += u32::cast_from(s_buf[u]);
            #[unroll]
            for i in 0..vc_a {
                acc += u32::cast_from(f32::cast_from(a_buf[u * vc_a + i][0]));
            }
        }
    }

    if acc == u32::new(0x5AFE_5AFEi64) {
        out[ABSOLUTE_POS as usize % out.len()] = acc;
    }
}

/// The same 25128 one-warp streams, with the REQUEST WIDTH as the only variable.
///
/// `swz_geom_stream` says what the lane's load stream costs; it cannot say
/// whether the cost is the number of concurrent streams or the 64-byte
/// granularity each one advances at, because it varies neither. This kernel
/// varies exactly one: warp `n_tile` still walks its own contiguous 16 KiB of
/// codes and 2 KiB of scales, in order, but each lane reads `NB` words at a
/// time, so one warp instruction covers `32 * NB * 4` bytes instead of 64.
///
/// A is dropped, deliberately. It is L2-resident at `m_pad = 16` and accounts
/// for 6% of this lane's L2 sectors and none of its DRAM, so including it would
/// add a second variable for a fixed 6%. Read this arm against `geom, 1
/// warp/cube` at the same width, not against the real lane.
#[cube(launch)]
pub fn swz_width_stream<S: Scalar, NB: Size, NS: Size>(
    b: &Tensor<Vector<u32, NB>>,
    b_sc: &Tensor<Vector<S, NS>>,
    out: &mut Tensor<u32>,
    #[comptime] k_tiles: usize,
    #[comptime] planes: usize,
    #[comptime] words: usize,
    #[comptime] selems: usize,
) {
    let lane = UNIT_POS_PLANE as usize;
    let n_tile = CUBE_POS_Y as usize * comptime!(planes) + UNIT_POS_Y as usize;

    // Codes: `SWZ_BLOCK_CODES / 4` words per k-tile block, `k_tiles` of them
    // contiguous for this n-tile.
    let wpb = comptime!(SWZ_BLOCK_CODES / 4);
    let vecs = comptime!(k_tiles * wpb / words);
    let steps = comptime!(vecs / 32);
    let base = n_tile * vecs;

    let mut acc = u32::new(0i64);
    for j in 0..steps {
        acc += b[base + j * 32 + lane][0];
    }

    // Scales: `NTILE` E4M3 bytes per k-tile block, likewise contiguous.
    let svecs = comptime!(k_tiles * NTILE / selems);
    let ssteps = comptime!(svecs / 32);
    let sbase = n_tile * svecs;
    for j in 0..ssteps {
        acc += u32::cast_from(f32::cast_from(b_sc[sbase + j * 32 + lane][0]));
    }

    if acc == u32::new(0x5AFE_5AFEi64) {
        out[ABSOLUTE_POS as usize % out.len()] = acc;
    }
}

fn swz_width_launch(
    client: &ComputeClient<Rt>,
    b: &Handle,
    b_sc: &Handle,
    out: &Handle,
    k: usize,
    n: usize,
    words: usize,
    scale_elems: usize,
) {
    let k_tiles = k / KTILE;
    let wpb = SWZ_BLOCK_CODES / 4;
    let vecs = n * k_tiles * wpb / words;
    let svecs = n * k_tiles * NTILE / scale_elems;
    unsafe {
        swz_width_stream::launch::<e4m3, Rt>(
            client,
            CubeCount::Static(1, (n / NTILE) as u32, 1),
            CubeDim::new_2d(32, 1),
            words,
            scale_elems,
            TensorArg::from_raw_parts(b.clone(), [1].into(), [vecs].into()),
            TensorArg::from_raw_parts(b_sc.clone(), [1].into(), [svecs].into()),
            TensorArg::from_raw_parts(out.clone(), [1].into(), [1024].into()),
            k_tiles,
            1,
            words,
            scale_elems,
        )
    };
}

#[allow(clippy::too_many_arguments)]
fn swz_geom_launch(
    client: &ComputeClient<Rt>,
    a: &Handle,
    b: &Handle,
    b_sc: &Handle,
    out: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    planes: usize,
    kunroll: usize,
    m_live: usize,
    read_a: bool,
    read_sc: bool,
) {
    assert!(m_live <= MTILE / 2, "this control assumes hi_dead");
    assert_eq!(
        n % (NTILE * planes),
        0,
        "n {n} does not divide {planes} planes"
    );
    let vs = 32 / bf16::cube_type().size_bits();
    let wpr = k / CODES_PER_WORD;
    let spr = k / GROUP;
    unsafe {
        swz_geom_stream::launch::<bf16, e4m3, Rt>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE / planes) as u32, 1),
            CubeDim::new_2d(32, planes as u32),
            vs,
            TensorArg::from_raw_parts(a.clone(), [k, 1].into(), [m_pad, k].into()),
            TensorArg::from_raw_parts(b.clone(), [wpr, 1].into(), [n, wpr].into()),
            TensorArg::from_raw_parts(b_sc.clone(), [spr, 1].into(), [n, spr].into()),
            TensorArg::from_raw_parts(out.clone(), [1].into(), [1024].into()),
            k,
            kunroll,
            planes,
            read_a,
            read_sc,
            m_live as u32,
        )
    };
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// p50 of a slice, and the spread as (min, max).
fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

fn main() {
    let client = Rt::client(&Default::default());

    let n = env("INK_GC_N", 201024);
    let k = env("INK_GC_K", 4096);
    let m_pad = env("INK_GC_M", 16);
    let m_live = env("INK_GC_MLIVE", 1);
    let reps = env("INK_GC_REPS", 8).max(3);
    let planes = env("INK_GC_PLANES", 8);
    let depth = mary::models::inkling::w4a16gemm::swz_unroll();

    let codes = n * (k / 2);
    let scales = n * (k / GROUP);
    let bytes = codes + scales;

    // One shared pair of planes for every arm. At the head's 442 MiB against a
    // 24 MiB L2 nothing survives between two reads of it, so sharing costs no
    // residency asymmetry -- the confound `w4a16_swz_probe`'s header warns about
    // at SINK shapes, and which is why this harness refuses those.
    assert!(
        bytes >= (64usize << 20),
        "[{n}, {k}] is {} MiB, small enough for L2 residency to decide the answer; \
         this harness is for the head shape",
        bytes >> 20
    );

    let a = client.empty(m_pad * k * 2);
    let b = client.empty(codes);
    let b_sc = client.empty(scales);
    let dst = client.empty(4096);

    let vectors = codes / 16;
    let threads = vectors / PER;
    let blocks = (threads as u32).div_ceil(BLOCK);

    println!("=== W4A16 head lane: what the ACCESS PATTERN costs, arithmetic removed ===");
    println!(
        "shape [{m_pad}, {k}] x [{n}, {k}]^T, {m_live} live row(s), swz load depth {depth}, \
         wide arm {planes} warps/cube"
    );
    println!(
        "weight table {:.3} GiB ({bytes} B); GB/s is over it, per LAUNCH, one launch and one \
         sync each\n",
        bytes as f64 / (1u64 << 30) as f64
    );

    let names = [
        "coalesced (242 control)",
        "geom, 1 warp/cube      ",
        "geom, N warps/cube     ",
        "real w4a16_linear_swz  ",
        "geom width 4 B/lane    ",
        "geom width 16 B/lane   ",
        "geom, no A             ",
        "geom, no A no scales   ",
    ];
    let mut per_rep: Vec<Vec<f64>> = vec![Vec::new(); names.len()];

    for _rep in 0..reps {
        for arm in [0usize, 1, 2, 3, 4, 5, 6, 7] {
            let t0 = Instant::now();
            match arm {
                0 => {
                    unsafe {
                        stream_planes::launch::<Rt>(
                            &client,
                            CubeCount::Static(blocks, 1, 1),
                            CubeDim::new_1d(BLOCK),
                            4,
                            TensorArg::from_raw_parts(b.clone(), [1].into(), [vectors].into()),
                            TensorArg::from_raw_parts(
                                b_sc.clone(),
                                [1].into(),
                                [scales / 16].into(),
                            ),
                            TensorArg::from_raw_parts(dst.clone(), [1].into(), [1024].into()),
                            threads,
                            PER,
                        )
                    };
                    let _ = future::block_on(client.sync());
                }
                1 => {
                    swz_geom_launch(
                        &client, &a, &b, &b_sc, &dst, m_pad, k, n, 1, depth, m_live, true, true,
                    );
                    let _ = future::block_on(client.sync());
                }
                2 => {
                    swz_geom_launch(
                        &client, &a, &b, &b_sc, &dst, m_pad, k, n, planes, depth, m_live, true,
                        true,
                    );
                    let _ = future::block_on(client.sync());
                }
                4 => {
                    swz_width_launch(&client, &b, &b_sc, &dst, k, n, 1, 1);
                    let _ = future::block_on(client.sync());
                }
                5 => {
                    swz_width_launch(&client, &b, &b_sc, &dst, k, n, 4, 4);
                    let _ = future::block_on(client.sync());
                }
                6 => {
                    swz_geom_launch(
                        &client, &a, &b, &b_sc, &dst, m_pad, k, n, 1, depth, m_live, false, true,
                    );
                    let _ = future::block_on(client.sync());
                }
                7 => {
                    swz_geom_launch(
                        &client, &a, &b, &b_sc, &dst, m_pad, k, n, 1, depth, m_live, false, false,
                    );
                    let _ = future::block_on(client.sync());
                }
                _ => {
                    let o = w4a16_linear_swz_launch::<Rt>(
                        &client,
                        &a,
                        &b,
                        &b_sc,
                        m_pad,
                        k,
                        n,
                        true,
                        1.0,
                        Some(m_live),
                    );
                    let _ = future::block_on(client.sync());
                    drop(o);
                }
            }
            per_rep[arm].push(t0.elapsed().as_secs_f64());
        }
    }

    println!("  arm                        p50 ms      min      max      GB/s     % of coalesced");
    let warm: Vec<Vec<f64>> = per_rep.iter().map(|v| v[2..].to_vec()).collect();
    let (base, _, _) = stats(&warm[0]);
    for (i, nm) in names.iter().enumerate() {
        let (p50, lo, hi) = stats(&warm[i]);
        println!(
            "  {nm}  {:8.3} {:8.3} {:8.3}  {:8.1}   {:8.1}%",
            p50 * 1e3,
            lo * 1e3,
            hi * 1e3,
            bytes as f64 / p50 / 1e9,
            100.0 * base / p50
        );
    }
    println!(
        "\nframing: GB/s over the {:.3} GiB weight table of ONE [{n}, {k}] operand, per LAUNCH, \
         one launch and one sync each, p50 of {} warm reps of {reps}, GB10. Not a step figure.",
        bytes as f64 / (1u64 << 30) as f64,
        reps - 2
    );
}
