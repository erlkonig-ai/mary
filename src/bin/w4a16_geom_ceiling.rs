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
//! ## What the shuffle rungs measured (spark2 AND spark, 2026-08-27, both boxes
//! locked and idle)
//!
//! Same framing as above -- GB/s over the 0.431 GiB weight table THAT ARM READS,
//! PER LAUNCH, one launch and one sync each, `m_pad = 16` with one live row, p50
//! of 10 warm reps of 12, arms round-robined with the order reversed on odd
//! reps. Three processes a box; the ms/GB/s columns are the spark2 median of
//! three, and spark agrees on every ratio while sitting ~8% slower in absolute
//! terms (its coalesced control is 218-223 GB/s against spark2's 239-240).
//!
//! ```text
//!   arm                          ms     GB/s   % of coalesced   sect/req    L2 req   reg
//!   coalesced (the 242)       1.928    240.2       100.0            16      3.623M    30
//!   geom width 16 B/lane      2.044    226.6        94.3            12      3.625M    80
//!   geom width 4 B/lane       2.021    229.1        95.4             3      4.530M    48
//!   width 16 B/lane + shuffle 1.953    237.1        98.7            12      3.628M    44
//!   width 4 B/lane + shuffle  2.010    230.4        95.9             3      4.636M    38
//!   geom, no A                2.918    158.7        66.1             1     10.539M    37
//!   geom, 1 warp/cube         3.001    154.3        64.2             1     11.097M    48
//!   real w4a16_linear_swz     3.216    144.0        60.0             1     13.406M    73
//! ```
//!
//! The `real` row is the noisiest here: its p50 moves 2.98-3.31 ms between
//! processes while its min sits at 2.86-3.01. Nothing below is quoted against
//! it -- the ratios are against `geom, 1 warp/cube`, which is the same load
//! stream without the math, and against the coalesced control.
//!
//! THE REDISTRIBUTION IS FREE, and that is the whole answer. The paired
//! within-process ratio -- every arm run in one process on one pair of buffers,
//! ratioed rep by rep, which is the only figure here that resolves ~1.1% -- is
//! the p50 of nine processes across both boxes:
//!
//! ```text
//!   shuffle cost, 4 B/lane   (arm 8 / arm 4)   1.011   (0.994 .. 1.052)
//!   shuffle cost, 16 B/lane  (arm 9 / arm 5)   1.001   (0.940 .. 1.028)
//! ```
//!
//! Both sit ON the harness's resolution, not above it. The saturated shuffle
//! count is 768 warp-shuffles a lane per n-tile in BOTH arms (19.3M warp
//! instructions a launch, `smsp__inst_executed.sum` 67.1M against the flat arm's
//! 39.7M at 4 B/lane and 48.4M against 9.7M at 16 B/lane) and it buys no time
//! back and costs none. It lands in the same memory shadow the entire
//! dequantise and MMA already live in.
//!
//! IT COSTS NO TRANSACTIONS EITHER, which is the part an instruction count would
//! have missed. `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` and `sect/req`
//! are IDENTICAL between each flat arm and its shuffle arm (4.825M at 3.00,
//! 1.206M at 12.00) and `lts__d_sectors_fill_sysmem.sum` is 463.7 MB for every
//! arm within 0.2%. The shuffle moves bytes that are already in registers.
//!
//! AND IT COSTS NO REGISTERS. `launch__registers_per_thread` is 38 for the 4
//! B/lane shuffle arm and 44 for the 16 B/lane one, against 48 for the
//! fragment-shaped arm and 73 for the shipped lane; `launch__occupancy_limit_
//! blocks` stays 24 and achieved occupancy 49.6%, unchanged. The staged
//! coalesced words are the COMPRESSED form -- four words hold what sixteen
//! fragment words would -- so shuffling out of a staging register just-in-time
//! is a smaller live set than holding the fragments, not a larger one. That is
//! evidence about the MACHINERY, not a measurement of the shipped lane: these
//! rungs carry no A operand, no accumulator and no MMA. What it says is that
//! the shuffle path costs fewer registers than the fragment-shaped load path it
//! would replace (38 against 48, like for like), so the ~80-register ceiling is
//! not the thing that stops this.
//!
//! NOR DOES IT SERIALISE AGAINST THE LOADS. `lg_throttle` per issue-active
//! COLLAPSES when the shuffles are added -- 18.33 to 0.24 at 4 B/lane and 176.98
//! to 0.27 at 16 B/lane -- with `barrier` 0.00 everywhere and
//! `math_pipe_throttle` 0.05 and 0.01. The shuffles give the warp something to
//! issue instead of queueing loads.
//!
//! WHAT IT BUYS, same paired p50 across the nine processes: the 4 B/lane
//! redistributed arm runs at 0.70 of the fragment-shaped arm and the 16 B/lane
//! one at 0.66 -- 1.43x and 1.52x. Against the coalesced control they are 1.07
//! and 1.01, so the wide redistributed arm is within 1% of the 242 ceiling
//! while reading the same bytes the lane reads.
//!
//! THIS IS THE OPPOSITE OF THE DEPTH EXPERIMENT, and the two must not be
//! conflated. Load DEPTH issues MORE requests EARLIER, and depth 8 failed by
//! losing miss-merging (14.96M L2 requests against depth 4's 13.47M, landing on
//! depth 1's number). Width issues FEWER, WIDER requests: the 16 B/lane shuffle
//! arm's 3.628M L2 requests are the lowest of any arm here and equal the
//! coalesced control's 3.625M, for the same DRAM bytes.
//!
//! WHAT THE WIDTH ROWS DID NOT SAY, AND THE SHUFFLE ROWS DO. `geom width` is not
//! a proposed kernel: it reads the right bytes in the right order and hands them
//! to the WRONG LANES. The `+ shuffle` arms add exactly the missing step and
//! nothing else — the coalesced load, then a warp shuffle per fragment word, then
//! discard — so `+ shuffle` over `width` is the PRICE OF THE REDISTRIBUTION and
//! `+ shuffle` over `geom, 1 warp/cube` is what it BUYS.
//!
//! No shared memory and no barrier, because the lane runs ONE WARP PER CUBE:
//! there is nothing to synchronise on (the stall decomposition reads `barrier`
//! 0), so a warp that loads coalesced can redistribute to its own lanes with
//! register-to-register shuffles. `cubecl` lowers `plane_shuffle` to
//! `__shfl_sync` on the CUDA dialect; `MmaDefinition::load_matrix` (`ldmatrix`)
//! is also reachable but requires the operand to be in SHARED memory, so it
//! would need the stage the shuffle path avoids.
//!
//! THE SHUFFLE COUNT IS EXACTLY SATURATED, not a guess. One warp step at
//! `words` u32 per lane covers `2 * words` `(n_tile, k_tile)` blocks; the
//! `m16n8k16` B map gives lane `l` column `l >> 2` and elements
//! `2 * (l & 3) + 8 * i`, so lane `l` wants two words out of each block —
//! `4 * words` words a step. Each shuffle instruction delivers one word to one
//! lane and each word is wanted by four lanes, so `4 * words` shuffles a step is
//! the floor and the arms issue exactly that. At `words = 1` the source lanes
//! are the true ones (`col`, `col + 8`, `col + 16`, `col + 24`) and the arm is
//! the real redistribution over the SHIPPED swizzle. At `words = 4` the current
//! swizzle puts all sixteen wanted words in one vector component, which sixteen
//! shuffles cannot reach; that arm issues the saturated count over a
//! representative source-lane pattern, so it prices a 16 B/lane redistribution
//! ASSUMING A RE-SWIZZLE that spreads the wanted words across components. Read
//! row 8 as achievable today and row 9 as achievable after a format change.
//!
//! ## And then the rung became the kernel (2026-08-27, both boxes locked and idle)
//!
//! Everything above is the LOAD HALF -- rungs carrying no A operand, no
//! accumulator and no MMA -- so none of it was a claim about the shipped lane.
//! `real swz + shuffle` is that claim: the redistribution built into
//! `w4a16_linear_swz` itself, behind `w4a16gemm::swz_shuffle`
//! (`INK_W4A16_SWZ_SHUFFLE`, default off), round-robined against the flag-off
//! lane in the SAME process so the paired ratio applies to it.
//!
//! ```text
//!   arm                      spark2 p50 ms  GB/s     spark p50 ms  GB/s
//!   coalesced (the ceiling)      1.940     238.8        2.121     218.4
//!   real w4a16_linear_swz        3.121     148.4        3.093     149.7
//!   real swz + shuffle           2.658     174.2        2.642     175.3
//! ```
//!
//! Medians of three processes a box, same framing rule as above. Paired
//! within-process p50 of arm 10 over arm 3, over six processes across both
//! boxes: 0.860, range 0.818 .. 0.898. So the shipped lane's weight read is
//! 1.16x, and the lane moves from 62% of the coalesced control to 73%.
//!
//! IT IS NOT THE 95% THE LOAD RUNGS REACHED, and it was never going to be: arms
//! 8 and 9 drop the A operand, the accumulator and the MMA, and the kernel does
//! not. What the rungs bought was the right to expect a large win rather than a
//! small one; the size of it is only knowable by running the whole lane, which
//! is what arm 10 is for.
//!
//! Registers went DOWN, 73 to 61, with `launch__occupancy_limit_blocks` still 24
//! and the register limit moving off it (24 -> 32 blocks). L2 requests 13.454M
//! -> 5.451M for byte-identical DRAM traffic (14.485M `lts__d_sectors_fill_
//! sysmem` sectors either way, 0.014% apart). The L1 saving is exact rather than
//! approximate: codes 12.866M -> 3.216M requests, scales 6.433M -> 1.608M, A
//! untouched at 12.866M, predicted 14.475M saved and 14.474M measured.
//!
//! Timing only, deliberately: every arm reads the same bytes in the same order
//! as the lane it stands in for, so an uninitialised table exercises the
//! identical access pattern. Numerics are `w4a16_swz_probe`'s job, and it gates
//! the flag bit-for-bit.
//!
//! `INK_GC_ORDER=fwd` pins the arm order to the committed protocol; the default
//! reverses it on odd reps so no arm keeps the same slot in the rep.
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
    CODES_PER_WORD, GROUP, KTILE, MTILE, NTILE, SWZ_BLOCK_CODES, w4a16_linear_swz_launch_redist,
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
    #[comptime] redist: bool,
) {
    let lane = UNIT_POS_PLANE as usize;
    let n_tile = CUBE_POS_Y as usize * comptime!(planes) + UNIT_POS_Y as usize;

    // The n column this lane's B fragment sits in: `col = lane >> 2`, the closed
    // form `mma16_frag_map` dumped off sm_121a. It is the whole reason four lanes
    // land on one word, and it is also what makes the redistribution a shuffle.
    let col = UNIT_POS_PLANE / 4;

    // Codes: `SWZ_BLOCK_CODES / 4` words per k-tile block, `k_tiles` of them
    // contiguous for this n-tile.
    let wpb = comptime!(SWZ_BLOCK_CODES / 4);
    let vecs = comptime!(k_tiles * wpb / words);
    let steps = comptime!(vecs / 32);
    let base = n_tile * vecs;

    let mut acc = u32::new(0i64);
    for j in 0..steps {
        let v = b[base + j * 32 + lane];
        if comptime!(redist) {
            // One warp step covers `32 * words` words = `2 * words` k-tile
            // blocks, and every lane needs two words out of each of them
            // (`i = 0, 1` of the m16n8k16 B fragment). That is `4 * words`
            // words a lane, so `4 * words` shuffles a step -- exactly
            // saturated, because each shuffle instruction delivers one word to
            // one lane and each word is wanted by four lanes.
            #[unroll]
            for comp in 0..words {
                #[unroll]
                for r in 0..4usize {
                    acc += plane_shuffle(v[comp], col + (8 * r) as u32);
                }
            }
        } else {
            acc += v[0];
        }
    }

    // Scales: `NTILE` E4M3 bytes per k-tile block, likewise contiguous.
    let svecs = comptime!(k_tiles * NTILE / selems);
    let ssteps = comptime!(svecs / 32);
    let sbase = n_tile * svecs;
    for j in 0..ssteps {
        let sv = b_sc[sbase + j * 32 + lane];
        if comptime!(redist) {
            // One step covers `4 * selems` k-tile blocks and every lane needs
            // one scale out of each: `4 * selems` shuffles, same shape.
            #[unroll]
            for comp in 0..selems {
                let s = u32::cast_from(f32::cast_from(sv[comp]));
                #[unroll]
                for r in 0..4usize {
                    acc += plane_shuffle(s, col + (8 * r) as u32);
                }
            }
        } else {
            acc += u32::cast_from(f32::cast_from(sv[0]));
        }
    }

    if acc == u32::new(0x5AFE_5AFEi64) {
        out[ABSOLUTE_POS as usize % out.len()] = acc;
    }
}

#[allow(clippy::too_many_arguments)]
fn swz_width_launch(
    client: &ComputeClient<Rt>,
    b: &Handle,
    b_sc: &Handle,
    out: &Handle,
    k: usize,
    n: usize,
    words: usize,
    scale_elems: usize,
    redist: bool,
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
            redist,
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
        "coalesced (242 control)  ",
        "geom, 1 warp/cube        ",
        "geom, N warps/cube       ",
        "real w4a16_linear_swz    ",
        "geom width 4 B/lane      ",
        "geom width 16 B/lane     ",
        "geom, no A               ",
        "geom, no A no scales     ",
        "width 4 B/lane + shuffle ",
        "width 16 B/lane + shuffle",
        "real swz + shuffle       ",
    ];
    let mut per_rep: Vec<Vec<f64>> = vec![Vec::new(); names.len()];

    // Arm ORDER is alternated rep to rep unless `INK_GC_ORDER=fwd` pins it. The
    // committed table was taken in a fixed order, which leaves every arm's
    // position in the rep confounded with the arm; reversing on odd reps
    // balances that out without giving up the paired within-process delta,
    // which is where this harness's ~1.1% resolution lives. `fwd` reproduces
    // the committed protocol exactly.
    let order_fwd = std::env::var("INK_GC_ORDER")
        .map(|v| v == "fwd")
        .unwrap_or(false);
    let fwd: Vec<usize> = (0..names.len()).collect();
    let rev: Vec<usize> = (0..names.len()).rev().collect();

    for _rep in 0..reps {
        let sched: &[usize] = if order_fwd || _rep % 2 == 0 {
            &fwd
        } else {
            &rev
        };
        for &arm in sched {
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
                    swz_width_launch(&client, &b, &b_sc, &dst, k, n, 1, 1, false);
                    let _ = future::block_on(client.sync());
                }
                5 => {
                    swz_width_launch(&client, &b, &b_sc, &dst, k, n, 4, 4, false);
                    let _ = future::block_on(client.sync());
                }
                8 => {
                    swz_width_launch(&client, &b, &b_sc, &dst, k, n, 1, 1, true);
                    let _ = future::block_on(client.sync());
                }
                9 => {
                    swz_width_launch(&client, &b, &b_sc, &dst, k, n, 4, 4, true);
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
                // Arms 3 and 10 are the SHIPPED lane with the weight-load form
                // as the only variable, said explicitly rather than read from
                // the environment, so both live in one process and the paired
                // ratio 10/3 resolves what a two-process comparison cannot.
                3 | 10 => {
                    let o = w4a16_linear_swz_launch_redist::<Rt>(
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
                        arm == 10,
                    );
                    let _ = future::block_on(client.sync());
                    drop(o);
                }
                _ => unreachable!("arm {arm} has no launch"),
            }
            per_rep[arm].push(t0.elapsed().as_secs_f64());
        }
    }

    println!(
        "  arm                          p50 ms      min      max      GB/s     % of coalesced"
    );
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
    // THE PAIRED DELTA, which is the only figure here that resolves ~1.1%.
    // Every rep runs every arm inside one process on one pair of buffers, so a
    // rep-by-rep ratio cancels the drift that makes the absolute rows move 5-10%
    // between processes. `a / b` is the cost of what `a` does and `b` does not.
    let pair = |a: usize, b: usize| -> (f64, f64, f64) {
        let r: Vec<f64> = warm[a]
            .iter()
            .zip(warm[b].iter())
            .map(|(x, y)| x / y)
            .collect();
        stats(&r)
    };
    println!("\n  paired within-process ratio (per rep, then p50 / min / max)");
    for (lbl, a, b) in [
        // THE SHIPPED LANE, flag on over flag off. Everything else here is a
        // rung; this row is the kernel.
        ("REAL: shuffle on/off     (10/3)", 10usize, 3usize),
        ("real+shfl vs coalesced   (10/0)", 10, 0),
        ("real+shfl vs fragment map(10/1)", 10, 1),
        ("shuffle cost, 4 B/lane   (8/4)", 8usize, 4usize),
        ("shuffle cost, 16 B/lane  (9/5)", 9, 5),
        ("4 B+shfl vs no-A fragment(8/6)", 8, 6),
        ("16 B+shfl vs no-A frag   (9/6)", 9, 6),
        ("4 B+shfl vs coalesced    (8/0)", 8, 0),
        ("16 B+shfl vs coalesced   (9/0)", 9, 0),
        ("4 B+shfl vs fragment map (8/1)", 8, 1),
        ("16 B+shfl vs fragment map(9/1)", 9, 1),
        ("4 B+shfl vs real lane    (8/3)", 8, 3),
        ("width 4 B vs fragment map(4/1)", 4, 1),
        ("width 16 B vs coalesced  (5/0)", 5, 0),
    ] {
        let (p50, lo, hi) = pair(a, b);
        println!("    {lbl}   {p50:7.4}   {lo:7.4}   {hi:7.4}");
    }

    println!(
        "\nframing: GB/s over the {:.3} GiB weight table of ONE [{n}, {k}] operand, per LAUNCH, \
         one launch and one sync each, p50 of {} warm reps of {reps}, GB10. Not a step figure.",
        bytes as f64 / (1u64 << 30) as f64,
        reps - 2
    );
}
