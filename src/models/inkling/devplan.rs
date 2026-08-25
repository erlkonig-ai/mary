//! The routed-expert row plan built ON THE DEVICE, from a decision that never
//! comes back.
//!
//! # What this deletes
//!
//! [`super::routetopk`] already picks the experts where the logits are. What it
//! could not delete is the READ: `inkling_forward` pulled its
//! `[n, 2k + shared + 1]` answer to the host every layer, because the host was
//! the only thing that knew how to turn a decision into a
//! [`super::moegroup::RowPlan`] and a table of weight offsets. That read is the
//! only blocking point left in a decode layer, and a blocking point per layer
//! is a queue that never runs deeper than one layer.
//!
//! Measured on spark-zt at `INK_KV=1 INK_LAYERS=0:16`, nine interleaved rounds
//! of the `INK_ROUTE_STALE=1` probe against the same binary with the probe off:
//! **55.7 -> 51.2 ms per decode pass at p50**, 17.95 -> 19.53 tok/s. That is
//! the CEILING this module aims at and not a fraction of the read's own
//! 19.3 ms bucket: only 4.5 ms of that bucket is serialisation, the rest
//! resurfaces at the one sync after the stack and as launch overhead once the
//! queue runs deep.
//!
//! # What it measured
//!
//! `INK_DEV_PLAN=ab:10` alternates the two arms every ten decode passes inside
//! ONE process, which is the only honest pairing: the difference is about four
//! milliseconds and pass-to-pass drift on this box is two to three. Three runs
//! on spark-zt, `INK_KV=1 INK_LAYERS=0:16`, a five-token prompt grown to 707,
//! 350 decode passes an arm, p50 of the whole pass:
//!
//! ```text
//!   run   host plan   device plan   device against host
//!    1     57.4 ms      52.9 ms       -4.5 ms, +8.5% tok/s
//!    2     55.2 ms      52.3 ms       -2.9 ms, +5.6% tok/s
//!    3     56.5 ms      52.8 ms       -3.7 ms, +7.0% tok/s
//! ```
//!
//! The DEVICE arm is the reproducible one — 52.3 to 52.9 ms across the three,
//! a 0.6 ms spread — and the host arm is what moves, 55.2 to 57.4. That is the
//! expected shape: the arm with a blocking read per layer is the arm whose
//! wall clock depends on when the host gets scheduled.
//!
//! Against the `INK_ROUTE_STALE=1` probe, which is the CEILING for deleting
//! this sync and not a target to beat: 55.7 -> 51.2 ms p50, +8.8%. This lane
//! reaches 52.3-52.9, so it captures roughly four fifths of what the probe said
//! was there, and the 1.1-1.7 ms it does not reach is the fourteen plan
//! launches it adds. Anyone quoting the readback's own 19.3 ms bucket as the
//! win is wrong by a factor of four; only 4.5 ms of that bucket is
//! serialisation and the rest resurfaces at the one sync after the stack.
//!
//! # Why it is possible at `n == 1` and not at a prefill
//!
//! Six of `RowPlan`'s seven fields are a function of `n` and `top_k` ALONE at
//! `n == 1`, and this is measured rather than reasoned: `INK_PLAN_CHECK=1` in
//! `inkling_forward` keeps the first plan it sees at each `n` and compares
//! every later one against it. Over 512 plans of a 40-step decode on this box,
//! `row_tok`, `blk_slot`, `blk_tile0`, `blk_cnt`, `tok_rows`, `tok_cnt` and
//! `kmax` differed from the first plan ZERO times, and `row_wgt` differed 511
//! times. The reason is that one token routes to `top_k` DISTINCT experts of
//! one row each, so every expert occupies exactly one padded
//! [`super::fp4gemm::MTILE`]-row tile and the stacking is the identity.
//!
//! At a prefill it is false in the loudest possible way, and the same check
//! shows it: at `n == 5` the number of distinct experts moved between 17 and
//! 24 from one layer to the next, so `blk_slot` changed LENGTH. A prefill keeps
//! the host lane, and that is a property of the shape rather than a limitation
//! to be lifted.
//!
//! # `n > 1`: what the `n == 1` restriction actually was
//!
//! It was DEDUPLICATION, not width. The sentence above is true of a plan that
//! gives each DISTINCT expert one tile, and at `n == 1` that is automatic
//! because one token's `top_k` picks are distinct by construction. The moment
//! two rows can pick the same expert, the distinct count is data and the shape
//! follows it.
//!
//! So this lane stopped deduplicating. `devroute_new` builds `n * top_k` SLOTS
//! of one token each -- slot `s` is row `s / top_k`'s `s % top_k`-th pick --
//! and every field of `RowPlan` is a function of `n` and `top_k` again, at
//! every width up to [`super::fp4gemm::MTILE`]. Above `MTILE` a slot would need
//! more than one tile and the count would be data-dependent once more, which is
//! where a prefill lives and where the host lane still belongs.
//!
//! **What it costs and what it buys, both measured on spark2-zt (GB10), one
//! node, layers 0:21, a 3772-token prompt grown by 12 cached decode steps,
//! interleaved arms, idle- and memory-gated.** The cost is re-reading a shared
//! expert's plane: the log's own `expert slabs decoded` counter reads 114 a
//! pass at `n = 1`, ~200 at `n = 2` as the router actually routes, and 228
//! without dedup -- so the charge is 28 slabs, about 0.40 GiB a pass, ~2 ms at
//! this part's achievable bandwidth. The purchase is the readback, and its
//! ceiling is what `INK_ROUTE_STALE=1` measures against an interleaved control:
//!
//! ```text
//!   width   base ms/step   probe ms/step   the readback is worth
//!     1         50.7            50.5              0.2 ms
//!     2         72.2            64.1              8.1 ms
//!     3         76.4            70.4              6.0 ms
//! ```
//!
//! Width 1 is 0.2 ms because this lane was ALREADY on there -- there is no
//! readback left to delete, which is the control that says the probe measures
//! what it claims. And note what the 8.1 ms is NOT: the per-layer `BLOCKING
//! read` bucket reads 22.6 ms at width 2, so the bucket overstates the prize by
//! 2.8x for exactly the reason this module already records at `n == 1`. Most of
//! a blocking read is device time the host would have waited for somewhere.
//!
//! `INK_DEV_PLAN_MAXN=1` restores the old gate exactly, which is what makes the
//! widening an A/B rather than a replacement.
//!
//! # The sort, which is the part to get right
//!
//! The host lane's expert order is `BTreeMap` order — ASCENDING expert id — and
//! that order is not a detail. It is the order the scatter accumulates a
//! token's `top_k` contributions in, and floating-point addition is not
//! associative, so a different order is a different sum. [`super::moegroup`]
//! makes the point in its own words: the routing, the gather, the operand order
//! and the accumulation order "are not approximations of anything, so a
//! difference in any of them would be a defect".
//!
//! `router_topk_launch` emits its picks in DESCENDING SCORE order, so the
//! kernel below sorts them ascending by id before it does anything else. It
//! sorts by RANK — each id counts how many of the others are smaller — which is
//! `k^2 = 36` comparisons in one thread, needs no swaps, and has no
//! data-dependent control flow to get wrong. Two equal ids would collide onto
//! one rank, so the kernel raises the fault flag rather than writing a plan
//! with a hole in it.
//!
//! # What it costs
//!
//! One launch per layer, one unit wide. It writes `top_k * MTILE` floats,
//! `4 * top_k` offsets and `2 * top_k` scales — under a kilobyte — and reads
//! `top_k` rows of a table that was uploaded once for the whole run.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// One layer's weight table, over EVERY routed expert, uploaded once.
///
/// The host lane looked up four numbers per ACTIVE expert per layer per pass,
/// which it could only do because it knew which experts were active. Nothing on
/// the host knows that any more, so the table covers all of them and the kernel
/// indexes it with the ids the device already holds. It is
/// `n_routed * (2 * 8 + 2 * 8 + 4 + 4)` bytes — 10 KiB at 256 experts — held
/// for the run.
pub struct ExpertTable {
    /// `[n_routed * 2]` u64: `w13`'s code and scale plane offsets, per expert.
    pub off13: Handle,
    /// `[n_routed * 2]` u64: `w2`'s, per expert.
    pub off2: Handle,
    /// `[n_routed]` f32: `w13`'s second-level quantisation constant.
    pub sc13: Handle,
    /// `[n_routed]` f32: `w2`'s.
    pub sc2: Handle,
    /// The one registered mapping every plane of this layer lives in.
    pub wmap: Handle,
    /// Its length, which is what the GEMM binds.
    pub wmap_bytes: usize,
    /// Bytes in one expert's four planes. The alias accounting is charged
    /// `top_k` of these a layer, because the host no longer sees the binds it
    /// used to count one at a time. Every expert of a layer has the same shape,
    /// so this is the exact figure and not an average.
    pub expert_bytes: usize,
    /// Routed experts in the layer.
    pub n_routed: usize,
    /// Offset-table entries per expert: TWO on an NVFP4 layer (a code plane
    /// and a scale plane) and ONE on a BF16 layer (nothing quantised it, so
    /// there is no scale plane and no second-level constant either). It is the
    /// stride of both the table and the plan, so the two grouped GEMMs get the
    /// shape each already expects.
    pub stride: usize,
    /// Whether `sc13`/`sc2` mean anything. A BF16 layer fills them with zeros
    /// so the kernel needs no second form, and its GEMM never reads them.
    pub scaled: bool,
}

/// What one launch of [`plan_from_topk`] produced for one layer of one pass.
///
/// The same five buffers `RowPlan::build` and the offset loop used to upload,
/// with the same contents and without the round trip.
pub struct DevRowPlan {
    /// `[top_k * MTILE]` f32, the routing weight at each stacked row.
    pub row_wgt: Handle,
    /// `[2 * top_k]` u64.
    pub off13: Handle,
    /// `[2 * top_k]` u64.
    pub off2: Handle,
    /// `[top_k]` f32.
    pub sc13: Handle,
    /// `[top_k]` f32.
    pub sc2: Handle,
    /// `[top_k]` u32: the chosen expert ids IN THE ORDER THE PLAN STACKED THEM.
    ///
    /// Written for one reason and it is not the GEMM, which never reads it: it
    /// is the only place the sort's answer exists as INTEGERS. Everything else
    /// this kernel emits is a consequence of the order — offsets, scales,
    /// weights — and a wrong order shows up in those as a permutation you have
    /// to decode. `INK_DEVPLAN_CHECK=1` reads this and compares it to the
    /// host's `BTreeMap` keys directly, so a mis-sort names itself.
    ///
    /// It is six stores in a kernel that already does ninety-six, so it is
    /// written on every pass rather than behind a flag: a debug output that
    /// only exists in debug builds is a debug output that was never tested.
    pub ids: Handle,
}

/// Turn one token's top-k answer into the grouped lane's row plan.
///
/// One unit. `top_k` is six and the whole body is `k^2` comparisons plus
/// `k * MTILE` stores, so a wider cube would spend more on scheduling than the
/// work is worth — and a single unit is what makes the ordering below a plain
/// sequence rather than a race.
///
/// `fault` is a RUN-SCOPED accumulator, not a per-layer output: `1 + row` when
/// `routetopk` saw a non-finite logit, `0xdup` when two of the picks were the
/// same expert. The host reads it once, after the run, which is the whole point
/// — a per-layer read of it would be the read this module exists to delete.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn plan_from_topk(
    topk: &Array<f32>,
    tab_off13: &Array<u64>,
    tab_off2: &Array<u64>,
    tab_sc13: &Array<f32>,
    tab_sc2: &Array<f32>,
    row_wgt: &mut Array<f32>,
    off13: &mut Array<u64>,
    off2: &mut Array<u64>,
    sc13: &mut Array<f32>,
    sc2: &mut Array<f32>,
    ids: &mut Array<u32>,
    fault: &mut Array<u32>,
    #[comptime] top_k: u32,
    #[comptime] mtile: u32,
    #[comptime] width: u32,
    #[comptime] stride: u32,
    #[comptime] rows: u32,
) {
    // One unit, so no guard and no barrier: the cube is one wide (see
    // [`plan_from_topk_launch`]) and everything below is a plain sequence.
    //
    // The pad rows first, so the scatter below writes over a known field rather
    // than into whatever the allocator last held. `row_wgt` is `top_k * mtile`
    // floats, 96 of them at the shape this lane runs in.
    let m_total = comptime!(rows * top_k * mtile);
    for i in 0..m_total {
        row_wgt[i as usize] = 0.0f32;
    }

    let mut bad = fault[0];

    // ASCENDING EXPERT ID, by rank rather than by swaps. `routetopk` hands the
    // picks over in descending score order and the accumulation order is
    // defined by the id, so this is the whole correctness of the module in a
    // dozen lines. See the module doc.
    for row in 0..rows {
        // The row's own window of `routetopk`'s output, and the row's own run of
        // `top_k` slots. Row `t` owns slots `t * top_k .. (t + 1) * top_k` and no
        // other row's, which is what keeps every SHAPE below a function of `rows`
        // and `top_k` alone -- see the module doc's `n > 1` section.
        let rb = row * comptime!(width);
        let sb = row * comptime!(top_k);
        let flag = u32::cast_from(topk[(rb + comptime!(width - 1)) as usize]);
        if flag != 0u32 {
            bad = flag;
        }
        for j in 0..top_k {
        let e = u32::cast_from(topk[(rb + j) as usize]);
        let w = topk[(rb + comptime!(top_k) + j) as usize];
        let mut r = u32::new(0);
        for q in 0..top_k {
            let o = u32::cast_from(topk[(rb + q) as usize]);
            if o < e {
                r += 1u32;
            }
            if q != j {
                if o == e {
                    // Two picks on one expert would land on one rank and leave
                    // a slot unwritten. It cannot happen -- `routetopk` takes
                    // the next-largest score `top_k` times and masks each pick
                    // out -- and "cannot happen" is exactly the class of thing
                    // worth a comparison and a flag.
                    bad = 0xdu32;
                }
            }
        }
        let es = e as usize;
        let rs = (sb + r) as usize;
        // One entry an expert on a BF16 layer, two on an NVFP4 one. See
        // [`ExpertTable::stride`].
        for q in 0..stride {
            let qs = q as usize;
            off13[stride as usize * rs + qs] = tab_off13[stride as usize * es + qs];
            off2[stride as usize * rs + qs] = tab_off2[stride as usize * es + qs];
        }
        sc13[rs] = tab_sc13[es];
        sc2[rs] = tab_sc2[es];
        ids[rs] = e;
        row_wgt[rs * mtile as usize] = w;
        }
    }
    fault[0] = bad;
}

/// Launch [`plan_from_topk`] for one layer.
///
/// `topk` is `router_topk_launch`'s own output buffer, unread. `tab` is the
/// layer's [`ExpertTable`]. `fault` is the run-scoped accumulator.
#[allow(clippy::too_many_arguments)]
pub fn plan_from_topk_launch<R: Runtime>(
    client: &ComputeClient<R>,
    topk: &Handle,
    tab: &ExpertTable,
    fault: &Handle,
    top_k: usize,
    mtile: usize,
    width: usize,
    rows: usize,
) -> DevRowPlan {
    let slots = rows * top_k;
    let m_total = slots * mtile;
    let st = tab.stride;
    let row_wgt = client.empty(m_total * core::mem::size_of::<f32>());
    let off13 = client.empty(st * slots * core::mem::size_of::<u64>());
    let off2 = client.empty(st * slots * core::mem::size_of::<u64>());
    let sc13 = client.empty(slots * core::mem::size_of::<f32>());
    let sc2 = client.empty(slots * core::mem::size_of::<f32>());
    let ids = client.empty(slots * core::mem::size_of::<u32>());
    unsafe {
        plan_from_topk::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(topk.clone(), rows * width),
            ArrayArg::from_raw_parts(tab.off13.clone(), st * tab.n_routed),
            ArrayArg::from_raw_parts(tab.off2.clone(), st * tab.n_routed),
            ArrayArg::from_raw_parts(tab.sc13.clone(), tab.n_routed),
            ArrayArg::from_raw_parts(tab.sc2.clone(), tab.n_routed),
            ArrayArg::from_raw_parts(row_wgt.clone(), m_total),
            ArrayArg::from_raw_parts(off13.clone(), st * slots),
            ArrayArg::from_raw_parts(off2.clone(), st * slots),
            ArrayArg::from_raw_parts(sc13.clone(), slots),
            ArrayArg::from_raw_parts(sc2.clone(), slots),
            ArrayArg::from_raw_parts(ids.clone(), slots),
            ArrayArg::from_raw_parts(fault.clone(), 1),
            top_k as u32,
            mtile as u32,
            width as u32,
            st as u32,
            rows as u32,
        )
    };
    DevRowPlan {
        row_wgt,
        off13,
        off2,
        sc13,
        sc2,
        ids,
    }
}
