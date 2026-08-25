//! `fp4_linear_grouped` against its shared-memory-staged sibling, both arms in
//! ONE process, on ONE weight table, at the routed-expert shape.
//!
//! # Why both arms are in one binary
//!
//! Burn points cubecl's autotune cache at `$CWD/target/autotune`, so two
//! worktrees on this box keep two caches and a cache entry records WHICH KERNEL
//! WON A TIMING RACE. Measured on this machine: of 64 shapes present in both
//! caches, 58 name a different winner. A cross-worktree before/after is
//! therefore not a measurement of a code change. So both arms live here, run
//! back to back, on the same handles, seconds apart — the only thing that
//! differs between them is the kernel.
//!
//! # The shape
//!
//! A batched decode's routed-expert lane, which is what the 2.7 ms of headroom
//! is about. `INK_PROBE_EXPERTS` distinct experts each contribute
//! `INK_PROBE_ROWS` real rows padded to a whole 16-row tile, so at the default
//! 114 x 1 the layer stacks 114 tiles holding 114 real rows — 1 row a tile,
//! which is the decode regime [`grouped_nrep`]'s FILL gate is about, and which
//! takes `nrep` to 1. The weight table is then 114 x 9 MiB = 1.0 GiB against a
//! 24 MiB L2, i.e. 43x, so nothing here can be served warm.
//!
//! At that shape every block has `cnt = 1`: one m tile per expert, so three of
//! a four-plane cube's planes have nothing to do. The baseline `terminate!()`s
//! them. The staged arm cannot — a cube-wide barrier with exited threads is
//! undefined — so it keeps them in and they fill shared memory instead. That is
//! not a workaround that costs something; at decode it is free load bandwidth.
//!
//! # What is checked, and why it is bit equality
//!
//! The staged kernel changes WHERE the B fragment is read from and nothing
//! else: the same `execute_scaled` calls in the same k order on the same
//! operands. So the two arms must agree BIT FOR BIT, and anything less is a
//! defect in the staging, not a rounding difference. Both outputs are read back
//! and compared word for word.
//!
//! `INK_PROBE_K` / `INK_PROBE_N` set the GEMM (default 4096 x 4096, the w13
//! stage; the w2 stage is `k = 2048`, `n = 4096`). `INK_MOE_KC`, `INK_MOE_PAD`
//! and `INK_MOE_PLANES` are read by the kernels themselves.

use std::time::Instant;

use cubecl::prelude::*;
use mary::models::inkling::moegroup::{
    BlockPlanDev, RowPlan, fp4_linear_grouped_launch_as, fp4_linear_grouped_smem_launch_tuned,
    grouped_nrep,
};

type Rt = cubecl::cuda::CudaRuntime;

/// The little-endian bytes of a `[u32]` — no `bytemuck` in this crate's graph.
fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The little-endian bytes of a `[u64]`.
fn le64(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The little-endian bytes of a `[f32]`.
fn lef32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn env(name: &str, dflt: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(dflt)
}

/// Best and first-pass seconds over `reps` timed launches after two warmups.
///
/// The first pass is printed beside the best because a best that beats its own
/// first by more than jitter would mean the table was small enough to cache.
fn best_first(mut run: impl FnMut() -> f64, reps: usize) -> (f64, f64) {
    for _ in 0..2 {
        run();
    }
    let (mut best, mut first) = (f64::MAX, 0.0);
    for i in 0..reps {
        let dt = run();
        if i == 0 {
            first = dt;
        }
        best = best.min(dt);
    }
    (best, first)
}

fn main() {
    let client = Rt::client(&Default::default());
    let k = env("INK_PROBE_K", 4096);
    let n = env("INK_PROBE_N", 4096);
    let experts = env("INK_PROBE_EXPERTS", 114);
    let rows = env("INK_PROBE_ROWS", 1);
    let reps = env("INK_PROBE_REPS", 5);

    // One expert's two planes inside the registered mapping, laid out exactly
    // as the pile's are: `[n, k/2]` packed codes then `[n, k/16]` E4M3 scales.
    let codes = n * (k / 2);
    let scales = n * (k / 16);
    let per_expert = codes + scales;
    let wmap_bytes = experts * per_expert;

    // A deterministic fill rather than `client.empty`. Uninitialised scale
    // bytes can be E4M3 NaN, and a NaN output would make the bit comparison
    // below pass for the wrong reason -- it never compares equal to itself, so
    // an arm that produced garbage would be indistinguishable from one that
    // produced a different garbage. 0x7f and 0xff are the two NaN encodings and
    // are the only bytes excluded.
    let mut w = vec![0u8; wmap_bytes];
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    for (i, b) in w.iter_mut().enumerate() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let byte = (s >> 33) as u8;
        // Which plane of which expert this byte is in decides whether it has to
        // be a finite E4M3.
        *b = if i % per_expert >= codes {
            byte & 0x6f
        } else {
            byte
        };
    }
    let wmap = client.create_from_slice(&w);
    drop(w);

    // The routing: `experts` distinct experts, `rows` real rows each. One
    // expert contributes one block at `rows <= 16`.
    let toks: Vec<Vec<(usize, f32)>> = (0..experts)
        .map(|e| (0..rows).map(|r| (e * rows + r, 0.5f32)).collect())
        .collect();
    // Byte offsets into the mapping, `[codes, scales]` a slot, and the
    // second-level constants.
    let off: Vec<u64> = (0..experts)
        .flat_map(|e| [(e * per_expert) as u64, (e * per_expert + codes) as u64])
        .collect();
    let off_h = client.create_from_slice(&le64(&off));
    let sc2: Vec<f32> = (0..experts).map(|_| 1.0f32).collect();
    let sc2_h = client.create_from_slice(&lef32(&sc2));

    let bytes = experts * per_expert;
    println!("=== fp4_linear_grouped: global B against staged B, one process ===");
    println!("  shape        k={k} n={n}, {experts} experts x {rows} real rows");
    println!(
        "  weights      {:.3} GiB, {:.1}x a 24 MiB L2",
        bytes as f64 / (1 << 30) as f64,
        bytes as f64 / (24.0 * 1024.0 * 1024.0)
    );
    println!(
        "  every arm below reads THAT table, in this process, seconds apart; GB/s is over it\n"
    );
    println!("  {:<50} {:>9} {:>9} {:>9}", "arm", "GB/s", "ms", "1st ms");

    let report = |label: &str, best: f64, first: f64| {
        println!(
            "  {:<50} {:>9.1} {:>9.3} {:>9.3}",
            label,
            bytes as f64 / best / 1e9,
            best * 1e3,
            first * 1e3
        );
        bytes as f64 / best / 1e9
    };

    // The plane count is a HOST-side plan parameter -- it decides how many m
    // tiles share a cube -- so each value needs its own uploaded block plan.
    // Every other axis is a launch parameter and varies inside one plan.
    let planes_list: Vec<usize> = std::env::var("INK_PROBE_PLANES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);
    let kc_list: Vec<usize> = std::env::var("INK_PROBE_KC")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 8, 16]);

    let mut best_base = (0.0f64, String::new());
    let mut best_smem = (0.0f64, String::new());
    let mut checked: Option<(usize, usize, usize)> = None;

    for &planes in &planes_list {
        let plan = RowPlan::build(toks.iter(), experts * rows, planes);
        let m_total = plan.m_total();
        let rows_real = plan.rows_real();
        let blocks = plan.blk_slot.len();
        let blk = BlockPlanDev {
            slot: client.create_from_slice(&le32(&plan.blk_slot)),
            tile0: client.create_from_slice(&le32(&plan.blk_tile0)),
            cnt: client.create_from_slice(&le32(&plan.blk_cnt)),
            blocks,
            planes,
            rows_real,
        };
        let nrep = grouped_nrep(n, m_total, rows_real);
        // The stacked A, at this plan's height. `m_total` does not actually
        // depend on the plane count -- padding is per expert, not per cube --
        // but deriving it here rather than assuming that is one fewer thing to
        // be wrong about.
        let a_bytes = m_total * (k / 2);
        let asc_bytes = m_total * (k / 16);
        let mut av = vec![0u8; a_bytes + asc_bytes];
        for (i, b) in av.iter_mut().enumerate() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let byte = (s >> 33) as u8;
            *b = if i >= a_bytes { byte & 0x6f } else { byte };
        }
        let a = client.create_from_slice(&av[..a_bytes]);
        let a_sc = client.create_from_slice(&av[a_bytes..]);
        drop(av);

        // How many warps an SM can actually put to work. This part gives an SM
        // 1536 threads and at most 24 cubes, and a decode block is ONE m tile,
        // so `planes - 1` of every cube's planes have no `mma` to do: the
        // baseline exits them and the staged arm keeps them for the fill. It is
        // the axis the numbers below turn out to move with, so it is printed.
        let cubes_sm = (1536 / (32 * planes)).min(24);
        println!(
            "  -- {planes} planes/cube: m_total {m_total} ({rows_real} real), {blocks} blocks, nrep {nrep}, {cubes_sm} cubes/SM, {} working warps/SM",
            cubes_sm
        );

        let (bb, bf) = best_first(
            || {
                let t0 = Instant::now();
                let o = fp4_linear_grouped_launch_as::<f32, Rt>(
                    &client, &a, &a_sc, &wmap, wmap_bytes, &blk, &off_h, &sc2_h, experts, m_total,
                    k, n,
                );
                let _ = cubecl::future::block_on(client.sync());
                let dt = t0.elapsed().as_secs_f64();
                drop(o);
                dt
            },
            reps,
        );
        let g = report(
            &format!("baseline  B out of global, {planes} planes"),
            bb,
            bf,
        );
        if g > best_base.0 {
            best_base = (g, format!("{planes} planes"));
        }

        for &kc in &kc_list {
            if (k / 64) % kc != 0 {
                continue;
            }
            for &pad in &[4usize, 0usize] {
                let staged_rows = 8 * nrep;
                let smem = staged_rows * (kc * 8 + pad) * 4 + staged_rows * (kc + 1) * 4;
                let (sb, sf) = best_first(
                    || {
                        let t0 = Instant::now();
                        let o = fp4_linear_grouped_smem_launch_tuned::<f32, Rt>(
                            &client, &a, &a_sc, &wmap, wmap_bytes, &blk, &off_h, &sc2_h, experts,
                            m_total, k, n, kc, pad,
                        );
                        let _ = cubecl::future::block_on(client.sync());
                        let dt = t0.elapsed().as_secs_f64();
                        drop(o);
                        dt
                    },
                    reps,
                );
                let tag = format!("staged    kc={kc} pad={pad}, {planes} planes  [{smem} B smem]");
                let g = report(&tag, sb, sf);
                if g > best_smem.0 {
                    best_smem = (g, format!("kc={kc} pad={pad} {planes} planes"));
                    checked = Some((planes, kc, pad));
                }
            }
        }
        println!();
    }

    println!(
        "  best baseline {:.1} GB/s ({}), best staged {:.1} GB/s ({}): {:.3}x",
        best_base.0,
        best_base.1,
        best_smem.0,
        best_smem.1,
        best_smem.0 / best_base.0
    );

    // ---- the bit comparison, at the winning configuration ------------------
    //
    // The staged kernel changes WHERE the B fragment is read from and nothing
    // else -- the same `execute_scaled` calls, in the same k order, on the same
    // operands -- so the two arms must agree BIT FOR BIT. Anything less is a
    // defect in the staging, not a rounding difference.
    let (planes, kc, pad) = checked.expect("no staged configuration ran");
    let plan = RowPlan::build(toks.iter(), experts * rows, planes);
    let m_total = plan.m_total();
    let rows_real = plan.rows_real();
    let blocks = plan.blk_slot.len();
    let blk = BlockPlanDev {
        slot: client.create_from_slice(&le32(&plan.blk_slot)),
        tile0: client.create_from_slice(&le32(&plan.blk_tile0)),
        cnt: client.create_from_slice(&le32(&plan.blk_cnt)),
        blocks,
        planes,
        rows_real,
    };
    let a_bytes = m_total * (k / 2);
    let asc_bytes = m_total * (k / 16);
    let mut av = vec![0u8; a_bytes + asc_bytes];
    for (i, b) in av.iter_mut().enumerate() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let byte = (s >> 33) as u8;
        *b = if i >= a_bytes { byte & 0x6f } else { byte };
    }
    let a = client.create_from_slice(&av[..a_bytes]);
    let a_sc = client.create_from_slice(&av[a_bytes..]);
    drop(av);
    let o_base = fp4_linear_grouped_launch_as::<f32, Rt>(
        &client, &a, &a_sc, &wmap, wmap_bytes, &blk, &off_h, &sc2_h, experts, m_total, k, n,
    );
    let o_smem = fp4_linear_grouped_smem_launch_tuned::<f32, Rt>(
        &client, &a, &a_sc, &wmap, wmap_bytes, &blk, &off_h, &sc2_h, experts, m_total, k, n, kc,
        pad,
    );
    let vb = client.read_one(o_base).expect("read the baseline output");
    let vs = client.read_one(o_smem).expect("read the staged output");
    assert_eq!(vb.len(), vs.len(), "the two arms disagree on output size");
    let diff = vb.iter().zip(vs.iter()).filter(|(x, y)| x != y).count();
    println!(
        "  bit equality at kc={kc} pad={pad} {planes} planes: {} of {} output bytes differ",
        diff,
        vb.len()
    );
    assert_eq!(
        diff, 0,
        "the staged arm is NOT bit-identical to the baseline"
    );
    println!("  OK: bit-identical");
}
