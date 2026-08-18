//! `inkling_bf16_gemm_bench` — the plain-BF16 GEMM lane at the shapes the
//! forward issues, every candidate kernel, timed with a sync.
//!
//! ## What is measured
//!
//! Wall time around `iters` launches, synchronised by reading one element back
//! at the end — an unsynchronised timer measures the enqueue, not the work.
//! Reported as TFLOP/s (`2 m n k` over the time) and as GB/s of WEIGHT
//! (`n k 2` bytes), because the same shape is compute-bound at `m = 512` and
//! bandwidth-bound at `m = 16` and one column cannot say both.
//!
//! cuBLAS on this box, measured: 95.9 TFLOP/s at 4096^3. DRAM tops out near
//! 273 GB/s (236 measured), so a bucket implying more than that is measuring
//! cache, not memory.
//!
//! ## Read the small shapes with suspicion — and the recorded negative
//!
//! Looping one weight buffer makes anything under a few tens of MB L2-resident.
//! `shared gate+up` is 16.8 MB and reports well above what the forward sees,
//! because in a real pass 4 GiB of other weights cycle through between two
//! touches of the same one. So exactly the shapes with the least grid
//! parallelism — the ones an access-pattern change is aimed at — are the ones
//! this harness hands a warm cache. Two k-loop changes were measured positive
//! here and negative end to end (two accumulator chains: -10.4% here, +2.5 ms
//! slower end to end; `execute_inplace`: inside the noise). Nothing lands on
//! this binary's word alone.
//!
//! Build: `--features inkling-cuda,cuda-backend`
//! Run:   `inkling_bf16_gemm_bench [iters] [m]`

use anyhow::Result;
use cubecl::prelude::*;
use half::bf16;

use mary::models::inkling::bf16gemm::{
    bf16_linear_launch, try_bf16_linear_cubek_launch, Lane, KTILE, MTILE, NTILE,
};

type Rt = cubecl::cuda::CudaRuntime;

/// A deterministic BF16 slab. Values near unit scale so nothing denormalises.
fn slab(n: usize, seed: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 2);
    for i in 0..n {
        let f = ((i as f32 * 0.017_31 + seed).sin() * 0.5) as f32;
        v.extend_from_slice(&bf16::from_f32(f).to_le_bytes());
    }
    v
}

fn main() -> Result<()> {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let m: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(MTILE);
    let client = Rt::client(&Default::default());
    // The hand lane's grid IS its tiling, so it needs m a multiple of 16. The
    // tuned lanes bounds-check their own tiles, so `m = 1` -- what a decode step
    // actually feeds -- is a shape they can be asked about and it was not.
    let hand_ok = m % MTILE == 0;

    // Every BF16 GEMM a 20-layer node issues, by shape. `calls` is how many of
    // each one pass makes, so the last column is what the pass pays.
    let shapes: &[(&str, usize, usize, usize)] = &[
        // name,                    k,     n,      calls per pass on this node
        ("attn wq            ", 4096, 4096, 20),
        ("attn wk/wv         ", 4096, 1024, 40),
        ("attn wr            ", 4096, 512, 20),
        ("attn wo            ", 4096, 4096, 20),
        ("shared gate+up     ", 4096, 2048, 18),
        ("shared down        ", 2048, 4096, 36),
        ("dense w13          ", 4096, 16384, 2),
        ("dense down         ", 8192, 4096, 2),
        ("unembed            ", 4096, 201024, 1),
    ];

    println!("=== the plain-BF16 lane, m {m}, {iters} iters each, synchronised ===");
    println!("  cuBLAS reference on this box: 95.9 TFLOP/s at 4096^3\n");

    // Per-lane totals over the pass, so the last table answers "which kernel
    // would the forward want" rather than "which won this shape".
    let mut totals: Vec<(Lane, f64, usize)> = Lane::ALL.iter().map(|&l| (l, 0.0, 0)).collect();

    for &(name, k, n, calls) in shapes {
        assert_eq!(k % KTILE, 0, "{name}: k {k} does not tile");
        assert_eq!(n % NTILE, 0, "{name}: n {n} does not tile");
        let a = client.create_from_slice(&slab(m * k, 0.3));
        let b = client.create_from_slice(&slab(n * k, 1.7));

        println!("  {name} k {k:>5} n {n:>7}   ({:.1} MB of weight)", (n * k * 2) as f64 / 1e6);
        // The first lane that runs is the reference the rest are checked against,
        // over row 0 only.
        let mut reference: Option<Vec<f32>> = None;

        for (slot, &lane) in Lane::ALL.iter().enumerate() {
            let launch = |lane: Lane| -> Option<cubecl::server::Handle> {
                if lane == Lane::Hand {
                    if !hand_ok {
                        return None;
                    }
                    Some(bf16_linear_launch(&client, &a, &b, m, k, n))
                } else {
                    try_bf16_linear_cubek_launch(&client, &a, &b, m, k, n, lane).ok()
                }
            };

            // Warm: the first launch of a shape compiles it. A strategy that
            // declines this shape declines it here.
            let mut ok = true;
            for _ in 0..3 {
                if launch(lane).is_none() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                println!("    {:<24} unavailable at this shape", lane.name());
                continue;
            }
            let warm = launch(lane).expect("checked");
            let bytes = client.read_one(warm).expect("read back");
            let got: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Agreement, not bit-equality: these accumulate in different orders
            // and the operands are BF16. A lane that is WRONG is wrong by
            // orders of magnitude (a transposed B, a half-consumed operand), so
            // a relative check against the largest element separates the two
            // without pretending the orders match.
            let rel = match &reference {
                None => {
                    reference = Some(got.clone());
                    0.0f32
                }
                Some(r) => {
                    let scale = r[..n].iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-6);
                    r[..n]
                        .iter()
                        .zip(&got[..n])
                        .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()))
                        / scale
                }
            };

            let t0 = std::time::Instant::now();
            let mut last = None;
            for _ in 0..iters {
                last = launch(lane);
            }
            // The sync. Without it this times the enqueue.
            let _ = client.read_one(last.expect("iters > 0"));
            let secs = t0.elapsed().as_secs_f64() / iters as f64;

            let tflops = 2.0 * (m * n * k) as f64 / secs / 1e12;
            let gbs = (n * k * 2) as f64 / secs / 1e9;
            let ms_pass = secs * 1e3 * calls as f64;
            totals[slot].1 += ms_pass;
            totals[slot].2 += 1;
            println!(
                "    {:<24} {:>9.1} us  {:>7.2} TFLOP/s  {:>7.1} GB/s   rel {:>8.1e}",
                lane.name(),
                secs * 1e6,
                tflops,
                gbs,
                rel
            );
        }
        println!();
    }

    println!("  === ms per pass, summed over the shapes a lane could run ===");
    let mut ranked: Vec<_> = totals.iter().filter(|t| t.2 == shapes.len()).collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (lane, ms, _) in &ranked {
        println!("    {:<24} {ms:>9.2} ms", lane.name());
    }
    let partial: Vec<_> = totals.iter().filter(|t| t.2 > 0 && t.2 < shapes.len()).collect();
    if !partial.is_empty() {
        println!("  (ran only some shapes, not comparable:)");
        for (lane, ms, c) in partial {
            println!("    {:<24} {ms:>9.2} ms over {c}/{} shapes", lane.name(), shapes.len());
        }
    }
    Ok(())
}
