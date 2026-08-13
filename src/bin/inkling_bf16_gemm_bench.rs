//! `inkling_bf16_gemm_bench` — [`bf16_linear`] at the shapes a decode step
//! actually issues, timed with a sync.
//!
//! The kernel beats itself by 1.75x on different inputs — 89-100 GB/s on the
//! shared and dense projections against 175 GB/s on the unembedding, same
//! instruction, same weights-in-BF16 — and a difference that large between two
//! calls to one kernel is a property of the access pattern, not of the weights.
//! Reading it out of a full forward means paying a 50 s pile open and then
//! disentangling it from eighteen other things; this issues the same launches
//! against scratch buffers of the same shapes and syncs on the result.
//!
//! ## What is measured
//!
//! Wall time around `iters` launches, synchronised by reading one element back
//! at the end — an unsynchronised timer measures the enqueue, not the work. The
//! quoted bandwidth is `n * k * 2` bytes of WEIGHT per call over that time,
//! which is the figure the forward's own report uses, and it is a lower bound
//! on what the kernel moves: every cube also re-reads all of A, so at
//! `m_pad = 16` and `k = 4096` there are another 128 KiB per cube going through
//! L2 that this number does not count.
//!
//! On this box DRAM tops out near 273 GB/s, so a bucket implying more than that
//! is measuring cache, not memory.
//!
//! ## Read the small shapes with suspicion — and the recorded negative
//!
//! Looping one weight buffer makes anything under a few tens of MB L2-resident.
//! `shared gate+up` is 16.8 MB and reports 121-191 GB/s here against the 77 the
//! forward sees, because in a real pass 4 GiB of other weights cycle through
//! between two touches of the same one. So exactly the shapes with the least
//! grid parallelism — the ones an access-pattern change is aimed at — are the
//! ones this harness hands a warm cache.
//!
//! That is not hypothetical. Two attempts at the k loop were measured with this
//! binary and then end to end, and the two disagreed:
//!
//!   * **Two accumulator chains** over even and odd k tiles, all six loads
//!     issued before either `mma`, to break the dependency and double the
//!     requests one warp has outstanding. Here: a repeatable **-10.4%** on the
//!     two 4096x4096 attention projections (429.7/429.5 -> 385.1/384.2 us),
//!     flat on everything else, and consistently **+2% worse** on the
//!     unembedding, whose 25128 cubes already saturate. End to end, three
//!     interleaved pairs: **-2.3, -3.5, -3.5 ms, i.e. slower**, median 104.2 ->
//!     106.9. The extra live registers cost more on the many small
//!     DRAM-bound GEMMs than the extra parallelism won on the two large ones,
//!     and this harness could not see that because it had cached the small ones.
//!   * **`execute_inplace` alone**, dropping the copy-back with no extra live
//!     registers: end to end -2.60 and -0.10 ms over two pairs, mins 100.9
//!     against 100.5. Inside the noise, no benefit.
//!
//! Neither landed. The grid was already known to be a weak, non-monotone knob
//! (256 cubes 77 GB/s, 1024 96, 2048 89, 25128 175); per-warp parallelism is now
//! measured too, and it is negative. What has NOT been tried is cutting the A
//! traffic: every cube re-reads all of A, so at 256 cubes the kernel moves
//! 32 MB of activation against 16.8 MB of weight, and a cube covering several
//! n tiles would load A once and reuse it. That needs several accumulator
//! arrays, which is what `MmaDefinition::execute` taking a whole `Array` makes
//! awkward, and it is the next thing to try rather than the grid.
//!
//! Build: `--features inkling-cuda,cuda-backend`
//! Run:   `inkling_bf16_gemm_bench [iters]`

use anyhow::Result;
use cubecl::prelude::*;
use half::bf16;

use mary::models::inkling::bf16gemm::{bf16_linear_launch, KTILE, MTILE, NTILE};

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
    let client = Rt::client(&Default::default());
    let m_pad = MTILE;

    // Every BF16 GEMM a 20-layer decode step issues, by shape. `calls` is how
    // many of each one pass makes, so the last column is what the pass pays.
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

    println!("=== bf16_linear, m_pad {m_pad}, {iters} iters each, synchronised ===");
    println!(
        "  {:<20} {:>7} {:>8} {:>7} {:>10} {:>9} {:>7} {:>10}",
        "shape", "k", "n", "cubes", "us/call", "GB/s", "calls", "ms/pass"
    );

    let mut total = 0f64;
    for &(name, k, n, calls) in shapes {
        assert_eq!(k % KTILE, 0, "{name}: k {k} does not tile");
        assert_eq!(n % NTILE, 0, "{name}: n {n} does not tile");
        let a = client.create_from_slice(&slab(m_pad * k, 0.3));
        let b = client.create_from_slice(&slab(n * k, 1.7));

        // Warm: the first launch of a shape compiles it.
        for _ in 0..3 {
            let _ = bf16_linear_launch(&client, &a, &b, m_pad, k, n);
        }
        let warm = bf16_linear_launch(&client, &a, &b, m_pad, k, n);
        let _ = client.read_one(warm);

        let t0 = std::time::Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(bf16_linear_launch(&client, &a, &b, m_pad, k, n));
        }
        // The sync. Without it this times the enqueue.
        let _ = client.read_one(last.expect("iters > 0"));
        let secs = t0.elapsed().as_secs_f64() / iters as f64;

        let bytes = (n * k * 2) as f64;
        let gbs = bytes / secs / 1e9;
        let ms_pass = secs * 1e3 * calls as f64;
        total += ms_pass;
        println!(
            "  {name} {k:>7} {n:>8} {:>7} {:>10.1} {gbs:>9.1} {calls:>7} {ms_pass:>10.2}",
            n / NTILE,
            secs * 1e6
        );
    }
    println!("  {:<20} {:>53} {:>17.2}", "TOTAL", "", total);
    Ok(())
}
