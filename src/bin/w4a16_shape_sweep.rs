//! Why the SAME hand `w4a16_linear` kernel reads ~98 GB/s at the head shape and
//! ~64 GB/s at the sink shapes.
//!
//! The head is ONE GEMM at `n = 201024, k = 4096`; the sinks are ~42 small
//! per-layer GEMMs at `n = 8192, k = 4096` (gate_up) and `n = 4096, k = 2048`
//! (down). `w4a16_linear_launch` puts N on grid y at `NTILE = 8` and M on grid
//! x at `MTILE = 16`, with `CubeDim = 32` — ONE WARP per output tile. So at
//! decode (`m_pad = 16`, one m-tile) the launch is exactly `n / 8` warps:
//!
//!   head    n = 201024  ->  25128 cubes
//!   gate_up n =   8192  ->   1024 cubes
//!   down    n =   4096  ->    512 cubes
//!
//! against 48 SMs. That is the shape difference stated as a number, and this
//! binary is the four probes that decide whether it is the CAUSE.
//!
//! * `n` sweep at fixed `k`  — cubes vary, bytes vary with them.
//! * `k` sweep at fixed `n`  — **cubes held FIXED at 512**, bytes vary. This is
//!   the discriminator. A fixed per-launch overhead `t = t0 + B/BW` must show
//!   GB/s CLIMBING as `k` grows (the same `t0` amortised over more bytes). A
//!   concurrency limit must show it FLAT (512 warps pull what 512 warps pull,
//!   whatever the trip count).
//! * `m_pad` sweep at fixed `n, k` — cubes vary, weight bytes held fixed.
//! * a null launch at the same grids, and a fused-vs-split run of the REAL sink
//!   shapes: 42 launches against one launch of identical total cubes and bytes.
//!
//! Every store is sentinel-guarded so no probe charges write traffic to a read
//! figure. Min-of-N warm launches, launch + sync, no host readback.

use std::time::Instant;

use cubecl::future;
use cubecl::prelude::*;
use mary::models::inkling::w4a16gemm::w4a16_linear_launch;

type Rt = cubecl::cuda::CudaRuntime;

/// Bytes the weight side of a `[n, k]` W4A16 operand occupies: `k/2` packed
/// code bytes plus `k/16` E4M3 scale bytes per row.
fn table_bytes(n: usize, k: usize) -> usize {
    n * (k / 8) * 4 + n * (k / 16)
}

/// A launch that reads nothing and writes nothing — the fixed cost of getting a
/// grid onto the device, and nothing else.
#[cube(launch)]
pub fn null_kernel(out: &mut Tensor<u32>) {
    if ABSOLUTE_POS == u32::new(0xFFFF_FFFFi64) {
        out[0] = ABSOLUTE_POS;
    }
}

struct Timed {
    ms: f64,
    gbs: f64,
    cubes: u32,
}

/// Time `w4a16_linear_launch` at one shape. Allocates its own operands, warms
/// twice, keeps the min of the rest.
fn time_shape(client: &ComputeClient<Rt>, m_pad: usize, k: usize, n: usize, iters: usize) -> Timed {
    let bytes = table_bytes(n, k);
    let a = client.empty(m_pad * k * 2);
    let b = client.empty(n * (k / 8) * 4);
    let b_sc = client.empty(n * (k / 16));
    let mut best = f64::MAX;
    for i in 0..iters {
        let t0 = Instant::now();
        let out = w4a16_linear_launch::<Rt>(client, &a, &b, &b_sc, m_pad, k, n, 1.0);
        let _ = future::block_on(client.sync());
        let dt = t0.elapsed().as_secs_f64();
        drop(out);
        if i >= 2 {
            best = best.min(dt);
        }
    }
    Timed {
        ms: best * 1e3,
        gbs: bytes as f64 / best / 1e9,
        cubes: ((m_pad / 16) * (n / 8)) as u32,
    }
}

fn main() {
    let client = Rt::client(&Default::default());
    let iters: usize = std::env::var("INK_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    println!("=== w4a16_linear: one 32-thread cube per (m_tile 16, n_tile 8) ===");
    println!("48 SMs on GB10. cubes = (m_pad/16) * (n/8).\n");

    // ---- 1. n sweep at k = 4096, m_pad = 16 --------------------------------
    // Cubes and bytes move together. This is the curve: where does 64 become 98?
    println!("--- n sweep   (k=4096, m_pad=16)  cubes = n/8 ---");
    println!(
        "{:>8}  {:>8}  {:>10}  {:>9}  {:>8}",
        "n", "cubes", "bytes MiB", "ms", "GB/s"
    );
    for n in [
        512usize, 1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768,
        49152, 65536, 98304, 131072, 201024,
    ] {
        let t = time_shape(&client, 16, 4096, n, iters);
        println!(
            "{:>8}  {:>8}  {:>10.2}  {:>9.4}  {:>8.1}",
            n,
            t.cubes,
            table_bytes(n, 4096) as f64 / (1u64 << 20) as f64,
            t.ms,
            t.gbs
        );
    }

    // ---- 2. k sweep at n = 4096, m_pad = 16 --------------------------------
    // THE DISCRIMINATOR. 512 cubes at every row; only the k-trip count moves.
    println!("\n--- k sweep   (n=4096, m_pad=16)  cubes FIXED at 512 ---");
    println!(
        "{:>8}  {:>8}  {:>10}  {:>9}  {:>8}",
        "k", "cubes", "bytes MiB", "ms", "GB/s"
    );
    for k in [1024usize, 2048, 4096, 8192, 16384, 32768, 65536] {
        let t = time_shape(&client, 16, k, 4096, iters);
        println!(
            "{:>8}  {:>8}  {:>10.2}  {:>9.4}  {:>8.1}",
            k,
            t.cubes,
            table_bytes(4096, k) as f64 / (1u64 << 20) as f64,
            t.ms,
            t.gbs
        );
    }

    // Same at the gate_up n, so the discriminator is not a property of n=4096.
    println!("\n--- k sweep   (n=8192, m_pad=16)  cubes FIXED at 1024 ---");
    println!(
        "{:>8}  {:>8}  {:>10}  {:>9}  {:>8}",
        "k", "cubes", "bytes MiB", "ms", "GB/s"
    );
    for k in [1024usize, 2048, 4096, 8192, 16384, 32768] {
        let t = time_shape(&client, 16, k, 8192, iters);
        println!(
            "{:>8}  {:>8}  {:>10.2}  {:>9.4}  {:>8.1}",
            k,
            t.cubes,
            table_bytes(8192, k) as f64 / (1u64 << 20) as f64,
            t.ms,
            t.gbs
        );
    }

    // ---- 3. m_pad sweep at n = 4096, k = 4096 ------------------------------
    // Cubes rise; the weight table does NOT. GB/s below counts the weight only,
    // so a rise means the extra cubes are being served without extra DRAM.
    println!(
        "\n--- m_pad sweep  (n=4096, k=4096)  weight bytes FIXED at {:.2} MiB ---",
        table_bytes(4096, 4096) as f64 / (1u64 << 20) as f64
    );
    println!(
        "{:>8}  {:>8}  {:>9}  {:>10}",
        "m_pad", "cubes", "ms", "GB/s wt"
    );
    for m in [16usize, 32, 64, 128, 256, 512] {
        let t = time_shape(&client, m, 4096, 4096, iters);
        println!("{:>8}  {:>8}  {:>9.4}  {:>10.1}", m, t.cubes, t.ms, t.gbs);
    }

    // ---- 4. fixed per-launch cost ------------------------------------------
    // Back-to-back null launches with ONE sync at the end: the amortised cost of
    // a launch with no memory traffic under it, which is what candidate 2 needs.
    println!("\n--- null launch cost (no traffic) ---");
    let dst = client.empty(4096);
    for cubes in [512u32, 1024, 25128] {
        let reps = 200usize;
        let mut best = f64::MAX;
        for i in 0..6 {
            let t0 = Instant::now();
            for _ in 0..reps {
                unsafe {
                    null_kernel::launch::<Rt>(
                        &client,
                        CubeCount::Static(1, cubes, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(dst.clone(), [1].into(), [1024].into()),
                    )
                };
            }
            let _ = future::block_on(client.sync());
            let dt = t0.elapsed().as_secs_f64();
            if i >= 2 {
                best = best.min(dt);
            }
        }
        println!(
            "  grid y = {:>6} cubes : {:>7.2} us per launch  ({reps} back to back, one sync)",
            cubes,
            best / reps as f64 * 1e6
        );
    }
    // And the output allocation the GEMM launcher does per call.
    {
        let mut best = f64::MAX;
        for i in 0..8 {
            let t0 = Instant::now();
            let h = client.empty(16 * 8192 * 4);
            let _ = future::block_on(client.sync());
            let dt = t0.elapsed().as_secs_f64();
            drop(h);
            if i >= 2 {
                best = best.min(dt);
            }
        }
        println!(
            "  out alloc + sync (m_pad=16, n=8192) : {:>7.2} us",
            best * 1e6
        );
    }

    // ---- 5. the real sink pass: 42 split launches vs 1 fused ---------------
    // Identical total cubes and identical total bytes. If splitting is the cost,
    // the fused row is faster by exactly that much.
    println!("\n--- real sink shapes: split vs fused (same bytes, same total cubes) ---");
    let n_moe = 14usize;
    for (label, per_n, k, count) in [
        ("gate_up [8192,4096]", 8192usize, 4096usize, n_moe),
        ("down    [4096,2048]", 4096usize, 2048usize, n_moe * 2),
    ] {
        // split: `count` independent launches of `per_n`.
        let a = client.empty(16 * k * 2);
        let bs: Vec<_> = (0..count)
            .map(|_| {
                (
                    client.empty(per_n * (k / 8) * 4),
                    client.empty(per_n * (k / 16)),
                )
            })
            .collect();
        let mut split = f64::MAX;
        for i in 0..6 {
            let t0 = Instant::now();
            let outs: Vec<_> = bs
                .iter()
                .map(|(b, sc)| w4a16_linear_launch::<Rt>(&client, &a, b, sc, 16, k, per_n, 1.0))
                .collect();
            let _ = future::block_on(client.sync());
            let dt = t0.elapsed().as_secs_f64();
            drop(outs);
            if i >= 2 {
                split = split.min(dt);
            }
        }
        drop(bs);
        // fused: one launch, n = per_n * count.
        let big_n = per_n * count;
        let t = time_shape(&client, 16, k, big_n, 6);
        let bytes = table_bytes(per_n, k) * count;
        println!(
            "{label}  x{count}   bytes {:.1} MiB   cubes {:>6}",
            bytes as f64 / (1u64 << 20) as f64,
            (per_n / 8) * count
        );
        println!(
            "    split  {count} launches : {:>8.4} ms   {:>6.1} GB/s",
            split * 1e3,
            bytes as f64 / split / 1e9
        );
        println!(
            "    fused  1 launch n={big_n:<7}: {:>8.4} ms   {:>6.1} GB/s",
            t.ms, t.gbs
        );
    }

    // ---- 6. the head row, last, for the anchor -----------------------------
    let h = time_shape(&client, 16, 4096, 201024, iters);
    println!(
        "\nhead  n=201024 k=4096 m_pad=16 : {:.4} ms  {:.1} GB/s  ({} cubes)",
        h.ms, h.gbs, h.cubes
    );
}
