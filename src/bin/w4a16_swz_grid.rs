//! Is the `m16n8k16` permutation worth anything at the SINK grid, and why not?
//!
//! `swz_grid_scaling` asked this of the W4A4 `fp4_linear` pair and closed with
//! a caveat it could not lift: "the permutation MULTIPLIER for w4a16 has not
//! been measured and should not be assumed to be these numbers". This is that
//! measurement. It exists because a real regression was priced against the
//! missing number: the head's 95.9 -> 116.3 GB/s was carried to a shape with
//! 25x fewer cubes and the sign turned over.
//!
//! ## Why not just run `w4a16_swz_probe` at the sink shape
//!
//! Because at a sink shape that probe measures the L2, not the kernel. Its four
//! arms share buffers — arm 3 (`stream ceiling`) reads the SAME `b_row`/`bs_row`
//! the row-major arm reads, immediately before it — and the two plane sets
//! together are 36 MiB against a 24 MiB L2. So the row-major arm runs L2-warm
//! every rep and the swizzled arms run L2-cold every rep, and the gap between
//! them is a cache-residency artifact of the arm ORDER. At the head's 463 MiB
//! nothing is resident and the confound does not exist, which is why the same
//! probe is sound there and not here.
//!
//! This harness removes it three ways, all of them copied from
//! `swz_grid_scaling`, which had already learned them:
//!
//! * **Rotating buffers.** Each arm cycles through enough distinct tables that
//!   the gap between two reads of one clears L2 by a wide margin. Both arms
//!   rotate identically, so neither is the warm one.
//! * **Pipelined launches.** `REPS` launches enqueued back to back and ONE sync
//!   at the end, divided. `w4a16_swz_probe` syncs per launch, and at a sink
//!   shape the host round trip is ~0.12 ms against a ~0.06 ms kernel — the
//!   floor is twice the signal. (Visible in that probe's own output: its
//!   coalesced `stream ceiling` reads 18 MiB at 99 GB/s, a quarter of this
//!   part's bus, because it is timing the round trip.)
//! * **Round-robin over shapes AND arms**, min over rounds, first two rounds
//!   discarded, so a clock ramp or a neighbour lands on every arm equally.
//!
//! Timing only. The two kernels read the same bytes in a different ORDER, so an
//! uninitialised table exercises the identical access pattern; that the two
//! agree to the last bit is `w4a16_swz_probe`'s numerics check, not this one's.
//!
//! `INK_K` sets k (default 4096). `INK_NS` overrides the n sweep as a
//! comma-separated list. `INK_M` sets `m_pad`, which is the warp-count control.
//!
//! # What it measured (GB10, sm_121a, spark-zt, 2026-08-26, box verified idle)
//!
//! FRAMING RULE: `ratio` is row-major time / swizzled time for ONE launch of a
//! `[m_pad, k] x [n, k]^T` product at `m_pad = 16` — the decode case, one
//! m-tile — GB/s over the weight table only. Not per step, not per node.
//!
//! ```text
//!   k=2048     512 cubes   70.7 ->  62.0 GB/s   0.88   <- sink `down`, ACTUAL
//!             1024         92.6 -> 100.5        1.09
//!            25128        116.7 -> 133.0        1.14
//!   k=4096     256         49.0 ->  34.3        0.70
//!              512         77.9 ->  65.7        0.84
//!              768         90.7 ->  93.0        1.03   <- crossover
//!             1024         94.3 -> 106.6        1.13   <- sink `gate_up`, ACTUAL
//!             2048         90.9 -> 113.5        1.25   <- dense `g`/`u`
//!            25128        107.3 -> 132.6        1.24   <- the head
//!   k=16384    512         46.9 ->  58.4        1.25   <- dense `down`, ACTUAL
//!             1024         66.4 ->  95.1        1.43
//!             2048         67.8 ->  98.3        1.45
//! ```
//!
//! The multiplier rises monotonically with cube count, crosses 1.0 near 750
//! cubes (0.65 of a 1152-cube wave), and saturates at a value set by K: ~1.10 at
//! k=2048, ~1.24 at k=4096, ~1.45 at k=16384. `INK_M` isolates the cube count
//! from `n` — same `[4096, 4096]` table, same bytes, only m-tiles varied:
//!
//! ```text
//!   m_pad  16 ->  512 cubes  0.86       m_pad  64 -> 2048 cubes  1.15
//!   m_pad  32 -> 1024 cubes  1.09       m_pad 128 -> 4096 cubes  1.20
//! ```
//!
//! The mechanism, and what it means for each consumer, is written out beside the
//! head's own figure in `w4a16gemm::swizzle_w4a16`'s doc. The short version:
//! row-major's eight-sector burst is an incidental four-k-tile L1 prefetch, the
//! permutation trades it for 8x fewer requests, and which side of that trade
//! pays depends on whether there are enough resident warps to hide the latency
//! the prefetch was covering.

use std::time::Instant;

use cubecl::future;
use cubecl::prelude::*;
use cubecl::server::Handle;
use mary::models::inkling::w4a16gemm::{w4a16_linear_launch, w4a16_linear_swz_launch};

type Rt = cubecl::cuda::CudaRuntime;

const REPS: usize = 20;
const ROUNDS: usize = 6;

/// Weight-side bytes of a `[n, k]` NVFP4 operand: `k/2` codes + `k/16` scales.
fn table_bytes(n: usize, k: usize) -> usize {
    n * (k / 2) + n * (k / 16)
}

struct Arm {
    a: Handle,
    rot: Vec<(Handle, Handle)>,
}

impl Arm {
    fn new(client: &ComputeClient<Rt>, m_pad: usize, k: usize, n: usize) -> Self {
        let bytes = table_bytes(n, k);
        // The GAP between two reads of one buffer must clear the 24 MiB L2 by a
        // wide margin, or the small shapes are quietly served from cache --
        // which is exactly the bias this harness exists to remove.
        let rot = (1 + (256usize << 20).div_ceil(bytes.max(1))).clamp(2, 130);
        let rot = rot.min(((3usize << 30) / bytes.max(1)).max(2));
        Arm {
            // A is BF16 here, not NVFP4: this is the W4A16 lane.
            a: client.empty(m_pad * k * 2),
            rot: (0..rot)
                .map(|_| (client.empty(n * (k / 2)), client.empty(n * (k / 16))))
                .collect(),
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let client = Rt::client(&Default::default());
    let k = env_usize("INK_K", 4096);
    // `INK_M` is the WARP-COUNT control. Cubes are `(m_pad / 16) * (n / 8)`, so
    // raising it multiplies the grid over an UNCHANGED weight table -- which is
    // the one knob that separates "too few warps" from "the wrong n". Not a
    // pure control: it also multiplies A traffic and the output store, and each
    // B byte is then shared by `m_pad / 16` cubes out of L2. The RATIO is what
    // it is for; the levels move for reasons that are not the permutation.
    let m_pad = env_usize("INK_M", 16);

    let ns: Vec<usize> = match std::env::var("INK_NS") {
        Ok(s) => s
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect::<Vec<usize>>(),
        Err(_) => vec![4096, 8192, 16384, 32768, 65536, 131072, 201024],
    };

    println!("=== W4A16: pre-permuted vs row-major, swept across grid size ===");
    println!("both lanes: one 32-thread cube per (m_tile 16, n_tile 8), so cubes = n/8");
    println!(
        "k = {k}, m_pad = {m_pad}, swz load depth = {}; GB/s over the weight table only; min of \
         {ROUNDS} rounds of {REPS} pipelined launches\n",
        mary::models::inkling::w4a16gemm::swz_unroll(),
    );
    println!(
        "{:>8} {:>7} {:>9} {:>4} {:>10} {:>10} {:>10} {:>10} {:>7}",
        "n", "cubes", "MiB", "rot", "row ms", "swz ms", "row GB/s", "swz GB/s", "ratio"
    );

    let arms: Vec<Arm> = ns.iter().map(|&n| Arm::new(&client, m_pad, k, n)).collect();
    let mut base = vec![f64::MAX; ns.len()];
    let mut swz = vec![f64::MAX; ns.len()];

    for r in 0..ROUNDS + 2 {
        for (i, (&n, arm)) in ns.iter().zip(&arms).enumerate() {
            let t0 = Instant::now();
            for j in 0..REPS {
                let (b, sc) = &arm.rot[j % arm.rot.len()];
                let o = w4a16_linear_launch::<Rt>(&client, &arm.a, b, sc, m_pad, k, n, 1.0);
                drop(o);
            }
            let _ = future::block_on(client.sync());
            let db = t0.elapsed().as_secs_f64() / REPS as f64;

            let t1 = Instant::now();
            for j in 0..REPS {
                let (b, sc) = &arm.rot[j % arm.rot.len()];
                let o =
                    w4a16_linear_swz_launch::<Rt>(&client, &arm.a, b, sc, m_pad, k, n, true, 1.0);
                drop(o);
            }
            let _ = future::block_on(client.sync());
            let ds = t1.elapsed().as_secs_f64() / REPS as f64;

            if r >= 2 {
                base[i] = base[i].min(db);
                swz[i] = swz[i].min(ds);
            }
        }
    }

    for (i, &n) in ns.iter().enumerate() {
        let bytes = table_bytes(n, k);
        println!(
            "{:>8} {:>7} {:>9.2} {:>4} {:>10.4} {:>10.4} {:>10.1} {:>10.1} {:>7.2}",
            n,
            (m_pad / 16) * (n / 8),
            bytes as f64 / (1u64 << 20) as f64,
            arms[i].rot.len(),
            base[i] * 1e3,
            swz[i] * 1e3,
            bytes as f64 / base[i] / 1e9,
            bytes as f64 / swz[i] / 1e9,
            base[i] / swz[i]
        );
    }
    println!(
        "\nframing: per LAUNCH of one [{m_pad}, {k}] x [n, {k}]^T product, GB/s over the weight \
         planes only, {REPS} pipelined launches over rotating buffers divided by {REPS}, min of \
         {ROUNDS} rounds after 2 discarded, arms round-robined, one GB10 box. Not a step figure \
         and not a two-node figure."
    );
    println!("ratio > 1 -> the permutation is a win at that cube count; < 1 -> a loss.");
}
