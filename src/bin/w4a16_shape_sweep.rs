//! Why the SAME hand `w4a16_linear` kernel reads ~98 GB/s at the head shape and
//! ~64 GB/s at the sink shapes.
//!
//! The head is ONE GEMM at `n = 201024, k = 4096`; the sinks are ~42 small
//! per-layer GEMMs at `n = 8192, k = 4096` (gate_up) and `n = 4096, k = 2048`
//! (down). `w4a16_linear_launch` puts N on grid y at `NTILE = 8` and M on grid
//! x at `MTILE = 16`, with `CubeDim = 32` — ONE WARP per output tile. So at
//! decode (`m_pad = 16`, one m-tile) the launch is exactly `n / 8` warps:
//!
//!   head    n = 201024  ->  25128 cubes      (523 per SM, ~22 full waves)
//!   gate_up n =   8192  ->   1024 cubes      (21 per SM, under one wave)
//!   down    n =   4096  ->    512 cubes      (11 per SM, under half a wave)
//!
//! against 48 SMs. That is the shape difference stated as a number.
//!
//! # Instrument
//!
//! Two things contaminated the first version of this sweep and both are fixed
//! here.
//!
//! * **The output allocation.** `w4a16_linear_launch` calls `client.empty` for
//!   its own `out` on every call, and that measures ~20 us — which is 15% of a
//!   9 MiB sink GEMM and invisible at the head. [`launch_into`] is the same
//!   launch against a PRE-ALLOCATED output, so nothing in a timing loop
//!   allocates.
//! * **Drift.** Sweeping shapes in sequential blocks lets a clock ramp or a
//!   neighbouring process land entirely on one end of a curve. Every sweep here
//!   is ROUND-ROBIN: round `r` touches every shape once, and each shape keeps
//!   the min over rounds.
//!
//! Each measurement is reported two ways, and the pair is the point:
//!
//! * `pipe` — `REPS` launches back to back with ONE sync at the end, divided by
//!   `REPS`. Host-side launch cost overlaps the device, so this is the device's
//!   steady-state cost for that grid.
//! * `solo` — one launch, one sync. `solo - pipe` is what a launch costs that a
//!   pipelined launch does not pay.
//!
//! # Probes
//!
//! * `n` sweep at fixed `k` — cubes vary, bytes vary with them. The curve.
//! * `k` sweep at fixed `n` — **cubes held FIXED**, bytes vary. The
//!   discriminator between a fixed per-launch overhead (`t = t0 + B/BW`, which
//!   must show GB/s CLIMBING as `k` grows) and a concurrency limit (flat).
//! * split vs fused on the REAL sink shapes: 14 and 28 launches against one
//!   launch of identical total cubes and identical total bytes.
//!
//! # What it measured (GB10, sm_121a, 2026-08-25; two runs, 4-5 sibling
//! # processes on the GPU throughout)
//!
//! FRAMING RULE for every figure below: GB/s is over the WEIGHT TABLE ONLY
//! (packed codes + E4M3 block scales) of ONE GEMM, at `m_pad = 16` — the decode
//! case, one m-tile — timed as `REPS` pipelined launches over rotating buffers
//! and divided. It is not per decode step and not per pass. Absolute rates moved
//! ~10% between the two runs because the box was shared; every RATIO below
//! reproduced, and ratios are what the conclusions rest on.
//!
//! The rate is a smooth saturating function of CUBE COUNT and of almost nothing
//! else. At `k = 2048`: 384 cubes 62, 768 cubes 86, 1152 cubes 94, 2304 cubes
//! 102, 6144 cubes 101-110, 24576 cubes 107-120 GB/s. No sawtooth at any
//! multiple of a resident wave (~1152 single-warp cubes on 48 SMs) and still
//! climbing at 21 waves, so this is NOT wave quantisation — it is a fill.
//!
//! The `split-k emulation` rows are the proof, because they hold the bytes
//! EXACTLY constant and move only the grid:
//!
//!   down    4.5 MiB:   512 cubes  67-73 GB/s  ->  4096 cubes  97-110  (1.34-1.57x)
//!   gate_up  18 MiB:  1024 cubes  78-88 GB/s  ->  8192 cubes 113-118  (1.34-1.46x)
//!
//! Identical bytes, up to 1.57x less time. That is the exact converse of the
//! result this investigation started from.
//!
//! Two candidates died here:
//!
//! * **Per-launch overhead is not it.** At FIXED cube count the rate does not
//!   climb with `k` — 512 cubes reads 58, 71, 74, 58, 46, 45, 45 GB/s as `k`
//!   goes 1024..65536, i.e. flat then falling. A fixed `t0` would force it to
//!   climb toward the asymptote over a 64x range of bytes. Independently, the
//!   LIBRARY arm — which calls `client.empty` on every one of the 42 calls, as
//!   the real stack does — is not slower than the pre-allocated arm at all; the
//!   allocator is pooled and the tax is at or below zero.
//! * **The host launch path is real but small.** `w4a16_linear` at 8 cubes and
//!   4.5 KiB, where no device work is left to hide behind, costs 23.68-23.76 us
//!   per launch (very stable across runs) against the null kernel's 7.2 us —
//!   four tensor args instead of one. 42 of those is ~1.0 ms, and it is hidden
//!   whenever launches are pipelined, which is why `split` and `fused` differ by
//!   1.5x rather than by 42 x 24 us.

use std::time::Instant;

use cubecl::e4m3;
use cubecl::future;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::bf16;
use mary::models::inkling::w4a16gemm::{
    CODES_PER_WORD, GROUP, MTILE, NTILE, w4a16_linear, w4a16_linear_launch,
};

type Rt = cubecl::cuda::CudaRuntime;

/// Repeats inside one pipelined burst.
const REPS: usize = 20;
/// Round-robin rounds over a sweep; each shape keeps its min.
const ROUNDS: usize = 6;

/// Bytes the weight side of a `[n, k]` W4A16 operand occupies: `k/2` packed
/// code bytes plus `k/16` E4M3 scale bytes per row.
fn table_bytes(n: usize, k: usize) -> usize {
    n * (k / 8) * 4 + n * (k / 16)
}

/// [`w4a16_linear_launch`] with the output handed IN.
///
/// Byte-identical launch to the library's, minus the `client.empty` — which is
/// the whole reason this exists. See the module header.
#[allow(clippy::too_many_arguments)]
fn launch_into(
    client: &ComputeClient<Rt>,
    a: &Handle,
    b: &Handle,
    b_sc: &Handle,
    out: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
) {
    let vs = 32 / bf16::cube_type().size_bits();
    let wpr = k / CODES_PER_WORD;
    let spr = k / GROUP;
    unsafe {
        w4a16_linear::launch::<bf16, e4m3, Rt>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            TensorArg::from_raw_parts(a.clone(), [k, 1].into(), [m_pad, k].into()),
            TensorArg::from_raw_parts(b.clone(), [wpr, 1].into(), [n, wpr].into()),
            TensorArg::from_raw_parts(b_sc.clone(), [spr, 1].into(), [n, spr].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_pad, n].into()),
            k,
            n,
            1.0f32,
        )
    };
}

/// Operands for one shape, allocated once and reused across every round.
///
/// `rot` holds SEVERAL distinct weight tables and the repeats cycle through
/// them. Without that, `REPS` back-to-back launches over one buffer let any
/// working set under the 24 MiB L2 be served from L2 on repeats 2..REPS — which
/// flatters exactly the small shapes this is trying to judge, and is why v2 read
/// a 9 MiB table at 100 GB/s and a 36 MiB one at 45. Enough copies are made to
/// exceed 2x L2, so every launch reads memory the previous one evicted.
struct Shape {
    m_pad: usize,
    k: usize,
    n: usize,
    a: Handle,
    rot: Vec<(Handle, Handle)>,
    out: Handle,
    pipe: f64,
    solo: f64,
}

impl Shape {
    fn new(client: &ComputeClient<Rt>, m_pad: usize, k: usize, n: usize) -> Self {
        let bytes = table_bytes(n, k);
        // What has to exceed L2 is the GAP between two reads of the SAME buffer,
        // which is (rot - 1) * bytes, not rot * bytes. v3 targeted 48 MiB of
        // total rotation and so left a 9 MiB table with a 45 MiB gap -- under 2x
        // L2, i.e. still partly resident, and the inflation lands on exactly the
        // small shapes under judgement. In situ there is NO reuse at all: GBs of
        // model stream past between one pass's sink read and the next. 256 MiB
        // of gap is ~11x L2 and models that.
        let rot = (1 + (256usize << 20).div_ceil(bytes.max(1))).clamp(2, 130);
        // ... but never allocate more than 4 GiB for one shape.
        let rot = rot.min(((4usize << 30) / bytes.max(1)).max(1));
        Shape {
            m_pad,
            k,
            n,
            a: client.empty(m_pad * k * 2),
            rot: (0..rot)
                .map(|_| (client.empty(n * (k / 8) * 4), client.empty(n * (k / 16))))
                .collect(),
            out: client.empty(m_pad * n * 4),
            pipe: f64::MAX,
            solo: f64::MAX,
        }
    }
    fn cubes(&self) -> usize {
        (self.m_pad / MTILE) * (self.n / NTILE)
    }
    fn bytes(&self) -> usize {
        table_bytes(self.n, self.k)
    }
    fn round(&mut self, client: &ComputeClient<Rt>, keep: bool) {
        let t0 = Instant::now();
        for r in 0..REPS {
            let (b, sc) = &self.rot[r % self.rot.len()];
            launch_into(
                client, &self.a, b, sc, &self.out, self.m_pad, self.k, self.n,
            );
        }
        let _ = future::block_on(client.sync());
        let dt = t0.elapsed().as_secs_f64() / REPS as f64;
        let (b0, sc0) = &self.rot[0];
        let t1 = Instant::now();
        launch_into(
            client, &self.a, b0, sc0, &self.out, self.m_pad, self.k, self.n,
        );
        let _ = future::block_on(client.sync());
        let ds = t1.elapsed().as_secs_f64();
        if keep {
            self.pipe = self.pipe.min(dt);
            self.solo = self.solo.min(ds);
        }
    }
}

/// Run a set of shapes round-robin and print one table.
fn sweep(client: &ComputeClient<Rt>, title: &str, mut shapes: Vec<Shape>) -> Vec<Shape> {
    println!("\n--- {title} ---");
    // Two unkept rounds: compile every kernel and let the clock settle before
    // any shape is charged for it.
    for r in 0..ROUNDS + 2 {
        for s in shapes.iter_mut() {
            s.round(client, r >= 2);
        }
    }
    println!(
        "{:>7} {:>7} {:>7} {:>7} {:>9} {:>4} {:>9} {:>9} {:>8} {:>8}",
        "m_pad", "k", "n", "cubes", "MiB", "rot", "pipe ms", "solo ms", "GB/s", "ovh us"
    );
    for s in &shapes {
        println!(
            "{:>7} {:>7} {:>7} {:>7} {:>9.2} {:>4} {:>9.4} {:>9.4} {:>8.1} {:>8.1}",
            s.m_pad,
            s.k,
            s.n,
            s.cubes(),
            s.bytes() as f64 / (1u64 << 20) as f64,
            s.rot.len(),
            s.pipe * 1e3,
            s.solo * 1e3,
            s.bytes() as f64 / s.pipe / 1e9,
            (s.solo - s.pipe) * 1e6
        );
    }
    shapes.clear();
    shapes
}

fn main() {
    let client = Rt::client(&Default::default());
    println!("=== w4a16_linear: one 32-thread cube per (m_tile 16, n_tile 8) ===");
    println!("48 SMs on GB10.  cubes = (m_pad/16) * (n/8).");
    println!("pipe = {REPS} launches, one sync, divided.  solo = 1 launch + 1 sync.");
    println!("GB/s is over the WEIGHT table (codes + E4M3 scales) at pipe.");

    // ---- 1. n sweep at k = 4096, m_pad = 16 --------------------------------
    // Cubes and bytes move together. Where does 64 become 98?
    let ns = [
        1024usize, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152,
        65536, 98304, 131072, 201024,
    ];
    sweep(
        &client,
        "n sweep (k=4096, m_pad=16): cubes = n/8",
        ns.iter()
            .map(|&n| Shape::new(&client, 16, 4096, n))
            .collect(),
    );

    // ---- 2. k sweep, cubes held fixed --------------------------------------
    // THE DISCRIMINATOR. If a fixed per-launch cost t0 were the shortfall, GB/s
    // must climb toward the asymptote as k (and so the bytes) grow at constant
    // grid. If it is a concurrency limit, it must not.
    sweep(
        &client,
        "k sweep (n=4096, m_pad=16): cubes FIXED at 512",
        [1024usize, 2048, 4096, 8192, 16384, 32768, 65536]
            .iter()
            .map(|&k| Shape::new(&client, 16, k, 4096))
            .collect(),
    );
    sweep(
        &client,
        "k sweep (n=8192, m_pad=16): cubes FIXED at 1024",
        [1024usize, 2048, 4096, 8192, 16384, 32768]
            .iter()
            .map(|&k| Shape::new(&client, 16, k, 8192))
            .collect(),
    );
    // And at a grid that DOES fill the device, as the control: here the same
    // k axis should be flat and high, because concurrency is not the binding
    // constraint any more.
    sweep(
        &client,
        "k sweep (n=65536, m_pad=16): cubes FIXED at 8192 (fills the device)",
        [1024usize, 2048, 4096, 8192]
            .iter()
            .map(|&k| Shape::new(&client, 16, k, 65536))
            .collect(),
    );

    // A grid sweep at the DOWN k, where the per-cube trip count is halved and
    // the ramp is therefore a larger share of the launch.
    sweep(
        &client,
        "n sweep (k=2048, m_pad=16): the down GEMM's own k",
        [2048usize, 4096, 8192, 16384, 32768, 65536, 114688]
            .iter()
            .map(|&n| Shape::new(&client, 16, 2048, n))
            .collect(),
    );

    // Fine sweep across the low end at the down k. A pure WAVE-QUANTISATION
    // effect must saturate once the launch covers one resident wave (48 SMs x
    // ~24 single-warp cubes = ~1152) and sawtooth around multiples of it. A
    // latency/occupancy fill must be smooth and keep climbing well past it.
    sweep(
        &client,
        "fine cube sweep (k=2048, m_pad=16): is there wave structure near 1152?",
        [
            384usize, 768, 1152, 1536, 2304, 3072, 4608, 6144, 9216, 12288, 18432, 24576,
        ]
        .iter()
        .map(|&c| Shape::new(&client, 16, 2048, c * 8))
        .collect(),
    );

    // ---- 3. the real sink shapes and the head, side by side ----------------
    sweep(
        &client,
        "the real shapes",
        vec![
            Shape::new(&client, 16, 2048, 4096),   // down
            Shape::new(&client, 16, 4096, 8192),   // gate_up
            Shape::new(&client, 16, 4096, 201024), // head
        ],
    );

    // ---- 3b. what a launch of THIS kernel costs the host ---------------------
    // The n=2048,k=2048 fit implies a fixed ~35 us per launch that is NOT the
    // output allocation (the library arm below prices that at ~0). Either it is
    // an on-device ramp, or it is cubecl's launch path -- which calls
    // cuFuncSetAttribute before EVERY launch and takes four tensor args here
    // against the null kernel's one. This shape is 8 cubes and 4.5 KiB: there is
    // no device work left to hide behind, so whatever this costs is the host.
    {
        let s = Shape::new(&client, 16, 1024, 64);
        let (b0, sc0) = &s.rot[0];
        let reps = 300usize;
        let mut best = f64::MAX;
        for r in 0..8 {
            let t0 = Instant::now();
            for _ in 0..reps {
                launch_into(&client, &s.a, b0, sc0, &s.out, 16, 1024, 64);
            }
            let _ = future::block_on(client.sync());
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            if r >= 2 {
                best = best.min(dt);
            }
        }
        println!(
            "\n--- host launch cost of w4a16_linear itself (n=64, k=1024: 8 cubes, 4.5 KiB) ---"
        );
        println!(
            "  {:.2} us per launch, {reps} back to back, one sync",
            best * 1e6
        );
    }

    // ---- 3c. SPLIT-K, emulated exactly ---------------------------------------
    // If the shortfall is that the grid is too small, the fix is not a better
    // layout -- it is more cubes over the SAME bytes, which is what split-k buys:
    // each cube takes a k-slice and a cheap f32 reduction adds the partials.
    //
    // A split-k of S over [n, k] launches S*n/8 cubes that each walk k/S. That is
    // shape-identical to ONE launch at [S*n, k/S]: same total cubes, same
    // per-cube trip count, byte-for-byte the same table. So these rows price the
    // fix without writing the reduction, and the pairs below hold BYTES EXACTLY
    // constant -- only the grid moves.
    sweep(
        &client,
        "split-k emulation for down [4096,2048]: 4.5 MiB in every row",
        vec![
            Shape::new(&client, 16, 2048, 4096), // S=1, as shipped: 512 cubes
            Shape::new(&client, 16, 1024, 8192), // S=2: 1024 cubes
            Shape::new(&client, 16, 512, 16384), // S=4: 2048 cubes
            Shape::new(&client, 16, 256, 32768), // S=8: 4096 cubes
        ],
    );
    sweep(
        &client,
        "split-k emulation for gate_up [8192,4096]: 18 MiB in every row",
        vec![
            Shape::new(&client, 16, 4096, 8192), // S=1, as shipped: 1024 cubes
            Shape::new(&client, 16, 2048, 16384), // S=2: 2048 cubes
            Shape::new(&client, 16, 1024, 32768), // S=4: 4096 cubes
            Shape::new(&client, 16, 512, 65536), // S=8: 8192 cubes
        ],
    );

    // ---- 4. split vs fused, identical bytes and identical total cubes ------
    // 14 (28) launches of a sink shape against ONE launch covering the same n.
    // Both pipelined, both with pre-allocated outputs, so the only difference
    // is where the kernel boundaries fall.
    println!("\n--- split vs fused (same bytes, same total cubes, no allocation in the loop) ---");
    for (label, per_n, k, count) in [
        ("gate_up [8192,4096]", 8192usize, 4096usize, 14usize),
        ("down    [4096,2048]", 4096usize, 2048usize, 28usize),
    ] {
        let a = client.empty(16 * k * 2);
        let parts: Vec<(Handle, Handle, Handle)> = (0..count)
            .map(|_| {
                (
                    client.empty(per_n * (k / 8) * 4),
                    client.empty(per_n * (k / 16)),
                    client.empty(16 * per_n * 4),
                )
            })
            .collect();
        let big_n = per_n * count;
        let mut fused = Shape::new(&client, 16, k, big_n);
        let bytes = table_bytes(per_n, k) * count;
        let mut split = f64::MAX;
        // The THIRD arm is what the real stack runs: the library launcher, which
        // calls client.empty for its own output on every one of the 42 calls.
        let mut lib = f64::MAX;
        for r in 0..ROUNDS + 2 {
            let t0 = Instant::now();
            for (b, sc, out) in &parts {
                launch_into(&client, &a, b, sc, out, 16, k, per_n);
            }
            let _ = future::block_on(client.sync());
            let dt = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let outs: Vec<Handle> = parts
                .iter()
                .map(|(b, sc, _)| {
                    w4a16_linear_launch::<Rt>(&client, &a, b, sc, 16, k, per_n, 1.0, None)
                })
                .collect();
            let _ = future::block_on(client.sync());
            let dl = t1.elapsed().as_secs_f64();
            drop(outs);
            fused.round(&client, r >= 2);
            if r >= 2 {
                split = split.min(dt);
                lib = lib.min(dl);
            }
        }
        println!(
            "{label}  x{count}   {:.1} MiB   {} cubes total",
            bytes as f64 / (1u64 << 20) as f64,
            (per_n / 8) * count
        );
        println!(
            "    split {count:>2} launches of {:>5} cubes : {:>8.4} ms   {:>6.1} GB/s",
            per_n / 8,
            split * 1e3,
            bytes as f64 / split / 1e9
        );
        println!(
            "    fused  1 launch  of {:>5} cubes : {:>8.4} ms   {:>6.1} GB/s   ({:.2}x)",
            big_n / 8,
            fused.pipe * 1e3,
            bytes as f64 / fused.pipe / 1e9,
            split / fused.pipe
        );
        println!(
            "    split, LIBRARY launcher (allocates out per call) : {:>8.4} ms   {:>6.1} GB/s",
            lib * 1e3,
            bytes as f64 / lib / 1e9
        );
        println!(
            "      the allocation tax alone: {:>7.4} ms over {count} calls = {:.1} us each",
            (lib - split) * 1e3,
            (lib - split) / count as f64 * 1e6
        );
    }
}
