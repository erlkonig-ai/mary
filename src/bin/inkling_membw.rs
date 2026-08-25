//! Read bandwidth on this part, measured on a working set FAR larger than L2,
//! for the two places a weight can live: device-allocated memory, and host
//! `mmap` pages the GPU addresses in place.
//!
//! This exists because a 992 GB/s figure was in circulation for "a kernel
//! reading device memory", and 992 exceeds this part's LPDDR5X bus (~273 GB/s).
//! Nothing read from memory can beat the memory bus, so that was not a memory
//! measurement — it was an 8.4 MB slab re-read out of L2. The tell is exactly
//! that: any read figure above the bus spec means the working set fit in cache.
//!
//! So the rules here are the ones that make a number interpretable:
//!
//! * the working set is gibibytes, not megabytes, and is printed with the
//!   result rather than left to the reader to guess;
//! * each timed run STREAMS it once, so nothing is served warm from L2 on a
//!   second pass — and a second device pass is timed anyway, because if it
//!   comes back faster the set was too small and the whole table is cache;
//! * ONE kernel serves both cases. The only difference between the device run
//!   and the host-mapped run is where the `Handle` came from, so a gap between
//!   them is a property of the pool and not of two different benchmarks;
//! * the cold case is measured first, on pages nothing has touched, with
//!   `/proc/self/io` printed beside it — a "cold" number with zero disk reads
//!   is a warm number that has been mislabelled.
//!
//! `INK_BW_GIB` working set (default 8), `INK_BW_SHARD` the file to map.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use cubecl::prelude::*;
use half::bf16;
use memmap2::Mmap;

use mary::models::inkling::fp4gemm::Aliases;

type Rt = cubecl::cuda::CudaRuntime;

/// Elements each thread reads. Unrolled, so it is a run of coalesced loads.
const PER: usize = 32;
const BLOCK: u32 = 256;
const GIB: f64 = (1u64 << 30) as f64;

/// Stream `src` once and reduce it, so nothing can be elided and nothing is
/// re-read.
///
/// Thread `t` reads `src[t], src[t + T], src[t + 2T], …` for `T` threads, which
/// is fully coalesced: at every step a warp covers 32 consecutive floats. The
/// one store per thread adds 1/32 to the traffic and is NOT added to the
/// reported bytes, so the figure is if anything conservative.
#[cube(launch)]
pub fn stream_read(
    src: &Tensor<f32>,
    out: &mut Tensor<f32>,
    #[comptime] threads: usize,
    #[comptime] per: usize,
) {
    let t = ABSOLUTE_POS as usize;
    if t < out.len() {
        let mut acc = 0.0f32;
        #[unroll]
        for i in 0..per {
            acc += src[t + i * threads];
        }
        out[t] = acc;
    }
}

fn io_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("read_bytes:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

/// Launch [`stream_read`] over `src` and return the seconds it took.
///
/// `read_one` on the output is the synchronisation. Without it the timer would
/// measure enqueueing, which is the other classic way to publish a bandwidth
/// figure that beats the bus.
fn time_read(
    client: &cubecl::prelude::ComputeClient<Rt>,
    src: &cubecl::server::Handle,
    n_f32: usize,
) -> f64 {
    let threads = n_f32 / PER;
    let blocks = threads.div_ceil(BLOCK as usize) as u32;
    let out = client.empty(threads * 4);
    let t0 = Instant::now();
    unsafe {
        stream_read::launch::<Rt>(
            client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(BLOCK),
            TensorArg::from_raw_parts(src.clone(), [threads, 1].into(), [PER, threads].into()),
            TensorArg::from_raw_parts(out.clone(), [threads, 1].into(), [1, threads].into()),
            threads,
            PER,
        )
    };
    let _ = client.read_one(out).expect("read out");
    t0.elapsed().as_secs_f64()
}

fn main() -> Result<()> {
    // The element-width / access-pattern matrix is a different question from
    // the device-vs-host-mapped one below and needs no checkpoint on disk.
    if std::env::var("INK_BW_AXES").is_ok() {
        return run_axes();
    }
    let gib: f64 = std::env::var("INK_BW_GIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8.0);
    let shard = mary::paths::model(
        std::env::var("INK_BW_SHARD").ok().as_deref(),
        "thinkingmachines-inkling-small-nvfp4/model-00005-of-00009.safetensors",
    )?;

    // A whole number of PER * BLOCK * 4-byte grains, so no thread runs off the
    // end and the bounds check is never the thing being measured.
    let grain = PER * BLOCK as usize * 4;
    let bytes = ((gib * GIB) as usize / grain) * grain;
    let n_f32 = bytes / 4;

    let client = Rt::client(&Default::default());
    println!("=== read bandwidth, one streamed pass ===");
    println!(
        "  working set     {:.2} GiB  ({n_f32} f32)",
        bytes as f64 / GIB
    );
    println!("  the LPDDR5X bus here is ~273 GB/s; anything above it is cache, not memory");
    println!(
        "  device can address host memory directly : {}",
        cubecl::cuda::supports_zero_copy_host(0)
    );

    // Compile the kernel on something small, so the first timed run is a read
    // and not a compile.
    {
        let warm = client.empty(grain);
        let _ = time_read(&client, &warm, grain / 4);
    }

    // ---- host-mapped, COLD -------------------------------------------------
    // First, and from the far end of a shard nothing in this process has
    // touched. The disk figure beside it is what makes the label honest.
    let file =
        std::fs::File::open(&shard).with_context(|| format!("opening {}", shard.display()))?;
    // SAFETY: the checkpoint is read-only and nothing else writes it.
    let map = Arc::new(unsafe { Mmap::map(&file) }?);
    anyhow::ensure!(
        map.len() > bytes + 4096,
        "{} is smaller than the working set",
        shard.display()
    );
    // Page-aligned: aliasing needs 4-byte alignment and a page boundary is the
    // least surprising way to be sure of it.
    let off = ((map.len() - bytes) / 4096) * 4096;
    let slice: &[u8] = &map[off..off + bytes];

    let io0 = io_read_bytes();
    let al = Aliases::register(
        &client,
        vec![(
            map.as_ptr() as usize,
            map.len(),
            map.clone() as Arc<dyn std::any::Any + Send + Sync>,
        )],
    )
    .context("the device cannot address host memory directly")?;
    let h_host = al.slice(slice).context("aliasing refused")?;
    let cold = time_read(&client, &h_host, n_f32);
    let cold_disk = io_read_bytes() - io0;
    println!(
        "\n  host-mapped, COLD   {:8.1} GB/s   {:8.1} ms   disk {:.2} GiB",
        bytes as f64 / cold / 1e9,
        cold * 1e3,
        cold_disk as f64 / GIB
    );

    // ---- host-mapped, COLD to the GPU but CPU-PREFAULTED -------------------
    // A DIFFERENT 8 GiB region of the same shard, which this process has not
    // touched from either side. The CPU walks it first, one byte per 4 KiB, so
    // every page is present in this process's page table before the GPU asks
    // for it. If the cold figure above is the cost of establishing those
    // mappings, this one is fast; if it is a property of the pool, it is not.
    {
        let off2 = 0usize;
        let slice2: &[u8] = &map[off2..off2 + bytes];
        let h2 = al.slice(slice2).context("aliasing refused")?;
        let t_pf = Instant::now();
        let mut sink = 0u64;
        let mut i = 0usize;
        while i < slice2.len() {
            sink = sink.wrapping_add(slice2[i] as u64);
            i += 4096;
        }
        std::hint::black_box(sink);
        let pf = t_pf.elapsed().as_secs_f64();
        let c2 = time_read(&client, &h2, n_f32);
        println!(
            "  host-mapped, CPU-PREFAULTED then GPU-cold  {:8.1} GB/s   {:8.1} ms  (CPU walk took {:.1} ms = {:.1} GB/s)",
            bytes as f64 / c2 / 1e9,
            c2 * 1e3,
            pf * 1e3,
            bytes as f64 / pf / 1e9
        );
        let c3 = time_read(&client, &h2, n_f32);
        println!(
            "  same region, 2nd GPU pass                  {:8.1} GB/s   {:8.1} ms",
            bytes as f64 / c3 / 1e9,
            c3 * 1e3
        );
    }

    // ---- host-mapped, WARM -------------------------------------------------
    let io1 = io_read_bytes();
    let warm = time_read(&client, &h_host, n_f32);
    let warm_disk = io_read_bytes() - io1;
    println!(
        "  host-mapped, WARM   {:8.1} GB/s   {:8.1} ms   disk {:.2} GiB",
        bytes as f64 / warm / 1e9,
        warm * 1e3,
        warm_disk as f64 / GIB
    );

    // ---- device-allocated --------------------------------------------------
    // Copied from the SAME bytes, so the pools are compared on identical
    // content and only the provenance of the handle differs.
    let t0 = Instant::now();
    let h_dev = client.create_from_slice(slice);
    let _ = client.read_one(client.empty(4)).expect("sync the upload");
    let upload = t0.elapsed().as_secs_f64();
    let dev = time_read(&client, &h_dev, n_f32);
    println!(
        "  device-allocated    {:8.1} GB/s   {:8.1} ms   (the upload itself ran at {:.1} GB/s)",
        bytes as f64 / dev / 1e9,
        dev * 1e3,
        bytes as f64 / upload / 1e9
    );

    let dev2 = time_read(&client, &h_dev, n_f32);
    println!(
        "  device, 2nd pass    {:8.1} GB/s   {:8.1} ms   (a jump here would mean L2 -- set too small)",
        bytes as f64 / dev2 / 1e9,
        dev2 * 1e3
    );

    println!("\n  device vs host-mapped warm : {:.2}x", warm / dev);
    println!("  device vs host-mapped cold : {:.2}x", cold / dev);
    Ok(())
}

// ---------------------------------------------------------------------------
// The axis matrix: is there ONE ceiling, or one per element width and pattern?
// ---------------------------------------------------------------------------
//
// Two agents measured this box's achievable read bandwidth and disagreed: a
// BF16/f32 in-pass control read 232-247 GB/s, a `stream_packed` four-bit
// control read 168-172 GB/s, and each was used to judge the SAME kernel
// (`fp4_linear_grouped`, 154-171 GB/s) — once as "3 ms of headroom", once as
// "at the ceiling". A ceiling is only a ceiling if the thing it bounds and the
// thing that measured it ran on the same bus under the same conditions, so
// every row below runs BACK TO BACK IN ONE PROCESS, and rows 1-5 read the
// SAME 1 GiB handle bound at different element types: identical physical
// pages, so a gap between them cannot be placement.
//
// What actually varies, and therefore what the answer can be about:
//
// * element type at a fixed 128-bit load (f32 / BF16 / packed u32). The memory
//   system does not see types, so these MUST agree; they are here to prove the
//   232-vs-172 gap is not "four-bit reads are slower".
// * LOAD WIDTH (128-bit vs 32-bit), which the memory system very much does
//   see. `fp4_linear_grouped` issues 32-bit loads: `vs = 32 / e2m1x2 bits` = 4
//   packed bytes per load, `SCALE_VEC` = 4 E4M3 bytes per load.
// * ACCESS PATTERN: a flat coalesced stream versus the `m16n8k64` B-operand
//   footprint the GEMM actually walks — a warp covering 8 weight rows, 32
//   contiguous bytes of each per k tile, the rows `k/2` bytes apart, stepping
//   along k for `k/64` iterations.
// * the SECOND stream. NVFP4 is two planes, not one: `k/2` bytes of codes and
//   `k/16` bytes of E4M3 scales per row, read alternately inside the k loop.
// * the scale plane's LAYOUT: row-major `[n][k/16]`, where a warp's eight
//   4-byte scale reads land in eight different 32-byte sectors, against a
//   k-tile-major `[k/64][n][4]`, where the same eight reads are one contiguous
//   32-byte segment. That is the separable half of the previous bullet.
//
// Cache discipline: every buffer is >= 1 GiB against a 24 MiB L2 (two
// asymmetric instances on this part), i.e. ~42x, so nothing here can be served
// warm; each row is timed five times after two warmups and the FIRST timed
// pass is printed beside the best, because a best that beats its own first
// pass by more than jitter would mean the set was small enough to cache. No
// row loops a small weight buffer — the trap this repo already documents,
// which flatters exactly the shapes with the least grid parallelism.
//
// Stores are sentinel-guarded (`if acc == <never>`) rather than unconditional,
// so write traffic is zero in every row and the figures are pure read. Without
// that the 32-bit-load rows would carry a 1/8 write tax the 128-bit rows do
// not, which is itself a way to manufacture this disagreement.
//
// `INK_BW_AXES=1` runs it; `INK_BW_AXIS_GIB` sizes the buffer (default 1).

/// A coalesced stream at a chosen vector width, reading `f32`.
#[cube(launch)]
pub fn axis_f32<NW: Size>(
    src: &Tensor<Vector<f32, NW>>,
    out: &mut Tensor<f32>,
    #[comptime] threads: usize,
    #[comptime] per: usize,
    #[comptime] nw: usize,
) {
    let t = ABSOLUTE_POS as usize;
    let mut acc = 0.0f32;
    #[unroll]
    for i in 0..per {
        let v = src[t + i * threads];
        #[unroll]
        for j in 0..nw {
            acc += v[j];
        }
    }
    if acc == -1.234_567_9e-31f32 {
        out[t % out.len()] = acc;
    }
}

/// The same stream reading BF16 — same bytes, same addresses, half the width
/// per element and twice the elements per load.
#[cube(launch)]
pub fn axis_bf16<NW: Size>(
    src: &Tensor<Vector<bf16, NW>>,
    out: &mut Tensor<f32>,
    #[comptime] threads: usize,
    #[comptime] per: usize,
    #[comptime] nw: usize,
) {
    let t = ABSOLUTE_POS as usize;
    let mut acc = 0.0f32;
    #[unroll]
    for i in 0..per {
        let v = src[t + i * threads];
        #[unroll]
        for j in 0..nw {
            acc += f32::cast_from(v[j]);
        }
    }
    if acc == -1.234_567_9e-31f32 {
        out[t % out.len()] = acc;
    }
}

/// The same stream reading raw words — what a packed-4-bit plane is to the
/// memory system. `with_sc` adds the E4M3 scale plane at NVFP4's own 8:1 byte
/// ratio, also coalesced, which is the "scales laid out contiguously" arm.
#[cube(launch)]
pub fn axis_u32<NW: Size>(
    src: &Tensor<Vector<u32, NW>>,
    sc: &Tensor<Vector<u32, NW>>,
    out: &mut Tensor<u32>,
    #[comptime] threads: usize,
    #[comptime] per: usize,
    #[comptime] nw: usize,
    #[comptime] with_sc: bool,
) {
    let t = ABSOLUTE_POS as usize;
    let mut acc = u32::new(0i64);
    #[unroll]
    for i in 0..per {
        let v = src[t + i * threads];
        #[unroll]
        for j in 0..nw {
            acc += v[j];
        }
    }
    if comptime![with_sc] {
        // One scale vector per `per` code vectors is exactly NVFP4's ratio when
        // `per` = 8: `k/2` code bytes against `k/16` scale bytes.
        let s = sc[t];
        #[unroll]
        for j in 0..nw {
            acc += s[j];
        }
    }
    if acc == u32::new(0x5AFE_5AFEi64) {
        out[t % out.len()] = acc;
    }
}

/// The B-operand footprint of [`mary::models::inkling::moegroup::fp4_linear_grouped`]
/// with the `mma` deleted and nothing else changed.
///
/// One plane per n tile. The PTX B layout for `m16n8k64` is `col = lane >> 2`,
/// `row = (lane & 3) * 8 + i` with a second group 32 k-elements on — so the
/// plane's 32 lanes cover EIGHT weight rows, four lanes to a row, and each
/// lane's two 32-bit loads sit 16 bytes apart inside one 32-byte segment. Per
/// k tile the plane therefore touches 8 rows x 32 contiguous bytes, the rows
/// `k/2` bytes apart, and `t` walks along k so each 128-byte line is consumed
/// over four consecutive iterations out of L1. 32-bit is the width the real
/// kernel issues: `vs = 32 / e2m1x2::size_bits()` = 4 packed bytes.
///
/// `sc_mode`: 0 = codes only (isolates the pattern from the second stream),
/// 1 = row-major `[n][k/16]` scales, one 32-bit load per row per k tile,
/// 2 = k-tile-major `[k/64][n][4]` scales, where the plane's eight scale reads
/// are eight consecutive words instead of eight scattered sectors.
///
/// `ut` k tiles are unrolled into one iteration. At `ut = 1` that is exactly
/// the real loop, which leaves a warp with three loads in flight; raising it
/// raises memory-level parallelism WITHOUT changing a single address, so a
/// figure that climbs with `ut` says the pattern is latency-bound and a figure
/// that does not says the pattern itself is the ceiling.
#[cube(launch)]
pub fn axis_frag_b(
    codes: &Tensor<u32>,
    scales: &Tensor<u32>,
    out: &mut Tensor<u32>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] sc_mode: u32,
    #[comptime] ut: usize,
    #[comptime] tpw: usize,
) {
    let warp = ABSOLUTE_POS as usize / 32;
    let lane = ABSOLUTE_POS as usize % 32;
    // Four lanes per weight row, eight rows per plane -- the n tile is 8 wide.
    let sub = lane % 4;
    let cw = comptime!(size_k / 8); // u32 words in one row of codes
    let sw = comptime!(size_k / 64); // u32 words in one row of scales
    let steps = comptime!(size_k / 64 / ut);
    let mut acc = u32::new(0i64);
    // `tpw` consecutive n tiles per plane. It reads exactly the same addresses
    // in a different ORDER and, crucially, with `tpw` times fewer resident
    // planes -- which is the one thing separating this model from the kernel
    // it models. `fp4_linear` launches `CubeDim::new_1d(32)`, one plane to a
    // cube, so its occupancy is capped by the per-SM CUBE limit; this model's
    // 256-thread cubes put eight planes on that same budget.
    for g in 0..tpw {
        let row = (warp * tpw + g) * 8 + lane / 4;
        for s in 0..steps {
            #[unroll]
            for u in 0..ut {
                let t = s * ut + u;
                let base = row * cw + t * 8 + sub;
                acc += codes[base];
                acc += codes[base + 4];
                if comptime![sc_mode == 1] {
                    acc += scales[row * sw + t];
                }
                if comptime![sc_mode == 2] {
                    acc += scales[t * size_n + row];
                }
            }
        }
    }
    if acc == u32::new(0x5AFE_5AFEi64) {
        out[warp % out.len()] = acc;
    }
}

/// Best and first-pass seconds over `reps` timed launches after two warmups.
fn best_first(mut run: impl FnMut() -> f64, reps: usize) -> (f64, f64) {
    for _ in 0..2 {
        run();
    }
    let mut best = f64::MAX;
    let mut first = 0.0;
    for i in 0..reps {
        let dt = run();
        if i == 0 {
            first = dt;
        }
        best = best.min(dt);
    }
    (best, first)
}

fn run_axes() -> Result<()> {
    let gib: f64 = std::env::var("INK_BW_AXIS_GIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let client = Rt::client(&Default::default());
    let reps: usize = 5;

    // The GEMM's own shape. `k` is Inkling's hidden width; `n` is chosen so the
    // code plane is the requested size, and is a multiple of 64 so that the
    // fragment grid is a whole number of 256-thread blocks.
    let k = 4096usize;
    // A multiple of 64 so the fragment grid is a whole number of 256-thread
    // blocks, and `n / NTILE <= 65535` so the real kernel below accepts the
    // same table -- it has to be the SAME table, or the row that anchors this
    // whole matrix is measuring a different problem.
    let n = ((((gib * GIB) as usize / (k / 2)) / 64) * 64).min(524224);
    let codes_b = n * (k / 2);
    let scales_b = n * (k / 16);

    // ONE handle serves rows 1-5. Same pages, same placement, only the binding
    // changes — so an element-width difference cannot be an allocator artefact.
    let big = client.empty(codes_b);
    let sc = client.empty(scales_b);
    let sink = client.empty(4096);

    println!("=== achievable read bandwidth, one process, back to back ===");
    println!(
        "  code plane      {:.3} GiB  ({n} rows x {} B)",
        codes_b as f64 / GIB,
        k / 2
    );
    println!(
        "  scale plane     {:.3} GiB  ({n} rows x {} B)",
        scales_b as f64 / GIB,
        k / 16
    );
    println!(
        "  L2 is 24 MiB in 2 asymmetric instances: the code plane is {:.0}x it",
        codes_b as f64 / (24.0 * 1024.0 * 1024.0)
    );
    println!("  rows 1-5 read the SAME handle, bound at different element types");
    println!("  the LPDDR5X bus here is ~273 GB/s; anything above it is cache, not memory\n");
    println!("  {:<46} {:>9} {:>9} {:>9}", "row", "GB/s", "ms", "1st ms");

    let report = |label: &str, bytes: usize, best: f64, first: f64| {
        println!(
            "  {:<46} {:>9.1} {:>9.3} {:>9.3}",
            label,
            bytes as f64 / best / 1e9,
            best * 1e3,
            first * 1e3
        );
    };

    // ---- rows 1-5: flat coalesced streams over the same 1 GiB -------------
    let per = 8usize;
    for (label, nw) in [("1  f32   coalesced, 128-bit loads", 4usize)] {
        let words = codes_b / (4 * nw);
        let threads = words / per;
        let blocks = (threads as u32).div_ceil(BLOCK);
        let (b, f) = best_first(
            || {
                let t0 = Instant::now();
                unsafe {
                    axis_f32::launch::<Rt>(
                        &client,
                        CubeCount::Static(blocks, 1, 1),
                        CubeDim::new_1d(BLOCK),
                        nw,
                        TensorArg::from_raw_parts(big.clone(), [1].into(), [words].into()),
                        TensorArg::from_raw_parts(sink.clone(), [1].into(), [1024].into()),
                        threads,
                        per,
                        nw,
                    )
                };
                let _ = cubecl::future::block_on(client.sync());
                t0.elapsed().as_secs_f64()
            },
            reps,
        );
        report(label, codes_b, b, f);
    }
    for (label, nw) in [("2  f32   coalesced,  32-bit loads", 1usize)] {
        let words = codes_b / (4 * nw);
        let threads = words / per;
        let blocks = (threads as u32).div_ceil(BLOCK);
        let (b, f) = best_first(
            || {
                let t0 = Instant::now();
                unsafe {
                    axis_f32::launch::<Rt>(
                        &client,
                        CubeCount::Static(blocks, 1, 1),
                        CubeDim::new_1d(BLOCK),
                        nw,
                        TensorArg::from_raw_parts(big.clone(), [1].into(), [words].into()),
                        TensorArg::from_raw_parts(sink.clone(), [1].into(), [1024].into()),
                        threads,
                        per,
                        nw,
                    )
                };
                let _ = cubecl::future::block_on(client.sync());
                t0.elapsed().as_secs_f64()
            },
            reps,
        );
        report(label, codes_b, b, f);
    }
    for (label, nw) in [("3  BF16  coalesced, 128-bit loads", 8usize)] {
        let words = codes_b / (2 * nw);
        let threads = words / per;
        let blocks = (threads as u32).div_ceil(BLOCK);
        let (b, f) = best_first(
            || {
                let t0 = Instant::now();
                unsafe {
                    axis_bf16::launch::<Rt>(
                        &client,
                        CubeCount::Static(blocks, 1, 1),
                        CubeDim::new_1d(BLOCK),
                        nw,
                        TensorArg::from_raw_parts(big.clone(), [1].into(), [words].into()),
                        TensorArg::from_raw_parts(sink.clone(), [1].into(), [1024].into()),
                        threads,
                        per,
                        nw,
                    )
                };
                let _ = cubecl::future::block_on(client.sync());
                t0.elapsed().as_secs_f64()
            },
            reps,
        );
        report(label, codes_b, b, f);
    }
    for (label, nw, with_sc) in [
        ("4  NVFP4 codes only, 128-bit loads", 4usize, false),
        ("5  NVFP4 codes only,  32-bit loads", 1usize, false),
        (
            "6  NVFP4 codes + scales, both coalesced, 32-bit",
            1usize,
            true,
        ),
    ] {
        let words = codes_b / (4 * nw);
        let threads = words / per;
        let blocks = (threads as u32).div_ceil(BLOCK);
        let sc_words = scales_b / (4 * nw);
        let bytes = if with_sc { codes_b + scales_b } else { codes_b };
        let (b, f) = best_first(
            || {
                let t0 = Instant::now();
                unsafe {
                    axis_u32::launch::<Rt>(
                        &client,
                        CubeCount::Static(blocks, 1, 1),
                        CubeDim::new_1d(BLOCK),
                        nw,
                        TensorArg::from_raw_parts(big.clone(), [1].into(), [words].into()),
                        TensorArg::from_raw_parts(sc.clone(), [1].into(), [sc_words].into()),
                        TensorArg::from_raw_parts(sink.clone(), [1].into(), [1024].into()),
                        threads,
                        per,
                        nw,
                        with_sc,
                    )
                };
                let _ = cubecl::future::block_on(client.sync());
                t0.elapsed().as_secs_f64()
            },
            reps,
        );
        report(label, bytes, b, f);
    }

    // ---- rows 7-9: the GEMM's own B footprint ------------------------------
    let threads0 = n * 4; // 32 lanes per n tile of 8 rows
    // `bd` is the cube width, and on this part it IS the occupancy knob: 48 SMs
    // with maxThreadsPerSM 1536 and maxBlocksPerSM 24, so a 256-thread cube
    // puts 6 cubes x 8 planes = 48 planes on an SM while a 32-thread cube --
    // which is what `fp4_linear` launches -- is capped by the cube limit at 24.
    // Each plane holds eight 128-byte code lines and eight scale lines it must
    // keep across four k iterations or refetch them, so 48 planes want 96 KiB
    // of a 128 KiB L1 and 24 planes want 48 KiB. That is the one axis the model
    // did not share with the kernel, and `tpw` does NOT test it: giving a plane
    // more n tiles launches fewer planes but leaves the SM just as full.
    for (label, mode, with_sc, ut, tpw, bd) in [
        (
            "7  m16n8k64 B footprint, codes only",
            0u32,
            false,
            1usize,
            1usize,
            256u32,
        ),
        (
            "8  m16n8k64 B footprint + row-major scales",
            1u32,
            true,
            1usize,
            1usize,
            256u32,
        ),
        (
            "9  m16n8k64 B footprint + k-tile-major scales",
            2u32,
            true,
            1usize,
            1usize,
            256u32,
        ),
        (
            "10 row 8, 2 k tiles per iteration",
            1u32,
            true,
            2usize,
            1usize,
            256u32,
        ),
        (
            "11 row 8, 4 k tiles per iteration",
            1u32,
            true,
            4usize,
            1usize,
            256u32,
        ),
        (
            "12 row 8, 8 k tiles per iteration",
            1u32,
            true,
            8usize,
            1usize,
            256u32,
        ),
        (
            "13 row 9, 8 k tiles per iteration",
            2u32,
            true,
            8usize,
            1usize,
            256u32,
        ),
        (
            "14 row 7, 8 k tiles per iteration",
            0u32,
            false,
            8usize,
            1usize,
            256u32,
        ),
        (
            "16 row 8, 2 n tiles per plane",
            1u32,
            true,
            1usize,
            2usize,
            256u32,
        ),
        (
            "17 row 8, 8 n tiles per plane",
            1u32,
            true,
            1usize,
            8usize,
            256u32,
        ),
        (
            "18 row 8, 32-thread cubes (24 planes/SM, as fp4_linear)",
            1u32,
            true,
            1usize,
            1usize,
            32u32,
        ),
        (
            "19 row 7, 32-thread cubes",
            0u32,
            false,
            1usize,
            1usize,
            32u32,
        ),
        (
            "20 row 9, 32-thread cubes",
            2u32,
            true,
            1usize,
            1usize,
            32u32,
        ),
        (
            "21 row 8, 64-thread cubes",
            1u32,
            true,
            1usize,
            1usize,
            64u32,
        ),
        (
            "22 row 8, 128-thread cubes",
            1u32,
            true,
            1usize,
            1usize,
            128u32,
        ),
        (
            "23 row 8, 512-thread cubes",
            1u32,
            true,
            1usize,
            1usize,
            512u32,
        ),
    ] {
        let threads = threads0 / tpw;
        let blocks = (threads as u32).div_ceil(bd);
        let bytes = if with_sc { codes_b + scales_b } else { codes_b };
        let (b, f) = best_first(
            || {
                let t0 = Instant::now();
                unsafe {
                    axis_frag_b::launch::<Rt>(
                        &client,
                        CubeCount::Static(blocks, 1, 1),
                        CubeDim::new_1d(bd),
                        TensorArg::from_raw_parts(big.clone(), [1].into(), [codes_b / 4].into()),
                        TensorArg::from_raw_parts(sc.clone(), [1].into(), [scales_b / 4].into()),
                        TensorArg::from_raw_parts(sink.clone(), [1].into(), [1024].into()),
                        k,
                        n,
                        mode,
                        ut,
                        tpw,
                    )
                };
                let _ = cubecl::future::block_on(client.sync());
                t0.elapsed().as_secs_f64()
            },
            reps,
        );
        report(label, bytes, b, f);
    }

    // ---- row 15: the kernel itself, on the same table, in the same process --
    //
    // Rows 7-14 are a MODEL of what `fp4_linear_grouped` reads, and a model is
    // only worth its agreement with the thing modelled. `fp4_linear` is that
    // thing: the grouped kernel is line for line this one plus an expert
    // offset, so its B loads and its k loop are identical, and at `m_pad = 16`
    // it is the decode case -- one m tile, no reuse to find. It reads the SAME
    // 1 GiB handle rows 1-14 read, on the same bus, seconds apart.
    //
    // Its time includes the launcher's own `[m_pad, n]` f32 output allocation
    // and the store into it, which rows 7-14 do not pay; both are printed
    // below so the comparison can be made net of them.
    {
        use mary::models::inkling::fp4gemm::fp4_linear_launch;
        let m_pad = 16usize;
        let qa = client.empty(m_pad * k / 2);
        let qa_sc = client.empty(m_pad * (k / 16));
        let bytes = codes_b + scales_b;
        let (b, f) = best_first(
            || {
                let t0 = Instant::now();
                let o = fp4_linear_launch::<Rt>(&client, &qa, &qa_sc, &big, &sc, m_pad, k, n, 1.0);
                let _ = cubecl::future::block_on(client.sync());
                let dt = t0.elapsed().as_secs_f64();
                drop(o);
                dt
            },
            reps,
        );
        report(
            "15 fp4_linear ITSELF, m_pad=16 (the decode case)",
            bytes,
            b,
            f,
        );
        let (ab, af) = best_first(
            || {
                let t0 = Instant::now();
                let h = client.empty(m_pad * n * 4);
                let _ = cubecl::future::block_on(client.sync());
                let dt = t0.elapsed().as_secs_f64();
                drop(h);
                dt
            },
            reps,
        );
        println!(
            "     of which output alloc + sync {:.3} ms, so the read alone is {:.1} GB/s",
            ab * 1e3,
            bytes as f64 / (b - ab) / 1e9
        );
        let _ = af;
    }

    Ok(())
}
