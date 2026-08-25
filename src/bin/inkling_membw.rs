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

/// [`axis_frag_b`]'s footprint again — the SAME addresses, the same bytes, the
/// same fragment shape at the consumer — with the global read STAGED THROUGH
/// SHARED MEMORY.
///
/// This is the whole experiment. `axis_frag_b` reads the B fragment straight
/// out of global, and a warp instruction there spans eight weight rows `k/2`
/// bytes apart: eight sector requests for 128 useful bytes, where a coalesced
/// 32-bit stream issues four for the same 128. No instruction can be a
/// contiguous read while lanes map onto eight separate rows, so the only way
/// to recover the coalesced rate is to break that correspondence — which is
/// what this does. The GLOBAL read becomes a per-row contiguous stream (a warp
/// covers `kc * 32` consecutive bytes of ONE row), shared memory is filled
/// cooperatively, and only the SMEM read keeps the fragment's eight-row shape.
///
/// What is NOT settled going in, and is the reason this row exists next to the
/// unstaged one rather than replacing it: how much of that 2x survives the
/// round trip. The staged arm pays two `sync_cube` per chunk, an smem write
/// and an smem read per word, and — unless the row stride is padded — an
/// eight-way bank conflict on every fragment read.
///
/// `kc` k tiles are staged per chunk, so a row contributes `kc * 32` bytes to
/// each cooperative pass. `kc = 4` makes that exactly one 128-byte line per row
/// per pass, which is the smallest chunk that reads whole lines; `kc = 8` makes
/// it two, and also makes the SCALE row long enough (32 B) to fill a sector.
///
/// `pad` is the padding, in words, added to the code row stride, and it is the
/// bank-conflict knob. A fragment read has lane `l` at row `l / 4` and word
/// `(l & 3)` of that row, so its bank is `(l / 4) * cs + (l & 3)` mod 32 for a
/// row stride of `cs` words. At `pad = 0`, `cs = kc * 8` is a multiple of 32
/// and all eight rows collapse onto banks 0-3: an eight-way conflict. At
/// `pad = 4`, `cs = 4 * (2 * kc + 1)` is four times an ODD number, so the eight
/// row offsets are eight distinct multiples of four and lane `l` lands in bank
/// `l`. Conflict-free, for 4 words a row of extra shared memory. Both are
/// measured, because "what the re-layout costs" is exactly the open question.
///
/// `stage_sc` extends the staging to the E4M3 scale plane. Its cost per row per
/// k tile is one word, so a chunk stages `kc` words a row — 16 B at `kc = 4`,
/// which is still half a sector, and 32 B at `kc = 8`, which is a whole one.
/// Left unstaged (`with_sc` alone) the scale read keeps the eight-row shape it
/// has in the kernel: eight sector requests for 32 useful bytes, i.e. the same
/// sector count as the codes for an eighth of the bytes.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn axis_frag_b_smem(
    codes: &Tensor<u32>,
    scales: &Tensor<u32>,
    out: &mut Tensor<u32>,
    #[comptime] size_k: usize,
    #[comptime] with_sc: bool,
    #[comptime] stage_sc: bool,
    #[comptime] kc: usize,
    #[comptime] pad: usize,
    #[comptime] pad_s: usize,
    #[comptime] threads: usize,
) {
    // Weight rows a cube stages: one n tile of 8 per plane, as in `axis_frag_b`.
    let rows = comptime!(threads / 32 * 8);
    let cs = comptime!(kc * 8 + pad);
    let ss = comptime!(kc + pad_s);
    let cw = comptime!(size_k / 8); // u32 words in one row of codes
    let sw = comptime!(size_k / 64); // u32 words in one row of scales
    let chunks = comptime!(size_k / 64 / kc);
    // Words a thread moves per cooperative pass: `rows * kc * 8 / threads`,
    // which is `2 * kc` for any cube width — the cube grows with the rows it
    // has to fill.
    let per_c = comptime!(rows * kc * 8 / threads);
    let per_s = comptime!(rows * kc / threads);

    let mut sm = SharedMemory::<u32>::new(comptime!(rows * cs));
    let mut sm_sc = SharedMemory::<u32>::new(comptime!(if stage_sc { rows * ss } else { 1usize }));

    let unit = UNIT_POS as usize;
    let row0 = CUBE_POS as usize * rows;
    // The consumer's fragment shape, unchanged: four lanes to a weight row,
    // eight rows to a plane, two 32-bit reads 16 bytes apart.
    let rl = unit / 32 * 8 + (unit % 32) / 4;
    let sub = unit % 4;

    let mut acc = u32::new(0i64);
    for c in 0..chunks {
        // The cooperative load. Thread `t` takes flat word `t + j * threads` of
        // the chunk, and the chunk is laid out row-major, so a warp covers
        // `min(kc * 8, 32)` consecutive words of one row — 128 consecutive
        // bytes at `kc >= 4`, which is the fully-coalesced case.
        #[unroll]
        for j in 0..per_c {
            let f = unit + j * threads;
            let r = f / comptime!(kc * 8);
            let o = f % comptime!(kc * 8);
            sm[r * cs + o] = codes[(row0 + r) * cw + c * comptime!(kc * 8) + o];
        }
        if comptime![stage_sc] {
            #[unroll]
            for j in 0..per_s {
                let f = unit + j * threads;
                let r = f / kc;
                let o = f % kc;
                sm_sc[r * ss + o] = scales[(row0 + r) * sw + c * kc + o];
            }
        }
        sync_cube();
        #[unroll]
        for t in 0..kc {
            let base = rl * cs + t * 8 + sub;
            acc += sm[base];
            acc += sm[base + 4];
            if comptime![stage_sc] {
                acc += sm_sc[rl * ss + t];
            } else if comptime![with_sc] {
                acc += scales[(row0 + rl) * sw + c * kc + t];
            }
        }
        // The chunk is overwritten on the next pass, so the readers have to be
        // done with it. This is the second of the two barriers a single-buffered
        // stage pays, and it is part of what the figure below is measuring.
        sync_cube();
    }
    if acc == u32::new(0x5AFE_5AFEi64) {
        out[unit % out.len()] = acc;
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
    println!("  {:<62} {:>9} {:>9} {:>9}", "row", "GB/s", "ms", "1st ms");

    let report = |label: &str, bytes: usize, best: f64, first: f64| {
        println!(
            "  {:<62} {:>9.1} {:>9.3} {:>9.3}",
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

    // ---- rows 24+: the same footprint, STAGED THROUGH SHARED MEMORY --------
    //
    // These are the experiment the retraction in `moegroup`'s header asks for.
    // They read the SAME 1 GiB handle rows 1-23 read, seconds later, in the
    // same process, and they present the consumer with the SAME eight-row
    // fragment shape — the only thing that changes is that the GLOBAL read is
    // a per-row contiguous stream and the eight-row scatter happens in shared
    // memory instead. Each is printed next to the unstaged row it replaces:
    // 24 against 7, 25 against 8, 26 against 8 with the scale plane staged too.
    //
    // The `pad = 0` rows are not an oversight. `pad` is the bank-conflict knob
    // (see the kernel's header) and its two settings are the measured price of
    // the re-layout, which is half of what "how much of the 2x survives" means.
    // The smem-per-cube column is the other half: this part gives an SM 1536
    // threads and ~100 KiB of shared memory, so a 256-thread cube can have six
    // residents on the thread budget and `100 KiB / smem` on the other, and
    // once the second is the smaller number the stage has bought coalescing
    // with occupancy.
    for (label, with_sc, stage_sc, kc, pad, pad_s, bd) in [
        (
            "24 STAGED codes only, kc=4, padded",
            false,
            false,
            4usize,
            4usize,
            1usize,
            256u32,
        ),
        (
            "25 STAGED codes, kc=4, padded + row-major scales",
            true,
            false,
            4,
            4,
            1,
            256,
        ),
        (
            "26 STAGED codes + scales, kc=8, padded",
            true,
            true,
            8,
            4,
            1,
            256,
        ),
        (
            "27 STAGED codes only, kc=8, padded",
            false,
            false,
            8,
            4,
            1,
            256,
        ),
        (
            "28 STAGED codes, kc=8, padded + row-major scales",
            true,
            false,
            8,
            4,
            1,
            256,
        ),
        (
            "29 row 24 with pad=0 (8-way bank conflict)",
            false,
            false,
            4,
            0,
            1,
            256,
        ),
        (
            "30 row 26 with pad=0 (8-way bank conflict)",
            true,
            true,
            8,
            0,
            0,
            256,
        ),
        (
            "31 row 26, kc=16 (twice the smem, half the cubes)",
            true,
            true,
            16,
            4,
            1,
            256,
        ),
        ("32 row 26, 128-thread cubes", true, true, 8, 4, 1, 128),
        ("33 row 26, 64-thread cubes", true, true, 8, 4, 1, 64),
        ("34 row 26, 32-thread cubes", true, true, 8, 4, 1, 32),
    ] {
        let rows_per_cube = bd as usize / 4;
        let blocks = (n / rows_per_cube) as u32;
        let bytes = if with_sc { codes_b + scales_b } else { codes_b };
        let smem = rows_per_cube * (kc * 8 + pad) * 4
            + if stage_sc {
                rows_per_cube * (kc + pad_s) * 4
            } else {
                4
            };
        let (b, f) = best_first(
            || {
                let t0 = Instant::now();
                unsafe {
                    axis_frag_b_smem::launch::<Rt>(
                        &client,
                        CubeCount::Static(blocks, 1, 1),
                        CubeDim::new_1d(bd),
                        TensorArg::from_raw_parts(big.clone(), [1].into(), [codes_b / 4].into()),
                        TensorArg::from_raw_parts(sc.clone(), [1].into(), [scales_b / 4].into()),
                        TensorArg::from_raw_parts(sink.clone(), [1].into(), [1024].into()),
                        k,
                        with_sc,
                        stage_sc,
                        kc,
                        pad,
                        pad_s,
                        bd as usize,
                    )
                };
                let _ = cubecl::future::block_on(client.sync());
                t0.elapsed().as_secs_f64()
            },
            reps,
        );
        report(&format!("{label}  [{smem} B smem/cube]"), bytes, b, f);
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

        // ---- rows 35+: the same kernel with B staged --------------------
        //
        // `fp4_linear_smem` is row 15's kernel with the B fragment read out of
        // shared memory, and the cube is ONE plane here, so the fill is the
        // warp's own eight rows and there is no cross-plane barrier. It reads
        // the same table, allocates the same output, pays the same store, and
        // is bit-identical — so the difference from row 15 is the staging and
        // nothing else.
        //
        // `stage_sc` is separate because the two defects are separate: this
        // kernel reads its four E4M3 block scales as four INDIVIDUAL bytes,
        // four instructions each spanning the same eight rows, i.e. 32 sector
        // requests a k tile against the codes' 16. The grouped kernel already
        // reads them as one 32-bit vector; this one does not, so leaving the
        // staging off measures the code staging against an unchanged baseline
        // and turning it on says what the scale plane was costing.
        use mary::models::inkling::fp4gemm::fp4_linear_smem_launch;
        for (label, kc, pad, st) in [
            ("35 STAGED codes, kc=4 pad=0", 4usize, 0usize, false),
            ("36 STAGED codes, kc=4 pad=4", 4usize, 4usize, false),
            ("37 STAGED codes, kc=8 pad=0", 8usize, 0usize, false),
            ("38 STAGED codes + scales, kc=4 pad=0", 4usize, 0usize, true),
            ("39 STAGED codes + scales, kc=8 pad=0", 8usize, 0usize, true),
        ] {
            let (sb, sf) = best_first(
                || {
                    let t0 = Instant::now();
                    let o = fp4_linear_smem_launch::<Rt>(
                        &client, &qa, &qa_sc, &big, &sc, m_pad, k, n, 1.0, kc, pad, st,
                    );
                    let _ = cubecl::future::block_on(client.sync());
                    let dt = t0.elapsed().as_secs_f64();
                    drop(o);
                    dt
                },
                reps,
            );
            report(label, bytes, sb, sf);
        }

        // ---- rows 40+: the same kernel on a PRE-PERMUTED weight ---------
        //
        // The third arm, and the one that costs nothing at runtime. Rows 35-39
        // recover the coalesced rate by moving the scattered read into shared
        // memory; `fp4_linear_swz` recovers it by having the bytes already be
        // in fragment order, which for a STATIC weight is a property of how it
        // was written down rather than of what the kernel does. Same 1 GiB
        // handle, same output allocation, same store — so the only difference
        // from row 15 is the B address expression.
        //
        // Reusing `big`/`sc` is deliberate: the swizzled kernel reads exactly
        // the same volume out of exactly the same pages, so an address-pattern
        // difference cannot be an allocator or placement artefact. Its CONTENTS
        // would have to be permuted for the arithmetic to agree, which is why
        // bit-identity is proved in `gemm_grid_parity` on host-created bytes
        // and not here on an uninitialised handle.
        //
        // First: the fragment map itself, off the device, because every claim
        // about the permutation rests on it.
        {
            use mary::models::inkling::fp4gemm::fp4_frag_b_map_launch;
            let h = fp4_frag_b_map_launch::<Rt>(&client);
            let raw = client.read_one(h).expect("frag map");
            let w: Vec<u32> = raw
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let (vc, vs, ec, scn) = (w[256], w[257], w[258], w[259]);
            println!(
                "\n  m16n8k64 B fragment, from position_of_nth: elems/lane {ec}, vector {vs}, \
                 loads/lane {vc}, scales/lane {scn}"
            );
            let mut closed = true;
            for l in 0..32usize {
                for i in 0..vc as usize {
                    let (row, col) = (w[(l * 4 + i) * 2], w[(l * 4 + i) * 2 + 1]);
                    closed &= col as usize == l >> 2 && row as usize == (l & 3) * 8 + i * 32;
                }
                closed &= w[261 + l] as usize == l >> 2;
            }
            let show = |lo: usize, hi: usize| -> String {
                let mut parts = Vec::new();
                for l in lo..hi {
                    for i in 0..vc as usize {
                        parts.push(format!(
                            "l{l}/{i} k={} n={}",
                            w[(l * 4 + i) * 2],
                            w[(l * 4 + i) * 2 + 1]
                        ));
                    }
                }
                parts.join("  ")
            };
            println!("  lanes 0..4:  {}", show(0, 4));
            println!("  lanes 4..8:  {}", show(4, 8));
            println!(
                "  closed form col = lane>>2, k = (lane&3)*8 + 32*i, scale row = lane>>2: {}",
                if closed {
                    "HOLDS for all 32 lanes"
                } else {
                    "DOES NOT HOLD -- the swizzle is wrong"
                }
            );
            assert!(
                closed,
                "the fragment map is not what swizzle_b_codes assumes"
            );
        }

        use mary::models::inkling::fp4gemm::fp4_linear_swz_launch;
        for (label, swz_sc) in [
            ("40 PRE-PERMUTED codes (scales untouched)", false),
            ("41 PRE-PERMUTED codes + scales", true),
        ] {
            let (zb, zf) = best_first(
                || {
                    let t0 = Instant::now();
                    let o = fp4_linear_swz_launch::<Rt>(
                        &client, &qa, &qa_sc, &big, &sc, m_pad, k, n, 1.0, swz_sc,
                    );
                    let _ = cubecl::future::block_on(client.sync());
                    let dt = t0.elapsed().as_secs_f64();
                    drop(o);
                    dt
                },
                reps,
            );
            report(label, bytes, zb, zf);
            println!(
                "     net of the {:.3} ms output alloc: {:.1} GB/s",
                ab * 1e3,
                bytes as f64 / (zb - ab) / 1e9
            );
        }

        // ---- the shootout: the three arms INTERLEAVED -------------------
        //
        // Rows 15, 35-39 and 40-41 are measured in blocks, one kernel's five
        // reps before the next kernel's, and this part drifts: two runs of this
        // binary minutes apart, one with a sibling on the GPU and one on an
        // idle one, put row 15 at 102.6 and 94.5 GB/s and the coalesced control
        // (row 5) at 247.7 and 237.4. The whole table moves together, so the
        // ratios survive — but rows 40-41 are always LAST, which on a drifting
        // part is a systematic handicap and not noise.
        //
        // So: alternate the three arms inside one loop, so each sees the same
        // clocks on average and position cannot favour any of them. Min over
        // rounds, as everywhere else here.
        {
            use mary::models::inkling::fp4gemm::fp4_linear_swz_launch;
            let rounds = 9usize;
            let (mut u, mut g, mut z) = (f64::MAX, f64::MAX, f64::MAX);
            let mut time = |f: &mut dyn FnMut()| {
                let t0 = Instant::now();
                f();
                let _ = cubecl::future::block_on(client.sync());
                t0.elapsed().as_secs_f64()
            };
            for _ in 0..rounds {
                u = u.min(time(&mut || {
                    drop(fp4_linear_launch::<Rt>(
                        &client, &qa, &qa_sc, &big, &sc, m_pad, k, n, 1.0,
                    ))
                }));
                g = g.min(time(&mut || {
                    drop(fp4_linear_smem_launch::<Rt>(
                        &client, &qa, &qa_sc, &big, &sc, m_pad, k, n, 1.0, 8, 0, true,
                    ))
                }));
                z = z.min(time(&mut || {
                    drop(fp4_linear_swz_launch::<Rt>(
                        &client, &qa, &qa_sc, &big, &sc, m_pad, k, n, 1.0, true,
                    ))
                }));
            }
            let gbs = |t: f64| bytes as f64 / t / 1e9;
            println!("\n  === the three arms interleaved, {rounds} rounds, min each ===");
            println!(
                "  A  unstaged  (fp4_linear)          {:8.1} GB/s  {:7.3} ms   1.00x",
                gbs(u),
                u * 1e3
            );
            println!(
                "  B  STAGED    (fp4_linear_smem)     {:8.1} GB/s  {:7.3} ms   {:.2}x",
                gbs(g),
                g * 1e3,
                u / g
            );
            println!(
                "  C  PRE-PERM  (fp4_linear_swz)      {:8.1} GB/s  {:7.3} ms   {:.2}x   \
                 (C/B = {:.3}x)",
                gbs(z),
                z * 1e3,
                u / z,
                g / z
            );
        }

        // ---- row 42: what the permutation COSTS, on the host ------------
        //
        // The only price option (b) — permute once at load — actually pays.
        // `PileSource::copy_share` already memcpys every routed expert into an
        // anonymous arena at startup and the GPU handles alias THAT, not the
        // pile mmap, so permuting during a copy that already happens replaces
        // a `copy_from_slice` with this. The question is therefore not "what
        // does a copy cost" but "what does this copy cost against a plain one",
        // which is what the two figures below are.
        //
        // Sized as one NVFP4 expert's `w13` — [4096, 4096], 8.4 MB of codes and
        // 1.05 MB of scales — because that is the granularity `copy_share`
        // moves and the working set it moves it in.
        {
            use mary::models::inkling::fp4gemm::{swizzle_b_codes, swizzle_b_scales};
            let (en, ek) = (4096usize, 4096usize);
            let src = vec![0x5Au8; en * ek / 2];
            let ssc = vec![0x38u8; en * (ek / 16)];
            let total = src.len() + ssc.len();
            let mut best = f64::MAX;
            let mut best_cp = f64::MAX;
            for _ in 0..5 {
                let t0 = Instant::now();
                let c = swizzle_b_codes(&src, en, ek);
                let d = swizzle_b_scales(&ssc, en, ek);
                best = best.min(t0.elapsed().as_secs_f64());
                std::hint::black_box((&c, &d));
                let t1 = Instant::now();
                let c2 = src.clone();
                let d2 = ssc.clone();
                best_cp = best_cp.min(t1.elapsed().as_secs_f64());
                std::hint::black_box((&c2, &d2));
            }
            println!(
                "\n  42 HOST permutation of one expert w13 ({:.1} MB): {:.2} ms = {:.1} GB/s",
                total as f64 / 1e6,
                best * 1e3,
                total as f64 / best / 1e9
            );
            println!(
                "     the plain copy it replaces:              {:.2} ms = {:.1} GB/s  ({:.2}x)",
                best_cp * 1e3,
                total as f64 / best_cp / 1e9,
                best / best_cp
            );
            // 39 NVFP4 routed layers x 256 experts x (w13 + w2) is the whole
            // model; a process holds its layer share of that.
            let model = 39.0 * 256.0 * (14_155_784.0);
            println!(
                "     extrapolated over all 39 NVFP4 routed layers ({:.0} GiB): {:.1} s of startup, \
                 against {:.1} s for the copy already paid",
                model / GIB,
                model / (total as f64 / best) / 1.0,
                model / (total as f64 / best_cp)
            );
        }

        // Bit equality against row 15. The staged kernel changes WHERE the B
        // fragment is read from and nothing else, so anything but identical
        // output is a defect in the staging rather than a rounding difference.
        let o_base = fp4_linear_launch::<Rt>(&client, &qa, &qa_sc, &big, &sc, m_pad, k, n, 1.0);
        let o_smem = fp4_linear_smem_launch::<Rt>(
            &client, &qa, &qa_sc, &big, &sc, m_pad, k, n, 1.0, 4, 0, true,
        );
        let vb = client.read_one(o_base).expect("baseline output");
        let vs = client.read_one(o_smem).expect("staged output");
        let diff = vb.iter().zip(vs.iter()).filter(|(x, y)| x != y).count();
        println!(
            "     bit equality against row 15: {diff} of {} output bytes differ",
            vb.len()
        );
        assert_eq!(diff, 0, "the staged head lane is NOT bit-identical");
    }

    Ok(())
}
