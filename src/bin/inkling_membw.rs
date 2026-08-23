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
