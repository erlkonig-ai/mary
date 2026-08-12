//! How much DEVICE-allocated memory does this part actually hand out, and does
//! a LARGE resident set still stream at device-memory bandwidth?
//!
//! Both questions gate the residency decision and neither can be asked of the
//! driver here: on this unified part `nvidia-smi` answers `[N/A]` for total and
//! for free memory, so the only honest instrument is to allocate and watch
//! `/proc/meminfo`. That is what this does.
//!
//! The second question is the one that is easy to skip and expensive to skip.
//! A device-memory read bandwidth measured on a single 12 MB expert slab is a
//! measurement of cache, not of memory: it says what a working set that fits in
//! L2 costs. A residency plan sized at tens of gibibytes needs to know the
//! rate at tens of gibibytes, because if the figure collapses once the set
//! stops fitting in cache then every per-token estimate built on it is wrong by
//! the same factor. So the read is re-timed over the WHOLE held set after every
//! chunk, and the table below is a curve rather than a number.
//!
//! Guards, because this deliberately allocates until it cannot:
//!
//! * it stops at a `MemAvailable` floor rather than at an allocation failure,
//!   so the machine is never handed to the OOM killer to make a point;
//! * every chunk is FORCED with a reduction before it counts — Burn allocates
//!   lazily and an unforced tensor measures nothing (the trap `resident_final`
//!   documents);
//! * the force is `max`, not `sum`: summing 500 M ones in f32 saturates at 2^24
//!   and the flat result reads exactly like a failed allocation.
//!
//! `INK_CEIL_CHUNK` GiB per chunk, `INK_CEIL_CAP` GiB total, `INK_CEIL_FLOOR`
//! GiB of MemAvailable to leave standing.

use std::time::Instant;

use burn::backend::Cuda;
use burn::prelude::*;

type B = Cuda<f32>;
const GIB: f64 = (1u64 << 30) as f64;

/// One `/proc/meminfo` field, in bytes.
fn meminfo(key: &str) -> f64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix(key) {
            if let Some(kb) = v.split_whitespace().next().and_then(|x| x.parse::<f64>().ok()) {
                return kb * 1024.0;
            }
        }
    }
    0.0
}

/// This process's resident anonymous+file pages, in bytes.
///
/// Worth printing beside the system figure: if device allocations on a unified
/// part do NOT appear in the process's own RSS, then every residency budget
/// reasoned from RSS is reasoning about the wrong pool.
fn rss() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        .map(|pages| pages * 4096.0)
        .unwrap_or(0.0)
}

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Read every byte of `held` on the device and return the elapsed seconds.
///
/// One reduction per tensor, combined into a single scalar so that exactly one
/// synchronisation covers the whole set — timing each tensor separately would
/// measure the sync, which at these sizes is most of it.
///
/// This is a FLOOR on read bandwidth, not a peak: it is Burn's `max` reduction,
/// whose kernel has its own efficiency, and a reduction that runs at half the
/// memory rate reports half the memory rate. Quoted as a floor for that reason.
/// The clean bandwidth number in this probe is the fill rate beside it, which
/// is a flat elementwise write with nothing between it and memory.
fn read_all(held: &[Tensor<B, 1>]) -> f64 {
    let t0 = Instant::now();
    let mut acc = held[0].clone().max();
    for t in &held[1..] {
        acc = acc + t.clone().max();
    }
    let _ = acc.into_scalar();
    t0.elapsed().as_secs_f64()
}

fn main() {
    let chunk_gib = env_f64("INK_CEIL_CHUNK", 2.0);
    let cap_gib = env_f64("INK_CEIL_CAP", 96.0);
    let floor_gib = env_f64("INK_CEIL_FLOOR", 12.0);

    let dev = burn::backend::cuda::CudaDevice::default();
    let elems = (chunk_gib * GIB / 4.0) as usize;
    let chunk_bytes = elems as f64 * 4.0;

    let total = meminfo("MemTotal:");
    println!("=== device-allocation ceiling, GB10 ===");
    println!("  MemTotal        {:8.1} GiB", total / GIB);
    println!("  MemAvailable    {:8.1} GiB  at start", meminfo("MemAvailable:") / GIB);
    println!("  chunk           {:8.2} GiB f32 ({elems} elements)", chunk_bytes / GIB);
    println!("  cap {cap_gib:.0} GiB, stop at {floor_gib:.0} GiB MemAvailable\n");

    // Warm the reduction kernel so the first timed read is not a compile.
    {
        let w = Tensor::<B, 1>::ones([1 << 20], &dev);
        let _ = w.max().into_scalar();
    }

    let mut held: Vec<Tensor<B, 1>> = Vec::new();
    let mut stopped = "cap reached";
    // Reading the WHOLE set costs O(held) and doing it every chunk costs
    // O(held^2), which at a 2 GiB chunk and a 90 GiB ceiling is 2 TiB of reads
    // through a reduction — it dominated the probe and told us nothing the
    // curve at four points does not.
    let checkpoints = [8.0f64, 24.0, 48.0, 72.0];
    let mut next_cp = 0usize;

    println!("   held GiB    fill s   fill GB/s   MemAvail GiB   Cached GiB   RSS GiB");
    loop {
        let avail = meminfo("MemAvailable:");
        if avail < floor_gib * GIB {
            stopped = "MemAvailable floor";
            break;
        }
        if held.len() as f64 * chunk_bytes + chunk_bytes > cap_gib * GIB {
            break;
        }

        let t0 = Instant::now();
        let t = Tensor::<B, 1>::ones([elems], &dev);
        // Force it. An unforced Burn tensor has not necessarily been allocated,
        // and an unallocated tensor measures nothing. One element is enough to
        // synchronise the stream, and the fill kernel that ran before it
        // touched every byte — so this times a flat device-memory WRITE of the
        // whole chunk, which is the cleanest bandwidth figure here.
        let probe = t.clone().slice([0..1]).into_scalar();
        let fill_s = t0.elapsed().as_secs_f64();
        assert_eq!(probe, 1.0f32, "the chunk did not come back as ones");
        held.push(t);

        let bytes = held.len() as f64 * chunk_bytes;
        println!(
            "  {:9.1} {:9.3} {:11.1} {:14.1} {:12.1} {:9.2}",
            bytes / GIB,
            fill_s,
            chunk_bytes / fill_s / 1e9,
            meminfo("MemAvailable:") / GIB,
            (meminfo("Cached:") + meminfo("Buffers:")) / GIB,
            rss() / GIB,
        );

        if next_cp < checkpoints.len() && bytes / GIB >= checkpoints[next_cp] {
            next_cp += 1;
            let s = read_all(&held);
            println!(
                "     .. full read of {:.1} GiB: {:.0} ms  >= {:.0} GB/s (reduction floor)",
                bytes / GIB,
                s * 1e3,
                bytes / s / 1e9
            );
        }
    }

    let bytes = held.len() as f64 * chunk_bytes;
    println!("\n  stopped: {stopped}");
    println!("  held {:.1} GiB of DEVICE-allocated f32 in {} tensors", bytes / GIB, held.len());
    println!("  MemAvailable {:8.1} GiB at the end", meminfo("MemAvailable:") / GIB);
    // What the whole set costs to read once, which is the per-token figure a
    // resident dense set would pay.
    if !held.is_empty() {
        let s = read_all(&held);
        println!(
            "  one full read of the held set: {:.1} ms  =  {:.1} GB/s",
            s * 1e3,
            bytes / s / 1e9
        );
    }
}
