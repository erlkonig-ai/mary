//! Cross-device depformer probe -- the CPU half of a PersonaPlex frame.
//!
//! The temporal transformer runs on the GPU and has been measured. The depth
//! transformer is `depth_fast.rs`, "the fast CPU predictor", and it is the
//! missing half of the 80 ms budget. This benchmarks it ALONE, on synthetic
//! weights (`DepthFast::synthetic`) so no pile is needed and both machines run
//! byte-identical work.
//!
//! This is an M4 Max CPU versus GB10 (10x Cortex-X925 + 10x Cortex-A725)
//! comparison, which is a completely different axis from the GPU numbers.
use std::time::Instant;

use mary::models::personaplex::config as cfg;
use mary::models::personaplex::depth_fast::DepthFast;

fn loadavg() -> f64 {
    std::process::Command::new("uptime")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.rsplit("average").next().map(|t| {
                t.trim_start_matches(|c: char| !c.is_ascii_digit())
                    .to_string()
            })
        })
        .and_then(|t| t.split(|c| c == ',' || c == ' ').next().map(str::to_string))
        .and_then(|t| t.parse().ok())
        .unwrap_or(f64::NAN)
}

fn main() {
    let frames: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let f16 = std::env::args().nth(2).map(|s| s == "f16").unwrap_or(false);

    println!(
        "depformer  frames={frames}  weights={}",
        if f16 { "f16" } else { "f32" }
    );
    let load0 = loadavg();

    let mut d = DepthFast::synthetic(f16);
    println!(
        "frame weight bytes: {:.2} MiB",
        d.frame_weight_bytes() as f64 / 1048576.0
    );

    // deterministic transformer output, identical on both machines
    let mut s = 0x2545_F491u32;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s as i32 as f32) / (i32::MAX as f32)
    };
    let x: Vec<f32> = (0..cfg::DIM).map(|_| rnd() * 0.5).collect();
    let forced = [None; cfg::DEP_Q];

    // warm, then drain the counters so warmup is excluded
    for _ in 0..8 {
        let _ = d.frame(&x, 42, &forced, None, None);
    }
    let _ = d.take_bench();

    let t0 = Instant::now();
    for _ in 0..frames {
        let _ = d.frame(&x, 42, &forced, None, None);
    }
    let wall = t0.elapsed().as_secs_f64();
    let (n, total, cond, gemv, head, rest) = d.take_bench();
    let load1 = loadavg();

    println!("\nframes measured  : {n}");
    println!("per frame total  : {total:.3} ms");
    println!("  conditioning   : {cond:.3} ms");
    println!(
        "  stack gemv     : {gemv:.3} ms   ({} sequential steps)",
        16
    );
    println!("  head           : {head:.3} ms");
    println!("  scalar rest    : {rest:.3} ms");
    println!(
        "wall clock       : {wall:.2}s  ({:.3} ms/frame)",
        wall / frames as f64 * 1e3
    );
    println!("\nshare of the 80ms frame budget: {:.1}%", total / 0.8);
    println!("machine load: {load0:.2} -> {load1:.2}");
    if load0 > 2.0 || load1 > 2.0 {
        println!("  !! LOADED -- CPU timings on a contended machine are not a measurement");
    }
}
