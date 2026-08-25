//! `inkling_gemm_autotune_gate` — does TIMING the BF16 lane order pick a
//! different lane than WALKING it, at the widths a speculative verify feeds?
//!
//! [`bf16_gemm`](mary::models::inkling::bf16gemm::bf16_gemm) caches the first
//! lane of a static order that does not return a setup error. Nothing in that
//! measures. This gate asks, per shape and per width, what a timed walk would
//! have chosen instead — and prints BOTH picks on the same line, so the answer
//! to "is the static order already right?" is a column and not an inference.
//!
//! ## What a CHANGED line is, and what it is not
//!
//! It is evidence that the static order is not the timed order at that shape.
//! It is NOT evidence that changing it makes a pass faster. The timing here is
//! an isolated one: the shape had the whole device, and four of a layer's
//! projections are independent and overlap in a real pass. `PREFERENCE_NARROW`'s
//! own doc records a per-shape tuner that won in isolation and lost end to end,
//! and the same trap is open here. The arbiter is `scripts/bench-decode.sh`
//! with an `INK_GEMM_AUTOTUNE=1` arm against a bare one.
//!
//! The weight slab is looped, so it is L2-resident in a way it never is in a
//! pass — see `inkling_bf16_gemm_bench`'s doc, which records two changes that
//! measured positive on exactly this hazard and negative end to end. The tune
//! narrows it (round-robin, median-of-rounds, first round discarded) and cannot
//! close it.
//!
//! Build: `--features inkling-cuda,cuda-backend`
//! Run:   `inkling_gemm_autotune_gate [widths] [--all]`
//!        widths default to `1,2,3,4,5,8,16`; `--all` adds the unembed shape,
//!        whose operands are a 1.6 GB host slab and a minute of `sin`.

use anyhow::Result;
use cubecl::prelude::*;
use half::bf16;

use mary::models::inkling::bf16gemm::{Lane, autotune_params, static_lane, tuned_lane};

type Rt = cubecl::cuda::CudaRuntime;

/// A deterministic BF16 slab. Values near unit scale so nothing denormalises.
/// The same generator `inkling_bf16_gemm_bench` uses, so the two binaries are
/// asking about the same bytes.
fn slab(n: usize, seed: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 2);
    for i in 0..n {
        let f = (i as f32 * 0.017_31 + seed).sin() * 0.5;
        v.extend_from_slice(&bf16::from_f32(f).to_le_bytes());
    }
    v
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let all = args.iter().any(|a| a == "--all");
    // `--core` is the SHORT run: the three shapes that dominate a pass, for a
    // box that is shared. Four agents build and measure on this machine, and a
    // forty-pair sweep is not a short correctness run -- it is a sweep, and it
    // lands in the middle of somebody's p50.
    let core = args.iter().any(|a| a == "--core");
    let widths: Vec<usize> = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 3, 4, 5, 8, 16]);

    // Every BF16 GEMM a 20-layer node issues, by shape.
    let mut shapes: Vec<(&str, usize, usize)> = vec![
        ("attn wq       ", 4096, 4096),
        ("attn wk/wv    ", 4096, 1024),
        ("attn wr       ", 4096, 512),
        ("attn wo       ", 4096, 4096),
        ("shared gate+up", 4096, 2048),
        ("shared down   ", 2048, 4096),
        ("dense w13     ", 4096, 16384),
        ("dense down    ", 8192, 4096),
    ];
    if core {
        shapes.retain(|&(name, _, _)| {
            matches!(name.trim(), "attn wq" | "shared gate+up" | "shared down")
        });
    }
    if all {
        shapes.push(("unembed       ", 4096, 201024));
    }

    let client = Rt::client(&Default::default());
    println!("=== static walk vs timed walk, per shape, per width ===");
    // From the tune itself, never from the env-or-a-literal: a literal in this
    // format string is a framing rule for a build that may not be the one
    // running, and this line printed "iters 2" on a run doing eight.
    let (rounds, iters, margin) = autotune_params();
    println!(
        "  rounds {rounds} (first discarded), iters {iters} per round, margin {:.0}%; \
         round-robin with a rotating start, so cache state AND position are common-mode.",
        margin * 100.0
    );
    println!(
        "  The weight is looped and therefore L2-warm. A CHANGED row is a \
         disagreement, not a speedup.\n"
    );

    let mut changed = 0usize;
    let mut rows = 0usize;
    // Shape outer, width inner. The WEIGHT slab does not depend on `m`, and it
    // is the expensive one -- `dense w13` is 134 MB of host `sin` -- so hoisting
    // it out of the width loop is the difference between one generation per
    // shape and one per shape and width. It also puts the question this binary
    // exists to answer on adjacent lines: for ONE shape, does the pick move as
    // the verify width moves?
    for &(name, k, n) in &shapes {
        let b = client.create_from_slice(&slab(n * k, 1.7));
        println!(
            "  --- {} k {k} n {n}   ({:.1} MB of weight) ---",
            name.trim(),
            (n * k * 2) as f64 / 1e6
        );
        println!(
            "  {:>3}  {:<24} {:<24} {:>9}  {:>9}  {:>7}",
            "m", "static walk", "timed walk", "static us", "timed us", "delta"
        );
        for &m in &widths {
            let a = client.create_from_slice(&slab(m * k, 0.3));
            let stat: Option<Lane> = static_lane::<Rt>(&client, &a, &b, m, k, n);
            let tuned = tuned_lane::<Rt>(&client, &a, &b, m, k, n);
            match (stat, tuned) {
                (Some(s), Some((t, ranked))) => {
                    let us = |l: Lane| {
                        ranked
                            .iter()
                            .find(|(x, _)| *x == l)
                            .map(|(_, secs)| secs * 1e6)
                            .unwrap_or(f64::NAN)
                    };
                    let (ss, ts) = (us(s), us(t));
                    rows += 1;
                    if s != t {
                        changed += 1;
                    }
                    println!(
                        "  {m:>3}  {:<24} {:<24} {ss:>9.1}  {ts:>9.1}  {:>6.1}%{}",
                        s.name(),
                        t.name(),
                        100.0 * (ts - ss) / ss,
                        if s == t { "" } else { "   CHANGED" },
                    );
                }
                (s, t) => {
                    println!("  {m:>3}  static {s:?}, timed {t:?}  (the tune declined this shape)")
                }
            }
        }
        println!();
    }
    println!("  {changed} of {rows} shape/width pairs disagree.");
    println!(
        "  A disagreement is a question for `scripts/bench-decode.sh`, not an answer from it."
    );
    Ok(())
}
