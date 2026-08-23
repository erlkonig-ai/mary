//! `inkling_align_gate` — would EVERY weight plane in this source alias, or
//! would some of them be copied, and why?
//!
//! The runtime answer to that question is a sample: a forward pass binds the
//! planes the router happened to route to, so a five-token run touches a few
//! hundred of the 41 928 planes the model has. That is enough to notice a
//! catastrophe and not enough to certify anything. This walks ALL of them.
//!
//! It asks through [`Aliases::classify`] — the same predicate
//! [`Aliases::slice_or_copy`] binds on, not a second transcription of the
//! 4-byte rule — so "every plane aliases" here and "every bind aliased" in a
//! forward are statements about one function.
//!
//! # Why the layout is the DATA
//!
//! safetensors packed tensors back to back with no padding, so where a plane
//! landed was whatever the sum of the preceding tensors' lengths happened to
//! be. A pile does not: a V3 record's data begins at `record_start + 256` with
//! every record a 256-byte multiple, and a tensor blob's header is exactly 256
//! wide, so a leaf's payload is at an absolutely 256-aligned file offset —
//! by construction, for every leaf, forever.
//!
//! That was the whole of the difference, and the checkpoint side of it is now
//! history: the runtime loads from a pile and nothing else, so this gate does
//! too. What keeps it falsifiable is `--mutate N`, not the comparison it used
//! to be able to run.
//!
//!   inkling_align_gate <pile> [--mutate N] [--limit N]
//!
//! `--mutate N` shifts every plane N bytes forward before classifying it. It
//! exists so this gate can be seen to FAIL: a check that has never failed is
//! not evidence, and N=1..3 is exactly the corruption it claims to detect.
//!
//! Build: `--features inkling-cuda,cuda-backend,import`

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};

use mary::models::inkling::fp4gemm::{Aliases, Bind};
use mary::models::inkling::source::Weights;

type Rt = cubecl::cuda::CudaRuntime;

/// Per-plane-class tallies. Ordered, so the report is stable run to run.
#[derive(Default)]
struct Tally {
    planes: u64,
    bytes: u64,
    alias: u64,
    unaligned: BTreeMap<usize, u64>,
    unmapped: u64,
    /// Alignment the plane's ADDRESS actually achieved: the largest power of
    /// two up to 256 that divides it. Reported because "4-aligned" is the
    /// binding bound but not the interesting fact — if every plane is 256, the
    /// property has headroom and a future kernel wanting 16-byte vector loads
    /// costs nothing.
    align: BTreeMap<usize, u64>,
    /// One example of a plane that did NOT alias, for the error message.
    worst: Option<(String, usize)>,
}

impl Tally {
    fn note(&mut self, name: &str, data: &[u8], kind: Bind) {
        self.planes += 1;
        self.bytes += data.len() as u64;
        let p = data.as_ptr() as usize;
        let mut a = 1usize;
        while a < 256 && p % (a * 2) == 0 {
            a *= 2;
        }
        *self.align.entry(a).or_default() += 1;
        match kind {
            Bind::Alias => self.alias += 1,
            Bind::CopyUnaligned(r) => {
                *self.unaligned.entry(r).or_default() += 1;
                self.worst.get_or_insert_with(|| (name.to_string(), p));
            }
            Bind::CopyUnmapped => {
                self.unmapped += 1;
                self.worst.get_or_insert_with(|| (name.to_string(), p));
            }
            // An empty plane binds nothing; it is not a failure and not a pass.
            Bind::Empty => self.planes -= 1,
        }
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .context("usage: inkling_align_gate <pile> [--mutate N] [--limit N]")?;
    let (mut mutate, mut limit) = (0usize, usize::MAX);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--mutate" => mutate = args.next().context("--mutate needs N")?.parse()?,
            "--limit" => limit = args.next().context("--limit needs N")?.parse()?,
            other => anyhow::bail!("unexpected argument {other:?}"),
        }
    }

    let t0 = std::time::Instant::now();
    let src = Weights::open(&path)?;
    println!("=== alignment gate ===");
    println!("  source     : pile {}", path.display());
    println!("  {}", src.inventory());
    println!("  index built in {:.1}s", t0.elapsed().as_secs_f64());

    let client = <Rt as cubecl::prelude::Runtime>::client(&Default::default());
    let maps = src.mappings()?;
    let n_maps = maps.len();
    let map_bytes: usize = maps.iter().map(|(_, l, _)| *l).sum();
    let al = Aliases::register(&client, maps)
        .context("this device cannot address host memory directly — nothing can alias")?;
    println!(
        "  registered : {n_maps} mapping(s), {:.2} GiB",
        map_bytes as f64 / (1u64 << 30) as f64
    );
    if mutate > 0 {
        println!("  MUTATION   : every plane shifted {mutate} byte(s) forward");
    }
    println!();

    let mut by_kind: BTreeMap<&'static str, Tally> = BTreeMap::new();
    let mut seen = 0usize;
    let mut stopped = false;
    let r = src.for_each_bindable(|name, kind, data| {
        if seen >= limit {
            stopped = true;
            // A sentinel, not a failure: `for_each_bindable` has no early exit
            // and inventing one for a debug flag is not worth the surface.
            anyhow::bail!("__limit__");
        }
        seen += 1;
        // The mutation is applied to the SLICE, so it moves the pointer — which
        // is the thing classified. Shortening from the front is exactly how a
        // one-byte layout slip would present.
        let data = if mutate > 0 && data.len() > mutate {
            &data[mutate..]
        } else {
            data
        };
        by_kind
            .entry(kind)
            .or_default()
            .note(name, data, al.classify(data));
        Ok(())
    });
    match r {
        Ok(()) => {}
        Err(e) if format!("{e}") == "__limit__" => {}
        Err(e) => return Err(e),
    }
    if stopped {
        println!("  (stopped at --limit {limit})");
    }

    println!(
        "  {:<20} {:>8} {:>12} {:>8}  {}",
        "plane class", "planes", "GiB", "aliased", "address alignment"
    );
    let (mut planes, mut aliased, mut bytes) = (0u64, 0u64, 0u64);
    for (kind, t) in &by_kind {
        let aligns: Vec<String> = t.align.iter().map(|(a, n)| format!("{n}x{a}B")).collect();
        println!(
            "  {:<20} {:>8} {:>12.2} {:>7.1}%  {}",
            kind,
            t.planes,
            t.bytes as f64 / (1u64 << 30) as f64,
            if t.planes == 0 {
                f64::NAN
            } else {
                t.alias as f64 * 100.0 / t.planes as f64
            },
            aligns.join(" ")
        );
        planes += t.planes;
        aliased += t.alias;
        bytes += t.bytes;
    }
    println!();
    println!(
        "  TOTAL      : {aliased} of {planes} planes alias ({:.2}%), {:.2} GiB",
        if planes == 0 {
            f64::NAN
        } else {
            aliased as f64 * 100.0 / planes as f64
        },
        bytes as f64 / (1u64 << 30) as f64
    );

    // Why the failures failed, in the terms that name the repair.
    let mut failed = false;
    for (kind, t) in &by_kind {
        for (r, n) in &t.unaligned {
            failed = true;
            println!("  {kind}: {n} plane(s) COPY -- address % 4 == {r}");
        }
        if t.unmapped > 0 {
            failed = true;
            println!(
                "  {kind}: {} plane(s) COPY -- outside every registered mapping",
                t.unmapped
            );
        }
        if let Some((name, p)) = &t.worst {
            println!("       e.g. {name} at {p:#x}");
        }
    }

    if failed || planes == 0 {
        println!("\nFAIL — not every plane can be aliased.");
        std::process::exit(1);
    }
    println!(
        "\nPASS — all {planes} planes are 4-byte aligned inside a registered \
         mapping, so every weight bind of a full pass aliases."
    );
    Ok(())
}
