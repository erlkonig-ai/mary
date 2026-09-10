//! Synthetic TP2-shaped cached attention, not a full-model throughput estimate.
//!
//! Compares BF16 expansion on every call with direct NVFP4 reading, and the
//! single-query tile with larger query tiles. Every timed sample includes the
//! reader's allocations, optional dequantization, attention, combine, and one
//! stream synchronization; repeats > 1 report amortized batch time. Uploads,
//! compilation, correctness readbacks, and
//! model projections are excluded. Inputs are reused and may be cache-hot.
//! The timed order rotates and reverses within each sample to balance drift.
//! Packed/dense readers at the SAME tile must agree bit-for-bit; changing the
//! tile also changes key splitting, so that comparison reports numerical error.
//!
//! Example: --keys 65536 --queries 512 --samples 5 --repeats 1 --tiles 4,32
//! No model weights, pile, tokenizer, generation, or training is involved.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use cubecl::prelude::*;
use cubecl::server::Handle;
use mary::models::inkling::flash::{self, KeyRun, KvElem};
use mary::models::inkling::fp4quant::dequantize_nvfp4_bf16;
use serde_json::json;
use std::time::Instant;

type Rt = cubecl::cuda::CudaRuntime;
const HEADS: usize = 16;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const WIDTH: usize = KV_HEADS * HEAD_DIM;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 65536)]
    keys: usize,
    #[arg(long, default_value_t = 512)]
    queries: usize,
    #[arg(long, default_value_t = 5)]
    samples: usize,
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    #[arg(long, value_delimiter = ',', default_value = "4,32")]
    tiles: Vec<usize>,
    #[arg(long)]
    window: Option<usize>,
}

#[derive(Clone, Copy)]
struct Arm {
    packed: bool,
    rows: usize,
}

impl Arm {
    fn name(self) -> String {
        format!("{}_rows{}", if self.packed { "packed" } else { "dense" }, self.rows)
    }
}

fn packed_data(n: usize, seed: u32) -> (Vec<u32>, Vec<u8>) {
    let words = (0..n / 8)
        .map(|i| {
            let mut x = (i as u32).wrapping_add(seed).wrapping_mul(0x9e37_79b9);
            x ^= x >> 16;
            x = x.wrapping_mul(0x85eb_ca6b);
            x ^ (x >> 13)
        })
        .collect();
    // Finite E4M3 scales with nonzero mantissas, not just powers of two.
    const SCALES: [u8; 6] = [0x21, 0x25, 0x29, 0x2d, 0x31, 0x35];
    let scales = (0..n / 16)
        .map(|i| SCALES[(i.wrapping_mul(5).wrapping_add(seed as usize)) % SCALES.len()])
        .collect();
    (words, scales)
}

fn dense_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % 65521) as f32 * 0.01731 + seed).sin() * 0.25).collect()
}

struct Inputs {
    q: Handle,
    rel: Handle,
    kc: Handle,
    ks: Handle,
    vc: Handle,
    vs: Handle,
    capacity: usize,
    live_rows: usize,
    base: usize,
    eff: usize,
}

fn launch(client: &ComputeClient<Rt>, inputs: &Inputs, args: &Args, arm: Arm) -> Handle {
    let dense = if arm.packed {
        None
    } else {
        Some((
            dequantize_nvfp4_bf16(client, &inputs.kc, &inputs.ks, inputs.capacity, WIDTH),
            dequantize_nvfp4_bf16(client, &inputs.vc, &inputs.vs, inputs.capacity, WIDTH),
        ))
    };
    let (k, v, k_scales, v_scales, rows, elem) = match &dense {
        Some((k, v)) => (k, v, None, None, inputs.capacity, KvElem::Bf16),
        None => (
            &inputs.kc, &inputs.vc, Some(&inputs.ks), Some(&inputs.vs),
            inputs.capacity, KvElem::Nvfp4,
        ),
    };
    flash::flash_attention_launch(
        client,
        &inputs.q,
        &[KeyRun {
            k, v, k_scales, v_scales, rows,
            base: inputs.base, lo: 0, hi: inputs.live_rows, row0: 0,
        }],
        &inputs.rel,
        elem,
        args.queries,
        args.keys - args.queries,
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        inputs.eff,
        args.window,
        1.0 / HEAD_DIM as f32,
        arm.rows,
        None,
    )
}

fn differences(reference: &[f32], actual: &[f32]) -> Result<serde_json::Value> {
    ensure!(reference.len() == actual.len(), "output lengths differ");
    let mut max_abs = 0.0f64;
    let mut square_error = 0.0f64;
    let mut square_reference = 0.0f64;
    let mut different_bits = 0usize;
    for (&a, &b) in reference.iter().zip(actual) {
        ensure!(a.is_finite() && b.is_finite(), "nonfinite attention output");
        let error = a as f64 - b as f64;
        max_abs = max_abs.max(error.abs());
        square_error += error * error;
        square_reference += (a as f64) * (a as f64);
        different_bits += usize::from(a.to_bits() != b.to_bits());
    }
    Ok(json!({
        "max_abs": max_abs,
        "relative_rms": (square_error / square_reference.max(1e-30)).sqrt(),
        "different_bits": different_bits,
        "elements": reference.len(),
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!((1..=1_048_576).contains(&args.keys), "keys must be in 1..=1048576");
    ensure!((1..=512).contains(&args.queries) && args.queries <= args.keys,
        "queries must be in 1..=min(keys, 512)");
    ensure!((1..=20).contains(&args.samples) && (1..=100).contains(&args.repeats),
        "samples must be 1..=20 and repeats 1..=100");
    ensure!(!args.tiles.is_empty() && args.tiles.len() <= 8, "provide 1..=8 tile sizes");
    ensure!(args.window != Some(0), "window must be nonzero");
    for (i, &rows) in args.tiles.iter().enumerate() {
        ensure!(flash::applies(HEADS, KV_HEADS, HEAD_DIM, rows), "unsupported tile {rows}");
        ensure!(!args.tiles[..i].contains(&rows), "duplicate tile {rows}");
    }
    let arms: Vec<Arm> = args.tiles.iter().flat_map(|&rows| {
        [Arm { packed: false, rows }, Arm { packed: true, rows }]
    }).collect();
    let client = Rt::client(&Default::default());
    // A local layer keeps enough keys for the earliest query in the batch,
    // not the whole absolute context followed by a mask. Head slack is zero
    // here; fixed-cache read extents still round to the 512-row epoch.
    let live_rows = match args.window {
        Some(window) => args.keys.min(window.checked_add(args.queries - 1)
            .context("window plus query width overflows")?),
        None => args.keys,
    };
    let capacity = live_rows.next_multiple_of(512);
    let eff = args.keys.min(args.window.unwrap_or(1024));
    let upload = |seed| {
        let (words, scales) = packed_data(capacity * WIDTH, seed);
        (client.create_from_slice(u32::as_bytes(&words)), client.create_from_slice(&scales))
    };
    let (kc, ks) = upload(1);
    let (vc, vs) = upload(7);
    let inputs = Inputs {
        q: client.create_from_slice(f32::as_bytes(&dense_data(args.queries * HEADS * HEAD_DIM, 0.1))),
        rel: client.create_from_slice(f32::as_bytes(&dense_data(args.queries * HEADS * eff, 0.7))),
        kc, ks, vc, vs, capacity, live_rows, base: args.keys - live_rows, eff,
    };
    println!("{}", json!({
        "event": "framing", "scope": "one synthetic TP2-shaped cached attention layer-call",
        "absolute_context": args.keys, "live_keys": live_rows,
        "key_base": inputs.base, "physical_and_dense_read_rows": capacity,
        "queries": args.queries,
        "heads": HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM, "eff": eff,
        "window": args.window, "samples": args.samples, "repeats": args.repeats,
        "dense_kv_bytes_per_call": 2 * capacity * WIDTH * 2,
        "packed_kv_backing_bytes": 2 * capacity * WIDTH * 9 / 16,
        "includes": "optional dequant, allocations, flash, combine, stream sync",
        "excludes": "uploads, compilation, readbacks, projections, model and TP collectives",
        "cache_state": "reused inputs, possibly cache-hot",
        "cache_geometry": "one retained run, no dead-prefix slack, 512-row read epoch; global head geometry",
        "timing": if args.repeats == 1 { "synchronized call latency" }
            else { "amortized enqueue batch plus one synchronization" },
        "arms": arms.iter().map(|a| a.name()).collect::<Vec<_>>(),
    }));
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for arm in &arms {
        let out = launch(&client, &inputs, &args, *arm);
        let bytes = client.read_one(out).context("read correctness output")?;
        let got = f32::from_bytes(&bytes).to_vec();
        ensure!(got.len() == args.queries * HEADS * HEAD_DIM, "wrong output size");
        ensure!(got.iter().any(|x| x.abs() > 1e-6), "degenerate all-zero output");
        let diff = differences(outputs.first().unwrap_or(&got), &got)?;
        if arm.packed {
            let paired = differences(outputs.last().expect("dense partner"), &got)?;
            ensure!(paired["different_bits"] == 0, "packed reader mismatch: {paired}");
        }
        println!("{}", json!({"event": "correctness", "arm": arm.name(),
            "against": arms[0].name(), "difference": diff,
            "blake3": blake3::hash(f32::as_bytes(&got)).to_hex().to_string()}));
        outputs.push(got);
    }
    drop(outputs);
    let mut times = vec![Vec::<f64>::new(); arms.len()];
    for sample in 0..args.samples {
        let mut order: Vec<usize> = (0..arms.len()).map(|i| (i + sample) % arms.len()).collect();
        order.extend(order.clone().into_iter().rev());
        for (position, index) in order.into_iter().enumerate() {
            cubecl::future::block_on(client.sync()).context("drain before timing")?;
            let started = Instant::now();
            let mut last = None;
            for _ in 0..args.repeats {
                last = Some(launch(&client, &inputs, &args, arms[index]));
            }
            cubecl::future::block_on(client.sync()).context("synchronize timed work")?;
            let seconds = started.elapsed().as_secs_f64() / args.repeats as f64;
            drop(last);
            times[index].push(seconds);
            println!("{}", json!({"event": "sample", "sample": sample,
                "position": position, "arm": arms[index].name(), "seconds_per_call": seconds}));
        }
    }
    for (arm, samples) in arms.iter().zip(&mut times) {
        samples.sort_by(f64::total_cmp);
        let median = (samples[(samples.len() - 1) / 2] + samples[samples.len() / 2]) / 2.0;
        println!("{}", json!({"event": "summary", "arm": arm.name(),
            "timed_batches": samples.len(), "calls": samples.len() * args.repeats,
            "median_seconds_per_call": median, "min_seconds_per_call": samples[0],
            "max_seconds_per_call": samples[samples.len() - 1]}));
    }
    Ok(())
}
