//! Does bf16 weight storage let 93 layers fit? Attempt six, with all five
//! previous confounds designed out.
//!
//! What defeated each earlier attempt, and how this avoids it:
//!   1. nvidia-smi reports no used-memory figure on this unified-memory part
//!      -> read /proc/meminfo instead.
//!   2. RSS deltas across sequential probes are absorbed by allocator reuse
//!      -> one measurement series, monotonically growing, never freeing.
//!   3. MemAvailable moves with PAGE CACHE while reading a multi-GiB
//!      checkpoint -> track Cached explicitly and subtract it, so what is
//!      reported is ANONYMOUS allocation (tensors) rather than file cache.
//!   4. Burn allocates lazily, so RSS read straight after construction
//!      measures nothing -> every tensor is forced with a reduction.
//!   5. Summing bf16 ones saturates near 2^27 and looks like a memory failure
//!      -> force with max(), which touches every element and cannot saturate.
//!
//! And the failure that was worse than any of them: a probe once printed
//! "BF16 storage is therefore NARROW" while the allocation had panicked,
//! because the verdict was an unconditional println rather than a computed
//! result. Here every conclusion is derived from the measured bytes, the
//! per-tensor check is asserted before it is trusted, and a partial run
//! reports what it got rather than what it hoped.

use burn::backend::Cuda;
use burn::prelude::*;
use burn::tensor::FloatDType;
use mary::models::k3::ckpt::Ckpt;
use mary::models::k3::K3Config;

type B = Cuda<f32>;

/// (MemFree, Cached+Buffers) in bytes.
fn meminfo() -> (f64, f64) {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let get = |k: &str| -> f64 {
        for line in s.lines() {
            if let Some(v) = line.strip_prefix(k) {
                if let Some(kb) = v
                    .split_whitespace()
                    .next()
                    .and_then(|x| x.parse::<f64>().ok())
                {
                    return kb * 1024.0;
                }
            }
        }
        0.0
    };
    (get("MemFree:"), get("Cached:") + get("Buffers:"))
}

const GIB: f64 = (1u64 << 30) as f64;

fn main() {
    let model = mary::paths::model(std::env::var("K3_MODEL_DIR").ok().as_deref(), "kimi-k3")
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
    let n_layers: usize = std::env::var("K3_PROBE_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let cfg: K3Config = serde_json::from_str(
        &std::fs::read_to_string(model.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let t = &cfg.text_config;
    let ck = Ckpt::open(&model);
    let dev = Device::<B>::default();

    // Tensor names from the index. NOTE the trailing dot on the prefix: an
    // earlier probe matched "layers.1" against layers.10..19 and reported
    // attention as 8 GiB/layer against a known 0.72.
    let mut names: Vec<String> = Vec::new();
    for e in std::fs::read_dir(&model).expect("read model dir").flatten() {
        let p = e.path();
        if p.to_string_lossy().ends_with(".safetensors.index.json") {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            if let Some(m) = v.get("weight_map").and_then(|m| m.as_object()) {
                names.extend(m.keys().cloned());
            }
        }
    }
    names.sort();

    println!("holding REAL K3 attention weights as BF16 on a Cuda<f32> backend");
    println!("  layers: {n_layers} of {}\n", t.num_hidden_layers);

    let (free0, cache0) = meminfo();
    let mut held: Vec<Tensor<B, 2>> = Vec::new();
    let mut elems: u64 = 0;
    let mut disk: u64 = 0;

    for layer in 0..n_layers {
        let prefix = format!("language_model.model.layers.{layer}.");
        for name in names.iter().filter(|n| n.starts_with(&prefix)) {
            if !(name.contains("attn") || name.contains("kda")) {
                continue;
            }
            let (dt, shape, bytes) = ck.raw(name);
            if dt != "BF16" || shape.len() != 2 {
                continue;
            }
            disk += bytes.len() as u64;
            let vals: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let n = vals.len();
            let tt = Tensor::<B, 2>::from_data(TensorData::new(vals, [shape[0], shape[1]]), &dev)
                .cast(FloatDType::BF16);
            // Force materialisation with max: touches every element, cannot
            // saturate the way a sum of many values would.
            let m: f32 = tt.clone().max().into_scalar().elem();
            assert!(
                m.is_finite(),
                "{name}: max is not finite — tensor did not materialise"
            );
            held.push(tt);
            elems += n as u64;
        }
        if layer % 3 == 2 || layer == n_layers - 1 {
            let (free, cache) = meminfo();
            let anon = (free0 - free) - (cache - cache0);
            println!(
                "  through L{layer:02}  {:>10} elems  disk {:6.2} GiB  anon {:6.2} GiB  {:.3} B/elem",
                elems,
                disk as f64 / GIB,
                anon / GIB,
                anon / elems as f64
            );
        }
    }

    let (free1, cache1) = meminfo();
    let anon = (free1_placeholder(free0, free1)) - (cache1 - cache0);
    let per = anon / elems as f64;
    println!("\n  tensors held    : {}", held.len());
    println!("  elements        : {elems}");
    println!("  disk (BF16)     : {:.2} GiB", disk as f64 / GIB);
    println!(
        "  anonymous mem   : {:.2} GiB  (MemFree drop minus Cached growth)",
        anon / GIB
    );
    println!("  measured        : {per:.3} B/elem");

    // Verdict computed, never asserted.
    let verdict = if per < 3.0 {
        "NARROW (~2 B/elem)"
    } else {
        "WIDE (~4 B/elem)"
    };
    println!("  VERDICT         : {verdict}");
    println!(
        "  => 106.55 GiB of resident weights would cost {:.1} GiB against 119 GiB",
        106.55 * per / 2.0
    );
    drop(held);
}

fn free1_placeholder(free0: f64, free1: f64) -> f64 {
    free0 - free1
}
