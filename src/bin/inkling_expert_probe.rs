//! What ONE routed expert costs, in parts — and the gate on the fused decode.
//!
//! The routed lane is 833 expert loads in a five-token forward and 240 in a
//! cached decode step, so everything per-expert is multiplied by three orders
//! of magnitude and nothing else about it matters. A measured forward put the
//! lane at 17.4 s: 5.2 s slicing from the mapping, 11.6 s widening and
//! uploading, 0.5 s of enqueue and sync. That is 20.9 ms an expert of which
//! 20.4 is host time the device never waits on.
//!
//! This binary times the SAME work both ways in one process and one clock:
//! the mapping opened per call versus opened once, the Burn dequant chain
//! versus one fused kernel. It also gates them — the fused kernel is BITWISE
//! identical to the chain, which is a claim a tolerance would hide.
//!
//!   inkling_expert_probe <ckpt> [layer] [experts] [tokens]

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

use burn::prelude::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use mary::models::inkling::burn::{
    deinterleave_rows_device, expert_ffn, expert_weight_from_packed,
};
use mary::models::inkling::config::InklingConfig;
use mary::models::inkling::dequant_cuda::expert_weight_fused;
use mary::models::inkling::load::Checkpoint;

type Bk = burn::backend::Cuda<f32>;

/// One expert's packed bytes, read the way the lane read them before the
/// mapping was cached: open, mmap, parse the shard header, copy the slice out,
/// unmap. Four of these per weight, eight per expert.
///
/// Kept here rather than in the library so the before/after is one binary and
/// one clock — a comparison across two builds measures the build too.
struct Uncached {
    dir: PathBuf,
    shard_of: HashMap<String, String>,
}

impl Uncached {
    fn open(dir: &PathBuf) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("model.safetensors.index.json"))?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let map = v.get("weight_map").and_then(|m| m.as_object()).context("weight_map")?;
        Ok(Uncached {
            dir: dir.clone(),
            shard_of: map
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect(),
        })
    }

    fn with_view<R>(
        &self,
        name: &str,
        f: impl FnOnce(&[u8], &[usize]) -> Result<R>,
    ) -> Result<R> {
        let shard = self.shard_of.get(name).with_context(|| format!("{name} not in index"))?;
        let file = std::fs::File::open(self.dir.join(shard))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let mmap = unsafe { Mmap::map(&file) }?;
        let st = SafeTensors::deserialize(&mmap)?;
        let v = st.tensor(name)?;
        f(v.data(), v.shape())
    }

    /// The old `expert_slice_packed`, verbatim in its costs.
    fn packed(&self, base: &str, e: usize) -> Result<(Vec<u8>, Vec<u8>, f32, usize, usize)> {
        let shape = self.with_view(base, |_, s| Ok(s.to_vec()))?;
        let (experts, rows, cols) = (shape[0], shape[1], shape[2]);
        let scales_per_row = cols * 2 / 16;
        let scale2 = self.with_view(&format!("{base}.scale2"), |d, _| {
            Ok(d.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>())
        })?;
        anyhow::ensure!(scale2.len() == experts, "scale2 is {}", scale2.len());
        let codes =
            self.with_view(base, |d, _| Ok(d[e * rows * cols..(e + 1) * rows * cols].to_vec()))?;
        let scales = self.with_view(&format!("{base}.scale"), |d, _| {
            let s0 = e * rows * scales_per_row;
            Ok(d[s0..s0 + rows * scales_per_row].to_vec())
        })?;
        Ok((codes, scales, scale2[e], rows, cols))
    }
}

fn ms(t: f64, n: usize) -> String {
    format!("{:7.3}", t * 1e3 / n as f64)
}

fn main() -> Result<()> {
    let ckpt: PathBuf = std::env::args().nth(1).map(PathBuf::from).context("usage: <ckpt> [layer] [experts] [tokens]")?;
    let layer: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(3);
    let n_experts: usize = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(21);
    let tokens: usize = std::env::args().nth(4).and_then(|v| v.parse().ok()).unwrap_or(5);

    let cfg: InklingConfig =
        InklingConfig::from_json(&std::fs::read_to_string(ckpt.join("config.json"))?)?;
    let t = &cfg.text_config;
    let h = t.hidden_size;
    let inter = t.intermediate_size;

    let cp = Checkpoint::open(&ckpt)?;
    let un = Uncached::open(&ckpt)?;
    let p = format!("model.llm.layers.{layer}.");
    let n13 = format!("{p}mlp.experts.w13_weight");
    let n2 = format!("{p}mlp.experts.w2_weight");
    anyhow::ensure!(cp.is_nvfp4(&n13), "layer {layer} is not NVFP4 — pick another");

    let dev = burn::backend::cuda::CudaDevice::default();
    let sync = || <Bk as Backend>::sync(&dev).expect("device sync");

    println!("=== one expert, in parts ===");
    println!("  checkpoint {}", ckpt.display());
    println!("  layer {layer}   experts {n_experts}   tokens {tokens}   hidden {h}   intermediate {inter}");
    let r0 = cp.expert_packed_ref(&n13, 0)?;
    let r1 = cp.expert_packed_ref(&n2, 0)?;
    let bytes = r0.codes().len() + r0.scales().len() + r1.codes().len() + r1.scales().len();
    println!(
        "  w13 {}x{} packed + {} scale bytes, w2 {}x{} + {}  =  {:.2} MB an expert",
        r0.rows, r0.cols, r0.scales().len(), r1.rows, r1.cols, r1.scales().len(),
        bytes as f64 / 1e6
    );

    // ---- the decode, both ways, bitwise ------------------------------------
    //
    // The fused kernel keeps the chain's multiply order — (fp4 * block) *
    // scale2 in f32 — so agreement is exact, not approximate. Checked on the
    // whole 16.7 M-value weight, not a sample: a wrong nibble order or scale
    // stride is a small, localised difference that a sample can miss.
    println!("\n=== gate: fused kernel vs the Burn chain ===");
    for (name, base, permute) in [("w13", &n13, true), ("w2", &n2, false)] {
        for e in [0usize, 7, 200] {
            let q = cp.expert_packed_ref(base, e)?;
            let chain = expert_weight_from_packed::<Bk>(
                q.codes(), q.scales(), q.scale2, q.rows, q.cols, &dev,
            );
            let chain = if permute { deinterleave_rows_device(chain) } else { chain };
            let fused = expert_weight_fused(
                q.codes(), q.scales(), q.scale2, q.rows, q.cols, permute, &dev,
            );
            anyhow::ensure!(chain.dims() == fused.dims(), "{name}: dims differ");
            let a = chain.into_data().convert::<f32>().to_vec::<f32>().expect("chain");
            let b = fused.into_data().convert::<f32>().to_vec::<f32>().expect("fused");
            let diff = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            let (mut worst, mut at) = (0f32, 0usize);
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                if (x - y).abs() > worst {
                    worst = (x - y).abs();
                    at = i;
                }
            }
            anyhow::ensure!(
                diff == 0,
                "{name} expert {e}: {diff} of {} values differ, worst {worst:e} at {at} ({} vs {})",
                a.len(), a[at], b[at]
            );
            println!("  {name} expert {e:3}: {} values BITWISE identical", a.len());
        }
    }

    // ---- host: the mapping opened per call, versus opened once -------------
    println!("\n=== host, per expert (ms) ===");
    let experts: Vec<usize> = (0..n_experts).collect();
    // Warm both: the first touch of a shard faults its header in, and the
    // question is the steady-state cost, not the first one.
    for &e in &experts {
        let _ = un.packed(&n13, e)?;
        let _ = cp.expert_packed_ref(&n13, e)?;
    }

    let t0 = Instant::now();
    let mut keep = 0usize;
    for &e in &experts {
        let (c, s, _, _, _) = un.packed(&n13, e)?;
        let (c2, s2, _, _, _) = un.packed(&n2, e)?;
        keep += c.len() + s.len() + c2.len() + s2.len();
    }
    let t_uncached = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let mut keep2 = 0usize;
    for &e in &experts {
        let a = cp.expert_packed_ref(&n13, e)?;
        let b = cp.expert_packed_ref(&n2, e)?;
        keep2 += a.codes().len() + a.scales().len() + b.codes().len() + b.scales().len();
    }
    let t_cached = t0.elapsed().as_secs_f64();
    anyhow::ensure!(keep == keep2, "the two host paths read different byte counts");
    println!("  open+mmap+parse+copy per call : {}   (8 opens, 12.6 MB copied)", ms(t_uncached, n_experts));
    println!("  mapping cached, bytes borrowed: {}   (no copy at all)", ms(t_cached, n_experts));

    // ---- device: the chain versus the fused kernel --------------------------
    //
    // Both loops sync per expert, so neither can hide behind the other's
    // enqueueing, and both start from bytes already in hand — this is the
    // decode and its upload, not the slice.
    println!("\n=== device, per expert (ms, synced) ===");
    let mut refs = Vec::new();
    for &e in &experts {
        refs.push((cp.expert_packed_ref(&n13, e)?, cp.expert_packed_ref(&n2, e)?));
    }

    for pass in 0..2 {
        let t0 = Instant::now();
        for (a, b) in &refs {
            let w13 = expert_weight_from_packed::<Bk>(a.codes(), a.scales(), a.scale2, a.rows, a.cols, &dev);
            let w13 = deinterleave_rows_device(w13);
            let w2 = expert_weight_from_packed::<Bk>(b.codes(), b.scales(), b.scale2, b.rows, b.cols, &dev);
            std::hint::black_box((&w13, &w2));
            sync();
        }
        let chain = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        for (a, b) in &refs {
            let w13 = expert_weight_fused(a.codes(), a.scales(), a.scale2, a.rows, a.cols, true, &dev);
            let w2 = expert_weight_fused(b.codes(), b.scales(), b.scale2, b.rows, b.cols, false, &dev);
            std::hint::black_box((&w13, &w2));
            sync();
        }
        let fused = t0.elapsed().as_secs_f64();

        // Upload alone: the same bytes to the device, no decode. The floor
        // every decode path shares.
        let t0 = Instant::now();
        for (a, b) in &refs {
            mary::models::inkling::dequant_cuda::upload_only(a.codes(), a.scales(), &dev);
            mary::models::inkling::dequant_cuda::upload_only(b.codes(), b.scales(), &dev);
            sync();
        }
        let up = t0.elapsed().as_secs_f64();

        if pass == 1 {
            println!("  Burn chain  (46 launches/weight): {}", ms(chain, n_experts));
            println!("  fused kernel (1 launch/weight)  : {}", ms(fused, n_experts));
            println!("  upload only, no decode          : {}", ms(up, n_experts));
        }
    }

    // ---- what the upload is actually made of --------------------------------
    //
    // `create_from_slice` copies the slice into a `Vec`, then `do_create_from_
    // slices` copies THAT into `Bytes` — two full host copies of 14.2 MB before
    // a byte moves to the device. This times one such copy against the whole
    // upload, so the driver's share is what is left rather than what is assumed.
    println!("\n=== upload anatomy, per expert (ms) ===");
    let t0 = Instant::now();
    let mut sunk = 0usize;
    for (a, b) in &refs {
        let c1 = a.codes().to_vec();
        let s1 = a.scales().to_vec();
        let c2 = b.codes().to_vec();
        let s2 = b.scales().to_vec();
        sunk += c1.len() + s1.len() + c2.len() + s2.len();
        std::hint::black_box((&c1, &s1, &c2, &s2));
    }
    let memcpy = t0.elapsed().as_secs_f64();
    println!(
        "  ONE host copy of the bytes      : {}   ({:.1} GB/s)",
        ms(memcpy, n_experts),
        sunk as f64 / 1e9 / memcpy
    );
    let t0 = Instant::now();
    for (a, b) in &refs {
        mary::models::inkling::dequant_cuda::upload_only(a.codes(), a.scales(), &dev);
        mary::models::inkling::dequant_cuda::upload_only(b.codes(), b.scales(), &dev);
        sync();
    }
    let up = t0.elapsed().as_secs_f64();
    println!("  create_from_slice + sync        : {}", ms(up, n_experts));
    println!(
        "  => driver + DMA, by difference  : {}   ({:.1} GB/s)",
        ms(up - 2.0 * memcpy, n_experts),
        sunk as f64 / 1e9 / (up - 2.0 * memcpy).max(1e-9)
    );

    // ---- the whole per-expert lane -----------------------------------------
    //
    // Slice, decode, gather this expert's tokens, both matmuls, scatter back:
    // what the forward actually pays, times 833.
    println!("\n=== whole lane, per expert (ms, synced) ===");
    let hn: Vec<f32> = (0..tokens * h).map(|i| ((i % 97) as f32 - 48.0) / 64.0).collect();
    let hn_dev: Tensor<Bk, 2> = Tensor::from_data(TensorData::new(hn, [tokens, h]), &dev);
    // Two tokens an expert is the measured average for a five-token prefill
    // (833 loads for 40 layers x 30 (token, expert) pairs).
    let toks: Vec<i32> = (0..tokens.min(2) as i32).collect();
    let k = toks.len();

    for (label, fused_path) in [("chain", false), ("fused", true)] {
        let t0 = Instant::now();
        let mut acc: Tensor<Bk, 2> = Tensor::zeros([tokens, h], &dev);
        for &e in &experts {
            let a = cp.expert_packed_ref(&n13, e)?;
            let b = cp.expert_packed_ref(&n2, e)?;
            let (gu, dn) = if fused_path {
                (
                    expert_weight_fused(a.codes(), a.scales(), a.scale2, a.rows, a.cols, true, &dev),
                    expert_weight_fused(b.codes(), b.scales(), b.scale2, b.rows, b.cols, false, &dev),
                )
            } else {
                (
                    deinterleave_rows_device(expert_weight_from_packed::<Bk>(
                        a.codes(), a.scales(), a.scale2, a.rows, a.cols, &dev,
                    )),
                    expert_weight_from_packed::<Bk>(
                        b.codes(), b.scales(), b.scale2, b.rows, b.cols, &dev,
                    ),
                )
            };
            let idx: Tensor<Bk, 1, Int> =
                Tensor::from_data(TensorData::new(toks.clone(), [k]), &dev);
            let xs = hn_dev.clone().select(0, idx.clone());
            let ys = expert_ffn(xs, gu, dn);
            acc = acc.select_assign(0, idx, ys, burn::tensor::IndexingUpdateOp::Add);
        }
        let out = acc.into_data().convert::<f32>().to_vec::<f32>().expect("acc");
        let el = t0.elapsed().as_secs_f64();
        let rms = (out.iter().map(|v| (v * v) as f64).sum::<f64>() / out.len() as f64).sqrt();
        println!("  {label}: {} per expert   ({:.2}s for {n_experts}, rms {rms:.5})", ms(el, n_experts), el);
    }

    println!("\n  (a five-token forward loads 833 experts; a cached decode step 240)");
    Ok(())
}
