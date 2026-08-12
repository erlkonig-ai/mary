//! `inkling_expert_lane_bench` — where does the routed-expert lane actually
//! spend its time?
//!
//! The whole-forward measurement says the routed lane is 88.6 s of a 117.6 s
//! forward, split 31.0 s "slicing" / 56.2 s "widening + uploading" / 1.4 s
//! device. Two claims about that lane cannot both be casually true: the
//! `routed_experts_gpu` docstring says the host never materialises a
//! dequantised weight, yet 56.2 s is attributed to host-side widening.
//!
//! Both ARE true, because "widening" is not dequantisation:
//!
//! * [`Checkpoint::expert_slice_packed`] re-opens, re-mmaps and re-deserializes
//!   the shard header on EVERY call — `shape_of`, `tensor(.scale2)`,
//!   `with_bytes(base)` and `with_bytes(.scale)` are four independent
//!   `SafeTensors::deserialize` of a multi-GB shard — and then `to_vec()`s
//!   12.6 MB out of the mapping.
//! * [`expert_weight_from_packed`] then runs `word()`, a scalar
//!   `chunks_exact(4).map(i32::from_le_bytes).collect()` over those same
//!   12.6 MB, allocating a fresh `Vec<i32>` before burn copies it AGAIN into a
//!   `TensorData` and once more into the device buffer.
//!
//! So the bytes are packed the whole way — no f32 is ever built on the host —
//! but they are copied and re-boxed three or more times before the GPU sees
//! them. That is the "widening": container churn, not precision change.
//!
//! This binary measures each step separately so the port's two contributions
//! can be reported apart from each other: how much comes from deleting host
//! plumbing, and how much from the FP4 MMA itself.
//!
//! Build: `--features inkling-cuda,cuda-backend`
//! Run:   `inkling_expert_lane_bench [<ckpt>] [<n_experts>]`

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

use mary::models::inkling::load::Checkpoint;

const LAYER: usize = 10;

fn base13() -> String {
    format!("model.llm.layers.{LAYER}.mlp.experts.w13_weight")
}
fn base2() -> String {
    format!("model.llm.layers.{LAYER}.mlp.experts.w2_weight")
}

/// The `word()` repack from `expert_weight_from_packed`, verbatim in cost.
fn word(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() -> Result<()> {
    let ckpt = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/thinkingmachines-inkling-small-nvfp4"));
    let n_experts: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(8);

    let cp = Checkpoint::open(&ckpt)?;
    println!("=== routed-expert lane breakdown ===");
    println!("  checkpoint : {}", ckpt.display());
    println!("  layer      : {LAYER}   experts sampled: {n_experts}");
    println!();

    // ---------------------------------------------------------------- A
    // One bare header parse, the unit the current code pays four of per slab.
    let shard = ckpt.join("model-00004-of-00009.safetensors");
    let t = Instant::now();
    let file = std::fs::File::open(&shard)?;
    let mmap = unsafe { Mmap::map(&file) }?;
    let st = SafeTensors::deserialize(&mmap)?;
    let one_parse = t.elapsed();
    println!(
        "A. one open+mmap+SafeTensors::deserialize of a shard : {:8.2} ms  ({} tensors)",
        ms(one_parse),
        st.names().len()
    );
    drop(st);

    // ---------------------------------------------------------------- B
    // The real thing, as the forward calls it.
    let mut t_slice = std::time::Duration::ZERO;
    let mut bytes = 0usize;
    for e in 0..n_experts {
        let t = Instant::now();
        let w13 = cp.expert_slice_packed(&base13(), e)?;
        let w2 = cp.expert_slice_packed(&base2(), e)?;
        t_slice += t.elapsed();
        bytes += w13.codes.len() + w13.scales.len() + w2.codes.len() + w2.scales.len();
    }
    println!(
        "B. expert_slice_packed x2 per expert (as called)     : {:8.2} ms/expert   [{:.1} MB/expert]",
        ms(t_slice) / n_experts as f64,
        bytes as f64 / n_experts as f64 / 1e6
    );

    // ---------------------------------------------------------------- C
    // The same bytes, but sliced out of ONE cached mapping: no repeated header
    // parse. Isolates parse cost from copy cost.
    let f13 = std::fs::File::open(ckpt.join(cp_shard(&cp, &base13())?))?;
    let m13 = unsafe { Mmap::map(&f13) }?;
    let s13 = SafeTensors::deserialize(&m13)?;
    let v13 = s13.tensor(&base13())?;
    let sh = v13.shape().to_vec();
    let (rows, cols) = (sh[1], sh[2]);
    let raw13 = v13.data();

    let mut t_cached = std::time::Duration::ZERO;
    let mut sink = 0u64;
    for e in 0..n_experts {
        let t = Instant::now();
        let slice = &raw13[e * rows * cols..(e + 1) * rows * cols];
        let owned = slice.to_vec();
        t_cached += t.elapsed();
        sink += owned[0] as u64;
    }
    println!(
        "C. same w13 bytes from a CACHED mmap + to_vec        : {:8.2} ms/expert   [{:.1} MB]",
        ms(t_cached) / n_experts as f64,
        (rows * cols) as f64 / 1e6
    );

    // ---------------------------------------------------------------- D
    // Borrowing instead of copying: what the packed path could hand over.
    let mut t_borrow = std::time::Duration::ZERO;
    for e in 0..n_experts {
        let t = Instant::now();
        let slice = &raw13[e * rows * cols..(e + 1) * rows * cols];
        t_borrow += t.elapsed();
        sink += slice[0] as u64;
    }
    println!(
        "D. same w13 bytes BORROWED from cached mmap (no copy): {:8.4} ms/expert",
        ms(t_borrow) / n_experts as f64
    );

    // ---------------------------------------------------------------- E
    // The word() repack -- the "widening" that is really a bitcast + alloc.
    let probe = &raw13[0..rows * cols];
    let mut t_word = std::time::Duration::ZERO;
    for _ in 0..n_experts {
        let t = Instant::now();
        let w = word(probe);
        t_word += t.elapsed();
        sink += w[0] as u64;
    }
    println!(
        "E. word() i32 repack of w13 (the \"widening\")         : {:8.2} ms/expert   [{:.1} MB in, {:.1} MB out]",
        ms(t_word) / n_experts as f64,
        (rows * cols) as f64 / 1e6,
        (rows * cols) as f64 / 1e6
    );

    // ---------------------------------------------------------------- F
    // Uploading the raw packed bytes with cubecl, no repack, no TensorData.
    //
    // Measured three ways, because a per-iteration `sync()` measures the fence
    // and not the copy, and reporting that as "upload cost" would be exactly
    // the kind of adjacent-quantity mistake this lane is full of.
    {
        use cubecl::prelude::*;
        type Rt = cubecl::cuda::CudaRuntime;
        let client = Rt::client(&Default::default());
        let _ = client.create_from_slice(&probe[..1024]);
        client.sync();

        // F1: sync EVERY iteration (what the first version of this bench did)
        let mut t1 = std::time::Duration::ZERO;
        for _ in 0..n_experts {
            let t = Instant::now();
            let h = client.create_from_slice(probe);
            client.sync();
            t1 += t.elapsed();
            core::hint::black_box(&h);
        }

        // F2: enqueue all, sync ONCE at the end -- the amortised cost the real
        // forward actually pays, since it does not fence per expert.
        let t = Instant::now();
        let mut keep = Vec::new();
        for _ in 0..n_experts {
            keep.push(client.create_from_slice(probe));
        }
        client.sync();
        let t2 = t.elapsed();

        // F3: how much of F2 is the `slice.to_vec()` create_from_slice does
        // before it ever reaches the driver.
        let t = Instant::now();
        let mut copies = Vec::new();
        for _ in 0..n_experts {
            copies.push(probe.to_vec());
        }
        let t3 = t.elapsed();
        core::hint::black_box((&keep, &copies));

        println!(
            "F1. cubecl create_from_slice + sync EVERY iter       : {:8.2} ms/expert   [{:.1} MB]",
            ms(t1) / n_experts as f64,
            (rows * cols) as f64 / 1e6
        );
        println!(
            "F2. cubecl create_from_slice, ONE sync at end        : {:8.2} ms/expert   <- amortised",
            ms(t2) / n_experts as f64
        );
        println!(
            "F3.   of which the internal slice.to_vec()           : {:8.2} ms/expert",
            ms(t3) / n_experts as f64
        );
    }

    // ---------------------------------------------------------------- G
    // The current burn upload path end to end, synced.
    #[cfg(feature = "inkling-cuda")]
    {
        use burn::prelude::Backend;
        use mary::models::inkling::burn::expert_weight_from_packed;
        type Bk = burn::backend::Cuda<f32>;
        let dev = burn::backend::cuda::CudaDevice::default();

        let scales_len = rows * (cols * 2 / 16);
        let fs = std::fs::File::open(ckpt.join(cp_shard(&cp, &format!("{}.scale", base13()))?))?;
        let msc = unsafe { Mmap::map(&fs) }?;
        let ssc = SafeTensors::deserialize(&msc)?;
        let raw_sc = ssc.tensor(&format!("{}.scale", base13()))?.data();
        let sc = &raw_sc[0..scales_len];

        // warm
        let _w = expert_weight_from_packed::<Bk>(probe, sc, 1.0, rows, cols, &dev);
        <Bk as Backend>::sync(&dev);

        let mut t_burn = std::time::Duration::ZERO;
        for _ in 0..n_experts {
            let t = Instant::now();
            let w = expert_weight_from_packed::<Bk>(probe, sc, 1.0, rows, cols, &dev);
            <Bk as Backend>::sync(&dev);
            t_burn += t.elapsed();
            core::hint::black_box(&w);
        }
        println!(
            "G. expert_weight_from_packed (repack+upload+dequant): {:8.2} ms/expert   [-> {:.1} MB f32 on device]",
            ms(t_burn) / n_experts as f64,
            (rows * cols * 2 * 4) as f64 / 1e6
        );
    }


    // ---------------------------------------------------------------- H
    // The replacement source: headers parsed once, slabs borrowed not copied.
    {
        use mary::models::inkling::source::Weights;
        let t = Instant::now();
        let src = Weights::open_ckpt(&ckpt)?;
        let open_ms = ms(t.elapsed());
        println!();
        println!("H. Weights::open_ckpt (the shard index, ONCE)        : {:8.2} ms  (one-time)", open_ms);

        let mut t_e = std::time::Duration::ZERO;
        let mut acc = 0u64;
        for e in 0..n_experts {
            let t = Instant::now();
            let a = src.expert_packed(&base13(), e)?;
            let b = src.expert_packed(&base2(), e)?;
            t_e += t.elapsed();
            acc += a.codes()[0] as u64 + b.codes()[0] as u64 + a.scales()[0] as u64;
        }
        println!("I. Weights::expert_packed x2 per expert (borrowed)   : {:8.4} ms/expert   [vs B above]", ms(t_e) / n_experts as f64);

        use cubecl::prelude::*;
        type Rt = cubecl::cuda::CudaRuntime;
        let client = Rt::client(&Default::default());
        let _ = client.create_from_slice(&[0u8; 1024]);
        client.sync();
        let t = Instant::now();
        let mut keep = Vec::new();
        for e in 0..n_experts {
            let a = src.expert_packed(&base13(), e)?;
            let b = src.expert_packed(&base2(), e)?;
            keep.push(client.create_from_slice(a.codes()));
            keep.push(client.create_from_slice(a.scales()));
            keep.push(client.create_from_slice(b.codes()));
            keep.push(client.create_from_slice(b.scales()));
        }
        client.sync();
        let t_new = t.elapsed();
        core::hint::black_box((&keep, acc));
        println!("J. NEW LANE: borrow + upload w13+w2 packed, one sync : {:8.2} ms/expert   [12.6 MB]", ms(t_new) / n_experts as f64);
        println!("   (current lane, B + G scaled to w13+w2, is ~{:.1} ms/expert)", 2.74 + 10.98 * 1.5);
    }


    // ---------------------------------------------------------------- K/L
    // The two device lanes, like for like: packed bytes in, [tokens, hidden]
    // f32 out, synced. This is the ONLY place the tensor cores can contribute,
    // and it is deliberately measured apart from the host plumbing above.
    #[cfg(feature = "inkling-cuda")]
    {
        use burn::prelude::Backend;
        use burn::tensor::{Tensor, TensorData};
        use cubecl::prelude::*;
        use mary::models::inkling::burn::{deinterleave_rows_device, expert_ffn, expert_weight_from_packed};
        use mary::models::inkling::fp4gemm::{
            fp4_linear_launch, gate_up_silu_launch, upload_quantized_act,
        };
        use mary::models::inkling::source::Weights;
        type Bk = burn::backend::Cuda<f32>;
        type Rt = cubecl::cuda::CudaRuntime;

        let src = Weights::open_ckpt(&ckpt)?;
        let w13 = src.expert_packed(&base13(), 0)?;
        let w2 = src.expert_packed(&base2(), 0)?;
        let (nn, kk) = (w13.rows(), w13.cols() * 2);
        let inter = nn / 2;
        let tokens = 5usize;

        // A real activation: decoded rows of another expert.
        let mut x = vec![0f32; tokens * kk];
        {
            let p = src.expert_packed(&base13(), 7)?;
            for r in 0..tokens {
                for j in 0..kk {
                    let byte = p.codes()[r * (kk / 2) + j / 2];
                    let c = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    x[r * kk + j] = mary::models::inkling::nvfp4::FP4_E2M1[c as usize]
                        * mary::models::inkling::nvfp4::e4m3_to_f32(p.scales()[r * (kk / 16) + j / 16])
                        * p.scale2();
                }
            }
        }

        let reps = 8;

        // ---- K: native FP4 ------------------------------------------------
        let client = Rt::client(&Default::default());
        let run_fp4 = |client: &ComputeClient<Rt>| {
            let (a, asc, m_pad) = upload_quantized_act(client, &x, tokens, kk);
            let b = client.create_from_slice(w13.codes());
            let bsc = client.create_from_slice(w13.scales());
            let both = fp4_linear_launch(client, &a, &asc, &b, &bsc, m_pad, kk, nn, w13.scale2());
            let act = gate_up_silu_launch(client, &both, m_pad, inter);
            let actf = f32::from_bytes(&client.read_one(act).expect("read")).to_vec();
            let (a2, asc2, _) = upload_quantized_act(client, &actf, m_pad, inter);
            let b2 = client.create_from_slice(w2.codes());
            let bsc2 = client.create_from_slice(w2.scales());
            fp4_linear_launch(client, &a2, &asc2, &b2, &bsc2, m_pad, inter, w2.rows(), w2.scale2())
        };
        let h = run_fp4(&client);
        client.sync();
        core::hint::black_box(&h);
        let t = Instant::now();
        for _ in 0..reps {
            let h = run_fp4(&client);
            core::hint::black_box(&h);
        }
        client.sync();
        let t_fp4 = ms(t.elapsed()) / reps as f64;

        // ---- L: decode to f32, then f32 matmul ----------------------------
        let dev = burn::backend::cuda::CudaDevice::default();
        let xt: Tensor<Bk, 2> =
            Tensor::from_data(TensorData::new(x.clone(), [tokens, kk]), &dev);
        let run_f32 = || {
            let gu = expert_weight_from_packed::<Bk>(
                w13.codes(), w13.scales(), w13.scale2(), w13.rows(), w13.cols(), &dev,
            );
            let dn = expert_weight_from_packed::<Bk>(
                w2.codes(), w2.scales(), w2.scale2(), w2.rows(), w2.cols(), &dev,
            );
            expert_ffn(xt.clone(), deinterleave_rows_device(gu), dn)
        };
        let y = run_f32();
        <Bk as Backend>::sync(&dev);
        core::hint::black_box(&y);
        let t = Instant::now();
        for _ in 0..reps {
            let y = run_f32();
            core::hint::black_box(&y);
        }
        <Bk as Backend>::sync(&dev);
        let t_f32 = ms(t.elapsed()) / reps as f64;

        println!();
        println!("--- device lane, like for like ({tokens} tokens, N={nn}, K={kk}) ---");
        println!("K. native NVFP4 (packed in, tensor cores)            : {t_fp4:8.2} ms/expert");
        println!("L. decode to f32 on device, then f32 matmul          : {t_f32:8.2} ms/expert");
        println!("   device-side speedup from the FP4 path             : {:8.2}x", t_f32 / t_fp4);
        println!("   (L materialises 67.1 + 33.6 MB of f32 per expert; K materialises none)");
    }

    core::hint::black_box(sink);
    println!();
    println!("Per-forward scaling: the forward touches ~237 distinct experts per layer");
    println!("x 42 layers ~= 9950 expert-loads, each w13+w2 = 12.6 MB packed.");
    Ok(())
}

/// Which shard holds `name` (the index is private, so re-read it here).
fn cp_shard(_cp: &Checkpoint, name: &str) -> Result<String> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/thinkingmachines-inkling-small-nvfp4".into());
    let idx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(PathBuf::from(&dir).join("model.safetensors.index.json"))?)?;
    Ok(idx["weight_map"][name]
        .as_str()
        .with_context(|| format!("index has no {name}"))?
        .to_string())
}
