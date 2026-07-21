//! AUDIO-TOWER PARITY GATE — the trust gate for the audio path.
//!
//! Asserts mary's Gemma 4 audio path against goldens captured from HF
//! transformers (`google/gemma-4-E4B-it`), per stage:
//!
//!   1. features — `AudioFeatureExtractor::extract` vs
//!      `Gemma4AudioFeatureExtractor` (`{name}.features.bin`)
//!   2. tower    — `AudioModel::forward` ON THE GOLDEN FEATURES vs
//!      `Gemma4AudioModel` (`{name}.tower.bin`) — isolated from stage 1
//!   3. embedder — `AudioEmbedder::forward` ON THE GOLDEN TOWER OUTPUT vs
//!      `Gemma4MultimodalEmbedder` (`{name}.embed.bin`) — isolated from stage 2
//!   4. cascade  — mary features → mary tower → mary embedder vs golden embed
//!      (the end-to-end number production actually lives on)
//!
//! Goldens are `.bin` files (u32 LE ndim, u32 dims..., f32 LE data) written by
//! `capture_goldens.py`. Reference wavs must be 16 kHz mono with a sample
//! count that is a multiple of 128 (no padded frames on either side).
//!
//! With `--pile <gemma.pile>` the tower + embedder are ALSO loaded from the
//! persisted pile (the runtime path since f3833b9) and scored against the same
//! goldens — gating the pile wiring, including any f16-leaf quantization the
//! pile stores. Pile stages are labeled `tower(pile)` / `embed(pile)` /
//! `cascade(pile)`.
//!
//! Usage:
//!   cargo run --release --features gemma,import --bin gemma_audio_parity -- \
//!     --model-dir <hf-snapshot-dir> --wavs /tmp/gemma_audio_work/wavs \
//!     --goldens /tmp/gemma_audio_work/goldens [--pile <gemma.pile>] \
//!     [--threshold 0.999]

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::prelude::*;
use mary::models::gemma::gemma4::audio::{AudioEmbedder, AudioModel};
use mary::models::gemma::gemma4::audio_load::load_audio_16k_mono;
use mary::models::gemma::gemma4::audio_preprocess::AudioFeatureExtractor;
use mary::models::gemma::gemma4::config::Gemma4Config;
use safetensors::SafeTensors;
use std::path::Path;

type B = Wgpu;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|s| s == k).map(|i| args[i + 1].clone())
}

/// Read a golden .bin: u32 LE ndim, u32 dims..., f32 LE data.
fn read_bin(path: &Path) -> (Vec<f32>, Vec<usize>) {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let ndim = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mut dims = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let o = 4 + i * 4;
        dims.push(u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize);
    }
    let data_off = 4 + ndim * 4;
    let n: usize = dims.iter().product();
    assert_eq!(raw.len() - data_off, n * 4, "size mismatch in {path:?}");
    let data: Vec<f32> = raw[data_off..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (data, dims)
}

struct Stat {
    cos: f64,
    max_abs: f32,
    rmse: f64,
}

fn compare(ours: &[f32], golden: &[f32]) -> Stat {
    assert_eq!(ours.len(), golden.len(), "length mismatch");
    let (mut dot, mut na, mut nb, mut se) = (0f64, 0f64, 0f64, 0f64);
    let mut max_abs = 0f32;
    for (&a, &b) in ours.iter().zip(golden) {
        dot += a as f64 * b as f64;
        na += a as f64 * a as f64;
        nb += b as f64 * b as f64;
        let d = a - b;
        se += d as f64 * d as f64;
        if d.abs() > max_abs {
            max_abs = d.abs();
        }
    }
    Stat {
        cos: dot / (na.sqrt() * nb.sqrt()).max(1e-30),
        max_abs,
        rmse: (se / ours.len() as f64).sqrt(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = arg(&args, "--model-dir").expect("need --model-dir <hf snapshot dir>");
    let wav_dir = arg(&args, "--wavs").unwrap_or_else(|| "/tmp/gemma_audio_work/wavs".into());
    let golden_dir = arg(&args, "--goldens").unwrap_or_else(|| "/tmp/gemma_audio_work/goldens".into());
    let threshold: f64 = arg(&args, "--threshold").and_then(|s| s.parse().ok()).unwrap_or(0.999);

    let device = WgpuDevice::default();
    let config = Gemma4Config::load(&Path::new(&model_dir).join("config.json"));
    let audio_cfg = config.audio_config.as_ref().expect("model has no audio_config").clone();

    // Tower + embedder from the SAME checkpoint the goldens came from.
    let mut shard_paths: Vec<String> = std::fs::read_dir(&model_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().to_string())
        .filter(|p| p.ends_with(".safetensors"))
        .collect();
    shard_paths.sort();
    eprintln!("Loading tower + embedder from {} shard(s)...", shard_paths.len());
    let shard_data: Vec<Vec<u8>> = shard_paths.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let shards: Vec<SafeTensors> = shard_data
        .iter()
        .map(|d| SafeTensors::deserialize(d).unwrap())
        .collect();
    let tower = AudioModel::<B>::load_from_shards(audio_cfg.clone(), &shards, &device);
    let embedder = AudioEmbedder::<B>::load_from_shards(&shards, audio_cfg.rms_norm_eps, &device);
    let fe = AudioFeatureExtractor::new();

    // Optional: the SAME components loaded from the persisted pile (runtime path).
    let pile_audio = arg(&args, "--pile").map(|p| {
        eprintln!("Loading tower + embedder from pile {p}...");
        mary::persist::load_gemma4_audio_from_pile::<B>(Path::new(&p), audio_cfg.clone(), &device)
            .unwrap_or_else(|e| panic!("pile audio load: {e}"))
    });

    let mut wavs: Vec<String> = std::fs::read_dir(&wav_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().to_string())
        .filter(|p| p.ends_with(".wav"))
        .collect();
    wavs.sort();
    assert!(!wavs.is_empty(), "no wavs in {wav_dir}");

    let mut all_pass = true;
    println!("stage thresholds: cos >= {threshold}\n");
    for wav in &wavs {
        let name = Path::new(wav).file_stem().unwrap().to_string_lossy().to_string();
        let g = |suffix: &str| read_bin(&Path::new(&golden_dir).join(format!("{name}.{suffix}.bin")));

        let wave = load_audio_16k_mono(Path::new(wav)).unwrap();
        println!("== {name} ({:.2}s) ==", wave.len() as f32 / 16000.0);

        // --- Stage 1: features ---
        let (feat, mask, n_frames) = fe.extract(&wave);
        assert!(mask.iter().all(|&m| m), "{name}: padded frames in mary extract");
        let (g_feat, g_fdims) = g("features");
        assert_eq!(
            (n_frames, fe.feature_size),
            (g_fdims[0], g_fdims[1]),
            "{name}: feature shape mismatch"
        );
        let s = compare(&feat, &g_feat);
        let pass = s.cos >= threshold;
        all_pass &= pass;
        println!(
            "  features [{n_frames}x{}]  cos={:.7}  max|d|={:.3e}  rmse={:.3e}  {}",
            fe.feature_size, s.cos, s.max_abs, s.rmse, if pass { "PASS" } else { "FAIL" }
        );

        // --- Stage 2: tower on GOLDEN features (isolated) ---
        let gf = Tensor::<B, 1>::from_floats(&g_feat[..], &device).reshape([1, g_fdims[0], g_fdims[1]]);
        let tower_out = tower.forward(gf);
        let [_, t4, hid] = tower_out.dims();
        let ours: Vec<f32> = tower_out.clone().to_data().to_vec().unwrap();
        let (g_tower, g_tdims) = g("tower");
        assert_eq!((t4, hid), (g_tdims[0], g_tdims[1]), "{name}: tower shape mismatch");
        let s = compare(&ours, &g_tower);
        let pass = s.cos >= threshold;
        all_pass &= pass;
        println!(
            "  tower    [{t4}x{hid}]  cos={:.7}  max|d|={:.3e}  rmse={:.3e}  {}",
            s.cos, s.max_abs, s.rmse, if pass { "PASS" } else { "FAIL" }
        );

        // --- Stage 3: embedder on GOLDEN tower output (isolated) ---
        let gt = Tensor::<B, 1>::from_floats(&g_tower[..], &device).reshape([g_tdims[0], g_tdims[1]]);
        let emb_out = embedder.forward(gt);
        let [n_tok, text_h] = emb_out.dims();
        let ours: Vec<f32> = emb_out.to_data().to_vec().unwrap();
        let (g_emb, g_edims) = g("embed");
        assert_eq!((n_tok, text_h), (g_edims[0], g_edims[1]), "{name}: embed shape mismatch");
        let s = compare(&ours, &g_emb);
        let pass = s.cos >= threshold;
        all_pass &= pass;
        println!(
            "  embedder [{n_tok}x{text_h}]  cos={:.7}  max|d|={:.3e}  rmse={:.3e}  {}",
            s.cos, s.max_abs, s.rmse, if pass { "PASS" } else { "FAIL" }
        );

        // --- Stage 4: full mary cascade vs golden embed ---
        let mf = Tensor::<B, 1>::from_floats(&feat[..], &device).reshape([1, n_frames, fe.feature_size]);
        let cascade = embedder.forward(tower.forward(mf).reshape([t4, hid]));
        let ours: Vec<f32> = cascade.to_data().to_vec().unwrap();
        let s = compare(&ours, &g_emb);
        let pass = s.cos >= threshold;
        all_pass &= pass;
        println!(
            "  cascade  [{t4}x{text_h}]  cos={:.7}  max|d|={:.3e}  rmse={:.3e}  {}",
            s.cos, s.max_abs, s.rmse, if pass { "PASS" } else { "FAIL" }
        );

        // --- Pile stages: same three components, loaded from the pile ---
        if let Some((p_tower, p_embedder)) = &pile_audio {
            let gf = Tensor::<B, 1>::from_floats(&g_feat[..], &device)
                .reshape([1, g_fdims[0], g_fdims[1]]);
            let ours: Vec<f32> = p_tower.forward(gf).to_data().to_vec().unwrap();
            let s = compare(&ours, &g_tower);
            let pass = s.cos >= threshold;
            all_pass &= pass;
            println!(
                "  tower(pile)    cos={:.7}  max|d|={:.3e}  rmse={:.3e}  {}",
                s.cos, s.max_abs, s.rmse, if pass { "PASS" } else { "FAIL" }
            );

            let gt = Tensor::<B, 1>::from_floats(&g_tower[..], &device)
                .reshape([g_tdims[0], g_tdims[1]]);
            let ours: Vec<f32> = p_embedder.forward(gt).to_data().to_vec().unwrap();
            let s = compare(&ours, &g_emb);
            let pass = s.cos >= threshold;
            all_pass &= pass;
            println!(
                "  embed(pile)    cos={:.7}  max|d|={:.3e}  rmse={:.3e}  {}",
                s.cos, s.max_abs, s.rmse, if pass { "PASS" } else { "FAIL" }
            );

            let mf = Tensor::<B, 1>::from_floats(&feat[..], &device)
                .reshape([1, n_frames, fe.feature_size]);
            let cascade = p_embedder.forward(p_tower.forward(mf).reshape([t4, hid]));
            let ours: Vec<f32> = cascade.to_data().to_vec().unwrap();
            let s = compare(&ours, &g_emb);
            let pass = s.cos >= threshold;
            all_pass &= pass;
            println!(
                "  cascade(pile)  cos={:.7}  max|d|={:.3e}  rmse={:.3e}  {}",
                s.cos, s.max_abs, s.rmse, if pass { "PASS" } else { "FAIL" }
            );
        }
        println!();
    }

    if all_pass {
        println!("PARITY GATE: PASS (all stages, {} wavs)", wavs.len());
    } else {
        println!("PARITY GATE: FAIL — see stages above");
        std::process::exit(1);
    }
}
