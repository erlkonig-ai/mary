//! ZERO-COPY Metal/f16 parity gate for `nomic-embed-multimodal-7b`
//! (`BiQwen2_5`). Loads the combined Qwen2.5-VL backbone + vision tower through
//! the **aliased-from-pile** path ([`mary::persist::load_nomic_mm7b_aliased_from_pile`]):
//! every f16 weight blob is mmap'd from the pile straight onto the Metal GPU — no
//! copy, no f32 materialization — and the backbone then runs in f16 *compute* on
//! the GPU. This is the production "no daemon needed" path; the NdArray gates
//! (`nomic_mm7b_real_parity`, `nomic_mm7b_image_parity`) prove the same maths in
//! f32-on-CPU.
//!
//! The bar is intentionally looser than the f32 gates: f16 GPU compute (not just
//! f16 *storage*) is the dominant residual, so we expect a hair below the ~0.9999
//! f32 number. Anything >= 0.999 is good; the printed cosine is the honest margin.
//!
//! Prereqs (disk-gated; SKIPS cleanly if missing):
//!   - the combined f16 pile (text+vision), default `/private/tmp/nomic_mm7b_combined.pile`
//!     or `NOMIC_MM7B_PILE=<pile>` (build steps: see `nomic_mm7b_image_parity.rs` header)
//!   - the float32 reference goldens under `tests/golden/nomic_mm7b/`
//!     (`nomic_mm7b_probe.py` + `nomic_mm7b_image_dump.py`)
//!
//!   cargo test --release --features gemma --test nomic_mm7b_aliased_parity -- --nocapture
//!
//! macOS / Metal only (the whole file is `target_os = "macos"` gated).
#![cfg(target_os = "macos")]

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::models::gemma::metal_device::init_metal_device_16gb;
use mary::nn::backend::B;
use mary::nn::npy;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/nomic_mm7b")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn load_ids(dir: &Path, name: &str) -> Vec<i64> {
    let (d, _) = npy::load_npy(&dir.join(format!("{name}.npy"))).unwrap();
    d.iter().map(|&v| v.round() as i64).collect()
}

/// A `tokenizer.json` for the embedder constructor (we feed input_ids directly,
/// so any Qwen2.5 tokenizer works). Best-effort from the HF cache.
fn nomic_tokenizer() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    for pat in [
        "models--nomic-ai--nomic-embed-multimodal-7b",
        "models--Qwen--Qwen2.5-VL-7B-Instruct",
    ] {
        let base = PathBuf::from(&home).join(".cache/huggingface/hub").join(pat).join("snapshots");
        if let Ok(rd) = std::fs::read_dir(&base) {
            for e in rd.flatten() {
                let p = e.path().join("tokenizer.json");
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }
    None
}

const IMAGE_TOKEN_ID: i64 = 151655;

#[test]
fn aliased_metal_embed_parity() {
    let pile_path = std::env::var("NOMIC_MM7B_PILE")
        .unwrap_or_else(|_| "/private/tmp/nomic_mm7b_combined.pile".to_string());
    if !Path::new(&pile_path).exists() {
        eprintln!("SKIP: no combined pile at {pile_path} (set NOMIC_MM7B_PILE; see header)");
        return;
    }
    let dir = golden_dir();
    if !dir.join("query_emb.npy").exists() {
        eprintln!("SKIP: run scripts/nomic_mm7b_probe.py to dump reference goldens");
        return;
    }
    let Some(tok_path) = nomic_tokenizer() else {
        eprintln!("SKIP: no tokenizer.json in HF cache (nomic or Qwen2.5-VL-7B)");
        return;
    };

    let device = init_metal_device_16gb();

    // --- cold aliased load (timed): mmap f16 -> Metal, zero copy ---
    let t0 = Instant::now();
    let embedder = mary::persist::load_nomic_mm7b_aliased_from_pile(
        Path::new(&pile_path),
        &tok_path,
        device.clone(),
    )
    .expect("aliased load");
    let load_ms = t0.elapsed().as_millis();
    eprintln!("[aliased] cold aliased load (mmap f16 -> Metal): {load_ms} ms");

    let mut worst = 1.0f64;

    // --- TEXT: query + document, via the SAME input ids the f32 probe used ---
    for (ids_name, ref_name) in [
        ("query_input_ids", "query_emb"),
        ("doc_text_input_ids", "doc_text_emb"),
    ] {
        if !dir.join(format!("{ref_name}.npy")).exists() {
            continue;
        }
        let ids = load_ids(&dir, ids_name);
        let seq = ids.len();
        let want = npy::load_npy(&dir.join(format!("{ref_name}.npy"))).unwrap().0;
        let t = Instant::now();
        let got = embedder.embed_ids(&ids);
        let ms = t.elapsed().as_millis();
        assert_eq!(got.len(), want.len(), "{ref_name}: dim mismatch");
        let cos = cosine(&got, &want);
        let n: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("[aliased] {ref_name} (seq={seq}): cosine={cos:.7}  |emb|={n:.5}  ({ms} ms)");
        worst = worst.min(cos);
        assert!(cos >= 0.999, "{ref_name}: f16-Metal cosine={cos:.7} < 0.999");
    }

    // --- IMAGE: vision tower -> splice -> M-RoPE backbone, all f16 on Metal ---
    let vdir = dir.join("vision");
    if dir.join("image_emb.npy").exists() && vdir.join("pixel_values.npy").exists() {
        let input_ids = load_ids(&dir, "image_input_ids");
        let grid = vec![(1usize, 4usize, 4usize)]; // 56x56 probe: grid_thw (1,4,4)
        let (pd, ps) = npy::load_npy(&vdir.join("pixel_values.npy")).unwrap();
        let pixel_values =
            Tensor::<B, 2>::from_data(TensorData::new(pd, ps), &device);
        let want = npy::load_npy(&dir.join("image_emb.npy")).unwrap().0;
        let t = Instant::now();
        let got = embedder
            .embed_image_pixels(pixel_values, &grid, &input_ids)
            .expect("embed_image");
        let ms = t.elapsed().as_millis();
        assert_eq!(got.len(), want.len(), "image_emb dim mismatch");
        let cos = cosine(&got, &want);
        let n_img = input_ids.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
        let nrm: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "[aliased] image_emb (seq={}, img_tokens={n_img}): cosine={cos:.7}  |emb|={nrm:.5}  ({ms} ms)",
            input_ids.len()
        );
        worst = worst.min(cos);
        assert!(cos >= 0.999, "image_emb: f16-Metal cosine={cos:.7} < 0.999");
    } else {
        eprintln!("SKIP image: run scripts/nomic_mm7b_image_dump.py for image goldens");
    }

    eprintln!("[aliased] worst cosine = {worst:.7}  (f16-Metal bar 0.999)  OK");

    // --- O(n^2) attention at a realistic sequence length (text cap 2048) ---
    // No golden here (the probe dumps short prompts); we validate the long-context
    // attention runs on Metal and yields a finite, unit-norm embedding.
    let base = load_ids(&dir, "doc_text_input_ids");
    let mut long_ids = Vec::with_capacity(2048);
    while long_ids.len() < 2048 {
        long_ids.extend_from_slice(&base);
    }
    long_ids.truncate(2048);
    let t = Instant::now();
    let long_emb = embedder.embed_ids(&long_ids);
    let ms = t.elapsed().as_millis();
    let nrm: f32 = long_emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(long_emb.iter().all(|v| v.is_finite()), "long-seq embedding has non-finite values");
    assert!((nrm - 1.0).abs() < 1e-2, "long-seq embedding not unit-norm: |emb|={nrm}");
    eprintln!("[aliased] O(n^2) attention @ seq=2048: ran on Metal, |emb|={nrm:.5}  ({ms} ms)");
}
