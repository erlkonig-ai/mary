//! Pure-Rust image-preprocessing + prompt-assembly parity for
//! `nomic-embed-multimodal-7b` (`BiQwen2_5`). Closes the last Python dependency
//! of the IMAGE path: turns raw image bytes into the exact
//! `(pixel_values, image_grid_thw)` the vision tower consumes, plus the image
//! prompt `input_ids`.
//!
//! Anchors (the first two need NO pile — pure preprocessing logic):
//!   1. `pixel_values`  vs `tests/golden/nomic_mm7b/vision/pixel_values.npy`
//!      (max_abs; resize is a no-op for the 56×56 probe, so this is ~exact)
//!   2. image `input_ids`  EXACT vs `tests/golden/nomic_mm7b/image_input_ids.npy`
//!   3. (disk-gated) full `embed_image(bytes)` end-to-end cos vs `image_emb.npy`
//!
//! The probe image matches `scripts/nomic_mm7b_image_dump.py` exactly:
//!   `Image.new("RGB", (56, 56), (123, 200, 90))`.
//!
//! Anchor 3 needs the COMBINED text+vision native model-collection pile (same
//! as the image-parity gate):
//!   NOMIC_MM7B_PILE=<combined pile> cargo test --release --features gemma \
//!     --test nomic_mm7b_preprocess_parity -- --nocapture

use burn_ndarray::NdArray;
use mary::models::qwen2_5_vl::preprocess::{build_image_prompt, preprocess_image, PATCH_DIM};
use mary::nn::npy;
use std::path::{Path, PathBuf};

type B = NdArray;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/nomic_mm7b")
}

/// Encode the same synthetic probe image the Python dump used: 56×56 solid RGB
/// `(123, 200, 90)`, as PNG bytes (the input to `preprocess_image`).
fn probe_image_png() -> Vec<u8> {
    use image::ImageEncoder;
    let mut img = image::RgbImage::new(56, 56);
    for p in img.pixels_mut() {
        *p = image::Rgb([123, 200, 90]);
    }
    let mut bytes: Vec<u8> = Vec::new();
    image::codecs::png::PngEncoder::new(&mut std::io::Cursor::new(&mut bytes))
        .write_image(&img, 56, 56, image::ExtendedColorType::Rgb8)
        .expect("encode png");
    bytes
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
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

/// ANCHOR 1+2: preprocessing logic, NO pile required.
#[test]
fn preprocess_and_prompt_parity() {
    let dir = golden_dir();
    let pv_path = dir.join("vision/pixel_values.npy");
    if !pv_path.exists() {
        eprintln!("SKIP: run scripts/nomic_mm7b_image_dump.py first");
        return;
    }

    // --- ANCHOR 1: pixel_values + grid ---
    let (pixels, grid) = preprocess_image(&probe_image_png()).expect("preprocess");
    assert_eq!(grid, (1, 4, 4), "image_grid_thw");
    let seq = grid.0 * grid.1 * grid.2;
    assert_eq!(pixels.len(), seq * PATCH_DIM, "pixel_values flat length");

    let (want_pv, want_shape) = npy::load_npy(&pv_path).unwrap();
    assert_eq!(
        want_shape,
        vec![seq, PATCH_DIM],
        "golden pixel_values shape"
    );
    let ma = max_abs(&pixels, &want_pv);
    let cos = cosine(&pixels, &want_pv);
    eprintln!("[preprocess] pixel_values: max_abs={ma:e}  cosine={cos:.9}  ({seq}×{PATCH_DIM})");
    assert!(ma < 1e-2, "pixel_values max_abs {ma:e} >= 1e-2");

    // --- ANCHOR 2: image prompt input_ids (needs a tokenizer; skip if absent) ---
    let want_ids: Vec<i64> = {
        let (d, _) = npy::load_npy(&dir.join("image_input_ids.npy")).unwrap();
        d.iter().map(|&v| v.round() as i64).collect()
    };
    let prompt = build_image_prompt(grid);
    if let Some(tok_path) = nomic_tokenizer() {
        let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");
        let enc = tok.encode(prompt.as_str(), false).expect("tokenize");
        let got_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
        assert_eq!(got_ids, want_ids, "image input_ids EXACT mismatch");
        eprintln!(
            "[preprocess] input_ids: EXACT match ({} tokens)  OK",
            got_ids.len()
        );
    } else {
        eprintln!("[preprocess] input_ids: SKIP exact check (no tokenizer.json in HF cache)");
        // Structural sanity even without a tokenizer: 4 image-pad placeholders.
        assert_eq!(
            prompt.matches("<|image_pad|>").count(),
            4,
            "image-pad count"
        );
    }
    eprintln!("[preprocess] ANCHOR 1 (pixels) + ANCHOR 2 (ids)  OK");
}

/// ANCHOR 3: full `embed_image(bytes)` end-to-end vs the golden embedding.
/// Disk-gated on the combined pile (text + vision weights).
#[test]
fn embed_image_bytes_parity() {
    use mary::models::qwen2_5_vl::embedder::NomicMultimodalEmbedder;
    let Ok(pile_path) = std::env::var("NOMIC_MM7B_PILE") else {
        eprintln!("SKIP: set NOMIC_MM7B_PILE=<combined pile> (see file header)");
        return;
    };
    let dir = golden_dir();
    if !dir.join("image_emb.npy").exists() {
        eprintln!("SKIP: run scripts/nomic_mm7b_image_dump.py first");
        return;
    }
    let Some(tok_path) = nomic_tokenizer() else {
        eprintln!("SKIP: no tokenizer.json in HF cache");
        return;
    };

    let device = burn_ndarray::NdArrayDevice::default();
    let (_, snapshot) =
        mary::model_collection::load_sole_model_collection_local_latest(Path::new(&pile_path))
            .expect("load native model collection snapshot");
    let map = mary::persist::load_nomic_mm7b_keymap_from_snapshot(snapshot)
        .expect("select and materialize the Nomic text + vision components");
    let w = KeymapW { map, device };
    let embedder =
        NomicMultimodalEmbedder::<B>::load_with_vision(&w, &tok_path, device).expect("embedder");

    let got = embedder
        .embed_image(&probe_image_png())
        .expect("embed_image(bytes)");
    let want = npy::load_npy(&dir.join("image_emb.npy")).unwrap().0;
    assert_eq!(got.len(), want.len(), "image_emb dim");
    let cos = cosine(&got, &want);
    let n: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!("[preprocess] embed_image(bytes) END-TO-END: cosine={cos:.7}  |emb|={n:.5}");
    assert!(cos >= 0.999, "embed_image(bytes) cosine {cos:.7} < 0.999");
}

// --- weight source over a pile keymap (mirrors nomic_mm7b_image_parity.rs) ---
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArrayDevice;
use mary::models::qwen2_5_vl::layers::QwenWeights;
use mary::models::qwen2_5_vl::vision::VisionWeights;
use std::collections::HashMap;

struct KeymapW {
    map: HashMap<String, (Vec<f32>, Vec<usize>)>,
    device: NdArrayDevice,
}
impl KeymapW {
    fn get(&self, name: &str) -> &(Vec<f32>, Vec<usize>) {
        self.map
            .get(name)
            .unwrap_or_else(|| panic!("missing weight {name}"))
    }
}
impl QwenWeights<B> for KeymapW {
    fn t1(&self, name: &str) -> Tensor<B, 1> {
        let (d, s) = self.get(name);
        Tensor::from_data(TensorData::new(d.clone(), s.clone()), &self.device)
    }
    fn t2(&self, name: &str) -> Tensor<B, 2> {
        let (d, s) = self.get(name);
        Tensor::from_data(TensorData::new(d.clone(), s.clone()), &self.device)
    }
}
impl VisionWeights<B> for KeymapW {
    fn t1(&self, name: &str) -> Tensor<B, 1> {
        <Self as QwenWeights<B>>::t1(self, name)
    }
    fn t2(&self, name: &str) -> Tensor<B, 2> {
        <Self as QwenWeights<B>>::t2(self, name)
    }
    fn patch_proj(&self, name: &str, embed: usize, in_flat: usize) -> Tensor<B, 2> {
        let (d, _) = self.get(name);
        Tensor::from_data(TensorData::new(d.clone(), [embed, in_flat]), &self.device)
    }
}

/// Best-effort tokenizer.json from the HF cache (same paths as the image gate).
fn nomic_tokenizer() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    for pat in [
        "models--nomic-ai--nomic-embed-multimodal-7b",
        "models--Qwen--Qwen2.5-VL-7B-Instruct",
    ] {
        let base = PathBuf::from(&home)
            .join(".cache/huggingface/hub")
            .join(pat)
            .join("snapshots");
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
