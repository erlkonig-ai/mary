//! End-to-end IMAGE-embedding parity gate for `nomic-embed-multimodal-7b`
//! (`BiQwen2_5`). Splices the parity-verified vision-tower tokens into the text
//! sequence at the `<|image_pad|>` positions, builds M-RoPE 3D position-ids
//! (`get_rope_index`), runs the 28-layer backbone, last-token pool + L2 — and
//! compares against the float32 reference dumped by `nomic_mm7b_image_dump.py`.
//!
//! The hard step (the multimodal splice + section-wise M-RoPE) is de-risked by
//! verifying THREE intermediate anchors independently of the final number:
//!   1. get_rope_index  vs `image_position_ids.npy`   (pure logic, no pile)
//!   2. spliced embeds  vs `image_inputs_embeds.npy`   (vision tower + splice)
//!   3. backbone hidden vs `image_last_hidden.npy`     (M-RoPE through 28 layers)
//! then the final `image_emb` itself (bar cosine >= 0.9999).
//!
//! Prereqs (the embed path needs BOTH text + vision weights in one collection,
//! under their explicit component coordinates; disk-gated, SKIPS cleanly if
//! missing):
//!   1. python3 scripts/nomic_mm7b_merge.py        <SCRATCH>/text/merged_text_backbone.safetensors
//!   2. python3 scripts/nomic_mm7b_vision_dump.py  <SCRATCH>/vision/vision_tower.safetensors
//!   3. python3 scripts/nomic_mm7b_image_dump.py   # dumps the image reference goldens
//!   4. cargo run --release --features import,hub --bin mary -- import <SCRATCH>/text --pile <SCRATCH>/combined.pile --key <SCRATCH>/model.key --name 'nomic-ai/nomic-embed-multimodal-7b#text' --dtype f16
//!   5. cargo run --release --features import,hub --bin mary -- import <SCRATCH>/vision --pile <SCRATCH>/combined.pile --key <SCRATCH>/model.key --name 'nomic-ai/nomic-embed-multimodal-7b#vision' --dtype f16
//!   6. NOMIC_MM7B_PILE=<SCRATCH>/combined.pile cargo test --release --features gemma --test nomic_mm7b_image_parity -- --nocapture

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::{NdArray, NdArrayDevice};
use mary::models::qwen2_5_vl::config::{Qwen2_5VlTextConfig, Qwen2_5VlVisionConfig};
use mary::models::qwen2_5_vl::layers::{get_rope_index, QwenTextModel, QwenWeights};
use mary::models::qwen2_5_vl::vision::{VisionTransformer, VisionWeights};
use mary::nn::npy;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

type B = NdArray;

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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The 56x56 probe image: 15-token prompt, grid_thw (1,4,4) -> 4 image-pad rows.
const IMAGE_TOKEN_ID: i64 = 151655;
const MERGE: usize = 2;
fn probe_grid() -> Vec<(usize, usize, usize)> {
    vec![(1, 4, 4)]
}

fn load_ids(dir: &Path, name: &str) -> Vec<i64> {
    let (d, _) = npy::load_npy(&dir.join(format!("{name}.npy"))).unwrap();
    d.iter().map(|&v| v.round() as i64).collect()
}

/// get_rope_index is pure index logic — verify it with NO pile/weights at all.
#[test]
fn get_rope_index_parity() {
    let dir = golden_dir();
    if !dir.join("image_position_ids.npy").exists() {
        eprintln!("SKIP: run scripts/nomic_mm7b_image_dump.py");
        return;
    }
    let input_ids = load_ids(&dir, "image_input_ids");
    let pos = get_rope_index(&input_ids, &probe_grid(), IMAGE_TOKEN_ID, MERGE);
    let s = input_ids.len();
    assert_eq!(pos.len(), s);

    // reference is [3, 1, S] row-major: axis-major (t row, h row, w row)
    let want = load_ids(&dir, "image_position_ids");
    assert_eq!(want.len(), 3 * s, "position_ids shape");
    for (i, p) in pos.iter().enumerate() {
        for axis in 0..3 {
            assert_eq!(
                p[axis],
                want[axis * s + i],
                "position_ids[axis={axis}][token={i}]: got {} want {}",
                p[axis],
                want[axis * s + i]
            );
        }
    }
    eprintln!("[image-parity] get_rope_index: EXACT match ({s} tokens, 3 axes)  OK");
}

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

fn text_cfg() -> Qwen2_5VlTextConfig {
    Qwen2_5VlTextConfig::nomic_mm7b()
}
fn vision_cfg() -> Qwen2_5VlVisionConfig {
    Qwen2_5VlVisionConfig::nomic_mm7b()
}

fn load2(dir: &Path, name: &str, device: &NdArrayDevice) -> Tensor<B, 2> {
    let (d, s) = npy::load_npy(&dir.join(format!("{name}.npy"))).unwrap();
    Tensor::from_data(TensorData::new(d, s), device)
}

#[test]
fn image_embed_parity() {
    let Ok(pile_path) = std::env::var("NOMIC_MM7B_PILE") else {
        eprintln!("SKIP: set NOMIC_MM7B_PILE=<combined pile> (see file header)");
        return;
    };
    let dir = golden_dir();
    let vdir = dir.join("vision");
    if !dir.join("image_emb.npy").exists() {
        eprintln!("SKIP: run scripts/nomic_mm7b_image_dump.py first");
        return;
    }
    let device = NdArrayDevice::default();
    eprintln!("[image-parity] loading combined keymap from {pile_path} ...");
    let (_, snapshot) =
        mary::model_collection::load_sole_model_collection_local_latest(Path::new(&pile_path))
            .expect("load native model collection snapshot");
    let map = mary::persist::load_nomic_mm7b_keymap_from_snapshot(snapshot)
        .expect("select and materialize the Nomic text + vision components");
    eprintln!(
        "[image-parity] keymap has {} tensors; building text + vision ...",
        map.len()
    );
    let w = KeymapW { map, device };
    let text = QwenTextModel::<B>::load(&w, &text_cfg(), &device);
    let vision = VisionTransformer::<B>::load(&w, &vision_cfg(), &device);

    let input_ids = load_ids(&dir, "image_input_ids");
    let s = input_ids.len();
    let grid = probe_grid();
    let pixel_values = load2(&vdir, "pixel_values", &device);

    // --- vision tower -> merged image tokens ([4, 3584]) ---
    let vision_tokens = vision.forward(pixel_values, &grid);

    // --- ANCHOR 1: get_rope_index (already covered in its own test, re-pin here) ---
    let position_ids = get_rope_index(&input_ids, &grid, IMAGE_TOKEN_ID, MERGE);

    // --- ANCHOR 2: spliced input embeds vs reference ---
    let h = vision_tokens.dims()[1];
    let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(input_ids.clone(), [1, s]), &device);
    let mut embeds = text.embed_tokens(ids);
    let mut k = 0usize;
    for (p, &tok) in input_ids.iter().enumerate() {
        if tok == IMAGE_TOKEN_ID {
            let row = vision_tokens.clone().narrow(0, k, 1).reshape([1, 1, h]);
            embeds = embeds.slice_assign([0..1, p..p + 1, 0..h], row);
            k += 1;
        }
    }
    assert_eq!(k, vision_tokens.dims()[0], "spliced {k} vision tokens");
    let got_splice = embeds.clone().into_data().to_vec::<f32>().unwrap();
    let want_splice = npy::load_npy(&dir.join("image_inputs_embeds.npy"))
        .unwrap()
        .0;
    let cos_splice = cosine(&got_splice, &want_splice);
    let ma_splice = max_abs(&got_splice, &want_splice);
    eprintln!(
        "[image-parity] ANCHOR splice (inputs_embeds): cosine={cos_splice:.7} max_abs={ma_splice:e}"
    );
    assert!(cos_splice >= 0.999, "splice cosine {cos_splice:.7} < 0.999");

    // --- ANCHOR 3: backbone hidden (M-RoPE through 28 layers) vs reference ---
    let hidden = text.run_embeds(embeds.clone(), &position_ids);
    let got_hidden = hidden.into_data().to_vec::<f32>().unwrap();
    let want_hidden = npy::load_npy(&dir.join("image_last_hidden.npy")).unwrap().0;
    let cos_hidden = cosine(&got_hidden, &want_hidden);
    eprintln!("[image-parity] ANCHOR backbone (last_hidden): cosine={cos_hidden:.7}");
    assert!(
        cos_hidden >= 0.999,
        "backbone hidden cosine {cos_hidden:.7} < 0.999"
    );

    // --- FINAL: dense image embedding (pool + L2) vs reference ---
    let got = text
        .embed_from_embeds(embeds, &position_ids)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let want = npy::load_npy(&dir.join("image_emb.npy")).unwrap().0;
    assert_eq!(got.len(), want.len(), "image_emb dim");
    let cos = cosine(&got, &want);
    let n: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!("[image-parity] FINAL image_emb: cosine={cos:.7}  |emb|={n:.5}  (bar 0.9999)");
    assert!(cos >= 0.9999, "image_emb cosine={cos:.7} < 0.9999");

    // --- the wired entry point: NomicMultimodalEmbedder::embed_image must match ---
    use mary::models::qwen2_5_vl::embedder::NomicMultimodalEmbedder;
    let tok = nomic_tokenizer();
    if let Some(tok_path) = tok {
        let embedder = NomicMultimodalEmbedder::<B>::load_with_vision(&w, &tok_path, device)
            .expect("load embedder");
        let pixel_values = load2(&vdir, "pixel_values", &device);
        let via_api = embedder
            .embed_image_pixels(pixel_values, &grid, &input_ids)
            .expect("embed_image");
        let cos_api = cosine(&via_api, &want);
        eprintln!("[image-parity] embed_image() API: cosine={cos_api:.7}");
        assert!(
            cos_api >= 0.9999,
            "embed_image API cosine={cos_api:.7} < 0.9999"
        );
    }
    eprintln!("[image-parity] ALL anchors + final + API  OK");
}

/// A tokenizer.json for the embedder constructor (embed_image takes input_ids
/// directly, so any Qwen2.5 tokenizer works). Best-effort from the HF cache.
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
