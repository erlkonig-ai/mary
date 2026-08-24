//! Real-weight parity gate for the Qwen2.5-VL vision tower of
//! `nomic-embed-multimodal-7b` (base `visual.*` weights; no vision LoRA).
//!
//! Loads the vision tower from a pile and reproduces the merged image tokens
//! (`merger.pooler_output`) for the 56x56 probe image, comparing against the
//! float32 reference dumped by `scripts/nomic_mm7b_vision_dump.py`. Also pins the
//! patch-embed and block-0 intermediates.
//!
//! Prereqs (disk-gated; SKIPS cleanly if missing):
//!   1. python3 scripts/nomic_mm7b_vision_dump.py <SCRATCH>/vision_merged/vision_tower.safetensors
//!   2. cargo run --release --features import,hub --bin mary -- import <SCRATCH>/vision_merged --pile <SCRATCH>/nomic_mm7b_vision.pile --key <SCRATCH>/model.key --name 'nomic-ai/nomic-embed-multimodal-7b#vision' --dtype f16
//!   3. NOMIC_MM7B_VISION_PILE=<SCRATCH>/nomic_mm7b_vision.pile cargo test --release --features gemma --test nomic_mm7b_vision_parity -- --nocapture

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::{NdArray, NdArrayDevice};
use mary::models::qwen2_5_vl::config::Qwen2_5VlVisionConfig;
use mary::models::qwen2_5_vl::vision::{VisionTransformer, VisionWeights};
use mary::nn::npy;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

type B = NdArray;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/nomic_mm7b/vision")
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

struct KeymapW {
    map: HashMap<String, (Vec<f32>, Vec<usize>)>,
    device: NdArrayDevice,
}
impl KeymapW {
    fn get(&self, name: &str) -> &(Vec<f32>, Vec<usize>) {
        self.map
            .get(name)
            .unwrap_or_else(|| panic!("missing vision weight {name}"))
    }
}
impl VisionWeights<B> for KeymapW {
    fn t1(&self, name: &str) -> Tensor<B, 1> {
        let (d, s) = self.get(name);
        Tensor::from_data(TensorData::new(d.clone(), s.clone()), &self.device)
    }
    fn t2(&self, name: &str) -> Tensor<B, 2> {
        let (d, s) = self.get(name);
        Tensor::from_data(TensorData::new(d.clone(), s.clone()), &self.device)
    }
    fn patch_proj(&self, name: &str, embed: usize, in_flat: usize) -> Tensor<B, 2> {
        let (d, _) = self.get(name); // stored shape [embed, in, t, ph, pw]; reshape flat
        Tensor::from_data(TensorData::new(d.clone(), [embed, in_flat]), &self.device)
    }
}

fn vision_cfg() -> Qwen2_5VlVisionConfig {
    Qwen2_5VlVisionConfig {
        depth: 32,
        hidden_size: 1280,
        hidden_act: "silu".into(),
        intermediate_size: 3420,
        num_heads: 16,
        in_channels: 3,
        patch_size: 14,
        spatial_merge_size: 2,
        temporal_patch_size: 2,
        window_size: 112,
        out_hidden_size: 3584,
        fullatt_block_indexes: vec![7, 15, 23, 31],
    }
}

fn load2(dir: &Path, name: &str, device: &NdArrayDevice) -> Tensor<B, 2> {
    let (d, s) = npy::load_npy(&dir.join(format!("{name}.npy"))).unwrap();
    Tensor::from_data(TensorData::new(d, s), device)
}

#[test]
fn vision_tower_parity() {
    let Ok(pile_path) = std::env::var("NOMIC_MM7B_VISION_PILE") else {
        eprintln!("SKIP: set NOMIC_MM7B_VISION_PILE=<pile> (see file header)");
        return;
    };
    let dir = golden_dir();
    if !dir.join("merger_out.npy").exists() {
        eprintln!("SKIP: run scripts/nomic_mm7b_vision_dump.py first");
        return;
    }
    let device = NdArrayDevice::default();
    let (_, snapshot) =
        mary::model_collection::load_sole_model_collection_local_latest(Path::new(&pile_path))
            .expect("load native model collection snapshot");
    let map = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.reader(),
        mary::selection::ModelSelector::Source {
            source: mary::models::qwen2_5_vl::NOMIC_MM7B_VISION_SOURCE,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .expect("select and materialize the Nomic vision component");
    eprintln!(
        "[vision-parity] keymap has {} tensors; building tower ...",
        map.len()
    );
    let w = KeymapW { map, device };
    let model = VisionTransformer::<B>::load(&w, &vision_cfg(), &device);

    let pixel_values = load2(&dir, "pixel_values", &device);
    let grid = vec![(1usize, 4usize, 4usize)]; // grid_thw of the 56x56 probe image

    let merged = model.forward(pixel_values, &grid);
    let got = merged.into_data().to_vec::<f32>().unwrap();
    let want = npy::load_npy(&dir.join("merger_out.npy")).unwrap().0;
    assert_eq!(got.len(), want.len(), "merger_out dim mismatch");
    let cos = cosine(&got, &want);
    let ma = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    eprintln!("[vision-parity] merger_out: cosine={cos:.7} max_abs={ma:e}  (bar 0.999)");
    assert!(cos >= 0.999, "vision merger_out cosine={cos:.7} < 0.999");
}

/// MULTI-WINDOW vision parity: a 140x140 image -> 10x10 patch grid -> 5x5 merged
/// units, which SPANS 2 windows per axis (4 windows + padding). Unlike the 56x56
/// single-window probe, this actually exercises the window-partition reorder,
/// `cu_window_seqlens` block-diagonal mask, and the scatter-back-to-raster path.
/// Accepts the combined `NOMIC_MM7B_PILE` (has vision keys) or the vision pile.
#[test]
fn vision_tower_multiwindow_parity() {
    let pile_path =
        std::env::var("NOMIC_MM7B_PILE").or_else(|_| std::env::var("NOMIC_MM7B_VISION_PILE"));
    let Ok(pile_path) = pile_path else {
        eprintln!("SKIP: set NOMIC_MM7B_PILE or NOMIC_MM7B_VISION_PILE (see file header)");
        return;
    };
    let mw = golden_dir().parent().unwrap().join("vision_mw");
    if !mw.join("merger_out.npy").exists() {
        eprintln!("SKIP: run scripts/nomic_mm7b_vision_mw_dump.py first");
        return;
    }
    let device = NdArrayDevice::default();
    let (_, snapshot) =
        mary::model_collection::load_sole_model_collection_local_latest(Path::new(&pile_path))
            .expect("load native model collection snapshot");
    let map = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.reader(),
        mary::selection::ModelSelector::Source {
            source: mary::models::qwen2_5_vl::NOMIC_MM7B_VISION_SOURCE,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .expect("select and materialize the Nomic vision component");
    let w = KeymapW { map, device };
    let model = VisionTransformer::<B>::load(&w, &vision_cfg(), &device);

    let pixel_values = load2(&mw, "pixel_values", &device);
    let grid = vec![(1usize, 10usize, 10usize)]; // 5x5 merged units -> 4 windows

    let got = model
        .forward(pixel_values, &grid)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let want = npy::load_npy(&mw.join("merger_out.npy")).unwrap().0;
    assert_eq!(
        got.len(),
        want.len(),
        "mw merger_out dim mismatch ({} vs {})",
        got.len(),
        want.len()
    );
    let cos = cosine(&got, &want);
    let ma = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    eprintln!(
        "[vision-parity] MULTI-WINDOW merger_out (25 tokens): cosine={cos:.7} max_abs={ma:e}  (bar 0.999)"
    );
    assert!(cos >= 0.999, "mw vision merger_out cosine={cos:.7} < 0.999");
}
