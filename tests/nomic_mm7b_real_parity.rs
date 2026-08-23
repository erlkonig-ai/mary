//! Real-weight, full-scale (28-layer) parity gate for the Qwen2.5-VL text
//! backbone of `nomic-embed-multimodal-7b` (`BiQwen2_5`).
//!
//! Unlike the tiny-golden gate (`nomic_mm7b_parity.rs`, fixed-seed reference
//! classes, no download), this loads the REAL merged backbone from a pile and
//! reproduces the dense `embed_query`/`embed_document` vectors end-to-end, then
//! compares against float32 reference embeddings dumped by the colpali model
//! (`scripts/nomic_mm7b_probe.py`).
//!
//! Prereqs (all disk-gated; the test SKIPS cleanly if missing):
//!   1. python3 scripts/nomic_mm7b_merge.py  <SCRATCH>/merged/merged_text_backbone.safetensors
//!   2. python3 scripts/nomic_mm7b_probe.py   # dumps real reference goldens
//!   3. cargo run --release --features import,hub --bin mary -- import <SCRATCH>/merged --pile <SCRATCH>/nomic_mm7b.pile --key <SCRATCH>/model.key --name 'nomic-ai/nomic-embed-multimodal-7b#text' --dtype f16
//!   4. NOMIC_MM7B_PILE=<SCRATCH>/nomic_mm7b.pile cargo test --release --features gemma --test nomic_mm7b_real_parity -- --nocapture
//!
//! The reference is float32; we store/run the backbone as f16-in-pile upcast to
//! f32 for the NdArray compute. The bar is cosine >= 0.9999 (f16 weight rounding
//! is the dominant residual; see the printed number for the honest margin).

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::{NdArray, NdArrayDevice};
use mary::models::qwen2_5_vl::config::Qwen2_5VlTextConfig;
use mary::models::qwen2_5_vl::layers::{QwenTextModel, QwenWeights};
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

/// f32 keymap (materialized from the pile) as a `QwenWeights` source. Logical
/// dotted names map 1:1 to the persisted safetensors key names because the merge
/// script already strips the `model.` prefix to QwenTextModel naming.
struct KeymapW {
    map: HashMap<String, (Vec<f32>, Vec<usize>)>,
    device: NdArrayDevice,
}
impl KeymapW {
    fn get(&self, name: &str) -> &(Vec<f32>, Vec<usize>) {
        self.map
            .get(name)
            .unwrap_or_else(|| panic!("missing weight {name} in pile keymap"))
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

/// The real Qwen2.5-VL-7B-Instruct text config (the nomic backbone).
fn real_cfg() -> Qwen2_5VlTextConfig {
    Qwen2_5VlTextConfig {
        vocab_size: 152064,
        hidden_size: 3584,
        intermediate_size: 18944,
        num_hidden_layers: 28,
        num_attention_heads: 28,
        num_key_value_heads: 4,
        head_dim: Some(128),
        hidden_act: "silu".into(),
        max_position_embeddings: 128000,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        rope_scaling: None,
        tie_word_embeddings: false,
        attention_dropout: 0.0,
    }
}

fn load_ids(dir: &Path, name: &str, device: &NdArrayDevice) -> (Tensor<B, 2, Int>, usize) {
    let (d, s) = npy::load_npy(&dir.join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("load {name}: {e}"));
    let ids: Vec<i64> = d.iter().map(|&v| v as i64).collect();
    let (b, seq) = (s[0], s[1]);
    (
        Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [b, seq]), device),
        seq,
    )
}

#[test]
fn real_text_embed_parity() {
    let Ok(pile_path) = std::env::var("NOMIC_MM7B_PILE") else {
        eprintln!("SKIP: set NOMIC_MM7B_PILE=<pile> (see file header for the build steps)");
        return;
    };
    let dir = golden_dir();
    if !dir.join("query_emb.npy").exists() {
        eprintln!("SKIP: run scripts/nomic_mm7b_probe.py to dump real reference goldens");
        return;
    }

    let device = NdArrayDevice::default();
    eprintln!("[real-parity] loading merged backbone keymap from pile {pile_path} ...");
    let (_, snapshot) =
        mary::model_collection::load_sole_model_collection_local_latest(Path::new(&pile_path))
            .expect("load native model collection snapshot");
    let map = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.reader(),
        mary::selection::ModelSelector::Source {
            source: mary::models::qwen2_5_vl::NOMIC_MM7B_TEXT_SOURCE,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .expect("select and materialize the Nomic text component");
    eprintln!(
        "[real-parity] keymap has {} tensors; building model ...",
        map.len()
    );
    let w = KeymapW { map, device };
    let model = QwenTextModel::<B>::load(&w, &real_cfg(), &device);

    // Compare every (ids, reference-embedding) pair the probe dumped.
    let cases: &[(&str, &str)] = &[
        ("query_input_ids", "query_emb"),
        ("doc_text_input_ids", "doc_text_emb"),
    ];
    let mut worst = 1.0f64;
    for (ids_name, ref_name) in cases {
        if !dir.join(format!("{ref_name}.npy")).exists() {
            continue;
        }
        let (ids, seq) = load_ids(&dir, ids_name, &device);
        let want = npy::load_npy(&dir.join(format!("{ref_name}.npy")))
            .unwrap()
            .0;
        let got = model.embed(ids).into_data().to_vec::<f32>().unwrap();
        assert_eq!(got.len(), want.len(), "{ref_name}: dim mismatch");
        let cos = cosine(&got, &want);
        let n: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("  {ref_name} (seq={seq}): cosine={cos:.7}  |emb|={n:.5}");
        worst = worst.min(cos);
        assert!(cos >= 0.9999, "{ref_name}: cosine={cos:.7} < 0.9999");
    }
    eprintln!("[real-parity] worst cosine = {worst:.7}  (bar 0.9999)  OK");
}
