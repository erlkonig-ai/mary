//! Numeric-parity gate for the Qwen2.5-VL text backbone of
//! `nomic-embed-multimodal-7b` (`BiQwen2_5`), against deterministic tiny goldens
//! dumped from transformers' real Qwen2.5-VL layer classes.
//!
//! Regenerate goldens (no 16 GB download needed):
//!   python3 scripts/nomic_mm7b_dump_tiny.py
//! Then: cargo test --features ndarray --test nomic_mm7b_parity
//!
//! Each component asserts cosine > 0.999999 vs the PyTorch reference. The full
//! 1-layer text-model check exercises embedding + GQA attention + 1D-RoPE
//! (M-RoPE collapses to standard RoPE for text) + SwiGLU MLP + two RMSNorms +
//! final norm end-to-end. Real-weight, full-scale (28-layer) goldens are a
//! disk-gated follow-up; the math is what this pins.

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::{NdArray, NdArrayDevice};
use mary::models::qwen2_5_vl::config::Qwen2_5VlTextConfig;
use mary::models::qwen2_5_vl::layers::{
    QwenEmbedding, QwenMlp, QwenRmsNorm, QwenTextModel, QwenWeights,
};
use mary::nn::npy;
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
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// npy-dir-backed weight source: logical `a.b.c` → file `a__b__c.npy`.
struct NpyDir<'a> {
    dir: &'a Path,
    device: NdArrayDevice,
}
impl<'a> NpyDir<'a> {
    fn raw(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let file = self.dir.join(format!("{}.npy", name.replace('.', "__")));
        npy::load_npy(&file).unwrap_or_else(|e| panic!("load {}: {e}", file.display()))
    }
}
impl QwenWeights<B> for NpyDir<'_> {
    fn t1(&self, name: &str) -> Tensor<B, 1> {
        let (d, s) = self.raw(name);
        Tensor::from_data(TensorData::new(d, s), &self.device)
    }
    fn t2(&self, name: &str) -> Tensor<B, 2> {
        let (d, s) = self.raw(name);
        Tensor::from_data(TensorData::new(d, s), &self.device)
    }
}

fn load_dir<const D: usize>(dir: &Path, name: &str, device: &NdArrayDevice) -> Tensor<B, D> {
    let (d, s) = npy::load_npy(&dir.join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("load {name}: {e}"));
    Tensor::from_data(TensorData::new(d, s), device)
}

fn assert_parity(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let cos = cosine(got, want);
    let ma = max_abs(got, want);
    assert!(cos > 0.999_999, "{what}: cosine={cos:.9}, max_abs={ma:e}");
    assert!(ma < 2e-4, "{what}: cosine={cos:.9}, max_abs={ma:e}");
    eprintln!("  {what}: cosine={cos:.9}, max_abs={ma:e}  OK");
}

/// Tiny config matching scripts/nomic_mm7b_dump_tiny.py.
fn tiny_cfg() -> Qwen2_5VlTextConfig {
    Qwen2_5VlTextConfig {
        vocab_size: 100,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: Some(16),
        hidden_act: "silu".into(),
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        rope_scaling: None,
        tie_word_embeddings: false,
        attention_dropout: 0.0,
    }
}

fn skip_if_missing(dir: &Path) -> bool {
    if !dir.join("rms_out.npy").exists() {
        eprintln!("skipping: run `python3 scripts/nomic_mm7b_dump_tiny.py` to create {}", dir.display());
        return true;
    }
    false
}

#[test]
fn rms_norm_parity() {
    let dir = golden_dir();
    if skip_if_missing(&dir) { return; }
    let device = NdArrayDevice::default();
    let x = load_dir::<3>(&dir, "rms_in", &device);
    let w = load_dir::<1>(&dir, "rms_weight", &device);
    let want = npy::load_npy(&dir.join("rms_out.npy")).unwrap().0;
    let got = QwenRmsNorm::<B>::from_weight(w, 1e-6).forward(x).into_data().to_vec::<f32>().unwrap();
    assert_parity(&got, &want, "rms_norm");
}

#[test]
fn mlp_parity() {
    let dir = golden_dir();
    if skip_if_missing(&dir) { return; }
    let device = NdArrayDevice::default();
    // Re-key the mlp_* goldens into the names QwenMlp::load expects.
    let x = load_dir::<3>(&dir, "mlp_in", &device);
    let want = npy::load_npy(&dir.join("mlp_out.npy")).unwrap().0;
    struct MlpW<'a> { dir: &'a Path, device: NdArrayDevice }
    impl QwenWeights<B> for MlpW<'_> {
        fn t1(&self, _: &str) -> Tensor<B, 1> { unreachable!() }
        fn t2(&self, name: &str) -> Tensor<B, 2> {
            let key = match name { // map "mlp.gate_proj.weight" -> "mlp_gate_w"
                n if n.ends_with("gate_proj.weight") => "mlp_gate_w",
                n if n.ends_with("up_proj.weight") => "mlp_up_w",
                n if n.ends_with("down_proj.weight") => "mlp_down_w",
                other => panic!("unexpected {other}"),
            };
            load_dir::<2>(self.dir, key, &self.device)
        }
    }
    let w = MlpW { dir: &dir, device };
    let got = QwenMlp::<B>::load(&w, "mlp").forward(x).into_data().to_vec::<f32>().unwrap();
    assert_parity(&got, &want, "swiglu_mlp");
}

#[test]
fn embedding_parity() {
    let dir = golden_dir();
    if skip_if_missing(&dir) { return; }
    let device = NdArrayDevice::default();
    let weight = load_dir::<2>(&dir, "embed_tokens__weight", &device);
    let ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(vec![5i64, 9, 2, 41, 17, 3, 88], [1, 7]), &device);
    let want = npy::load_npy(&dir.join("emb_out.npy")).unwrap().0;
    let got = QwenEmbedding::new(weight).forward(ids).into_data().to_vec::<f32>().unwrap();
    assert_parity(&got, &want, "embedding");
}

#[test]
fn text_model_parity() {
    let dir = golden_dir();
    if skip_if_missing(&dir) { return; }
    let device = NdArrayDevice::default();
    let w = NpyDir { dir: &dir, device };
    let model = QwenTextModel::load(&w, &tiny_cfg(), &device);
    let ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(vec![5i64, 9, 2, 41, 17, 3, 88], [1, 7]), &device);

    // Final hidden states (after the single decoder layer + model.norm).
    let want_final = npy::load_npy(&dir.join("final_out.npy")).unwrap().0;
    let got_final = model.hidden(ids.clone()).into_data().to_vec::<f32>().unwrap();
    assert_parity(&got_final, &want_final, "text_model_final (attn+rope+mlp+norms)");

    // Dense embedding: last-token pool + L2-norm; sanity-check unit norm.
    let emb = model.embed(ids).into_data().to_vec::<f32>().unwrap();
    let n: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((n - 1.0).abs() < 1e-4, "dense embedding not L2-normalized: |emb|={n}");
}
