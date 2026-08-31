//! Seam test for `NomicMultimodalEmbedder`: the tokenizer + query-augmentation
//! glue must reproduce the exact ids the colpali processor produced (so the
//! already-proven backbone parity, cos 0.9999999 in `nomic_mm7b_real_parity`,
//! carries through end-to-end), and — if the merged pile is available —
//! `embed_query`/`embed_document` match the reference embeddings.
//!
//! Tokenizer path: NOMIC_MM7B_TOKENIZER=<...>/tokenizer.json (the nomic adapter
//! cache copy). Optional model check:
//! NOMIC_MM7B_PILE=<native merged-backbone model-collection pile>.

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::{NdArray, NdArrayDevice};
use mary::models::qwen2_5_vl::config::Qwen2_5VlTextConfig;
use mary::models::qwen2_5_vl::embedder::NomicMultimodalEmbedder;
use mary::models::qwen2_5_vl::layers::{QwenTextModel, QwenWeights};
use mary::nn::npy;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

type B = NdArray;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/nomic_mm7b")
}

fn ids_golden(name: &str) -> Vec<i64> {
    let (d, _) = npy::load_npy(&golden_dir().join(format!("{name}.npy"))).unwrap();
    d.iter().map(|&v| v as i64).collect()
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
impl QwenWeights<B> for KeymapW {
    fn t1(&self, name: &str) -> Tensor<B, 1> {
        let (d, s) = self
            .map
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        Tensor::from_data(TensorData::new(d.clone(), s.clone()), &self.device)
    }
    fn t2(&self, name: &str) -> Tensor<B, 2> {
        let (d, s) = self
            .map
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        Tensor::from_data(TensorData::new(d.clone(), s.clone()), &self.device)
    }
}

const QUERY_TEXT: &str = "What is the capital of France?";
const DOC_TEXT: &str = "The capital of France is Paris, a major European city.";

#[test]
fn query_and_doc_tokenization_matches_reference() {
    let Ok(tok_path) = std::env::var("NOMIC_MM7B_TOKENIZER") else {
        eprintln!("SKIP: set NOMIC_MM7B_TOKENIZER=<.../tokenizer.json>");
        return;
    };
    let device = NdArrayDevice::default();
    // Build an embedder with a 0-layer stub model: we only exercise tokenization.
    let mut cfg = Qwen2_5VlTextConfig::nomic_mm7b();
    cfg.num_hidden_layers = 0;
    cfg.vocab_size = 1; // tiny embed table; we never call embed here
    let w = KeymapW {
        map: stub_weights(&cfg),
        device,
    };
    let model = QwenTextModel::<B>::load(&w, &cfg, &device);
    let tokenizer = Tokenizer::from_file(&tok_path).expect("tokenizer");
    let emb = NomicMultimodalEmbedder::new(model, tokenizer, device);

    let q = emb.embed_query_ids(QUERY_TEXT).expect("tokenize query");
    let d = emb.embed_document_ids(DOC_TEXT).expect("tokenize doc");
    assert_eq!(
        q,
        ids_golden("query_input_ids"),
        "query ids match colpali processor"
    );
    assert_eq!(
        d,
        ids_golden("doc_text_input_ids"),
        "doc ids match colpali processor"
    );
    eprintln!(
        "  query ids ({}) + doc ids ({}) match reference  OK",
        q.len(),
        d.len()
    );
}

#[test]
fn embed_query_and_document_parity() {
    let (Ok(tok_path), Ok(pile)) = (
        std::env::var("NOMIC_MM7B_TOKENIZER"),
        std::env::var("NOMIC_MM7B_PILE"),
    ) else {
        eprintln!("SKIP: set NOMIC_MM7B_TOKENIZER and NOMIC_MM7B_PILE");
        return;
    };
    let device = NdArrayDevice::default();
    let snapshot = mary::model_collection::load_model_collection_local_latest(Path::new(&pile))
        .expect("load native model collection snapshot");
    let map = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        mary::selection::ModelSelector::Source {
            source: mary::models::qwen2_5_vl::NOMIC_MM7B_TEXT_SOURCE,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .expect("select and materialize the Nomic text component");
    let w = KeymapW { map, device };
    let model = QwenTextModel::<B>::load(&w, &Qwen2_5VlTextConfig::nomic_mm7b(), &device);
    let tokenizer = Tokenizer::from_file(&tok_path).expect("tokenizer");
    let emb = NomicMultimodalEmbedder::new(model, tokenizer, device);

    let q = emb.embed_query(QUERY_TEXT).expect("embed query");
    let d = emb.embed_document(DOC_TEXT).expect("embed doc");
    let qcos = cosine(
        &q,
        &npy::load_npy(&golden_dir().join("query_emb.npy"))
            .unwrap()
            .0,
    );
    let dcos = cosine(
        &d,
        &npy::load_npy(&golden_dir().join("doc_text_emb.npy"))
            .unwrap()
            .0,
    );
    eprintln!("  embed_query cos={qcos:.7}  embed_document cos={dcos:.7}  (bar 0.9999)");
    assert!(
        qcos >= 0.9999 && dcos >= 0.9999,
        "embedder text parity below bar"
    );
}

/// Minimal weight set so a 0-layer `QwenTextModel` builds (tokenization-only).
fn stub_weights(cfg: &Qwen2_5VlTextConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut m = HashMap::new();
    m.insert(
        "embed_tokens.weight".into(),
        (
            vec![0.0; cfg.vocab_size * cfg.hidden_size],
            vec![cfg.vocab_size, cfg.hidden_size],
        ),
    );
    m.insert(
        "norm.weight".into(),
        (vec![1.0; cfg.hidden_size], vec![cfg.hidden_size]),
    );
    m
}
