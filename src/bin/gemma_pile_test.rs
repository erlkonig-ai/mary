//! Gemma 4 → mary pile → Gemma 4 round-trip parity gate.
//!
//! Loads google/gemma-4-E2B-it two ways and asserts they generate the EXACT
//! same token-id sequence for a fixed prompt:
//!   1. the direct safetensors path (`GemmaLM::load` → `load_gemma4`).
//!   2. the pile path: ingest the safetensors into an in-memory mary pile
//!      (`save_safetensors` — each tensor a content-addressed f32 leaf),
//!      materialize the weights back into a keymap (`load_keymap`), and build the
//!      model from the keymap (`GemmaLM::from_keymap` → `load_gemma4_from_keymap`).
//!
//! The f32 blobs store weights exactly, so the round-trip is lossless and the two
//! token streams must be identical. This is the "the brain is a trible graph"
//! proof for the text LLM, mirroring `smolvla_pile_test`.
//!
//!   cargo run --release --features gemma --bin gemma_pile_test

use mary::ingest::{load_keymap, save_safetensors};
use mary::models::gemma::gemma4::config::Gemma4Config;
use mary::models::gemma::gemma4::lm::GemmaLM;
use mary::nn::backend::{WgpuDevice, B};
use mary::nn::weight_loader::read_safetensors_file;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use triblespace::prelude::*;

const MODEL_ID: &str = "google/gemma-4-E2B-it";
const PROMPT: &str = "What is 17 times 23? Answer with just the number.";
const MAX_NEW: usize = 10;

fn find_hf_file(model_id: &str, filename: &str) -> String {
    let o = Command::new("python3")
        .args([
            "-c",
            &format!(
                "from huggingface_hub import hf_hub_download; print(hf_hub_download('{}', '{}'))",
                model_id, filename
            ),
        ])
        .output()
        .unwrap();
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

fn main() {
    let device = WgpuDevice::default();

    // Resolve config + tokenizer + the single safetensors shard from the HF cache.
    let config_path = find_hf_file(MODEL_ID, "config.json");
    let _ = find_hf_file(MODEL_ID, "model.safetensors"); // force snapshot to hold the shard
    let tokenizer_path = find_hf_file(MODEL_ID, "tokenizer.json");
    let config = Gemma4Config::load(Path::new(&config_path));

    let snapshot_dir = Path::new(&config_path).parent().unwrap();
    let mut shard_paths: Vec<String> = std::fs::read_dir(snapshot_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().to_string())
        .filter(|p| p.ends_with(".safetensors"))
        .collect();
    shard_paths.sort();
    let paths: Vec<&Path> = shard_paths.iter().map(|s| Path::new(s.as_str())).collect();

    // The chat string both paths decode (greedy, deterministic).
    let chat = format!("<bos><|turn>user\n{PROMPT}<turn|>\n<|turn>model\n");

    // ── Path 1: direct safetensors ─────────────────────────────────────────
    eprintln!(
        "[safetensors] loading {MODEL_ID} from {} shard(s)...",
        paths.len()
    );
    let lm_safe = GemmaLM::<B>::load(
        config.clone(),
        &paths,
        Path::new(&tokenizer_path),
        device.clone(),
    );
    let ids_safe = lm_safe.complete_ids(&chat, MAX_NEW);
    let text_safe = lm_safe.decode(&ids_safe);
    drop(lm_safe);

    // ── Path 2: pile round-trip ────────────────────────────────────────────
    eprintln!("[pile] ingesting safetensors → in-memory pile...");
    let mut blobs = MemoryBlobStore::new();
    let mut tribles = TribleSet::new();
    let mut model_ids: Vec<Id> = Vec::new();
    for shard in &paths {
        let bytes = read_safetensors_file(shard);
        let name = shard
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("gemma4");
        let frag = save_safetensors(&bytes, name, &mut blobs, mary::ingest::LeafDtype::F16)
            .expect("ingest");
        model_ids.push(frag.root().expect("model root"));
        tribles += frag;
    }
    eprintln!("[pile] ingested → {} tribles", tribles.len());
    let reader = BlobStore::reader(&mut blobs).expect("reader");
    let mut keymap: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for id in model_ids {
        keymap.extend(load_keymap(&tribles, &reader, id));
    }
    eprintln!("[pile] materialized {} tensors from pile", keymap.len());

    let lm_pile = GemmaLM::<B>::from_keymap(
        config.clone(),
        keymap,
        Path::new(&tokenizer_path),
        device.clone(),
    );
    let ids_pile = lm_pile.complete_ids(&chat, MAX_NEW);
    let text_pile = lm_pile.decode(&ids_pile);

    // ── Parity gate ────────────────────────────────────────────────────────
    println!("\nprompt: {PROMPT}");
    println!("safetensors → ids {ids_safe:?}\n              text {text_safe:?}");
    println!("pile        → ids {ids_pile:?}\n              text {text_pile:?}");

    if ids_safe == ids_pile {
        println!("\nPASS — token-id sequences identical ({} tokens). Gemma 4 round-trips through the mary pile.", ids_safe.len());
    } else {
        println!("\nFAIL — token-id sequences DIVERGE.");
        std::process::exit(1);
    }
}
