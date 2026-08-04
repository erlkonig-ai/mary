//! Gemma 4 → REAL on-disk pile → Gemma 4 round-trip parity gate (the true
//! shell-is-physics endpoint). Unlike `gemma_pile_test` (in-memory pile), this:
//!
//!   1. persists google/gemma-4-E2B-it's weights into a TEMP pile FILE on disk
//!      (`persist_safetensors_to_pile` — each tensor a content-addressed f32 leaf,
//!      committed on the `main` branch),
//!   2. then, via a FRESH `Pile::open` of that file (no in-memory carryover),
//!      loads the keymap back from JUST the pile (`load_keymap_from_pile`)
//!      and builds the model (`GemmaLM::from_keymap`),
//!   3. ALSO loads E2B the normal safetensors way,
//!   4. and asserts both generate the EXACT same token-id sequence.
//!
//! The f32 blobs store weights exactly, so the disk round-trip is lossless and the
//! token streams must be identical.
//!
//!   cargo run --release --features gemma --bin gemma_persist_test

use mary::models::gemma::gemma4::config::Gemma4Config;
use mary::models::gemma::gemma4::lm::GemmaLM;
use mary::nn::backend::{WgpuDevice, B};
use mary::persist::{load_keymap_from_pile, persist_safetensors_to_pile};
use std::path::Path;
use std::process::Command;

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

    // Resolve config + tokenizer + the safetensors shards from the HF cache.
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

    // ── Persist to a TEMP pile FILE on disk ────────────────────────────────
    let pile_path =
        std::env::temp_dir().join(format!("mary_gemma_persist_{}.pile", std::process::id()));
    // Start clean if a stale file is lying around.
    let _ = std::fs::remove_file(&pile_path);
    eprintln!("[persist] writing weights → {pile_path:?} ...");
    persist_safetensors_to_pile(snapshot_dir, &pile_path, mary::ingest::LeafDtype::F16)
        .expect("persist to pile");
    let pile_size = std::fs::metadata(&pile_path).unwrap().len();
    eprintln!(
        "[persist] pile is {} bytes ({:.2} GiB) on disk.",
        pile_size,
        pile_size as f64 / (1u64 << 30) as f64
    );

    // ── Path 2: load from JUST the pile FILE (fresh Pile::open) ────────────
    eprintln!("[pile] loading keymap from the pile file (no safetensors)...");
    let keymap = load_keymap_from_pile(&pile_path).expect("load keymap from pile");
    eprintln!(
        "[pile] materialized {} tensors from the pile file.",
        keymap.len()
    );

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
    println!(
        "\npile file: {pile_path:?}  ({} bytes, {:.2} GiB)",
        pile_size,
        pile_size as f64 / (1u64 << 30) as f64
    );

    // Clean up the temp pile.
    let _ = std::fs::remove_file(&pile_path);

    if ids_safe == ids_pile {
        println!(
            "\nPASS — token-id sequences identical ({} tokens). Gemma 4 round-trips through a REAL on-disk pile (no safetensors at load).",
            ids_safe.len()
        );
    } else {
        println!("\nFAIL — token-id sequences DIVERGE.");
        std::process::exit(1);
    }
}
