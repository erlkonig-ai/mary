//! Parity gate for the STREAMING pile load: a model streamed from a pile
//! (handles indexed once, each tensor read on demand, peak CPU = one tensor)
//! must generate token-for-token identically to the materialized-keymap load
//! from the same pile. This is the path that scales weights-as-tribles to the
//! 31B; here it's proven lossless on the small E2B.
//!
//!   cargo run --release --features gemma --bin gemma_stream_test

use mary::models::gemma::gemma4::config::Gemma4Config;
use mary::models::gemma::gemma4::lm::GemmaLM;
use mary::nn::backend::{WgpuDevice, B};
use mary::persist::{load_keymap_from_pile, persist_safetensors_to_pile};
use std::path::Path;
use std::process::Command;

fn hf(model_id: &str, file: &str) -> String {
    let o = Command::new("python3")
        .args(["-c", &format!(
            "from huggingface_hub import hf_hub_download; print(hf_hub_download('{model_id}', '{file}'))"
        )])
        .output()
        .unwrap();
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

fn main() {
    let model_id = "google/gemma-4-E2B-it";
    let config_path = hf(model_id, "config.json");
    let _ = hf(model_id, "model.safetensors");
    let tokenizer_path = hf(model_id, "tokenizer.json");
    let snapshot_dir = Path::new(&config_path).parent().unwrap();
    let device = WgpuDevice::default();

    let pile = std::env::temp_dir().join(format!("mary_gemma_stream_{}.pile", std::process::id()));
    println!("[persist] {snapshot_dir:?} -> {pile:?}");
    persist_safetensors_to_pile(snapshot_dir, &pile, mary::ingest::LeafDtype::F16).expect("persist");
    println!("[persist] pile is {} bytes", std::fs::metadata(&pile).map(|m| m.len()).unwrap_or(0));

    let chat = "<bos><|turn>user\nWhat is 17 times 23? Answer with just the number.<turn|>\n<|turn>model\n";

    println!("[stream] loading via from_streaming_pile (peak = one tensor)...");
    let streamed = GemmaLM::<B>::from_streaming_pile(
        Gemma4Config::load(Path::new(&config_path)),
        &pile,
        Path::new(&tokenizer_path),
        device,
    );
    let ids_stream = streamed.complete_ids(chat, 10);
    drop(streamed);

    println!("[keymap] loading via materialized load_keymap_from_pile...");
    let keymap = load_keymap_from_pile(&pile).expect("keymap");
    let materialized = GemmaLM::<B>::from_keymap(
        Gemma4Config::load(Path::new(&config_path)),
        keymap,
        Path::new(&tokenizer_path),
        WgpuDevice::default(),
    );
    let ids_keymap = materialized.complete_ids(chat, 10);

    let _ = std::fs::remove_file(&pile);

    println!("streaming   ids {ids_stream:?} -> {:?}", streamed_text(&materialized, &ids_stream));
    println!("materialized ids {ids_keymap:?} -> {:?}", streamed_text(&materialized, &ids_keymap));
    assert_eq!(ids_stream, ids_keymap, "streaming pile load diverged from materialized");
    println!("=== PASS — streaming pile load is token-identical to materialized ===");
}

fn streamed_text(lm: &GemmaLM<B>, ids: &[u32]) -> String {
    lm.decode(ids)
}
