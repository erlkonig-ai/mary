//! Parity gate for the STREAMING native-collection load: a selected model
//! streamed from a pile (handles indexed once, each tensor read on demand,
//! peak CPU = one tensor) must generate token-for-token identically to the
//! explicitly selected materialized-keymap load from the same collection.
//! This is the path that scales weights-as-tribles to the 31B; here it's proven
//! lossless on the small E2B.
//!
//!   cargo run --release --features gemma,import --bin gemma_stream_test

#[path = "support/native_model_fixture.rs"]
mod native_model_fixture;

use crate::native_model_fixture::import_native_model_fixture;
use mary::models::gemma::gemma4::config::Gemma4Config;
use mary::models::gemma::gemma4::lm::GemmaLM;
use mary::nn::backend::{B, WgpuDevice};
use mary::selection::{ModelSelector, SelectedModelIndex};
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
    let _ = std::fs::remove_file(&pile);
    println!("[persist] {snapshot_dir:?} -> {pile:?} (native collection, f16)");
    let imported_root =
        import_native_model_fixture(snapshot_dir, &pile, mary::ingest::LeafDtype::F16, model_id)
            .expect("import native model collection");
    println!(
        "[persist] pile is {} bytes",
        std::fs::metadata(&pile).map(|m| m.len()).unwrap_or(0)
    );

    let chat = "<bos><|turn>user\nWhat is 17 times 23? Answer with just the number.<turn|>\n<|turn>model\n";

    println!("[stream] selecting the native root and loading (peak = one tensor)...");
    let (_, snapshot) = mary::model_collection::load_sole_model_collection_local_latest(&pile)
        .expect("load native model collection snapshot for streaming");
    let selected = SelectedModelIndex::from_snapshot(
        snapshot,
        ModelSelector::Source {
            source: model_id,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .expect("select imported model for streaming");
    assert_eq!(selected.single_root(), Some(imported_root));
    let (streamed_model, _vision) = mary::persist::load_gemma4_streaming_from_index::<B, _>(
        selected,
        Gemma4Config::load(Path::new(&config_path)),
        &device,
    );
    let streamed = GemmaLM::<B>::from_model(
        Gemma4Config::load(Path::new(&config_path)),
        streamed_model,
        Path::new(&tokenizer_path),
        device.clone(),
    );
    let ids_stream = streamed.complete_ids(chat, 10);
    drop(streamed);

    println!("[keymap] materializing the explicitly selected native root...");
    let (_, snapshot) = mary::model_collection::load_sole_model_collection_local_latest(&pile)
        .expect("load native model collection snapshot for materialization");
    let keymap = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Source {
            source: model_id,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .expect("materialize selected model keymap");
    let materialized = GemmaLM::<B>::from_keymap(
        Gemma4Config::load(Path::new(&config_path)),
        keymap,
        Path::new(&tokenizer_path),
        WgpuDevice::default(),
    );
    let ids_keymap = materialized.complete_ids(chat, 10);

    let _ = std::fs::remove_file(&pile);

    println!(
        "streaming   ids {ids_stream:?} -> {:?}",
        streamed_text(&materialized, &ids_stream)
    );
    println!(
        "materialized ids {ids_keymap:?} -> {:?}",
        streamed_text(&materialized, &ids_keymap)
    );
    assert_eq!(
        ids_stream, ids_keymap,
        "streaming native-collection load diverged from materialized"
    );
    println!("=== PASS — streaming native load is token-identical to materialized ===");
}

fn streamed_text(lm: &GemmaLM<B>, ids: &[u32]) -> String {
    lm.decode(ids)
}
