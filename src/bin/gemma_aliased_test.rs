//! The payoff: load Gemma 4 from a pile with ZERO-COPY weights (each tensor's
//! mmap'd f16 blob aliased straight onto the Metal GPU — no copy, no f32
//! materialization) and prove it generates token-for-token identically to the
//! streaming load. Both run f16 (`BHalf`).
//!
//!   cargo run --release --features gemma --bin gemma_aliased_test
//! macOS / Metal only.

#[cfg(target_os = "macos")]
mod imp {

use mary::models::gemma::gemma4::config::Gemma4Config;
use mary::models::gemma::gemma4::lm::GemmaLM;
use mary::nn::backend::BHalf;
use mary::persist::persist_safetensors_to_pile;
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

pub fn run() {
    let model_id = "google/gemma-4-E2B-it";
    let config_path = hf(model_id, "config.json");
    let _ = hf(model_id, "model.safetensors");
    let tok = hf(model_id, "tokenizer.json");
    let snapshot = Path::new(&config_path).parent().unwrap();
    let device = mary::models::gemma::metal_device::init_metal_device_16gb();

    let pile = std::env::temp_dir().join(format!("mary_gemma_aliased_{}.pile", std::process::id()));
    println!("[persist] {snapshot:?} -> {pile:?} (f16)");
    persist_safetensors_to_pile(snapshot, &pile, mary::ingest::LeafDtype::F16).expect("persist");

    let chat = "<bos><|turn>user\nWhat is 17 times 23? Answer with just the number.<turn|>\n<|turn>model\n";

    println!("[stream f16] loading...");
    let streamed = GemmaLM::<BHalf>::from_streaming_pile(
        Gemma4Config::load(Path::new(&config_path)),
        &pile,
        Path::new(&tok),
        device.clone(),
    );
    let ids_stream = streamed.complete_ids(chat, 10);
    drop(streamed);

    println!("[aliased] loading ZERO-COPY (weights aliased from the pile mmap)...");
    let aliased = GemmaLM::<BHalf>::from_aliased_pile(
        Gemma4Config::load(Path::new(&config_path)),
        &pile,
        Path::new(&tok),
        device,
    );
    let ids_alias = aliased.complete_ids(chat, 10);

    let _ = std::fs::remove_file(&pile);

    println!("streamed ids {ids_stream:?} -> {:?}", aliased.decode(&ids_stream));
    println!("aliased  ids {ids_alias:?} -> {:?}", aliased.decode(&ids_alias));
    if ids_stream == ids_alias {
        println!("=== PASS — zero-copy aliased load is token-identical to streamed (f16) ===");
    } else {
        println!("=== FAIL — aliased diverged from streamed ===");
        std::process::exit(1);
    }
}
}

#[cfg(target_os = "macos")]
fn main() {
    imp::run()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemma_aliased_test: macOS/Metal-only lane (zero-copy GPU aliasing).");
    std::process::exit(2);
}
