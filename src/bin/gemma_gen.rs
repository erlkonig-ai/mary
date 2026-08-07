//! Gemma 4 as a pure TEXT LLM, in-process inside mary — no ollama, no HTTP, no
//! OpenAI shim. The brain runs in the substrate (shell-is-physics). This bin is
//! a thin CLI over the real seam, `mary::models::gemma::gemma4::lm::GemmaLM` —
//! the same warm handle the playground's `ModelBackend::Local` calls in-process.
//!
//! Weights come ONLY from a persisted pile (write one with `gemma_persist`);
//! `config.json` + `tokenizer.json` stay small files resolved from the local
//! HF snapshot of `--model`.
//!
//!   cargo run --release --features gemma --bin gemma_gen -- \
//!     --pile /path/to/gemma.pile --prompt "Explain what a trible is." \
//!     [--model e4b] [--tokens 120]
//!
//! `--pile` falls back to the `GEMMA_PILE` env var. Pass --model to point the
//! config/tokenizer resolution at any Gemma 4 variant — short aliases
//! (e2b|e4b|12b|26b|31b) or a full HF id both work. The dense 12B runs f32 on
//! 128GB; build with the extra `f16gen` feature to run f16 weights (the 31B
//! fits 128GB only at f16 — streamed from the pile either way).

use mary::models::gemma::gemma4::config::Gemma4Config;
use mary::models::gemma::gemma4::lm::GemmaLM;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

// Half-precision (f16) weights: 31B dense becomes ~62GB instead of ~124GB f32,
// the only way the flagship fits 128GB. f16 inference is standard; validated
// to match f32 output on the small E2B.
#[cfg(feature = "f16gen")]
use mary::nn::backend::BHalf as B;
#[cfg(not(feature = "f16gen"))]
use mary::nn::backend::B;

/// Resolve a SMALL side-file (config.json / tokenizer.json) from the local HF
/// snapshot. Weights never come from here — they load from the pile.
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
        .unwrap_or_else(|e| panic!("hf_hub_download {model_id}/{filename}: {e}"));
    let p = String::from_utf8(o.stdout)
        .expect("utf8")
        .trim()
        .to_string();
    if p.is_empty() || !Path::new(&p).exists() {
        panic!("hf_hub_download failed for {model_id}/{filename}");
    }
    p
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|s| s == k)
        .map(|i| args[i + 1].clone())
}

/// Expand short variant aliases to full HF model ids; anything else (a full
/// id) passes through untouched.
fn resolve_model_id(arg: &str) -> String {
    match arg.to_ascii_lowercase().as_str() {
        "e2b" => "google/gemma-4-E2B-it".into(),
        "e4b" => "google/gemma-4-E4B-it".into(),
        "12b" => "google/gemma-4-12B-it".into(),
        "26b" => "google/gemma-4-26B-A4B-it".into(),
        "31b" => "google/gemma-4-31B-it".into(),
        _ => arg.to_string(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = arg(&args, "--prompt")
        .unwrap_or_else(|| "Explain what a trible is in one sentence.".into());
    let model_id =
        resolve_model_id(&arg(&args, "--model").unwrap_or_else(|| "google/gemma-4-E2B-it".into()));
    let max_new = arg(&args, "--tokens")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(120);
    let pile = arg(&args, "--pile")
        .or_else(|| std::env::var("GEMMA_PILE").ok())
        .unwrap_or_else(|| {
            eprintln!("gemma_gen: pass --pile <gemma.pile> or set GEMMA_PILE (write one with gemma_persist)");
            std::process::exit(2);
        });
    // Use a Metal device with the storage-buffer-binding cap raised past wgpu's
    // 4 GiB default. The dense 31B's embedding is 5.6 GB at f32 / 2.8 GB at f16;
    // at f32 it exceeds the cap and panics in cubecl's dispatch (server.rs:270).
    // Harmless for the small models. (Pair with the f16gen feature for 31B so
    // 60 GB f16 fits 128 GB — f32 31B is ~120 GB and will not.)
    let device = mary::models::gemma::metal_device::init_metal_device_16gb();

    // Resolve config + tokenizer (small files) from the HF snapshot.
    let config_path = find_hf_file(&model_id, "config.json");
    let tokenizer_path = find_hf_file(&model_id, "tokenizer.json");
    let mut config = Gemma4Config::load(Path::new(&config_path));
    // Text-only bin: GemmaLM discards the vision encoder and never touches
    // audio, so skip loading those weights entirely. This is also what lets
    // the encoder-free "unified" 12B run the shared text path — its vision
    // embedder is a different structure the text loader shouldn't parse.
    config.vision_config = None;
    config.audio_config = None;

    eprintln!(
        "Loading {model_id} ({} hidden, {} layers, vocab {}) from pile {pile} (streaming)...",
        config.text_config.hidden_size,
        config.text_config.num_hidden_layers,
        config.text_config.vocab_size
    );
    let t_load = Instant::now();
    let lm = GemmaLM::<B>::from_streaming_pile(
        config,
        Path::new(&pile),
        Path::new(&tokenizer_path),
        device,
    );
    eprintln!("Loaded in {:.1}s.\n", t_load.elapsed().as_secs_f64());

    let t_gen = Instant::now();
    let out = lm.generate(&prompt, max_new);
    println!("{prompt}\n→ {out}");
    eprintln!("\n(generated in {:.2}s)", t_gen.elapsed().as_secs_f64());
}
