//! Exercise the playground↔mary seam (mary::local) exactly as the playground's
//! ModelBackend::Local will: load a warm engine, then call generate() with
//! roles+content+params. Proves the trait, chat template, stop strings, and
//! token counts end-to-end.
//!
//! Weights come ONLY from a persisted pile (write one with `gemma_persist`);
//! pass it via GEMMA_PILE. config.json/tokenizer.json stay small HF side-files.
//!
//!   GEMMA_PILE=/path/to/gemma_e2b.pile \
//!   cargo run --release --features local-model --bin local_demo

use mary::local::{load_gemma4_from_persisted_pile_f16, LocalChatTurn, LocalGenParams, LocalRole};
use mary::nn::backend::WgpuDevice;
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
    let tokenizer_path = hf(model_id, "tokenizer.json");
    let pile = std::env::var("GEMMA_PILE")
        .expect("set GEMMA_PILE to a persisted gemma pile (write one with gemma_persist)");

    eprintln!("loading engine from pile {pile}...");
    let mut engine = load_gemma4_from_persisted_pile_f16(
        Path::new(&pile),
        Path::new(&config_path),
        Path::new(&tokenizer_path),
        WgpuDevice::default(),
    )
    .expect("load engine");
    eprintln!("ready.\n");

    let turns = vec![
        LocalChatTurn {
            role: LocalRole::System,
            content: "You are a unix shell. Reply with ONLY the single command, no prose.".into(),
        },
        LocalChatTurn {
            role: LocalRole::User,
            content: "list files in the current directory in long format".into(),
        },
    ];
    let params = LocalGenParams {
        max_tokens: 24,
        stop: vec!["\n".into()],
        ..Default::default()
    };

    let g = engine.generate(&turns, &params).expect("generate");
    println!("command: {:?}", g.text);
    println!(
        "prompt_tokens={} completion_tokens={}",
        g.prompt_tokens, g.completion_tokens
    );
}
