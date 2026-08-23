//! Dump gemma-4-31B-it's final-position logits (loaded from the pile, streamed,
//! f16 weights) for a fixed chat string, so a Python HF-transformers reference
//! can compare them (cos + argmax). The house-standard cross-check that the
//! ported 31B is numerically faithful, not just self-consistent.
//!
//!   cargo run --release --features gemma,f16gen --bin gemma31b_parity -- \
//!     --pile models/gemma_31b.pile [--out /tmp/gemma31b_parity/mary_logits.bin]
//!
//! Writes raw little-endian f32[vocab] to `--out`. Also prints the argmax token.

use burn::prelude::*;
use mary::models::gemma::gemma4::config::Gemma4Config;
use std::io::Write;
use std::path::Path;
use std::process::Command;

#[cfg(feature = "f16gen")]
use mary::nn::backend::BHalf as B;
#[cfg(not(feature = "f16gen"))]
use mary::nn::backend::B;

const PROMPT: &str = "What is the capital of France? Answer in one word.";

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
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|s| s == k)
        .map(|i| args[i + 1].clone())
}

fn argmax(v: &[f32]) -> usize {
    let mut i = 0;
    let mut b = f32::NEG_INFINITY;
    for (k, &x) in v.iter().enumerate() {
        if x > b {
            b = x;
            i = k;
        }
    }
    i
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_id = "google/gemma-4-31B-it";
    let pile = arg(&args, "--pile").expect("--pile <gemma_31b.pile>");
    let out = arg(&args, "--out").unwrap_or_else(|| "/tmp/gemma31b_parity/mary_logits.bin".into());

    let config_path = find_hf_file(model_id, "config.json");
    let tokenizer_path = find_hf_file(model_id, "tokenizer.json");
    let mut config = Gemma4Config::load(Path::new(&config_path));
    // Text-only forward: the LM path discards vision/audio (parity is on the
    // text decoder + lm_head, which is what the background LLM runs).
    config.vision_config = None;
    config.audio_config = None;

    let device = mary::models::gemma::metal_device::init_metal_device_16gb();

    eprintln!(
        "[parity] loading {model_id} ({} hidden, {} layers, vocab {}) streaming from {pile}...",
        config.text_config.hidden_size,
        config.text_config.num_hidden_layers,
        config.text_config.vocab_size
    );
    // Build the model directly (not via GemmaLM) so we can grab the raw logits.
    let (model, _vision) = mary::persist::load_gemma4_streaming_from_pile::<B>(
        Path::new(&pile),
        mary::selection::ModelSelector::Source {
            source: model_id,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
        config.clone(),
        &device,
    )
    .expect("stream 31b from pile");

    // Tokenize the exact mary chat template (special_tokens=false, so the <bos>
    // literal in the string is the only bos — matches the HF ref's
    // add_special_tokens=False).
    let tokenizer =
        tokenizers::Tokenizer::from_file(Path::new(&tokenizer_path)).expect("tokenizer");
    let chat = format!("<bos><|turn>user\n{PROMPT}<turn|>\n<|turn>model\n");
    let ids: Vec<i32> = tokenizer
        .encode(chat.as_str(), false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&x| x as i32)
        .collect();
    eprintln!("[parity] token ids ({}): {:?}", ids.len(), ids);

    let (rope_s, rope_g) = model.rope_tables(&device);
    let scale = (config.text_config.hidden_size as f64).sqrt() as f32;
    let n = ids.len();
    let tokens = Tensor::<B, 1, Int>::from_ints(&ids[..], &device).reshape([1, n]);
    let emb = model
        .decoder
        .embed
        .forward(tokens.clone())
        .mul_scalar(scale);
    let mut caches = model.new_caches();
    let l = model.forward_embeds(
        emb,
        tokens.clone(),
        &rope_s,
        &rope_g,
        &mut caches,
        &[],
        None,
    );
    let [_, sl, vv] = l.dims();
    let last: Vec<f32> = l
        .slice([0..1, (sl - 1)..sl, 0..vv])
        .reshape([vv])
        .to_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    let am = argmax(&last);
    eprintln!(
        "[parity] mary argmax token id = {am} -> {:?}",
        tokenizer.decode(&[am as u32], false).unwrap_or_default()
    );
    eprintln!("[parity] vocab = {}", last.len());

    // Write raw little-endian f32[vocab].
    let mut bytes = Vec::with_capacity(last.len() * 4);
    for &x in &last {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    if let Some(parent) = Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = std::fs::File::create(&out).expect("create out");
    f.write_all(&bytes).expect("write logits");
    eprintln!("[parity] wrote {} f32 logits -> {out}", last.len());
    println!("MARY_ARGMAX {am}");
}
