//! LoRA training smoke for the Gemma 4 E4B text decoder: overfit ONE sentence.
//!
//! The de-risk gate for finetuning the brain in-substrate (the gemma analog of
//! smolvla_overfit): load the full E4B in f32 under `Autodiff<Metal>`, attach
//! rank-16 LoRA adapters to every attention/MLP projection, and drive next-token
//! cross-entropy on a single hardcoded sentence with hand-rolled AdamW until the
//! loss collapses. The base model stays frozen — only the adapters (~50M params)
//! receive gradients. If the loss falls markedly, the full mechanic
//! (forward → CE → backward → AdamW → adapter update) is proven on this machine;
//! everything past this is data.
//!
//!   cargo run --release --features gemma --bin gemma_train_lora -- \
//!     [--model e4b|12b|<full HF id>] [--steps 200] [--lr 5e-4] [--rank 16] \
//!     [--alpha 16] [--text "..."] [--out /tmp/gemma_e4b_lora.safetensors]
//!
//! Memory: E4B f32 resident is ~30GB GPU + a transient CPU spike while the PLE
//! table is sliced; the autodiff graph over a ~20-token sequence adds little.
//! The dense 12B f32 is ~48GB of weights and trains within 128GB.
//! Do NOT point this at the 31B (f32 would be ~124GB).

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use burn::prelude::*;
use mary::models::gemma::gemma4::config::Gemma4Config;
use mary::models::gemma::gemma4::weights::load_gemma4;
use mary::models::gemma::lora::LoraWeights;
use mary::nn::backend::BTrain as B;
use mary::nn::backend::B as Inner;
use tokenizers::Tokenizer;

fn try_hf_file(model_id: &str, filename: &str) -> Option<String> {
    let o = Command::new("python3")
        .args([
            "-c",
            &format!(
                "from huggingface_hub import hf_hub_download; print(hf_hub_download('{}', '{}'))",
                model_id, filename
            ),
        ])
        .output()
        .ok()?;
    let p = String::from_utf8(o.stdout).ok()?.trim().to_string();
    if p.is_empty() || !Path::new(&p).exists() { None } else { Some(p) }
}

fn find_hf_file(model_id: &str, filename: &str) -> String {
    try_hf_file(model_id, filename)
        .unwrap_or_else(|| panic!("hf_hub_download failed for {model_id}/{filename}"))
}

/// Resolve the safetensors shard(s) for a model, robust to HF's split-snapshot
/// layout. Prefer the single-file `model.safetensors`; otherwise read
/// `model.safetensors.index.json`'s `weight_map` and resolve each unique shard
/// filename via hf_hub_download. The dense 12B/31B are sharded; E2B/E4B are not.
fn resolve_shards(model_id: &str) -> Vec<String> {
    if let Some(single) = try_hf_file(model_id, "model.safetensors") {
        return vec![single];
    }
    let index_path = find_hf_file(model_id, "model.safetensors.index.json");
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).expect("read index.json"))
            .expect("parse index.json");
    let mut names: Vec<String> = index["weight_map"]
        .as_object()
        .expect("index.json weight_map")
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    names.iter().map(|n| find_hf_file(model_id, n)).collect()
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|s| s == k).map(|i| args[i + 1].clone())
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

/// AdamW state for one parameter (f32 on the inner backend).
struct AdamWParamState {
    m: Tensor<Inner, 2>,
    v: Tensor<Inner, 2>,
}

/// Hand-rolled AdamW over raw f32 tensors (same as avatar's train_lora; no
/// loss-scaling bridge needed here — the whole path is f32).
struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
    states: HashMap<String, AdamWParamState>,
}

impl AdamW {
    fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step: 0,
            states: HashMap::new(),
        }
    }

    fn step_param(
        &mut self,
        key: &str,
        param: Tensor<Inner, 2>,
        grad: Tensor<Inner, 2>,
    ) -> Tensor<Inner, 2> {
        let state = self.states.entry(key.to_string()).or_insert_with(|| {
            let shape = param.dims();
            let device = param.device();
            AdamWParamState {
                m: Tensor::zeros(shape, &device),
                v: Tensor::zeros(shape, &device),
            }
        });

        // Update moments
        state.m = state.m.clone() * self.beta1 + grad.clone() * (1.0 - self.beta1);
        state.v = state.v.clone() * self.beta2 + grad.clone() * grad * (1.0 - self.beta2);

        // Bias correction
        let t = self.step as f32 + 1.0;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        let m_hat = state.m.clone() / bc1;
        let v_hat = state.v.clone() / bc2;

        // AdamW: param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
        let update = m_hat / (v_hat.sqrt() + self.eps) + param.clone() * self.weight_decay;
        param - update * self.lr
    }

    fn increment_step(&mut self) {
        self.step += 1;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_id = resolve_model_id(&arg(&args, "--model").unwrap_or_else(|| "google/gemma-4-E4B-it".into()));
    let steps: usize = arg(&args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(200);
    let lr: f32 = arg(&args, "--lr").and_then(|s| s.parse().ok()).unwrap_or(5e-4);
    let rank: usize = arg(&args, "--rank").and_then(|s| s.parse().ok()).unwrap_or(16);
    let alpha: f32 = arg(&args, "--alpha").and_then(|s| s.parse().ok()).unwrap_or(16.0);
    let weight_decay: f32 = arg(&args, "--weight-decay").and_then(|s| s.parse().ok()).unwrap_or(0.01);
    let text = arg(&args, "--text").unwrap_or_else(|| {
        "The quick brown fox jumps over the lazy dog beside the quiet riverbank.".into()
    });
    let out = arg(&args, "--out").unwrap_or_else(|| "/tmp/gemma_e4b_lora.safetensors".into());

    eprintln!("=== Gemma 4 LoRA overfit smoke ===");
    eprintln!("Model: {model_id}  rank={rank} alpha={alpha} lr={lr} steps={steps}");
    eprintln!("Text:  {text:?}");

    // Raised storage-buffer cap (the E4B embedding is 2.7GB f32; harmless here,
    // required for bigger tensors) — same device init as gemma_gen.
    let device = mary::models::gemma::metal_device::init_metal_device_16gb();

    // Resolve config + tokenizer + weight shard(s) from the HF snapshot.
    let config_path = find_hf_file(&model_id, "config.json");
    let tokenizer_path = find_hf_file(&model_id, "tokenizer.json");
    let shard_paths = resolve_shards(&model_id);
    let weight_paths: Vec<&Path> = shard_paths.iter().map(|s| Path::new(s.as_str())).collect();
    let mut config = Gemma4Config::load(Path::new(&config_path));
    // Text-only training: skip the vision encoder load (audio is never loaded
    // by load_gemma4). Saves ~4GB of weights we would never touch.
    config.vision_config = None;
    config.audio_config = None;

    eprintln!(
        "Loading {model_id} ({} hidden, {} layers, vocab {}) in f32 under Autodiff<Metal>...",
        config.text_config.hidden_size,
        config.text_config.num_hidden_layers,
        config.text_config.vocab_size
    );
    let t_load = Instant::now();
    let (model, _vision) = load_gemma4::<B>(config, &weight_paths, &device);
    eprintln!("Loaded in {:.1}s.", t_load.elapsed().as_secs_f64());

    let (rope_s, rope_g) = model.rope_tables(&device);
    let scale = (model.config.hidden_size as f64).sqrt() as f32;

    // Tokenize the one training sentence (with BOS so position 0 is standard).
    let tokenizer = Tokenizer::from_file(Path::new(&tokenizer_path))
        .unwrap_or_else(|e| panic!("tokenizer {tokenizer_path:?}: {e}"));
    let ids: Vec<i32> = tokenizer
        .encode(format!("<bos>{text}"), false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&x| x as i32)
        .collect();
    let s = ids.len();
    assert!(s >= 2, "need at least 2 tokens to form a next-token pair");
    eprintln!("Tokenized to {s} tokens.");
    let tokens = Tensor::<B, 1, Int>::from_ints(&ids[..], &device).reshape([1, s]);
    // Shifted next-token targets: predict ids[1..] from positions 0..s-1.
    let targets = Tensor::<B, 1, Int>::from_ints(&ids[1..], &device);

    // LoRA adapters as trainable leaves; base model stays frozen.
    let mut lora = LoraWeights::<B>::init_gemma4(&model.config, rank, alpha, &device);
    for adapter in lora.adapters.values_mut() {
        adapter.lora_a = adapter.lora_a.clone().require_grad();
        adapter.lora_b = adapter.lora_b.clone().require_grad();
    }
    eprintln!(
        "{} LoRA adapters, {} trainable parameters.",
        lora.adapters.len(),
        lora.num_params()
    );

    let ce = burn::nn::loss::CrossEntropyLoss::new(None, &device);
    let mut optimizer = AdamW::new(lr, weight_decay);
    let vocab = model.config.vocab_size;

    let t_train = Instant::now();
    let mut first_loss = f32::NAN;
    let mut last_loss = f32::NAN;
    for step in 0..steps {
        let t_step = Instant::now();

        // Fresh KV caches every step — prefill-only forward, no decode.
        let mut caches = model.new_caches();
        let emb = model.decoder.embed.forward(tokens.clone()).mul_scalar(scale);
        let logits = model.forward_embeds(
            emb,
            tokens.clone(),
            &rope_s,
            &rope_g,
            &mut caches,
            &[],
            Some(&lora),
        );

        // Next-token CE: logits[:, :-1] vs tokens[:, 1:].
        let logits_shift = logits
            .slice([0..1, 0..(s - 1), 0..vocab])
            .reshape([s - 1, vocab]);
        let loss = ce.forward(logits_shift, targets.clone());
        let loss_val: f32 = loss.clone().into_scalar().elem();
        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        let grads = loss.backward();

        // AdamW on every adapter; update on the inner backend and re-lift as a
        // fresh trainable leaf so the graph never accumulates across steps.
        for (key, adapter) in lora.adapters.iter_mut() {
            if let Some(grad_a) = adapter.lora_a.grad(&grads) {
                let updated = optimizer.step_param(
                    &format!("{key}.lora_A"),
                    adapter.lora_a.clone().inner(),
                    grad_a,
                );
                adapter.lora_a = Tensor::from_inner(updated).require_grad();
            }
            if let Some(grad_b) = adapter.lora_b.grad(&grads) {
                let updated = optimizer.step_param(
                    &format!("{key}.lora_B"),
                    adapter.lora_b.clone().inner(),
                    grad_b,
                );
                adapter.lora_b = Tensor::from_inner(updated).require_grad();
            }
        }
        optimizer.increment_step();

        eprintln!(
            "step {:>4}/{steps}  CE = {loss_val:.6}  ({:.2}s)",
            step + 1,
            t_step.elapsed().as_secs_f64()
        );
    }
    eprintln!(
        "\nCE {first_loss:.4} -> {last_loss:.6}  ({:.1}x reduction over {steps} AdamW steps, {:.1}s total)",
        first_loss / last_loss.max(1e-9),
        t_train.elapsed().as_secs_f64()
    );

    lora.save(Path::new(&out));

    assert!(
        last_loss < first_loss * 0.5,
        "loss did not fall markedly — LoRA training loop broken"
    );
    println!(
        "✓ LoRA overfit smoke passed: CE {first_loss:.4} -> {last_loss:.6}; adapters saved to {out}"
    );
}
