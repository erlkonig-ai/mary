//! Persist the four embedders from the local HF cache into their standalone
//! piles — the durable weight stores `mary::embed`'s `load_*_from_pile`
//! constructors (and downstream semantic search) load from. One pile PER
//! embedder: CLIP and SigLIP share tensor names (`text_model.*` etc.), so a
//! union pile would collide. f32 leaves — lossless, so pile-vs-safetensors
//! parity is exact (gated by `embed_pile_test`).
//!
//!   cargo run --release --features embed,import --bin embed_persist -- <out-dir>
//!
//! Writes `<out-dir>/{clip,siglip,nomic_text,nomic_vision}.pile`. The models
//! must already be in the HF cache (pure cache lookup, no download).

use mary::embed::hf_cache_resolve;
use mary::ingest::LeafDtype;
use mary::persist::persist_safetensors_files_to_pile;
use std::path::Path;
use std::time::Instant;

const MODELS: &[(&str, &str)] = &[
    ("openai/clip-vit-base-patch32", "clip"),
    ("google/siglip2-so400m-patch14-384", "siglip"),
    ("nomic-ai/nomic-embed-text-v1.5", "nomic_text"),
    ("nomic-ai/nomic-embed-vision-v1.5", "nomic_vision"),
];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: embed_persist <out-dir>");
        std::process::exit(2);
    }
    let out_dir = Path::new(&args[1]);
    std::fs::create_dir_all(out_dir)?;

    for (model_id, stem) in MODELS {
        let weights = hf_cache_resolve(model_id, "model.safetensors").ok_or_else(|| {
            anyhow::anyhow!(
                "model.safetensors not in HF cache for {model_id} — fetch it first (huggingface-cli download {model_id})"
            )
        })?;
        let pile_path = out_dir.join(format!("{stem}.pile"));
        let t = Instant::now();
        eprintln!("Persisting {model_id} → {pile_path:?} ...");
        persist_safetensors_files_to_pile(
            &[(weights, format!("{stem}.safetensors"))],
            &pile_path,
            LeafDtype::F32,
        )?;
        let size = std::fs::metadata(&pile_path)?.len();
        println!(
            "  {stem}: {:.1}s, {:.2} GiB",
            t.elapsed().as_secs_f64(),
            size as f64 / (1u64 << 30) as f64
        );
    }
    Ok(())
}
