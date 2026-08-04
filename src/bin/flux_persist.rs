//! Persist a FLUX.2 model directory (text_encoder + transformer + vae) into
//! ONE on-disk TribleSpace pile — the durable imagination weight store the
//! `Flux2Pipeline` loads from. Model-entity names are component-prefixed
//! (`text_encoder/…`, `transformer/…`, `vae/…`) so the pipeline materializes
//! one component's keymap per phase (`load_keymap_from_pile_prefixed`) and
//! keeps peak RAM at a single component. f16 leaves — the checkpoints are
//! bf16-native and the pipeline builds f32/f16 tensors from the leaves either
//! way; f16 halves the pile.
//!
//!   cargo run --release --features import --bin flux_persist -- \
//!     <model-dir> <pile-path>
//!
//! `<model-dir>` is the HF snapshot dir (klein or dev layout) with
//! `text_encoder/`, `transformer/`, and `vae/` subdirectories.

use mary::ingest::LeafDtype;
use mary::persist::persist_safetensors_files_to_pile;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The pipeline's three weight-bearing components (tokenizer/scheduler configs
/// stay small files next to the pile).
const COMPONENTS: &[&str] = &["text_encoder", "transformer", "vae"];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: flux_persist <model-dir> <pile-path>");
        std::process::exit(2);
    }
    let model_dir = Path::new(&args[1]);
    let pile_path = Path::new(&args[2]);

    for component in COMPONENTS {
        let dir = model_dir.join(component);
        let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        shards.sort();
        anyhow::ensure!(!shards.is_empty(), "no .safetensors shards in {dir:?}");
        let files: Vec<(PathBuf, String)> = shards
            .into_iter()
            .map(|p| {
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("model");
                let name = format!("{component}/{name}");
                (p, name)
            })
            .collect();
        let t = Instant::now();
        eprintln!(
            "Persisting {component} ({} file(s)) → {pile_path:?} ...",
            files.len()
        );
        persist_safetensors_files_to_pile(&files, pile_path, LeafDtype::F16)?;
        eprintln!("  {component}: {:.1}s", t.elapsed().as_secs_f64());
    }

    let size = std::fs::metadata(pile_path)?.len();
    println!(
        "Pile file {pile_path:?} is {} bytes ({:.2} GiB).",
        size,
        size as f64 / (1u64 << 30) as f64
    );
    Ok(())
}
