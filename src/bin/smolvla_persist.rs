//! Persist SmolVLA (lerobot/smolvla_base) into a REAL on-disk TribleSpace pile
//! — the durable action-model weight store `smolvla_infer` (and the eventual
//! embodiment loop) loads from. f32 leaves — lossless w.r.t. the bf16→f32
//! conversion the loaders do anyway, so pile-vs-safetensors parity is exact
//! (gated by `smolvla_pile_test`).
//!
//!   cargo run --release --features import --bin smolvla_persist -- \
//!     <model-dir-or-file> <pile-path>
//!
//! `<model-dir-or-file>` is the HF snapshot dir holding `model.safetensors`
//! (or the file itself).

use mary::ingest::LeafDtype;
use mary::persist::{persist_safetensors_files_to_pile, persist_safetensors_to_pile};
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: smolvla_persist <model-dir-or-file> <pile-path>");
        std::process::exit(2);
    }
    let src = Path::new(&args[1]);
    let pile_path = Path::new(&args[2]);

    let t = Instant::now();
    eprintln!("Persisting SmolVLA from {src:?} → {pile_path:?} ...");
    if src.is_file() {
        persist_safetensors_files_to_pile(
            &[(src.to_path_buf(), "smolvla_base.safetensors".to_string())],
            pile_path,
            LeafDtype::F32,
        )?;
    } else {
        persist_safetensors_to_pile(src, pile_path, LeafDtype::F32)?;
    }
    let secs = t.elapsed().as_secs_f64();

    let size = std::fs::metadata(pile_path)?.len();
    println!(
        "Persisted in {:.1}s. Pile file {pile_path:?} is {} bytes ({:.2} GiB).",
        secs,
        size,
        size as f64 / (1u64 << 30) as f64
    );
    Ok(())
}
