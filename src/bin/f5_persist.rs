//! Persist the F5-TTS transformer + Vocos vocoder into a REAL on-disk
//! TribleSpace pile — the durable voice-origin weight store `mary::say` loads
//! from. Both checkpoints go into ONE pile: their tensor-name namespaces are
//! disjoint (`ema_model.*` vs `backbone.*`/`head.*`/`feature_extractor.*`), so
//! `load_keymap_from_pile`'s union keymap serves both models. f32 leaves —
//! lossless (F5's published checkpoint is f32).
//!
//!   cargo run --release --features import --bin f5_persist -- \
//!     <f5.safetensors> <vocos.safetensors> <pile-path>

use mary::ingest::LeafDtype;
use mary::persist::persist_safetensors_files_to_pile;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: f5_persist <f5.safetensors> <vocos.safetensors> <pile-path>");
        std::process::exit(2);
    }
    let f5 = PathBuf::from(&args[1]);
    let vocos = PathBuf::from(&args[2]);
    let pile_path = Path::new(&args[3]);

    let t = Instant::now();
    eprintln!("Persisting F5 + Vocos → {pile_path:?} ...");
    persist_safetensors_files_to_pile(
        &[
            (f5, "f5tts_v1_base.safetensors".to_string()),
            (vocos, "vocos.safetensors".to_string()),
        ],
        pile_path,
        LeafDtype::F32,
    )?;
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
