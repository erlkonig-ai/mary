//! Persist Gemma 4 weights into a REAL on-disk TribleSpace pile. After this runs,
//! the pile file holds the full model as content-addressed tribles — no
//! safetensors needed to load it again (see `load_gemma4_from_persisted_pile`).
//!
//!   cargo run --release --features gemma --bin gemma_persist -- <model-dir> <pile-path>
//!
//! `<model-dir>` is a directory containing the `*.safetensors` shards (config.json
//! and tokenizer.json may live there too but aren't read by this step).

use mary::persist::persist_safetensors_to_pile;
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: gemma_persist <model-dir> <pile-path>");
        std::process::exit(2);
    }
    let model_dir = Path::new(&args[1]);
    let pile_path = Path::new(&args[2]);

    eprintln!("Persisting weights from {model_dir:?} → pile {pile_path:?} ...");
    let t = Instant::now();
    persist_safetensors_to_pile(model_dir, pile_path, mary::ingest::LeafDtype::F16)?;
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
