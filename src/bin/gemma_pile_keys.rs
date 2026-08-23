//! List the model entities + tensor keys inside a mary weight pile (read-only).
//! Component census for feasibility scouting: answers "which components does this pile
//! actually carry?" without loading any weights. Documented negative it was
//! built for (2026-07-11): gemma_31b.pile has ZERO model.audio_tower.* keys —
//! google/gemma-4-31B-it ships without the audio path (it lives in the E-lineage).
//!
//!   cargo run --release --features gemma --bin gemma_pile_keys -- <pile> <source> [filter]

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pile = args
        .get(1)
        .expect("usage: gemma_pile_keys <pile> <source> [filter]");
    let source = args
        .get(2)
        .expect("usage: gemma_pile_keys <pile> <source> [filter]");
    let filter = args.get(3).map(|s| s.as_str()).unwrap_or("");

    let (_, snapshot) =
        mary::model_collection::load_sole_model_collection_local_latest(Path::new(pile))
            .expect("load native model snapshot");
    let selected = mary::selection::SelectedModelIndex::from_snapshot(
        snapshot,
        mary::selection::ModelSelector::Source {
            source,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .expect("select the native model component");
    let mut keys: Vec<&String> = selected.handles().keys().collect();
    keys.sort();
    println!("total tensor keys: {}", keys.len());

    let matched: Vec<&&String> = keys.iter().filter(|k| k.contains(filter)).collect();
    println!("matching {filter:?}: {}", matched.len());
    for k in matched.iter().take(60) {
        println!("  {k}");
    }
    if matched.len() > 60 {
        println!("  ... ({} more)", matched.len() - 60);
    }

    // Prefix histogram (top-level component census).
    use std::collections::BTreeMap;
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for k in &keys {
        let prefix: String = k.split('.').take(2).collect::<Vec<_>>().join(".");
        *hist.entry(prefix).or_default() += 1;
    }
    println!("\nprefix census:");
    for (p, n) in hist {
        println!("  {n:5}  {p}");
    }
}
