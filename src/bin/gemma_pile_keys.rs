//! List the model entities + tensor keys inside a mary weight pile (read-only).
//! Component census for feasibility scouting: answers "which components does this pile
//! actually carry?" without loading any weights. Documented negative it was
//! built for (2026-07-11): gemma_31b.pile has ZERO model.audio_tower.* keys —
//! google/gemma-4-31B-it ships without the audio path (it lives in the E-lineage).
//!
//!   cargo run --release --features gemma --bin gemma_pile_keys -- <pile> [filter]

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pile = args.get(1).expect("usage: gemma_pile_keys <pile> [filter]");
    let filter = args.get(2).map(|s| s.as_str()).unwrap_or("");

    let (f16, f32_, _reader) =
        mary::persist::load_split_index_from_pile(Path::new(pile), "").expect("open pile index");
    let mut keys: Vec<&String> = f16.keys().chain(f32_.keys()).collect();
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
