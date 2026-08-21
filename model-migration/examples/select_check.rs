//! READ-ONLY: does a migrated pile answer the exact selection a runtime
//! consumer makes?  `select_check <pile> <source> <quantization> [tokenizer]`
//!
//! Reproduces the two-step every native consumer performs — discover the sole
//! model-graph team, load its locally-admitted latest snapshot — and then the
//! `ModelSelector::Source` lookup plus the strict tensor-handle index. This is
//! the collection/selection boundary only; it does not construct a model.

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let pile = args.next().context("usage: select_check <pile> <source> <quantization> [tokenizer-name]")?;
    let source = args.next().context("missing <source>")?;
    let quantization = args.next().context("missing <quantization>")?;
    let tokenizer = args.next();
    let pile = std::path::PathBuf::from(pile);

    let team = mary::model_collection::model_graph_team_at(&pile)
        .context("read the sole model-graph team")?;
    let snapshot = mary::model_collection::load_model_collection_local_latest(&pile, team)
        .context("load locally admitted native collection")?;
    println!("pile {}", pile.display());
    println!("team {}", hex(&team.to_bytes()));
    println!("{} commit(s) in ticket, {} facts", snapshot.commits().len(), snapshot.facts().len());

    let root = mary::selection::select_model_root(
        snapshot.facts(),
        snapshot.reader(),
        mary::selection::ModelSelector::Source {
            source: &source,
            quantization: &quantization,
        },
    )
    .with_context(|| format!("select model root for ({source:?}, {quantization:?})"))?;
    println!("selected root {root}");

    let keymap =
        mary::selection::index_keymap_for_root(snapshot.facts(), snapshot.reader(), root)
            .context("index the selected root's tensor handles")?;
    println!("{} tensor handles", keymap.len());
    let mut names: Vec<&String> = keymap.keys().collect();
    names.sort();
    for n in names.iter().take(3) {
        println!("  {n}");
    }

    if let Some(name) = tokenizer {
        let t = mary::selection::select_tokenizer_root(
            snapshot.facts(),
            snapshot.reader(),
            mary::selection::TokenizerSelector::Name(&name),
        )
        .with_context(|| format!("select tokenizer named {name:?}"))?;
        println!("tokenizer root {t}");
    }

    println!("OK");
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
