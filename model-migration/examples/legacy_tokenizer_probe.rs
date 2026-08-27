//! READ-ONLY: does a pile's LEGACY branch DAG still carry a tokenizer graph
//! that its native collection does not?
//!
//! The bundle migration adopted one exact weight commit. A pile is append-only,
//! so whatever the legacy branch held is still in the file even when no
//! collection reaches it. This asks the difference directly: walk `main`'s
//! ancestor closure, and report the tokenizer nodes and SentencePiece pieces
//! found there.
//!
//! Opens the pile, closes it, and writes nothing.

use anyhow::{anyhow, Context, Result};
use std::collections::BTreeSet;

use triblespace::core::repo::{content, parent, BlobStoreGet};
use triblespace::macros::{find, pattern};
use triblespace::prelude::*;

type Commit = Inline<inlineencodings::Handle<blobencodings::SimpleArchive>>;

fn parents_and_content(
    reader: &impl BlobStoreGet,
    commit: Commit,
) -> Result<(Vec<Commit>, Option<Commit>)> {
    let meta: TribleSet = reader
        .get(commit)
        .map_err(|e| anyhow!("read commit: {e:?}"))?;
    let mut contents = find!((c: Inline<_>), pattern!(&meta, [{ content: ?c }]));
    let content_handle = contents.next().map(|(handle,)| handle);
    let parents: Vec<Commit> = find!((p: Inline<_>), pattern!(&meta, [{ parent: ?p }]))
        .map(|(p,)| p)
        .collect();
    Ok((parents, content_handle))
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: legacy_tokenizer_probe <pile>")?;
    let path = std::path::PathBuf::from(path);
    println!("pile {}", path.display());

    let mut pile = Pile::open(&path).map_err(|e| anyhow!("open: {e:?}"))?;
    let result = probe(&mut pile);
    let close = pile.close().map_err(|e| anyhow!("close: {e:?}"));
    result?;
    close?;
    Ok(())
}

fn probe(pile: &mut Pile) -> Result<()> {
    let frozen =
        mary_model_migration::freeze_legacy_model_main(pile).context("freeze legacy main")?;
    let head = frozen.head;
    let reader = frozen.reader;
    println!("legacy main head {head:?}");

    let mut all = TribleSet::new();
    let mut seen = BTreeSet::new();
    let mut pending = vec![head];
    while let Some(c) = pending.pop() {
        if !seen.insert(c) {
            continue;
        }
        let (parents, content_handle) = parents_and_content(&reader, c)?;
        if let Some(h) = content_handle {
            let contribution: TribleSet =
                reader.get(h).map_err(|e| anyhow!("read content: {e:?}"))?;
            all += contribution;
        }
        pending.extend(parents);
    }
    println!("legacy closure facts (raw): {}", all.len());
    let all = mary::model_collection::project_legacy_model_attributes(&all).facts;
    println!("legacy closure facts (projected): {}", all.len());

    let toks: Vec<_> = mary::tokenizer::find_tokenizers(&all).collect();
    println!("tokenizer nodes in legacy closure: {}", toks.len());
    for t in &toks {
        let pieces = mary::tokenizer::load_spm_pieces(&all, &reader, *t);
        let adp = mary::tokenizer::has_add_prefix_space(&all, *t);
        println!(
            "  node {t}  spm pieces {}  add_prefix_space {adp}",
            pieces.len()
        );
        if !pieces.is_empty() {
            let spm = mary::models::personaplex::spm::SpmTokenizer::from_pieces(&pieces, adp);
            println!("  -> SpmTokenizer vocab_size {}", spm.vocab_size());
            let probe = "Hello, this is a round trip.";
            let ids = spm.encode(probe);
            let back: String = ids
                .iter()
                .map(|&i| String::from_utf8_lossy(spm.piece_bytes(i)).to_string())
                .collect::<String>()
                .replace('\u{2581}', " ");
            println!("  -> encode({probe:?}) = {} ids", ids.len());
            println!("  -> decode = {:?}", back.trim_start());
        }
    }
    Ok(())
}
