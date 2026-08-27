//! READ-ONLY: is the loose `.model` file the SAME tokenizer the pile's legacy
//! branch already carries?
//!
//! The tokenizer content is present in `personaplex.pile`'s pre-collection
//! branch DAG but reachable by no collection. Republishing it needs a source,
//! and there are two candidates: the legacy graph itself, and the `.model` file
//! on disk. They are only interchangeable if they are equal, and "the vocab
//! sizes match" is not equality — a tokenizer with the right count and the
//! wrong scores encodes prompts into garbage.
//!
//! Compares piece bytes, ids, scores, type tags, and the add-prefix-space flag,
//! one by one. Opens the pile, closes it, and writes nothing.

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
    let mut args = std::env::args().skip(1);
    let pile_path = args
        .next()
        .context("usage: spm_source_parity <pile> <model-file>")?;
    let model_path = args
        .next()
        .context("usage: spm_source_parity <pile> <model-file>")?;
    let pile_path = std::path::PathBuf::from(pile_path);
    let model_path = std::path::PathBuf::from(model_path);

    let mut pile = Pile::open(&pile_path).map_err(|e| anyhow!("open: {e:?}"))?;
    let from_pile = legacy_pieces(&mut pile);
    let close = pile.close().map_err(|e| anyhow!("close: {e:?}"));
    let (pieces_pile, adp_pile) = from_pile?;
    close?;

    let (pieces_file, adp_file, byte_fallback_file) =
        mary::models::personaplex::spm::SpmTokenizer::parse_model(&model_path);

    println!(
        "legacy graph : {} pieces, add_dummy_prefix {adp_pile}",
        pieces_pile.len()
    );
    println!(
        "{:<13}: {} pieces, add_dummy_prefix {adp_file}, byte_fallback {byte_fallback_file}",
        model_path.file_name().unwrap_or_default().to_string_lossy(),
        pieces_file.len()
    );

    anyhow::ensure!(
        pieces_pile.len() == pieces_file.len(),
        "piece COUNT differs: pile {} vs file {}",
        pieces_pile.len(),
        pieces_file.len()
    );

    let mut mismatches = 0usize;
    let mut first: Option<String> = None;
    for (i, (a, b)) in pieces_pile.iter().zip(pieces_file.iter()).enumerate() {
        // bytes, score, type — all three, because any one of them being wrong
        // silently changes what a prompt encodes to.
        let same = a.0 == b.0 && a.1.to_bits() == b.1.to_bits() && a.2 == b.2;
        if !same {
            mismatches += 1;
            if first.is_none() {
                first = Some(format!(
                    "id {i}: pile ({:?}, {}, {}) vs file ({:?}, {}, {})",
                    String::from_utf8_lossy(&a.0),
                    a.1,
                    a.2,
                    String::from_utf8_lossy(&b.0),
                    b.1,
                    b.2
                ));
            }
        }
    }
    println!("mismatching pieces: {mismatches}");
    if let Some(detail) = first {
        println!("first mismatch -> {detail}");
    }
    println!("add_prefix_space equal: {}", adp_pile == adp_file);
    anyhow::ensure!(mismatches == 0, "tokenizer sources are NOT equal");
    anyhow::ensure!(adp_pile == adp_file, "add_prefix_space differs");
    println!("PARITY: the file and the pile's legacy graph are the same tokenizer");
    Ok(())
}

type Pieces = Vec<(Vec<u8>, f32, u64)>;

fn legacy_pieces(pile: &mut Pile) -> Result<(Pieces, bool)> {
    let frozen =
        mary_model_migration::freeze_legacy_model_main(pile).context("freeze legacy main")?;
    let reader = frozen.reader;
    let mut all = TribleSet::new();
    let mut seen = BTreeSet::new();
    let mut pending = vec![frozen.head];
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
    let all = mary::model_collection::project_legacy_model_attributes(&all).facts;
    let toks: Vec<_> = mary::tokenizer::find_tokenizers(&all).collect();
    anyhow::ensure!(
        toks.len() == 1,
        "expected exactly one legacy tokenizer, found {}",
        toks.len()
    );
    let tok = toks[0];
    Ok((
        mary::tokenizer::load_spm_pieces(&all, &reader, tok),
        mary::tokenizer::has_add_prefix_space(&all, tok),
    ))
}
