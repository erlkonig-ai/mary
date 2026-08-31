//! READ-ONLY: decode a pile's `mary-model-bundles` archive H and report what
//! the bundle's complete fact set actually contains — tensors, tokenizer
//! nodes, and (if a SentencePiece tokenizer is there) its piece count.
//!
//! This is the question the loader failure cannot answer: `mary-model-graph`
//! being absent says nothing about whether the tokenizer CONTENT is present
//! under the bundle shape. If it is, the fix is in the loader and no pile is
//! written; if it is not, the pile needs an append.
//!
//! Opens the pile, closes it, and writes nothing.

use anyhow::{anyhow, Context, Result};

use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::core::metadata;
use triblespace::core::trible::TribleSet;
use triblespace::prelude::*;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: bundle_tokenizer_probe <pile>")?;
    let path = std::path::PathBuf::from(path);
    println!("pile {}", path.display());

    let mut pile = Pile::open(&path).map_err(|e| anyhow!("open {path:?}: {e:?}"))?;
    let observed = mary::model_collection::snapshot_model_bundle_collection_local_latest(&mut pile);
    let snapshot = match observed {
        Ok(snapshot) => snapshot,
        Err(error) => {
            let _ = pile.close();
            return Err(anyhow!("freeze sole bundle snapshot: {error}"));
        }
    };
    let (_, cover, reader) = snapshot.into_parts();
    println!("cover members {}", cover.len());

    for member in cover.members() {
        let token_blob: Blob<SimpleArchive> = reader
            .get(member)
            .map_err(|e| anyhow!("read bundle token: {e}"))?;
        let token = TribleSet::try_from_blob(token_blob).context("decode bundle token")?;
        println!("  token rows {}", token.len());
        for fact in token.iter() {
            if fact.a() != &metadata::archive.id() {
                println!("    (row is not metadata::archive)");
                continue;
            }
            let root = *fact.e();
            let h = inlineencodings::Handle::<SimpleArchive>::to_hash(
                *fact.v::<inlineencodings::Handle<SimpleArchive>>(),
            );
            let archive: Blob<SimpleArchive> = reader
                .get(inlineencodings::Handle::<SimpleArchive>::from_hash(h))
                .map_err(|e| anyhow!("read archive H: {e}"))?;
            let facts = TribleSet::try_from_blob(archive).context("decode archive H")?;
            println!("    root {root}  H facts (raw) {}", facts.len());
            let facts = mary::model_collection::project_legacy_model_attributes(&facts).facts;
            println!("    root {root}  H facts (projected) {}", facts.len());

            // What tokenizer nodes does H hold, and of which kind?
            let toks: Vec<_> = mary::tokenizer::find_tokenizers(&facts).collect();
            println!("    tokenizer nodes in H: {}", toks.len());
            for t in &toks {
                let tags = mary::tokenizer::node_tags(&facts, *t);
                let pieces = mary::tokenizer::load_spm_pieces(&facts, &reader, *t);
                println!(
                    "      node {t}  tags {}  spm pieces {}",
                    tags.len(),
                    pieces.len()
                );
            }
            // And what does mary's own selector make of it?
            match mary::selection::select_tokenizer_root(
                &facts,
                &reader,
                mary::selection::TokenizerSelector::Only,
            ) {
                Ok(id) => println!("    select_tokenizer_root -> {id}"),
                Err(e) => println!("    select_tokenizer_root FAILS -- {e:#}"),
            }
        }
    }
    drop(reader);
    pile.close().map_err(|e| anyhow!("close: {e:?}"))?;
    Ok(())
}
