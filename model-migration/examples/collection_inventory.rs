//! READ-ONLY: name every collection a pile actually publishes, and report
//! whether a SentencePiece tokenizer graph is reachable through each.
//!
//! The loader asks a pile for one collection by NAME. When that name is absent
//! the failure says only "no collection named X", which does not distinguish
//! "this pile carries the content under a different name" from "the content is
//! not here at all". Those are different problems with different fixes, so the
//! inventory has to be taken rather than inferred from the error.
//!
//! Opens the pile, closes it, and writes nothing.

use anyhow::{anyhow, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::blob::encodings::utf8string::UTF8String;
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::core::collection::descriptor;
use triblespace::core::collection::CollectionRecord;
use triblespace::core::trible::TribleSet;
use triblespace::prelude::*;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: collection_inventory <pile>")?;
    let path = std::path::PathBuf::from(path);
    println!("pile {}", path.display());

    let mut pile = Pile::open(&path).map_err(|e| anyhow!("open {path:?}: {e:?}"))?;
    let result = inventory(&mut pile);
    let close = pile.close().map_err(|e| anyhow!("close {path:?}: {e:?}"));
    result?;
    close?;

    // The tokenizer question, asked through mary's own public loader so the
    // answer is about the shipping code path and not a re-implementation.
    match mary::persist::load_spm_tokenizer_from_pile(&path) {
        Ok(spm) => println!(
            "spm    : loads, vocab_size {} (TEXT_CARD {})",
            spm.vocab_size(),
            mary::models::personaplex::config::TEXT_CARD
        ),
        Err(error) => println!("spm    : FAILS -- {error:#}"),
    }
    Ok(())
}

fn inventory(pile: &mut Pile) -> Result<()> {
    let mut per_descriptor: BTreeMap<[u8; 32], usize> = BTreeMap::new();
    let mut order = Vec::new();
    for record in pile.records().map_err(|e| anyhow!("scan records: {e}"))? {
        if let CollectionRecord::Commit(commit) = record.map_err(|e| anyhow!("record: {e}"))? {
            let key = commit.collection().raw;
            if per_descriptor.insert(key, 0).is_none() {
                order.push(commit.collection());
            }
            *per_descriptor.get_mut(&key).expect("just inserted") += 1;
        }
    }
    let reader = pile.reader().map_err(|e| anyhow!("reader: {e}"))?;
    println!("collections ({} distinct descriptors):", order.len());
    let mut named = BTreeSet::new();
    for handle in order {
        let commits = per_descriptor[&handle.raw];
        let Ok(blob) = reader.get::<Blob<SimpleArchive>, _>(handle.transmute()) else {
            println!("  <descriptor blob absent from this pile>  commits {commits}");
            continue;
        };
        let Ok(facts) = <TribleSet as TryFromBlob<SimpleArchive>>::try_from_blob(blob) else {
            println!("  <descriptor blob undecodable>            commits {commits}");
            continue;
        };
        let name = descriptor::name(&facts)
            .ok()
            .flatten()
            .and_then(|handle| reader.get::<Blob<UTF8String>, _>(handle).ok())
            .and_then(|blob| std::str::from_utf8(&blob.bytes).ok().map(str::to_owned))
            .unwrap_or_else(|| "<unnamed>".to_string());
        let authority = descriptor::authority(&facts)
            .map(|key| {
                key.as_bytes()
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            })
            .unwrap_or_else(|_| "<no authority>".to_string());
        println!(
            "  {name:<24} authority {}…  commits {commits}",
            &authority[..16.min(authority.len())]
        );
        named.insert(name);
    }
    for want in ["mary-model-graph", "mary-model-bundles"] {
        println!(
            "  -> {want:<20} {}",
            if named.contains(want) {
                "PRESENT"
            } else {
                "ABSENT"
            }
        );
    }
    Ok(())
}
