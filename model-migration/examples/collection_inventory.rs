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
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::core::collection::records::{collection_name, collection_namespace};
use triblespace::core::collection::CollectionRecord;
use triblespace::core::inline::encodings::ed25519::ED25519PublicKey;
use triblespace::core::inline::encodings::shortstring::ShortString;
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
        let mut name = None;
        let mut team = None;
        for fact in facts.iter() {
            if *fact.a() == collection_name.id() {
                name = fact.v::<ShortString>().try_from_inline::<String>().ok();
            } else if *fact.a() == collection_namespace.id() {
                team = fact
                    .v::<ED25519PublicKey>()
                    .raw
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
                    .into();
            }
        }
        let name = name.unwrap_or_else(|| "<unnamed>".to_string());
        let team = team.unwrap_or_else(|| "<no team>".to_string());
        println!(
            "  {name:<24} team {}…  commits {commits}",
            &team[..16.min(team.len())]
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
