//! List what a pile actually holds: its branches, and its signed model
//! collection.
//!
//! Written because a converter guessed at a pile's branch layout twice and was
//! wrong twice. The pile is the authority on its own shape; ask it.
//!
//! BOTH readers, because a model pile now has two and they do not agree. The
//! deprecated branch reader is what `pile_leaf_migrate` and `qwen3tts_say`
//! use; the COLLECTION is what `mary::speak` — the production voice — selects
//! from. A pile can be perfectly readable through one and absent through the
//! other, and reading only the branch pins reports a pile as fine when the
//! seam that matters cannot open it at all.
//!
//! The leaf census answers the question those two readers exist to settle:
//! is this pile on the typed `Tensor<T, RANK>` encoding, or still on the
//! two-blob `{data|data_f16, shape}` form? The typed attribute ids are
//! DERIVED from (anchor, encoding), so they are asked of `mary::leaf` rather
//! than restated here — a table copied into a tool is a table that drifts.

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use triblespace::core::repo::{ancestors, Repository};
use triblespace::macros::{find, pattern};
use triblespace::prelude::*;

/// One fact set's attribute histogram, labelled from mary's own schema.
///
/// Shared by the branch walk and the collection view deliberately: the whole
/// point of printing both readers is that their fact sets can DIFFER, and a
/// difference is only visible when the same census is applied to each. It also
/// makes the epoch legible — a pre-epoch fact set shows bare `Id(...)` where a
/// post-epoch one shows `data`/`shape`/`weight`, because the historical
/// literal ids have no current declaration to be named by.
fn print_attribute_census(
    set: &TribleSet,
    leaf_names: &std::collections::HashMap<Id, String>,
) {
    let mut counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for t in set.iter() {
        let a = t.a();
        let label = match a {
            x if *x == mary::format::attrs::data.id() => "data".to_string(),
            x if *x == mary::format::attrs::data_f16.id() => "data_f16".to_string(),
            x if *x == mary::format::attrs::shape.id() => "shape".to_string(),
            x if *x == mary::format::attrs::data_q4.id() => "data_q4".to_string(),
            x if *x == mary::format::attrs::data_q8.id() => "data_q8".to_string(),
            x if *x == mary::format::attrs::q_scales.id() => "q_scales".to_string(),
            x if *x == mary::format::attrs::weight.id() => "weight".to_string(),
            x if *x == mary::format::attrs::bias.id() => "bias".to_string(),
            x if *x == mary::format::attrs::kind.id() => "kind".to_string(),
            x if *x == mary::format::attrs::safetensor_path.id() => "safetensor_path".to_string(),
            x if *x == mary::format::attrs::member.id() => "member".to_string(),
            x if *x == mary::format::attrs::model_name.id() => "model_name".to_string(),
            x if *x == mary::format::attrs::source.id() => "source".to_string(),
            x if *x == mary::format::attrs::quantization.id() => "quantization".to_string(),
            x if *x == mary::format::attrs::index.id() => "index".to_string(),
            other => leaf_names
                .get(other)
                .cloned()
                .unwrap_or_else(|| format!("{other}")),
        };
        *counts.entry(label).or_default() += 1;
    }
    for (k, v) in counts {
        println!("      {v:6}  {k}");
    }
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: pile_branches <pile>")?;
    let path = std::path::Path::new(&path);
    let mut pile = Pile::open(path).map_err(|e| anyhow::anyhow!("open {path:?}: {e:?}"))?;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("load {path:?}: {e:?}"))?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;

    // What ids does the COMPILED schema actually resolve to? The declared hex
    // in `attributes!` is not necessarily the attribute id — the anchored form
    // derives it from (anchor, value encoding) — so print both and let the
    // comparison be visible rather than assumed.
    println!("compiled schema ids:");
    println!("  data            {}", mary::format::attrs::data.id());
    println!("  data_f16        {}", mary::format::attrs::data_f16.id());
    println!("  shape           {}", mary::format::attrs::shape.id());
    println!("  data_q4         {}", mary::format::attrs::data_q4.id());
    println!("  data_q8         {}", mary::format::attrs::data_q8.id());
    println!("  q_scales        {}", mary::format::attrs::q_scales.id());
    for (name, id) in mary::leaf::typed_leaf_attrs() {
        println!("  {name:15} {id}");
    }
    println!("  weight          {}", mary::format::attrs::weight.id());
    println!("  kind            {}", mary::format::attrs::kind.id());
    println!("  safetensor_path {}", mary::format::attrs::safetensor_path.id());
    println!("  member          {}", mary::format::attrs::member.id());
    println!("  model_name      {}", mary::format::attrs::model_name.id());
    println!("  source          {}", mary::format::attrs::source.id());
    println!("  quantization    {}", mary::format::attrs::quantization.id());
    println!();

    let leaf_names: std::collections::HashMap<Id, String> = mary::leaf::typed_leaf_attrs()
        .into_iter()
        .map(|(n, id)| (id, n))
        .collect();

    let pins: Vec<Id> = repo
        .storage_mut()
        .pins()
        .map_err(|e| anyhow::anyhow!("pins: {e:?}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("pins: {e:?}"))?;
    println!("{} pin(s)", pins.len());

    for branch_id in pins {
        let head = match repo
            .storage_mut()
            .head(branch_id)
            .map_err(|e| anyhow::anyhow!("head: {e:?}"))?
        {
            Some(h) => h,
            None => {
                println!("  {branch_id}  <no head>");
                continue;
            }
        };
        let reader = repo
            .storage_mut()
            .reader()
            .map_err(|e| anyhow::anyhow!("reader: {e:?}"))?;
        let meta: TribleSet = reader
            .get(head)
            .map_err(|e| anyhow::anyhow!("meta: {e:?}"))?;
        let name = triblespace::core::repo::branch::branch_entity(&meta, branch_id)
            .ok()
            .and_then(|be| {
                find!(
                    n: Inline<inlineencodings::Handle<blobencodings::LongString>>,
                    pattern!(&meta, [{ be @ triblespace::core::metadata::name: ?n }])
                )
                .next()
            })
            .and_then(|n| reader.get::<anybytes::View<str>, _>(n).ok())
            .map(|v| v.to_string())
            .unwrap_or_else(|| "<unnamed>".into());

        let mut ws = repo
            .pull(branch_id)
            .map_err(|e| anyhow::anyhow!("pull: {e:?}"))?;
        let facts = match ws.head() {
            Some(h) => ws
                .checkout(ancestors(h))
                .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
                .facts()
                .len(),
            None => 0,
        };
        println!("  {branch_id}  {name:20}  {facts} facts");

        // Attribute histogram: what this branch actually holds, by attribute
        // id. Names come from mary's own schema so the output is readable
        // without a lookup table.
        if let Some(h) = ws.head() {
            let set = ws
                .checkout(ancestors(h))
                .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
                .facts()
                .clone();
            print_attribute_census(&set, &leaf_names);
        }
    }

    repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;

    // ── the collection, and the leaf census through it ──────────────────────
    // Separate open: the collection reader takes the pile by value, and the
    // branch walk above has just finished with it.
    let mut pile = Pile::open(path).map_err(|e| anyhow::anyhow!("reopen {path:?}: {e:?}"))?;
    match mary::model_collection::sole_model_graph_team(&mut pile) {
        Err(e) => println!("\nmodel collection: none ({e:?}) — `mary::speak` cannot open this pile"),
        Ok(team) => {
            match mary::model_collection::snapshot_model_collection_local_latest(&mut pile, team) {
                Err(e) => println!("\nmodel collection: snapshot failed: {e}"),
                Ok(snapshot) => {
                    let facts = snapshot.facts();
                    let typed = mary::leaf::index_typed_all(facts, snapshot.reader()).len();
                    let two_blob = find!(
                        (e: Id),
                        pattern!(facts, [{ ?e @ mary::format::attrs::data: _?d }])
                    )
                    .count()
                        + find!(
                            (e: Id),
                            pattern!(facts, [{ ?e @ mary::format::attrs::data_f16: _?d }])
                        )
                        .count();
                    println!("\nmodel collection: {} facts", facts.len());
                    println!("  typed leaves      {typed}");
                    println!("  two-blob leaves   {two_blob}");
                    print_attribute_census(facts, &leaf_names);
                }
            }
        }
    }
    let _ = pile.close();
    Ok(())
}
