//! The version graph: a learned model as a root in the model collection,
//! with a parent, and what it takes to write one and to find the heads.
//! Backend-free, so a Mac or a Pi build reads the graph a Spark wrote.

use anyhow::{Context, Result};
use triblespace::prelude::Id;

use super::load::PackedExpert;
pub use super::resident::{Persisted, VersionRecipe};

/// One whole expert, every rank's cut joined, as the pile would store it.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedExpert {
    pub name: String,
    pub layer: i64,
    pub expert: i64,
    pub packed: PackedExpert,
}

// ── the learned VERSION: a root in the model graph, with a parent ──────────
//
// A learned model is a new ROOT in the same collection as the model it grew
// from: the parent's `member` edges, except that the experts that moved are
// new leaves, plus a `parent` edge to the root it was learned from and the
// recipe that took it there. Nothing about the parent changes, and the leaves
// that did not move are the same entities, so a version costs only the
// experts it moved. Two versions with one parent are a branch. The versions no
// other version names as `parent` are the heads, and the loader takes the
// one head as the model when nothing names a root. A root's id is its member
// set, so a version that learned nothing IS its parent.
//
// The checkpoint was imported in pieces (41 partial roots; experts that belong
// to no root), so the first version's parent is minted here once: the GENESIS
// root, whose members are every leaf the loader takes from the whole
// collection. It is intrinsic, so a second run mints the same id.
//
// JP, 2026-09-03 17:35Z: not a collection per snapshot -- a DAG in one graph,
// so a model can branch off a model, and the experiment's metadata and its
// explanation sit on the graph where a later analysis can query them.

pub mod attrs {
    use triblespace::prelude::blobencodings::UTF8String;
    use triblespace::prelude::inlineencodings::{F64, GenId, Handle, ShortString, U256BE};
    use triblespace::prelude::*;

    attributes! {
        /// The root this version was learned from. Minted 2026-09-03.
        "914320431BD23350DEE18D3D54FA84F1" as parent: GenId;
        /// The learner's step size on the routed experts' codes.
        "EA48C51D05FC180A5855106F0C0CCCAE" as learn_lr: F64;
        /// How hard her own rows were held to the distribution that said
        /// them (the anchor's weight). Absent when unanchored.
        "8BF7CAB60E71E7186955ACDAF6EF3867" as learn_anchor: F64;
        /// Where the stochastic rounding's step counter started.
        "E8C7ED409F235DA6A787B0B094269DD8" as learn_seed: U256BE;
        /// Scored passes learned from, parent to this version.
        "361889279FA26F7AD22187DF37DC7EC7" as learn_steps: U256BE;
        /// What was learned from, in words: a corpus and its lines, or a
        /// span of the archive.
        "9F65DE719752C3AB83F709B55567B247" as learned_span: Handle<UTF8String>;
        /// Why this version exists, in its author's words.
        "365C41C24AB9FF7DF2C78105DB1FDE97" as explanation: Handle<UTF8String>;
        /// The code that learned it: a git revision.
        "6CEEC82A5A1A1EC65A4609C025E42767" as code_revision: ShortString;
    }
}



/// A version assembled from learned experts, before or after it is committed.
pub struct LearnedVersion {
    /// The version root.
    pub root: Id,
    /// The root it was learned from.
    pub parent: Id,
    /// Whether `parent` is the genesis root, minted by this assembly.
    pub genesis: bool,
    /// A label: the model's name and the moment.
    pub name: String,
    /// The facts to ADD: new leaves, the version root (and the genesis root
    /// when minted), their annotations. Never the parent's facts, which the
    /// collection already holds.
    pub facts: triblespace::core::trible::TribleSet,
    /// Leaves whose bytes moved and were replaced by new leaf entities.
    pub replaced: usize,
    /// Members of the version root.
    pub members: usize,
}

/// The heads of the version DAG in `facts`: every root that carries `parent`
/// or is named as one, minus those some version names as its parent. Empty
/// when no version exists yet.
pub fn version_heads(facts: &triblespace::core::trible::TribleSet) -> Vec<Id> {
    use std::collections::BTreeSet;
    use triblespace::macros::{find, pattern};
    let mut nodes: BTreeSet<Id> = BTreeSet::new();
    let mut parents: BTreeSet<Id> = BTreeSet::new();
    for (v, p) in find!((v: Id, p: Id), pattern!(facts, [{ ?v @ attrs::parent: ?p }])) {
        nodes.insert(v);
        nodes.insert(p);
        parents.insert(p);
    }
    nodes.difference(&parents).copied().collect()
}

/// Every leaf the loader takes from the whole collection: the packed experts,
/// and the dense tensors of every element type and rank it sweeps.
fn all_leaves(facts: &triblespace::core::trible::TribleSet) -> std::collections::BTreeSet<Id> {
    use super::pile::attrs as ink;
    use triblespace::core::blob::encodings::tensor::elements::{BF16, F32};
    use triblespace::core::metadata;
    use triblespace::macros::{find, pattern};
    let mut out = std::collections::BTreeSet::new();
    for (e,) in find!(
        (e: Id),
        pattern!(facts, [{ ?e @ metadata::name: _?n, ink::expert_index: _?i, ink::weight_nvfp4_2: _?h }])
    ) {
        out.insert(e);
    }
    macro_rules! dense {
        ($ty:ty, $rank:literal) => {
            for (e,) in find!(
                (e: Id),
                pattern!(facts, [{ ?e @ metadata::name: _?n, ink::weight::<$ty, $rank>(): _?h }])
            ) {
                out.insert(e);
            }
        };
    }
    dense!(BF16, 0);
    dense!(BF16, 1);
    dense!(BF16, 2);
    dense!(BF16, 3);
    dense!(BF16, 4);
    dense!(F32, 0);
    dense!(F32, 1);
    dense!(F32, 2);
    dense!(F32, 3);
    out
}

/// The moment, as `YYYYMMDDTHHMMSSZ`, from the system clock with no crate.
pub fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, m, s) = (rem / 3_600, rem % 3_600 / 60, rem % 60);
    // Civil date from days since the epoch (Hinnant).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(mo <= 2);
    format!("{y:04}{mo:02}{d:02}T{h:02}{m:02}{s:02}Z")
}

/// Assemble a version from the learned experts, as a child of `parent` -- or
/// of the graph's one head when `parent` is `None`, or of the genesis root
/// minted here when the graph has no version yet. Writes the new leaves'
/// blobs into the pile (content addressed, so a repeat is a no-op) and
/// nothing else: no record is committed here.
pub fn learned_version(
    pile: &mut triblespace::prelude::Pile,
    learned: &[LearnedExpert],
    parent: Option<Id>,
    recipe: &VersionRecipe,
) -> Result<LearnedVersion> {
    use super::pile::attrs as ink;
    use crate::format::attrs as model;
    use std::collections::{BTreeSet, HashMap};
    use triblespace::core::inline::encodings::hash::Handle;
    use triblespace::core::metadata;
    use triblespace::core::trible::TribleSet;
    use triblespace::macros::{entity, find, pattern};
    use triblespace::prelude::*;

    let graph = crate::model_collection::mary_model_graph_name();
    let snapshot = crate::model_collection::snapshot_model_collection_named_local_latest(pile, graph)
        .with_context(|| format!("the model collection '{graph}'"))?;
    let facts = crate::model_collection::project_legacy_model_attributes(snapshot.facts()).facts;
    let (_, _, reader) = snapshot.into_parts();
    let mut added = TribleSet::new();

    // The model's name, functional per root: the one name the checkpoint's
    // roots agree on, carried onto every version.
    let names: BTreeSet<Inline<Handle<blobencodings::UTF8String>>> = find!(
        (mn: Inline<Handle<blobencodings::UTF8String>>),
        pattern!(&facts, [{ _?r @ model::model_name: ?mn }])
    )
    .map(|(mn,)| mn)
    .collect();
    let model_name = match names.len() {
        1 => names.into_iter().next(),
        _ => None,
    };
    let label_of = |what: &str| -> String {
        match &model_name {
            Some(h) => {
                let name: Result<anybytes::View<str>, _> = reader.get(*h);
                name.map(|s| format!("{} {what}", &*s))
                    .unwrap_or_else(|_| what.to_string())
            }
            None => what.to_string(),
        }
    };

    // 1. The parent, and its members.
    let heads = version_heads(&facts);
    let (parent, genesis, parent_members): (Id, bool, Vec<Id>) = match parent {
        Some(p) => (p, false, Vec::new()),
        None if heads.len() == 1 => (heads[0], false, Vec::new()),
        None if heads.len() > 1 => anyhow::bail!(
            "the model graph has {} heads ({}); name the parent",
            heads.len(),
            heads.iter().map(|i| format!("{i:X}")).collect::<Vec<_>>().join(", ")
        ),
        None => {
            let leaves = all_leaves(&facts);
            anyhow::ensure!(!leaves.is_empty(), "'{graph}' holds no tensor leaves");
            let root = entity! { _ @ model::member*: leaves.iter() };
            let id = root.root().expect("a root has a root");
            added += root;
            let name_h = pile
                .put::<blobencodings::UTF8String, _>(label_of("checkpoint"))
                .map_err(|e| anyhow::anyhow!("store the genesis label: {e:?}"))?;
            added += entity! { ExclusiveId::force_ref(&id) @ metadata::name: name_h };
            if let Some(mn) = model_name {
                added += entity! { ExclusiveId::force_ref(&id) @ model::model_name: mn };
            }
            (id, true, leaves.into_iter().collect())
        }
    };
    let parent_members: Vec<Id> = match parent_members.is_empty() {
        false => parent_members,
        true => find!((m: Id), pattern!(&facts, [{ (parent) @ model::member: ?m }]))
            .map(|(m,)| m)
            .collect(),
    };
    anyhow::ensure!(!parent_members.is_empty(), "parent {parent:X} has no members");
    let parent_set: BTreeSet<Id> = parent_members.iter().copied().collect();

    // 2. New leaves, and the parent's leaves they replace.
    let mut subst: HashMap<Id, Id> = HashMap::new();
    for x in learned {
        let blob = super::pile::expert_blob(&x.packed)
            .with_context(|| format!("{}[{}] as a pile leaf", x.name, x.expert))?;
        let handle = pile
            .put(blob)
            .map_err(|e| anyhow::anyhow!("store {}[{}]: {e:?}", x.name, x.expert))?;
        let name_h = pile
            .put::<blobencodings::UTF8String, _>(x.name.clone())
            .map_err(|e| anyhow::anyhow!("store the name of {}: {e:?}", x.name))?;
        let old: Vec<Id> = find!(
            (e: Id),
            pattern!(&facts, [{ ?e @ metadata::name: (name_h), ink::expert_index: (x.expert), ink::layer: (x.layer) }])
        )
        .map(|(e,)| e)
        .filter(|e| parent_set.contains(e))
        .collect();
        anyhow::ensure!(
            old.len() == 1,
            "{}[{}] (layer {}) names {} leaves among the parent's members, not one",
            x.name,
            x.expert,
            x.layer,
            old.len()
        );
        let leaf = entity! { _ @
            ink::weight_nvfp4_2: handle,
            ink::expert_index: x.expert,
            metadata::name: name_h,
            ink::layer: x.layer,
        };
        let new_id = leaf.root().expect("a leaf has a root");
        added += leaf;
        subst.insert(old[0], new_id);
    }

    // 3. The version root: the parent's members with the moved leaves
    //    replaced. Its id is that set, so a version that moved nothing is
    //    its parent and is not annotated as a child of itself.
    let members: Vec<Id> = parent_members
        .iter()
        .map(|m| subst.get(m).copied().unwrap_or(*m))
        .collect();
    let root_e = entity! { _ @ model::member*: members.iter() };
    let root = root_e.root().expect("a root has a root");
    let name = label_of(&format!("learned {}", utc_stamp()));
    if root != parent {
        added += root_e;
        let name_h = pile
            .put::<blobencodings::UTF8String, _>(name.clone())
            .map_err(|e| anyhow::anyhow!("store the version label: {e:?}"))?;
        let span_h = pile
            .put::<blobencodings::UTF8String, _>(recipe.span.clone())
            .map_err(|e| anyhow::anyhow!("store the learned span: {e:?}"))?;
        let why_h = pile
            .put::<blobencodings::UTF8String, _>(recipe.explanation.clone())
            .map_err(|e| anyhow::anyhow!("store the explanation: {e:?}"))?;
        added += entity! { ExclusiveId::force_ref(&root) @
            attrs::parent: parent,
            metadata::name: name_h,
            attrs::learn_lr: recipe.lr,
            attrs::learn_seed: recipe.seed,
            attrs::learn_steps: recipe.steps,
            attrs::learned_span: span_h,
            attrs::explanation: why_h,
            attrs::code_revision: recipe.code_revision.as_str(),
        };
        if let Some(w) = recipe.anchor {
            added += entity! { ExclusiveId::force_ref(&root) @ attrs::learn_anchor: w };
        }
        if let Some(mn) = model_name {
            added += entity! { ExclusiveId::force_ref(&root) @ model::model_name: mn };
        }
    }
    Ok(LearnedVersion {
        root,
        parent,
        genesis,
        name,
        facts: added,
        replaced: subst.len(),
        members: members.len(),
    })
}

/// Commit the version's facts into the model graph, signed with `key`. The
/// collection's WRITE policy decides whether that signature counts.
pub fn publish_version(
    pile: &mut triblespace::prelude::Pile,
    key: &ed25519_dalek::SigningKey,
    version: LearnedVersion,
) -> Result<()> {
    use triblespace::prelude::*;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("refresh before publishing '{}': {e:?}", version.name))?;
    let graph = crate::model_collection::mary_model_graph_name();
    let collection = crate::model_collection::collection_or_create(pile, key, graph)?;
    let fragment: Fragment = version.facts.into();
    pile.commit(collection, key, fragment)
        .map_err(|e| anyhow::anyhow!("publish '{}': {e}", version.name))?;
    Ok(())
}
