//! Strictly additive migration from Mary's legacy Repository model piles.
//!
//! The legacy `main` branch remains authoritative, byte-for-byte evidence. A
//! migration captures one exact branch head, checks out only that head's
//! ancestry, projects the audited pre-epoch attribute aliases, adds the
//! requested canonical selection coordinates at the already-existing model
//! root, and publishes the resulting union as one native model-collection
//! commit. It never pushes, creates, deletes, or otherwise advances a branch.
//!
//! Storage policy stays with the caller: this crate takes an already-open
//! [`Pile`], never reopens it, and neither flushes nor closes it. The caller
//! supplies the signing key and chooses the explicit durability boundary.
//!
//! # Why this is a crate and not a `mary` module
//!
//! The migration is a bridge across the `Repository` -> `Collection` cutover,
//! so it is the one place that must still speak the legacy `Repository` API.
//! `mary` itself is past that cutover. Living outside the library — a separate
//! package with its own lockfile and its own `[patch.crates-io]` table, not a
//! workspace member sharing mary's — is what lets this crate later pin an
//! older `triblespace` (and an older `mary`) once `Repository`/`PinStore` are
//! dropped from the main line, so legacy piles stay migratable without the
//! library carrying a shape that exists only for them.

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, Context};
use ed25519_dalek::{SigningKey, VerifyingKey};
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::core::collection::{CollectionCommit, CollectionData};
use triblespace::core::repo::pile::Pile;
use triblespace::core::repo::{ancestors, content, parent, BlobStore, CommitHandle, Repository};
use triblespace::prelude::blobencodings::LongString;
use triblespace::prelude::inlineencodings::{Handle, ShortString};
use triblespace::prelude::*;

use mary::format::attrs;
use mary::model_collection::{
    model_bundle_team_or_own, model_graph_team_or_own, prepare_model_bundle_fragment,
    project_legacy_model_attributes, publish_model_fragment,
    snapshot_model_bundle_collection_local_latest,
};
use mary::models::personaplex::{PersonaPlexWeights, SOURCE as PERSONAPLEX_SOURCE};
use mary::selection::{select_model_root, select_tokenizer_root, ModelSelector, TokenizerSelector};

const PERSONAPLEX_LM_FILE: &str = "model.safetensors";
const PERSONAPLEX_MIMI_FILE: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
const PERSONAPLEX_LM_MEMBERS: usize = 475;
const PERSONAPLEX_MIMI_MEMBERS: usize = 318;
const PERSONAPLEX_MEMBERS: usize = 793;

/// Data needed to turn one legacy model graph into a selectable native graph.
///
/// `model` resolves to exactly one entity carrying at least one `member` edge,
/// through the same exact-cardinality selectors the native runtime already
/// uses. [`ModelSelector::Name`] matches the canonical (possibly projected)
/// name and still fails closed when one pile holds two roots under one name;
/// [`ModelSelector::Root`] names the content address itself, which is how such
/// a pile states an unambiguous choice instead of being unmigratable. Neither
/// direction has a first-match fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegacyModelMigration<'a> {
    /// Which existing legacy weight root this migration publishes.
    pub model: ModelSelector<'a>,
    /// Canonical model-source coordinate to add or verify.
    pub source: &'a str,
    /// Canonical weight-format coordinate to add or verify.
    pub quantization: &'a str,
    /// If present, require exactly one tokenizer carrying this name.
    ///
    /// The audited legacy-attribute projection supplies the canonical
    /// `model_name` alias for pre-epoch tokenizers. This slice deliberately
    /// does not guess an unnamed tokenizer root.
    pub tokenizer_name: Option<&'a str>,
}

/// Exact evidence returned by one successful migration publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegacyModelMigrationResult {
    /// Complete signed native collection ticket, suitable for exact reads.
    pub commit: CollectionCommit,
    /// Team whose native model-graph collection received the commit.
    pub team: VerifyingKey,
    /// Legacy branch id whose head was frozen.
    pub legacy_branch: Id,
    /// Exact legacy commit head used for the checkout.
    pub legacy_head: CommitHandle,
    /// Existing model entity selected without recomputing its identity.
    pub model_root: Id,
    /// Existing tokenizer entity selected when `tokenizer_name` was supplied.
    pub tokenizer_root: Option<Id>,
    /// Fact count in the frozen legacy checkout.
    pub legacy_facts: usize,
    /// Missing canonical pre-epoch aliases added by the audited projection.
    pub aliases_added: usize,
    /// Missing explicit source/quantization coordinates added at the model root.
    pub selector_facts_added: usize,
}

/// Result of adopting the exact legacy PersonaPlex weight commit as one bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersonaPlexLegacyAdoptionResult {
    /// Existing bundle COMMIT on a no-op retry, or the newly published COMMIT.
    pub commit: CollectionCommit,
    /// Whether this invocation finalized a new COMMIT.
    pub published: bool,
    /// Team whose `mary-model-bundles` collection contains the token.
    pub team: VerifyingKey,
    /// Exact legacy Repository commit selected by the caller.
    pub legacy_commit: CommitHandle,
    /// Existing legacy LM model root selected by its exact file name.
    pub legacy_lm_root: Id,
    /// Existing legacy Mimi model root selected by its exact file name.
    pub legacy_mimi_root: Id,
    /// New intrinsic root derived only from the union of member edges.
    pub model_root: Id,
    /// Canonical model-fact archive `H` named by the one-row bundle token.
    pub model_archive_data: CollectionData,
    /// Fact count in the exact legacy checkout before alias projection.
    pub legacy_facts: usize,
    /// Missing canonical pre-epoch aliases added by the audited projection.
    pub aliases_added: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PersonaPlexMemberPolicy {
    legacy_facts: usize,
    lm: usize,
    mimi: usize,
    total: usize,
}

const PERSONAPLEX_MEMBER_POLICY: PersonaPlexMemberPolicy = PersonaPlexMemberPolicy {
    legacy_facts: 4_700,
    lm: PERSONAPLEX_LM_MEMBERS,
    mimi: PERSONAPLEX_MIMI_MEMBERS,
    total: PERSONAPLEX_MEMBERS,
};

fn freeze_legacy_main(
    pile: &mut Pile,
    signing_key: &SigningKey,
) -> anyhow::Result<(Id, CommitHandle, TribleSet, <Pile as BlobStore>::Reader)> {
    pile.refresh()
        .context("refresh legacy model pile before freezing 'main'")?;

    // Mary's inspected legacy writers constructed Repository with empty
    // commit metadata, so their target piles already contain this exact
    // content-addressed archive and the put deduplicates. Repository itself
    // permits other metadata; this migration surface is deliberately scoped
    // to Mary's legacy model piles. In every case Repository::new neither
    // creates nor advances a branch.
    let mut repo = Repository::new(&mut *pile, signing_key.clone(), TribleSet::new())
        .map_err(|error| anyhow!("open borrowed legacy repository view: {error}"))?;
    let branch = repo
        .lookup_branch("main")
        .map_err(|error| anyhow!("lookup legacy 'main' branch: {error:?}"))?
        .ok_or_else(|| anyhow!("legacy model pile has no 'main' branch"))?;
    let mut workspace = repo
        .pull(branch)
        .map_err(|error| anyhow!("pull legacy 'main' branch: {error:?}"))?;
    let head = workspace
        .head()
        .ok_or_else(|| anyhow!("legacy 'main' branch has no commits"))?;

    // The captured head, not a moving branch name, is the checkout authority.
    // A concurrent later branch advance therefore cannot widen this snapshot.
    let facts = workspace
        .checkout(ancestors(head))
        .map_err(|error| anyhow!("checkout frozen legacy 'main' head: {error}"))?
        .into_facts();
    let reader = repo
        .storage_mut()
        .reader()
        .context("open blob reader for frozen legacy model snapshot")?;
    Ok((branch, head, facts, reader))
}

/// Check out exactly `commit` and its ancestors from one immutable blob view.
///
/// This is the detached equivalent of `Workspace::checkout(ancestors(commit))`:
/// every reachable commit contributes its optional content archive, while a
/// merge-only commit contributes no facts. It deliberately does not construct
/// a `Repository` or `Workspace`, because both are branch-oriented surfaces and
/// pulling one would make an otherwise resident exact commit depend on a live,
/// well-formed branch pin.
fn checkout_legacy_commit_ancestors(
    reader: &impl BlobStoreGet,
    commit: CommitHandle,
) -> anyhow::Result<TribleSet> {
    let mut visited = BTreeSet::new();
    let mut pending = vec![commit];
    let mut facts = TribleSet::new();

    while let Some(commit) = pending.pop() {
        if !visited.insert(commit) {
            continue;
        }

        let metadata: TribleSet = reader
            .get(commit)
            .map_err(|error| anyhow!("read exact legacy commit {commit:?}: {error}"))?;
        let mut content_handles = find!(
            (content_handle: Inline<_>),
            pattern!(&metadata, [{ content: ?content_handle }])
        );
        let content_handle = content_handles.next().map(|(handle,)| handle);
        anyhow::ensure!(
            content_handles.next().is_none(),
            "exact legacy commit {commit:?} has ambiguous content"
        );
        if let Some(content_handle) = content_handle {
            let contribution: TribleSet = reader.get(content_handle).map_err(|error| {
                anyhow!("read content of exact legacy commit {commit:?}: {error}")
            })?;
            facts += contribution;
        }

        pending.extend(
            find!(
                (parent_handle: Inline<_>),
                pattern!(&metadata, [{ parent: ?parent_handle }])
            )
            .map(|(parent_handle,)| parent_handle),
        );
    }

    Ok(facts)
}

/// Check out exactly `commit` and its ancestors from the legacy pile.
///
/// The caller-supplied content handle is the complete checkout authority. No
/// branch id, branch metadata, or current head participates in the read.
fn freeze_legacy_commit(
    pile: &mut Pile,
    commit: CommitHandle,
) -> anyhow::Result<(TribleSet, <Pile as BlobStore>::Reader)> {
    pile.refresh()
        .context("refresh legacy model pile before exact commit checkout")?;

    let reader = pile
        .reader()
        .context("open blob reader for exact legacy model snapshot")?;
    let facts = checkout_legacy_commit_ancestors(&reader, commit)
        .with_context(|| format!("checkout exact legacy commit {commit:?}"))?;
    Ok((facts, reader))
}

fn members_for_root(facts: &TribleSet, root: Id) -> BTreeSet<Id> {
    find!(
        (member: Id),
        pattern!(facts, [{ root @ attrs::member: ?member }])
    )
    .map(|(member,)| member)
    .collect()
}

struct PreparedPersonaPlexCandidate {
    fragment: Fragment,
    legacy_lm_root: Id,
    legacy_mimi_root: Id,
    model_root: Id,
    legacy_facts: usize,
    aliases_added: usize,
}

/// Construct the unpublished PersonaPlex bundle graph without reading or
/// copying tensor payloads.
fn prepare_legacy_personaplex_candidate(
    legacy: TribleSet,
    reader: &impl BlobStoreGet,
    policy: PersonaPlexMemberPolicy,
) -> anyhow::Result<PreparedPersonaPlexCandidate> {
    let legacy_facts = legacy.len();
    anyhow::ensure!(
        legacy_facts == policy.legacy_facts,
        "legacy PersonaPlex checkout has {legacy_facts} facts, expected exactly {}",
        policy.legacy_facts
    );
    let projection = project_legacy_model_attributes(&legacy);
    let aliases_added = projection.aliases_added;

    let legacy_lm_root = select_model_root(
        &projection.facts,
        reader,
        ModelSelector::Name(PERSONAPLEX_LM_FILE),
    )
    .context("select exactly one legacy PersonaPlex LM root")?;
    let legacy_mimi_root = select_model_root(
        &projection.facts,
        reader,
        ModelSelector::Name(PERSONAPLEX_MIMI_FILE),
    )
    .context("select exactly one legacy PersonaPlex Mimi root")?;
    anyhow::ensure!(
        legacy_lm_root != legacy_mimi_root,
        "PersonaPlex LM and Mimi names resolve to the same legacy root {legacy_lm_root}"
    );

    let lm_members = members_for_root(&projection.facts, legacy_lm_root);
    let mimi_members = members_for_root(&projection.facts, legacy_mimi_root);
    anyhow::ensure!(
        lm_members.len() == policy.lm,
        "legacy PersonaPlex LM root {legacy_lm_root} has {} members, expected {}",
        lm_members.len(),
        policy.lm
    );
    anyhow::ensure!(
        mimi_members.len() == policy.mimi,
        "legacy PersonaPlex Mimi root {legacy_mimi_root} has {} members, expected {}",
        mimi_members.len(),
        policy.mimi
    );
    if let Some(overlap) = lm_members.intersection(&mimi_members).next() {
        bail!("legacy PersonaPlex LM and Mimi roots share member {overlap}");
    }
    let members: BTreeSet<_> = lm_members.union(&mimi_members).copied().collect();
    anyhow::ensure!(
        members.len() == policy.total,
        "legacy PersonaPlex member union has {} members, expected {}",
        members.len(),
        policy.total
    );

    // Re-wrap raw projected facts with an intentionally empty attachment
    // store. Every historical tensor/name/shape handle remains a reference to
    // bytes already resident in the pile; only the new source label below
    // enters this Fragment's MemoryBlobStore.
    let mut fragment = Fragment::from_facts_and_blobs(
        projection.facts,
        triblespace::core::blob::MemoryBlobStore::new(),
    );
    let root = entity! { _ @ attrs::member*: members.iter() };
    let model_root = root.root().expect("non-empty member set yields one root");
    fragment += root;

    let source = fragment.put::<LongString, _>(PERSONAPLEX_SOURCE.to_owned());
    fragment += entity! { ExclusiveId::force_ref(&model_root) @
        attrs::source: source,
        attrs::quantization: mary::persist::QUANTIZATION_NATIVE,
    };
    anyhow::ensure!(
        fragment.root() == Some(model_root),
        "PersonaPlex candidate lost its unique intrinsic root export"
    );

    Ok(PreparedPersonaPlexCandidate {
        fragment,
        legacy_lm_root,
        legacy_mimi_root,
        model_root,
        legacy_facts,
        aliases_added,
    })
}

/// Adopt one exact legacy PersonaPlex Repository commit as a signed model bundle.
///
/// The commit handle supplied by the caller is the only legacy history
/// selector. The current `main` head cannot widen the checkout. Existing facts,
/// entity ids, and attachment handles are preserved byte-for-byte; the only new
/// graph identity is one root derived from the union of the two legacy member
/// sets. No `mary-model-graph` COMMIT is published.
///
/// The caller owns the already-open pile and its eventual close/flush boundary.
pub fn adopt_legacy_personaplex_bundle(
    pile: &mut Pile,
    signing_key: &SigningKey,
    legacy_commit: CommitHandle,
) -> anyhow::Result<PersonaPlexLegacyAdoptionResult> {
    adopt_legacy_personaplex_bundle_with_policy(
        pile,
        signing_key,
        legacy_commit,
        PERSONAPLEX_MEMBER_POLICY,
    )
}

fn adopt_legacy_personaplex_bundle_with_policy(
    pile: &mut Pile,
    signing_key: &SigningKey,
    legacy_commit: CommitHandle,
    policy: PersonaPlexMemberPolicy,
) -> anyhow::Result<PersonaPlexLegacyAdoptionResult> {
    let (legacy, reader) = freeze_legacy_commit(pile, legacy_commit)?;
    let candidate = prepare_legacy_personaplex_candidate(legacy, &reader, policy)?;
    drop(reader);

    let team = model_bundle_team_or_own(pile, signing_key)
        .context("select the existing PersonaPlex bundle team or found it under this signer")?;
    let prepared = prepare_model_bundle_fragment(team, candidate.model_root, candidate.fragment)
        .context("prepare canonical PersonaPlex bundle token")?;
    let model_archive_data = prepared.model_archive_data();

    // Freeze exactly the current same-team bundle ticket before staging any
    // dependency. A matching `(root, H)` makes the operation a strict no-op;
    // a different PersonaPlex authority fails before a COMMIT can be exposed.
    let existing = snapshot_model_bundle_collection_local_latest(pile, team)
        .context("freeze existing same-team model bundles")?;
    if let Some(existing) = PersonaPlexWeights::find_in_bundle_snapshot(team, existing)
        .context("inspect existing same-team PersonaPlex bundle")?
    {
        anyhow::ensure!(
            existing.authority().model_root() == candidate.model_root
                && existing.authority().model_archive_data() == model_archive_data,
            "a different PersonaPlex bundle is already authoritative in this team"
        );
        let token_data = existing.authority().bundle_token_data();
        let commit = existing
            .authority()
            .ticket()
            .iter()
            .copied()
            .find(|commit| commit.data() == token_data)
            .expect("validated bundle authority names one of its ticket commits");
        return Ok(PersonaPlexLegacyAdoptionResult {
            commit,
            published: false,
            team,
            legacy_commit,
            legacy_lm_root: candidate.legacy_lm_root,
            legacy_mimi_root: candidate.legacy_mimi_root,
            model_root: candidate.model_root,
            model_archive_data,
            legacy_facts: candidate.legacy_facts,
            aliases_added: candidate.aliases_added,
        });
    }

    let mut staged = prepared
        .into_prepared_commit()
        .stage(pile, signing_key)
        .map_err(|error| anyhow!("stage PersonaPlex bundle dependencies: {error}"))?;

    // The source LongString is one of the just-staged dependencies, while the
    // legacy tensor blobs were already resident. Validate that exact combined
    // view through the same zero-copy resolver used by runtime loading before
    // the signed COMMIT becomes visible. Dropping `staged` on failure leaves
    // only inert content-addressed dependencies.
    let staged_reader = staged
        .store_mut()
        .reader()
        .context("open reader over staged PersonaPlex dependencies")?;
    let archive: Blob<blobencodings::SimpleArchive> = staged_reader
        .get(inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(model_archive_data))
        .map_err(|error| anyhow!("read staged PersonaPlex model archive H: {error}"))?;
    let staged_facts =
        TribleSet::try_from_blob(archive).context("decode staged PersonaPlex model archive H")?;
    let weights = PersonaPlexWeights::from_graph(&staged_facts, staged_reader)
        .context("validate staged but unpublished legacy PersonaPlex candidate")?;
    anyhow::ensure!(
        weights.root() == candidate.model_root,
        "validated PersonaPlex root {} differs from constructed root {}",
        weights.root(),
        candidate.model_root
    );
    anyhow::ensure!(
        weights.count() == policy.total,
        "validated PersonaPlex candidate has {} unique tensors, expected {}",
        weights.count(),
        policy.total
    );
    drop(weights);

    let commit = staged
        .finalize()
        .map_err(|error| anyhow!("finalize validated PersonaPlex bundle: {error}"))?;

    Ok(PersonaPlexLegacyAdoptionResult {
        commit,
        published: true,
        team,
        legacy_commit,
        legacy_lm_root: candidate.legacy_lm_root,
        legacy_mimi_root: candidate.legacy_mimi_root,
        model_root: candidate.model_root,
        model_archive_data,
        legacy_facts: candidate.legacy_facts,
        aliases_added: candidate.aliases_added,
    })
}

fn source_coordinate_is_missing(
    fragment: &Fragment,
    reader: &impl BlobStoreGet,
    root: Id,
    wanted: &str,
) -> anyhow::Result<bool> {
    let handles: Vec<Inline<Handle<LongString>>> = fragment
        .facts()
        .iter()
        .filter(|fact| fact.e() == &root && fact.a() == &attrs::source.id())
        .map(|fact| *fact.v::<Handle<LongString>>())
        .collect();

    match handles.as_slice() {
        [] => Ok(true),
        [handle] => {
            let actual: anybytes::View<str> = reader
                .get(*handle)
                .map_err(|error| anyhow!("read source on legacy model root {root}: {error}"))?;
            if &*actual != wanted {
                bail!(
                    "legacy model root {root} already has conflicting source {:?}",
                    &*actual
                );
            }
            Ok(false)
        }
        _ => bail!("legacy model root {root} has ambiguous source coordinates"),
    }
}

fn quantization_coordinate_is_missing(
    fragment: &Fragment,
    root: Id,
    wanted: &str,
) -> anyhow::Result<bool> {
    let values: Vec<Inline<ShortString>> = fragment
        .facts()
        .iter()
        .filter(|fact| fact.e() == &root && fact.a() == &attrs::quantization.id())
        .map(|fact| *fact.v::<ShortString>())
        .collect();

    match values.as_slice() {
        [] => {
            let _: Inline<ShortString> = wanted.try_to_inline().map_err(|error| {
                anyhow!("quantization {wanted:?} is not a valid ShortString coordinate: {error:?}")
            })?;
            Ok(true)
        }
        [value] => {
            let value = ShortString::validate(*value).map_err(|error| {
                anyhow!("invalid quantization bytes on legacy model root {root}: {error:?}")
            })?;
            let actual: &str = value.try_from_inline().map_err(|error| {
                anyhow!("invalid quantization UTF-8 on legacy model root {root}: {error}")
            })?;
            if actual != wanted {
                bail!("legacy model root {root} already has conflicting quantization {actual:?}");
            }
            Ok(false)
        }
        _ => bail!("legacy model root {root} has ambiguous quantization coordinates"),
    }
}

/// Publish one frozen legacy `main` snapshot into Mary's native collection.
///
/// The returned [`CollectionCommit`] is the complete signed ticket, not merely
/// its intrinsic id. Existing facts are copied as raw tribles, which preserves
/// entity ids, value bytes, unknown attributes, and resident attachment
/// handles. Only canonical aliases plus the requested selector coordinates are
/// added. The caller retains `pile` on both success and failure and must choose
/// its own explicit `flush`/`close` policy.
pub fn migrate_legacy_model_main(
    pile: &mut Pile,
    signing_key: &SigningKey,
    request: LegacyModelMigration<'_>,
) -> anyhow::Result<LegacyModelMigrationResult> {
    let (legacy_branch, legacy_head, legacy, reader) = freeze_legacy_main(pile, signing_key)?;
    let legacy_facts = legacy.len();
    let projection = project_legacy_model_attributes(&legacy);
    let aliases_added = projection.aliases_added;

    let model_root =
        select_model_root(&projection.facts, &reader, request.model).with_context(|| {
            format!(
                "select exactly one legacy weight root matching {:?}",
                request.model
            )
        })?;

    let tokenizer_root = request
        .tokenizer_name
        .map(|name| {
            select_tokenizer_root(&projection.facts, &reader, TokenizerSelector::Name(name))
                .with_context(|| format!("select exactly one legacy tokenizer named {name:?}"))
        })
        .transpose()?;

    // Build at explicit, already-selected ids. No intrinsic entity id is ever
    // recomputed from the enriched coordinate set.
    let mut fragment = Fragment::from_facts_and_blobs(
        projection.facts,
        triblespace::core::blob::MemoryBlobStore::new(),
    );
    let add_source = source_coordinate_is_missing(&fragment, &reader, model_root, request.source)?;
    let add_quantization =
        quantization_coordinate_is_missing(&fragment, model_root, request.quantization)?;
    let selector_facts_added = usize::from(add_source) + usize::from(add_quantization);

    // Match Mary's ordinary model-root construction idiom: new coordinates
    // are described `entity!` facts at one forced existing id. The explicit
    // subject prevents intrinsic-id recomputation while the returned Fragment
    // carries the attributes' metafacts into the native commit metadata.
    if selector_facts_added != 0 {
        let source = add_source.then(|| fragment.put::<LongString, _>(request.source.to_owned()));
        let quantization = add_quantization.then_some(request.quantization);
        fragment += entity! { ExclusiveId::force_ref(&model_root) @
            attrs::source?: source,
            attrs::quantization?: quantization,
        };
    }

    // Join the existing model-graph team when there is one; otherwise this
    // durable signer founds it. The result records that selected team, so
    // exact readers never need to rediscover or restate it.
    let team = model_graph_team_or_own(pile, signing_key)
        .context("select the existing model-graph team or found it under this signer")?;
    let commit = publish_model_fragment(pile, team, signing_key, fragment)
        .context("publish migrated model graph to Mary's native collection")?;

    Ok(LegacyModelMigrationResult {
        commit,
        team,
        legacy_branch,
        legacy_head,
        model_root,
        tokenizer_root,
        legacy_facts,
        aliases_added,
        selector_facts_added,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::OpenOptions;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use anybytes::{Bytes, View};
    use triblespace::core::blob::MemoryBlobStore;
    use triblespace::core::inline::encodings::UnknownInline;
    use triblespace::core::metadata;
    use triblespace::core::repo::pile::PileReader;
    use triblespace::core::repo::pile::WantRewritePolicy;
    use triblespace::core::repo::{PinStore, PushResult, RetentionRoots};
    use triblespace::prelude::blobencodings::RawBytes;

    use super::*;
    use mary::format::{F32Array, U64Array};
    use mary::model_collection::{
        local_model_bundle_ticket, publish_model_bundle_fragment,
        snapshot_model_bundle_collection_exact, snapshot_model_collection_exact,
    };

    static NEXT_TEMP_PILE: AtomicU64 = AtomicU64::new(0);
    const LEGACY_MODEL_NAME: &str = "legacy-weights.safetensors";
    const CANONICAL_SOURCE: &str = "example/model-v1";
    const CANONICAL_TOKENIZER: &str = "example/model-v1";
    const QUANTIZATION: &str = "native";

    struct TempPilePath(PathBuf);

    impl TempPilePath {
        fn new(label: &str) -> Self {
            let ordinal = NEXT_TEMP_PILE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mary-model-migration-{label}-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create disposable pile");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPilePath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    struct LegacyFixture {
        pile: TempPilePath,
        branch: Id,
        head: CommitHandle,
        model_root: Id,
        /// The second root sharing `model_root`'s name, when the fixture was
        /// asked for the ambiguous shape.
        duplicate_root: Option<Id>,
        tokenizer_root: Id,
        attachment: Inline<Handle<LongString>>,
        unknown_fact: Trible,
        facts: TribleSet,
    }

    struct LegacyPersonaPlexFixture {
        pile: TempPilePath,
        branch: Id,
        weight_commit: CommitHandle,
        later_commit: CommitHandle,
        lm_root: Id,
        mimi_root: Id,
        facts: TribleSet,
        later_fact: Trible,
        leaf_ids: Vec<Id>,
        data: Vec<Inline<Handle<F32Array>>>,
        shapes: Vec<Inline<Handle<U64Array>>>,
        orphan: Inline<Handle<RawBytes>>,
    }

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn test_id(byte: u8) -> Id {
        Id::new([byte; 16]).expect("nonzero test id")
    }

    fn historical_attribute(label: &str) -> Id {
        mary::model_collection::legacy_model_attribute_aliases()
            .into_iter()
            .find(|alias| alias.label == label)
            .unwrap_or_else(|| panic!("missing legacy alias {label}"))
            .historical
    }

    fn replace_canonical_attributes_with_historical(facts: &TribleSet) -> TribleSet {
        let reverse: HashMap<Id, Id> = mary::model_collection::legacy_model_attribute_aliases()
            .into_iter()
            .map(|alias| (alias.canonical, alias.historical))
            .collect();
        facts
            .iter()
            .map(|fact| match reverse.get(fact.a()) {
                Some(historical) => Trible::force(fact.e(), historical, fact.v::<UnknownInline>()),
                None => *fact,
            })
            .collect()
    }

    fn legacy_fixture(label: &str, duplicate_model_name: bool) -> LegacyFixture {
        let pile = TempPilePath::new(label);
        let mut blobs = MemoryBlobStore::new();
        let tokenizer = mary::tokenizer::save_tokenizer_json(
            br###"{
                "model": {
                    "type": "WordPiece",
                    "vocab": {"[UNK]": 0, "hello": 1},
                    "unk_token": "[UNK]",
                    "continuing_subword_prefix": "##",
                    "max_input_chars_per_word": 100
                },
                "added_tokens": []
            }"###,
            CANONICAL_TOKENIZER,
            &mut blobs,
        )
        .expect("build legacy tokenizer fixture");
        let tokenizer_root = tokenizer.root().expect("tokenizer root");
        let mut facts = replace_canonical_attributes_with_historical(&tokenizer.into_facts());

        let model_root = test_id(0x11);
        let member = test_id(0x12);
        let name = blobs
            .put::<LongString, _>(LEGACY_MODEL_NAME.to_owned())
            .expect("put legacy model name");
        let attachment = blobs
            .put::<LongString, _>("resident legacy attachment".to_owned())
            .expect("put resident attachment");
        let member_value = inlineencodings::GenId::inline_from(member);
        let tokenizer_value = inlineencodings::GenId::inline_from(tokenizer_root);
        facts.insert(&Trible::force(
            &model_root,
            &historical_attribute("format.model_name"),
            &name,
        ));
        facts.insert(&Trible::force(
            &model_root,
            &historical_attribute("format.member"),
            &member_value,
        ));
        facts.insert(&Trible::force(
            &model_root,
            &historical_attribute("tokenizer.tokenizer"),
            &tokenizer_value,
        ));

        // This attribute is deliberately outside the audited mapping. Its
        // exact row and LongString handle must survive the native migration.
        let unknown_fact = Trible::force(&member, &test_id(0x7f), &attachment);
        facts.insert(&unknown_fact);

        let mut duplicate_root = None;
        if duplicate_model_name {
            let other_root = test_id(0x21);
            duplicate_root = Some(other_root);
            let other_member = inlineencodings::GenId::inline_from(test_id(0x22));
            facts.insert(&Trible::force(
                &other_root,
                &historical_attribute("format.model_name"),
                &name,
            ));
            facts.insert(&Trible::force(
                &other_root,
                &historical_attribute("format.member"),
                &other_member,
            ));
        }

        let fragment = Fragment::from_facts_and_blobs(facts.clone(), blobs);
        let pile_store = Pile::open(pile.path()).expect("open disposable pile");
        let mut repo = Repository::new(pile_store, key(3), TribleSet::new())
            .expect("create legacy repository");
        let branch = *repo
            .create_branch("main", None)
            .expect("create legacy main branch");
        let mut workspace = repo.pull(branch).expect("pull legacy main");
        workspace.commit(fragment, "legacy model graph");
        let head = workspace.head().expect("legacy commit head");
        repo.push(&mut workspace).expect("push legacy main");
        repo.close().expect("close legacy fixture pile");

        LegacyFixture {
            pile,
            branch,
            head,
            model_root,
            duplicate_root,
            tokenizer_root,
            attachment,
            unknown_fact,
            facts,
        }
    }

    fn add_legacy_tensor(
        blobs: &mut MemoryBlobStore,
        facts: &mut TribleSet,
        root: Id,
        member: Id,
        leaf: Id,
        tensor_name: &str,
        value: f32,
    ) -> (Inline<Handle<F32Array>>, Inline<Handle<U64Array>>) {
        let data = blobs
            .put::<F32Array, _>(vec![value])
            .expect("put legacy f32 payload");
        let shape = blobs
            .put::<U64Array, _>(vec![1_u64])
            .expect("put legacy tensor shape");
        let path = blobs
            .put::<LongString, _>(tensor_name.to_owned())
            .expect("put legacy tensor path");

        facts.insert(&Trible::force(
            &leaf,
            &historical_attribute("format.data"),
            &data,
        ));
        facts.insert(&Trible::force(
            &leaf,
            &historical_attribute("format.shape"),
            &shape,
        ));
        facts.insert(&Trible::force(
            &member,
            &historical_attribute("format.safetensor_path"),
            &path,
        ));
        facts.insert(&Trible::force(
            &member,
            &historical_attribute("format.weight"),
            &inlineencodings::GenId::inline_from(leaf),
        ));
        facts.insert(&Trible::force(
            &root,
            &historical_attribute("format.member"),
            &inlineencodings::GenId::inline_from(member),
        ));
        (data, shape)
    }

    fn legacy_personaplex_fixture(
        label: &str,
        overlap: bool,
        duplicate_tensor_name: bool,
    ) -> LegacyPersonaPlexFixture {
        let pile = TempPilePath::new(label);
        let lm_root = test_id(0x31);
        let mimi_root = test_id(0x32);
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let lm_name = blobs
            .put::<LongString, _>(PERSONAPLEX_LM_FILE.to_owned())
            .expect("put legacy LM name");
        let mimi_name = blobs
            .put::<LongString, _>(PERSONAPLEX_MIMI_FILE.to_owned())
            .expect("put legacy Mimi name");
        facts.insert(&Trible::force(
            &lm_root,
            &historical_attribute("format.model_name"),
            &lm_name,
        ));
        facts.insert(&Trible::force(
            &mimi_root,
            &historical_attribute("format.model_name"),
            &mimi_name,
        ));

        let lm_member_a = test_id(0x41);
        let lm_member_b = test_id(0x42);
        let lm_leaf_a = test_id(0x51);
        let lm_leaf_b = test_id(0x52);
        let mut leaf_ids = vec![lm_leaf_a, lm_leaf_b];
        let mut data = Vec::new();
        let mut shapes = Vec::new();
        let (payload, shape) = add_legacy_tensor(
            &mut blobs,
            &mut facts,
            lm_root,
            lm_member_a,
            lm_leaf_a,
            "lm.layer.0.weight",
            1.0,
        );
        data.push(payload);
        shapes.push(shape);
        let (payload, shape) = add_legacy_tensor(
            &mut blobs,
            &mut facts,
            lm_root,
            lm_member_b,
            lm_leaf_b,
            if duplicate_tensor_name {
                "lm.layer.0.weight"
            } else {
                "lm.layer.1.weight"
            },
            2.0,
        );
        data.push(payload);
        shapes.push(shape);

        if overlap {
            facts.insert(&Trible::force(
                &mimi_root,
                &historical_attribute("format.member"),
                &inlineencodings::GenId::inline_from(lm_member_a),
            ));
        } else {
            let mimi_member = test_id(0x43);
            let mimi_leaf = test_id(0x53);
            leaf_ids.push(mimi_leaf);
            let (payload, shape) = add_legacy_tensor(
                &mut blobs,
                &mut facts,
                mimi_root,
                mimi_member,
                mimi_leaf,
                "mimi.encoder.weight",
                3.0,
            );
            data.push(payload);
            shapes.push(shape);
        }

        let fragment = Fragment::from_facts_and_blobs(facts.clone(), blobs);
        let pile_store = Pile::open(pile.path()).expect("open PersonaPlex fixture pile");
        let mut repo = Repository::new(pile_store, key(0x33), TribleSet::new())
            .expect("create legacy PersonaPlex repository");
        let branch = *repo
            .create_branch("main", None)
            .expect("create legacy PersonaPlex main branch");
        let mut workspace = repo.pull(branch).expect("pull PersonaPlex main");
        workspace.commit(fragment, "legacy PersonaPlex weights");
        let weight_commit = workspace.head().expect("legacy weight commit");
        repo.push(&mut workspace)
            .expect("push legacy PersonaPlex weights");

        let mut later_blobs = MemoryBlobStore::new();
        let orphan = later_blobs
            .put::<RawBytes, _>(b"later unrelated orphan".to_vec())
            .expect("put later unrelated attachment");
        let later_fact = Trible::force(&test_id(0x61), &test_id(0x62), &orphan);
        let later_facts = std::iter::once(later_fact).collect();
        workspace.commit(
            Fragment::from_facts_and_blobs(later_facts, later_blobs),
            "later unrelated graph",
        );
        let later_commit = workspace.head().expect("later unrelated commit");
        repo.push(&mut workspace)
            .expect("push later unrelated commit");
        repo.close().expect("close PersonaPlex fixture pile");

        LegacyPersonaPlexFixture {
            pile,
            branch,
            weight_commit,
            later_commit,
            lm_root,
            mimi_root,
            facts,
            later_fact,
            leaf_ids,
            data,
            shapes,
            orphan,
        }
    }

    fn personaplex_policy(fixture: &LegacyPersonaPlexFixture) -> PersonaPlexMemberPolicy {
        PersonaPlexMemberPolicy {
            legacy_facts: fixture.facts.len(),
            lm: 2,
            mimi: 1,
            total: 3,
        }
    }

    fn conflicting_personaplex_fragment() -> Fragment {
        let mut fragment = Fragment::empty();
        let shape = fragment.put::<U64Array, _>(vec![1_u64]);
        let data = fragment.put::<F32Array, _>(vec![99.0]);
        let leaf = entity! { _ @ attrs::data: data, attrs::shape: shape };
        let leaf_id = leaf.root().expect("conflicting tensor leaf");
        fragment += leaf;
        let name = fragment.put::<LongString, _>("other.weight".to_owned());
        let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: leaf_id };
        let member_id = member.root().expect("conflicting tensor member");
        fragment += member;
        let root = entity! { _ @ attrs::member: member_id };
        let root_id = root.root().expect("conflicting PersonaPlex root");
        fragment += root;
        let source = fragment.put::<LongString, _>(PERSONAPLEX_SOURCE.to_owned());
        fragment += entity! { ExclusiveId::force_ref(&root_id) @
            attrs::source: source,
            attrs::quantization: mary::persist::QUANTIZATION_NATIVE,
        };
        let (_, facts, metafacts, blobs) = fragment.into_parts();
        Fragment::rooted_from_parts(root_id, facts, metafacts, blobs)
    }

    fn read_main_identity(path: &Path) -> (Id, CommitHandle) {
        let pile = Pile::open(path).expect("open pile to inspect main");
        let mut repo =
            Repository::new(pile, key(4), TribleSet::new()).expect("open repository view");
        let branch = repo
            .lookup_branch("main")
            .expect("lookup main")
            .expect("main exists");
        let head = repo
            .pull(branch)
            .expect("pull main")
            .head()
            .expect("main head");
        repo.close().expect("close inspected pile");
        (branch, head)
    }

    fn exact_snapshot(
        path: &Path,
        team: VerifyingKey,
        commit: CollectionCommit,
    ) -> triblespace::core::collection::CollectionSnapshot<PileReader> {
        let mut pile = Pile::open(path).expect("open pile for exact native read");
        // `team` is the exact value returned by publication; do not replace
        // that authority with a second ambient discovery pass.
        let snapshot = snapshot_model_collection_exact(&mut pile, team, &[commit])
            .expect("materialize exact migration ticket");
        pile.close().expect("close exact-read pile");
        snapshot
    }

    fn exact_bundle_snapshot(
        path: &Path,
        team: VerifyingKey,
        commit: CollectionCommit,
    ) -> triblespace::core::collection::CollectionSnapshot<PileReader> {
        let mut pile = Pile::open(path).expect("open pile for exact bundle read");
        let snapshot = snapshot_model_bundle_collection_exact(&mut pile, team, &[commit])
            .expect("materialize exact model-bundle ticket");
        pile.close().expect("close exact bundle-read pile");
        snapshot
    }

    #[test]
    fn migration_is_additive_exact_selectable_resident_and_idempotent() {
        let fixture = legacy_fixture("complete", false);
        let before = std::fs::read(fixture.pile.path()).expect("read legacy prefix");
        let migration_key = key(9);
        let request = LegacyModelMigration {
            model: ModelSelector::Name(LEGACY_MODEL_NAME),
            source: CANONICAL_SOURCE,
            quantization: QUANTIZATION,
            tokenizer_name: Some(CANONICAL_TOKENIZER),
        };

        let mut pile = Pile::open(fixture.pile.path()).expect("open migration pile");
        let first = migrate_legacy_model_main(&mut pile, &migration_key, request)
            .expect("migrate legacy graph");
        pile.close().expect("durably close migrated pile");

        assert_eq!(first.legacy_branch, fixture.branch);
        assert_eq!(first.legacy_head, fixture.head);
        assert_eq!(first.model_root, fixture.model_root);
        assert_eq!(first.tokenizer_root, Some(fixture.tokenizer_root));
        assert_eq!(first.legacy_facts, fixture.facts.len());
        assert_eq!(first.selector_facts_added, 2);

        let after_first = std::fs::read(fixture.pile.path()).expect("read migrated pile");
        assert!(
            after_first.starts_with(&before),
            "legacy byte prefix changed"
        );
        assert_eq!(
            read_main_identity(fixture.pile.path()),
            (fixture.branch, fixture.head),
            "legacy branch identity/head changed"
        );

        let snapshot = exact_snapshot(fixture.pile.path(), first.team, first.commit);
        let projection = project_legacy_model_attributes(&fixture.facts);
        assert_eq!(first.aliases_added, projection.aliases_added);
        let mut expected = projection.facts;
        let mut expected_blobs = MemoryBlobStore::new();
        let source = expected_blobs
            .put::<LongString, _>(CANONICAL_SOURCE.to_owned())
            .expect("derive expected source handle");
        expected += entity! { ExclusiveId::force_ref(&fixture.model_root) @
            attrs::source: source,
            attrs::quantization: QUANTIZATION,
        }
        .into_facts();
        assert_eq!(
            snapshot.facts(),
            &expected,
            "native facts are not exactly legacy union projected aliases union selector coordinates"
        );
        assert!(snapshot.facts().contains(&fixture.unknown_fact));

        assert_eq!(
            select_model_root(
                snapshot.facts(),
                snapshot.reader(),
                ModelSelector::Source {
                    source: CANONICAL_SOURCE,
                    quantization: QUANTIZATION,
                },
            )
            .expect("select migrated model from exact ticket"),
            fixture.model_root
        );
        assert_eq!(
            select_tokenizer_root(
                snapshot.facts(),
                snapshot.reader(),
                TokenizerSelector::Name(CANONICAL_TOKENIZER),
            )
            .expect("select migrated tokenizer from exact ticket"),
            fixture.tokenizer_root
        );
        let attachment: View<str> = snapshot
            .reader()
            .get(fixture.attachment)
            .expect("legacy attachment remains resident through exact snapshot");
        assert_eq!(&*attachment, "resident legacy attachment");

        let mut pile = Pile::open(fixture.pile.path()).expect("reopen for idempotence run");
        let second =
            migrate_legacy_model_main(&mut pile, &migration_key, request).expect("rerun migration");
        pile.close().expect("close idempotence run");
        assert_eq!(second.commit, first.commit, "rerun changed exact ticket");
        assert_eq!(
            std::fs::read(fixture.pile.path()).expect("read rerun pile"),
            after_first,
            "rerun appended bytes despite identical content-addressed publication"
        );
    }

    /// A legacy pile can hold two weight roots under one `model_name`, which is
    /// exactly what the name selector must refuse. Naming the content address
    /// is how such a pile still migrates. The root selected here is the SECOND
    /// of the two, so a first-match implementation cannot pass by accident.
    #[test]
    fn root_selection_migrates_one_root_of_an_ambiguous_name() {
        let fixture = legacy_fixture("by-root", true);
        let wanted = fixture.duplicate_root.expect("ambiguous fixture");
        assert_ne!(wanted, fixture.model_root);

        let mut pile = Pile::open(fixture.pile.path()).expect("open ambiguous fixture");
        let result = migrate_legacy_model_main(
            &mut pile,
            &key(9),
            LegacyModelMigration {
                model: ModelSelector::Root(wanted),
                source: CANONICAL_SOURCE,
                quantization: QUANTIZATION,
                tokenizer_name: None,
            },
        )
        .expect("the content address resolves what the shared name cannot");
        pile.close().expect("close migrated pile");
        assert_eq!(result.model_root, wanted);
        assert_eq!(result.selector_facts_added, 2);

        // The coordinates must land on the requested root and nowhere else, and
        // the shared name must still fail closed afterwards.
        let snapshot = exact_snapshot(fixture.pile.path(), result.team, result.commit);
        assert_eq!(
            select_model_root(
                snapshot.facts(),
                snapshot.reader(),
                ModelSelector::Source {
                    source: CANONICAL_SOURCE,
                    quantization: QUANTIZATION,
                },
            )
            .expect("the migrated root is selectable by its new coordinates"),
            wanted
        );
        let error = select_model_root(
            snapshot.facts(),
            snapshot.reader(),
            ModelSelector::Name(LEGACY_MODEL_NAME),
        )
        .expect_err("the shared name is still ambiguous after migration")
        .to_string();
        assert!(error.contains("ambiguous"), "{error}");
    }

    #[test]
    fn ambiguous_legacy_weight_roots_fail_before_publication() {
        let fixture = legacy_fixture("ambiguous", true);
        let before = std::fs::read(fixture.pile.path()).expect("read ambiguous fixture");
        let mut pile = Pile::open(fixture.pile.path()).expect("open ambiguous fixture");
        let error = migrate_legacy_model_main(
            &mut pile,
            &key(9),
            LegacyModelMigration {
                model: ModelSelector::Name(LEGACY_MODEL_NAME),
                source: CANONICAL_SOURCE,
                quantization: QUANTIZATION,
                tokenizer_name: None,
            },
        )
        .expect_err("ambiguous roots must fail");
        pile.close().expect("close failed migration pile");
        assert!(error
            .to_string()
            .contains("select exactly one legacy weight root"));
        assert_eq!(
            std::fs::read(fixture.pile.path()).expect("read failed migration pile"),
            before,
            "failed validation mutated the legacy pile"
        );
        assert_eq!(
            read_main_identity(fixture.pile.path()),
            (fixture.branch, fixture.head)
        );
    }

    #[test]
    fn audited_personaplex_policy_pins_the_real_commit_shape() {
        assert_eq!(PERSONAPLEX_MEMBER_POLICY.legacy_facts, 4_700);
        assert_eq!(PERSONAPLEX_MEMBER_POLICY.lm, 475);
        assert_eq!(PERSONAPLEX_MEMBER_POLICY.mimi, 318);
        assert_eq!(PERSONAPLEX_MEMBER_POLICY.total, 793);
    }

    #[test]
    fn personaplex_adoption_uses_exact_commit_and_is_a_zero_byte_retry() {
        let fixture = legacy_personaplex_fixture("personaplex-exact", false, false);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x71);

        // The later head still contains both weight roots and all three
        // members. Exact legacy fact cardinality is therefore an independent
        // fail-closed guard against accidentally selecting the moving head.
        let before_later_attempt = std::fs::read(fixture.pile.path()).unwrap();
        let mut pile = Pile::open(fixture.pile.path()).unwrap();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.later_commit,
            policy,
        )
        .expect_err("later unrelated head must not pass the audited checkout shape");
        assert!(
            error.to_string().contains("facts, expected exactly"),
            "{error}"
        );
        assert!(
            local_model_bundle_ticket(&mut pile, migration_key.verifying_key())
                .unwrap()
                .is_empty(),
            "failed exact-head validation exposed a bundle COMMIT"
        );
        assert_eq!(
            std::fs::read(fixture.pile.path()).unwrap(),
            before_later_attempt,
            "later-head rejection appended dependencies or a COMMIT"
        );

        let first = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("adopt exact legacy PersonaPlex weight commit");
        assert!(first.published);
        assert_eq!(first.legacy_commit, fixture.weight_commit);
        assert_eq!(first.legacy_lm_root, fixture.lm_root);
        assert_eq!(first.legacy_mimi_root, fixture.mimi_root);
        assert_ne!(first.model_root, fixture.lm_root);
        assert_ne!(first.model_root, fixture.mimi_root);

        let len_after_first = std::fs::metadata(fixture.pile.path()).unwrap().len();
        let repeated = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("repeat exact PersonaPlex adoption");
        assert!(!repeated.published);
        assert_eq!(repeated.commit, first.commit);
        assert_eq!(repeated.model_root, first.model_root);
        assert_eq!(repeated.model_archive_data, first.model_archive_data);
        assert_eq!(
            std::fs::metadata(fixture.pile.path()).unwrap().len(),
            len_after_first,
            "same-team exact `(root, H)` retry appended bytes"
        );
        pile.close().unwrap();

        let snapshot = exact_bundle_snapshot(fixture.pile.path(), first.team, first.commit);
        let archive: Blob<blobencodings::SimpleArchive> = snapshot
            .reader()
            .get(
                inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(
                    first.model_archive_data,
                ),
            )
            .unwrap();
        let model_facts = TribleSet::try_from_blob(archive).unwrap();
        let projection = project_legacy_model_attributes(&fixture.facts);
        assert!(
            projection
                .facts
                .iter()
                .all(|fact| model_facts.contains(fact)),
            "adoption rewrote or omitted exact legacy facts/ids"
        );
        assert!(
            !model_facts.contains(&fixture.later_fact),
            "exact weight checkout widened to the later current head"
        );
        assert_eq!(
            select_model_root(
                &model_facts,
                snapshot.reader(),
                ModelSelector::Source {
                    source: PERSONAPLEX_SOURCE,
                    quantization: mary::persist::QUANTIZATION_NATIVE,
                },
            )
            .unwrap(),
            first.model_root
        );
        assert_eq!(
            select_model_root(
                &model_facts,
                snapshot.reader(),
                ModelSelector::Name(PERSONAPLEX_LM_FILE),
            )
            .unwrap(),
            fixture.lm_root
        );
        assert_eq!(
            select_model_root(
                &model_facts,
                snapshot.reader(),
                ModelSelector::Name(PERSONAPLEX_MIMI_FILE),
            )
            .unwrap(),
            fixture.mimi_root
        );
        assert!(
            model_facts.iter().all(|fact| {
                fact.e() != &first.model_root || fact.a() != &attrs::model_name.id()
            }),
            "the unified root must not acquire two values for functional model_name"
        );
        let weights = PersonaPlexWeights::from_graph(&model_facts, snapshot.reader().clone())
            .expect("load adopted PersonaPlex graph through runtime resolver");
        assert_eq!(weights.root(), first.model_root);
        assert_eq!(weights.count(), policy.total);
    }

    #[test]
    fn exact_personaplex_commit_does_not_require_a_live_legacy_branch() {
        let fixture = legacy_personaplex_fixture("personaplex-detached", false, false);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x76);
        let mut pile = Pile::open(fixture.pile.path()).unwrap();

        let branch_metadata = pile
            .head(fixture.branch)
            .unwrap()
            .expect("live legacy main pin");
        assert!(matches!(
            pile.update(fixture.branch, Some(branch_metadata), None)
                .unwrap(),
            PushResult::Success()
        ));
        assert_eq!(pile.head(fixture.branch).unwrap(), None);
        assert_eq!(pile.pins().unwrap().count(), 0);

        let adopted = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("resident exact commit remains adoptable without a branch pin");
        assert!(adopted.published);
        assert_eq!(adopted.legacy_commit, fixture.weight_commit);
        pile.close().unwrap();

        let snapshot = exact_bundle_snapshot(fixture.pile.path(), adopted.team, adopted.commit);
        let archive: Blob<blobencodings::SimpleArchive> = snapshot
            .reader()
            .get(
                inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(
                    adopted.model_archive_data,
                ),
            )
            .unwrap();
        let model_facts = TribleSet::try_from_blob(archive).unwrap();
        let weights = PersonaPlexWeights::from_graph(&model_facts, snapshot.reader().clone())
            .expect("load branch-independent adopted PersonaPlex graph");
        assert_eq!(weights.root(), adopted.model_root);
        assert_eq!(weights.count(), policy.total);
    }

    #[test]
    fn personaplex_count_and_overlap_fail_before_any_publication() {
        let count_fixture = legacy_personaplex_fixture("personaplex-count", false, false);
        let mut wrong_count = personaplex_policy(&count_fixture);
        wrong_count.lm += 1;
        let before = std::fs::read(count_fixture.pile.path()).unwrap();
        let mut pile = Pile::open(count_fixture.pile.path()).unwrap();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &key(0x72),
            count_fixture.weight_commit,
            wrong_count,
        )
        .expect_err("wrong audited LM count must fail");
        assert!(error.to_string().contains("members, expected"), "{error}");
        assert!(
            local_model_bundle_ticket(&mut pile, key(0x72).verifying_key())
                .unwrap()
                .is_empty()
        );
        pile.close().unwrap();
        assert_eq!(std::fs::read(count_fixture.pile.path()).unwrap(), before);

        let overlap_fixture = legacy_personaplex_fixture("personaplex-overlap", true, false);
        let policy = personaplex_policy(&overlap_fixture);
        let before = std::fs::read(overlap_fixture.pile.path()).unwrap();
        let mut pile = Pile::open(overlap_fixture.pile.path()).unwrap();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &key(0x73),
            overlap_fixture.weight_commit,
            policy,
        )
        .expect_err("shared LM/Mimi member must fail");
        assert!(error.to_string().contains("share member"), "{error}");
        assert!(
            local_model_bundle_ticket(&mut pile, key(0x73).verifying_key())
                .unwrap()
                .is_empty()
        );
        pile.close().unwrap();
        assert_eq!(std::fs::read(overlap_fixture.pile.path()).unwrap(), before);
    }

    #[test]
    fn malformed_personaplex_fails_after_staging_without_a_bundle_commit() {
        let fixture = legacy_personaplex_fixture("personaplex-malformed", false, true);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x77);
        let before = std::fs::metadata(fixture.pile.path()).unwrap().len();
        let mut pile = Pile::open(fixture.pile.path()).unwrap();

        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect_err("duplicate tensor names must fail runtime validation");
        assert!(
            format!("{error:#}").contains("duplicate tensor name"),
            "{error:#}"
        );
        assert!(
            std::fs::metadata(fixture.pile.path()).unwrap().len() > before,
            "the falsifier must cross the dependency-staging boundary"
        );
        assert!(
            local_model_bundle_ticket(&mut pile, migration_key.verifying_key())
                .unwrap()
                .is_empty(),
            "failed staged validation finalized a bundle COMMIT"
        );
        pile.close().unwrap();
    }

    #[test]
    fn personaplex_conflict_is_detected_before_staging_dependencies() {
        let fixture = legacy_personaplex_fixture("personaplex-conflict", false, false);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x74);
        let conflicting = conflicting_personaplex_fragment();
        let conflicting_root = conflicting.root().unwrap();
        let mut pile = Pile::open(fixture.pile.path()).unwrap();
        let existing = publish_model_bundle_fragment(
            &mut pile,
            migration_key.verifying_key(),
            &migration_key,
            conflicting_root,
            conflicting,
        )
        .expect("publish conflicting same-team PersonaPlex bundle");
        let before = std::fs::metadata(fixture.pile.path()).unwrap().len();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect_err("a different authoritative PersonaPlex bundle must conflict");
        assert!(
            error
                .to_string()
                .contains("different PersonaPlex bundle is already authoritative"),
            "{error}"
        );
        assert_eq!(
            std::fs::metadata(fixture.pile.path()).unwrap().len(),
            before,
            "conflict staged dependencies before failing"
        );
        assert_eq!(
            local_model_bundle_ticket(&mut pile, migration_key.verifying_key()).unwrap(),
            vec![existing]
        );
        pile.close().unwrap();
    }

    #[test]
    fn same_personaplex_root_with_different_archive_conflicts_before_staging() {
        let fixture = legacy_personaplex_fixture("personaplex-same-root-conflict", false, false);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x78);
        let mut pile = Pile::open(fixture.pile.path()).unwrap();

        pile.refresh().unwrap();
        let reader = pile.reader().unwrap();
        let legacy = checkout_legacy_commit_ancestors(&reader, fixture.weight_commit).unwrap();
        let mut conflicting =
            prepare_legacy_personaplex_candidate(legacy, &reader, policy).unwrap();
        drop(reader);

        // Preserve the exact intrinsic PersonaPlex root while changing H with
        // one unrelated, valid row. This distinguishes pair equality from an
        // incorrect `same root || same archive` retry test.
        let unrelated = Trible::force(
            &test_id(0x79),
            &test_id(0x7a),
            &inlineencodings::GenId::inline_from(test_id(0x7b)),
        );
        conflicting.fragment += Fragment::from_facts_and_blobs(
            std::iter::once(unrelated).collect(),
            MemoryBlobStore::new(),
        );
        assert_eq!(conflicting.fragment.root(), Some(conflicting.model_root));
        let existing = publish_model_bundle_fragment(
            &mut pile,
            migration_key.verifying_key(),
            &migration_key,
            conflicting.model_root,
            conflicting.fragment,
        )
        .expect("publish same-root different-H authority");

        let before = std::fs::metadata(fixture.pile.path()).unwrap().len();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect_err("same root with a different H must conflict");
        assert!(
            error
                .to_string()
                .contains("different PersonaPlex bundle is already authoritative"),
            "{error}"
        );
        assert_eq!(
            std::fs::metadata(fixture.pile.path()).unwrap().len(),
            before,
            "same-root different-H conflict staged dependencies"
        );
        assert_eq!(
            local_model_bundle_ticket(&mut pile, migration_key.verifying_key()).unwrap(),
            vec![existing]
        );
        pile.close().unwrap();
    }

    #[test]
    fn retained_rewrite_reaches_legacy_tensor_blobs_only_through_bundle_commit() {
        let fixture = legacy_personaplex_fixture("personaplex-retention", false, false);
        let destination = TempPilePath::new("personaplex-retained");
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x75);
        let mut source = Pile::open(fixture.pile.path()).unwrap();
        let adopted = adopt_legacy_personaplex_bundle_with_policy(
            &mut source,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("adopt bundle before retained rewrite");

        // Remove the legacy Repository as an independent ownership path. If
        // the test left `main` active, its commit chain would retain every old
        // attachment even if COMMIT -> T -> H traversal were broken.
        let branch_metadata = source
            .head(fixture.branch)
            .unwrap()
            .expect("live legacy main pin");
        assert_ne!(
            branch_metadata, fixture.later_commit,
            "the strong pin names branch metadata, not the content commit"
        );
        assert!(matches!(
            source
                .update(fixture.branch, Some(branch_metadata), None)
                .unwrap(),
            PushResult::Success()
        ));
        assert_eq!(source.head(fixture.branch).unwrap(), None);
        assert_eq!(
            source.pins().unwrap().count(),
            0,
            "an active strong pin confounds retention"
        );

        let mut retained = Pile::open(destination.path()).unwrap();
        source
            .rewrite_retained_into(
                &mut retained,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            )
            .expect("rewrite native bundle closure");
        source.close().unwrap();

        let snapshot =
            snapshot_model_bundle_collection_exact(&mut retained, adopted.team, &[adopted.commit])
                .expect("materialize retained exact bundle");
        let token_archive: Blob<blobencodings::SimpleArchive> = snapshot
            .reader()
            .get(
                inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(
                    adopted.commit.data(),
                ),
            )
            .expect("COMMIT retained its one-row token T");
        let token = TribleSet::try_from_blob(token_archive).unwrap();
        assert_eq!(&token, snapshot.facts());
        assert_eq!(token.len(), 1);
        let token_row = token.iter().next().unwrap();
        assert_eq!(token_row.e(), &adopted.model_root);
        assert_eq!(token_row.a(), &metadata::archive.id());
        let model_archive = *token_row.v::<inlineencodings::Handle<blobencodings::SimpleArchive>>();
        assert_eq!(
            inlineencodings::Handle::<blobencodings::SimpleArchive>::to_hash(model_archive),
            adopted.model_archive_data
        );
        let archive: Blob<blobencodings::SimpleArchive> = snapshot
            .reader()
            .get(model_archive)
            .expect("T retained the exact model archive H");
        let facts = TribleSet::try_from_blob(archive).unwrap();
        assert!(
            fixture.facts.iter().all(|fact| facts.contains(fact)),
            "retained H omitted original historical rows"
        );
        for (index, ((leaf_id, data), shape)) in fixture
            .leaf_ids
            .iter()
            .zip(&fixture.data)
            .zip(&fixture.shapes)
            .enumerate()
        {
            let payload: View<[f32]> = snapshot.reader().get(*data).unwrap();
            let dimensions: View<[u64]> = snapshot.reader().get(*shape).unwrap();
            assert_eq!(&*payload, &[index as f32 + 1.0]);
            assert_eq!(&*dimensions, &[1]);
            let leaf = mary::leaf::resolve(&facts, snapshot.reader(), *leaf_id)
                .unwrap()
                .expect("legacy two-blob leaf remains resolvable");
            assert_eq!(leaf.dims(), &[1]);
            assert_eq!(&*leaf.view_f32().unwrap(), &[index as f32 + 1.0]);
        }
        assert!(
            snapshot.reader().get::<Bytes, _>(fixture.orphan).is_err(),
            "unrelated later-commit attachment survived without an ownership path"
        );
        drop(snapshot);
        retained.close().unwrap();
    }
}
