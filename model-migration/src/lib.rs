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

use anyhow::{anyhow, bail, Context};
use ed25519_dalek::SigningKey;
use triblespace::core::collection::CollectionCommit;
use triblespace::core::repo::pile::Pile;
use triblespace::core::repo::{ancestors, BlobStore, CommitHandle, Repository};
use triblespace::prelude::blobencodings::LongString;
use triblespace::prelude::inlineencodings::{Handle, ShortString};
use triblespace::prelude::*;

use mary::format::attrs;
use mary::model_collection::{project_legacy_model_attributes, publish_model_fragment};
use mary::selection::{
    select_model_root, select_tokenizer_root, ModelSelector, TokenizerSelector,
};

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

    let model_root = select_model_root(&projection.facts, &reader, request.model)
        .with_context(|| {
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

    // Whose collection the migrated graph joins: the pile's existing
    // model-graph team if it already has one, else this signer owns it. Same
    // rule every other publisher in mary follows, so a migrated root and an
    // imported one land in one collection rather than two.
    let team = mary::model_collection::model_graph_team_or_own(pile, signing_key)
        .context("determine the model-graph team for the migrated collection")?;
    let commit = publish_model_fragment(pile, team, signing_key, fragment)
        .context("publish migrated model graph to Mary's native collection")?;

    Ok(LegacyModelMigrationResult {
        commit,
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

    use anybytes::View;
    use triblespace::core::blob::MemoryBlobStore;
    use triblespace::core::inline::encodings::UnknownInline;
    use triblespace::core::repo::pile::PileReader;

    use super::*;
    use mary::model_collection::snapshot_model_collection_exact;

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
        commit: CollectionCommit,
    ) -> triblespace::core::collection::CollectionSnapshot<PileReader> {
        let mut pile = Pile::open(path).expect("open pile for exact native read");
        // Read the team back out of the pile rather than restating it here, so
        // the test cannot pass against a collection published under a team the
        // migration did not actually choose.
        let team = mary::model_collection::sole_model_graph_team(&mut pile)
            .expect("the migrated pile publishes exactly one model-graph team");
        let snapshot = snapshot_model_collection_exact(&mut pile, team, &[commit])
            .expect("materialize exact migration ticket");
        pile.close().expect("close exact-read pile");
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

        let snapshot = exact_snapshot(fixture.pile.path(), first.commit);
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
        let snapshot = exact_snapshot(fixture.pile.path(), result.commit);
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
}
