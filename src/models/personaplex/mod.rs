//! PersonaPlex-7B full-duplex model port. Phase 1: the Mimi neural audio codec
//! ([`mimi`]). Phase 0 of the LM port: verified architecture constants in
//! [`config`] (from the real checkpoint's tensor shapes) and the weight pile
//! (`personaplex_persist`). LM part 1: the 7B [`temporal`] transformer
//! forward. LM part 2: the [`depth`] transformer (depformer) and the
//! [`lmgen`] delay/undelay step machinery (all CPU-f32 parity vs the moshi
//! oracle — `personaplex_probe`). LM part 3: the end-to-end [`pipeline`] —
//! input WAV → Mimi encode → LM free-run → agent streams 1..=8 → Mimi decode
//! → 24 kHz audio out. Realtime lane A: the [`temporal_metal`] q4/Metal
//! decode build (feature `q4`; gated by `personaplex_rt_probe`). Realtime
//! lane B: [`depth_fast`] — the Accelerate/NEON CPU depformer predictor
//! (preloaded per-step weight sets, fixed buffers, optional f16 storage with
//! f32 accumulate; gate+bench `moshi_depth_probe`). Phase 5: the prompt
//! machinery — [`spm`] (pure-Rust SentencePiece unigram text tokenizer, encode
//! + decode), [`voice_prompt`] (packaged voice `.pt` reader) and [`prompt`]
//! (system prompts assembled from primary sources instead of golden npys;
//! gated by `personaplex_probe prompt`). Realtime foundation: [`sampling`]
//! (seedable temperature / top-k / top-p over the text + audio heads; greedy
//! stays the parity default) and the `reset_session` seam on the [`lmgen`] /
//! [`pipeline`] step machines (a new conversation without a weight reload).

use crate::leaf::{Elem, Leaf};
use crate::nn::weight_loader::WeightLoader;
use crate::selection::{ModelSelector, SelectedModelIndex};
use ed25519_dalek::VerifyingKey;
use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::core::collection::{CollectionCommit, CollectionData, CollectionSnapshot};
use triblespace::core::metadata;
use triblespace::core::repo::pile::PileReader;
use triblespace::prelude::{blobencodings, inlineencodings, BlobStoreGet, Id, Inline, TribleSet};

/// Canonical source coordinate of the complete PersonaPlex LM + Mimi model.
pub const SOURCE: &str = "nvidia/personaplex-7b-v1";

/// One exact PersonaPlex model selected from a self-contained model graph.
///
/// The root is the union of the LM and Mimi checkpoint tensors. Runtime code
/// deliberately accepts only faithful f32 leaves under [`SOURCE`] and
/// `quantization="native"`. Steady-state loads enter through signed model
/// bundles; direct graph selection exists only for validating a staged
/// candidate before its token COMMIT is finalized.
pub struct PersonaPlexWeights<R> {
    selected: SelectedModelIndex<R>,
}

/// Exact signed bundle authority retained alongside one PersonaPlex loader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersonaPlexAuthority {
    team: VerifyingKey,
    model_root: Id,
    model_archive_data: CollectionData,
    bundle_token_data: CollectionData,
    ticket: Vec<CollectionCommit>,
}

impl PersonaPlexAuthority {
    pub fn team(&self) -> VerifyingKey {
        self.team
    }

    pub fn model_root(&self) -> Id {
        self.model_root
    }

    /// Complete canonical model-fact archive `H`.
    pub fn model_archive_data(&self) -> CollectionData {
        self.model_archive_data
    }

    /// Source-collection data `T`: the one-row archive containing `τ`.
    pub fn bundle_token_data(&self) -> CollectionData {
        self.bundle_token_data
    }

    /// Complete exact source ticket frozen with this loader.
    pub fn ticket(&self) -> &[CollectionCommit] {
        &self.ticket
    }
}

/// PersonaPlex weights plus the exact authority from which they were loaded.
pub struct PersonaPlexBundle<R> {
    weights: PersonaPlexWeights<R>,
    authority: PersonaPlexAuthority,
}

/// Bundle-bound authority and loader for deterministic runtime transforms.
///
/// Its fields are private so a caller cannot label weights from one bundle
/// with another bundle's `(team, root, H, τ)` identity.
pub struct PersonaPlexRuntimeSource {
    authority: PersonaPlexAuthority,
    loader: WeightLoader,
}

impl PersonaPlexRuntimeSource {
    pub fn authority(&self) -> &PersonaPlexAuthority {
        &self.authority
    }

    pub(crate) fn loader(&self) -> &WeightLoader {
        &self.loader
    }
}

impl<R> PersonaPlexBundle<R> {
    pub fn authority(&self) -> &PersonaPlexAuthority {
        &self.authority
    }

    pub fn weights(&self) -> &PersonaPlexWeights<R> {
        &self.weights
    }
}

impl<R: BlobStoreGet> PersonaPlexWeights<R> {
    fn admit(selected: SelectedModelIndex<R>) -> anyhow::Result<Self> {
        if let Some(name) = selected
            .handles()
            .iter()
            .filter_map(|(name, leaf)| (leaf.elem() == Elem::F16).then_some(name))
            .min()
        {
            anyhow::bail!("PersonaPlex exact tensor {name:?} is not f32");
        }
        Ok(Self { selected })
    }

    /// Select the canonical exact root from an explicit graph and owning
    /// reader. This supports commit-last validation of staged candidate facts.
    pub fn from_graph(facts: &TribleSet, reader: R) -> anyhow::Result<Self> {
        let selected = SelectedModelIndex::from_graph(
            facts,
            reader,
            ModelSelector::Source {
                source: SOURCE,
                quantization: crate::persist::QUANTIZATION_NATIVE,
            },
        )?;
        Self::admit(selected)
    }

    /// Select PersonaPlex from individually self-contained signed bundle
    /// tokens, retaining the frozen ticket and exact `(root, H, τ)` identity.
    pub fn find_in_bundle_snapshot(
        team: VerifyingKey,
        snapshot: CollectionSnapshot<R>,
    ) -> anyhow::Result<Option<PersonaPlexBundle<R>>>
    where
        R: Clone,
    {
        use anyhow::{anyhow, Context};
        use std::collections::BTreeSet;

        let (_, ticket, reader) = snapshot.into_parts();
        let expected_collection =
            crate::model_collection::model_bundle_collection_handle(team);
        anyhow::ensure!(
            ticket
                .iter()
                .all(|commit| commit.collection() == expected_collection),
            "bundle snapshot contains a COMMIT outside the collection derived from the supplied team"
        );
        let mut seen_tokens = BTreeSet::new();
        let mut selected: Option<(Self, Id, CollectionData, CollectionData)> = None;
        for commit in &ticket {
            if !seen_tokens.insert(commit.data()) {
                continue;
            }
            let token_blob: Blob<SimpleArchive> = reader
                .get(inlineencodings::Handle::<SimpleArchive>::from_hash(commit.data()))
                .map_err(|error| anyhow!("read bundle token {}: {error}", commit.id()))?;
            let token = TribleSet::try_from_blob(token_blob)
                .with_context(|| format!("decode bundle token {}", commit.id()))?;
            anyhow::ensure!(
                token.len() == 1,
                "bundle COMMIT {} data must be exactly one token row, found {}",
                commit.id(),
                token.len()
            );
            let fact = token.iter().next().expect("one-row bundle token");
            anyhow::ensure!(
                fact.a() == &metadata::archive.id(),
                "bundle COMMIT {} token does not use metadata::archive",
                commit.id()
            );
            let root = *fact.e();
            let model_archive_data = inlineencodings::Handle::<SimpleArchive>::to_hash(
                *fact.v::<inlineencodings::Handle<SimpleArchive>>(),
            );
            let source_blob: Blob<SimpleArchive> = reader
                .get(inlineencodings::Handle::<SimpleArchive>::from_hash(model_archive_data))
                .map_err(|error| anyhow!("read model archive H for bundle {}: {error}", commit.id()))?;
            triblespace::core::collection::simplearchive_union::validate_element(&source_blob)
                .map_err(|error| anyhow!("model archive H for bundle {} is not canonical: {error}", commit.id()))?;
            let facts = TribleSet::try_from_blob(source_blob)
                .with_context(|| format!("decode model archive H for bundle {}", commit.id()))?;
            anyhow::ensure!(
                facts.iter().any(|row| row.e() == &root),
                "bundle COMMIT {} asserts root {root} absent from H",
                commit.id()
            );

            let native = triblespace::prelude::exists!(triblespace::prelude::pattern!(
                &facts,
                [{ root @ crate::format::attrs::quantization: crate::persist::QUANTIZATION_NATIVE }]
            ));
            if !native {
                continue;
            }
            let mut source_matches = false;
            for (handle,) in triblespace::prelude::find!(
                (source: Inline<inlineencodings::Handle<blobencodings::LongString>>),
                triblespace::prelude::pattern!(&facts, [{ root @ crate::format::attrs::source: ?source }])
            ) {
                let value: anybytes::View<str> = reader
                    .get(handle)
                    .map_err(|error| anyhow!("read PersonaPlex source coordinate: {error}"))?;
                source_matches |= value.as_ref() == SOURCE;
            }
            if !source_matches {
                continue;
            }

            let weights = Self::from_graph(&facts, reader.clone())
                .with_context(|| format!("validate PersonaPlex bundle {}", commit.id()))?;
            anyhow::ensure!(
                weights.root() == root,
                "bundle COMMIT {} asserts root {root}, but H selects {}",
                commit.id(),
                weights.root()
            );
            if let Some((_, existing_root, existing_h, _)) = &selected {
                anyhow::ensure!(
                    *existing_root == root && *existing_h == model_archive_data,
                    "ambiguous PersonaPlex bundle authority: ({existing_root}, {existing_h:?}) and ({root}, {model_archive_data:?})"
                );
            } else {
                selected = Some((weights, root, model_archive_data, commit.data()));
            }
        }

        let Some((weights, model_root, model_archive_data, bundle_token_data)) = selected else {
            return Ok(None);
        };
        Ok(Some(PersonaPlexBundle {
            weights,
            authority: PersonaPlexAuthority {
                team,
                model_root,
                model_archive_data,
                bundle_token_data,
                ticket,
            },
        }))
    }

    /// Required form of [`Self::find_in_bundle_snapshot`].
    pub fn from_bundle_snapshot(
        team: VerifyingKey,
        snapshot: CollectionSnapshot<R>,
    ) -> anyhow::Result<PersonaPlexBundle<R>>
    where
        R: Clone,
    {
        Self::find_in_bundle_snapshot(team, snapshot)?
            .ok_or_else(|| anyhow::anyhow!("no signed PersonaPlex bundle in exact ticket"))
    }

    /// Content-addressed root of the complete exact model.
    pub fn root(&self) -> triblespace::prelude::Id {
        self.selected.root()
    }

    /// Number of tensors in the LM + Mimi union.
    pub fn count(&self) -> usize {
        self.selected.handles().len()
    }

    /// Exact tensor index retained for source-parity gates.
    pub fn exact(&self) -> &std::collections::HashMap<String, Leaf> {
        self.selected.handles()
    }

    /// Reader owning every attachment named by [`Self::exact`].
    pub fn reader(&self) -> &R {
        self.selected.reader()
    }
}

impl PersonaPlexWeights<PileReader> {
    /// Consume the exact model into the platform's established lazy loader.
    pub fn into_loader(self) -> WeightLoader {
        let (_, exact, _reader) = self.selected.into_parts();
        #[cfg(target_os = "macos")]
        {
            return WeightLoader::Aliased(crate::nn::weight_loader::AliasedPile::new(
                std::collections::HashMap::new(),
                exact,
                crate::nn::backend::WgpuDevice::default(),
            ));
        }
        #[cfg(not(target_os = "macos"))]
        {
            let keymap = exact
                .into_iter()
                .map(|(name, leaf)| (name, leaf.to_f32_shape()))
                .collect();
            WeightLoader::Pile(keymap)
        }
    }
}

impl PersonaPlexBundle<PileReader> {
    /// Split a bundle-native load into its immutable authority and runtime
    /// loader without reopening the pile or observing a wider prefix.
    pub fn into_parts(self) -> (PersonaPlexAuthority, WeightLoader) {
        (self.authority, self.weights.into_loader())
    }

    /// Preserve the authority/loader binding for runtime transforms.
    pub fn into_runtime_source(self) -> PersonaPlexRuntimeSource {
        PersonaPlexRuntimeSource {
            authority: self.authority,
            loader: self.weights.into_loader(),
        }
    }
}

pub mod config;
pub mod depth;
pub mod depth_fast;
pub mod lmgen;
pub mod mimi;
pub mod pipeline;
pub mod prompt;
pub mod sampling;
pub mod spm;
pub mod temporal;
#[cfg(feature = "q4")]
pub mod temporal_metal;
pub mod voice_prompt;

#[cfg(test)]
mod native_authority_tests {
    use super::*;
    use crate::format::{attrs, F32Array, U64Array};
    use ed25519_dalek::SigningKey;
    use std::fs::OpenOptions;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use triblespace::core::repo::pile::Pile;
    use triblespace::prelude::blobencodings::LongString;
    use triblespace::prelude::*;

    static NEXT_TEST_PILE: AtomicU64 = AtomicU64::new(0);

    /// The one team these fixtures publish under. Fixed, because a snapshot
    /// must name the same team the commit was published to.
    fn test_team() -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[0x50; 32]).verifying_key()
    }

    struct TestPile(PathBuf);

    impl TestPile {
        fn new() -> Self {
            let ordinal = NEXT_TEST_PILE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mary-personaplex-native-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create synthetic PersonaPlex pile");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestPile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn model_fragment(tensors: &[(&str, &[f32], &[u64])], f16: bool) -> Fragment {
        let mut fragment = Fragment::empty();
        let mut members = Vec::new();
        for &(tensor, values, dimensions) in tensors {
            let shape = fragment.put::<U64Array, _>(dimensions.to_vec());
            let leaf = if f16 {
                let values: Vec<_> = values.iter().copied().map(half::f16::from_f32).collect();
                let data = fragment.put::<crate::f16enc::F16Array, _>(values);
                entity! { _ @ attrs::data_f16: data, attrs::shape: shape }
            } else {
                let data = fragment.put::<F32Array, _>(values.to_vec());
                entity! { _ @ attrs::data: data, attrs::shape: shape }
            };
            let leaf_id = leaf.root().expect("tensor leaf root");
            fragment += leaf;
            let name = fragment.put::<LongString, _>(tensor.to_owned());
            let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: leaf_id };
            members.push(member.root().expect("model member root"));
            fragment += member;
        }
        let root = entity! { _ @ attrs::member*: members.iter() };
        let root_id = root.root().expect("model root");
        fragment += root;
        let source = fragment.put::<LongString, _>(SOURCE.to_owned());
        fragment += entity! { ExclusiveId::force_ref(&root_id) @
            attrs::source: source,
            attrs::quantization: crate::persist::QUANTIZATION_NATIVE,
        };
        let (_, facts, metafacts, blobs) = fragment.into_parts();
        Fragment::rooted_from_parts(root_id, facts, metafacts, blobs)
    }

    #[test]
    fn exact_bundle_is_selected_and_repeated_publication_appends_nothing() {
        let file = TestPile::new();
        let fragment = model_fragment(
            &[
                ("transformer.weight", &[1.0, 2.0], &[2]),
                ("encoder.weight", &[3.0, 4.0], &[1, 2]),
            ],
            false,
        );
        let root = fragment.root().expect("PersonaPlex model root");
        let signing_key = SigningKey::from_bytes(&[0x50; 32]);
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let first = crate::model_collection::publish_model_bundle_fragment(
            &mut pile,
            test_team(),
            &signing_key,
            root,
            fragment.clone(),
        )
        .expect("publish exact PersonaPlex root");
        let len_after_first = std::fs::metadata(file.path()).unwrap().len();
        let repeated = crate::model_collection::publish_model_bundle_fragment(
            &mut pile,
            test_team(),
            &signing_key,
            root,
            fragment,
        )
        .expect("repeat exact PersonaPlex publication");
        let len_after_retry = std::fs::metadata(file.path()).unwrap().len();
        assert_eq!(first, repeated);
        assert_eq!(len_after_first, len_after_retry);

        let snapshot = crate::model_collection::snapshot_model_bundle_collection_local_latest(&mut pile, test_team())
            .expect("freeze exact PersonaPlex prefix");
        let bundle = PersonaPlexWeights::from_bundle_snapshot(test_team(), snapshot)
            .expect("select exact PersonaPlex bundle");
        let authority = bundle.authority();
        assert_eq!(authority.model_root(), root);
        assert_eq!(authority.bundle_token_data(), first.data());
        assert_eq!(authority.ticket(), &[first]);
        let weights = bundle.weights();
        assert_eq!(weights.root(), root);
        assert_eq!(weights.count(), 2);
        assert!(weights.exact().contains_key("transformer.weight"));
        assert!(weights.exact().contains_key("encoder.weight"));
        drop(bundle);
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn bundle_snapshot_cannot_be_relabelled_as_another_team() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let fragment = model_fragment(&[("weight", &[1.0], &[1])], false);
        let root = fragment.root().unwrap();
        crate::model_collection::publish_model_bundle_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x50; 32]),
            root,
            fragment,
        )
        .expect("publish PersonaPlex bundle");
        let snapshot =
            crate::model_collection::snapshot_model_bundle_collection_local_latest(
                &mut pile,
                test_team(),
            )
            .expect("freeze PersonaPlex bundle");
        let foreign_team = SigningKey::from_bytes(&[0x60; 32]).verifying_key();
        let error = PersonaPlexWeights::from_bundle_snapshot(foreign_team, snapshot)
            .err()
            .expect("foreign team must not be paired with the snapshot");
        assert!(
            error.to_string().contains("supplied team"),
            "{error:#}"
        );
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn non_f32_exact_coordinate_fails_closed() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let fragment = model_fragment(&[("weight", &[1.0], &[1])], true);
        let root = fragment.root().unwrap();
        crate::model_collection::publish_model_bundle_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x50; 32]),
            root,
            fragment,
        )
        .expect("publish incompatible PersonaPlex root");
        let snapshot = crate::model_collection::snapshot_model_bundle_collection_local_latest(&mut pile, test_team())
            .expect("freeze incompatible PersonaPlex prefix");
        let error = PersonaPlexWeights::from_bundle_snapshot(test_team(), snapshot)
            .err()
            .expect("f16 exact coordinate must fail");
        assert!(format!("{error:#}").contains("is not f32"), "{error:#}");
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn conflicting_bundle_roots_fail_closed() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let existing = model_fragment(&[("weight", &[1.0], &[1])], false);
        let existing_root = existing.root().unwrap();
        let existing_commit = crate::model_collection::publish_model_bundle_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x51; 32]),
            existing_root,
            existing,
        )
        .expect("publish existing PersonaPlex authority");

        let candidate = model_fragment(&[("weight", &[2.0], &[1])], false);
        let candidate_root = candidate.root().unwrap();
        let candidate_commit = crate::model_collection::publish_model_bundle_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x52; 32]),
            candidate_root,
            candidate,
        )
        .expect("publish conflicting PersonaPlex authority");
        let snapshot = crate::model_collection::snapshot_model_bundle_collection_local_latest(&mut pile, test_team())
            .expect("freeze conflicting bundle authority");
        let error = PersonaPlexWeights::from_bundle_snapshot(test_team(), snapshot)
            .err()
            .expect("conflicting Source/native roots must fail closed");
        assert!(format!("{error:#}").contains("ambiguous"), "{error:#}");
        assert_ne!(existing_commit.data(), candidate_commit.data());
        pile.close().expect("close synthetic PersonaPlex pile");
    }
}
