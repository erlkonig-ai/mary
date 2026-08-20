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
use triblespace::core::collection::CollectionSnapshot;
use triblespace::core::repo::pile::PileReader;
use triblespace::prelude::{BlobStoreGet, TribleSet};

/// Canonical source coordinate of the complete PersonaPlex LM + Mimi model.
pub const SOURCE: &str = "nvidia/personaplex-7b-v1";

/// One exact PersonaPlex model selected from an immutable native collection
/// snapshot.
///
/// The root is the union of the LM and Mimi checkpoint tensors. Runtime code
/// deliberately accepts only faithful f32 leaves under [`SOURCE`] and
/// `quantization="native"`; sibling-pile runtime derivatives remain a
/// separate representation layered on top of this authority.
pub struct PersonaPlexWeights<R> {
    selected: SelectedModelIndex<R>,
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

    /// Select the canonical exact root and reject incompatible leaf widths.
    pub fn from_snapshot(snapshot: CollectionSnapshot<R>) -> anyhow::Result<Self> {
        let selected = SelectedModelIndex::from_snapshot(
            snapshot,
            ModelSelector::Source {
                source: SOURCE,
                quantization: crate::persist::QUANTIZATION_NATIVE,
            },
        )?;
        Self::admit(selected)
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

pub mod config;
pub mod depth;
pub mod depth_fast;
pub mod lmgen;
pub mod mimi;
pub mod pipeline;
pub mod prompt;
pub mod sampling;
// Derived runtime-format sibling piles (zero-copy load seam): the format
// marker/ABI, the derive step, and the auto-discovery loaders.
#[cfg(all(feature = "q4", target_os = "macos"))]
pub mod qpile;
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

    fn pile_model_fragment(pile: &mut Pile, tensors: &[(&str, &[f32], &[u64])]) -> Fragment {
        let mut fragment = Fragment::empty();
        let mut members = Vec::new();
        for &(tensor, values, dimensions) in tensors {
            let leaf = crate::format::put_raw(pile, values, dimensions).expect("put tensor blobs");
            let leaf_id = leaf.root().expect("tensor leaf root");
            fragment += leaf;
            let name = pile
                .put::<LongString, _>(tensor.to_owned())
                .expect("put tensor name");
            let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: leaf_id };
            members.push(member.root().expect("model member root"));
            fragment += member;
        }
        let root = entity! { _ @ attrs::member*: members.iter() };
        let root_id = root.root().expect("model root");
        fragment += root;
        let source = pile
            .put::<LongString, _>(SOURCE.to_owned())
            .expect("put source coordinate");
        fragment += entity! { ExclusiveId::force_ref(&root_id) @
            attrs::source: source,
            attrs::quantization: crate::persist::QUANTIZATION_NATIVE,
        };
        let (_, facts, metafacts, blobs) = fragment.into_parts();
        Fragment::rooted_from_parts(root_id, facts, metafacts, blobs)
    }

    #[test]
    fn exact_union_is_selected_and_repeated_publication_appends_nothing() {
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
        let first = crate::model_collection::publish_model_fragment(
            &mut pile,
            test_team(),
            &signing_key,
            fragment.clone(),
        )
        .expect("publish exact PersonaPlex root");
        let len_after_first = std::fs::metadata(file.path()).unwrap().len();
        let repeated = crate::model_collection::publish_model_fragment(
            &mut pile,
            test_team(),
            &signing_key,
            fragment,
        )
        .expect("repeat exact PersonaPlex publication");
        let len_after_retry = std::fs::metadata(file.path()).unwrap().len();
        assert_eq!(first, repeated);
        assert_eq!(len_after_first, len_after_retry);

        let snapshot = crate::model_collection::snapshot_model_collection_local_latest(&mut pile, test_team())
            .expect("freeze exact PersonaPlex prefix");
        let weights =
            PersonaPlexWeights::from_snapshot(snapshot).expect("select exact PersonaPlex root");
        assert_eq!(weights.root(), root);
        assert_eq!(weights.count(), 2);
        assert!(weights.exact().contains_key("transformer.weight"));
        assert!(weights.exact().contains_key("encoder.weight"));
        drop(weights);
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn non_f32_exact_coordinate_fails_closed() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        crate::model_collection::publish_model_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x50; 32]),
            model_fragment(&[("weight", &[1.0], &[1])], true),
        )
        .expect("publish incompatible PersonaPlex root");
        let snapshot = crate::model_collection::snapshot_model_collection_local_latest(&mut pile, test_team())
            .expect("freeze incompatible PersonaPlex prefix");
        let error = PersonaPlexWeights::from_snapshot(snapshot)
            .err()
            .expect("f16 exact coordinate must fail");
        assert!(error.to_string().contains("is not f32"), "{error:#}");
        pile.close().expect("close synthetic PersonaPlex pile");
    }

    #[test]
    fn conflicting_staged_root_does_not_publish_authority() {
        let file = TestPile::new();
        let mut pile = Pile::open(file.path()).expect("open synthetic PersonaPlex pile");
        let existing = pile_model_fragment(&mut pile, &[("weight", &[1.0], &[1])]);
        let existing_commit = crate::model_collection::publish_model_fragment(
            &mut pile,
            test_team(),
            &SigningKey::from_bytes(&[0x51; 32]),
            existing,
        )
        .expect("publish existing PersonaPlex authority");

        let candidate = pile_model_fragment(&mut pile, &[("weight", &[2.0], &[1])]);
        let snapshot = crate::model_collection::snapshot_model_collection_local_latest(&mut pile, test_team())
            .expect("freeze preexisting authority after staging candidate blobs");
        assert_eq!(snapshot.commits(), &[existing_commit]);
        let (mut candidate_view, _, reader) = snapshot.into_parts();
        candidate_view += candidate.facts().clone();
        let error = PersonaPlexWeights::from_graph(&candidate_view, reader)
            .err()
            .expect("conflicting Source/native roots must fail closed");
        assert!(format!("{error:#}").contains("ambiguous"), "{error:#}");

        let unchanged = crate::model_collection::snapshot_model_collection_local_latest(&mut pile, test_team())
            .expect("re-read authority after rejected candidate");
        assert_eq!(unchanged.commits(), &[existing_commit]);
        drop(unchanged);
        pile.close().expect("close synthetic PersonaPlex pile");
    }
}
