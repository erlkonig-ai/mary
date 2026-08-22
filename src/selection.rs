//! Deterministic model and tokenizer selection over an already-open graph.
//!
//! A consolidated collection can contain several models and tokenizers. The
//! old pile wrappers predated that: they selected the first matching root or
//! extended a `HashMap`, making iteration order decide ambiguous data. This
//! module keeps storage out of the decision. Callers pass a materialized
//! [`TribleSet`], its blob reader, and an explicit selector; every selector and
//! every functional model field has exact-cardinality semantics.

use crate::format::attrs;
use crate::leaf::Leaf;
use anyhow::{anyhow, bail, Context};
use std::collections::{BTreeSet, HashMap};
use triblespace::core::collection::CollectionSnapshot;
use triblespace::prelude::*;

/// How to identify one model root in a consolidated graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSelector<'a> {
    /// Succeed only when the graph contains exactly one model root.
    Only,
    /// Select the exact content-addressed root.
    Root(Id),
    /// Select the one legacy/root entity carrying this exact `model_name`.
    Name(&'a str),
    /// Select the one content-addressed root carrying this source and weight
    /// format label.
    Source {
        source: &'a str,
        quantization: &'a str,
    },
}

/// How to identify one tokenizer root in a consolidated graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerSelector<'a> {
    /// Succeed only when the graph contains exactly one tokenizer root.
    Only,
    /// Select the exact content-addressed root.
    Root(Id),
    /// Select the one tokenizer carrying this exact `model_name`.
    Name(&'a str),
}

/// One explicitly selected model root, its strict tensor-handle index, and the
/// blob reader that owns every indexed attachment.
///
/// Selection consumes a frozen [`CollectionSnapshot`]. Once the root and its
/// functional fields have been validated, the collection facts and commit
/// ticket are no longer needed by weight loading; the reader is retained so
/// the content handles remain resolvable without reopening storage. This is
/// the storage-policy-free boundary for lazy, streaming, and mmap-aliased
/// loaders.
pub struct SelectedModelIndex<R> {
    root: Id,
    handles: HashMap<String, Leaf>,
    reader: R,
}

impl<R> SelectedModelIndex<R> {
    /// Exact content-addressed model root selected from the frozen graph.
    pub fn root(&self) -> Id {
        self.root
    }

    /// Strict `tensor name -> leaf` index for the selected root.
    pub fn handles(&self) -> &HashMap<String, Leaf> {
        &self.handles
    }

    /// Reader that owns the attachment snapshot named by [`Self::handles`].
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// Consume the selection into its root, leaf index, and owned reader.
    pub fn into_parts(self) -> (Id, HashMap<String, Leaf>, R) {
        (self.root, self.handles, self.reader)
    }
}

fn read_long_string(
    blobs: &impl BlobStoreGet,
    handle: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
    field: &str,
) -> anyhow::Result<String> {
    let value: anybytes::View<str> = blobs
        .get(handle)
        .map_err(|error| anyhow!("read {field} blob: {error}"))?;
    Ok(value.to_string())
}

fn exactly_one<T>(
    values: impl IntoIterator<Item = T>,
    what: impl std::fmt::Display,
) -> anyhow::Result<T> {
    let mut values = values.into_iter();
    let Some(value) = values.next() else {
        bail!("no {what}");
    };
    if values.next().is_some() {
        bail!("ambiguous {what}");
    }
    Ok(value)
}

fn model_roots(tribles: &TribleSet) -> BTreeSet<Id> {
    let named = find!(
        (model: Id),
        pattern!(tribles, [{ ?model @ attrs::model_name: _?name, attrs::member: _?member }])
    );
    let sourced = find!(
        (model: Id),
        pattern!(tribles, [{ ?model @ attrs::source: _?source, attrs::member: _?member }])
    );
    named.chain(sourced).map(|(model,)| model).collect()
}

fn validate_source_coordinates(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    root: Id,
    wanted_source: &str,
    wanted_quantization: &str,
) -> anyhow::Result<()> {
    let source_handle = exactly_one(
        find!(
            (source: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(tribles, [{ root @ attrs::source: ?source }])
        )
        .map(|(source,)| source),
        format_args!("source field on model root {root}"),
    )?;
    let source = read_long_string(blobs, source_handle, "source")?;
    let quantization = exactly_one(
        find!(
            (quantization: String),
            pattern!(tribles, [{ root @ attrs::quantization: ?quantization }])
        )
        .map(|(quantization,)| quantization),
        format_args!("quantization field on model root {root}"),
    )?;
    if source != wanted_source || quantization != wanted_quantization {
        bail!(
            "model root {root} coordinates changed while selecting: ({source:?}, {quantization:?})"
        );
    }
    Ok(())
}

fn validate_model_name(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    root: Id,
    wanted: &str,
) -> anyhow::Result<()> {
    let handle = exactly_one(
        find!(
            (name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(tribles, [{ root @ attrs::model_name: ?name }])
        )
        .map(|(name,)| name),
        format_args!("model_name field on model root {root}"),
    )?;
    let name = read_long_string(blobs, handle, "model_name")?;
    if name != wanted {
        bail!("model root {root} name changed while selecting");
    }
    Ok(())
}

/// Resolve exactly one model root according to `selector`.
pub fn select_model_root(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: ModelSelector<'_>,
) -> anyhow::Result<Id> {
    match selector {
        ModelSelector::Only => exactly_one(model_roots(tribles), "model root in graph"),
        ModelSelector::Root(root) => {
            if model_roots(tribles).contains(&root) {
                Ok(root)
            } else {
                bail!("model root {root} is absent or has no members")
            }
        }
        ModelSelector::Name(wanted) => {
            let matches = find!(
                (model: Id, name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ ?model @ attrs::model_name: ?name, attrs::member: _?member }])
            )
            .filter_map(
                |(model, name)| match read_long_string(blobs, name, "model_name") {
                    Ok(name) if name == wanted => Some(Ok(model)),
                    Ok(_) => None,
                    Err(error) => Some(Err(error)),
                },
            )
            .collect::<anyhow::Result<BTreeSet<_>>>()?;
            let root = exactly_one(matches, format_args!("model root named {wanted:?}"))?;
            validate_model_name(tribles, blobs, root, wanted)?;
            Ok(root)
        }
        ModelSelector::Source {
            source: wanted_source,
            quantization,
        } => {
            let matches = find!(
                (model: Id, source: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ ?model @
                    attrs::source: ?source,
                    attrs::quantization: quantization,
                    attrs::member: _?member,
                }])
            )
            .filter_map(
                |(model, source)| match read_long_string(blobs, source, "source") {
                    Ok(source) if source == wanted_source => Some(Ok(model)),
                    Ok(_) => None,
                    Err(error) => Some(Err(error)),
                },
            )
            .collect::<anyhow::Result<BTreeSet<_>>>()?;
            let root = exactly_one(
                matches,
                format_args!(
                    "model root with source {wanted_source:?} and quantization {quantization:?}"
                ),
            )?;
            validate_source_coordinates(tribles, blobs, root, wanted_source, quantization)?;
            Ok(root)
        }
    }
}

fn model_members(tribles: &TribleSet, root: Id) -> anyhow::Result<BTreeSet<Id>> {
    let members: BTreeSet<_> = find!(
        (member: Id),
        pattern!(tribles, [{ root @ attrs::member: ?member }])
    )
    .map(|(member,)| member)
    .collect();
    if members.is_empty() {
        bail!("model root {root} has no members");
    }
    Ok(members)
}

/// Build a strict `name -> leaf` index for one model root.
///
/// Every module must have exactly one name and weight edge, every weight entity
/// must carry exactly one readable tensor leaf, and tensor names must be
/// globally unique within the model. Violations are errors rather than
/// iteration-order-dependent `HashMap` overwrites.
///
/// A leaf's element format and shape come out of the leaf itself — the
/// attribute's id names the element and rank, and the blob header names the
/// dims — so there is nothing here that pairs a payload with a shape and hopes.
pub fn index_keymap_for_root(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    root: Id,
) -> anyhow::Result<HashMap<String, Leaf>> {
    let mut map = HashMap::new();
    for member in model_members(tribles, root)? {
        let name_handle = exactly_one(
            find!(
                (name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ member @ attrs::safetensor_path: ?name }])
            )
            .map(|(name,)| name),
            format_args!("safetensor_path on model member {member}"),
        )?;
        let weight = exactly_one(
            find!(
                (weight: Id),
                pattern!(tribles, [{ member @ attrs::weight: ?weight }])
            )
            .map(|(weight,)| weight),
            format_args!("weight edge on model member {member}"),
        )?;
        let handles = crate::leaf::resolve(tribles, blobs, weight)
            .with_context(|| format!("read tensor leaf {weight}"))?
            .ok_or_else(|| anyhow!("tensor leaf {weight} carries no readable tensor"))?;

        let name = read_long_string(blobs, name_handle, "safetensor_path")?;
        if map.insert(name.clone(), handles).is_some() {
            bail!("duplicate tensor name {name:?} under model root {root}");
        }
    }
    Ok(map)
}

impl<R: BlobStoreGet> SelectedModelIndex<R> {
    /// Resolve and strictly index one model from an explicit graph plus its
    /// owning reader.
    ///
    /// This is the unpublished-candidate seam: callers may union staged facts
    /// with an authorized snapshot, validate the resulting view, and only then
    /// publish authority. Storage admission remains entirely with the caller.
    pub fn from_graph(
        facts: &TribleSet,
        reader: R,
        selector: ModelSelector<'_>,
    ) -> anyhow::Result<Self> {
        let root = select_model_root(facts, &reader, selector)
            .context("select model root from explicit graph")?;
        let handles = index_keymap_for_root(facts, &reader, root)
            .with_context(|| format!("index model root {root} from explicit graph"))?;
        Ok(Self {
            root,
            handles,
            reader,
        })
    }

    /// Resolve and strictly index one model from an already-frozen collection.
    ///
    /// Storage opening and admission policy remain with the caller. Ambiguous
    /// selectors, malformed functional fields, mixed data encodings on one
    /// leaf, and duplicate tensor names all fail before the snapshot is
    /// consumed.
    pub fn from_snapshot(
        snapshot: CollectionSnapshot<R>,
        selector: ModelSelector<'_>,
    ) -> anyhow::Result<Self> {
        let (facts, _, reader) = snapshot.into_parts();
        Self::from_graph(&facts, reader, selector)
            .context("select model from native collection snapshot")
    }
}

/// Select one model and materialize its tensor keymap from an already-open
/// graph and blob reader.
pub fn load_keymap_from_graph(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: ModelSelector<'_>,
) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let root = select_model_root(tribles, blobs, selector)?;
    Ok(index_keymap_for_root(tribles, blobs, root)?
        .into_iter()
        .map(|(name, leaf)| (name, leaf.to_f32_shape()))
        .collect())
}

fn tokenizer_roots(tribles: &TribleSet) -> BTreeSet<Id> {
    crate::tokenizer::find_tokenizers(tribles).collect()
}

fn tokenizer_name(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    root: Id,
) -> anyhow::Result<String> {
    let handle = exactly_one(
        find!(
            (name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!(tribles, [{ root @ crate::tokenizer::attrs::model_name: ?name }])
        )
        .map(|(name,)| name),
        format_args!("model_name field on tokenizer root {root}"),
    )?;
    read_long_string(blobs, handle, "tokenizer model_name")
}

/// Resolve exactly one tokenizer root according to `selector`.
pub fn select_tokenizer_root(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: TokenizerSelector<'_>,
) -> anyhow::Result<Id> {
    match selector {
        TokenizerSelector::Only => {
            let root = exactly_one(tokenizer_roots(tribles), "tokenizer root in graph")?;
            tokenizer_name(tribles, blobs, root)?;
            Ok(root)
        }
        TokenizerSelector::Root(root) => {
            if tokenizer_roots(tribles).contains(&root) {
                tokenizer_name(tribles, blobs, root)?;
                Ok(root)
            } else {
                bail!("tokenizer root {root} is absent or lacks a tokenizer kind/name")
            }
        }
        TokenizerSelector::Name(wanted) => {
            let roots = tokenizer_roots(tribles);
            let matches = find!(
                (tokenizer: Id, name: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
                pattern!(tribles, [{ ?tokenizer @ crate::tokenizer::attrs::model_name: ?name }])
            )
            .filter(|(tokenizer, _)| roots.contains(tokenizer))
            .filter_map(|(tokenizer, name)| {
                match read_long_string(blobs, name, "tokenizer model_name") {
                    Ok(name) if name == wanted => Some(Ok(tokenizer)),
                    Ok(_) => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .collect::<anyhow::Result<BTreeSet<_>>>()?;
            let root = exactly_one(matches, format_args!("tokenizer root named {wanted:?}"))?;
            let name = tokenizer_name(tribles, blobs, root)?;
            if name != wanted {
                bail!("tokenizer root {root} name changed while selecting");
            }
            Ok(root)
        }
    }
}

/// Select and construct one HuggingFace tokenizer from an already-open graph.
#[cfg(feature = "tokenizer")]
pub fn load_tokenizer_from_graph(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    selector: TokenizerSelector<'_>,
) -> anyhow::Result<tokenizers::Tokenizer> {
    let root = select_tokenizer_root(tribles, blobs, selector)?;
    crate::tokenizer::build_tokenizer(tribles, blobs, root)
        .map_err(|error| anyhow!("build tokenizer {root}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace::core::blob::MemoryBlobStore;
    #[cfg(feature = "tokenizer")]
    use triblespace::core::metadata;

    struct ModelFixture {
        root: Id,
        members: Vec<Id>,
    }

    /// How a fixture writes its leaves. Both forms are in the wild: the typed
    /// tensor every importer writes now, and the two-blob pair every existing
    /// model pile holds. Selection has to work on both, so the tests say which.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum LeafForm {
        Typed,
        TwoBlob,
    }

    fn add_leaf(
        facts: &mut TribleSet,
        blobs: &mut MemoryBlobStore,
        form: LeafForm,
        value: f32,
    ) -> Id {
        let leaf = match form {
            LeafForm::Typed => crate::format::put_raw(blobs, &[value], &[1]).unwrap(),
            LeafForm::TwoBlob => {
                let data = blobs.put::<crate::format::F32Array, _>(vec![value]).unwrap();
                let shape = blobs.put::<crate::format::U64Array, _>(vec![1]).unwrap();
                entity! { _ @ attrs::data: data, attrs::shape: shape }
            }
        };
        let leaf_id = leaf.root().unwrap();
        *facts += leaf.into_facts();
        leaf_id
    }

    fn add_model(
        facts: &mut TribleSet,
        blobs: &mut MemoryBlobStore,
        name: &str,
        source: &str,
        quantization: &str,
        tensors: &[(&str, f32)],
    ) -> ModelFixture {
        add_model_as(
            facts,
            blobs,
            LeafForm::Typed,
            name,
            source,
            quantization,
            tensors,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn add_model_as(
        facts: &mut TribleSet,
        blobs: &mut MemoryBlobStore,
        form: LeafForm,
        name: &str,
        source: &str,
        quantization: &str,
        tensors: &[(&str, f32)],
    ) -> ModelFixture {
        let mut members = Vec::new();
        for &(tensor_name, value) in tensors {
            let leaf_id = add_leaf(facts, blobs, form, value);

            let name = blobs
                .put::<blobencodings::UTF8String, _>(tensor_name.to_string())
                .unwrap();
            let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: &leaf_id };
            let member_id = member.root().unwrap();
            *facts += member.into_facts();
            members.push(member_id);
        }

        let name = blobs
            .put::<blobencodings::UTF8String, _>(name.to_string())
            .unwrap();
        let source = blobs
            .put::<blobencodings::UTF8String, _>(source.to_string())
            .unwrap();
        let model = entity! { _ @
            attrs::model_name: name,
            attrs::source: source,
            attrs::quantization: quantization,
            attrs::member*: members.iter(),
        };
        let root = model.root().unwrap();
        *facts += model.into_facts();
        ModelFixture { root, members }
    }

    #[test]
    fn model_selectors_are_exact_in_a_consolidated_graph() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let alpha = add_model(
            &mut facts,
            &mut blobs,
            "alpha",
            "org/alpha",
            "native",
            &[("alpha.weight", 1.0)],
        );
        let beta = add_model(
            &mut facts,
            &mut blobs,
            "beta",
            "org/beta",
            "fp4",
            &[("beta.weight", 2.0)],
        );
        let reader = BlobStore::reader(&mut blobs).unwrap();

        assert!(select_model_root(&facts, &reader, ModelSelector::Only).is_err());
        assert_eq!(
            select_model_root(&facts, &reader, ModelSelector::Name("alpha")).unwrap(),
            alpha.root
        );
        assert_eq!(
            select_model_root(
                &facts,
                &reader,
                ModelSelector::Source {
                    source: "org/beta",
                    quantization: "fp4",
                },
            )
            .unwrap(),
            beta.root
        );
        assert_eq!(
            select_model_root(&facts, &reader, ModelSelector::Root(beta.root)).unwrap(),
            beta.root
        );

        let keymap = load_keymap_from_graph(
            &facts,
            &reader,
            ModelSelector::Source {
                source: "org/alpha",
                quantization: "native",
            },
        )
        .unwrap();
        assert_eq!(keymap["alpha.weight"], (vec![1.0], vec![1]));
        assert!(select_model_root(&facts, &reader, ModelSelector::Name("missing")).is_err());
    }

    #[test]
    fn duplicate_model_names_and_functional_fields_are_errors() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let first = add_model(
            &mut facts,
            &mut blobs,
            "same",
            "org/one",
            "native",
            &[("weight", 1.0)],
        );
        add_model(
            &mut facts,
            &mut blobs,
            "same",
            "org/two",
            "native",
            &[("weight", 2.0)],
        );
        let reader = BlobStore::reader(&mut blobs).unwrap();
        assert!(select_model_root(&facts, &reader, ModelSelector::Name("same")).is_err());

        let extra_leaf_id = add_leaf(&mut facts, &mut blobs, LeafForm::Typed, 3.0);
        let member = first.members[0];
        facts += entity! { ExclusiveId::force_ref(&member) @ attrs::weight: &extra_leaf_id }
            .into_facts();
        let reader = BlobStore::reader(&mut blobs).unwrap();
        let error = match index_keymap_for_root(&facts, &reader, first.root) {
            Ok(_) => panic!("duplicate weight edge was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("ambiguous weight edge"), "{error}");
    }

    #[test]
    fn duplicate_tensor_names_are_errors() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let model = add_model(
            &mut facts,
            &mut blobs,
            "duplicate",
            "org/duplicate",
            "native",
            &[("weight", 1.0), ("weight", 2.0)],
        );
        let reader = BlobStore::reader(&mut blobs).unwrap();
        let error = match index_keymap_for_root(&facts, &reader, model.root) {
            Ok(_) => panic!("duplicate tensor name was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("duplicate tensor name"), "{error}");
    }

    #[test]
    fn name_selector_rejects_a_root_with_multiple_names() {
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let model = add_model(
            &mut facts,
            &mut blobs,
            "primary",
            "org/model",
            "native",
            &[("weight", 1.0)],
        );
        let alias = blobs
            .put::<blobencodings::UTF8String, _>("alias".to_string())
            .unwrap();
        facts +=
            entity! { ExclusiveId::force_ref(&model.root) @ attrs::model_name: alias }.into_facts();
        let reader = BlobStore::reader(&mut blobs).unwrap();

        let error = select_model_root(&facts, &reader, ModelSelector::Name("primary"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("ambiguous model_name field"), "{error}");
    }

    /// The model piles that exist today hold the two-blob form, and selection
    /// must keep reading them. Same graph, same keymap, one leaf form apart.
    #[test]
    fn a_two_blob_pile_indexes_the_same_as_a_typed_one() {
        let mut typed_facts = TribleSet::new();
        let mut typed_blobs = MemoryBlobStore::new();
        let typed = add_model_as(
            &mut typed_facts,
            &mut typed_blobs,
            LeafForm::Typed,
            "m",
            "org/m",
            "native",
            &[("a.weight", 1.5), ("b.weight", -2.5)],
        );
        let typed_reader = BlobStore::reader(&mut typed_blobs).unwrap();

        let mut old_facts = TribleSet::new();
        let mut old_blobs = MemoryBlobStore::new();
        let old = add_model_as(
            &mut old_facts,
            &mut old_blobs,
            LeafForm::TwoBlob,
            "m",
            "org/m",
            "native",
            &[("a.weight", 1.5), ("b.weight", -2.5)],
        );
        let old_reader = BlobStore::reader(&mut old_blobs).unwrap();

        let typed_map = index_keymap_for_root(&typed_facts, &typed_reader, typed.root).unwrap();
        let old_map = index_keymap_for_root(&old_facts, &old_reader, old.root).unwrap();
        assert_eq!(typed_map.len(), 2);
        assert_eq!(old_map.len(), 2);
        for name in ["a.weight", "b.weight"] {
            assert_eq!(typed_map[name].to_f32_shape(), old_map[name].to_f32_shape());
            assert_eq!(typed_map[name].elem(), old_map[name].elem());
        }
        // The roots differ, and that is correct: the two forms are different
        // bytes, so they are different content addresses.
        assert_ne!(typed.root, old.root);
    }

    #[cfg(feature = "tokenizer")]
    const WORDPIECE: &str = r###"{
      "added_tokens": [],
      "normalizer": {"type": "BertNormalizer", "clean_text": true,
                     "handle_chinese_chars": true, "strip_accents": null,
                     "lowercase": true},
      "pre_tokenizer": {"type": "BertPreTokenizer"},
      "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "hello": 1}}
    }"###;

    #[cfg(feature = "tokenizer")]
    #[test]
    fn tokenizer_name_selection_disambiguates_consolidated_graphs() {
        let mut blobs = MemoryBlobStore::new();
        let alpha =
            crate::tokenizer::save_tokenizer_json(WORDPIECE.as_bytes(), "org/alpha", &mut blobs)
                .unwrap();
        let alpha_root = alpha.root().unwrap();
        let beta =
            crate::tokenizer::save_tokenizer_json(WORDPIECE.as_bytes(), "org/beta", &mut blobs)
                .unwrap();
        let beta_root = beta.root().unwrap();
        let mut facts = alpha.into_facts();
        facts += beta.into_facts();
        let reader = BlobStore::reader(&mut blobs).unwrap();

        assert!(select_tokenizer_root(&facts, &reader, TokenizerSelector::Only).is_err());
        assert_eq!(
            select_tokenizer_root(&facts, &reader, TokenizerSelector::Name("org/alpha")).unwrap(),
            alpha_root
        );
        assert_eq!(
            select_tokenizer_root(&facts, &reader, TokenizerSelector::Root(beta_root)).unwrap(),
            beta_root
        );
        let tokenizer =
            load_tokenizer_from_graph(&facts, &reader, TokenizerSelector::Name("org/beta"))
                .unwrap();
        assert_eq!(tokenizer.token_to_id("hello"), Some(1));
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn duplicate_tokenizer_functional_edge_is_an_error() {
        let mut blobs = MemoryBlobStore::new();
        let tokenizer = crate::tokenizer::save_tokenizer_json(
            WORDPIECE.as_bytes(),
            "org/tokenizer",
            &mut blobs,
        )
        .unwrap();
        let root = tokenizer.root().unwrap();
        let mut facts = tokenizer.into_facts();
        let other = entity! { _ @ metadata::tag: metadata::KIND_MULTI };
        let other_root = other.root().unwrap();
        facts += other.into_facts();
        facts += entity! { ExclusiveId::force_ref(&root) @
            crate::tokenizer::attrs::normalizer: &other_root
        }
        .into_facts();
        let reader = BlobStore::reader(&mut blobs).unwrap();

        let error = load_tokenizer_from_graph(&facts, &reader, TokenizerSelector::Only)
            .unwrap_err()
            .to_string();
        assert!(error.contains("more than one attrs::normalizer"), "{error}");
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn duplicate_vocab_entry_functional_field_is_an_error() {
        let mut blobs = MemoryBlobStore::new();
        let tokenizer = crate::tokenizer::save_tokenizer_json(
            WORDPIECE.as_bytes(),
            "org/tokenizer",
            &mut blobs,
        )
        .unwrap();
        let root = tokenizer.root().unwrap();
        let mut facts = tokenizer.into_facts();
        let entry = find!(
            (entry: Id),
            pattern!(&facts, [{ root @ crate::tokenizer::attrs::vocab: ?entry }])
        )
        .next()
        .unwrap()
        .0;
        facts += entity! { ExclusiveId::force_ref(&entry) @
            crate::tokenizer::attrs::token_id: 99_u64
        }
        .into_facts();
        let reader = BlobStore::reader(&mut blobs).unwrap();

        let error = load_tokenizer_from_graph(&facts, &reader, TokenizerSelector::Only)
            .unwrap_err()
            .to_string();
        assert!(error.contains("more than one token_id"), "{error}");
    }
}
