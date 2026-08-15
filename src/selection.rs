//! Deterministic model and tokenizer selection over an already-open graph.
//!
//! A consolidated collection can contain several models and tokenizers. The
//! old pile wrappers predated that: they selected the first matching root or
//! extended a `HashMap`, making iteration order decide ambiguous data. This
//! module keeps storage out of the decision. Callers pass a materialized
//! [`TribleSet`], its blob reader, and an explicit selector; every selector and
//! every functional model field has exact-cardinality semantics.

use crate::format::{F32Array, U64Array, attrs};
use crate::ingest::LeafHandles;
use anyhow::{Context, anyhow, bail};
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
    handles: HashMap<String, LeafHandles>,
    reader: R,
}

impl<R> SelectedModelIndex<R> {
    /// Exact content-addressed model root selected from the frozen graph.
    pub fn root(&self) -> Id {
        self.root
    }

    /// Strict `tensor name -> leaf handles` index for the selected root.
    pub fn handles(&self) -> &HashMap<String, LeafHandles> {
        &self.handles
    }

    /// Reader that owns the attachment snapshot named by [`Self::handles`].
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// Consume the selection into its root, handle index, and owned reader.
    pub fn into_parts(self) -> (Id, HashMap<String, LeafHandles>, R) {
        (self.root, self.handles, self.reader)
    }
}

fn read_long_string(
    blobs: &impl BlobStoreGet,
    handle: Inline<inlineencodings::Handle<blobencodings::LongString>>,
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
            (source: Inline<inlineencodings::Handle<blobencodings::LongString>>),
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
            (name: Inline<inlineencodings::Handle<blobencodings::LongString>>),
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
                (model: Id, name: Inline<inlineencodings::Handle<blobencodings::LongString>>),
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
                (model: Id, source: Inline<inlineencodings::Handle<blobencodings::LongString>>),
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

/// Build a strict `name -> leaf handles` index for one model root.
///
/// Every module must have exactly one name and weight edge; every tensor leaf
/// exactly one shape and exactly one of `data`/`data_f16`; tensor names must be
/// globally unique within the model. Violations are errors rather than
/// iteration-order-dependent `HashMap` overwrites.
pub fn index_keymap_for_root(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    root: Id,
) -> anyhow::Result<HashMap<String, LeafHandles>> {
    let mut map = HashMap::new();
    for member in model_members(tribles, root)? {
        let name_handle = exactly_one(
            find!(
                (name: Inline<inlineencodings::Handle<blobencodings::LongString>>),
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
        let shape = exactly_one(
            find!(
                (shape: Inline<inlineencodings::Handle<U64Array>>),
                pattern!(tribles, [{ weight @ attrs::shape: ?shape }])
            )
            .map(|(shape,)| shape),
            format_args!("shape on tensor leaf {weight}"),
        )?;

        let f32_data: Vec<_> = find!(
            (data: Inline<inlineencodings::Handle<F32Array>>),
            pattern!(tribles, [{ weight @ attrs::data: ?data }])
        )
        .map(|(data,)| data)
        .collect();
        let f16_data: Vec<_> = find!(
            (data: Inline<inlineencodings::Handle<crate::f16enc::F16Array>>),
            pattern!(tribles, [{ weight @ attrs::data_f16: ?data }])
        )
        .map(|(data,)| data)
        .collect();
        let handles = match (f32_data.as_slice(), f16_data.as_slice()) {
            ([data], []) => LeafHandles::F32(*data, shape),
            ([], [data]) => LeafHandles::F16(*data, shape),
            ([], []) => bail!("tensor leaf {weight} has neither data nor data_f16"),
            _ => bail!(
                "tensor leaf {weight} must have exactly one of data/data_f16 (found {} f32, {} f16)",
                f32_data.len(),
                f16_data.len()
            ),
        };

        let name = read_long_string(blobs, name_handle, "safetensor_path")?;
        if map.insert(name.clone(), handles).is_some() {
            bail!("duplicate tensor name {name:?} under model root {root}");
        }
    }
    Ok(map)
}

impl<R: BlobStoreGet> SelectedModelIndex<R> {
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
        let root = select_model_root(snapshot.facts(), snapshot.reader(), selector)
            .context("select model root from native collection snapshot")?;
        let handles = index_keymap_for_root(snapshot.facts(), snapshot.reader(), root)
            .with_context(|| format!("index model root {root} from native collection snapshot"))?;
        let (_, _, reader) = snapshot.into_parts();
        Ok(Self {
            root,
            handles,
            reader,
        })
    }
}

fn read_shape(
    blobs: &impl BlobStoreGet,
    handle: Inline<inlineencodings::Handle<U64Array>>,
) -> anyhow::Result<Vec<usize>> {
    let bytes: anybytes::Bytes = blobs
        .get(handle)
        .map_err(|error| anyhow!("read shape blob: {error}"))?;
    let values = bytes.view::<[u64]>().context("decode shape blob")?;
    Ok(values.iter().map(|&value| value as usize).collect())
}

fn materialize_leaf(
    blobs: &impl BlobStoreGet,
    handles: LeafHandles,
) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    match handles {
        LeafHandles::F32(data, shape) => {
            let bytes: anybytes::Bytes = blobs
                .get(data)
                .map_err(|error| anyhow!("read f32 tensor blob: {error}"))?;
            let values = bytes.view::<[f32]>().context("decode f32 tensor blob")?;
            Ok((values.to_vec(), read_shape(blobs, shape)?))
        }
        LeafHandles::F16(data, shape) => {
            let bytes: anybytes::Bytes = blobs
                .get(data)
                .map_err(|error| anyhow!("read f16 tensor blob: {error}"))?;
            let values = bytes
                .view::<[half::f16]>()
                .context("decode f16 tensor blob")?;
            Ok((
                values.iter().map(|value| value.to_f32()).collect(),
                read_shape(blobs, shape)?,
            ))
        }
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
    index_keymap_for_root(tribles, blobs, root)?
        .into_iter()
        .map(|(name, handles)| Ok((name, materialize_leaf(blobs, handles)?)))
        .collect()
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
            (name: Inline<inlineencodings::Handle<blobencodings::LongString>>),
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
                (tokenizer: Id, name: Inline<inlineencodings::Handle<blobencodings::LongString>>),
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
    use triblespace::core::metadata;

    struct ModelFixture {
        root: Id,
        members: Vec<Id>,
    }

    fn add_model(
        facts: &mut TribleSet,
        blobs: &mut MemoryBlobStore,
        name: &str,
        source: &str,
        quantization: &str,
        tensors: &[(&str, f32)],
    ) -> ModelFixture {
        let mut members = Vec::new();
        for &(tensor_name, value) in tensors {
            let data = blobs.put::<F32Array, _>(vec![value]).unwrap();
            let shape = blobs.put::<U64Array, _>(vec![1]).unwrap();
            let leaf = entity! { _ @ attrs::data: data, attrs::shape: shape };
            let leaf_id = leaf.root().unwrap();
            *facts += leaf.into_facts();

            let name = blobs
                .put::<blobencodings::LongString, _>(tensor_name.to_string())
                .unwrap();
            let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: &leaf_id };
            let member_id = member.root().unwrap();
            *facts += member.into_facts();
            members.push(member_id);
        }

        let name = blobs
            .put::<blobencodings::LongString, _>(name.to_string())
            .unwrap();
        let source = blobs
            .put::<blobencodings::LongString, _>(source.to_string())
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

        let extra_data = blobs.put::<F32Array, _>(vec![3.0]).unwrap();
        let extra_shape = blobs.put::<U64Array, _>(vec![1]).unwrap();
        let extra_leaf = entity! { _ @ attrs::data: extra_data, attrs::shape: extra_shape };
        let extra_leaf_id = extra_leaf.root().unwrap();
        facts += extra_leaf.into_facts();
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
            .put::<blobencodings::LongString, _>("alias".to_string())
            .unwrap();
        facts +=
            entity! { ExclusiveId::force_ref(&model.root) @ attrs::model_name: alias }.into_facts();
        let reader = BlobStore::reader(&mut blobs).unwrap();

        let error = select_model_root(&facts, &reader, ModelSelector::Name("primary"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("ambiguous model_name field"), "{error}");
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
