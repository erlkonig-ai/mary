//! Persist Gemma 4 weights into a REAL on-disk TribleSpace pile and load them
//! back from JUST the pile file — no safetensors needed at load time. This is the
//! "shell-is-physics endpoint": the weights live as content-addressed tribles on
//! disk, and a fresh process reconstructs the model from the pile alone.
//!
//! `persist_safetensors_to_pile` ingests every `*.safetensors` shard into the pile
//! (each tensor a content-addressed f32 leaf via [`crate::ingest::save_safetensors`]),
//! committing the model-graph facts on the `main` branch. The weight *blobs* are
//! written straight into the pile's storage (the `Pile` is itself a
//! `BlobStorePut`), so there is no giant in-memory buffer — only the small fact
//! set rides through the workspace commit. It is fully model-agnostic — the
//! `gemma4` lineage in the old name was history; CLIP/SigLIP use the same path.
//!
//! `load_keymap_from_pile` opens the pile fresh, resolves `main`, checks
//! out the full history into a `TribleSet`, finds every model entity (the ones
//! carrying `attrs::model_name`), and materializes the union keymap via
//! [`crate::ingest::load_keymap`]. The f32 blobs store weights exactly, so the
//! round-trip is lossless.

use crate::ingest::load_keymap;
#[cfg(feature = "import")]
use crate::ingest::LeafDtype;
#[cfg(feature = "import")]
use crate::nn::weight_loader::read_safetensors_file;
#[cfg(any(
    feature = "import",
    feature = "qwen3tts",
    feature = "tokenizer",
    feature = "gemma"
))]
use anyhow::Context;
use ed25519_dalek::SigningKey;
#[cfg(feature = "import")]
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::Path;
#[cfg(feature = "import")]
use triblespace::core::collection::CollectionCommit;
use triblespace::prelude::*;

/// Resolve the sorted `*.safetensors` shards in a model directory.
#[cfg(feature = "import")]
fn shard_paths(dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    shards.sort();
    if shards.is_empty() {
        anyhow::bail!("no .safetensors shards in {dir:?}");
    }
    Ok(shards)
}

/// Persist every safetensors shard under `safetensors_dir` into a real on-disk
/// pile at `pile_path` (see [`persist_safetensors_files_to_pile`]). Each model
/// entity is named by its shard's file name.
#[cfg(feature = "import")]
pub fn persist_safetensors_to_pile(
    safetensors_dir: &Path,
    pile_path: &Path,
    dtype: LeafDtype,
) -> anyhow::Result<()> {
    let shards = shard_paths(safetensors_dir)?;
    let files: Vec<(std::path::PathBuf, String)> = shards
        .into_iter()
        .map(|p| {
            let name = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string();
            (p, name)
        })
        .collect();
    persist_safetensors_files_to_pile(&files, pile_path, dtype)
}

/// Persist explicit safetensors files into a real on-disk pile at `pile_path`,
/// each under a caller-chosen model-entity name (component-prefixed names like
/// `"text_encoder/model-00001.safetensors"` let a loader materialize one
/// component at a time via [`load_keymap_from_pile_prefixed`]). Creates the
/// pile file if it does not exist. Each tensor becomes a content-addressed
/// leaf; the model-graph facts are committed on the `main` branch. After this
/// returns, the pile file holds the full weights — no safetensors are needed
/// to load the model again.
#[cfg(feature = "import")]
pub fn persist_safetensors_files_to_pile(
    files: &[(std::path::PathBuf, String)],
    pile_path: &Path,
    dtype: LeafDtype,
) -> anyhow::Result<()> {
    persist_files_to_pile(files, pile_path, dtype)
}

/// Ingest one model directory and build its unpublished rooted fragment.
///
/// Tensor and label blobs are written as content-addressed exhaust, but no
/// [`CollectionCommit`] is published. This is the commit-last seam for callers
/// that must validate a complete candidate against existing collection state
/// before granting it authority. The caller owns the open pile and eventual
/// durability boundary; this function never creates, opens, flushes, or closes
/// storage.
///
/// `source` is the model's canonical label — the HF id it was imported from, or
/// a `--name` for a local-dir import. The root id is content-derived from only
/// its weight members, so byte-identical weights converge independently of
/// source container or provenance. For a multi-shard model, the one root
/// composes every shard's tensor members as an order-independent set; source,
/// quantization, and shard names are queryable non-core coordinates on it.
///
/// `quantization` tags the weight format ("native" for the faithful import).
#[cfg(feature = "import")]
pub fn ingest_model_fragment(
    pile: &mut Pile,
    model_dir: &Path,
    dtype: LeafDtype,
    source: &str,
    quantization: &str,
) -> anyhow::Result<Fragment> {
    // Detect the container the directory actually ships (safetensors / gguf /
    // pytorch pickle) and gather its weight files. Every format funnels into the
    // SAME content-addressed member path below, so the model-root id stays the
    // pure hash of the f32 tensors regardless of source format.
    let (fmt, weight_files) = crate::formats::detect_format(model_dir)?;
    let files: Vec<(std::path::PathBuf, String)> = weight_files
        .into_iter()
        .map(|p| {
            let name = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string();
            (p, name)
        })
        .collect();
    eprintln!(
        "[persist] detected {fmt:?} — {} weight file(s)",
        files.len()
    );

    // Validate the caller's observed prefix before writing any imported blob.
    // A corrupt tail must fail loud: repair remains an explicit operator act.
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "model pile failed to load ({e:?}); refusing to auto-truncate — \
             if the tail is a genuinely torn write, amputate explicitly with \
             `trible pile amputate`"
        )
    })?;

    // Ingest EVERY shard's weight blobs straight into the pile storage (no
    // in-memory carryover), gathering ALL shards' tensor members under ONE root
    // whose id is content-derived from (model_id, quantization, members). Each
    // shard's file name becomes non-core provenance on that root.
    let mut members: Vec<Id> = Vec::new();
    let mut facts = TribleSet::new();
    let mut provenance: Vec<String> = Vec::new();
    let mut tensor_names = BTreeSet::new();
    for (path, name) in &files {
        let (mut shard_members, shard_facts) = match fmt {
            crate::formats::WeightFormat::Safetensors => {
                let bytes = std::fs::read(path)
                    .with_context(|| format!("read safetensors file {path:?}"))?;
                eprintln!(
                    "[persist] ingesting {name} ({} bytes, safetensors)...",
                    bytes.len()
                );
                let tensors = safetensors::SafeTensors::deserialize(&bytes)
                    .with_context(|| format!("decode safetensors file {path:?}"))?;
                for tensor_name in tensors.names() {
                    use safetensors::Dtype;
                    let tensor = tensors.tensor(tensor_name)?;
                    if matches!(
                        tensor.dtype(),
                        Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16
                    ) {
                        anyhow::ensure!(
                            tensor_names.insert(tensor_name.to_owned()),
                            "duplicate tensor name {tensor_name:?} across model weight files"
                        );
                    }
                }
                crate::ingest::ingest_members(&bytes, pile, dtype, |_| true)
                    .map_err(|e| anyhow::anyhow!("ingest {path:?}: {e}"))?
            }
            crate::formats::WeightFormat::Gguf | crate::formats::WeightFormat::Pickle => {
                let tensors = crate::formats::extract_tensors(fmt, path)
                    .map_err(|e| anyhow::anyhow!("extract {path:?}: {e}"))?;
                eprintln!(
                    "[persist] ingesting {name} ({} tensors, {fmt:?})...",
                    tensors.len()
                );
                for (tensor_name, _, _) in &tensors {
                    anyhow::ensure!(
                        tensor_names.insert(tensor_name.clone()),
                        "duplicate tensor name {tensor_name:?} across model weight files"
                    );
                }
                crate::ingest::ingest_tensors(tensors.into_iter(), pile, dtype)
                    .map_err(|e| anyhow::anyhow!("ingest {path:?}: {e}"))?
            }
        };
        members.append(&mut shard_members);
        facts += shard_facts;
        provenance.push(name.clone());
    }
    crate::ingest::build_model_root(pile, source, quantization, members, facts, &provenance)
        .map_err(|e| anyhow::anyhow!("build model root: {e}"))
}

/// Import one model directory into Mary's native append-only model collection.
///
/// This is the library seam behind `mary import`. It stages the complete model
/// with [`ingest_model_fragment`] and immediately publishes that fragment under
/// the caller's durable signing identity. Callers with additional domain gates
/// should stage, validate, and call [`crate::model_collection::publish_model_fragment`]
/// themselves so publication remains the final authority transition.
#[cfg(feature = "import")]
pub fn import_model_to_collection(
    pile: &mut Pile,
    signing_key: &SigningKey,
    model_dir: &Path,
    dtype: LeafDtype,
    source: &str,
    quantization: &str,
) -> anyhow::Result<(Id, CollectionCommit)> {
    let root = ingest_model_fragment(pile, model_dir, dtype, source, quantization)?;
    let root_id = root.root().expect("model root id");
    let team = crate::model_collection::model_graph_team_or_own(pile, signing_key)?;
    let commit = crate::model_collection::publish_model_fragment(pile, team, signing_key, root)
        .map_err(|error| anyhow::anyhow!("publish model collection commit: {error}"))?;
    Ok((root_id, commit))
}

/// Derive one complete f16 model root from an already-selected exact f32 root
/// and publish it into the caller's same open native model collection.
///
/// The selected snapshot owns the immutable source reader. Tensors are visited
/// in deterministic name order and each f32 payload is materialized, converted,
/// and persisted before the next is read, so peak host weight memory is one
/// tensor rather than the model. The caller supplies both the target coordinate
/// and signing identity and retains the pile/durability boundary.
#[cfg(feature = "import")]
pub fn derive_selected_f16_to_collection<R: BlobStoreGet>(
    pile: &mut Pile,
    signing_key: &SigningKey,
    selected: crate::selection::SelectedModelIndex<R>,
    source: &str,
    quantization: &str,
) -> anyhow::Result<(Id, CollectionCommit, usize, usize)> {
    let (_, mut index, reader) = selected.into_parts();
    let _ = &reader;
    anyhow::ensure!(!index.is_empty(), "cannot derive an empty f16 model root");
    for (name, leaf) in &index {
        anyhow::ensure!(
            leaf.elem() == crate::leaf::Elem::F32,
            "{name}: f16 derivation requires an exact f32 source root"
        );
    }

    let mut names: Vec<_> = index.keys().cloned().collect();
    names.sort_unstable();
    let mut members = Vec::with_capacity(names.len());
    let mut facts = TribleSet::new();
    let mut elements = 0;
    for (ordinal, name) in names.into_iter().enumerate() {
        let leaf = index.remove(&name).expect("name collected from index");
        // The leaf states its own width and shape, so there is nothing to
        // cross-check here: `elem()` already decided the branch above.
        let data = leaf.to_f32();
        let shape = leaf.shape();
        elements += data.len();
        let (mut tensor_members, tensor_facts) = crate::ingest::ingest_tensors(
            std::iter::once((name, data, shape)),
            pile,
            crate::ingest::LeafDtype::F16,
        )
        .map_err(|error| anyhow::anyhow!("derive f16 tensor: {error}"))?;
        members.append(&mut tensor_members);
        facts += tensor_facts;
        if (ordinal + 1) % 100 == 0 {
            eprintln!("[persist] {} tensors derived → f16 ...", ordinal + 1);
        }
    }
    let tensor_count = members.len();
    let root = crate::ingest::build_model_root(pile, source, quantization, members, facts, &[])
        .map_err(|error| anyhow::anyhow!("build derived f16 model root: {error}"))?;
    let root_id = root.root().expect("derived f16 model root");
    let team = crate::model_collection::model_graph_team_or_own(pile, signing_key)?;
    let commit = crate::model_collection::publish_model_fragment(pile, team, signing_key, root)
        .map_err(|error| anyhow::anyhow!("publish derived f16 model root: {error}"))?;
    Ok((root_id, commit, tensor_count, elements))
}

/// Import one safetensors file's selected float tensors into Mary's native
/// append-only model collection.
///
/// This is the component-sized counterpart to [`import_model_to_collection`]:
/// `keep` chooses tensors by their safetensors key, while `source` and
/// `quantization` label the resulting content-derived root. The file name is
/// retained as non-core provenance. An empty selection is rejected rather than
/// publishing the otherwise-valid empty-set root.
///
/// The caller owns both the already-open pile and the durable signing identity,
/// and chooses the eventual durability boundary. This function never opens,
/// flushes, or closes storage and never creates or advances a Repository branch.
#[cfg(feature = "import")]
pub fn import_safetensors_file_filtered_to_collection(
    pile: &mut Pile,
    signing_key: &SigningKey,
    file: &Path,
    dtype: LeafDtype,
    source: &str,
    quantization: &str,
    keep: impl Fn(&str) -> bool,
) -> anyhow::Result<(Id, CollectionCommit)> {
    let root =
        ingest_safetensors_file_filtered_fragment(pile, file, dtype, source, quantization, keep)?;
    let root_id = root.root().expect("model root id");
    let team = crate::model_collection::model_graph_team_or_own(pile, signing_key)?;
    let commit = crate::model_collection::publish_model_fragment(pile, team, signing_key, root)
        .map_err(|error| anyhow::anyhow!("publish model collection commit: {error}"))?;
    Ok((root_id, commit))
}

/// Ingest one explicit safetensors file into an unpublished native model root.
///
/// This is the commit-last counterpart to
/// [`import_safetensors_file_filtered_to_collection`]. The explicit path is
/// intentional: callers importing a known checkpoint file do not accidentally
/// absorb adapters or unrelated safetensors found beside it. Tensor blobs are
/// written as inert content-addressed exhaust, while the returned fragment
/// remains unpublished until its caller has completed every domain gate.
#[cfg(feature = "import")]
pub fn ingest_safetensors_file_filtered_fragment(
    pile: &mut Pile,
    file: &Path,
    dtype: LeafDtype,
    source: &str,
    quantization: &str,
    keep: impl Fn(&str) -> bool,
) -> anyhow::Result<Fragment> {
    ingest_weight_file_filtered_fragment(
        pile,
        file,
        crate::formats::WeightFormat::Safetensors,
        dtype,
        source,
        quantization,
        keep,
    )
}

/// Ingest one explicit supported weight file into an unpublished model root.
///
/// Safetensors and PyTorch-pickle inputs converge through the same canonical
/// `(name, f32 bytes, shape)` member representation. The caller supplies the
/// detected format explicitly, so a signed cohort never changes decoder merely
/// because another artifact later appears beside the chosen file.
#[cfg(feature = "import")]
pub fn ingest_weight_file_filtered_fragment(
    pile: &mut Pile,
    file: &Path,
    format: crate::formats::WeightFormat,
    dtype: LeafDtype,
    source: &str,
    quantization: &str,
    keep: impl Fn(&str) -> bool,
) -> anyhow::Result<Fragment> {
    // Validate the caller's observed prefix before writing imported blobs. A
    // corrupt tail must remain an explicit operator repair, never an importer
    // side effect.
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "model pile failed to load ({e:?}); refusing to auto-truncate — \
             if the tail is a genuinely torn write, amputate explicitly with \
             `trible pile amputate`"
        )
    })?;

    let (members, facts) = match format {
        crate::formats::WeightFormat::Safetensors => {
            let bytes =
                std::fs::read(file).with_context(|| format!("read safetensors file {file:?}"))?;
            crate::ingest::ingest_members(&bytes, pile, dtype, keep)
                .map_err(|error| anyhow::anyhow!("ingest {file:?}: {error}"))?
        }
        crate::formats::WeightFormat::Gguf | crate::formats::WeightFormat::Pickle => {
            let tensors = crate::formats::extract_tensors(format, file)
                .with_context(|| format!("extract {format:?} tensors from {file:?}"))?;
            crate::ingest::ingest_tensors(
                tensors
                    .into_iter()
                    .filter(|(name, _, _)| keep(name.as_str())),
                pile,
                dtype,
            )
            .map_err(|error| anyhow::anyhow!("ingest {file:?}: {error}"))?
        }
    };
    anyhow::ensure!(
        !members.is_empty(),
        "filtered {format:?} import selected no supported float tensors from {file:?}"
    );

    let provenance = file
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.weights")
        .to_owned();
    let root =
        crate::ingest::build_model_root(pile, source, quantization, members, facts, &[provenance])
            .map_err(|e| anyhow::anyhow!("build model root: {e}"))?;
    Ok(root)
}

#[cfg(all(test, feature = "import"))]
mod filtered_native_import_tests {
    use super::*;
    use crate::selection::ModelSelector;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TempFixture {
        dir: std::path::PathBuf,
    }

    impl TempFixture {
        fn new() -> Self {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "mary-filtered-native-import-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&dir).unwrap();
            Self { dir }
        }
    }

    impl Drop for TempFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn pickle_binunicode(out: &mut Vec<u8>, value: &str) {
        out.push(b'X');
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    fn pickle_global(out: &mut Vec<u8>, module: &str, name: &str) {
        out.push(b'c');
        out.extend_from_slice(module.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(name.as_bytes());
        out.push(b'\n');
    }

    fn pickle_binint1(out: &mut Vec<u8>, value: usize) {
        assert!(value <= u8::MAX as usize);
        out.extend_from_slice(&[b'K', value as u8]);
    }

    /// Emit the protocol-2 value produced by torch's
    /// `_rebuild_tensor_v2(storage, offset, shape, stride, ...)` reduction.
    fn pickle_tensor(
        out: &mut Vec<u8>,
        storage_key: &str,
        numel: usize,
        shape: &[usize],
        stride: &[usize],
    ) {
        pickle_global(out, "torch._utils", "_rebuild_tensor_v2");
        out.push(b'('); // MARK: reduction arguments

        out.push(b'('); // MARK: persistent storage id
        pickle_binunicode(out, "storage");
        pickle_global(out, "torch", "FloatStorage");
        pickle_binunicode(out, storage_key);
        pickle_binunicode(out, "cpu");
        pickle_binint1(out, numel);
        out.push(b't'); // TUPLE
        out.push(b'Q'); // BINPERSID

        pickle_binint1(out, 0); // storage offset
        out.push(b'(');
        for &dimension in shape {
            pickle_binint1(out, dimension);
        }
        out.push(b't'); // shape tuple
        out.push(b'(');
        for &dimension in stride {
            pickle_binint1(out, dimension);
        }
        out.push(b't'); // stride tuple
        out.push(b'\x89'); // NEWFALSE: requires_grad
        out.push(b'N'); // NONE: backward hooks
        out.push(b't'); // TUPLE: reduction arguments
        out.push(b'R'); // REDUCE
    }

    fn write_torch_pickle_fixture(path: &Path, tensors: &[(&str, &[f32], &[usize], &[usize])]) {
        let mut data_pickle = vec![b'\x80', 2]; // PROTO 2
        pickle_global(&mut data_pickle, "collections", "OrderedDict");
        data_pickle.extend_from_slice(&[b')', b'R', b'(']); // OrderedDict(), SETITEMS mark
        for (index, (name, values, shape, stride)) in tensors.iter().enumerate() {
            assert_eq!(values.len(), shape.iter().product::<usize>());
            pickle_binunicode(&mut data_pickle, name);
            pickle_tensor(
                &mut data_pickle,
                &index.to_string(),
                values.len(),
                shape,
                stride,
            );
        }
        data_pickle.extend_from_slice(&[b'u', b'.']); // SETITEMS, STOP

        let file = std::fs::File::create(path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        let options =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        archive.start_file("archive/data.pkl", options).unwrap();
        archive.write_all(&data_pickle).unwrap();
        archive.start_file("archive/byteorder", options).unwrap();
        archive.write_all(b"little").unwrap();
        for (index, (_, values, _, _)) in tensors.iter().enumerate() {
            archive
                .start_file(format!("archive/data/{index}"), options)
                .unwrap();
            archive.write_all(&f32_bytes(values)).unwrap();
        }
        archive.start_file("archive/version", options).unwrap();
        archive.write_all(b"3\n").unwrap();
        archive.finish().unwrap();
    }

    /// The Repository/workspace write path, end to end on a real pile file:
    /// tensors go into the pile's blob store as typed leaves, only the small
    /// fact set rides through the commit, and a cold reopen reads every tensor
    /// back with its shape.
    ///
    /// It is the seam the collection tests do not cover — blobs written
    /// straight to storage rather than staged through a prepared commit — and
    /// it is the one every `*_persist` binary uses.
    #[test]
    fn a_typed_pile_written_through_a_workspace_reads_back_cold() {
        let fixture = TempFixture::new();
        let weights_file = fixture.dir.join("model.safetensors");
        let matrix = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = vec![-0.5_f32, 0.5];
        let matrix_bytes = f32_bytes(&matrix);
        let vector_bytes = f32_bytes(&vector);
        serialize_to_file(
            [
                (
                    "block.weight",
                    TensorView::new(Dtype::F32, vec![3, 2], &matrix_bytes).unwrap(),
                ),
                (
                    "block.bias",
                    TensorView::new(Dtype::F32, vec![2], &vector_bytes).unwrap(),
                ),
            ],
            &None,
            &weights_file,
        )
        .unwrap();

        let pile_path = fixture.dir.join("weights.pile");
        persist_safetensors_files_to_pile(
            &[(weights_file, "model.safetensors".to_owned())],
            &pile_path,
            LeafDtype::F32,
        )
        .unwrap();

        // Cold reopen through the ordinary runtime loader.
        let keymap = load_keymap_from_pile(&pile_path).unwrap();
        assert_eq!(keymap.len(), 2);
        assert_eq!(keymap["block.weight"], (matrix.clone(), vec![3, 2]));
        assert_eq!(keymap["block.bias"], (vector.clone(), vec![2]));

        // And through the lazy index the aliasing loaders use: no data is
        // materialized to build it, and an f32 leaf serves a view rather than
        // a copy.
        let (f16, exact, _reader) = load_split_index_from_pile(&pile_path, "half_").unwrap();
        assert!(f16.is_empty(), "no half-width entity was written");
        assert_eq!(exact.len(), 2);
        let leaf = &exact["block.weight"];
        assert_eq!(leaf.elem(), crate::leaf::Elem::F32);
        assert_eq!(leaf.dims(), &[3, 2]);
        assert_eq!(&leaf.view_f32().expect("zero-copy f32 view")[..], &matrix[..]);
    }

    #[test]
    fn filtered_import_publishes_only_selected_tensors_and_rejects_empty_roots() {
        let fixture = TempFixture::new();
        let weights_file = fixture.dir.join("components.safetensors");
        let kept = vec![1.0_f32, 2.0, 3.0, 4.0];
        let dropped = vec![-1.0_f32, -2.0];
        let kept_bytes = f32_bytes(&kept);
        let dropped_bytes = f32_bytes(&dropped);
        serialize_to_file(
            [
                (
                    "talker.layer.weight",
                    TensorView::new(Dtype::F32, vec![2, 2], &kept_bytes).unwrap(),
                ),
                (
                    "codec.layer.bias",
                    TensorView::new(Dtype::F32, vec![2], &dropped_bytes).unwrap(),
                ),
            ],
            &None,
            &weights_file,
        )
        .unwrap();

        let pile_path = fixture.dir.join("models.pile");
        std::fs::File::create(&pile_path).unwrap();
        let signing_key = SigningKey::from_bytes(&[0x59; 32]);
        let mut pile = Pile::open(&pile_path).unwrap();
        let (root, commit) = import_safetensors_file_filtered_to_collection(
            &mut pile,
            &signing_key,
            &weights_file,
            LeafDtype::F32,
            "fixture/talker",
            "native",
            |name| name.starts_with("talker."),
        )
        .unwrap();

        let team = signing_key.verifying_key();
        let snapshot =
            crate::model_collection::snapshot_model_collection_exact(&mut pile, team, &[commit])
                .unwrap();
        let keymap = crate::selection::load_keymap_from_graph(
            snapshot.facts(),
            snapshot.reader(),
            ModelSelector::Root(root),
        )
        .unwrap();
        assert_eq!(keymap.len(), 1);
        assert_eq!(keymap["talker.layer.weight"], (kept, vec![2, 2]));
        assert!(!keymap.contains_key("codec.layer.bias"));

        let empty = import_safetensors_file_filtered_to_collection(
            &mut pile,
            &signing_key,
            &weights_file,
            LeafDtype::F32,
            "fixture/empty",
            "native",
            |_| false,
        )
        .unwrap_err();
        assert!(empty
            .to_string()
            .contains("selected no supported float tensors"));
        pile.close().unwrap();

        let latest =
            crate::model_collection::load_model_collection_local_latest(&pile_path, team).unwrap();
        assert_eq!(latest.commits(), &[commit]);
    }

    #[test]
    fn filtered_ordered_dict_pickle_ingest_builds_only_selected_tensors() {
        let fixture = TempFixture::new();
        let weights_file = fixture.dir.join("pytorch_model.bin");
        let kept = [1.25_f32, -2.5, 3.75, 4.5];
        let dropped = [-8.0_f32, 9.0];
        write_torch_pickle_fixture(
            &weights_file,
            &[
                ("encoder.layer.weight", &kept, &[2, 2], &[2, 1]),
                ("decoder.layer.bias", &dropped, &[2], &[1]),
            ],
        );

        let pile_path = fixture.dir.join("models.pile");
        std::fs::File::create(&pile_path).unwrap();
        let mut pile = Pile::open(&pile_path).unwrap();
        let fragment = ingest_weight_file_filtered_fragment(
            &mut pile,
            &weights_file,
            crate::formats::WeightFormat::Pickle,
            LeafDtype::F32,
            "fixture/pickle",
            "native",
            |name| name.starts_with("encoder."),
        )
        .unwrap();

        let root = fragment.root().expect("filtered model root");
        let reader = pile.reader().unwrap();
        let keymap = crate::selection::load_keymap_from_graph(
            fragment.facts(),
            &reader,
            ModelSelector::Root(root),
        )
        .unwrap();
        assert_eq!(keymap.len(), 1);
        assert_eq!(keymap["encoder.layer.weight"], (kept.to_vec(), vec![2, 2]));
        assert!(!keymap.contains_key("decoder.layer.bias"));
        pile.close().unwrap();
    }
}

/// The engine behind [`persist_safetensors_files_to_pile`] (untagged, `main`):
/// ingest each file's weight blobs straight into `pile_path`'s storage (no
/// in-memory carryover) and commit ONE model entity PER file on `main`, creating
/// the pile and branch if absent. (The content-addressed model-ROOT path is
/// [`import_model_to_collection`], native model collection.)
#[cfg(feature = "import")]
fn persist_files_to_pile(
    files: &[(std::path::PathBuf, String)],
    pile_path: &Path,
    dtype: LeafDtype,
) -> anyhow::Result<()> {
    // Pile::open requires the file to exist; create an empty one if needed —
    // loudly, so a typo'd path to an existing pile is visible instead of
    // silently persisting into a fresh file somewhere else.
    if !pile_path.exists() {
        eprintln!("[persist] pile {pile_path:?} does not exist — creating a NEW empty pile");
        std::fs::File::create(pile_path)
            .map_err(|e| anyhow::anyhow!("create pile {pile_path:?}: {e}"))?;
    }
    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    // Non-mutating load; NEVER amputate here. A corrupt tail on a weights
    // pile must fail loud — truncation is an explicit operator decision
    // (`trible pile amputate`), not a persist side effect.
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "pile {pile_path:?} failed to load ({e:?}); refusing to auto-truncate — \
             if the tail is a genuinely torn write, amputate explicitly with \
             `trible pile amputate`"
        )
    })?;

    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    // Reuse the branch if it exists (append into an existing pile), else create it.
    let branch_id = match repo
        .lookup_branch("main")
        .map_err(|e| anyhow::anyhow!("lookup main: {e:?}"))?
    {
        Some(id) => id,
        None => *repo
            .create_branch("main", None)
            .map_err(|e| anyhow::anyhow!("create main: {e:?}"))?,
    };
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;

    // Ingest each file's weight blobs straight into the pile storage (no
    // in-memory carryover), accumulating only the model-graph facts.
    let mut facts = TribleSet::new();
    for (path, name) in files {
        let bytes = read_safetensors_file(path);
        eprintln!("[persist] ingesting {name} ({} bytes)...", bytes.len());
        let frag = crate::ingest::save_safetensors_filtered(
            &bytes,
            name,
            repo.storage_mut(),
            dtype,
            |_| true,
        )
        .map_err(|e| anyhow::anyhow!("ingest {path:?}: {e}"))?;
        facts += frag.into_facts();
    }

    ws.commit(facts, "ingest model weights");
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok(())
}

/// Persist ONE safetensors file's tensors that pass `keep` into the pile under
/// `entity_name` — the per-component variant of [`persist_safetensors_files_to_pile`]
/// (e.g. the qwen3tts talker as a half-width `talker_f16` entity next to the
/// exact f32 leaves; content-addressing dedups the shape blobs it shares with
/// the f32 ingest). Appends to an existing pile or creates a fresh one.
#[cfg(feature = "import")]
pub fn persist_safetensors_file_filtered_to_pile(
    file: &Path,
    entity_name: &str,
    pile_path: &Path,
    dtype: LeafDtype,
    keep: impl Fn(&str) -> bool,
) -> anyhow::Result<()> {
    // Pile::open requires the file to exist; create an empty one if needed —
    // loudly, so a typo'd path to an existing pile is visible instead of
    // silently persisting into a fresh file somewhere else.
    if !pile_path.exists() {
        eprintln!("[persist] pile {pile_path:?} does not exist — creating a NEW empty pile");
        std::fs::File::create(pile_path)
            .map_err(|e| anyhow::anyhow!("create pile {pile_path:?}: {e}"))?;
    }
    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    // Non-mutating load; NEVER amputate here. A corrupt tail on a weights
    // pile must fail loud — truncation is an explicit operator decision
    // (`trible pile amputate`), not a persist side effect.
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "pile {pile_path:?} failed to load ({e:?}); refusing to auto-truncate — \
             if the tail is a genuinely torn write, amputate explicitly with \
             `trible pile amputate`"
        )
    })?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = match repo
        .lookup_branch("main")
        .map_err(|e| anyhow::anyhow!("lookup main: {e:?}"))?
    {
        Some(id) => id,
        None => *repo
            .create_branch("main", None)
            .map_err(|e| anyhow::anyhow!("create main: {e:?}"))?,
    };
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;

    let bytes = read_safetensors_file(file);
    eprintln!(
        "[persist] ingesting {entity_name} (filtered from {} bytes)...",
        bytes.len()
    );
    let frag = crate::ingest::save_safetensors_filtered(
        &bytes,
        entity_name,
        repo.storage_mut(),
        dtype,
        keep,
    )
    .map_err(|e| anyhow::anyhow!("ingest {file:?}: {e}"))?;

    ws.commit(
        frag.into_facts(),
        "ingest model weights (filtered component)",
    );
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok(())
}

/// Add the audited pre-epoch attribute aliases to a checked-out fact set.
///
/// TribleSpace commit `6b65f278` changed `"hex" as attribute: Encoding` from a
/// literal attribute id to one derived from `(hex, Encoding)`. Every pile
/// written before that carries the literal ids, so the declarations in
/// [`crate::format::attrs`] name nothing in it and every query returns empty.
/// [`crate::model_collection::project_legacy_model_attributes`] is the audited
/// historical-to-canonical table; applying it here makes those piles readable
/// where they lie. The projection is additive and purely in memory - the pile
/// on disk is never written, and a post-epoch pile gains nothing and loses
/// nothing.
fn pre_epoch_aliased(facts: &TribleSet) -> TribleSet {
    crate::model_collection::project_legacy_model_attributes(facts).facts
}

/// Drop every pre-epoch attribute spelling whose canonical alias is already
/// stated, and report how many went.
///
/// The exact inverse of [`pre_epoch_aliased`], and the reason it exists is that
/// the projection is a READ-side convenience which becomes a WRITE-side defect
/// the moment its output is persisted. A pile-to-pile conversion reads through
/// the projection and would otherwise carry both spellings of every fact into
/// the new file — a converted model pile stating `kind` 173 times AND its
/// historical literal 173 times, for a fact set exactly twice the size it
/// means. (Both spellings are already on disk in the qwen3tts collections,
/// which were published from projected facts for the same reason.)
///
/// A drop happens only where the canonical twin is present, so nothing this
/// removes is information: the fact survives under the name every current
/// reader actually queries. A historical id with no canonical twin in the set
/// is an error rather than a silent drop — that would mean the projection
/// table and the pile disagree, which is exactly the case where guessing loses
/// data. Attributes absent from the table are untouched, historical or not.
pub fn strip_projected_legacy_attributes(facts: &TribleSet) -> anyhow::Result<(TribleSet, usize)> {
    let aliases = crate::model_collection::legacy_model_attribute_aliases();
    let canonical_of: std::collections::HashMap<Id, Id> = aliases
        .iter()
        .map(|alias| (alias.historical, alias.canonical))
        .collect();
    let mut out = TribleSet::new();
    let mut dropped = 0usize;
    for fact in facts.iter() {
        if let Some(canonical) = canonical_of.get(fact.a()) {
            let twin = Trible::force(fact.e(), canonical, fact.v::<UnknownInline>());
            anyhow::ensure!(
                facts.contains(&twin),
                "pre-epoch fact {} on {} has no canonical twin — refusing to drop it",
                fact.a(),
                fact.e()
            );
            dropped += 1;
            continue;
        }
        out.insert(fact);
    }
    Ok((out, dropped))
}

/// Open a pile and build the leaf indexes for two families of model entities —
/// the ones whose name starts with `f16_prefix` (half-width leaves for the fast
/// native-width GPU load) and ALL OTHERS (the exact leaves) — plus a
/// [`PileReader`](triblespace::core::repo::pile::PileReader) the caller may keep.
/// The first index comes back empty if no entity matches the prefix.
///
/// No tensor bytes are COPIED: each leaf's payload is a slice of the pile's
/// mapping, which stays valid after the repository is closed because every blob
/// keeps the mapping alive. Building the index does, however, resolve every
/// leaf, and a pile validates a record's hash the first time it hands it out —
/// so the integrity check that used to happen at first touch now happens here,
/// for the whole family at once. Same total work for the ordinary case (a model
/// whose tensors all get loaded), front-loaded rather than interleaved.
pub fn load_split_index_from_pile(
    pile_path: &Path,
    f16_prefix: &str,
) -> anyhow::Result<(
    HashMap<String, crate::leaf::Leaf>,
    HashMap<String, crate::leaf::Leaf>,
    triblespace::core::repo::pile::PileReader,
)> {
    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    // Read path: non-mutating load, NEVER amputate. A corrupt tail fails
    // loud; truncation is an explicit operator decision (`trible pile
    // amputate`), never a side effect of loading weights.
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "pile {pile_path:?} failed to load ({e:?}); refusing to auto-truncate on a \
             read path — if the tail is a genuinely torn write, amputate explicitly \
             with `trible pile amputate`"
        )
    })?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = repo
        .lookup_branch("main")
        .map_err(|e| anyhow::anyhow!("lookup main: {e:?}"))?
        .ok_or_else(|| anyhow::anyhow!("no 'main' branch in pile {pile_path:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;
    let head = ws
        .head()
        .ok_or_else(|| anyhow::anyhow!("'main' has no commits"))?;
    let tribles: TribleSet = pre_epoch_aliased(
        ws.checkout(ancestors(head))
            .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
            .facts(),
    );
    let reader = repo
        .storage_mut()
        .reader()
        .map_err(|e| anyhow::anyhow!("pile reader: {e:?}"))?;

    let mut f16 = HashMap::new();
    let mut f32_ = HashMap::new();
    for (m, n) in find!(
        (m: Id, n: Inline<inlineencodings::Handle<blobencodings::LongString>>),
        pattern!(&tribles, [{ ?m @ crate::format::attrs::model_name: ?n }])
    ) {
        let name: anybytes::View<str> = reader
            .get(n)
            .map_err(|e| anyhow::anyhow!("model name blob: {e:?}"))?;
        if !f16_prefix.is_empty() && name.starts_with(f16_prefix) {
            f16.extend(crate::ingest::index_keymap(&tribles, &reader, m));
        } else {
            f32_.extend(crate::ingest::index_keymap(&tribles, &reader, m));
        }
    }
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok((f16, f32_, reader))
}

/// The runtime weight loader for a pile that carries a half-width alias entity
/// (e.g. the qwen3tts pile after `qwen3tts_persist --f16-talker-only`): on
/// macOS the fast [`WeightLoader::Aliased`] — requests on the fused Metal
/// backends load the half-width blobs DIRECTLY at native width (no f32
/// materialization/cast) — and elsewhere (or under `MARY_SPEAK_MATERIALIZE=1`,
/// the A/B switch) the exact materialized keymap. In BOTH cases the
/// `f16_prefix` entities are excluded from the exact/f32 side, so
/// materialized loads never see half-width leaves.
#[cfg(feature = "qwen3tts")]
pub fn load_aliased_loader_from_pile(
    pile_path: &Path,
    f16_prefix: &str,
) -> anyhow::Result<crate::nn::weight_loader::WeightLoader> {
    use crate::nn::weight_loader::WeightLoader;
    let (f16, f32_, _reader) = load_split_index_from_pile(pile_path, f16_prefix)?;
    anyhow::ensure!(
        !f32_.is_empty(),
        "no exact (f32) model entities in pile {pile_path:?}"
    );
    let materialize = std::env::var("MARY_SPEAK_MATERIALIZE").is_ok();
    #[cfg(target_os = "macos")]
    if !materialize {
        if f16.is_empty() {
            eprintln!(
                "[mary] pile has no '{f16_prefix}' entity — f16-backend tensors will \
                 materialize+cast (append it with: qwen3tts_persist <model-dir> <pile> --f16-talker-only)"
            );
        }
        return Ok(WeightLoader::Aliased(
            crate::nn::weight_loader::AliasedPile::new(
                f16,
                f32_,
                crate::nn::backend::WgpuDevice::default(),
            ),
        ));
    }
    if materialize {
        eprintln!("[mary] MARY_SPEAK_MATERIALIZE set — using the fully materialized load");
    }
    let _ = f16; // half-width leaves are only for aliasing; exact leaves feed the keymap
    let keymap = f32_
        .into_iter()
        .map(|(k, leaf)| (k, leaf.to_f32_shape()))
        .collect();
    Ok(WeightLoader::Pile(keymap))
}

/// Load the canonical PersonaPlex bundle with its exact signed authority.
///
/// Each admitted COMMIT must contain one atomic `(root, archive, H)` token;
/// `H` is decoded and validated independently rather than reconstructed from
/// a broad graph union. The returned bundle retains the frozen ticket and
/// `(root, H, τ)` identity alongside the exact loader.
#[cfg(feature = "qwen3tts")]
pub fn personaplex_bundle(
    pile_path: &Path,
) -> anyhow::Result<
    crate::models::personaplex::PersonaPlexBundle<
        triblespace::core::repo::pile::PileReader,
    >,
> {
    // Team discovery and ticket selection share one observed record prefix.
    // The 32 GiB model also stays under one open pile, avoiding a second
    // validate/close/reopen cycle merely to learn the publishing team.
    let mut pile = Pile::open(pile_path)
        .with_context(|| format!("open PersonaPlex bundle pile {pile_path:?}"))?;
    let (team, snapshot) = match
        crate::model_collection::snapshot_sole_model_bundle_collection_local_latest(&mut pile)
    {
        Ok(observation) => observation,
        Err(error) => {
            let _ = pile.close();
            return Err(error).with_context(|| {
                format!("freeze the sole PersonaPlex bundle snapshot in {pile_path:?}")
            });
        }
    };
    pile.close()
        .with_context(|| format!("close PersonaPlex bundle pile {pile_path:?}"))?;
    crate::models::personaplex::PersonaPlexWeights::from_bundle_snapshot(team, snapshot)
        .with_context(|| format!("select exact PersonaPlex bundle from {pile_path:?}"))
}

/// Convenience weight-loader projection backed exclusively by bundle authority.
/// Runtime consumers should use [`personaplex_bundle`] when the exact ticket
/// and identity must stay paired with the loader.
#[cfg(feature = "qwen3tts")]
pub fn personaplex_loader(
    pile_path: &Path,
) -> anyhow::Result<crate::nn::weight_loader::WeightLoader> {
    personaplex_bundle(pile_path).map(|bundle| bundle.into_parts().1)
}

/// The derived FOLDED half-width sibling of a qwen3tts weights pile:
/// `<stem>_folded_f16.pile` next to it (`models/qwen3tts.pile` →
/// `models/qwen3tts_folded_f16.pile`; the 0.6B sibling maps analogously).
/// It holds the talker's GPU tensors in their FINAL runtime layout — the
/// load-time folds (wide fused `[qk | R(qk) | v]` with rotate_half
/// pre-applied to weight rows, norm weights folded into matmul rows,
/// pre-transposed Linears, the q/k-norm × 1/√d RoPE chain weights) applied
/// ONCE at derive time — so the raw-backend talker aliases every tensor
/// zero-copy straight from the mmap'd pile pages. Written by
/// `qwen3tts_persist --fold-derive`; consumed by
/// [`load_qwen3tts_talker_folded`] (the speak talker lane — raw/zero-copy is
/// the ONLY talker lane since 2026-07-12).
/// (Extends the voxtral `<stem>_f16.pile` sibling convention: `_f16` =
/// same names, half width; `_folded_f16` = derived layout, half width.)
#[cfg(feature = "qwen3tts")]
pub fn qwen3tts_folded_sibling_path(pile_path: &Path) -> std::path::PathBuf {
    let stem = pile_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("weights");
    pile_path.with_file_name(format!("{stem}_folded_f16.pile"))
}

/// The `(leaf name, f16 bits, dims)` readback of every fold-transformed GPU
/// tensor a [`Talker`](crate::models::qwen3tts::talker::Talker) holds — the
/// single source of truth for the folded pile's name scheme, shared by the
/// derive (write these), the derive gate, and `qwen3tts_raw_gate` (compare
/// lanes against these). CPU stages (codec_head, codec_embedding_cpu) and
/// the computed RoPE tables are not included; the untransformed embeddings
/// are ALSO excluded — they alias the canonical pile's `talker_f16` leaves
/// directly, no duplication. `B` must be an f16-storage backend
/// (`BFusedHalf` / `BHalf`).
#[cfg(all(feature = "qwen3tts", target_os = "macos"))]
pub fn qwen3tts_folded_readback<B: burn::prelude::Backend>(
    talker: &crate::models::qwen3tts::talker::Talker<B>,
) -> Vec<(String, Vec<half::f16>, Vec<u64>)> {
    use burn::prelude::*;
    fn rb<B: Backend, const D: usize>(
        out: &mut Vec<(String, Vec<half::f16>, Vec<u64>)>,
        name: String,
        t: &Tensor<B, D>,
    ) {
        let dims: Vec<u64> = t.dims().iter().map(|&d| d as u64).collect();
        let bits: Vec<half::f16> = t
            .clone()
            .into_data()
            .to_vec()
            .expect("f16 readback (f16-storage backend required)");
        out.push((name, bits, dims));
    }
    let mut out = Vec::new();
    rb(
        &mut out,
        "talker.folded.text_projection.linear_fc1.weight_t".into(),
        &talker.text_fc1.weight_t,
    );
    rb(
        &mut out,
        "talker.folded.text_projection.linear_fc1.bias".into(),
        talker.text_fc1.bias.as_ref().expect("fc1 bias"),
    );
    rb(
        &mut out,
        "talker.folded.text_projection.linear_fc2.weight_t".into(),
        &talker.text_fc2.weight_t,
    );
    rb(
        &mut out,
        "talker.folded.text_projection.linear_fc2.bias".into(),
        talker.text_fc2.bias.as_ref().expect("fc2 bias"),
    );
    for (i, layer) in talker.layers.iter().enumerate() {
        rb(
            &mut out,
            format!("talker.folded.layers.{i}.attn.wide_t"),
            &layer.attn.wide_t,
        );
        rb(
            &mut out,
            format!("talker.folded.layers.{i}.attn.w"),
            &layer.attn.w,
        );
        rb(
            &mut out,
            format!("talker.folded.layers.{i}.attn.w_rot"),
            &layer.attn.w_rot,
        );
        rb(
            &mut out,
            format!("talker.folded.layers.{i}.attn.o_proj.weight_t"),
            &layer.attn.o_proj.weight_t,
        );
        rb(
            &mut out,
            format!("talker.folded.layers.{i}.gate_up_t"),
            &layer.gate_up_t,
        );
        rb(
            &mut out,
            format!("talker.folded.layers.{i}.down_proj.weight_t"),
            &layer.down_proj.weight_t,
        );
    }
    rb(
        &mut out,
        "talker.folded.norm.weight".into(),
        &talker.norm.weight,
    );
    out
}

/// Derive the folded zero-copy sibling pile for a qwen3tts weights pile (see
/// [`qwen3tts_folded_sibling_path`]). Loads the talker EXACTLY as the
/// production fused-f16 lane does (same fold code, same f16 leaves, same
/// arithmetic), reads back every derived GPU tensor, and writes each as an
/// f16 leaf into a NEW sibling pile under `talker.folded.*` names — so the
/// sibling's bytes are bit-identical to the weights the production lane
/// computes at load time. Gate inline: the sibling is re-opened and every
/// leaf compared bit-for-bit against the readback. Returns
/// `(tensor count, total f16 bytes)`.
///
/// RAILS: `dst_pile` must not exist (only NEW files are written); `src_pile`
/// is opened read-only through the ordinary load path and its byte length is
/// verified unchanged afterwards.
#[cfg(all(feature = "qwen3tts", target_os = "macos"))]
pub fn derive_qwen3tts_folded_pile(
    src_pile: &Path,
    dst_pile: &Path,
) -> anyhow::Result<(usize, usize)> {
    use crate::format::attrs;
    use crate::models::qwen3tts::talker::Talker;
    use crate::nn::backend::BFusedHalf;

    anyhow::ensure!(
        !dst_pile.exists(),
        "dst pile {dst_pile:?} already exists — --fold-derive writes only NEW sibling piles \
         (delete it first if you mean to re-derive)"
    );
    let src_len_before = std::fs::metadata(src_pile)?.len();

    eprintln!("[fold-derive] loading production talker (BFusedHalf) from {src_pile:?} ...");
    let loader = load_aliased_loader_from_pile(src_pile, "talker_f16")?;
    let dev = Default::default();
    let talker = Talker::<BFusedHalf>::load(&loader, &dev);
    drop(loader);
    eprintln!("[fold-derive] reading back folded tensors ...");
    let tensors = qwen3tts_folded_readback(&talker);
    drop(talker);

    eprintln!("[fold-derive] pile {dst_pile:?} does not exist — creating a NEW sibling pile");
    std::fs::File::create(dst_pile)
        .map_err(|e| anyhow::anyhow!("create pile {dst_pile:?}: {e}"))?;
    let pile =
        Pile::open(dst_pile).map_err(|e| anyhow::anyhow!("open pile {dst_pile:?}: {e:?}"))?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = *repo
        .create_branch("main", None)
        .map_err(|e| anyhow::anyhow!("create main: {e:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;

    let mut members: Vec<Id> = Vec::new();
    let mut facts = TribleSet::new();
    let (mut count, mut bytes) = (0usize, 0usize);
    for (name, bits, dims) in &tensors {
        // put_raw_f16 re-rounds f32→f16; f16→f32→f16 round-trips bit-exactly,
        // so the stored leaf is the exact production readback.
        let data: Vec<f32> = bits.iter().map(|h| h.to_f32()).collect();
        let leaf = crate::format::put_raw_f16(repo.storage_mut(), &data, dims)
            .map_err(|e| anyhow::anyhow!("{name}: put f16 leaf: {e}"))?;
        let leaf_id = leaf.root().expect("leaf root");
        facts += leaf.into_facts();
        let kind = match dims.len() {
            1 => "vector",
            2 => "matrix",
            3 => "conv",
            _ => "tensor",
        };
        let name_h = repo
            .storage_mut()
            .put::<blobencodings::LongString, _>(name.clone())
            .map_err(|e| anyhow::anyhow!("{name}: put name blob: {e:?}"))?;
        let m = entity! { _ @ attrs::kind: kind, attrs::safetensor_path: name_h, attrs::weight: leaf_id };
        members.push(m.root().expect("module root"));
        facts += m.into_facts();
        count += 1;
        bytes += bits.len() * 2;
    }
    let mn = repo
        .storage_mut()
        .put::<blobencodings::LongString, _>("talker_folded_f16".to_string())
        .map_err(|e| anyhow::anyhow!("put entity name blob: {e:?}"))?;
    let model = entity! { _ @ attrs::model_name: mn, attrs::member*: members.iter() };
    facts += model.into_facts();
    ws.commit(
        facts,
        "derive folded talker weights (production readback) for zero-copy alias",
    );
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;

    // ── rails: the canonical pile must be byte-length unchanged ──
    let src_len_after = std::fs::metadata(src_pile)?.len();
    anyhow::ensure!(
        src_len_before == src_len_after,
        "canonical pile {src_pile:?} changed length during derive ({src_len_before} → {src_len_after}) — investigate immediately"
    );

    // ── gate: re-open the sibling, alias every leaf, compare bit-for-bit ──
    eprintln!("[fold-derive] gate: aliasing every leaf back and comparing bits ...");
    let (_, folded, _folded_reader) = load_split_index_from_pile(dst_pile, "")?;
    for (name, bits, dims) in &tensors {
        let leaf = match folded.get(name.as_str()) {
            Some(leaf) if leaf.elem() == crate::leaf::Elem::F16 => leaf,
            other => anyhow::bail!(
                "{name}: bad folded leaf after derive ({})",
                if other.is_none() { "missing" } else { "f32" }
            ),
        };
        anyhow::ensure!(
            leaf.dims() == &dims[..],
            "{name}: shape mismatch {:?} vs {dims:?}",
            leaf.dims()
        );
        let t = crate::nn::alias::alias_flat_raw::<half::f16>(
            leaf.payload().clone(),
            &crate::nn::backend::WgpuDevice::default(),
        )
        .map_err(|e| anyhow::anyhow!("{name}: alias failed: {e}"))?;
        let back: Vec<half::f16> = t.into_data().to_vec().expect("aliased readback");
        anyhow::ensure!(back.len() == bits.len(), "{name}: length mismatch");
        let mism = back
            .iter()
            .zip(bits)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        anyhow::ensure!(
            mism == 0,
            "{name}: {mism} f16 elements differ after alias round-trip"
        );
    }
    eprintln!(
        "[fold-derive] gate PASSED: {count} tensors / {bytes} f16 bytes bit-identical through \
         the zero-copy alias"
    );
    Ok((count, bytes))
}

/// Load the qwen3tts talker for the RAW f16 backend (`BHalf`) with every GPU
/// tensor a ZERO-COPY alias of an mmap'd pile blob: the folded tensors from
/// the derived sibling pile (see [`derive_qwen3tts_folded_pile`]) and the
/// untransformed embeddings from the canonical pile's `talker_f16` leaves.
/// No fold math runs at load time and no weight bytes are copied on the GPU
/// path — the model's buffers ARE the pile's pages (first-touch page-in,
/// evictable, shared across processes). The CPU stages (codec-head gemv,
/// codec-embedding rows) materialize from the canonical exact-f32 leaves as
/// in every other lane.
#[cfg(all(feature = "qwen3tts", target_os = "macos"))]
pub fn load_qwen3tts_talker_folded(
    src_pile: &Path,
    folded_pile: &Path,
) -> anyhow::Result<crate::models::qwen3tts::talker::Talker<crate::nn::backend::BHalf>> {
    let (f16, f32_, _reader) = load_split_index_from_pile(src_pile, "talker_f16")?;
    anyhow::ensure!(
        !f16.is_empty(),
        "no 'talker_f16' leaves in {src_pile:?} (append with qwen3tts_persist --f16-talker-only)"
    );
    let (_, folded, _folded_reader) = load_split_index_from_pile(folded_pile, "")?;
    anyhow::ensure!(
        !folded.is_empty(),
        "no leaves in folded pile {folded_pile:?}"
    );
    load_qwen3tts_talker_folded_from_indexes(&f16, &f32_, &folded)
}

/// Construct the raw f16 Qwen3-TTS talker from already-selected native model
/// indexes. The three may all come from one frozen collection: the split is
/// semantic (base/talker/folded roots), not a storage or file boundary.
///
/// No blob reader: a leaf already holds its bytes as a view over the pile's
/// mapping, so aliasing one onto the GPU needs nothing but the leaf.
#[cfg(all(feature = "qwen3tts", target_os = "macos"))]
pub fn load_qwen3tts_talker_folded_from_indexes(
    f16: &HashMap<String, crate::leaf::Leaf>,
    f32_: &HashMap<String, crate::leaf::Leaf>,
    folded: &HashMap<String, crate::leaf::Leaf>,
) -> anyhow::Result<crate::models::qwen3tts::talker::Talker<crate::nn::backend::BHalf>> {
    use crate::models::qwen3tts::config::{
        TALKER_EPS, TALKER_HEAD_DIM, TALKER_LAYERS, TALKER_ROPE_THETA,
    };
    use crate::models::qwen3tts::layers::{
        Attention, DecoderLayer, Embedding, Linear, RmsNorm, RopeTable,
    };
    use crate::models::qwen3tts::talker::{talker_attn_config, Talker};
    use crate::nn::backend::BHalf;
    use burn::prelude::*;

    let dev = crate::nn::backend::WgpuDevice::default();
    fn alias(
        idx: &HashMap<String, crate::leaf::Leaf>,
        name: &str,
        dev: &crate::nn::backend::WgpuDevice,
    ) -> anyhow::Result<(Tensor<BHalf, 1>, Vec<usize>)> {
        let leaf = match idx.get(name) {
            Some(leaf) if leaf.elem() == crate::leaf::Elem::F16 => leaf,
            Some(_) => anyhow::bail!("{name}: expected an f16 leaf, found f32"),
            None => anyhow::bail!("{name}: missing from pile index"),
        };
        let t = crate::nn::alias::alias_flat_raw::<half::f16>(leaf.payload().clone(), dev)
            .map_err(|e| anyhow::anyhow!("{name}: zero-copy alias failed: {e}"))?;
        Ok((t, leaf.shape()))
    }
    let f3 = |name: &str| -> anyhow::Result<Tensor<BHalf, 3>> {
        let (t, s) = alias(folded, name, &dev)?;
        anyhow::ensure!(s.len() == 3, "{name}: rank {} != 3", s.len());
        Ok(t.reshape([s[0], s[1], s[2]]))
    };
    let f4 = |name: &str| -> anyhow::Result<Tensor<BHalf, 4>> {
        let (t, s) = alias(folded, name, &dev)?;
        anyhow::ensure!(s.len() == 4, "{name}: rank {} != 4", s.len());
        Ok(t.reshape([s[0], s[1], s[2], s[3]]))
    };

    let cfg = talker_attn_config();
    let (ce, ce_shape) = alias(f16, "talker.model.codec_embedding.weight", &dev)?;
    anyhow::ensure!(ce_shape.len() == 2, "codec_embedding rank != 2");
    let codec_embedding = Embedding {
        weight: ce.reshape([ce_shape[0], ce_shape[1]]),
    };
    let hidden = ce_shape[1];
    let (te, te_shape) = alias(f16, "talker.model.text_embedding.weight", &dev)?;
    anyhow::ensure!(te_shape.len() == 2, "text_embedding rank != 2");
    let text_embedding = Embedding {
        weight: te.reshape([te_shape[0], te_shape[1]]),
    };

    let linear =
        |wt: Tensor<BHalf, 3>, bias: Option<Tensor<BHalf, 3>>| Linear { weight_t: wt, bias };
    let text_fc1 = linear(
        f3("talker.folded.text_projection.linear_fc1.weight_t")?,
        Some(f3("talker.folded.text_projection.linear_fc1.bias")?),
    );
    let text_fc2 = linear(
        f3("talker.folded.text_projection.linear_fc2.weight_t")?,
        Some(f3("talker.folded.text_projection.linear_fc2.bias")?),
    );
    let mut layers = Vec::with_capacity(TALKER_LAYERS);
    for i in 0..TALKER_LAYERS {
        let attn = Attention::from_parts(
            f3(&format!("talker.folded.layers.{i}.attn.wide_t"))?,
            f4(&format!("talker.folded.layers.{i}.attn.w"))?,
            f4(&format!("talker.folded.layers.{i}.attn.w_rot"))?,
            linear(
                f3(&format!("talker.folded.layers.{i}.attn.o_proj.weight_t"))?,
                None,
            ),
            cfg,
        );
        layers.push(DecoderLayer::from_parts(
            attn,
            f3(&format!("talker.folded.layers.{i}.gate_up_t"))?,
            linear(
                f3(&format!("talker.folded.layers.{i}.down_proj.weight_t"))?,
                None,
            ),
            TALKER_EPS,
        ));
    }
    let (nw, nw_shape) = alias(folded, "talker.folded.norm.weight", &dev)?;
    anyhow::ensure!(nw_shape.len() == 1, "norm.weight rank != 1");
    let norm = RmsNorm {
        weight: nw,
        eps: TALKER_EPS,
    };

    // CPU stages: exact f32 leaves from the canonical pile, as in every lane.
    let codec_embedding_cpu = f32_
        .get("talker.model.codec_embedding.weight")
        .ok_or_else(|| anyhow::anyhow!("talker.model.codec_embedding.weight: missing f32 leaf"))?
        .to_f32();
    let codec_head = f32_
        .get("talker.codec_head.weight")
        .ok_or_else(|| anyhow::anyhow!("talker.codec_head.weight: missing f32 leaf"))?
        .to_f32();

    Ok(Talker {
        hidden,
        codec_embedding,
        codec_embedding_cpu,
        text_embedding,
        text_fc1,
        text_fc2,
        layers,
        norm,
        codec_head,
        rope: RopeTable::new(TALKER_ROPE_THETA, TALKER_HEAD_DIM, 8192, &dev),
    })
}

/// Load the weight keymap from JUST the pile file — no safetensors. Opens
/// the pile, resolves the `main` branch, checks out the full history, finds every
/// model entity, and materializes the union `name → (f32, shape)` keymap.
pub fn load_keymap_from_pile(
    pile_path: &Path,
) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    load_keymap_from_pile_prefixed(pile_path, "")
}

/// Like [`load_keymap_from_pile`], but materializes only the model entities
/// whose `model_name` starts with `name_prefix` — the per-component load for
/// piles that hold several components (e.g. flux's `"text_encoder/"`,
/// `"transformer/"`, `"vae/"`), so one phase's weights peak in RAM at a time.
pub fn load_keymap_from_pile_prefixed(
    pile_path: &Path,
    name_prefix: &str,
) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    // Read path: non-mutating load, NEVER amputate. A corrupt tail fails
    // loud; truncation is an explicit operator decision (`trible pile
    // amputate`), never a side effect of loading weights.
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "pile {pile_path:?} failed to load ({e:?}); refusing to auto-truncate on a \
             read path — if the tail is a genuinely torn write, amputate explicitly \
             with `trible pile amputate`"
        )
    })?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;

    let branch_id = repo
        .lookup_branch("main")
        .map_err(|e| anyhow::anyhow!("lookup main: {e:?}"))?
        .ok_or_else(|| anyhow::anyhow!("no 'main' branch in pile {pile_path:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;
    let head = ws
        .head()
        .ok_or_else(|| anyhow::anyhow!("'main' has no commits"))?;

    // Full history → all the model-graph facts.
    let checkout = ws
        .checkout(ancestors(head))
        .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
    let tribles: TribleSet = pre_epoch_aliased(checkout.facts());

    // A reader over the pile blobs (where the weights live).
    let reader = repo
        .storage_mut()
        .reader()
        .map_err(|e| anyhow::anyhow!("pile reader: {e:?}"))?;

    // Every model entity (one per persisted shard) carries `attrs::model_name`.
    let model_ids: Vec<Id> = find!(
        (m: Id, n: Inline<inlineencodings::Handle<blobencodings::LongString>>),
        pattern!(&tribles, [{ ?m @ crate::format::attrs::model_name: ?n }])
    )
    .filter(|&(_m, n)| {
        let v: anybytes::View<str> = reader.get(n).expect("model name blob");
        v.starts_with(name_prefix)
    })
    .map(|(m, _n)| m)
    .collect();
    if model_ids.is_empty() {
        anyhow::bail!("no model entity (attrs::model_name matching {name_prefix:?}) found in pile");
    }

    let mut keymap: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for id in model_ids {
        keymap.extend(load_keymap(&tribles, &reader, id));
    }
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    if keymap.is_empty() {
        anyhow::bail!("keymap empty after materializing from pile");
    }
    Ok(keymap)
}

/// The default weight-format tag: the faithful import (no derived quantization).
pub const QUANTIZATION_NATIVE: &str = "native";

#[cfg(test)]
mod legacy_spelling_tests {
    use super::*;
    use crate::model_collection::{
        legacy_model_attribute_aliases, project_legacy_model_attributes, ModelAttributeAlias,
    };
    use triblespace::core::id_hex;

    fn raw_fact(entity: Id, attribute: Id, value: u8) -> Trible {
        Trible::force(
            &entity,
            &attribute,
            &Inline::<UnknownInline>::new([value; 32]),
        )
    }

    fn mapping(label: &str) -> ModelAttributeAlias {
        legacy_model_attribute_aliases()
            .into_iter()
            .find(|mapping| mapping.label == label)
            .unwrap_or_else(|| panic!("missing mapping {label}"))
    }

    /// Projecting then stripping is the identity on the canonical facts, and
    /// leaves nothing pre-epoch behind.
    ///
    /// This is the round trip a conversion actually performs: it reads a
    /// pre-epoch pile through the projection and writes what comes out. Without
    /// the strip it writes both spellings — which is what the qwen3tts
    /// collections on disk hold, 17208 facts stating 8604.
    #[test]
    fn projecting_then_stripping_leaves_exactly_one_spelling() {
        let model = id_hex!("A4C3D3D77C7C63A0E9CE1A45A1F3B4B5");
        let leaf = id_hex!("2A0DE5F9E2A56AB6D5A21A3E9BB2F0C1");
        let legacy = [
            raw_fact(model, mapping("format.member").historical, 0x11),
            raw_fact(model, mapping("format.model_name").historical, 0x12),
            raw_fact(leaf, mapping("format.shape").historical, 0x22),
            raw_fact(leaf, mapping("format.weight").historical, 0x23),
        ];
        // One attribute with no mapping at all, which must survive untouched.
        let unmapped = raw_fact(leaf, id_hex!("0B51DA3E67216213871743E045590DBC"), 0x44);
        let input: TribleSet = legacy.iter().copied().chain([unmapped]).collect();

        let projected = project_legacy_model_attributes(&input).facts;
        assert_eq!(projected.len(), 9, "four aliases added beside five facts");

        let (stripped, dropped) = strip_projected_legacy_attributes(&projected).unwrap();
        assert_eq!(dropped, legacy.len());
        assert_eq!(stripped.len(), 5);
        assert!(stripped.contains(&unmapped));
        for source in legacy {
            let alias_mapping = legacy_model_attribute_aliases()
                .into_iter()
                .find(|mapping| mapping.historical == *source.a())
                .expect("audited mapping");
            let canonical = Trible::force(
                source.e(),
                &alias_mapping.canonical,
                source.v::<UnknownInline>(),
            );
            assert!(!stripped.contains(&source), "pre-epoch spelling survived");
            assert!(stripped.contains(&canonical), "canonical spelling lost");
        }

        // Idempotent: nothing pre-epoch is left to drop.
        let (again, dropped_again) = strip_projected_legacy_attributes(&stripped).unwrap();
        assert_eq!(dropped_again, 0);
        assert_eq!(again, stripped);
    }

    /// A pre-epoch fact whose canonical twin is missing is an error, not a
    /// silent drop.
    ///
    /// The whole safety of the strip rests on the twin existing. If the
    /// projection table and the pile ever disagree, the difference between
    /// refusing and shrugging is the difference between a loud failure and a
    /// converted model quietly missing a fact.
    #[test]
    fn an_unprojected_pre_epoch_fact_is_refused_rather_than_dropped() {
        let leaf = id_hex!("2A0DE5F9E2A56AB6D5A21A3E9BB2F0C1");
        let orphan: TribleSet = [raw_fact(leaf, mapping("format.shape").historical, 0x22)]
            .into_iter()
            .collect();
        let error = strip_projected_legacy_attributes(&orphan)
            .err()
            .expect("an unprojected pre-epoch fact must not be dropped");
        assert!(
            error.to_string().contains("no canonical twin"),
            "unexpected diagnostic: {error}"
        );
    }
}

/// A model pile's complete facts, read through whichever reader it actually
/// has, plus the identity of the collection that authorized them.
///
/// `collection` is `Some((team, handle))` exactly when the pile publishes a
/// `mary-model-graph` collection, and the handle is taken from the pile's OWN
/// commits rather than recomputed from the team — so a conversion comparing
/// source and destination compares what is on disk, not one function against
/// itself.
pub struct ModelPileSource {
    /// Every fact the model is stated by, canonical aliases projected in.
    pub facts: TribleSet,
    /// Reader over the source's immutable mapping; valid after the pile closes.
    pub reader: triblespace::core::repo::pile::PileReader,
    /// `(team, collection handle)`, absent for a pre-collection pile.
    pub collection: Option<(
        ed25519_dalek::VerifyingKey,
        triblespace::core::collection::CollectionHandle,
    )>,
    /// Which reader answered: `"collection"`, or the branch name.
    pub via: &'static str,
}

/// Read a model pile's complete facts, PREFERRING its signed collection.
///
/// The two readers a model pile has do not agree, and the collection is the
/// one that is right. `qwen3tts.pile` carries two pins both named `main`;
/// `lookup_branch` picks one, and that one states 1292 of the model's 1465
/// weight entities. The collection states all 1465 — the pins are fragments of
/// a history, the collection is the model. Anything deciding what a pile
/// contains (a converter, a verifier) has to ask the collection or it is
/// deciding about a fraction.
///
/// The branch fallback is for the pre-collection piles only (f5, smolvla,
/// siglip). It is never consulted when a collection is present.
pub fn read_model_pile(path: &Path) -> anyhow::Result<ModelPileSource> {
    let mut pile = Pile::open(path).map_err(|e| anyhow::anyhow!("open {path:?}: {e:?}"))?;
    let team = match crate::model_collection::sole_model_graph_team(&mut pile) {
        Ok(team) => Some(team),
        Err(crate::model_collection::SoleModelGraphTeamError::None) => None,
        Err(e) => {
            let _ = pile.close();
            anyhow::bail!("{path:?}: model-graph team: {e}");
        }
    };
    let Some(team) = team else {
        let _ = pile.close();
        let (branch, facts, reader) = checkout_any_branch(path)?;
        return Ok(ModelPileSource {
            facts,
            reader,
            collection: None,
            via: branch,
        });
    };

    let snapshot = crate::model_collection::snapshot_model_collection_local_latest(&mut pile, team)
        .map_err(|e| anyhow::anyhow!("{path:?}: snapshot model collection: {e}"))?;
    let facts = pre_epoch_aliased(snapshot.facts());
    let (_, commits, reader) = snapshot.into_parts();
    let handle = commits
        .first()
        .ok_or_else(|| anyhow::anyhow!("{path:?}: a model collection with no commits"))?
        .collection();
    let _ = pile.close();
    Ok(ModelPileSource {
        facts,
        reader,
        collection: Some((team, handle)),
        via: "collection",
    })
}

/// Open a pile and return `(branch name, facts, blob reader)` from whichever
/// branch holds the model — `mary` if present, else `main`.
///
/// Detected, not assumed. Both layouts are in the wild, and a loader that
/// guesses wrong reports "no model" for a pile that plainly has one.
/// NEVER amputates: a corrupt tail fails loud.
pub fn checkout_any_branch(
    pile_path: &Path,
) -> anyhow::Result<(
    &'static str,
    TribleSet,
    triblespace::core::repo::pile::PileReader,
)> {
    for branch in ["mary", "main"] {
        let mut pile =
            Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open {pile_path:?}: {e:?}"))?;
        pile.refresh().map_err(|e| {
            anyhow::anyhow!("{pile_path:?} failed to load ({e:?}); refusing to auto-truncate")
        })?;
        let mut repo = Repository::new(
            pile,
            SigningKey::generate(&mut rand::rngs::OsRng),
            TribleSet::new(),
        )
        .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
        let found = repo
            .lookup_branch(branch)
            .map_err(|e| anyhow::anyhow!("lookup {branch}: {e:?}"))?;
        let Some(branch_id) = found else {
            repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;
            continue;
        };
        let mut ws = repo
            .pull(branch_id)
            .map_err(|e| anyhow::anyhow!("pull {branch}: {e:?}"))?;
        let Some(head) = ws.head() else {
            repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;
            continue;
        };
        let tribles: TribleSet = pre_epoch_aliased(
            ws.checkout(ancestors(head))
                .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
                .facts(),
        );
        let reader = repo
            .storage_mut()
            .reader()
            .map_err(|e| anyhow::anyhow!("pile reader: {e:?}"))?;
        repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;
        return Ok((branch, tribles, reader));
    }
    anyhow::bail!("no 'mary' or 'main' branch with commits in {pile_path:?}")
}

/// Index a pile's TYPED tensor leaves by name, without materializing any of
/// them.
///
/// The typed twin of [`load_keymap_from_pile`], and the difference is the whole
/// point: that one returns `(Vec<f32>, Vec<usize>)` per tensor — every weight in
/// the model, in RAM, as owned copies — while this returns views over the
/// pile's mapping. Loading becomes a decision the caller makes per tensor
/// rather than a cost paid up front for all of them.
///
/// Returns an empty map for a pile with no typed leaves (i.e. one not yet
/// converted); callers should treat empty as "not a typed pile" and fall back.
pub fn load_typed_keymap_from_pile(
    pile_path: &Path,
) -> anyhow::Result<std::collections::HashMap<String, crate::leaf::Leaf>> {
    let (_, tribles, reader) = checkout_any_branch(pile_path)?;
    crate::leaf::index_by_name(&tribles, &reader)
}

/// Reconstruct a SentencePiece UNIGRAM tokenizer from a pile's tokenizer graph.
/// The `.model` file is not needed — the pieces ARE the model.
///
/// NOTE: this mirrors [`load_tokenizer_from_pile`]'s pile-opening prologue
/// rather than sharing it. Factoring the two would need a callback trait,
/// because the blob reader is an associated type and `BlobStoreGet` is not
/// dyn-compatible — more machinery than the ~25 duplicated lines are worth,
/// and it would put the proven reader at risk for no behavioural gain.
#[cfg(feature = "qwen3tts")]
pub fn load_spm_tokenizer_from_pile(
    pile_path: &Path,
) -> anyhow::Result<crate::models::personaplex::spm::SpmTokenizer> {
    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("pile {pile_path:?} failed to load ({e:?})"))?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = repo
        .lookup_branch("main")
        .map_err(|e| anyhow::anyhow!("lookup main: {e:?}"))?
        .ok_or_else(|| anyhow::anyhow!("no 'main' branch in pile {pile_path:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;
    let head = ws
        .head()
        .ok_or_else(|| anyhow::anyhow!("'main' has no commits"))?;
    let checkout = ws
        .checkout(ancestors(head))
        .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
    let tribles: TribleSet = pre_epoch_aliased(checkout.facts());
    let reader = repo
        .storage_mut()
        .reader()
        .map_err(|e| anyhow::anyhow!("pile reader: {e:?}"))?;
    // Close BEFORE anything fallible: the reader keeps its own mmap alive, so
    // it outlives the repository, and every bail below would otherwise drop the
    // pile unclosed ("data may not be persisted").
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    let tok_id = crate::selection::select_tokenizer_root(
        &tribles,
        &reader,
        crate::selection::TokenizerSelector::Only,
    )
    .with_context(|| format!("select SentencePiece tokenizer in pile {pile_path:?}"))?;
    let pieces = crate::tokenizer::load_spm_pieces(&tribles, &reader, tok_id);
    if pieces.is_empty() {
        anyhow::bail!("tokenizer graph in {pile_path:?} has no scored pieces — not UNIGRAM?");
    }
    let adp = crate::tokenizer::has_add_prefix_space(&tribles, tok_id);
    Ok(crate::models::personaplex::spm::SpmTokenizer::from_pieces(
        &pieces, adp,
    ))
}

/// Ingest a SentencePiece `.model` file into a pile as a tokenizer GRAPH — the
/// write side of [`load_spm_tokenizer_from_pile`].
///
/// The `.proto` is parsed and DISCARDED: what lands in the pile is one entity
/// per piece (bytes, id, score, type tag) plus a tokenizer node, not the
/// original file as an opaque blob. That is the whole point — a blob would
/// persist the tokenizer, but only the graph makes it *queryable*, and only the
/// graph can be diffed, merged, or partially reused the way every other fact in
/// the pile can.
///
/// Refuses to write a second tokenizer into a pile that already has one:
/// `find_tokenizer` returns a single node, so two would make which-one-you-get
/// depend on iteration order.
#[cfg(feature = "qwen3tts")]
pub fn ingest_spm_tokenizer(
    pile_path: &Path,
    model_file: &Path,
    source_name: &str,
) -> anyhow::Result<usize> {
    let (pieces, add_dummy_prefix, byte_fallback) =
        crate::models::personaplex::spm::SpmTokenizer::parse_model(model_file);
    if pieces.is_empty() {
        anyhow::bail!("{model_file:?} parsed to zero pieces");
    }

    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("pile {pile_path:?} failed to load ({e:?})"))?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = match repo
        .lookup_branch("main")
        .map_err(|e| anyhow::anyhow!("lookup main: {e:?}"))?
    {
        Some(id) => id,
        None => *repo
            .create_branch("main", None)
            .map_err(|e| anyhow::anyhow!("create main: {e:?}"))?,
    };
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;

    // A pile is append-only, so a duplicate tokenizer cannot be taken back.
    // Close before bailing — an early return here would drop the pile unclosed.
    let existing = match ws.head() {
        Some(head) => {
            let checkout = ws
                .checkout(ancestors(head))
                .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
            crate::tokenizer::find_tokenizer(checkout.facts())
        }
        None => None,
    };
    if let Some(existing) = existing {
        repo.close()
            .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
        anyhow::bail!(
            "pile {pile_path:?} already contains tokenizer {existing:?}; \
             refusing to add a second (a pile is append-only — this cannot be undone)"
        );
    }

    let frag = crate::tokenizer::save_spm_unigram(
        &pieces,
        add_dummy_prefix,
        byte_fallback,
        source_name,
        repo.storage_mut(),
    )
    .map_err(|e| anyhow::anyhow!("build tokenizer graph: {e}"))?;
    let facts = frag.into_facts();
    let n = facts.len();
    ws.commit(facts, "ingest SentencePiece UNIGRAM tokenizer graph");
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok(n)
}

/// Construct a ready-to-encode `tokenizers::Tokenizer` from the tokenizer
/// GRAPH in a pile — the HuggingFace (BPE/WordPiece) counterpart to
/// [`load_spm_tokenizer_from_pile`].
#[cfg(feature = "tokenizer")]
pub fn load_tokenizer_from_pile(pile_path: &Path) -> anyhow::Result<tokenizers::Tokenizer> {
    load_tokenizer_from_pile_on(pile_path, "main")
}

/// [`load_tokenizer_from_pile`] on a NAMED branch.
///
/// A model pile does not have to keep its facts on `main` — Inkling's weights,
/// and now its tokenizer, live on `inkling` — and a loader that can only read
/// one branch name forces a second pile for the tokenizer, which is the
/// side-file problem this module exists to remove.
#[cfg(feature = "tokenizer")]
pub fn load_tokenizer_from_pile_on(
    pile_path: &Path,
    branch: &str,
) -> anyhow::Result<tokenizers::Tokenizer> {
    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    // Read path: non-mutating load, NEVER amputate (see load_keymap_from_pile).
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "pile {pile_path:?} failed to load ({e:?}); refusing to auto-truncate on a \
             read path — if the tail is a genuinely torn write, amputate explicitly \
             with `trible pile amputate`"
        )
    })?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = repo
        .lookup_branch(branch)
        .map_err(|e| anyhow::anyhow!("lookup {branch}: {e:?}"))?
        .ok_or_else(|| anyhow::anyhow!("no {branch:?} branch in pile {pile_path:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull {branch}: {e:?}"))?;
    let head = ws
        .head()
        .ok_or_else(|| anyhow::anyhow!("{branch:?} has no commits"))?;
    let checkout = ws
        .checkout(ancestors(head))
        .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
    let tribles: TribleSet = pre_epoch_aliased(checkout.facts());
    let reader = repo
        .storage_mut()
        .reader()
        .map_err(|e| anyhow::anyhow!("pile reader: {e:?}"))?;

    let tok = crate::selection::load_tokenizer_from_graph(
        &tribles,
        &reader,
        crate::selection::TokenizerSelector::Only,
    )
    .with_context(|| format!("select tokenizer on {branch:?} in pile {pile_path:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok(tok)
}

/// What one tokenizer ingest cost, so the rate can be reported rather than
/// guessed at.
#[cfg(feature = "tokenizer")]
#[derive(Debug, Clone, Copy)]
pub struct TokenizerIngest {
    pub facts: usize,
    /// Blob puts the ingest issued. Counted at the store, not derived from the
    /// JSON's shape.
    pub puts: u64,
    /// Nanoseconds spent inside those puts.
    pub put_nanos: u64,
    /// Nanoseconds for the whole ingest, puts included.
    pub total_nanos: u64,
    /// Bytes the pile file grew by. The interesting comparison is against the
    /// ~2 MB of distinct text a tokenizer actually is: a V3 record is a
    /// 256-byte header plus data padded to a 256-byte multiple, so a vocabulary
    /// of short strings costs far more in framing than in content, and knowing
    /// which of the two is the cost decides whether there is anything to fix.
    pub file_growth: u64,
}

#[cfg(feature = "tokenizer")]
impl TokenizerIngest {
    /// Puts per second measured against the time actually spent putting.
    pub fn put_rate(&self) -> Option<f64> {
        match self.put_nanos {
            0 => None,
            n => Some(self.puts as f64 * 1e9 / n as f64),
        }
    }
}

/// Ingest a HuggingFace `tokenizer.json` into a pile as a tokenizer GRAPH — the
/// write side of [`load_tokenizer_from_pile_on`], and the BPE/WordPiece
/// counterpart of [`ingest_spm_tokenizer`].
///
/// `save_tokenizer_json` has existed and been tested since 2026-07-16 and has
/// never had a caller that writes to disk; this is it. The gap mattered: an
/// in-memory `MemoryBlobStore` answers a put in about the time it takes to
/// hash, and a pile answers it by appending a record, so every performance
/// claim about ingest made against the in-memory path was about a different
/// operation.
///
/// Refuses to write a second tokenizer onto a branch that already has one: a
/// pile is append-only, `find_tokenizer` returns a single node, and two would
/// make which-one-you-get depend on iteration order.
#[cfg(feature = "tokenizer")]
pub fn ingest_hf_tokenizer(
    pile_path: &Path,
    tokenizer_json: &Path,
    source_name: &str,
    branch: &str,
) -> anyhow::Result<TokenizerIngest> {
    let json = std::fs::read(tokenizer_json)
        .map_err(|e| anyhow::anyhow!("read {tokenizer_json:?}: {e}"))?;
    let before = std::fs::metadata(pile_path).map(|m| m.len()).unwrap_or(0);

    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("pile {pile_path:?} failed to load ({e:?})"))?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = repo
        .ensure_branch(branch, None)
        .map_err(|e| anyhow::anyhow!("ensure {branch}: {e:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull {branch}: {e:?}"))?;

    let existing = match ws.head() {
        Some(head) => {
            let checkout = ws
                .checkout(ancestors(head))
                .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
            crate::tokenizer::find_tokenizer(checkout.facts())
        }
        None => None,
    };
    if let Some(existing) = existing {
        // Close before bailing: an early return would drop the pile unclosed.
        repo.close()
            .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
        anyhow::bail!(
            "pile {pile_path:?} branch {branch:?} already contains tokenizer \
             {existing:?}; refusing to add a second (a pile is append-only — \
             this cannot be undone)"
        );
    }

    let t0 = std::time::Instant::now();
    // Straight into the pile's own store, not the workspace's staging one: the
    // point of the measurement is the on-disk put.
    let mut counting = crate::tokenizer::CountingBlobs::new(repo.storage_mut());
    let frag = crate::tokenizer::save_tokenizer_json(&json, source_name, &mut counting)
        .map_err(|e| anyhow::anyhow!("build tokenizer graph: {e}"))?;
    let (puts, put_nanos) = (counting.puts, counting.nanos);
    let total_nanos = t0.elapsed().as_nanos() as u64;

    let facts = frag.into_facts();
    let n = facts.len();
    ws.commit(facts, "ingest HuggingFace BPE tokenizer graph");
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;

    let after = std::fs::metadata(pile_path)
        .map(|m| m.len())
        .unwrap_or(before);
    Ok(TokenizerIngest {
        facts: n,
        puts,
        put_nanos,
        total_nanos,
        file_growth: after.saturating_sub(before),
    })
}

/// A branch's facts and a reader for the blobs they name.
///
/// The prologue every pile reader in this file repeats, offered once. Note the
/// order: the repository is CLOSED before the pair is returned, because the
/// reader holds its own mapping and outlives it — and a bail before the close
/// leaves the pile "unclosed", which is a warning nobody reads and a habit that
/// eventually loses a write.
pub fn pile_facts(
    pile_path: &Path,
    branch: &str,
) -> anyhow::Result<(TribleSet, triblespace::core::repo::pile::PileReader)> {
    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    // Read path: never amputate. A torn tail is an operator decision.
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("pile {pile_path:?} failed to load ({e:?})"))?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = repo
        .lookup_branch(branch)
        .map_err(|e| anyhow::anyhow!("lookup {branch}: {e:?}"))?
        .ok_or_else(|| anyhow::anyhow!("no {branch:?} branch in pile {pile_path:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull {branch}: {e:?}"))?;
    let head = ws
        .head()
        .ok_or_else(|| anyhow::anyhow!("{branch:?} has no commits"))?;
    let facts: TribleSet = ws
        .checkout(ancestors(head))
        .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
        .facts()
        .clone();
    let reader = repo
        .storage_mut()
        .reader()
        .map_err(|e| anyhow::anyhow!("pile reader: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok((facts, reader))
}

/// Ingest a checkpoint's JSON sidecars into a pile as facts.
///
/// `json_docs` are parsed; `text_docs` are stored as documents whose root is a
/// JSON string, so there is one storage mechanism rather than two.
///
/// Idempotent by construction rather than by a skip list: a document node's id
/// derives from `(tag, name, root)` and the root's from its content, so
/// re-ingesting the same file yields the same entity and merging it is a no-op.
/// What is NOT harmless is ingesting a DIFFERENT `config.json` under the same
/// name — two documents, and which one a reader gets depends on iteration
/// order — so that is refused rather than appended.
#[cfg(feature = "tokenizer")]
pub fn ingest_json_documents(
    pile_path: &Path,
    dir: &Path,
    json_docs: &[&str],
    text_docs: &[&str],
    branch: &str,
) -> anyhow::Result<usize> {
    let mut pending: Vec<(String, serde_json::Value)> = Vec::new();
    for name in json_docs {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| anyhow::anyhow!("read {path:?}: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("parse {path:?}: {e}"))?;
        pending.push((name.to_string(), v));
    }
    for name in text_docs {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| anyhow::anyhow!("read {path:?}: {e}"))?;
        pending.push((name.to_string(), serde_json::Value::String(text)));
    }
    anyhow::ensure!(!pending.is_empty(), "no sidecars found in {dir:?}");

    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("pile {pile_path:?} failed to load ({e:?})"))?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = repo
        .ensure_branch(branch, None)
        .map_err(|e| anyhow::anyhow!("ensure {branch}: {e:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull {branch}: {e:?}"))?;

    // What the branch already says, compared by VALUE rather than by node id.
    // Re-ingesting the identical file is silent (the ids derive from the
    // content, so the facts merge to nothing new); a file whose CONTENT changed
    // is refused, because a second document under the same name makes which one
    // a reader gets depend on iteration order.
    let mut clash: Option<String> = None;
    if let Some(head) = ws.head() {
        let checkout = ws
            .checkout(ancestors(head))
            .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
        let facts = checkout.facts().clone();
        let reader = repo
            .storage_mut()
            .reader()
            .map_err(|e| anyhow::anyhow!("reader: {e:?}"))?;
        for (name, v) in &pending {
            if let Ok(have) = crate::jsonfacts::load_document(&facts, &reader, name) {
                if &have != v {
                    clash = Some(name.clone());
                    break;
                }
            }
        }
    }
    if let Some(name) = clash {
        repo.close()
            .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
        anyhow::bail!(
            "pile {pile_path:?} branch {branch:?} already holds a DIFFERENT \
             document named {name:?}; a pile is append-only, so writing a \
             second would make which one a reader gets depend on iteration \
             order"
        );
    }

    let mut change = TribleSet::new();
    for (name, v) in &pending {
        if let Err(e) = crate::jsonfacts::save_document(name, v, repo.storage_mut(), &mut change) {
            repo.close()
                .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
            anyhow::bail!("{name}: {e}");
        }
    }

    let n = change.len();
    ws.commit(change, "ingest checkpoint JSON sidecars as facts");
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok(n)
}

#[cfg(feature = "gemma")]
fn select_native_model_index(
    pile_path: &Path,
    selector: crate::selection::ModelSelector<'_>,
) -> anyhow::Result<crate::selection::SelectedModelIndex<triblespace::core::repo::pile::PileReader>>
{
    let team = crate::model_collection::model_graph_team_at(pile_path)?;
    let snapshot = crate::model_collection::load_model_collection_local_latest(pile_path, team)
        .with_context(|| format!("load local-latest native model snapshot from {pile_path:?}"))?;
    crate::selection::SelectedModelIndex::from_snapshot(snapshot, selector)
        .with_context(|| format!("select one native model root in {pile_path:?}"))
}

/// Stream a Gemma 4 model from one already-selected native model index: load
/// each tensor on demand and drop it after upload, so peak CPU is one tensor
/// rather than the whole f32 keymap. Each leaf keeps the pile's mapping alive
/// for as long as the index holds it, so the build needs no separate reader.
#[cfg(feature = "gemma")]
pub fn load_gemma4_streaming_from_index<
    B: burn::prelude::Backend,
    R: triblespace::prelude::BlobStoreGet,
>(
    selected: crate::selection::SelectedModelIndex<R>,
    config: crate::models::gemma::gemma4::config::Gemma4Config,
    device: &B::Device,
) -> (
    crate::models::gemma::gemma4::decoder::Gemma4Model<B>,
    Option<crate::models::gemma::gemma4::vision::Gemma4VisionEncoder<B>>,
) {
    let (_, index, _reader) = selected.into_parts();
    crate::models::gemma::gemma4::weights::load_gemma4_streaming::<B>(config, index, device)
}

/// Local-latest path convenience for [`load_gemma4_streaming_from_index`].
/// The caller selects exactly one model root from the native collection.
#[cfg(feature = "gemma")]
pub fn load_gemma4_streaming_from_pile<B: burn::prelude::Backend>(
    pile_path: &Path,
    selector: crate::selection::ModelSelector<'_>,
    config: crate::models::gemma::gemma4::config::Gemma4Config,
    device: &B::Device,
) -> anyhow::Result<(
    crate::models::gemma::gemma4::decoder::Gemma4Model<B>,
    Option<crate::models::gemma::gemma4::vision::Gemma4VisionEncoder<B>>,
)> {
    Ok(load_gemma4_streaming_from_index(
        select_native_model_index(pile_path, selector)?,
        config,
        device,
    ))
}

/// The full HEARING stack from one selected native model index: text decoder
/// (+vision when present), audio tower, and multimodal embedder.
#[cfg(feature = "gemma")]
pub fn load_gemma4_hearing_from_index<
    B: burn::prelude::Backend,
    R: triblespace::prelude::BlobStoreGet,
>(
    selected: crate::selection::SelectedModelIndex<R>,
    config: crate::models::gemma::gemma4::config::Gemma4Config,
    device: &B::Device,
) -> anyhow::Result<(
    crate::models::gemma::gemma4::decoder::Gemma4Model<B>,
    Option<crate::models::gemma::gemma4::vision::Gemma4VisionEncoder<B>>,
    crate::models::gemma::gemma4::audio::AudioModel<B>,
    crate::models::gemma::gemma4::audio::AudioEmbedder<B>,
)> {
    let audio_cfg = config.audio_config.clone().ok_or_else(|| {
        anyhow::anyhow!("config has no audio_config — this checkpoint has no stt")
    })?;
    let (_, index, _reader) = selected.into_parts();
    let fetch = |name: &str| index.get(name).map(crate::leaf::Leaf::to_f32_shape);
    let tower = crate::models::gemma::gemma4::audio::AudioModel::<B>::load_with(
        audio_cfg.clone(),
        &fetch,
        device,
    );
    let embedder = crate::models::gemma::gemma4::audio::AudioEmbedder::<B>::load_with(
        &fetch,
        audio_cfg.rms_norm_eps,
        device,
    );
    let (model, vision) =
        crate::models::gemma::gemma4::weights::load_gemma4_streaming::<B>(config, index, device);
    Ok((model, vision, tower, embedder))
}

/// Local-latest path convenience for [`load_gemma4_hearing_from_index`]. The
/// caller selects exactly one model root from the native collection.
#[cfg(feature = "gemma")]
pub fn load_gemma4_hearing_from_pile<B: burn::prelude::Backend>(
    pile_path: &Path,
    selector: crate::selection::ModelSelector<'_>,
    config: crate::models::gemma::gemma4::config::Gemma4Config,
    device: &B::Device,
) -> anyhow::Result<(
    crate::models::gemma::gemma4::decoder::Gemma4Model<B>,
    Option<crate::models::gemma::gemma4::vision::Gemma4VisionEncoder<B>>,
    crate::models::gemma::gemma4::audio::AudioModel<B>,
    crate::models::gemma::gemma4::audio::AudioEmbedder<B>,
)> {
    load_gemma4_hearing_from_index(
        select_native_model_index(pile_path, selector)?,
        config,
        device,
    )
}

/// Just the STT stack from one selected native model index: audio tower plus
/// multimodal embedder, without the decoder.
#[cfg(feature = "gemma")]
pub fn load_gemma4_audio_from_index<
    B: burn::prelude::Backend,
    R: triblespace::prelude::BlobStoreGet,
>(
    selected: crate::selection::SelectedModelIndex<R>,
    audio_cfg: crate::models::gemma::gemma4::config::Gemma4AudioConfig,
    device: &B::Device,
) -> (
    crate::models::gemma::gemma4::audio::AudioModel<B>,
    crate::models::gemma::gemma4::audio::AudioEmbedder<B>,
) {
    let (_, index, _reader) = selected.into_parts();
    let fetch = |name: &str| index.get(name).map(crate::leaf::Leaf::to_f32_shape);
    let tower = crate::models::gemma::gemma4::audio::AudioModel::<B>::load_with(
        audio_cfg.clone(),
        &fetch,
        device,
    );
    let embedder = crate::models::gemma::gemma4::audio::AudioEmbedder::<B>::load_with(
        &fetch,
        audio_cfg.rms_norm_eps,
        device,
    );
    (tower, embedder)
}

/// Local-latest path convenience for [`load_gemma4_audio_from_index`]. The
/// caller selects exactly one model root from the native collection.
#[cfg(feature = "gemma")]
pub fn load_gemma4_audio_from_pile<B: burn::prelude::Backend>(
    pile_path: &Path,
    selector: crate::selection::ModelSelector<'_>,
    audio_cfg: crate::models::gemma::gemma4::config::Gemma4AudioConfig,
    device: &B::Device,
) -> anyhow::Result<(
    crate::models::gemma::gemma4::audio::AudioModel<B>,
    crate::models::gemma::gemma4::audio::AudioEmbedder<B>,
)> {
    Ok(load_gemma4_audio_from_index(
        select_native_model_index(pile_path, selector)?,
        audio_cfg,
        device,
    ))
}

/// Load a Gemma 4 model from one selected native model index with zero-copy
/// weights. Each mmap-backed f16 blob is aliased onto Metal, and each GPU buffer
/// retains its own mmap keepalive after the selected index is consumed.
///
/// Unlike the streaming loaders, this accepts the concrete
/// [`PileReader`](triblespace::core::repo::pile::PileReader) capability: an
/// arbitrary [`BlobStoreGet`] may return owned bytes that cannot be registered
/// as an external Metal buffer.
#[cfg(all(feature = "gemma", target_os = "macos"))]
pub fn load_gemma4_aliased_from_index(
    selected: crate::selection::SelectedModelIndex<triblespace::core::repo::pile::PileReader>,
    config: crate::models::gemma::gemma4::config::Gemma4Config,
    device: burn::backend::wgpu::WgpuDevice,
) -> anyhow::Result<crate::models::gemma::gemma4::decoder::Gemma4Model<crate::nn::backend::BHalf>> {
    use crate::models::gemma::gemma4::weights::{load_gemma4_from_source, WeightCtx};
    use crate::nn::backend::BHalf;
    use burn::backend::wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
    use burn::tensor::{DType, Tensor, TensorPrimitive};
    use cubecl::Runtime;
    use memmap2::MmapRaw;
    use std::sync::Arc;
    const PAGE: u64 = 16384;

    require_f16_model_index(&selected)?;
    let (_, index, _reader) = selected.into_parts();

    let client = WgpuRuntime::client(&device);
    let ctx = WeightCtx::<BHalf> {
        has: Box::new(|name: &str| index.contains_key(name)),
        get: Box::new(|name: &str, device: &WgpuDevice| {
            let leaf = index.get(name)?;
            if leaf.elem() != crate::leaf::Elem::F16 {
                return None;
            }
            let shape = leaf.shape();
            let bytes: anybytes::Bytes = leaf.payload().clone();
            let blob_ptr = bytes.as_ptr() as u64;
            let nbytes = bytes.len() as u64;
            let n = (nbytes / 2) as usize; // f16 element count
                                           // The owner downcast = capability check (mmap?) + region bounds + keepalive.
            let mmap = bytes.downcast_to_owner::<MmapRaw>().ok()?;
            let region_end = mmap.as_ptr() as u64 + mmap.len() as u64;
            let page_start = blob_ptr & !(PAGE - 1);
            let off_in_page = blob_ptr - page_start;
            let page_len =
                ((blob_ptr + nbytes + PAGE - 1) & !(PAGE - 1)).min(region_end) - page_start;
            let keepalive: Arc<dyn std::any::Any + Send + Sync> = mmap.clone();
            // SAFETY: page_start/page_len is a page-aligned superset of the blob,
            // inside the (page-aligned) mmap which `keepalive` pins for the buffer's life.
            let handle = unsafe {
                client.register_external_aliased(
                    page_start as *mut core::ffi::c_void,
                    page_len,
                    off_in_page,
                    nbytes,
                    keepalive,
                )
            };
            let cube = CubeTensor::<WgpuRuntime>::new_contiguous(
                client.clone(),
                device.clone(),
                [n].into(),
                handle,
                DType::F16,
            );
            Some((
                Tensor::<BHalf, 1>::from_primitive(TensorPrimitive::Float(cube)),
                shape,
            ))
        }),
        raw: None,
    };
    let (model, _vision) = load_gemma4_from_source::<BHalf>(config, &ctx, &device);
    drop(ctx);
    Ok(model)
}

/// Local-latest path convenience for [`load_gemma4_aliased_from_index`]. The
/// caller selects exactly one native f16 model root from the collection.
#[cfg(all(feature = "gemma", target_os = "macos"))]
pub fn load_gemma4_aliased_from_pile(
    pile_path: &Path,
    selector: crate::selection::ModelSelector<'_>,
    config: crate::models::gemma::gemma4::config::Gemma4Config,
    device: burn::backend::wgpu::WgpuDevice,
) -> anyhow::Result<crate::models::gemma::gemma4::decoder::Gemma4Model<crate::nn::backend::BHalf>> {
    load_gemma4_aliased_from_index(
        select_native_model_index(pile_path, selector)?,
        config,
        device,
    )
}

/// Alias ONE f16 tensor leaf's mmap'd pile blob straight onto the Metal GPU —
/// no copy, no f32 materialization. Returns the flat `[n]` `BHalf` tensor (the
/// caller reshapes to the weight's rank) plus the stored shape. Mirrors the
/// per-tensor body of [`load_gemma4_aliased_from_pile`]; factored out so the
/// Qwen2.5-VL `QwenWeights`/`VisionWeights` aliasing source can reuse it.
#[cfg(all(feature = "gemma", target_os = "macos"))]
fn alias_f16_leaf(
    leaf: &crate::leaf::Leaf,
    device: &burn::backend::wgpu::WgpuDevice,
) -> (burn::tensor::Tensor<crate::nn::backend::B, 1>, Vec<usize>) {
    use crate::nn::backend::B;
    use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
    use burn::tensor::{DType, Tensor, TensorPrimitive};
    use cubecl::Runtime;
    use memmap2::MmapRaw;
    use std::sync::Arc;
    const PAGE: u64 = 16384;

    assert!(
        leaf.elem() == crate::leaf::Elem::F16,
        "aliased path requires f16 leaves; found f32"
    );
    let shape = leaf.shape();
    let bytes: anybytes::Bytes = leaf.payload().clone();
    let blob_ptr = bytes.as_ptr() as u64;
    let nbytes = bytes.len() as u64;
    let n = (nbytes / 2) as usize; // f16 element count
                                   // The owner downcast = capability check (mmap?) + region bounds + keepalive.
    let mmap = bytes
        .downcast_to_owner::<MmapRaw>()
        .expect("aliased path requires an mmap-backed pile blob");
    let region_end = mmap.as_ptr() as u64 + mmap.len() as u64;
    let page_start = blob_ptr & !(PAGE - 1);
    let off_in_page = blob_ptr - page_start;
    let page_len = ((blob_ptr + nbytes + PAGE - 1) & !(PAGE - 1)).min(region_end) - page_start;
    let keepalive: Arc<dyn std::any::Any + Send + Sync> = mmap.clone();
    let client = WgpuRuntime::client(device);
    // SAFETY: page_start/page_len is a page-aligned superset of the blob, inside
    // the (page-aligned) mmap which `keepalive` pins for the buffer's life.
    let handle = unsafe {
        client.register_external_aliased(
            page_start as *mut core::ffi::c_void,
            page_len,
            off_in_page,
            nbytes,
            keepalive,
        )
    };
    let cube = CubeTensor::<WgpuRuntime>::new_contiguous(
        client.clone(),
        device.clone(),
        [n].into(),
        handle,
        DType::F16,
    );
    (
        Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(cube)),
        shape,
    )
}

/// A zero-copy, aliased-from-pile [`QwenWeights`]/[`VisionWeights`] source. Each
/// `t1`/`t2`/`patch_proj` aliases the requested f16 blob's mmap straight onto the
/// Metal GPU (via [`alias_f16_leaf`], an `F16`-dtype tensor) and reshapes — the
/// Metal-specific counterpart of the test harnesses' `KeymapW`. The weights stay
/// f16 in GPU memory (zero-copy); the model upcasts them per-op so activations
/// run in f32 (this bf16-native model overflows f16's range — see
/// `QwenRmsNorm`/`Linear`). Names resolve EXACTLY (the merge scripts already
/// strip the `model.` prefix to QwenTextModel naming), so no prefix munging.
#[cfg(all(feature = "gemma", target_os = "macos"))]
struct AliasedQwenWeights<'a> {
    index: &'a HashMap<String, crate::leaf::Leaf>,
    device: burn::backend::wgpu::WgpuDevice,
}

#[cfg(all(feature = "gemma", target_os = "macos"))]
impl<'a> AliasedQwenWeights<'a> {
    fn flat(&self, name: &str) -> (burn::tensor::Tensor<crate::nn::backend::B, 1>, Vec<usize>) {
        let leaf = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("missing weight {name} in pile index"));
        alias_f16_leaf(leaf, &self.device)
    }
}

#[cfg(all(feature = "gemma", target_os = "macos"))]
impl<'a> crate::models::qwen2_5_vl::layers::QwenWeights<crate::nn::backend::B>
    for AliasedQwenWeights<'a>
{
    fn t1(&self, name: &str) -> burn::tensor::Tensor<crate::nn::backend::B, 1> {
        self.flat(name).0
    }
    fn t2(&self, name: &str) -> burn::tensor::Tensor<crate::nn::backend::B, 2> {
        let (t, s) = self.flat(name);
        t.reshape([s[0], s[1]])
    }
}

#[cfg(all(feature = "gemma", target_os = "macos"))]
impl<'a> crate::models::qwen2_5_vl::vision::VisionWeights<crate::nn::backend::B>
    for AliasedQwenWeights<'a>
{
    fn t1(&self, name: &str) -> burn::tensor::Tensor<crate::nn::backend::B, 1> {
        self.flat(name).0
    }
    fn t2(&self, name: &str) -> burn::tensor::Tensor<crate::nn::backend::B, 2> {
        let (t, s) = self.flat(name);
        t.reshape([s[0], s[1]])
    }
    fn patch_proj(
        &self,
        name: &str,
        embed: usize,
        in_flat: usize,
    ) -> burn::tensor::Tensor<crate::nn::backend::B, 2> {
        self.flat(name).0.reshape([embed, in_flat])
    }
}

#[cfg(all(feature = "gemma", target_os = "macos"))]
fn require_f16_model_index<R>(
    selected: &crate::selection::SelectedModelIndex<R>,
) -> anyhow::Result<()> {
    // The aliased Metal ABI is f16. Reject a structurally valid but incompatible
    // native import before model construction can turn it into a distant panic.
    if let Some(name) = selected
        .handles()
        .iter()
        .filter_map(|(name, leaf)| (leaf.elem() == crate::leaf::Elem::F32).then_some(name))
        .min()
    {
        anyhow::bail!(
            "model root {} is not an aliased-f16 model: tensor {name:?} has an f32 leaf",
            selected.root()
        );
    }
    Ok(())
}

/// Load `nomic-embed-multimodal-7b` (Qwen2.5-VL backbone + vision tower) from
/// one already-frozen native model-collection snapshot and an explicit model
/// selector.
///
/// Every selected f16 tensor blob is aliased straight from the snapshot's mmap
/// onto the Metal GPU (no copy, no f32 materialization). Each GPU buffer clones
/// the mmap owner into `register_external_aliased`'s keepalive, so the temporary
/// key index and [`PileReader`](triblespace::core::repo::pile::PileReader) may
/// be dropped after construction while the mappings remain valid for the
/// embedder's life. Weights stay f16 in GPU memory; activations run in f32.
///
/// Snapshot acquisition and admission policy are deliberately caller-owned.
/// This constructor neither opens storage nor falls back to a Repository
/// branch, and ambiguous or incompatible model selections fail closed.
#[cfg(all(feature = "gemma", target_os = "macos"))]
pub fn load_nomic_mm7b_aliased_from_snapshot(
    snapshot: CollectionSnapshot<triblespace::core::repo::pile::PileReader>,
    selector: crate::selection::ModelSelector<'_>,
    tokenizer_path: &Path,
    device: burn::backend::wgpu::WgpuDevice,
) -> anyhow::Result<
    crate::models::qwen2_5_vl::embedder::NomicMultimodalEmbedder<crate::nn::backend::B>,
> {
    use crate::models::qwen2_5_vl::embedder::NomicMultimodalEmbedder;
    use crate::nn::backend::B;

    let selected = crate::selection::SelectedModelIndex::from_snapshot(snapshot, selector)?;
    require_f16_model_index(&selected)?;
    let (_, index, reader) = selected.into_parts();

    let weights = AliasedQwenWeights {
        index: &index,
        device: device.clone(),
    };
    let embedder =
        NomicMultimodalEmbedder::<B>::load_with_vision(&weights, tokenizer_path, device)?;
    drop(weights);
    // Every constructed GPU tensor owns an Arc to the mmap region it aliases;
    // dropping this reader cannot invalidate the returned embedder.
    drop(reader);
    Ok(embedder)
}

#[cfg(all(test, feature = "gemma", target_os = "macos"))]
mod native_model_snapshot_tests {
    use super::*;
    use crate::format::attrs;
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use triblespace::prelude::blobencodings::LongString;

    static NEXT_TEST_PILE: AtomicU64 = AtomicU64::new(0);

    struct TestPile {
        path: PathBuf,
    }

    impl TestPile {
        fn new(label: &str) -> Self {
            let ordinal = NEXT_TEST_PILE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mary-native-model-{label}-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create isolated model pile");
            Self { path }
        }
    }

    impl Drop for TestPile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn add_model(
        pile: &mut Pile,
        signing_key: &SigningKey,
        source: &str,
        tensor_name: &str,
        value: f32,
        f16: bool,
    ) {
        let leaf = if f16 {
            crate::format::put_raw_f16(pile, &[value], &[1]).unwrap()
        } else {
            crate::format::put_raw(pile, &[value], &[1]).unwrap()
        };
        let leaf_id = leaf.root().unwrap();
        let mut facts = leaf.into_facts();
        let name = pile.put::<LongString, _>(tensor_name.to_owned()).unwrap();
        let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: leaf_id };
        let member_id = member.root().unwrap();
        facts += member.into_facts();
        let root = entity! { _ @ attrs::member: member_id };
        let root_id = root.root().unwrap();
        facts += root.into_facts();
        let source = pile.put::<LongString, _>(source.to_owned()).unwrap();
        facts += entity! { ExclusiveId::force_ref(&root_id) @
            attrs::source: source,
            attrs::quantization: QUANTIZATION_NATIVE,
        }
        .into_facts();
        let team = signing_key.verifying_key();
        crate::model_collection::publish_model_fragment(
            pile,
            team,
            signing_key,
            Fragment::rooted(root_id, facts),
        )
        .unwrap();
    }

    fn open_test_pile(file: &TestPile) -> Pile {
        let mut pile = Pile::open(&file.path).unwrap();
        pile.refresh().unwrap();
        pile
    }

    #[test]
    fn native_snapshot_selection_keeps_the_reader_and_rejects_ambiguity() {
        let file = TestPile::new("selection");
        let mut pile = open_test_pile(&file);
        let signer = SigningKey::from_bytes(&[0xA1; 32]);
        add_model(
            &mut pile,
            &signer,
            "example/target",
            "target.weight",
            1.5,
            true,
        );
        add_model(
            &mut pile,
            &signer,
            "example/other",
            "other.weight",
            2.5,
            true,
        );
        pile.close().unwrap();

        let error =
            match select_native_model_index(&file.path, crate::selection::ModelSelector::Only) {
                Ok(_) => panic!("the Gemma path frontdoor merged two native roots"),
                Err(error) => format!("{error:#}"),
            };
        assert!(error.contains("ambiguous model root"), "{error}");

        let snapshot = crate::model_collection::load_model_collection_local_latest(
            &file.path,
            signer.verifying_key(),
        )
        .expect("native snapshot");
        let selected = crate::selection::SelectedModelIndex::from_snapshot(
            snapshot,
            crate::selection::ModelSelector::Source {
                source: "example/target",
                quantization: QUANTIZATION_NATIVE,
            },
        )
        .expect("strict source selection");
        require_f16_model_index(&selected).expect("selected model is f16");
        let (_, index, _reader) = selected.into_parts();
        assert_eq!(index.len(), 1);
        let leaf = &index["target.weight"];
        assert_eq!(leaf.elem(), crate::leaf::Elem::F16);
        // The leaf's bytes outlive the pile handle: its payload is a view over
        // the mapping and keeps it alive.
        assert_eq!(leaf.view_f16().expect("f16 view")[0].to_f32(), 1.5);
        assert_eq!(leaf.dims(), &[1]);

        let mut pile = open_test_pile(&file);
        add_model(
            &mut pile,
            &signer,
            "example/target",
            "second.weight",
            3.5,
            true,
        );
        pile.close().unwrap();
        let snapshot = crate::model_collection::load_model_collection_local_latest(
            &file.path,
            signer.verifying_key(),
        )
        .expect("ambiguous native snapshot");
        let error = match crate::selection::SelectedModelIndex::from_snapshot(
            snapshot,
            crate::selection::ModelSelector::Source {
                source: "example/target",
                quantization: QUANTIZATION_NATIVE,
            },
        ) {
            Ok(_) => panic!("ambiguous coordinates were accepted"),
            Err(error) => format!("{error:#}"),
        };
        assert!(error.contains("ambiguous model root"), "{error}");
    }

    #[test]
    fn aliased_snapshot_rejects_f32_before_model_construction() {
        let file = TestPile::new("f32-rejection");
        let mut pile = open_test_pile(&file);
        add_model(
            &mut pile,
            &SigningKey::from_bytes(&[0xB2; 32]),
            "example/f32",
            "f32.weight",
            4.0,
            false,
        );
        pile.close().unwrap();

        let selected = select_native_model_index(&file.path, crate::selection::ModelSelector::Only)
            .expect("the only native model root is selected exactly");
        let error = require_f16_model_index(&selected)
            .expect_err("f32 leaf was accepted by aliased constructor boundary")
            .to_string();
        assert!(
            error.contains("tensor \"f32.weight\" has an f32 leaf"),
            "{error}"
        );
    }
}
