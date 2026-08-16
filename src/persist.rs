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
use anyhow::Context;
use ed25519_dalek::SigningKey;
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

/// Import one model directory into Mary's native append-only model collection.
///
/// This is the library seam behind `mary import`. The caller owns the already
/// open pile, supplies a durable signing identity, and chooses the eventual
/// durability boundary. This function never creates, opens, flushes, or closes
/// storage, and it never creates or advances a Repository branch.
///
/// `source` is the model's canonical label — the HF id it was imported from, or
/// a `--name` for a local-dir import. The root id is content-derived from only
/// its weight members, so byte-identical weights converge independently of
/// source container or provenance. For a multi-shard model, the one root
/// composes every shard's tensor members as an order-independent set; source,
/// quantization, and shard names are queryable non-core coordinates on it.
///
/// `quantization` tags the weight format ("native" for the faithful import).
/// The return value is the imported root's entity id plus the complete signed
/// 192-byte [`CollectionCommit`] needed by an exact collection reader.
#[cfg(feature = "import")]
pub fn import_model_to_collection(
    pile: &mut Pile,
    signing_key: &SigningKey,
    model_dir: &Path,
    dtype: LeafDtype,
    source: &str,
    quantization: &str,
) -> anyhow::Result<(Id, CollectionCommit)> {
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
    for (path, name) in &files {
        let (mut shard_members, shard_facts) = match fmt {
            crate::formats::WeightFormat::Safetensors => {
                let bytes = read_safetensors_file(path);
                eprintln!(
                    "[persist] ingesting {name} ({} bytes, safetensors)...",
                    bytes.len()
                );
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
                crate::ingest::ingest_tensors(tensors.into_iter(), pile, dtype)
                    .map_err(|e| anyhow::anyhow!("ingest {path:?}: {e}"))?
            }
        };
        members.append(&mut shard_members);
        facts += shard_facts;
        provenance.push(name.clone());
    }
    let root =
        crate::ingest::build_model_root(pile, source, quantization, members, facts, &provenance)
            .map_err(|e| anyhow::anyhow!("build model root: {e}"))?;
    let root_id = root.root().expect("model root id");
    let commit = crate::model_collection::publish_model_fragment(pile, signing_key, root)
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
    anyhow::ensure!(!index.is_empty(), "cannot derive an empty f16 model root");
    for (name, handles) in &index {
        anyhow::ensure!(
            matches!(handles, crate::ingest::LeafHandles::F32(..)),
            "{name}: f16 derivation requires an exact f32 source root"
        );
    }

    let mut names: Vec<_> = index.keys().cloned().collect();
    names.sort_unstable();
    let mut members = Vec::with_capacity(names.len());
    let mut facts = TribleSet::new();
    let mut elements = 0;
    for (ordinal, name) in names.into_iter().enumerate() {
        let handles = index.remove(&name).expect("name collected from index");
        let (data_handle, shape_handle) = match handles {
            crate::ingest::LeafHandles::F32(data, shape) => (data, shape),
            crate::ingest::LeafHandles::F16(..) => unreachable!("validated exact width"),
        };
        let data_bytes: anybytes::Bytes = reader
            .get(data_handle)
            .map_err(|error| anyhow::anyhow!("read exact tensor {name:?}: {error}"))?;
        let data = data_bytes
            .view::<[f32]>()
            .with_context(|| format!("decode exact tensor {name:?}"))?
            .to_vec();
        let shape_bytes: anybytes::Bytes = reader
            .get(shape_handle)
            .map_err(|error| anyhow::anyhow!("read shape for exact tensor {name:?}: {error}"))?;
        let shape = shape_bytes
            .view::<[u64]>()
            .with_context(|| format!("decode shape for exact tensor {name:?}"))?
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension).with_context(|| {
                    format!("shape dimension {dimension} for exact tensor {name:?} exceeds usize")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
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
    let commit = crate::model_collection::publish_model_fragment(pile, signing_key, root)
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

    let bytes = std::fs::read(file).with_context(|| format!("read safetensors file {file:?}"))?;
    let (members, facts) = crate::ingest::ingest_members(&bytes, pile, dtype, keep)
        .map_err(|e| anyhow::anyhow!("ingest {file:?}: {e}"))?;
    anyhow::ensure!(
        !members.is_empty(),
        "filtered safetensors import selected no supported float tensors from {file:?}"
    );

    let provenance = file
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors")
        .to_owned();
    let root =
        crate::ingest::build_model_root(pile, source, quantization, members, facts, &[provenance])
            .map_err(|e| anyhow::anyhow!("build model root: {e}"))?;
    let root_id = root.root().expect("model root id");
    let commit = crate::model_collection::publish_model_fragment(pile, signing_key, root)
        .map_err(|error| anyhow::anyhow!("publish model collection commit: {error}"))?;
    Ok((root_id, commit))
}

#[cfg(all(test, feature = "import"))]
mod filtered_native_import_tests {
    use super::*;
    use crate::selection::ModelSelector;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
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

        let snapshot =
            crate::model_collection::snapshot_model_collection_exact(&mut pile, &[commit]).unwrap();
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
            crate::model_collection::load_model_collection_local_latest(&pile_path).unwrap();
        assert_eq!(latest.commits(), &[commit]);
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

/// Open a pile and build the CHEAP handle indexes for two families of model
/// entities — the ones whose name starts with `f16_prefix` (half-width leaves
/// for the fast native-width GPU load) and ALL OTHERS (the exact leaves) —
/// plus a long-lived
/// [`PileReader`](triblespace::core::repo::pile::PileReader) to resolve them
/// through. No tensor data is read here; the fast loader uploads straight
/// from the reader's mmap'd blobs, and the mmap stays valid after the
/// repository is closed (each blob keeps the mapping alive). The first index
/// comes back empty if no entity matches the prefix.
pub fn load_split_index_from_pile(
    pile_path: &Path,
    f16_prefix: &str,
) -> anyhow::Result<(
    HashMap<String, crate::ingest::LeafHandles>,
    HashMap<String, crate::ingest::LeafHandles>,
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
    let tribles: TribleSet = ws
        .checkout(ancestors(head))
        .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
        .facts()
        .clone();
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
    let (f16, f32_, reader) = load_split_index_from_pile(pile_path, f16_prefix)?;
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
                reader.clone(),
                reader,
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
        .map(|(k, h)| (k, crate::ingest::read_leaf(&reader, h)))
        .collect();
    Ok(WeightLoader::Pile(keymap))
}

/// Open a weights pile as the lazy handle-indexed runtime loader
/// ([`WeightLoader::Aliased`] — nothing materialized wholesale; tensors
/// resolve on demand through the pile mmap, and `view_f32` serves zero-copy
/// slices). The one loader every PersonaPlex probe and the realtime
/// pipeline share.
/// Non-macOS sibling of [`personaplex_loader`]. There is no Metal aliasing
/// seam off macOS, so weights are materialized through `WeightLoader::Pile`
/// instead: with an empty `f16_prefix` the split predicate routes every leaf to
/// the f32 side, which is exactly the fallback the macOS path already takes
/// when aliasing is refused. Slower to load, identical semantics.
#[cfg(all(feature = "qwen3tts", not(target_os = "macos")))]
pub fn personaplex_loader(
    pile_path: &Path,
) -> anyhow::Result<crate::nn::weight_loader::WeightLoader> {
    load_aliased_loader_from_pile(pile_path, "")
}

#[cfg(all(feature = "qwen3tts", target_os = "macos"))]
pub fn personaplex_loader(
    pile_path: &Path,
) -> anyhow::Result<crate::nn::weight_loader::WeightLoader> {
    let (f16, f32_, reader) = load_split_index_from_pile(pile_path, "")?;
    Ok(crate::nn::weight_loader::WeightLoader::Aliased(
        crate::nn::weight_loader::AliasedPile::new(
            f16,
            f32_,
            reader.clone(), // one union pile: both leaf families share the reader
            reader,
            crate::nn::backend::WgpuDevice::default(),
        ),
    ))
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
    let (_, folded, folded_reader) = load_split_index_from_pile(dst_pile, "")?;
    use triblespace::prelude::BlobStoreGet;
    for (name, bits, dims) in &tensors {
        let (dh, sh) = match folded.get(name.as_str()) {
            Some(crate::ingest::LeafHandles::F16(d, s)) => (*d, *s),
            other => anyhow::bail!(
                "{name}: bad folded leaf after derive ({})",
                if other.is_none() { "missing" } else { "f32" }
            ),
        };
        let got_dims: Vec<u64> = crate::ingest::read_shape(&folded_reader, sh)
            .iter()
            .map(|&d| d as u64)
            .collect();
        anyhow::ensure!(
            &got_dims == dims,
            "{name}: shape mismatch {got_dims:?} vs {dims:?}"
        );
        let blob: anybytes::Bytes = folded_reader
            .get(dh)
            .map_err(|e| anyhow::anyhow!("{name}: data blob: {e:?}"))?;
        let t = crate::nn::alias::alias_flat_raw::<half::f16>(
            blob,
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
    let (f16, f32_, reader) = load_split_index_from_pile(src_pile, "talker_f16")?;
    anyhow::ensure!(
        !f16.is_empty(),
        "no 'talker_f16' leaves in {src_pile:?} (append with qwen3tts_persist --f16-talker-only)"
    );
    let (_, folded, folded_reader) = load_split_index_from_pile(folded_pile, "")?;
    anyhow::ensure!(
        !folded.is_empty(),
        "no leaves in folded pile {folded_pile:?}"
    );
    load_qwen3tts_talker_folded_from_indexes(&f16, &f32_, &reader, &folded, &folded_reader)
}

/// Construct the raw f16 Qwen3-TTS talker from already-selected native model
/// indexes. Both readers may be the same frozen collection reader: the split is
/// semantic (base/talker/folded roots), not a storage or file boundary.
#[cfg(all(feature = "qwen3tts", target_os = "macos"))]
pub fn load_qwen3tts_talker_folded_from_indexes<R: BlobStoreGet, F: BlobStoreGet>(
    f16: &HashMap<String, crate::ingest::LeafHandles>,
    f32_: &HashMap<String, crate::ingest::LeafHandles>,
    reader: &R,
    folded: &HashMap<String, crate::ingest::LeafHandles>,
    folded_reader: &F,
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
    fn alias<R: BlobStoreGet>(
        idx: &HashMap<String, crate::ingest::LeafHandles>,
        rd: &R,
        name: &str,
        dev: &crate::nn::backend::WgpuDevice,
    ) -> anyhow::Result<(Tensor<BHalf, 1>, Vec<usize>)> {
        let (dh, sh) = match idx.get(name) {
            Some(crate::ingest::LeafHandles::F16(d, s)) => (*d, *s),
            Some(crate::ingest::LeafHandles::F32(..)) => {
                anyhow::bail!("{name}: expected an f16 leaf, found f32")
            }
            None => anyhow::bail!("{name}: missing from pile index"),
        };
        let bytes: anybytes::Bytes = rd
            .get(dh)
            .map_err(|e| anyhow::anyhow!("{name}: data blob: {e:?}"))?;
        let shape = crate::ingest::read_shape(rd, sh);
        let t = crate::nn::alias::alias_flat_raw::<half::f16>(bytes, dev)
            .map_err(|e| anyhow::anyhow!("{name}: zero-copy alias failed: {e}"))?;
        Ok((t, shape))
    }
    let f3 = |name: &str| -> anyhow::Result<Tensor<BHalf, 3>> {
        let (t, s) = alias(folded, folded_reader, name, &dev)?;
        anyhow::ensure!(s.len() == 3, "{name}: rank {} != 3", s.len());
        Ok(t.reshape([s[0], s[1], s[2]]))
    };
    let f4 = |name: &str| -> anyhow::Result<Tensor<BHalf, 4>> {
        let (t, s) = alias(folded, folded_reader, name, &dev)?;
        anyhow::ensure!(s.len() == 4, "{name}: rank {} != 4", s.len());
        Ok(t.reshape([s[0], s[1], s[2], s[3]]))
    };

    let cfg = talker_attn_config();
    let (ce, ce_shape) = alias(f16, reader, "talker.model.codec_embedding.weight", &dev)?;
    anyhow::ensure!(ce_shape.len() == 2, "codec_embedding rank != 2");
    let codec_embedding = Embedding {
        weight: ce.reshape([ce_shape[0], ce_shape[1]]),
    };
    let hidden = ce_shape[1];
    let (te, te_shape) = alias(f16, reader, "talker.model.text_embedding.weight", &dev)?;
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
    let (nw, nw_shape) = alias(folded, folded_reader, "talker.folded.norm.weight", &dev)?;
    anyhow::ensure!(nw_shape.len() == 1, "norm.weight rank != 1");
    let norm = RmsNorm {
        weight: nw,
        eps: TALKER_EPS,
    };

    // CPU stages: exact f32 leaves from the canonical pile, as in every lane.
    let ce_cpu = f32_
        .get("talker.model.codec_embedding.weight")
        .copied()
        .ok_or_else(|| anyhow::anyhow!("talker.model.codec_embedding.weight: missing f32 leaf"))?;
    let codec_embedding_cpu = crate::ingest::read_leaf(reader, ce_cpu).0;
    let ch = f32_
        .get("talker.codec_head.weight")
        .copied()
        .ok_or_else(|| anyhow::anyhow!("talker.codec_head.weight: missing f32 leaf"))?;
    let codec_head = crate::ingest::read_leaf(reader, ch).0;

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
    let tribles: TribleSet = checkout.facts().clone();

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

/// Load a model's keymap from the shared `mary` branch of a consolidated model
/// pile — the mary-branch twin of [`load_keymap_from_pile`]. Resolves the ONE
/// content-addressed model-ROOT labelled with `source` at
/// `quantization="native"`, out of a pile that holds many, and materializes ALL
/// its members. For a non-native format use
/// [`load_keymap_from_mary_branch_quantized`]; to address a root directly by its
/// entity id use [`load_keymap_from_mary_branch_by_root`]. This is a retained
/// legacy Repository reader; new callers should materialize
/// [`crate::model_collection::load_model_collection_local_latest`] and apply
/// [`crate::selection::ModelSelector`] directly.
pub fn load_keymap_from_mary_branch(
    pile_path: &Path,
    source: &str,
) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    load_keymap_from_mary_branch_quantized(pile_path, source, QUANTIZATION_NATIVE)
}

/// Like [`load_keymap_from_mary_branch`], but selects the root by BOTH `source`
/// AND `quantization`. `quantization` is a CORE identity coordinate; `source` is
/// a non-core label — together they name one root (a given `(source,
/// quantization)` pair maps to a single import). `native` and `fp4` of the same
/// model are distinct roots; this picks the requested one.
pub fn load_keymap_from_mary_branch_quantized(
    pile_path: &Path,
    source: &str,
    quantization: &str,
) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let (tribles, reader) = checkout_mary_branch(pile_path)?;
    crate::selection::load_keymap_from_graph(
        &tribles,
        &reader,
        crate::selection::ModelSelector::Source {
            source,
            quantization,
        },
    )
    .with_context(|| format!("select model on the 'mary' branch in pile {pile_path:?}"))
}

/// Load a model's keymap from the `mary` branch by the model-root's ENTITY ID
/// directly — the content address itself, no `(model_id, quantization)` lookup.
/// The complement to [`load_keymap_from_mary_branch_quantized`]: the id is what
/// the historical branch importer returned, so a caller that recorded it can
/// round-trip straight back to the exact weights.
pub fn load_keymap_from_mary_branch_by_root(
    pile_path: &Path,
    root: Id,
) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let (tribles, reader) = checkout_mary_branch(pile_path)?;
    crate::selection::load_keymap_from_graph(
        &tribles,
        &reader,
        crate::selection::ModelSelector::Root(root),
    )
    .with_context(|| format!("select model on the 'mary' branch in pile {pile_path:?}"))
}

/// Open a pile, resolve its `mary` branch, and return `(full-history facts, blob
/// reader)` — the shared read-side plumbing behind the mary-branch loaders. The
/// repo is closed before returning; the reader's mmap stays valid afterward (each
/// blob keeps the mapping alive, as in [`load_split_index_from_pile`]). NEVER
/// amputates: a corrupt tail fails loud (see [`load_keymap_from_pile`]).
fn checkout_mary_branch(
    pile_path: &Path,
) -> anyhow::Result<(TribleSet, triblespace::core::repo::pile::PileReader)> {
    let mut pile =
        Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open pile {pile_path:?}: {e:?}"))?;
    // Read path: non-mutating load, NEVER amputate (see load_keymap_from_pile).
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "pile {pile_path:?} failed to load ({e:?}); refusing to auto-truncate on a \
             read path — amputate explicitly with `trible pile amputate` if the tail is torn"
        )
    })?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = repo
        .lookup_branch("mary")
        .map_err(|e| anyhow::anyhow!("lookup mary: {e:?}"))?
        .ok_or_else(|| anyhow::anyhow!("no 'mary' branch in pile {pile_path:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull mary: {e:?}"))?;
    let head = ws
        .head()
        .ok_or_else(|| anyhow::anyhow!("'mary' branch has no commits"))?;
    let checkout = ws
        .checkout(ancestors(head))
        .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
    let tribles: TribleSet = checkout.facts().clone();
    let reader = repo
        .storage_mut()
        .reader()
        .map_err(|e| anyhow::anyhow!("pile reader: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok((tribles, reader))
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
        let tribles: TribleSet = ws
            .checkout(ancestors(head))
            .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
            .facts()
            .clone();
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
) -> anyhow::Result<std::collections::HashMap<String, crate::leaf::TypedLeaf>> {
    let (_, tribles, reader) = checkout_any_branch(pile_path)?;
    Ok(crate::leaf::index_typed_by_name(&tribles, &reader))
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
    let tribles: TribleSet = checkout.facts().clone();
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
    let tribles: TribleSet = checkout.facts().clone();
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
    let snapshot = crate::model_collection::load_model_collection_local_latest(pile_path)
        .with_context(|| format!("load local-latest native model snapshot from {pile_path:?}"))?;
    crate::selection::SelectedModelIndex::from_snapshot(snapshot, selector)
        .with_context(|| format!("select one native model root in {pile_path:?}"))
}

/// Stream a Gemma 4 model from one already-selected native model index: load
/// each tensor on demand and drop it after upload, so peak CPU is one tensor
/// rather than the whole f32 keymap. The index owns its blob reader across the
/// complete build.
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
    let (_, index, reader) = selected.into_parts();
    crate::models::gemma::gemma4::weights::load_gemma4_streaming::<B>(
        config, index, &reader, device,
    )
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
    let (_, index, reader) = selected.into_parts();
    let fetch = |name: &str| {
        index
            .get(name)
            .map(|&h| crate::ingest::read_leaf(&reader, h))
    };
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
    let (model, vision) = crate::models::gemma::gemma4::weights::load_gemma4_streaming::<B>(
        config, index, &reader, device,
    );
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
    let (_, index, reader) = selected.into_parts();
    let fetch = |name: &str| {
        index
            .get(name)
            .map(|&h| crate::ingest::read_leaf(&reader, h))
    };
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
    use crate::ingest::LeafHandles;
    use crate::models::gemma::gemma4::weights::{load_gemma4_from_source, WeightCtx};
    use crate::nn::backend::BHalf;
    use burn::backend::wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
    use burn::tensor::{DType, Tensor, TensorPrimitive};
    use cubecl::Runtime;
    use memmap2::MmapRaw;
    use std::sync::Arc;
    const PAGE: u64 = 16384;

    require_f16_model_index(&selected)?;
    let (_, index, reader) = selected.into_parts();

    let client = WgpuRuntime::client(&device);
    let ctx = WeightCtx::<BHalf> {
        has: Box::new(|name: &str| index.contains_key(name)),
        get: Box::new(|name: &str, device: &WgpuDevice| {
            let (dh, sh) = match index.get(name)? {
                LeafHandles::F16(d, s) => (*d, *s),
                LeafHandles::F32(..) => return None,
            };
            let sh_bytes: anybytes::Bytes = reader.get(sh).ok()?;
            let shape: Vec<usize> = sh_bytes
                .view::<[u64]>()
                .ok()?
                .iter()
                .map(|&x| x as usize)
                .collect();
            let bytes: anybytes::Bytes = reader.get(dh).ok()?;
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
fn alias_f16_leaf<R: triblespace::prelude::BlobStoreGet>(
    reader: &R,
    handles: crate::ingest::LeafHandles,
    device: &burn::backend::wgpu::WgpuDevice,
) -> (burn::tensor::Tensor<crate::nn::backend::B, 1>, Vec<usize>) {
    use crate::ingest::LeafHandles;
    use crate::nn::backend::B;
    use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
    use burn::tensor::{DType, Tensor, TensorPrimitive};
    use cubecl::Runtime;
    use memmap2::MmapRaw;
    use std::sync::Arc;
    const PAGE: u64 = 16384;

    let (dh, sh) = match handles {
        LeafHandles::F16(d, s) => (d, s),
        LeafHandles::F32(..) => panic!("aliased path requires f16 leaves; found f32"),
    };
    let sh_bytes: anybytes::Bytes = reader.get(sh).expect("shape blob");
    let shape: Vec<usize> = sh_bytes
        .view::<[u64]>()
        .expect("shape view")
        .iter()
        .map(|&x| x as usize)
        .collect();
    let bytes: anybytes::Bytes = reader.get(dh).expect("data_f16 blob");
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
struct AliasedQwenWeights<'a, R: triblespace::prelude::BlobStoreGet> {
    index: &'a HashMap<String, crate::ingest::LeafHandles>,
    reader: &'a R,
    device: burn::backend::wgpu::WgpuDevice,
}

#[cfg(all(feature = "gemma", target_os = "macos"))]
impl<'a, R: triblespace::prelude::BlobStoreGet> AliasedQwenWeights<'a, R> {
    fn flat(&self, name: &str) -> (burn::tensor::Tensor<crate::nn::backend::B, 1>, Vec<usize>) {
        let handles = *self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("missing weight {name} in pile index"));
        alias_f16_leaf(self.reader, handles, &self.device)
    }
}

#[cfg(all(feature = "gemma", target_os = "macos"))]
impl<'a, R: triblespace::prelude::BlobStoreGet>
    crate::models::qwen2_5_vl::layers::QwenWeights<crate::nn::backend::B>
    for AliasedQwenWeights<'a, R>
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
impl<'a, R: triblespace::prelude::BlobStoreGet>
    crate::models::qwen2_5_vl::vision::VisionWeights<crate::nn::backend::B>
    for AliasedQwenWeights<'a, R>
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
        .filter_map(|(name, handles)| {
            matches!(handles, crate::ingest::LeafHandles::F32(..)).then_some(name)
        })
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
        reader: &reader,
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
        crate::model_collection::publish_model_fragment(
            pile,
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

        let snapshot = crate::model_collection::load_model_collection_local_latest(&file.path)
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
        let (_, index, reader) = selected.into_parts();
        assert_eq!(index.len(), 1);
        let crate::ingest::LeafHandles::F16(data, shape) = index["target.weight"] else {
            panic!("selected leaf was not f16");
        };
        let values: anybytes::Bytes = reader.get(data).expect("data after pile close");
        let shape: anybytes::Bytes = reader.get(shape).expect("shape after pile close");
        assert_eq!(values.view::<[half::f16]>().unwrap()[0].to_f32(), 1.5);
        assert_eq!(&*shape.view::<[u64]>().unwrap(), &[1]);

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
        let snapshot = crate::model_collection::load_model_collection_local_latest(&file.path)
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
