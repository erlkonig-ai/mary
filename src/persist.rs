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
use ed25519_dalek::SigningKey;
use std::collections::HashMap;
use std::path::Path;
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

/// Import a model directory's safetensors into a SHARED model pile on the `mary`
/// branch as ONE content-addressed model-ROOT entity — so many models coexist in
/// one pile, each loadable by `(source [, quantization])` via
/// [`load_keymap_from_mary_branch`] (or by the root's entity id via
/// [`load_keymap_from_mary_branch_by_root`]). This is the consolidated-MODEL_PILE
/// front door (`mary import`): the model becomes a proper addressable entity AT
/// import, so no separate consolidation step exists — appending another model is
/// just another import.
///
/// `source` is the model's canonical NAME — the HF id it was imported from, or a
/// `--name` for a local-dir import. The root's id is CONTENT-DERIVED from its
/// identity `(source, quantization, weight members)`: importing the same
/// `(source, quantization, weights)` twice yields the SAME root id (dedup), while
/// a different `quantization` of the same model is a DISTINCT entity. For a
/// MULTI-shard model, the ONE root composes EVERY shard's tensor members
/// (order-independent set), and each shard's file name is recorded as NON-core
/// `model_name` provenance on the root.
///
/// `quantization` tags the weight format ("native" for the faithful import).
/// Returns the imported root's entity id (the content address).
#[cfg(feature = "import")]
pub fn persist_model_to_pile(
    model_dir: &Path,
    pile_path: &Path,
    dtype: LeafDtype,
    source: &str,
    quantization: &str,
) -> anyhow::Result<Id> {
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
    // Reuse the mary branch if it exists (append into an existing pile), else create it.
    let branch_id = match repo
        .lookup_branch("mary")
        .map_err(|e| anyhow::anyhow!("lookup mary: {e:?}"))?
    {
        Some(id) => id,
        None => *repo
            .create_branch("mary", None)
            .map_err(|e| anyhow::anyhow!("create mary: {e:?}"))?,
    };
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull mary: {e:?}"))?;

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
                crate::ingest::ingest_members(&bytes, repo.storage_mut(), dtype, |_| true)
                    .map_err(|e| anyhow::anyhow!("ingest {path:?}: {e}"))?
            }
            crate::formats::WeightFormat::Gguf | crate::formats::WeightFormat::Pickle => {
                let tensors = crate::formats::extract_tensors(fmt, path)
                    .map_err(|e| anyhow::anyhow!("extract {path:?}: {e}"))?;
                eprintln!(
                    "[persist] ingesting {name} ({} tensors, {fmt:?})...",
                    tensors.len()
                );
                crate::ingest::ingest_tensors(tensors.into_iter(), repo.storage_mut(), dtype)
                    .map_err(|e| anyhow::anyhow!("ingest {path:?}: {e}"))?
            }
        };
        members.append(&mut shard_members);
        facts += shard_facts;
        provenance.push(name.clone());
    }
    let root = crate::ingest::build_model_root(
        repo.storage_mut(),
        source,
        quantization,
        members,
        facts,
        &provenance,
    )
    .map_err(|e| anyhow::anyhow!("build model root: {e}"))?;
    let root_id = root.root().expect("model root id");

    ws.commit(
        root.into_facts(),
        &format!("ingest model {source} ({quantization})"),
    );
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok(root_id)
}

/// The engine behind [`persist_safetensors_files_to_pile`] (untagged, `main`):
/// ingest each file's weight blobs straight into `pile_path`'s storage (no
/// in-memory carryover) and commit ONE model entity PER file on `main`, creating
/// the pile and branch if absent. (The content-addressed model-ROOT path is
/// [`persist_model_to_pile`], `mary` branch.)
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

/// The derived half-width SIBLING of a weights pile: `<stem>_f16.pile` next to
/// it (`models/voxtral_mini.pile` → `models/voxtral_mini_f16.pile`). Written
/// by `voxtral_persist --f16-derive`; auto-discovered by
/// [`load_loader_with_f16_sibling`].
pub fn f16_sibling_path(pile_path: &Path) -> std::path::PathBuf {
    let stem = pile_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("weights");
    pile_path.with_file_name(format!("{stem}_f16.pile"))
}

/// The runtime weight loader for an exact-f32 pile that MAY have a derived
/// half-width sibling pile next to it (see [`f16_sibling_path`] /
/// [`derive_f16_pile`]) — the two-pile variant of
/// [`load_aliased_loader_from_pile`]. On macOS this is the fast
/// [`WeightLoader::Aliased`]: `BFusedHalf` requests upload the sibling's f16
/// leaves at native width (no f32 materialization, half the I/O — this is
/// what deletes the ~2×-weights transient host spike and most of the load
/// time), `BHalf` (raw, unfused) requests alias the sibling's f16 leaves
/// ZERO-COPY onto the GPU (mmap'd pile pages, no upload at all — the
/// `rawhalf` ear lane), `BFused` requests upload the exact f32 leaves, and
/// everything else materializes lazily one tensor at a time. When the
/// sibling pile is absent the same loader still works — f16-backend tensors
/// materialize from the exact leaves and cast on upload (bit-identical
/// result, just slower) — so piles without a derived sibling keep loading
/// unchanged. Elsewhere (or under `MARY_SPEAK_MATERIALIZE=1`, the A/B
/// switch) the exact materialized keymap.
#[cfg(any(feature = "qwen3tts", feature = "voxtral"))]
pub fn load_loader_with_f16_sibling(
    pile_path: &Path,
    f16_entity: &str,
) -> anyhow::Result<crate::nn::weight_loader::WeightLoader> {
    use crate::nn::weight_loader::WeightLoader;
    let (_, f32_, f32_reader) = load_split_index_from_pile(pile_path, "")?;
    anyhow::ensure!(
        !f32_.is_empty(),
        "no exact (f32) model entities in pile {pile_path:?}"
    );
    let sibling = f16_sibling_path(pile_path);
    let (f16, f16_reader) = if sibling.exists() {
        let (f16, _, r) = load_split_index_from_pile(&sibling, f16_entity)?;
        anyhow::ensure!(
            !f16.is_empty(),
            "sibling pile {sibling:?} exists but has no '{f16_entity}' entity"
        );
        eprintln!(
            "[mary] half-width sibling {sibling:?}: {} f16 leaves",
            f16.len()
        );
        (f16, r)
    } else {
        eprintln!(
            "[mary] no half-width sibling {sibling:?} — f16-backend tensors will \
             materialize+cast (derive it with: voxtral_persist --f16-derive <pile>)"
        );
        (HashMap::new(), f32_reader.clone())
    };
    let materialize = std::env::var("MARY_SPEAK_MATERIALIZE").is_ok();
    #[cfg(target_os = "macos")]
    if !materialize {
        return Ok(WeightLoader::Aliased(
            crate::nn::weight_loader::AliasedPile::new(
                f16,
                f32_,
                f16_reader,
                f32_reader,
                crate::nn::backend::WgpuDevice::default(),
            ),
        ));
    }
    if materialize {
        eprintln!("[mary] MARY_SPEAK_MATERIALIZE set — using the fully materialized load");
    }
    let _ = (f16, f16_reader); // half-width leaves are only for aliasing
    let keymap = f32_
        .into_iter()
        .map(|(k, h)| (k, crate::ingest::read_leaf(&f32_reader, h)))
        .collect();
    Ok(WeightLoader::Pile(keymap))
}

/// Derive a HALF-WIDTH weights pile from an existing exact-f32 pile: every
/// f32 leaf of every model entity in `src_pile` is read back (one tensor at a
/// time — peak host RAM is one tensor, not the model), cast host-side to f16
/// (`f16::from_f32` — the exact rounding the materializing loader applies on
/// an f16 backend, so fast-loaded weights stay bit-identical to the old
/// cast-on-load path), and persisted as a `data_f16` leaf under ONE model
/// entity named `entity_name` in `dst_pile`. The source pile is only ever
/// read; the destination pile is created (or appended — piles only grow).
/// Separate-pile-now / merge-later is the cheap direction: piles union by
/// `cat` + consolidate, and the f16 sibling gets its own lifecycle (a
/// deployment machine can carry just the half-width pile). Returns
/// `(tensor count, element count)`.
pub fn derive_f16_pile(
    src_pile: &Path,
    dst_pile: &Path,
    entity_name: &str,
) -> anyhow::Result<(usize, usize)> {
    use crate::format::attrs;
    anyhow::ensure!(
        src_pile.canonicalize()?
            != dst_pile
                .canonicalize()
                .unwrap_or_else(|_| dst_pile.to_path_buf()),
        "src and dst are the same pile file {src_pile:?}"
    );
    let (_, src_idx, src_reader) = load_split_index_from_pile(src_pile, "")?;
    anyhow::ensure!(!src_idx.is_empty(), "no model entities in {src_pile:?}");

    if !dst_pile.exists() {
        eprintln!("[persist] pile {dst_pile:?} does not exist — creating a NEW empty pile");
        std::fs::File::create(dst_pile)
            .map_err(|e| anyhow::anyhow!("create pile {dst_pile:?}: {e}"))?;
    }
    let mut pile =
        Pile::open(dst_pile).map_err(|e| anyhow::anyhow!("open pile {dst_pile:?}: {e:?}"))?;
    // Non-mutating load; NEVER amputate here (see persist_safetensors_files_to_pile).
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "pile {dst_pile:?} failed to load ({e:?}); refusing to auto-truncate — \
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

    // Deterministic order (the source index is a HashMap).
    let mut names: Vec<&String> = src_idx.keys().collect();
    names.sort();
    let mut members: Vec<Id> = Vec::new();
    let mut facts = TribleSet::new();
    let (mut count, mut elems) = (0usize, 0usize);
    for name in names {
        let handles = src_idx[name];
        anyhow::ensure!(
            matches!(handles, crate::ingest::LeafHandles::F32(..)),
            "{name}: source leaf is not f32 — derive_f16_pile expects an exact-f32 source pile"
        );
        let (data, shape) = crate::ingest::read_leaf(&src_reader, handles);
        let shp: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let leaf = crate::format::put_raw_f16(repo.storage_mut(), &data, &shp)
            .map_err(|e| anyhow::anyhow!("{name}: put f16 leaf: {e}"))?;
        let leaf_id = leaf.root().expect("leaf root");
        facts += leaf.into_facts();
        let kind = match shape.len() {
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
        elems += data.len();
        if count % 100 == 0 {
            eprintln!("[persist] {count} tensors cast → f16 ...");
        }
    }
    let mn = repo
        .storage_mut()
        .put::<blobencodings::LongString, _>(entity_name.to_string())
        .map_err(|e| anyhow::anyhow!("put entity name blob: {e:?}"))?;
    let model = entity! { _ @ attrs::model_name: mn, attrs::member*: members.iter() };
    facts += model.into_facts();

    ws.commit(
        facts,
        "derive f16 weights (host-side f32→f16) from the exact pile",
    );
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok((count, elems))
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
    use crate::models::qwen3tts::config::{
        TALKER_EPS, TALKER_HEAD_DIM, TALKER_LAYERS, TALKER_ROPE_THETA,
    };
    use crate::models::qwen3tts::layers::{
        Attention, DecoderLayer, Embedding, Linear, RmsNorm, RopeTable,
    };
    use crate::models::qwen3tts::talker::{talker_attn_config, Talker};
    use crate::nn::backend::BHalf;
    use burn::prelude::*;
    use triblespace::prelude::BlobStoreGet;

    let dev = crate::nn::backend::WgpuDevice::default();
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

    let alias = |idx: &HashMap<String, crate::ingest::LeafHandles>,
                 rd: &triblespace::core::repo::pile::PileReader,
                 name: &str|
     -> anyhow::Result<(Tensor<BHalf, 1>, Vec<usize>)> {
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
        let t = crate::nn::alias::alias_flat_raw::<half::f16>(bytes, &dev)
            .map_err(|e| anyhow::anyhow!("{name}: zero-copy alias failed: {e}"))?;
        Ok((t, shape))
    };
    let f3 = |name: &str| -> anyhow::Result<Tensor<BHalf, 3>> {
        let (t, s) = alias(&folded, &folded_reader, name)?;
        anyhow::ensure!(s.len() == 3, "{name}: rank {} != 3", s.len());
        Ok(t.reshape([s[0], s[1], s[2]]))
    };
    let f4 = |name: &str| -> anyhow::Result<Tensor<BHalf, 4>> {
        let (t, s) = alias(&folded, &folded_reader, name)?;
        anyhow::ensure!(s.len() == 4, "{name}: rank {} != 4", s.len());
        Ok(t.reshape([s[0], s[1], s[2], s[3]]))
    };

    let cfg = talker_attn_config();
    let (ce, ce_shape) = alias(&f16, &reader, "talker.model.codec_embedding.weight")?;
    anyhow::ensure!(ce_shape.len() == 2, "codec_embedding rank != 2");
    let codec_embedding = Embedding {
        weight: ce.reshape([ce_shape[0], ce_shape[1]]),
    };
    let hidden = ce_shape[1];
    let (te, te_shape) = alias(&f16, &reader, "talker.model.text_embedding.weight")?;
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
    let (nw, nw_shape) = alias(&folded, &folded_reader, "talker.folded.norm.weight")?;
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
    let codec_embedding_cpu = crate::ingest::read_leaf(&reader, ce_cpu).0;
    let ch = f32_
        .get("talker.codec_head.weight")
        .copied()
        .ok_or_else(|| anyhow::anyhow!("talker.codec_head.weight: missing f32 leaf"))?;
    let codec_head = crate::ingest::read_leaf(&reader, ch).0;

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
/// entity id use [`load_keymap_from_mary_branch_by_root`].
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
    // Constrain `quantization` engine-side (a ShortString value on the root),
    // then match the projected `source` label by its blob string. `source` is a
    // `Handle<LongString>` so its value lives in a blob, not inline — resolve it
    // through the reader.
    let root: Id = find!(
        (m: Id, s: Inline<inlineencodings::Handle<blobencodings::LongString>>),
        pattern!(&tribles, [{ ?m @
            crate::format::attrs::quantization: quantization,
            crate::format::attrs::source: ?s,
        }])
    )
    .filter(|&(_m, s)| {
        let v: anybytes::View<str> = reader.get(s).expect("source blob");
        &*v == source
    })
    .map(|(m, _s)| m)
    .next()
    .ok_or_else(|| {
        anyhow::anyhow!(
            "no model root (source={source:?}, quantization={quantization:?}) on the \
             'mary' branch in pile {pile_path:?}"
        )
    })?;

    let keymap = load_keymap(&tribles, &reader, root);
    if keymap.is_empty() {
        anyhow::bail!("keymap empty after materializing model root {root} from the mary branch");
    }
    Ok(keymap)
}

/// Load a model's keymap from the `mary` branch by the model-root's ENTITY ID
/// directly — the content address itself, no `(model_id, quantization)` lookup.
/// The complement to [`load_keymap_from_mary_branch_quantized`]: the id is what
/// `persist_model_to_pile` returns, so a caller that recorded it can round-trip
/// straight back to the exact weights.
pub fn load_keymap_from_mary_branch_by_root(
    pile_path: &Path,
    root: Id,
) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let (tribles, reader) = checkout_mary_branch(pile_path)?;
    let keymap = load_keymap(&tribles, &reader, root);
    if keymap.is_empty() {
        anyhow::bail!(
            "no members under model root {root} on the 'mary' branch in pile {pile_path:?} \
             (unknown root id, or an empty model)"
        );
    }
    Ok(keymap)
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

/// Reconstruct a SentencePiece UNIGRAM tokenizer from a pile's tokenizer graph.
/// The `.model` file is not needed — the pieces ARE the model.
///
/// NOTE: this mirrors [`load_tokenizer_from_pile`]'s pile-opening prologue
/// rather than sharing it. Factoring the two would need a callback trait,
/// because the blob reader is an associated type and `BlobStoreGet` is not
/// dyn-compatible — more machinery than the ~25 duplicated lines are worth,
/// and it would put the proven reader at risk for no behavioural gain.
#[cfg(feature = "tokenizer")]
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
    let tok_id = crate::tokenizer::find_tokenizer(&tribles)
        .ok_or_else(|| anyhow::anyhow!("no tokenizer graph in pile {pile_path:?}"))?;
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
#[cfg(feature = "tokenizer")]
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

    let tok_id = crate::tokenizer::find_tokenizer(&tribles).ok_or_else(|| {
        anyhow::anyhow!(
            "no tokenizer graph in pile {pile_path:?} — ingest one from its \
             tokenizer.json (e.g. `memory ingest-tokenizer`)"
        )
    })?;
    let tok = crate::tokenizer::build_tokenizer(&tribles, &reader, tok_id)
        .map_err(|e| anyhow::anyhow!("build tokenizer from graph: {e}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok(tok)
}

/// Stream a Gemma 4 model directly from a pile: index the blob handles (cheap),
/// then load each tensor on demand and drop it after upload — peak CPU is one
/// tensor, NOT the whole f32 keymap. This is the path that scales weights-as-
/// tribles to the dense 31B (the materialized `load_keymap_from_pile` would OOM).
/// The pile reader is held alive across the whole build.
#[cfg(feature = "gemma")]
pub fn load_gemma4_streaming_from_pile<B: burn::prelude::Backend>(
    pile_path: &Path,
    config: crate::models::gemma::gemma4::config::Gemma4Config,
    device: &B::Device,
) -> anyhow::Result<(
    crate::models::gemma::gemma4::decoder::Gemma4Model<B>,
    Option<crate::models::gemma::gemma4::vision::Gemma4VisionEncoder<B>>,
)> {
    let (index, reader) = pile_weight_index(pile_path)?;
    Ok(
        crate::models::gemma::gemma4::weights::load_gemma4_streaming::<B>(
            config, index, &reader, device,
        ),
    )
}

/// The full HEARING stack from ONE pile open: text decoder (+vision when the
/// checkpoint has one) AND the audio tower + multimodal embedder. This is
/// `gemma_hear`'s pile seam — audio inference without any safetensors on the
/// load path, matching the text-only `load_gemma4_streaming_from_pile`.
#[cfg(feature = "gemma")]
pub fn load_gemma4_hearing_from_pile<B: burn::prelude::Backend>(
    pile_path: &Path,
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
    let (index, reader) = pile_weight_index(pile_path)?;
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

/// JUST the stt from a pile: audio tower + multimodal embedder, no decoder.
/// Cheap relative to the full model (~0.75B of the E4B's ~8B parameters) —
/// the parity gate uses this to score the pile-loaded audio path against the
/// HF goldens without streaming in the text stack.
#[cfg(feature = "gemma")]
pub fn load_gemma4_audio_from_pile<B: burn::prelude::Backend>(
    pile_path: &Path,
    audio_cfg: crate::models::gemma::gemma4::config::Gemma4AudioConfig,
    device: &B::Device,
) -> anyhow::Result<(
    crate::models::gemma::gemma4::audio::AudioModel<B>,
    crate::models::gemma::gemma4::audio::AudioEmbedder<B>,
)> {
    let (index, reader) = pile_weight_index(pile_path)?;
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
    Ok((tower, embedder))
}

/// Open a pile, resolve `main`, and index every persisted model tensor by
/// exact name (cheap: two ~32-byte handles per tensor, no weight data read).
/// The returned blob reader is standalone — it stays valid after the repo is
/// closed, so callers stream leaves through it on demand.
#[cfg(feature = "gemma")]
fn pile_weight_index(
    pile_path: &Path,
) -> anyhow::Result<(
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
    let checkout = ws
        .checkout(ancestors(head))
        .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
    let tribles: TribleSet = checkout.facts().clone();
    let reader = repo
        .storage_mut()
        .reader()
        .map_err(|e| anyhow::anyhow!("pile reader: {e:?}"))?;

    let model_ids: Vec<Id> = find!(
        (m: Id, n: Inline<inlineencodings::Handle<blobencodings::LongString>>),
        pattern!(&tribles, [{ ?m @ crate::format::attrs::model_name: ?n }])
    )
    .map(|(m, _n)| m)
    .collect();
    if model_ids.is_empty() {
        anyhow::bail!("no model entity (attrs::model_name) found in pile");
    }

    // Cheap: union the handle-indices across shards (no f32 data read here).
    let mut index = HashMap::new();
    for id in model_ids {
        index.extend(crate::ingest::index_keymap(&tribles, &reader, id));
    }
    if index.is_empty() {
        anyhow::bail!("empty model index from pile");
    }

    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok((index, reader))
}

/// Load a Gemma 4 model from a pile with ZERO-COPY weights: each tensor's mmap'd
/// f16 blob is aliased straight onto the Metal GPU — no copy, no f32
/// materialization. Each weight's GPU buffer carries the pile mmap alive (the
/// `register_external_aliased` keepalive), so the pile/reader are dropped after
/// the build and the mapping persists for the model's life. Metal / `BHalf` only.
#[cfg(all(feature = "gemma", target_os = "macos"))]
pub fn load_gemma4_aliased_from_pile(
    pile_path: &Path,
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
    let model_ids: Vec<Id> = find!(
        (m: Id, n: Inline<inlineencodings::Handle<blobencodings::LongString>>),
        pattern!(&tribles, [{ ?m @ crate::format::attrs::model_name: ?n }])
    )
    .map(|(m, _n)| m)
    .collect();
    let mut index = HashMap::new();
    for id in model_ids {
        index.extend(crate::ingest::index_keymap(&tribles, &reader, id));
    }
    if index.is_empty() {
        anyhow::bail!("empty model index from pile");
    }

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
    drop(ctx); // release the reader borrow before closing the pile
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok(model)
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

/// Load `nomic-embed-multimodal-7b` (Qwen2.5-VL backbone + vision tower) from a
/// combined f16 pile with ZERO-COPY weights: every tensor's mmap'd f16 blob is
/// aliased straight onto the Metal GPU (no copy, no f32 materialization). Each
/// weight's GPU buffer carries the pile mmap alive (the `register_external_aliased`
/// keepalive), so the pile/reader are dropped after the build and the mappings
/// persist for the embedder's life. Metal only. Weights stay f16 in GPU memory
/// (zero-copy); activations run in f32 (the model upcasts weights per-op, as this
/// bf16-native model's activations exceed f16's range). This is the per-call
/// "no daemon needed" path: cold mmap + one embed, no multi-GB f32 weight upload.
#[cfg(all(feature = "gemma", target_os = "macos"))]
pub fn load_nomic_mm7b_aliased_from_pile(
    pile_path: &Path,
    tokenizer_path: &Path,
    device: burn::backend::wgpu::WgpuDevice,
) -> anyhow::Result<
    crate::models::qwen2_5_vl::embedder::NomicMultimodalEmbedder<crate::nn::backend::B>,
> {
    use crate::models::qwen2_5_vl::embedder::NomicMultimodalEmbedder;
    use crate::nn::backend::B;

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
    let model_ids: Vec<Id> = find!(
        (m: Id, n: Inline<inlineencodings::Handle<blobencodings::LongString>>),
        pattern!(&tribles, [{ ?m @ crate::format::attrs::model_name: ?n }])
    )
    .map(|(m, _n)| m)
    .collect();
    let mut index = HashMap::new();
    for id in model_ids {
        index.extend(crate::ingest::index_keymap(&tribles, &reader, id));
    }
    if index.is_empty() {
        anyhow::bail!("empty model index from pile");
    }

    let weights = AliasedQwenWeights {
        index: &index,
        reader: &reader,
        device: device.clone(),
    };
    let embedder =
        NomicMultimodalEmbedder::<B>::load_with_vision(&weights, tokenizer_path, device)?;
    drop(weights); // release the reader borrow before closing the pile
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok(embedder)
}
