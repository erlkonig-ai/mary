//! `mary` — the command-line front door to mary.
//!
//! mary stores neural-network weights as content-addressed graphs in one native
//! append-only TribleSpace collection: a tensor is a self-describing leaf, a
//! module is an entity, composition is role-edges. Every mary runtime loads
//! weights from such a pile, never from safetensors. This binary is how weights
//! get *into* one.
//!
//! Bootstrap a model in one line:
//!
//! ```text
//! # many models in one shared MODEL_PILE — each a signed collection member:
//! mary import openai/clip-vit-base-patch32 --pile models.pile --key model.key
//! mary import HuggingFaceTB/SmolLM2-135M --pile models.pile --key model.key --dtype f16
//! mary import ./my-model-dir --pile models.pile --key model.key --name my_model
//! ```
//!
//! The source is either a HuggingFace model id (auto-downloaded from the hub if
//! not already in the local cache) or a local directory of weight files (which
//! needs a `--name`, since there is no hf-id to label it with). The format is
//! auto-detected — `.safetensors` shards, a `.gguf` (dequantized to f32), or a
//! pytorch pickle `state_dict` — and every format funnels into the SAME
//! content-addressed member path, so the root id is the pure hash of the tensor
//! set regardless of source format (a model imported from GGUF/f16 or from
//! safetensors with the same weights resolves to the same root). The model
//! becomes a content-addressed ROOT entity in Mary's model collection, so many
//! models coexist in one pile, each loaded back by its `source` label (the hf-id
//! / `--name`) or by its entity id. Re-importing with the same signer is
//! byte-idempotent. The resulting pile is self-contained: no weight files are
//! needed at load time.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Args, Parser, Subcommand, ValueEnum};
use mary::ingest::LeafDtype;
use mary::selection::ModelSelector;
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;
use triblespace::prelude::Id;

#[derive(Parser)]
#[command(
    name = "mary",
    about = "Neural-network models as content-addressed graphs in TribleSpace — import weights into a pile.",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Import a model's weights into a pile (the durable weight store). Accepts
    /// safetensors, GGUF (.gguf), or pytorch pickle (pytorch_model*.bin / .pth).
    Import(ImportArgs),
    /// Load ONE model from the native model collection — by `--source` (+
    /// optional `--quantization`) or by `--root` entity id — and print its
    /// tensor count + a sample. A round-trip check of the content-addressed store.
    Keys(KeysArgs),
}

#[derive(Args)]
struct ImportArgs {
    /// A HuggingFace model id (resolved from the local HF cache, or downloaded)
    /// OR a local directory holding the model's weight files. The format is
    /// auto-detected: `.safetensors` shards, a `.gguf`, or a pytorch pickle
    /// `state_dict` (`pytorch_model*.bin` / `.pth`).
    source: String,
    /// Output pile path. Created if absent; a model added to an existing pile
    /// is appended as a native collection commit (content addressing dedups).
    #[arg(long)]
    pile: PathBuf,
    /// Existing private signing-key file (strict 64-hex, owner-private mode).
    /// Import never generates or infers an author identity.
    #[arg(long)]
    key: PathBuf,
    /// Leaf storage dtype. `f32` is lossless (the faithful original, whatever
    /// width the source used); `f16` halves the pile for 16-bit-native weights.
    #[arg(long, value_enum, default_value_t = Dtype::F32)]
    dtype: Dtype,
    /// The model's canonical `source` label in the model collection (a
    /// queryable non-core name; the root id addresses only the weights).
    /// Defaults to the hf-id `source` argument; REQUIRED for a local directory,
    /// which has no hf-id to label it with.
    #[arg(long)]
    name: Option<String>,
    /// Weight-format tag recorded as a non-core label on the root (e.g. "native",
    /// "fp4"). Defaults to "native" — the faithful import.
    #[arg(long, default_value = "native")]
    quantization: String,
}

#[derive(Args)]
struct KeysArgs {
    /// The consolidated model pile to read from.
    #[arg(long)]
    pile: PathBuf,
    /// The `source` label to load from the native collection (the hf-id or the
    /// `--name` used at import). Mutually exclusive with `--root`.
    #[arg(long, conflicts_with = "root")]
    source: Option<String>,
    /// The weight-format label to disambiguate when the same `source` was
    /// imported under several tags. Defaults to "native".
    #[arg(long, default_value = "native")]
    quantization: String,
    /// Load the model directly by its ROOT entity id (hex) — the content address
    /// `mary import` printed. Mutually exclusive with `--source`.
    #[arg(long, conflicts_with = "source")]
    root: Option<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Dtype {
    F32,
    F16,
}

impl From<Dtype> for LeafDtype {
    fn from(d: Dtype) -> Self {
        match d {
            Dtype::F32 => LeafDtype::F32,
            Dtype::F16 => LeafDtype::F16,
        }
    }
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Import(a) => import(a),
        Cmd::Keys(a) => keys(a),
    }
}

fn keys(a: KeysArgs) -> anyhow::Result<()> {
    // One observed local collection prefix supplies both the facts and reader;
    // selector policy stays explicit and no Repository branch or fallback
    // storage participates.
    let snapshot = mary::model_collection::load_model_collection_local_latest(&a.pile)?;
    let (selector, label) = match (&a.source, &a.root) {
        (Some(source), None) => (
            ModelSelector::Source {
                source,
                quantization: &a.quantization,
            },
            format!("source={source} quantization={}", a.quantization),
        ),
        (None, Some(hex)) => {
            let root = Id::from_hex(hex)
                .ok_or_else(|| anyhow::anyhow!("--root {hex:?} is not a valid 32-hex entity id"))?;
            (ModelSelector::Root(root), format!("root={hex}"))
        }
        _ => anyhow::bail!("mary keys: pass exactly one of --source or --root"),
    };
    let km =
        mary::selection::load_keymap_from_graph(snapshot.facts(), snapshot.reader(), selector)?;
    let mut names: Vec<&String> = km.keys().collect();
    names.sort();
    eprintln!(
        "mary keys: {label} in the model collection of {} -> {} tensors",
        a.pile.display(),
        km.len()
    );
    for n in names.iter().take(8) {
        let (data, shape) = &km[*n];
        eprintln!("  {n}  shape={shape:?}  ({} f32)", data.len());
    }
    Ok(())
}

fn import(a: ImportArgs) -> anyhow::Result<()> {
    let dir = resolve_source(&a.source)?;
    let dt = match a.dtype {
        Dtype::F32 => "f32",
        Dtype::F16 => "f16",
    };
    // The model's `source` LABEL: an explicit `--name`, else the hf-id argument.
    // A local directory has no hf-id, so `--name` is required there.
    let label = match &a.name {
        Some(n) => n.clone(),
        None => {
            if Path::new(&a.source).is_dir() {
                anyhow::bail!(
                    "mary import: a local directory has no hf-id to use as its `source` label — \
                     pass `--name <n>`"
                );
            }
            a.source.clone()
        }
    };
    eprintln!(
        "mary import: {} -> {} ({dt} leaves, source={label} quantization={} in the native model collection)",
        dir.display(),
        a.pile.display(),
        a.quantization,
    );
    let signing_key = signing_key_file::load_existing(&a.key)
        .with_context(|| format!("load existing signing key {:?}", a.key))?;
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a.pile)
    {
        Ok(_) => eprintln!("mary import: created new empty pile {:?}", a.pile),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error).with_context(|| format!("create model pile {:?}", a.pile)),
    }
    let mut pile = Pile::open(&a.pile).with_context(|| format!("open model pile {:?}", a.pile))?;
    let imported = mary::persist::import_model_to_collection(
        &mut pile,
        &signing_key,
        &dir,
        a.dtype.into(),
        &label,
        &a.quantization,
    );
    let close = pile.close();
    let (root, commit) = match (imported, close) {
        (Ok(imported), Ok(())) => imported,
        (Ok(_), Err(error)) => return Err(anyhow::anyhow!("close model pile: {error}")),
        (Err(error), Ok(())) => return Err(error),
        (Err(error), Err(close_error)) => {
            return Err(error.context(format!(
                "import also failed to close the pile: {close_error}"
            )))
        }
    };
    eprintln!(
        "mary import: done — model root {root:X}, native commit {} in pile {}",
        commit.id(),
        a.pile.display(),
    );
    println!("{}", lowercase_hex(&commit.to_bytes()));
    Ok(())
}

fn lowercase_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String is infallible");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticket_hex_is_exact_and_lowercase() {
        let bytes: Vec<u8> = (0..192).map(|index| index as u8).collect();
        let encoded = lowercase_hex(&bytes);
        assert_eq!(encoded.len(), 384);
        assert!(encoded
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        assert_eq!(&encoded[..8], "00010203");
        assert_eq!(&encoded[encoded.len() - 8..], "bcbdbebf");
    }
}

/// Resolve the import source to a directory of importable weight files. A local
/// directory is used as-is; otherwise the source is a HuggingFace model id and
/// we locate (or download) its snapshot directory in the local HF cache. Any of
/// the three supported formats counts — safetensors, GGUF, or a pytorch pickle
/// `state_dict` — since `mary::formats::detect_format` picks the decoder at
/// import time. Kept inline (rather than via the embedding cache helper) so
/// `mary import` stays behind the lean `import` feature and never drags in the
/// `embed` stack.
fn resolve_source(source: &str) -> anyhow::Result<PathBuf> {
    let p = Path::new(source);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    if let Some(dir) = hf_snapshot_dir(source) {
        return Ok(dir);
    }
    // Not a directory and not cached → treat as a HuggingFace model id and pull
    // whichever weight format it actually ships into the local cache, then resolve.
    eprintln!(
        "mary import: '{source}' not in the local cache — downloading from the HuggingFace hub..."
    );
    download_hf_weights(source)?;
    hf_snapshot_dir(source).ok_or_else(|| {
        anyhow::anyhow!(
            "downloaded '{source}' but found no importable weights (.safetensors / .gguf / \
             pytorch_model*.bin) in its cache snapshot"
        )
    })
}

/// Pull a model's weight files from the HuggingFace hub into the local cache,
/// in whichever format the repo ships — preferring safetensors, then GGUF, then
/// pytorch pickle. Weights only: mary loads from the pile, so config/tokenizer
/// files are skipped. Sharded checkpoints (safetensors or pytorch) are resolved
/// through their `*.index.json` weight-map; single-file ones fetched directly.
/// The repo's actual file list (`info().siblings`) drives GGUF selection so we
/// don't have to guess arbitrary `.gguf` filenames.
fn download_hf_weights(id: &str) -> anyhow::Result<()> {
    use hf_hub::api::sync::Api;
    let repo = Api::new()
        .map_err(|e| anyhow::anyhow!("hf-hub api init: {e}"))?
        .model(id.to_string());

    // List the repo's files up front (best-effort; falls back to name probing).
    let files: Vec<String> = repo
        .info()
        .map(|i| i.siblings.into_iter().map(|s| s.rfilename).collect())
        .unwrap_or_default();
    let has = |name: &str| files.iter().any(|f| f == name);

    // 1) safetensors (the faithful native path) — sharded via index, else single-file.
    if has("model.safetensors.index.json") {
        let index_path = repo.get("model.safetensors.index.json")?;
        fetch_sharded(&repo, &index_path)?;
        return Ok(());
    }
    let safetensors: Vec<String> = files
        .iter()
        .filter(|f| f.ends_with(".safetensors"))
        .cloned()
        .collect();
    if !safetensors.is_empty() {
        // single-file, or an un-indexed multi-file safetensors set
        for s in &safetensors {
            eprintln!("  fetching {s} ...");
            repo.get(s).map_err(|e| anyhow::anyhow!("fetch {s}: {e}"))?;
        }
        return Ok(());
    }
    if files.is_empty() && repo.get("model.safetensors").is_ok() {
        eprintln!("  fetched model.safetensors");
        return Ok(());
    }

    // 2) GGUF — pick every `.gguf` file the repo actually lists. A repo may ship
    //    several quant variants under arbitrary names; grab them all (import
    //    selects one, and detect_format confirms the magic). If we couldn't list
    //    the repo, fall back to probing common conventional names.
    let ggufs: Vec<&String> = files.iter().filter(|f| f.ends_with(".gguf")).collect();
    if !ggufs.is_empty() {
        for g in ggufs {
            eprintln!("  fetching {g} ...");
            repo.get(g).map_err(|e| anyhow::anyhow!("fetch {g}: {e}"))?;
        }
        return Ok(());
    }
    if files.is_empty() {
        if let Some(fetched) = probe_gguf_names(&repo, id)? {
            eprintln!("  fetched {fetched}");
            return Ok(());
        }
    }

    // 3) pytorch pickle — sharded via index, else single-file `pytorch_model.bin`.
    if has("pytorch_model.bin.index.json") {
        let index_path = repo.get("pytorch_model.bin.index.json")?;
        fetch_sharded(&repo, &index_path)?;
        return Ok(());
    }
    if has("pytorch_model.bin") || repo.get("pytorch_model.bin").is_ok() {
        eprintln!("  fetched pytorch_model.bin");
        return Ok(());
    }

    anyhow::bail!(
        "'{id}': the hub repo ships no recognized weight file \
         (*.safetensors[.index.json], *.gguf, or pytorch_model.bin[.index.json])"
    )
}

/// Fetch every shard named by a `*.index.json` weight-map (shared by the
/// safetensors and pytorch sharded layouts — both use the `weight_map` schema).
fn fetch_sharded(repo: &hf_hub::api::sync::ApiRepo, index_path: &Path) -> anyhow::Result<()> {
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(index_path)?)?;
    let shards: std::collections::BTreeSet<String> = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .map(|m| {
            m.values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    if shards.is_empty() {
        anyhow::bail!("weight-map index lists no shards");
    }
    for shard in &shards {
        eprintln!("  fetching {shard} ...");
        repo.get(shard)
            .map_err(|e| anyhow::anyhow!("fetch {shard}: {e}"))?;
    }
    Ok(())
}

/// Fallback when the repo file list is unavailable: probe a short list of
/// conventional single-file GGUF names and return the first that resolves.
fn probe_gguf_names(repo: &hf_hub::api::sync::ApiRepo, id: &str) -> anyhow::Result<Option<String>> {
    let base = id.rsplit('/').next().unwrap_or(id);
    let candidates = [
        format!("{base}.gguf"),
        format!("{}.gguf", base.to_lowercase()),
        "model.gguf".to_string(),
        "ggml-model-f16.gguf".to_string(),
        "ggml-model-q4_0.gguf".to_string(),
    ];
    for name in candidates {
        if repo.get(&name).is_ok() {
            return Ok(Some(name));
        }
    }
    Ok(None)
}

/// Locate the HF cache snapshot directory for `id` (e.g. `org/name`) that holds
/// the model's weight files. Mirrors the standard hub layout
/// `<HF_HOME|~/.cache/huggingface>/hub/models--<org>--<name>/snapshots/<rev>/`,
/// picking the snapshot that actually contains an importable weight file
/// (safetensors, gguf, or pytorch pickle).
fn hf_snapshot_dir(id: &str) -> Option<PathBuf> {
    let hf_home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
            PathBuf::from(home).join(".cache/huggingface")
        });
    let repo = format!("models--{}", id.replace('/', "--"));
    let snapshots = hf_home.join("hub").join(repo).join("snapshots");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter().find(|d| has_importable_weights(d))
}

/// True iff `dir` contains at least one importable weight file (following the
/// symlinks the HF cache uses from `snapshots/` into `blobs/`) — i.e.
/// `mary::formats::detect_format` would succeed.
fn has_importable_weights(dir: &Path) -> bool {
    mary::formats::detect_format(dir).is_ok()
}
