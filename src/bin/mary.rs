//! `mary` — the command-line front door to mary.
//!
//! mary stores neural-network weights as content-addressed graphs in a
//! TribleSpace pile: a tensor is a self-describing leaf, a module is an entity,
//! composition is role-edges. Every mary runtime loads weights from such a
//! pile, never from safetensors. This binary is how weights get *into* one.
//!
//! Bootstrap a model in one line:
//!
//! ```text
//! # one model per pile (legacy, main branch):
//! mary import ./my-model-dir --pile my.pile --dtype f16
//! # many models in one shared MODEL_PILE, each addressable by --model-id:
//! mary import openai/clip-vit-base-patch32 --pile models.pile --model-id clip_vit_base
//! mary import HuggingFaceTB/SmolLM2-135M    --pile models.pile --model-id smollm2
//! ```
//!
//! The source is either a HuggingFace model id (auto-downloaded from the hub if
//! not already in the local cache) or a local directory of `.safetensors`
//! shards. With `--model-id`, the model becomes a proper entity
//! on the pile's `mary` branch, so many models coexist in one pile (each loaded
//! back by id) — no separate consolidation step: appending a model is just
//! another import. The resulting pile is self-contained: no safetensors needed.

use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use mary::ingest::LeafDtype;
use mary::persist::persist_safetensors_to_pile;

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
    /// Import a model's safetensors weights into a pile (the durable weight store).
    Import(ImportArgs),
    /// Load ONE model from a pile's `mary` branch by `model_id` and print its
    /// tensor count + a sample — a round-trip check of the consolidated store.
    Keys(KeysArgs),
}

#[derive(Args)]
struct ImportArgs {
    /// A HuggingFace model id (resolved from the local HF cache) OR a local
    /// directory holding the model's `.safetensors` shards.
    source: String,
    /// Output pile path. Created if absent; a model added to an existing pile
    /// is appended on the `main` branch (content-addressing dedups shared blobs).
    #[arg(long)]
    pile: PathBuf,
    /// Leaf storage dtype. `f32` is lossless (the faithful original, whatever
    /// width the source used); `f16` halves the pile for 16-bit-native weights.
    #[arg(long, value_enum, default_value_t = Dtype::F32)]
    dtype: Dtype,
    /// Model id for a shared MODEL_PILE: the model becomes a proper entity
    /// carrying this id on the pile's `mary` branch, so many models coexist in
    /// one pile (each loadable by id). Omit for a legacy single-model pile
    /// (`main` branch, untagged).
    #[arg(long)]
    model_id: Option<String>,
}

#[derive(Args)]
struct KeysArgs {
    /// The consolidated model pile to read from.
    #[arg(long)]
    pile: PathBuf,
    /// The `model_id` to load from the pile's `mary` branch.
    #[arg(long)]
    model: String,
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
    let km = mary::persist::load_keymap_from_mary_branch(&a.pile, &a.model)?;
    let mut names: Vec<&String> = km.keys().collect();
    names.sort();
    eprintln!(
        "mary keys: model_id={} on the mary branch of {} -> {} tensors",
        a.model,
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
    match &a.model_id {
        Some(id) => {
            eprintln!(
                "mary import: {} -> {} ({dt} leaves, model_id={id} on the mary branch)",
                dir.display(),
                a.pile.display()
            );
            mary::persist::persist_model_to_pile(&dir, &a.pile, a.dtype.into(), id)?;
        }
        None => {
            eprintln!(
                "mary import: {} -> {} ({dt} leaves, main branch)",
                dir.display(),
                a.pile.display()
            );
            persist_safetensors_to_pile(&dir, &a.pile, a.dtype.into())?;
        }
    }
    eprintln!("mary import: done — pile at {}", a.pile.display());
    Ok(())
}

/// Resolve the import source to a directory of `.safetensors` shards. A local
/// directory is used as-is; otherwise the source is a HuggingFace model id and
/// we locate its snapshot directory in the local HF cache. Kept inline (rather
/// than via `mary::embed::hf_cache_resolve`) so `mary import` stays behind the
/// lean `import` feature and never drags in the `embed` stack.
fn resolve_source(source: &str) -> anyhow::Result<PathBuf> {
    let p = Path::new(source);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    if let Some(dir) = hf_snapshot_dir(source) {
        return Ok(dir);
    }
    // Not a directory and not cached → treat as a HuggingFace model id and pull
    // its safetensors from the hub into the local cache, then resolve.
    eprintln!("mary import: '{source}' not in the local cache — downloading from the HuggingFace hub...");
    download_hf_safetensors(source)?;
    hf_snapshot_dir(source).ok_or_else(|| {
        anyhow::anyhow!(
            "downloaded '{source}' but found no .safetensors in its cache snapshot — the \
             model may ship weights in another format (pytorch .bin / gguf) not yet supported."
        )
    })
}

/// Pull a model's safetensors weights from the HuggingFace hub into the local
/// cache — single-file, or every shard named by the `.index.json` weight-map.
/// Weights only: mary loads from the pile, so config/tokenizer files are skipped.
fn download_hf_safetensors(id: &str) -> anyhow::Result<()> {
    use hf_hub::api::sync::Api;
    let repo = Api::new()
        .map_err(|e| anyhow::anyhow!("hf-hub api init: {e}"))?
        .model(id.to_string());
    // Sharded checkpoints carry a weight-map index; single-file ones don't.
    match repo.get("model.safetensors.index.json") {
        Ok(index_path) => {
            let index: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let shards: std::collections::BTreeSet<String> = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .map(|m| m.values().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if shards.is_empty() {
                anyhow::bail!("safetensors index for '{id}' lists no shards");
            }
            for shard in &shards {
                eprintln!("  fetching {shard} ...");
                repo.get(shard).map_err(|e| anyhow::anyhow!("fetch {shard}: {e}"))?;
            }
        }
        Err(_) => {
            eprintln!("  fetching model.safetensors ...");
            repo.get("model.safetensors").map_err(|e| {
                anyhow::anyhow!("fetch model.safetensors for '{id}': {e} (and no sharded index)")
            })?;
        }
    }
    Ok(())
}

/// Locate the HF cache snapshot directory for `id` (e.g. `org/name`) that holds
/// the model's safetensors shards. Mirrors the standard hub layout
/// `<HF_HOME|~/.cache/huggingface>/hub/models--<org>--<name>/snapshots/<rev>/`,
/// picking the snapshot that actually contains `.safetensors` files.
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
    dirs.into_iter().find(|d| has_safetensors(d))
}

/// True iff `dir` contains at least one `.safetensors` file (following the
/// symlinks the HF cache uses from `snapshots/` into `blobs/`).
fn has_safetensors(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        })
        .unwrap_or(false)
}
