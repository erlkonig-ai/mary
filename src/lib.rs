//! mary — our Burn model zoo, held in TribleSpace.
//!
//! Named for Mary — Mary Wollstonecraft Shelley, who wrote the creature into
//! being, and her mother Mary Wollstonecraft. The crate brings dead weights to
//! life as composable model-graphs.
//!
//!   - [`format`] — the model-in-TribleSpace storage format (self-describing
//!     tensor leaves, modules as entities, role-edges). The substrate.
//!   - [`nn`] — shared Burn toolkit (backend, weight loader, npy, norms).
//!   - `models::*` — the ports, named by lineage: `f5` (voice), `flux` (face),
//!     `gemma` (senses), `smolvla` (body/action).
//!   - `stitch` — the franken-stitch: graph surgery across models' role-edges.

/// A generic, modality-blind ML dataset / sample-management schema on
/// TribleSpace: Dataset snapshots, multimodal Samples, TrainingRuns, and
/// Preferences. Domain-free — it knows nothing about any one dataset's labels.
pub mod dataset;
/// CLIP ViT-B/32 multi-modal embedder behind `LocalEmbedder` — image+text into
/// one L2-normalized contrastive space (cosine == dot). Gated behind `embed`.
#[cfg(feature = "embed")]
pub mod embed;
pub mod f16enc;
pub mod format;
/// Non-safetensors weight-file importers (GGUF, pickled PyTorch `state_dict`).
/// Each decodes to the same `(name, f32-data, shape)` tuples [`ingest`] consumes,
/// so every format lands in one content-addressed graph. Import-only.
#[cfg(feature = "import")]
pub mod formats;
pub mod ingest;
#[cfg(feature = "local-model")]
pub mod local;
pub mod models;
pub mod nn;
/// Persist/load model weights to a real on-disk pile (the shell-is-physics
/// endpoint). Loading from a pile is THE runtime weight path — every model
/// family goes through here — so the module is unconditional. The
/// safetensors → pile persist direction inside it is `import`-gated.
pub mod persist;
/// In-process F5-TTS voice synthesis — zero-shot, cloning the speaker from a
/// reference clip. A library seam shared by the `say` example — no separately-
/// built production binary that can drift stale against the pile format. The
/// production speak path is [`speak`] (Qwen3-TTS).
pub mod say;
/// In-process Qwen3-TTS voice synthesis (the production speak path):
/// clone the reference kit, speak arbitrary text, weights loaded from a durable
/// standalone pile — no safetensors, no separate binary in the path.
#[cfg(feature = "speak")]
pub mod speak;
/// Tokenizers as content-addressed graphs (the companion to [`format`]): vocab,
/// merges, and added-tokens as tribles so the tokenizer travels with the pile
/// instead of as a HuggingFace-cache `tokenizer.json` side-file.
pub mod tokenizer;

pub use format::*;
