//! qwen3tts — Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0), streaming-TTS
//! **candidate A**: a discrete multi-codebook LM TTS ported to Burn, cloned
//! from `ref_voice.wav`. Candidate B (Voxtral) is ported in a parallel lane;
//! same runtime, same reference, judged by ears.
//!
//! Four components (dims measured from the checkpoint — see PORT_NOTES.md):
//!   - [`talker`]  — 28×2048 Qwen3-style GQA decoder over 12.5 Hz codec
//!     frames, text interleaved into the frame stream (dual-track streaming).
//!   - [`predictor`] — 5×1024 sub-talker: codebooks 1..15 per frame. Runs on
//!     the **CPU** (Accelerate, [`cpu`]) — its 15 sequential single-token
//!     steps are dominated by per-op GPU submission overhead, not math.
//!   - [`speaker`] — ECAPA-TDNN x-vector (2048) + slaney-mel front end.
//!   - [`codec`]   — the 12 Hz tokenizer-v2 **decoder**: split-RVQ →
//!     sliding-window transformer → ConvNeXt ×4 → SnakeBeta SEANet ×480.
//!
//! The codec encoder is ported in [`encoder`] and used by the standalone Qwen
//! paths. Production `mary::speak` still consumes a captured reference-code
//! artifact so its selected voice profile remains explicit and reproducible.

pub mod codec;
pub mod config;
pub mod cpu;
pub mod encoder;
pub mod layers;
#[cfg(feature = "megakernel")]
pub mod megakernel;
pub mod pipeline;
pub mod predictor;
pub mod speaker;
pub mod talker;
pub mod tokenizer;
