//! Model ports, indexed by what the model *is* (its lineage), not by the role
//! it plays in a deployment: `f5` (TTS), `gemma` (LLM + audio), `flux` (image
//! generation). Each is a graph of `mary::format` modules built with `mary::nn`.

pub mod f5;

// Qwen2.5-VL text backbone (BiQwen2_5 / nomic-embed-multimodal-7b). Reuses
// gemma's RoPE table, so it rides the `gemma` feature.
#[cfg(feature = "gemma")]
pub mod qwen2_5_vl;

#[cfg(feature = "flux")]
pub mod flux;

#[cfg(feature = "gemma")]
pub mod gemma;

#[cfg(feature = "smolvla")]
pub mod smolvla;

#[cfg(feature = "qwen3tts")]
pub mod qwen3tts;

// Mimi neural audio codec (PersonaPlex Phase 1). Reuses the qwen3tts CPU
// primitives (Accelerate sgemm + libm gelu), so it rides the same feature.
#[cfg(feature = "qwen3tts")]
pub mod personaplex;

#[cfg(feature = "voxtral")]
pub mod voxtral;

// Kimi-K3 (2.78 T MoE vision-language). Loading skeleton only so far: config +
// checkpoint-name -> module-slot layout, no forward pass. Pure serde: no backend,
// no new dependencies.
#[cfg(feature = "k3")]
pub mod k3;
