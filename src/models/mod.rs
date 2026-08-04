//! Model ports, indexed by what the model *is* (its lineage), not by the role
//! it plays in a deployment: `f5` (TTS), `gemma` (LLM + audio), `flux` (image
//! generation). Each is a graph of `mary::format` modules built with `mary::nn`.

pub mod f5;

// Kimi K3 (Moonshot) -- 2.78 T-param hybrid-attention MoE, landing primitive
// by primitive. Ungated: pure Burn tensor ops, no dependency of its own.
pub mod kimi_k3;

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
