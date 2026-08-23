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

// Kimi-K3 (`kimi_linear`, 2.78 T MoE). Config + checkpoint-name->module-slot
// layout, the ported operators (SiTU, KDA, MLA, AttnRes, router, latent MoE)
// and the whole decoder layer that composes them.
#[cfg(feature = "k3")]
pub mod k3;

// Inkling (Thinking Machines, 975 B / 276 B MoE, natively multimodal). One
// feature, and it names CUDA: every lane below is a Blackwell tensor-core lane.
#[cfg(feature = "inkling-cuda")]
pub mod inkling;
