//! gemma — Gemma 4 multimodal, ported to Burn 0.21 inside mary: audio and
//! vision encoders feed the text decoder (speech + image understanding).
//!
//! Text decoder + audio encoder + vision encoder for Gemma 4 (E2B/E4B/…).
//! Its own self-contained toolkit (config/rope/layers/weights/…) — deliberately
//! NOT deduped against `mary::nn`, whose primitives differ.
//!
//! Supported architectures:
//! - Mistral family (text decoder shell)
//! - Gemma 4: E2B, E4B, 26B-A4B (MoE), 31B (text + vision + audio)
//!
//! Features: GQA, RoPE, SwiGLU/GELU, RMSNorm, KV cache, sliding window,
//! safetensors loading.

pub mod config;
pub mod rope;
pub mod layers;
pub mod decoder;
pub mod weights;
pub mod turbo_quant;
pub mod gpu_quant;
pub mod metal_device;
pub mod dyntensor;
pub mod gemma4;
pub mod lora;

// Re-export KV cache types for convenience.
pub use layers::{KvCache, LayerCaches};
pub use layers::{QuantizedKvCache, QuantizedLayerCaches, QuantConfig, QuantBits};
pub use turbo_quant::{TurboQuantConfig, TurboQuantCtx};
pub use layers::{TurboQuantKvCache, TurboQuantLayerCaches};
pub use gpu_quant::{GpuQuantKvCache, GpuQuantLayerCaches};
pub use gpu_quant::{GpuTurboQuantKvCache, GpuTurboQuantLayerCaches, GpuTurboQuantCtx};
