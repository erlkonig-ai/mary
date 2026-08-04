//! Gemma 4 multimodal model support.
//!
//! Supports all Gemma 4 variants: E2B, E4B, 26B-A4B (MoE), 31B.
//! Text + Vision + Audio (E2B/E4B only).
//!
//! Architecture:
//! - Interleaved sliding-window and full attention layers
//! - Dual RoPE (standard local, proportional global with partial rotation)
//! - Per-Layer Embeddings (PLE) for parameter-efficient small models
//! - K=V optimization in global attention (larger models)
//! - Logit softcapping via Burn's attention kernel
//! - GELU (pytorch tanh) activation throughout

pub mod audio;
pub mod audio_load;
pub mod audio_preprocess;
pub mod config;
pub mod decoder;
pub mod hear;
pub mod layers;
pub mod lm;
pub mod preprocess;
pub mod vision;
pub mod weights;
