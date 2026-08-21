//! Mimi neural audio codec (Kyutai Moshi / NVIDIA PersonaPlex) — Phase 1 of the
//! full-duplex PersonaPlex-7B port. Only the codec (no 7B LM) is ported here.
//!
//! The ungated checkpoint is `kyutai/moshiko-pytorch-bf16` /
//! `tokenizer-e351c8d8-checkpoint125.safetensors` (moshi state_dict layout).
//! Both encode and decode run on the CPU (Accelerate sgemm, reusing the
//! qwen3tts CPU primitives) for a single deterministic numeric path — the
//! parity-first choice for Phase 1. A Burn/CubeCL/Metal decoder is the
//! throughput follow-up; the CPU decode here is the reference to gate it
//! against.

pub mod config;
pub mod decoder;
pub mod encoder;
#[cfg(feature = "q4")]
pub mod encoder_gpu;

pub use decoder::MimiDecoder;
pub use encoder::{MimiEncoder, MimiEncoderState};
#[cfg(feature = "q4")]
pub use encoder_gpu::MimiEncoderGpu;
