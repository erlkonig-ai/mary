//! Shared Burn toolkit reused across model ports: the concrete backend alias,
//! the safetensors `WeightLoader`, `.npy` I/O, normalization primitives, and
//! the 4-bit weight codecs (`q4` for mary-quantized weights, `mxfp4` for the
//! microscaling format checkpoints ship in).
//! Model-specific layers live with their model under `mary::models`.

#[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
pub mod alias;
pub mod backend;
pub mod mxfp4;
pub mod npy;
pub mod norm;
#[cfg(feature = "q4")]
pub mod q4;
pub mod weight_loader;
