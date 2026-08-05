//! Shared Burn toolkit reused across model ports: the concrete backend alias,
//! the safetensors `WeightLoader`, `.npy`/`.npz` I/O, and normalization primitives.
//! Model-specific layers live with their model under `mary::models`.

#[cfg(all(any(feature = "qwen3tts", feature = "voxtral"), target_os = "macos"))]
pub mod alias;
pub mod backend;
pub mod npy;
pub mod npz;
pub mod norm;
#[cfg(feature = "q4")]
pub mod q4;
pub mod weight_loader;
