//! Qwen2.5-VL / BiQwen2.5 building blocks for `nomic-embed-multimodal-7b`.
//!
//! This port starts gate-first: `BiQwen2_5` is a dense retriever, so the first
//! Rust pieces are the Qwen backbone components needed to reproduce its single
//! normalized last-token embedding.

pub mod config;
pub mod embedder;
pub mod layers;
pub mod preprocess;
pub mod vision;
