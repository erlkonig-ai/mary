//! Inkling (Thinking Machines) — a 42-layer (276 B / 12 B active) and 66-layer
//! (975 B / 41 B active) sparse-MoE decoder with native audio and vision input.
//!
//! What is here so far: the configuration and the checkpoint-name to
//! module-slot layout, gated as a bijection against a real checkpoint by
//! `inkling_layout_gate`.
//!
//! Relative to Kimi-K3 the attention is conventional — GQA, not KDA plus MLA —
//! but the block adds a depthwise short convolution on the attention input, on
//! K, on V and into the MLP; QK-norm; and a rank-16 learned relative-position
//! logit path in place of RoPE. The MoE block is close enough to K3's
//! sigmoid router with shared experts and gate bias to port with parameter
//! changes. The FP4 decode is *not*: K3 is MXFP4 (E8M0 scales, block 32),
//! Inkling is NVFP4 (E4M3 scales, block 16, plus a per-expert F32 second level).

#[cfg(feature = "inkling-burn")]
pub mod burn;
pub mod attn;
pub mod block;
pub mod config;
pub mod layer;
pub mod layout;
pub mod load;
pub mod mlp;
pub mod nvfp4;
pub mod stack;
pub mod vision;

pub use config::InklingConfig;
