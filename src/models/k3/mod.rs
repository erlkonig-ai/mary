//! Kimi-K3 — a 2.78 T-parameter MoE vision-language model.
//!
//! What is here so far is the **loading skeleton**: the config, and the map from
//! every safetensors tensor name to the module slot it fills. No forward pass,
//! no tensor data. The point of landing this half first is that it is the half
//! that fails quietly — a mis-shifted layer index or a mistyped weight name
//! yields a model that loads and runs and is simply wrong, whereas a broken
//! kernel usually announces itself.
//!
//! Shape of the model, for orientation:
//!
//! * 93 decoder layers, `hidden_size` 7168. 69 run KDA (a gated-delta linear
//!   recurrence with kernel-4 short convolutions); the other 24 run MLA, at
//!   every fourth layer plus the last one. MLA is NoPE — `mla_use_nope` is set
//!   and `rotary_emb` is `None` — so *all* position information in the model
//!   comes from the KDA recurrence and its convolutions.
//! * Layer 0 has a dense MLP; layers 1..93 are MoE, 896 routed experts each,
//!   16 active per token, plus 2 shared experts fused into one wide MLP. The
//!   routed experts work in a 3584-wide latent, not the residual stream.
//! * The routed experts are MXFP4: 4-bit E2M1 codes, two per byte, with one
//!   E8M0 exponent per 32 elements. That is 92.67% of the checkpoint's bytes.
//! * Every sublayer's input is an AttnRes mix: a per-token softmax over a bank
//!   of depth checkpoints plus the running accumulator, rather than a plain
//!   residual add.
//!
//! See [`config`] for the layer-index base trap, and [`layout`] for how the
//! name mapping is checked in both directions.

pub mod config;
pub mod layout;
pub mod mla;

pub use config::{AttnKind, K3Config, K3TextConfig, K3VisionConfig, LinearAttnConfig};
pub use layout::{describe, for_each_slot, Dtype, Shape, Slot, TensorSlot};
pub use mla::{MlaBlock, MlaConfig, MlaKvCache, MlaTrace, MlaWeights, Precision};
