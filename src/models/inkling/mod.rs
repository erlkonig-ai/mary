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

// Nothing here is conditional any more. The module tree used to be gated three
// ways — `inkling` for the headers, `inkling-burn` for the Burn lane,
// `inkling-cuda` for the device one — and the gating was already a fiction:
// `fp4gemm` below sat outside all of it and needs cubecl, so the header-only
// build had not compiled in some time. One feature, one lane, no cfg.
pub mod burn;
pub mod attn;
pub mod block;
pub mod config;
// The NVFP4 ACTIVATION quantiser, on the device. The routed-expert lane calls
// it twice per expert; there is no host twin in the data plane to select
// between, so there is nothing to gate.
pub mod fp4quant;
pub mod layer;
pub mod layout;
pub mod load;
pub mod mlp;
pub mod mtp;
pub mod nvfp4;
pub mod pile;
// Where a Burn tensor and a raw cubecl handle are admitted to be the same
// bytes. Two functions; it is what lets the residual stream stay on the device
// across a lane boundary that is a dialect boundary and nothing more.
pub mod seam;
// The short convolution's decode step, as one kernel instead of nineteen Burn
// ops. Four run per layer and they were a third of every launch in a decode
// step; the arithmetic is 16384 multiply-adds.
pub mod sconv;
// One interface over the two places a running model's weights can come from —
// a safetensors checkpoint or a pile — plus the residency cache and the byte
// counters, which belong to the asking rather than to either storage.
pub mod source;
pub mod stack;
pub mod vision;

pub use config::InklingConfig;
// Layer 2's experts are BF16 and have no scales; the same tiling, the same
// device residency, the unscaled sibling of the instruction.
pub mod bf16gemm;
pub mod fp4gemm;
