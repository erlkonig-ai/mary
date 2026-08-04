//! Kimi K3 (`model_type: kimi_linear`) — port in progress.
//!
//! 2.78 T parameters over 93 layers at hidden size 7168, of which 92 are MoE
//! (896 experts, 16 active, 2 shared) stored as MXFP4 — 92.67% of the
//! checkpoint's bytes are packed 4-bit codes plus E8M0 scales, so the config's
//! `dtype: bfloat16` describes almost nothing that is actually on disk.
//!
//! The attention stack alternates: 69 [`kda`] linear-attention layers and 24
//! full MLA layers at every fourth position. The MLA layers run NoPE
//! (`mla_use_nope: true`, no rotary embedding at all), so the KDA recurrence
//! and its four-tap short convolutions carry *all* of the model's positional
//! information — which is why [`kda`] is the first piece ported and the first
//! gated against a third-party oracle.
//!
//! Present: [`kda`] — the decay gate, the gated delta-rule recurrence, the
//! short convolution and the output gate, gated by the `kimi_kda_gate` binary
//! against `flash-linear-attention` 0.5.2's own kernels and float64 references.
//! Absent: MLA, AttnRes, the MoE router and the projections.

pub mod kda;
