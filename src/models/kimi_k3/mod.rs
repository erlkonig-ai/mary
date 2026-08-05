//! Kimi K3 (Moonshot) — a 2.78 T-parameter hybrid-attention MoE. Too large to
//! port in one move, so it lands operator by operator: each primitive arrives
//! with a gate against vectors captured from the shipped implementation, and
//! nothing composes until its parts have passed.
//!
//! Landed:
//!
//! * [`situ`] — the soft-clipped gated activation every MLP and expert uses.
//! * [`moe`] — the latent MoE block: the 3584-wide bottleneck, the sigmoid
//!   router with its `noaux_tc` selection bias, 16 of 896 routed experts read
//!   as MXFP4, and the 2 always-on shared experts.

pub mod moe;
pub mod situ;

pub use moe::{ActRound, LatentMoe, MoeDims};
pub use situ::{Situ, K3_BETA, K3_LINEAR_BETA};
