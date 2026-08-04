//! Kimi K3 (Moonshot) — a 2.78 T-parameter hybrid-attention MoE. Too large to
//! port in one move, so it lands operator by operator: each primitive arrives
//! with a gate against vectors captured from the shipped implementation, and
//! nothing composes until its parts have passed.
//!
//! Landed:
//!
//! * [`situ`] — the soft-clipped gated activation every MLP and expert uses.

pub mod situ;

pub use situ::{Situ, K3_BETA, K3_LINEAR_BETA};
