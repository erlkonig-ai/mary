//! Kimi K3 (`model_type: kimi_linear`) — port in progress.
//!
//! 2.78 T parameters over 93 layers at hidden size 7168, of which 92 are MoE
//! (896 experts, 16 active, 2 shared) stored as MXFP4. The attention stack
//! alternates 69 KDA linear-attention layers with 24 full MLA layers.
//!
//! Present here: [`router`] — the `noaux_tc` sigmoid gate with its trained
//! `e_score_correction_bias`, gated by the `k3_router_gate` binary against the
//! whole-layer oracle's forward-hook captures of the shipped
//! `KimiMoEGate.forward`.

pub mod router;
