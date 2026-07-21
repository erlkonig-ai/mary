//! smolvla — SmolVLA (`lerobot/smolvla_base`): a
//! vision-language-action model ported to Burn and held in TribleSpace,
//! alongside `f5` (TTS), `gemma` (LLM/audio), `flux` (image generation).
//!
//! It closes the embodied loop `observe -> reason -> act`: given the
//! current camera frame + proprioceptive state + a language instruction, it
//! denoises (flow-matching) a chunk of future expressive poses for the Reachy
//! Mini (head 6-DOF + body yaw + 2 antennas).
//!
//! Architecture (frozen VLM backbone + trained action expert), every dimension
//! measured from the checkpoint — see [`config`]:
//!   1. VLM (SmolVLM2, 350M, frozen): SigLIP vision tower + SmolLM2 text
//!      decoder, width 960, 16 layers, GQA 15:5. Produces cached KV.
//!   2. Action expert (98M, width 720, 16 layers): cross-attends the VLM KV in
//!      head-space (q 720->960, kv 720->320, o 960->720) and self-attends the
//!      action tokens; flow-matching denoiser over the action chunk.
//!   3. Projections: state 32->960, action_in 32->720, action_out 720->32,
//!      action_time_mlp 1440->720->720 (action ⊕ time embedding).
//!
//! Ported the same way as the others — the "Flux method": author the
//! architecture from the resolved shapes, then drive each layer to golden-
//! output parity against the PyTorch reference (vla-venv, lerobot 0.5.1),
//! weights stored as `mary::format` tribles. The towers + cross-attention +
//! flow-matching sampler fill in layer-by-layer behind probe gates.

pub mod config;
pub mod projections;
pub mod time;
pub mod suffix;
pub mod sampler;
pub mod rope;
pub mod layers;
pub mod denoiser;
pub mod vlm;
pub mod vision;
pub mod pipeline;
