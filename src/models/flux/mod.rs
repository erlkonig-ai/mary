//! Flux.2 diffusion-transformer port (folded in from the `avatar` crate).
//!
//! The DiT lives in [`transformer`]; the surrounding generation pipeline pulls
//! in the [`vae`], the [`text_encoder`]/[`mistral_encoder`] conditioning stacks,
//! the [`scheduler`], [`tokenizer`], and image [`utils`]. Weights come through
//! [`crate::nn::weight_loader::WeightLoader`] — including the `Pile` variant, so
//! a model reconstructed out of a `mary` pile (via [`crate::ingest`]) drops in
//! unchanged. Reuses [`crate::nn`] for backend/loader/npy/norm primitives.

pub mod mistral_encoder;
pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod utils;
pub mod vae;
