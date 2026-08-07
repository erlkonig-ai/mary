//! The SmolVLA projection heads — the small linear maps that bridge the
//! padded action/state space (32) and the two tower widths (VLM 960, expert
//! 720), plus the action⊕time MLP that conditions the flow-matching denoiser
//! on the diffusion timestep.
//!
//! Shapes are measured from the checkpoint. Exact parameter key names and bias
//! presence are pinned during the first probe-parity pass against the PyTorch
//! reference; `load` takes the prefix so the names live in one place.

use super::config::SmolVlaConfig;
use crate::nn::weight_loader::WeightLoader;
use burn::prelude::*;

/// `y = x @ wᵀ (+ b)` for a `[.., in]` tensor against a `[out, in]` weight
/// (PyTorch `nn.Linear` layout).
fn linear<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    w: &Tensor<B, 2>,
    b: Option<&Tensor<B, 1>>,
) -> Tensor<B, D> {
    let out = x.matmul(w.clone().transpose().unsqueeze());
    match b {
        Some(b) => out + b.clone().unsqueeze(),
        None => out,
    }
}

/// A single `nn.Linear` (weight + optional bias) held as raw tensors.
pub struct Linear<B: Backend> {
    pub weight: Tensor<B, 2>,       // [out, in]
    pub bias: Option<Tensor<B, 1>>, // [out]
}

impl<B: Backend> Linear<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, has_bias: bool, device: &B::Device) -> Self {
        Self {
            weight: loader.load_tensor(&format!("{prefix}.weight"), device),
            bias: has_bias.then(|| loader.load_tensor(&format!("{prefix}.bias"), device)),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        linear(x, &self.weight, self.bias.as_ref())
    }
}

/// All of SmolVLA's bridge projections, gathered.
pub struct Projections<B: Backend> {
    /// state 32 -> 960 (proprioception into the VLM width).
    pub state_proj: Linear<B>,
    /// action 32 -> 720 (noised action into the expert width).
    pub action_in_proj: Linear<B>,
    /// expert 720 -> 32 (denoiser output back to action space).
    pub action_out_proj: Linear<B>,
    /// (action ⊕ time) 1440 -> 720.
    pub action_time_mlp_in: Linear<B>,
    /// 720 -> 720.
    pub action_time_mlp_out: Linear<B>,
}

impl<B: Backend> Projections<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, device: &B::Device) -> Self {
        Self {
            state_proj: Linear::load(loader, &format!("{prefix}.state_proj"), true, device),
            action_in_proj: Linear::load(loader, &format!("{prefix}.action_in_proj"), true, device),
            action_out_proj: Linear::load(
                loader,
                &format!("{prefix}.action_out_proj"),
                true,
                device,
            ),
            action_time_mlp_in: Linear::load(
                loader,
                &format!("{prefix}.action_time_mlp_in"),
                true,
                device,
            ),
            action_time_mlp_out: Linear::load(
                loader,
                &format!("{prefix}.action_time_mlp_out"),
                true,
                device,
            ),
        }
    }
}

/// Compile-time check that the projection in/out widths line up with the config
/// (documents the contract; the real numeric gate is the probe-parity pass).
pub fn assert_shapes(cfg: &SmolVlaConfig) {
    debug_assert_eq!(cfg.action_dim, 32);
    debug_assert_eq!(cfg.vlm.width, 960);
    debug_assert_eq!(cfg.expert.width, 720);
}
