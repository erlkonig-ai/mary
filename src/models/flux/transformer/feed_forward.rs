use burn::prelude::*;
use burn::tensor::activation::silu;

use super::embeddings::linear3d;
use crate::nn::weight_loader::WeightLoader;

/// Flux2FeedForward: Linear(dim, inner_dim*2) -> SwiGLU -> Linear(inner_dim, dim)
///
/// inner_dim = dim * mlp_ratio (e.g. 3072 * 3.0 = 9216)
/// SwiGLU: split input in half, silu(first_half) * second_half
/// All linear layers are bias=False.
pub struct Flux2FeedForward<B: Backend> {
    pub linear_in_weight: Tensor<B, 2>,  // [inner_dim * 2, dim]
    pub linear_out_weight: Tensor<B, 2>, // [dim, inner_dim]
}

impl<B: Backend> Flux2FeedForward<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, device: &B::Device) -> Self {
        Self {
            linear_in_weight: loader.load_tensor(&format!("{prefix}.linear_in.weight"), device),
            linear_out_weight: loader.load_tensor(&format!("{prefix}.linear_out.weight"), device),
        }
    }

    /// Forward: linear_in -> SwiGLU -> linear_out
    /// x: [B, S, dim]
    /// Returns: [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // linear_in: [B, S, dim] -> [B, S, inner_dim*2]
        let h = linear3d(x, self.linear_in_weight.clone());

        // SwiGLU: split in half along last dim
        let h = swiglu(h);

        // linear_out: [B, S, inner_dim] -> [B, S, dim]
        linear3d(h, self.linear_out_weight.clone())
    }
}

/// SwiGLU activation: split input in half along last dim, silu(first) * second.
/// x: [B, S, 2*D]
/// Returns: [B, S, D]
pub fn swiglu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, s, total] = x.dims();
    let half = total / 2;
    let x1 = x.clone().slice([0..b, 0..s, 0..half]);
    let x2 = x.slice([0..b, 0..s, half..total]);
    silu(x1) * x2
}
