use burn::prelude::*;
use burn::tensor::activation::silu;

use super::embeddings::linear2d;
use crate::nn::weight_loader::WeightLoader;

// Affine-free LayerNorm is shared across ports; reuse mary::nn::norm.
// Re-exported here so siblings keep using `super::norm::layer_norm_no_affine`.
pub use crate::nn::norm::layer_norm_no_affine;

/// AdaLayerNormContinuous: adaptive normalization for transformer output.
///
/// Projects conditioning embedding to (scale, shift), then:
/// out = norm(x) * (1 + scale) + shift
///
/// linear projects from inner_dim to inner_dim * 2 (bias=False for klein).
/// norm is LayerNorm without affine.
pub struct AdaLayerNormContinuous<B: Backend> {
    pub linear_weight: Tensor<B, 2>, // [inner_dim * 2, inner_dim]
    pub eps: f64,
}

impl<B: Backend> AdaLayerNormContinuous<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, eps: f64, device: &B::Device) -> Self {
        Self {
            linear_weight: loader.load_tensor(&format!("{prefix}.linear.weight"), device),
            eps,
        }
    }

    /// Forward pass.
    /// x: [B, S, inner_dim]
    /// conditioning: [B, inner_dim] (temb)
    /// Returns: [B, S, inner_dim]
    pub fn forward(&self, x: Tensor<B, 3>, conditioning: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _dim] = conditioning.dims();

        // silu(conditioning) -> linear -> [B, inner_dim * 2]
        let emb = silu(conditioning);
        let emb = linear2d(emb, self.linear_weight.clone()); // [B, inner_dim * 2]

        // Split into scale, shift: each [B, inner_dim]
        let inner_dim = emb.dims()[1] / 2;
        let scale = emb.clone().slice([0..batch, 0..inner_dim]); // [B, inner_dim]
        let shift = emb.slice([0..batch, inner_dim..inner_dim * 2]); // [B, inner_dim]

        // Unsqueeze to [B, 1, inner_dim] for broadcasting
        let scale = scale.unsqueeze_dim::<3>(1); // [B, 1, inner_dim]
        let shift = shift.unsqueeze_dim::<3>(1); // [B, 1, inner_dim]

        // norm(x) * (1 + scale) + shift
        let normed = layer_norm_no_affine(x, self.eps);
        normed * (scale + 1.0) + shift
    }
}
