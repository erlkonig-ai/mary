use burn::prelude::*;
use burn::tensor::activation::silu;

use super::embeddings::linear2d;
use crate::nn::weight_loader::WeightLoader;

/// Flux2Modulation: SiLU -> Linear -> produces modulation params (shift, scale, gate).
///
/// For double-stream blocks: mod_param_sets=2 -> 6 outputs per block
/// For single-stream blocks: mod_param_sets=1 -> 3 outputs per block
pub struct Flux2Modulation<B: Backend> {
    pub linear_weight: Tensor<B, 2>, // [dim * 3 * mod_param_sets, dim]
    pub mod_param_sets: usize,
}

impl<B: Backend> Flux2Modulation<B> {
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        mod_param_sets: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            linear_weight: loader.load_tensor(&format!("{prefix}.linear.weight"), device),
            mod_param_sets,
        }
    }

    /// Forward: silu(temb) -> linear -> modulation parameters.
    /// temb: [B, dim]
    /// Returns: [B, dim * 3 * mod_param_sets]
    pub fn forward(&self, temb: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = silu(temb);
        linear2d(h, self.linear_weight.clone())
    }
}

/// Split modulation output into (shift, scale, gate) tuples.
///
/// mod_output: [B, dim * 3 * num_sets] (from Flux2Modulation::forward)
/// Returns: Vec of (shift, scale, gate) tuples, each element is [B, 1, dim]
///
/// The unsqueeze to [B, 1, dim] allows broadcasting with [B, S, dim] hidden states.
pub fn split_modulation<B: Backend>(
    mod_output: Tensor<B, 2>,
    num_sets: usize,
) -> Vec<(Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>)> {
    let [batch, total_dim] = mod_output.dims();
    let chunk_size = total_dim / (3 * num_sets);

    // Reshape to [B, 1, total_dim] first, then split
    let mod_3d = mod_output.unsqueeze_dim::<3>(1); // [B, 1, total_dim]

    let mut result = Vec::with_capacity(num_sets);
    for i in 0..num_sets {
        let base = i * 3 * chunk_size;
        let shift = mod_3d
            .clone()
            .slice([0..batch, 0..1, base..base + chunk_size]); // [B, 1, dim]
        let scale =
            mod_3d
                .clone()
                .slice([0..batch, 0..1, base + chunk_size..base + 2 * chunk_size]);
        let gate =
            mod_3d
                .clone()
                .slice([0..batch, 0..1, base + 2 * chunk_size..base + 3 * chunk_size]);
        result.push((shift, scale, gate));
    }
    result
}
