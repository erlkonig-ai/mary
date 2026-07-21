use burn::prelude::*;

use super::attention::Flux2ParallelSelfAttention;
use super::norm::layer_norm_no_affine;
use crate::nn::weight_loader::WeightLoader;

/// Flux2SingleTransformerBlock: single-stream block that processes
/// concatenated text+image as a single stream.
///
/// Uses parallel attention + FF (Flux2ParallelSelfAttention) with shared modulation.
/// No per-block norm weights -- LayerNorm without affine is used.
pub struct Flux2SingleTransformerBlock<B: Backend> {
    pub attn: Flux2ParallelSelfAttention<B>,
    pub eps: f64,
}

impl<B: Backend> Flux2SingleTransformerBlock<B> {
    pub fn load(
        loader: &WeightLoader,
        block_idx: usize,
        num_heads: usize,
        head_dim: usize,
        inner_dim: usize,
        mlp_hidden_dim: usize,
        eps: f64,
        device: &B::Device,
    ) -> Self {
        let prefix = format!("single_transformer_blocks.{block_idx}");
        Self {
            attn: Flux2ParallelSelfAttention::load(
                loader,
                &format!("{prefix}.attn"),
                num_heads,
                head_dim,
                inner_dim,
                mlp_hidden_dim,
                device,
            ),
            eps,
        }
    }

    /// Forward pass.
    ///
    /// hidden_states: [B, S_total, inner_dim] (concatenated text + image)
    /// mod_shift: [B, 1, inner_dim]
    /// mod_scale: [B, 1, inner_dim]
    /// mod_gate: [B, 1, inner_dim]
    /// rope_cos: [S_total, D_rope]
    /// rope_sin: [S_total, D_rope]
    ///
    /// Returns: [B, S_total, inner_dim]
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        mod_shift: Tensor<B, 3>,
        mod_scale: Tensor<B, 3>,
        mod_gate: Tensor<B, 3>,
        rope_cos: Tensor<B, 2>,
        rope_sin: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        // 1. Norm + modulate
        let norm_hidden = layer_norm_no_affine(hidden_states.clone(), self.eps);
        let norm_hidden = norm_hidden * (mod_scale + 1.0) + mod_shift;

        // 2. Parallel attention + FF (fused)
        let attn_output = self.attn.forward(norm_hidden, rope_cos, rope_sin);

        // 3. Residual with gating
        hidden_states + mod_gate * attn_output
    }
}
