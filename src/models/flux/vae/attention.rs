use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::resnet::group_norm_4d;
use crate::nn::weight_loader::WeightLoader;

/// VAE mid-block self-attention.
///
/// This implements the deprecated-style attention block used in diffusers VAE:
/// GroupNorm -> reshape to [B, C, H*W] -> Q, K, V projections -> scaled dot-product
/// attention (single head, head_dim = channels) -> output projection -> reshape back.
///
/// Weight names follow the pattern:
///   {prefix}.group_norm.weight/bias
///   {prefix}.to_q.weight/bias
///   {prefix}.to_k.weight/bias
///   {prefix}.to_v.weight/bias
///   {prefix}.to_out.0.weight/bias
pub struct VaeAttention<B: Backend> {
    pub group_norm_weight: Tensor<B, 1>,
    pub group_norm_bias: Tensor<B, 1>,
    pub to_q_weight: Tensor<B, 2>,
    pub to_q_bias: Tensor<B, 1>,
    pub to_k_weight: Tensor<B, 2>,
    pub to_k_bias: Tensor<B, 1>,
    pub to_v_weight: Tensor<B, 2>,
    pub to_v_bias: Tensor<B, 1>,
    pub to_out_weight: Tensor<B, 2>,
    pub to_out_bias: Tensor<B, 1>,
    pub channels: usize,
    pub num_groups: usize,
}

impl<B: Backend> VaeAttention<B> {
    /// Load from safetensors.
    /// Example prefix: "decoder.mid_block.attentions.0"
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        channels: usize,
        num_groups: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            group_norm_weight: loader.load_tensor(&format!("{prefix}.group_norm.weight"), device),
            group_norm_bias: loader.load_tensor(&format!("{prefix}.group_norm.bias"), device),
            to_q_weight: loader.load_tensor(&format!("{prefix}.to_q.weight"), device),
            to_q_bias: loader.load_tensor(&format!("{prefix}.to_q.bias"), device),
            to_k_weight: loader.load_tensor(&format!("{prefix}.to_k.weight"), device),
            to_k_bias: loader.load_tensor(&format!("{prefix}.to_k.bias"), device),
            to_v_weight: loader.load_tensor(&format!("{prefix}.to_v.weight"), device),
            to_v_bias: loader.load_tensor(&format!("{prefix}.to_v.bias"), device),
            to_out_weight: loader.load_tensor(&format!("{prefix}.to_out.0.weight"), device),
            to_out_bias: loader.load_tensor(&format!("{prefix}.to_out.0.bias"), device),
            channels,
            num_groups,
        }
    }

    /// Linear projection: x @ W^T + b
    /// x: [B, S, C], weight: [C_out, C_in], bias: [C_out]
    /// Returns: [B, S, C_out]
    fn linear_3d(x: Tensor<B, 3>, weight: Tensor<B, 2>, bias: Tensor<B, 1>) -> Tensor<B, 3> {
        // weight: [C_out, C_in] -> transpose -> [C_in, C_out]
        // x: [B, S, C_in] @ [C_in, C_out] = [B, S, C_out]
        let [_batch, _seq, _c_in] = x.dims();
        let [c_out, _] = weight.dims();
        let out = x.matmul(weight.transpose().unsqueeze::<3>()); // [B, S, C_out]
        // Add bias: [C_out] -> [1, 1, C_out]
        out + bias.reshape([1, 1, c_out])
    }

    /// Forward pass with residual connection.
    /// Input: [B, C, H, W] -> Output: [B, C, H, W]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let seq_len = height * width;
        let eps = 1e-6;

        // Store residual
        let residual = x.clone();

        // Group normalization
        let h = group_norm_4d(
            x,
            self.num_groups,
            self.group_norm_weight.clone(),
            self.group_norm_bias.clone(),
            eps,
        );

        // Reshape to [B, C, H*W] then transpose to [B, H*W, C]
        let h = h.reshape([batch, channels, seq_len]).swap_dims(1, 2);

        // Q, K, V projections: [B, H*W, C]
        let q = Self::linear_3d(h.clone(), self.to_q_weight.clone(), self.to_q_bias.clone());
        let k = Self::linear_3d(h.clone(), self.to_k_weight.clone(), self.to_k_bias.clone());
        let v = Self::linear_3d(h, self.to_v_weight.clone(), self.to_v_bias.clone());

        // Scaled dot-product attention (single head, head_dim = channels)
        let scale = (channels as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(1, 2)) / scale; // [B, H*W, H*W]
        let attn_weights = softmax(attn_weights, 2);
        let attn_output = attn_weights.matmul(v); // [B, H*W, C]

        // Output projection
        let attn_output = Self::linear_3d(
            attn_output,
            self.to_out_weight.clone(),
            self.to_out_bias.clone(),
        );

        // Reshape back to [B, C, H, W]
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, channels, height, width]);

        // Residual connection
        residual + attn_output
    }
}
