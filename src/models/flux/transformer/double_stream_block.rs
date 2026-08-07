use burn::prelude::*;

use super::attention::Flux2Attention;
use super::feed_forward::Flux2FeedForward;
use super::norm::layer_norm_no_affine;
use crate::nn::weight_loader::WeightLoader;

/// Flux2TransformerBlock: double-stream block processing image and text streams
/// with joint cross-stream attention.
///
/// Each block has:
/// - norm1, norm1_context (LayerNorm without affine) for pre-attention normalization
/// - norm2, norm2_context (LayerNorm without affine) for pre-FF normalization
/// - Joint attention (Flux2Attention)
/// - Separate feedforward networks for image and text
///
/// Modulation parameters (shift, scale, gate) come from the shared modulation layers,
/// not per-block parameters.
pub struct Flux2TransformerBlock<B: Backend> {
    pub attn: Flux2Attention<B>,
    pub ff: Flux2FeedForward<B>,
    pub ff_context: Flux2FeedForward<B>,
    pub eps: f64,
}

impl<B: Backend> Flux2TransformerBlock<B> {
    pub fn load(
        loader: &WeightLoader,
        block_idx: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f64,
        device: &B::Device,
    ) -> Self {
        let prefix = format!("transformer_blocks.{block_idx}");
        Self {
            attn: Flux2Attention::load(
                loader,
                &format!("{prefix}.attn"),
                num_heads,
                head_dim,
                device,
            ),
            ff: Flux2FeedForward::load(loader, &format!("{prefix}.ff"), device),
            ff_context: Flux2FeedForward::load(loader, &format!("{prefix}.ff_context"), device),
            eps,
        }
    }

    /// Forward pass for a double-stream block.
    ///
    /// hidden_states: [B, S_img, inner_dim] (image stream)
    /// encoder_hidden_states: [B, S_txt, inner_dim] (text stream)
    /// img_mod: ((shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp))
    /// txt_mod: ((shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp))
    /// rope_cos: [S_total, D_rope]
    /// rope_sin: [S_total, D_rope]
    ///
    /// Returns: (encoder_hidden_states, hidden_states) - text first, image second
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        // Image modulation: (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp)
        img_shift_msa: Tensor<B, 3>,
        img_scale_msa: Tensor<B, 3>,
        img_gate_msa: Tensor<B, 3>,
        img_shift_mlp: Tensor<B, 3>,
        img_scale_mlp: Tensor<B, 3>,
        img_gate_mlp: Tensor<B, 3>,
        // Text modulation
        txt_shift_msa: Tensor<B, 3>,
        txt_scale_msa: Tensor<B, 3>,
        txt_gate_msa: Tensor<B, 3>,
        txt_shift_mlp: Tensor<B, 3>,
        txt_scale_mlp: Tensor<B, 3>,
        txt_gate_mlp: Tensor<B, 3>,
        // RoPE
        rope_cos: Tensor<B, 2>,
        rope_sin: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // 1. Norm + modulate image stream for attention
        let norm_img = layer_norm_no_affine(hidden_states.clone(), self.eps);
        let norm_img = norm_img * (img_scale_msa + 1.0) + img_shift_msa;

        // 2. Norm + modulate text stream for attention
        let norm_txt = layer_norm_no_affine(encoder_hidden_states.clone(), self.eps);
        let norm_txt = norm_txt * (txt_scale_msa + 1.0) + txt_shift_msa;

        // 3. Joint attention (concatenate text+image internally)
        let (attn_img, attn_txt) = self.attn.forward(norm_img, norm_txt, rope_cos, rope_sin);

        // 4. Residual with gating for image attention
        let hidden_states = hidden_states + img_gate_msa * attn_img;

        // 5. Residual with gating for text attention
        let encoder_hidden_states = encoder_hidden_states + txt_gate_msa * attn_txt;

        // 6. Norm + modulate + FF for image
        let norm_img = layer_norm_no_affine(hidden_states.clone(), self.eps);
        let norm_img = norm_img * (img_scale_mlp + 1.0) + img_shift_mlp;
        let ff_img = self.ff.forward(norm_img);
        let hidden_states = hidden_states + img_gate_mlp * ff_img;

        // 7. Norm + modulate + FF for text
        let norm_txt = layer_norm_no_affine(encoder_hidden_states.clone(), self.eps);
        let norm_txt = norm_txt * (txt_scale_mlp + 1.0) + txt_shift_mlp;
        let ff_txt = self.ff_context.forward(norm_txt);
        let encoder_hidden_states = encoder_hidden_states + txt_gate_mlp * ff_txt;

        // Return text first, image second (matches Python)
        (encoder_hidden_states, hidden_states)
    }
}
