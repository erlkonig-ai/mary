use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::embeddings::linear3d;
use super::feed_forward::swiglu;
use super::rope::apply_rotary_emb;
use crate::nn::weight_loader::WeightLoader;

/// RMS norm for per-head normalization: x: [B, S, H, D_head], weight: [D_head]
fn rms_norm_head<B: Backend>(x: Tensor<B, 4>, weight: Tensor<B, 1>, eps: f64) -> Tensor<B, 4> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(3); // [B, S, H, 1]
    let inv_rms = (variance + eps).sqrt().recip();
    let normed = x * inv_rms;
    // weight: [D_head] -> [1, 1, 1, D_head]
    normed
        * weight
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
}

/// Flux2Attention for double-stream blocks.
///
/// Contains separate Q/K/V projections for the image stream and
/// add_{q,k,v}_proj for the text stream. QK-norm with RMSNorm.
///
/// Joint attention: concatenate text+image Q/K/V along sequence dim,
/// apply RoPE, SDPA, split output back.
pub struct Flux2Attention<B: Backend> {
    // Image stream projections
    pub to_q_weight: Tensor<B, 2>,   // [inner_dim, inner_dim]
    pub to_k_weight: Tensor<B, 2>,   // [inner_dim, inner_dim]
    pub to_v_weight: Tensor<B, 2>,   // [inner_dim, inner_dim]
    pub to_out_weight: Tensor<B, 2>, // [inner_dim, inner_dim] (to_out.0)

    // Text stream projections
    pub add_q_proj_weight: Tensor<B, 2>, // [inner_dim, inner_dim]
    pub add_k_proj_weight: Tensor<B, 2>, // [inner_dim, inner_dim]
    pub add_v_proj_weight: Tensor<B, 2>, // [inner_dim, inner_dim]
    pub to_add_out_weight: Tensor<B, 2>, // [inner_dim, inner_dim]

    // QK norms (RMSNorm with learned weight)
    pub norm_q_weight: Tensor<B, 1>,       // [head_dim]
    pub norm_k_weight: Tensor<B, 1>,       // [head_dim]
    pub norm_added_q_weight: Tensor<B, 1>, // [head_dim]
    pub norm_added_k_weight: Tensor<B, 1>, // [head_dim]

    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> Flux2Attention<B> {
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        num_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            to_q_weight: loader.load_tensor(&format!("{prefix}.to_q.weight"), device),
            to_k_weight: loader.load_tensor(&format!("{prefix}.to_k.weight"), device),
            to_v_weight: loader.load_tensor(&format!("{prefix}.to_v.weight"), device),
            to_out_weight: loader.load_tensor(&format!("{prefix}.to_out.0.weight"), device),
            add_q_proj_weight: loader.load_tensor(&format!("{prefix}.add_q_proj.weight"), device),
            add_k_proj_weight: loader.load_tensor(&format!("{prefix}.add_k_proj.weight"), device),
            add_v_proj_weight: loader.load_tensor(&format!("{prefix}.add_v_proj.weight"), device),
            to_add_out_weight: loader.load_tensor(&format!("{prefix}.to_add_out.weight"), device),
            norm_q_weight: loader.load_tensor(&format!("{prefix}.norm_q.weight"), device),
            norm_k_weight: loader.load_tensor(&format!("{prefix}.norm_k.weight"), device),
            norm_added_q_weight: loader
                .load_tensor(&format!("{prefix}.norm_added_q.weight"), device),
            norm_added_k_weight: loader
                .load_tensor(&format!("{prefix}.norm_added_k.weight"), device),
            num_heads,
            head_dim,
        }
    }

    /// Joint attention forward pass.
    ///
    /// hidden_states: [B, S_img, inner_dim] (image stream, already modulated+normed)
    /// encoder_hidden_states: [B, S_txt, inner_dim] (text stream, already modulated+normed)
    /// rope_cos: [S_txt+S_img, D_rope] (concatenated text+image RoPE cos)
    /// rope_sin: [S_txt+S_img, D_rope] (concatenated text+image RoPE sin)
    ///
    /// Returns: (img_attn_output, txt_attn_output) each [B, S_*, inner_dim]
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        rope_cos: Tensor<B, 2>,
        rope_sin: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, s_img, _dim] = hidden_states.dims();
        let s_txt = encoder_hidden_states.dims()[1];
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        // Project image stream Q/K/V: [B, S_img, inner_dim]
        let q_img = linear3d(hidden_states.clone(), self.to_q_weight.clone());
        let k_img = linear3d(hidden_states.clone(), self.to_k_weight.clone());
        let v_img = linear3d(hidden_states, self.to_v_weight.clone());

        // Project text stream Q/K/V: [B, S_txt, inner_dim]
        let q_txt = linear3d(encoder_hidden_states.clone(), self.add_q_proj_weight.clone());
        let k_txt = linear3d(encoder_hidden_states.clone(), self.add_k_proj_weight.clone());
        let v_txt = linear3d(encoder_hidden_states, self.add_v_proj_weight.clone());

        // Reshape to [B, S, H, D_head] (unflatten last dim)
        let q_img = q_img.reshape([batch, s_img, num_heads, head_dim]);
        let k_img = k_img.reshape([batch, s_img, num_heads, head_dim]);
        let v_img = v_img.reshape([batch, s_img, num_heads, head_dim]);

        let q_txt = q_txt.reshape([batch, s_txt, num_heads, head_dim]);
        let k_txt = k_txt.reshape([batch, s_txt, num_heads, head_dim]);
        let v_txt = v_txt.reshape([batch, s_txt, num_heads, head_dim]);

        // QK-norm (RMSNorm per head)
        let q_img = rms_norm_head(q_img, self.norm_q_weight.clone(), 1e-6);
        let k_img = rms_norm_head(k_img, self.norm_k_weight.clone(), 1e-6);
        let q_txt = rms_norm_head(q_txt, self.norm_added_q_weight.clone(), 1e-6);
        let k_txt = rms_norm_head(k_txt, self.norm_added_k_weight.clone(), 1e-6);

        // Concatenate text + image along sequence dim: [B, S_txt+S_img, H, D]
        let q = Tensor::cat(vec![q_txt, q_img], 1);
        let k = Tensor::cat(vec![k_txt, k_img], 1);
        let v = Tensor::cat(vec![v_txt, v_img], 1);

        // Apply RoPE (sequence_dim=1 format: [B, S, H, D])
        let q = apply_rotary_emb(q, rope_cos.clone(), rope_sin.clone());
        let k = apply_rotary_emb(k, rope_cos, rope_sin);

        // SDPA: transpose to [B, H, S, D] for matmul
        let q = q.swap_dims(1, 2); // [B, H, S, D]
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let scale = (head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.transpose()) / scale; // [B, H, S_total, S_total]
        let attn_weights = softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v); // [B, H, S_total, D]

        // Transpose back and flatten heads: [B, S_total, H*D]
        let s_total = s_txt + s_img;
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, s_total, num_heads * head_dim]);

        // Split back into text and image outputs
        let txt_output = attn_output
            .clone()
            .slice([0..batch, 0..s_txt, 0..num_heads * head_dim]);
        let img_output =
            attn_output.slice([0..batch, s_txt..s_total, 0..num_heads * head_dim]);

        // Output projections
        let img_output = linear3d(img_output, self.to_out_weight.clone());
        let txt_output = linear3d(txt_output, self.to_add_out_weight.clone());

        (img_output, txt_output)
    }
}

/// Flux2ParallelSelfAttention for single-stream blocks.
///
/// Fused projection: to_qkv_mlp_proj outputs QKV + MLP input simultaneously.
/// QK-norm + RoPE, SDPA, SwiGLU on MLP path, concatenate attn+mlp, final projection.
pub struct Flux2ParallelSelfAttention<B: Backend> {
    pub to_qkv_mlp_proj_weight: Tensor<B, 2>, // [inner_dim*3 + mlp_hidden_dim*2, inner_dim]
    pub to_out_weight: Tensor<B, 2>,           // [inner_dim, inner_dim + mlp_hidden_dim]
    pub norm_q_weight: Tensor<B, 1>,           // [head_dim]
    pub norm_k_weight: Tensor<B, 1>,           // [head_dim]

    pub num_heads: usize,
    pub head_dim: usize,
    pub inner_dim: usize,
    pub mlp_hidden_dim: usize,
}

impl<B: Backend> Flux2ParallelSelfAttention<B> {
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        num_heads: usize,
        head_dim: usize,
        inner_dim: usize,
        mlp_hidden_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            to_qkv_mlp_proj_weight: loader
                .load_tensor(&format!("{prefix}.to_qkv_mlp_proj.weight"), device),
            to_out_weight: loader.load_tensor(&format!("{prefix}.to_out.weight"), device),
            norm_q_weight: loader.load_tensor(&format!("{prefix}.norm_q.weight"), device),
            norm_k_weight: loader.load_tensor(&format!("{prefix}.norm_k.weight"), device),
            num_heads,
            head_dim,
            inner_dim,
            mlp_hidden_dim,
        }
    }

    /// Forward pass.
    ///
    /// hidden_states: [B, S, inner_dim] (already modulated+normed)
    /// rope_cos: [S, D_rope]
    /// rope_sin: [S, D_rope]
    ///
    /// Returns: [B, S, inner_dim]
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        rope_cos: Tensor<B, 2>,
        rope_sin: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _dim] = hidden_states.dims();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let inner_dim = self.inner_dim;
        let mlp_hidden_dim = self.mlp_hidden_dim;

        // Fused projection: [B, S, inner_dim] -> [B, S, inner_dim*3 + mlp_hidden_dim*2]
        let projected = linear3d(hidden_states, self.to_qkv_mlp_proj_weight.clone());

        // Split into QKV and MLP input
        let qkv_dim = inner_dim * 3;
        let mlp_dim = mlp_hidden_dim * 2;

        let qkv = projected
            .clone()
            .slice([0..batch, 0..seq_len, 0..qkv_dim]);
        let mlp_input =
            projected.slice([0..batch, 0..seq_len, qkv_dim..qkv_dim + mlp_dim]);

        // Split QKV into Q, K, V: each [B, S, inner_dim]
        let q = qkv
            .clone()
            .slice([0..batch, 0..seq_len, 0..inner_dim]);
        let k = qkv
            .clone()
            .slice([0..batch, 0..seq_len, inner_dim..inner_dim * 2]);
        let v =
            qkv.slice([0..batch, 0..seq_len, inner_dim * 2..inner_dim * 3]);

        // Reshape to [B, S, H, D_head]
        let q = q.reshape([batch, seq_len, num_heads, head_dim]);
        let k = k.reshape([batch, seq_len, num_heads, head_dim]);
        let v = v.reshape([batch, seq_len, num_heads, head_dim]);

        // QK-norm
        let q = rms_norm_head(q, self.norm_q_weight.clone(), 1e-6);
        let k = rms_norm_head(k, self.norm_k_weight.clone(), 1e-6);

        // Apply RoPE (sequence_dim=1: [B, S, H, D])
        let q = apply_rotary_emb(q, rope_cos.clone(), rope_sin.clone());
        let k = apply_rotary_emb(k, rope_cos, rope_sin);

        // SDPA: transpose to [B, H, S, D]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let scale = (head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.transpose()) / scale;
        let attn_weights = softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v); // [B, H, S, D]

        // Flatten heads: [B, S, H*D]
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, num_heads * head_dim]);

        // SwiGLU on MLP path
        let mlp_output = swiglu(mlp_input); // [B, S, mlp_hidden_dim]

        // Concatenate attn + mlp outputs: [B, S, inner_dim + mlp_hidden_dim]
        let combined = Tensor::cat(vec![attn_output, mlp_output], 2);

        // Final output projection: [B, S, inner_dim]
        linear3d(combined, self.to_out_weight.clone())
    }
}
