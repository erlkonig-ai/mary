use burn::prelude::*;
use burn::tensor::TensorData;

use super::config::Mistral3Config;
use crate::models::flux::text_encoder::layers::rms_norm_3d;
use crate::models::flux::text_encoder::rope::RotaryEmbedding;
use crate::nn::weight_loader::WeightLoader;

/// Apply a linear layer to a 3D tensor: x @ W^T
/// x: [B, L, in_dim], weight: [out_dim, in_dim] -> [B, L, out_dim]
fn linear3d<B: Backend>(x: Tensor<B, 3>, weight: Tensor<B, 2>) -> Tensor<B, 3> {
    let wt = weight.transpose().unsqueeze::<3>(); // [1, in_dim, out_dim]
    x.matmul(wt)
}

/// Mistral3 attention with Grouped Query Attention (GQA).
/// Unlike Qwen3, Mistral does NOT have QK-norm.
/// head_dim is explicit (128), so Q projects to num_heads*head_dim=4096 (not hidden_size=5120).
pub struct MistralAttention<B: Backend> {
    pub q_proj_weight: Tensor<B, 2>, // [num_heads * head_dim, hidden_size]  = [4096, 5120]
    pub k_proj_weight: Tensor<B, 2>, // [num_kv_heads * head_dim, hidden_size] = [1024, 5120]
    pub v_proj_weight: Tensor<B, 2>, // [num_kv_heads * head_dim, hidden_size] = [1024, 5120]
    pub o_proj_weight: Tensor<B, 2>, // [hidden_size, num_heads * head_dim] = [5120, 4096]
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> MistralAttention<B> {
    pub fn load(
        loader: &WeightLoader,
        config: &Mistral3Config,
        layer_idx: usize,
        device: &B::Device,
    ) -> Self {
        let prefix = format!("language_model.model.layers.{}.self_attn", layer_idx);
        Self {
            q_proj_weight: loader.load_tensor(&format!("{prefix}.q_proj.weight"), device),
            k_proj_weight: loader.load_tensor(&format!("{prefix}.k_proj.weight"), device),
            v_proj_weight: loader.load_tensor(&format!("{prefix}.v_proj.weight"), device),
            o_proj_weight: loader.load_tensor(&format!("{prefix}.o_proj.weight"), device),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
        }
    }

    /// Forward pass with GQA and RoPE (no QK-norm, unlike Qwen3).
    /// x: [B, L, hidden_size]
    /// attn_mask: optional pre-computed [1, 1, L, L] combined causal + padding mask
    /// Returns: [B, L, hidden_size]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        attn_mask: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let kv_groups = num_heads / num_kv_heads;

        // Project Q, K, V
        let q = linear3d(x.clone(), self.q_proj_weight.clone());
        let k = linear3d(x.clone(), self.k_proj_weight.clone());
        let v = linear3d(x, self.v_proj_weight.clone());

        // Reshape to [B, L, num_heads, head_dim] then transpose to [B, num_heads, L, head_dim]
        let q = q
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, num_kv_heads, head_dim])
            .swap_dims(1, 2);

        // No QK-norm for Mistral (unlike Qwen3)

        // Apply RoPE
        let q = rope.apply(q, 0);
        let k = rope.apply(k, 0);

        // GQA: repeat K,V heads to match Q heads
        let k = Self::repeat_kv(k, kv_groups);
        let v = Self::repeat_kv(v, kv_groups);

        // Scaled dot-product attention
        let scale = (head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.transpose()) / scale; // [B, H, L, L]

        // Apply attention mask (combined causal + padding, or causal only)
        let attn_weights = match attn_mask {
            Some(mask) => attn_weights + mask.clone(),
            None => Self::apply_causal_mask(attn_weights, seq_len),
        };

        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v); // [B, H, L, D]

        // Reshape back: [B, H, L, D] -> [B, L, H*D]
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, num_heads * head_dim]);

        // Output projection: [B, L, num_heads*head_dim] -> [B, L, hidden_size]
        linear3d(attn_output, self.o_proj_weight.clone())
    }

    /// Repeat KV heads: [B, num_kv_heads, L, D] -> [B, num_heads, L, D]
    fn repeat_kv(x: Tensor<B, 4>, num_groups: usize) -> Tensor<B, 4> {
        if num_groups == 1 {
            return x;
        }
        let [b, kv_heads, l, d] = x.dims();
        x.unsqueeze_dim::<5>(2)
            .repeat_dim(2, num_groups)
            .reshape([b, kv_heads * num_groups, l, d])
    }

    /// Apply causal attention mask (upper triangular = -inf).
    fn apply_causal_mask(attn: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
        let device = attn.device();
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::<B, 2>::from_data(
            TensorData::new(mask_data, [seq_len, seq_len]),
            &device,
        );
        let mask = mask.reshape([1, 1, seq_len, seq_len]);
        attn + mask
    }
}

/// Mistral3 MLP with SwiGLU activation (same structure as Qwen3).
pub struct MistralMLP<B: Backend> {
    pub gate_proj_weight: Tensor<B, 2>, // [intermediate_size, hidden_size]
    pub up_proj_weight: Tensor<B, 2>,   // [intermediate_size, hidden_size]
    pub down_proj_weight: Tensor<B, 2>, // [hidden_size, intermediate_size]
}

impl<B: Backend> MistralMLP<B> {
    pub fn load(loader: &WeightLoader, layer_idx: usize, device: &B::Device) -> Self {
        let prefix = format!("language_model.model.layers.{}.mlp", layer_idx);
        Self {
            gate_proj_weight: loader.load_tensor(&format!("{prefix}.gate_proj.weight"), device),
            up_proj_weight: loader.load_tensor(&format!("{prefix}.up_proj.weight"), device),
            down_proj_weight: loader.load_tensor(&format!("{prefix}.down_proj.weight"), device),
        }
    }

    /// Forward: SwiGLU(gate_proj(x)) * up_proj(x) -> down_proj
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = linear3d(x.clone(), self.gate_proj_weight.clone());
        let gate = burn::tensor::activation::silu(gate);
        let up = linear3d(x, self.up_proj_weight.clone());
        let hidden = gate * up;
        linear3d(hidden, self.down_proj_weight.clone())
    }
}

/// A single Mistral3 decoder layer.
pub struct MistralDecoderLayer<B: Backend> {
    pub self_attn: MistralAttention<B>,
    pub mlp: MistralMLP<B>,
    pub input_layernorm_weight: Tensor<B, 1>,
    pub post_attention_layernorm_weight: Tensor<B, 1>,
    pub eps: f64,
}

impl<B: Backend> MistralDecoderLayer<B> {
    pub fn load(
        loader: &WeightLoader,
        config: &Mistral3Config,
        layer_idx: usize,
        device: &B::Device,
    ) -> Self {
        let prefix = format!("language_model.model.layers.{}", layer_idx);
        Self {
            self_attn: MistralAttention::load(loader, config, layer_idx, device),
            mlp: MistralMLP::load(loader, layer_idx, device),
            input_layernorm_weight: loader
                .load_tensor(&format!("{prefix}.input_layernorm.weight"), device),
            post_attention_layernorm_weight: loader
                .load_tensor(&format!("{prefix}.post_attention_layernorm.weight"), device),
            eps: config.rms_norm_eps,
        }
    }

    /// Forward pass: pre-norm attention + pre-norm MLP with residuals.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        attn_mask: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        // Pre-attention norm
        let normed = rms_norm_3d(x.clone(), self.input_layernorm_weight.clone(), self.eps);
        let attn_out = self.self_attn.forward(normed, rope, attn_mask);
        let x = x + attn_out;

        // Pre-MLP norm
        let normed = rms_norm_3d(
            x.clone(),
            self.post_attention_layernorm_weight.clone(),
            self.eps,
        );
        let mlp_out = self.mlp.forward(normed);
        x + mlp_out
    }
}
