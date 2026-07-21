//! Gemma 4 Vision Encoder: ViT with 2D RoPE, spatial pooling, QKV-norm.
//!
//! Architecture:
//!   PatchEmbedder: flatten 16x16 patches → linear proj + learned 2D position embedding
//!   Encoder: 16 transformer layers (bidirectional attention, same norm pattern as text)
//!   Pooler: 3x3 spatial average pooling + scale by sqrt(hidden_size)
//!   Projection: linear + RMSNorm to text decoder hidden_size

use burn::prelude::*;
use burn::nn::{Linear, RmsNorm};
use serde::Deserialize;

/// Clipping bounds for Gemma4ClippableLinear layers.
#[derive(Debug, Clone)]
pub struct ClipBounds {
    pub input_min: f32,
    pub input_max: f32,
    pub output_min: f32,
    pub output_max: f32,
}

impl ClipBounds {
    /// Apply input clipping, linear forward, then output clipping.
    pub fn apply<B: Backend>(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 3> {
        let x = x.clamp(self.input_min, self.input_max);
        let x = linear.forward(x);
        x.clamp(self.output_min, self.output_max)
    }
}

/// Vision encoder configuration (from config.json vision_config).
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,          // 768
    pub intermediate_size: usize,    // 3072
    pub num_hidden_layers: usize,    // 16
    pub num_attention_heads: usize,  // 12
    pub num_key_value_heads: usize,  // 12
    pub head_dim: usize,             // 64
    pub patch_size: usize,           // 16
    pub pooling_kernel_size: usize,  // 3
    #[serde(default = "default_pos_embed_size")]
    pub position_embedding_size: usize, // 10240
    pub rms_norm_eps: f64,           // 1e-6
    #[serde(default)]
    pub standardize: bool,           // false for E2B
}

fn default_pos_embed_size() -> usize { 10240 }

/// Patch embedder: flattens 16x16 RGB patches and projects to hidden_size.
/// Adds learned 2D positional embedding via one-hot lookup table.
pub struct Gemma4PatchEmbedder<B: Backend> {
    pub input_proj: Linear<B>,  // [3*16*16=768, hidden_size=768]
    /// Position embedding table: [2, position_embedding_size, hidden_size]
    /// Index 0 = x positions, index 1 = y positions
    pub position_embedding_table: Tensor<B, 3>,
}

/// A single vision encoder layer: bidirectional attention + MLP with 4 norms.
pub struct Gemma4VisionLayer<B: Backend> {
    pub q_proj: Linear<B>,
    pub q_clip: Option<ClipBounds>,
    pub k_proj: Linear<B>,
    pub k_clip: Option<ClipBounds>,
    pub v_proj: Linear<B>,
    pub v_clip: Option<ClipBounds>,
    pub o_proj: Linear<B>,
    pub o_clip: Option<ClipBounds>,
    pub q_norm: RmsNorm<B>,
    pub k_norm: RmsNorm<B>,
    pub v_norm: RmsNorm<B>,
    pub gate_proj: Linear<B>,
    pub gate_clip: Option<ClipBounds>,
    pub up_proj: Linear<B>,
    pub up_clip: Option<ClipBounds>,
    pub down_proj: Linear<B>,
    pub down_clip: Option<ClipBounds>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub pre_feedforward_layernorm: RmsNorm<B>,
    pub post_feedforward_layernorm: RmsNorm<B>,
    pub n_heads: usize,
    pub head_dim: usize,
}

/// Complete vision encoder.
pub struct Gemma4VisionEncoder<B: Backend> {
    pub patch_embedder: Gemma4PatchEmbedder<B>,
    pub layers: Vec<Gemma4VisionLayer<B>>,
    /// Pre-projection RMSNorm (no learned weights).
    pub embedding_pre_projection_norm: RmsNorm<B>,
    /// Projection from vision hidden_size to text hidden_size.
    pub embedding_projection: Linear<B>,
    /// Per-channel standardization buffers (31B/larger variants). Applied
    /// after pooling + scaling, before the pre-projection norm:
    ///   hidden = (hidden - std_bias) * std_scale.
    pub std_bias: Option<Tensor<B, 1>>,
    pub std_scale: Option<Tensor<B, 1>>,
    pub config: Gemma4VisionConfig,
}

impl<B: Backend> Gemma4VisionLayer<B> {
    /// Forward: bidirectional attention + MLP (same norm pattern as text decoder).
    /// attn_mask: additive mask [batch, 1, 1, seq] with 0 for valid, -inf for padding.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 3>,
        sin: &Tensor<B, 3>,
        attn_mask: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        // Attention block
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);

        let q = self.clipped_forward(&self.q_proj, &self.q_clip, h.clone())
            .reshape([batch, seq_len, self.n_heads, self.head_dim]).swap_dims(1, 2);
        let k = self.clipped_forward(&self.k_proj, &self.k_clip, h.clone())
            .reshape([batch, seq_len, self.n_heads, self.head_dim]).swap_dims(1, 2);
        let v = self.clipped_forward(&self.v_proj, &self.v_clip, h)
            .reshape([batch, seq_len, self.n_heads, self.head_dim]).swap_dims(1, 2);

        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);
        let v = self.v_norm.forward(v);

        // Apply 2D RoPE: split head_dim into 2 spatial parts, apply 1D RoPE to each
        let half = self.head_dim / 2;
        let q = Self::apply_2d_rope(q, cos, sin, half);
        let k = Self::apply_2d_rope(k, cos, sin, half);

        // Bidirectional attention with scale=1.0 (QKV norms handle scaling)
        let mut attn_scores = q.matmul(k.swap_dims(2, 3)); // No 1/sqrt(d) scaling

        // Apply padding mask (0 for valid, -inf for padding positions)
        if let Some(mask) = attn_mask {
            attn_scores = attn_scores + mask.clone();
        }

        let attn_weights = burn::tensor::activation::softmax(attn_scores, 3);
        let attn_out = attn_weights.matmul(v);

        let attn_out = attn_out.swap_dims(1, 2)
            .reshape([batch, seq_len, self.n_heads * self.head_dim]);
        let h = self.clipped_forward(&self.o_proj, &self.o_clip, attn_out);
        let h = self.post_attention_layernorm.forward(h);
        let x = residual + h;

        // MLP block
        let residual = x.clone();
        let h = self.pre_feedforward_layernorm.forward(x);
        let gate = burn::tensor::activation::gelu_approximate(
            self.clipped_forward(&self.gate_proj, &self.gate_clip, h.clone()));
        let up = self.clipped_forward(&self.up_proj, &self.up_clip, h);
        let h = self.clipped_forward(&self.down_proj, &self.down_clip, gate * up);
        let h = self.post_feedforward_layernorm.forward(h);
        residual + h
    }

    /// Apply a linear layer with optional clipping (Gemma4ClippableLinear).
    fn clipped_forward(&self, linear: &Linear<B>, clip: &Option<ClipBounds>, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match clip {
            Some(bounds) => bounds.apply(x, linear),
            None => linear.forward(x),
        }
    }

    /// Apply 2D RoPE by splitting head_dim into two halves (x, y) and applying 1D RoPE to each.
    pub fn apply_2d_rope(
        x: Tensor<B, 4>,       // [B, H, S, D]
        cos: &Tensor<B, 3>,    // [B, S, D] (concatenated x+y cos)
        sin: &Tensor<B, 3>,    // [B, S, D]
        half: usize,           // D/2 = size of each spatial part
    ) -> Tensor<B, 4> {
        let [_batch, _n_heads, _seq_len, _head_dim] = x.dims();

        // Split x into two spatial halves
        let x_spatial = x.clone().narrow(3, 0, half);        // [B, H, S, half]
        let y_spatial = x.narrow(3, half, half);              // [B, H, S, half]

        // Split cos/sin into two spatial halves [B, S, half] each
        let cos_x = cos.clone().narrow(2, 0, half).unsqueeze_dim::<4>(1);  // [B, 1, S, half]
        let sin_x = sin.clone().narrow(2, 0, half).unsqueeze_dim::<4>(1);
        let cos_y = cos.clone().narrow(2, half, half).unsqueeze_dim::<4>(1);
        let sin_y = sin.clone().narrow(2, half, half).unsqueeze_dim::<4>(1);

        // Apply standard rotate_half to each spatial part
        let x_rot = Self::rotate_half_apply(x_spatial, cos_x, sin_x);
        let y_rot = Self::rotate_half_apply(y_spatial, cos_y, sin_y);

        Tensor::cat(vec![x_rot, y_rot], 3)
    }

    /// Standard rotate_half + apply: x * cos + rotate_half(x) * sin
    fn rotate_half_apply(
        x: Tensor<B, 4>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [b, h, s, d] = x.dims();
        let half = d / 2;
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.clone().narrow(3, half, half);
        let rotated = Tensor::cat(vec![x2.neg(), x1], 3);

        let cos = cos.expand([b, h, s, d]);
        let sin = sin.expand([b, h, s, d]);
        x * cos + rotated * sin
    }
}

impl<B: Backend> Gemma4VisionEncoder<B> {
    /// Encode an image to soft tokens.
    ///
    /// pixel_values: [batch, num_patches, 3*patch_size*patch_size] (flattened patches)
    /// pixel_position_ids: [batch, num_patches, 2] (x, y coordinates per patch)
    ///
    /// Returns: [total_pooled_tokens, text_hidden_size] (projected to text space)
    pub fn forward(
        &self,
        pixel_values: Tensor<B, 3>,
        pixel_position_ids: Tensor<B, 3, Int>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let [batch, num_patches, _] = pixel_values.dims();
        let config = &self.config;

        // Patch embedding (normalization happens inside patch_embedder)
        let mut h = self.patch_embedder.forward(pixel_values, pixel_position_ids.clone(), device);

        // Compute 2D RoPE cos/sin
        let (cos, sin) = self.compute_2d_rope(&pixel_position_ids, device);

        // Build 2D padding mask: valid queries can attend to valid keys only
        // Mask shape: [batch, 1, seq, seq] — 0 for valid pair, -inf if either is padding
        let pos_x: Vec<i32> = pixel_position_ids.clone()
            .slice([0..batch, 0..num_patches, 0..1])
            .reshape([num_patches])
            .to_data().to_vec().unwrap();
        let is_valid: Vec<bool> = pos_x.iter().map(|&x| x >= 0).collect();

        let mut mask_data = vec![0.0f32; num_patches * num_patches];
        for i in 0..num_patches {
            for j in 0..num_patches {
                if !is_valid[i] || !is_valid[j] {
                    mask_data[i * num_patches + j] = -1e9; // Large negative (not -inf to avoid NaN in softmax)
                }
            }
        }
        let attn_mask = Tensor::<B, 1>::from_floats(&mask_data[..], device)
            .reshape([1, 1, num_patches, num_patches]);

        // Encoder layers (bidirectional)
        // Note: Python uses SDPA with a bidirectional mask that masks padding.
        // Using 2D mask [B,1,S,S] with -1e9 for padding pairs.
        for layer in self.layers.iter() {
            h = layer.forward(h, &cos, &sin, Some(&attn_mask));
        }

        // Count valid (non-padding) patches (for later stripping)
        let pos_data: Vec<i32> = pixel_position_ids.clone()
            .slice([0..1, 0..num_patches, 0..1])
            .reshape([num_patches])
            .to_data().to_vec().unwrap();
        let n_valid = pos_data.iter().filter(|&&v| v >= 0).count();

        // Spatial pooling: average k*k patches → output_length tokens
        let k = config.pooling_kernel_size;
        let output_length = num_patches / (k * k);
        let h = self.spatial_pool(h, pixel_position_ids, output_length);

        // Scale by sqrt(hidden_size) (pooler scaling)
        let scale = (config.hidden_size as f64).sqrt() as f32;
        let dt = h.dtype();
        let h = h.cast(burn::tensor::FloatDType::F32) * scale;

        // Strip padding tokens: only keep n_valid/k² valid pooled tokens
        let n_valid_pooled = n_valid / (k * k);
        let h = h.slice([0..batch, 0..n_valid_pooled, 0..config.hidden_size]);

        // Optional standardization (31B): per-channel shift + scale.
        let h = match (&self.std_bias, &self.std_scale) {
            (Some(b), Some(s)) => {
                let b = b.clone().reshape([1, 1, config.hidden_size]).cast(burn::tensor::FloatDType::F32);
                let s = s.clone().reshape([1, 1, config.hidden_size]).cast(burn::tensor::FloatDType::F32);
                (h - b) * s
            }
            _ => h,
        };

        // Norm (run in f32 to avoid f16 overflow on large scaled values)
        let h = self.embedding_pre_projection_norm.forward(h);
        
        // Cast back to original dtype before projection
        let h = h.cast(dt);
        let h = self.embedding_projection.forward(h);

        // Reshape: [batch, n_valid_pooled, text_hidden_size] → [n_valid_pooled, text_hidden_size]
        let [_, out_len, text_dim] = h.dims();
        h.reshape([batch * out_len, text_dim])
    }

    /// Compute 2D RoPE cos/sin from patch position IDs.
    pub fn compute_2d_rope(
        &self,
        position_ids: &Tensor<B, 3, Int>, // [batch, num_patches, 2]
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let head_dim = self.config.head_dim;
        let spatial_dim = head_dim / 2; // 32 per spatial dimension
        let half_spatial = spatial_dim / 2; // 16 frequencies
        let theta = 100.0f64; // vision RoPE theta

        // Compute inv_freq: 1 / (theta ^ (2i / spatial_dim))
        let inv_freq: Vec<f32> = (0..half_spatial)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / spatial_dim as f64)) as f32)
            .collect();
        let inv_freq_t = Tensor::<B, 1>::from_floats(&inv_freq[..], device)
            .reshape([1, half_spatial, 1]); // [1, freq, 1]

        let [batch, num_patches, _] = position_ids.dims();

        // For each spatial dim, compute freqs = inv_freq * positions
        let mut all_cos = Vec::new();
        let mut all_sin = Vec::new();

        for dim in 0..2 {
            // Extract positions for this spatial dimension
            let pos = position_ids.clone()
                .slice([0..batch, 0..num_patches, dim..dim+1])
                .reshape([batch, num_patches])
                .float()
                .clamp_min(0.0); // Clamp -1 padding to 0

            // Outer product: [batch, 1, num_patches] × [1, freq, 1] → [batch, freq, num_patches]
            let pos_expanded = pos.unsqueeze_dim::<3>(1); // [batch, 1, num_patches]
            let freqs = inv_freq_t.clone().expand([batch, half_spatial, 1])
                .matmul(pos_expanded); // [batch, freq, num_patches]
            let freqs = freqs.swap_dims(1, 2); // [batch, num_patches, freq]

            // Duplicate: [batch, num_patches, freq] → [batch, num_patches, 2*freq=spatial_dim]
            let cos = Tensor::cat(vec![freqs.clone().cos(), freqs.clone().cos()], 2);
            let sin = Tensor::cat(vec![freqs.clone().sin(), freqs.sin()], 2);

            all_cos.push(cos);
            all_sin.push(sin);
        }

        // Concatenate x and y: [batch, num_patches, head_dim]
        let cos = Tensor::cat(all_cos, 2);
        let sin = Tensor::cat(all_sin, 2);

        (cos, sin)
    }

    /// Simple spatial average pooling by position.
    pub fn spatial_pool(
        &self,
        hidden_states: Tensor<B, 3>, // [batch, num_patches, hidden]
        position_ids: Tensor<B, 3, Int>,
        _output_length: usize,
    ) -> Tensor<B, 3> {
        let [batch, num_patches, hidden] = hidden_states.dims();
        let k = self.config.pooling_kernel_size;

        // Determine grid dimensions from position_ids (max x+1 by max y+1)
        let device = hidden_states.device();
        let pos: Vec<i32> = position_ids.to_data().to_vec().unwrap();
        let mut max_x = 0i32;
        let mut max_y = 0i32;
        for p in 0..num_patches {
            let x = pos[p * 2];
            let y = pos[p * 2 + 1];
            if x > max_x { max_x = x; }
            if y > max_y { max_y = y; }
        }
        let pw = (max_x + 1) as usize;
        let ph = (max_y + 1) as usize;

        // For padded inputs: scatter patches to their grid positions, padding contributes 0.
        // Convert to f32 first: under the BHalf backend the tensor data is f16,
        // and a bare to_vec::<f32>() TypeMismatches. This CPU pooling scatter is
        // backend-width-agnostic.
        let hidden_states_data: Vec<f32> = hidden_states.to_data().convert::<f32>().to_vec().unwrap();
        
        let out_ph = ph / k;
        let out_pw = pw / k;
        let out_num_patches = out_ph * out_pw;
        let mut out_grid = vec![0.0f32; batch * out_num_patches * hidden];
        let mut count_grid = vec![0; batch * out_num_patches];

        for b in 0..batch {
            for p in 0..num_patches {
                let x = pos[b * num_patches * 2 + p * 2];
                let y = pos[b * num_patches * 2 + p * 2 + 1];
                if x < 0 || y < 0 { continue; } // padding
                let (x, y) = (x as usize, y as usize);
                
                let out_x = x / k;
                let out_y = y / k;
                let out_p = out_y * out_pw + out_x;
                
                count_grid[b * out_num_patches + out_p] += 1;
                
                for h in 0..hidden {
                    let src = b * num_patches * hidden + p * hidden + h;
                    let dst = b * out_num_patches * hidden + out_p * hidden + h;
                    out_grid[dst] += hidden_states_data[src];
                }
            }
        }

        let k_squared = (k * k) as f32;
        for b in 0..batch {
            for out_p in 0..out_num_patches {
                // To perfectly match PyTorch `mean_dim(2)` where padding contributed 0
                // but the denominator is always k_squared:
                for h in 0..hidden {
                    let dst = b * out_num_patches * hidden + out_p * hidden + h;
                    out_grid[dst] /= k_squared;
                }
            }
        }

        Tensor::<B, 1>::from_floats(&out_grid[..], &device)
            .reshape([batch, out_num_patches, hidden])
    }
}

impl<B: Backend> Gemma4PatchEmbedder<B> {
    /// Forward: normalize, project flattened patches + add position embeddings.
    pub fn forward(
        &self,
        pixel_values: Tensor<B, 3>,        // [batch, num_patches, 3*16*16] in [0, 1]
        position_ids: Tensor<B, 3, Int>,    // [batch, num_patches, 2]
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [batch, num_patches, _] = pixel_values.dims();
        let hidden_size = self.position_embedding_table.dims()[2];
        let pos_embed_size = self.position_embedding_table.dims()[1];

        // Normalize: 2 * (x - 0.5) = 2x - 1 (matches Python's patch_embedder.forward)
        let pixel_values = pixel_values * 2.0 - 1.0;

        // Linear projection of patches
        let h = self.input_proj.forward(pixel_values);

        // Position embeddings via one-hot lookup
        // For each spatial dim (x, y), one-hot encode position, matmul with embedding table
        let clamped = position_ids.clamp_min(0); // clamp -1 padding to 0

        let dt = h.dtype();
        let mut pos_embed = Tensor::<B, 3>::zeros([batch, num_patches, hidden_size], device)
            .cast(burn::tensor::FloatDType::F32);
        for dim in 0..2usize {
            let dim_pos = clamped.clone()
                .slice([0..batch, 0..num_patches, dim..dim+1])
                .reshape([batch, num_patches]); // [batch, num_patches]

            // One-hot: [batch, num_patches] → [batch, num_patches, pos_embed_size]
            let one_hot = burn::tensor::Tensor::<B, 2, Int>::one_hot(dim_pos, pos_embed_size)
                .float(); // [batch, num_patches, pos_embed_size]

            // Lookup: [batch, num_patches, pos_embed_size] @ [pos_embed_size, hidden_size]
            let table_slice = self.position_embedding_table.clone()
                .slice([dim..dim+1, 0..pos_embed_size, 0..hidden_size])
                .reshape([pos_embed_size, hidden_size]); // [pos_embed_size, hidden_size]

            let dim_embed = one_hot.cast(burn::tensor::FloatDType::F32)
                .matmul(table_slice.cast(burn::tensor::FloatDType::F32).unsqueeze::<3>()); // [batch, num_patches, hidden_size]
            pos_embed = pos_embed + dim_embed;
        }

        h + pos_embed.cast(dt)
    }
}
