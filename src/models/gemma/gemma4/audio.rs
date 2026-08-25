//! Gemma 4 audio encoder (Conformer).
//!
//! Architecture: log-mel audio features → SubSampleConvProjection (2x stride-2
//! Conv2d + LayerNorm + ReLU, then reshape + Linear to hidden_size) →
//! relative positional encoding → 12× Conformer layers (half-step FF, chunked
//! local attention with relative-position bias, depthwise conv1d, half-step
//! FF) → output_proj (Linear) → multimodal embedder (RMSNorm + Linear to text
//! hidden_size). Mirrors the vision encoder's role but with a Conformer stack
//! and chunked attention instead of ViT.

use burn::nn::{
    LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d, PaddingConfig2d, RmsNorm,
    RmsNormConfig,
    conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig},
};
use burn::prelude::*;
use burn::tensor::Tensor;

use super::config::Gemma4AudioConfig;
use super::vision::ClipBounds;

/// Wrap safetensors shards as a named-tensor fetch for the `load_with`
/// constructors (EXACT names, f32 data + shape).
#[cfg(feature = "import")]
fn shard_fetch<'a>(
    shards: &'a [safetensors::SafeTensors<'a>],
) -> impl Fn(&str) -> Option<(Vec<f32>, Vec<usize>)> + 'a {
    use crate::models::gemma::weights::bytes_to_f32_pub;
    move |name: &str| {
        shards.iter().find_map(|st| {
            st.tensor(name)
                .ok()
                .map(|v| (bytes_to_f32_pub(v.data(), v.dtype()), v.shape().to_vec()))
        })
    }
}

/// One stride-2 Conv2d + LayerNorm(across channel dim) + ReLU layer.
///
/// Input:  [B, Cin,  T,  F]
/// Output: [B, Cout, T/2, F/2]  (stride=2, padding=1, kernel=3)
pub struct SubSampleConvLayer<B: Backend> {
    pub conv: Conv2d<B>,
    pub norm: LayerNorm<B>,
}

impl<B: Backend> SubSampleConvLayer<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Conv2d: [B, Cin, T, F] → [B, Cout, T/2, F/2]
        let h = self.conv.forward(x);
        // LayerNorm over the channel dim: permute to [B, T/2, F/2, C] for norm
        let h = h.permute([0, 2, 3, 1]);
        let h = self.norm.forward(h);
        // ReLU, then permute back to [B, C, T/2, F/2]
        let h = burn::tensor::activation::relu(h);
        h.permute([0, 3, 1, 2])
    }
}

/// Two-stage sub-sample conv projection + flatten + Linear to hidden_size.
pub struct SubSampleConvProjection<B: Backend> {
    pub layer0: SubSampleConvLayer<B>,
    pub layer1: SubSampleConvLayer<B>,
    /// Flattened projection: [ (c0/4) * c1, hidden_size ]. Here c0/4 is the
    /// residual feature-dim after two stride-2 conv passes over the feature
    /// axis starting from feat_dim=128 (so feat_dim/4 = 32, times c1=32 = 1024).
    pub input_proj: burn::nn::Linear<B>,
    pub input_feat_dim: usize,
    pub hidden_size: usize,
}

impl<B: Backend> SubSampleConvProjection<B> {
    /// input_features: [B, T, F] (log-mel).
    /// Returns: [B, T/4, hidden_size].
    pub fn forward(&self, input_features: Tensor<B, 3>) -> Tensor<B, 3> {
        // [B, T, F] → [B, 1, T, F] (add channel dim)
        let x = input_features.unsqueeze_dim::<4>(1);
        let x = self.layer0.forward(x); // [B, c0, T/2, F/2]
        let x = self.layer1.forward(x); // [B, c1, T/4, F/4]

        // Reshape to [B, T/4, C * F/4] — Python: permute(0,2,3,1).reshape(B,T,-1)
        let [b, c, t2, f2] = x.dims();
        let x = x.permute([0, 2, 3, 1]).reshape([b, t2, c * f2]);
        // input_proj: [..., c1 * f/4] → [..., hidden_size]
        self.input_proj.forward(x)
    }
}

/// Construct a SubSampleConvProjection from pre-loaded weight tensors.
pub fn build_subsample<B: Backend>(
    config: &Gemma4AudioConfig,
    layer0_conv: Tensor<B, 4>,
    layer0_norm: Tensor<B, 1>,
    layer1_conv: Tensor<B, 4>,
    layer1_norm: Tensor<B, 1>,
    input_proj_w: Tensor<B, 2>,
    input_feat_dim: usize,
    device: &B::Device,
) -> SubSampleConvProjection<B> {
    use burn::module::Param;

    let c0 = config.subsampling_conv_channels[0];
    let c1 = config.subsampling_conv_channels[1];

    // Conv2d shapes: [Cout, Cin, kH, kW] (same as PyTorch)
    let conv0 = {
        let mut m = Conv2dConfig::new([1, c0], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(false)
            .init::<B>(device);
        m.weight = Param::from_tensor(layer0_conv);
        m
    };
    let conv1 = {
        let mut m = Conv2dConfig::new([c0, c1], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(false)
            .init::<B>(device);
        m.weight = Param::from_tensor(layer1_conv);
        m
    };
    let norm0 = {
        let mut m = LayerNormConfig::new(c0)
            .with_epsilon(config.rms_norm_eps)
            .init::<B>(device);
        m.gamma = Param::from_tensor(layer0_norm);
        // Python uses `bias=False` — no beta. Match that.
        m.beta = None;
        m
    };
    let norm1 = {
        let mut m = LayerNormConfig::new(c1)
            .with_epsilon(config.rms_norm_eps)
            .init::<B>(device);
        m.gamma = Param::from_tensor(layer1_norm);
        m.beta = None;
        m
    };

    // input_proj_linear: Python weight shape [out, in] = [hidden, c1 * F/4]
    // Burn's Linear weight is [in, out] — we need to transpose.
    let [out_dim, in_dim] = input_proj_w.dims();
    assert_eq!(out_dim, config.hidden_size);
    let input_proj = {
        let mut m = burn::nn::LinearConfig::new(in_dim, out_dim)
            .with_bias(false)
            .init::<B>(device);
        m.weight = Param::from_tensor(input_proj_w.swap_dims(0, 1));
        m
    };

    SubSampleConvProjection {
        layer0: SubSampleConvLayer {
            conv: conv0,
            norm: norm0,
        },
        layer1: SubSampleConvLayer {
            conv: conv1,
            norm: norm1,
        },
        input_proj,
        input_feat_dim,
        hidden_size: config.hidden_size,
    }
}

// ---------------------------------------------------------------------------
// Relative positional encoding — fixed sinusoidal table
// ---------------------------------------------------------------------------

/// Precompute the audio encoder's relative position embeddings.
///
/// Python (`Gemma4AudioRelPositionalEncoding.forward`) hard-codes a
/// `torch.arange(12, -1, -1)` position range regardless of
/// `attention_context_size`, producing a [13, 1, hidden_size] embedding with
/// [sin..., cos...] concatenation. We reproduce the same table once.
pub fn rel_positional_encoding<B: Backend>(
    config: &Gemma4AudioConfig,
    device: &B::Device,
) -> Tensor<B, 3> {
    let hidden = config.hidden_size;
    let num_timescales = hidden / 2;
    let log_increment = (10_000.0f64.ln()) / ((num_timescales.max(2) - 1) as f64);
    let inv_timescales: Vec<f32> = (0..num_timescales)
        .map(|i| (-log_increment * i as f64).exp() as f32)
        .collect();

    // position_ids = [12, 11, ..., 0], length 13
    let positions: Vec<f32> = (0..13).rev().map(|i| i as f32).collect();

    let mut sin_table = vec![0.0f32; 13 * num_timescales];
    let mut cos_table = vec![0.0f32; 13 * num_timescales];
    for (pi, &p) in positions.iter().enumerate() {
        for (fi, &inv) in inv_timescales.iter().enumerate() {
            let t = p * inv;
            sin_table[pi * num_timescales + fi] = t.sin();
            cos_table[pi * num_timescales + fi] = t.cos();
        }
    }

    // Concat sin + cos along last dim → [13, hidden]. Unsqueeze to [13, 1, hidden].
    let mut out = vec![0.0f32; 13 * hidden];
    for pi in 0..13 {
        let row_out = pi * hidden;
        let row_in = pi * num_timescales;
        out[row_out..row_out + num_timescales]
            .copy_from_slice(&sin_table[row_in..row_in + num_timescales]);
        out[row_out + num_timescales..row_out + hidden]
            .copy_from_slice(&cos_table[row_in..row_in + num_timescales]);
    }

    Tensor::<B, 1>::from_floats(&out[..], device).reshape([1, 13, hidden])
}

// ---------------------------------------------------------------------------
// Clippable linear (shared pattern with vision)
// ---------------------------------------------------------------------------

/// A Linear wrapped with input/output clamping bounds from the checkpoint.
pub struct ClippableLinear<B: Backend> {
    pub linear: Linear<B>,
    pub clip: Option<ClipBounds>,
}

impl<B: Backend> ClippableLinear<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match &self.clip {
            Some(c) => c.apply(x, &self.linear),
            None => self.linear.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Feed-forward (half-step, Conformer-style)
// ---------------------------------------------------------------------------

/// Conformer half-step feed-forward:
///   y = residual + post_layer_scale * post_ln(ffw2(silu(ffw1(pre_ln(x)))))
pub struct AudioFeedForward<B: Backend> {
    pub pre_layer_norm: RmsNorm<B>,
    pub ffw_layer_1: ClippableLinear<B>,
    pub ffw_layer_2: ClippableLinear<B>,
    pub post_layer_norm: RmsNorm<B>,
    pub post_layer_scale: f32,
    pub gradient_clipping: f32,
}

impl<B: Backend> AudioFeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let g = self.gradient_clipping;
        let residual = x.clone();
        let h = x.clamp(-g, g);
        let h = self.pre_layer_norm.forward(h);
        let h = self.ffw_layer_1.forward(h);
        let h = burn::tensor::activation::silu(h);
        let h = self.ffw_layer_2.forward(h);
        let h = h.clamp(-g, g);
        let h = self.post_layer_norm.forward(h);
        residual + h.mul_scalar(self.post_layer_scale)
    }
}

// ---------------------------------------------------------------------------
// Light conv1d module (GLU + depthwise causal conv1d)
// ---------------------------------------------------------------------------

/// Lightweight conv block:
///   y = residual + linear_end(silu(conv_norm(depthwise_conv1d(glu(linear_start(pre_ln(x)))))))
/// `depthwise_conv1d` is causal with `groups=hidden_size` and left-padding by
/// `kernel_size - 1` so the output length matches the input.
pub struct AudioLightConv1d<B: Backend> {
    pub pre_layer_norm: RmsNorm<B>,
    pub linear_start: ClippableLinear<B>,
    pub depthwise_conv1d: Conv1d<B>,
    pub conv_norm: RmsNorm<B>,
    pub linear_end: ClippableLinear<B>,
    pub gradient_clipping: f32,
    pub kernel_size: usize,
}

impl<B: Backend> AudioLightConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = self.pre_layer_norm.forward(x);
        let h = self.linear_start.forward(h); // [B, T, 2*hidden]
        // GLU along last dim: split (a, b), y = a * sigmoid(b)
        let [b_, t, two_h] = h.dims();
        let half = two_h / 2;
        let a = h.clone().slice([0..b_, 0..t, 0..half]);
        let b = h.slice([0..b_, 0..t, half..two_h]);
        let h = a * burn::tensor::activation::sigmoid(b); // [B, T, hidden]

        // Depthwise causal Conv1d: [B, T, H] → [B, H, T] → pad-left (k-1) → conv → [B, H, T'] → [B, T', H]
        let h = h.swap_dims(1, 2); // [B, H, T]
        let left_pad = self.kernel_size - 1;
        // Pad only on the left of the time axis by prepending zeros.
        let [bb, hh, _tt] = h.dims();
        let pad = Tensor::<B, 3>::zeros([bb, hh, left_pad], &h.device());
        let h = Tensor::<B, 3>::cat(vec![pad, h], 2); // [B, H, T + k-1]
        let h = self.depthwise_conv1d.forward(h); // [B, H, T]
        let h = h.swap_dims(1, 2); // [B, T, H]

        let g = self.gradient_clipping;
        let h = h.clamp(-g, g);
        let h = self.conv_norm.forward(h);
        let h = burn::tensor::activation::silu(h);
        let h = self.linear_end.forward(h);
        residual + h
    }
}

// ---------------------------------------------------------------------------
// Builders for these pieces from loaded weight tensors
// ---------------------------------------------------------------------------

/// Build a ClippableLinear from a raw 2D weight tensor (shape [out, in],
/// as stored by HuggingFace safetensors) plus optional clip bounds.
pub fn build_clippable_linear<B: Backend>(
    weight_out_in: Tensor<B, 2>,
    clip: Option<ClipBounds>,
    device: &B::Device,
) -> ClippableLinear<B> {
    use burn::module::Param;
    let [out_dim, in_dim] = weight_out_in.dims();
    let mut linear = LinearConfig::new(in_dim, out_dim)
        .with_bias(false)
        .init::<B>(device);
    linear.weight = Param::from_tensor(weight_out_in.swap_dims(0, 1));
    ClippableLinear { linear, clip }
}

/// Build an RmsNorm with a preloaded 1D gamma tensor.
pub fn build_rms_norm<B: Backend>(gamma: Tensor<B, 1>, eps: f64, device: &B::Device) -> RmsNorm<B> {
    use burn::module::Param;
    let dim = gamma.dims()[0];
    let mut m = RmsNormConfig::new(dim).with_epsilon(eps).init::<B>(device);
    m.gamma = Param::from_tensor(gamma);
    m
}

/// Build a depthwise Conv1d with preloaded [out, 1, kernel] weight. Left-pad
/// happens in `AudioLightConv1d::forward`, so the conv itself uses valid
/// padding.
pub fn build_depthwise_conv1d<B: Backend>(
    weight: Tensor<B, 3>,
    channels: usize,
    kernel_size: usize,
    device: &B::Device,
) -> Conv1d<B> {
    use burn::module::Param;
    let mut m = Conv1dConfig::new(channels, channels, kernel_size)
        .with_groups(channels)
        .with_padding(PaddingConfig1d::Valid)
        .with_bias(false)
        .init::<B>(device);
    m.weight = Param::from_tensor(weight);
    m
}

// ---------------------------------------------------------------------------
// Audio attention (chunked local with relative-position bias + tanh softcap)
// ---------------------------------------------------------------------------

/// Gemma 4 audio's chunked local attention. Queries are split into
/// non-overlapping blocks of `chunk_size`; keys/values provide a context
/// window of `chunk_size + max_past + max_future` around each block. Adds a
/// relative-position bias via a Transformer-XL style shift trick and softcaps
/// attention logits through `tanh`.
pub struct AudioAttention<B: Backend> {
    pub q_proj: ClippableLinear<B>,
    pub k_proj: ClippableLinear<B>,
    pub v_proj: ClippableLinear<B>,
    pub post: ClippableLinear<B>,
    pub relative_k_proj: Linear<B>,
    pub per_dim_scale: Tensor<B, 1>, // [head_dim]
    pub num_heads: usize,
    pub head_dim: usize,
    pub chunk_size: usize,
    pub max_past_horizon: usize,
    pub max_future_horizon: usize,
    pub context_size: usize,
    pub softcap: f32,
    pub q_scale: f32,
    pub k_scale: f32,
    pub invalid_logits_value: f32,
}

impl<B: Backend> AudioAttention<B> {
    /// Debug-only passthrough to internal helpers.
    pub fn convert_to_block_pub(&self, x: Tensor<B, 4>) -> Tensor<B, 5> {
        self.convert_to_block(x)
    }
    pub fn extract_block_context_pub(&self, x: Tensor<B, 4>) -> Tensor<B, 5> {
        self.extract_block_context(x)
    }
    pub fn rel_shift_pub(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        self.rel_shift(x)
    }

    /// Split [B, T, H, D] → [B, num_blocks, chunk_size, H, D] (pad tail).
    fn convert_to_block(&self, x: Tensor<B, 4>) -> Tensor<B, 5> {
        let [b, t, h, d] = x.dims();
        let cs = self.chunk_size;
        let num_blocks = (t + cs - 1) / cs;
        let pad = num_blocks * cs - t;
        let x = if pad > 0 {
            let padding = Tensor::<B, 4>::zeros([b, pad, h, d], &x.device());
            Tensor::<B, 4>::cat(vec![x, padding], 1)
        } else {
            x
        };
        x.reshape([b, num_blocks, cs, h, d])
    }

    /// Extract overlapping context windows: [B, T, H, D] → [B, num_blocks, context_size, H, D].
    /// Pads time by `max_past_horizon` on the left and
    /// `max_future_horizon + chunk_size - 1` on the right, then unfolds.
    fn extract_block_context(&self, x: Tensor<B, 4>) -> Tensor<B, 5> {
        let [b, _t, h, d] = x.dims();
        let cs = self.chunk_size;
        let past = self.max_past_horizon;
        let future = self.max_future_horizon;
        let device = x.device();

        let pad_left = Tensor::<B, 4>::zeros([b, past, h, d], &device);
        let pad_right = Tensor::<B, 4>::zeros([b, future + cs - 1, h, d], &device);
        let padded = Tensor::<B, 4>::cat(vec![pad_left, x, pad_right], 1);

        // unfold along dim=1, window=context_size, step=chunk_size
        // Input shape [B, T', H, D] → [B, num_windows, H, D, context_size]
        let unfolded: Tensor<B, 5> = padded.unfold::<5, _>(1usize, self.context_size, cs);
        // Move the last dim (context_size) to position 2 → [B, num_windows, context_size, H, D]
        unfolded.permute([0, 1, 4, 2, 3])
    }

    /// Transformer-XL relative position shift for blocked attention.
    /// Input [B, H, nb, cs, position_length=13] → [B, H, nb, cs, context_size].
    fn rel_shift(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [b, h, nb, cs, pl] = x.dims();
        let ctx = self.context_size;
        let pad_amount = ctx + 1 - pl;
        let device = x.device();
        let pad = Tensor::<B, 5>::zeros([b, h, nb, cs, pad_amount], &device);
        let padded = Tensor::<B, 5>::cat(vec![x, pad], 4); // [B, H, nb, cs, ctx+1]
        let flat = padded.reshape([b, h, nb, cs * (ctx + 1)]);
        let sliced = flat.slice([0..b, 0..h, 0..nb, 0..(cs * ctx)]);
        sliced.reshape([b, h, nb, cs, ctx])
    }

    /// Forward. `hidden_states`: [B, T, hidden]. `position_embeddings`:
    /// [1, 13, hidden]. `mask_5d`: [B, 1, num_blocks, chunk_size, context_size]
    /// boolean (true = valid pair) or None.
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_embeddings: Tensor<B, 3>,
        mask_5d: Option<Tensor<B, 5>>, // additive mask: 0 or -inf at last dim
    ) -> Tensor<B, 3> {
        let [b, t, _] = hidden_states.dims();
        let h = self.num_heads;
        let d = self.head_dim;

        let q = self
            .q_proj
            .forward(hidden_states.clone())
            .reshape([b, t, h, d]);
        let k = self
            .k_proj
            .forward(hidden_states.clone())
            .reshape([b, t, h, d]);
        let v = self.v_proj.forward(hidden_states).reshape([b, t, h, d]);

        // q = q * q_scale * softplus(per_dim_scale)
        let softplus = {
            let x = self.per_dim_scale.clone();
            // softplus(x) = ln(1 + exp(x))
            let ones = Tensor::<B, 1>::ones([d], &x.device());
            (x.exp() + ones).log()
        };
        let q = q.mul_scalar(self.q_scale) * softplus.reshape([1, 1, 1, d]);
        let k = k.mul_scalar(self.k_scale);

        let q_blocks = self.convert_to_block(q); // [B, nb, cs, H, D]
        let k_ctx = self.extract_block_context(k); // [B, nb, ctx, H, D]
        let v_ctx = self.extract_block_context(v); // [B, nb, ctx, H, D]
        let [_, nb, _, _, _] = q_blocks.dims();
        let cs = self.chunk_size;
        let ctx = self.context_size;

        // rel_k: relative_k_proj(position_embeddings) → [1, 13, H*D] → view [13, H, D]
        let rel_k = self.relative_k_proj.forward(position_embeddings);
        let rel_k = rel_k.reshape([13, h, d]);

        // queries: [B, H, nb, cs, D]
        let queries = q_blocks.permute([0, 3, 1, 2, 4]);
        // keys for matrix_ac: [B, H, nb, D, ctx]
        let keys = k_ctx.clone().permute([0, 3, 1, 4, 2]);

        // matrix_ac = queries @ keys → [B, H, nb, cs, ctx]
        let matrix_ac: Tensor<B, 5> = {
            // Collapse [B, H, nb] into a batch dim for matmul:
            // queries: [B*H*nb, cs, D], keys: [B*H*nb, D, ctx]
            let q4 = queries.clone().reshape([b * h * nb, cs, d]);
            let k4 = keys.reshape([b * h * nb, d, ctx]);
            let out = q4.matmul(k4); // [B*H*nb, cs, ctx]
            out.reshape([b, h, nb, cs, ctx])
        };

        // matrix_bd = queries_flat @ rel_k.permute(1, 2, 0) → [B, H, nb*cs, 13]
        let queries_flat = queries.reshape([b, h, nb * cs, d]);
        // rel_k.permute(1, 2, 0): [H, D, 13]
        let rk_perm = rel_k.permute([1, 2, 0]);
        // For batched matmul: [B, H, nb*cs, D] @ [1, H, D, 13] = [B, H, nb*cs, 13]
        let rk_b = rk_perm.unsqueeze::<4>(); // [1, H, D, 13]
        let matrix_bd_2d = queries_flat.matmul(rk_b); // broadcasts on dim 0
        let matrix_bd = matrix_bd_2d.reshape([b, h, nb, cs, 13]);
        let matrix_bd = self.rel_shift(matrix_bd); // [B, H, nb, cs, ctx]

        // softcap * tanh((ac + bd) / softcap)
        let attn = matrix_ac + matrix_bd;
        let attn = (attn.div_scalar(self.softcap))
            .tanh()
            .mul_scalar(self.softcap);

        // Apply mask: set invalid positions to invalid_logits_value
        let attn = if let Some(m) = mask_5d {
            attn + m // caller supplies additive mask (0 valid, invalid_logits_value otherwise)
        } else {
            attn
        };

        let attn_weights = burn::tensor::activation::softmax(attn, 4); // softmax over context dim
        // out = attn_weights @ v.permute(0, 3, 1, 2, 4) = [B, H, nb, cs, ctx] @ [B, H, nb, ctx, D]
        let values = v_ctx.permute([0, 3, 1, 2, 4]); // [B, H, nb, ctx, D]
        let aw4 = attn_weights.reshape([b * h * nb, cs, ctx]);
        let v4 = values.reshape([b * h * nb, ctx, d]);
        let out5 = aw4.matmul(v4).reshape([b, h, nb, cs, d]);

        // Reshape → [B, nb, cs, H, D] → [B, nb*cs, H*D], then crop to T, then post
        let out = out5.permute([0, 2, 3, 1, 4]).reshape([b, nb * cs, h * d]);
        let out = out.slice([0..b, 0..t, 0..(h * d)]);
        self.post.forward(out)
    }
}

// ---------------------------------------------------------------------------
// Multimodal embedder for audio (RMSNorm without learnable scale + Linear)
// ---------------------------------------------------------------------------

/// Projects audio tower output features into the text decoder's embedding
/// space. Mirrors `Gemma4MultimodalEmbedder` in Python: RMSNorm with
/// `with_scale=False` (no learned gamma) followed by an unbiased Linear.
pub struct AudioEmbedder<B: Backend> {
    pub embedding_projection: Linear<B>,
    pub eps: f64,
}

impl<B: Backend> AudioEmbedder<B> {
    /// Input:  [N, multimodal_hidden_size]
    /// Output: [N, text_hidden_size]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // RMSNorm without learned scale: x * rsqrt(mean(x^2) + eps)
        let eps = self.eps as f32;
        let sq = x.clone() * x.clone();
        let [_, _d] = sq.dims();
        let mean_sq = sq.mean_dim(1); // [N, 1]
        let rms = (mean_sq + eps).sqrt();
        let normed = x / rms;
        self.embedding_projection.forward(normed)
    }

    /// Build from any named-tensor fetch (EXACT names, f32 data + shape).
    /// Shared by the safetensors and pile loaders.
    pub fn load_with(
        fetch: &dyn Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
        eps: f64,
        device: &B::Device,
    ) -> Self {
        use burn::module::Param;
        let name = "model.embed_audio.embedding_projection.weight";
        let (data, shape) = fetch(name).unwrap_or_else(|| panic!("tensor not found: {name}"));
        let w = Tensor::<B, 1>::from_floats(&data[..], device).reshape([shape[0], shape[1]]);
        let [out_dim, in_dim] = w.dims();
        let mut lin = LinearConfig::new(in_dim, out_dim)
            .with_bias(false)
            .init::<B>(device);
        lin.weight = Param::from_tensor(w.swap_dims(0, 1));
        AudioEmbedder {
            embedding_projection: lin,
            eps,
        }
    }

    #[cfg(feature = "import")]
    pub fn load_from_shards(
        shards: &[safetensors::SafeTensors<'_>],
        eps: f64,
        device: &B::Device,
    ) -> Self {
        Self::load_with(&shard_fetch(shards), eps, device)
    }
}

// ---------------------------------------------------------------------------
// Conformer layer + stacked model
// ---------------------------------------------------------------------------

pub struct AudioLayer<B: Backend> {
    pub feed_forward1: AudioFeedForward<B>,
    pub feed_forward2: AudioFeedForward<B>,
    pub self_attn: AudioAttention<B>,
    pub lconv1d: AudioLightConv1d<B>,
    pub norm_pre_attn: RmsNorm<B>,
    pub norm_post_attn: RmsNorm<B>,
    pub norm_out: RmsNorm<B>,
    pub gradient_clipping: f32,
}

impl<B: Backend> AudioLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        pos: &Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 5>>,
    ) -> Tensor<B, 3> {
        let g = self.gradient_clipping;

        // FF1 (half-step)
        let x = self.feed_forward1.forward(x);
        let residual = x.clone();

        // Pre-attn norm + attention + post-attn norm + residual
        let h = x.clamp(-g, g);
        let h = self.norm_pre_attn.forward(h);
        let h = self.self_attn.forward(h, pos.clone(), attn_mask);
        let h = h.clamp(-g, g);
        let h = self.norm_post_attn.forward(h);
        let x = residual + h;

        // LightConv1d
        let x = self.lconv1d.forward(x);
        // FF2 (half-step)
        let x = self.feed_forward2.forward(x);
        // Final norm
        let x = x.clamp(-g, g);
        self.norm_out.forward(x)
    }
}

/// Full audio encoder: subsample → pos_emb → 12× Conformer → output_proj.
pub struct AudioModel<B: Backend> {
    pub subsample: SubSampleConvProjection<B>,
    pub rel_pos: Tensor<B, 3>, // [1, 13, hidden]
    pub layers: Vec<AudioLayer<B>>,
    pub output_proj: Linear<B>, // hidden_size → output_proj_dims (with bias)
    pub config: Gemma4AudioConfig,
}

impl<B: Backend> AudioModel<B> {
    /// Build the 5D additive attention mask for a given input sequence
    /// length. No padding is assumed — the mask enforces the sliding
    /// window (past=context_left-1, future=context_right) after T has been
    /// subsampled by 4x.
    ///
    /// Matches Python's `create_bidirectional_mask` + sliding-window-mask +
    /// `_convert_4d_mask_to_blocked_5d` composition, but for no-padding
    /// inputs (which is all we need for inference with real waveforms).
    pub fn build_mask_5d(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 5> {
        let cs = self.config.attention_chunk_size;
        let past = self.config.attention_context_left - 1;
        let future = self.config.attention_context_right;
        let ctx = cs + past + future;
        let num_blocks = (seq_len + cs - 1) / cs;
        let padded = num_blocks * cs;
        let invalid = self.config.attention_invalid_logits_value;

        // For each block b (rows chunk_size) and each offset o (ctx positions),
        // kv position in the padded-then-extended time axis = b*cs + o - past.
        // Valid iff:
        //   kv_pos in [0, seq_len)                   (not padding)
        //   AND |q - kv| window: q = b*cs + row; kv = b*cs + o - past
        //     => q - kv = past - o + row ∈ [-future, past]
        //     => past - o + row ≥ -future   => o ≤ past + future + row
        //     => past - o + row ≤ past      => o ≥ row
        //   (simplifies to: row ≤ o ≤ row + past + future, i.e. sliding window)
        //
        // But Python also masks positions where the underlying 4D mask
        // (q, kv) pair has q beyond seq_len (end-padded). We enforce both.
        let mut data = vec![invalid; num_blocks * cs * ctx];
        for b in 0..num_blocks {
            for row in 0..cs {
                let q_pos = b * cs + row;
                if q_pos >= seq_len {
                    continue;
                } // query past end → row fully masked
                for o in 0..ctx {
                    let kv_pos_raw = (b * cs + o) as isize - past as isize;
                    if kv_pos_raw < 0 || (kv_pos_raw as usize) >= seq_len {
                        continue;
                    }
                    // Python (sliding_window_mask_function, gemma4):
                    //   dist = q - kv
                    //   left  := dist >= 0 && dist < past
                    //   right := dist <  0 && -dist < future
                    //   valid = left || right
                    // Note future=0 kills right_mask entirely (dist=0 stays valid via left).
                    let delta = q_pos as isize - kv_pos_raw;
                    let left = delta >= 0 && delta < past as isize;
                    let right = delta < 0 && -delta < future as isize;
                    if !(left || right) {
                        continue;
                    }
                    data[(b * cs + row) * ctx + o] = 0.0;
                }
            }
        }
        let _ = padded;

        Tensor::<B, 1>::from_floats(&data[..], device).reshape([1, 1, num_blocks, cs, ctx])
    }

    /// Load the full audio tower from HuggingFace safetensors shards.
    ///
    /// `shards` must contain all weights prefixed by `model.audio_tower.*`
    /// and `model.embed_audio.*` (caller typically just passes all shards
    /// for the checkpoint).
    #[cfg(feature = "import")]
    pub fn load_from_shards(
        config: Gemma4AudioConfig,
        shards: &[safetensors::SafeTensors<'_>],
        device: &B::Device,
    ) -> Self {
        Self::load_with(config, &shard_fetch(shards), device)
    }

    /// Build the full audio tower from any named-tensor fetch (EXACT HF names,
    /// f32 data + shape). Shared by the safetensors loader and the pile path
    /// (`persist::load_gemma4_hearing_from_pile` wraps a handle-index +
    /// `ingest::read_leaf` into a fetch).
    pub fn load_with(
        config: Gemma4AudioConfig,
        fetch: &dyn Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
        device: &B::Device,
    ) -> Self {
        use burn::module::Param;

        let head_dim = config.hidden_size / config.num_attention_heads;

        type Fetch<'f> = dyn Fn(&str) -> Option<(Vec<f32>, Vec<usize>)> + 'f;
        // Generic tensor loader parameterized over shape dimensions.
        fn t<const D: usize, B: Backend>(
            fetch: &Fetch<'_>,
            name: &str,
            device: &B::Device,
        ) -> Tensor<B, D> {
            let (data, shape) = fetch(name).unwrap_or_else(|| panic!("tensor not found: {name}"));
            let dims: [usize; D] = std::array::from_fn(|i| shape[i]);
            Tensor::<B, 1>::from_floats(&data[..], device).reshape(dims)
        }
        fn scalar(fetch: &Fetch<'_>, name: &str) -> f32 {
            let (data, _) = fetch(name).unwrap_or_else(|| panic!("scalar not found: {name}"));
            data[0]
        }
        fn clip(fetch: &Fetch<'_>, prefix: &str) -> Option<ClipBounds> {
            Some(ClipBounds {
                input_min: scalar(fetch, &format!("{prefix}.input_min")),
                input_max: scalar(fetch, &format!("{prefix}.input_max")),
                output_min: scalar(fetch, &format!("{prefix}.output_min")),
                output_max: scalar(fetch, &format!("{prefix}.output_max")),
            })
        }

        // --- SubsampleConvProjection ---
        let ss_p = "model.audio_tower.subsample_conv_projection";
        let subsample = build_subsample::<B>(
            &config,
            t::<4, B>(fetch, &format!("{ss_p}.layer0.conv.weight"), device),
            t::<1, B>(fetch, &format!("{ss_p}.layer0.norm.weight"), device),
            t::<4, B>(fetch, &format!("{ss_p}.layer1.conv.weight"), device),
            t::<1, B>(fetch, &format!("{ss_p}.layer1.norm.weight"), device),
            t::<2, B>(fetch, &format!("{ss_p}.input_proj_linear.weight"), device),
            128,
            device,
        );

        // --- Layers ---
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let lp = format!("model.audio_tower.layers.{i}");
            let eps = config.rms_norm_eps;

            // Feed-forward blocks (FF1 and FF2)
            let make_ff = |sub: &str| -> AudioFeedForward<B> {
                let p = format!("{lp}.{sub}");
                AudioFeedForward {
                    pre_layer_norm: build_rms_norm(
                        t::<1, B>(fetch, &format!("{p}.pre_layer_norm.weight"), device),
                        eps,
                        device,
                    ),
                    post_layer_norm: build_rms_norm(
                        t::<1, B>(fetch, &format!("{p}.post_layer_norm.weight"), device),
                        eps,
                        device,
                    ),
                    ffw_layer_1: build_clippable_linear(
                        t::<2, B>(fetch, &format!("{p}.ffw_layer_1.linear.weight"), device),
                        clip(fetch, &format!("{p}.ffw_layer_1")),
                        device,
                    ),
                    ffw_layer_2: build_clippable_linear(
                        t::<2, B>(fetch, &format!("{p}.ffw_layer_2.linear.weight"), device),
                        clip(fetch, &format!("{p}.ffw_layer_2")),
                        device,
                    ),
                    post_layer_scale: config.residual_weight,
                    gradient_clipping: config.gradient_clipping,
                }
            };

            // LightConv1d
            let lconv_p = format!("{lp}.lconv1d");
            let lconv = AudioLightConv1d {
                pre_layer_norm: build_rms_norm(
                    t::<1, B>(fetch, &format!("{lconv_p}.pre_layer_norm.weight"), device),
                    eps,
                    device,
                ),
                conv_norm: build_rms_norm(
                    t::<1, B>(fetch, &format!("{lconv_p}.conv_norm.weight"), device),
                    eps,
                    device,
                ),
                linear_start: build_clippable_linear(
                    t::<2, B>(
                        fetch,
                        &format!("{lconv_p}.linear_start.linear.weight"),
                        device,
                    ),
                    clip(fetch, &format!("{lconv_p}.linear_start")),
                    device,
                ),
                linear_end: build_clippable_linear(
                    t::<2, B>(
                        fetch,
                        &format!("{lconv_p}.linear_end.linear.weight"),
                        device,
                    ),
                    clip(fetch, &format!("{lconv_p}.linear_end")),
                    device,
                ),
                depthwise_conv1d: build_depthwise_conv1d(
                    t::<3, B>(fetch, &format!("{lconv_p}.depthwise_conv1d.weight"), device),
                    config.hidden_size,
                    config.conv_kernel_size,
                    device,
                ),
                gradient_clipping: config.gradient_clipping,
                kernel_size: config.conv_kernel_size,
            };

            // Attention
            let sa_p = format!("{lp}.self_attn");
            let rel_k_w = t::<2, B>(fetch, &format!("{sa_p}.relative_k_proj.weight"), device);
            let [out_dim, in_dim] = rel_k_w.dims();
            let relative_k_proj = {
                let mut m = LinearConfig::new(in_dim, out_dim)
                    .with_bias(false)
                    .init::<B>(device);
                m.weight = Param::from_tensor(rel_k_w.swap_dims(0, 1));
                m
            };
            let q_scale = (head_dim as f32).powf(-0.5) / 2.0f32.ln();
            let k_scale = (1.0 + std::f32::consts::E).ln() / 2.0f32.ln();
            let self_attn = AudioAttention {
                q_proj: build_clippable_linear(
                    t::<2, B>(fetch, &format!("{sa_p}.q_proj.linear.weight"), device),
                    clip(fetch, &format!("{sa_p}.q_proj")),
                    device,
                ),
                k_proj: build_clippable_linear(
                    t::<2, B>(fetch, &format!("{sa_p}.k_proj.linear.weight"), device),
                    clip(fetch, &format!("{sa_p}.k_proj")),
                    device,
                ),
                v_proj: build_clippable_linear(
                    t::<2, B>(fetch, &format!("{sa_p}.v_proj.linear.weight"), device),
                    clip(fetch, &format!("{sa_p}.v_proj")),
                    device,
                ),
                post: build_clippable_linear(
                    t::<2, B>(fetch, &format!("{sa_p}.post.linear.weight"), device),
                    clip(fetch, &format!("{sa_p}.post")),
                    device,
                ),
                relative_k_proj,
                per_dim_scale: t::<1, B>(fetch, &format!("{sa_p}.per_dim_scale"), device),
                num_heads: config.num_attention_heads,
                head_dim,
                chunk_size: config.attention_chunk_size,
                max_past_horizon: config.attention_context_left - 1,
                max_future_horizon: config.attention_context_right,
                context_size: config.attention_chunk_size
                    + (config.attention_context_left - 1)
                    + config.attention_context_right,
                softcap: config.attention_logit_cap,
                q_scale,
                k_scale,
                invalid_logits_value: config.attention_invalid_logits_value,
            };

            layers.push(AudioLayer {
                feed_forward1: make_ff("feed_forward1"),
                feed_forward2: make_ff("feed_forward2"),
                self_attn,
                lconv1d: lconv,
                norm_pre_attn: build_rms_norm(
                    t::<1, B>(fetch, &format!("{lp}.norm_pre_attn.weight"), device),
                    eps,
                    device,
                ),
                norm_post_attn: build_rms_norm(
                    t::<1, B>(fetch, &format!("{lp}.norm_post_attn.weight"), device),
                    eps,
                    device,
                ),
                norm_out: build_rms_norm(
                    t::<1, B>(fetch, &format!("{lp}.norm_out.weight"), device),
                    eps,
                    device,
                ),
                gradient_clipping: config.gradient_clipping,
            });
        }

        // --- Output projection (has bias in this checkpoint) ---
        let out_w = t::<2, B>(fetch, "model.audio_tower.output_proj.weight", device);
        let out_b = t::<1, B>(fetch, "model.audio_tower.output_proj.bias", device);
        let [out_dim, in_dim] = out_w.dims();
        let mut output_proj = LinearConfig::new(in_dim, out_dim)
            .with_bias(true)
            .init::<B>(device);
        output_proj.weight = Param::from_tensor(out_w.swap_dims(0, 1));
        output_proj.bias = Some(Param::from_tensor(out_b));

        let rel_pos = rel_positional_encoding::<B>(&config, device);

        AudioModel {
            subsample,
            rel_pos,
            layers,
            output_proj,
            config,
        }
    }

    /// Forward: [B, T, feat_dim] log-mel → [B, T/4, output_proj_dims].
    pub fn forward(&self, input_features: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.subsample.forward(input_features);
        let [_, t, _] = h.dims();
        let device = h.device();
        let mask = self.build_mask_5d(t, &device);
        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(h, &self.rel_pos, Some(mask.clone()));
        }
        self.output_proj.forward(h)
    }
}
