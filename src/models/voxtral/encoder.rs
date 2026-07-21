//! Voxtral causal audio encoder (~970M): causal conv stem (÷2) → 32 pre-norm
//! transformer layers (MHA 32×64 with q/v/o biases, k un-biased, RoPE θ=1e6,
//! sliding window 750, SwiGLU 5120 with biased down_proj) → final RMSNorm →
//! 4-frame stack → projector (5120→3072 GELU 3072→3072) to decoder space.
//!
//! Tensor names: `audio_tower.*`, `multi_modal_projector.*` (HF layout).

use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

use super::config::*;
use super::layers::{Attention, AttnConfig, KvCache, Linear, Mlp, RmsNorm, RopeTable};
use crate::nn::weight_loader::WeightLoader;

/// Causal Conv1d for the stem: left-pad `(k−1) − (stride−1)` zeros, no right pad.
/// (`pub(crate)` so the folded fast lane reuses the stem op-for-op.)
pub(crate) struct CausalConv<B: Backend> {
    weight: Tensor<B, 3>, // [out, in, k]
    bias: Tensor<B, 1>,
    stride: usize,
    left_pad: usize,
}

impl<B: Backend> CausalConv<B> {
    pub(crate) fn load(loader: &WeightLoader, prefix: &str, stride: usize, device: &B::Device) -> Self {
        let weight: Tensor<B, 3> = loader.load_tensor(&format!("{prefix}.weight"), device);
        let k = weight.dims()[2];
        Self {
            weight,
            bias: loader.load_tensor(&format!("{prefix}.bias"), device),
            stride,
            left_pad: k - stride, // (k−1)·d + 1 − stride, dilation 1
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, c, _] = x.dims();
        let x = if self.left_pad > 0 {
            let zeros = Tensor::zeros([b, c, self.left_pad], &x.device());
            Tensor::cat(vec![zeros, x], 2)
        } else {
            x
        };
        conv1d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([self.stride], [0], [1], 1),
        )
    }
}

struct EncoderLayer<B: Backend> {
    attn_norm: RmsNorm<B>,
    attn: Attention<B>,
    mlp_norm: RmsNorm<B>,
    mlp: Mlp<B>,
}

pub struct AudioEncoder<B: Backend> {
    conv1: CausalConv<B>,
    conv2: CausalConv<B>,
    layers: Vec<EncoderLayer<B>>,
    norm: RmsNorm<B>,
    rope: RopeTable<B>,
    proj1: Linear<B>,
    proj2: Linear<B>,
}

/// Per-layer KV caches for incremental encoding.
pub struct EncoderCaches<B: Backend>(pub Vec<KvCache<B>>);

impl<B: Backend> AudioEncoder<B> {
    pub fn load(loader: &WeightLoader, max_positions: usize, device: &B::Device) -> Self {
        let cfg = AttnConfig {
            heads: ENC_HEADS,
            kv_heads: ENC_HEADS,
            head_dim: ENC_HEAD_DIM,
            qvo_bias: true,
            window: ENC_WINDOW,
        };
        let layers = (0..ENC_LAYERS)
            .map(|i| {
                let p = format!("audio_tower.layers.{i}");
                EncoderLayer {
                    attn_norm: RmsNorm::load(loader, &format!("{p}.self_attn_layer_norm.weight"), EPS, device),
                    attn: Attention::load(loader, &format!("{p}.self_attn"), cfg, device),
                    mlp_norm: RmsNorm::load(loader, &format!("{p}.final_layer_norm.weight"), EPS, device),
                    mlp: Mlp::load(loader, &format!("{p}.mlp"), true, device),
                }
            })
            .collect();
        Self {
            conv1: CausalConv::load(loader, "audio_tower.embedder.conv1", 1, device),
            conv2: CausalConv::load(loader, "audio_tower.embedder.conv2", 2, device),
            layers,
            norm: RmsNorm::load(loader, "audio_tower.norm.weight", EPS, device),
            rope: RopeTable::new(ROPE_THETA, ENC_HEAD_DIM, max_positions, device),
            proj1: Linear::load(loader, "multi_modal_projector.linear_1", false, device),
            proj2: Linear::load(loader, "multi_modal_projector.linear_2", false, device),
        }
    }

    pub fn new_caches(&self) -> EncoderCaches<B> {
        EncoderCaches((0..ENC_LAYERS).map(|_| KvCache::empty()).collect())
    }

    /// mel `[1, 128, T_mel]` → conv-stem embeds `[1, T_mel/2, 1280]`.
    /// (Streaming later chunks re-enter here with the convs' tail state —
    /// batch mode runs the whole mel at once.)
    pub fn stem(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = gelu(self.conv1.forward(mel));
        let x = gelu(self.conv2.forward(x));
        x.swap_dims(1, 2)
    }

    /// Encoder transformer over the next `l` stem positions (append-only KV;
    /// pass the full stem with fresh caches for batch mode). Returns the
    /// final-normed hidden states `[1, l, 1280]`.
    pub fn forward(&self, embeds: Tensor<B, 3>, caches: &mut EncoderCaches<B>) -> Tensor<B, 3> {
        let l = embeds.dims()[1];
        let offset = caches.0[0].seq_len();
        let (cos, sin) = self.rope.slices(offset, l);
        let device = embeds.device();
        let mut x = embeds;
        for (layer, cache) in self.layers.iter().zip(caches.0.iter_mut()) {
            let att = layer
                .attn
                .forward(layer.attn_norm.forward(x.clone()), &cos, &sin, cache, &device);
            let x1 = x + att;
            let mlp = layer.mlp.forward(layer.mlp_norm.forward(x1.clone()));
            x = x1 + mlp;
        }
        self.norm.forward(x)
    }

    /// Encoder hidden `[1, l, 1280]` (l a multiple of 4) → audio embeds in
    /// decoder space `[1, l/4, 3072]`.
    pub fn project(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, l, _] = hidden.dims();
        assert!(l % DOWNSAMPLE == 0, "project needs a multiple of {DOWNSAMPLE} positions");
        let stacked = hidden.reshape([b, l / DOWNSAMPLE, ENC_HIDDEN * DOWNSAMPLE]);
        self.proj2.forward(gelu(self.proj1.forward(stacked)))
    }
}
