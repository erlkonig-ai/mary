//! Voxtral text decoder (~3.4B, Ministral-3-3B shape): 26 pre-norm GQA layers
//! (32 q / 8 kv heads × 128, RoPE θ=1e6, sliding window 8192, SwiGLU 9216, no
//! biases) with **ada-RMS-norm delay conditioning**: a per-channel scale
//! `1 + linear2(gelu(linear1(t_cond)))` applied between the post-attention
//! norm and the MLP in every layer. `t_cond` is the sinusoidal embedding of
//! the delay-token count — constant per session, so the 26 scale vectors are
//! precomputed once per delay setting.
//!
//! Tied lm_head: logits = h @ embed_tokensᵀ.
//! Tensor names: `language_model.model.*` (HF layout).

use burn::prelude::*;
use burn::tensor::activation::gelu;

use super::config::*;
use super::layers::{Attention, AttnConfig, Embedding, KvCache, Linear, Mlp, RmsNorm, RopeTable};
use crate::nn::weight_loader::WeightLoader;

/// Sinusoidal time embedding of the delay-token count, mirroring the
/// reference op-for-op in f32: `inv_freq = exp(−ln θ · i / (dim/2))`,
/// `t_cond = cat(cos(t·inv_freq), sin(t·inv_freq))` → `[3072]`.
pub fn time_embedding(num_delay_tokens: usize) -> Vec<f32> {
    let half = DEC_HIDDEN / 2;
    let log_theta = (TIME_THETA).ln() as f32; // f64 ln, f32 constant (torch order)
    let t = num_delay_tokens as f32;
    let mut out = vec![0f32; DEC_HIDDEN];
    for i in 0..half {
        let inv_freq = (-log_theta * i as f32 / half as f32).exp();
        let e = t * inv_freq;
        out[i] = e.cos();
        out[half + i] = e.sin();
    }
    out
}

struct AdaRmsNorm<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> AdaRmsNorm<B> {
    /// `1 + linear2(gelu(linear1(t_cond)))` → `[1, 1, 3072]`.
    fn scale(&self, t_cond: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear2
            .forward(gelu(self.linear1.forward(t_cond)))
            .add_scalar(1.0)
    }
}

struct DecoderLayer<B: Backend> {
    input_norm: RmsNorm<B>,
    attn: Attention<B>,
    post_norm: RmsNorm<B>,
    ada: AdaRmsNorm<B>,
    mlp: Mlp<B>,
}

pub struct Decoder<B: Backend> {
    pub embed: Embedding<B>,
    layers: Vec<DecoderLayer<B>>,
    norm: RmsNorm<B>,
    /// Tied lm_head: embed weight pre-transposed `[1, 3072, VOCAB]`.
    head_t: Tensor<B, 3>,
    rope: RopeTable<B>,
}

/// Per-layer KV caches.
pub struct DecoderCaches<B: Backend>(pub Vec<KvCache<B>>);

/// The 26 per-layer conditioning scales for one delay setting.
pub struct AdaScales<B: Backend>(pub Vec<Tensor<B, 3>>);

impl<B: Backend> Decoder<B> {
    pub fn load(loader: &WeightLoader, max_positions: usize, device: &B::Device) -> Self {
        let cfg = AttnConfig {
            heads: DEC_HEADS,
            kv_heads: DEC_KV_HEADS,
            head_dim: DEC_HEAD_DIM,
            qvo_bias: false,
            window: DEC_WINDOW,
        };
        let layers = (0..DEC_LAYERS)
            .map(|i| {
                let p = format!("language_model.model.layers.{i}");
                DecoderLayer {
                    input_norm: RmsNorm::load(
                        loader,
                        &format!("{p}.input_layernorm.weight"),
                        EPS,
                        device,
                    ),
                    attn: Attention::load(loader, &format!("{p}.self_attn"), cfg, device),
                    post_norm: RmsNorm::load(
                        loader,
                        &format!("{p}.post_attention_layernorm.weight"),
                        EPS,
                        device,
                    ),
                    ada: AdaRmsNorm {
                        linear1: Linear::load(
                            loader,
                            &format!("{p}.ada_rms_norm.linear1"),
                            false,
                            device,
                        ),
                        linear2: Linear::load(
                            loader,
                            &format!("{p}.ada_rms_norm.linear2"),
                            false,
                            device,
                        ),
                    },
                    mlp: Mlp::load(loader, &format!("{p}.mlp"), false, device),
                }
            })
            .collect();
        let embed = Embedding::load(loader, "language_model.model.embed_tokens.weight", device);
        let [v, d] = embed.weight.dims();
        let head_t = embed.weight.clone().transpose().reshape([1, d, v]);
        Self {
            embed,
            layers,
            norm: RmsNorm::load(loader, "language_model.model.norm.weight", EPS, device),
            head_t,
            rope: RopeTable::new(ROPE_THETA, DEC_HEAD_DIM, max_positions, device),
        }
    }

    pub fn new_caches(&self) -> DecoderCaches<B> {
        DecoderCaches((0..DEC_LAYERS).map(|_| KvCache::empty()).collect())
    }

    /// Precompute the 26 conditioning scales for `num_delay_tokens`.
    pub fn ada_scales(&self, num_delay_tokens: usize, device: &B::Device) -> AdaScales<B> {
        let t = time_embedding(num_delay_tokens);
        let t = Tensor::<B, 1>::from_floats(t.as_slice(), device).reshape([1, 1, DEC_HIDDEN]);
        AdaScales(self.layers.iter().map(|l| l.ada.scale(t.clone())).collect())
    }

    /// One decoder pass over `embeds [1, l, 3072]` (prompt-prefill or a single
    /// step), appending to the caches. Returns the FINAL-normed hidden states.
    pub fn forward(
        &self,
        embeds: Tensor<B, 3>,
        ada: &AdaScales<B>,
        caches: &mut DecoderCaches<B>,
    ) -> Tensor<B, 3> {
        let l = embeds.dims()[1];
        let offset = caches.0[0].seq_len();
        let (cos, sin) = self.rope.slices(offset, l);
        let device = embeds.device();
        let mut x = embeds;
        for (i, (layer, cache)) in self.layers.iter().zip(caches.0.iter_mut()).enumerate() {
            let att = layer.attn.forward(
                layer.input_norm.forward(x.clone()),
                &cos,
                &sin,
                cache,
                &device,
            );
            let x1 = x + att;
            let h = layer.post_norm.forward(x1.clone());
            let mlp = layer.mlp.forward(h.mul(ada.0[i].clone()));
            x = x1 + mlp;
        }
        self.norm.forward(x)
    }

    /// Logits for the LAST position of a hidden-state batch: `[VOCAB]`.
    pub fn logits_last(&self, hidden: Tensor<B, 3>) -> Tensor<B, 1> {
        let [_, l, _d] = hidden.dims();
        hidden
            .narrow(1, l - 1, 1)
            .matmul(self.head_t.clone())
            .reshape([VOCAB])
    }
}
