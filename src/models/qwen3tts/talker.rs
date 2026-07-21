//! The talker — Qwen3-TTS's 28-layer codec-frame LM (hidden 2048, GQA 16:8,
//! q/k-norm, rope θ=1e6). Operates purely on `inputs_embeds`: the pipeline
//! assembles text-side and codec-side embeddings and sums them per position
//! (see `pipeline.rs`); the talker turns the running sequence into the next
//! frame's hidden state and codebook-0 logits.

use burn::prelude::*;

use super::config::*;
use super::layers::{AttnConfig, DecoderLayer, Embedding, KvCache, Linear, RmsNorm, RopeTable};
use crate::nn::weight_loader::WeightLoader;

pub fn talker_attn_config() -> AttnConfig {
    AttnConfig {
        hidden: TALKER_HIDDEN,
        heads: TALKER_HEADS,
        kv_heads: TALKER_KV_HEADS,
        head_dim: TALKER_HEAD_DIM,
        rope_theta: TALKER_ROPE_THETA,
        eps: TALKER_EPS,
        window: None,
        qk_norm: true,
        layer_scale: false,
    }
}

pub struct Talker<B: Backend> {
    /// Talker hidden width, from the checkpoint (1.7B: 2048, 0.6B: 1024).
    pub hidden: usize,
    /// Codec-frame embedding [3072, hidden] — codebook-0 + control tokens.
    pub codec_embedding: Embedding<B>,
    /// CPU copy of the codec embedding — the decode loop assembles the next
    /// frame's input embedding host-side (one upload/frame instead of a
    /// select+add chain).
    pub codec_embedding_cpu: Vec<f32>,
    /// Text embedding [151936, 2048] (raw text space, pre text_projection).
    pub text_embedding: Embedding<B>,
    /// ResizeMLP text-space → talker-space: fc1 (silu) → fc2, both biased.
    pub text_fc1: Linear<B>,
    pub text_fc2: Linear<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    /// Codebook-0 head [3072 × 2048] — CPU: it runs once per frame on the
    /// read-back hidden state, right where the sampler needs the logits.
    pub codec_head: Vec<f32>,
    pub rope: RopeTable<B>,
}

impl<B: Backend> Talker<B> {
    pub fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let cfg = talker_attn_config();
        let codec_embedding: Embedding<B> =
            Embedding::load(loader, "talker.model.codec_embedding.weight", device);
        Self {
            hidden: codec_embedding.weight.dims()[1],
            codec_embedding,
            codec_embedding_cpu: loader.load_f32("talker.model.codec_embedding.weight").0,
            text_embedding: Embedding::load(loader, "talker.model.text_embedding.weight", device),
            text_fc1: Linear::load(loader, "talker.text_projection.linear_fc1", true, device),
            text_fc2: Linear::load(loader, "talker.text_projection.linear_fc2", true, device),
            layers: (0..TALKER_LAYERS)
                .map(|i| DecoderLayer::load(loader, &format!("talker.model.layers.{i}"), cfg, device))
                .collect(),
            norm: RmsNorm::load(loader, "talker.model.norm.weight", TALKER_EPS, device),
            codec_head: loader.load_f32("talker.codec_head.weight").0,
            rope: RopeTable::new(TALKER_ROPE_THETA, TALKER_HEAD_DIM, 8192, device),
        }
    }

    /// Talker-width CPU row of the codec embedding for one id.
    pub fn codec_row(&self, id: u32) -> &[f32] {
        &self.codec_embedding_cpu[id as usize * self.hidden..][..self.hidden]
    }

    /// Text ids → talker-space embeddings (text_embedding + ResizeMLP).
    pub fn embed_text(&self, ids: &[u32], device: &B::Device) -> Tensor<B, 3> {
        let e = self.text_embedding.forward(ids, device);
        self.text_fc2.forward(burn::tensor::activation::silu(self.text_fc1.forward(e)))
    }

    /// Codec ids → talker-space embeddings.
    pub fn embed_codec(&self, ids: &[u32], device: &B::Device) -> Tensor<B, 3> {
        self.codec_embedding.forward(ids, device)
    }

    /// Run the stack over new embeddings (prefill or one step), returning the
    /// **normed** hidden states `[1, L, 2048]`.
    pub fn forward(
        &self,
        embeds: Tensor<B, 3>,
        caches: &mut Vec<KvCache<B>>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let l = embeds.dims()[1];
        let (cos, sin) = self.rope.slices(caches[0].seq_len(), l);
        let mut h = embeds;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(h, &cos, &sin, cache, device);
        }
        self.norm.forward(h)
    }

    pub fn new_caches(&self) -> Vec<KvCache<B>> {
        (0..TALKER_LAYERS).map(|_| KvCache::empty()).collect()
    }

    /// Read back the last position's hidden state `[2048]` (one sync).
    pub fn last_hidden(&self, hidden: Tensor<B, 3>) -> Vec<f32> {
        let [_, l, _] = hidden.dims();
        hidden
            .narrow(1, l - 1, 1)
            .reshape([self.hidden])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap()
    }

    /// Codebook-0 logits from a read-back hidden state (CPU gemv,
    /// row-split across the pool — [3072×2048] is bit-exactness-verified).
    pub fn logits_from(&self, hidden: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; CODEC_VOCAB];
        super::cpu::sgemv_mt(&self.codec_head, CODEC_VOCAB, self.hidden, hidden, &mut y);
        y
    }

    /// Codebook-0 logits for the last position: `[1, L, 2048] → [3072]`.
    pub fn logits_last(&self, hidden: Tensor<B, 3>) -> Vec<f32> {
        self.logits_from(&self.last_hidden(hidden))
    }
}
