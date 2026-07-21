//! The VLM perceptual tower's text side: token embedding + the 16-layer
//! SmolLM2 decoder. The decoder self-attends the prefix (image ⊕ language ⊕
//! state) and *produces* the per-layer KV cache the action expert later
//! attends. (The SigLIP vision encoder that yields the image embeddings lives
//! in `vision.rs`; here the image embeddings are an input to `embed_prefix`.)
//!
//! The decoder layer is architecturally identical to the action expert's
//! self-attention layer (SmolLM2 block, GQA 15:5) — just full width (960) and
//! cache-producing — so it reuses [`ExpertLayer`] via `forward_vlm`.

use burn::prelude::*;

use super::config::SmolVlaConfig;
use super::layers::ExpertLayer;
use crate::nn::weight_loader::WeightLoader;

pub struct VlmTower<B: Backend> {
    embed_tokens: Tensor<B, 2>, // [vocab, 960]
    layers: Vec<ExpertLayer<B>>,
}

impl<B: Backend> VlmTower<B> {
    pub fn load(loader: &WeightLoader, cfg: &SmolVlaConfig, device: &B::Device) -> Self {
        let p = "model.vlm_with_expert.vlm.model.text_model";
        let embed_tokens = loader.load_tensor(&format!("{p}.embed_tokens.weight"), device);
        let layers = (0..cfg.vlm.n_layers)
            .map(|i| ExpertLayer::load(loader, &format!("{p}.layers.{i}"), cfg.vlm, device))
            .collect();
        Self { embed_tokens, layers }
    }

    /// Embed token ids `[B,L]` → `[B,L,960]` (row lookup into embed_tokens).
    pub fn embed_language_tokens(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, l] = ids.dims();
        let dim = self.embed_tokens.dims()[1];
        let flat = ids.reshape([b * l]);
        self.embed_tokens.clone().select(0, flat).reshape([b, l, dim])
    }

    /// Run the prefix through the decoder, returning the stacked per-layer KV
    /// cache `(k, v)`, each `[n_layers, B, L, Hkv, Dh]`. `mask: [B,L,L]`.
    pub fn forward_decoder(
        &self,
        prefix: Tensor<B, 3>,
        positions: Tensor<B, 2>,
        mask: Tensor<B, 3, Bool>,
        device: &B::Device,
    ) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let mut x = prefix;
        let mut ks = Vec::with_capacity(self.layers.len());
        let mut vs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (out, k, v) = layer.forward_vlm(x, positions.clone(), mask.clone(), device);
            x = out;
            ks.push(k);
            vs.push(v);
        }
        (Tensor::stack(ks, 0), Tensor::stack(vs, 0))
    }
}
