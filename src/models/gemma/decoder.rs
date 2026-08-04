//! Full Mistral decoder model: embedding + N transformer layers + output head.

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::models::gemma::config::MistralConfig;
use crate::models::gemma::layers::{
    LayerCaches, QuantConfig, QuantizedLayerCaches, RmsNorm, TransformerLayer,
    TurboQuantLayerCaches,
};
use crate::models::gemma::rope::RopeTable;
use crate::models::gemma::turbo_quant::TurboQuantConfig;

/// The complete Mistral decoder.
#[derive(Module, Debug)]
pub struct MistralDecoder<B: Backend> {
    pub embed: Embedding<B>,
    pub layers: Vec<TransformerLayer<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
}

/// Decoder plus its config (config is not a Module, kept separately).
pub struct MistralModel<B: Backend> {
    pub decoder: MistralDecoder<B>,
    pub config: MistralConfig,
}

impl<B: Backend> MistralModel<B> {
    /// Initialize with random weights (for testing architecture).
    pub fn new(config: MistralConfig, device: &B::Device) -> Self {
        let embed = EmbeddingConfig::new(config.vocab_size, config.hidden_dim).init(device);

        let layers = (0..config.n_layers)
            .map(|_| TransformerLayer::new(&config, device))
            .collect();

        let norm = crate::models::gemma::layers::RmsNormConfig::new(config.hidden_dim)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        let lm_head = LinearConfig::new(config.hidden_dim, config.vocab_size)
            .with_bias(false)
            .init(device);

        let decoder = MistralDecoder {
            embed,
            layers,
            norm,
            lm_head,
        };
        Self { decoder, config }
    }

    /// Forward pass without KV cache: token IDs → logits.
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        rope: &RopeTable<B>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let mut h = self.decoder.embed.forward(tokens);

        for layer in &self.decoder.layers {
            h = layer.forward(h, rope, offset);
        }

        h = self.decoder.norm.forward(h);
        self.decoder.lm_head.forward(h)
    }

    /// Forward pass with KV cache for incremental decoding.
    /// tokens: [batch, new_len] — new token IDs only
    /// caches: mutable layer caches (position offset is derived from cache state)
    /// Returns: [batch, new_len, vocab_size] logits
    pub fn forward_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        rope: &RopeTable<B>,
        caches: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        let mut h = self.decoder.embed.forward(tokens);

        for (layer, cache) in self.decoder.layers.iter().zip(caches.caches.iter_mut()) {
            h = layer.forward_cached(h, rope, cache);
        }

        h = self.decoder.norm.forward(h);
        self.decoder.lm_head.forward(h)
    }

    /// Forward pass with quantized KV cache for incremental decoding.
    /// Same as `forward_cached` but stores K/V in compressed format.
    pub fn forward_quantized_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        rope: &RopeTable<B>,
        caches: &mut QuantizedLayerCaches<B>,
    ) -> Tensor<B, 3> {
        let mut h = self.decoder.embed.forward(tokens);

        for (layer, cache) in self.decoder.layers.iter().zip(caches.caches.iter_mut()) {
            h = layer.forward_quantized_cached(h, rope, cache);
        }

        h = self.decoder.norm.forward(h);
        self.decoder.lm_head.forward(h)
    }

    /// Create empty KV caches for all layers.
    pub fn new_caches(&self) -> LayerCaches<B> {
        LayerCaches::new(self.config.n_layers)
    }

    /// Create empty quantized KV caches for all layers.
    pub fn new_quantized_caches(&self, config: QuantConfig) -> QuantizedLayerCaches<B> {
        QuantizedLayerCaches::new(self.config.n_layers, config)
    }

    /// Forward pass with TurboQuant KV cache for incremental decoding.
    pub fn forward_turbo_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        rope: &RopeTable<B>,
        caches: &mut TurboQuantLayerCaches<B>,
    ) -> Tensor<B, 3> {
        let mut h = self.decoder.embed.forward(tokens);

        for (layer, cache) in self.decoder.layers.iter().zip(caches.caches.iter_mut()) {
            h = layer.forward_turbo_cached(h, rope, cache);
        }

        h = self.decoder.norm.forward(h);
        self.decoder.lm_head.forward(h)
    }

    /// Create empty TurboQuant KV caches for all layers.
    pub fn new_turbo_caches(&self, config: TurboQuantConfig) -> TurboQuantLayerCaches<B> {
        TurboQuantLayerCaches::new(self.config.n_layers, config)
    }

    /// Forward pass with GPU-native quantized KV cache for incremental decoding.
    /// Stores K/V as packed int8 on GPU — no CPU involvement.
    pub fn forward_gpu_quant_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        rope: &RopeTable<B>,
        caches: &mut crate::models::gemma::gpu_quant::GpuQuantLayerCaches<B>,
    ) -> Tensor<B, 3>
    where
        B: Backend<IntElem = i32>,
    {
        let mut h = self.decoder.embed.forward(tokens);

        for (layer, cache) in self.decoder.layers.iter().zip(caches.caches.iter_mut()) {
            h = layer.forward_gpu_quant_cached(h, rope, cache);
        }

        h = self.decoder.norm.forward(h);
        self.decoder.lm_head.forward(h)
    }

    /// Create empty GPU-native quantized KV caches for all layers.
    pub fn new_gpu_quant_caches(&self) -> crate::models::gemma::gpu_quant::GpuQuantLayerCaches<B> {
        crate::models::gemma::gpu_quant::GpuQuantLayerCaches::new(self.config.n_layers)
    }

    /// Forward pass with GPU-native TurboQuant KV cache for incremental decoding.
    /// Stores K/V using TurboQuant (rotation + Lloyd-Max quantization) entirely on GPU.
    pub fn forward_gpu_turbo_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        rope: &RopeTable<B>,
        caches: &mut crate::models::gemma::gpu_quant::GpuTurboQuantLayerCaches<B>,
    ) -> Tensor<B, 3>
    where
        B: Backend<IntElem = i32>,
    {
        let mut h = self.decoder.embed.forward(tokens);

        for (layer, cache) in self.decoder.layers.iter().zip(caches.caches.iter_mut()) {
            h = layer.forward_gpu_turbo_cached(h, rope, cache);
        }

        h = self.decoder.norm.forward(h);
        self.decoder.lm_head.forward(h)
    }

    /// Create empty GPU-native TurboQuant KV caches for all layers.
    pub fn new_gpu_turbo_caches(
        &self,
        config: &TurboQuantConfig,
        device: &B::Device,
    ) -> crate::models::gemma::gpu_quant::GpuTurboQuantLayerCaches<B>
    where
        B: Backend<IntElem = i32>,
    {
        crate::models::gemma::gpu_quant::GpuTurboQuantLayerCaches::new(
            self.config.n_layers,
            config,
            device,
        )
    }

    /// Build the RoPE table for this model's configuration.
    /// Uses YaRN scaling if configured, for extended context.
    pub fn rope_table(&self, device: &B::Device) -> RopeTable<B> {
        let max_len = self.config.effective_max_seq_len();
        if self.config.yarn_max_seq_len.is_some() {
            let yarn = if self.config.qk_norm {
                crate::models::gemma::rope::YarnConfig::qwen3_8b()
            } else {
                crate::models::gemma::rope::YarnConfig::ministral()
            };
            RopeTable::with_yarn(
                self.config.head_dim,
                max_len,
                self.config.rope_theta,
                &yarn,
                device,
            )
        } else {
            RopeTable::new(
                self.config.head_dim,
                max_len,
                self.config.rope_theta,
                device,
            )
        }
    }
}
