pub mod config;
pub mod layers;

use burn::prelude::*;
use config::Mistral3Config;
use layers::MistralDecoderLayer;

use crate::models::flux::text_encoder::rope::RotaryEmbedding;
use crate::nn::weight_loader::WeightLoader;

/// Mistral3 language model (text encoder for FLUX.2-dev).
/// Extracts hidden states at specified layer indices, same pattern as Qwen3 for Klein.
pub struct Mistral3Model<B: Backend> {
    pub embed_tokens: Tensor<B, 2>, // [vocab_size, hidden_size]
    pub layers: Vec<MistralDecoderLayer<B>>,
    pub norm: Tensor<B, 1>, // final RMSNorm weight [hidden_size]
    pub config: Mistral3Config,
}

impl<B: Backend> Mistral3Model<B> {
    /// Load from multi-shard safetensors.
    /// Only loads layers 0..max_layer where max_layer = max(extract_layers).
    pub fn load(
        loader: &WeightLoader,
        config: Mistral3Config,
        extract_layers: &[usize],
        device: &B::Device,
    ) -> Self {
        let max_layer = *extract_layers.iter().max().unwrap();

        let embed_tokens: Tensor<B, 2> =
            loader.load_tensor("language_model.model.embed_tokens.weight", device);

        let mut layers = Vec::with_capacity(max_layer);
        for i in 0..max_layer {
            eprintln!("    Loading Mistral3 layer {}/{}", i + 1, max_layer);
            layers.push(MistralDecoderLayer::load(loader, &config, i, device));
        }

        let norm: Tensor<B, 1> = loader.load_tensor("language_model.model.norm.weight", device);

        Self {
            embed_tokens,
            layers,
            norm,
            config,
        }
    }

    /// Run forward pass, extracting hidden states at specified layer indices.
    /// `extract_layers` should be [10, 20, 30] for FLUX.2-dev.
    /// `attention_mask` is optional: &[u32] of length L with 1=real token, 0=padding.
    /// Returns concatenated hidden states: [B, L, num_layers * hidden_size].
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>, // [B, L]
        extract_layers: &[usize],
        attention_mask: Option<&[u32]>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [batch, seq_len] = input_ids.dims();
        let hidden_size = self.config.hidden_size;

        // Token embedding lookup
        let input_ids_flat = input_ids.clone().reshape([batch * seq_len]);
        let hidden_states = self
            .embed_tokens
            .clone()
            .select(0, input_ids_flat)
            .reshape([batch, seq_len, hidden_size]);

        // Precompute RoPE (using halved convention, same as Qwen3)
        let rope = RotaryEmbedding::new(
            self.config.head_dim,
            seq_len,
            self.config.rope_theta,
            device,
        );

        // Build combined causal + padding mask if attention_mask is provided
        let combined_mask = attention_mask.map(|am| {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i || am[j] == 0 {
                        mask_data[i * seq_len + j] = f32::MIN;
                    }
                }
            }
            Tensor::<B, 4>::from_data(
                burn::tensor::TensorData::new(mask_data, [1, 1, seq_len, seq_len]),
                device,
            )
        });

        // Run through layers, collecting hidden states
        let mut h = hidden_states;
        let mut collected: Vec<Tensor<B, 3>> = Vec::new();

        // hidden_states indexing: index 0 = embedding output, index i = output after layer i-1
        // So hidden_states[10] = output after layer 9 (after processing 10 layers: 0..=9)
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h, &rope, combined_mask.as_ref());

            let hs_index = i + 1;
            if extract_layers.contains(&hs_index) {
                collected.push(h.clone());
            }
        }

        assert_eq!(
            collected.len(),
            extract_layers.len(),
            "Did not collect all requested layers"
        );

        // Concatenate along the last dimension: [B, L, num_layers * hidden_size]
        Tensor::cat(collected, 2)
    }
}
