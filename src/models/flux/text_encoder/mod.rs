pub mod config;
pub mod layers;
pub mod rope;

use burn::prelude::*;
use config::Qwen3Config;
use layers::Qwen3DecoderLayer;
use rope::RotaryEmbedding;

use crate::nn::weight_loader::WeightLoader;

/// Qwen-3 causal language model (text encoder for FLUX.2-klein).
/// Only runs through enough layers to extract hidden states at the specified layer indices.
pub struct Qwen3Model<B: Backend> {
    pub embed_tokens: Tensor<B, 2>, // [vocab_size, hidden_size]
    pub layers: Vec<Qwen3DecoderLayer<B>>,
    pub norm: Tensor<B, 1>, // final RMSNorm weight [hidden_size]
    pub config: Qwen3Config,
}

impl<B: Backend> Qwen3Model<B> {
    /// Load from multi-shard safetensors.
    /// Only loads layers 0..max_layer where max_layer = max(extract_layers).
    pub fn load(loader: &WeightLoader, config: Qwen3Config, device: &B::Device) -> Self {
        let max_layer = 27; // We need hidden_states[27] = output after layer 26

        let embed_tokens: Tensor<B, 2> = loader.load_tensor("model.embed_tokens.weight", device);

        let mut layers = Vec::with_capacity(max_layer);
        for i in 0..max_layer {
            layers.push(Qwen3DecoderLayer::load(loader, &config, i, device));
        }

        let norm: Tensor<B, 1> = loader.load_tensor("model.norm.weight", device);

        Self {
            embed_tokens,
            layers,
            norm,
            config,
        }
    }

    /// Run forward pass, extracting hidden states at specified layer indices.
    /// `extract_layers` should be [9, 18, 27] for FLUX.2-klein.
    /// `attention_mask` is optional: &[u32] of length L with 1=real token, 0=padding.
    /// When provided, creates a combined causal + padding mask so padding tokens
    /// produce correct hidden states matching Python's Qwen3ForCausalLM.
    /// Returns concatenated hidden states: [B, L, 3 * hidden_size].
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
        // input_ids: [B, L] -> gather from embed_tokens: [vocab, hidden]
        let input_ids_flat = input_ids.clone().reshape([batch * seq_len]);
        let hidden_states = self
            .embed_tokens
            .clone()
            .select(0, input_ids_flat)
            .reshape([batch, seq_len, hidden_size]);

        // Precompute RoPE
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
                    // Mask if: future token (causal) OR padding token
                    if j > i || am[j] == 0 {
                        mask_data[i * seq_len + j] = f32::MIN; // Match HuggingFace's min_dtype
                    }
                }
            }
            Tensor::<B, 4>::from_data(TensorData::new(mask_data, [1, 1, seq_len, seq_len]), device)
        });

        // Run through layers, collecting hidden states
        let mut h = hidden_states;
        let mut collected: Vec<Tensor<B, 3>> = Vec::new();

        // hidden_states indexing: index 0 = embedding output, index i = output after layer i-1
        // So hidden_states[9] = output after layer 8 (after processing 9 layers: 0..=8)
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h, &rope, combined_mask.as_ref());

            // Check if we need to collect (layer i produces hidden_states[i+1])
            let hs_index = i + 1;
            if extract_layers.contains(&hs_index) {
                collected.push(h.clone());
            }
        }

        // Stack collected hidden states along a new dimension then reshape
        // Each is [B, L, hidden_size], we want [B, L, 3*hidden_size]
        assert_eq!(
            collected.len(),
            extract_layers.len(),
            "Did not collect all requested layers"
        );

        // Concatenate along the last dimension
        Tensor::cat(collected, 2)
    }
}
