//! Gemma 4 text decoder: embedding + N transformer layers + output head.

use burn::nn::{Embedding, Linear, RmsNorm};
use burn::prelude::*;

use super::config::Gemma4TextConfig;
use super::layers::Gemma4DecoderLayer;
use crate::models::gemma::layers::LayerCaches;
use crate::models::gemma::lora::LoraWeights;
use crate::models::gemma::rope::RopeTable;

/// The complete Gemma 4 text decoder.
pub struct Gemma4Decoder<B: Backend> {
    pub embed: Embedding<B>,
    /// Shared PLE embedding table [vocab, ple_dim * n_layers] (E2B/E4B only).
    pub embed_per_layer: Option<Embedding<B>>,
    /// PLE model-level projection: projects main embeddings to PLE space [hidden → ple_dim * n_layers].
    pub per_layer_model_projection: Option<Linear<B>>,
    /// PLE projection norm.
    pub per_layer_projection_norm: Option<RmsNorm<B>>,
    /// PLE projection scale factor.
    pub per_layer_model_projection_scale: f32,
    /// PLE input combination scale (1/sqrt(2)).
    pub per_layer_input_scale: f32,
    pub layers: Vec<Gemma4DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    /// Separate lm_head for models that don't tie to the input embedding.
    /// When tied, this is None and we reuse `embed.weight` via a matmul
    /// with its transpose.
    pub lm_head: Option<Linear<B>>,
}

/// Decoder plus config.
pub struct Gemma4Model<B: Backend> {
    pub decoder: Gemma4Decoder<B>,
    pub config: Gemma4TextConfig,
}

impl<B: Backend> Gemma4Model<B> {
    /// Forward pass with pre-computed input embeddings (for multimodal).
    /// inputs_embeds already includes merged text+vision+audio tokens.
    /// `mm_ranges` lists half-open [start, end) spans of merged multimodal
    /// soft-token positions (image or audio). PLE lookup substitutes
    /// pad_token_id at those positions; larger Gemma 4 variants with
    /// use_bidirectional_attention="vision" would also build a sliding-layer
    /// mask unmasking within each span (not yet wired).
    /// `lora`: optional trainable LoRA adapters threaded to every attention/MLP
    /// projection; `None` is the plain inference path, bit-identical to before.
    pub fn forward_embeds(
        &self,
        inputs_embeds: Tensor<B, 3>,
        tokens: Tensor<B, 2, Int>,
        rope_sliding: &RopeTable<B>,
        rope_global: &RopeTable<B>,
        caches: &mut LayerCaches<B>,
        mm_ranges: &[(usize, usize)],
        lora: Option<&LoraWeights<B>>,
    ) -> Tensor<B, 3> {
        self.forward_inner(
            inputs_embeds,
            tokens,
            rope_sliding,
            rope_global,
            caches,
            mm_ranges,
            lora,
        )
    }

    /// Forward pass with KV cache for incremental decoding.
    /// token_ids: [batch, new_len] — for PLE lookup and embedding.
    pub fn forward_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        rope_sliding: &RopeTable<B>,
        rope_global: &RopeTable<B>,
        caches: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        let scale = (self.config.hidden_size as f64).sqrt() as f32;
        let inputs_embeds = self.decoder.embed.forward(tokens.clone()).mul_scalar(scale);
        self.forward_inner(
            inputs_embeds,
            tokens,
            rope_sliding,
            rope_global,
            caches,
            &[],
            None,
        )
    }

    /// Inner forward: shared between forward_cached and forward_embeds.
    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        inputs_embeds: Tensor<B, 3>,
        tokens: Tensor<B, 2, Int>,
        rope_sliding: &RopeTable<B>,
        rope_global: &RopeTable<B>,
        caches: &mut LayerCaches<B>,
        mm_ranges: &[(usize, usize)],
        lora: Option<&LoraWeights<B>>,
    ) -> Tensor<B, 3> {
        let device = inputs_embeds.device();
        // Compute PLE inputs.
        //
        // Python (Gemma4Model.forward, modeling_gemma4.py:2188-2195) splits:
        //   - PLE table lookup uses input_ids with pad_token_id at image
        //     positions (image ids are OOV for the PLE table, which has
        //     vocab_size_per_layer_input < text vocab_size).
        //   - per_layer_model_projection takes the FULL merged inputs_embeds
        //     (with vision features at image positions) — this carries
        //     vision information into PLE.
        let ple_inputs: Vec<Option<Tensor<B, 3>>> = if let (Some(proj), Some(proj_norm)) = (
            &self.decoder.per_layer_model_projection,
            &self.decoder.per_layer_projection_norm,
        ) {
            let [batch, seq] = tokens.dims();
            let n_layers = self.config.num_hidden_layers;
            let ple_dim = self.config.hidden_size_per_layer_input;
            let pad_id = 0i64; // Gemma 4 pad_token_id

            // PLE table lookup: pad-substituted tokens at every multimodal
            // span (image and/or audio).
            let ple_tokens = if !mm_ranges.is_empty() {
                let mut tok_data: Vec<i32> = tokens
                    .clone()
                    .reshape([batch * seq])
                    .to_data()
                    .to_vec()
                    .unwrap();
                for &(s, e) in mm_ranges {
                    let s = s.min(seq);
                    let e = e.min(seq);
                    for b in 0..batch {
                        for p in s..e {
                            tok_data[b * seq + p] = pad_id as i32;
                        }
                    }
                }
                Tensor::<B, 1, Int>::from_ints(&tok_data[..], &device).reshape([batch, seq])
            } else {
                tokens.clone()
            };

            // Project the MERGED inputs_embeds (vision features included) so
            // that PLE for image positions is informed by vision.
            let projected =
                proj.forward(inputs_embeds.clone()) * self.decoder.per_layer_model_projection_scale;
            let projected = projected.reshape([batch, seq, n_layers, ple_dim]);
            let projected = proj_norm.forward(projected);

            let flat_ids = ple_tokens.reshape([batch * seq]);
            self.decoder
                .layers
                .iter()
                .enumerate()
                .map(|(i, layer)| {
                    layer.ple.as_ref().map(|ple| {
                        let ple_embed = ple
                            .embed_slice
                            .clone()
                            .select(0, flat_ids.clone())
                            .reshape([batch, seq, ple_dim]);
                        let proj_slice = projected
                            .clone()
                            .slice([0..batch, 0..seq, i..i + 1, 0..ple_dim])
                            .reshape([batch, seq, ple_dim]);
                        (proj_slice + ple_embed) * self.decoder.per_layer_input_scale
                    })
                })
                .collect()
        } else {
            self.decoder.layers.iter().map(|_| None).collect()
        };

        // Embeddings are already scaled
        let mut h = inputs_embeds;

        // Bidirectional-vision mask (31B+): sliding layers unmask every
        // (q, kv) pair within the same multimodal span in addition to the
        // standard causal constraint. Full-attention layers stay causal.
        // E2B/E4B have use_bidirectional_attention=None, so no mask is
        // built and the attention layer falls back to its builtin causal.
        let bidir_vision = self.config.use_bidirectional_attention.as_deref() == Some("vision");
        let [_, seq_len, _] = h.dims();
        let sliding_mask: Option<Tensor<B, 4>> = if bidir_vision
            && seq_len > 1
            && !mm_ranges.is_empty()
        {
            // Start from pure causal, then OR-unmask within each span.
            let mut data = vec![f32::NEG_INFINITY; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    data[i * seq_len + j] = 0.0;
                }
            }
            for &(s, e) in mm_ranges {
                let s = s.min(seq_len);
                let e = e.min(seq_len);
                for i in s..e {
                    for j in s..e {
                        data[i * seq_len + j] = 0.0;
                    }
                }
            }
            Some(Tensor::<B, 1>::from_floats(&data[..], &device).reshape([1, 1, seq_len, seq_len]))
        } else {
            None
        };

        // Plain causal prefill mask, built ONCE per forward and shared by every
        // layer. Without this each attention layer rebuilt an identical S²
        // mask on the CPU row-by-row (42 CPU round-trips per prefill on E4B);
        // the attention-internal builder now only remains as a fallback for
        // direct callers. Valid because forward_inner prefills from position 0
        // (offset 0) — the decode path (seq_len == 1) needs no mask at all.
        let causal_mask: Option<Tensor<B, 4>> = if seq_len > 1 {
            let mut data = vec![f32::NEG_INFINITY; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    data[i * seq_len + j] = 0.0;
                }
            }
            Some(Tensor::<B, 1>::from_floats(&data[..], &device).reshape([1, 1, seq_len, seq_len]))
        } else {
            None
        };

        // KV sharing: layers 15+ reuse K/V from source layers (13 or 14).
        // We store the K/V after source layers process and inject them for shared layers.
        let first_shared = self.config.first_shared_kv_layer();
        let mut shared_kv: std::collections::HashMap<usize, (Tensor<B, 4>, Tensor<B, 4>)> =
            std::collections::HashMap::new();

        for (i, (layer, cache)) in self
            .decoder
            .layers
            .iter()
            .zip(caches.caches.iter_mut())
            .enumerate()
        {
            let rope = match self.config.layer_type(i) {
                super::config::LayerType::SlidingAttention => rope_sliding,
                super::config::LayerType::FullAttention => rope_global,
            };

            // For KV-shared layers: inject source layer's KV into this layer's cache
            if i >= first_shared {
                // Source layer: sliding layers (15-18, 20-23, ...) use layer first_shared-2 (13)
                // Global layers (19, 24, 29, 34) use layer first_shared-1 (14)
                let source = match self.config.layer_type(i) {
                    super::config::LayerType::SlidingAttention => first_shared - 2,
                    super::config::LayerType::FullAttention => first_shared - 1,
                };
                if let Some((k, v)) = shared_kv.get(&source) {
                    // Pre-populate cache with source layer's KV
                    cache.k = Some(k.clone());
                    cache.v = Some(v.clone());
                }
            }

            // Sliding layers get the bidirectional-vision mask when enabled;
            // everything else gets the shared plain-causal prefill mask
            // (None during decode — new_len == 1 needs no mask).
            let layer_mask = match self.config.layer_type(i) {
                super::config::LayerType::SlidingAttention => {
                    sliding_mask.as_ref().or(causal_mask.as_ref())
                }
                super::config::LayerType::FullAttention => causal_mask.as_ref(),
            };
            h = layer.forward(
                h,
                rope,
                cache,
                ple_inputs[i].as_ref(),
                layer_mask,
                lora.map(|l| (l, i)),
            );

            // Debug: dump per-layer output when GAZE_DUMP_LAYERS env is set.
            if std::env::var("GAZE_DUMP_LAYERS").is_ok() && seq_len > 1 {
                let dir =
                    std::env::var("GAZE_DUMP_DIR").unwrap_or_else(|_| "/tmp/rust_layers".into());
                std::fs::create_dir_all(&dir).ok();
                let data: Vec<f32> = h.clone().to_data().to_vec().unwrap();
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/layer_{i:02}.bin"), bytes).ok();
            }

            // Store KV from source layers for sharing
            if i == first_shared - 2 || i == first_shared - 1 {
                if let (Some(k), Some(v)) = (&cache.k, &cache.v) {
                    shared_kv.insert(i, (k.clone(), v.clone()));
                }
            }
        }

        // Final norm
        h = self.decoder.norm.forward(h);

        // Logit projection with softcapping

        // Logit projection with softcapping
        let logits = match &self.decoder.lm_head {
            Some(lm_head) => lm_head.forward(h),
            None => {
                // Tied lm_head: h @ W.T where W is [vocab, hidden].
                let [b, t, _] = h.dims();
                let hidden = self.config.hidden_size;
                let w = self.decoder.embed.weight.val().swap_dims(0, 1); // [H, V]
                let [_h, v] = w.dims();
                h.reshape([b * t, hidden]).matmul(w).reshape([b, t, v])
            }
        };

        // Final logit softcapping: softcap * tanh(logits / softcap)
        let softcap = self.config.final_logit_softcapping as f32;
        burn::tensor::activation::tanh(logits / softcap) * softcap
    }

    /// Create empty KV caches for all layers.
    pub fn new_caches(&self) -> LayerCaches<B> {
        LayerCaches::new(self.config.num_hidden_layers)
    }

    /// Build the two RoPE tables for this model.
    /// Returns (sliding_rope, global_rope).
    pub fn rope_tables(&self, device: &B::Device) -> (RopeTable<B>, RopeTable<B>) {
        let sliding_config = &self.config.rope_parameters.sliding_attention;
        let global_config = &self.config.rope_parameters.full_attention;

        // Sliding: standard RoPE with full head_dim. The table must cover the
        // FULL prefill length, not sliding_window*2 — RoPE positions are
        // ABSOLUTE over the sequence (apply() slices [offset..offset+seq_len]),
        // not relative to the sliding window. sliding_window*2 (=1024 for a 512
        // window) panicked at rope.rs on any prompt longer than that, and real
        // agent prompts (system prompt alone ~1.4k tok, window 32k) are far
        // larger. Size it like the global table.
        let max_len = 131072.min(self.config.vocab_size);
        let sliding_rope = RopeTable::new(
            self.config.head_dim,
            max_len,
            sliding_config.rope_theta,
            device,
        );

        // Global: proportional RoPE with partial rotation
        // Must be full head_dim (512) with zeros for non-rotated frequencies,
        // because rotate_half operates on the full dimension.
        let global_head_dim = self.config.global_head_dim();
        let global_max_len = 131072.min(self.config.vocab_size);
        let global_rope = RopeTable::with_partial_rotation(
            global_head_dim,
            global_max_len,
            global_config.rope_theta,
            self.config
                .rope_parameters
                .full_attention
                .partial_rotary_factor,
            device,
        );

        (sliding_rope, global_rope)
    }
}
