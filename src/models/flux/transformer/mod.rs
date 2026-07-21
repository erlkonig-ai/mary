pub mod attention;
pub mod config;
pub mod double_stream_block;
pub mod embeddings;
pub mod feed_forward;
pub mod modulation;
pub mod norm;
pub mod rope;
pub mod single_stream_block;

use burn::prelude::*;

use config::Flux2TransformerConfig;
use double_stream_block::Flux2TransformerBlock;
use embeddings::{linear3d, Flux2TimestepGuidanceEmbeddings};
use modulation::{split_modulation, Flux2Modulation};
use norm::AdaLayerNormContinuous;
use rope::flux2_rope;
use single_stream_block::Flux2SingleTransformerBlock;

use crate::nn::weight_loader::WeightLoader;

/// Flux2Transformer2DModel: full diffusion transformer.
///
/// Assembles all components:
/// - Sinusoidal timestep embedding + MLP
/// - 3 shared modulation layers (double_stream_img, double_stream_txt, single_stream)
/// - Input projections (x_embedder, context_embedder)
/// - 5 double-stream blocks
/// - 20 single-stream blocks
/// - Output norm (AdaLayerNormContinuous) + proj_out
pub struct Flux2Transformer2DModel<B: Backend> {
    // Timestep embedding
    pub time_guidance_embed: Flux2TimestepGuidanceEmbeddings<B>,

    // Shared modulation layers
    pub double_stream_modulation_img: Flux2Modulation<B>,
    pub double_stream_modulation_txt: Flux2Modulation<B>,
    pub single_stream_modulation: Flux2Modulation<B>,

    // Input projections
    pub x_embedder_weight: Tensor<B, 2>,       // [inner_dim, in_channels]
    pub context_embedder_weight: Tensor<B, 2>,  // [inner_dim, joint_attention_dim]

    // Transformer blocks
    pub transformer_blocks: Vec<Flux2TransformerBlock<B>>,
    pub single_transformer_blocks: Vec<Flux2SingleTransformerBlock<B>>,

    // Output layers
    pub norm_out: AdaLayerNormContinuous<B>,
    pub proj_out_weight: Tensor<B, 2>, // [out_channels, inner_dim]

    // Config
    pub config: Flux2TransformerConfig,
}

impl<B: Backend> Flux2Transformer2DModel<B> {
    /// Load all weights (header + all blocks) into memory.
    /// Suitable for models that fit entirely in GPU memory (e.g. Klein ~12GB f32).
    pub fn load(
        loader: &WeightLoader,
        config: Flux2TransformerConfig,
        device: &B::Device,
    ) -> Self {
        let inner_dim = config.inner_dim();
        let mlp_hidden_dim = config.mlp_hidden_dim();
        let num_heads = config.num_attention_heads;
        let head_dim = config.attention_head_dim;
        let eps = config.eps;

        // Time embedding
        let time_guidance_embed =
            Flux2TimestepGuidanceEmbeddings::load(loader, &config, device);

        // Shared modulation layers
        let double_stream_modulation_img =
            Flux2Modulation::load(loader, "double_stream_modulation_img", 2, device);
        let double_stream_modulation_txt =
            Flux2Modulation::load(loader, "double_stream_modulation_txt", 2, device);
        let single_stream_modulation =
            Flux2Modulation::load(loader, "single_stream_modulation", 1, device);

        // Input projections
        let x_embedder_weight: Tensor<B, 2> =
            loader.load_tensor("x_embedder.weight", device);
        let context_embedder_weight: Tensor<B, 2> =
            loader.load_tensor("context_embedder.weight", device);

        // Double-stream transformer blocks
        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            transformer_blocks.push(Flux2TransformerBlock::load(
                loader, i, num_heads, head_dim, eps, device,
            ));
        }

        // Single-stream transformer blocks
        let mut single_transformer_blocks = Vec::with_capacity(config.num_single_layers);
        for i in 0..config.num_single_layers {
            single_transformer_blocks.push(Flux2SingleTransformerBlock::load(
                loader,
                i,
                num_heads,
                head_dim,
                inner_dim,
                mlp_hidden_dim,
                eps,
                device,
            ));
        }

        // Output layers
        let norm_out = AdaLayerNormContinuous::load(loader, "norm_out", eps, device);
        let proj_out_weight: Tensor<B, 2> =
            loader.load_tensor("proj_out.weight", device);

        Self {
            time_guidance_embed,
            double_stream_modulation_img,
            double_stream_modulation_txt,
            single_stream_modulation,
            x_embedder_weight,
            context_embedder_weight,
            transformer_blocks,
            single_transformer_blocks,
            norm_out,
            proj_out_weight,
            config,
        }
    }

    /// Load only header weights (no blocks). ~3GB f32 for Dev.
    /// Use with `forward_streaming()` which loads blocks on-the-fly.
    pub fn load_header_only(
        loader: &WeightLoader,
        config: Flux2TransformerConfig,
        device: &B::Device,
    ) -> Self {
        let time_guidance_embed =
            Flux2TimestepGuidanceEmbeddings::load(loader, &config, device);

        let double_stream_modulation_img =
            Flux2Modulation::load(loader, "double_stream_modulation_img", 2, device);
        let double_stream_modulation_txt =
            Flux2Modulation::load(loader, "double_stream_modulation_txt", 2, device);
        let single_stream_modulation =
            Flux2Modulation::load(loader, "single_stream_modulation", 1, device);

        let x_embedder_weight: Tensor<B, 2> =
            loader.load_tensor("x_embedder.weight", device);
        let context_embedder_weight: Tensor<B, 2> =
            loader.load_tensor("context_embedder.weight", device);

        let norm_out = AdaLayerNormContinuous::load(loader, "norm_out", config.eps, device);
        let proj_out_weight: Tensor<B, 2> =
            loader.load_tensor("proj_out.weight", device);

        Self {
            time_guidance_embed,
            double_stream_modulation_img,
            double_stream_modulation_txt,
            single_stream_modulation,
            x_embedder_weight,
            context_embedder_weight,
            transformer_blocks: Vec::new(),
            single_transformer_blocks: Vec::new(),
            norm_out,
            proj_out_weight,
            config,
        }
    }

    /// Forward pass.
    ///
    /// hidden_states: [B, S_img, in_channels] (packed latent noise)
    /// encoder_hidden_states: [B, S_txt, joint_attention_dim] (text encoder output)
    /// timestep: [B] (scalar timestep, e.g. 0.0 to 1.0)
    /// img_ids: [S_img, 4] (position IDs for image tokens, batch dim already squeezed)
    /// txt_ids: [S_txt, 4] (position IDs for text tokens, batch dim already squeezed)
    ///
    /// Returns: [B, S_img, out_channels]
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        timestep: Tensor<B, 1>,
        guidance: Option<Tensor<B, 1>>,
        img_ids: Tensor<B, 2>, // [S_img, 4]
        txt_ids: Tensor<B, 2>, // [S_txt, 4]
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [batch, _s_img, _in_ch] = hidden_states.dims();
        let s_txt = encoder_hidden_states.dims()[1];

        // 0. Scale timestep
        let timestep = timestep * 1000.0;

        // 1. Timestep + guidance embedding -> [B, inner_dim]
        let temb = self.time_guidance_embed.forward(timestep, guidance, device);

        // 2. Compute all modulation params upfront (shared)
        let double_mod_img = self.double_stream_modulation_img.forward(temb.clone());
        let double_mod_txt = self.double_stream_modulation_txt.forward(temb.clone());
        let single_mod = self.single_stream_modulation.forward(temb.clone());

        // Split modulation into per-set (shift, scale, gate) tuples
        let double_img_mods = split_modulation(double_mod_img, 2);
        let double_txt_mods = split_modulation(double_mod_txt, 2);
        let single_mods = split_modulation(single_mod, 1);

        // 3. Input projections
        let mut hidden_states =
            linear3d(hidden_states, self.x_embedder_weight.clone());
        let mut encoder_hidden_states =
            linear3d(encoder_hidden_states, self.context_embedder_weight.clone());

        // 4. Compute RoPE from position IDs
        let (img_cos, img_sin) = flux2_rope(
            img_ids,
            &self.config.axes_dims_rope,
            self.config.rope_theta,
            device,
        );
        let (txt_cos, txt_sin) = flux2_rope(
            txt_ids,
            &self.config.axes_dims_rope,
            self.config.rope_theta,
            device,
        );
        // Concatenate text + image RoPE (text first)
        let concat_cos = Tensor::cat(vec![txt_cos, img_cos], 0); // [S_txt+S_img, D_rope]
        let concat_sin = Tensor::cat(vec![txt_sin, img_sin], 0);

        // 5. Double-stream transformer blocks
        for block in &self.transformer_blocks {
            let (img_msa_shift, img_msa_scale, img_msa_gate) = double_img_mods[0].clone();
            let (img_mlp_shift, img_mlp_scale, img_mlp_gate) = double_img_mods[1].clone();
            let (txt_msa_shift, txt_msa_scale, txt_msa_gate) = double_txt_mods[0].clone();
            let (txt_mlp_shift, txt_mlp_scale, txt_mlp_gate) = double_txt_mods[1].clone();

            let (enc_out, hid_out) = block.forward(
                hidden_states,
                encoder_hidden_states,
                img_msa_shift,
                img_msa_scale,
                img_msa_gate,
                img_mlp_shift,
                img_mlp_scale,
                img_mlp_gate,
                txt_msa_shift,
                txt_msa_scale,
                txt_msa_gate,
                txt_mlp_shift,
                txt_mlp_scale,
                txt_mlp_gate,
                concat_cos.clone(),
                concat_sin.clone(),
            );
            encoder_hidden_states = enc_out;
            hidden_states = hid_out;
        }

        // 6. Concatenate text + image for single-stream blocks
        let mut combined =
            Tensor::cat(vec![encoder_hidden_states, hidden_states], 1);

        // 7. Single-stream transformer blocks
        let (ss_shift, ss_scale, ss_gate) = single_mods[0].clone();
        for block in &self.single_transformer_blocks {
            combined = block.forward(
                combined,
                ss_shift.clone(),
                ss_scale.clone(),
                ss_gate.clone(),
                concat_cos.clone(),
                concat_sin.clone(),
            );
        }

        // 8. Remove text tokens (keep only image portion)
        let s_total = combined.dims()[1];
        let inner_dim = self.config.inner_dim();
        let hidden_states =
            combined.slice([0..batch, s_txt..s_total, 0..inner_dim]);

        // 9. Output norm + projection
        let hidden_states = self.norm_out.forward(hidden_states, temb);
        linear3d(hidden_states, self.proj_out_weight.clone())
    }

    /// Streaming forward pass: loads each block on demand from the loader.
    /// Only header weights (modulations, embedders, norms) need to be in memory.
    /// Each block is loaded, used, then dropped — peak ~3.7GB per block instead of ~120GB total.
    pub fn forward_streaming(
        &self,
        loader: &WeightLoader,
        hidden_states: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        timestep: Tensor<B, 1>,
        guidance: Option<Tensor<B, 1>>,
        img_ids: Tensor<B, 2>,
        txt_ids: Tensor<B, 2>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [batch, _s_img, _in_ch] = hidden_states.dims();
        let s_txt = encoder_hidden_states.dims()[1];
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.attention_head_dim;
        let inner_dim = self.config.inner_dim();
        let mlp_hidden_dim = self.config.mlp_hidden_dim();
        let eps = self.config.eps;

        // 0-4: Same as forward() — uses header weights only
        let timestep = timestep * 1000.0;
        let temb = self.time_guidance_embed.forward(timestep, guidance, device);

        let double_mod_img = self.double_stream_modulation_img.forward(temb.clone());
        let double_mod_txt = self.double_stream_modulation_txt.forward(temb.clone());
        let single_mod = self.single_stream_modulation.forward(temb.clone());

        let double_img_mods = split_modulation(double_mod_img, 2);
        let double_txt_mods = split_modulation(double_mod_txt, 2);
        let single_mods = split_modulation(single_mod, 1);

        let mut hidden_states = linear3d(hidden_states, self.x_embedder_weight.clone());
        let mut encoder_hidden_states =
            linear3d(encoder_hidden_states, self.context_embedder_weight.clone());

        let (img_cos, img_sin) = flux2_rope(
            img_ids,
            &self.config.axes_dims_rope,
            self.config.rope_theta,
            device,
        );
        let (txt_cos, txt_sin) = flux2_rope(
            txt_ids,
            &self.config.axes_dims_rope,
            self.config.rope_theta,
            device,
        );
        let concat_cos = Tensor::cat(vec![txt_cos, img_cos], 0);
        let concat_sin = Tensor::cat(vec![txt_sin, img_sin], 0);

        // 5. Double-stream blocks — loaded one at a time
        for i in 0..self.config.num_layers {
            let block = Flux2TransformerBlock::load(loader, i, num_heads, head_dim, eps, device);

            let (img_msa_shift, img_msa_scale, img_msa_gate) = double_img_mods[0].clone();
            let (img_mlp_shift, img_mlp_scale, img_mlp_gate) = double_img_mods[1].clone();
            let (txt_msa_shift, txt_msa_scale, txt_msa_gate) = double_txt_mods[0].clone();
            let (txt_mlp_shift, txt_mlp_scale, txt_mlp_gate) = double_txt_mods[1].clone();

            let (enc_out, hid_out) = block.forward(
                hidden_states,
                encoder_hidden_states,
                img_msa_shift,
                img_msa_scale,
                img_msa_gate,
                img_mlp_shift,
                img_mlp_scale,
                img_mlp_gate,
                txt_msa_shift,
                txt_msa_scale,
                txt_msa_gate,
                txt_mlp_shift,
                txt_mlp_scale,
                txt_mlp_gate,
                concat_cos.clone(),
                concat_sin.clone(),
            );
            encoder_hidden_states = enc_out;
            hidden_states = hid_out;
            // block is dropped here, freeing ~3.7GB
        }

        // 6. Concatenate for single-stream
        let mut combined = Tensor::cat(vec![encoder_hidden_states, hidden_states], 1);

        // 7. Single-stream blocks — loaded one at a time
        let (ss_shift, ss_scale, ss_gate) = single_mods[0].clone();
        for i in 0..self.config.num_single_layers {
            let block = Flux2SingleTransformerBlock::load(
                loader, i, num_heads, head_dim, inner_dim, mlp_hidden_dim, eps, device,
            );
            combined = block.forward(
                combined,
                ss_shift.clone(),
                ss_scale.clone(),
                ss_gate.clone(),
                concat_cos.clone(),
                concat_sin.clone(),
            );
            // block is dropped here, freeing ~1.9GB
        }

        // 8. Remove text tokens
        let s_total = combined.dims()[1];
        let hidden_states = combined.slice([0..batch, s_txt..s_total, 0..inner_dim]);

        // 9. Output norm + projection
        let hidden_states = self.norm_out.forward(hidden_states, temb);
        linear3d(hidden_states, self.proj_out_weight.clone())
    }
}
