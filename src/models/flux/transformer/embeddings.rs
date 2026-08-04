use burn::prelude::*;
use burn::tensor::activation::silu;

use super::config::Flux2TransformerConfig;
use crate::nn::weight_loader::WeightLoader;

/// Apply a linear layer without bias: x @ weight^T
/// For 3D input: x [B, S, D_in], weight [D_out, D_in] -> [B, S, D_out]
pub fn linear3d<B: Backend>(x: Tensor<B, 3>, weight: Tensor<B, 2>) -> Tensor<B, 3> {
    // weight: [D_out, D_in] -> transpose -> [D_in, D_out] -> unsqueeze -> [1, D_in, D_out]
    let wt = weight.transpose().unsqueeze::<3>(); // [1, D_in, D_out]
    x.matmul(wt)
}

/// Apply a linear layer without bias for 2D input: x @ weight^T
/// x [N, D_in], weight [D_out, D_in] -> [N, D_out]
pub fn linear2d<B: Backend>(x: Tensor<B, 2>, weight: Tensor<B, 2>) -> Tensor<B, 2> {
    x.matmul(weight.transpose())
}

/// Sinusoidal timestep embedding (matches diffusers `get_timestep_embedding`).
///
/// timesteps: [N] (1-D tensor of timestep values, may be fractional)
/// embedding_dim: total output dimension (e.g. 256)
/// flip_sin_to_cos: if true, output is [cos, sin] instead of [sin, cos]
/// downscale_freq_shift: controls delta between frequencies (0 for Flux2)
///
/// Returns: [N, embedding_dim]
pub fn get_timestep_embedding<B: Backend>(
    timesteps: Tensor<B, 1>,
    embedding_dim: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
    device: &B::Device,
) -> Tensor<B, 2> {
    let half_dim = embedding_dim / 2;
    let max_period: f64 = 10000.0;

    // exponent = -ln(max_period) * arange(0, half_dim) / (half_dim - downscale_freq_shift)
    let mut exponent_data = Vec::with_capacity(half_dim);
    let log_max = -max_period.ln();
    let denom = half_dim as f64 - downscale_freq_shift;
    for i in 0..half_dim {
        exponent_data.push((log_max * (i as f64) / denom) as f32);
    }
    let exponent = Tensor::<B, 1>::from_floats(exponent_data.as_slice(), device); // [half_dim]
    let freqs = exponent.exp(); // [half_dim]

    // emb = timesteps[:, None] * freqs[None, :]
    let t = timesteps.unsqueeze_dim::<2>(1); // [N, 1]
    let f = freqs.unsqueeze_dim::<2>(0); // [1, half_dim]
    let emb = t * f; // [N, half_dim]

    // [sin(emb), cos(emb)]
    let sin_emb = emb.clone().sin(); // [N, half_dim]
    let cos_emb = emb.cos(); // [N, half_dim]

    if flip_sin_to_cos {
        // [cos, sin]
        Tensor::cat(vec![cos_emb, sin_emb], 1)
    } else {
        // [sin, cos]
        Tensor::cat(vec![sin_emb, cos_emb], 1)
    }
    // [N, embedding_dim]
}

/// TimestepEmbedding MLP: Linear(in_channels, inner_dim) -> SiLU -> Linear(inner_dim, inner_dim)
/// All linear layers are bias=False for FLUX.2-klein.
pub struct TimestepEmbedding<B: Backend> {
    pub linear_1_weight: Tensor<B, 2>, // [inner_dim, in_channels]
    pub linear_2_weight: Tensor<B, 2>, // [inner_dim, inner_dim]
}

impl<B: Backend> TimestepEmbedding<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, device: &B::Device) -> Self {
        Self {
            linear_1_weight: loader.load_tensor(&format!("{prefix}.linear_1.weight"), device),
            linear_2_weight: loader.load_tensor(&format!("{prefix}.linear_2.weight"), device),
        }
    }

    /// Forward: linear_1 -> silu -> linear_2
    /// x: [N, in_channels]
    /// Returns: [N, inner_dim]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = linear2d(x, self.linear_1_weight.clone());
        let h = silu(h);
        linear2d(h, self.linear_2_weight.clone())
    }
}

/// Flux2TimestepGuidanceEmbeddings: time_proj + timestep_embedder + optional guidance_embedder.
/// Klein: guidance_embeds=false → no guidance_embedder.
/// Dev: guidance_embeds=true → guidance_embedder loaded and summed into temb.
pub struct Flux2TimestepGuidanceEmbeddings<B: Backend> {
    pub timestep_embedder: TimestepEmbedding<B>,
    pub guidance_embedder: Option<TimestepEmbedding<B>>,
    pub in_channels: usize, // 256 (num_channels for sinusoidal projection)
}

impl<B: Backend> Flux2TimestepGuidanceEmbeddings<B> {
    pub fn load(
        loader: &WeightLoader,
        config: &Flux2TransformerConfig,
        device: &B::Device,
    ) -> Self {
        let guidance_embedder = if config.guidance_embeds {
            Some(TimestepEmbedding::load(
                loader,
                "time_guidance_embed.guidance_embedder",
                device,
            ))
        } else {
            None
        };

        Self {
            timestep_embedder: TimestepEmbedding::load(
                loader,
                "time_guidance_embed.timestep_embedder",
                device,
            ),
            guidance_embedder,
            in_channels: config.timestep_guidance_channels,
        }
    }

    /// Forward pass.
    /// timestep: [N] scalar timestep values (already scaled by 1000 externally).
    /// guidance: Optional guidance scale as [N] tensor. Used when guidance_embeds=true.
    /// Returns: [N, inner_dim]
    pub fn forward(
        &self,
        timestep: Tensor<B, 1>,
        guidance: Option<Tensor<B, 1>>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        // Sinusoidal time projection: flip_sin_to_cos=true, downscale_freq_shift=0
        let timesteps_proj = get_timestep_embedding(timestep, self.in_channels, true, 0.0, device);
        let mut temb = self.timestep_embedder.forward(timesteps_proj);

        // Add guidance embedding if present
        if let (Some(guidance_embedder), Some(guidance)) = (&self.guidance_embedder, guidance) {
            let guidance_proj =
                get_timestep_embedding(guidance, self.in_channels, true, 0.0, device);
            temb = temb + guidance_embedder.forward(guidance_proj);
        }

        temb
    }
}
