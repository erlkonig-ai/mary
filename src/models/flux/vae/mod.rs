pub mod attention;
pub mod config;
pub mod decoder;
pub mod down_block;
pub mod encoder;
pub mod resnet;
pub mod up_block;

use burn::prelude::*;

use crate::nn::weight_loader::WeightLoader;
use config::VaeConfig;
use decoder::Decoder;
use encoder::Encoder;
use resnet::conv2d_forward;

/// AutoencoderKLFlux2 — VAE for FLUX.2 (encode + decode).
///
/// Structure:
///   - encoder: Encoder (conv_in -> down_blocks -> mid_block -> conv_out)
///   - quant_conv: Conv2d(2*latent_channels, 2*latent_channels, 1)
///   - post_quant_conv: Conv2d(latent_channels, latent_channels, 1)
///   - decoder: Decoder (conv_in -> mid_block -> up_blocks -> conv_out)
///   - bn_running_mean / bn_running_var: BatchNorm statistics [128]
///
/// Encode path:
///   1. encoder.forward(image) -> [B, 64, H/8, W/8]
///   2. quant_conv -> [B, 64, H/8, W/8]
///   3. Split into mean [B, 32, H/8, W/8] and logvar [B, 32, H/8, W/8]
///   4. Return mean (deterministic) or sample from N(mean, exp(logvar))
///
/// Decode path:
///   1. post_quant_conv(latents)
///   2. decoder.forward(latents)
///   3. Clamp output to [-1, 1]
pub struct AutoencoderKLFlux2<B: Backend> {
    pub encoder: Option<Encoder<B>>,
    pub decoder: Decoder<B>,

    // quant_conv: 1x1 convolution (2*latent_channels -> 2*latent_channels) for encode
    pub quant_conv_weight: Option<Tensor<B, 4>>,
    pub quant_conv_bias: Option<Tensor<B, 1>>,

    // post_quant_conv: 1x1 convolution (latent_channels -> latent_channels) for decode
    pub post_quant_conv_weight: Tensor<B, 4>,
    pub post_quant_conv_bias: Tensor<B, 1>,

    // BatchNorm running statistics for normalization/denormalization
    pub bn_running_mean: Tensor<B, 1>, // [128]
    pub bn_running_var: Tensor<B, 1>,  // [128]
    pub bn_eps: f64,

    pub config: VaeConfig,
}

impl<B: Backend> AutoencoderKLFlux2<B> {
    /// Load decode-only VAE (backward compatible — skips encoder weights).
    pub fn load(loader: &WeightLoader, config: VaeConfig, device: &B::Device) -> Self {
        let decoder = Decoder::load(loader, &config, device);

        let post_quant_conv_weight = loader.load_tensor::<B, 4>("post_quant_conv.weight", device);
        let post_quant_conv_bias = loader.load_tensor::<B, 1>("post_quant_conv.bias", device);

        let bn_running_mean = loader.load_tensor::<B, 1>("bn.running_mean", device);
        let bn_running_var = loader.load_tensor::<B, 1>("bn.running_var", device);
        let bn_eps = config.batch_norm_eps;

        Self {
            encoder: None,
            decoder,
            quant_conv_weight: None,
            quant_conv_bias: None,
            post_quant_conv_weight,
            post_quant_conv_bias,
            bn_running_mean,
            bn_running_var,
            bn_eps,
            config,
        }
    }

    /// Load full VAE with both encoder and decoder.
    pub fn load_with_encoder(loader: &WeightLoader, config: VaeConfig, device: &B::Device) -> Self {
        let encoder = Encoder::load(loader, &config, device);
        let decoder = Decoder::load(loader, &config, device);

        let quant_conv_weight = loader.load_tensor::<B, 4>("quant_conv.weight", device);
        let quant_conv_bias = loader.load_tensor::<B, 1>("quant_conv.bias", device);
        let post_quant_conv_weight = loader.load_tensor::<B, 4>("post_quant_conv.weight", device);
        let post_quant_conv_bias = loader.load_tensor::<B, 1>("post_quant_conv.bias", device);

        let bn_running_mean = loader.load_tensor::<B, 1>("bn.running_mean", device);
        let bn_running_var = loader.load_tensor::<B, 1>("bn.running_var", device);
        let bn_eps = config.batch_norm_eps;

        Self {
            encoder: Some(encoder),
            decoder,
            quant_conv_weight: Some(quant_conv_weight),
            quant_conv_bias: Some(quant_conv_bias),
            post_quant_conv_weight,
            post_quant_conv_bias,
            bn_running_mean,
            bn_running_var,
            bn_eps,
            config,
        }
    }

    /// Encode an image to latent space (deterministic — returns the mean).
    ///
    /// Input: [B, 3, H, W] normalized to [-1, 1]
    /// Output: [B, latent_channels=32, H/8, W/8]
    ///
    /// Steps:
    ///   1. Encoder forward -> [B, 64, H/8, W/8]
    ///   2. quant_conv -> [B, 64, H/8, W/8]
    ///   3. Split: mean = first 32 channels (deterministic encoding)
    pub fn encode(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let encoder = self
            .encoder
            .as_ref()
            .expect("Encoder not loaded — use load_with_encoder()");
        let qc_weight = self
            .quant_conv_weight
            .as_ref()
            .expect("quant_conv not loaded");
        let qc_bias = self
            .quant_conv_bias
            .as_ref()
            .expect("quant_conv not loaded");

        // 1. Encoder forward
        let h = encoder.forward(image);

        // 2. quant_conv: 1x1 convolution
        let h = conv2d_forward(h, qc_weight.clone(), Some(qc_bias.clone()), [1, 1], [0, 0]);

        // 3. Split into mean and logvar, return mean (deterministic)
        let [b, c, height, width] = h.dims();
        let latent_ch = c / 2;
        h.slice([0..b, 0..latent_ch, 0..height, 0..width])
    }

    /// Decode latents to image pixels.
    ///
    /// Input: [B, latent_channels, H, W]
    /// Output: [B, 3, H', W'] clamped to [-1, 1]
    pub fn decode(&self, latents: Tensor<B, 4>) -> Tensor<B, 4> {
        // 1. post_quant_conv: 1x1 convolution
        let z = conv2d_forward(
            latents,
            self.post_quant_conv_weight.clone(),
            Some(self.post_quant_conv_bias.clone()),
            [1, 1],
            [0, 0],
        );

        // 2. Decoder forward pass
        let dec = self.decoder.forward(z);

        // 3. Clamp to [-1, 1]
        dec.clamp(-1.0, 1.0)
    }
}
