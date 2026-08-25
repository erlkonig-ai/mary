use burn::prelude::*;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

use super::resnet::{ResnetBlock2D, conv2d_forward};
use crate::nn::weight_loader::WeightLoader;

/// Nearest-neighbor 2x upsampling followed by a Conv2d.
/// This matches the diffusers Upsample2D(use_conv=True) behavior:
/// nearest-neighbor interpolation to 2x size, then Conv2d(ch, ch, 3, padding=1).
pub struct Upsample2D<B: Backend> {
    pub conv_weight: Tensor<B, 4>, // [out_ch, in_ch, 3, 3]
    pub conv_bias: Tensor<B, 1>,   // [out_ch]
}

impl<B: Backend> Upsample2D<B> {
    /// Load from safetensors.
    /// Example prefix: "decoder.up_blocks.0.upsamplers.0"
    pub fn load(loader: &WeightLoader, prefix: &str, device: &B::Device) -> Self {
        Self {
            conv_weight: loader.load_tensor(&format!("{prefix}.conv.weight"), device),
            conv_bias: loader.load_tensor(&format!("{prefix}.conv.bias"), device),
        }
    }

    /// Forward: nearest-neighbor upsample 2x then conv2d 3x3 with padding 1.
    /// Input: [B, C, H, W] -> Output: [B, C, 2*H, 2*W]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, _channels, height, width] = x.dims();
        let new_h = height * 2;
        let new_w = width * 2;

        // Nearest-neighbor upsample 2x using burn's interpolate
        let x = interpolate(
            x,
            [new_h, new_w],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );

        // Conv2d 3x3 with padding 1
        conv2d_forward(
            x,
            self.conv_weight.clone(),
            Some(self.conv_bias.clone()),
            [1, 1],
            [1, 1],
        )
    }
}

/// An up-decoder block consisting of multiple ResNet blocks and an optional upsampler.
///
/// In the diffusers Decoder, up blocks process channels in reversed order:
///   block_out_channels = [128, 256, 512, 512]
///   reversed = [512, 512, 256, 128]
///
/// For each up block i:
///   - The first resnet takes in_channels (prev_output_channel) and outputs out_channels
///   - Remaining resnets take out_channels -> out_channels
///   - An upsampler is added unless this is the last block
///
/// Actual block configuration (from the safetensors):
///   up_block 0: 512->512, 3 resnets, has upsample
///   up_block 1: 512->512, 3 resnets, has upsample
///   up_block 2: 512->256, 3 resnets (first takes 512->256), has upsample
///   up_block 3: 256->128, 3 resnets (first takes 256->128), NO upsample
pub struct UpDecoderBlock2D<B: Backend> {
    pub resnets: Vec<ResnetBlock2D<B>>,
    pub upsampler: Option<Upsample2D<B>>,
}

impl<B: Backend> UpDecoderBlock2D<B> {
    /// Load from safetensors.
    /// `prefix`: e.g. "decoder.up_blocks.0"
    /// `in_channels`: input channels (from previous block)
    /// `out_channels`: output channels for this block
    /// `num_layers`: number of ResNet blocks (layers_per_block + 1 = 3)
    /// `add_upsample`: whether to add an upsampler
    /// `num_groups`: number of groups for GroupNorm
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        add_upsample: bool,
        num_groups: usize,
        device: &B::Device,
    ) -> Self {
        let mut resnets = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let resnet_in_ch = if i == 0 { in_channels } else { out_channels };
            let resnet_prefix = format!("{prefix}.resnets.{i}");
            resnets.push(ResnetBlock2D::load(
                loader,
                &resnet_prefix,
                resnet_in_ch,
                out_channels,
                num_groups,
                device,
            ));
        }

        let upsampler = if add_upsample {
            let upsample_prefix = format!("{prefix}.upsamplers.0");
            Some(Upsample2D::load(loader, &upsample_prefix, device))
        } else {
            None
        };

        Self { resnets, upsampler }
    }

    /// Forward pass: run through all ResNet blocks, then optionally upsample.
    /// Input: [B, C_in, H, W] -> Output: [B, C_out, H', W']
    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for resnet in &self.resnets {
            x = resnet.forward(x);
        }

        if let Some(ref upsampler) = self.upsampler {
            x = upsampler.forward(x);
        }

        x
    }
}
