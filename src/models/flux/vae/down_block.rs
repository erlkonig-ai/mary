use burn::prelude::*;

use super::resnet::{ResnetBlock2D, conv2d_forward};
use crate::nn::weight_loader::WeightLoader;

/// Stride-2 Conv2d downsampler with asymmetric padding (matches diffusers Downsample2D).
///
/// Mirrors `Upsample2D` from `up_block.rs`. Halves spatial dimensions.
/// Uses asymmetric padding [0,1,0,1] (right and bottom) before stride-2 conv.
pub struct Downsample2D<B: Backend> {
    pub conv_weight: Tensor<B, 4>, // [out_ch, in_ch, 3, 3]
    pub conv_bias: Tensor<B, 1>,   // [out_ch]
}

impl<B: Backend> Downsample2D<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, device: &B::Device) -> Self {
        Self {
            conv_weight: loader.load_tensor(&format!("{prefix}.conv.weight"), device),
            conv_bias: loader.load_tensor(&format!("{prefix}.conv.bias"), device),
        }
    }

    /// Forward: asymmetric pad (0,1,0,1) then Conv2d stride=2.
    /// Input: [B, C, H, W] -> Output: [B, C, H/2, W/2]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let device = x.device();

        // Asymmetric padding: add 1 pixel on right and bottom (matches diffusers)
        let right_pad = Tensor::zeros([b, c, h, 1], &device);
        let x = Tensor::cat(vec![x, right_pad], 3); // [B, C, H, W+1]
        let bottom_pad = Tensor::zeros([b, c, 1, w + 1], &device);
        let x = Tensor::cat(vec![x, bottom_pad], 2); // [B, C, H+1, W+1]

        // Conv2d with stride=2, no additional padding
        conv2d_forward(
            x,
            self.conv_weight.clone(),
            Some(self.conv_bias.clone()),
            [2, 2], // stride
            [0, 0], // padding
        )
    }
}

/// A down-encoder block: multiple ResNet blocks followed by an optional downsampler.
///
/// Mirrors `UpDecoderBlock2D` from `up_block.rs`.
///
/// Block configuration (from VAE config block_out_channels = [128, 256, 512, 512]):
///   down_block 0: 128→128, 2 resnets, has downsample
///   down_block 1: 128→256, 2 resnets (first has conv_shortcut), has downsample
///   down_block 2: 256→512, 2 resnets (first has conv_shortcut), has downsample
///   down_block 3: 512→512, 2 resnets, NO downsample
pub struct DownEncoderBlock2D<B: Backend> {
    pub resnets: Vec<ResnetBlock2D<B>>,
    pub downsample: Option<Downsample2D<B>>,
}

impl<B: Backend> DownEncoderBlock2D<B> {
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        add_downsample: bool,
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

        let downsample = if add_downsample {
            let ds_prefix = format!("{prefix}.downsamplers.0");
            Some(Downsample2D::load(loader, &ds_prefix, device))
        } else {
            None
        };

        Self {
            resnets,
            downsample,
        }
    }

    /// Forward: run through all ResNet blocks, then optionally downsample.
    /// Input: [B, C_in, H, W] -> Output: [B, C_out, H', W']
    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for resnet in &self.resnets {
            x = resnet.forward(x);
        }

        if let Some(ref downsample) = self.downsample {
            x = downsample.forward(x);
        }

        x
    }
}
