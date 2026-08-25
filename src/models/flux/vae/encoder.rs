use burn::prelude::*;
use burn::tensor::activation::silu;

use super::attention::VaeAttention;
use super::config::VaeConfig;
use super::down_block::DownEncoderBlock2D;
use super::resnet::{ResnetBlock2D, conv2d_forward, group_norm_4d};
use crate::nn::weight_loader::WeightLoader;

/// The VAE Encoder (mirrors the Decoder).
///
/// Architecture:
///   1. conv_in: Conv2d(3, block_out_channels[0]=128, 3, pad=1)
///   2. down_blocks: 4x DownEncoderBlock2D (in forward channel order)
///   3. mid_block: ResnetBlock2D(512) -> Attention(512) -> ResnetBlock2D(512)
///   4. conv_norm_out: GroupNorm(32, 512)
///   5. conv_out: Conv2d(512, 2*latent_channels=64, 3, pad=1)
///
/// Forward: conv_in -> down_blocks -> mid_block -> GroupNorm -> SiLU -> conv_out
pub struct Encoder<B: Backend> {
    // Input convolution: in_channels(3) -> block_out_channels[0](128)
    pub conv_in_weight: Tensor<B, 4>,
    pub conv_in_bias: Tensor<B, 1>,

    // Down blocks (4 of them)
    pub down_blocks: Vec<DownEncoderBlock2D<B>>,

    // Mid block: resnet0 -> attention -> resnet1
    pub mid_resnet0: ResnetBlock2D<B>,
    pub mid_attention: VaeAttention<B>,
    pub mid_resnet1: ResnetBlock2D<B>,

    // Output normalization and convolution
    pub conv_norm_out_weight: Tensor<B, 1>,
    pub conv_norm_out_bias: Tensor<B, 1>,
    pub conv_out_weight: Tensor<B, 4>,
    pub conv_out_bias: Tensor<B, 1>,

    pub norm_num_groups: usize,
}

impl<B: Backend> Encoder<B> {
    pub fn load(loader: &WeightLoader, config: &VaeConfig, device: &B::Device) -> Self {
        let num_groups = config.norm_num_groups;
        let block_out_channels = &config.block_out_channels; // [128, 256, 512, 512]
        let last_ch = *block_out_channels.last().unwrap(); // 512

        // conv_in: in_channels(3) -> block_out_channels[0](128)
        let conv_in_weight = loader.load_tensor::<B, 4>("encoder.conv_in.weight", device);
        let conv_in_bias = loader.load_tensor::<B, 1>("encoder.conv_in.bias", device);

        // Down blocks: process in forward channel order
        // block_out_channels = [128, 256, 512, 512]
        //
        // down_block 0: prev=128(from conv_in), out=128, has downsample
        // down_block 1: prev=128, out=256, has downsample
        // down_block 2: prev=256, out=512, has downsample
        // down_block 3: prev=512, out=512, NO downsample
        let num_down_blocks = block_out_channels.len();
        let num_layers = config.layers_per_block; // 2 (encoder uses layers_per_block, not +1)

        let mut down_blocks = Vec::with_capacity(num_down_blocks);
        let mut output_channel = block_out_channels[0]; // 128

        for i in 0..num_down_blocks {
            let input_channel = output_channel;
            output_channel = block_out_channels[i];
            let is_final_block = i == num_down_blocks - 1;
            let add_downsample = !is_final_block;

            let prefix = format!("encoder.down_blocks.{i}");
            down_blocks.push(DownEncoderBlock2D::load(
                loader,
                &prefix,
                input_channel,
                output_channel,
                num_layers,
                add_downsample,
                num_groups,
                device,
            ));
        }

        // Mid block: 2 resnets + 1 attention, all at last_ch=512
        let mid_resnet0 = ResnetBlock2D::load(
            loader,
            "encoder.mid_block.resnets.0",
            last_ch,
            last_ch,
            num_groups,
            device,
        );
        let mid_attention = VaeAttention::load(
            loader,
            "encoder.mid_block.attentions.0",
            last_ch,
            num_groups,
            device,
        );
        let mid_resnet1 = ResnetBlock2D::load(
            loader,
            "encoder.mid_block.resnets.1",
            last_ch,
            last_ch,
            num_groups,
            device,
        );

        // Output normalization and convolution
        let conv_norm_out_weight =
            loader.load_tensor::<B, 1>("encoder.conv_norm_out.weight", device);
        let conv_norm_out_bias = loader.load_tensor::<B, 1>("encoder.conv_norm_out.bias", device);
        let conv_out_weight = loader.load_tensor::<B, 4>("encoder.conv_out.weight", device);
        let conv_out_bias = loader.load_tensor::<B, 1>("encoder.conv_out.bias", device);

        Self {
            conv_in_weight,
            conv_in_bias,
            down_blocks,
            mid_resnet0,
            mid_attention,
            mid_resnet1,
            conv_norm_out_weight,
            conv_norm_out_bias,
            conv_out_weight,
            conv_out_bias,
            norm_num_groups: num_groups,
        }
    }

    /// Forward pass: conv_in -> down_blocks -> mid_block -> GroupNorm -> SiLU -> conv_out
    ///
    /// Input: [B, 3, H, W] -> Output: [B, 2*latent_channels, H/8, W/8]
    /// The output has 64 channels: first 32 = mean, last 32 = logvar.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // conv_in
        let mut h = conv2d_forward(
            x,
            self.conv_in_weight.clone(),
            Some(self.conv_in_bias.clone()),
            [1, 1],
            [1, 1],
        );

        // Down blocks
        for down_block in &self.down_blocks {
            h = down_block.forward(h);
        }

        // Mid block: resnet -> attention -> resnet
        h = self.mid_resnet0.forward(h);
        h = self.mid_attention.forward(h);
        h = self.mid_resnet1.forward(h);

        // Post-process: GroupNorm -> SiLU -> conv_out
        h = group_norm_4d(
            h,
            self.norm_num_groups,
            self.conv_norm_out_weight.clone(),
            self.conv_norm_out_bias.clone(),
            1e-6,
        );
        h = silu(h);
        conv2d_forward(
            h,
            self.conv_out_weight.clone(),
            Some(self.conv_out_bias.clone()),
            [1, 1],
            [1, 1],
        )
    }
}
