use burn::prelude::*;
use burn::tensor::activation::silu;

use super::attention::VaeAttention;
use super::config::VaeConfig;
use super::resnet::{conv2d_forward, group_norm_4d, ResnetBlock2D};
use super::up_block::UpDecoderBlock2D;
use crate::nn::weight_loader::WeightLoader;

/// The VAE Decoder.
///
/// Architecture:
///   1. conv_in: Conv2d(latent_channels=32, block_out_channels[-1]=512, 3, pad=1)
///   2. mid_block: ResnetBlock2D(512) -> Attention(512) -> ResnetBlock2D(512)
///   3. up_blocks: 4x UpDecoderBlock2D (in reverse channel order)
///   4. conv_norm_out: GroupNorm(32, 128)
///   5. conv_out: Conv2d(128, 3, 3, pad=1)
///
/// Forward: conv_in -> mid_block -> up_blocks -> GroupNorm -> SiLU -> conv_out
pub struct Decoder<B: Backend> {
    // Input convolution: latent_channels -> block_out_channels[-1]
    pub conv_in_weight: Tensor<B, 4>,
    pub conv_in_bias: Tensor<B, 1>,

    // Mid block: resnet0 -> attention -> resnet1
    pub mid_resnet0: ResnetBlock2D<B>,
    pub mid_attention: VaeAttention<B>,
    pub mid_resnet1: ResnetBlock2D<B>,

    // Up blocks (4 of them)
    pub up_blocks: Vec<UpDecoderBlock2D<B>>,

    // Output normalization and convolution
    pub conv_norm_out_weight: Tensor<B, 1>,
    pub conv_norm_out_bias: Tensor<B, 1>,
    pub conv_out_weight: Tensor<B, 4>,
    pub conv_out_bias: Tensor<B, 1>,

    pub norm_num_groups: usize,
}

impl<B: Backend> Decoder<B> {
    /// Load the decoder from a single safetensors file.
    pub fn load(
        loader: &WeightLoader,
        config: &VaeConfig,
        device: &B::Device,
    ) -> Self {
        let num_groups = config.norm_num_groups;
        let block_out_channels = &config.block_out_channels;
        let last_ch = *block_out_channels.last().unwrap(); // 512

        // conv_in: latent_channels -> last_ch
        let conv_in_weight =
            loader.load_tensor::<B, 4>("decoder.conv_in.weight", device);
        let conv_in_bias =
            loader.load_tensor::<B, 1>("decoder.conv_in.bias", device);

        // Mid block: 2 resnets + 1 attention, all operating at last_ch=512
        let mid_resnet0 = ResnetBlock2D::load(
            loader,
            "decoder.mid_block.resnets.0",
            last_ch,
            last_ch,
            num_groups,
            device,
        );
        let mid_attention = VaeAttention::load(
            loader,
            "decoder.mid_block.attentions.0",
            last_ch,
            num_groups,
            device,
        );
        let mid_resnet1 = ResnetBlock2D::load(
            loader,
            "decoder.mid_block.resnets.1",
            last_ch,
            last_ch,
            num_groups,
            device,
        );

        // Up blocks: process in reversed channel order
        // block_out_channels = [128, 256, 512, 512]
        // reversed = [512, 512, 256, 128]
        //
        // up_block 0: prev=512, out=512, has upsample
        // up_block 1: prev=512, out=512, has upsample
        // up_block 2: prev=512, out=256, has upsample
        // up_block 3: prev=256, out=128, NO upsample
        let reversed: Vec<usize> = block_out_channels.iter().copied().rev().collect();
        let num_up_blocks = reversed.len();
        let num_layers = config.layers_per_block + 1; // 3

        let mut up_blocks = Vec::with_capacity(num_up_blocks);
        let mut output_channel = reversed[0]; // 512

        for i in 0..num_up_blocks {
            let prev_output_channel = output_channel;
            output_channel = reversed[i];
            let is_final_block = i == num_up_blocks - 1;
            let add_upsample = !is_final_block;

            let prefix = format!("decoder.up_blocks.{i}");
            up_blocks.push(UpDecoderBlock2D::load(
                loader,
                &prefix,
                prev_output_channel,
                output_channel,
                num_layers,
                add_upsample,
                num_groups,
                device,
            ));
        }

        // Output normalization and convolution
        let conv_norm_out_weight =
            loader.load_tensor::<B, 1>("decoder.conv_norm_out.weight", device);
        let conv_norm_out_bias =
            loader.load_tensor::<B, 1>("decoder.conv_norm_out.bias", device);
        let conv_out_weight =
            loader.load_tensor::<B, 4>("decoder.conv_out.weight", device);
        let conv_out_bias =
            loader.load_tensor::<B, 1>("decoder.conv_out.bias", device);

        Self {
            conv_in_weight,
            conv_in_bias,
            mid_resnet0,
            mid_attention,
            mid_resnet1,
            up_blocks,
            conv_norm_out_weight,
            conv_norm_out_bias,
            conv_out_weight,
            conv_out_bias,
            norm_num_groups: num_groups,
        }
    }

    /// Forward pass: conv_in -> mid_block -> up_blocks -> GroupNorm -> SiLU -> conv_out
    /// Input: [B, latent_channels, H, W] -> Output: [B, 3, H', W']
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // conv_in
        let mut h = conv2d_forward(
            x,
            self.conv_in_weight.clone(),
            Some(self.conv_in_bias.clone()),
            [1, 1],
            [1, 1],
        );

        // Mid block: resnet -> attention -> resnet
        h = self.mid_resnet0.forward(h);
        h = self.mid_attention.forward(h);
        h = self.mid_resnet1.forward(h);

        // Up blocks
        for up_block in &self.up_blocks {
            h = up_block.forward(h);
        }

        // Post-process: GroupNorm -> SiLU -> conv_out
        h = group_norm_4d(
            h,
            self.norm_num_groups,
            self.conv_norm_out_weight.clone(),
            self.conv_norm_out_bias.clone(),
            1e-6,
        );
        h = silu(h);
        h = conv2d_forward(
            h,
            self.conv_out_weight.clone(),
            Some(self.conv_out_bias.clone()),
            [1, 1],
            [1, 1],
        );

        h
    }
}
