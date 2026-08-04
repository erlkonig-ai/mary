use burn::prelude::*;
use burn::tensor::activation::silu;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;

use crate::nn::weight_loader::WeightLoader;

/// Group normalization for 4D tensors [B, C, H, W].
/// Applies: y = (x - mean) / sqrt(var + eps) * weight + bias
/// where mean/var are computed over groups of channels and spatial dims.
pub fn group_norm_4d<B: Backend>(
    x: Tensor<B, 4>,
    num_groups: usize,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.dims();
    let channels_per_group = channels / num_groups;

    // Reshape to [B, G, C/G, H, W]
    let x = x.reshape([batch, num_groups, channels_per_group, height, width]);

    // Compute mean and variance over (C/G, H, W) dimensions
    let elements = (channels_per_group * height * width) as f64;
    let mean = x.clone().sum_dim(2).sum_dim(3).sum_dim(4) / elements; // [B, G, 1, 1, 1]
    let x_centered = x - mean;
    let var = x_centered
        .clone()
        .powf_scalar(2.0)
        .sum_dim(2)
        .sum_dim(3)
        .sum_dim(4)
        / elements;
    let inv_std = (var + eps).sqrt().recip(); // [B, G, 1, 1, 1]
    let normed = x_centered * inv_std;

    // Reshape back to [B, C, H, W]
    let normed = normed.reshape([batch, channels, height, width]);

    // Apply affine: weight [C] -> [1, C, 1, 1], bias [C] -> [1, C, 1, 1]
    let weight = weight.reshape([1, channels, 1, 1]);
    let bias = bias.reshape([1, channels, 1, 1]);
    normed * weight + bias
}

/// Convenience wrapper around burn's conv2d with padding and stride.
pub fn conv2d_forward<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Tensor<B, 4> {
    let options = ConvOptions::new(stride, padding, [1, 1], 1);
    conv2d(input, weight, bias, options)
}

/// A ResNet block used in the VAE decoder.
/// Structure: GroupNorm -> SiLU -> Conv2d -> GroupNorm -> SiLU -> Conv2d + skip connection
pub struct ResnetBlock2D<B: Backend> {
    // First normalization + convolution
    pub norm1_weight: Tensor<B, 1>,
    pub norm1_bias: Tensor<B, 1>,
    pub conv1_weight: Tensor<B, 4>,
    pub conv1_bias: Tensor<B, 1>,

    // Second normalization + convolution
    pub norm2_weight: Tensor<B, 1>,
    pub norm2_bias: Tensor<B, 1>,
    pub conv2_weight: Tensor<B, 4>,
    pub conv2_bias: Tensor<B, 1>,

    // Optional skip connection (1x1 conv when in_ch != out_ch)
    pub conv_shortcut_weight: Option<Tensor<B, 4>>,
    pub conv_shortcut_bias: Option<Tensor<B, 1>>,

    pub num_groups: usize,
}

impl<B: Backend> ResnetBlock2D<B> {
    /// Load a ResnetBlock2D from safetensors with a given weight name prefix.
    /// Example prefix: "decoder.mid_block.resnets.0"
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        num_groups: usize,
        device: &B::Device,
    ) -> Self {
        let norm1_weight = loader.load_tensor::<B, 1>(&format!("{prefix}.norm1.weight"), device);
        let norm1_bias = loader.load_tensor::<B, 1>(&format!("{prefix}.norm1.bias"), device);
        let conv1_weight = loader.load_tensor::<B, 4>(&format!("{prefix}.conv1.weight"), device);
        let conv1_bias = loader.load_tensor::<B, 1>(&format!("{prefix}.conv1.bias"), device);

        let norm2_weight = loader.load_tensor::<B, 1>(&format!("{prefix}.norm2.weight"), device);
        let norm2_bias = loader.load_tensor::<B, 1>(&format!("{prefix}.norm2.bias"), device);
        let conv2_weight = loader.load_tensor::<B, 4>(&format!("{prefix}.conv2.weight"), device);
        let conv2_bias = loader.load_tensor::<B, 1>(&format!("{prefix}.conv2.bias"), device);

        let (conv_shortcut_weight, conv_shortcut_bias) = if in_channels != out_channels {
            let w = loader.load_tensor::<B, 4>(&format!("{prefix}.conv_shortcut.weight"), device);
            let b = loader.load_tensor::<B, 1>(&format!("{prefix}.conv_shortcut.bias"), device);
            (Some(w), Some(b))
        } else {
            (None, None)
        };

        Self {
            norm1_weight,
            norm1_bias,
            conv1_weight,
            conv1_bias,
            norm2_weight,
            norm2_bias,
            conv2_weight,
            conv2_bias,
            conv_shortcut_weight,
            conv_shortcut_bias,
            num_groups,
        }
    }

    /// Forward pass: GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 + skip
    /// Input: [B, C_in, H, W] -> Output: [B, C_out, H, W]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let eps = 1e-6;

        // First block: norm1 -> silu -> conv1
        let h = group_norm_4d(
            x.clone(),
            self.num_groups,
            self.norm1_weight.clone(),
            self.norm1_bias.clone(),
            eps,
        );
        let h = silu(h);
        let h = conv2d_forward(
            h,
            self.conv1_weight.clone(),
            Some(self.conv1_bias.clone()),
            [1, 1],
            [1, 1],
        );

        // Second block: norm2 -> silu -> conv2
        let h = group_norm_4d(
            h,
            self.num_groups,
            self.norm2_weight.clone(),
            self.norm2_bias.clone(),
            eps,
        );
        let h = silu(h);
        let h = conv2d_forward(
            h,
            self.conv2_weight.clone(),
            Some(self.conv2_bias.clone()),
            [1, 1],
            [1, 1],
        );

        // Skip connection
        let skip = match (&self.conv_shortcut_weight, &self.conv_shortcut_bias) {
            (Some(w), Some(b)) => conv2d_forward(x, w.clone(), Some(b.clone()), [1, 1], [0, 0]),
            _ => x,
        };

        skip + h
    }
}
