//! F5's text encoder: a character embedding refined by ConvNeXt-V2 blocks
//! (see docs/F5_ARCH.md). Small and F5-specific — not a big LLM encoder. The
//! output [B, T, text_dim] is concatenated with mel ⊕ cond_mel to condition the
//! DiT.
//!
//! Checkpoint keys: `ema_model.transformer.text_embed.{text_embed, text_blocks.*}`.

use super::dit::Linear;
use crate::nn::weight_loader::WeightLoader;
use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

/// Affine LayerNorm over the last dim.
fn layer_norm<B: Backend>(x: Tensor<B, 3>, w: &Tensor<B, 1>, b: &Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let c = x.dims()[2];
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    let norm = centered / (var + eps).sqrt();
    norm * w.clone().reshape([1, 1, c]) + b.clone().reshape([1, 1, c])
}

/// ConvNeXt-V2 Global Response Normalization over the channel dim (last),
/// aggregating across time (dim 1).
fn grn<B: Backend>(x: Tensor<B, 3>, gamma: &Tensor<B, 1>, beta: &Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let c = x.dims()[2];
    let gx = x.clone().powf_scalar(2.0).sum_dim(1).sqrt(); // [B,1,C] — L2 over time
    let nx = gx.clone() / (gx.mean_dim(2) + eps); // normalise by mean over channels → [B,1,C]
    x.clone() * (nx * gamma.clone().reshape([1, 1, c])) + beta.clone().reshape([1, 1, c]) + x
}

/// One ConvNeXt-V2 block: depthwise conv (k7) → LayerNorm → pw↑ → GELU → GRN →
/// pw↓, residual. Channels-last [B, T, C].
pub struct ConvNeXtBlock<B: Backend> {
    dwconv_w: Tensor<B, 3>, // [C, 1, 7]
    dwconv_b: Tensor<B, 1>,
    norm_w: Tensor<B, 1>,
    norm_b: Tensor<B, 1>,
    pw1: Linear<B>, // C → 2C
    grn_gamma: Tensor<B, 1>,
    grn_beta: Tensor<B, 1>,
    pw2: Linear<B>, // 2C → C
    channels: usize,
}

impl<B: Backend> ConvNeXtBlock<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, channels: usize, device: &B::Device) -> Self {
        let g3: Tensor<B, 3> = loader.load_tensor(&format!("{prefix}.grn.gamma"), device); // [1,1,C]
        let b3: Tensor<B, 3> = loader.load_tensor(&format!("{prefix}.grn.beta"), device);
        let (gc, bc) = (g3.dims()[2], b3.dims()[2]);
        Self {
            dwconv_w: loader.load_tensor(&format!("{prefix}.dwconv.weight"), device),
            dwconv_b: loader.load_tensor(&format!("{prefix}.dwconv.bias"), device),
            norm_w: loader.load_tensor(&format!("{prefix}.norm.weight"), device),
            norm_b: loader.load_tensor(&format!("{prefix}.norm.bias"), device),
            pw1: Linear::load(loader, &format!("{prefix}.pwconv1"), true, device),
            grn_gamma: g3.reshape([gc]),
            grn_beta: b3.reshape([bc]),
            pw2: Linear::load(loader, &format!("{prefix}.pwconv2"), true, device),
            channels,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        // depthwise conv: [B,T,C] → [B,C,T] → conv (k7, pad3, groups=C) → [B,T,C]
        let xc = x.swap_dims(1, 2);
        let conv = conv1d(
            xc,
            self.dwconv_w.clone(),
            Some(self.dwconv_b.clone()),
            ConvOptions::new([1], [3], [1], self.channels),
        );
        let x = conv.swap_dims(1, 2);
        let x = layer_norm(x, &self.norm_w, &self.norm_b, 1e-6);
        let x = gelu(self.pw1.forward(x));
        let x = grn(x, &self.grn_gamma, &self.grn_beta, 1e-6);
        let x = self.pw2.forward(x);
        residual + x
    }
}

/// Char embedding + a stack of ConvNeXt-V2 blocks → [B, T, text_dim].
pub struct TextEmbed<B: Backend> {
    embed: Tensor<B, 2>, // [vocab, text_dim]
    blocks: Vec<ConvNeXtBlock<B>>,
}

impl<B: Backend> TextEmbed<B> {
    pub fn load(loader: &WeightLoader, n_blocks: usize, text_dim: usize, device: &B::Device) -> Self {
        let base = "ema_model.transformer.text_embed";
        let blocks = (0..n_blocks)
            .map(|i| ConvNeXtBlock::load(loader, &format!("{base}.text_blocks.{i}"), text_dim, device))
            .collect();
        Self {
            embed: loader.load_tensor(&format!("{base}.text_embed.weight"), device),
            blocks,
        }
    }

    /// ids: [B, T] character ids → [B, T, text_dim]. `drop_text` (CFG uncond)
    /// replaces every token with the filler embedding but keeps the real text's
    /// padding mask — matching F5's `drop_text=True`.
    pub fn forward(&self, ids: Tensor<B, 2, Int>, drop_text: bool) -> Tensor<B, 3> {
        let [b, t] = ids.dims();
        let text_dim = self.embed.dims()[1];
        let device = ids.device();
        // F5 text_mask: padding positions (raw id −1) are zeroed before and after
        // each ConvNeXt block, so padding can't leak through the text convs into
        // the valid region. Derived from the REAL ids even when dropping text.
        let keep = (ids.clone().lower_elem(0).float() * -1.0 + 1.0).reshape([b, t, 1]);
        // +1: index 0 is the filler/pad token. drop_text → all filler.
        let eids = if drop_text {
            Tensor::<B, 2, Int>::zeros([b, t], &device) - 1
        } else {
            ids
        };
        let flat = (eids + 1).reshape([b * t]);
        let x = self.embed.clone().select(0, flat).reshape([b, t, text_dim]);
        // absolute sinusoidal pos-emb (non-persistent buffer; computed, not loaded).
        let freqs = freqs_cis::<B>(t, text_dim, &device).reshape([1, t, text_dim]);
        let mut x = (x + freqs) * keep.clone();
        for block in &self.blocks {
            x = block.forward(x) * keep.clone();
        }
        x
    }
}

/// F5's `precompute_freqs_cis(dim, end)` = cat([cos(outer(t,invfreq)),
/// sin(outer(t,invfreq))]); invfreq = 1/10000^(arange(0,dim,2)/dim). → [end, dim].
fn freqs_cis<B: Backend>(end: usize, dim: usize, device: &B::Device) -> Tensor<B, 2> {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (1.0 / 10000f64.powf((2 * i) as f64 / dim as f64)) as f32)
        .collect();
    let inv = Tensor::<B, 1>::from_floats(inv.as_slice(), device).reshape([1, half]);
    let pos = Tensor::<B, 1, Int>::arange(0..end as i64, device).float().reshape([end, 1]);
    let f = pos * inv; // [end, half]
    Tensor::cat(vec![f.clone().cos(), f.sin()], 1) // [end, dim]
}
