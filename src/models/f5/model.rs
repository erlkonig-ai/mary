//! The full F5 flow-matching DiT (see docs/F5_ARCH.md).
//!
//! forward(noised_mel, cond_mel, text_ids, time) → velocity field over the mel.
//! Assembles: input embed (proj 712→dim + grouped conv positional embed),
//! sinusoidal time MLP, the ConvNeXt-V2 text encoder, `depth` AdaLN-zero DiT
//! blocks, a final AdaLN, and proj_out → n_mel.

use super::config::F5Config;
use super::dit::{f5_rope, F5Block, Linear};
use super::text::TextEmbed;
use crate::nn::norm::layer_norm_no_affine;
use crate::nn::weight_loader::WeightLoader;
use burn::prelude::*;
use burn::tensor::activation::silu;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

fn mish<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let softplus = (x.clone().exp() + 1.0).log();
    x * softplus.tanh()
}

pub struct F5Transformer<B: Backend> {
    cfg: F5Config,
    proj: Linear<B>, // 712 → dim
    pos_w1: Tensor<B, 3>,
    pos_b1: Tensor<B, 1>,
    pos_w2: Tensor<B, 3>,
    pos_b2: Tensor<B, 1>,
    pos_groups: usize,
    time_mlp0: Linear<B>, // 256 → dim
    time_mlp2: Linear<B>, // dim → dim
    text: TextEmbed<B>,
    blocks: Vec<F5Block<B>>,
    norm_out: Linear<B>, // dim → 2·dim
    proj_out: Linear<B>, // dim → n_mel
    inv_freq: Tensor<B, 1>,
}

impl<B: Backend> F5Transformer<B> {
    pub fn load(loader: &WeightLoader, cfg: F5Config, device: &B::Device) -> Self {
        let t = "ema_model.transformer";
        let pos_w1: Tensor<B, 3> = loader.load_tensor(&format!("{t}.input_embed.conv_pos_embed.conv1d.0.weight"), device);
        let pos_groups = cfg.dim / pos_w1.dims()[1];
        let blocks = (0..cfg.depth)
            .map(|i| F5Block::load(loader, i, cfg.heads, cfg.head_dim(), device))
            .collect();
        Self {
            proj: Linear::load(loader, &format!("{t}.input_embed.proj"), true, device),
            pos_b1: loader.load_tensor(&format!("{t}.input_embed.conv_pos_embed.conv1d.0.bias"), device),
            pos_w2: loader.load_tensor(&format!("{t}.input_embed.conv_pos_embed.conv1d.2.weight"), device),
            pos_b2: loader.load_tensor(&format!("{t}.input_embed.conv_pos_embed.conv1d.2.bias"), device),
            pos_w1,
            pos_groups,
            time_mlp0: Linear::load(loader, &format!("{t}.time_embed.time_mlp.0"), true, device),
            time_mlp2: Linear::load(loader, &format!("{t}.time_embed.time_mlp.2"), true, device),
            text: TextEmbed::load(loader, cfg.conv_layers, cfg.text_dim, device),
            blocks,
            norm_out: Linear::load(loader, &format!("{t}.norm_out.linear"), true, device),
            proj_out: Linear::load(loader, &format!("{t}.proj_out"), true, device),
            inv_freq: loader.load_tensor(&format!("{t}.rotary_embed.inv_freq"), device),
            cfg,
        }
    }

    /// Grouped conv positional embedding (residual on the caller's side).
    fn pos_conv(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let pad = self.pos_w1.dims()[2] / 2;
        let xc = x.swap_dims(1, 2); // [B, dim, T]
        let h = conv1d(xc, self.pos_w1.clone(), Some(self.pos_b1.clone()), ConvOptions::new([1], [pad], [1], self.pos_groups));
        let h = mish(h);
        let h = conv1d(h, self.pos_w2.clone(), Some(self.pos_b2.clone()), ConvOptions::new([1], [pad], [1], self.pos_groups));
        let h = mish(h); // F5 has a Mish after the 2nd conv too
        h.swap_dims(1, 2)
    }

    fn time_embed(&self, time: Tensor<B, 1>, device: &B::Device) -> Tensor<B, 2> {
        // F5 SinusPositionEmbedding(dim=256, scale=1000): freqs = exp(arange(half)
        // · −log(10000)/(half−1)); emb = scale·t·freqs; cat(sin, cos).
        let half = 128usize;
        let b = time.dims()[0];
        let logc = (10000f64).ln();
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-logc * i as f64 / (half as f64 - 1.0)).exp() as f32)
            .collect();
        let inv = Tensor::<B, 1>::from_floats(freqs.as_slice(), device).reshape([1, half]);
        let emb = (time.reshape([b, 1]) * 1000.0) * inv; // [B, half]
        let sin = Tensor::cat(vec![emb.clone().sin(), emb.cos()], 1).unsqueeze_dim::<3>(1); // [B,1,256]
        let h = silu(self.time_mlp0.forward(sin));
        let h = self.time_mlp2.forward(h); // [B,1,dim]
        h.squeeze_dim::<2>(1)
    }

    /// noised/cond: [B,T,n_mel]; text_ids: [B,T]; time: [B] → velocity [B,T,n_mel].
    pub fn forward(
        &self,
        noised: Tensor<B, 3>,
        cond: Tensor<B, 3>,
        text_ids: Tensor<B, 2, Int>,
        time: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        self.forward_cfg(noised, cond, text_ids, time, false, false)
    }

    /// As `forward`, with F5's CFG drop flags: `drop_audio_cond` zeros the cond
    /// mel, `drop_text` replaces text with filler. The uncond pass sets both.
    pub fn forward_cfg(
        &self,
        noised: Tensor<B, 3>,
        cond: Tensor<B, 3>,
        text_ids: Tensor<B, 2, Int>,
        time: Tensor<B, 1>,
        drop_audio_cond: bool,
        drop_text: bool,
    ) -> Tensor<B, 3> {
        let device = noised.device();
        let [b, s, _] = noised.dims();
        let cond = if drop_audio_cond { cond.zeros_like() } else { cond };
        // curtail/pad text to the mel length with filler (−1 → +1 = 0), as F5 does
        let n = text_ids.dims()[1];
        let text_ids = if n < s {
            Tensor::cat(vec![text_ids, Tensor::<B, 2, Int>::zeros([b, s - n], &device) - 1], 1)
        } else if n > s {
            text_ids.slice([0..b, 0..s])
        } else {
            text_ids
        };
        let text = self.text.forward(text_ids, drop_text); // [B,T,text_dim]
        let x = Tensor::cat(vec![noised, cond, text], 2); // [B,T,712]
        let x = self.proj.forward(x); // [B,T,dim]
        let x = self.pos_conv(x.clone()) + x; // residual conv-pos
        let t = self.time_embed(time, &device); // [B,dim]
        let (cos, sin) = f5_rope(s, self.inv_freq.clone(), &device);

        let mut h = x;
        for block in &self.blocks {
            h = block.forward(h, t.clone(), cos.clone(), sin.clone());
        }

        // final AdaLN: (scale, shift) = norm_out.linear(silu(t)) — scale first
        let m = self.norm_out.forward(silu(t).unsqueeze_dim::<3>(1)); // [B,1,2·dim]
        let dim = self.cfg.dim;
        let scale = m.clone().slice([0..b, 0..1, 0..dim]);
        let shift = m.slice([0..b, 0..1, dim..2 * dim]);
        let h = layer_norm_no_affine(h, 1e-6) * (scale + 1.0) + shift;
        self.proj_out.forward(h) // [B,T,n_mel]
    }

    /// Like `forward`, but also returns named intermediate activations as
    /// (name, flat data, shape) for numerical-parity probing against the
    /// reference F5-TTS. Tap points mirror `scripts/probe_f5.py`'s hooks.
    #[allow(clippy::type_complexity)]
    pub fn forward_probed(
        &self,
        noised: Tensor<B, 3>,
        cond: Tensor<B, 3>,
        text_ids: Tensor<B, 2, Int>,
        time: Tensor<B, 1>,
    ) -> (Tensor<B, 3>, Vec<(String, Vec<f32>, Vec<usize>)>) {
        let device = noised.device();
        let [b, s, _] = noised.dims();
        let mut p: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        fn tap<B: Backend, const D: usize>(
            p: &mut Vec<(String, Vec<f32>, Vec<usize>)>,
            name: &str,
            t: &Tensor<B, D>,
        ) {
            let data = t.clone().into_data();
            let shape = data.shape.to_vec();
            p.push((name.to_string(), data.to_vec::<f32>().unwrap(), shape));
        }

        let text = self.text.forward(text_ids, false);
        tap(&mut p, "text_embed", &text);
        let x = Tensor::cat(vec![noised, cond, text], 2);
        let x = self.proj.forward(x);
        let x = self.pos_conv(x.clone()) + x;
        tap(&mut p, "input_embed", &x);
        let t = self.time_embed(time, &device);
        tap(&mut p, "time_embed", &t);
        let (cos, sin) = f5_rope(s, self.inv_freq.clone(), &device);

        let mut h = x;
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(h, t.clone(), cos.clone(), sin.clone());
            if i == 0 {
                tap(&mut p, "block0", &h);
            }
        }
        tap(&mut p, "block21", &h);

        let m = self.norm_out.forward(silu(t).unsqueeze_dim::<3>(1));
        let dim = self.cfg.dim;
        let scale = m.clone().slice([0..b, 0..1, 0..dim]);
        let shift = m.slice([0..b, 0..1, dim..2 * dim]);
        let h = layer_norm_no_affine(h, 1e-6) * (scale + 1.0) + shift;
        tap(&mut p, "norm_out", &h);
        let out = self.proj_out.forward(h);
        tap(&mut p, "output", &out);
        (out, p)
    }
}
