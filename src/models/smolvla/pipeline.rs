//! The assembled SmolVLA: perception (SigLIP + VLM decoder → prefix KV) and
//! action (flow-matching expert → action chunk). This is the reusable entry
//! point — the embodied loop calls `perceive` once per observation and `act`
//! to roll out a chunk; the pile round-trip test exercises the same path.

use burn::prelude::*;

use super::config::SmolVlaConfig;
use super::denoiser::ExpertDenoiser;
use super::sampler::sample_actions;
use super::suffix::embed_suffix;
use super::vision::VisionEncoder;
use super::vlm::VlmTower;
use crate::models::smolvla::projections::Projections;
use crate::nn::weight_loader::WeightLoader;
use burn::tensor::TensorData;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

/// SmolVLA image preprocessing (`prepare_images`): resize-with-pad to
/// `target×target` keeping aspect ratio (pad on left/top with 0), then map
/// pixel range [0,1] → [-1,1] for SigLIP. `img: [B,C,H,W]` in [0,1].
pub fn preprocess_image<B: Backend>(img: Tensor<B, 4>, target: usize) -> Tensor<B, 4> {
    let [_, _, h, w] = img.dims();
    let ratio = (w as f64 / target as f64).max(h as f64 / target as f64);
    let (rh, rw) = ((h as f64 / ratio) as usize, (w as f64 / ratio) as usize);
    let resized = if rh == h && rw == w {
        img
    } else {
        interpolate(
            img,
            [rh, rw],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        )
    };
    // pad left/top to target (right/bottom stay 0)
    let padded = if rh < target || rw < target {
        resized.pad((target - rw, 0, target - rh, 0), 0.0.elem::<B::FloatElem>())
    } else {
        resized
    };
    padded.mul_scalar(2.0).sub_scalar(1.0)
}

/// Prefix attention (SmolVLA `embed_prefix` + `make_att_2d_masks`): image and
/// language tokens attend each other bidirectionally; the trailing state token
/// attends everything. Returns `(positions [1,Lp], mask [1,Lp,Lp])`.
fn prefix_masks<B: Backend>(
    n_img: usize,
    n_lang: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 3, Bool>) {
    let lp = n_img + n_lang + 1;
    let state = lp - 1; // the single state token, the only `att_mask=1` entry
    let mut m = vec![0f32; lp * lp];
    for i in 0..lp {
        for j in 0..lp {
            // make_att_2d_masks with att=[0]*（lp-1)+[1]: attend image+lang block
            // always; the state column only by the state row.
            if j < state || i == state {
                m[i * lp + j] = 1.0;
            }
        }
    }
    let pos: Vec<f32> = (0..lp).map(|i| i as f32).collect();
    (
        Tensor::from_data(TensorData::new(pos, [1, lp]), device),
        Tensor::<B, 3>::from_data(TensorData::new(m, [1, lp, lp]), device).greater_elem(0.5),
    )
}

/// Denoise attention: the action chunk attends the full prefix, and is causal
/// among itself. Returns `(positions [1,chunk], mask [1,chunk,Lp+chunk])`,
/// positions continuing the prefix.
fn denoise_masks<B: Backend>(
    lp: usize,
    chunk: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 3, Bool>) {
    let total = lp + chunk;
    let mut m = vec![0f32; chunk * total];
    for i in 0..chunk {
        for j in 0..lp {
            m[i * total + j] = 1.0; // attend all prefix
        }
        for j in 0..=i {
            m[i * total + lp + j] = 1.0; // causal over the chunk
        }
    }
    let pos: Vec<f32> = (0..chunk).map(|i| (lp + i) as f32).collect();
    (
        Tensor::from_data(TensorData::new(pos, [1, chunk]), device),
        Tensor::<B, 3>::from_data(TensorData::new(m, [1, chunk, total]), device).greater_elem(0.5),
    )
}

pub struct SmolVla<B: Backend> {
    pub vision: VisionEncoder<B>,
    pub vlm: VlmTower<B>,
    pub denoiser: ExpertDenoiser<B>,
    pub proj: Projections<B>,
    pub cfg: SmolVlaConfig,
}

impl<B: Backend> SmolVla<B> {
    pub fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let cfg = SmolVlaConfig::smolvla_base();
        Self {
            vision: VisionEncoder::load(loader, &cfg, device),
            vlm: VlmTower::load(loader, &cfg, device),
            denoiser: ExpertDenoiser::load(loader, &cfg, device),
            proj: Projections::load(loader, "model", device),
            cfg,
        }
    }

    /// Perceive: image `[B,3,512,512]` + language ids `[B,Lt]` + state `[B,1,32]`
    /// → the per-layer prefix KV cache `(k, v)` the action expert attends. The
    /// prefix positions and block-attention mask are computed internally.
    pub fn perceive(
        &self,
        image: Tensor<B, 4>,
        lang_ids: Tensor<B, 2, Int>,
        state: Tensor<B, 3>,
        device: &B::Device,
    ) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let s = (self.cfg.vlm.width as f64).sqrt();
        let img = self.vision.embed_image(image).mul_scalar(s);
        let lang = self.vlm.embed_language_tokens(lang_ids).mul_scalar(s);
        let (n_img, n_lang) = (img.dims()[1], lang.dims()[1]);
        let st = self.proj.state_proj.forward(state);
        let prefix = Tensor::cat(vec![img, lang, st], 1);
        let (pos, mask) = prefix_masks::<B>(n_img, n_lang, device);
        self.vlm.forward_decoder(prefix, pos, mask, device)
    }

    /// Act: roll out an action chunk by flow-matching over the prefix caches.
    /// `noise: [B,chunk,32]`. Positions + masks are computed internally.
    pub fn act(
        &self,
        caches_k: Tensor<B, 5>,
        caches_v: Tensor<B, 5>,
        noise: Tensor<B, 3>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let lp = caches_k.dims()[2];
        let chunk = noise.dims()[1];
        let (suffix_positions, suffix_mask) = denoise_masks::<B>(lp, chunk, device);
        let cross_mask = suffix_mask
            .clone()
            .float()
            .narrow(2, 0, lp)
            .greater_elem(0.5);
        let denoise = |x_t: Tensor<B, 3>, t: f64| {
            let tt = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(vec![t as f32], vec![1]),
                device,
            );
            let sfx = embed_suffix::<B>(
                &self.proj,
                &self.cfg,
                self.cfg.min_period,
                self.cfg.max_period,
                x_t,
                tt,
                device,
            );
            let e = self.denoiser.forward(
                sfx,
                suffix_positions.clone(),
                caches_k.clone(),
                caches_v.clone(),
                suffix_mask.clone(),
                cross_mask.clone(),
                device,
            );
            self.proj.action_out_proj.forward(e)
        };
        sample_actions(noise, self.cfg.num_steps, denoise)
    }
}
