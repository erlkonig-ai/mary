//! The action expert as a whole: 16 decoder layers (self/cross interleaved) +
//! final RMSNorm. This is `denoise_step` minus the VLM — it consumes the
//! per-layer prefix KV cache the (frozen) VLM tower produces, runs the suffix
//! action-tokens through the stack, and the caller projects the result with
//! `action_out_proj` to the flow velocity `v_t`.
//!
//! Dispatch: `self_attn_every_n_layers = 2` → even layers self-attend (and
//! reach the prefix cache), odd layers cross-attend the reprojected VLM KV.

use burn::prelude::*;

use super::config::SmolVlaConfig;
use super::layers::{ExpertLayer, RmsNorm};
use crate::nn::weight_loader::WeightLoader;

pub struct ExpertDenoiser<B: Backend> {
    layers: Vec<ExpertLayer<B>>,
    norm: RmsNorm<B>,
}

impl<B: Backend> ExpertDenoiser<B> {
    pub fn load(loader: &WeightLoader, cfg: &SmolVlaConfig, device: &B::Device) -> Self {
        let prefix = "model.vlm_with_expert.lm_expert";
        let layers = (0..cfg.expert.n_layers)
            .map(|i| ExpertLayer::load(loader, &format!("{prefix}.layers.{i}"), cfg.expert, device))
            .collect();
        let norm = RmsNorm::load(loader, &format!("{prefix}.norm.weight"), cfg.expert.rms_norm_eps, device);
        Self { layers, norm }
    }

    /// `suffix: [B,L,width]` action tokens (from `embed_suffix`).
    /// `positions: [B,L]` the suffix positions (continuing the prefix).
    /// `caches_k/v: [n_layers, B, Lp, Hkv, Dh]` the per-layer VLM prefix cache.
    /// `self_mask: [B,L,Lp+L]`, `cross_mask: [B,L,Lp]`.
    /// Returns the expert output `[B,L,width]` (post final-norm).
    pub fn forward(
        &self,
        suffix: Tensor<B, 3>,
        positions: Tensor<B, 2>,
        caches_k: Tensor<B, 5>,
        caches_v: Tensor<B, 5>,
        self_mask: Tensor<B, 3, Bool>,
        cross_mask: Tensor<B, 3, Bool>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        // cross-attn queries use positions renormalized to start at 0
        let mn = positions.clone().min().into_scalar().elem::<f64>();
        let qpos = positions.clone().sub_scalar(mn);

        let [_, b, lp, hkv, dh] = caches_k.dims();
        let layer_cache = |c: &Tensor<B, 5>, i: usize| c.clone().narrow(0, i, 1).reshape([b, lp, hkv, dh]);

        let mut x = suffix;
        for (i, layer) in self.layers.iter().enumerate() {
            let pk = layer_cache(&caches_k, i);
            let pv = layer_cache(&caches_v, i);
            x = if i % 2 == 0 {
                layer.forward(x, positions.clone(), pk, pv, self_mask.clone(), device)
            } else {
                layer.forward_cross(x, qpos.clone(), pk, pv, cross_mask.clone(), device)
            };
        }
        self.norm.forward(x)
    }
}
