//! The action expert's transformer layer (SmolLM2-style decoder block, GQA),
//! and the eager GQA attention it shares with the VLM tower.
//!
//! A layer (reference `SmolVLMWithExpertModel.forward`, expert branch):
//!   h    = input_layernorm(x)
//!   q,k,v= proj(h)                       (q 720→960=15·64, k/v 720→320=5·64)
//!   q,k  = rope(q), rope(k)
//!   k,v  = cat([prefix_cache, k/v])      (self-attn layers still attend prefix)
//!   a    = eager_gqa_attention(mask, q, k, v)        → [B,L,960]
//!   x    = x + o_proj(a)                              (residual 1)
//!   x    = x + mlp(post_attention_layernorm(x))       (residual 2, SwiGLU)
//!
//! Attention projections are bias-free; MLP is gate/up/down SwiGLU. Layer 0 is
//! a self-attn layer (self_attn_every_n_layers=2) but still concatenates the
//! prefix KV cache, so the action tokens attend [prefix ++ suffix] under GQA.

use burn::prelude::*;
use burn::tensor::activation::{silu, softmax};

use super::config::TowerConfig;
use super::projections::Linear;
use super::rope::apply_rope;
use crate::nn::weight_loader::WeightLoader;

pub const ROPE_MAX_WAVELENGTH: f64 = 10_000.0;

/// Llama/SmolLM2 RMSNorm: `x · rsqrt(mean(x²) + eps) · weight`.
pub struct RmsNorm<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn load(loader: &WeightLoader, name: &str, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: loader.load_tensor(name, device),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let var = x.clone().powf_scalar(2.0).mean_dim(2);
        let normed = x.mul(var.add_scalar(self.eps).sqrt().recip());
        let w = self.weight.clone().reshape([1, 1, self.weight.dims()[0]]);
        normed.mul(w)
    }
}

/// Eager GQA attention, fp32. `q: [B,Lq,Hq,Dh]`, `k/v: [B,Lk,Hkv,Dh]`,
/// `mask: [B,Lq,Lk]` (bool, true = attend) → `[B,Lq,Hq·Dh]`.
pub fn eager_gqa_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Tensor<B, 3, Bool>,
) -> Tensor<B, 3> {
    let [b, lq, hq, dh] = q.dims();
    let [_, lk, hkv, _] = k.dims();
    let groups = hq / hkv;

    // repeat_kv: [B,Lk,Hkv,Dh] -> [B,Lk,Hkv,1,Dh] -> expand -> [B,Lk,Hq,Dh]
    let repeat = |t: Tensor<B, 4>| {
        t.reshape([b, lk, hkv, 1, dh])
            .expand([b, lk, hkv, groups, dh])
            .reshape([b, lk, hq, dh])
    };
    let k = repeat(k);
    let v = repeat(v);

    // -> [B,Hq,Lq,Dh] / [B,Hq,Lk,Dh]
    let q = q.swap_dims(1, 2);
    let k = k.swap_dims(1, 2);
    let v = v.swap_dims(1, 2);

    // scores = q·kᵀ · Dh^-0.5  -> [B,Hq,Lq,Lk]
    let scores = q
        .matmul(k.swap_dims(2, 3))
        .mul_scalar((dh as f64).powf(-0.5));

    // mask: [B,Lq,Lk] -> [B,Hq,Lq,Lk]; fill ¬mask with big_neg
    let mask4 = mask.reshape([b, 1, lq, lk]).expand([b, hq, lq, lk]);
    let scores = scores.mask_fill(mask4.bool_not(), f32::MIN);

    let probs = softmax(scores, 3);
    // probs·v -> [B,Hq,Lq,Dh] -> [B,Lq,Hq,Dh] -> [B,Lq,Hq·Dh]
    probs.matmul(v).swap_dims(1, 2).reshape([b, lq, hq * dh])
}

/// One action-expert decoder layer.
pub struct ExpertLayer<B: Backend> {
    pub input_layernorm: RmsNorm<B>,
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
    cfg: TowerConfig,
}

impl<B: Backend> ExpertLayer<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, cfg: TowerConfig, device: &B::Device) -> Self {
        let lin = |n: &str| Linear::load(loader, &format!("{prefix}.{n}"), false, device);
        Self {
            input_layernorm: RmsNorm::load(
                loader,
                &format!("{prefix}.input_layernorm.weight"),
                cfg.rms_norm_eps,
                device,
            ),
            q_proj: lin("self_attn.q_proj"),
            k_proj: lin("self_attn.k_proj"),
            v_proj: lin("self_attn.v_proj"),
            o_proj: lin("self_attn.o_proj"),
            post_attention_layernorm: RmsNorm::load(
                loader,
                &format!("{prefix}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
                device,
            ),
            gate_proj: lin("mlp.gate_proj"),
            up_proj: lin("mlp.up_proj"),
            down_proj: lin("mlp.down_proj"),
            cfg,
        }
    }

    /// Shared tail: o_proj + residual-1 + post-norm + SwiGLU MLP + residual-2.
    /// `x` is the (un-normed) layer input; `att` the attention output `[B,L,960]`.
    fn residual_mlp(&self, x: Tensor<B, 3>, att: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.add(self.o_proj.forward(att)); // residual 1
        let after = x.clone();
        let h = self.post_attention_layernorm.forward(x);
        let mlp = self
            .down_proj
            .forward(silu(self.gate_proj.forward(h.clone())).mul(self.up_proj.forward(h)));
        after.add(mlp) // residual 2
    }

    /// Self-attention layer (even index): the action tokens self-attend and also
    /// reach into the prefix KV cache. `x: [B,L,width]`, `positions: [B,L]`,
    /// prefix cache `[B,Lp,Hkv,Dh]`, `mask: [B,L,Lp+L]`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        positions: Tensor<B, 2>,
        prefix_k: Tensor<B, 4>,
        prefix_v: Tensor<B, 4>,
        mask: Tensor<B, 3, Bool>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, l, _] = x.dims();
        let (hq, hkv, dh) = (self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim);

        let h = self.input_layernorm.forward(x.clone());
        let q = self.q_proj.forward(h.clone()).reshape([b, l, hq, dh]);
        let k = self.k_proj.forward(h.clone()).reshape([b, l, hkv, dh]);
        let v = self.v_proj.forward(h).reshape([b, l, hkv, dh]);

        let q = apply_rope(q, positions.clone(), ROPE_MAX_WAVELENGTH, device);
        let k = apply_rope(k, positions, ROPE_MAX_WAVELENGTH, device);
        let k = Tensor::cat(vec![prefix_k, k], 1);
        let v = Tensor::cat(vec![prefix_v, v], 1);

        let att = eager_gqa_attention(q, k, v, mask);
        self.residual_mlp(x, att)
    }

    /// VLM tower layer: plain self-attention over the prefix sequence (no prior
    /// cache — this layer *produces* the cache). Returns `(out, k, v)` where
    /// `k` (RoPE'd) and `v` are stored as the per-layer prefix KV cache the
    /// action expert later attends. `mask: [B,L,L]` the prefix 2D mask.
    pub fn forward_vlm(
        &self,
        x: Tensor<B, 3>,
        positions: Tensor<B, 2>,
        mask: Tensor<B, 3, Bool>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let [b, l, _] = x.dims();
        let (hq, hkv, dh) = (self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim);

        let h = self.input_layernorm.forward(x.clone());
        let q = apply_rope(
            self.q_proj.forward(h.clone()).reshape([b, l, hq, dh]),
            positions.clone(),
            ROPE_MAX_WAVELENGTH,
            device,
        );
        let k = apply_rope(
            self.k_proj.forward(h.clone()).reshape([b, l, hkv, dh]),
            positions,
            ROPE_MAX_WAVELENGTH,
            device,
        );
        let v = self.v_proj.forward(h).reshape([b, l, hkv, dh]);

        let att = eager_gqa_attention(q, k.clone(), v.clone(), mask);
        let out = self.residual_mlp(x, att);
        (out, k, v)
    }

    /// Cross-attention layer (odd index): the action tokens attend ONLY the
    /// prefix. Keys/values are the VLM's cached prefix KV **reprojected** through
    /// this layer's (320→320) k/v_proj — and are *not* re-RoPE'd (they were
    /// RoPE'd at prefix positions when cached). The query is RoPE'd at positions
    /// normalized to start from 0. `vlm_k/v: [B,Lp,Hkv,Dh]`, `q_positions:
    /// [B,L]`, `mask: [B,L,Lp]`.
    pub fn forward_cross(
        &self,
        x: Tensor<B, 3>,
        q_positions: Tensor<B, 2>,
        vlm_k: Tensor<B, 4>,
        vlm_v: Tensor<B, 4>,
        mask: Tensor<B, 3, Bool>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, l, _] = x.dims();
        let (hq, hkv, dh) = (self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim);
        let lp = vlm_k.dims()[1];

        let h = self.input_layernorm.forward(x.clone());
        let q = self.q_proj.forward(h).reshape([b, l, hq, dh]);
        let q = apply_rope(q, q_positions, ROPE_MAX_WAVELENGTH, device);

        // reproject the cached VLM KV (flattened to [B,Lp,Hkv·Dh]); not RoPE'd
        let k = self
            .k_proj
            .forward(vlm_k.reshape([b, lp, hkv * dh]))
            .reshape([b, lp, hkv, dh]);
        let v = self
            .v_proj
            .forward(vlm_v.reshape([b, lp, hkv * dh]))
            .reshape([b, lp, hkv, dh]);

        let att = eager_gqa_attention(q, k, v, mask);
        self.residual_mlp(x, att)
    }
}
