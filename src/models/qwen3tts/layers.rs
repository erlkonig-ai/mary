//! Shared transformer primitives for the Qwen3-TTS port: Linear, RMSNorm,
//! embedding lookup, RoPE, KV cache, and the Qwen3-style GQA decoder layer
//! (optional per-head q/k-norm, optional sliding window, optional LayerScale —
//! the codec's pre_transformer reuses the same block with different knobs).
//!
//! The decode loop is **CPU-submission-bound**: every burn op costs ~15-25 µs
//! of host time (graph capture + buffer management + encoding), while the GPU
//! drains in a fraction of that. The layout here is therefore op-count-driven:
//!
//! - one **wide fused matmul** `[q‖k | R(q‖k) | v]` per attention, where `R`
//!   is rotate_half pre-applied to the weight rows — RoPE becomes a short
//!   elementwise chain `(qk·w·cos + qkR·w_rot·sin)·s` with no narrow/cat ops;
//! - q and k share one variance pass (`s`), their RMS weights (and the
//!   attention's 1/√d) are folded into the chain's `w`;
//! - the input/post-attention RMSNorm **weights** are folded into the
//!   following matmul's rows, LayerScale into o_proj/down_proj columns —
//!   the norms in the layer are weightless rsqrt chains;
//! - cos/sin position slices are computed **once per forward** by the stack
//!   owner and passed down, not recomputed per layer;
//! - all Linear weights are stored pre-transposed + pre-unsqueezed;
//! - single-token decode folds the GQA groups onto the query axis
//!   ([B,H,1,D] ≅ [B,Hkv,G,D]) so k/v need no expand and causality no mask.

use burn::prelude::*;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::FloatDType;

use crate::nn::weight_loader::WeightLoader;

/// Bias-optional Linear, stored ready-to-matmul: `[1, in, out]`.
pub struct Linear<B: Backend> {
    pub weight_t: Tensor<B, 3>, // [1, in, out]
    pub bias: Option<Tensor<B, 3>>, // [1, 1, out]
}

impl<B: Backend> Linear<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, bias: bool, device: &B::Device) -> Self {
        let w: Tensor<B, 2> = loader.load_tensor(&format!("{prefix}.weight"), device); // [out, in]
        let [o, i] = w.dims();
        Self {
            weight_t: w.transpose().reshape([1, i, o]),
            bias: bias.then(|| {
                let b: Tensor<B, 1> = loader.load_tensor(&format!("{prefix}.bias"), device);
                b.reshape([1, 1, o])
            }),
        }
    }

    /// Fold a per-input-channel scale into the weight (for absorbing a
    /// preceding RMSNorm's weight): `x·diag(s) @ W  ≡  x @ (diag(s)·W)`.
    pub fn fold_in(mut self, s: Tensor<B, 1>) -> Self {
        let i = s.dims()[0];
        self.weight_t = self.weight_t.mul(s.reshape([1, i, 1]));
        self
    }

    /// Fold a per-output-channel scale into weight + bias (for absorbing a
    /// following LayerScale): `(x @ W)·diag(s) ≡ x @ (W·diag(s))`.
    pub fn fold_out(mut self, s: Tensor<B, 1>) -> Self {
        let o = s.dims()[0];
        let s = s.reshape([1, 1, o]);
        self.weight_t = self.weight_t.mul(s.clone());
        self.bias = self.bias.map(|b| b.mul(s));
        self
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = x.matmul(self.weight_t.clone());
        match &self.bias {
            Some(bias) => y + bias.clone(),
            None => y,
        }
    }
}

/// Weightless RMS normalization: `x · rsqrt(mean(x², dim=2) + eps)` — the
/// weight lives folded into whatever matmul consumes the normed activations.
/// The variance chain runs in f32 and casts back — LLM activation outliers
/// overflow f16 both per-element (x² > 65504) and in the mean's accumulator.
pub fn rms<B: Backend>(x: Tensor<B, 3>, eps: f64) -> Tensor<B, 3> {
    let dt = x.dtype();
    let x32 = x.cast(FloatDType::F32);
    let var = x32.clone().powf_scalar(2.0).mean_dim(2);
    x32.mul(var.add_scalar(eps).sqrt().recip()).cast(dt)
}

/// RMSNorm over the last dim: `x · rsqrt(mean(x²)+eps) · weight` — kept for
/// stack-final norms whose output feeds more than one consumer.
pub struct RmsNorm<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn load(loader: &WeightLoader, name: &str, eps: f64, device: &B::Device) -> Self {
        Self { weight: loader.load_tensor(name, device), eps }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let d = self.weight.dims()[0];
        rms(x, self.eps).mul(self.weight.clone().reshape([1, 1, d]))
    }
}

/// Embedding table `[vocab, dim]`; lookup by u32 ids → `[1, L, dim]`.
pub struct Embedding<B: Backend> {
    pub weight: Tensor<B, 2>,
}

impl<B: Backend> Embedding<B> {
    pub fn load(loader: &WeightLoader, name: &str, device: &B::Device) -> Self {
        Self { weight: loader.load_tensor(name, device) }
    }

    pub fn forward(&self, ids: &[u32], device: &B::Device) -> Tensor<B, 3> {
        let idx: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let idx = Tensor::<B, 1, Int>::from_ints(idx.as_slice(), device);
        let (l, d) = (ids.len(), self.weight.dims()[1]);
        self.weight.clone().select(0, idx).reshape([1, l, d])
    }
}

/// Precomputed RoPE tables in **full-width rotate_half form**: `[max_len, D]`
/// with the half-tables duplicated (`cos_full = [cos‖cos]`, `sin = [sin‖sin]`),
/// so `rope(x) = x·cos + rotate_half(x)·sin` — and `rotate_half` itself is
/// pre-applied to the qkv weight rows (see `Attention`), never to activations.
pub struct RopeTable<B: Backend> {
    cos: Tensor<B, 2>, // [max_len, d]
    sin: Tensor<B, 2>,
    d: usize,
}

impl<B: Backend> RopeTable<B> {
    pub fn new(theta: f64, head_dim: usize, max_len: usize, device: &B::Device) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0f32; max_len * head_dim];
        let mut sin = vec![0f32; max_len * head_dim];
        for p in 0..max_len {
            for i in 0..half {
                let r = p as f64 * theta.powf(-2.0 * i as f64 / head_dim as f64);
                cos[p * head_dim + i] = r.cos() as f32;
                cos[p * head_dim + half + i] = r.cos() as f32;
                sin[p * head_dim + i] = r.sin() as f32;
                sin[p * head_dim + half + i] = r.sin() as f32;
            }
        }
        Self {
            cos: Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([max_len, head_dim]),
            sin: Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([max_len, head_dim]),
            d: head_dim,
        }
    }

    /// cos/sin slices for query positions `offset..offset+l`, `[1,1,l,D]` —
    /// computed once per stack forward and shared by all layers.
    pub fn slices(&self, offset: usize, l: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        (
            self.cos.clone().narrow(0, offset, l).reshape([1, 1, l, self.d]),
            self.sin.clone().narrow(0, offset, l).reshape([1, 1, l, self.d]),
        )
    }
}

/// Growing KV cache: `[B, Hkv, L, D]` per tensor.
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }

    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dims()[2])
    }

    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (fk, fv) = match (self.k.take(), self.v.take()) {
            (Some(pk), Some(pv)) => (Tensor::cat(vec![pk, k], 2), Tensor::cat(vec![pv, v], 2)),
            _ => (k, v),
        };
        self.k = Some(fk.clone());
        self.v = Some(fv.clone());
        (fk, fv)
    }
}

/// Geometry + behavior knobs for one attention stack.
#[derive(Clone, Copy)]
pub struct AttnConfig {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub eps: f64,
    /// Sliding-window size (codec pre_transformer: 72); None = full causal.
    pub window: Option<usize>,
    /// Qwen3 per-head q/k RMSNorm (talker + predictor: true, codec: false).
    pub qk_norm: bool,
    /// LayerScale on both residual branches (codec: Some(init from ckpt)).
    pub layer_scale: bool,
}

pub struct Attention<B: Backend> {
    /// Wide fused projection `[1, hidden, (2·(H+Hkv)+Hkv)·D]`, output layout
    /// `[ qk | R(qk) | v ]` where `R` = rotate_half applied to the weight rows
    /// of the qk block. The preceding RMSNorm's weight is folded in.
    pub wide_t: Tensor<B, 3>,
    /// RoPE chain scale for the `qk` block `[1, H+Hkv, 1, D]`: the per-head
    /// q/k-norm weights (or ones) with 1/√D folded into the q section.
    pub w: Tensor<B, 4>,
    /// Same for the `R(qk)` block: half-permuted norm weights (rotate_half
    /// moves dim `i` to `i±half`, so the elementwise weight moves with it).
    pub w_rot: Tensor<B, 4>,
    pub o_proj: Linear<B>,
    cfg: AttnConfig,
}

impl<B: Backend> Attention<B> {
    /// Assemble from pre-folded parts — the zero-copy aliased path, where the
    /// fold math ran once at derive time (`qwen3tts_persist --fold-derive`)
    /// and the tensors arrive in their final `load`-produced layout.
    pub(crate) fn from_parts(
        wide_t: Tensor<B, 3>,
        w: Tensor<B, 4>,
        w_rot: Tensor<B, 4>,
        o_proj: Linear<B>,
        cfg: AttnConfig,
    ) -> Self {
        Self { wide_t, w, w_rot, o_proj, cfg }
    }

    /// `fold_in`: the preceding RMSNorm's weight (folded into `wide_t` rows).
    /// `fold_out`: LayerScale for the attention branch (into o_proj columns).
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        cfg: AttnConfig,
        fold_in: Tensor<B, 1>,
        fold_out: Option<Tensor<B, 1>>,
        device: &B::Device,
    ) -> Self {
        let (h, hkv, d) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let half = d / 2;
        let w2 = |n: &str| -> Tensor<B, 2> { loader.load_tensor(&format!("{prefix}.{n}.weight"), device) };

        // rotate_half on OUTPUT rows: rows [heads, d, in] → [-rows[half..] ‖ rows[..half]]
        let qk: Tensor<B, 2> = Tensor::cat(vec![w2("q_proj"), w2("k_proj")], 0); // [(H+Hkv)D, in]
        let hidden = qk.dims()[1];
        let qk3 = qk.clone().reshape([h + hkv, d, hidden]);
        let qk_rot = Tensor::cat(
            vec![qk3.clone().narrow(1, half, half).neg(), qk3.narrow(1, 0, half)],
            1,
        )
        .reshape([(h + hkv) * d, hidden]);
        let wide = Tensor::cat(vec![qk, qk_rot, w2("v_proj")], 0); // [(2(H+Hkv)+Hkv)D, in]
        let n_out = wide.dims()[0];
        let wide_t = wide
            .transpose()
            .mul(fold_in.reshape([hidden, 1]))
            .reshape([1, hidden, n_out]);

        // chain weights: per-head q/k norm weight (or ones), 1/√D on the q part
        let scale = (d as f64).powf(-0.5);
        let (qn, kn): (Tensor<B, 1>, Tensor<B, 1>) = if cfg.qk_norm {
            (
                loader.load_tensor(&format!("{prefix}.q_norm.weight"), device),
                loader.load_tensor(&format!("{prefix}.k_norm.weight"), device),
            )
        } else {
            (Tensor::ones([d], device), Tensor::ones([d], device))
        };
        let perm = |t: Tensor<B, 1>| -> Tensor<B, 1> {
            Tensor::cat(vec![t.clone().narrow(0, half, half), t.narrow(0, 0, half)], 0)
        };
        let heads_w = |q: Tensor<B, 1>, k: Tensor<B, 1>| -> Tensor<B, 4> {
            Tensor::cat(
                vec![
                    q.mul_scalar(scale).reshape([1, 1, 1, d]).expand([1, h, 1, d]),
                    k.reshape([1, 1, 1, d]).expand([1, hkv, 1, d]),
                ],
                1,
            )
        };
        let w = heads_w(qn.clone(), kn.clone());
        let w_rot = heads_w(perm(qn), perm(kn));

        let mut o_proj = Linear::load(loader, &format!("{prefix}.o_proj"), false, device);
        if let Some(s) = fold_out {
            o_proj = o_proj.fold_out(s);
        }
        Self { wide_t, w, w_rot, o_proj, cfg }
    }

    /// Causal (optionally sliding-window) self-attention with cache.
    /// `x: [B, L, hidden]` (pre-normed, weightless); `cos`/`sin` are the
    /// caller's position slices `[1,1,L,D]` for `offset..offset+L` where
    /// `offset = cache.seq_len()`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        cache: &mut KvCache<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, l, _] = x.dims();
        let (h, hkv, d) = (self.cfg.heads, self.cfg.kv_heads, self.cfg.head_dim);
        let hh = h + hkv;
        let offset = cache.seq_len();

        let qkv = x.matmul(self.wide_t.clone()); // [B,L,(2(H+Hkv)+Hkv)·D]
        // [B,L,heads·D] → [B,heads,L,D]; for L=1 the reshape alone is exact.
        let heads = |t: Tensor<B, 3>, n: usize| -> Tensor<B, 4> {
            if l == 1 {
                t.reshape([b, n, 1, d])
            } else {
                t.reshape([b, l, n, d]).swap_dims(1, 2)
            }
        };
        let qk = heads(qkv.clone().narrow(2, 0, hh * d), hh);
        let qkr = heads(qkv.clone().narrow(2, hh * d, hh * d), hh);
        let v = heads(qkv.narrow(2, 2 * hh * d, hkv * d), hkv);

        // norm + RoPE as one elementwise chain (variance in f32, see `rms`)
        let roped = if self.cfg.qk_norm {
            let s = qk
                .clone()
                .cast(FloatDType::F32)
                .powf_scalar(2.0)
                .mean_dim(3)
                .add_scalar(self.cfg.eps)
                .sqrt()
                .recip()
                .cast(qk.dtype());
            (qk.mul(self.w.clone()).mul(cos) + qkr.mul(self.w_rot.clone()).mul(sin)).mul(s)
        } else {
            qk.mul(self.w.clone()).mul(cos) + qkr.mul(self.w_rot.clone()).mul(sin)
        };
        let q = roped.clone().narrow(1, 0, h);
        let k = roped.narrow(1, h, hkv);

        let (k, v) = cache.update(k, v);
        let lk = k.dims()[2];
        let groups = h / hkv;

        if l == 1 && self.cfg.window.is_none() {
            // Decode step: fold the GQA groups onto the (length-1) query axis —
            // [B,H,1,D] ≅ [B,Hkv,G,D] — no kv expand, no mask.
            let q = q.reshape([b, hkv, groups, d]);
            let scores = q.matmul(k.swap_dims(2, 3)); // [B,Hkv,G,Lk], 1/√D folded
            let probs = softmax(scores, 3);
            let out = probs.matmul(v).reshape([b, 1, h * d]);
            return self.o_proj.forward(out);
        }

        // GQA: expand kv heads
        let expand = |t: Tensor<B, 4>| {
            t.reshape([b, hkv, 1, lk, d])
                .expand([b, hkv, groups, lk, d])
                .reshape([b, h, lk, d])
        };
        let k = expand(k);
        let v = expand(v);

        let scores = q.matmul(k.swap_dims(2, 3)); // [B,H,L,Lk], 1/√D folded into w

        // mask: query position offset+i may attend key j iff j <= offset+i
        // (and offset+i - j < window when sliding). Built host-side — masks are
        // small (≤ a few hundred squared) and this sidesteps bool-op APIs.
        let scores = {
            let w = self.cfg.window.unwrap_or(usize::MAX);
            let mut blocked = vec![false; l * lk];
            for i in 0..l {
                let qp = offset + i;
                for j in 0..lk {
                    blocked[i * lk + j] = j > qp || qp - j >= w;
                }
            }
            let mask = Tensor::<B, 2, Bool>::from_data(
                burn::tensor::TensorData::new(blocked, [l, lk]),
                device,
            )
            .reshape([1, 1, l, lk])
            .expand([b, h, l, lk]);
            scores.mask_fill(mask, f32::MIN)
        };

        let probs = softmax(scores, 3);
        let out = probs.matmul(v).swap_dims(1, 2).reshape([b, l, h * d]);
        self.o_proj.forward(out)
    }
}

/// One Qwen3-style decoder block: pre-norm attention + SwiGLU MLP. The norm
/// weights are folded into wide_t/gate_up_t; LayerScale into o_proj/down_proj.
pub struct DecoderLayer<B: Backend> {
    pub attn: Attention<B>,
    /// Fused gate‖up `[1, hidden, 2·inter]` with the post-attention norm
    /// weight folded into the rows.
    pub gate_up_t: Tensor<B, 3>,
    pub down_proj: Linear<B>,
    eps: f64,
}

impl<B: Backend> DecoderLayer<B> {
    /// Assemble from pre-folded parts — see [`Attention::from_parts`].
    pub(crate) fn from_parts(
        attn: Attention<B>,
        gate_up_t: Tensor<B, 3>,
        down_proj: Linear<B>,
        eps: f64,
    ) -> Self {
        Self { attn, gate_up_t, down_proj, eps }
    }

    pub fn load(loader: &WeightLoader, prefix: &str, cfg: AttnConfig, device: &B::Device) -> Self {
        let w = |n: &str| -> Tensor<B, 2> { loader.load_tensor(&format!("{prefix}.{n}.weight"), device) };
        let w1 = |n: &str| -> Tensor<B, 1> { loader.load_tensor(&format!("{prefix}.{n}.weight"), device) };
        let scale = |n: &str| -> Option<Tensor<B, 1>> {
            cfg.layer_scale.then(|| loader.load_tensor(&format!("{prefix}.{n}.scale"), device))
        };
        let attn = Attention::load(
            loader,
            &format!("{prefix}.self_attn"),
            cfg,
            w1("input_layernorm"),
            scale("self_attn_layer_scale"),
            device,
        );
        let gu = Tensor::cat(vec![w("mlp.gate_proj"), w("mlp.up_proj")], 0); // [2I, hidden]
        let [o2, hidden] = gu.dims();
        let gate_up_t = gu
            .transpose()
            .mul(w1("post_attention_layernorm").reshape([hidden, 1]))
            .reshape([1, hidden, o2]);
        let mut down_proj = Linear::load(loader, &format!("{prefix}.mlp.down_proj"), false, device);
        if let Some(s) = scale("mlp_layer_scale") {
            down_proj = down_proj.fold_out(s);
        }
        Self { attn, gate_up_t, down_proj, eps: cfg.eps }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        cache: &mut KvCache<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let att = self
            .attn
            .forward(rms(x.clone(), self.eps), cos.clone(), sin.clone(), cache, device);
        let x = x + att;
        let h = rms(x.clone(), self.eps);
        let gu = h.matmul(self.gate_up_t.clone());
        let inter = self.gate_up_t.dims()[2] / 2;
        let mlp = self
            .down_proj
            .forward(silu(gu.clone().narrow(2, 0, inter)).mul(gu.narrow(2, inter, inter)));
        x + mlp
    }
}
