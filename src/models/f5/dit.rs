//! F5's flow-matching DiT — a textbook AdaLN-zero DiT (see docs/F5_ARCH.md).
//!
//! Each block: AdaLN-zero modulation (time → 6×dim) gates a plain MHSA sublayer
//! (to_q/k/v/o, RoPE, no q/k norm) and a plain MLP sublayer (Linear→GELU→Linear).
//! Reuses avatar's `apply_rotary_emb`, `layer_norm_no_affine`, and the
//! safetensors `WeightLoader`; F5's block is its own (simpler than Flux's).
//!
//! Key prefix in the checkpoint: `ema_model.transformer.`.

use crate::nn::norm::layer_norm_no_affine;
use crate::nn::weight_loader::WeightLoader;
use burn::prelude::*;
use burn::tensor::activation::{gelu, silu, softmax};

/// F5's interleaved rotate (x_transformers convention): view as pairs
/// [.., d/2, 2] = (a,b) → (−b, a). NOT the Llama half-split.
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, s, h, d] = x.dims();
    let half = d / 2;
    let xr = x.reshape([b, s, h, half, 2]);
    let a = xr.clone().slice([0..b, 0..s, 0..h, 0..half, 0..1]);
    let bb = xr.slice([0..b, 0..s, 0..h, 0..half, 1..2]);
    Tensor::cat(vec![-bb, a], 4).reshape([b, s, h, d])
}

/// Apply F5's RoPE (interleaved convention). x: [B,S,H,D]; cos/sin: [S,D].
fn apply_rope<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 2>, sin: Tensor<B, 2>) -> Tensor<B, 4> {
    let cos = cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(2); // [1,S,1,D]
    let sin = sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(2);
    x.clone() * cos + rotate_half(x) * sin
}

/// Linear `y = x · Wᵀ + b`. weight stored [out, in]; we keep it transposed.
pub struct Linear<B: Backend> {
    weight_t: Tensor<B, 2>, // [in, out]
    bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> Linear<B> {
    pub fn load(loader: &WeightLoader, prefix: &str, bias: bool, device: &B::Device) -> Self {
        let w: Tensor<B, 2> = loader.load_tensor(&format!("{prefix}.weight"), device); // [out,in]
        let weight_t = w.transpose();
        let bias = if bias {
            Some(loader.load_tensor(&format!("{prefix}.bias"), device))
        } else {
            None
        };
        Self { weight_t, bias }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.weight_t.dims()[1];
        let wt = self.weight_t.clone().unsqueeze_dim::<3>(0); // [1, in, out]
        let mut y = x.matmul(wt); // [B, S, out]
        if let Some(b) = &self.bias {
            y = y + b.clone().reshape([1, 1, out]);
        }
        y
    }
}

/// Compute F5's 1-D RoPE tables. `inv_freq` is [head_dim/2]; returns cos/sin of
/// shape [S, head_dim] ready for `apply_rotary_emb`.
pub fn f5_rope<B: Backend>(
    seq_len: usize,
    inv_freq: Tensor<B, 1>,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half = inv_freq.dims()[0];
    let pos = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device).float(); // [S]
    let freqs = pos.reshape([seq_len, 1]) * inv_freq.reshape([1, half]); // [S, half]
    // interleaved repeat: [f0,f0,f1,f1,…] to match the pair-wise rotate_half.
    let cf = freqs.clone().cos().reshape([seq_len, half, 1]);
    let sf = freqs.sin().reshape([seq_len, half, 1]);
    let cos = Tensor::cat(vec![cf.clone(), cf], 2).reshape([seq_len, 2 * half]);
    let sin = Tensor::cat(vec![sf.clone(), sf], 2).reshape([seq_len, 2 * half]);
    (cos, sin)
}

/// Plain multi-head self-attention with RoPE (no q/k norm).
pub struct F5Attention<B: Backend> {
    to_q: Linear<B>,
    to_k: Linear<B>,
    to_v: Linear<B>,
    to_out: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> F5Attention<B> {
    pub fn load(
        loader: &WeightLoader,
        prefix: &str,
        heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            to_q: Linear::load(loader, &format!("{prefix}.to_q"), true, device),
            to_k: Linear::load(loader, &format!("{prefix}.to_k"), true, device),
            to_v: Linear::load(loader, &format!("{prefix}.to_v"), true, device),
            to_out: Linear::load(loader, &format!("{prefix}.to_out.0"), true, device),
            heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, cos: Tensor<B, 2>, sin: Tensor<B, 2>) -> Tensor<B, 3> {
        let [b, s, dim] = x.dims();
        let (h, d) = (self.heads, self.head_dim);
        let q = self.to_q.forward(x.clone()).reshape([b, s, h, d]);
        let k = self.to_k.forward(x.clone()).reshape([b, s, h, d]);
        let v = self.to_v.forward(x).reshape([b, s, h, d]);
        let q = apply_rope(q, cos.clone(), sin.clone());
        let k = apply_rope(k, cos, sin);
        // → [B, H, S, D] for scaled dot-product attention
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        let scale = 1.0 / (d as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) * scale; // [B,H,S,S]
        let out = softmax(scores, 3).matmul(v); // [B,H,S,D]
        let out = out.swap_dims(1, 2).reshape([b, s, dim]);
        self.to_out.forward(out)
    }
}

/// One AdaLN-zero DiT block.
pub struct F5Block<B: Backend> {
    attn_norm: Linear<B>, // dim → 6·dim
    attn: F5Attention<B>,
    ff_in: Linear<B>,  // dim → ff
    ff_out: Linear<B>, // ff → dim
    eps: f64,
}

impl<B: Backend> F5Block<B> {
    pub fn load(
        loader: &WeightLoader,
        idx: usize,
        heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let p = format!("ema_model.transformer.transformer_blocks.{idx}");
        Self {
            attn_norm: Linear::load(loader, &format!("{p}.attn_norm.linear"), true, device),
            attn: F5Attention::load(loader, &format!("{p}.attn"), heads, head_dim, device),
            ff_in: Linear::load(loader, &format!("{p}.ff.ff.0.0"), true, device),
            ff_out: Linear::load(loader, &format!("{p}.ff.ff.2"), true, device),
            eps: 1e-6,
        }
    }

    /// x: [B,S,dim]; time_emb: [B,dim]; rope cos/sin: [S, head_dim].
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        time_emb: Tensor<B, 2>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let dim = x.dims()[2] as i64;
        // AdaLN-zero: modulation = Linear(SiLU(t)) → [B, 6·dim] → [B,1,6·dim]
        let m = self.attn_norm.forward(silu(time_emb).unsqueeze_dim::<3>(1));
        let chunk = |i: i64| {
            m.clone().slice([
                0..m.dims()[0],
                0..1,
                (i * dim) as usize..((i + 1) * dim) as usize,
            ])
        };
        let (sa_shift, sa_scale, sa_gate) = (chunk(0), chunk(1), chunk(2));
        let (mlp_shift, mlp_scale, mlp_gate) = (chunk(3), chunk(4), chunk(5));

        // attention sublayer
        let h = layer_norm_no_affine(x.clone(), self.eps) * (sa_scale + 1.0) + sa_shift;
        let h = self.attn.forward(h, cos, sin);
        let x = x + h * sa_gate;

        // MLP sublayer
        let h = layer_norm_no_affine(x.clone(), self.eps) * (mlp_scale + 1.0) + mlp_shift;
        let h = self.ff_out.forward(gelu(self.ff_in.forward(h)));
        x + h * mlp_gate
    }
}
