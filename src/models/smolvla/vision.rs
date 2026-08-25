//! SigLIP vision encoder + the SmolVLM pixel-shuffle connector — `embed_image`.
//! The frozen perceptual front-end: a 512×512 image → 1024 patch tokens (conv
//! patch embed + learned position table) → 12 ViT layers (LayerNorm + full MHA
//! + GELU-tanh MLP) → post-LayerNorm → pixel-shuffle (×16) → 12288→960 proj →
//! 64 image tokens in the text embedding space.
//!
//! Unlike the text towers this is plain ViT: standard LayerNorm (not RMSNorm),
//! full bidirectional attention (no GQA, no RoPE, no mask), GELU-tanh.

use burn::prelude::*;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;

use super::config::SmolVlaConfig;
use super::projections::Linear;
use crate::nn::weight_loader::WeightLoader;

/// Standard affine LayerNorm over the last dim.
pub struct LayerNorm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    fn load(loader: &WeightLoader, prefix: &str, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: loader.load_tensor(&format!("{prefix}.weight"), device),
            bias: loader.load_tensor(&format!("{prefix}.bias"), device),
            eps,
        }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let d = self.weight.dims()[0];
        let mean = x.clone().mean_dim(2);
        let xc = x.sub(mean);
        let var = xc.clone().powf_scalar(2.0).mean_dim(2);
        let norm = xc.div(var.add_scalar(self.eps).sqrt());
        norm.mul(self.weight.clone().reshape([1, 1, d]))
            .add(self.bias.clone().reshape([1, 1, d]))
    }
}

/// GELU (tanh approximation) — `gelu_pytorch_tanh`.
fn gelu_tanh<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let x3 = x.clone().powf_scalar(3.0);
    let inner = x
        .clone()
        .add(x3.mul_scalar(0.044715))
        .mul_scalar(0.7978845608028654);
    x.mul(inner.tanh().add_scalar(1.0)).mul_scalar(0.5)
}

struct VisionLayer<B: Backend> {
    ln1: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    ln2: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> VisionLayer<B> {
    fn load(
        loader: &WeightLoader,
        p: &str,
        eps: f64,
        n_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let lin = |n: &str| Linear::load(loader, &format!("{p}.{n}"), true, device);
        Self {
            ln1: LayerNorm::load(loader, &format!("{p}.layer_norm1"), eps, device),
            q: lin("self_attn.q_proj"),
            k: lin("self_attn.k_proj"),
            v: lin("self_attn.v_proj"),
            out: lin("self_attn.out_proj"),
            ln2: LayerNorm::load(loader, &format!("{p}.layer_norm2"), eps, device),
            fc1: lin("mlp.fc1"),
            fc2: lin("mlp.fc2"),
            n_heads,
            head_dim,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let (h, hd) = (self.n_heads, self.head_dim);
        // self-attention (full, no mask)
        let hidden = self.ln1.forward(x.clone());
        let shape = |t: Tensor<B, 3>| t.reshape([b, s, h, hd]).swap_dims(1, 2); // [b,h,s,hd]
        let q = shape(self.q.forward(hidden.clone()));
        let k = shape(self.k.forward(hidden.clone()));
        let v = shape(self.v.forward(hidden));
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((hd as f64).powf(-0.5));
        let probs = burn::tensor::activation::softmax(scores, 3);
        let att = probs.matmul(v).swap_dims(1, 2).reshape([b, s, d]);
        let x = x.add(self.out.forward(att)); // residual 1
        // MLP
        let h2 = self.ln2.forward(x.clone());
        let mlp = self.fc2.forward(gelu_tanh(self.fc1.forward(h2)));
        x.add(mlp) // residual 2
    }
}

pub struct VisionEncoder<B: Backend> {
    patch_weight: Tensor<B, 4>, // [768,3,16,16]
    patch_bias: Tensor<B, 1>,
    position_embedding: Tensor<B, 2>, // [1024,768]
    layers: Vec<VisionLayer<B>>,
    post_ln: LayerNorm<B>,
    connector: Linear<B>, // [960, 12288], no bias
    patch_size: usize,
    scale_factor: usize,
}

impl<B: Backend> VisionEncoder<B> {
    pub fn load(loader: &WeightLoader, _cfg: &SmolVlaConfig, device: &B::Device) -> Self {
        let vp = "model.vlm_with_expert.vlm.model.vision_model";
        // SigLIP vision config (fixed for SmolVLM2-500M): 768/12 layers/12 heads/64, eps 1e-6
        let (n_layers, n_heads, head_dim, eps) = (12usize, 12usize, 64usize, 1e-6);
        let layers = (0..n_layers)
            .map(|i| {
                VisionLayer::load(
                    loader,
                    &format!("{vp}.encoder.layers.{i}"),
                    eps,
                    n_heads,
                    head_dim,
                    device,
                )
            })
            .collect();
        Self {
            patch_weight: loader
                .load_tensor(&format!("{vp}.embeddings.patch_embedding.weight"), device),
            patch_bias: loader
                .load_tensor(&format!("{vp}.embeddings.patch_embedding.bias"), device),
            position_embedding: loader.load_tensor(
                &format!("{vp}.embeddings.position_embedding.weight"),
                device,
            ),
            layers,
            post_ln: LayerNorm::load(loader, &format!("{vp}.post_layernorm"), eps, device),
            connector: Linear::load(
                loader,
                "model.vlm_with_expert.vlm.model.connector.modality_projection.proj",
                false,
                device,
            ),
            patch_size: 16,
            scale_factor: 4,
        }
    }

    /// Patch + position embedding: image `[B,3,H,W]` → `[B, num_patches, 768]`.
    pub fn embeddings(&self, image: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, _, _, _] = image.dims();
        let opts = ConvOptions::new([self.patch_size, self.patch_size], [0, 0], [1, 1], 1);
        let patch = conv2d(
            image,
            self.patch_weight.clone(),
            Some(self.patch_bias.clone()),
            opts,
        ); // [b,768,gh,gw]
        let [_, c, gh, gw] = patch.dims();
        let patch = patch.reshape([b, c, gh * gw]).swap_dims(1, 2); // [b, np, 768]
        let np = gh * gw;
        patch.add(self.position_embedding.clone().reshape([1, np, c]))
    }

    /// SmolVLM pixel-shuffle: `[B, seq, embed]` → `[B, seq/sf², embed·sf²]`.
    fn pixel_shuffle(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, seq, embed] = x.dims();
        let sf = self.scale_factor;
        let hw = (seq as f64).sqrt() as usize;
        x.reshape([b, hw, hw, embed])
            .reshape([b, hw, hw / sf, embed * sf])
            .swap_dims(1, 2)
            .reshape([b, hw / sf, hw / sf, embed * sf * sf])
            .swap_dims(1, 2)
            .reshape([b, seq / (sf * sf), embed * sf * sf])
    }

    /// Full `embed_image`: image `[B,3,512,512]` → image tokens `[B,64,960]`.
    pub fn embed_image(&self, image: Tensor<B, 4>) -> Tensor<B, 3> {
        let mut x = self.embeddings(image);
        for layer in &self.layers {
            x = layer.forward(x);
        }
        let x = self.post_ln.forward(x);
        self.connector.forward(self.pixel_shuffle(x))
    }

    /// Expose one encoder layer for probing.
    pub fn layer0(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.layers[0].forward(x)
    }

    /// Post-encoder hidden state (all layers + post-LayerNorm), pre-connector.
    pub fn encode(&self, image: Tensor<B, 4>) -> Tensor<B, 3> {
        let mut x = self.embeddings(image);
        for layer in &self.layers {
            x = layer.forward(x);
        }
        self.post_ln.forward(x)
    }
}
