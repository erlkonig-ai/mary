//! Qwen2.5-VL text-decoder primitives — the backbone of `BiQwen2_5`
//! (`nomic-embed-multimodal-7b`, a DENSE 3584-d last-token embedder).
//!
//! Per `Qwen/Qwen2.5-VL-7B-Instruct`: RMSNorm (variance-only, eps 1e-6), GQA
//! attention with **bias on q/k/v** and bias-free `o_proj`, SwiGLU MLP (`silu`),
//! 1D RoPE (M-RoPE collapses to standard RoPE for pure-text sequences), two norms
//! per decoder layer with plain residuals. The embedder only does single forward
//! passes (no autoregression), so there is no KV cache here.

use burn::prelude::*;
use burn::tensor::activation::{silu, softmax};

use super::config::Qwen2_5VlTextConfig;

/// Apply RoPE to `x` `[b, n_heads, seq, head_dim]` given per-token half-width
/// `cos`/`sin` `[seq, head_dim/2]` (rotate-half convention; matches HF's
/// `apply_rotary_pos_emb` with duplicated cos/sin). M-RoPE differs from 1D RoPE
/// only in how those `cos`/`sin` were built (see [`QwenTextModel::build_cos_sin`]).
fn apply_rope<B: Backend>(x: Tensor<B, 4>, cos: &Tensor<B, 2>, sin: &Tensor<B, 2>) -> Tensor<B, 4> {
    let [b, nh, seq, hd] = x.dims();
    let half = hd / 2;
    let cos = cos.clone().reshape([1, 1, seq, half]).expand([b, nh, seq, half]);
    let sin = sin.clone().reshape([1, 1, seq, half]).expand([b, nh, seq, half]);
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);
    let out1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let out2 = x1 * sin + x2 * cos;
    Tensor::cat(vec![out1, out2], 3)
}

/// Source of named weights for building the backbone (npy dir, safetensors, or
/// pile keymap). Ranks are explicit so callers stay simple.
pub trait QwenWeights<B: Backend> {
    fn t1(&self, name: &str) -> Tensor<B, 1>;
    fn t2(&self, name: &str) -> Tensor<B, 2>;
}

/// `y = x @ wᵀ (+ b)` against a PyTorch `[out, in]` weight — the same local
/// `Linear` idiom `mary::embed` uses (no dependency on `burn::nn::Linear`
/// internals). `bias` is `[out]`.
struct Linear<B: Backend> {
    weight: Tensor<B, 2>, // [out, in]
    bias: Option<Tensor<B, 1>>,
}
impl<B: Backend> Linear<B> {
    fn new(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>) -> Self {
        Self { weight, bias }
    }
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // f32-accumulated matmul (PyTorch f16 semantics: f16 weights, f32 tensor-
        // core accumulation). cubecl's Metal matmul accumulates in f16, which
        // overflows to `inf` for some 7B/28-layer residual streams (-> NaN). The
        // weights stay f16 in the aliased GPU buffer; only this matmul upcasts.
        // Identity on an f32 backend, so f32 parity is preserved.
        let dt = x.dtype();
        let xf = x.cast(burn::tensor::FloatDType::F32);
        let wf = self.weight.clone().cast(burn::tensor::FloatDType::F32);
        let out = xf.matmul(wf.transpose().unsqueeze());
        let out = match &self.bias {
            Some(b) => out + b.clone().cast(burn::tensor::FloatDType::F32).unsqueeze(),
            None => out,
        };
        out.cast(dt)
    }
}

/// Qwen RMSNorm over the last dim: `x * rsqrt(mean(x²)+eps) * weight`
/// (variance-only, no mean-subtraction). Mirrors `mary::models::smolvla`'s
/// proven manual RmsNorm.
pub struct QwenRmsNorm<B: Backend> {
    weight: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> QwenRmsNorm<B> {
    pub fn from_weight(weight: Tensor<B, 1>, eps: f64) -> Self {
        Self { weight, eps }
    }
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // HF computes RMSNorm in fp32 even for f16 models: `x²` overflows f16 for
        // deep residual streams (-> NaN). Upcast, normalize, cast back. For an f32
        // backend this is identity, so f32 parity is preserved exactly.
        let dt = x.dtype();
        let x = x.cast(burn::tensor::FloatDType::F32);
        let var = x.clone().powf_scalar(2.0).mean_dim(2);
        let normed = x.mul(var.add_scalar(self.eps).sqrt().recip());
        let d = self.weight.dims()[0];
        let w = self.weight.clone().cast(burn::tensor::FloatDType::F32).reshape([1, 1, d]);
        normed.mul(w).cast(dt)
    }
}

/// Token-embedding lookup: ids `[B, S]` → `[B, S, H]`.
pub struct QwenEmbedding<B: Backend> {
    weight: Tensor<B, 2>, // [vocab, hidden]
}
impl<B: Backend> QwenEmbedding<B> {
    pub fn new(weight: Tensor<B, 2>) -> Self {
        Self { weight }
    }
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, s] = ids.dims();
        let h = self.weight.dims()[1];
        let flat = ids.reshape([b * s]);
        // Activations run in f32: this bf16-native model (Qwen2.5) has "massive
        // activations" whose magnitudes exceed f16's 65504 range (-> inf/NaN).
        // The embedding weight stays f16 (zero-copy aliased); only the gathered
        // rows upcast, seeding an f32 residual stream. Identity on an f32 backend.
        self.weight
            .clone()
            .select(0, flat)
            .cast(burn::tensor::FloatDType::F32)
            .reshape([b, s, h])
    }
}

/// GQA attention: q/k/v projections (q/k/v have bias, o does not), 1D RoPE,
/// causal mask, scale `1/sqrt(head_dim)`. Single-pass prefill (no cache).
pub struct QwenAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> QwenAttention<B> {
    pub fn load(w: &impl QwenWeights<B>, prefix: &str, cfg: &Qwen2_5VlTextConfig) -> Self {
        let g = |n: &str| w.t2(&format!("{prefix}.{n}.weight"));
        let b = |n: &str| w.t1(&format!("{prefix}.{n}.bias"));
        Self {
            q_proj: Linear::new(g("q_proj"), Some(b("q_proj"))),
            k_proj: Linear::new(g("k_proj"), Some(b("k_proj"))),
            v_proj: Linear::new(g("v_proj"), Some(b("v_proj"))),
            o_proj: Linear::new(g("o_proj"), None),
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
        }
    }

    fn repeat_kv(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        if n_rep == 1 {
            return x;
        }
        let [b, kv, s, hd] = x.dims();
        x.unsqueeze_dim::<5>(2)
            .expand([b, kv, n_rep, s, hd])
            .reshape([b, kv * n_rep, s, hd])
    }

    pub fn forward(&self, x: Tensor<B, 3>, cos: &Tensor<B, 2>, sin: &Tensor<B, 2>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        let to_heads = |t: Tensor<B, 3>, h: usize| t.reshape([b, s, h, hd]).swap_dims(1, 2);

        let q = apply_rope(to_heads(self.q_proj.forward(x.clone()), nh), cos, sin);
        let k = apply_rope(to_heads(self.k_proj.forward(x.clone()), nkv), cos, sin);
        let v = to_heads(self.v_proj.forward(x), nkv);

        let n_rep = nh / nkv;
        let k = Self::repeat_kv(k, n_rep);
        let v = Self::repeat_kv(v, n_rep);

        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar((hd as f64).powf(-0.5));
        let scores = scores.clone() + Self::causal_mask(s, &scores.device());
        // HF softmaxes attention in fp32 then casts back to the model dtype; in
        // f16 the masked scores + exp can otherwise lose/blow precision. Identity
        // on an f32 backend, so f32 parity is preserved.
        let dt = scores.dtype();
        let probs = softmax(scores.cast(burn::tensor::FloatDType::F32), 3).cast(dt);
        let out = probs.matmul(v).swap_dims(1, 2).reshape([b, s, nh * hd]);
        self.o_proj.forward(out)
    }

    /// Additive causal mask `[1, 1, S, S]` (0 on/below diagonal, -inf above).
    fn causal_mask(s: usize, device: &B::Device) -> Tensor<B, 4> {
        let rows: Vec<Tensor<B, 2>> = (0..s)
            .map(|i| {
                let row: Vec<f32> = (0..s)
                    .map(|j| if j <= i { 0.0 } else { f32::NEG_INFINITY })
                    .collect();
                Tensor::<B, 1>::from_floats(&row[..], device).unsqueeze::<2>()
            })
            .collect();
        Tensor::<B, 2>::cat(rows, 0).reshape([1, 1, s, s])
    }
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`, all bias-free.
pub struct QwenMlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}
impl<B: Backend> QwenMlp<B> {
    pub fn load(w: &impl QwenWeights<B>, prefix: &str) -> Self {
        let g = |n: &str| Linear::new(w.t2(&format!("{prefix}.{n}.weight")), None);
        Self {
            gate_proj: g("gate_proj"),
            up_proj: g("up_proj"),
            down_proj: g("down_proj"),
        }
    }
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        self.down_proj.forward(gate * self.up_proj.forward(x))
    }
}

/// One Qwen2.5 decoder layer: `x += attn(in_ln(x)); x += mlp(post_attn_ln(x))`.
pub struct QwenDecoderLayer<B: Backend> {
    input_layernorm: QwenRmsNorm<B>,
    attn: QwenAttention<B>,
    post_attention_layernorm: QwenRmsNorm<B>,
    mlp: QwenMlp<B>,
}
impl<B: Backend> QwenDecoderLayer<B> {
    pub fn load(w: &impl QwenWeights<B>, idx: usize, cfg: &Qwen2_5VlTextConfig) -> Self {
        let p = format!("layers.{idx}");
        Self {
            input_layernorm: QwenRmsNorm::from_weight(
                w.t1(&format!("{p}.input_layernorm.weight")),
                cfg.rms_norm_eps,
            ),
            attn: QwenAttention::load(w, &format!("{p}.self_attn"), cfg),
            post_attention_layernorm: QwenRmsNorm::from_weight(
                w.t1(&format!("{p}.post_attention_layernorm.weight")),
                cfg.rms_norm_eps,
            ),
            mlp: QwenMlp::load(w, &format!("{p}.mlp")),
        }
    }
    pub fn forward(&self, x: Tensor<B, 3>, cos: &Tensor<B, 2>, sin: &Tensor<B, 2>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.input_layernorm.forward(x), cos, sin);
        x.clone() + self.mlp.forward(self.post_attention_layernorm.forward(x))
    }
}

/// Qwen2.5-VL M-RoPE 3D position-ids (`get_rope_index`). Returns one `[t, h, w]`
/// triple per token of a single interleaved text+image sequence. Text tokens
/// advance all three axes together; the image tokens of each `<|image_pad|>`
/// block get 2D `(h, w)` grid positions (constant `t`), all offset past the
/// running max — exactly HF's `Qwen2_5_VLModel.get_rope_index` for the image
/// case (no video). `image_grids` are the `(t, h, w)` patch grids per image, in
/// order of appearance; `merge` is `spatial_merge_size`.
pub fn get_rope_index(
    input_ids: &[i64],
    image_grids: &[(usize, usize, usize)],
    image_token_id: i64,
    merge: usize,
) -> Vec<[i64; 3]> {
    let n = input_ids.len();
    let mut pos: Vec<[i64; 3]> = Vec::with_capacity(n);
    let mut st = 0usize; // cursor into input_ids
    let mut next_start: i64 = 0; // st_idx: running max position + 1
    for &(t, h, w) in image_grids {
        // ed = first image_pad token at/after st (start of this image block)
        let Some(off) = input_ids[st..].iter().position(|&x| x == image_token_id) else {
            break;
        };
        let ed = st + off;
        let (lt, lh, lw) = (t, h / merge, w / merge);
        // text run [st, ed): sequential positions on all three axes
        let text_len = ed - st;
        for k in 0..text_len {
            let p = next_start + k as i64;
            pos.push([p, p, p]);
        }
        let base = next_start + text_len as i64; // vision-block origin
        for ti in 0..lt {
            for hi in 0..lh {
                for wi in 0..lw {
                    pos.push([base + ti as i64, base + hi as i64, base + wi as i64]);
                }
            }
        }
        let block_max = base + (lt.max(lh).max(lw) as i64) - 1;
        next_start = block_max + 1;
        st = ed + lt * lh * lw;
    }
    // trailing text run after the last image (or the whole sequence if no image)
    if st < n {
        let text_len = n - st;
        for k in 0..text_len {
            let p = next_start + k as i64;
            pos.push([p, p, p]);
        }
    }
    debug_assert_eq!(pos.len(), n, "get_rope_index produced {} of {n} positions", pos.len());
    pos
}

/// Qwen2.5-VL text backbone → dense embedding. Embedding lookup, N decoder
/// layers, final `model.norm`, then **last-token pool + L2-normalize** (the
/// `BiQwen2_5` dense head). RoPE is section-wise M-RoPE over 3D position-ids
/// (which collapses to standard 1D RoPE for pure-text sequences, where the three
/// axes coincide).
pub struct QwenTextModel<B: Backend> {
    embed: QwenEmbedding<B>,
    layers: Vec<QwenDecoderLayer<B>>,
    norm: QwenRmsNorm<B>,
    inv_freq: Vec<f64>,         // [head_dim/2]
    mrope_section: [usize; 3],  // sums to head_dim/2
    device: B::Device,
}

impl<B: Backend> QwenTextModel<B> {
    pub fn load(w: &impl QwenWeights<B>, cfg: &Qwen2_5VlTextConfig, device: &B::Device) -> Self {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| QwenDecoderLayer::load(w, i, cfg))
            .collect();
        let head_dim = cfg.head_dim();
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();
        // Section split across the (t, h, w) axes; sums to half. Pure text never
        // depends on the split (axes coincide), so default everything to axis 0.
        let mrope_section = cfg
            .rope_scaling
            .as_ref()
            .map(|s| s.mrope_section)
            .unwrap_or([half, 0, 0]);
        assert_eq!(
            mrope_section.iter().sum::<usize>(),
            half,
            "mrope_section {mrope_section:?} must sum to head_dim/2 = {half}"
        );
        Self {
            embed: QwenEmbedding::new(w.t2("embed_tokens.weight")),
            layers,
            norm: QwenRmsNorm::from_weight(w.t1("norm.weight"), cfg.rms_norm_eps),
            inv_freq,
            mrope_section,
            device: device.clone(),
        }
    }

    /// Build per-token half-width `cos`/`sin` `[S, head_dim/2]` from 3D
    /// position-ids via section-wise M-RoPE: frequency `j` reads the position
    /// axis its `mrope_section` chunk selects (t for the first chunk, h for the
    /// second, w for the third), then `angle = pos[axis] * inv_freq[j]`.
    pub fn build_cos_sin(&self, position_ids: &[[i64; 3]]) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let half = self.inv_freq.len();
        // axis index per frequency from the cumulative section split
        let mut axis_of = vec![0usize; half];
        let mut j = 0usize;
        for (ax, &count) in self.mrope_section.iter().enumerate() {
            for _ in 0..count {
                axis_of[j] = ax;
                j += 1;
            }
        }
        let s = position_ids.len();
        let mut cos = vec![0f32; s * half];
        let mut sin = vec![0f32; s * half];
        for (ti, p) in position_ids.iter().enumerate() {
            for j in 0..half {
                let angle = p[axis_of[j]] as f64 * self.inv_freq[j];
                cos[ti * half + j] = angle.cos() as f32;
                sin[ti * half + j] = angle.sin() as f32;
            }
        }
        (
            Tensor::<B, 1>::from_floats(&cos[..], &self.device).reshape([s, half]),
            Tensor::<B, 1>::from_floats(&sin[..], &self.device).reshape([s, half]),
        )
    }

    /// Core: run the decoder stack + final norm over input embeds `[B, S, H]`
    /// with explicit 3D position-ids. (B is 1 for the embedder.)
    pub fn run_embeds(&self, mut x: Tensor<B, 3>, position_ids: &[[i64; 3]]) -> Tensor<B, 3> {
        let (cos, sin) = self.build_cos_sin(position_ids);
        for layer in &self.layers {
            x = layer.forward(x, &cos, &sin);
        }
        self.norm.forward(x)
    }

    /// Default text positions: sequential `[i, i, i]` for each of `s` tokens.
    fn text_positions(s: usize) -> Vec<[i64; 3]> {
        (0..s).map(|i| { let p = i as i64; [p, p, p] }).collect()
    }

    /// Run the backbone over token ids `[B, S]` → final hidden states `[B, S, H]`
    /// (pure-text 1D RoPE).
    pub fn hidden(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let s = ids.dims()[1];
        self.run_embeds(self.embed.forward(ids), &Self::text_positions(s))
    }

    /// Token-embedding lookup `[B, S]` → `[B, S, H]` (exposed for the multimodal
    /// splice, which overwrites the image-pad rows with vision tokens).
    pub fn embed_tokens(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed.forward(ids)
    }

    /// Last-token pool of final hidden states, L2-normalized — the dense head.
    fn pool(h: Tensor<B, 3>) -> Tensor<B, 2> {
        let [b, s, d] = h.dims();
        let pooled = h.narrow(1, s - 1, 1).reshape([b, d]); // last token (left-padded)
        let norm = pooled.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        pooled / norm
    }

    /// Dense embedding: last-token pool of the final hidden states, L2-normalized.
    pub fn embed(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        Self::pool(self.hidden(ids))
    }

    /// Dense embedding from already-spliced input embeds `[B, S, H]` + 3D
    /// position-ids (the multimodal image path).
    pub fn embed_from_embeds(&self, x: Tensor<B, 3>, position_ids: &[[i64; 3]]) -> Tensor<B, 2> {
        Self::pool(self.run_embeds(x, position_ids))
    }
}
