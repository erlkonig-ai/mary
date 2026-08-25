//! Gemma 4 transformer layers: attention (sliding + full), MLP, PLE.
//!
//! Uses Burn's built-in attention kernel with softcap support.

use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;

use super::config::{Gemma4TextConfig, LayerType};
use crate::models::gemma::lora::{LoraWeights, maybe_lora};
use crate::models::gemma::rope::RopeTable;

// ---------------------------------------------------------------------------
// Gemma 4 Attention (supports both sliding-window and full)
// ---------------------------------------------------------------------------

/// Attention layer that handles both sliding-window (local) and full (global) modes.
/// Global layers can use K=V optimization and different head dimensions.
///
/// Config fields (non-Module) are stored separately from weight fields.
pub struct Gemma4Attention<B: Backend> {
    // Weights
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub q_norm: RmsNorm<B>,
    pub k_norm: RmsNorm<B>,
    pub v_norm: RmsNorm<B>,
    // Config (plain values)
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub layer_type: LayerType,
    pub sliding_window: usize,
    pub k_eq_v: bool,
    pub softcap: f64,
    pub rope_dim: usize,
    /// If true, this layer shares KV from a source layer (skip k_proj/v_proj).
    pub is_kv_shared: bool,
}

impl<B: Backend> Gemma4Attention<B> {
    pub fn new(config: &Gemma4TextConfig, layer_idx: usize, device: &B::Device) -> Self {
        let layer_type = config.layer_type(layer_idx);
        let (n_kv_heads, head_dim) = match layer_type {
            LayerType::SlidingAttention => (config.num_key_value_heads, config.head_dim),
            LayerType::FullAttention => (config.global_kv_heads(), config.global_head_dim()),
        };
        let q_dim = config.num_attention_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        Self {
            q_proj: LinearConfig::new(config.hidden_size, q_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(q_dim, config.hidden_size)
                .with_bias(false)
                .init(device),
            q_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            v_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            n_heads: config.num_attention_heads,
            n_kv_heads,
            head_dim,
            layer_type,
            sliding_window: config.sliding_window,
            k_eq_v: config.attention_k_eq_v && layer_type == LayerType::FullAttention,
            softcap: config.final_logit_softcapping,
            rope_dim: config.rope_dim(layer_type),
            is_kv_shared: layer_idx >= config.first_shared_kv_layer(),
        }
    }

    /// Forward pass with KV cache.
    /// For KV-shared layers, cache must be pre-populated with the source layer's KV.
    /// `lora` optionally carries trainable adapters plus this layer's index (for
    /// key derivation); `None` is the plain inference path, bit-identical to before.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut crate::models::gemma::layers::KvCache<B>,
        attn_mask: Option<&Tensor<B, 4>>,
        lora: Option<(&LoraWeights<B>, usize)>,
    ) -> Tensor<B, 3> {
        let [batch, new_len, _] = x.dims();
        // Offset = absolute position of the first new Q token.
        //   Non-shared: cache.seq_len() is pre-update, so = prefix length.
        //   KV-shared: cache has just been overwritten with source's
        //   POST-update state (source ran earlier in the stack and appended
        //   new_len already), so we subtract new_len to get the same pre-
        //   update prefix length. Previously hardcoded to 0, which happens
        //   to be correct only during the very first prefill
        //   (cache_len == new_len).
        let offset = if self.is_kv_shared {
            cache.seq_len().saturating_sub(new_len)
        } else {
            cache.seq_len()
        };

        // Per-projection LoRA keys, built only when adapters ride along.
        let lora_w = lora.map(|(l, _)| l);
        let keys = lora.map(|(_, i)| {
            [
                format!("layers.{i}.self_attn.q_proj"),
                format!("layers.{i}.self_attn.k_proj"),
                format!("layers.{i}.self_attn.v_proj"),
                format!("layers.{i}.self_attn.o_proj"),
            ]
        });
        let key = |j: usize| keys.as_ref().map_or("", |k| k[j].as_str());

        // Project Q (always needed)
        let q = maybe_lora(&self.q_proj, x.clone(), lora_w, key(0));
        let q = q
            .reshape([batch, new_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let q = self.q_norm.forward(q);
        let q = rope.apply(q, offset);

        // K/V: compute from input OR reuse from pre-populated cache (KV sharing)
        let (full_k, full_v) = if self.is_kv_shared {
            // KV-shared layer: cache was pre-populated by the decoder with source layer's KV.
            // Don't compute k_proj/v_proj — just use the cache directly.
            let k = cache
                .k
                .clone()
                .expect("KV-shared layer needs pre-populated cache");
            let v = cache
                .v
                .clone()
                .expect("KV-shared layer needs pre-populated cache");
            (k, v)
        } else {
            // Normal layer: compute K/V from input. With K=V (full-attention
            // layers of the dense 12B/31B) the checkpoint carries no v_proj —
            // the loader fills v_proj with the k_proj weights — and k is
            // defined as v, so running k_proj would be dead compute whose
            // output is discarded (and whose LoRA adapter could never
            // receive a gradient). Skip it outright.
            let v = maybe_lora(&self.v_proj, x.clone(), lora_w, key(2));
            let k = if self.k_eq_v {
                v.clone()
            } else {
                maybe_lora(&self.k_proj, x, lora_w, key(1))
            };

            let k = k
                .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
                .swap_dims(1, 2);
            let v = v
                .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
                .swap_dims(1, 2);

            let k = self.k_norm.forward(k);
            let v = self.v_norm.forward(v);

            let k = rope.apply(k, offset);

            // Update KV cache
            cache.update(k, v)
        };

        // GQA: expand KV heads
        let n_rep = self.n_heads / self.n_kv_heads;
        let full_k = Self::repeat_kv(full_k, n_rep);
        let full_v = Self::repeat_kv(full_v, n_rep);

        // Manual attention with scale=1.0 (Gemma 4: QKV-norm handles scaling)
        // Using manual implementation to avoid kernel scale issues.
        let out = {
            // QK^T (no 1/sqrt(d) scaling — QKV norms handle normalization)
            let attn_scores = q.matmul(full_k.swap_dims(2, 3));

            // Prefill: caller-provided mask (e.g. causal + bidirectional
            // image-block unmask for Gemma 4 sliding layers), else plain
            // causal. Decode (new_len == 1) needs no mask.
            let attn_scores = if let Some(mask) = attn_mask {
                attn_scores + mask.clone()
            } else if new_len > 1 {
                Self::causal_mask(attn_scores, new_len, offset)
            } else {
                attn_scores
            };

            let attn_weights = burn::tensor::activation::softmax(attn_scores, 3);
            attn_weights.matmul(full_v)
        };

        // Reshape back and project output
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, new_len, self.n_heads * self.head_dim]);
        maybe_lora(&self.o_proj, out, lora_w, key(3))
    }

    /// Build a sliding window + causal attention mask.
    fn sliding_window_mask<B2: Backend>(
        &self,
        new_len: usize,
        total_len: usize,
        offset: usize,
        device: &B2::Device,
    ) -> Tensor<B2, 4> {
        let rows: Vec<Tensor<B2, 2>> = (0..new_len)
            .map(|i| {
                let query_pos = offset + i;
                let window_start = query_pos.saturating_sub(self.sliding_window - 1);
                let row: Vec<f32> = (0..total_len)
                    .map(|j| {
                        if j <= query_pos && j >= window_start {
                            0.0
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect();
                Tensor::<B2, 1>::from_floats(&row[..], device).unsqueeze::<2>()
            })
            .collect();
        Tensor::<B2, 2>::cat(rows, 0).reshape([1, 1, new_len, total_len])
    }

    /// Causal attention mask: future positions get -inf.
    fn causal_mask(attn: Tensor<B, 4>, seq_len: usize, _offset: usize) -> Tensor<B, 4> {
        if seq_len <= 1 {
            return attn;
        }
        let device = attn.device();
        let rows: Vec<Tensor<B, 2>> = (0..seq_len)
            .map(|i| {
                let row: Vec<f32> = (0..seq_len)
                    .map(|j| if j <= i { 0.0 } else { f32::NEG_INFINITY })
                    .collect();
                Tensor::<B, 1>::from_floats(&row[..], &device).unsqueeze::<2>()
            })
            .collect();
        let mask = Tensor::<B, 2>::cat(rows, 0).reshape([1, 1, seq_len, seq_len]);
        attn + mask
    }

    fn repeat_kv(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        if n_rep == 1 {
            return x;
        }
        let [batch, n_kv_heads, seq_len, head_dim] = x.dims();
        x.unsqueeze_dim::<5>(2)
            .expand([batch, n_kv_heads, n_rep, seq_len, head_dim])
            .reshape([batch, n_kv_heads * n_rep, seq_len, head_dim])
    }
}

// ---------------------------------------------------------------------------
// Gemma 4 MoE — router + experts (26B-A4B)
// ---------------------------------------------------------------------------

/// Token-level router. Matches Gemma4TextRouter (modeling_gemma4.py:1289).
///
///   x = RMSNorm(x, no_scale)        # [B*S, H]
///   x = x * scale[H] / sqrt(H)      # learned + fixed rescaling
///   scores = proj(x)                # [B*S, E]
///   probs  = softmax(scores, dim=-1)
///   top_k_w, top_k_i = topk(probs, k)
///   top_k_w /= top_k_w.sum(-1)      # re-normalize chosen subset
///   top_k_w *= per_expert_scale[top_k_i]
pub struct Gemma4Router<B: Backend> {
    pub norm: RmsNorm<B>,               // with_scale=False
    pub proj: Linear<B>,                // [H, E], no bias
    pub scale: Tensor<B, 1>,            // [H]
    pub per_expert_scale: Tensor<B, 1>, // [E]
    pub inv_sqrt_hidden: f32,
    pub top_k: usize,
    pub num_experts: usize,
}

impl<B: Backend> Gemma4Router<B> {
    /// Returns (top_k_weights [B*S, K], top_k_index [B*S, K] as i32).
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Vec<Vec<usize>>) {
        let [bs, h] = x.dims();
        let h_normed = self.norm.forward(x);
        let device = h_normed.device();
        let h_scaled = h_normed
            * self
                .scale
                .clone()
                .reshape([1, h])
                .mul_scalar(self.inv_sqrt_hidden);
        let scores = self.proj.forward(h_scaled); // [B*S, E]
        let probs = burn::tensor::activation::softmax(scores, 1);

        // Pull probs to host and pick top-k per row (Burn lacks a native top_k on D=2).
        let probs_host: Vec<f32> = probs.to_data().to_vec().unwrap();
        let k = self.top_k;
        let e = self.num_experts;
        let per_expert: Vec<f32> = self.per_expert_scale.clone().to_data().to_vec().unwrap();
        let mut top_k_weights = vec![0.0f32; bs * k];
        let mut top_k_index: Vec<Vec<usize>> = vec![Vec::with_capacity(k); bs];
        for row in 0..bs {
            let base = row * e;
            let mut idx: Vec<(usize, f32)> = (0..e).map(|i| (i, probs_host[base + i])).collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let picks = &idx[..k];
            let sum: f32 = picks.iter().map(|(_, p)| *p).sum();
            for (j, (i, p)) in picks.iter().enumerate() {
                let w = (p / sum) * per_expert[*i];
                top_k_weights[row * k + j] = w;
                top_k_index[row].push(*i);
            }
        }
        let weights = Tensor::<B, 1>::from_floats(&top_k_weights[..], &device).reshape([bs, k]);
        (weights, top_k_index)
    }
}

/// Stacked expert weights — one fused gate_up plus one down per expert.
pub struct Gemma4Experts<B: Backend> {
    pub gate_up_proj: Tensor<B, 3>, // [E, 2*I, H]
    pub down_proj: Tensor<B, 3>,    // [E, H, I]
    pub hidden_size: usize,
    pub intermediate_size: usize, // moe_intermediate_size
    pub num_experts: usize,
}

impl<B: Backend> Gemma4Experts<B> {
    /// Input:  x [B*S, H], top_k_weights [B*S, K], top_k_index host-side.
    /// Output: [B*S, H]
    ///
    /// Python groups tokens by expert via one-hot then loops each expert
    /// that was hit, running a single batched matmul per expert. We do the
    /// same — extract the rows assigned to each expert, apply the gate_up
    /// (fused gate || up), GELU(gate) * up, down-project, multiply by the
    /// token's weight for that expert, then scatter-add back.
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        top_k_weights: Tensor<B, 2>,
        top_k_index: &[Vec<usize>],
    ) -> Tensor<B, 2> {
        let [bs, h] = x.dims();
        let device = x.device();
        let _k = top_k_weights.dims()[1];

        // Pull tokens + routing to host for grouping.
        let x_host: Vec<f32> = x.to_data().to_vec().unwrap();
        let w_host: Vec<f32> = top_k_weights.to_data().to_vec().unwrap();

        // expert_idx → Vec<(token_idx, pos_in_topk, weight)>
        let mut per_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.num_experts];
        for (tok, picks) in top_k_index.iter().enumerate() {
            for (j, &exp) in picks.iter().enumerate() {
                per_expert[exp].push((tok, w_host[tok * picks.len() + j]));
            }
        }

        let i = self.intermediate_size;
        let mut out = vec![0.0f32; bs * h];
        for exp in 0..self.num_experts {
            let tokens = &per_expert[exp];
            if tokens.is_empty() {
                continue;
            }
            let t = tokens.len();

            // Gather the token rows assigned to this expert.
            let mut src = vec![0.0f32; t * h];
            for (row, (tok, _)) in tokens.iter().enumerate() {
                src[row * h..row * h + h].copy_from_slice(&x_host[tok * h..tok * h + h]);
            }
            let src_t = Tensor::<B, 1>::from_floats(&src[..], &device).reshape([t, h]);

            // gate_up_proj[exp] has shape [2I, H]; matmul via [t, H] @ [H, 2I].
            let gu = self
                .gate_up_proj
                .clone()
                .slice([exp..exp + 1, 0..2 * i, 0..h])
                .reshape([2 * i, h])
                .swap_dims(0, 1); // [H, 2I]
            let gu_out = src_t.matmul(gu); // [t, 2I]
            // Chunk into (gate, up) along last dim.
            let gate = gu_out.clone().slice([0..t, 0..i]);
            let up = gu_out.slice([0..t, i..2 * i]);
            let acted = burn::tensor::activation::gelu_approximate(gate) * up;

            // down_proj[exp] has shape [H, I]; matmul via [t, I] @ [I, H].
            let dp = self
                .down_proj
                .clone()
                .slice([exp..exp + 1, 0..h, 0..i])
                .reshape([h, i])
                .swap_dims(0, 1); // [I, H]
            let acted_out = acted.matmul(dp); // [t, H]
            let acted_host: Vec<f32> = acted_out.to_data().to_vec().unwrap();

            // Scatter-add weighted output back into [BS, H] buffer.
            for (row, (tok, w)) in tokens.iter().enumerate() {
                let src_base = row * h;
                let dst_base = tok * h;
                for d in 0..h {
                    out[dst_base + d] += acted_host[src_base + d] * w;
                }
            }
        }

        Tensor::<B, 1>::from_floats(&out[..], &device).reshape([bs, h])
    }
}

// ---------------------------------------------------------------------------
// Gemma 4 MLP (GELU with optional double-wide)
// ---------------------------------------------------------------------------

/// Gated MLP with GELU (pytorch tanh approximation).
/// Double-wide mode (E2B): intermediate_size is doubled.
pub struct Gemma4MLP<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Gemma4MLP<B> {
    pub fn new(config: &Gemma4TextConfig, device: &B::Device) -> Self {
        let intermediate = config.intermediate_size;
        Self {
            gate_proj: LinearConfig::new(config.hidden_size, intermediate)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(config.hidden_size, intermediate)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(intermediate, config.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }

    /// `lora` optionally carries trainable adapters plus this layer's index;
    /// `None` is the plain inference path, bit-identical to before.
    pub fn forward(&self, x: Tensor<B, 3>, lora: Option<(&LoraWeights<B>, usize)>) -> Tensor<B, 3> {
        let lora_w = lora.map(|(l, _)| l);
        let keys = lora.map(|(_, i)| {
            [
                format!("layers.{i}.mlp.gate_proj"),
                format!("layers.{i}.mlp.up_proj"),
                format!("layers.{i}.mlp.down_proj"),
            ]
        });
        let key = |j: usize| keys.as_ref().map_or("", |k| k[j].as_str());

        let gate = burn::tensor::activation::gelu_approximate(maybe_lora(
            &self.gate_proj,
            x.clone(),
            lora_w,
            key(0),
        ));
        let up = maybe_lora(&self.up_proj, x, lora_w, key(1));
        maybe_lora(&self.down_proj, gate * up, lora_w, key(2))
    }
}

// ---------------------------------------------------------------------------
// Per-Layer Embedding (PLE) — E2B, E4B only
// ---------------------------------------------------------------------------

/// Per-Layer Embedding (PLE): per-token signal added to each layer's input.
///
/// Actual structure from weights:
///   - gate: Linear [ple_dim, hidden_size] — context-dependent gating
///   - projection: Linear [hidden_size, ple_dim] — project back to hidden
///   - post_norm: RmsNorm [hidden_size]
///   - layer_scalar: f32 — per-layer residual scaling
///
/// The shared embedding table lives on the model (not per-layer).
pub struct PerLayerInput<B: Backend> {
    /// Per-layer slice of the shared PLE embedding [vocab_size, ple_dim].
    pub embed_slice: Tensor<B, 2>,
    pub gate: Linear<B>,
    pub projection: Linear<B>,
    pub post_norm: RmsNorm<B>,
    pub layer_scalar: f32,
}

impl<B: Backend> PerLayerInput<B> {
    /// Apply PLE matching HuggingFace Gemma4TextDecoderLayer exactly:
    ///   gate = GELU(gate_proj(x))  (hidden→ple_dim)
    ///   h = gate * per_layer_input  (element-wise in ple_dim)
    ///   h = projection(h)  (ple_dim→hidden)
    ///   h = post_norm(h)
    ///   x = (residual + h) * layer_scalar
    pub fn forward(&self, x: Tensor<B, 3>, per_layer_input: &Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();

        // Gate: GELU(gate(x)) — maps hidden→ple_dim
        let gate_signal = burn::tensor::activation::gelu_approximate(self.gate.forward(x));
        let gated = gate_signal * per_layer_input.clone();

        // Project back: ple_dim→hidden, then norm
        let projected = self.projection.forward(gated);
        let normed = self.post_norm.forward(projected);

        // Residual — the overall per-layer scalar is applied at the
        // Gemma4DecoderLayer level instead (Python does it there, not
        // here), so E2B keeps its byte-exact parity and 31B picks up the
        // correct scaling in one place.
        residual + normed
    }
}

// ---------------------------------------------------------------------------
// Gemma 4 Transformer Layer
// ---------------------------------------------------------------------------

/// A single Gemma 4 decoder layer: PLE + attention + MLP with pre/post norms.
///
/// Actual norm structure from weights:
///   input_layernorm → attention → post_attention_layernorm
///   pre_feedforward_layernorm → MLP → post_feedforward_layernorm
///   post_per_layer_input_norm (PLE only)
pub struct Gemma4DecoderLayer<B: Backend> {
    pub input_layernorm: RmsNorm<B>,
    pub attention: Gemma4Attention<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub pre_feedforward_layernorm: RmsNorm<B>,
    pub mlp: Gemma4MLP<B>,
    pub post_feedforward_layernorm: RmsNorm<B>,
    pub ple: Option<PerLayerInput<B>>,
    /// MoE path alongside the dense MLP (26B-A4B). When set:
    ///   h_1 = post_ffn_ln_1(mlp(pre_ffn_ln(residual)))
    ///   h_2 = post_ffn_ln_2(experts(pre_ffn_ln_2(residual)))
    ///   h   = post_ffn_ln(h_1 + h_2) + residual
    pub moe: Option<Gemma4MoeBlock<B>>,
    /// Per-layer output scale (Gemma4TextDecoderLayer.layer_scalar in Python).
    /// Always applied at the very end of the layer. For E2B this is 1.0;
    /// 31B/larger variants have per-layer learned values (often <1).
    pub layer_scalar: f32,
}

/// Bundle of extra norms + router + experts for a MoE decoder layer.
pub struct Gemma4MoeBlock<B: Backend> {
    pub router: Gemma4Router<B>,
    pub experts: Gemma4Experts<B>,
    pub post_ffn_norm_1: RmsNorm<B>,
    pub pre_ffn_norm_2: RmsNorm<B>,
    pub post_ffn_norm_2: RmsNorm<B>,
}

impl<B: Backend> Gemma4DecoderLayer<B> {
    /// Forward pass matching HuggingFace Gemma4TextDecoderLayer exactly.
    /// `lora` optionally carries trainable adapters plus this layer's index;
    /// `None` is the plain inference path, bit-identical to before.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut crate::models::gemma::layers::KvCache<B>,
        ple_embed: Option<&Tensor<B, 3>>,
        attn_mask: Option<&Tensor<B, 4>>,
        lora: Option<(&LoraWeights<B>, usize)>,
    ) -> Tensor<B, 3> {
        // Attention block: norm → attn → post-norm → residual
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let h = self.attention.forward(h, rope, cache, attn_mask, lora);
        let h = self.post_attention_layernorm.forward(h);
        let mut x = residual + h;

        // MLP / MoE block.
        let residual = x.clone();
        let h_mlp = self
            .mlp
            .forward(self.pre_feedforward_layernorm.forward(x), lora);

        let combined = if let Some(moe) = &self.moe {
            // Dense MLP output normed once, MoE output normed independently,
            // then summed. Python's Gemma4TextDecoderLayer:1384-1396.
            let h1 = moe.post_ffn_norm_1.forward(h_mlp);

            let [b, s, hidden] = residual.dims();
            let flat = residual.clone().reshape([b * s, hidden]);
            let flat_normed = moe.pre_ffn_norm_2.forward(flat.clone());
            let (top_k_w, top_k_i) = moe.router.forward(flat_normed.clone());
            let moe_out = moe.experts.forward(flat_normed, top_k_w, &top_k_i);
            let h2 = moe.post_ffn_norm_2.forward(moe_out.reshape([b, s, hidden]));

            h1 + h2
        } else {
            h_mlp
        };
        x = residual + self.post_feedforward_layernorm.forward(combined);

        // PLE: gate(GELU) → multiply embedding → project → norm → residual
        if let (Some(ple), Some(emb)) = (&self.ple, ple_embed) {
            x = ple.forward(x, emb);
        }

        // Per-layer output scaling. Python applies this unconditionally at
        // the end of every Gemma4TextDecoderLayer forward. For E2B this is
        // 1.0 so it's a no-op; 31B has per-layer values like 0.09 at layer 0
        // or 0.04 at layer 59 — skipping it inflates hidden state ~11× and
        // completely breaks generation.
        x * self.layer_scalar
    }
}
