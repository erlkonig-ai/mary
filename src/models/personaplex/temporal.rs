//! PersonaPlex-7B **temporal transformer** (moshi `transformer.*` plus the
//! embedding / logit surfaces `text_emb`, `emb.{0..15}`, `out_norm`,
//! `text_linear`) — Phase "LM part 1" of the port.
//!
//! The stack REUSES [`crate::models::qwen3tts::layers`] (`DecoderLayer` /
//! `Attention` / `RopeTable` / `KvCache`) — the moshi layer is the same
//! pre-norm block with different knobs, all of which `AttnConfig` already
//! has: full MHA (`kv_heads == heads == 32`), RMSNorm eps `1e-8`, no
//! q/k-norm, no LayerScale, no sliding window (context 3000 never binds in
//! this phase). What differs is *convention*, and both differences are
//! resolved at WEIGHT-ADAPTATION time so the layer itself runs its fused
//! path unchanged:
//!
//! - **Tensor names / fusions** (moshi → mary): `self_attn.in_proj_weight
//!   [12288, 4096]` row-splits into q/k/v at `[0:4096) / [4096:8192) /
//!   [8192:12288)`; `out_proj` → `o_proj`; `norm1.alpha`/`norm2.alpha`
//!   `[1,1,4096]` squeeze into `input_layernorm`/`post_attention_layernorm`
//!   weights `[4096]`; `gating.linear_in [22528, 4096]` row-splits into
//!   `mlp.gate_proj` (rows `[0:11264)`, the SiLU branch) and `mlp.up_proj`
//!   (rows `[11264:22528)`); `gating.linear_out` → `mlp.down_proj`.
//!
//! - **RoPE convention**: moshi rotates INTERLEAVED pairs — `(x[2i],
//!   x[2i+1])` is the complex pair for frequency `θ^(-2i/D)` — while
//!   `layers::RopeTable` implements the split-half rotate_half convention
//!   (pair = `(x[i], x[half+i])`, same frequency `θ^(-2i/D)`). The two are
//!   conjugate under the per-head de-interleave permutation `P`:
//!   `P(x)[i] = x[2i]`, `P(x)[half+i] = x[2i+1]`, giving
//!   `rope_split_half(P·x) = P(rope_interleaved(x))` exactly (same
//!   frequency table, elements just live at permuted indices). Attention
//!   scores are inner products `q·k`, and a shared permutation of q and k
//!   cancels: `(P·q)ᵀ(P·k) = qᵀk`. So applying `P` to the ROWS of the
//!   q_proj and k_proj weight blocks (v untouched) makes the port
//!   numerically EXACT with zero runtime cost and zero layer changes —
//!   the same trick that converts GPT-J-style checkpoints to NeoX-style
//!   kernels. `personaplex_probe` gates this against the moshi oracle.
//!
//! Everything loads through a NAME-ADAPTING step: each moshi layer's tensors
//! are fetched from the weight pile, split/permuted/squeezed on the host,
//! and served to `DecoderLayer::load` as a transient `WeightLoader::Pile`
//! map in mary's qwen3tts naming convention — one layer at a time, so the
//! adaptation working set stays ~1 GiB while the model itself is ~26 GiB f32.
//!
//! CPU-f32 numeric parity comes first (run under `burn_ndarray::NdArray` —
//! see `personaplex_probe`); Metal / q4 throughput is a later increment.

use burn::prelude::*;
use std::collections::HashMap;

use super::config as cfg;
use crate::models::qwen3tts::layers::{
    AttnConfig, DecoderLayer, Embedding, KvCache, Linear, RmsNorm, RopeTable,
};
use crate::nn::weight_loader::WeightLoader;

/// The temporal stack's geometry, expressed as `layers::AttnConfig` knobs.
fn attn_config() -> AttnConfig {
    AttnConfig {
        hidden: cfg::DIM,
        heads: cfg::NUM_HEADS,
        kv_heads: cfg::NUM_HEADS, // full MHA — no GQA anywhere in this fork
        head_dim: cfg::HEAD_DIM,
        rope_theta: cfg::ROPE_THETA,
        eps: cfg::RMS_EPS,
        window: None,
        qk_norm: false,
        layer_scale: false,
    }
}

/// Per-head de-interleave permutation on the ROWS of a `[DIM, DIM]` q or k
/// projection block (see module docs): output row `(h, j)` = input row
/// `(h, 2j)` for `j < 64`, output row `(h, 64 + j)` = input row `(h, 2j+1)`.
fn deinterleave_rows(w: &[f32], cols: usize) -> Vec<f32> {
    let (h, d) = (cfg::NUM_HEADS, cfg::HEAD_DIM);
    let half = d / 2;
    assert_eq!(w.len(), h * d * cols);
    let mut out = vec![0f32; w.len()];
    for head in 0..h {
        for j in 0..half {
            let dst_re = (head * d + j) * cols;
            let src_re = (head * d + 2 * j) * cols;
            out[dst_re..dst_re + cols].copy_from_slice(&w[src_re..src_re + cols]);
            let dst_im = (head * d + half + j) * cols;
            let src_im = (head * d + 2 * j + 1) * cols;
            out[dst_im..dst_im + cols].copy_from_slice(&w[src_im..src_im + cols]);
        }
    }
    out
}

/// Fetch one moshi temporal layer from `loader`, apply the row-splits /
/// squeezes / RoPE-permutation, and serve it back as a transient map in
/// mary's layer naming convention (all keys under the prefix `"l"`).
fn adapt_layer(loader: &WeightLoader, i: usize) -> WeightLoader {
    let src = format!("transformer.layers.{i}");
    let mut map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let (d, fh) = (cfg::DIM, cfg::FFN_HIDDEN);

    // fused q|k|v — split, de-interleave q and k for the RoPE convention
    let (in_proj, s) = loader.load_f32(&format!("{src}.self_attn.in_proj_weight"));
    assert_eq!(s, vec![3 * d, d], "{src}: in_proj_weight shape");
    let d2 = d * d;
    map.insert(
        "l.self_attn.q_proj.weight".into(),
        (deinterleave_rows(&in_proj[..d2], d), vec![d, d]),
    );
    map.insert(
        "l.self_attn.k_proj.weight".into(),
        (deinterleave_rows(&in_proj[d2..2 * d2], d), vec![d, d]),
    );
    map.insert(
        "l.self_attn.v_proj.weight".into(),
        (in_proj[2 * d2..].to_vec(), vec![d, d]),
    );

    let (o, s) = loader.load_f32(&format!("{src}.self_attn.out_proj.weight"));
    assert_eq!(s, vec![d, d], "{src}: out_proj shape");
    map.insert("l.self_attn.o_proj.weight".into(), (o, vec![d, d]));

    // norm alphas [1,1,D] → [D]
    for (moshi, mary) in [
        ("norm1", "input_layernorm"),
        ("norm2", "post_attention_layernorm"),
    ] {
        let (a, s) = loader.load_f32(&format!("{src}.{moshi}.alpha"));
        assert_eq!(s, vec![1, 1, d], "{src}: {moshi}.alpha shape");
        map.insert(format!("l.{mary}.weight"), (a, vec![d]));
    }

    // fused gate|up — rows [0:11264) SiLU (gate), rows [11264:22528) linear (up)
    let (gu, s) = loader.load_f32(&format!("{src}.gating.linear_in.weight"));
    assert_eq!(
        s,
        vec![cfg::FFN_FUSED_IN, d],
        "{src}: gating.linear_in shape"
    );
    map.insert(
        "l.mlp.gate_proj.weight".into(),
        (gu[..fh * d].to_vec(), vec![fh, d]),
    );
    map.insert(
        "l.mlp.up_proj.weight".into(),
        (gu[fh * d..].to_vec(), vec![fh, d]),
    );

    let (down, s) = loader.load_f32(&format!("{src}.gating.linear_out.weight"));
    assert_eq!(s, vec![d, fh], "{src}: gating.linear_out shape");
    map.insert("l.mlp.down_proj.weight".into(), (down, vec![d, fh]));

    WeightLoader::Pile(map)
}

/// The non-layer surfaces: final norm, text logit head, and the 17 input
/// embedding tables, renamed/squeezed into a transient map.
fn adapt_globals(loader: &WeightLoader) -> WeightLoader {
    let mut map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let d = cfg::DIM;

    let (a, s) = loader.load_f32("out_norm.alpha");
    assert_eq!(s, vec![1, 1, d], "out_norm.alpha shape");
    map.insert("out_norm.weight".into(), (a, vec![d]));

    let (w, s) = loader.load_f32("text_linear.weight");
    assert_eq!(s, vec![cfg::TEXT_LOGITS, d], "text_linear shape");
    map.insert("text_linear.weight".into(), (w, s));

    let (w, s) = loader.load_f32("text_emb.weight");
    assert_eq!(s, vec![cfg::TEXT_VOCAB, d], "text_emb shape");
    map.insert("text_emb.weight".into(), (w, s));

    for cb in 0..cfg::N_Q {
        let (w, s) = loader.load_f32(&format!("emb.{cb}.weight"));
        assert_eq!(s, vec![cfg::AUDIO_VOCAB, d], "emb.{cb} shape");
        map.insert(format!("emb.{cb}.weight"), (w, s));
    }

    WeightLoader::Pile(map)
}

/// The 7B temporal transformer with its embedding and logit surfaces, plus
/// per-layer KV caches and the stream position (`offset`) — the streaming
/// state one voice session owns.
///
/// NOTE for the decode loop to come: `layers::KvCache` GROWS (cat per step),
/// mirroring the oracle within its 3000-frame context; moshi's `RingKVCache`
/// only differs once a session exceeds 3000 frames (4 min), at which point
/// the ring overwrite + the `delta < context` mask must be ported. Parity
/// windows and short sessions never get there.
pub struct TemporalTransformer<B: Backend> {
    pub layers: Vec<DecoderLayer<B>>,
    pub caches: Vec<KvCache<B>>,
    pub rope: RopeTable<B>,
    pub out_norm: RmsNorm<B>,
    /// `text_linear [32000, 4096]` — NO extra row (PAD id 3 is reused).
    pub text_linear: Linear<B>,
    /// `text_emb [32001, 4096]` — row 32000 = text BOS-of-stream.
    pub text_emb: Embedding<B>,
    /// `emb.{0..15} [2049, 4096]` — row 2048 = audio initial token.
    pub audio_emb: Vec<Embedding<B>>,
    offset: usize,
}

impl<B: Backend> TemporalTransformer<B> {
    /// Load the 475-tensor LM checkpoint's temporal side from any
    /// `WeightLoader` (the weight pile in production), adapting names and
    /// conventions layer-by-layer (see module docs).
    pub fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let cfg_a = attn_config();
        let mut layers = Vec::with_capacity(cfg::NUM_LAYERS);
        for i in 0..cfg::NUM_LAYERS {
            let adapted = adapt_layer(loader, i);
            layers.push(DecoderLayer::load(&adapted, "l", cfg_a, device));
        }

        let g = adapt_globals(loader);
        let out_norm = RmsNorm::load(&g, "out_norm.weight", cfg::RMS_EPS, device);
        let text_linear = Linear::load(&g, "text_linear", false, device);
        let text_emb = Embedding::load(&g, "text_emb.weight", device);
        let audio_emb = (0..cfg::N_Q)
            .map(|cb| Embedding::load(&g, &format!("emb.{cb}.weight"), device))
            .collect();

        Self {
            layers,
            caches: (0..cfg::NUM_LAYERS).map(|_| KvCache::empty()).collect(),
            rope: RopeTable::new(cfg::ROPE_THETA, cfg::HEAD_DIM, cfg::CONTEXT, device),
            out_norm,
            text_linear,
            text_emb,
            audio_emb,
            offset: 0,
        }
    }

    /// moshi `embed_codes`: one step's 17 tokens (`[text, agent audio 1..8,
    /// user audio 1..8]`, delays already applied by the caller) → the summed
    /// input embedding `[1, 1, 4096]`. Sum order mirrors the oracle (audio
    /// codebooks 0..15 first, text LAST). Token `-1` is moshi's
    /// `zero_token_id`: that stream contributes exactly zero.
    pub fn embed_codes(&self, tokens: &[i64], device: &B::Device) -> Tensor<B, 3> {
        assert_eq!(tokens.len(), cfg::NUM_STREAMS, "expected 17 stream tokens");
        let mut acc: Option<Tensor<B, 2>> = None;
        let add = |acc: &mut Option<Tensor<B, 2>>, row: Tensor<B, 2>| {
            *acc = Some(match acc.take() {
                Some(a) => a + row,
                None => row,
            });
        };
        for cb in 0..cfg::N_Q {
            let t = tokens[1 + cb];
            if t >= 0 {
                assert!(
                    (t as usize) < cfg::AUDIO_VOCAB,
                    "audio token {t} out of range"
                );
                add(
                    &mut acc,
                    self.audio_emb[cb].weight.clone().narrow(0, t as usize, 1),
                );
            }
        }
        let t = tokens[0];
        if t >= 0 {
            assert!(
                (t as usize) < cfg::TEXT_VOCAB,
                "text token {t} out of range"
            );
            add(
                &mut acc,
                self.text_emb.weight.clone().narrow(0, t as usize, 1),
            );
        }
        match acc {
            Some(a) => a.reshape([1, 1, cfg::DIM]),
            None => Tensor::zeros([1, 1, cfg::DIM], device),
        }
    }

    /// moshi `forward_embeddings`: one temporal step (or a prefill window)
    /// of pre-summed input embeddings `[1, L, 4096]` → `(transformer_out,
    /// text_logits)` where `transformer_out` is the post-`out_norm` hidden
    /// `[1, L, 4096]` (the depformer's conditioning input) and `text_logits`
    /// is `[1, L, 32000]`. Advances the KV caches and stream position.
    pub fn forward_embeddings(
        &mut self,
        x: Tensor<B, 3>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let l = x.dims()[1];
        let (cos, sin) = self.rope.slices(self.offset, l);
        let mut h = x;
        for (layer, cache) in self.layers.iter().zip(self.caches.iter_mut()) {
            h = layer.forward(h, &cos, &sin, cache, device);
        }
        self.offset += l;
        let hidden = self.out_norm.forward(h);
        let logits = self.text_linear.forward(hidden.clone());
        (hidden, logits)
    }

    /// moshi `forward_codes` = `forward_embeddings ∘ embed_codes`.
    pub fn forward_codes(
        &mut self,
        tokens: &[i64],
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let x = self.embed_codes(tokens, device);
        self.forward_embeddings(x, device)
    }

    /// Steps consumed so far (the RoPE position of the next step).
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Reset the streaming state (KV caches + position) for a new session.
    pub fn reset(&mut self) {
        self.caches = (0..cfg::NUM_LAYERS).map(|_| KvCache::empty()).collect();
        self.offset = 0;
    }
}
