//! PersonaPlex-7B **depth transformer** (moshi `depformer.*` plus its
//! projection / embedding / logit surfaces `depformer_in.{0..15}`,
//! `depformer_text_emb`, `depformer_emb.{0..14}`, `linears.{0..15}`) —
//! Phase "LM part 2" of the port.
//!
//! Per temporal frame the depformer generates the 16 audio codebooks
//! autoregressively (`dep_q = 16`: 8 agent + 8 user-prediction streams).
//! Step `s` of the in-frame sequence (moshi `LMModel.forward_depformer` +
//! `LMGen.depformer_step`):
//!
//! ```text
//! input_s  = depformer_in.{s}(transformer_out)            [1024 ← 4096]
//!          + emb(prev)     s=0: depformer_text_emb(next_text_token)
//!                          s>0: depformer_emb.{s-1}(prev audio token)
//! h_s      = 6 shared-geometry layers with PER-STEP weight sets
//! logits_s = linears.{s}(h_s)                             [2048 ← 1024]
//! ```
//!
//! **Per-step vs shared** (from `modules/transformer.py`, verified against
//! the checkpoint's tensor shapes): per-step ×16 — the attention projections
//! (`self_attn.in_proj_weight [16·3·1024, 1024]` and `out_proj.weight
//! [16·1024, 1024]`, row-sliced by `multi_linear` at the in-frame step
//! offset) and the FFN (`gating.{0..15}`, a `ModuleList` indexed by the same
//! offset). Shared across the 16 steps — only `norm1.alpha` / `norm2.alpha`
//! per layer. Folding the shared norm weight into each of the 16 per-step
//! weight sets (what `layers::DecoderLayer::load` does) is therefore exact.
//!
//! **No positional embedding** (`depformer_pos_emb: "none"` in loaders.py):
//! no sin embedding, no RoPE. The stack still runs through
//! [`crate::models::qwen3tts::layers`] by feeding an IDENTITY RoPE slice
//! (cos = 1, sin = 0): `qk·w·1 + R(qk)·w_rot·0 = qk·w` exactly in f32
//! (the 1/√64 = 2⁻³ attention scale folded in `w` is a power of two, and
//! the zeroed rotate branch adds ±0.0). Zero layer edits, again.
//!
//! **Attention: causal over the in-frame sequence, effective window 15.**
//! Two traps stacked here, both verified empirically against the oracle's
//! `RingKVCache` (and the hard way — the wrong window was THE parity gap):
//! (1) loaders.py ships `depformer_context: 8`, but `LMModel.__init__`
//! OVERRIDES it — `kwargs_dep["context"] = None` (lm.py:321) — so the knob
//! is dead: no `delta < context` mask, ring capacity = `weights_per_step`
//! = 16, and steps 0..=14 attend fully causally. (2) The ring's position
//! labeling has an off-by-one at the wrap: `complete()` assigns the slot at
//! `end_index = end_offset % capacity` the FUTURE position `end_offset`
//! (`delta <= 0` branch), so the moment the ring is full the oldest entry
//! is causally masked — at in-frame step 15, key 0 (the text-conditioned
//! step) is dropped: visible = `{1..=15}`. Both behaviors together are
//! exactly a sliding window of `capacity - 1 = 15`, which
//! `AttnConfig { window: Some(15) }` reproduces over mary's growing cache
//! (softmax-identical: masked keys contribute exactly zero either way).
//! The KV state is fresh per frame (`with depformer.streaming(B)` around
//! each frame in the oracle).
//!
//! The depformer runs at moshi's knobs: full MHA 16 heads × 64, RMS eps 1e-8,
//! no q/k-norm, no LayerScale, and there is NO final norm — `linears.{s}`
//! applies straight to the last residual stream.

use burn::prelude::*;

use super::config as cfg;
use super::sampling::Sampler;
use crate::models::qwen3tts::layers::{AttnConfig, DecoderLayer, Embedding, KvCache, Linear};
use crate::nn::weight_loader::WeightLoader;

/// The depth stack's geometry as `layers::AttnConfig` knobs.
fn attn_config() -> AttnConfig {
    AttnConfig {
        hidden: cfg::DEP_DIM,
        heads: cfg::DEP_HEADS,
        kv_heads: cfg::DEP_HEADS, // full MHA
        head_dim: cfg::DEP_HEAD_DIM,
        rope_theta: cfg::ROPE_THETA, // unused — identity cos/sin fed at forward
        eps: cfg::RMS_EPS,
        // The oracle ring's effective window (see module docs): context is
        // OVERRIDDEN to None (lm.py:321, `depformer_context: 8` is dead) and
        // the RingKVCache wrap quirk masks the oldest key at step 15.
        window: Some(cfg::WEIGHTS_PER_STEP - 1),
        qk_norm: false,
        layer_scale: false,
    }
}

/// First-index-wins argmax (torch.argmax CPU tie behavior).
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Fetch one depformer layer from `loader` and serve the 16 per-step weight
/// sets as transient maps in mary's layer naming convention (key prefix
/// `"l"`), one per step: `in_proj_weight` row-block `[t·3072, (t+1)·3072)`
/// splits q/k/v (each `[1024, 1024]`, NO de-interleave — there is no RoPE),
/// `out_proj` row-block `[t·1024, (t+1)·1024)`, `gating.{t}.linear_in
/// [5632, 1024]` row-splits gate `[0:2816)` / up `[2816:5632)` (gating.py
/// `x.view(B,T,2,-1)`: first half is the SiLU branch), and the SHARED
/// `norm{1,2}.alpha [1,1,1024]` squeeze into every step's map.
fn adapt_layer_steps(loader: &WeightLoader, i: usize) -> Vec<WeightLoader> {
    let src = format!("depformer.layers.{i}");
    let (d, fh) = (cfg::DEP_DIM, cfg::DEP_FFN_HIDDEN);
    let n = cfg::WEIGHTS_PER_STEP;

    let (in_proj, s) = loader.load_f32(&format!("{src}.self_attn.in_proj_weight"));
    assert_eq!(s, vec![n * 3 * d, d], "{src}: in_proj_weight shape");
    let (out_proj, s) = loader.load_f32(&format!("{src}.self_attn.out_proj.weight"));
    assert_eq!(s, vec![n * d, d], "{src}: out_proj shape");
    let mut norms: Vec<(String, Vec<f32>)> = Vec::new();
    for (moshi, mary) in [
        ("norm1", "input_layernorm"),
        ("norm2", "post_attention_layernorm"),
    ] {
        let (a, s) = loader.load_f32(&format!("{src}.{moshi}.alpha"));
        assert_eq!(s, vec![1, 1, d], "{src}: {moshi}.alpha shape");
        norms.push((format!("l.{mary}.weight"), a));
    }

    (0..n)
        .map(|t| {
            let mut map = std::collections::HashMap::new();
            let block = &in_proj[t * 3 * d * d..(t + 1) * 3 * d * d];
            for (j, name) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
                map.insert(
                    format!("l.self_attn.{name}.weight"),
                    (block[j * d * d..(j + 1) * d * d].to_vec(), vec![d, d]),
                );
            }
            map.insert(
                "l.self_attn.o_proj.weight".into(),
                (out_proj[t * d * d..(t + 1) * d * d].to_vec(), vec![d, d]),
            );
            for (name, a) in &norms {
                map.insert(name.clone(), (a.clone(), vec![d]));
            }

            let (gu, s) = loader.load_f32(&format!("{src}.gating.{t}.linear_in.weight"));
            assert_eq!(s, vec![2 * fh, d], "{src}: gating.{t}.linear_in shape");
            map.insert(
                "l.mlp.gate_proj.weight".into(),
                (gu[..fh * d].to_vec(), vec![fh, d]),
            );
            map.insert(
                "l.mlp.up_proj.weight".into(),
                (gu[fh * d..].to_vec(), vec![fh, d]),
            );
            let (down, s) = loader.load_f32(&format!("{src}.gating.{t}.linear_out.weight"));
            assert_eq!(s, vec![d, fh], "{src}: gating.{t}.linear_out shape");
            map.insert("l.mlp.down_proj.weight".into(), (down, vec![d, fh]));

            WeightLoader::Pile(map)
        })
        .collect()
}

/// The depth transformer: 16 per-step weight sets over 6 layers, plus the
/// per-step conditioning projections, prev-token embeddings and logit heads.
/// KV state lives per frame (fresh in [`Self::frame`]), so the struct itself
/// is stateless between frames.
pub struct DepthTransformer<B: Backend> {
    /// `[step 0..16][layer 0..6]` — step-major so one in-frame step walks a
    /// contiguous weight set.
    steps: Vec<Vec<DecoderLayer<B>>>,
    /// `depformer_in.{0..15} [1024, 4096]` — per-step conditioning of
    /// `transformer_out`.
    dep_in: Vec<Linear<B>>,
    /// `depformer_text_emb [32001, 1024]` — prev-token embedding for step 0.
    text_emb: Embedding<B>,
    /// `depformer_emb.{0..14} [2049, 1024]` — prev-token embedding for
    /// steps 1..16 (step s embeds with table s-1).
    audio_emb: Vec<Embedding<B>>,
    /// `linears.{0..15} [2048, 1024]` — per-step logit heads.
    heads: Vec<Linear<B>>,
    /// Identity RoPE slice (cos = 1, sin = 0): depformer_pos_emb is "none".
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
}

impl<B: Backend> DepthTransformer<B> {
    pub fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let cfg_a = attn_config();
        let mut steps: Vec<Vec<DecoderLayer<B>>> = (0..cfg::WEIGHTS_PER_STEP)
            .map(|_| Vec::with_capacity(cfg::DEP_LAYERS))
            .collect();
        for i in 0..cfg::DEP_LAYERS {
            for (t, adapted) in adapt_layer_steps(loader, i).iter().enumerate() {
                steps[t].push(DecoderLayer::load(adapted, "l", cfg_a, device));
            }
        }

        let mut g = std::collections::HashMap::new();
        for t in 0..cfg::WEIGHTS_PER_STEP {
            let (w, s) = loader.load_f32(&format!("depformer_in.{t}.weight"));
            assert_eq!(s, vec![cfg::DEP_DIM, cfg::DIM], "depformer_in.{t} shape");
            g.insert(format!("depformer_in.{t}.weight"), (w, s));
            let (w, s) = loader.load_f32(&format!("linears.{t}.weight"));
            assert_eq!(s, vec![cfg::CARD, cfg::DEP_DIM], "linears.{t} shape");
            g.insert(format!("linears.{t}.weight"), (w, s));
        }
        let (w, s) = loader.load_f32("depformer_text_emb.weight");
        assert_eq!(
            s,
            vec![cfg::TEXT_VOCAB, cfg::DEP_DIM],
            "depformer_text_emb shape"
        );
        g.insert("depformer_text_emb.weight".into(), (w, s));
        for t in 0..cfg::WEIGHTS_PER_STEP - 1 {
            let (w, s) = loader.load_f32(&format!("depformer_emb.{t}.weight"));
            assert_eq!(
                s,
                vec![cfg::AUDIO_VOCAB, cfg::DEP_DIM],
                "depformer_emb.{t} shape"
            );
            g.insert(format!("depformer_emb.{t}.weight"), (w, s));
        }
        let g = WeightLoader::Pile(g);

        let d = cfg::DEP_HEAD_DIM;
        Self {
            steps,
            dep_in: (0..cfg::WEIGHTS_PER_STEP)
                .map(|t| Linear::load(&g, &format!("depformer_in.{t}"), false, device))
                .collect(),
            text_emb: Embedding::load(&g, "depformer_text_emb.weight", device),
            audio_emb: (0..cfg::WEIGHTS_PER_STEP - 1)
                .map(|t| Embedding::load(&g, &format!("depformer_emb.{t}.weight"), device))
                .collect(),
            heads: (0..cfg::WEIGHTS_PER_STEP)
                .map(|t| Linear::load(&g, &format!("linears.{t}"), false, device))
                .collect(),
            cos: Tensor::ones([1, 1, 1, d], device),
            sin: Tensor::zeros([1, 1, 1, d], device),
        }
    }

    /// One temporal frame's depformer pass (moshi `LMGen.depformer_step`,
    /// greedy): `transformer_out` is the temporal stack's post-`out_norm`
    /// hidden `[1, 1, 4096]`, `text_token` the frame's `next_text_token`.
    ///
    /// The prev-token chain follows the oracle's forcing rule per step:
    /// `prev = forced[s]` when the LMGen cache provided the target (prompt /
    /// user-audio phases), else the step's own sampled (greedy) token — or
    /// `teacher[s]` when teacher-forcing against oracle tokens, which pins
    /// every step's INPUT to the oracle trajectory so the 16 logit
    /// comparisons stay independent of upstream argmax flips.
    ///
    /// Returns the 16 tokens (sampled via `sampler` when `Some`, else greedy
    /// argmax) and the 16 × 2048 logit rows.
    pub fn frame(
        &self,
        transformer_out: &Tensor<B, 3>,
        text_token: i64,
        forced: &[Option<i64>; cfg::DEP_Q],
        teacher: Option<&[i64]>,
        mut sampler: Option<&mut Sampler>,
        device: &B::Device,
    ) -> ([i64; cfg::DEP_Q], Vec<Vec<f32>>) {
        let mut caches: Vec<KvCache<B>> = (0..cfg::DEP_LAYERS).map(|_| KvCache::empty()).collect();
        let mut tokens = [0i64; cfg::DEP_Q];
        let mut logits_out = Vec::with_capacity(cfg::DEP_Q);
        let mut prev = text_token;
        for s in 0..cfg::DEP_Q {
            let emb = if s == 0 {
                assert!(
                    (0..cfg::TEXT_VOCAB as i64).contains(&prev),
                    "text token {prev}"
                );
                self.text_emb.weight.clone().narrow(0, prev as usize, 1)
            } else {
                assert!(
                    (0..cfg::AUDIO_VOCAB as i64).contains(&prev),
                    "audio token {prev}"
                );
                self.audio_emb[s - 1]
                    .weight
                    .clone()
                    .narrow(0, prev as usize, 1)
            }
            .reshape([1, 1, cfg::DEP_DIM]);
            let mut h = self.dep_in[s].forward(transformer_out.clone()) + emb;
            for (layer, cache) in self.steps[s].iter().zip(caches.iter_mut()) {
                h = layer.forward(h, &self.cos, &self.sin, cache, device);
            }
            let logits: Vec<f32> = self.heads[s]
                .forward(h)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            tokens[s] = match sampler.as_deref_mut() {
                Some(smp) => smp.token(&logits) as i64,
                None => argmax(&logits) as i64,
            };
            prev = forced[s].unwrap_or_else(|| teacher.map_or(tokens[s], |t| t[s]));
            logits_out.push(logits);
        }
        (tokens, logits_out)
    }
}
