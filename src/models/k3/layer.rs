//! `KimiDecoderLayer` — the whole thing, composed out of the ported primitives.
//!
//! This module contains no arithmetic of its own beyond two reshapes. Every
//! operation is one already gated somewhere else:
//!
//! | sublayer | comes from |
//! |---|---|
//! | the depth-axis mixtures and the snapshot boundary | [`super::attn_res`] |
//! | `input_layernorm`, `post_attention_layernorm`, every `nn.Linear` | [`super::ops`] |
//! | MLA (NoPE, gated output) | [`super::mla`] |
//! | KDA (projections, convolutions, recurrence, gated norm) | [`super::kda_attn`] over [`super::kda`] |
//! | the latent MoE block, its router and its MXFP4 experts | [`super::moe`] |
//! | the dense MLP of layer 0 | [`super::moe::LatentMoe::shared_traced`] — `KimiMLP` and the fused shared-expert MLP are the same module with a different width |
//! | the SiTU activation inside all of those | [`super::situ`] |
//!
//! # The control flow, and why it is not a residual stream
//!
//! A conventional decoder layer is `x + attn(norm(x))` then `x + mlp(norm(x))`.
//! K3's is not, and the difference is the whole architecture:
//!
//! ```text
//! prefix_sum  = layer_in
//! hidden      = AttnRes(prefix_sum, bank)          -- unless the bank is empty
//! if boundary: bank.push(layer_in); prefix_sum = None
//! hidden      = attn(input_layernorm(hidden))
//! prefix_sum  = prefix_sum + hidden                -- or = hidden, if it was reset
//! hidden      = AttnRes(prefix_sum, bank)
//! hidden      = ffn(post_attention_layernorm(hidden))
//! layer_out   = prefix_sum + hidden
//! ```
//!
//! So the "residual" a sublayer reads is not the previous sublayer's output but
//! a per-token softmax mixture over a bank of depth checkpoints — attention
//! along the depth axis. The accumulator resets at every twelfth layer, and the
//! bank is what carries information across the reset. [`DepthMixer`] owns that
//! state, which is why it is a parameter here rather than a field: it spans
//! layers, and a layer that owned its own copy would silently start every layer
//! with an empty bank.
//!
//! # What a caller must still supply
//!
//! Weights. This layer holds tensors, not a loader — the checkpoint is 1.5 TB
//! over 96 shards and how they are streamed is the caller's problem, not the
//! layer's. `k3_layer_gate` streams one layer at a time and materialises only
//! the routed experts the router selected; that is a policy, and it belongs
//! outside the arithmetic.

use burn::prelude::*;

use super::attn_res::{AttnResMix, AttnResParams, DepthMixer};
use super::config::{AttnKind, K3TextConfig};
use super::kda_attn::{KdaAttention, KdaCache, KdaTrace};
use super::mla::{MlaBlock, MlaKvCache, MlaTrace};
use super::moe::{
    BlockTrace, ExpertWeights, LatentMoe, LatentMoeWeights, MoeDims, SharedExpertWeights,
    SharedTrace,
};
use super::ops::{rms_norm, ActRound};

/// The attention sublayer: 69 layers run KDA, 24 run MLA.
///
/// Which one a layer runs is read from the config's `kda_layers` /
/// `full_attn_layers` lists — **one-indexed**, so layer 3 is MLA and layer 4 is
/// KDA. See [`K3TextConfig::attn_kind`].
#[derive(Debug, Clone)]
pub enum K3Attn<B: Backend> {
    Mla(Box<MlaBlock<B>>),
    Kda(Box<KdaAttention<B>>),
}

impl<B: Backend> K3Attn<B> {
    /// Which kind this is, for a caller that needs to build the matching cache.
    pub fn kind(&self) -> AttnKind {
        match self {
            K3Attn::Mla(_) => AttnKind::Mla,
            K3Attn::Kda(_) => AttnKind::Kda,
        }
    }
}

/// The attention sublayer's carried state.
///
/// The two are not interchangeable and they are not the same size: MLA's grows
/// linearly in sequence length (the shipped `KimiDynamicCache` stores the
/// *expanded* `[B, H, S, 192]` keys, not the 512-wide latent), KDA's does not
/// grow at all. That asymmetry is the reason the architecture is 69:24 and not
/// 93:0.
#[derive(Debug, Clone)]
pub enum K3AttnCache<B: Backend> {
    Mla(MlaKvCache<B>),
    Kda(KdaCache),
}

/// Everything the attention sublayer computed.
#[derive(Debug, Clone)]
pub enum K3AttnTrace<B: Backend> {
    Mla(Box<MlaTrace<B>>),
    Kda(Box<KdaTrace<B>>),
}

impl<B: Backend> K3AttnTrace<B> {
    /// `[tokens, hidden]` — the sublayer output, whichever kind ran.
    pub fn out(&self, batch: usize) -> Tensor<B, 2> {
        match self {
            K3AttnTrace::Kda(t) => t.out.clone(),
            K3AttnTrace::Mla(t) => {
                let [b, s, h] = t.out.dims();
                assert_eq!(b, batch, "MLA trace batch");
                t.out.clone().reshape([b * s, h])
            }
        }
    }
}

/// The feed-forward sublayer. Layer 0 is a dense MLP
/// (`first_k_dense_replace = 1`); every layer after it is the latent MoE block.
#[derive(Debug, Clone)]
pub enum K3Ffn<B: Backend> {
    Moe(Box<LatentMoeWeights<B>>),
    Dense(Box<SharedExpertWeights<B>>),
}

/// Everything the feed-forward sublayer computed.
#[derive(Debug, Clone)]
pub enum K3FfnTrace<B: Backend> {
    Moe(Box<BlockTrace<B>>),
    Dense(Box<SharedTrace<B>>),
}

impl<B: Backend> K3FfnTrace<B> {
    /// `[tokens, hidden]` — the sublayer output, whichever kind ran.
    pub fn out(&self) -> Tensor<B, 2> {
        match self {
            K3FfnTrace::Moe(t) => t.out.clone(),
            K3FfnTrace::Dense(t) => t.out.clone(),
        }
    }
}

/// Every boundary inside one decoder layer.
///
/// A gate that only compares `out` against the oracle can tell you a layer is
/// wrong; it cannot tell you *where*. Every field here is a boundary the oracle
/// captured, so a failure localises to one sublayer.
#[derive(Debug, Clone)]
pub struct K3LayerTrace<B: Backend> {
    /// The entry mixture, or `None` at layer 0 where the bank is still empty
    /// and the shipped code skips the call entirely.
    pub entry_mix: Option<AttnResMix<B>>,
    /// `[tokens, hidden]` — what `input_layernorm` was given.
    pub to_attention: Tensor<B, 2>,
    /// `[tokens, hidden]`.
    pub input_layernorm_out: Tensor<B, 2>,
    /// The attention sublayer.
    pub attn: K3AttnTrace<B>,
    /// `[tokens, hidden]` — the accumulator after the attention output was
    /// folded in. This is `prefix_sum` at the point the shipped code takes the
    /// MLP mixture, and it is the slot the mixture's last candidate comes from.
    pub prefix_sum_after_attn: Tensor<B, 2>,
    /// The MLP-side depth mixture.
    pub mlp_mix: AttnResMix<B>,
    /// `[tokens, hidden]`.
    pub post_attention_layernorm_out: Tensor<B, 2>,
    /// The feed-forward sublayer.
    pub ffn: K3FfnTrace<B>,
    /// `[tokens, hidden]` — the layer output, i.e. the accumulator.
    pub out: Tensor<B, 2>,
    /// The snapshot bank *after* this layer, oldest first. Empty entries are
    /// impossible; a boundary layer's own input is the last element.
    pub bank_len: usize,
}

/// One K3 decoder layer.
#[derive(Debug, Clone)]
pub struct K3DecoderLayer<B: Backend> {
    /// Which layer this is. Used only for error messages and for the
    /// `DepthMixer` order assertions — the boundary schedule itself lives in
    /// the mixer, built from the config, so this field cannot disagree with it.
    pub layer_idx: usize,
    pub hidden_size: usize,
    pub rms_norm_eps: f64,
    pub round: ActRound,
    /// `input_layernorm.weight`, `[hidden]`.
    pub input_layernorm: Tensor<B, 1>,
    /// `post_attention_layernorm.weight`, `[hidden]`.
    pub post_attention_layernorm: Tensor<B, 1>,
    /// `self_attention_res_norm` × `self_attention_res_proj`.
    pub sa_res: AttnResParams<B>,
    /// `mlp_res_norm` × `mlp_res_proj`.
    pub mlp_res: AttnResParams<B>,
    pub attn: K3Attn<B>,
    pub ffn: K3Ffn<B>,
    /// The MoE block's arithmetic. Held even for a dense layer, because
    /// `KimiMLP` and the fused shared-expert MLP are the same module and
    /// [`LatentMoe::shared_traced`] is the one transcription of it.
    moe: LatentMoe,
}

impl<B: Backend> K3DecoderLayer<B> {
    /// Assemble a layer. `dims` is the MoE block's shape even when `ffn` is
    /// dense — the dense MLP still needs the SiTU parameters and the rounding
    /// policy, which live there.
    pub fn new(
        layer_idx: usize,
        dims: MoeDims,
        round: ActRound,
        input_layernorm: Tensor<B, 1>,
        post_attention_layernorm: Tensor<B, 1>,
        sa_res: AttnResParams<B>,
        mlp_res: AttnResParams<B>,
        attn: K3Attn<B>,
        ffn: K3Ffn<B>,
    ) -> Self {
        let h = dims.hidden_size;
        assert_eq!(input_layernorm.dims()[0], h, "input_layernorm width");
        assert_eq!(post_attention_layernorm.dims()[0], h, "post_attention_layernorm width");
        assert_eq!(sa_res.hidden(), h, "self_attention_res site width");
        assert_eq!(mlp_res.hidden(), h, "mlp_res site width");
        let rms_norm_eps = dims.rms_norm_eps;
        let moe = match round {
            ActRound::Bf16 => LatentMoe::new(dims),
            ActRound::None => LatentMoe::new_f32(dims),
        };
        Self {
            layer_idx,
            hidden_size: h,
            rms_norm_eps,
            round,
            input_layernorm,
            post_attention_layernorm,
            sa_res,
            mlp_res,
            attn,
            ffn,
            moe,
        }
    }

    /// Whether this layer's config says it should run MLA or KDA.
    pub fn attn_kind(&self) -> AttnKind {
        self.attn.kind()
    }

    /// A zero cache matching this layer's attention kind.
    pub fn new_cache(&self, batch: usize) -> K3AttnCache<B> {
        match &self.attn {
            K3Attn::Mla(_) => K3AttnCache::Mla(MlaKvCache::new()),
            K3Attn::Kda(k) => K3AttnCache::Kda(KdaCache::zeros(k, batch)),
        }
    }

    /// Run the layer.
    ///
    /// `hidden` is `[tokens, hidden_size]` with `tokens = batch · seq` in
    /// row-major `[b, t] -> b·seq + t` order — the flattening the shipped
    /// module does on its own first line, and the layout every oracle array is
    /// stored in.
    ///
    /// `mixer` is advanced in place through all three of its stages; its
    /// internal stage machine will panic rather than let a caller run the
    /// sublayers out of order. `expert` is called once per selected routed
    /// expert; for a dense layer it is never called.
    pub fn forward<F>(
        &self,
        mixer: &mut DepthMixer<B>,
        hidden: Tensor<B, 2>,
        batch: usize,
        cache: &mut K3AttnCache<B>,
        expert: F,
    ) -> K3LayerTrace<B>
    where
        F: FnMut(usize) -> ExpertWeights<B>,
    {
        let [tokens, dh] = hidden.dims();
        assert_eq!(dh, self.hidden_size, "layer input width {dh}");
        assert!(tokens > 0, "layer forward over zero tokens");
        assert!(batch > 0 && tokens % batch == 0, "{tokens} tokens, {batch} sequences");
        assert_eq!(
            mixer.layer(),
            self.layer_idx,
            "layer {} driven with a mixer sitting at layer {}",
            self.layer_idx,
            mixer.layer()
        );
        let seq = tokens / batch;

        // --- entry mixture + snapshot boundary -----------------------------
        let entry = mixer.enter_layer(hidden, &self.sa_res);
        let to_attention = entry.to_attention.clone();

        // --- attention ------------------------------------------------------
        let input_layernorm_out = rms_norm(
            to_attention.clone(),
            &self.input_layernorm,
            self.rms_norm_eps,
            self.round,
        );
        let attn = match (&self.attn, &mut *cache) {
            (K3Attn::Kda(k), K3AttnCache::Kda(c)) => {
                assert_eq!(c.batch(), batch, "KDA cache holds {} sequences", c.batch());
                K3AttnTrace::Kda(Box::new(k.forward(input_layernorm_out.clone(), c)))
            }
            (K3Attn::Mla(m), K3AttnCache::Mla(c)) => {
                let x3 = input_layernorm_out
                    .clone()
                    .reshape([batch, seq, self.hidden_size]);
                let offset = c.len();
                let mask = MlaBlock::<B>::causal_mask(batch, seq, offset + seq, offset, &x3.device());
                K3AttnTrace::Mla(Box::new(m.forward(x3, Some(mask), Some(c))))
            }
            (a, _) => panic!(
                "attention kind {:?} was given the wrong cache at layer {}",
                a.kind(),
                self.layer_idx
            ),
        };

        // --- fold the attention output in, mix for the MLP -------------------
        let mlp_mix = mixer.after_attention(attn.out(batch), &self.mlp_res);
        let prefix_sum_after_attn = mixer
            .accumulator()
            .expect("after_attention leaves the accumulator set")
            .clone();

        // --- feed-forward ----------------------------------------------------
        let post_attention_layernorm_out = rms_norm(
            mlp_mix.out.clone(),
            &self.post_attention_layernorm,
            self.rms_norm_eps,
            self.round,
        );
        let ffn = match &self.ffn {
            K3Ffn::Moe(w) => K3FfnTrace::Moe(Box::new(self.moe.forward_traced(
                post_attention_layernorm_out.clone(),
                w,
                expert,
            ))),
            K3Ffn::Dense(w) => K3FfnTrace::Dense(Box::new(
                self.moe.shared_traced(post_attention_layernorm_out.clone(), w),
            )),
        };

        let out = mixer.after_mlp(ffn.out());

        K3LayerTrace {
            entry_mix: entry.mix,
            to_attention,
            input_layernorm_out,
            attn,
            prefix_sum_after_attn,
            mlp_mix,
            post_attention_layernorm_out,
            ffn,
            out,
            bank_len: mixer.bank_len(),
        }
    }
}

/// Build the AttnRes parameter pair for one call site from the two checkpoint
/// tensors, at their checkpoint ranks.
///
/// A convenience so a caller does not have to remember which of the two is the
/// `[1, hidden]` one — [`AttnResParams::new`] asserts it, but only if it is
/// handed them the right way round.
pub fn attn_res_site<B: Backend>(
    norm_weight: Tensor<B, 1>,
    proj_weight: Tensor<B, 2>,
    cfg: &K3TextConfig,
) -> AttnResParams<B> {
    AttnResParams::new(norm_weight, proj_weight, cfg.rms_norm_eps)
}
