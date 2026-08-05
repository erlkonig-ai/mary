//! Kimi-K3 configuration, parsed from the checkpoint's `config.json`.
//!
//! K3 is a vision-language model: a ViT tower and a patch-merger projector feed
//! a `kimi_linear` text decoder. The decoder is the interesting part — 93 layers
//! that alternate KDA (linear/recurrent attention) with MLA (full attention),
//! and whose MLP is a 896-expert MoE from layer 1 onwards.
//!
//! # The layer-index base trap
//!
//! `linear_attn_config.kda_layers` and `linear_attn_config.full_attn_layers` are
//! **1-based**, while `first_k_dense_replace` is compared against a **0-based**
//! index. Both conventions live in the same shipped file, three lines apart in
//! spirit, and getting the attention one wrong silently linearises the wrong 69
//! layers — a model that still runs and still emits fluent garbage. The evidence
//! that the attention lists are 1-based, all three independent:
//!
//! 1. `configuration_kimi_k3.py::KimiLinearConfig::is_kda_layer` reads
//!    `(layer_idx + 1) in self.linear_attn_config["kda_layers"]`, and its caller
//!    (`modeling_kimi_linear.py::KimiDecoderLayer`) is constructed from
//!    `range(config.num_hidden_layers)`, i.e. 0-based.
//! 2. `full_attn_layers` ends `[..., 92, 93]`. With `num_hidden_layers = 93` the
//!    0-based indices stop at 92, so a 0-based reading would name a layer that
//!    does not exist.
//! 3. The shipped weights settle it: the layers carrying `self_attn.q_a_proj`
//!    (MLA-only) are exactly `{x - 1 | x in full_attn_layers}` and the layers
//!    carrying `self_attn.A_log` (KDA-only) are exactly `{x - 1 | x in
//!    kda_layers}`. Verified over all 93 layers by `k3_layout_gate`.
//!
//! [`K3TextConfig::attn_kind`] is the single place that applies the shift, and
//! [`K3TextConfig::validate`] refuses a config whose shifted lists do not
//! partition `0..num_hidden_layers` — so a base error is a hard error at parse
//! time rather than a quiet mislinearisation at inference time.

use serde::Deserialize;

/// Which attention a decoder layer runs.
///
/// There is no third option: the two lists partition the layers (checked in
/// [`K3TextConfig::validate`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttnKind {
    /// Kimi Delta Attention — the gated-delta linear recurrence, with kernel-4
    /// short convolutions on q/k/v. 69 of the 93 layers.
    Kda,
    /// Multi-head Latent Attention — full attention over a 512-wide latent KV
    /// cache. 24 of the 93 layers, and NoPE: `mla_use_nope` is true and
    /// `rotary_emb` is `None`, so every bit of position information in the model
    /// arrives through the KDA recurrence and its short convolutions.
    Mla,
}

/// The `linear_attn_config` block: everything the KDA layers need, plus the two
/// layer lists that decide which layers are KDA at all.
#[derive(Debug, Clone, Deserialize)]
pub struct LinearAttnConfig {
    /// 1-based indices of the full-attention (MLA) layers. See the module docs.
    pub full_attn_layers: Vec<usize>,
    /// 1-based indices of the linear-attention (KDA) layers. See the module docs.
    pub kda_layers: Vec<usize>,
    /// Floor on the log-space forget gate. When set, the fla kernel switches
    /// from `-exp(A_log) * softplus(g + dt_bias)` to
    /// `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, which bounds the
    /// per-step retention below at `exp(lower_bound)`.
    pub gate_lower_bound: Option<f32>,
    /// Width of one KDA head (also the rank of the low-rank decay gate).
    pub head_dim: usize,
    /// Number of KDA heads.
    pub num_heads: usize,
    /// Width of the causal depthwise convolution on q, k and v.
    pub short_conv_kernel_size: usize,
    /// Whether the **output** gate is full rank (`g_proj`) rather than low-rank
    /// (`g_a_proj` then `g_b_proj`). This does not describe the decay gate,
    /// which is low-rank (`f_a_proj` then `f_b_proj`) regardless.
    pub use_full_rank_gate: bool,
}

impl LinearAttnConfig {
    /// Width of the fused q/k/v/gate projections: every KDA head, concatenated.
    pub fn proj_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// The `text_config` block — the `kimi_linear` decoder.
#[derive(Debug, Clone, Deserialize)]
pub struct K3TextConfig {
    /// Model width.
    pub hidden_size: usize,
    /// FFN width of the dense layers (only layer 0 here).
    pub intermediate_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Number of MLA query heads.
    pub num_attention_heads: usize,
    /// Number of MLA key/value heads. Equal to `num_attention_heads` — MLA
    /// shares one latent cache rather than grouping heads.
    pub num_key_value_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon, shared by every norm in the decoder.
    pub rms_norm_eps: f64,
    /// Longest position the model was trained to address.
    pub max_position_embeddings: usize,

    /// Layers `0..first_k_dense_replace` keep a dense MLP; the rest are MoE.
    /// Compared against the **0-based** layer index — unlike the attention
    /// lists. See the module docs.
    pub first_k_dense_replace: usize,
    /// Only every `moe_layer_freq`-th layer is MoE. 1 here, so the test reduces
    /// to `layer >= first_k_dense_replace`.
    pub moe_layer_freq: usize,

    /// Number of routed experts per MoE layer.
    pub num_experts: usize,
    /// Number of routed experts activated per token.
    pub num_experts_per_token: usize,
    /// Number of always-on shared experts, fused into one wide MLP.
    pub num_shared_experts: Option<usize>,
    /// FFN width of a single expert.
    pub moe_intermediate_size: usize,
    /// Width of the latent the routed experts operate in. The MoE block projects
    /// `hidden_size` down to this before routing and back up after, so the
    /// experts are narrower than the residual stream.
    pub routed_expert_hidden_size: Option<usize>,
    /// Whether the routed-expert latent is RMSNormed.
    pub latent_moe_use_norm: bool,
    /// Whether the top-k routing weights are renormalised to sum to one.
    pub moe_renormalize: bool,
    /// Router activation. `"sigmoid"` here, not softmax.
    pub moe_router_activation_func: String,
    /// Top-k selection algorithm. `"noaux_tc"` — grouped top-k biased by a
    /// trained `e_score_correction_bias` instead of an auxiliary loss.
    pub topk_method: String,
    /// Number of expert groups for grouped top-k.
    pub num_expert_group: usize,
    /// Number of groups kept by grouped top-k.
    pub topk_group: usize,
    /// Scale applied to the combined routed-expert output.
    pub routed_scaling_factor: f64,

    /// Rank of the MLA query down-projection. `None` means an undecomposed
    /// `q_proj`; K3 sets it, so the MLA layers carry `q_a_proj`/`q_b_proj`.
    pub q_lora_rank: Option<usize>,
    /// Width of the MLA latent KV cache.
    pub kv_lora_rank: usize,
    /// Per-head width of the position-independent part of the MLA query/key.
    pub qk_nope_head_dim: usize,
    /// Per-head width of the rotary part of the MLA query/key. Allocated in the
    /// projections but never rotated — see `mla_use_nope`.
    pub qk_rope_head_dim: usize,
    /// Per-head width of the MLA value.
    pub v_head_dim: usize,
    /// Whether MLA runs without positional encoding. True here, and
    /// `KimiMLAAttention.__init__` asserts it.
    pub mla_use_nope: bool,
    /// Whether MLA has a sigmoid output gate (`g_proj`).
    pub mla_use_output_gate: bool,

    /// Depth period of the AttnRes checkpoint bank: every layer whose index is a
    /// multiple of this snapshots the running accumulator into the bank and
    /// restarts it. With 93 layers and a period of 12 the bank grows to 8
    /// entries (layers 0, 12, 24, ..., 84).
    pub attn_res_block_size: Option<usize>,

    /// Activation of the expert / MLP FFNs. `"situ"` — a SwiGLU with both
    /// branches soft-clipped.
    pub hidden_act: String,
    /// Soft-clip sharpness of the gated branch of `situ`.
    pub activation_situ_beta: f64,
    /// Soft-clip sharpness of the linear branch of `situ`.
    pub activation_situ_linear_beta: f64,

    /// Whether `lm_head` reuses the embedding matrix. False here — the
    /// checkpoint ships both.
    pub tie_word_embeddings: bool,

    /// KDA hyper-parameters and the layer-kind lists.
    pub linear_attn_config: LinearAttnConfig,
}

impl K3TextConfig {
    /// Which attention layer `layer` (**0-based**) runs.
    ///
    /// This is the single place the 1-based config lists are shifted; see the
    /// module docs for why the shift is there and how it was established.
    /// Panics on an out-of-range layer, and on a layer named by neither list —
    /// both are impossible after [`Self::validate`].
    pub fn attn_kind(&self, layer: usize) -> AttnKind {
        assert!(
            layer < self.num_hidden_layers,
            "layer {layer} out of range for {} layers",
            self.num_hidden_layers,
        );
        let one_based = layer + 1;
        if self.linear_attn_config.kda_layers.contains(&one_based) {
            AttnKind::Kda
        } else if self.linear_attn_config.full_attn_layers.contains(&one_based) {
            AttnKind::Mla
        } else {
            panic!("layer {layer} is in neither kda_layers nor full_attn_layers");
        }
    }

    /// Whether layer `layer` (**0-based**) has a MoE block rather than a dense
    /// MLP. Mirrors `KimiDecoderLayer.__init__`, which compares the 0-based
    /// index directly — no `+ 1`.
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        layer >= self.first_k_dense_replace && layer % self.moe_layer_freq == 0
    }

    /// Whether layer `layer` snapshots the AttnRes accumulator into the
    /// checkpoint bank on entry.
    pub fn is_attn_res_checkpoint(&self, layer: usize) -> bool {
        self.attn_res_block_size
            .is_some_and(|period| layer % period == 0)
    }

    /// Number of AttnRes depth checkpoints the bank holds after the last layer.
    pub fn attn_res_bank_size(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&l| self.is_attn_res_checkpoint(l))
            .count()
    }

    /// Per-head width of an MLA query/key: the NoPE part plus the (unrotated)
    /// rope part.
    pub fn mla_q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Combined FFN width of the fused shared-expert MLP.
    pub fn shared_expert_intermediate_size(&self) -> usize {
        self.moe_intermediate_size * self.num_shared_experts.unwrap_or(0)
    }

    /// Width the routed experts operate in — the latent if the MoE block has
    /// one, otherwise the residual stream itself.
    pub fn moe_hidden_size(&self) -> usize {
        self.routed_expert_hidden_size.unwrap_or(self.hidden_size)
    }

    /// Reject a config whose invariants the layout code silently depends on.
    ///
    /// The load-bearing one is the partition check: it is what turns a
    /// layer-index base error from a silent mislinearisation into a parse
    /// failure. A 0-based reading of these lists leaves layer 92 unclaimed and
    /// names a layer 93, and both show up here.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.num_hidden_layers;
        let la = &self.linear_attn_config;

        let mut seen = vec![0u32; n];
        for (list, name) in [(&la.kda_layers, "kda_layers"), (&la.full_attn_layers, "full_attn_layers")] {
            for &one_based in list {
                if one_based == 0 || one_based > n {
                    return Err(format!(
                        "{name} names layer {one_based}, outside the 1-based range 1..={n} \
                         (a 0-based reading of these lists is wrong — see K3TextConfig docs)",
                    ));
                }
                seen[one_based - 1] += 1;
            }
        }
        for (layer, &count) in seen.iter().enumerate() {
            if count != 1 {
                return Err(format!(
                    "layer {layer} (0-based) is claimed {count} times by kda_layers/full_attn_layers, \
                     expected exactly 1",
                ));
            }
        }

        if la.proj_dim() != self.num_attention_heads * self.v_head_dim {
            return Err(format!(
                "KDA projection width {} differs from MLA value width {}; o_proj and g_proj are \
                 shared between the two layer kinds and would need separate shapes",
                la.proj_dim(),
                self.num_attention_heads * self.v_head_dim,
            ));
        }
        if self.moe_hidden_size() % 32 != 0 {
            return Err(format!(
                "routed-expert width {} is not a multiple of the MXFP4 block size 32",
                self.moe_hidden_size(),
            ));
        }
        if self.moe_intermediate_size % 32 != 0 {
            return Err(format!(
                "expert FFN width {} is not a multiple of the MXFP4 block size 32",
                self.moe_intermediate_size,
            ));
        }
        Ok(())
    }
}

/// The `vision_config` block — the ViT tower and the patch-merger projector.
#[derive(Debug, Clone, Deserialize)]
pub struct K3VisionConfig {
    /// Side length of a square patch, in pixels.
    pub patch_size: usize,
    /// ViT model width.
    pub vt_hidden_size: usize,
    /// ViT FFN width.
    pub vt_intermediate_size: usize,
    /// Number of ViT blocks.
    pub vt_num_hidden_layers: usize,
    /// Number of ViT attention heads.
    pub vt_num_attention_heads: usize,
    /// Width of ONE of q/k/v inside a ViT block; `wqkv` emits three of these.
    /// Larger than `vt_hidden_size`, so the ViT attention is not square.
    pub qkv_hidden_size: usize,
    /// Width the projector receives per merged patch group, before the
    /// `merge_kernel_size` fan-in.
    pub mm_hidden_size: usize,
    /// Width the projector emits — the text decoder's `hidden_size`.
    pub text_hidden_size: usize,
    /// Height of the learned position-embedding grid, in patches.
    pub init_pos_emb_height: usize,
    /// Width of the learned position-embedding grid, in patches.
    pub init_pos_emb_width: usize,
    /// Patch-group the projector merges, as `[height, width]`.
    pub merge_kernel_size: Vec<usize>,
}

impl K3VisionConfig {
    /// Width entering the projector: one merged patch group's worth of ViT
    /// output, i.e. `mm_hidden_size` times the merge fan-in.
    pub fn projector_input_dim(&self) -> usize {
        self.mm_hidden_size * self.merge_kernel_size.iter().product::<usize>()
    }
}

/// The whole `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct K3Config {
    /// Beginning-of-sequence token.
    pub bos_token_id: u32,
    /// End-of-sequence token.
    pub eos_token_id: u32,
    /// Padding token.
    pub pad_token_id: u32,
    /// Placeholder token whose embedding the vision features replace.
    pub media_placeholder_token_id: u32,
    /// The `kimi_linear` text decoder.
    pub text_config: K3TextConfig,
    /// The ViT tower and projector.
    pub vision_config: K3VisionConfig,
}

impl K3Config {
    /// Parse and validate a `config.json`.
    ///
    /// Validation is not optional here: the layer-kind lists are the one place
    /// this architecture can go wrong invisibly, so a config that fails
    /// [`K3TextConfig::validate`] never becomes a `K3Config`.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let cfg: K3Config = serde_json::from_str(json).map_err(|e| format!("config.json: {e}"))?;
        cfg.text_config.validate()?;
        Ok(cfg)
    }
}
