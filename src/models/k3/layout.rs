//! The K3 checkpoint layout: every safetensors tensor name, and the module slot
//! it fills.
//!
//! This is names, shapes and dtypes only — no tensor data is read here. It is
//! the skeleton a loader hangs weights on, and the thing a port gets silently
//! wrong: a mis-shifted layer index or a mistyped suffix produces a model that
//! loads, runs, and emits fluent nonsense.
//!
//! The mapping is written **twice, in opposite directions**, on purpose:
//!
//! * [`for_each_slot`] walks the config and emits the slots the model needs,
//!   each with the tensor name it expects to find — module tree to checkpoint.
//! * [`Slot::parse`] takes a checkpoint tensor name and recovers the slot it
//!   fills — checkpoint to module tree.
//!
//! Neither shares code with the other, so `k3_layout_gate` can assert they agree
//! on the real headers and get a genuine cross-check rather than a tautology. A
//! typo in a `format!` shows up as an unfilled slot; a typo in the parser shows
//! up as an unmapped tensor.
//!
//! Shapes are always in **checkpoint order** — the row-major `[out, in]` of a
//! `torch.nn.Linear`, not the `[in, out]` a Burn `Linear` wants. Transposition
//! belongs to the loader, not to the layout.

use crate::models::k3::config::{AttnKind, K3Config};

/// On-disk element type of a checkpoint tensor.
///
/// Deliberately mary's own enum and not `safetensors::Dtype`: the layout has to
/// be describable in builds without the `import` feature, since a pile-backed
/// runtime loader wants the same slot list without a safetensors reader
/// anywhere in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// bfloat16 — everything dense: norms, projections, embeddings, the router.
    Bf16,
    /// float32 — the handful of tensors kept in full precision: the KDA decay
    /// parameters, the short-convolution kernels, the KDA output-norm gain, and
    /// the router's score-correction bias.
    F32,
    /// uint8 — the MXFP4 expert weights: two 4-bit codes per byte in
    /// `weight_packed`, one E8M0 exponent per 32-element block in
    /// `weight_scale`.
    U8,
}

impl Dtype {
    /// Bytes per stored element.
    pub fn size(self) -> usize {
        match self {
            Dtype::Bf16 => 2,
            Dtype::F32 => 4,
            Dtype::U8 => 1,
        }
    }
}

/// A checkpoint tensor shape.
///
/// Rank never exceeds 4 (the ViT patch embedding is `[1024, 3, 14, 14]`), so
/// this stays on the stack — the layout enumerates roughly half a million slots
/// and a heap allocation each would be half a million allocations for four
/// numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [usize; 4],
    rank: u8,
}

impl Shape {
    /// Build a shape from up to four dimensions.
    pub fn new(dims: &[usize]) -> Self {
        assert!(dims.len() <= 4, "K3 has no tensors above rank 4");
        let mut d = [0usize; 4];
        d[..dims.len()].copy_from_slice(dims);
        Shape {
            dims: d,
            rank: dims.len() as u8,
        }
    }

    /// The dimensions, outermost first.
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank as usize]
    }

    /// Total element count.
    pub fn numel(&self) -> usize {
        self.dims().iter().product()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.dims())
    }
}

/// Which of an expert's three matrices.
///
/// Named `w1`/`w2`/`w3` because the checkpoint does; the roles are SwiGLU's
/// gate, down and up respectively (`KimiBlockSparseMLP` comments them so).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertMat {
    /// Gate projection, latent to FFN width.
    W1,
    /// Down projection, FFN width back to latent.
    W2,
    /// Up projection, latent to FFN width.
    W3,
}

/// The two halves of an MXFP4-quantised expert matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantPart {
    /// Two E2M1 codes per byte, low nibble first, packed along the input axis.
    Packed,
    /// One E8M0 exponent byte per 32-element block along the input axis.
    Scale,
}

/// A gate/up/down triple, shared by the dense MLP and the fused shared experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlpPart {
    /// SwiGLU gate branch.
    GateProj,
    /// SwiGLU linear branch.
    UpProj,
    /// Output projection.
    DownProj,
}

/// A tensor inside a `block_sparse_moe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoePart {
    /// Router logits projection.
    GateWeight,
    /// Trained per-expert bias added to the router scores before top-k
    /// selection (`noaux_tc`), and dropped again before the weights are used.
    GateScoreCorrectionBias,
    /// Residual stream down into the routed-expert latent.
    RoutedExpertDownProj,
    /// Routed-expert latent back up into the residual stream.
    RoutedExpertUpProj,
    /// RMSNorm on the routed-expert latent.
    RoutedExpertNorm,
    /// The shared experts, fused into one MLP of
    /// `num_shared_experts * moe_intermediate_size`.
    Shared(MlpPart),
    /// One matrix of one routed expert.
    Expert {
        /// Expert index.
        expert: usize,
        /// Which matrix.
        mat: ExpertMat,
        /// Codes or scales.
        part: QuantPart,
    },
}

/// A tensor inside a `self_attn`, of either attention kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttnPart {
    // --- present on every layer, whichever attention it runs ---
    /// Sigmoid output gate. KDA always has one (`use_full_rank_gate`); MLA has
    /// one because `mla_use_output_gate` is set. Same shape either way.
    GProj,
    /// Attention output projection.
    OProj,

    // --- KDA only ---
    /// Query projection.
    QProj,
    /// Key projection.
    KProj,
    /// Value projection.
    VProj,
    /// Causal depthwise convolution on the query.
    QConv1d,
    /// Causal depthwise convolution on the key.
    KConv1d,
    /// Causal depthwise convolution on the value.
    VConv1d,
    /// Log of the per-head decay rate.
    ///
    /// The checkpoint ships `[128]` while `KimiDeltaAttention.__init__`
    /// declares `torch.empty(self.num_heads)` = `[96]`, and the fla kernel
    /// indexes `A_log + i_hv` with `i_hv` a *head* index.
    ///
    /// **RESOLVED BY MEASUREMENT, 2026-08-05 — and the earlier warning here was
    /// exactly backwards.** On all **69/69** KDA layers, `A_log[0..96]` is
    /// entirely non-zero and lies in the `log(Uniform(1,16))` range, while
    /// `A_log[96..128]` is **exactly 0.0**. Controls on the same layers
    /// (`o_norm`, the `f_a_proj`/`f_b_proj` 128-axes, `dt_bias`) show no such
    /// trailing zeros, so it is not an artefact of how the file was written.
    ///
    /// It is **96 per-head entries zero-padded to the next power of two**,
    /// which on this model coincides with `head_dim`. So:
    ///
    /// - **Taking the first 96 is CORRECT.**
    /// - The port that gets a wrong decay is the one that BELIEVES the old
    ///   warning and uses all 128 — because `exp(0) = 1`, the padding is
    ///   decay-rate **1**, i.e. no decay at all, not a harmless no-op.
    ///
    /// This doc previously asserted the opposite, the gate printed it every
    /// run, and it was routed to another window as needing a forward-pass
    /// oracle to settle. It needed 512 bytes per layer. Worth noting that an
    /// architecture fragment written a day earlier already said "one parameter
    /// per head (96 of them)" — two artifacts in the same repository
    /// disagreeing about one tensor, with nothing watching for that.
    ALog,
    /// Per-channel bias added to the decay-gate input before the nonlinearity.
    DtBias,
    /// Decay-gate down-projection (low rank, rank `head_dim`).
    FAProj,
    /// Decay-gate up-projection.
    FBProj,
    /// Per-head beta (delta-rule write strength) logits.
    BProj,
    /// Gain of the gated RMSNorm on the attention output, per head channel.
    ONorm,

    // --- MLA only ---
    /// Query down-projection to `q_lora_rank`.
    QAProj,
    /// RMSNorm on the query latent.
    QALayernorm,
    /// Query up-projection to all heads.
    QBProj,
    /// Fused key/value down-projection: the `kv_lora_rank` latent plus the
    /// `qk_rope_head_dim` shared key channels.
    KvAProjWithMqa,
    /// RMSNorm on the key/value latent.
    KvALayernorm,
    /// Key/value up-projection to all heads.
    KvBProj,
}

impl AttnPart {
    /// Which attention kind carries this tensor, or `None` if both do.
    pub fn kind(self) -> Option<AttnKind> {
        match self {
            AttnPart::GProj | AttnPart::OProj => None,
            AttnPart::QProj
            | AttnPart::KProj
            | AttnPart::VProj
            | AttnPart::QConv1d
            | AttnPart::KConv1d
            | AttnPart::VConv1d
            | AttnPart::ALog
            | AttnPart::DtBias
            | AttnPart::FAProj
            | AttnPart::FBProj
            | AttnPart::BProj
            | AttnPart::ONorm => Some(AttnKind::Kda),
            AttnPart::QAProj
            | AttnPart::QALayernorm
            | AttnPart::QBProj
            | AttnPart::KvAProjWithMqa
            | AttnPart::KvALayernorm
            | AttnPart::KvBProj => Some(AttnKind::Mla),
        }
    }
}

/// A tensor inside a decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerPart {
    /// Pre-attention RMSNorm.
    InputLayernorm,
    /// Pre-MLP RMSNorm.
    PostAttentionLayernorm,
    /// RMSNorm of the AttnRes mix taken before attention.
    SelfAttentionResNorm,
    /// Score direction of the AttnRes mix taken before attention. Shape
    /// `[1, hidden]`: it is a single direction dotted with each normalised
    /// candidate to produce one logit per bank entry, not a bank of logits.
    SelfAttentionResProj,
    /// RMSNorm of the AttnRes mix taken before the MLP.
    MlpResNorm,
    /// Score direction of the AttnRes mix taken before the MLP.
    MlpResProj,
    /// Attention.
    Attn(AttnPart),
    /// The dense MLP, on layers below `first_k_dense_replace`.
    DenseMlp(MlpPart),
    /// The sparse MoE block, on the remaining layers.
    Moe(MoePart),
}

/// A tensor inside a ViT block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisionBlockPart {
    /// Pre-attention RMSNorm.
    Norm0,
    /// Fused query/key/value projection.
    Wqkv,
    /// Attention output projection.
    Wo,
    /// Pre-MLP RMSNorm.
    Norm1,
    /// MLP up-projection.
    MlpFc0,
    /// MLP down-projection.
    MlpFc1,
}

/// A tensor inside the ViT tower.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisionPart {
    /// Patch embedding convolution, `[vt_hidden, 3, patch, patch]`.
    PatchEmbedProj,
    /// Learned position-embedding grid, interpolated to the actual image size.
    PatchEmbedPosEmb,
    /// One tensor of one ViT block.
    Block {
        /// Block index.
        block: usize,
        /// Which tensor.
        part: VisionBlockPart,
    },
    /// RMSNorm after the last block.
    FinalLayernorm,
}

/// A tensor inside the vision-to-text projector.
///
/// The projector is a `Sequential` of `Linear, activation, Linear`, so the
/// checkpoint names its two weights `proj.0` and `proj.2` — index 1 is the
/// activation and carries no parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MmProjPart {
    /// First projector Linear, square at the merged-patch width.
    Proj0,
    /// Second projector Linear, out to the text `hidden_size`.
    Proj2,
    /// RMSNorm on the projected features.
    PostNorm,
}

/// A position in the K3 module tree that one checkpoint tensor fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slot {
    /// Token embedding table.
    EmbedTokens,
    /// Final RMSNorm of the decoder.
    FinalNorm,
    /// RMSNorm of the model-level AttnRes mix applied after the last layer.
    OutputAttnResNorm,
    /// Score direction of the model-level AttnRes mix.
    OutputAttnResProj,
    /// Vocabulary projection. Untied — `tie_word_embeddings` is false and the
    /// checkpoint ships a separate matrix.
    LmHead,
    /// A tensor of decoder layer `layer`.
    Layer {
        /// 0-based layer index.
        layer: usize,
        /// Which tensor.
        part: LayerPart,
    },
    /// A tensor of the ViT tower.
    Vision(VisionPart),
    /// A tensor of the vision-to-text projector.
    MmProjector(MmProjPart),
}

/// A slot together with everything needed to fetch it from the checkpoint.
#[derive(Debug, Clone)]
pub struct TensorSlot {
    /// Where in the module tree it goes.
    pub slot: Slot,
    /// The safetensors tensor name it comes from.
    pub name: String,
    /// Expected shape, in checkpoint order.
    pub shape: Shape,
    /// Expected on-disk element type.
    pub dtype: Dtype,
}

fn mlp_leaf(part: MlpPart) -> &'static str {
    match part {
        MlpPart::GateProj => "gate_proj.weight",
        MlpPart::UpProj => "up_proj.weight",
        MlpPart::DownProj => "down_proj.weight",
    }
}

fn attn_leaf(part: AttnPart) -> &'static str {
    match part {
        AttnPart::GProj => "g_proj.weight",
        AttnPart::OProj => "o_proj.weight",
        AttnPart::QProj => "q_proj.weight",
        AttnPart::KProj => "k_proj.weight",
        AttnPart::VProj => "v_proj.weight",
        AttnPart::QConv1d => "q_conv1d.weight",
        AttnPart::KConv1d => "k_conv1d.weight",
        AttnPart::VConv1d => "v_conv1d.weight",
        AttnPart::ALog => "A_log",
        AttnPart::DtBias => "dt_bias",
        AttnPart::FAProj => "f_a_proj.weight",
        AttnPart::FBProj => "f_b_proj.weight",
        AttnPart::BProj => "b_proj.weight",
        AttnPart::ONorm => "o_norm.weight",
        AttnPart::QAProj => "q_a_proj.weight",
        AttnPart::QALayernorm => "q_a_layernorm.weight",
        AttnPart::QBProj => "q_b_proj.weight",
        AttnPart::KvAProjWithMqa => "kv_a_proj_with_mqa.weight",
        AttnPart::KvALayernorm => "kv_a_layernorm.weight",
        AttnPart::KvBProj => "kv_b_proj.weight",
    }
}

fn vision_block_leaf(part: VisionBlockPart) -> &'static str {
    match part {
        VisionBlockPart::Norm0 => "norm0.weight",
        VisionBlockPart::Wqkv => "wqkv.weight",
        VisionBlockPart::Wo => "wo.weight",
        VisionBlockPart::Norm1 => "norm1.weight",
        VisionBlockPart::MlpFc0 => "mlp.fc0.weight",
        VisionBlockPart::MlpFc1 => "mlp.fc1.weight",
    }
}

impl Slot {
    /// The safetensors tensor name that fills this slot.
    ///
    /// The forward half of the mapping. [`Slot::parse`] is its inverse and is
    /// written independently; `k3_layout_gate` asserts they compose to the
    /// identity on every name in the real checkpoint.
    pub fn tensor_name(&self) -> String {
        match self {
            Slot::EmbedTokens => "language_model.model.embed_tokens.weight".to_string(),
            Slot::FinalNorm => "language_model.model.norm.weight".to_string(),
            Slot::OutputAttnResNorm => {
                "language_model.model.output_attn_res_norm.weight".to_string()
            }
            Slot::OutputAttnResProj => {
                "language_model.model.output_attn_res_proj.weight".to_string()
            }
            Slot::LmHead => "language_model.lm_head.weight".to_string(),
            Slot::Layer { layer, part } => {
                let head = format!("language_model.model.layers.{layer}");
                match part {
                    LayerPart::InputLayernorm => format!("{head}.input_layernorm.weight"),
                    LayerPart::PostAttentionLayernorm => {
                        format!("{head}.post_attention_layernorm.weight")
                    }
                    LayerPart::SelfAttentionResNorm => {
                        format!("{head}.self_attention_res_norm.weight")
                    }
                    LayerPart::SelfAttentionResProj => {
                        format!("{head}.self_attention_res_proj.weight")
                    }
                    LayerPart::MlpResNorm => format!("{head}.mlp_res_norm.weight"),
                    LayerPart::MlpResProj => format!("{head}.mlp_res_proj.weight"),
                    LayerPart::Attn(p) => format!("{head}.self_attn.{}", attn_leaf(*p)),
                    LayerPart::DenseMlp(m) => format!("{head}.mlp.{}", mlp_leaf(*m)),
                    LayerPart::Moe(m) => {
                        let moe = format!("{head}.block_sparse_moe");
                        match m {
                            MoePart::GateWeight => format!("{moe}.gate.weight"),
                            MoePart::GateScoreCorrectionBias => {
                                format!("{moe}.gate.e_score_correction_bias")
                            }
                            MoePart::RoutedExpertDownProj => {
                                format!("{moe}.routed_expert_down_proj.weight")
                            }
                            MoePart::RoutedExpertUpProj => {
                                format!("{moe}.routed_expert_up_proj.weight")
                            }
                            MoePart::RoutedExpertNorm => format!("{moe}.routed_expert_norm.weight"),
                            MoePart::Shared(p) => format!("{moe}.shared_experts.{}", mlp_leaf(*p)),
                            MoePart::Expert { expert, mat, part } => {
                                let mat = match mat {
                                    ExpertMat::W1 => "w1",
                                    ExpertMat::W2 => "w2",
                                    ExpertMat::W3 => "w3",
                                };
                                let leaf = match part {
                                    QuantPart::Packed => "weight_packed",
                                    QuantPart::Scale => "weight_scale",
                                };
                                format!("{moe}.experts.{expert}.{mat}.{leaf}")
                            }
                        }
                    }
                }
            }
            Slot::Vision(v) => match v {
                VisionPart::PatchEmbedProj => "vision_tower.patch_embed.proj.weight".to_string(),
                VisionPart::PatchEmbedPosEmb => {
                    "vision_tower.patch_embed.pos_emb.weight".to_string()
                }
                VisionPart::FinalLayernorm => {
                    "vision_tower.encoder.final_layernorm.weight".to_string()
                }
                VisionPart::Block { block, part } => {
                    format!(
                        "vision_tower.encoder.blocks.{block}.{}",
                        vision_block_leaf(*part)
                    )
                }
            },
            Slot::MmProjector(p) => match p {
                MmProjPart::Proj0 => "mm_projector.proj.0.weight".to_string(),
                MmProjPart::Proj2 => "mm_projector.proj.2.weight".to_string(),
                MmProjPart::PostNorm => "mm_projector.post_norm.weight".to_string(),
            },
        }
    }

    /// Recover the slot a checkpoint tensor name fills, or `None` if the name
    /// belongs to no slot this port knows about.
    ///
    /// The reverse half of the mapping, written without reference to
    /// [`Slot::tensor_name`]. It is deliberately strict: an unrecognised suffix
    /// is `None`, never a silent fallthrough, because the whole point of the
    /// gate is that an unmapped tensor gets *named*.
    ///
    /// This checks nothing about the config — a returned slot may still name a
    /// layer that does not exist or an MLA tensor on a KDA layer. [`describe`]
    /// is what decides whether a slot is legal for a given config.
    pub fn parse(name: &str) -> Option<Slot> {
        if let Some(rest) = name.strip_prefix("language_model.") {
            if rest == "lm_head.weight" {
                return Some(Slot::LmHead);
            }
            let rest = rest.strip_prefix("model.")?;
            match rest {
                "embed_tokens.weight" => return Some(Slot::EmbedTokens),
                "norm.weight" => return Some(Slot::FinalNorm),
                "output_attn_res_norm.weight" => return Some(Slot::OutputAttnResNorm),
                "output_attn_res_proj.weight" => return Some(Slot::OutputAttnResProj),
                _ => {}
            }
            let rest = rest.strip_prefix("layers.")?;
            let (layer, rest) = split_index(rest)?;
            let part = parse_layer_part(rest)?;
            return Some(Slot::Layer { layer, part });
        }
        if let Some(rest) = name.strip_prefix("vision_tower.") {
            return Some(Slot::Vision(parse_vision_part(rest)?));
        }
        if let Some(rest) = name.strip_prefix("mm_projector.") {
            let part = match rest {
                "proj.0.weight" => MmProjPart::Proj0,
                "proj.2.weight" => MmProjPart::Proj2,
                "post_norm.weight" => MmProjPart::PostNorm,
                _ => return None,
            };
            return Some(Slot::MmProjector(part));
        }
        None
    }
}

/// Split a leading `"<digits>."` off a name, returning the number and the rest.
fn split_index(s: &str) -> Option<(usize, &str)> {
    let (head, rest) = s.split_once('.')?;
    Some((head.parse().ok()?, rest))
}

fn parse_mlp_part(leaf: &str) -> Option<MlpPart> {
    match leaf {
        "gate_proj.weight" => Some(MlpPart::GateProj),
        "up_proj.weight" => Some(MlpPart::UpProj),
        "down_proj.weight" => Some(MlpPart::DownProj),
        _ => None,
    }
}

fn parse_layer_part(rest: &str) -> Option<LayerPart> {
    match rest {
        "input_layernorm.weight" => return Some(LayerPart::InputLayernorm),
        "post_attention_layernorm.weight" => return Some(LayerPart::PostAttentionLayernorm),
        "self_attention_res_norm.weight" => return Some(LayerPart::SelfAttentionResNorm),
        "self_attention_res_proj.weight" => return Some(LayerPart::SelfAttentionResProj),
        "mlp_res_norm.weight" => return Some(LayerPart::MlpResNorm),
        "mlp_res_proj.weight" => return Some(LayerPart::MlpResProj),
        _ => {}
    }
    if let Some(leaf) = rest.strip_prefix("self_attn.") {
        let part = match leaf {
            "g_proj.weight" => AttnPart::GProj,
            "o_proj.weight" => AttnPart::OProj,
            "q_proj.weight" => AttnPart::QProj,
            "k_proj.weight" => AttnPart::KProj,
            "v_proj.weight" => AttnPart::VProj,
            "q_conv1d.weight" => AttnPart::QConv1d,
            "k_conv1d.weight" => AttnPart::KConv1d,
            "v_conv1d.weight" => AttnPart::VConv1d,
            "A_log" => AttnPart::ALog,
            "dt_bias" => AttnPart::DtBias,
            "f_a_proj.weight" => AttnPart::FAProj,
            "f_b_proj.weight" => AttnPart::FBProj,
            "b_proj.weight" => AttnPart::BProj,
            "o_norm.weight" => AttnPart::ONorm,
            "q_a_proj.weight" => AttnPart::QAProj,
            "q_a_layernorm.weight" => AttnPart::QALayernorm,
            "q_b_proj.weight" => AttnPart::QBProj,
            "kv_a_proj_with_mqa.weight" => AttnPart::KvAProjWithMqa,
            "kv_a_layernorm.weight" => AttnPart::KvALayernorm,
            "kv_b_proj.weight" => AttnPart::KvBProj,
            _ => return None,
        };
        return Some(LayerPart::Attn(part));
    }
    if let Some(leaf) = rest.strip_prefix("mlp.") {
        return Some(LayerPart::DenseMlp(parse_mlp_part(leaf)?));
    }
    if let Some(leaf) = rest.strip_prefix("block_sparse_moe.") {
        return Some(LayerPart::Moe(parse_moe_part(leaf)?));
    }
    None
}

fn parse_moe_part(leaf: &str) -> Option<MoePart> {
    match leaf {
        "gate.weight" => return Some(MoePart::GateWeight),
        "gate.e_score_correction_bias" => return Some(MoePart::GateScoreCorrectionBias),
        "routed_expert_down_proj.weight" => return Some(MoePart::RoutedExpertDownProj),
        "routed_expert_up_proj.weight" => return Some(MoePart::RoutedExpertUpProj),
        "routed_expert_norm.weight" => return Some(MoePart::RoutedExpertNorm),
        _ => {}
    }
    if let Some(l) = leaf.strip_prefix("shared_experts.") {
        return Some(MoePart::Shared(parse_mlp_part(l)?));
    }
    let l = leaf.strip_prefix("experts.")?;
    let (expert, l) = split_index(l)?;
    let (mat, l) = l.split_once('.')?;
    let mat = match mat {
        "w1" => ExpertMat::W1,
        "w2" => ExpertMat::W2,
        "w3" => ExpertMat::W3,
        _ => return None,
    };
    let part = match l {
        "weight_packed" => QuantPart::Packed,
        "weight_scale" => QuantPart::Scale,
        _ => return None,
    };
    Some(MoePart::Expert { expert, mat, part })
}

fn parse_vision_part(rest: &str) -> Option<VisionPart> {
    match rest {
        "patch_embed.proj.weight" => return Some(VisionPart::PatchEmbedProj),
        "patch_embed.pos_emb.weight" => return Some(VisionPart::PatchEmbedPosEmb),
        "encoder.final_layernorm.weight" => return Some(VisionPart::FinalLayernorm),
        _ => {}
    }
    let l = rest.strip_prefix("encoder.blocks.")?;
    let (block, l) = split_index(l)?;
    let part = match l {
        "norm0.weight" => VisionBlockPart::Norm0,
        "wqkv.weight" => VisionBlockPart::Wqkv,
        "wo.weight" => VisionBlockPart::Wo,
        "norm1.weight" => VisionBlockPart::Norm1,
        "mlp.fc0.weight" => VisionBlockPart::MlpFc0,
        "mlp.fc1.weight" => VisionBlockPart::MlpFc1,
        _ => return None,
    };
    Some(VisionPart::Block { block, part })
}

/// The shape and dtype a slot must have, or `None` if the slot does not exist
/// in this config.
///
/// Every dimension here is derived from `config.json` — nothing is a literal
/// read off the headers. That is the real test of the config struct: if a shape
/// cannot be computed from the config, the config is missing a field, and the
/// gate finds out.
///
/// Returning `None` is how legality is expressed: a layer past the end, an MLA
/// tensor on a KDA layer, a dense MLP on a MoE layer, an expert index past
/// `num_experts`. The gate uses this as the second, independent opinion on
/// which slots exist.
pub fn describe(cfg: &K3Config, slot: Slot) -> Option<(Shape, Dtype)> {
    let t = &cfg.text_config;
    let v = &cfg.vision_config;
    let h = t.hidden_size;

    let spec = match slot {
        Slot::EmbedTokens | Slot::LmHead => (Shape::new(&[t.vocab_size, h]), Dtype::Bf16),
        Slot::FinalNorm | Slot::OutputAttnResNorm => (Shape::new(&[h]), Dtype::Bf16),
        Slot::OutputAttnResProj => (Shape::new(&[1, h]), Dtype::Bf16),

        Slot::Layer { layer, part } => {
            if layer >= t.num_hidden_layers {
                return None;
            }
            match part {
                LayerPart::InputLayernorm
                | LayerPart::PostAttentionLayernorm
                | LayerPart::SelfAttentionResNorm
                | LayerPart::MlpResNorm => (Shape::new(&[h]), Dtype::Bf16),
                LayerPart::SelfAttentionResProj | LayerPart::MlpResProj => {
                    (Shape::new(&[1, h]), Dtype::Bf16)
                }
                LayerPart::Attn(p) => {
                    if let Some(required) = p.kind() {
                        if t.attn_kind(layer) != required {
                            return None;
                        }
                    }
                    describe_attn(cfg, p)?
                }
                LayerPart::DenseMlp(m) => {
                    if t.is_moe_layer(layer) {
                        return None;
                    }
                    let ffn = t.intermediate_size;
                    match m {
                        MlpPart::GateProj | MlpPart::UpProj => (Shape::new(&[ffn, h]), Dtype::Bf16),
                        MlpPart::DownProj => (Shape::new(&[h, ffn]), Dtype::Bf16),
                    }
                }
                LayerPart::Moe(m) => {
                    if !t.is_moe_layer(layer) {
                        return None;
                    }
                    describe_moe(cfg, m)?
                }
            }
        }

        Slot::Vision(p) => {
            let vh = v.vt_hidden_size;
            match p {
                VisionPart::PatchEmbedProj => (
                    Shape::new(&[vh, 3, v.patch_size, v.patch_size]),
                    Dtype::Bf16,
                ),
                VisionPart::PatchEmbedPosEmb => (
                    Shape::new(&[v.init_pos_emb_height, v.init_pos_emb_width, vh]),
                    Dtype::Bf16,
                ),
                VisionPart::FinalLayernorm => (Shape::new(&[vh]), Dtype::Bf16),
                VisionPart::Block { block, part } => {
                    if block >= v.vt_num_hidden_layers {
                        return None;
                    }
                    match part {
                        VisionBlockPart::Norm0 | VisionBlockPart::Norm1 => {
                            (Shape::new(&[vh]), Dtype::Bf16)
                        }
                        // One `wqkv` emits query, key and value side by side,
                        // each `qkv_hidden_size` wide.
                        VisionBlockPart::Wqkv => {
                            (Shape::new(&[3 * v.qkv_hidden_size, vh]), Dtype::Bf16)
                        }
                        VisionBlockPart::Wo => (Shape::new(&[vh, v.qkv_hidden_size]), Dtype::Bf16),
                        VisionBlockPart::MlpFc0 => {
                            (Shape::new(&[v.vt_intermediate_size, vh]), Dtype::Bf16)
                        }
                        VisionBlockPart::MlpFc1 => {
                            (Shape::new(&[vh, v.vt_intermediate_size]), Dtype::Bf16)
                        }
                    }
                }
            }
        }

        Slot::MmProjector(p) => {
            let din = v.projector_input_dim();
            match p {
                MmProjPart::Proj0 => (Shape::new(&[din, din]), Dtype::Bf16),
                MmProjPart::Proj2 => (Shape::new(&[v.text_hidden_size, din]), Dtype::Bf16),
                MmProjPart::PostNorm => (Shape::new(&[v.text_hidden_size]), Dtype::Bf16),
            }
        }
    };
    Some(spec)
}

fn describe_attn(cfg: &K3Config, part: AttnPart) -> Option<(Shape, Dtype)> {
    let t = &cfg.text_config;
    let la = &t.linear_attn_config;
    let h = t.hidden_size;
    // KDA concatenates its heads and MLA concatenates its values to the same
    // width, which is why `o_proj` and `g_proj` can be shared between the two
    // layer kinds. `K3TextConfig::validate` refuses a config where they differ.
    let attn_out = la.proj_dim();

    Some(match part {
        AttnPart::GProj => (Shape::new(&[attn_out, h]), Dtype::Bf16),
        AttnPart::OProj => (Shape::new(&[h, attn_out]), Dtype::Bf16),

        AttnPart::QProj | AttnPart::KProj | AttnPart::VProj => {
            (Shape::new(&[la.proj_dim(), h]), Dtype::Bf16)
        }
        // Depthwise: one kernel per channel, hence the singleton middle axis.
        AttnPart::QConv1d | AttnPart::KConv1d | AttnPart::VConv1d => (
            Shape::new(&[la.proj_dim(), 1, la.short_conv_kernel_size]),
            Dtype::F32,
        ),
        // See the `AttnPart::ALog` docs: 96 live per-head entries zero-padded
        // to the next power of two, which here equals `head_dim`. The layout
        // follows the checkpoint because that is what a loader must read.
        // (was: the checkpoint ships `head_dim`, the
        // shipped modelling code declares `num_heads`, and they disagree here.
        AttnPart::ALog => (Shape::new(&[la.head_dim]), Dtype::F32),
        AttnPart::DtBias => (Shape::new(&[la.proj_dim()]), Dtype::F32),
        AttnPart::FAProj => (Shape::new(&[la.head_dim, h]), Dtype::Bf16),
        AttnPart::FBProj => (Shape::new(&[la.proj_dim(), la.head_dim]), Dtype::Bf16),
        AttnPart::BProj => (Shape::new(&[la.num_heads, h]), Dtype::Bf16),
        AttnPart::ONorm => (Shape::new(&[la.head_dim]), Dtype::F32),

        AttnPart::QAProj => (Shape::new(&[t.q_lora_rank?, h]), Dtype::Bf16),
        AttnPart::QALayernorm => (Shape::new(&[t.q_lora_rank?]), Dtype::Bf16),
        AttnPart::QBProj => (
            Shape::new(&[t.num_attention_heads * t.mla_q_head_dim(), t.q_lora_rank?]),
            Dtype::Bf16,
        ),
        // The latent, plus the rope key channels that are shared across heads.
        AttnPart::KvAProjWithMqa => (
            Shape::new(&[t.kv_lora_rank + t.qk_rope_head_dim, h]),
            Dtype::Bf16,
        ),
        AttnPart::KvALayernorm => (Shape::new(&[t.kv_lora_rank]), Dtype::Bf16),
        AttnPart::KvBProj => (
            Shape::new(&[
                t.num_key_value_heads * (t.qk_nope_head_dim + t.v_head_dim),
                t.kv_lora_rank,
            ]),
            Dtype::Bf16,
        ),
    })
}

/// MXFP4 block size: 32 elements share one E8M0 exponent, and two 4-bit codes
/// share one byte. Both apply along the *input* axis of the expert matrices.
const MXFP4_BLOCK: usize = 32;

fn describe_moe(cfg: &K3Config, part: MoePart) -> Option<(Shape, Dtype)> {
    let t = &cfg.text_config;
    let h = t.hidden_size;
    let latent = t.moe_hidden_size();
    let ffn = t.moe_intermediate_size;

    Some(match part {
        MoePart::GateWeight => (Shape::new(&[t.num_experts, h]), Dtype::Bf16),
        MoePart::GateScoreCorrectionBias => (Shape::new(&[t.num_experts]), Dtype::F32),
        MoePart::RoutedExpertDownProj => (Shape::new(&[latent, h]), Dtype::Bf16),
        MoePart::RoutedExpertUpProj => (Shape::new(&[h, latent]), Dtype::Bf16),
        MoePart::RoutedExpertNorm => {
            if !t.latent_moe_use_norm || t.routed_expert_hidden_size.is_none() {
                return None;
            }
            (Shape::new(&[latent]), Dtype::Bf16)
        }
        MoePart::Shared(m) => {
            t.num_shared_experts?;
            let shared = t.shared_expert_intermediate_size();
            match m {
                MlpPart::GateProj | MlpPart::UpProj => (Shape::new(&[shared, h]), Dtype::Bf16),
                MlpPart::DownProj => (Shape::new(&[h, shared]), Dtype::Bf16),
            }
        }
        MoePart::Expert { expert, mat, part } => {
            if expert >= t.num_experts {
                return None;
            }
            // Experts are quantised, so the stored shape is the logical
            // `[out, in]` with the input axis divided: by two for the packed
            // nibble pairs, by 32 for the per-block exponents.
            let (out, inp) = match mat {
                ExpertMat::W1 | ExpertMat::W3 => (ffn, latent),
                ExpertMat::W2 => (latent, ffn),
            };
            match part {
                QuantPart::Packed => (Shape::new(&[out, inp / 2]), Dtype::U8),
                QuantPart::Scale => (Shape::new(&[out, inp / MXFP4_BLOCK]), Dtype::U8),
            }
        }
    })
}

fn emit(cfg: &K3Config, slot: Slot, f: &mut impl FnMut(TensorSlot)) {
    // A slot the enumeration produces but `describe` rejects means the two
    // halves of this module disagree about what exists — a bug here, not in the
    // checkpoint, so it is an assertion rather than a gate finding.
    let (shape, dtype) = describe(cfg, slot)
        .unwrap_or_else(|| panic!("layout enumerated a slot describe() rejects: {slot:?}"));
    f(TensorSlot {
        slot,
        name: slot.tensor_name(),
        shape,
        dtype,
    });
}

/// Walk every slot the model needs, in load order.
///
/// A callback rather than an iterator: there are close to half a million slots
/// (494,592 of them are expert matrices), and a loader wants to stream them
/// against the checkpoint rather than materialise the list.
pub fn for_each_slot(cfg: &K3Config, mut f: impl FnMut(TensorSlot)) {
    let t = &cfg.text_config;
    let f = &mut f;

    emit(cfg, Slot::EmbedTokens, f);

    for layer in 0..t.num_hidden_layers {
        let mut layer_slot = |part: LayerPart| emit(cfg, Slot::Layer { layer, part }, f);

        layer_slot(LayerPart::InputLayernorm);
        layer_slot(LayerPart::PostAttentionLayernorm);
        layer_slot(LayerPart::SelfAttentionResNorm);
        layer_slot(LayerPart::SelfAttentionResProj);
        layer_slot(LayerPart::MlpResNorm);
        layer_slot(LayerPart::MlpResProj);

        layer_slot(LayerPart::Attn(AttnPart::GProj));
        layer_slot(LayerPart::Attn(AttnPart::OProj));
        match t.attn_kind(layer) {
            AttnKind::Kda => {
                for p in [
                    AttnPart::QProj,
                    AttnPart::KProj,
                    AttnPart::VProj,
                    AttnPart::QConv1d,
                    AttnPart::KConv1d,
                    AttnPart::VConv1d,
                    AttnPart::ALog,
                    AttnPart::DtBias,
                    AttnPart::FAProj,
                    AttnPart::FBProj,
                    AttnPart::BProj,
                    AttnPart::ONorm,
                ] {
                    layer_slot(LayerPart::Attn(p));
                }
            }
            AttnKind::Mla => {
                for p in [
                    AttnPart::QAProj,
                    AttnPart::QALayernorm,
                    AttnPart::QBProj,
                    AttnPart::KvAProjWithMqa,
                    AttnPart::KvALayernorm,
                    AttnPart::KvBProj,
                ] {
                    layer_slot(LayerPart::Attn(p));
                }
            }
        }

        if t.is_moe_layer(layer) {
            layer_slot(LayerPart::Moe(MoePart::GateWeight));
            layer_slot(LayerPart::Moe(MoePart::GateScoreCorrectionBias));
            if t.routed_expert_hidden_size.is_some() {
                layer_slot(LayerPart::Moe(MoePart::RoutedExpertDownProj));
                layer_slot(LayerPart::Moe(MoePart::RoutedExpertUpProj));
                if t.latent_moe_use_norm {
                    layer_slot(LayerPart::Moe(MoePart::RoutedExpertNorm));
                }
            }
            if t.num_shared_experts.is_some() {
                for m in [MlpPart::GateProj, MlpPart::UpProj, MlpPart::DownProj] {
                    layer_slot(LayerPart::Moe(MoePart::Shared(m)));
                }
            }
            for expert in 0..t.num_experts {
                for mat in [ExpertMat::W1, ExpertMat::W2, ExpertMat::W3] {
                    for part in [QuantPart::Packed, QuantPart::Scale] {
                        layer_slot(LayerPart::Moe(MoePart::Expert { expert, mat, part }));
                    }
                }
            }
        } else {
            for m in [MlpPart::GateProj, MlpPart::UpProj, MlpPart::DownProj] {
                layer_slot(LayerPart::DenseMlp(m));
            }
        }
    }

    emit(cfg, Slot::FinalNorm, f);
    emit(cfg, Slot::OutputAttnResNorm, f);
    emit(cfg, Slot::OutputAttnResProj, f);
    if !t.tie_word_embeddings {
        emit(cfg, Slot::LmHead, f);
    }

    let v = &cfg.vision_config;
    emit(cfg, Slot::Vision(VisionPart::PatchEmbedProj), f);
    emit(cfg, Slot::Vision(VisionPart::PatchEmbedPosEmb), f);
    for block in 0..v.vt_num_hidden_layers {
        for part in [
            VisionBlockPart::Norm0,
            VisionBlockPart::Wqkv,
            VisionBlockPart::Wo,
            VisionBlockPart::Norm1,
            VisionBlockPart::MlpFc0,
            VisionBlockPart::MlpFc1,
        ] {
            emit(cfg, Slot::Vision(VisionPart::Block { block, part }), f);
        }
    }
    emit(cfg, Slot::Vision(VisionPart::FinalLayernorm), f);

    for p in [MmProjPart::Proj0, MmProjPart::Proj2, MmProjPart::PostNorm] {
        emit(cfg, Slot::MmProjector(p), f);
    }
}
