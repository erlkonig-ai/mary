//! Checkpoint-name to module-slot mapping for Inkling.
//!
//! Same discipline as the K3 layout: `Slot::tensor_name` and `Slot::parse` are
//! written *independently* so that composing them is a real test rather than a
//! tautology, and `describe` predicts a shape and dtype for every slot from the
//! config alone. `inkling_layout_gate` then asserts the mapping is a bijection
//! against a real checkpoint.
//!
//! Four towers share one namespace — `model.llm`, `model.mtp`, `model.visual`,
//! `model.audio` — and all four use `layers.` in their names, so any pattern
//! that is not anchored to its tower will silently sum one tower's layers into
//! another's. (It did: the first census reported LLM layers whose index ranges
//! overlapped, which cannot describe one tower.)
//!
//! One thing here is deliberately *not* predicted from the config: whether a
//! given MoE layer's experts are NVFP4 or BF16. The released small checkpoint
//! quantizes 39 of its 40 MoE layers and leaves layer 2 in BF16, and nothing in
//! `config.json` says so. Rather than hardcode `2`, the layout treats expert
//! quantization as a per-layer fact supplied by the caller, and the gate
//! asserts the *structural* invariant instead: a layer carries either all four
//! quantization sidecars or none of them, never a mix.

use std::collections::BTreeSet;

use crate::models::inkling::config::{AttnKind, InklingConfig};

/// The dtypes an Inkling checkpoint holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// bfloat16 — everything dense, plus the BF16 expert layers.
    Bf16,
    /// float32 — the router bias and global scale, and the per-expert `scale2`.
    F32,
    /// uint8 — NVFP4 expert weights, two 4-bit codes per byte.
    U8,
    /// float8 E4M3 — the NVFP4 block scales (one per 16 logical elements).
    F8E4M3,
    /// int64 — the `original_shape` sidecars.
    I64,
}

impl Dtype {
    /// Parse safetensors' own spelling. Matched on the debug rendering rather
    /// than on enum variants because `F8_E4M3` is not present in every
    /// safetensors release, and a checkpoint that holds one must still be
    /// readable rather than fail to compile.
    pub fn from_safetensors_debug(s: &str) -> Option<Dtype> {
        match s {
            "BF16" => Some(Dtype::Bf16),
            "F32" => Some(Dtype::F32),
            "U8" => Some(Dtype::U8),
            "F8_E4M3" | "F8E4M3" => Some(Dtype::F8E4M3),
            "I64" => Some(Dtype::I64),
            _ => None,
        }
    }
}

/// A tensor shape, up to rank 3 (Inkling has nothing wider).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; 3],
    rank: u8,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        assert!(dims.len() <= 3, "Inkling has no tensors above rank 3");
        let mut d = [0usize; 3];
        d[..dims.len()].copy_from_slice(dims);
        Shape { dims: d, rank: dims.len() as u8 }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank as usize]
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.dims())
    }
}

/// Which of an expert's two stacked matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertMat {
    /// Gate and up concatenated: logical `[experts, hidden, 2 * intermediate]`.
    W13,
    /// Down: logical `[experts, hidden, intermediate]`.
    W2,
}

impl ExpertMat {
    /// The name stem this matrix is stored under.
    pub fn suffix(self) -> &'static str {
        match self {
            ExpertMat::W13 => "w13_weight",
            ExpertMat::W2 => "w2_weight",
        }
    }
}

/// The pieces an NVFP4 matrix is stored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QuantPart {
    /// The packed codes themselves (or the plain BF16 weight, unquantized).
    Weight,
    /// Per-block E4M3 scale, one per 16 logical elements.
    Scale,
    /// Per-expert F32 second-level scale.
    Scale2,
    /// Activation calibration maximum.
    InputAmax,
    /// The logical shape before 4-bit packing.
    OriginalShape,
}

impl QuantPart {
    /// The name suffix this piece carries; the codes themselves carry none.
    pub fn suffix(self) -> &'static str {
        match self {
            QuantPart::Weight => "",
            QuantPart::Scale => ".scale",
            QuantPart::Scale2 => ".scale2",
            QuantPart::InputAmax => ".input_amax",
            QuantPart::OriginalShape => ".original_shape",
        }
    }

    /// The four sidecars a quantized matrix carries beyond its codes.
    pub fn sidecars() -> [QuantPart; 4] {
        [QuantPart::Scale, QuantPart::Scale2, QuantPart::InputAmax, QuantPart::OriginalShape]
    }
}

/// A dense MLP's tensors — used by layers below `dense_mlp_idx` and by every
/// MTP block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DensePart {
    /// `[2 * dense_intermediate, hidden]` — gate and up concatenated.
    W13,
    /// `[hidden, dense_intermediate]`.
    W2,
    /// Scalar.
    GlobalScale,
}

/// A MoE MLP's tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MoePart {
    /// `[gate_rows, hidden]` — wider than `n_routed_experts` when
    /// `shared_expert_sink` is set.
    GateWeight,
    /// `[n_routed_experts]` — routed experts only, never the sinks.
    GateBias,
    /// Scalar.
    GateGlobalScale,
    /// One of the stacked expert matrices, or one of its sidecars.
    Expert(ExpertMat, QuantPart),
    /// `[n_shared, hidden, 2 * intermediate]`.
    SharedW13,
    /// `[n_shared, hidden, intermediate]`.
    SharedW2,
}

/// Either kind of MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MlpPart {
    Dense(DensePart),
    Moe(MoePart),
}

/// The attention block's tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttnPart {
    /// `[q_dim, hidden]`.
    Wq,
    /// `[kv_dim, hidden]`.
    Wk,
    /// `[kv_dim, hidden]`.
    Wv,
    /// `[hidden, q_dim]`.
    Wo,
    /// `[rel_dim, hidden]` — the rank-`d_rel`-per-head relative path.
    Wr,
    /// `[d_rel, rel_extent / 2]`.
    RelLogitsProj,
    /// `[head_dim]`.
    QNorm,
    /// `[head_dim]`.
    KNorm,
    /// `[kv_dim, 1, sconv_kernel_size]`.
    KSconv,
    /// `[kv_dim, 1, sconv_kernel_size]`.
    VSconv,
}

impl AttnPart {
    fn suffix(self) -> &'static str {
        match self {
            AttnPart::Wq => "wq_du.weight",
            AttnPart::Wk => "wk_dv.weight",
            AttnPart::Wv => "wv_dv.weight",
            AttnPart::Wo => "wo_ud.weight",
            AttnPart::Wr => "wr_du.weight",
            AttnPart::RelLogitsProj => "rel_logits_proj.proj",
            AttnPart::QNorm => "q_norm.weight",
            AttnPart::KNorm => "k_norm.weight",
            AttnPart::KSconv => "k_sconv.weight",
            AttnPart::VSconv => "v_sconv.weight",
        }
    }

    fn all() -> [AttnPart; 10] {
        [
            AttnPart::Wq,
            AttnPart::Wk,
            AttnPart::Wv,
            AttnPart::Wo,
            AttnPart::Wr,
            AttnPart::RelLogitsProj,
            AttnPart::QNorm,
            AttnPart::KNorm,
            AttnPart::KSconv,
            AttnPart::VSconv,
        ]
    }
}

/// Everything inside one transformer block, LLM or MTP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlockPart {
    /// `[hidden]`.
    AttnNorm,
    /// `[hidden, 1, sconv_kernel_size]` — short conv on the attention input.
    AttnSconv,
    /// `[hidden]`.
    MlpNorm,
    /// `[hidden, 1, sconv_kernel_size]` — short conv into the MLP.
    MlpSconv,
    Attn(AttnPart),
    Mlp(MlpPart),
}

/// An MTP layer wraps a block with its own norms and an input projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MtpPart {
    /// `[hidden]`.
    EmbedNorm,
    /// `[hidden]`.
    HiddenNorm,
    /// `[hidden, 2 * hidden]` — embedding and hidden state concatenated.
    InputProj,
    /// The wrapped transformer block; always dense.
    Block(BlockPart),
}

/// The HMLP vision pyramid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VisionPart {
    /// `layers.linear_<i>.weight`.
    Linear(usize),
    /// `layers.norm_<i>.weight`.
    Norm(usize),
    /// `[decoder_dmodel]`.
    FinalNorm,
}

/// The dMel audio input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AudioPart {
    /// `[n_mel_bins * mel_vocab_size, decoder_dmodel]`.
    Encoder,
    /// `[decoder_dmodel]`.
    FinalNorm,
}

/// One tensor's worth of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Slot {
    /// `[vocab, hidden]`.
    Embed,
    /// `[hidden]`.
    EmbedNorm,
    /// `[hidden]` — the final norm before the unembedding.
    FinalNorm,
    /// `[vocab, hidden]`.
    Unembed,
    Llm(usize, BlockPart),
    Mtp(usize, MtpPart),
    Vision(VisionPart),
    Audio(AudioPart),
}

/// A slot together with what the config says must fill it.
#[derive(Debug, Clone, Copy)]
pub struct TensorSlot {
    pub slot: Slot,
    pub shape: Shape,
    pub dtype: Dtype,
}

fn block_suffix(part: BlockPart) -> String {
    match part {
        BlockPart::AttnNorm => "attn_norm.weight".to_string(),
        BlockPart::AttnSconv => "attn_sconv.weight".to_string(),
        BlockPart::MlpNorm => "mlp_norm.weight".to_string(),
        BlockPart::MlpSconv => "mlp_sconv.weight".to_string(),
        BlockPart::Attn(a) => format!("attn.{}", a.suffix()),
        BlockPart::Mlp(MlpPart::Dense(d)) => match d {
            DensePart::W13 => "mlp.w13_dn.weight".to_string(),
            DensePart::W2 => "mlp.w2_md.weight".to_string(),
            DensePart::GlobalScale => "mlp.global_scale".to_string(),
        },
        BlockPart::Mlp(MlpPart::Moe(m)) => match m {
            MoePart::GateWeight => "mlp.gate.weight".to_string(),
            MoePart::GateBias => "mlp.gate.bias".to_string(),
            MoePart::GateGlobalScale => "mlp.gate.global_scale".to_string(),
            MoePart::SharedW13 => "mlp.shared_experts.shared_w13_weight".to_string(),
            MoePart::SharedW2 => "mlp.shared_experts.shared_w2_weight".to_string(),
            MoePart::Expert(mat, q) => {
                format!("mlp.experts.{}{}", mat.suffix(), q.suffix())
            }
        },
    }
}

impl Slot {
    /// The checkpoint name this slot is filled from.
    ///
    /// Written independently of [`Slot::parse`]; the gate composes the two and
    /// requires the identity, so a typo in either surfaces rather than
    /// cancelling out.
    pub fn tensor_name(&self) -> String {
        match self {
            Slot::Embed => "model.llm.embed.weight".to_string(),
            Slot::EmbedNorm => "model.llm.embed_norm.weight".to_string(),
            Slot::FinalNorm => "model.llm.norm.weight".to_string(),
            Slot::Unembed => "model.llm.unembed.weight".to_string(),
            Slot::Llm(i, part) => format!("model.llm.layers.{i}.{}", block_suffix(*part)),
            Slot::Mtp(i, part) => match part {
                MtpPart::EmbedNorm => format!("model.mtp.layers.{i}.embed_norm.weight"),
                MtpPart::HiddenNorm => format!("model.mtp.layers.{i}.hidden_norm.weight"),
                MtpPart::InputProj => format!("model.mtp.layers.{i}.input_proj.weight"),
                MtpPart::Block(b) => {
                    format!("model.mtp.layers.{i}.transformer_block.{}", block_suffix(*b))
                }
            },
            Slot::Vision(v) => match v {
                VisionPart::Linear(i) => format!("model.visual.layers.linear_{i}.weight"),
                VisionPart::Norm(i) => format!("model.visual.layers.norm_{i}.weight"),
                VisionPart::FinalNorm => "model.visual.final_norm.weight".to_string(),
            },
            Slot::Audio(a) => match a {
                AudioPart::Encoder => "model.audio.encoder.weight".to_string(),
                AudioPart::FinalNorm => "model.audio.final_norm.weight".to_string(),
            },
        }
    }

    /// Recover a slot from a checkpoint name, or `None` if the name is not one
    /// this layout knows.
    pub fn parse(name: &str) -> Option<Slot> {
        match name {
            "model.llm.embed.weight" => return Some(Slot::Embed),
            "model.llm.embed_norm.weight" => return Some(Slot::EmbedNorm),
            "model.llm.norm.weight" => return Some(Slot::FinalNorm),
            "model.llm.unembed.weight" => return Some(Slot::Unembed),
            "model.visual.final_norm.weight" => {
                return Some(Slot::Vision(VisionPart::FinalNorm))
            }
            "model.audio.encoder.weight" => return Some(Slot::Audio(AudioPart::Encoder)),
            "model.audio.final_norm.weight" => return Some(Slot::Audio(AudioPart::FinalNorm)),
            _ => {}
        }

        if let Some(rest) = name.strip_prefix("model.visual.layers.") {
            let stem = rest.strip_suffix(".weight")?;
            if let Some(i) = stem.strip_prefix("linear_") {
                return Some(Slot::Vision(VisionPart::Linear(i.parse().ok()?)));
            }
            if let Some(i) = stem.strip_prefix("norm_") {
                return Some(Slot::Vision(VisionPart::Norm(i.parse().ok()?)));
            }
            return None;
        }

        if let Some(rest) = name.strip_prefix("model.llm.layers.") {
            let (idx, tail) = rest.split_once('.')?;
            return Some(Slot::Llm(idx.parse().ok()?, parse_block(tail)?));
        }

        if let Some(rest) = name.strip_prefix("model.mtp.layers.") {
            let (idx, tail) = rest.split_once('.')?;
            let i: usize = idx.parse().ok()?;
            return Some(Slot::Mtp(
                i,
                match tail {
                    "embed_norm.weight" => MtpPart::EmbedNorm,
                    "hidden_norm.weight" => MtpPart::HiddenNorm,
                    "input_proj.weight" => MtpPart::InputProj,
                    _ => MtpPart::Block(parse_block(tail.strip_prefix("transformer_block.")?)?),
                },
            ));
        }

        None
    }
}

fn parse_block(tail: &str) -> Option<BlockPart> {
    match tail {
        "attn_norm.weight" => return Some(BlockPart::AttnNorm),
        "attn_sconv.weight" => return Some(BlockPart::AttnSconv),
        "mlp_norm.weight" => return Some(BlockPart::MlpNorm),
        "mlp_sconv.weight" => return Some(BlockPart::MlpSconv),
        "mlp.w13_dn.weight" => return Some(BlockPart::Mlp(MlpPart::Dense(DensePart::W13))),
        "mlp.w2_md.weight" => return Some(BlockPart::Mlp(MlpPart::Dense(DensePart::W2))),
        "mlp.global_scale" => {
            return Some(BlockPart::Mlp(MlpPart::Dense(DensePart::GlobalScale)))
        }
        "mlp.gate.weight" => return Some(BlockPart::Mlp(MlpPart::Moe(MoePart::GateWeight))),
        "mlp.gate.bias" => return Some(BlockPart::Mlp(MlpPart::Moe(MoePart::GateBias))),
        "mlp.gate.global_scale" => {
            return Some(BlockPart::Mlp(MlpPart::Moe(MoePart::GateGlobalScale)))
        }
        "mlp.shared_experts.shared_w13_weight" => {
            return Some(BlockPart::Mlp(MlpPart::Moe(MoePart::SharedW13)))
        }
        "mlp.shared_experts.shared_w2_weight" => {
            return Some(BlockPart::Mlp(MlpPart::Moe(MoePart::SharedW2)))
        }
        _ => {}
    }

    if let Some(a) = tail.strip_prefix("attn.") {
        for part in AttnPart::all() {
            if part.suffix() == a {
                return Some(BlockPart::Attn(part));
            }
        }
        return None;
    }

    if let Some(e) = tail.strip_prefix("mlp.experts.") {
        for mat in [ExpertMat::W13, ExpertMat::W2] {
            if let Some(q) = e.strip_prefix(mat.suffix()) {
                for part in [
                    QuantPart::Weight,
                    QuantPart::Scale,
                    QuantPart::Scale2,
                    QuantPart::InputAmax,
                    QuantPart::OriginalShape,
                ] {
                    if part.suffix() == q {
                        return Some(BlockPart::Mlp(MlpPart::Moe(MoePart::Expert(mat, part))));
                    }
                }
            }
        }
        return None;
    }

    None
}

/// What the config says fills `slot`, or `None` if the slot does not exist for
/// this config.
///
/// `quantized_moe_layers` names the LLM layers whose experts are NVFP4; any
/// other MoE layer is expected in BF16 with no sidecars. It is a parameter
/// rather than a config field because `config.json` does not record it.
pub fn describe(
    cfg: &InklingConfig,
    quantized_moe_layers: &BTreeSet<usize>,
    slot: Slot,
) -> Option<(Shape, Dtype)> {
    let t = &cfg.text_config;
    let bf = Dtype::Bf16;
    let d = |dims: &[usize], dt: Dtype| Some((Shape::new(dims), dt));

    match slot {
        Slot::Embed | Slot::Unembed => d(&[t.vocab_size, t.hidden_size], bf),
        Slot::EmbedNorm | Slot::FinalNorm => d(&[t.hidden_size], bf),

        Slot::Vision(v) => {
            let vc = &cfg.vision_config;
            match v {
                VisionPart::FinalNorm => d(&[vc.decoder_dmodel], bf),
                VisionPart::Norm(i) => d(&[*vc.hidden_dims.get(i)?], bf),
                VisionPart::Linear(i) => {
                    // Stage i maps a group of the previous width to the next.
                    // Stage 0 consumes one sub-patch; the last stage consumes
                    // the whole temporal patch and emits the decoder width.
                    let out = if i + 1 == vc.n_layers {
                        vc.decoder_dmodel
                    } else {
                        *vc.hidden_dims.get(i)?
                    };
                    let inp = if i == 0 {
                        vc.subpatch_elems()
                    } else if i + 1 == vc.n_layers {
                        vc.patch_elems()
                    } else {
                        // A whole number of previous-stage vectors; the gate
                        // checks the grouping factor is integral.
                        let prev = *vc.hidden_dims.get(i - 1)?;
                        prev * grouping_factor(vc, i)?
                    };
                    d(&[out, inp], bf)
                }
            }
        }

        Slot::Audio(a) => match a {
            AudioPart::Encoder => {
                d(&[cfg.audio_config.encoder_rows(), cfg.audio_config.decoder_dmodel], bf)
            }
            AudioPart::FinalNorm => d(&[cfg.audio_config.decoder_dmodel], bf),
        },

        Slot::Llm(i, part) => {
            if i >= t.num_hidden_layers {
                return None;
            }
            describe_block(cfg, quantized_moe_layers, i, t.is_dense(i), t.attn_kind(i), part)
        }

        Slot::Mtp(i, part) => {
            if i >= cfg.mtp_config.num_nextn_predict_layers {
                return None;
            }
            match part {
                MtpPart::EmbedNorm | MtpPart::HiddenNorm => d(&[t.hidden_size], bf),
                MtpPart::InputProj => d(&[t.hidden_size, 2 * t.hidden_size], bf),
                // Every MTP block is dense, whatever `dense_mlp_idx` says.
                MtpPart::Block(b) => describe_block(
                    cfg,
                    quantized_moe_layers,
                    i,
                    true,
                    cfg.mtp_config.attn_kind(i),
                    b,
                ),
            }
        }
    }
}

/// How many previous-stage vectors stage `i` consumes. Derived from the patch
/// geometry so a wrong `hidden_dims` default cannot pass unnoticed.
fn grouping_factor(vc: &crate::models::inkling::config::InklingVisionConfig, i: usize) -> Option<usize> {
    // Sub-patches per frame, then the pyramid folds them in equal steps.
    let per_frame = (vc.patch_size / vc.subpatch_size).pow(2);
    let stages = vc.n_layers.checked_sub(1)?;
    if stages == 0 {
        return None;
    }
    // Two grouping stages on the released models: 64 = 4 * 16.
    let factors: Vec<usize> = match (per_frame, stages) {
        (64, 3) => vec![4, 16],
        _ => return None,
    };
    factors.get(i - 1).copied()
}

fn describe_block(
    cfg: &InklingConfig,
    quantized_moe_layers: &BTreeSet<usize>,
    layer: usize,
    dense: bool,
    kind: AttnKind,
    part: BlockPart,
) -> Option<(Shape, Dtype)> {
    let t = &cfg.text_config;
    let bf = Dtype::Bf16;
    let k = t.sconv_kernel_size;
    let d = |dims: &[usize], dt: Dtype| Some((Shape::new(dims), dt));

    match part {
        BlockPart::AttnNorm | BlockPart::MlpNorm => d(&[t.hidden_size], bf),
        BlockPart::AttnSconv | BlockPart::MlpSconv => d(&[t.hidden_size, 1, k], bf),

        BlockPart::Attn(a) => match a {
            AttnPart::Wq => d(&[t.q_dim(kind), t.hidden_size], bf),
            AttnPart::Wk | AttnPart::Wv => d(&[t.kv_dim(kind), t.hidden_size], bf),
            AttnPart::Wo => d(&[t.hidden_size, t.q_dim(kind)], bf),
            AttnPart::Wr => d(&[t.rel_dim(kind), t.hidden_size], bf),
            AttnPart::RelLogitsProj => d(&[t.d_rel, t.rel_span(kind)], bf),
            AttnPart::QNorm | AttnPart::KNorm => d(&[t.norm_dim(kind)], bf),
            AttnPart::KSconv | AttnPart::VSconv => d(&[t.kv_dim(kind), 1, k], bf),
        },

        BlockPart::Mlp(MlpPart::Dense(dp)) => {
            if !dense {
                return None;
            }
            match dp {
                DensePart::W13 => d(&[2 * t.dense_intermediate_size, t.hidden_size], bf),
                DensePart::W2 => d(&[t.hidden_size, t.dense_intermediate_size], bf),
                DensePart::GlobalScale => d(&[1], bf),
            }
        }

        BlockPart::Mlp(MlpPart::Moe(m)) => {
            if dense {
                return None;
            }
            let e = t.n_routed_experts;
            let h = t.hidden_size;
            let inter = t.intermediate_size;
            let quantized = quantized_moe_layers.contains(&layer);
            match m {
                MoePart::GateWeight => d(&[t.gate_rows(), h], bf),
                MoePart::GateBias => d(&[t.n_routed_experts], Dtype::F32),
                MoePart::GateGlobalScale => d(&[1], Dtype::F32),
                MoePart::SharedW13 => d(&[t.n_shared_experts, h, 2 * inter], bf),
                MoePart::SharedW2 => d(&[t.n_shared_experts, h, inter], bf),
                MoePart::Expert(mat, q) => {
                    // Logical width of this matrix's last dimension.
                    let logical = match mat {
                        ExpertMat::W13 => 2 * inter,
                        ExpertMat::W2 => inter,
                    };
                    if !quantized {
                        return match q {
                            QuantPart::Weight => d(&[e, h, logical], bf),
                            // A BF16 layer carries no sidecars at all.
                            _ => None,
                        };
                    }
                    match q {
                        // Two 4-bit codes per byte.
                        QuantPart::Weight => d(&[e, h, logical / 2], Dtype::U8),
                        // One E4M3 scale per NVFP4 block of 16.
                        QuantPart::Scale => d(&[e, h, logical / NVFP4_GROUP], Dtype::F8E4M3),
                        QuantPart::Scale2 => d(&[e], Dtype::F32),
                        QuantPart::InputAmax => d(&[1], bf),
                        QuantPart::OriginalShape => d(&[3], Dtype::I64),
                    }
                }
            }
        }
    }
}

/// NVFP4's block size — `hf_quant_config.json`'s `group_size`. K3's MXFP4 uses
/// 32 with an E8M0 scale and no second level; this is the one place the K3
/// decode does not carry over.
pub const NVFP4_GROUP: usize = 16;

/// Visit every slot the config implies, with the shape and dtype it must hold.
pub fn for_each_slot(
    cfg: &InklingConfig,
    quantized_moe_layers: &BTreeSet<usize>,
    mut f: impl FnMut(TensorSlot),
) {
    let t = &cfg.text_config;
    let emit = |slot: Slot, f: &mut dyn FnMut(TensorSlot)| {
        if let Some((shape, dtype)) = describe(cfg, quantized_moe_layers, slot) {
            f(TensorSlot { slot, shape, dtype });
        }
    };

    for s in [Slot::Embed, Slot::EmbedNorm, Slot::FinalNorm, Slot::Unembed] {
        emit(s, &mut f);
    }

    let mut block_parts: Vec<BlockPart> = vec![
        BlockPart::AttnNorm,
        BlockPart::AttnSconv,
        BlockPart::MlpNorm,
        BlockPart::MlpSconv,
    ];
    for a in AttnPart::all() {
        block_parts.push(BlockPart::Attn(a));
    }
    for dp in [DensePart::W13, DensePart::W2, DensePart::GlobalScale] {
        block_parts.push(BlockPart::Mlp(MlpPart::Dense(dp)));
    }
    for m in [
        MoePart::GateWeight,
        MoePart::GateBias,
        MoePart::GateGlobalScale,
        MoePart::SharedW13,
        MoePart::SharedW2,
    ] {
        block_parts.push(BlockPart::Mlp(MlpPart::Moe(m)));
    }
    for mat in [ExpertMat::W13, ExpertMat::W2] {
        for q in [
            QuantPart::Weight,
            QuantPart::Scale,
            QuantPart::Scale2,
            QuantPart::InputAmax,
            QuantPart::OriginalShape,
        ] {
            block_parts.push(BlockPart::Mlp(MlpPart::Moe(MoePart::Expert(mat, q))));
        }
    }

    for i in 0..t.num_hidden_layers {
        for &p in &block_parts {
            emit(Slot::Llm(i, p), &mut f);
        }
    }

    for i in 0..cfg.mtp_config.num_nextn_predict_layers {
        for s in [MtpPart::EmbedNorm, MtpPart::HiddenNorm, MtpPart::InputProj] {
            emit(Slot::Mtp(i, s), &mut f);
        }
        for &p in &block_parts {
            emit(Slot::Mtp(i, MtpPart::Block(p)), &mut f);
        }
    }

    let vc = &cfg.vision_config;
    for i in 0..vc.n_layers {
        emit(Slot::Vision(VisionPart::Linear(i)), &mut f);
        if i + 1 < vc.n_layers {
            emit(Slot::Vision(VisionPart::Norm(i)), &mut f);
        }
    }
    emit(Slot::Vision(VisionPart::FinalNorm), &mut f);

    for a in [AudioPart::Encoder, AudioPart::FinalNorm] {
        emit(Slot::Audio(a), &mut f);
    }
}
