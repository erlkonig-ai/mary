//! Inkling's multi-token-prediction (MTP) heads — the composition
//! `transformers` declines to define, and the reason that is not fatal.
//!
//! Each of the eight heads is one DENSE decoder block plus a small wrapper:
//! two RMS norms and an `input_proj` of `[hidden, 2 * hidden]`. The block half
//! is settled — `inkling_mtp_gate` runs `decoder_layer` on real MTP weights
//! against a reference to 1e-5. The wrapper half has no oracle at all:
//! `transformers` discards every `model.mtp.*` weight on load
//! (`_keys_to_ignore_on_load_unexpected`) and no class in the package uses
//! `input_proj` or `hidden_norm`, so nothing upstream defines how the pieces
//! compose.
//!
//! The names nearly determine it — an `[hidden, 2 * hidden]` projection over a
//! concatenation, one norm per operand — and `mtp_hidden_states_first` says the
//! hidden state comes first. But that is a config flag, not an observed
//! computation, so [`Concat`] keeps BOTH orders reachable rather than encoding
//! the guess as if it were knowledge.
//!
//! # Why a guess is safe here
//!
//! These heads exist to draft tokens for speculative decoding, and speculative
//! decoding verifies every draft against the full model, accepting the longest
//! correct prefix. Output is identical to not speculating for ANY draft — that
//! is the theorem, not a quality claim. So a wrong composition cannot produce
//! wrong text; it produces a low ACCEPTANCE RATE. Which makes acceptance the
//! measurement `transformers` does not ship: run both orders, and the data
//! picks. The failure mode is "slower than not speculating", visible in one
//! run, and never a wrong answer.

use crate::models::inkling::attn::{AttnDims, AttnWeights, LogScaling};
use crate::models::inkling::block::rms_norm;
use crate::models::inkling::layer::{decoder_layer, LayerMlp, LayerWeights};

/// Which operand occupies the first `hidden` columns of `input_proj`'s input.
///
/// `mtp_hidden_states_first` is true in every Inkling config seen so far, which
/// makes [`Concat::HiddenFirst`] the default reading — but the flag is not a
/// computation anyone has run, so the other order stays reachable and the
/// acceptance rate decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Concat {
    /// `[hidden_norm(h) ; embed_norm(e)]` — what `mtp_hidden_states_first` reads as.
    HiddenFirst,
    /// `[embed_norm(e) ; hidden_norm(h)]` — the alternative, kept to be measured.
    EmbedFirst,
}

impl Concat {
    /// Parse the environment/CLI spelling; anything unrecognised is an error at
    /// the caller rather than a silent default, because a control that quietly
    /// falls back is a control that reads as "made no difference".
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "hidden" | "hidden-first" => Some(Concat::HiddenFirst),
            "embed" | "embed-first" => Some(Concat::EmbedFirst),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Concat::HiddenFirst => "hidden-first",
            Concat::EmbedFirst => "embed-first",
        }
    }
}

/// One MTP head's weights: the wrapper, plus the ordinary decoder block it
/// wraps. Borrowed rather than owned — the caller already holds these through
/// the residency cache and copying 8 blocks would be for nothing.
pub struct MtpHead<'a> {
    pub embed_norm: &'a [f32],
    pub hidden_norm: &'a [f32],
    /// `[hidden, 2 * hidden]`, row-major: row `o` is output channel `o`.
    pub input_proj: &'a [f32],
    pub lw: LayerWeights<'a>,
    pub aw: AttnWeights<'a>,
    pub mlp: LayerMlp<'a>,
}

/// Run one MTP head over a whole sequence and return its hidden states.
///
/// `hidden` is `[tokens, hidden]` from the previous stage — the main stack for
/// head 0, the previous head thereafter. `embeds` is `[tokens, hidden]`, the
/// embedding of the token each position is being asked to predict FROM; the
/// caller does the shifting, because the shift is what distinguishes head `i`
/// from head `i+1` and hiding it in here would make the chain opaque.
///
/// The block carries attention, so this is a whole-sequence operation and not
/// a per-position one. That is why the sequence is an argument rather than the
/// last row: drafting from the final position still needs the positions before
/// it to attend over.
pub fn mtp_block(
    hidden: &[f32],
    embeds: &[f32],
    head: &MtpHead<'_>,
    dims: &AttnDims,
    log_scaling: Option<LogScaling>,
    mask: &[f32],
    tokens: usize,
    order: Concat,
) -> Vec<f32> {
    let h = dims.hidden;
    debug_assert_eq!(hidden.len(), tokens * h, "hidden is [tokens, hidden]");
    debug_assert_eq!(embeds.len(), tokens * h, "embeds is [tokens, hidden]");
    debug_assert_eq!(
        head.input_proj.len(),
        h * 2 * h,
        "input_proj is [hidden, 2 * hidden]"
    );

    // Each operand is normed by its OWN weight before the concat. Two norms is
    // the tell that they are joined rather than summed: a residual add would
    // want one.
    let hn = rms_norm(hidden, head.hidden_norm, dims.rms_eps, tokens, h);
    let en = rms_norm(embeds, head.embed_norm, dims.rms_eps, tokens, h);

    let mut x = vec![0f32; tokens * h];
    let mut cat = vec![0f32; 2 * h];
    for t in 0..tokens {
        let (first, second) = match order {
            Concat::HiddenFirst => (&hn[t * h..(t + 1) * h], &en[t * h..(t + 1) * h]),
            Concat::EmbedFirst => (&en[t * h..(t + 1) * h], &hn[t * h..(t + 1) * h]),
        };
        cat[..h].copy_from_slice(first);
        cat[h..].copy_from_slice(second);
        for o in 0..h {
            let w = &head.input_proj[o * 2 * h..(o + 1) * 2 * h];
            let mut acc = 0f32;
            for k in 0..2 * h {
                acc += w[k] * cat[k];
            }
            x[t * h + o] = acc;
        }
    }

    let (y, _) = decoder_layer(&x, &head.lw, &head.aw, dims, log_scaling, &head.mlp, mask, tokens);
    y
}
