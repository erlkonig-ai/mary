//! **The resident mind** — the vocabulary of an Inkling turn, and the
//! `drive::mind::Mind` that produces one from a `session::Session` held IN
//! THIS PROCESS.
//!
//! # What this used to be, and why it is not that any more
//!
//! Until 2026-08-30 this file was called `serve.rs` and it was a WIRE: a
//! framed-stream protocol with nine content types, a `ServeClient` that spawned
//! a model process and talked to it over two pipes, and a `ServePair` that
//! fanned one such stream out to two tensor-parallel ranks — the second of them
//! launched over passwordless `ssh`. Three process kinds (`inkling_drive`,
//! `inkling_serve_pair`, two `inkling_serve`) and a protocol between the mind
//! and the model.
//!
//! It existed for one reason, stated plainly in its own header: *"a `Session`
//! lives in one process's address space, and the process that wants it — drive
//! — must not link mary. Drive builds GPU-free in seconds and that is protected
//! on purpose."* That reason is retired. Drive links mary unconditionally now;
//! the resident mind runs where the model runs, which is a DGX Spark, where the
//! compile is fast anyway.
//!
//! # What it cost, measured, per generated token
//!
//! Framing rule for every number in this section: **seconds per GENERATED
//! TOKEN**, 42 layers, TP2 across both Sparks, greedy, measured INSIDE the
//! serving process around the `Session` calls — except the third row, which is
//! measured by Drive around a whole one-token consult and is therefore the only
//! one that includes the protocol.
//!
//! | | ms per token | what it is |
//! |---|---|---|
//! | decode inside one 32-token consult | 55.8 / 58.7 | the model, amortising one consult over 32 tokens |
//! | decode as Drive actually used it | p50 **82** (n=768, min 58, p25 65, p75 114, p95 186) | the same model, one token per consult |
//! | the difference | **~26** | not the model |
//!
//! Drive consults ONE TOKEN AT A TIME — deliberately, because a one-token
//! consult is the tensor-parallel-safe stop boundary (see
//! `InklingMind::turn`) — so every single token paid a framed-stream round
//! trip out through the pair proxy, a fan-out to two rank pipes, an `ssh`
//! channel to the second box, two JSON `TurnEnd` envelopes back, and a fresh
//! `Session` entry. **About a third of resident decode was transport.** That is
//! what collapsing to one process deletes, and it is the whole reason for the
//! change.
//!
//! It also deletes the `ssh` trust edge — rank 0 launching rank 1 over
//! passwordless `ssh` meant the two boxes were not independent failure domains,
//! and were not independent security domains either — and a whole class of
//! version skew, because there is now ONE binary and it is deployed unchanged
//! to both boxes.
//!
//! # The shape now
//!
//! ```text
//!   box A (rank 0)                          box B (rank 1)
//!   ─────────────                           ─────────────
//!   one binary                              THE SAME binary
//!     Drive shell + ledger + pile             (no Drive, no pile)
//!     InklingMind  ─────┐                     Engine::follow
//!     Engine            │                       │
//!     Session (rank 0)  │  rendezvous socket    Session (rank 1)
//!        │              └──── one pass command ─────▶ │
//!        └──────────────── NCCL on ConnectX ──────────┘
//! ```
//!
//! Two processes, both of them this program, plus the playground sandbox. The
//! only wire left is the one that was always there: the tensor-parallel
//! rendezvous socket `tpcomm::Group` already keeps, carrying one small
//! command per model PASS rather than one JSON envelope per token.
//!
//! # This module is still GPU-free, and that is still load-bearing
//!
//! `InklingMind` talks to a [`Model`], not to a `Session`. The real
//! implementation is `engine::Engine`, which holds the `Session`, the
//! tokenizer, the incremental detokenizer and the rank link, and which only
//! compiles on the CUDA lane. Everything HERE — the typed context vocabulary,
//! the native output parser, the carry accounting, the turn shaping, the
//! coverage arithmetic — compiles and is TESTED anywhere, against an in-memory
//! scripted model. The seam that used to be a pipe (and needed a shell-script
//! fixture and a real subprocess to test) is now a trait, so the same
//! behaviours are pinned by tests that allocate nothing and spawn nothing.
//!
//! # TWO PROPERTIES WERE DELIBERATELY LOST. Both are named here.
//!
//! `ServePair` did two things this design does not, and neither is an oversight.
//!
//! ## 1. Byte-for-byte cross-rank confirmation, BEFORE release
//!
//! **Where it used to live:** `serve::broker_rank_streams`, plus
//! `serve::compare_turn_ends`. The proxy drained BOTH ranks' token streams,
//! held each rank-0 fragment until rank 1's corresponding fragment arrived, and
//! released it downstream only if the two were byte-identical; at the end of a
//! turn it required `turn`, `tokens`, `token_ids`, `delta_tokens`, `carried`,
//! `stopped` and `position` to be equal, and classified any disagreement as
//! `ServePairFailure::Divergence`. A fragment that had not been confirmed had
//! not been spoken.
//!
//! **What replaces it:** nothing, at that granularity. One process per box
//! decodes from the argmax IT computed, and there is no second observer to
//! compare against. What makes that defensible is that the argmax is itself a
//! collective: `tpcomm::Group::argmax_across` gathers `(best value,
//! local row)` from every rank through an all-reduce and every rank reads back
//! THE SAME BUFFER. Two ranks that disagreed on a token id would mean the
//! all-reduce delivered different bytes to each rank — a fabric or NCCL fault,
//! not a model divergence. The old check could catch exactly that class and
//! nothing else, since both ranks were already fed identical inputs.
//!
//! **What still detects it, and what got weaker:** [`Model::agree_sequence`].
//! Each rank hashes every token id every pass returns, together with the
//! position after it, and the two ranks exchange that 32-byte digest at every
//! logical-turn boundary and at every reinitialization. It catches the same
//! class, it also covers CONTEXT passes the old check never saw, and it costs
//! one small round trip per Drive turn instead of one per token.
//!
//! What is genuinely gone is the GATING. The old check was a valve: a byte
//! reached the voice only after the peer confirmed it. The new one is an alarm:
//! the bytes are spoken, and divergence is reported at the end of the turn. A
//! run that diverges therefore SAYS something wrong before it is caught. That
//! is the price of not having a second process to arbitrate, it is paid
//! knowingly, and restoring the valve would mean re-introducing a per-token
//! peer round trip — i.e. re-introducing the 26 ms this change exists to
//! delete.
//!
//! ## 2. A bounded death for a peer that dies INSIDE a collective
//!
//! **Where it used to live:** `ServePair::consult` ran a reader thread per rank
//! and `broker_rank_streams` turned "rank 1's stream truncated" into a killed
//! rank 0 — the fifth invariant of the old design ("a dead rank must kill its
//! peer", because the survivor blocks in NCCL with no timeout).
//!
//! **What replaces it:** [`Model::agree_sequence`]'s transport plus a hangup
//! poll. `tpcomm::Group::peer_alive` polls the rendezvous socket for
//! `POLLHUP`/`POLLERR` at every pass boundary, and a write to a dead peer fails
//! with `EPIPE`, so a peer that dies BETWEEN passes is a loud error within one
//! pass.
//!
//! **What got weaker:** a peer that dies while both ranks are inside a
//! collective still hangs the survivor, because nothing here is watching the
//! link concurrently. Restoring the old bound needs one of two things, and
//! neither is in this change: a watchdog thread owning a `try_clone` of the
//! link socket that force-exits the process on EOF, or NCCL's own
//! `NCCL_ASYNC_ERROR_HANDLING=1` plus `ncclCommGetAsyncError` polled from the
//! layer loop. Written down here rather than discovered at 3am.
//!
//! # What did NOT change
//!
//! Everything that was model-side logic rather than transport: the typed
//! context codec and its content-only tokenizer view (a tool result spelling
//! `<|message_model|>` is still ordinary content, never structure), the
//! execution manifest, the context preflight, the one-token carry and its
//! accounting in [`TurnEnd::carried`], the native output parser and its strict
//! single-call rule, the monologue-coordinate discipline, and the
//! replacement/reinitialization boundary. Those were never the wire's; they are
//! what a turn of this model IS.

use anyhow::{Context, Result};

#[cfg(any(feature = "tokenizer", test))]
const MESSAGE_MODEL: &str = "<|message_model|>";
#[cfg(any(feature = "tokenizer", test))]
const MESSAGE_SYSTEM: &str = "<|message_system|>";
#[cfg(any(feature = "tokenizer", test))]
const MESSAGE_TOOL: &str = "<|message_tool|>";
#[cfg(any(feature = "tokenizer", test))]
const CONTENT_TEXT: &str = "<|content_text|>";
#[cfg(any(feature = "tokenizer", test))]
const CONTENT_XML: &str = "<|content_xml|>";
#[cfg(any(feature = "tokenizer", test))]
const CONTENT_THINKING: &str = "<|content_thinking|>";
#[cfg(any(feature = "tokenizer", test))]
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
#[cfg(any(feature = "tokenizer", test))]
const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
#[cfg(any(feature = "tokenizer", test))]
const END_MESSAGE: &str = "<|end_message|>";
/// The dMel audio input, as the shipped template renders an audio content
/// part: `<|content_audio_input|>`, one placeholder per 50 ms frame, then
/// `<|audio_end|>` (`processor_config.json`: `audio_bos_token`, `audio_token`).
const CONTENT_AUDIO_INPUT: &str = "<|content_audio_input|>";
const AUDIO_SLOT: &str = "<|unused_200053|>";
const AUDIO_END: &str = "<|audio_end|>";
/// The template's image part: `<|content_image|>`, one placeholder per
/// patch, then `<|end_message|>` -- there is no image end token
/// (`processor_config.json`: `image_bos_token`, `image_token`; the template's
/// `image` branch).
const CONTENT_IMAGE: &str = "<|content_image|>";
const IMAGE_SLOT: &str = "<|unused_200054|>";
/// One dMel frame is this many levels, each below `DMEL_LEVELS`
/// (`audio_config`: `n_mel_bins`, `mel_vocab_size`).
pub const DMEL_BINS: usize = 80;
pub const DMEL_LEVELS: usize = 16;

/// The single tool exposed to Inkling, in the exact compact/sorted JSON shape
/// produced by the checkpoint's shipped `chat_template.jinja`.
///
/// This is content, not framing: it is deliberately passed through the
/// content-only tokenizer even though it is trusted static text.
pub const EXEC_TOOL_DECLARATION: &str = concat!(
    "[{\"description\":\"Execute one shell command in Drive's sandbox.\",",
    "\"name\":\"exec\",\"parameters\":{\"properties\":{\"command\":{",
    "\"type\":\"string\"}},\"required\":[\"command\"],\"type\":\"object\"},",
    "\"type\":\"function\"}]"
);

#[cfg(any(feature = "tokenizer", test))]
const DEFAULT_THINKING_EFFORT: &str = "Thinking effort level: 0.9";

/// Runtime ids of the TML tokens this vocabulary understands.
///
/// No numeric vocabulary constants live in the adapter. The engine resolves
/// every field by token spelling from the exact tokenizer it loaded and
/// announces the result in [`Ready`], so `InklingMind` parses generated
/// structure by id without owning or reconstructing that tokenizer. It stays a
/// serializable value even though nothing serializes it between processes any
/// more: it is durable evidence (`super::telemetry`), and it is what a scripted
/// test model announces.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InklingSpecialIds {
    pub message_model: u32,
    pub message_system: u32,
    pub message_tool: u32,
    pub content_text: u32,
    pub content_xml: u32,
    pub content_thinking: u32,
    pub content_invoke_tool_json: u32,
    pub content_model_end_sampling: u32,
    pub end_message: u32,
    /// The three the audio input is spelled with; see [`SenseMedia::Dmel`].
    pub content_audio_input: u32,
    pub audio_slot: u32,
    pub audio_end: u32,
    /// The two the image input is spelled with; see [`SenseMedia::Patches`].
    pub content_image: u32,
    pub image_slot: u32,
    /// Every added token marked `special` by this tokenizer, including kinds
    /// this minimal vocabulary does not support. Generated unknown specials are
    /// rejected rather than leaked into visible text.
    pub all_special: Vec<u32>,
    /// Decoder contribution of each special id in isolation, obtained from
    /// the exact runtime tokenizer. A streaming decode can flush pending
    /// payload together with this suffix when a structural token arrives.
    pub decoded_special: Vec<(u32, String)>,
}

impl InklingSpecialIds {
    fn is_special(&self, id: u32) -> bool {
        self.all_special.contains(&id)
    }

    fn decoded_special(&self, id: u32) -> Option<&str> {
        self.decoded_special
            .iter()
            .find_map(|(candidate, decoded)| (*candidate == id).then_some(decoded.as_str()))
    }
}

/// One historical or live result of the sole `exec` tool.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecResultContext {
    /// Exact shell command whose result this is. It is used to reconstruct a
    /// historical assistant call and to bind a live result to its outstanding
    /// native call in `InklingMind`.
    pub command: String,
    /// Drive's deliberate text projection of its typed result, including any
    /// structural status annotation that adds information.
    pub content: String,
}

/// One ordered part of a completed historical model response.
///
/// Adjacent parts of the same textual kind are fragments of one model block.
/// The codec concatenates them *before* content tokenization, preserving exact
/// tokens regardless of archival fragment boundaries. An `Exec` part is
/// instead a complete native call and is valid only as the final model part of
/// a response.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InklingHistoryPart {
    Thinking { content: String },
    Text { content: String },
    Exec { command: String },
}

/// One completed historical model response and its optional native result.
///
/// The parts retain model-channel order. A response either contains only text
/// and thinking parts, or contains exactly one final `Exec` part followed by
/// `tool_result`. This shape deliberately stores the command once: the result
/// is structurally attached to that final call, so there is no second command
/// string that can disagree with it.
///
/// This representation can retain arbitrary thinking/text alternation. Drive's current
/// `Turn` cannot yet produce that full fidelity because it separates reasoning
/// from speech instead of carrying one ordered emitted-part vector; callers
/// must not infer an order from those two accumulated fields during rollover.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InklingHistoryResponse {
    pub parts: Vec<InklingHistoryPart>,
    pub tool_result: Option<String>,
}

impl InklingHistoryResponse {
    /// Lift the former exec-result-only history into the ordered response
    /// model without retaining a second compatibility representation.
    pub fn exec(result: ExecResultContext) -> Self {
        Self {
            parts: vec![InklingHistoryPart::Exec {
                command: result.command,
            }],
            tool_result: Some(result.content),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let execs = self
            .parts
            .iter()
            .filter(|part| matches!(part, InklingHistoryPart::Exec { .. }))
            .count();
        anyhow::ensure!(
            execs <= 1,
            "a historical response contains multiple exec calls"
        );
        let final_exec = matches!(self.parts.last(), Some(InklingHistoryPart::Exec { .. }));
        anyhow::ensure!(
            execs == 0 || final_exec,
            "a historical exec call must be the response's final model part"
        );
        if final_exec {
            anyhow::ensure!(
                self.tool_result.is_some(),
                "a completed historical exec response requires its tool result"
            );
        } else {
            anyhow::ensure!(
                self.tool_result.is_none(),
                "a historical tool result has no final exec call"
            );
        }
        Ok(())
    }
}

/// Typed context inserted into Inkling's retained conversation.
///
/// Initialization is one batch because Drive's memory cover is already a
/// history: the one generation prompt must come *after* every historical pair,
/// never between the system prefix and those pairs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InklingContext {
    Initialize {
        system: String,
        history: Vec<InklingHistoryResponse>,
    },
    /// Result of the native call already present in the retained KV sequence,
    /// and whatever was sensed while it ran: the result's tool message, then
    /// one tool message per sense record (the order the world released
    /// them), then the generation prompt. Sensing rides with a result rather
    /// than waiting behind it, because a mind that acts every turn would
    /// otherwise never hear or see at all.
    ToolResult {
        result: ExecResultContext,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        sensed: Vec<SenseRecord>,
    },
    /// A complete response which predates this live `InklingMind` (for example
    /// a Drive memory-cover response). Its model parts and optional result are
    /// inserted together.
    HistoricalResponse { response: InklingHistoryResponse },
    /// Start another autonomous assistant response after a completed text-only
    /// response. A tool result already carries this prompt itself.
    GenerationPrompt,
    /// Something was sensed and nothing else happened: one tool message per
    /// record, each named for the faculty that sensed it and holding the
    /// template's own part for its medium (the audio part for dMel levels,
    /// the image part for patches). The payloads are numbers inside typed
    /// records and never text, so free text cannot smuggle the structural
    /// markers in. Ends with the generation prompt, like a tool result: the
    /// world spoke or moved, now she thinks.
    Sensed { records: Vec<SenseRecord> },
}

/// What a sense delivered: one record from the faculty named `source`, in
/// one of the media the model can embed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SenseRecord {
    pub source: String,
    pub media: SenseMedia,
}

/// The media a sense record can carry, one per content type on the stream.
/// A new medium is a new variant here, a session input to embed it, and a
/// content type for the faculties that produce it; a new SOURCE of an
/// existing medium (a video call, a file) needs nothing here at all.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SenseMedia {
    /// `frames * DMEL_BINS` dMel levels (each below `DMEL_LEVELS`, 50 ms a
    /// frame, from the ear's front end): the template's audio part.
    Dmel { levels: Vec<u8> },
    /// Whole patches as the eye's front end lays them out
    /// (`super::patches`: little-endian f32, `PATCH_BYTES` each): the
    /// template's image part, one placeholder per patch.
    Patches { patches: Vec<u8> },
}

impl SenseRecord {
    /// How many placeholder slots this record spells: frames or patches.
    pub fn slots(&self) -> Result<usize> {
        match &self.media {
            SenseMedia::Dmel { levels } => heard_frames(levels),
            SenseMedia::Patches { patches } => super::patches::count(patches),
        }
    }
}

/// Where a typed context would be placed if its preflight succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextPlacement {
    /// Extend the resident sequence, including its one unfed carry token.
    Append,
    /// Replace the resident sequence; admission begins at an empty Session.
    Replace,
}

/// Price one exact typed delta and its whole response, changing nothing.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextPreflight {
    pub placement: ContextPlacement,
    pub context: InklingContext,
    pub max_response_tokens: usize,
}

/// The exact arithmetic used for context admission.
///
/// `required_end == None` is checked-arithmetic overflow and is necessarily a
/// rejection. A rejected preflight has changed neither the Session nor any of
/// the engine's pending context, carry, decoder, or turn state.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextPreflighted {
    pub placement: ContextPlacement,
    pub position: usize,
    pub carried: usize,
    pub delta_tokens: usize,
    pub max_response_tokens: usize,
    pub required_end: Option<usize>,
    pub context_budget: usize,
    pub fits: bool,
}

impl ContextPreflighted {
    pub fn validate_for(&self, request: &ContextPreflight, context_budget: usize) -> Result<()> {
        anyhow::ensure!(
            self.placement == request.placement,
            "context-preflight placement changed from {:?} to {:?}",
            request.placement,
            self.placement
        );
        anyhow::ensure!(
            self.max_response_tokens == request.max_response_tokens,
            "context-preflight response bound changed from {} to {}",
            request.max_response_tokens,
            self.max_response_tokens
        );
        anyhow::ensure!(
            self.context_budget == context_budget,
            "context-preflight budget {} disagrees with READY budget {context_budget}",
            self.context_budget
        );
        anyhow::ensure!(
            self.carried <= 1,
            "context-preflight reported {} carry tokens",
            self.carried
        );
        let recomputed = context_preflight(
            self.placement,
            self.position,
            self.carried,
            self.delta_tokens,
            self.max_response_tokens,
            self.context_budget,
        )?;
        anyhow::ensure!(
            *self == recomputed,
            "context-preflight evidence is not self-consistent: {self:?}"
        );
        Ok(())
    }
}

/// Compute the singular response-admission inequality.
///
/// A response's first token is predicted by the pass which appends carry and
/// delta, so only `R - 1` further positions have to be retained:
/// `p + carry + D + (R - 1) <= context_budget`.
pub fn context_preflight(
    placement: ContextPlacement,
    position: usize,
    carried: usize,
    delta_tokens: usize,
    max_response_tokens: usize,
    context_budget: usize,
) -> Result<ContextPreflighted> {
    anyhow::ensure!(
        max_response_tokens > 0,
        "context preflight requires a nonzero response bound"
    );
    let (base_position, base_carried) = match placement {
        ContextPlacement::Append => (position, carried),
        ContextPlacement::Replace => (0, 0),
    };
    let required_end = base_position
        .checked_add(base_carried)
        .and_then(|end| end.checked_add(delta_tokens))
        .and_then(|end| end.checked_add(max_response_tokens - 1));
    Ok(ContextPreflighted {
        placement,
        position: base_position,
        carried: base_carried,
        delta_tokens,
        max_response_tokens,
        required_end,
        context_budget,
        fits: required_end.is_some_and(|end| end <= context_budget),
    })
}

/// One completed sequence has been replaced against the same warm weights.
///
/// The replacement initialization has been tokenized and staged, but has not
/// yet been attended to: the next consult performs the fresh Session prefill.
/// `previous_*` name the sequence that was discarded. They make a rollover an
/// observable boundary without introducing another long-lived sequence id.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reinitialized {
    pub previous_position: usize,
    pub previous_turns: usize,
    pub initialization_tokens: usize,
}

#[cfg(any(feature = "tokenizer", test))]
fn canonical_exec_call_json(command: &str) -> String {
    // The shipped template fixes wrapper order as `name`, then `args`, while
    // recursively sorting the argument object. There is one argument, so JSON
    // string escaping is the only variable operation.
    let command = serde_json::to_string(command).expect("a Rust string always serializes to JSON");
    format!(r#"{{"name":"exec","args":{{"command":{command}}}}}"#)
}

/// Server-side TML encoder built from the exact checkpoint tokenizer.
///
/// `tokenizer` retains special added tokens for generated-token decoding and
/// id lookup. `content` is independently reconstructed from the *same JSON*
/// after removing every `added_tokens[*].special == true` entry. Consequently
/// a system prompt or tool result containing the literal spelling
/// `<|message_model|>` is ordinary content tokens, never model-role framing.
#[cfg(feature = "tokenizer")]
pub struct InklingContextCodec {
    content: tokenizers::Tokenizer,
    special_ids: InklingSpecialIds,
}

/// How many frames `levels` is, refusing a ragged or out-of-range record.
pub fn heard_frames(levels: &[u8]) -> Result<usize> {
    anyhow::ensure!(
        !levels.is_empty() && levels.len() % DMEL_BINS == 0,
        "a heard record is {DMEL_BINS} dMel levels per frame; {} arrived",
        levels.len()
    );
    if let Some(bad) = levels.iter().find(|&&l| usize::from(l) >= DMEL_LEVELS) {
        anyhow::bail!("dMel level {bad} is not below {DMEL_LEVELS}");
    }
    Ok(levels.len() / DMEL_BINS)
}

#[cfg(feature = "tokenizer")]
impl InklingContextCodec {
    /// Build both tokenizer views and resolve every structural id by spelling.
    pub fn from_json(tokenizer_json: &[u8]) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json)
            .map_err(|error| anyhow::anyhow!("load Inkling tokenizer: {error}"))?;
        let required = |spelling: &str| {
            tokenizer
                .token_to_id(spelling)
                .with_context(|| format!("Inkling tokenizer lacks required token {spelling:?}"))
        };
        let mut all_special = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, token)| token.special.then_some(id))
            .collect::<Vec<_>>();
        all_special.sort_unstable();
        all_special.dedup();
        let decoded_special = all_special
            .iter()
            .map(|id| {
                tokenizer
                    .decode(&[*id], false)
                    .map(|decoded| (*id, decoded))
                    .map_err(|error| {
                        anyhow::anyhow!("decode Inkling special token id {id}: {error}")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let special_ids = InklingSpecialIds {
            message_model: required(MESSAGE_MODEL)?,
            message_system: required(MESSAGE_SYSTEM)?,
            message_tool: required(MESSAGE_TOOL)?,
            content_text: required(CONTENT_TEXT)?,
            content_xml: required(CONTENT_XML)?,
            content_thinking: required(CONTENT_THINKING)?,
            content_invoke_tool_json: required(CONTENT_INVOKE_TOOL_JSON)?,
            content_model_end_sampling: required(CONTENT_MODEL_END_SAMPLING)?,
            end_message: required(END_MESSAGE)?,
            content_audio_input: required(CONTENT_AUDIO_INPUT)?,
            audio_slot: required(AUDIO_SLOT)?,
            audio_end: required(AUDIO_END)?,
            content_image: required(CONTENT_IMAGE)?,
            image_slot: required(IMAGE_SLOT)?,
            all_special,
            decoded_special,
        };
        for id in [
            special_ids.message_model,
            special_ids.message_system,
            special_ids.message_tool,
            special_ids.content_text,
            special_ids.content_xml,
            special_ids.content_thinking,
            special_ids.content_invoke_tool_json,
            special_ids.content_model_end_sampling,
            special_ids.end_message,
            special_ids.content_audio_input,
            special_ids.audio_slot,
            special_ids.audio_end,
            special_ids.content_image,
            special_ids.image_slot,
        ] {
            anyhow::ensure!(
                special_ids.is_special(id),
                "Inkling structural token id {id} is not declared special"
            );
        }

        let mut content_json: serde_json::Value = serde_json::from_slice(tokenizer_json)
            .context("parse Inkling tokenizer JSON for the content-only view")?;
        let added = content_json
            .get_mut("added_tokens")
            .and_then(serde_json::Value::as_array_mut)
            .context("Inkling tokenizer JSON has no added_tokens array")?;
        added.retain(|token| {
            !token
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        });
        let content_json = serde_json::to_vec(&content_json)
            .context("serialize Inkling content-only tokenizer JSON")?;
        let content = tokenizers::Tokenizer::from_bytes(&content_json)
            .map_err(|error| anyhow::anyhow!("load Inkling content-only tokenizer: {error}"))?;
        let codec = Self {
            content,
            special_ids,
        };

        // Fail at startup, rather than on the first hostile result, if removing
        // added tokens did not actually make their spellings content-only.
        for spelling in codec
            .special_ids
            .all_special
            .iter()
            .filter_map(|id| tokenizer.id_to_token(*id))
        {
            codec.encode_content(&spelling).with_context(|| {
                format!("special-token spelling {spelling:?} is not content-safe")
            })?;
        }
        Ok(codec)
    }

    /// Runtime ids announced in [`Ready`] and used by [`NativeOutputParser`].
    pub fn special_ids(&self) -> &InklingSpecialIds {
        &self.special_ids
    }

    /// Encode unframed probe text through the same safe content path.
    pub fn encode_raw_content(&self, text: &str) -> Result<Vec<usize>> {
        self.encode_content(text)
    }

    /// Encode one typed context record into exact model token ids.
    pub fn encode(&self, context: &InklingContext) -> Result<Vec<usize>> {
        let mut ids = Vec::new();
        match context {
            InklingContext::Initialize { system, history } => {
                // Shipped template order: tool declaration, system message,
                // default effort (immediately before the first non-system
                // message, or as the all-system fallback), history, prompt.
                ids.push(self.special_ids.message_system as usize);
                self.push_content(&mut ids, "tool_declare")?;
                ids.push(self.special_ids.content_xml as usize);
                self.push_content(&mut ids, EXEC_TOOL_DECLARATION)?;
                ids.push(self.special_ids.end_message as usize);

                ids.push(self.special_ids.message_system as usize);
                ids.push(self.special_ids.content_text as usize);
                self.push_content(&mut ids, system)?;
                ids.push(self.special_ids.end_message as usize);

                ids.push(self.special_ids.message_system as usize);
                ids.push(self.special_ids.content_text as usize);
                self.push_content(&mut ids, DEFAULT_THINKING_EFFORT)?;
                ids.push(self.special_ids.end_message as usize);

                for response in history {
                    self.push_historical_response(&mut ids, response)?;
                }
                ids.push(self.special_ids.message_model as usize);
            }
            InklingContext::ToolResult { result, sensed } => {
                self.push_tool_result(&mut ids, result)?;
                for record in sensed {
                    self.push_sense_part(&mut ids, record)?;
                }
                ids.push(self.special_ids.message_model as usize);
            }
            InklingContext::HistoricalResponse { response } => {
                self.push_historical_response(&mut ids, response)?;
                ids.push(self.special_ids.message_model as usize);
            }
            InklingContext::GenerationPrompt => {
                ids.push(self.special_ids.message_model as usize);
            }
            InklingContext::Sensed { records } => {
                anyhow::ensure!(!records.is_empty(), "a Sensed context with no records");
                for record in records {
                    self.push_sense_part(&mut ids, record)?;
                }
                ids.push(self.special_ids.message_model as usize);
            }
        }
        Ok(ids)
    }

    /// The sense records a context carries, in the order [`Self::encode`]
    /// emitted their slots: what the Session stages behind those slots.
    /// Empty for everything but `Sensed` and a `ToolResult` with senses.
    pub fn sensed<'a>(&self, context: &'a InklingContext) -> &'a [SenseRecord] {
        match context {
            InklingContext::Sensed { records } => records,
            InklingContext::ToolResult { sensed, .. } => sensed,
            _ => &[],
        }
    }

    /// The placeholder id a medium stands behind.
    pub fn slot_of(&self, media: &SenseMedia) -> usize {
        match media {
            SenseMedia::Dmel { .. } => self.special_ids.audio_slot as usize,
            SenseMedia::Patches { .. } => self.special_ids.image_slot as usize,
        }
    }

    /// One tool message named for the record's source whose content part is
    /// the template's part for its medium: the audio part
    /// (`content_audio_input`, one slot per frame, `audio_end`) or the image
    /// part (`content_image`, one slot per patch, no end token).
    fn push_sense_part(&self, ids: &mut Vec<usize>, record: &SenseRecord) -> Result<()> {
        let slots = record.slots()?;
        ids.push(self.special_ids.message_tool as usize);
        self.push_content(ids, &record.source)?;
        match &record.media {
            SenseMedia::Dmel { .. } => {
                ids.push(self.special_ids.content_audio_input as usize);
                ids.extend(std::iter::repeat_n(self.special_ids.audio_slot as usize, slots));
                ids.push(self.special_ids.audio_end as usize);
            }
            SenseMedia::Patches { .. } => {
                ids.push(self.special_ids.content_image as usize);
                ids.extend(std::iter::repeat_n(self.special_ids.image_slot as usize, slots));
            }
        }
        ids.push(self.special_ids.end_message as usize);
        Ok(())
    }

    fn push_historical_response(
        &self,
        ids: &mut Vec<usize>,
        response: &InklingHistoryResponse,
    ) -> Result<()> {
        response.validate()?;
        let mut parts = response.parts.iter().peekable();
        while let Some(part) = parts.next() {
            match part {
                InklingHistoryPart::Thinking { content } => {
                    let mut content = content.clone();
                    while let Some(InklingHistoryPart::Thinking { content: next }) = parts.peek() {
                        content.push_str(next);
                        parts.next();
                    }
                    self.push_historical_text_part(
                        ids,
                        self.special_ids.content_thinking,
                        &content,
                    )?;
                }
                InklingHistoryPart::Text { content } => {
                    let mut content = content.clone();
                    while let Some(InklingHistoryPart::Text { content: next }) = parts.peek() {
                        content.push_str(next);
                        parts.next();
                    }
                    self.push_historical_text_part(ids, self.special_ids.content_text, &content)?;
                }
                InklingHistoryPart::Exec { command } => {
                    ids.push(self.special_ids.message_model as usize);
                    self.push_content(ids, "exec")?;
                    ids.push(self.special_ids.content_invoke_tool_json as usize);
                    self.push_content(ids, &canonical_exec_call_json(command))?;
                    ids.push(self.special_ids.end_message as usize);
                }
            }
        }
        ids.push(self.special_ids.content_model_end_sampling as usize);
        if let Some(content) = &response.tool_result {
            self.push_tool_result_content(ids, content)?;
        }
        Ok(())
    }

    fn push_historical_text_part(
        &self,
        ids: &mut Vec<usize>,
        kind: u32,
        content: &str,
    ) -> Result<()> {
        ids.push(self.special_ids.message_model as usize);
        ids.push(kind as usize);
        self.push_content(ids, content)?;
        ids.push(self.special_ids.end_message as usize);
        Ok(())
    }

    fn push_tool_result(&self, ids: &mut Vec<usize>, result: &ExecResultContext) -> Result<()> {
        self.push_tool_result_content(ids, &result.content)
    }

    fn push_tool_result_content(&self, ids: &mut Vec<usize>, content: &str) -> Result<()> {
        ids.push(self.special_ids.message_tool as usize);
        self.push_content(ids, "exec")?;
        ids.push(self.special_ids.content_text as usize);
        self.push_content(ids, content)?;
        ids.push(self.special_ids.end_message as usize);
        Ok(())
    }

    fn push_content(&self, ids: &mut Vec<usize>, text: &str) -> Result<()> {
        ids.extend(self.encode_content(text)?);
        Ok(())
    }

    fn encode_content(&self, text: &str) -> Result<Vec<usize>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let ids = self
            .content
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!("encode Inkling content: {error}"))?
            .get_ids()
            .iter()
            .map(|id| *id as usize)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            ids.iter()
                .all(|id| !self.special_ids.is_special(*id as u32)),
            "content-only tokenizer produced a special token id"
        );
        Ok(ids)
    }
}

/// Canonical BLAKE3 input for one observed execution manifest.
///
/// A manifest is a fixed-order sequence of `(name, value)` byte strings. Every
/// string is preceded by its unsigned 64-bit big-endian length, including field
/// names, so neither concatenation nor a future variable-width field can make
/// two different manifests share an encoding. Callers choose and document the
/// field order; this type supplies the one encoding and hash implementation.
pub struct ExecutionManifest {
    hasher: blake3::Hasher,
}

impl ExecutionManifest {
    /// Begin a manifest under a named compatibility profile.
    pub fn new(profile: &str) -> Self {
        let mut manifest = Self {
            hasher: blake3::Hasher::new(),
        };
        manifest.field(
            "manifest-format",
            b"mary-execution-manifest-lp64be-blake3-v1",
        );
        manifest.field("profile", profile.as_bytes());
        manifest
    }

    /// Append one named byte string.
    pub fn field(&mut self, name: &str, value: &[u8]) {
        Self::length_prefixed(&mut self.hasher, name.as_bytes());
        Self::length_prefixed(&mut self.hasher, value);
    }

    /// Append an unsigned integer using its canonical decimal spelling.
    pub fn usize(&mut self, name: &str, value: usize) {
        self.field(name, value.to_string().as_bytes());
    }

    /// Append exactly `length` bytes from a reader without retaining them.
    ///
    /// This is used for `/proc/self/exe`: the executable itself, not a path,
    /// timestamp, build id, or separately defined checksum, enters the single
    /// manifest hash.
    pub fn reader(
        &mut self,
        name: &str,
        length: u64,
        mut reader: impl std::io::Read,
    ) -> Result<()> {
        Self::length_prefixed(&mut self.hasher, name.as_bytes());
        self.hasher.update(&length.to_be_bytes());
        let mut remaining = length;
        let mut buffer = [0u8; 64 * 1024];
        while remaining > 0 {
            let want = remaining.min(buffer.len() as u64) as usize;
            let read = reader
                .read(&mut buffer[..want])
                .context("read a length-prefixed execution-manifest field")?;
            anyhow::ensure!(
                read > 0,
                "execution-manifest field ended {remaining} bytes early"
            );
            self.hasher.update(&buffer[..read]);
            remaining -= read as u64;
        }
        let mut extra = [0u8; 1];
        anyhow::ensure!(
            reader.read(&mut extra)? == 0,
            "execution-manifest field grew after its length was observed"
        );
        Ok(())
    }

    /// Finish as the canonical uppercase 64-hex-digit identity.
    pub fn finish_hex(self) -> String {
        hex_32(*self.hasher.finalize().as_bytes())
    }

    fn length_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
        hasher.update(&(bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
}

fn hex_32(bytes: [u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut text = String::with_capacity(64);
    for byte in bytes {
        write!(&mut text, "{byte:02X}").expect("writing into a String is infallible");
    }
    text
}

/// What loaded, announced once when the weights are up.
///
/// This used to be a record on a wire, written after the load rather than in
/// the handshake so a client could tell "starting" from "ready". Nothing
/// transmits it now — the engine hands it to the mind by reference — but it
/// remains one serializable value for two reasons: it is the source of the
/// durable READY evidence in `super::telemetry`, and it is the whole of what a
/// `Model` implementation has to be able to say about itself.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Ready {
    /// The pile the weights came from, for diagnostics only.
    ///
    /// Two ranks may name different local paths for the same content. Runtime
    /// compatibility is decided by [`Ready::model_identity`], never by this
    /// spelling.
    pub pile: String,
    /// Canonical SimpleArchive handle of the projected model facts the runtime
    /// actually indexed, rendered as 64 hexadecimal digits.
    pub model_identity: String,
    /// RawBytes handle of the exact tokenizer bytes, rendered as 64
    /// hexadecimal digits.
    pub tokenizer_identity: String,
    /// TML ids resolved from that tokenizer by spelling at runtime. The client
    /// parses generated structure by these ids, never by decoded marker text.
    pub special_ids: InklingSpecialIds,
    /// Manifest compatibility profile. `sealed-v1` rejects runtime environment
    /// overrides before CUDA initialization; `observed-v1` records the same
    /// facts without making that exclusion claim.
    #[serde(default)]
    pub execution_profile: String,
    /// Canonical length-prefixed BLAKE3 digest of executable bytes, model and
    /// tokenizer identities, effective execution settings, GPU identity, and
    /// observable CUDA/NVRTC/NCCL facts.
    #[serde(default)]
    pub execution_identity: String,
    /// Runtime facts this phase could not observe. Every named absence also
    /// enters the digest as the literal `unavailable`; this list makes that
    /// boundary visible rather than letting the digest imply evidence it lacks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub execution_unavailable: Vec<String>,
    /// This rank's TP ordinal, absent on a non-TP process and on the pair's
    /// synthesized downstream READY. Rank is deliberately outside the shared
    /// digest; `tp_world` is inside it and the pair checks complementary roles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp_rank: Option<usize>,
    /// Effective TP world (one for a single-rank load).
    #[serde(default = "default_tp_world")]
    pub tp_world: usize,
    /// The layer range this rank runs, `[lo, hi)`.
    pub layers: [usize; 2],
    /// How many layers the whole stack has.
    pub stack: usize,
    /// Whether this is a STRICT SUBRANGE — if so its tokens are diagnostic, not
    /// the model's. Stated rather than inferred.
    pub partial: bool,
    /// Effective vocabulary width the head is sliced to.
    pub vocab: usize,
    /// Maximum token rows one prefill pass processes at once.
    pub prefill_budget: usize,
    /// Maximum positions the session may retain across all turns.
    pub context_budget: usize,
    /// Wall-clock seconds `Session::load` took. The number a RESIDENT process
    /// exists to pay ONCE.
    pub load_secs: f64,
}

fn default_tp_world() -> usize {
    1
}

/// Stop accumulating context and produce a turn.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Consult {
    /// Generate at most this many tokens. A turn always produces at least one:
    /// the token that follows what it was shown.
    pub max_tokens: usize,
}

impl Consult {
    /// A turn of at most `max_tokens` tokens.
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }
}

/// The turn is over, and this is how it went.
///
/// Every duration here is measured around the `Session` calls and nothing
/// else, so it is the model's time. That mattered when there was a pipe to
/// exclude; it still matters, because it is what makes these numbers
/// comparable ACROSS the collapse — the same quantity, measured at the same
/// place, before and after the transport was deleted. See
/// [`TurnEnd::first_token_secs`] for the framing rule that makes the warm-turn
/// number mean anything.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurnEnd {
    /// Turn ordinal on this session, from 0.
    pub turn: usize,
    /// Tokens this turn generated.
    pub tokens: usize,
    /// Exact token ids generated by the rank.
    ///
    /// Ids rather than text, because two different byte-level tokens can decode
    /// to the same bytes; the id is the unambiguous agreement signal. The
    /// deleted pair proxy compared these across ranks before accepting a turn;
    /// [`Model::agree_sequence`] now folds them into a running per-rank digest
    /// instead. They also let an end-to-end gate compare continuations across
    /// session boundaries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<u32>,
    /// Tokens of DELTA this turn attended to before generating — the new
    /// context, never a re-rendered transcript. Zero on a turn whose only input
    /// is the model's own previous output.
    pub delta_tokens: usize,
    /// The prequential score of the delta: for the LAST `delta_nll.len()` delta
    /// tokens, the negative log-likelihood in nats the model assigned each one
    /// given everything before it, measured before the model had seen it. One
    /// per delta token on every turn after turn 0 (the carry precedes the
    /// first); on turn 0 the first delta token has no predecessor and is not
    /// scored. Empty when scoring is off. THE framing rule: nats per token of
    /// DELTA, under the weights in force during this turn's first pass — not
    /// per generated token, and never the model's own words.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delta_nll: Vec<f32>,
    /// The CONTROL beside [`TurnEnd::delta_nll`]: the same delta tokens scored
    /// by the checkpoint's experts on the learned layer, over the same rows in
    /// the same pass, everything else identical. One entry per entry of
    /// `delta_nll` while a layer is frozen; empty when nothing learns. Same
    /// framing rule. The difference of the two is what learning did to this
    /// turn, measured without a second model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delta_nll_frozen: Vec<f32>,
    /// Tokens of the model's OWN previous turn this pass appended BEFORE the
    /// delta: `0` on turn 0 and `1` on every turn after it.
    ///
    /// A turn emits its last token and never feeds it back — the generation loop
    /// stops one step short rather than spend a decode step on an argmax nobody
    /// reads — so that token ends the turn in the consumer's stream and not in
    /// the KV cache. The next pass appends it at the head of its delta, and this
    /// counts it, so a reader can see the whole pass rather than the mind's
    /// half of it. A turn after turn 0 reporting `0` here is a turn whose model
    /// never heard its own last word.
    pub carried: usize,
    /// Why generation stopped: `"max_tokens"` or `"stop_token"`.
    pub stopped: String,
    /// Seconds to the FIRST token of this turn, `Session` calls only:
    /// `extend`/`prefill` over the already-tokenised ids and one forward.
    /// Tokenisation is deliberately outside this duration. On turn 0 this is
    /// the prompt's prefill; on every turn after it, it is what the KV cache
    /// saves. THE framing rule: seconds per FIRST TOKEN OF A TURN, over the
    /// layer range in [`Ready::layers`] — not per token and not per turn. Each
    /// rank reports its own wall clock, and under the collapse the number a
    /// caller sees is rank 0's rather than the slower of the two: the deleted
    /// proxy could take `max` across ranks because it held both, and nothing
    /// holds both any more.
    pub first_token_secs: f64,
    /// Seconds for the whole turn, `Session` calls only.
    pub turn_secs: f64,
    /// Positions the KV cache holds after this turn.
    pub position: usize,
}

/// Mean over the scored entries: a `NaN` entry is a row that was not scored
/// (its target was the audio slot) and is neither summed nor counted.
pub fn finite_mean(scores: &[f32]) -> Option<f64> {
    let (sum, count) = scores
        .iter()
        .filter(|x| x.is_finite())
        .fold((0f64, 0usize), |(s, c), &x| (s + x as f64, c + 1));
    (count > 0).then(|| sum / count as f64)
}

impl TurnEnd {
    /// Mean of [`TurnEnd::delta_nll`] in nats per scored delta token, `None`
    /// when nothing was scored.
    pub fn delta_mean_nll(&self) -> Option<f64> {
        finite_mean(&self.delta_nll)
    }

    /// Mean of [`TurnEnd::delta_nll_frozen`], the checkpoint's score of the
    /// same delta, `None` when no layer was frozen.
    pub fn delta_mean_nll_frozen(&self) -> Option<f64> {
        finite_mean(&self.delta_nll_frozen)
    }

    /// One line for a report, carrying its own framing rule.
    pub fn summary(&self) -> String {
        let score = match self.delta_mean_nll() {
            Some(mean) => format!(", delta nll {mean:.3} nats/token over {}", self.delta_nll.len()),
            None => String::new(),
        };
        format!(
            "turn {}: {} token(s) after a {}-token delta (+{} carried), first token {:.3}s, \
             turn {:.3}s, position {} ({}){}",
            self.turn,
            self.tokens,
            self.delta_tokens,
            self.carried,
            self.first_token_secs,
            self.turn_secs,
            self.position,
            self.stopped,
            score,
        )
    }
}

/// One native call extracted from a completed model response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeExecCall {
    pub command: String,
}

/// Content emitted by one generated token after structural parsing.
///
/// Archive currently preserves the channel and exact per-part bytes, but
/// Drive's split `said`/`Decision::reasoning` turn shape cannot retain an
/// arbitrary alternation of several thinking and text blocks within one response.
/// The clean follow-up is an ordered, provider-neutral emitted-part sequence on
/// `Turn`, populated here from parser transitions and staged verbatim. It must
/// not be replaced by a second accumulated-response string.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NativeTokenDelta {
    /// User-visible model text. Only `content_text` contributes here.
    pub text: String,
    /// Provider reasoning. Only `content_thinking` contributes here.
    pub thinking: String,
    /// The exact `content_model_end_sampling` id completed the response.
    pub completed: bool,
}

#[derive(Debug)]
enum NativeOutputState {
    /// The generation prompt has already inserted `message_model`; ordinary
    /// fragments here are the optional tool name preceding its content kind.
    Header(String),
    Text,
    Thinking,
    ToolJson(String),
    /// A block ended; another block must begin with `message_model`, or the
    /// response must end with `content_model_end_sampling`.
    Between,
}

/// Incremental parser for Inkling's generated typed blocks.
///
/// Structure is recognized exclusively by exact ids announced in READY.
/// Decoded strings are payload fragments only; a literal marker spelling made
/// from ordinary tokens remains ordinary content.
#[derive(Debug)]
pub struct NativeOutputParser {
    ids: InklingSpecialIds,
    state: NativeOutputState,
    call: Option<NativeExecCall>,
    completed: bool,
}

impl NativeOutputParser {
    pub fn new(ids: InklingSpecialIds) -> Self {
        Self {
            ids,
            // Initialize/tool-result/generation-prompt all end by inserting
            // message_model, so the first generated token is already in its
            // header/content-kind position.
            state: NativeOutputState::Header(String::new()),
            call: None,
            completed: false,
        }
    }

    /// Consume one exact generated id together with the decoder fragment that
    /// same one-token microturn produced.
    pub fn push(&mut self, id: u32, fragment: &str) -> Result<NativeTokenDelta> {
        anyhow::ensure!(!self.completed, "tokens arrived after model_end_sampling");
        let mut delta = NativeTokenDelta::default();

        if self.ids.is_special(id) {
            let decoded = self
                .ids
                .decoded_special(id)
                .with_context(|| format!("special token id {id} has no runtime decode"))?;
            let payload = fragment.strip_suffix(decoded).with_context(|| {
                format!(
                    "decoder fragment for special token id {id} did not end with its runtime decode {decoded:?}"
                )
            })?;
            // DecodeStream can hold an incomplete ordinary token and return it
            // only when this special token makes the prefix decodable. Those
            // bytes belong to the state *before* the structural transition.
            self.push_payload(payload, &mut delta)?;
        } else {
            self.push_payload(fragment, &mut delta)?;
            return Ok(delta);
        }

        if id == self.ids.content_model_end_sampling {
            match &self.state {
                NativeOutputState::Header(header) if header.is_empty() => {}
                NativeOutputState::Header(header) => {
                    anyhow::bail!("model_end_sampling left unclassified message header {header:?}")
                }
                NativeOutputState::ToolJson(_) => {
                    anyhow::bail!("model_end_sampling truncated an exec JSON block")
                }
                NativeOutputState::Text => {
                    anyhow::bail!("model_end_sampling truncated a text block before end_message")
                }
                NativeOutputState::Thinking => anyhow::bail!(
                    "model_end_sampling truncated a thinking block before end_message"
                ),
                NativeOutputState::Between => {}
            }
            self.state = NativeOutputState::Between;
            self.completed = true;
            delta.completed = true;
            return Ok(delta);
        }

        if id == self.ids.message_model {
            anyhow::ensure!(
                matches!(self.state, NativeOutputState::Between),
                "message_model appeared before the previous block ended"
            );
            anyhow::ensure!(
                self.call.is_none(),
                "model emitted another block after its exec call"
            );
            self.state = NativeOutputState::Header(String::new());
            return Ok(delta);
        }

        if id == self.ids.content_text {
            let NativeOutputState::Header(header) = &self.state else {
                anyhow::bail!("content_text appeared outside a model-message header")
            };
            anyhow::ensure!(
                header.is_empty(),
                "text block carried unexpected model-message header {header:?}"
            );
            self.state = NativeOutputState::Text;
            return Ok(delta);
        }

        if id == self.ids.content_thinking {
            let NativeOutputState::Header(header) = &self.state else {
                anyhow::bail!("content_thinking appeared outside a model-message header")
            };
            anyhow::ensure!(
                header.is_empty(),
                "thinking block carried unexpected model-message header {header:?}"
            );
            self.state = NativeOutputState::Thinking;
            return Ok(delta);
        }

        if id == self.ids.content_invoke_tool_json {
            let NativeOutputState::Header(header) = &self.state else {
                anyhow::bail!("content_invoke_tool_json appeared outside a model-message header")
            };
            anyhow::ensure!(
                header == "exec",
                "native tool header named {header:?}, expected exactly \"exec\""
            );
            self.state = NativeOutputState::ToolJson(String::new());
            return Ok(delta);
        }

        if id == self.ids.end_message {
            let state = std::mem::replace(&mut self.state, NativeOutputState::Between);
            match state {
                NativeOutputState::Text | NativeOutputState::Thinking => {}
                NativeOutputState::ToolJson(json) => {
                    anyhow::ensure!(self.call.is_none(), "model emitted multiple exec calls");
                    self.call = Some(parse_native_exec(&json)?);
                }
                NativeOutputState::Header(header) => {
                    anyhow::bail!("end_message closed an unclassified header {header:?}")
                }
                NativeOutputState::Between => {
                    anyhow::bail!("end_message appeared between model messages")
                }
            }
            return Ok(delta);
        }

        if self.ids.is_special(id) {
            anyhow::bail!("unsupported generated Inkling special token id {id}");
        }
        unreachable!("all non-special ids returned after applying payload")
    }

    fn push_payload(&mut self, fragment: &str, delta: &mut NativeTokenDelta) -> Result<()> {
        if fragment.is_empty() {
            return Ok(());
        }
        match &mut self.state {
            NativeOutputState::Header(header) => header.push_str(fragment),
            NativeOutputState::Text => delta.text.push_str(fragment),
            NativeOutputState::Thinking => delta.thinking.push_str(fragment),
            NativeOutputState::ToolJson(json) => json.push_str(fragment),
            NativeOutputState::Between => {
                anyhow::bail!("decoder flushed ordinary payload between model messages")
            }
        }
        Ok(())
    }

    /// Take the optional call after an exact end-of-sampling token and reset
    /// for the next generation prompt. Calling this on an incomplete response
    /// is a caller error.
    pub fn take_completed_call(&mut self) -> Result<Option<NativeExecCall>> {
        anyhow::ensure!(self.completed, "model response is not complete");
        self.completed = false;
        self.state = NativeOutputState::Header(String::new());
        Ok(self.call.take())
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeExecEnvelope {
    name: String,
    args: NativeExecArgs,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeExecArgs {
    command: String,
}

fn parse_native_exec(json: &str) -> Result<NativeExecCall> {
    let envelope: NativeExecEnvelope =
        serde_json::from_str(json).context("parse strict native exec JSON")?;
    anyhow::ensure!(
        envelope.name == "exec",
        "native tool JSON named {:?}, expected exactly \"exec\"",
        envelope.name
    );
    Ok(NativeExecCall {
        command: envelope.args.command,
    })
}

/// Append a canonical, marker-free transcript projection for one typed call.
/// The returned range is exact within `said` and can be shifted by the current
/// session monologue extent for `Decision::fire_projected`.
pub fn project_native_exec(said: &mut String, command: &str) -> std::ops::Range<usize> {
    // The separator is projection scaffold too. Naming it inside the range lets
    // Archive remove the entire synthetic suffix without retaining a newline
    // that the model never actually spoke.
    let start = said.len();
    if !said.is_empty() && !said.ends_with('\n') {
        said.push('\n');
    }
    said.push_str("$ ");
    said.push_str(command);
    if !said.ends_with('\n') {
        said.push('\n');
    }
    start..said.len()
}

/// Associate one arbitrated id with the fragments emitted by its one-token
/// consult. This check is the TP-safe microturn invariant: neither rank-local
/// stop policy nor callback record boundaries are allowed to guess the id.
pub fn one_token_association(end: &TurnEnd, fragments: String) -> Result<(u32, String)> {
    anyhow::ensure!(
        end.tokens == 1 && end.token_ids.len() == 1,
        "one-token consult returned {} token(s) and {} exact id(s)",
        end.tokens,
        end.token_ids.len()
    );
    Ok((end.token_ids[0], fragments))
}

// ── the model seam ──────────────────────────────────────────────────────────

/// What `InklingMind` needs of the model, and nothing more.
///
/// This is the seam that used to be a pipe. The nine framed-stream content
/// types of the deleted serving protocol encoded exactly these six operations
/// plus a preamble; expressing them as a trait costs one virtual call per model
/// pass — against a pass that is tens of milliseconds of GPU — and buys two
/// things the wire also bought and nothing else did:
///
///   1. **The mind stays GPU-free and therefore stays TESTED.** The real
///      implementation (`engine::Engine`) needs CUDA, a 171 GB pile
///      and two boxes. The turn shaping, the carry accounting, the native call
///      projection, the coverage arithmetic and the replacement boundary do
///      not, and they are where the expensive mistakes have historically been.
///      A scripted in-memory `Model` pins them anywhere.
///   2. **One place to say what a model owes a mind.** The old answer was
///      spread across `ServeClient`, `ServePair` and `inkling_serve`'s input
///      loop, and the pair's five invariants existed to keep the three in
///      agreement.
///
/// # The lockstep obligation
///
/// An implementation backed by tensor parallelism must make EVERY rank perform
/// the same `Session` passes in the same order — that is not this trait's
/// business, it is `engine::Engine`'s, and the way it is discharged
/// changed with the collapse. The deleted proxy mirrored INPUT BYTES to both
/// ranks and relied on both deriving the same passes from them (its third
/// invariant, and a fragile derivation). The engine now mirrors THE PASSES
/// THEMSELVES: rank 0 names each `prefill`/`extend`/`step`/`reset` on the
/// rendezvous socket immediately before making it, and rank 1 replays exactly
/// that. Lockstep is stated rather than inferred.
///
/// # `Send`, and why it is a supertrait rather than a bound at the use site
///
/// `drive::mind::Mind` requires `Send`, so `InklingMind` must be, so whatever
/// it holds must be. Saying it here means an implementation that is not `Send`
/// fails where it is written rather than a hundred lines away where it is
/// boxed.
pub trait Model: Send {
    /// What loaded, and whether its tokens are the model's.
    fn ready(&self) -> &Ready;

    /// Insert typed TML context into the pending delta.
    fn context(&mut self, context: &InklingContext) -> Result<()>;

    /// Price one exact typed context plus a whole bounded response, without
    /// queueing it or attending to it.
    ///
    /// Read-only: a rejection has changed neither the session nor the pending
    /// delta, carry, decoder or turn count.
    fn preflight_context(&mut self, request: &ContextPreflight) -> Result<ContextPreflighted>;

    /// Replace one completed sequence with a complete initialization, keeping
    /// the warm weights.
    ///
    /// `initialization` must be [`InklingContext::Initialize`]. Everything that
    /// can reject the replacement runs before anything is reset, so an error
    /// leaves the old sequence byte-for-byte alive.
    fn reinitialize(&mut self, initialization: &InklingContext) -> Result<Reinitialized>;

    /// Produce a turn, calling `on_token` with each token's text AS IT IS
    /// DECODED.
    ///
    /// The callback is the streaming seam: it runs while the model is still
    /// generating, which is what lets the voice start speaking on the first
    /// word. A callback that blocks blocks the model, and that is the honest
    /// coupling.
    ///
    /// `&mut dyn FnMut` rather than `impl FnMut` because this trait is used
    /// behind a `Box`.
    fn consult(
        &mut self,
        request: &Consult,
        on_token: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<TurnEnd>;

    /// Prove every rank still holds the same sequence, and fail loudly if not.
    ///
    /// This is what remains of `ServePair`'s byte-for-byte cross-rank check —
    /// see this module's header for exactly what weakened. A single-rank model
    /// has nothing to compare against and returns `Ok(())`.
    ///
    /// Called at every logical-turn boundary and after every reinitialization:
    /// points at which no collective is in flight, so a host-side round trip
    /// here cannot deadlock against one.
    fn agree_sequence(&mut self) -> Result<()> {
        Ok(())
    }

    /// Abandon the model TERMINALLY after a failure, releasing any peer.
    ///
    /// Write what the learner moved back into the model graph as a VERSION:
    /// a child of the root this model loaded from, carrying `recipe`, signed
    /// with the resident's key -- so that a day learned survives a restart.
    /// Called before a context is replaced and before shutdown. `Ok(None)`
    /// when this model does not learn, has no key to sign with, or nothing
    /// moved. A collective on a tensor-parallel pair.
    fn persist_learned(
        &mut self,
        recipe: &VersionRecipe,
    ) -> Result<Option<Persisted>> {
        let _ = recipe;
        Ok(None)
    }

    /// A tensor rank whose partner has stopped making collectives blocks in
    /// NCCL with no timeout, so this must reach the peer, not merely drop local
    /// state.
    fn kill(&mut self) -> Result<()>;

    /// End cleanly: the conversation is over and every rank should stop.
    ///
    /// Idempotent, because it is called both explicitly and from a `Drop` that
    /// cannot know whether it already ran.
    fn shutdown(&mut self) -> Result<()>;
}

// ── a learned version, as the turn vocabulary knows it ─────────────────────
//
// Backend-free on purpose: the mind asks for a version and records that one
// was written; assembling it is `version` on the CUDA lane.

/// How a version was learned. Facts on the version root, for analysis later.
#[derive(Clone, Debug, Default)]
pub struct VersionRecipe {
    pub lr: f64,
    pub anchor: Option<f64>,
    pub seed: u64,
    pub steps: u64,
    pub span: String,
    pub explanation: String,
    pub code_revision: String,
}

/// What a persisted version is, for the record that says it happened.
#[derive(Clone, Debug)]
pub struct Persisted {
    /// The version root, committed -- or equal to `parent` when nothing had
    /// moved and nothing was written.
    pub root: triblespace::prelude::Id,
    pub parent: triblespace::prelude::Id,
    pub name: String,
    /// Experts whose bytes moved.
    pub replaced: usize,
    /// Whether the parent was minted as the genesis root in the same commit.
    pub genesis: bool,
}

// ── the drive seam ──────────────────────────────────────────────────────────

/// What a turn proved about STREAMING, recorded per turn so the claim is a
/// measurement rather than an assertion.
///
/// The question is not "did the faculty get the words" — a batch would pass
/// that. It is "did the faculty produce OUTPUT while the mind was still
/// generating", and the only way to answer it is to look at the faculty's
/// return stream from inside `observe`, with the turn demonstrably unfinished.
#[derive(Debug, Clone)]
pub struct StreamProof {
    /// Turn ordinal.
    pub turn: usize,
    /// Tokens this turn produced in total.
    pub tokens: usize,
    /// How many tokens had been streamed when the voice faculty's FIRST record
    /// came back. `Some(k)` with `k < tokens` is the proof; `None` means the
    /// faculty produced nothing before the turn ended, which is a batch.
    pub tokens_at_first_return: Option<usize>,
    /// Records the faculty had returned by the end of the turn.
    pub records_at_end: u64,
    /// Seconds each of this turn's tokens cost INSIDE THE MODEL, in generation
    /// order — [`TurnEnd::turn_secs`] of each one-token consult.
    ///
    /// **Framing rule: seconds per GENERATED TOKEN, measured around the
    /// `Session` calls and nothing else**, over the layer range
    /// [`Ready::layers`] names, on the world size [`Ready::tp_world`] names.
    /// This is the same quantity the module header's FIRST rows report
    /// (55.8 / 58.7 ms within one 32-token consult) — the model's own time,
    /// which the collapse should NOT have changed.
    ///
    /// Retained only under `InklingMind::new_gate`: an unbounded resident run
    /// must not accumulate one `f64` per token forever.
    pub token_secs: Vec<f64>,
    /// Seconds each of this turn's tokens cost AS DRIVE PAYS FOR IT: this
    /// mind's own wall clock around the whole [`Model::consult`], in the same
    /// order.
    ///
    /// **Framing rule: seconds per GENERATED TOKEN including everything
    /// between the mind and the model**, same layers, same world. This is the
    /// quantity the module header's third row reports — p50 82 ms, n=768,
    /// through the deleted framed-stream proxy — and it is the one the collapse
    /// exists to move.
    ///
    /// # The pair of vectors IS the measurement
    ///
    /// `consult_secs[i] - token_secs[i]` is everything that was not the model:
    /// under the deleted arrangement, a framed-stream round trip out through a
    /// fan-out proxy, two rank pipes, an `ssh` channel to the second box, and
    /// two JSON `TurnEnd` envelopes back — about 26 ms, a third of resident
    /// decode. In this process it is a virtual call.
    ///
    /// Keeping both means a finite run is SELF-CONTAINED evidence: the
    /// before-number and the after-number for the overhead both come out of one
    /// invocation, rather than the after-number being compared against a
    /// remembered figure measured on a different day with a different framing.
    pub consult_secs: Vec<f64>,
}

impl StreamProof {
    /// Whether this turn STREAMED: the consumer produced output strictly before
    /// the mind stopped generating.
    pub fn streamed(&self) -> bool {
        matches!(self.tokens_at_first_return, Some(k) if k < self.tokens)
    }
}
