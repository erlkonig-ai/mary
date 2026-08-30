//! **The resident mind** — the vocabulary of an Inkling turn, and the
//! [`drive::mind::Mind`] that produces one from a `session::Session` held IN
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
//! [`InklingMind::turn`]) — so every single token paid a framed-stream round
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
//! [`InklingMind`] talks to a [`Model`], not to a `Session`. The real
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
/// announces the result in [`Ready`], so [`InklingMind`] parses generated
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
    /// native call in [`InklingMind`].
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

    fn validate(&self) -> Result<()> {
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
    /// Result of the native call already present in the retained KV sequence.
    ToolResult { result: ExecResultContext },
    /// A complete response which predates this live `InklingMind` (for example
    /// a Drive memory-cover response). Its model parts and optional result are
    /// inserted together.
    HistoricalResponse { response: InklingHistoryResponse },
    /// Start another autonomous assistant response after a completed text-only
    /// response. A tool result already carries this prompt itself.
    GenerationPrompt,
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
    fn validate_for(&self, request: &ContextPreflight, context_budget: usize) -> Result<()> {
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
            InklingContext::ToolResult { result } => {
                self.push_tool_result(&mut ids, result)?;
                ids.push(self.special_ids.message_model as usize);
            }
            InklingContext::HistoricalResponse { response } => {
                self.push_historical_response(&mut ids, response)?;
                ids.push(self.special_ids.message_model as usize);
            }
            InklingContext::GenerationPrompt => {
                ids.push(self.special_ids.message_model as usize);
            }
        }
        Ok(ids)
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

impl TurnEnd {
    /// One line for a report, carrying its own framing rule.
    pub fn summary(&self) -> String {
        format!(
            "turn {}: {} token(s) after a {}-token delta (+{} carried), first token {:.3}s, \
             turn {:.3}s, position {} ({})",
            self.turn,
            self.tokens,
            self.delta_tokens,
            self.carried,
            self.first_token_secs,
            self.turn_secs,
            self.position,
            self.stopped,
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
fn project_native_exec(said: &mut String, command: &str) -> std::ops::Range<usize> {
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
fn one_token_association(end: &TurnEnd, fragments: String) -> Result<(u32, String)> {
    anyhow::ensure!(
        end.tokens == 1 && end.token_ids.len() == 1,
        "one-token consult returned {} token(s) and {} exact id(s)",
        end.tokens,
        end.token_ids.len()
    );
    Ok((end.token_ids[0], fragments))
}

// ── the model seam ──────────────────────────────────────────────────────────

/// What [`InklingMind`] needs of the model, and nothing more.
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
/// `drive::mind::Mind` requires `Send`, so [`InklingMind`] must be, so whatever
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
    /// Seconds each of this turn's tokens cost, in generation order, as the
    /// model measured them around its own `Session` calls.
    ///
    /// **Framing rule: seconds per GENERATED TOKEN, at the granularity Drive
    /// actually generates them** — one one-token consult each — over the layer
    /// range [`Ready::layers`] names, on the world size [`Ready::tp_world`]
    /// names. That is deliberately the SAME quantity the module header's third
    /// row reports (p50 82 ms through the deleted framed-stream proxy), so the
    /// two are comparable without reconstructing anything: run a finite
    /// `inkling_drive --turns n` and the distribution over every element of
    /// every turn's vector is the after-number for the before-number in that
    /// table.
    ///
    /// Retained only under [`InklingMind::new_gate`]: an unbounded resident run
    /// must not accumulate one `f64` per token forever.
    pub token_secs: Vec<f64>,
}

impl StreamProof {
    /// Whether this turn STREAMED: the consumer produced output strictly before
    /// the mind stopped generating.
    pub fn streamed(&self) -> bool {
        matches!(self.tokens_at_first_return, Some(k) if k < self.tokens)
    }
}

/// The [`drive::mind::Mind`] backed by a real Inkling [`Model`].
///
/// This is the seam drive's `Mind` docs describe, filled in: it consumes a
/// causally ordered DELTA (not a re-rendered transcript), produces an utterance,
/// produces a decision always, and is stateful across turns — the state being a
/// KV cache that outlives every call.
///
/// That cache used to live in another process, reached over a framed stream.
/// It lives in THIS process now (see the module header for what that cost and
/// what it bought); the only thing that changed here is the type of one field.
///
/// # What it ignores, and why that is correct
///
/// `Payload::Monologue` events are the mind's OWN words from earlier turns. A
/// stateful backend already has them, and it takes TWO mechanisms rather than
/// one: all but the last token of a turn were fed back by its own `step()`, and
/// the last one — which the generation loop deliberately does not spend a decode
/// step to feed — is appended at the head of the NEXT turn's delta (see
/// `engine::Engine::generate` and [`TurnEnd::carried`]). Between them, every
/// token this mind has said is in the KV cache by the time it is consulted
/// again, so replaying the text would attend to the same words twice.
///
/// **That sentence used to say only the first half** — "they are literally the
/// tokens its own `step()` fed back" — and it was false by exactly one token per
/// turn, every turn with new context, for as long as it stood: the turn's final
/// word went to the consumer and never to the cache. It is spelled out here
/// because the comment is the whole reason the monologue is not re-fed, and a
/// justification whose premise nobody checks is how the defect survived. It left
/// no other trace: the cache stayed consistent, `position()` stayed exactly
/// right, and the only thing that disagreed was the model's answer.
///
/// They are ingested into a [`drive::mind::MonologueBuffer`] anyway — but only
/// for their COORDINATES, so the decision's span is expressed in the same
/// session-absolute byte space the loop verifies against.
///
/// `Payload::Result` events are untrusted output. They cross the typed context
/// seam as content-only tokens, paired with either the exact outstanding call
/// already retained in KV or a structural historical call during memory cover.
/// Marker-looking result bytes can therefore never become protocol structure.
///
/// # The decision
///
/// A completed, strictly parsed native call is projected canonically into this
/// same turn's `said` bytes and returned as [`drive::mind::Disposition::Fire`]
/// over that exact fresh range. Text-only responses remain audited `NoAction`
/// decisions.
pub struct InklingMind {
    model: Box<dyn Model>,
    /// Consecutive identity of the KV sequence the model currently holds. It
    /// changes only after a complete reinitialization acknowledgement.
    context_epoch: u64,
    /// Position reported by the final microturn of the latest consultation.
    /// `None` means the initialization has not yet been prefetched.
    position: Option<usize>,
    /// The voice, once the shell has been opened and has handed it over.
    /// `Shell::claim_voice` can only be called on a built shell, and the mind
    /// has to be built before the shell, so the handle arrives through a slot
    /// rather than through the constructor.
    voice: std::sync::Arc<std::sync::Mutex<Option<drive::stream::Voice>>>,
    /// The mind's own words, for their COORDINATES only.
    buffer: drive::mind::MonologueBuffer,
    /// Session-absolute byte offset up to which this mind has been shown its own
    /// monologue.
    scanned_abs: u64,
    /// Hard bound on one complete logical model response. This is the same
    /// output reservation Drive removes from the memory-cover budget, not a
    /// batching width exposed as extra Drive turns.
    max_response_tokens: usize,
    /// System prompt held until the first released memory cover can be inserted
    /// in the same typed initialization record.
    system: Option<String>,
    initialized: bool,
    /// Typed generated-output parser for the current logical response.
    output: NativeOutputParser,
    /// Checked after every one-token collective. The ordinary constructor never
    /// requests cancellation; the resident runner supplies its SIGINT flag.
    stop_requested: std::sync::Arc<dyn Fn() -> bool + Send + Sync>,
    /// An interrupted response cannot be resumed from Archive's completed
    /// response representation. Cancellation itself is a graceful `Continue`
    /// turn so Shell can observe its already-set stop flag; this guard makes an
    /// accidental direct reuse fail terminally instead of continuing a partial
    /// parser state.
    response_interrupted: bool,
    /// The native call already present in the model's retained sequence and
    /// awaiting Drive's result. Exact command matching prevents both duplicate
    /// historical-call insertion and a result being attached to the wrong act.
    outstanding_exec: Option<String>,
    /// A completed text-only response needs a fresh message_model prompt before
    /// another autonomous response. Initialization and tool results supply one
    /// themselves.
    needs_generation_prompt: bool,
    turns: usize,
    label: String,
    /// Native READY/TurnEnd evidence waiting for Drive to attach to the
    /// causative session or turn. This is a one-slot exhaust accumulator, not a
    /// report log: Drive drains it immediately through `Mind::take_exhaust`.
    pending_exhaust: triblespace::prelude::Fragment,
    /// Per-turn gate evidence. A resident run leaves both sinks absent: an
    /// unbounded conversation must not quietly retain one report per turn just
    /// because the finite correctness gate wants to print them afterward.
    log: Option<std::sync::Arc<std::sync::Mutex<Vec<TurnEnd>>>>,
    proofs: Option<std::sync::Arc<std::sync::Mutex<Vec<StreamProof>>>>,
}

/// A consultation that failed after producing zero or more final text bytes.
///
/// Keeping the partial utterance on the stack makes one failed call one value:
/// it cannot leak into a later turn, and failures before generation naturally
/// carry an empty string through `?`.
struct FailedTurn {
    error: anyhow::Error,
    said: String,
}

#[derive(Debug, PartialEq, Eq)]
enum PendingObservationContext {
    Context(InklingContext),
    WaitingForResult,
    AlreadyStaged,
}

impl From<anyhow::Error> for FailedTurn {
    fn from(error: anyhow::Error) -> Self {
        Self {
            error,
            said: String::new(),
        }
    }
}

fn text_result_content(
    content: &drive::content::Content,
    is_error: bool,
    exit_code: Option<i32>,
) -> String {
    // Inkling's serving wire is text/plain and Session accepts token ids, so it
    // cannot consume drive's resident image/audio parts yet. This is drive's
    // explicit compatibility seam for a text-only mind, not a second stored
    // representation: the typed Content stays intact in the World.
    let projected = content.text_projection();
    let mut delta = projected.into_owned();

    // A normal shell result already carries its `[exit 0]` line in the text
    // projection. Preserve the structural fields when they add information the
    // text need not carry: MCP `isError`, or a non-zero exit status.
    match (is_error, exit_code) {
        (true, Some(code)) => {
            delta.push_str(&format!(
                "\n[result status: isError=true, exit_code={code}]"
            ));
        }
        (true, None) => delta.push_str("\n[result status: isError=true]"),
        (false, Some(code)) => {
            if code != 0 {
                delta.push_str(&format!("\n[result status: exit_code={code}]"));
            }
        }
        (false, None) => {}
    }
    if !delta.ends_with('\n') {
        delta.push('\n');
    }
    delta
}

impl InklingMind {
    /// Wrap a loaded model as a potentially unbounded resident mind.
    ///
    /// Per-turn evidence belongs in Drive's ledger and optional telemetry. This
    /// constructor therefore retains no private report vector.
    pub fn new(
        model: Box<dyn Model>,
        max_response_tokens: usize,
        system: Option<String>,
    ) -> Result<Self> {
        Self::with_gate_evidence(model, max_response_tokens, system, false)
    }

    /// Wrap a loaded model for a FINITE run, retaining per-turn evidence.
    ///
    /// This deliberately retains a [`TurnEnd`] and a [`StreamProof`] — including
    /// its per-token seconds — for every turn, so a bounded run can report the
    /// decode distribution afterwards. That report is the measurement the
    /// one-binary collapse is judged by; see [`StreamProof::token_secs`].
    /// Resident (`--live`) callers must use [`Self::new`], which retains
    /// nothing.
    pub fn new_gate(
        model: Box<dyn Model>,
        max_response_tokens: usize,
        system: Option<String>,
    ) -> Result<Self> {
        Self::with_gate_evidence(model, max_response_tokens, system, true)
    }

    fn with_gate_evidence(
        mut model: Box<dyn Model>,
        max_response_tokens: usize,
        system: Option<String>,
        retain_gate_evidence: bool,
    ) -> Result<Self> {
        // This is also the validation boundary for the identities a `Model` may
        // announce as arbitrary strings: a malformed READY fails explicitly
        // before a mind exists, rather than panicking or silently omitting
        // native evidence later.
        let pending_exhaust = match super::telemetry::ready_fragment(model.ready()) {
            Ok(fragment) => fragment,
            Err(error) => {
                if let Err(teardown) = model.shutdown() {
                    return Err(error.context(format!(
                        "model teardown after invalid READY also failed: {teardown:#}"
                    )));
                }
                return Err(error);
            }
        };
        let label = match model.ready().partial {
            true => "inkling(partial)".to_string(),
            false => "inkling".to_string(),
        };
        let output = NativeOutputParser::new(model.ready().special_ids.clone());
        let log =
            retain_gate_evidence.then(|| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
        let proofs =
            retain_gate_evidence.then(|| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
        Ok(Self {
            model,
            context_epoch: 0,
            position: None,
            voice: std::sync::Arc::new(std::sync::Mutex::new(None)),
            buffer: drive::mind::MonologueBuffer::with_cap(64 * 1024),
            scanned_abs: 0,
            max_response_tokens,
            system,
            initialized: false,
            output,
            stop_requested: std::sync::Arc::new(|| false),
            response_interrupted: false,
            outstanding_exec: None,
            needs_generation_prompt: false,
            turns: 0,
            label,
            pending_exhaust,
            log,
            proofs,
        })
    }

    /// Check `stop_requested` between one-token collectives.
    ///
    /// Cancellation does not batch or delay output: the model still streams each
    /// token immediately, then returns one audited interrupted response at the
    /// next token boundary. A closure keeps the library independent of any
    /// particular signal implementation while allowing a resident binary to
    /// read its signal-safe atomic flag directly.
    pub fn with_cancellation(
        mut self,
        stop_requested: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Self {
        self.stop_requested = std::sync::Arc::new(stop_requested);
        self
    }

    /// The slot the shell's voice is dropped into after `Shell::open`.
    ///
    /// Claiming the voice transfers the WRITER ROLE: from then on the shell
    /// stops writing the mind's turns and this backend writes one record per
    /// token from inside `observe`. Both writing would make the faculty hear
    /// every turn twice.
    pub fn voice_slot(&self) -> std::sync::Arc<std::sync::Mutex<Option<drive::stream::Voice>>> {
        std::sync::Arc::clone(&self.voice)
    }

    /// Per-turn numbers, as the model measured them around its `Session` calls.
    pub fn log(&self) -> Option<std::sync::Arc<std::sync::Mutex<Vec<TurnEnd>>>> {
        self.log.as_ref().map(std::sync::Arc::clone)
    }

    /// What each turn proved about streaming.
    pub fn proofs(&self) -> Option<std::sync::Arc<std::sync::Mutex<Vec<StreamProof>>>> {
        self.proofs.as_ref().map(std::sync::Arc::clone)
    }

    /// What loaded on the far end.
    pub fn ready(&self) -> &Ready {
        self.model.ready()
    }

    /// Close the audited coverage this consultation was shown.
    ///
    /// Success and terminal failure use the same calculation. In particular,
    /// both clamp against `base_offset`: a bounded monologue may evict bytes
    /// behind `scanned_abs`, but neither path may claim coordinates it no
    /// longer holds.
    fn finish_coverage(&mut self) -> (u64, u64) {
        let span_end = self.buffer.end_offset();
        let span_start = self
            .scanned_abs
            .max(self.buffer.base_offset())
            .min(span_end);
        self.scanned_abs = span_end;
        (span_start, span_end)
    }

    /// Reconstruct the one typed delta this observation would send, without
    /// mutating either the adapter or the model.
    fn pending_context(&self, events: &[drive::world::Event]) -> Result<PendingObservationContext> {
        let mut results = Vec::new();
        for event in events {
            if let drive::world::Payload::Result {
                command,
                content,
                is_error,
                exit_code,
            } = &event.payload
            {
                results.push(ExecResultContext {
                    command: command.clone(),
                    content: text_result_content(content, *is_error, *exit_code),
                });
            }
        }

        if !self.initialized {
            return Ok(PendingObservationContext::Context(
                InklingContext::Initialize {
                    system: self.system.clone().unwrap_or_default(),
                    history: results
                        .into_iter()
                        .map(InklingHistoryResponse::exec)
                        .collect(),
                },
            ));
        }
        anyhow::ensure!(
            results.len() <= 1,
            "received {} exec results in one post-initialization view",
            results.len()
        );
        if let Some(result) = results.pop() {
            return match self.outstanding_exec.as_deref() {
                Some(command) => {
                    anyhow::ensure!(
                        command == result.command,
                        "exec result command {:?} did not match outstanding command {:?}",
                        result.command,
                        command
                    );
                    Ok(PendingObservationContext::Context(
                        InklingContext::ToolResult { result },
                    ))
                }
                None => {
                    anyhow::ensure!(
                        self.needs_generation_prompt,
                        "historical exec result arrived inside an unfinished assistant response"
                    );
                    Ok(PendingObservationContext::Context(
                        InklingContext::HistoricalResponse {
                            response: InklingHistoryResponse::exec(result),
                        },
                    ))
                }
            };
        }
        if self.outstanding_exec.is_some() {
            Ok(PendingObservationContext::WaitingForResult)
        } else if self.needs_generation_prompt {
            Ok(PendingObservationContext::Context(
                InklingContext::GenerationPrompt,
            ))
        } else {
            // A successful replacement has already staged its complete
            // initialization. Its first observation therefore needs no second
            // context record.
            Ok(PendingObservationContext::AlreadyStaged)
        }
    }
}

fn replacement_history(image: &drive::mind::ContextImage) -> Result<Vec<InklingHistoryResponse>> {
    image
        .responses
        .iter()
        .map(|response| {
            response.validate()?;
            let parts = response
                .parts
                .iter()
                .map(|part| match part {
                    drive::mind::ContextPart::Thinking(content) => InklingHistoryPart::Thinking {
                        content: content.clone(),
                    },
                    drive::mind::ContextPart::Text(content) => InklingHistoryPart::Text {
                        content: content.clone(),
                    },
                    drive::mind::ContextPart::ToolCall(command) => InklingHistoryPart::Exec {
                        command: command.clone(),
                    },
                })
                .collect::<Vec<_>>();
            let tool_result = response.tool_result.as_ref().map(|result| {
                text_result_content(&result.content, result.is_error, result.exit_code)
            });
            let response = InklingHistoryResponse { parts, tool_result };
            response.validate()?;
            Ok(response)
        })
        .collect()
}

fn validate_replacement_boundary(
    needs_generation_prompt: bool,
    outstanding_exec: Option<&str>,
    history: &[InklingHistoryResponse],
) -> Result<()> {
    anyhow::ensure!(
        needs_generation_prompt != outstanding_exec.is_some(),
        "cannot replace an Inkling context inside an unfinished response"
    );
    if let Some(command) = outstanding_exec {
        let final_command = history.last().and_then(|response| {
            response
                .tool_result
                .as_ref()
                .and_then(|_| response.parts.last())
                .and_then(|part| match part {
                    InklingHistoryPart::Exec { command } => Some(command.as_str()),
                    _ => None,
                })
        });
        anyhow::ensure!(
            final_command == Some(command),
            "replacement history does not close outstanding exec call {command:?}"
        );
    }
    Ok(())
}

/// The shell OWNS the mind (`Box<dyn Mind>`), so no caller can ever tell the
/// model the conversation is over by name. It is done here.
///
/// This still matters with the model in this process, for a different reason
/// than it did with the model in another one. There is no longer an
/// `END{aborted}` to avoid writing — but there is still a PEER, on the other
/// box, sitting in `engine::Follower::follow` waiting to be told the run
/// is over. Dropping local state would leave it alive, holding a 121 GiB arena
/// and half a communicator, and the next run would rendezvous against a
/// stranger. [`Model::shutdown`] is what reaches it.
impl Drop for InklingMind {
    fn drop(&mut self) {
        match self.model.shutdown() {
            Ok(()) => eprintln!("inkling: the model shut down cleanly"),
            Err(error) => eprintln!("inkling: bounded model shutdown failed: {error:#}"),
        }
    }
}

impl drive::mind::Mind for InklingMind {
    fn admit_observation(
        &mut self,
        events: &[drive::world::Event],
    ) -> Result<drive::mind::ObservationAdmission> {
        let PendingObservationContext::Context(context) = self.pending_context(events)? else {
            return Ok(drive::mind::ObservationAdmission::Ready);
        };
        let request = ContextPreflight {
            placement: ContextPlacement::Append,
            context,
            max_response_tokens: self.max_response_tokens.max(1),
        };
        let evidence = self.model.preflight_context(&request)?;
        // The evidence used to cross a pipe and was checked because of that.
        // It is checked still: a `Model` is a trait now, the arithmetic is the
        // definition of admission, and recomputing it here costs nothing.
        evidence.validate_for(&request, self.model.ready().context_budget)?;
        self.pending_exhaust += super::telemetry::context_preflight_fragment(&evidence);
        Ok(if evidence.fits {
            drive::mind::ObservationAdmission::Ready
        } else {
            drive::mind::ObservationAdmission::ReplaceContext
        })
    }

    fn observe(&mut self, view: drive::world::MergedView<'_>) -> drive::mind::Turn {
        match self.turn(view.events, view.watermark) {
            Ok(turn) => turn,
            Err(FailedTurn { error, said }) => {
                // Equal fragments may already have escaped into the streaming
                // voice. They remain this turn's truthful partial utterance;
                // the backend failure is orthogonal terminal state, never an
                // ordinary silent NoAction and never a forward `Gap` pretending
                // it can retract bytes.
                eprintln!("inkling: {error:#}");
                let mut rationale = format!("inkling model failed: {error:#}");
                // Terminal, and it must REACH THE PEER: a tensor rank whose
                // partner stops issuing collectives blocks in NCCL with no
                // timeout, so failing locally without telling the other box
                // leaves it wedged holding the whole machine.
                if let Err(kill) = self.model.kill() {
                    eprintln!("inkling: could not stop the failed model: {kill:#}");
                    rationale.push_str(&format!("; model teardown also failed: {kill:#}"));
                }
                let (span_start, span_end) = self.finish_coverage();
                drive::mind::Turn::terminal_failure(
                    said,
                    span_start,
                    span_end,
                    view.watermark,
                    rationale,
                )
            }
        }
    }

    fn context_epoch(&self) -> u64 {
        self.context_epoch
    }

    fn replace_context(
        &mut self,
        next_epoch: u64,
        image: &drive::mind::ContextImage,
    ) -> std::result::Result<(), drive::mind::ContextReplacementFailure> {
        let initialization = (|| -> Result<InklingContext> {
            anyhow::ensure!(
                next_epoch
                    == self
                        .context_epoch
                        .checked_add(1)
                        .context("Inkling context epoch overflow")?,
                "Inkling context replacement must advance by exactly one"
            );
            let history = replacement_history(image)?;
            if self.initialized {
                validate_replacement_boundary(
                    self.needs_generation_prompt,
                    self.outstanding_exec.as_deref(),
                    &history,
                )?;
            }
            Ok(InklingContext::Initialize {
                system: image.system.clone(),
                history,
            })
        })()
        .map_err(drive::mind::ContextReplacementFailure::unchanged)?;
        let placement = if self.initialized {
            ContextPlacement::Replace
        } else {
            ContextPlacement::Append
        };
        let request = ContextPreflight {
            placement,
            context: initialization.clone(),
            max_response_tokens: self.max_response_tokens.max(1),
        };
        let budget = self.model.ready().context_budget;
        let evidence = self
            .model
            .preflight_context(&request)
            .and_then(|evidence| {
                evidence.validate_for(&request, budget)?;
                Ok(evidence)
            })
            .map_err(|error| {
                let _ = self.model.kill();
                drive::mind::ContextReplacementFailure::terminal(
                    error.context("preflight the resident Inkling replacement"),
                )
            })?;
        self.pending_exhaust += super::telemetry::context_preflight_fragment(&evidence);
        if !evidence.fits {
            return Err(drive::mind::ContextReplacementFailure::unchanged(
                anyhow::anyhow!(
                    "the replacement needs {} token position(s), beyond the {}-token context budget",
                    evidence.required_end.map_or_else(
                        || "an overflowing number of".to_string(),
                        |end| end.to_string()
                    ),
                    evidence.context_budget,
                ),
            ));
        }

        let installed = if self.initialized {
            self.model
                .reinitialize(&initialization)
                .map(|acknowledged| (Some(acknowledged.initialization_tokens), Some(acknowledged)))
        } else {
            self.model.context(&initialization).map(|()| (None, None))
        };
        // A replacement is a sequence boundary on EVERY rank. Agreeing here is
        // the one point at which a rank that has silently drifted can still be
        // caught before the fresh prefix is built on top of it, and no
        // collective is in flight, so the host round trip is safe.
        let installed = installed.and_then(|installed| {
            self.model.agree_sequence()?;
            Ok(installed)
        });
        let (position, acknowledgement) = match installed {
            Ok(installed) => installed,
            Err(error) => {
                let _ = self.model.kill();
                return Err(drive::mind::ContextReplacementFailure::terminal(
                    error.context("replace the resident Inkling context"),
                ));
            }
        };
        if let Some(acknowledged) = acknowledgement {
            self.pending_exhaust +=
                super::telemetry::reinitialized_fragment(next_epoch, &acknowledged);
        }
        self.output = NativeOutputParser::new(self.model.ready().special_ids.clone());
        self.buffer.reset_empty_at(image.monologue_end);
        self.scanned_abs = image.monologue_end;
        self.outstanding_exec = None;
        self.needs_generation_prompt = false;
        self.position = position;
        self.system = None;
        self.initialized = true;
        self.context_epoch = next_epoch;
        Ok(())
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn take_exhaust(&mut self) -> triblespace::prelude::Fragment {
        std::mem::replace(
            &mut self.pending_exhaust,
            triblespace::prelude::Fragment::empty(),
        )
    }
}

fn aggregate_microturns(
    logical_turn: usize,
    microturns: &[TurnEnd],
    stopped: &str,
) -> Result<TurnEnd> {
    let first = microturns
        .first()
        .context("a logical turn made no consult")?;
    let last = microturns.last().expect("first proved nonempty");
    for (index, end) in microturns.iter().enumerate() {
        anyhow::ensure!(
            end.tokens == 1 && end.token_ids.len() == 1,
            "microturn {index} did not contain exactly one arbitrated token id"
        );
        if index > 0 {
            anyhow::ensure!(
                end.delta_tokens == 0 && end.carried == 1,
                "microturn {index} unexpectedly inserted {} delta token(s) and {} carry token(s)",
                end.delta_tokens,
                end.carried
            );
        }
    }
    Ok(TurnEnd {
        turn: logical_turn,
        tokens: microturns.len(),
        token_ids: microturns.iter().map(|end| end.token_ids[0]).collect(),
        delta_tokens: first.delta_tokens,
        carried: first.carried,
        stopped: stopped.to_string(),
        first_token_secs: first.first_token_secs,
        turn_secs: microturns.iter().map(|end| end.turn_secs).sum(),
        position: last.position,
    })
}

impl InklingMind {
    /// One turn, with the failure path lifted out so `observe` can stay total.
    fn turn(
        &mut self,
        events: &[drive::world::Event],
        watermark: drive::world::Coord,
    ) -> std::result::Result<drive::mind::Turn, FailedTurn> {
        if self.response_interrupted {
            return Err(anyhow::anyhow!("cannot resume an interrupted Inkling response").into());
        }

        let pending_context = self.pending_context(events)?;
        for event in events {
            match &event.payload {
                // Coordinates only. The model already attended to these tokens:
                // its own `step` fed back all but each turn's last, and the
                // model carries that last one into the next turn's delta.
                // See this type's doc for why both halves have to be true, and
                // what happened while only one was.
                drive::world::Payload::Monologue(text) => self.buffer.push_free(text),
                drive::world::Payload::Result { .. } => {}
            }
        }

        match pending_context {
            PendingObservationContext::Context(context) => {
                self.model.context(&context)?;
                match context {
                    InklingContext::Initialize { .. } => {
                        self.system = None;
                        self.initialized = true;
                    }
                    InklingContext::ToolResult { .. } => self.outstanding_exec = None,
                    InklingContext::HistoricalResponse { .. }
                    | InklingContext::GenerationPrompt => {}
                }
                self.needs_generation_prompt = false;
            }
            PendingObservationContext::WaitingForResult => {
                // A Fire's call tokens already live in KV. Generating again
                // before its exact result arrives would create two unresolved
                // calls and make result association ambiguous.
                let (span_start, span_end) = self.finish_coverage();
                return Ok(drive::mind::Turn::silent(drive::mind::Decision::no_action(
                    span_start,
                    span_end,
                    watermark,
                    "inkling: waiting for the outstanding native exec result",
                )));
            }
            PendingObservationContext::AlreadyStaged => {}
        }

        // One-token consults are the TP-safe stop boundary. Each collective
        // arbitrates exactly one id; only after it returns do we associate that
        // id with its decoder fragments and interpret its structure here.
        //
        // This granularity is the reason the one-binary collapse was worth
        // doing: it used to cost a whole framed-stream round trip through a
        // fan-out proxy and an `ssh` channel PER TOKEN — about 26 ms of the
        // measured 82 ms p50, i.e. a third of resident decode. The boundary is
        // kept, because the stop decision genuinely has to be made between
        // collectives; only the transport under it is gone.
        let voice = self.voice.lock().expect("voice slot").clone();
        let mut said = String::new();
        let mut thinking_this_turn = String::new();
        let mut microturns = Vec::new();
        let mut tokens = 0usize;
        let mut tokens_at_first_return = None;
        let turn = self.turns;
        let mut completed = false;
        let mut cancelled = false;
        for _ in 0..self.max_response_tokens.max(1) {
            let mut fragments = String::new();
            let end = match self.model.consult(&Consult::new(1), &mut |fragment: &str| {
                fragments.push_str(fragment);
                Ok(())
            }) {
                Ok(end) => end,
                Err(error) => return Err(FailedTurn { error, said }),
            };
            let (id, fragment) = match one_token_association(&end, fragments) {
                Ok(associated) => associated,
                Err(error) => return Err(FailedTurn { error, said }),
            };
            let delta = match self.output.push(id, &fragment) {
                Ok(delta) => delta,
                Err(error) => return Err(FailedTurn { error, said }),
            };
            tokens += 1;
            said.push_str(&delta.text);
            thinking_this_turn.push_str(&delta.thinking);
            if !delta.text.is_empty() {
                if let Some(voice) = &voice {
                    if let Err(error) = voice.say(&delta.text) {
                        return Err(FailedTurn {
                            error: error.into(),
                            said,
                        });
                    }
                    if tokens_at_first_return.is_none() && voice.report().records > 0 {
                        tokens_at_first_return = Some(tokens);
                    }
                }
            }
            microturns.push(end);
            if delta.completed {
                completed = true;
                break;
            }
            if (self.stop_requested)() {
                cancelled = true;
                break;
            }
        }

        let stopped = if completed {
            "content_model_end_sampling"
        } else if cancelled {
            "cancelled"
        } else {
            "max_response_tokens"
        };
        let end = match aggregate_microturns(turn, &microturns, stopped) {
            Ok(end) => end,
            Err(error) => return Err(FailedTurn { error, said }),
        };
        // THE LOGICAL-TURN BOUNDARY, and what is left of `ServePair`'s
        // cross-rank check. No collective is in flight here, so a host-side
        // round trip cannot deadlock against one; the ranks compare a running
        // digest of every token id every pass has returned, and a disagreement
        // is terminal. The module header records exactly what this is weaker
        // than: the deleted proxy withheld each fragment until the peer
        // confirmed it, so a divergent byte was never spoken. This one is an
        // alarm after the fact — the turn has already been said.
        if let Err(error) = self.model.agree_sequence() {
            return Err(FailedTurn { error, said });
        }
        self.position = Some(end.position);
        self.pending_exhaust += super::telemetry::turn_end_fragment(&end);
        let records_at_end = voice.as_ref().map(|v| v.report().records).unwrap_or(0);
        if let Some(proofs) = &self.proofs {
            proofs.lock().expect("proof log").push(StreamProof {
                turn,
                tokens,
                tokens_at_first_return,
                records_at_end,
                // One element per generated token, in order. `aggregate_microturns`
                // has already checked that each microturn was exactly one
                // arbitrated id, so these ARE per-token seconds and not per-turn
                // ones divided by something.
                token_secs: microturns.iter().map(|end| end.turn_secs).collect(),
            });
        }
        if let Some(log) = &self.log {
            log.lock().expect("turn log").push(end);
        }
        self.turns += 1;

        // The one Drive turn is the complete semantic response (or its truthful
        // interrupted prefix), even though one-token consults streamed it to
        // the voice as it was generated.
        let reasoning = thinking_this_turn;
        let mut decision = if completed {
            let call = match self.output.take_completed_call() {
                Ok(call) => call,
                Err(error) => return Err(FailedTurn { error, said }),
            };
            match call {
                Some(call) => {
                    let fresh_base = self.buffer.end_offset();
                    let range = project_native_exec(&mut said, &call.command);
                    let span = said[range.clone()].to_string();
                    let _ = self.finish_coverage();
                    self.outstanding_exec = Some(call.command.clone());
                    drive::mind::Decision::fire_projected(
                        call.command,
                        span,
                        fresh_base + range.start as u64,
                        fresh_base + range.end as u64,
                        watermark,
                        "inkling: strict native exec call projected into this same turn",
                    )
                }
                None => {
                    self.needs_generation_prompt = true;
                    let (span_start, span_end) = self.finish_coverage();
                    drive::mind::Decision::no_action(
                        span_start,
                        span_end,
                        watermark,
                        "inkling: completed a text-only assistant response",
                    )
                }
            }
        } else if cancelled {
            self.response_interrupted = true;
            let (span_start, span_end) = self.finish_coverage();
            drive::mind::Decision::no_action(
                span_start,
                span_end,
                watermark,
                "inkling: response interrupted by stop request",
            )
        } else {
            self.response_interrupted = true;
            let (span_start, span_end) = self.finish_coverage();
            drive::mind::Decision::no_action(
                span_start,
                span_end,
                watermark,
                format!(
                    "inkling response exceeded the configured {}-token response cap without content_model_end_sampling",
                    self.max_response_tokens.max(1)
                ),
            )
        };
        if !reasoning.is_empty() {
            decision = decision.with_reasoning(reasoning);
        }
        if completed {
            Ok(drive::mind::Turn::new(said, decision))
        } else if cancelled {
            Ok(drive::mind::Turn {
                said,
                decision,
                response_state: drive::mind::ResponseState::Interrupted,
                continuation: drive::mind::TurnContinuation::Continue,
            })
        } else {
            let error = decision.rationale.clone();
            Ok(drive::mind::Turn {
                said,
                decision,
                response_state: drive::mind::ResponseState::Interrupted,
                continuation: drive::mind::TurnContinuation::Terminal { error },
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── the scripted model ──────────────────────────────────────────────────
    //
    // Every mind test below used to need a real subprocess: a shell script
    // replaying a pre-encoded framed stream, two temporary files, and a reaped
    // child. That was not a choice, it was the seam — the model was in another
    // process, so a fake model had to be one too.
    //
    // With the model in this process the seam is [`Model`], and a fake model is
    // a struct. These tests now allocate nothing, spawn nothing, touch no
    // filesystem, and can assert on what the mind ASKED FOR rather than
    // decoding the bytes it wrote.

    /// One thing the mind asked the model to do.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ModelCall {
        Context(InklingContext),
        Preflight(ContextPlacement),
        Reinitialize,
        /// The requested `max_tokens`. Drive's mind must only ever say 1.
        Consult(usize),
        Agree,
        Kill,
        Shutdown,
    }

    /// A [`Model`] that answers from a script.
    ///
    /// The `TurnEnd` shaping deliberately reproduces what the deleted
    /// `inkling_serve` produced, because [`aggregate_microturns`] enforces it:
    /// the first microturn of a session carries the delta and no carry, and
    /// every microturn after it carries exactly one token and no delta.
    struct ScriptedModel {
        ready: Ready,
        /// `(token id, decoded fragment)` per one-token consult, in order.
        sequence: std::collections::VecDeque<(u32, String)>,
        /// Consult ordinal (global, from 0) at which `consult` fails instead of
        /// answering. Models a rank dying mid-response.
        fail_at_consult: Option<usize>,
        /// If set, `reinitialize` refuses with this message and changes nothing.
        reinitialize_error: Option<String>,
        /// If set, `agree_sequence` reports this cross-rank divergence.
        agree_error: Option<String>,
        /// If set, `preflight_context` answers with this evidence instead of
        /// computing it. The only way to hand the mind evidence that does not
        /// agree with the admission arithmetic.
        preflight_override: Option<ContextPreflighted>,
        /// Delta width every typed context is priced and charged at.
        context_tokens: usize,
        /// Positions the pretend session holds.
        position: usize,
        /// Consults answered so far, i.e. the global microturn ordinal.
        microturn: usize,
        log: std::sync::Arc<std::sync::Mutex<Vec<ModelCall>>>,
    }

    impl ScriptedModel {
        fn new(ready: Ready, sequence: &[(u32, &str)]) -> Self {
            Self {
                ready,
                sequence: sequence
                    .iter()
                    .map(|(id, text)| (*id, (*text).to_string()))
                    .collect(),
                fail_at_consult: None,
                reinitialize_error: None,
                agree_error: None,
                preflight_override: None,
                context_tokens: 3,
                position: 8,
                microturn: 0,
                log: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        /// The shared call log. Cloned BEFORE the model is boxed into the mind,
        /// which owns it from then on.
        fn log(&self) -> std::sync::Arc<std::sync::Mutex<Vec<ModelCall>>> {
            std::sync::Arc::clone(&self.log)
        }

        fn record(&self, call: ModelCall) {
            self.log.lock().expect("scripted model log").push(call);
        }

        /// One unfed carry token exists exactly when a turn has been generated.
        fn carried(&self) -> usize {
            usize::from(self.microturn > 0)
        }
    }

    /// What a mind's whole conversation with a scripted model looked like.
    fn calls(log: &std::sync::Arc<std::sync::Mutex<Vec<ModelCall>>>) -> Vec<ModelCall> {
        log.lock().expect("scripted model log").clone()
    }

    /// The exact shape `assert_one_initialization_and_microconsults` used to
    /// prove by decoding captured framed-stream bytes: one typed context, then
    /// `expected` one-token consults, then the per-turn agreement.
    fn assert_one_initialization_and_microconsults(
        log: &std::sync::Arc<std::sync::Mutex<Vec<ModelCall>>>,
        expected: usize,
    ) {
        let calls = calls(log);
        let mut calls = calls
            .iter()
            .filter(|call| !matches!(call, ModelCall::Preflight(_) | ModelCall::Shutdown));
        assert!(
            matches!(
                calls.next(),
                Some(ModelCall::Context(InklingContext::Initialize { .. }))
            ),
            "the first thing a mind installs is one typed initialization"
        );
        for index in 0..expected {
            assert_eq!(
                calls.next(),
                Some(&ModelCall::Consult(1)),
                "microturn {index} must be a one-token consult"
            );
        }
    }

    impl Model for ScriptedModel {
        fn ready(&self) -> &Ready {
            &self.ready
        }

        fn context(&mut self, context: &InklingContext) -> Result<()> {
            self.record(ModelCall::Context(context.clone()));
            Ok(())
        }

        fn preflight_context(&mut self, request: &ContextPreflight) -> Result<ContextPreflighted> {
            self.record(ModelCall::Preflight(request.placement));
            if let Some(evidence) = &self.preflight_override {
                return Ok(evidence.clone());
            }
            context_preflight(
                request.placement,
                self.position,
                self.carried(),
                self.context_tokens,
                request.max_response_tokens,
                self.ready.context_budget,
            )
        }

        fn reinitialize(&mut self, initialization: &InklingContext) -> Result<Reinitialized> {
            self.record(ModelCall::Reinitialize);
            anyhow::ensure!(
                matches!(initialization, InklingContext::Initialize { .. }),
                "a reinitialization requires one complete Initialize payload"
            );
            if let Some(message) = &self.reinitialize_error {
                anyhow::bail!("{message}");
            }
            let acknowledgement = Reinitialized {
                previous_position: self.position,
                previous_turns: self.microturn,
                initialization_tokens: self.context_tokens,
            };
            self.position = 0;
            self.microturn = 0;
            Ok(acknowledgement)
        }

        fn consult(
            &mut self,
            request: &Consult,
            on_token: &mut dyn FnMut(&str) -> Result<()>,
        ) -> Result<TurnEnd> {
            self.record(ModelCall::Consult(request.max_tokens));
            let microturn = self.microturn;
            self.microturn += 1;
            anyhow::ensure!(
                self.fail_at_consult != Some(microturn),
                "scripted model failed at consult {microturn}"
            );
            let (id, fragment) = self
                .sequence
                .pop_front()
                .context("the scripted model ran out of tokens")?;
            on_token(&fragment)?;
            let mut end = fake_end(&[id]);
            end.turn = microturn;
            if microturn > 0 {
                end.delta_tokens = 0;
                end.carried = 1;
            }
            if id == self.ready.special_ids.content_model_end_sampling {
                end.stopped = "stop_token".to_string();
            }
            end.position = self.position;
            Ok(end)
        }

        fn agree_sequence(&mut self) -> Result<()> {
            self.record(ModelCall::Agree);
            match &self.agree_error {
                Some(message) => anyhow::bail!("{message}"),
                None => Ok(()),
            }
        }

        fn kill(&mut self) -> Result<()> {
            self.record(ModelCall::Kill);
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            self.record(ModelCall::Shutdown);
            Ok(())
        }
    }

    /// A scripted mind, plus the log of everything it will ask for.
    fn scripted_mind(
        sequence: &[(u32, &str)],
        max_response_tokens: usize,
        system: &str,
    ) -> (
        InklingMind,
        std::sync::Arc<std::sync::Mutex<Vec<ModelCall>>>,
    ) {
        let model = ScriptedModel::new(fake_ready("scripted.pile"), sequence);
        let log = model.log();
        let mind = InklingMind::new_gate(
            Box::new(model),
            max_response_tokens,
            Some(system.to_string()),
        )
        .expect("valid READY evidence");
        (mind, log)
    }

    #[test]
    fn exact_context_admission_covers_initial_generation_and_tool_deltas() {
        let initial = context_preflight(ContextPlacement::Append, 0, 0, 7, 4, 10).unwrap();
        assert_eq!(initial.required_end, Some(10));
        assert!(initial.fits, "equality with the budget is admitted");
        assert!(
            !context_preflight(ContextPlacement::Append, 0, 0, 7, 4, 9)
                .unwrap()
                .fits,
            "one position beyond the budget is refused"
        );

        let generation = context_preflight(ContextPlacement::Append, 10, 1, 1, 4, 15).unwrap();
        assert_eq!(generation.required_end, Some(15));
        assert!(generation.fits);
        assert!(
            !context_preflight(ContextPlacement::Append, 10, 1, 1, 4, 14)
                .unwrap()
                .fits
        );

        let tool = context_preflight(ContextPlacement::Append, 10, 1, 20, 4, 34).unwrap();
        assert_eq!(tool.required_end, Some(34));
        assert!(tool.fits);
        assert!(
            !context_preflight(ContextPlacement::Append, 10, 1, 20, 4, 33)
                .unwrap()
                .fits
        );

        let replacement = context_preflight(ContextPlacement::Replace, 9_999, 1, 7, 4, 10).unwrap();
        assert_eq!(replacement.position, 0);
        assert_eq!(replacement.carried, 0);
        assert_eq!(replacement.required_end, Some(10));
        assert!(replacement.fits);
    }

    #[test]
    fn context_admission_overflow_is_a_pure_rejection() {
        let evidence =
            context_preflight(ContextPlacement::Append, usize::MAX, 1, 1, 1, usize::MAX).unwrap();
        assert_eq!(evidence.required_end, None);
        assert!(!evidence.fits);
    }

    #[test]
    fn execution_manifest_is_deterministic_and_length_delimited() {
        let mut first = ExecutionManifest::new("sealed-v1");
        first.field("a", b"bc");
        first.field("d", b"");
        let first = first.finish_hex();

        let mut same = ExecutionManifest::new("sealed-v1");
        same.field("a", b"bc");
        same.field("d", b"");
        assert_eq!(first, same.finish_hex());

        let mut different_boundary = ExecutionManifest::new("sealed-v1");
        different_boundary.field("ab", b"c");
        different_boundary.field("d", b"");
        assert_ne!(first, different_boundary.finish_hex());
    }

    #[test]
    fn execution_manifest_reader_hashes_the_exact_declared_bytes() {
        let mut direct = ExecutionManifest::new("sealed-v1");
        direct.field("executable-bytes", b"\0mary\xff");

        let mut streamed = ExecutionManifest::new("sealed-v1");
        streamed
            .reader("executable-bytes", 6, std::io::Cursor::new(b"\0mary\xff"))
            .expect("stream exact bytes");
        assert_eq!(direct.finish_hex(), streamed.finish_hex());
    }

    fn fake_ready(pile: &str) -> Ready {
        Ready {
            pile: pile.to_string(),
            model_identity: "11".repeat(32),
            tokenizer_identity: "22".repeat(32),
            special_ids: InklingSpecialIds {
                message_model: 101,
                message_system: 102,
                message_tool: 103,
                content_text: 104,
                content_xml: 105,
                content_thinking: 106,
                content_invoke_tool_json: 107,
                content_model_end_sampling: 108,
                end_message: 109,
                all_special: (101..=109).collect(),
                decoded_special: vec![
                    (101, MESSAGE_MODEL.to_string()),
                    (102, MESSAGE_SYSTEM.to_string()),
                    (103, MESSAGE_TOOL.to_string()),
                    (104, CONTENT_TEXT.to_string()),
                    (105, CONTENT_XML.to_string()),
                    (106, CONTENT_THINKING.to_string()),
                    (107, CONTENT_INVOKE_TOOL_JSON.to_string()),
                    (108, CONTENT_MODEL_END_SAMPLING.to_string()),
                    (109, END_MESSAGE.to_string()),
                ],
            },
            execution_profile: "sealed-v1".to_string(),
            execution_identity: "33".repeat(32),
            execution_unavailable: Vec::new(),
            tp_rank: None,
            tp_world: 1,
            layers: [0, 42],
            stack: 42,
            partial: false,
            vocab: 200_058,
            prefill_budget: 4096,
            context_budget: 65_536,
            load_secs: 1.0,
        }
    }

    fn special_fragment(ids: &InklingSpecialIds, id: u32, pending: &str) -> String {
        format!(
            "{pending}{}",
            ids.decoded_special(id).expect("fixture special decode")
        )
    }

    #[test]
    fn native_parser_routes_blocks_and_flushes_payload_before_special_transitions() {
        let ids = fake_ready("parser").special_ids;
        let mut parser = NativeOutputParser::new(ids.clone());

        parser
            .push(
                ids.content_thinking,
                &special_fragment(&ids, ids.content_thinking, ""),
            )
            .unwrap();
        assert_eq!(parser.push(7, "reason").unwrap().thinking, "reason");
        parser
            .push(
                ids.end_message,
                &special_fragment(&ids, ids.end_message, ""),
            )
            .unwrap();

        parser
            .push(
                ids.message_model,
                &special_fragment(&ids, ids.message_model, ""),
            )
            .unwrap();
        parser
            .push(
                ids.content_text,
                &special_fragment(&ids, ids.content_text, ""),
            )
            .unwrap();
        let literal = parser.push(8, CONTENT_INVOKE_TOOL_JSON).unwrap();
        assert_eq!(literal.text, CONTENT_INVOKE_TOOL_JSON);
        parser
            .push(
                ids.end_message,
                &special_fragment(&ids, ids.end_message, ""),
            )
            .unwrap();

        parser
            .push(
                ids.message_model,
                &special_fragment(&ids, ids.message_model, ""),
            )
            .unwrap();
        // `exec` was buffered by the decoder's preceding ordinary id. It is
        // returned with the content-kind spelling but belongs to Header.
        parser
            .push(
                ids.content_invoke_tool_json,
                &special_fragment(&ids, ids.content_invoke_tool_json, "exec"),
            )
            .unwrap();
        parser
            .push(9, r#"{"name":"exec","args":{"command":"printf"#)
            .unwrap();
        // The tail of an incomplete ordinary decode belongs to ToolJson even
        // though DecodeStream flushes it alongside end_message.
        parser
            .push(
                ids.end_message,
                &special_fragment(&ids, ids.end_message, r#" hi"}}"#),
            )
            .unwrap();
        let end = parser
            .push(
                ids.content_model_end_sampling,
                &special_fragment(&ids, ids.content_model_end_sampling, ""),
            )
            .unwrap();
        assert!(end.completed);
        assert_eq!(
            parser.take_completed_call().unwrap(),
            Some(NativeExecCall {
                command: "printf hi".to_string()
            })
        );
    }

    #[test]
    fn native_parser_rejects_malformed_incomplete_and_multiple_calls() {
        let ids = fake_ready("parser").special_ids;

        let mut malformed = NativeOutputParser::new(ids.clone());
        malformed.push(1, "exec").unwrap();
        malformed
            .push(
                ids.content_invoke_tool_json,
                &special_fragment(&ids, ids.content_invoke_tool_json, ""),
            )
            .unwrap();
        malformed
            .push(2, r#"{"name":"exec","args":{"command":"true"},"extra":1}"#)
            .unwrap();
        assert!(
            malformed
                .push(
                    ids.end_message,
                    &special_fragment(&ids, ids.end_message, ""),
                )
                .unwrap_err()
                .to_string()
                .contains("parse strict native exec JSON")
        );

        let mut incomplete = NativeOutputParser::new(ids.clone());
        incomplete.push(1, "exec").unwrap();
        incomplete
            .push(
                ids.content_invoke_tool_json,
                &special_fragment(&ids, ids.content_invoke_tool_json, ""),
            )
            .unwrap();
        incomplete.push(2, "{\"name\":").unwrap();
        assert!(
            incomplete
                .push(
                    ids.content_model_end_sampling,
                    &special_fragment(&ids, ids.content_model_end_sampling, ""),
                )
                .unwrap_err()
                .to_string()
                .contains("truncated an exec JSON block")
        );

        for (kind, expected) in [
            (ids.content_text, "truncated a text block"),
            (ids.content_thinking, "truncated a thinking block"),
        ] {
            let mut unclosed = NativeOutputParser::new(ids.clone());
            unclosed
                .push(kind, &special_fragment(&ids, kind, ""))
                .unwrap();
            unclosed.push(3, "payload").unwrap();
            assert!(
                unclosed
                    .push(
                        ids.content_model_end_sampling,
                        &special_fragment(&ids, ids.content_model_end_sampling, ""),
                    )
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
        }

        let mut empty = NativeOutputParser::new(ids.clone());
        assert!(
            empty
                .push(
                    ids.content_model_end_sampling,
                    &special_fragment(&ids, ids.content_model_end_sampling, ""),
                )
                .unwrap()
                .completed
        );

        let mut multiple = NativeOutputParser::new(ids.clone());
        multiple.push(1, "exec").unwrap();
        multiple
            .push(
                ids.content_invoke_tool_json,
                &special_fragment(&ids, ids.content_invoke_tool_json, ""),
            )
            .unwrap();
        multiple
            .push(2, r#"{"name":"exec","args":{"command":"true"}}"#)
            .unwrap();
        multiple
            .push(
                ids.end_message,
                &special_fragment(&ids, ids.end_message, ""),
            )
            .unwrap();
        let trailing = multiple
            .push(
                ids.message_model,
                &special_fragment(&ids, ids.message_model, ""),
            )
            .unwrap_err();
        assert!(
            trailing
                .to_string()
                .contains("another block after its exec call")
        );
    }

    #[test]
    fn native_parser_preserves_state_across_one_token_microconsults() {
        let ids = fake_ready("parser").special_ids;
        let mut parser = NativeOutputParser::new(ids.clone());
        parser
            .push(
                ids.content_text,
                &special_fragment(&ids, ids.content_text, ""),
            )
            .unwrap();
        assert_eq!(
            parser.push(1, "first fragment").unwrap().text,
            "first fragment"
        );
        assert!(parser.take_completed_call().is_err());

        assert_eq!(
            parser.push(2, " second fragment").unwrap().text,
            " second fragment"
        );
        parser
            .push(
                ids.end_message,
                &special_fragment(&ids, ids.end_message, ""),
            )
            .unwrap();
        assert!(
            parser
                .push(
                    ids.content_model_end_sampling,
                    &special_fragment(&ids, ids.content_model_end_sampling, ""),
                )
                .unwrap()
                .completed
        );
        assert_eq!(parser.take_completed_call().unwrap(), None);
    }

    #[test]
    fn native_call_projection_names_the_exact_fresh_same_turn_bytes() {
        let mut said = "I will inspect.".to_string();
        let range = project_native_exec(&mut said, "printf same-turn");
        assert_eq!(said, "I will inspect.\n$ printf same-turn\n");
        assert_eq!(&said[range.clone()], "\n$ printf same-turn\n");
        let prior_monologue_end = 73_u64;
        assert_eq!(prior_monologue_end + range.start as u64, 88);
        assert_eq!(prior_monologue_end + range.end as u64, 108);
    }

    #[test]
    fn projected_calls_carry_explicit_archive_semantics() {
        let decision = drive::mind::Decision::fire_projected(
            "printf same-turn",
            "\n$ printf same-turn\n",
            88,
            108,
            0,
            "native call projection",
        );
        assert_eq!(
            decision.command_span_origin,
            Some(drive::mind::CommandSpanOrigin::Projected)
        );
    }

    #[test]
    fn one_token_association_refuses_ambiguous_turn_ends() {
        assert_eq!(
            one_token_association(&fake_end(&[41]), "fragment".to_string()).unwrap(),
            (41, "fragment".to_string())
        );
        assert!(one_token_association(&fake_end(&[41, 42]), String::new()).is_err());
        let mut inconsistent = fake_end(&[41]);
        inconsistent.tokens = 2;
        assert!(one_token_association(&inconsistent, String::new()).is_err());
    }

    #[test]
    fn logical_turn_aggregation_requires_every_microturn_to_be_one_token() {
        let first = fake_end(&[41]);
        let mut second = fake_end(&[42]);
        second.turn = 1;
        second.delta_tokens = 0;
        second.carried = 1;
        let combined =
            aggregate_microturns(9, &[first.clone(), second.clone()], "max_response_tokens")
                .unwrap();
        assert_eq!(combined.turn, 9);
        assert_eq!(combined.tokens, 2);
        assert_eq!(combined.token_ids, [41, 42]);
        assert_eq!(combined.delta_tokens, first.delta_tokens);
        assert_eq!(combined.carried, first.carried);
        assert_eq!(combined.stopped, "max_response_tokens");

        second.tokens = 2;
        assert!(aggregate_microturns(9, &[first, second], "cancelled").is_err());
    }

    #[cfg(feature = "tokenizer")]
    fn miniature_tokenizer_json() -> Vec<u8> {
        let added_tokens = [
            MESSAGE_MODEL,
            MESSAGE_SYSTEM,
            MESSAGE_TOOL,
            CONTENT_TEXT,
            CONTENT_XML,
            CONTENT_THINKING,
            CONTENT_INVOKE_TOOL_JSON,
            CONTENT_MODEL_END_SAMPLING,
            END_MESSAGE,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, content)| {
            serde_json::json!({
                "id": index + 1,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect::<Vec<_>>();
        serde_json::to_vec(&serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added_tokens,
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0},
                "unk_token": "<unk>",
            },
        }))
        .expect("serialize miniature tokenizer")
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn content_only_codec_makes_marker_injection_non_structural() {
        let codec = InklingContextCodec::from_json(&miniature_tokenizer_json()).unwrap();
        let injected = format!("before {MESSAGE_MODEL} {END_MESSAGE} after");
        let raw = codec.encode_raw_content(&injected).unwrap();
        assert_eq!(raw, [0]);
        assert!(
            raw.iter()
                .all(|id| !codec.special_ids().is_special(*id as u32))
        );

        let response = InklingHistoryResponse {
            parts: vec![
                InklingHistoryPart::Thinking {
                    content: format!("think {MESSAGE_MODEL}"),
                },
                InklingHistoryPart::Text {
                    content: format!("say {END_MESSAGE}"),
                },
                InklingHistoryPart::Exec {
                    command: format!("printf {MESSAGE_TOOL}"),
                },
            ],
            tool_result: Some(format!("hostile {CONTENT_MODEL_END_SAMPLING}")),
        };
        let encoded = codec
            .encode(&InklingContext::HistoricalResponse { response })
            .unwrap();
        let special = codec.special_ids();
        assert_eq!(
            encoded,
            [
                special.message_model as usize,
                special.content_thinking as usize,
                0,
                special.end_message as usize,
                special.message_model as usize,
                special.content_text as usize,
                0,
                special.end_message as usize,
                special.message_model as usize,
                0,
                special.content_invoke_tool_json as usize,
                0,
                special.end_message as usize,
                special.content_model_end_sampling as usize,
                special.message_tool as usize,
                0,
                special.content_text as usize,
                0,
                special.end_message as usize,
                special.message_model as usize,
            ]
        );
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn initialization_sequence_matches_the_shipped_template_shape() {
        let codec = InklingContextCodec::from_json(&miniature_tokenizer_json()).unwrap();
        let ids = codec.special_ids();
        let encoded = codec
            .encode(&InklingContext::Initialize {
                system: "system".to_string(),
                history: vec![
                    InklingHistoryResponse {
                        parts: vec![
                            InklingHistoryPart::Thinking {
                                content: "two ".to_string(),
                            },
                            // One provider block may cross several Drive
                            // slices. Adjacent parts of the same kind must be
                            // token-identical to the unsliced block.
                            InklingHistoryPart::Thinking {
                                content: "slices".to_string(),
                            },
                            InklingHistoryPart::Text {
                                content: "answer".to_string(),
                            },
                            InklingHistoryPart::Exec {
                                command: "true".to_string(),
                            },
                        ],
                        tool_result: Some("ok".to_string()),
                    },
                    InklingHistoryResponse {
                        parts: vec![InklingHistoryPart::Text {
                            content: "after".to_string(),
                        }],
                        tool_result: None,
                    },
                ],
            })
            .unwrap();
        assert_eq!(
            encoded,
            [
                ids.message_system as usize,
                0,
                ids.content_xml as usize,
                0,
                ids.end_message as usize,
                ids.message_system as usize,
                ids.content_text as usize,
                0,
                ids.end_message as usize,
                ids.message_system as usize,
                ids.content_text as usize,
                0,
                ids.end_message as usize,
                ids.message_model as usize,
                ids.content_thinking as usize,
                0,
                ids.end_message as usize,
                ids.message_model as usize,
                ids.content_text as usize,
                0,
                ids.end_message as usize,
                ids.message_model as usize,
                0,
                ids.content_invoke_tool_json as usize,
                0,
                ids.end_message as usize,
                ids.content_model_end_sampling as usize,
                ids.message_tool as usize,
                0,
                ids.content_text as usize,
                0,
                ids.end_message as usize,
                ids.message_model as usize,
                ids.content_text as usize,
                0,
                ids.end_message as usize,
                ids.content_model_end_sampling as usize,
                ids.message_model as usize,
            ]
        );
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn adjacent_history_slices_are_token_identical_to_one_model_part() {
        let codec = InklingContextCodec::from_json(&miniature_tokenizer_json()).unwrap();
        let split = InklingContext::HistoricalResponse {
            response: InklingHistoryResponse {
                parts: vec![
                    InklingHistoryPart::Thinking {
                        content: "one ".to_string(),
                    },
                    InklingHistoryPart::Thinking {
                        content: "thought".to_string(),
                    },
                    InklingHistoryPart::Text {
                        content: "one ".to_string(),
                    },
                    InklingHistoryPart::Text {
                        content: "answer".to_string(),
                    },
                ],
                tool_result: None,
            },
        };
        let whole = InklingContext::HistoricalResponse {
            response: InklingHistoryResponse {
                parts: vec![
                    InklingHistoryPart::Thinking {
                        content: "one thought".to_string(),
                    },
                    InklingHistoryPart::Text {
                        content: "one answer".to_string(),
                    },
                ],
                tool_result: None,
            },
        };
        assert_eq!(codec.encode(&split).unwrap(), codec.encode(&whole).unwrap());
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn historical_response_validation_rejects_unpaired_or_nonfinal_execs() {
        let codec = InklingContextCodec::from_json(&miniature_tokenizer_json()).unwrap();
        let encode = |response| codec.encode(&InklingContext::HistoricalResponse { response });

        assert!(
            encode(InklingHistoryResponse {
                parts: vec![InklingHistoryPart::Exec {
                    command: "true".to_string(),
                }],
                tool_result: None,
            })
            .unwrap_err()
            .to_string()
            .contains("requires its tool result")
        );
        assert!(
            encode(InklingHistoryResponse {
                parts: vec![InklingHistoryPart::Text {
                    content: "answer".to_string(),
                }],
                tool_result: Some("orphan".to_string()),
            })
            .unwrap_err()
            .to_string()
            .contains("has no final exec call")
        );
        assert!(
            encode(InklingHistoryResponse {
                parts: vec![
                    InklingHistoryPart::Exec {
                        command: "true".to_string(),
                    },
                    InklingHistoryPart::Text {
                        content: "too late".to_string(),
                    },
                ],
                tool_result: Some("ok".to_string()),
            })
            .unwrap_err()
            .to_string()
            .contains("final model part")
        );
        assert!(
            encode(InklingHistoryResponse {
                parts: vec![
                    InklingHistoryPart::Exec {
                        command: "first".to_string(),
                    },
                    InklingHistoryPart::Exec {
                        command: "second".to_string(),
                    },
                ],
                tool_result: Some("ambiguous".to_string()),
            })
            .unwrap_err()
            .to_string()
            .contains("multiple exec calls")
        );
    }

    #[cfg(feature = "tokenizer")]
    fn shipped_tokenizer_bytes() -> Option<Vec<u8>> {
        let path = std::env::var_os("INKLING_TEST_TOKENIZER_JSON")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("/private/tmp/inkling-small-tokenizer.json")
            });
        std::fs::read(path).ok()
    }

    #[cfg(feature = "tokenizer")]
    fn shipped_template_bytes() -> Option<Vec<u8>> {
        let path = std::env::var_os("INKLING_TEST_CHAT_TEMPLATE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("/private/tmp/inkling-small-chat-template.jinja")
            });
        std::fs::read(path).ok()
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn codec_matches_the_exact_shipped_tokenizer_template_sequence() {
        let Some(bytes) = shipped_tokenizer_bytes() else {
            eprintln!("shipped tokenizer fixture unavailable; set INKLING_TEST_TOKENIZER_JSON");
            return;
        };
        let template = shipped_template_bytes().expect(
            "shipped template fixture unavailable; set INKLING_TEST_CHAT_TEMPLATE alongside the tokenizer",
        );
        // This is the BLAKE3 identity of the supplied template whose SHA-256
        // is 0aa1aa0c729d90176dcaa00c440c8faffca2957ffb2cc4b79456ee6d02bcf43b.
        assert_eq!(
            blake3::hash(&template).to_hex().as_str(),
            "54def8dad65b827478855ee10baa829d4fca0064d724d4bb22954dbe31f18321"
        );
        let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes).expect("shipped tokenizer");
        let codec = InklingContextCodec::from_json(&bytes).expect("shipped context codec");
        let result = ExecResultContext {
            command: "printf 'template'".to_string(),
            content: "template output".to_string(),
        };
        let context = InklingContext::Initialize {
            system: "template system".to_string(),
            history: vec![InklingHistoryResponse {
                parts: vec![
                    InklingHistoryPart::Thinking {
                        content: "template thought".to_string(),
                    },
                    InklingHistoryPart::Text {
                        content: "template answer".to_string(),
                    },
                    InklingHistoryPart::Exec {
                        command: result.command.clone(),
                    },
                ],
                tool_result: Some(result.content.clone()),
            }],
        };
        let rendered = format!(
            "{MESSAGE_SYSTEM}tool_declare{CONTENT_XML}{EXEC_TOOL_DECLARATION}{END_MESSAGE}\
             {MESSAGE_SYSTEM}{CONTENT_TEXT}template system{END_MESSAGE}\
             {MESSAGE_SYSTEM}{CONTENT_TEXT}{DEFAULT_THINKING_EFFORT}{END_MESSAGE}\
             {MESSAGE_MODEL}{CONTENT_THINKING}template thought{END_MESSAGE}\
             {MESSAGE_MODEL}{CONTENT_TEXT}template answer{END_MESSAGE}\
             {MESSAGE_MODEL}exec{CONTENT_INVOKE_TOOL_JSON}{}{END_MESSAGE}\
             {CONTENT_MODEL_END_SAMPLING}{MESSAGE_TOOL}exec{CONTENT_TEXT}{}\
             {END_MESSAGE}{MESSAGE_MODEL}",
            canonical_exec_call_json(&result.command),
            result.content,
        );
        let template_ids = tokenizer
            .encode(rendered, false)
            .expect("tokenize template rendering")
            .get_ids()
            .iter()
            .map(|id| *id as usize)
            .collect::<Vec<_>>();
        assert_eq!(codec.encode(&context).unwrap(), template_ids);

        let hostile = format!("system {MESSAGE_MODEL} {END_MESSAGE}");
        let safe = codec.encode_raw_content(&hostile).unwrap();
        assert!(
            safe.iter()
                .all(|id| !codec.special_ids().is_special(*id as u32))
        );
    }

    #[cfg(feature = "tokenizer")]
    #[test]
    fn shipped_decoder_flush_at_special_is_applied_before_transition() {
        let Some(bytes) = shipped_tokenizer_bytes() else {
            eprintln!("shipped tokenizer fixture unavailable; set INKLING_TEST_TOKENIZER_JSON");
            return;
        };
        let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes).expect("shipped tokenizer");
        let ids = InklingContextCodec::from_json(&bytes)
            .expect("shipped context codec")
            .special_ids()
            .clone();
        let incomplete = tokenizer
            .token_to_id("Ã")
            .expect("shipped byte-level incomplete UTF-8 token");
        let mut stream = tokenizer.decode_stream(false);
        let mut parser = NativeOutputParser::new(ids.clone());

        let kind = stream
            .step(ids.content_text)
            .expect("decode content kind")
            .expect("content kind fragment");
        parser.push(ids.content_text, &kind).unwrap();
        assert_eq!(
            stream.step(incomplete).expect("decode incomplete byte"),
            None
        );
        let flushed = stream
            .step(ids.end_message)
            .expect("decode end_message")
            .expect("end_message flush");
        assert!(
            flushed.len()
                > ids
                    .decoded_special(ids.end_message)
                    .expect("end-message runtime decode")
                    .len(),
            "fixture must exercise pending payload plus the structural suffix: {flushed:?}"
        );
        let delta = parser.push(ids.end_message, &flushed).unwrap();
        assert!(!delta.text.is_empty(), "pending payload was not discarded");
    }

    fn fake_end(token_ids: &[u32]) -> TurnEnd {
        TurnEnd {
            turn: 0,
            tokens: token_ids.len(),
            token_ids: token_ids.to_vec(),
            delta_tokens: 3,
            carried: 0,
            stopped: "max_tokens".to_string(),
            first_token_secs: 0.01,
            turn_secs: 0.02,
            position: 8,
        }
    }

    #[test]
    fn a_typed_drive_result_crosses_the_text_only_seam_deliberately() {
        let content = drive::content::Content::text("output\n[exit 7]");
        assert_eq!(
            text_result_content(&content, false, Some(7)),
            "output\n[exit 7]\n[result status: exit_code=7]\n"
        );
        assert_eq!(
            text_result_content(&content, true, Some(7)),
            "output\n[exit 7]\n[result status: isError=true, exit_code=7]\n"
        );

        // The native call already names its command structurally. Tool content
        // does not duplicate it as a shell-looking text scaffold, and ordinary
        // success does not grow a second status rendering.
        let ok = drive::content::Content::text("output\n[exit 0]");
        assert_eq!(
            text_result_content(&ok, false, Some(0)),
            "output\n[exit 0]\n"
        );
    }

    #[test]
    fn drive_replacement_history_preserves_parts_and_closes_the_exact_call() {
        let image = drive::mind::ContextImage {
            system: "system".to_string(),
            responses: vec![
                drive::mind::ContextResponse {
                    parts: vec![
                        drive::mind::ContextPart::Thinking("consider".to_string()),
                        drive::mind::ContextPart::Text("answer".to_string()),
                    ],
                    tool_result: None,
                },
                drive::mind::ContextResponse {
                    parts: vec![
                        drive::mind::ContextPart::Thinking("inspect".to_string()),
                        drive::mind::ContextPart::ToolCall("ls -la".to_string()),
                    ],
                    tool_result: Some(drive::mind::ContextToolResult {
                        content: drive::content::Content::text("listing"),
                        is_error: false,
                        exit_code: Some(0),
                    }),
                },
            ],
            monologue_end: 0,
        };

        let history = replacement_history(&image).expect("lower Drive context");
        assert_eq!(
            history,
            vec![
                InklingHistoryResponse {
                    parts: vec![
                        InklingHistoryPart::Thinking {
                            content: "consider".to_string(),
                        },
                        InklingHistoryPart::Text {
                            content: "answer".to_string(),
                        },
                    ],
                    tool_result: None,
                },
                InklingHistoryResponse {
                    parts: vec![
                        InklingHistoryPart::Thinking {
                            content: "inspect".to_string(),
                        },
                        InklingHistoryPart::Exec {
                            command: "ls -la".to_string(),
                        },
                    ],
                    tool_result: Some("listing\n".to_string()),
                },
            ]
        );
        validate_replacement_boundary(false, Some("ls -la"), &history)
            .expect("the exact archived result closes the outstanding call");
        let error = validate_replacement_boundary(false, Some("pwd"), &history)
            .expect_err("a different call must not be silently closed");
        assert!(error.to_string().contains("pwd"), "{error:#}");
        assert!(
            validate_replacement_boundary(true, None, &history).is_ok(),
            "a completed text response is also a replacement boundary"
        );
    }

    // ── the ported mind tests ───────────────────────────────────────────────

    /// Borrow an owned script as the `&[(u32, &str)]` [`ScriptedModel::new`]
    /// takes. Fragments are built with [`special_fragment`], which owns its
    /// strings, exactly as they were for the deleted subprocess fixture.
    fn script(owned: &[(u32, String)]) -> Vec<(u32, &str)> {
        owned
            .iter()
            .map(|(id, text)| (*id, text.as_str()))
            .collect()
    }

    #[test]
    fn pending_context_preview_is_typed_and_does_not_mutate_mind_state() {
        let model = ScriptedModel::new(fake_ready("scripted.pile"), &[]);
        let log = model.log();
        let mut mind = InklingMind::new(Box::new(model), 8, Some("system".to_string()))
            .expect("valid READY evidence");

        assert_eq!(
            mind.pending_context(&[]).unwrap(),
            PendingObservationContext::Context(InklingContext::Initialize {
                system: "system".to_string(),
                history: Vec::new(),
            })
        );
        assert!(!mind.initialized);
        assert_eq!(mind.system.as_deref(), Some("system"));

        mind.initialized = true;
        mind.needs_generation_prompt = true;
        assert_eq!(
            mind.pending_context(&[]).unwrap(),
            PendingObservationContext::Context(InklingContext::GenerationPrompt)
        );
        assert!(mind.needs_generation_prompt);

        mind.needs_generation_prompt = false;
        mind.outstanding_exec = Some("printf exact".to_string());
        let events = [drive::world::Event::text_result(7, "printf exact", "exact")];
        assert_eq!(
            mind.pending_context(&events).unwrap(),
            PendingObservationContext::Context(InklingContext::ToolResult {
                result: ExecResultContext {
                    command: "printf exact".to_string(),
                    content: "exact\n".to_string(),
                },
            })
        );
        assert_eq!(mind.outstanding_exec.as_deref(), Some("printf exact"));

        drop(mind);
        // The old fixture proved this by reaping a child that had exited
        // successfully and by checking its captured input ended `Complete`.
        // The call log says it directly, and says more: previewing the pending
        // context asked the model for NOTHING at all.
        assert_eq!(
            calls(&log),
            vec![ModelCall::Shutdown],
            "a preview installs nothing, and the drop ends the model cleanly"
        );
    }

    #[test]
    fn failed_microturn_becomes_terminal_after_one_typed_initialization_batch() {
        use drive::mind::Mind as _;

        let system = "system first";
        let result_delta =
            text_result_content(&drive::content::Content::text("output"), false, None);
        let max_tokens = 7;

        // The exact typed batch `observe` must install before it consults. The
        // old fixture measured this as a byte prefix of the framed-stream
        // request; it is now simply the call the model records.
        let initial_context = InklingContext::Initialize {
            system: system.to_string(),
            history: vec![InklingHistoryResponse::exec(ExecResultContext {
                command: "cmd".to_string(),
                content: result_delta.clone(),
            })],
        };

        // The scripted model dies inside the first one-token collective, which
        // is the failure the old fixture staged as an ABORTED framed stream.
        // (The old fixture also released a decoder fragment WITHOUT its TURN
        // id, to prove exact id/fragment association is the parser's TP-safe
        // trust boundary. A `Model` returns the two together by signature, so
        // that half is now structurally impossible rather than checked here;
        // `one_token_association_refuses_ambiguous_turn_ends` still pins it.)
        let mut model = ScriptedModel::new(fake_ready("fixture.pile"), &[]);
        model.fail_at_consult = Some(0);
        let log = model.log();
        let mut mind = InklingMind::new(Box::new(model), max_tokens, Some(system.to_string()))
            .expect("valid READY evidence");
        assert!(mind.log().is_none(), "resident minds retain no turn log");
        assert!(
            mind.proofs().is_none(),
            "resident minds retain no streaming-proof log"
        );
        let events = [
            drive::world::Event::monologue(1, "prior "),
            drive::world::Event::text_result(2, "cmd", "output"),
        ];
        let turn = mind.observe(drive::world::MergedView {
            events: &events,
            watermark: 3,
        });

        assert_eq!(turn.said, "");
        assert_eq!(
            turn.decision.disposition,
            drive::mind::Disposition::NoAction
        );
        assert_eq!(turn.decision.span_start, 0);
        assert_eq!(turn.decision.span_end, "prior ".len() as u64);
        assert_eq!(turn.decision.watermark, 3);
        let terminal = turn
            .continuation
            .terminal_error()
            .expect("backend failure is terminal");
        assert!(
            terminal.contains("scripted model failed at consult 0"),
            "{terminal}"
        );

        // One typed batch lets the model place tool declaration + system +
        // effort before history, and the sole generation prompt after it. Then
        // exactly one one-token consult, and — because the failure is terminal
        // and a stranded tensor rank blocks in NCCL forever — the kill that
        // reaches the peer. The old fixture could only observe that kill as a
        // non-zero exit status on a reaped child.
        assert_eq!(
            calls(&log),
            vec![
                ModelCall::Context(initial_context),
                ModelCall::Consult(1),
                ModelCall::Kill,
            ]
        );

        let drop_started = std::time::Instant::now();
        drop(mind);
        assert!(drop_started.elapsed() < std::time::Duration::from_secs(3));
    }

    #[test]
    fn one_observation_owns_one_complete_semantic_response() {
        use drive::mind::{Mind as _, ResponseState, TurnContinuation};

        let ids = fake_ready("scripted.pile").special_ids;
        let mut owned = vec![(
            ids.content_thinking,
            special_fragment(&ids, ids.content_thinking, ""),
        )];
        // Cross the former 64-token Drive slice boundary. This remains one
        // observe and one archival response while every token is still one
        // independently arbitrated microconsult.
        owned.extend((0..64).map(|_| (7, "x".to_string())));
        owned.push((ids.end_message, special_fragment(&ids, ids.end_message, "")));
        owned.push((
            ids.content_model_end_sampling,
            special_fragment(&ids, ids.content_model_end_sampling, ""),
        ));
        let sequence = script(&owned);

        let (mut mind, log) = scripted_mind(&sequence, sequence.len(), "system");
        let startup = mind.take_exhaust();
        assert_eq!(startup.exports().count(), 1, "READY has one exported root");
        assert_eq!(mind.take_exhaust(), triblespace::prelude::Fragment::empty());

        let turn = mind.observe(drive::world::MergedView::EMPTY);
        let sample = mind.take_exhaust();
        assert_eq!(sample.exports().count(), 1, "TurnEnd has one exported root");
        assert_eq!(mind.take_exhaust(), triblespace::prelude::Fragment::empty());

        let expected_reasoning = "x".repeat(64);
        assert_eq!(turn.response_state, ResponseState::Complete);
        assert_eq!(turn.continuation, TurnContinuation::Continue);
        assert_eq!(turn.said, "");
        assert_eq!(
            turn.decision.reasoning.as_deref(),
            Some(expected_reasoning.as_str())
        );

        drop(mind);
        assert_eq!(
            calls(&log).last(),
            Some(&ModelCall::Shutdown),
            "drop ends the model cleanly"
        );
        assert_one_initialization_and_microconsults(&log, sequence.len());
    }

    #[test]
    fn response_cap_publishes_exact_interrupted_prefix_without_an_extra_consult() {
        use drive::mind::{Mind as _, ResponseState};

        let ids = fake_ready("scripted.pile").special_ids;
        let owned = vec![
            (
                ids.content_thinking,
                special_fragment(&ids, ids.content_thinking, ""),
            ),
            (6, "why".to_string()),
            (ids.end_message, special_fragment(&ids, ids.end_message, "")),
            (
                ids.message_model,
                special_fragment(&ids, ids.message_model, ""),
            ),
            (
                ids.content_text,
                special_fragment(&ids, ids.content_text, ""),
            ),
            (7, "partial".to_string()),
            (8, " response".to_string()),
        ];
        let sequence = script(&owned);
        let (mut mind, log) = scripted_mind(&sequence, sequence.len(), "system");

        let turn = mind.observe(drive::world::MergedView::EMPTY);

        assert_eq!(turn.said, "partial response");
        assert_eq!(turn.response_state, ResponseState::Interrupted);
        let terminal = turn
            .continuation
            .terminal_error()
            .expect("exhausting the semantic response cap is terminal");
        assert!(terminal.contains("7-token response cap"), "{terminal}");
        assert_eq!(turn.decision.reasoning.as_deref(), Some("why"));
        assert!(mind.response_interrupted);

        drop(mind);
        assert_eq!(
            calls(&log).last(),
            Some(&ModelCall::Shutdown),
            "drop ends the model cleanly"
        );
        assert_one_initialization_and_microconsults(&log, sequence.len());
    }

    #[test]
    fn cancellation_between_microconsults_is_audited_and_gracefully_continuable() {
        use drive::mind::{Mind as _, ResponseState, TurnContinuation};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ids = fake_ready("scripted.pile").special_ids;
        let owned = vec![
            (
                ids.content_thinking,
                special_fragment(&ids, ids.content_thinking, ""),
            ),
            (7, "alpha".to_string()),
            (8, " beta".to_string()),
        ];
        let sequence = script(&owned);
        let checks = std::sync::Arc::new(AtomicUsize::new(0));
        let observed_checks = std::sync::Arc::clone(&checks);
        let (mind, log) = scripted_mind(&sequence, 32, "system");
        let mut mind =
            mind.with_cancellation(move || observed_checks.fetch_add(1, Ordering::SeqCst) >= 1);

        let turn = mind.observe(drive::world::MergedView::EMPTY);

        assert_eq!(checks.load(Ordering::SeqCst), 2);
        assert_eq!(turn.said, "");
        assert_eq!(turn.decision.reasoning.as_deref(), Some("alpha"));
        assert_eq!(turn.response_state, ResponseState::Interrupted);
        assert_eq!(turn.continuation, TurnContinuation::Continue);
        assert!(turn.decision.rationale.contains("stop request"));
        assert!(mind.response_interrupted);

        drop(mind);
        assert_eq!(
            calls(&log).last(),
            Some(&ModelCall::Shutdown),
            "drop ends the model cleanly"
        );
        assert_one_initialization_and_microconsults(&log, 2);
    }

    #[test]
    fn replacement_resets_projection_to_the_absolute_monologue_extent() {
        use drive::mind::Mind as _;

        let ready = fake_ready("fixture.pile");
        let ids = ready.special_ids.clone();
        let command_text = "printf after-rollover";
        let owned = vec![
            ("exec-id".len() as u32, "exec".to_string()),
            (
                ids.content_invoke_tool_json,
                special_fragment(&ids, ids.content_invoke_tool_json, ""),
            ),
            (
                9,
                format!(r#"{{"name":"exec","args":{{"command":"{command_text}"}}}}"#),
            ),
            (ids.end_message, special_fragment(&ids, ids.end_message, "")),
            (
                ids.content_model_end_sampling,
                special_fragment(&ids, ids.content_model_end_sampling, ""),
            ),
        ];
        let sequence = script(&owned);

        // The old fixture pre-baked the two replacement records the client
        // would read back: evidence for `context_preflight(Replace, 123, 1, 12,
        // 32, budget)` and the acknowledgement for it. The scripted model
        // COMPUTES both from the same numbers — a session holding 123
        // positions, replaced by a 12-token initialization — so the arithmetic
        // is the model's rather than the test's transcription of it. (The
        // acknowledgement's `previous_turns` is 0 rather than the fixture's
        // hand-written 1, because this model has genuinely generated nothing
        // yet; nothing here asserts on it.)
        let mut model = ScriptedModel::new(ready, &sequence);
        model.context_tokens = 12;
        model.position = 123;
        let log = model.log();
        let mut mind = InklingMind::new_gate(Box::new(model), 32, Some("old system".to_string()))
            .expect("valid READY evidence");
        let startup = mind.take_exhaust();
        assert_eq!(startup.exports().count(), 1, "READY has one exported root");
        mind.initialized = true;
        mind.needs_generation_prompt = true;
        mind.position = Some(123);
        let monologue_end = 4_096;
        let image = drive::mind::ContextImage {
            system: "replacement system".to_string(),
            responses: Vec::new(),
            monologue_end,
        };

        mind.replace_context(1, &image)
            .expect("replace resident context");
        let replacement = mind.take_exhaust();
        assert_eq!(
            replacement.exports().count(),
            2,
            "replacement emits its preflight and successful ACK"
        );
        assert_eq!(mind.take_exhaust(), triblespace::prelude::Fragment::empty());
        assert_eq!(mind.buffer.base_offset(), monologue_end);
        assert_eq!(mind.buffer.end_offset(), monologue_end);
        let turn = mind.observe(drive::world::MergedView::EMPTY);
        assert_eq!(turn.decision.disposition, drive::mind::Disposition::Fire);
        assert_eq!(turn.decision.command.as_deref(), Some(command_text));
        assert_eq!(turn.decision.span_start, monologue_end);
        assert_eq!(
            turn.decision.span_end,
            monologue_end + turn.said.len() as u64
        );
        assert_eq!(turn.decision.span, turn.said);

        drop(mind);
        assert_eq!(
            calls(&log).last(),
            Some(&ModelCall::Shutdown),
            "drop ends the model cleanly"
        );
    }

    #[test]
    fn failed_reinitialization_does_not_emit_an_installed_context() {
        use drive::mind::Mind as _;

        // The old fixture wrote the successful PREFLIGHTED record and then
        // ended the stream WITHOUT a REINITIALIZED one. The scripted model
        // states the same thing directly: the preflight succeeds, the
        // replacement itself refuses, and nothing was installed.
        let mut model = ScriptedModel::new(fake_ready("fixture.pile"), &[]);
        model.context_tokens = 12;
        model.position = 123;
        model.reinitialize_error =
            Some("the scripted model refused to install the replacement".to_string());
        let log = model.log();
        let mut mind = InklingMind::new_gate(Box::new(model), 32, Some("old system".to_string()))
            .expect("valid READY evidence");
        let _ready = mind.take_exhaust();
        mind.initialized = true;
        mind.needs_generation_prompt = true;
        mind.position = Some(123);
        let image = drive::mind::ContextImage {
            system: "replacement system".to_string(),
            responses: Vec::new(),
            monologue_end: 4_096,
        };

        let failure = mind
            .replace_context(1, &image)
            .expect_err("a refused reinitialization must fail");
        assert!(failure.is_terminal());
        assert_eq!(
            mind.context_epoch(),
            0,
            "failed replacement keeps the epoch"
        );
        let exhaust = mind.take_exhaust();
        assert_eq!(
            exhaust.exports().count(),
            1,
            "the successful preflight is evidence, but no ACK was received"
        );
        assert_eq!(mind.take_exhaust(), triblespace::prelude::Fragment::empty());
        // A terminal replacement failure has to reach the peer rank for the
        // same reason a terminal turn does.
        assert!(calls(&log).contains(&ModelCall::Kill));

        drop(mind);
    }

    #[test]
    fn native_call_crossing_microconsults_becomes_an_exact_same_turn_fire() {
        use drive::mind::Mind as _;

        let ready = fake_ready("fixture.pile");
        let ids = ready.special_ids.clone();
        let owned = vec![
            (
                ids.content_thinking,
                special_fragment(&ids, ids.content_thinking, ""),
            ),
            (7, "checked exact bytes".to_string()),
            (ids.end_message, special_fragment(&ids, ids.end_message, "")),
            (
                ids.message_model,
                special_fragment(&ids, ids.message_model, ""),
            ),
            (8, "exec".to_string()),
            (
                ids.content_invoke_tool_json,
                special_fragment(&ids, ids.content_invoke_tool_json, ""),
            ),
            (
                9,
                r#"{"name":"exec","args":{"command":"printf same-turn"}}"#.to_string(),
            ),
            (ids.end_message, special_fragment(&ids, ids.end_message, "")),
            (
                ids.content_model_end_sampling,
                special_fragment(&ids, ids.content_model_end_sampling, ""),
            ),
        ];
        let sequence = script(&owned);

        let model = ScriptedModel::new(ready, &sequence);
        let log = model.log();
        let mut mind = InklingMind::new_gate(Box::new(model), 32, Some("system".to_string()))
            .expect("valid READY evidence");
        assert_eq!(
            mind.reasoning_provenance(),
            drive::reason::ReasoningProvenance::Provider
        );

        let events = [drive::world::Event::monologue(1, "prior ")];
        let turn = mind.observe(drive::world::MergedView {
            events: &events,
            watermark: 3,
        });
        assert_eq!(turn.said, "$ printf same-turn\n");
        assert_eq!(turn.decision.disposition, drive::mind::Disposition::Fire);
        assert_eq!(turn.decision.command.as_deref(), Some("printf same-turn"));
        assert_eq!(
            turn.decision.command_span_origin,
            Some(drive::mind::CommandSpanOrigin::Projected)
        );
        assert_eq!(turn.decision.span, "$ printf same-turn\n");
        assert_eq!(turn.decision.span_start, "prior ".len() as u64);
        assert_eq!(
            turn.decision.span_end,
            ("prior ".len() + turn.said.len()) as u64
        );
        assert_eq!(
            turn.decision.reasoning.as_deref(),
            Some("checked exact bytes")
        );
        assert_eq!(turn.response_state, drive::mind::ResponseState::Complete);
        assert_eq!(
            mind.log()
                .expect("finite fixture retains turn evidence")
                .lock()
                .expect("turn log")[0]
                .token_ids
                .len(),
            9
        );

        drop(mind);
        // Exactly what the captured framed-stream input used to prove, frame by
        // frame: one typed initialization, then one one-token CONSULT per
        // microturn, then the logical-turn agreement — and the stream's final
        // `End(Complete)` is now the model's own clean shutdown on drop.
        let mut expected = vec![ModelCall::Context(InklingContext::Initialize {
            system: "system".to_string(),
            history: Vec::new(),
        })];
        expected.extend(sequence.iter().map(|_| ModelCall::Consult(1)));
        expected.push(ModelCall::Agree);
        expected.push(ModelCall::Shutdown);
        assert_eq!(calls(&log), expected);
    }

    /// Context admission is checked arithmetic, not a courtesy: a model that
    /// reports evidence disagreeing with the admission inequality is refused
    /// before anything is installed.
    ///
    /// This replaces `tensor_parallel_context_admission_requires_identical_evidence`,
    /// which compared two ranks' evidence through the deleted
    /// `matching_context_preflight`. With the model in this process there is no
    /// second rank's evidence to compare against — that check is gone, and the
    /// module header records what it cost — but the evidence is still measured
    /// against the definition of admission itself, which is the half that
    /// caught a wrong answer rather than merely a disagreeing one.
    #[test]
    fn preflight_evidence_that_is_not_self_consistent_is_refused() {
        use drive::mind::Mind as _;

        let mut model = ScriptedModel::new(fake_ready("scripted.pile"), &[]);
        let mut evidence = context_preflight(
            ContextPlacement::Append,
            7,
            1,
            5,
            3,
            model.ready.context_budget,
        )
        .unwrap();
        // The width the evidence CLAIMS no longer produces the extent it
        // reports — the same one-token divergence the deleted cross-rank test
        // used, now between a model and the arithmetic instead of between two
        // ranks.
        evidence.delta_tokens += 1;
        model.preflight_override = Some(evidence);
        let log = model.log();
        let mut mind = InklingMind::new(Box::new(model), 3, Some("system".to_string()))
            .expect("valid READY evidence");

        let error = mind
            .admit_observation(&[])
            .expect_err("evidence that disagrees with the admission arithmetic is refused");
        assert!(error.to_string().contains("self-consistent"), "{error:#}");
        assert_eq!(
            calls(&log),
            vec![ModelCall::Preflight(ContextPlacement::Append)],
            "a refused preflight installs nothing"
        );
    }
}
