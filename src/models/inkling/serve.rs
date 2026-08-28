//! **The serving protocol** — how a process that HOLDS a [`Session`] is talked
//! to, and the client half that talks to it.
//!
//! [`Session`]: super::session::Session
//!
//! # What this is for
//!
//! [`super::session`] made the model a value that survives across calls. It did
//! not make it a value another PROGRAM can reach: a `Session` lives in one
//! process's address space, and the process that wants it — `drive` — must not
//! link `mary`. Drive builds GPU-free in seconds and that is protected on
//! purpose; a cognition loop that took a CUDA toolchain to compile would be a
//! cognition loop nobody could run on a laptop.
//!
//! So the model runs in its own long-lived process (`inkling_serve`) and is
//! reached over a pipe. What goes over that pipe is not a new format: it is the
//! **framed-stream convention** (`framed_stream`, the crate `drive` factored out
//! for exactly this — "a convention only one program can link is not a
//! convention"). One framed stream each way, both directions carrying
//! `text/plain;charset=utf-8` counted in bytes.
//!
//! ```text
//!   drive ──framed text (context) ──▶ inkling_serve  ── holds a Session
//!         ◀─framed text (tokens) ───               ── one record PER TOKEN
//! ```
//!
//! # The control plane rides on the data plane
//!
//! The convention's first guarantee is that **every record carries its content
//! type**, and that "a heterogeneous stream — telemetry interleaved with
//! keyframes — needs no separate channel". That is used here rather than
//! worked around: a record whose content type is the stream's default is
//! CONTENT (raw probe context going in, a decoded fragment coming out), and a
//! record that overrides it is CONTROL. Six overrides exist, and there is no
//! second socket, no length-prefixed sidecar and no JSON-lines mode:
//!
//! | content type | direction | meaning |
//! |---|---|---|
//! | [`READY_TYPE`] | serve → client | the model is loaded; here is what it is |
//! | [`CONTEXT_TYPE`] | client → serve | insert typed TML context safely |
//! | [`CONSULT_TYPE`] | client → serve | the delta is complete — produce a turn |
//! | [`TURN_TYPE`] | serve → client | the turn is over; here is how it went |
//! | [`REINITIALIZE_TYPE`] | client → serve | replace one completed sequence with a complete initialization |
//! | [`REINITIALIZED_TYPE`] | serve → client | the replacement prefix is staged against reset warm weights |
//!
//! # The two clocks mean something here
//!
//! `index` counts records and `offset` counts BYTES of UTF-8, exactly as the
//! convention says for a text stream. On the return path that makes the offset
//! the number of bytes of the mind's own words so far, which is the same
//! coordinate space drive's monologue is addressed in — so a dropped token is a
//! continuity error at the reader rather than a sentence that silently reads
//! fine.
//!
//! And when the incremental detokenizer cannot honour that (see
//! [`ServeClient::consult`]'s note on resynchronisation), the producer declares
//! a **gap** naming what it could not deliver, which is the convention's third
//! guarantee and the only legal way to skip content.
//!
//! # One turn, end to end
//!
//! ```text
//!   client                                   serve
//!   ──────                                   ─────
//!   preamble ────────────────────────────▶
//!         ◀──────────────────────────────── preamble
//!         ◀──────────────────────────────── READY  {layers, partial, vocab}
//!   text  "…command output…"  ───────────▶   (tokenize, hold as the delta)
//!   text  "…more of it…"      ───────────▶
//!   CONSULT {max_tokens}      ───────────▶   extend(delta); step; step; …
//!         ◀──────────────────────────────── text  " The"     ← flushed AS PRODUCED
//!         ◀──────────────────────────────── text  " capital"
//!         ◀──────────────────────────────── TURN  {tokens, first_token_secs, …}
//! ```
//!
//! The token records are the whole point of the exercise: they are written and
//! flushed as each token is decoded, so a consumer can start speaking on the
//! first word of a sentence instead of the last. Batching them at the end of the
//! turn would be a legal framed stream and a useless one.
//!
//! # What is deliberately NOT here
//!
//! No HTTP and no sandbox. A serving process serves turns; the Drive adapter
//! owns the conversation lifecycle and executes nothing itself. The tokenizer
//! stays on the SERVE side: typed context crosses as JSON, free text is encoded
//! through its content-only view there, and generated exact ids return in TURN.
//!
//! # What this measured, and the framing rules that make the numbers evidence
//!
//! One GB10 (`spark2`, 121.63 GiB unified memory), idle at the time — 115 GiB
//! available, load average 0.06, no GPU compute apps — with the box's advisory
//! lock held on BOTH boxes. `work-inkling-complete.pile` (171 GB), the
//! checkpoint's own `tokenizer.json`, release build, features
//! `inkling-serve,drive-mind,cuda-backend,import`. Greedy, no `INK_*` switch
//! set beyond the layer range. Every duration below is measured INSIDE the
//! serving process around the `Session` calls.
//!
//! **Layers 0..21 of 42** — the shape the frontier benchmark's HEAD box runs,
//! 103.88 GiB admitted of 121.63 GiB:
//!
//! | | seconds | what it is |
//! |---|---|---|
//! | `Session::load` → READY | 35.3 | paid ONCE, per process |
//! | first token, turn 0 | 9.838 | 5-token prompt, cold session, every layer's first bind |
//! | first token, turn 1 | 0.579 | a 13-token DELTA, walked one position each — SUPERSEDED, see below |
//! | first token, turns 2–3 | 0.044, 0.045 | no delta: ONE step against a warm cache |
//! | decode | 0.709–0.716 / 16 tok | **44.3–44.8 ms PER STEP** |
//!
//! The framing rule on the middle rows is the one that is easy to lose: those
//! are seconds per FIRST TOKEN OF A TURN, not per token and not per turn.
//! **A no-delta turn's first token costs 0.044 s where turn 0's cost 9.838 s —
//! 221×** — and that ratio is what a held `Session` buys. Turn 1 shows the
//! delta's price honestly: 0.579 s is thirteen walked positions at ~44 ms, i.e.
//! exactly the per-step cost times the delta, and nothing for the 5 tokens
//! before them, which is the KV cache doing its job.
//!
//! The per-step figure cross-checks against the project's scoreboard: the
//! frontier measures **86.5 ms/step for the FULL 42-layer two-box pipeline**,
//! and half that stack on one box at ~44 ms/step is the consistent half.
//!
//! **Layers 0..4 of 42** — the cheap gate shape, 30.96 GiB admitted:
//! READY in 10.2–10.6 s, warm first token **0.015–0.016 s**, ~15.6 ms per step.
//!
//! Two cautions, both from watching the same shape twice:
//!
//!   - **Turn 0 is not reproducible; the warm turn is.** The same 0..4 run
//!     measured turn-0 first token at 7.823 s and then at 2.425 s, because the
//!     second run found the pile's pages already cached. The warm number was
//!     0.015 s and 0.016 s across those same two runs. So quote the warm number
//!     as a measurement and turn 0 as a range.
//!   - **A turn occasionally stalls for seconds.** One 24-token turn at 0..4
//!     took 4.270 s where its neighbour took 0.370 s, same delta, same shape.
//!     That is the intermittent multi-second decode stall
//!     [`super::stepstat`] exists to characterise; it is not introduced here and
//!     it is not explained here, but a serving process makes it USER-VISIBLE for
//!     the first time, because a conversation waits on it.
//!
//! And the protocol's own overhead, since it is the thing this file adds: the
//! serving process measured its first token at 0.015 s and the client saw that
//! token at 0.016 s — **~1 ms per turn for the pipe, the framing and the
//! detokenizer**, against a 44 ms step.
//!
//! ## What the CARRY changed, and why the turn-1 row above is superseded
//!
//! That row is two changes stale, and both moved the quantity it names rather
//! than the number: `Session::extend` now appends a delta in ONE BATCHED PASS
//! instead of walking it, and a turn's delta is now one token WIDER than the
//! client's, because the previous turn's last token rides at its head (see
//! [`TurnEnd::carried`]). "13 walked positions at ~44 ms" is no longer what turn
//! 1 does.
//!
//! Measured 2026-08-28, same box, same lock, same pile, same 0..21 range, the
//! same probe arguments byte for byte — an A/B of two `inkling_serve` binaries
//! that differ by the carry commit and nothing else:
//!
//! | | turn 0 | turn 1 | turns 2–3 |
//! |---|---|---|---|
//! | position, before | 20 | 48 | 64, 80 |
//! | position, after | 20 | **49** | 65, 81 |
//! | first token, before | 5.372 s | 1.248 s | 0.044 s |
//! | first token, after | 5.921 s | 0.505 s | 0.044 s |
//! | text | **byte-identical** | diverges | diverges |
//!
//! Read the position row first, because it is the whole mechanism: **+1 at turn
//! 1 and +1 thereafter, never +2.** One token per turn WITH NEW CONTEXT is what
//! was being lost, and turns 2–3 were never losing one — an empty delta reaches
//! `extend(&[])`, which shortcuts to `Session::step` and does feed the token
//! back. The defect was always exactly the turns that had something new to say.
//!
//! And the divergence begins where the mechanism says it must: turn 0 has no
//! previous turn to carry from, so it is byte-identical across the change, and
//! every turn after it differs. The seconds are one sample a cell and are not a
//! measurement of the carry — one row of a batched pass cannot be read out of a
//! turn that also pays a first-token latency — they are here to say the change
//! did not cost a decode step, which a `step()`-based fix would have.
//!
//! **Neither side's text is the model's.** Both are the degenerate output of a
//! PARTIAL STACK unembedding through layers it did not all run, which is
//! structural (see the TP section below), so this A/B shows WHERE the divergence
//! starts and cannot show whether the answer got better. The token-level
//! correctness claim lives in `inkling_session --carry`, which compares against a
//! session fed the identical sequence with the identical pass partition.
//!
//! # Tensor parallelism is above this, not inside it
//!
//! `Session::load` refuses `INK_TP`, so one `inkling_serve` is one RANK and a
//! single-box serving process necessarily runs a STRICT SUBRANGE of the stack
//! (`hi - lo < num_hidden_layers` is enforced: 144 GiB does not fit a 121 GiB
//! box). Its tokens are therefore DIAGNOSTIC, not the model's, and
//! [`Ready::partial`] says so on the wire rather than leaving it to be inferred
//! from fluent-looking wrong text.
//!
//! **So a single-box serving process can never produce the model's text, and
//! that is structural rather than a matter of effort.** The only route through a
//! `Session` to a real token is the TP pair, because the layer split
//! (`INK_PIPE`) is refused too and a `Session` that does not start at layer 0
//! has no embedding table. Saying that plainly is more useful than a serving
//! process that reads well and is wrong.
//!
//! ## The fan-out proxy boundary
//!
//! It speaks THIS protocol on both sides — two [`ServeClient`]s upstream, an
//! `inkling_serve`-shaped server downstream — so `drive` cannot tell the
//! difference and [`Ready::partial`] becomes `false` for the first time.
//! [`ServePair`] is that client-side fan-out; the rank process forms the Group
//! and hands it to `Session::load_with_group`. Five invariants define the seam:
//!
//! 1. **A TP Session accepts a group it did not form.** `Group::form`
//!    is a rendezvous (rank 0 `accept`s with no timeout, the others dial with a
//!    180 s deadline) and `set_external_comm` is process-global, so there is at
//!    most one TP session per process and a library call must not block on a
//!    peer booting elsewhere. The group is formed ABOVE and passed in.
//! 2. **The layer-range rule INVERTS.** Single-box requires
//!    `hi - lo < num_hidden_layers`; TP requires exactly `0:num_hidden_layers`,
//!    because each rank holds half of EVERY tensor rather than all of some
//!    layers. One of the two rules has to be selected by whether a group is
//!    present, and getting it backwards is a refusal at load, not a wrong
//!    answer — which is the good failure.
//! 3. **Lockstep is per PASS, not per turn.** Both ranks must make the same
//!    forward calls in the same order or the other blocks in NCCL forever, and
//!    a turn's passes depend on its delta: `extend` appends in chunks of
//!    `extend_batch` rows, and the pass is one row wider than the client's delta
//!    whenever a carried token rides at its head. So the proxy must feed both
//!    ranks the SAME context bytes and the same `max_tokens`, not merely the
//!    same turns.
//! 4. **Rank 1's stream must be DRAINED and CHECKED, not ignored.** Both ranks
//!    produce the same token (embedding and unembedding are replicated, so both
//!    unembed the whole table and take the same argmax). The proxy returns rank
//!    0's tokens — but it has to read rank 1's too, or its pipe fills and the
//!    rank blocks, and comparing them is the loudest available signal that the
//!    all-reduce has broken.
//! 5. **A dead rank must kill its peer.** If one rank dies mid-turn the other
//!    blocks in NCCL with no timeout. The convention already hands the proxy
//!    that signal for free — its fourth guarantee is that a truncated stream is
//!    distinguishable from a finished one — so the proxy has what it needs to
//!    act; it just has to act.

use anyhow::{Context, Result};

/// The stream's own content type, both ways: the mind's words are text.
pub const CONTENT_TYPE: &str = framed_stream::TEXT_PLAIN;
/// The unit both offsets are counted in: bytes of UTF-8.
pub const UNIT: &str = framed_stream::UNIT_BYTES;

/// Control record, serve → client, once: the model is loaded and ready.
pub const READY_TYPE: &str = "application/vnd.mary.inkling-ready+json";
/// Control record, client → serve: the delta is complete, produce a turn.
pub const CONSULT_TYPE: &str = "application/vnd.mary.inkling-consult+json";
/// Control record, client → serve: typed context to insert into the model's
/// TML conversation. Free text remains JSON data on the wire and is encoded by
/// the serving process's content-only tokenizer; it is never scanned for TML
/// marker spellings.
pub const CONTEXT_TYPE: &str = "application/vnd.mary.inkling-context+json";
/// Control record, serve → client: the turn is over, and here is how it went.
pub const TURN_TYPE: &str = "application/vnd.mary.inkling-turn+json";
/// Control record, client → serve: atomically replace a completed sequence
/// with a complete [`InklingContext::Initialize`] prefix.
pub const REINITIALIZE_TYPE: &str = "application/vnd.mary.inkling-reinitialize+json";
/// Control record, serve → client: the warm Session was reset and the
/// replacement initialization is staged for its first CONSULT.
pub const REINITIALIZED_TYPE: &str = "application/vnd.mary.inkling-reinitialized+json";

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

/// Runtime ids of the TML tokens this protocol understands.
///
/// No numeric vocabulary constants live in the adapter. The serving process
/// resolves every field by token spelling from the exact tokenizer it loaded,
/// then announces the result in READY so the GPU-free client can parse ids
/// without owning or reconstructing that tokenizer.
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
    /// this minimal protocol does not support. Generated unknown specials are
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
    /// native call on the client.
    pub command: String,
    /// Drive's deliberate text projection of its typed result, including any
    /// structural status annotation that adds information.
    pub content: String,
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
        historical_exec_results: Vec<ExecResultContext>,
    },
    /// Result of the native call already present in the retained KV sequence.
    ToolResult { result: ExecResultContext },
    /// A result whose assistant call predates this live `InklingMind` (for
    /// example a Drive memory-cover pair). Both sides are inserted.
    HistoricalExecResult { result: ExecResultContext },
    /// Start another autonomous assistant response after a completed text-only
    /// response. A tool result already carries this prompt itself.
    GenerationPrompt,
}

/// Serve → client: one completed sequence has been replaced in the same
/// warm serving process.
///
/// The replacement initialization has been tokenized and staged, but has not
/// yet been attended to: the next CONSULT performs the fresh Session prefill.
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

    /// Runtime ids announced in READY and used by the client parser.
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
            InklingContext::Initialize {
                system,
                historical_exec_results,
            } => {
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

                for result in historical_exec_results {
                    self.push_historical_result(&mut ids, result)?;
                }
                ids.push(self.special_ids.message_model as usize);
            }
            InklingContext::ToolResult { result } => {
                self.push_tool_result(&mut ids, result)?;
                ids.push(self.special_ids.message_model as usize);
            }
            InklingContext::HistoricalExecResult { result } => {
                self.push_historical_result(&mut ids, result)?;
                ids.push(self.special_ids.message_model as usize);
            }
            InklingContext::GenerationPrompt => {
                ids.push(self.special_ids.message_model as usize);
            }
        }
        Ok(ids)
    }

    fn push_historical_result(
        &self,
        ids: &mut Vec<usize>,
        result: &ExecResultContext,
    ) -> Result<()> {
        ids.push(self.special_ids.message_model as usize);
        self.push_content(ids, "exec")?;
        ids.push(self.special_ids.content_invoke_tool_json as usize);
        self.push_content(ids, &canonical_exec_call_json(&result.command))?;
        ids.push(self.special_ids.end_message as usize);
        ids.push(self.special_ids.content_model_end_sampling as usize);
        self.push_tool_result(ids, result)
    }

    fn push_tool_result(&self, ids: &mut Vec<usize>, result: &ExecResultContext) -> Result<()> {
        ids.push(self.special_ids.message_tool as usize);
        self.push_content(ids, "exec")?;
        ids.push(self.special_ids.content_text as usize);
        self.push_content(ids, &result.content)?;
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

    /// Finish as the protocol's uppercase 64-hex-digit identity.
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

/// What the serving process is, announced once when the weights are up.
///
/// Sent AFTER the load rather than in the preamble, because the load is minutes
/// and the preamble is the handshake: a client that got this in the preamble
/// could not tell "starting" from "ready", which is the one thing it needs to
/// know before it asks for a turn.
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
    /// Effective TP world (one for a non-TP serving process).
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
    /// Wall-clock seconds `Session::load` took. The number a serving process
    /// exists to pay ONCE.
    pub load_secs: f64,
}

fn default_tp_world() -> usize {
    1
}

/// Client → serve: stop accumulating context and produce a turn.
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

/// Serve → client: the turn is over, and this is how it went.
///
/// Every duration here is measured INSIDE the serving process, around the
/// `Session` calls, so it is the model's time and not the pipe's. See
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
    /// A tensor-parallel pair compares these before it accepts a turn. Text is
    /// also compared fragment by fragment for streaming, but two different
    /// byte-level tokens can decode to the same text; ids are the unambiguous
    /// agreement signal. A paired proxy forwards the already-arbitrated ids so
    /// an end-to-end gate can compare continuations across session boundaries.
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
    /// counts it, so a reader can see the whole pass rather than the client's
    /// half of it. A turn after turn 0 reporting `0` here is a turn whose model
    /// never heard its own last word.
    pub carried: usize,
    /// Why generation stopped: `"max_tokens"` or `"stop_token"`.
    pub stopped: String,
    /// Seconds to the FIRST token of this turn, `Session` calls only:
    /// `extend`/`prefill` over the already-tokenised ids and one forward.
    /// Tokenisation and pipe transit are deliberately outside this duration;
    /// clients may measure their own wall clock around [`ServeClient::consult`]
    /// when those costs matter. On turn 0 this is the prompt's prefill; on every
    /// turn after it, it is what the KV cache saves. THE framing rule: seconds
    /// per FIRST TOKEN OF A TURN, over the layer range in [`Ready::layers`] —
    /// not per token and not per turn. A single server reports its rank-local
    /// wall clock; a paired proxy reports the slower rank, the distributed
    /// critical path.
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
    /// for the next generation prompt. Calling this on a sliced/incomplete
    /// response is a protocol error; callers instead retain the parser.
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
/// session monologue extent for `Decision::fire`.
#[cfg(any(feature = "drive-mind", test))]
fn project_native_exec(said: &mut String, command: &str) -> std::ops::Range<usize> {
    if !said.is_empty() && !said.ends_with('\n') {
        said.push('\n');
    }
    let start = said.len();
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
#[cfg(any(feature = "drive-mind", test))]
fn one_token_association(end: &TurnEnd, fragments: String) -> Result<(u32, String)> {
    anyhow::ensure!(
        end.tokens == 1 && end.token_ids.len() == 1,
        "one-token consult returned {} token(s) and {} exact id(s)",
        end.tokens,
        end.token_ids.len()
    );
    Ok((end.token_ids[0], fragments))
}

// ── the client half ─────────────────────────────────────────────────────────

/// A running `inkling_serve`, with a framed stream in each direction.
///
/// Deliberately synchronous and strictly alternating: the client writes context
/// and one consult record, then reads until the turn ends. Neither side ever
/// writes while the other is writing, so the two-pipe deadlock `drive::stream`
/// spawns a thread to avoid cannot arise here and no thread is spawned.
pub struct ServeClient {
    child: ChildHandle,
    writer: Option<framed_stream::FramedWriter<std::process::ChildStdin>>,
    reader: framed_stream::FramedReader<std::process::ChildStdout>,
    ready: Ready,
    label: String,
}

#[derive(Clone)]
struct ChildHandle {
    child: std::sync::Arc<std::sync::Mutex<std::process::Child>>,
    label: std::sync::Arc<str>,
}

impl ChildHandle {
    fn new(child: std::process::Child, label: String) -> Self {
        Self {
            child: std::sync::Arc::new(std::sync::Mutex::new(child)),
            label: label.into(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, std::process::Child> {
        // A panic in diagnostic code must not make the OS process unkillable.
        // The Child itself remains the kernel's handle even if this mutex was
        // poisoned, so recover ownership and continue teardown.
        self.child
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn kill(&self) -> Result<()> {
        let mut child = self.lock();
        if child.try_wait()?.is_none() {
            child
                .kill()
                .with_context(|| format!("kill {}", self.label))?;
        }
        Ok(())
    }

    fn wait(&self) -> Result<std::process::ExitStatus> {
        self.lock()
            .wait()
            .with_context(|| format!("wait for {} to exit", self.label))
    }

    fn try_wait(&self) -> Result<Option<std::process::ExitStatus>> {
        self.lock()
            .try_wait()
            .with_context(|| format!("poll {} for exit", self.label))
    }
}

/// A child whose pipes exist but whose protocol has not been opened yet.
///
/// Pair startup deliberately separates `spawn(2)` from protocol/READY waits:
/// both rank processes therefore exist before either can block in its Group
/// rendezvous.
struct LaunchedServe {
    child: ChildHandle,
    stdin: Option<std::process::ChildStdin>,
    stdout: Option<std::process::ChildStdout>,
    label: String,
    kill_on_drop: bool,
}

impl LaunchedServe {
    fn launch(command: &mut std::process::Command) -> Result<Self> {
        let label = command.get_program().to_string_lossy().to_string();
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt as _;

            let expected_parent = std::process::id() as libc::pid_t;
            // SAFETY: only async-signal-safe libc calls run between fork and
            // exec. The parent check closes the fork-to-prctl race.
            unsafe {
                command.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::getppid() != expected_parent {
                        return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                    }
                    Ok(())
                });
            }
        }
        let mut child = command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            // Inherited on purpose: diagnostics must not share the protocol fd.
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .with_context(|| format!("start the serving process {label}"))?;
        let stdin = child.stdin.take().context("serving process has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("serving process has no stdout")?;
        Ok(Self {
            child: ChildHandle::new(child, label.clone()),
            stdin: Some(stdin),
            stdout: Some(stdout),
            label,
            kill_on_drop: true,
        })
    }

    fn open(mut self) -> Result<StartingServe> {
        // Open the WRITER first on both sides: each preamble is written before
        // either side reads, so neither blocks waiting for the other's.
        let writer = framed_stream::FramedWriter::open(
            self.stdin.take().context("serving process has no stdin")?,
            CONTENT_TYPE,
            UNIT,
        )
        .context("write the serving process's input preamble")?;
        let reader = framed_stream::FramedReader::open(
            self.stdout
                .take()
                .context("serving process has no stdout")?,
        )
        .context("read the serving process's output preamble")?;
        reader.require_content_type(CONTENT_TYPE)?;
        self.kill_on_drop = false;
        Ok(StartingServe {
            child: self.child.clone(),
            writer: Some(writer),
            reader: Some(reader),
            label: self.label.clone(),
            kill_on_drop: true,
        })
    }
}

impl Drop for LaunchedServe {
    fn drop(&mut self) {
        if self.kill_on_drop {
            // Keep the transport alive long enough to carry EOF to a remote
            // supervisor. Killing `ssh` before closing its pipes can orphan
            // the rank that the supervisor was meant to own.
            drop(self.stdin.take());
            drop(self.stdout.take());
            reap_child_after_channel_close(&self.child);
        }
    }
}

/// An open serving protocol that has not announced READY yet.
struct StartingServe {
    child: ChildHandle,
    writer: Option<framed_stream::FramedWriter<std::process::ChildStdin>>,
    reader: Option<framed_stream::FramedReader<std::process::ChildStdout>>,
    label: String,
    kill_on_drop: bool,
}

impl StartingServe {
    fn wait_ready(mut self) -> Result<ServeClient> {
        let ready = match self
            .reader
            .as_mut()
            .context("serving process has no output stream")?
            .next_frame()
            .context("wait for the model to load")?
        {
            framed_stream::Frame::Record(record) if record.content_type() == READY_TYPE => {
                serde_json::from_slice::<Ready>(&record.payload)
                    .context("parse the serving process's READY record")?
            }
            framed_stream::Frame::Record(record) => anyhow::bail!(
                "the serving process's first record is {}, not a READY record",
                record.content_type()
            ),
            framed_stream::Frame::Gap(gap) => {
                anyhow::bail!(
                    "the serving process declared a gap before READY: {}",
                    gap.reason
                )
            }
            framed_stream::Frame::End(status) => {
                anyhow::bail!("the serving process ended before READY: {status:?}")
            }
        };
        self.kill_on_drop = false;
        Ok(ServeClient {
            child: self.child.clone(),
            writer: self.writer.take(),
            reader: self
                .reader
                .take()
                .context("serving process has no output stream")?,
            ready,
            label: self.label.clone(),
        })
    }
}

impl Drop for StartingServe {
    fn drop(&mut self) {
        if self.kill_on_drop {
            // `FramedWriter::drop` emits an honest ABORTED end before closing
            // the pipe. Let a remote supervisor observe that channel closure
            // before forcefully terminating the local transport.
            drop(self.writer.take());
            drop(self.reader.take());
            reap_child_after_channel_close(&self.child);
        }
    }
}

impl ServeClient {
    /// How long a normal close waits for the serving process to honour END.
    pub const DEFAULT_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

    /// Start `command` and wait for it to say it is READY.
    ///
    /// This blocks for the whole model load — minutes on a real range — because
    /// there is nothing useful a caller can do with a half-loaded model and a
    /// client that returned early would only move the wait to the first turn,
    /// where it would look like a slow turn instead of a slow start.
    pub fn spawn(command: &mut std::process::Command) -> Result<Self> {
        LaunchedServe::launch(command)?.open()?.wait_ready()
    }

    /// What loaded, and whether its tokens are the model's.
    pub fn ready(&self) -> &Ready {
        &self.ready
    }

    /// Add text to the turn's DELTA: new context the session has not attended
    /// to. Never a re-rendered transcript — the KV cache is still holding
    /// everything before it.
    pub fn feed(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.writer()?
            .text(text)
            .with_context(|| format!("feed context to {}", self.label))
    }

    /// Insert typed TML context. Free text is JSON data here; only the serving
    /// process owns the tokenizer and turns this record into structural ids.
    pub fn context(&mut self, context: &InklingContext) -> Result<()> {
        let payload = serde_json::to_vec(context).context("encode typed Inkling context")?;
        let extent = payload.len() as u64;
        self.writer()?
            .record_as(CONTEXT_TYPE, &payload, extent)
            .with_context(|| format!("feed typed context to {}", self.label))
    }

    /// Replace one completed sequence while keeping the serving process and its
    /// warm weights alive.
    ///
    /// `initialization` must be [`InklingContext::Initialize`], not an
    /// incremental context record. The server accepts it only after a completed
    /// turn and before any context for the next turn has been queued. It first
    /// tokenizes and validates the complete replacement, then resets Session,
    /// tokenizer stream and carried-token state together. The acknowledgement
    /// therefore means the next CONSULT is a fresh prefill; an error before it
    /// leaves the old sequence untouched.
    ///
    /// This is deliberately not an automatic Drive policy. The caller above
    /// the wire owns the durable cognition pile and must construct the fresh
    /// cover represented by this initialization. In particular, a Drive
    /// rollover must rebuild the cover with its fail-loud `build_cover`, take
    /// `Cover::end_key` as the end of *contiguous* recalled coverage, and replay
    /// every live turn after that boundary. A later isolated memory does not
    /// make an earlier uncovered turn safe to discard.
    ///
    /// The serving process cannot see whether Drive still has a sliced response
    /// or tool execution in flight. Its completed-turn check is only the narrow
    /// mechanical boundary; the foreground runner owns that semantic boundary.
    pub fn reinitialize(&mut self, initialization: &InklingContext) -> Result<Reinitialized> {
        anyhow::ensure!(
            matches!(initialization, InklingContext::Initialize { .. }),
            "a serving-process reinitialization requires one complete Initialize payload"
        );
        let payload =
            serde_json::to_vec(initialization).context("encode Inkling reinitialization")?;
        let extent = payload.len() as u64;
        self.writer()?
            .record_as(REINITIALIZE_TYPE, &payload, extent)
            .with_context(|| format!("reinitialize {}", self.label))?;
        match self
            .reader
            .next_frame()
            .with_context(|| format!("wait for {} to reinitialize", self.label))?
        {
            framed_stream::Frame::Record(record) if record.content_type() == REINITIALIZED_TYPE => {
                serde_json::from_slice::<Reinitialized>(&record.payload)
                    .context("parse the serving process's REINITIALIZED record")
            }
            framed_stream::Frame::Record(record) => anyhow::bail!(
                "the serving process sent a {} record while reinitializing",
                record.content_type()
            ),
            framed_stream::Frame::Gap(gap) => anyhow::bail!(
                "the serving process could not acknowledge reinitialization: {} byte(s): {}",
                gap.extent,
                gap.reason
            ),
            framed_stream::Frame::End(status) => {
                anyhow::bail!("the serving process ended while reinitializing: {status:?}")
            }
        }
    }

    /// Ask for a turn, calling `on_token` with each token's text AS IT ARRIVES.
    ///
    /// The callback is the streaming seam: it runs while the serving process is
    /// still generating, which is what lets a consumer start speaking on the
    /// first word. A callback that blocks blocks the model, and that is the
    /// honest coupling — backpressure on a pipe means the consumer is slower
    /// than the mind, which is a fact rather than a hang.
    ///
    /// # Resynchronisation
    ///
    /// A byte-level BPE token can be a partial UTF-8 sequence. The serving
    /// process therefore keeps one tokenizer `DecodeStream` across the logical
    /// sequence: newly inserted delta ids advance that decoder but their text is
    /// discarded, while generated ids produce only newly completed text
    /// fragments. It never re-steps the carry and never rewrites bytes already
    /// emitted. A producer-declared gap is still surfaced as an error carrying
    /// its reason, because a consumer that has already SPOKEN bytes cannot
    /// silently repair them.
    pub fn consult(
        &mut self,
        request: &Consult,
        mut on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<TurnEnd> {
        let payload = serde_json::to_vec(request).context("encode the consult record")?;
        let extent = payload.len() as u64;
        self.writer()?
            .record_as(CONSULT_TYPE, &payload, extent)
            .with_context(|| format!("ask {} for a turn", self.label))?;
        loop {
            match self.reader.next_frame().with_context(|| {
                format!(
                    "read {}'s turn (the serving process may have died)",
                    self.label
                )
            })? {
                framed_stream::Frame::Record(record) if record.content_type() == TURN_TYPE => {
                    return serde_json::from_slice::<TurnEnd>(&record.payload)
                        .context("parse the serving process's TURN record");
                }
                framed_stream::Frame::Record(record) if record.content_type() == CONTENT_TYPE => {
                    on_token(record.text()?)?;
                }
                framed_stream::Frame::Record(record) => anyhow::bail!(
                    "the serving process sent a {} record mid-turn, which this client does \
                     not understand",
                    record.content_type()
                ),
                framed_stream::Frame::Gap(gap) => anyhow::bail!(
                    "the serving process could not deliver {} byte(s) of this turn: {}",
                    gap.extent,
                    gap.reason
                ),
                framed_stream::Frame::End(status) => {
                    anyhow::bail!("the serving process ended mid-turn: {status:?}")
                }
            }
        }
    }

    fn writer(&mut self) -> Result<&mut framed_stream::FramedWriter<std::process::ChildStdin>> {
        self.writer
            .as_mut()
            .context("the serving process's input stream is already ended")
    }

    /// End the input stream as COMPLETE — the conversation is over.
    ///
    /// Ending the stream is what tells the serving process to finish: it
    /// terminates its own output stream and exits. Idempotent, because it is
    /// called both by [`ServeClient::close`] and from a `Drop` that cannot know
    /// whether close already ran. Complete rather than aborted matters: a
    /// writer dropped without this writes `END{aborted}`, and a run that
    /// finished did not give up.
    pub fn end_input(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            let sink = writer
                .finish(framed_stream::EndStatus::Complete)
                .context("end the serving process's input stream")?;
            drop(sink);
        }
        Ok(())
    }

    /// End the input stream and wait for the serving process to exit.
    ///
    /// The wait is not politeness. The serving process holds tens of gibibytes
    /// of arena and the kernel takes a while to hand it back; a caller that
    /// starts the next one before this one is gone is how a unified-memory box
    /// OOM-kills a run that did nothing wrong.
    pub fn close(mut self) -> Result<std::process::ExitStatus> {
        self.shutdown_with_timeout(Self::DEFAULT_SHUTDOWN_TIMEOUT)
    }

    /// [`ServeClient::close`] with an explicit bound on the clean END wait.
    pub fn close_with_timeout(
        mut self,
        shutdown_timeout: std::time::Duration,
    ) -> Result<std::process::ExitStatus> {
        self.shutdown_with_timeout(shutdown_timeout)
    }

    /// End input and reap the serving process without moving this client.
    ///
    /// This is a terminal operation even though it borrows `self`: after END,
    /// the protocol cannot carry another turn. The non-consuming shape lets an
    /// owner call the exact same teardown from `Drop` without wrapping the
    /// client in an `Option` merely to move it out.
    ///
    /// A cooperative child gets `shutdown_timeout` to exit. If END cannot be
    /// written or that deadline expires, the child is killed and gets one
    /// finite reap grace. A kernel that still cannot report its exit does not
    /// make the caller unbounded: a detached thread retains the child handle
    /// and performs the eventual blocking reap.
    pub fn shutdown_with_timeout(
        &mut self,
        shutdown_timeout: std::time::Duration,
    ) -> Result<std::process::ExitStatus> {
        anyhow::ensure!(
            !shutdown_timeout.is_zero(),
            "the serving process shutdown timeout must be nonzero"
        );
        std::time::Instant::now()
            .checked_add(shutdown_timeout)
            .context("the serving process shutdown timeout is too large")?;

        let primary = match self.end_input() {
            Err(error) => error,
            Ok(()) => match wait_child_with_timeout(&self.child, shutdown_timeout) {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => anyhow::anyhow!(
                    "{} did not exit within {:?} after END",
                    self.label,
                    shutdown_timeout
                ),
                Err(error) => error.context(format!(
                    "could not wait for {} after ending its input",
                    self.label
                )),
            },
        };
        let mut teardown_errors = Vec::new();
        if let Err(error) = self.kill() {
            teardown_errors.push(format!("could not kill {}: {error:#}", self.label));
        }
        match wait_child_with_timeout(&self.child, SINGLE_REAP_GRACE) {
            Ok(Some(_)) => {}
            Ok(None) => {
                if let Err(error) = detach_child_reaper(self.child.clone()) {
                    teardown_errors.push(format!(
                        "could not start the detached reaper for {}: {error}",
                        self.label
                    ));
                }
            }
            Err(error) => {
                teardown_errors.push(format!(
                    "could not poll {} after kill: {error:#}",
                    self.label
                ));
                if let Err(error) = detach_child_reaper(self.child.clone()) {
                    teardown_errors.push(format!(
                        "could not start the detached reaper for {}: {error}",
                        self.label
                    ));
                }
            }
        }
        if teardown_errors.is_empty() {
            Err(primary)
        } else {
            Err(primary.context(format!(
                "forced teardown also reported: {}",
                teardown_errors.join("; ")
            )))
        }
    }

    /// Kill the serving process. Its output stream reads as truncated, which is
    /// the honest report: it was killed, it did not finish.
    pub fn kill(&mut self) -> Result<()> {
        // Once killed, this protocol cannot accept a later COMPLETE end. Drop
        // the writer now so `end_input` remains idempotent and teardown does
        // not make a second, misleading write to a dead process.
        let writer = self.writer.take();
        let result = self.child.kill();
        drop(writer);
        result
    }
}

/// The post-kill grace is deliberately finite. Ten seconds matches the pair
/// teardown's existing bound while leaving enough room for a CUDA process to
/// release a large unified-memory arena after SIGKILL.
const SINGLE_REAP_GRACE: std::time::Duration = std::time::Duration::from_secs(10);

fn wait_child_with_timeout(
    child: &ChildHandle,
    timeout: std::time::Duration,
) -> Result<Option<std::process::ExitStatus>> {
    anyhow::ensure!(!timeout.is_zero(), "the child wait timeout must be nonzero");
    let deadline = std::time::Instant::now()
        .checked_add(timeout)
        .context("the child wait timeout is too large")?;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            return Ok(None);
        }
        std::thread::sleep(
            deadline
                .duration_since(now)
                .min(std::time::Duration::from_millis(5)),
        );
    }
}

fn detach_child_reaper(child: ChildHandle) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("inkling-serve-reaper".to_string())
        .spawn(move || {
            let _ = child.wait();
        })
        .map(drop)
}

// ── a full-stack tensor-parallel pair ──────────────────────────────────────

/// One rank process, expressed as argv and environment rather than as a shell
/// command string.
///
/// A remote rank is launched through a small `inkling_serve_pair __supervise`
/// process on the remote host. OpenSSH ultimately hands that argv to a login
/// shell, but callers supply the host, supervisor, rank program, arguments, and
/// environment as separate values. This type performs the unavoidable quoting
/// in one audited place; no caller constructs an injection-prone shell blob.
#[derive(Debug, Clone)]
pub struct RankCommand {
    host: Option<String>,
    ssh_program: std::ffi::OsString,
    remote_supervisor: std::ffi::OsString,
    remote_shutdown_timeout: std::time::Duration,
    program: std::ffi::OsString,
    args: Vec<std::ffi::OsString>,
    env: Vec<(std::ffi::OsString, std::ffi::OsString)>,
}

impl RankCommand {
    /// A rank launched on this host.
    pub fn local(program: impl Into<std::ffi::OsString>) -> Self {
        Self {
            host: None,
            ssh_program: "ssh".into(),
            remote_supervisor: "inkling_serve_pair".into(),
            remote_shutdown_timeout: std::time::Duration::from_secs(45),
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
        }
    }

    /// A rank launched through `ssh HOST`.
    pub fn ssh(host: impl Into<String>, program: impl Into<std::ffi::OsString>) -> Self {
        Self {
            host: Some(host.into()),
            ..Self::local(program)
        }
    }

    /// Override the local OpenSSH executable used for a remote rank.
    pub fn ssh_program(mut self, program: impl Into<std::ffi::OsString>) -> Self {
        self.ssh_program = program.into();
        self
    }

    /// Override the `inkling_serve_pair` executable used to supervise a remote
    /// rank. The supervisor makes loss of the SSH channel a bounded teardown
    /// event instead of leaving an NCCL process alive on the remote host.
    pub fn remote_supervisor(mut self, program: impl Into<std::ffi::OsString>) -> Self {
        self.remote_supervisor = program.into();
        self
    }

    /// Bound remote cleanup after the SSH channel closes. Keep this below the
    /// outer pair shutdown deadline so the supervisor can report completion.
    pub fn remote_shutdown_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.remote_shutdown_timeout = timeout;
        self
    }

    /// Append one rank-process argument. Repetition preserves argv order.
    pub fn arg(mut self, argument: impl Into<std::ffi::OsString>) -> Self {
        self.args.push(argument.into());
        self
    }

    /// Set one rank-process environment variable.
    pub fn env(
        mut self,
        key: impl Into<std::ffi::OsString>,
        value: impl Into<std::ffi::OsString>,
    ) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    fn command(&self) -> Result<std::process::Command> {
        anyhow::ensure!(!self.program.is_empty(), "a rank program may not be empty");
        match &self.host {
            None => {
                let mut command = std::process::Command::new(&self.program);
                command.args(&self.args).envs(self.env.iter().cloned());
                Ok(command)
            }
            Some(host) => {
                validate_ssh_host(host)?;
                anyhow::ensure!(
                    !self.remote_shutdown_timeout.is_zero(),
                    "the remote supervisor shutdown timeout must be nonzero"
                );
                let supervisor = remote_word(&self.remote_supervisor, "rank supervisor")?;
                let program = remote_word(&self.program, "rank program")?;
                let mut words = Vec::with_capacity(7 + self.env.len() * 2 + self.args.len() * 2);
                words.push("exec".to_string());
                words.push(supervisor.to_string());
                words.push("__supervise".to_string());
                words.push("--shutdown-timeout-secs".to_string());
                words.push(self.remote_shutdown_timeout.as_secs().max(1).to_string());
                words.push("--program".to_string());
                words.push(program.to_string());
                for (key, value) in &self.env {
                    let key = remote_word(key, "environment name")?;
                    anyhow::ensure!(
                        valid_env_name(key),
                        "invalid remote environment name {key:?}"
                    );
                    let value = remote_word(value, "environment value")?;
                    words.push("--env".to_string());
                    words.push(format!("{key}={value}"));
                }
                for argument in &self.args {
                    words.push("--arg".to_string());
                    words.push(remote_word(argument, "rank argument")?.to_string());
                }
                let remote = words
                    .iter()
                    .map(|word| shell_quote(word))
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut command = std::process::Command::new(&self.ssh_program);
                command
                    // A user-level RequestTTY setting must never put terminal
                    // line discipline between two binary framed streams.
                    .arg("-T")
                    .arg("-o")
                    .arg("BatchMode=yes")
                    .arg("--")
                    .arg(host)
                    .arg(remote);
                Ok(command)
            }
        }
    }
}

fn validate_ssh_host(host: &str) -> Result<()> {
    anyhow::ensure!(!host.is_empty(), "an ssh host may not be empty");
    anyhow::ensure!(!host.starts_with('-'), "an ssh host may not start with '-'");
    anyhow::ensure!(
        !host.chars().any(char::is_whitespace) && !host.chars().any(char::is_control),
        "an ssh host may not contain whitespace or control characters"
    );
    Ok(())
}

fn remote_word<'a>(value: &'a std::ffi::OsStr, what: &str) -> Result<&'a str> {
    value
        .to_str()
        .with_context(|| format!("a remote {what} must be valid UTF-8"))
}

fn valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some('_' | 'a'..='z' | 'A'..='Z'))
        && chars.all(|c| matches!(c, '_' | 'a'..='z' | 'A'..='Z' | '0'..='9'))
}

fn shell_quote(word: &str) -> String {
    if word.is_empty() {
        return "''".to_string();
    }
    format!("'{}'", word.replace('\'', "'\"'\"'"))
}

/// Why a paired consult failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServePairFailure {
    /// The two complete rank streams disagreed.
    Divergence,
    /// A rank process or its framed stream failed.
    Rank,
    /// The downstream streaming callback failed.
    Consumer,
}

/// A paired-turn failure, including the exact amount of already-confirmed text.
///
/// That extent is diagnostic, not a candidate `Gap`: framed-stream gaps move
/// forward over missing extent and cannot retract bytes. Equal fragments remain
/// truthful partial speech; divergence or rank death then aborts the stream and
/// must not be rewritten into a model-level result.
#[derive(Debug)]
pub struct ServePairError {
    kind: ServePairFailure,
    confirmed_extent: u64,
    message: String,
}

impl ServePairError {
    fn new(kind: ServePairFailure, confirmed_extent: u64, message: impl Into<String>) -> Self {
        Self {
            kind,
            confirmed_extent,
            message: message.into(),
        }
    }

    /// Failure class, for faithful downstream framed-stream handling.
    pub fn kind(&self) -> ServePairFailure {
        self.kind
    }

    /// Bytes already checked equal and handed to the downstream consumer.
    pub fn confirmed_extent(&self) -> u64 {
        self.confirmed_extent
    }
}

impl std::fmt::Display for ServePairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ServePairError {}

/// Two `inkling_serve` ranks acting as one full-stack serving process.
///
/// The pair is intentionally above `Session`: it mirrors one input stream,
/// drains both output streams concurrently, and only releases a rank-0 text
/// fragment after the corresponding rank-1 fragment agrees. There is no second
/// scheduling policy here; the ranks still execute the serving protocol's one
/// strictly alternating turn at a time.
pub struct ServePair {
    rank0: ServeClient,
    rank1: ServeClient,
    ready: Ready,
    terminated: bool,
}

/// Own both rank transports from the moment the second spawn succeeds until
/// they become a [`ServePair`].
///
/// READY [`ServeClient`]s live inside the guard, whose destructor explicitly
/// closes their framed channels while SSH is still alive. That gives each
/// remote supervisor an authoritative EOF on every `?` or panic. Only then
/// does the guard wait, force-kill if necessary, and reap. Successful
/// construction takes both clients and explicitly disarms the guard when
/// ownership moves into `ServePair`.
struct PairStartupGuard {
    children: [ChildHandle; 2],
    clients: [Option<ServeClient>; 2],
    armed: bool,
}

impl PairStartupGuard {
    fn new(children: [ChildHandle; 2]) -> Self {
        Self {
            children,
            clients: [None, None],
            armed: true,
        }
    }

    fn ready(&mut self, rank: usize, client: ServeClient) {
        self.clients[rank] = Some(client);
    }

    fn take(&mut self, rank: usize) -> Option<ServeClient> {
        self.clients[rank].take()
    }

    /// Break a READY wait that cannot finish. Reaping remains the guard's one
    /// responsibility after the startup workers and their pipes have dropped.
    fn abort(&self) {
        kill_children(&self.children);
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PairStartupGuard {
    fn drop(&mut self) {
        if self.armed {
            for client in &mut self.clients {
                drop(client.take());
            }
            reap_children_after_channel_close(&self.children);
        }
    }
}

impl ServePair {
    /// Long enough for the measured multi-minute model load, finite enough that
    /// a live-but-stuck rendezvous cannot occupy both boxes forever.
    pub const DEFAULT_STARTUP_TIMEOUT: std::time::Duration =
        std::time::Duration::from_secs(15 * 60);

    /// Bound the clean END handshake as well as process teardown. A rank that
    /// remains live but stuck in a collective must not hold its peer forever.
    pub const DEFAULT_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

    /// Launch both ranks, wait for both READY records concurrently, and require
    /// compatible full-stack sessions.
    pub fn spawn(commands: [RankCommand; 2]) -> Result<Self> {
        Self::spawn_with_timeout(commands, Self::DEFAULT_STARTUP_TIMEOUT)
    }

    /// [`ServePair::spawn`] with an explicit deadline for both READY records.
    pub fn spawn_with_timeout(
        commands: [RankCommand; 2],
        startup_timeout: std::time::Duration,
    ) -> Result<Self> {
        anyhow::ensure!(
            !startup_timeout.is_zero(),
            "the pair startup timeout must be nonzero"
        );
        let deadline = std::time::Instant::now()
            .checked_add(startup_timeout)
            .context("the pair startup timeout is too large")?;
        let [rank0, rank1] = commands;
        let (launched0, launched1) = concurrently(
            move || {
                let mut command = rank0.command()?;
                LaunchedServe::launch(&mut command)
            },
            move || {
                let mut command = rank1.command()?;
                LaunchedServe::launch(&mut command)
            },
        )?;
        let (launched0, launched1) = match (launched0, launched1) {
            (Ok(rank0), Ok(rank1)) => (rank0, rank1),
            (rank0, rank1) => {
                // Any successfully launched side is still a `LaunchedServe`;
                // consuming these Results below invokes its channel-first
                // teardown rather than duplicating ownership here.
                let error0 = rank0.err().map(|e| format!("rank 0: {e:#}"));
                let error1 = rank1.err().map(|e| format!("rank 1: {e:#}"));
                anyhow::bail!(
                    "could not launch the serving pair: {}",
                    [error0, error1]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                        .join("; ")
                );
            }
        };

        let mut startup = PairStartupGuard::new([launched0.child.clone(), launched1.child.clone()]);
        let (send, receive) = std::sync::mpsc::channel();
        let send0 = send.clone();
        let worker0 = std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                launched0.open().and_then(StartingServe::wait_ready)
            }))
            .unwrap_or_else(|_| Err(anyhow::anyhow!("rank 0 startup worker panicked")));
            let _ = send0.send((0usize, result));
        });
        let send1 = send.clone();
        let worker1 = std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                launched1.open().and_then(StartingServe::wait_ready)
            }))
            .unwrap_or_else(|_| Err(anyhow::anyhow!("rank 1 startup worker panicked")));
            let _ = send1.send((1usize, result));
        });
        drop(send);

        let mut startup_error = None;
        for _ in 0..2 {
            let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) else {
                startup_error.get_or_insert_with(|| {
                    format!("serving pair did not become ready within {startup_timeout:?}")
                });
                startup.abort();
                break;
            };
            match receive.recv_timeout(remaining) {
                Ok((rank, Ok(client))) => startup.ready(rank, client),
                Ok((rank, Err(error))) => {
                    startup_error.get_or_insert_with(|| {
                        format!("rank {rank} did not become ready: {error:#}")
                    });
                    startup.abort();
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    startup_error.get_or_insert_with(|| {
                        format!("serving pair did not become ready within {startup_timeout:?}")
                    });
                    startup.abort();
                    break;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    startup_error.get_or_insert_with(|| {
                        "rank startup workers disappeared before READY".to_string()
                    });
                    startup.abort();
                    break;
                }
            }
        }
        let worker0_panicked = worker0.join().is_err();
        let worker1_panicked = worker1.join().is_err();
        if worker0_panicked || worker1_panicked {
            startup_error.get_or_insert_with(|| "a rank startup worker panicked".to_string());
            startup.abort();
        }
        if let Some(error) = startup_error {
            anyhow::bail!(error);
        }
        let rank0 = startup.take(0).context("rank 0 returned no client")?;
        let rank1 = startup.take(1).context("rank 1 returned no client")?;
        let ready = compatible_ready(rank0.ready(), rank1.ready())?;
        startup.disarm();
        Ok(Self {
            rank0,
            rank1,
            ready,
            terminated: false,
        })
    }

    /// The synthesized full-stack READY announced to the downstream client.
    pub fn ready(&self) -> &Ready {
        &self.ready
    }

    /// The individual rank READY records, useful for diagnostics.
    pub fn rank_ready(&self) -> [&Ready; 2] {
        [self.rank0.ready(), self.rank1.ready()]
    }

    /// Mirror one exact context fragment to both ranks.
    pub fn feed(&mut self, text: &str) -> Result<()> {
        anyhow::ensure!(!self.terminated, "the serving pair is terminated");
        if let Err(error) = self.rank0.feed(text) {
            self.fail_and_reap();
            return Err(error.context("feed rank 0"));
        }
        if let Err(error) = self.rank1.feed(text) {
            self.fail_and_reap();
            return Err(error.context("feed rank 1"));
        }
        Ok(())
    }

    /// Mirror one typed context record to both ranks.
    pub fn context(&mut self, context: &InklingContext) -> Result<()> {
        anyhow::ensure!(!self.terminated, "the serving pair is terminated");
        if let Err(error) = self.rank0.context(context) {
            self.fail_and_reap();
            return Err(error.context("feed typed context to rank 0"));
        }
        if let Err(error) = self.rank1.context(context) {
            self.fail_and_reap();
            return Err(error.context("feed typed context to rank 1"));
        }
        Ok(())
    }

    /// Reinitialize both ranks at the same completed-turn boundary.
    ///
    /// Session reset has no collective, so the requests may be acknowledged in
    /// rank order. No downstream acknowledgement is valid until both ranks
    /// report the same discarded position, turn count and replacement width;
    /// one failure or disagreement terminates the pair instead of letting its
    /// sequences diverge.
    pub fn reinitialize(&mut self, initialization: &InklingContext) -> Result<Reinitialized> {
        anyhow::ensure!(!self.terminated, "the serving pair is terminated");
        let rank0 = match self.rank0.reinitialize(initialization) {
            Ok(ack) => ack,
            Err(error) => {
                self.fail_and_reap();
                return Err(error.context("reinitialize rank 0"));
            }
        };
        let rank1 = match self.rank1.reinitialize(initialization) {
            Ok(ack) => ack,
            Err(error) => {
                self.fail_and_reap();
                return Err(error.context("reinitialize rank 1"));
            }
        };
        if rank0 != rank1 {
            self.fail_and_reap();
            anyhow::bail!("rank reinitialization mismatch: rank 0 {rank0:?}, rank 1 {rank1:?}");
        }
        Ok(rank0)
    }

    /// Consult both ranks concurrently and stream only equal fragment pairs.
    pub fn consult(
        &mut self,
        request: &Consult,
        mut on_token: impl FnMut(&str) -> Result<()>,
    ) -> std::result::Result<TurnEnd, ServePairError> {
        if self.terminated {
            return Err(ServePairError::new(
                ServePairFailure::Rank,
                0,
                "the serving pair is terminated",
            ));
        }
        let child0 = self.rank0.child.clone();
        let child1 = self.rank1.child.clone();
        let children = [child0.clone(), child1.clone()];
        let result = std::thread::scope(|scope| {
            // Sixty-four unconfirmed fragments is a hard memory/liveness bound,
            // not a batching target. Healthy TP ranks perform each collective
            // together and normally differ by zero or one pipe records.
            const MAX_UNCONFIRMED_FRAGMENTS: usize = 64;
            let (send, receive) =
                std::sync::mpsc::sync_channel::<RankEvent>(MAX_UNCONFIRMED_FRAGMENTS * 2);
            let send0 = send.clone();
            let rank0 = &mut self.rank0;
            scope.spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    rank0.consult(request, |text| {
                        send0
                            .send(RankEvent::text(0, text))
                            .map_err(|_| anyhow::anyhow!("the pair broker stopped"))
                    })
                }))
                .unwrap_or_else(|_| Err(anyhow::anyhow!("rank 0 consult worker panicked")));
                let event = match result {
                    Ok(end) => RankEvent::end(0, end),
                    Err(error) => RankEvent::error(0, error),
                };
                let _ = send0.send(event);
            });
            let send1 = send.clone();
            let rank1 = &mut self.rank1;
            scope.spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    rank1.consult(request, |text| {
                        send1
                            .send(RankEvent::text(1, text))
                            .map_err(|_| anyhow::anyhow!("the pair broker stopped"))
                    })
                }))
                .unwrap_or_else(|_| Err(anyhow::anyhow!("rank 1 consult worker panicked")));
                let event = match result {
                    Ok(end) => RankEvent::end(1, end),
                    Err(error) => RankEvent::error(1, error),
                };
                let _ = send1.send(event);
            });
            drop(send);

            let result = broker_rank_streams(receive, MAX_UNCONFIRMED_FRAGMENTS, &mut on_token);
            if result.is_err() {
                let _ = child0.kill();
                let _ = child1.kill();
            }
            result
        });
        if result.is_err() {
            kill_children(&children);
            wait_children(&children);
            self.terminated = true;
        }
        result
    }

    /// End both input streams before waiting for either rank, then wait for the
    /// two processes concurrently.
    pub fn close(self) -> Result<[std::process::ExitStatus; 2]> {
        self.close_with_timeout(Self::DEFAULT_SHUTDOWN_TIMEOUT)
    }

    /// [`ServePair::close`] with an explicit bound on the END handshake.
    pub fn close_with_timeout(
        mut self,
        shutdown_timeout: std::time::Duration,
    ) -> Result<[std::process::ExitStatus; 2]> {
        anyhow::ensure!(!self.terminated, "the serving pair is already terminated");
        anyhow::ensure!(
            !shutdown_timeout.is_zero(),
            "the pair shutdown timeout must be nonzero"
        );
        let end0 = self.rank0.end_input();
        let end1 = self.rank1.end_input();
        let children = [self.rank0.child.clone(), self.rank1.child.clone()];
        if end0.is_err() || end1.is_err() {
            kill_children(&children);
        }
        let statuses = wait_children_result_with_timeout(&children, shutdown_timeout);
        if statuses.is_err() {
            kill_children(&children);
            wait_children(&children);
        }
        self.terminated = true;
        end0.context("end rank 0 input")?;
        end1.context("end rank 1 input")?;
        statuses
    }

    fn fail_and_reap(&mut self) {
        if self.terminated {
            return;
        }
        let children = [self.rank0.child.clone(), self.rank1.child.clone()];
        kill_children(&children);
        wait_children(&children);
        self.terminated = true;
    }
}

fn concurrently<T: Send, L, R>(left: L, right: R) -> Result<(Result<T>, Result<T>)>
where
    L: FnOnce() -> Result<T> + Send,
    R: FnOnce() -> Result<T> + Send,
{
    std::thread::scope(|scope| {
        let left = scope.spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(left))
                .unwrap_or_else(|_| Err(anyhow::anyhow!("left worker panicked")))
        });
        let right = scope.spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(right))
                .unwrap_or_else(|_| Err(anyhow::anyhow!("right worker panicked")))
        });
        Ok((
            left.join()
                .map_err(|_| anyhow::anyhow!("left worker escaped panic containment"))?,
            right
                .join()
                .map_err(|_| anyhow::anyhow!("right worker escaped panic containment"))?,
        ))
    })
}

impl Drop for ServePair {
    fn drop(&mut self) {
        self.fail_and_reap();
    }
}

fn compatible_ready(rank0: &Ready, rank1: &Ready) -> Result<Ready> {
    for (rank, ready) in [(0usize, rank0), (1usize, rank1)] {
        anyhow::ensure!(ready.stack > 0, "rank {rank} announced an empty stack");
        anyhow::ensure!(
            ready.prefill_budget > 0,
            "rank {rank} announced a zero-token prefill budget"
        );
        anyhow::ensure!(
            ready.prefill_budget <= ready.context_budget,
            "rank {rank} announced a {}-token prefill budget wider than its {}-token context budget",
            ready.prefill_budget,
            ready.context_budget
        );
        for (name, identity) in [
            ("model", ready.model_identity.as_str()),
            ("tokenizer", ready.tokenizer_identity.as_str()),
            ("execution", ready.execution_identity.as_str()),
        ] {
            anyhow::ensure!(
                identity.len() == 64 && identity.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "rank {rank} announced an invalid {name} identity {identity:?}"
            );
        }
        anyhow::ensure!(
            ready.execution_profile == "sealed-v1",
            "rank {rank} announced execution profile {:?}; a paired computation requires \
             sealed-v1 so ambient execution overrides cannot differ between ranks",
            ready.execution_profile
        );
        anyhow::ensure!(
            ready.tp_rank == Some(rank) && ready.tp_world == 2,
            "rank {rank} announced TP role {:?} of {}, expected rank {rank} of 2",
            ready.tp_rank,
            ready.tp_world
        );
        anyhow::ensure!(
            !ready.partial && ready.layers == [0, ready.stack],
            "rank {rank} is not a full stack: layers {}..{} of {}, partial={}",
            ready.layers[0],
            ready.layers[1],
            ready.stack,
            ready.partial
        );
    }
    anyhow::ensure!(
        rank0.layers == rank1.layers && rank0.stack == rank1.stack,
        "rank READY stack mismatch: {:?}/{} vs {:?}/{}",
        rank0.layers,
        rank0.stack,
        rank1.layers,
        rank1.stack
    );
    anyhow::ensure!(
        rank0.vocab == rank1.vocab,
        "rank READY vocabulary mismatch: {} vs {}",
        rank0.vocab,
        rank1.vocab
    );
    anyhow::ensure!(
        rank0.model_identity == rank1.model_identity,
        "rank READY model identity mismatch: {} vs {}",
        rank0.model_identity,
        rank1.model_identity
    );
    anyhow::ensure!(
        rank0.tokenizer_identity == rank1.tokenizer_identity,
        "rank READY tokenizer identity mismatch: {} vs {}",
        rank0.tokenizer_identity,
        rank1.tokenizer_identity
    );
    anyhow::ensure!(
        rank0.special_ids == rank1.special_ids,
        "rank READY special-token ids mismatch: {:?} vs {:?}",
        rank0.special_ids,
        rank1.special_ids
    );
    anyhow::ensure!(
        rank0.execution_profile == rank1.execution_profile,
        "rank READY execution profile mismatch: {:?} vs {:?}",
        rank0.execution_profile,
        rank1.execution_profile
    );
    anyhow::ensure!(
        rank0.execution_identity == rank1.execution_identity,
        "rank READY execution identity mismatch: {} vs {}",
        rank0.execution_identity,
        rank1.execution_identity
    );
    anyhow::ensure!(
        rank0.execution_unavailable == rank1.execution_unavailable,
        "rank READY unavailable execution facts mismatch: {:?} vs {:?}",
        rank0.execution_unavailable,
        rank1.execution_unavailable
    );
    anyhow::ensure!(
        rank0.prefill_budget == rank1.prefill_budget,
        "rank READY prefill budget mismatch: {} vs {}",
        rank0.prefill_budget,
        rank1.prefill_budget
    );
    anyhow::ensure!(
        rank0.context_budget == rank1.context_budget,
        "rank READY context budget mismatch: {} vs {}",
        rank0.context_budget,
        rank1.context_budget
    );
    Ok(Ready {
        pile: rank0.pile.clone(),
        model_identity: rank0.model_identity.clone(),
        tokenizer_identity: rank0.tokenizer_identity.clone(),
        special_ids: rank0.special_ids.clone(),
        execution_profile: rank0.execution_profile.clone(),
        execution_identity: rank0.execution_identity.clone(),
        execution_unavailable: rank0.execution_unavailable.clone(),
        tp_rank: None,
        tp_world: rank0.tp_world,
        layers: rank0.layers,
        stack: rank0.stack,
        partial: false,
        vocab: rank0.vocab,
        prefill_budget: rank0.prefill_budget,
        context_budget: rank0.context_budget,
        load_secs: rank0.load_secs.max(rank1.load_secs),
    })
}

enum RankPart {
    Text(String),
    End(TurnEnd),
    Error(String),
}

struct RankEvent {
    rank: usize,
    part: RankPart,
}

impl RankEvent {
    fn text(rank: usize, text: &str) -> Self {
        Self {
            rank,
            part: RankPart::Text(text.to_string()),
        }
    }

    fn end(rank: usize, end: TurnEnd) -> Self {
        Self {
            rank,
            part: RankPart::End(end),
        }
    }

    fn error(rank: usize, error: anyhow::Error) -> Self {
        Self {
            rank,
            part: RankPart::Error(format!("{error:#}")),
        }
    }
}

fn broker_rank_streams(
    receive: std::sync::mpsc::Receiver<RankEvent>,
    max_unconfirmed: usize,
    on_token: &mut impl FnMut(&str) -> Result<()>,
) -> std::result::Result<TurnEnd, ServePairError> {
    let mut pending: [std::collections::VecDeque<RankPart>; 2] = Default::default();
    let mut confirmed_extent = 0u64;
    loop {
        let event = receive.recv().map_err(|error| {
            ServePairError::new(
                ServePairFailure::Rank,
                confirmed_extent,
                format!("both rank streams ended without a complete turn: {error}"),
            )
        })?;
        if event.rank > 1 {
            return Err(ServePairError::new(
                ServePairFailure::Divergence,
                confirmed_extent,
                format!("invalid rank number {}", event.rank),
            ));
        }
        match event.part {
            RankPart::Error(message) => {
                return Err(ServePairError::new(
                    ServePairFailure::Rank,
                    confirmed_extent,
                    format!("rank {} failed: {message}", event.rank),
                ));
            }
            part => pending[event.rank].push_back(part),
        }
        if pending[event.rank].len() > max_unconfirmed {
            return Err(ServePairError::new(
                ServePairFailure::Divergence,
                confirmed_extent,
                format!(
                    "rank {} exceeded the {}-fragment agreement window",
                    event.rank, max_unconfirmed
                ),
            ));
        }

        while !pending[0].is_empty() && !pending[1].is_empty() {
            let left = pending[0].pop_front().expect("checked nonempty");
            let right = pending[1].pop_front().expect("checked nonempty");
            match (left, right) {
                (RankPart::Text(left), RankPart::Text(right)) => {
                    if left != right {
                        return Err(ServePairError::new(
                            ServePairFailure::Divergence,
                            confirmed_extent,
                            format!("rank token text diverged: rank 0 {left:?}, rank 1 {right:?}"),
                        ));
                    }
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_token(&left)))
                        .map_err(|_| {
                            ServePairError::new(
                                ServePairFailure::Consumer,
                                confirmed_extent,
                                "the paired-turn consumer panicked",
                            )
                        })?
                        .map_err(|error| {
                            ServePairError::new(
                                ServePairFailure::Consumer,
                                confirmed_extent,
                                format!("the paired-turn consumer failed: {error:#}"),
                            )
                        })?;
                    confirmed_extent += left.len() as u64;
                }
                (RankPart::End(mut left), RankPart::End(right)) => {
                    compare_turn_ends(&left, &right).map_err(|message| {
                        ServePairError::new(ServePairFailure::Divergence, confirmed_extent, message)
                    })?;
                    // The pair is one distributed computation. State must
                    // agree exactly, while elapsed time is a rank-local
                    // observation; its honest critical path is the slower
                    // rank, not whichever rank happens to be numbered zero.
                    left.first_token_secs = left.first_token_secs.max(right.first_token_secs);
                    left.turn_secs = left.turn_secs.max(right.turn_secs);
                    return Ok(left);
                }
                (RankPart::Text(_), RankPart::End(_)) => {
                    return Err(ServePairError::new(
                        ServePairFailure::Divergence,
                        confirmed_extent,
                        "rank 1 ended while rank 0 still had token text",
                    ));
                }
                (RankPart::End(_), RankPart::Text(_)) => {
                    return Err(ServePairError::new(
                        ServePairFailure::Divergence,
                        confirmed_extent,
                        "rank 0 ended while rank 1 still had token text",
                    ));
                }
                (RankPart::Error(_), _) | (_, RankPart::Error(_)) => {
                    unreachable!("rank errors are handled before queueing")
                }
            }
        }
    }
}

fn compare_turn_ends(rank0: &TurnEnd, rank1: &TurnEnd) -> std::result::Result<(), String> {
    if rank0.token_ids.len() != rank0.tokens || rank1.token_ids.len() != rank1.tokens {
        return Err(format!(
            "rank TURN omitted exact token ids: rank 0 {}/{}, rank 1 {}/{}",
            rank0.token_ids.len(),
            rank0.tokens,
            rank1.token_ids.len(),
            rank1.tokens
        ));
    }
    macro_rules! same {
        ($field:ident) => {
            if rank0.$field != rank1.$field {
                return Err(format!(
                    "rank TURN field `{}` diverged: {:?} vs {:?}",
                    stringify!($field),
                    rank0.$field,
                    rank1.$field
                ));
            }
        };
    }
    same!(turn);
    same!(tokens);
    same!(token_ids);
    same!(delta_tokens);
    same!(carried);
    same!(stopped);
    same!(position);
    // Timings are observations made by separate processes, not model state.
    Ok(())
}

fn kill_children(children: &[ChildHandle; 2]) {
    let _ = children[0].kill();
    let _ = children[1].kill();
}

/// Give a closed serving channel one finite chance to shut its process down,
/// then force termination and retain a reaper even if the kernel cannot report
/// the exit within the second finite grace.
fn reap_child_after_channel_close(child: &ChildHandle) {
    match wait_child_with_timeout(child, SINGLE_REAP_GRACE) {
        Ok(Some(_)) => return,
        Ok(None) | Err(_) => {}
    }
    let _ = child.kill();
    match wait_child_with_timeout(child, SINGLE_REAP_GRACE) {
        Ok(Some(_)) => {}
        Ok(None) | Err(_) => {
            let _ = detach_child_reaper(child.clone());
        }
    }
}

/// Pair form of [`reap_child_after_channel_close`]. The first grace begins
/// only after all later-owned protocol pipes have dropped, so remote
/// supervisors can observe EOF before their local SSH transports are killed.
fn reap_children_after_channel_close(children: &[ChildHandle; 2]) {
    if wait_children_result_with_timeout(children, SINGLE_REAP_GRACE).is_ok() {
        return;
    }
    force_reap_children(children);
}

fn force_reap_children(children: &[ChildHandle; 2]) {
    kill_children(children);
    if wait_children_result_with_timeout(children, SINGLE_REAP_GRACE).is_ok() {
        return;
    }
    for child in children {
        let _ = detach_child_reaper(child.clone());
    }
}

fn wait_children(children: &[ChildHandle; 2]) {
    force_reap_children(children);
}

#[cfg(test)]
fn wait_children_result(children: &[ChildHandle; 2]) -> Result<[std::process::ExitStatus; 2]> {
    wait_children_result_with_timeout(children, ServePair::DEFAULT_SHUTDOWN_TIMEOUT)
}

fn wait_children_result_with_timeout(
    children: &[ChildHandle; 2],
    timeout: std::time::Duration,
) -> Result<[std::process::ExitStatus; 2]> {
    anyhow::ensure!(!timeout.is_zero(), "the child wait timeout must be nonzero");
    let deadline = std::time::Instant::now()
        .checked_add(timeout)
        .context("the child wait timeout is too large")?;
    let mut statuses = [None, None];
    loop {
        for rank in 0..2 {
            if statuses[rank].is_none() {
                statuses[rank] = match children[rank].try_wait() {
                    Ok(status) => status,
                    Err(error) => {
                        kill_children(children);
                        return Err(error);
                    }
                };
                if statuses[rank]
                    .as_ref()
                    .is_some_and(|status| !status.success())
                    && statuses[1 - rank].is_none()
                {
                    // No blocking wait owns the sibling's mutex, so a failing
                    // rank can always tear its peer down immediately.
                    let _ = children[1 - rank].kill();
                }
            }
        }
        if let [Some(rank0), Some(rank1)] = &statuses {
            return Ok([*rank0, *rank1]);
        }
        if std::time::Instant::now() >= deadline {
            kill_children(children);
            anyhow::bail!("serving pair did not exit within {timeout:?}");
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

// ── the drive seam ──────────────────────────────────────────────────────────

/// What a turn proved about STREAMING, recorded per turn so the claim is a
/// measurement rather than an assertion.
///
/// The question is not "did the faculty get the words" — a batch would pass
/// that. It is "did the faculty produce OUTPUT while the mind was still
/// generating", and the only way to answer it is to look at the faculty's
/// return stream from inside `observe`, with the turn demonstrably unfinished.
#[cfg(feature = "drive-mind")]
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
}

#[cfg(feature = "drive-mind")]
impl StreamProof {
    /// Whether this turn STREAMED: the consumer produced output strictly before
    /// the mind stopped generating.
    pub fn streamed(&self) -> bool {
        matches!(self.tokens_at_first_return, Some(k) if k < self.tokens)
    }
}

/// The [`drive::mind::Mind`] backed by a real Inkling `Session` in another
/// process.
///
/// This is the seam drive's `Mind` docs describe, filled in: it consumes a
/// causally ordered DELTA (not a re-rendered transcript), produces an utterance,
/// produces a decision always, and is stateful across turns — the state being a
/// KV cache in a process that outlives every call.
///
/// # What it ignores, and why that is correct
///
/// `Payload::Monologue` events are the mind's OWN words from earlier turns. A
/// stateful backend already has them, and it takes TWO mechanisms rather than
/// one: all but the last token of a turn were fed back by its own `step()`, and
/// the last one — which the generation loop deliberately does not spend a decode
/// step to feed — is appended at the head of the NEXT turn's delta (see
/// `inkling_serve::serve_turn` and [`TurnEnd::carried`]). Between them, every
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
/// over that exact fresh range. Text-only and sliced responses remain audited
/// `NoAction` decisions.
#[cfg(feature = "drive-mind")]
pub struct InklingMind {
    client: ServeClient,
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
    /// Tokens per turn.
    max_tokens: usize,
    /// System prompt held until the first released memory cover can be inserted
    /// in the same typed initialization record.
    system: Option<String>,
    initialized: bool,
    /// Typed generated-output parser, retained when a Drive turn's token slice
    /// ends before the model's logical response does.
    output: NativeOutputParser,
    /// Provider reasoning accumulated across Drive token slices for the
    /// current logical assistant response.
    response_thinking: String,
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
    /// Per-turn numbers, shared so the caller can report them after the run.
    log: std::sync::Arc<std::sync::Mutex<Vec<TurnEnd>>>,
    proofs: std::sync::Arc<std::sync::Mutex<Vec<StreamProof>>>,
}

/// A consultation that failed after producing zero or more final text bytes.
///
/// Keeping the partial utterance on the stack makes one failed call one value:
/// it cannot leak into a later turn, and failures before generation naturally
/// carry an empty string through `?`.
#[cfg(feature = "drive-mind")]
struct FailedTurn {
    error: anyhow::Error,
    said: String,
}

#[cfg(feature = "drive-mind")]
impl From<anyhow::Error> for FailedTurn {
    fn from(error: anyhow::Error) -> Self {
        Self {
            error,
            said: String::new(),
        }
    }
}

#[cfg(feature = "drive-mind")]
fn text_result_delta(
    command: &str,
    content: &drive::content::Content,
    is_error: bool,
    exit_code: Option<i32>,
) -> String {
    // Inkling's serving wire is text/plain and Session accepts token ids, so it
    // cannot consume drive's resident image/audio parts yet. This is drive's
    // explicit compatibility seam for a text-only mind, not a second stored
    // representation: the typed Content stays intact in the World.
    let projected = content.text_projection();
    let mut delta = format!("\n$ {command}\n{projected}");

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
    delta.push('\n');
    delta
}

#[cfg(feature = "drive-mind")]
impl InklingMind {
    /// Wrap a running serving process as a mind.
    pub fn new(client: ServeClient, max_tokens: usize, system: Option<String>) -> Self {
        let label = match client.ready().partial {
            true => "inkling(partial)".to_string(),
            false => "inkling".to_string(),
        };
        let output = NativeOutputParser::new(client.ready().special_ids.clone());
        Self {
            client,
            voice: std::sync::Arc::new(std::sync::Mutex::new(None)),
            buffer: drive::mind::MonologueBuffer::with_cap(64 * 1024),
            scanned_abs: 0,
            max_tokens,
            system,
            initialized: false,
            output,
            response_thinking: String::new(),
            outstanding_exec: None,
            needs_generation_prompt: false,
            turns: 0,
            label,
            log: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            proofs: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
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

    /// Per-turn numbers, as the serving process measured them.
    pub fn log(&self) -> std::sync::Arc<std::sync::Mutex<Vec<TurnEnd>>> {
        std::sync::Arc::clone(&self.log)
    }

    /// What each turn proved about streaming.
    pub fn proofs(&self) -> std::sync::Arc<std::sync::Mutex<Vec<StreamProof>>> {
        std::sync::Arc::clone(&self.proofs)
    }

    /// What loaded on the far end.
    pub fn ready(&self) -> &Ready {
        self.client.ready()
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
}

/// The shell OWNS the mind (`Box<dyn Mind>`), so no caller can ever hand the
/// serving process a clean goodbye by name. It is done here.
///
/// Without this, dropping the writer would write `END{aborted}` — the
/// convention's best-effort "the producer died" — and a run that finished did
/// not die. And the WAIT is load-bearing on a unified-memory box: the serving
/// process holds tens of gibibytes of arena that the kernel takes a while to
/// reclaim, and returning from `Shell::finish` while it is still held is how the
/// next run gets OOM-killed for no fault of its own.
#[cfg(feature = "drive-mind")]
impl Drop for InklingMind {
    fn drop(&mut self) {
        match self
            .client
            .shutdown_with_timeout(ServeClient::DEFAULT_SHUTDOWN_TIMEOUT)
        {
            Ok(status) => eprintln!("inkling_serve: the serving process exited: {status}"),
            Err(error) => {
                eprintln!("inkling_serve: bounded serving-process shutdown failed: {error:#}")
            }
        }
    }
}

#[cfg(feature = "drive-mind")]
impl drive::mind::Mind for InklingMind {
    fn observe(&mut self, view: drive::world::MergedView<'_>) -> drive::mind::Turn {
        match self.turn(view.events, view.watermark) {
            Ok(turn) => turn,
            Err(FailedTurn { error, said }) => {
                // Equal fragments may already have escaped into the streaming
                // voice. They remain this turn's truthful partial utterance;
                // the backend failure is orthogonal terminal state, never an
                // ordinary silent NoAction and never a forward `Gap` pretending
                // it can retract bytes.
                eprintln!("inkling_serve: {error:#}");
                let mut rationale = format!("inkling serving process failed: {error:#}");
                if let Err(kill) = self.client.kill() {
                    eprintln!("inkling_serve: could not kill the failed backend: {kill:#}");
                    rationale.push_str(&format!("; backend teardown also failed: {kill:#}"));
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

    fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(feature = "drive-mind")]
fn aggregate_microturns(
    logical_turn: usize,
    microturns: &[TurnEnd],
    completed: bool,
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
        stopped: if completed {
            "content_model_end_sampling".to_string()
        } else {
            "max_tokens".to_string()
        },
        first_token_secs: first.first_token_secs,
        turn_secs: microturns.iter().map(|end| end.turn_secs).sum(),
        position: last.position,
    })
}

#[cfg(feature = "drive-mind")]
impl InklingMind {
    /// One turn, with the failure path lifted out so `observe` can stay total.
    fn turn(
        &mut self,
        events: &[drive::world::Event],
        watermark: drive::world::Coord,
    ) -> std::result::Result<drive::mind::Turn, FailedTurn> {
        let mut results = Vec::new();
        for event in events {
            match &event.payload {
                // Coordinates only. The model already attended to these tokens:
                // its own `step` fed back all but each turn's last, and the
                // serving process carries that last one into the next turn's
                // delta. See this type's doc for why both halves have to be
                // true, and what happened while only one was.
                drive::world::Payload::Monologue(text) => self.buffer.push_free(text),
                drive::world::Payload::Result {
                    command,
                    content,
                    is_error,
                    exit_code,
                } => results.push(ExecResultContext {
                    command: command.clone(),
                    content: text_result_delta(command, content, *is_error, *exit_code),
                }),
            }
        }

        if !self.initialized {
            // Initialization is one server-composed batch so memory-cover
            // history sits after system/effort but before the sole generation
            // prompt, exactly as the shipped chat template requires.
            self.client.context(&InklingContext::Initialize {
                system: self.system.take().unwrap_or_default(),
                historical_exec_results: results,
            })?;
            self.initialized = true;
            self.needs_generation_prompt = false;
        } else {
            if results.len() > 1 {
                return Err(anyhow::anyhow!(
                    "received {} exec results in one post-initialization view",
                    results.len()
                )
                .into());
            }
            if let Some(result) = results.pop() {
                match self.outstanding_exec.as_deref() {
                    Some(command) => {
                        if command != result.command {
                            return Err(anyhow::anyhow!(
                                "exec result command {:?} did not match outstanding command {:?}",
                                result.command,
                                command
                            )
                            .into());
                        }
                        self.client
                            .context(&InklingContext::ToolResult { result })?;
                        self.outstanding_exec = None;
                        self.needs_generation_prompt = false;
                    }
                    None => {
                        if !self.needs_generation_prompt {
                            return Err(anyhow::anyhow!(
                                "historical exec result arrived inside an unfinished assistant response"
                            )
                            .into());
                        }
                        self.client
                            .context(&InklingContext::HistoricalExecResult { result })?;
                        self.needs_generation_prompt = false;
                    }
                }
            } else if self.outstanding_exec.is_some() {
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
            } else if self.needs_generation_prompt {
                self.client.context(&InklingContext::GenerationPrompt)?;
                self.needs_generation_prompt = false;
            }
        }

        // One-token consults are the TP-safe stop boundary. Each collective
        // arbitrates exactly one id; only after it returns do we associate that
        // id with its decoder fragments and interpret it client-side.
        let voice = self.voice.lock().expect("voice slot").clone();
        let mut said = String::new();
        let mut thinking_this_turn = String::new();
        let mut microturns = Vec::new();
        let mut tokens = 0usize;
        let mut tokens_at_first_return = None;
        let turn = self.turns;
        let mut completed = false;
        for _ in 0..self.max_tokens.max(1) {
            let mut fragments = String::new();
            let end = match self.client.consult(&Consult::new(1), |fragment| {
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
            self.response_thinking.push_str(&delta.thinking);
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
        }

        let end = match aggregate_microturns(turn, &microturns, completed) {
            Ok(end) => end,
            Err(error) => return Err(FailedTurn { error, said }),
        };
        let records_at_end = voice.as_ref().map(|v| v.report().records).unwrap_or(0);
        self.proofs.lock().expect("proof log").push(StreamProof {
            turn,
            tokens,
            tokens_at_first_return,
            records_at_end,
        });
        self.log.lock().expect("turn log").push(end);
        self.turns += 1;

        let mut reasoning = thinking_this_turn;
        let mut decision = if completed {
            let call = match self.output.take_completed_call() {
                Ok(call) => call,
                Err(error) => return Err(FailedTurn { error, said }),
            };
            reasoning = std::mem::take(&mut self.response_thinking);
            match call {
                Some(call) => {
                    let fresh_base = self.buffer.end_offset();
                    let range = project_native_exec(&mut said, &call.command);
                    let span = said[range.clone()].to_string();
                    let _ = self.finish_coverage();
                    self.outstanding_exec = Some(call.command.clone());
                    drive::mind::Decision::fire(
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
        } else {
            let (span_start, span_end) = self.finish_coverage();
            drive::mind::Decision::no_action(
                span_start,
                span_end,
                watermark,
                "inkling: assistant response continues in the next token slice",
            )
        };
        if !reasoning.is_empty() {
            decision = decision.with_reasoning(reasoning);
        }
        Ok(drive::mind::Turn::new(said, decision))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[derive(Clone, Default)]
    struct SharedSink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    #[cfg(unix)]
    impl std::io::Write for SharedSink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("fixture sink")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[cfg(unix)]
    impl SharedSink {
        fn len(&self) -> usize {
            self.0.lock().expect("fixture sink").len()
        }

        fn bytes(&self) -> Vec<u8> {
            self.0.lock().expect("fixture sink").clone()
        }
    }

    /// The control records are the protocol, so their encoding is worth pinning:
    /// a rename that a compiler cannot see would be a wire break.
    #[test]
    fn the_control_records_round_trip() {
        let consult = Consult::new(16);
        let bytes = serde_json::to_vec(&consult).expect("encode");
        let back: Consult = serde_json::from_slice(&bytes).expect("decode");
        assert_eq!(back.max_tokens, 16);

        let end = TurnEnd {
            turn: 3,
            tokens: 10,
            token_ids: (0..10).collect(),
            delta_tokens: 4,
            carried: 1,
            stopped: "max_tokens".to_string(),
            first_token_secs: 0.021,
            turn_secs: 0.43,
            position: 62,
        };
        let back: TurnEnd = serde_json::from_slice(&serde_json::to_vec(&end).unwrap()).unwrap();
        assert_eq!(back.position, 62);
        // The summary carries its own framing: "first token", not "per token".
        assert!(
            back.summary().contains("first token 0.021s"),
            "{}",
            back.summary()
        );
        // And it says how wide the pass actually was, not just the client's half
        // of it: a turn whose model did not hear its own last word is a turn
        // that reported `+0 carried` after turn 0, and that has to be readable.
        assert!(
            back.summary().contains("4-token delta (+1 carried)"),
            "{}",
            back.summary()
        );

        let context = InklingContext::Initialize {
            system: "literal <|message_model|>".to_string(),
            historical_exec_results: vec![ExecResultContext {
                command: "printf hi".to_string(),
                content: "literal <|content_invoke_tool_json|>".to_string(),
            }],
        };
        let encoded = serde_json::to_vec(&context).expect("encode context");
        let decoded: InklingContext = serde_json::from_slice(&encoded).expect("decode context");
        assert_eq!(decoded, context);

        let acknowledgement = Reinitialized {
            previous_position: 65_000,
            previous_turns: 41,
            initialization_tokens: 12_345,
        };
        let decoded: Reinitialized =
            serde_json::from_slice(&serde_json::to_vec(&acknowledgement).unwrap()).unwrap();
        assert_eq!(decoded, acknowledgement);
    }

    /// The control types must be distinct from each other and from the
    /// stream's own type, or a control record would read as content.
    #[test]
    fn the_control_types_are_distinct_from_content() {
        let types = [
            READY_TYPE,
            CONSULT_TYPE,
            CONTEXT_TYPE,
            TURN_TYPE,
            REINITIALIZE_TYPE,
            REINITIALIZED_TYPE,
        ];
        for t in types {
            assert_ne!(t, CONTENT_TYPE);
        }
        assert_eq!(types.len(), {
            let mut set: Vec<&str> = types.to_vec();
            set.sort_unstable();
            set.dedup();
            set.len()
        });
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

    #[cfg(unix)]
    #[test]
    fn client_reinitialization_is_one_initialize_request_and_one_acknowledgement() {
        let response = SharedSink::default();
        let mut writer = framed_stream::FramedWriter::open(response.clone(), CONTENT_TYPE, UNIT)
            .expect("response preamble");
        let ready_payload = serde_json::to_vec(&fake_ready("fixture.pile")).expect("READY json");
        writer
            .record_as(READY_TYPE, &ready_payload, ready_payload.len() as u64)
            .expect("READY record");
        let expected = Reinitialized {
            previous_position: 65_000,
            previous_turns: 41,
            initialization_tokens: 12_345,
        };
        let ack_payload = serde_json::to_vec(&expected).expect("REINITIALIZED json");
        writer
            .record_as(REINITIALIZED_TYPE, &ack_payload, ack_payload.len() as u64)
            .expect("REINITIALIZED record");
        writer
            .finish(framed_stream::EndStatus::Complete)
            .expect("complete response");

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let response_path =
            std::env::temp_dir().join(format!("mary-reinitialize-response-{nonce}"));
        let capture_path = std::env::temp_dir().join(format!("mary-reinitialize-capture-{nonce}"));
        std::fs::write(&response_path, response.bytes()).expect("write fake response");
        let mut command = std::process::Command::new("sh");
        command
            .arg("-c")
            .arg("cat \"$1\"; cat >\"$2\"")
            .arg("fake-inkling")
            .arg(&response_path)
            .arg(&capture_path);
        let mut client = ServeClient::spawn(&mut command).expect("spawn fake serving process");
        let initialization = InklingContext::Initialize {
            system: "replacement system".to_string(),
            historical_exec_results: vec![ExecResultContext {
                command: "memory old..new".to_string(),
                content: "replacement cover".to_string(),
            }],
        };
        assert_eq!(
            client.reinitialize(&initialization).expect("reinitialize"),
            expected
        );
        assert!(client.close().expect("close fixture").success());

        let mut captured = framed_stream::FramedReader::open(
            std::fs::File::open(&capture_path).expect("open captured input"),
        )
        .expect("captured preamble");
        captured.require_content_type(CONTENT_TYPE).unwrap();
        let framed_stream::Frame::Record(record) = captured.next_frame().unwrap() else {
            panic!("first input frame was not REINITIALIZE")
        };
        assert_eq!(record.content_type(), REINITIALIZE_TYPE);
        assert_eq!(
            serde_json::from_slice::<InklingContext>(&record.payload).unwrap(),
            initialization
        );
        assert_eq!(
            captured.next_frame().unwrap(),
            framed_stream::Frame::End(framed_stream::EndStatus::Complete)
        );
        let _ = std::fs::remove_file(response_path);
        let _ = std::fs::remove_file(capture_path);
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
    fn native_parser_survives_a_drive_token_slice_without_resetting() {
        let ids = fake_ready("parser").special_ids;
        let mut parser = NativeOutputParser::new(ids.clone());
        parser
            .push(
                ids.content_text,
                &special_fragment(&ids, ids.content_text, ""),
            )
            .unwrap();
        assert_eq!(parser.push(1, "first slice").unwrap().text, "first slice");
        assert!(parser.take_completed_call().is_err());

        assert_eq!(
            parser.push(2, " second slice").unwrap().text,
            " second slice"
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
        assert_eq!(&said[range.clone()], "$ printf same-turn\n");
        let prior_monologue_end = 73_u64;
        assert_eq!(prior_monologue_end + range.start as u64, 89);
        assert_eq!(prior_monologue_end + range.end as u64, 108);
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

    #[cfg(feature = "drive-mind")]
    #[test]
    fn logical_turn_aggregation_requires_every_microturn_to_be_one_token() {
        let first = fake_end(&[41]);
        let mut second = fake_end(&[42]);
        second.turn = 1;
        second.delta_tokens = 0;
        second.carried = 1;
        let combined = aggregate_microturns(9, &[first.clone(), second.clone()], false).unwrap();
        assert_eq!(combined.turn, 9);
        assert_eq!(combined.tokens, 2);
        assert_eq!(combined.token_ids, [41, 42]);
        assert_eq!(combined.delta_tokens, first.delta_tokens);
        assert_eq!(combined.carried, first.carried);
        assert_eq!(combined.stopped, "max_tokens");

        second.tokens = 2;
        assert!(aggregate_microturns(9, &[first, second], false).is_err());
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

        let result = ExecResultContext {
            command: format!("printf {MESSAGE_TOOL}"),
            content: format!("hostile {CONTENT_MODEL_END_SAMPLING}"),
        };
        let encoded = codec
            .encode(&InklingContext::HistoricalExecResult { result })
            .unwrap();
        let special = codec.special_ids();
        assert_eq!(
            encoded,
            [
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
                historical_exec_results: vec![ExecResultContext {
                    command: "true".to_string(),
                    content: "ok".to_string(),
                }],
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
            ]
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
            historical_exec_results: vec![result.clone()],
        };
        let rendered = format!(
            "{MESSAGE_SYSTEM}tool_declare{CONTENT_XML}{EXEC_TOOL_DECLARATION}{END_MESSAGE}\
             {MESSAGE_SYSTEM}{CONTENT_TEXT}template system{END_MESSAGE}\
             {MESSAGE_SYSTEM}{CONTENT_TEXT}{DEFAULT_THINKING_EFFORT}{END_MESSAGE}\
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

    fn fake_pair_ready(left_pile: &str, right_pile: &str) -> (Ready, Ready) {
        let mut left = fake_ready(left_pile);
        left.tp_rank = Some(0);
        left.tp_world = 2;
        let mut right = fake_ready(right_pile);
        right.tp_rank = Some(1);
        right.tp_world = 2;
        (left, right)
    }

    #[cfg(unix)]
    fn supervised_ready_fixture(
        ready: &Ready,
        stem: &str,
    ) -> (RankCommand, std::path::PathBuf, std::path::PathBuf) {
        let response = SharedSink::default();
        let mut writer = framed_stream::FramedWriter::open(response.clone(), CONTENT_TYPE, UNIT)
            .expect("response preamble");
        let payload = serde_json::to_vec(ready).expect("READY json");
        writer
            .record_as(READY_TYPE, &payload, payload.len() as u64)
            .expect("READY record");
        let ready_len = response.len();
        drop(writer);
        let mut bytes = response.bytes();
        bytes.truncate(ready_len);

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let base = format!("mary-{stem}-{}-{nonce}", std::process::id());
        let response_path = std::env::temp_dir().join(format!("{base}-response"));
        let pid_path = std::env::temp_dir().join(format!("{base}-pids"));
        std::fs::write(&response_path, bytes).expect("write READY fixture");

        // This shell is a stand-in for the remote supervisor and its `sleep`
        // child for the expensive rank. It can reap the rank only when the
        // serving input reaches EOF while the supervisor is still alive. The
        // old validation path killed this shell first and orphaned `sleep`.
        let script = r#"
sleep 30 &
rank=$!
printf '%s %s\n' "$$" "$rank" >"$1"
dd if="$2" bs=1 count="$3" 2>/dev/null
cat >/dev/null
kill "$rank" 2>/dev/null || true
wait "$rank" 2>/dev/null || true
exit 0
"#;
        let command = RankCommand::local("sh")
            .arg("-c")
            .arg(script)
            .arg("fake-supervisor")
            .arg(pid_path.as_os_str())
            .arg(response_path.as_os_str())
            .arg(ready_len.to_string());
        (command, pid_path, response_path)
    }

    #[cfg(unix)]
    fn fixture_pids(path: &std::path::Path) -> [libc::pid_t; 2] {
        let text = std::fs::read_to_string(path).expect("read fixture pids");
        let mut pids = text
            .split_whitespace()
            .map(|pid| pid.parse::<libc::pid_t>().expect("numeric fixture pid"));
        let result = [
            pids.next().expect("supervisor pid"),
            pids.next().expect("rank pid"),
        ];
        assert!(pids.next().is_none(), "unexpected extra fixture pid");
        result
    }

    #[cfg(unix)]
    fn process_exists(pid: libc::pid_t) -> bool {
        // SAFETY: signal 0 does not alter the target; it only probes whether
        // this test-owned numeric pid still denotes a process.
        let result = unsafe { libc::kill(pid, 0) };
        result == 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
    }

    #[cfg(unix)]
    struct FixtureProcessCleanup(Vec<libc::pid_t>);

    #[cfg(unix)]
    impl Drop for FixtureProcessCleanup {
        fn drop(&mut self) {
            for pid in &self.0 {
                if process_exists(*pid) {
                    // SAFETY: these pids were written by this test's live
                    // supervisor fixtures and are retained only for cleanup.
                    unsafe {
                        libc::kill(*pid, libc::SIGKILL);
                    }
                }
            }
        }
    }

    #[cfg(unix)]
    fn spawn_shutdown_fixture(
        name: &str,
        hang_after_end: bool,
    ) -> (
        ServeClient,
        ChildHandle,
        std::path::PathBuf,
        std::path::PathBuf,
    ) {
        let response = SharedSink::default();
        let mut response_writer =
            framed_stream::FramedWriter::open(response.clone(), CONTENT_TYPE, UNIT)
                .expect("response preamble");
        let ready_payload = serde_json::to_vec(&fake_ready("fixture.pile")).expect("READY json");
        response_writer
            .record_as(READY_TYPE, &ready_payload, ready_payload.len() as u64)
            .expect("READY record");
        let ready_len = response.len();
        drop(response_writer);

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let stem = format!("mary-{name}-{}-{nonce}", std::process::id());
        let response_path = std::env::temp_dir().join(format!("{stem}-response"));
        let capture_path = std::env::temp_dir().join(format!("{stem}-capture"));
        std::fs::write(&response_path, response.bytes()).expect("write fake response");

        let script = if hang_after_end {
            r#"
dd if="$1" bs=1 count="$2" 2>/dev/null
cat >"$3"
exec sleep 30
"#
        } else {
            r#"
dd if="$1" bs=1 count="$2" 2>/dev/null
cat >"$3"
"#
        };
        let mut command = std::process::Command::new("sh");
        command
            .arg("-c")
            .arg(script)
            .arg("fake-inkling")
            .arg(&response_path)
            .arg(ready_len.to_string())
            .arg(&capture_path);
        let client = ServeClient::spawn(&mut command).expect("spawn fake serving process");
        let child = client.child.clone();
        (client, child, response_path, capture_path)
    }

    #[cfg(unix)]
    fn assert_complete_input(path: &std::path::Path) {
        let mut captured = framed_stream::FramedReader::open(
            std::fs::File::open(path).expect("open captured input"),
        )
        .expect("captured preamble");
        captured
            .require_content_type(CONTENT_TYPE)
            .expect("captured content type");
        assert_eq!(
            captured.next_frame().expect("captured END"),
            framed_stream::Frame::End(framed_stream::EndStatus::Complete)
        );
    }

    #[cfg(unix)]
    #[test]
    fn single_shutdown_sends_complete_end_and_reaps_clean_exit() {
        let (mut client, child, response_path, capture_path) =
            spawn_shutdown_fixture("clean-shutdown", false);
        let status = client
            .shutdown_with_timeout(std::time::Duration::from_secs(1))
            .expect("clean serving process shutdown");
        assert!(status.success());
        assert!(
            child.try_wait().expect("poll reaped fixture").is_some(),
            "shutdown must reap before returning"
        );
        assert_complete_input(&capture_path);
        let _ = std::fs::remove_file(response_path);
        let _ = std::fs::remove_file(capture_path);
    }

    #[cfg(unix)]
    #[test]
    fn single_close_bounds_a_child_that_hangs_after_complete_end() {
        let (client, child, response_path, capture_path) =
            spawn_shutdown_fixture("hung-shutdown", true);
        let started = std::time::Instant::now();
        let error = client
            .close_with_timeout(std::time::Duration::from_millis(50))
            .expect_err("hung serving process must hit the deadline");
        assert!(error.to_string().contains("did not exit"), "{error:#}");
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
        let status = child
            .try_wait()
            .expect("poll killed fixture")
            .expect("forced shutdown must reap before returning");
        assert!(!status.success());
        assert_complete_input(&capture_path);
        let _ = std::fs::remove_file(response_path);
        let _ = std::fs::remove_file(capture_path);
    }

    #[test]
    fn pair_compatibility_is_content_identity_not_pile_path() {
        let (left, right) = fake_pair_ready("/models/left.pile", "/different/host/right.pile");
        let ready = compatible_ready(&left, &right).expect("same runtime content");
        assert_eq!(ready.model_identity, left.model_identity);
        assert_eq!(ready.tokenizer_identity, left.tokenizer_identity);
        assert_eq!(ready.pile, left.pile, "pile is rank-0 diagnostics only");
    }

    #[test]
    fn pair_compatibility_refuses_model_or_tokenizer_mismatch() {
        let (left, mut right) = fake_pair_ready("left", "right");
        right.model_identity = "33".repeat(32);
        let error = compatible_ready(&left, &right).expect_err("different model facts");
        assert!(error.to_string().contains("model identity mismatch"));

        let (_, mut right) = fake_pair_ready("left", "right");
        right.tokenizer_identity = "44".repeat(32);
        let error = compatible_ready(&left, &right).expect_err("different tokenizer bytes");
        assert!(error.to_string().contains("tokenizer identity mismatch"));
    }

    #[test]
    fn pair_compatibility_refuses_execution_manifest_mismatch() {
        let (left, mut right) = fake_pair_ready("left", "right");
        right.execution_identity = "44".repeat(32);
        let error = compatible_ready(&left, &right).expect_err("different executable/runtime");
        assert!(
            error.to_string().contains("execution identity mismatch"),
            "{error:#}"
        );
    }

    #[test]
    fn pair_compatibility_refuses_resource_budget_mismatch() {
        let (left, mut right) = fake_pair_ready("left", "right");
        right.prefill_budget /= 2;
        let error = compatible_ready(&left, &right).expect_err("different prefill budgets");
        assert!(error.to_string().contains("prefill budget mismatch"));

        let (_, mut right) = fake_pair_ready("left", "right");
        right.context_budget /= 2;
        let error = compatible_ready(&left, &right).expect_err("different context budgets");
        assert!(error.to_string().contains("context budget mismatch"));
    }

    #[cfg(unix)]
    #[test]
    fn post_ready_rejection_closes_supervisors_then_reaps_both_rank_trees() {
        let (mut rank0, mut rank1) = fake_pair_ready("left", "right");
        rank0.execution_profile = "observed-v1".to_string();
        rank1.execution_profile = "observed-v1".to_string();
        let (command0, pids0, response0) = supervised_ready_fixture(&rank0, "reject-rank0");
        let (command1, pids1, response1) = supervised_ready_fixture(&rank1, "reject-rank1");

        let started = std::time::Instant::now();
        let error =
            ServePair::spawn_with_timeout([command0, command1], std::time::Duration::from_secs(2))
                .err()
                .expect("an observed execution profile must be rejected");
        assert!(error.to_string().contains("execution profile"), "{error:#}");

        let pids = [fixture_pids(&pids0), fixture_pids(&pids1)]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let _cleanup = FixtureProcessCleanup(pids.clone());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while pids.iter().any(|pid| process_exists(*pid)) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        for pid in pids {
            assert!(
                !process_exists(pid),
                "post-READY rejection left supervisor/rank pid {pid} alive"
            );
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(3));

        for path in [pids0, pids1, response0, response1] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[cfg(all(feature = "drive-mind", unix))]
    #[test]
    fn failed_microturn_becomes_terminal_after_one_typed_initialization_batch() {
        use drive::mind::Mind as _;

        let system = "system first";
        let result_delta =
            text_result_delta("cmd", &drive::content::Content::text("output"), false, None);
        let max_tokens = 7;

        // Complete fake server output. `ServeClient::spawn` receives only the
        // preamble + READY prefix at first; the shell fixture releases the
        // partial turn after it has captured the client's whole request.
        let response = SharedSink::default();
        let mut response_writer =
            framed_stream::FramedWriter::open(response.clone(), CONTENT_TYPE, UNIT)
                .expect("response preamble");
        let ready_payload = serde_json::to_vec(&fake_ready("fixture.pile")).expect("READY json");
        response_writer
            .record_as(READY_TYPE, &ready_payload, ready_payload.len() as u64)
            .expect("READY record");
        let ready_len = response.len();
        // A fragment without its TURN id is deliberately not released: exact
        // id/fragment association is the parser's TP-safe trust boundary.
        response_writer
            .text("unassociated")
            .expect("unassociated fragment");
        response_writer
            .finish(framed_stream::EndStatus::Aborted(
                "rank divergence".to_string(),
            ))
            .expect("aborted response");

        // Count the exact prefix observe must send. The writer stays alive at
        // the measurement point, so its eventual drop-only ABORTED trailer is
        // deliberately outside `input_len`.
        let expected_input = SharedSink::default();
        let mut input_writer =
            framed_stream::FramedWriter::open(expected_input.clone(), CONTENT_TYPE, UNIT)
                .expect("input preamble");
        let initial_context = InklingContext::Initialize {
            system: system.to_string(),
            historical_exec_results: vec![ExecResultContext {
                command: "cmd".to_string(),
                content: result_delta.clone(),
            }],
        };
        let context_payload = serde_json::to_vec(&initial_context).expect("CONTEXT json");
        input_writer
            .record_as(CONTEXT_TYPE, &context_payload, context_payload.len() as u64)
            .expect("CONTEXT record");
        let consult_payload = serde_json::to_vec(&Consult::new(1)).expect("CONSULT json");
        input_writer
            .record_as(CONSULT_TYPE, &consult_payload, consult_payload.len() as u64)
            .expect("CONSULT record");
        let input_len = expected_input.len();

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let response_path = std::env::temp_dir().join(format!("mary-failed-turn-response-{nonce}"));
        let capture_path = std::env::temp_dir().join(format!("mary-failed-turn-capture-{nonce}"));
        std::fs::write(&response_path, response.bytes()).expect("write fake response");

        let script = r#"
dd if="$1" bs=1 count="$2" 2>/dev/null
dd of="$3" bs=1 count="$4" 2>/dev/null
dd if="$1" bs=1 skip="$2" 2>/dev/null
exec cat >/dev/null
"#;
        let mut command = std::process::Command::new("sh");
        command
            .arg("-c")
            .arg(script)
            .arg("fake-inkling")
            .arg(&response_path)
            .arg(ready_len.to_string())
            .arg(&capture_path)
            .arg(input_len.to_string());
        let client = ServeClient::spawn(&mut command).expect("spawn fake serving process");
        let child = client.child.clone();
        assert!(child.try_wait().expect("poll fixture").is_none());

        let mut mind = InklingMind::new(client, max_tokens, Some(system.to_string()));
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
        assert!(terminal.contains("rank divergence"), "{terminal}");

        // One typed batch lets the server place tool declaration + system +
        // effort before history, and the sole generation prompt after it.
        let mut captured = framed_stream::FramedReader::open(
            std::fs::File::open(&capture_path).expect("open captured input"),
        )
        .expect("captured preamble");
        captured
            .require_content_type(CONTENT_TYPE)
            .expect("captured content type");
        let framed_stream::Frame::Record(context_record) =
            captured.next_frame().expect("CONTEXT frame")
        else {
            panic!("first input frame was not CONTEXT")
        };
        assert_eq!(context_record.content_type(), CONTEXT_TYPE);
        let context: InklingContext =
            serde_json::from_slice(&context_record.payload).expect("captured CONTEXT");
        assert_eq!(context, initial_context);
        let framed_stream::Frame::Record(consult_record) =
            captured.next_frame().expect("CONSULT frame")
        else {
            panic!("second input frame was not CONSULT")
        };
        assert_eq!(consult_record.content_type(), CONSULT_TYPE);
        let consult: Consult =
            serde_json::from_slice(&consult_record.payload).expect("captured CONSULT");
        assert_eq!(consult.max_tokens, 1);

        let drop_started = std::time::Instant::now();
        drop(mind);
        assert!(drop_started.elapsed() < std::time::Duration::from_secs(3));
        let status = child
            .try_wait()
            .expect("poll failed fixture after mind drop")
            .expect("InklingMind::drop must reap the killed fixture");
        assert!(!status.success(), "observe must kill the live fixture");
        let _ = std::fs::remove_file(response_path);
        let _ = std::fs::remove_file(capture_path);
    }

    #[cfg(all(feature = "drive-mind", unix))]
    #[test]
    fn complete_fake_native_sequence_becomes_an_exact_same_turn_fire() {
        use drive::mind::Mind as _;

        let ready = fake_ready("fixture.pile");
        let ids = ready.special_ids.clone();
        let sequence = vec![
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

        let response = SharedSink::default();
        let mut writer = framed_stream::FramedWriter::open(response.clone(), CONTENT_TYPE, UNIT)
            .expect("response preamble");
        let ready_payload = serde_json::to_vec(&ready).expect("READY json");
        writer
            .record_as(READY_TYPE, &ready_payload, ready_payload.len() as u64)
            .expect("READY record");
        for (turn, (id, fragment)) in sequence.iter().enumerate() {
            writer.text(fragment).expect("token decoder fragment");
            let mut end = fake_end(&[*id]);
            end.turn = turn;
            if turn > 0 {
                end.delta_tokens = 0;
                end.carried = 1;
            }
            if *id == ids.content_model_end_sampling {
                end.stopped = "stop_token".to_string();
            }
            let payload = serde_json::to_vec(&end).expect("TURN json");
            writer
                .record_as(TURN_TYPE, &payload, payload.len() as u64)
                .expect("TURN record");
        }
        writer
            .finish(framed_stream::EndStatus::Complete)
            .expect("complete response");

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let response_path = std::env::temp_dir().join(format!("mary-fire-response-{nonce}"));
        let capture_path = std::env::temp_dir().join(format!("mary-fire-capture-{nonce}"));
        std::fs::write(&response_path, response.bytes()).expect("write fake response");
        let mut command = std::process::Command::new("sh");
        command
            .arg("-c")
            .arg("cat \"$1\"; cat >\"$2\"")
            .arg("fake-inkling")
            .arg(&response_path)
            .arg(&capture_path);
        let client = ServeClient::spawn(&mut command).expect("spawn fake serving process");
        let child = client.child.clone();
        let mut mind = InklingMind::new(client, 32, Some("system".to_string()));
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
        assert_eq!(mind.log().lock().expect("turn log")[0].token_ids.len(), 9);

        drop(mind);
        assert!(
            child
                .try_wait()
                .expect("poll fake server")
                .expect("drop reaps fake server")
                .success()
        );
        let mut captured = framed_stream::FramedReader::open(
            std::fs::File::open(&capture_path).expect("open captured input"),
        )
        .expect("captured preamble");
        captured.require_content_type(CONTENT_TYPE).unwrap();
        let framed_stream::Frame::Record(context) = captured.next_frame().unwrap() else {
            panic!("first frame was not typed initialization")
        };
        assert_eq!(context.content_type(), CONTEXT_TYPE);
        let context: InklingContext = serde_json::from_slice(&context.payload).unwrap();
        assert_eq!(
            context,
            InklingContext::Initialize {
                system: "system".to_string(),
                historical_exec_results: Vec::new(),
            }
        );
        for _ in &sequence {
            let framed_stream::Frame::Record(consult) = captured.next_frame().unwrap() else {
                panic!("microturn frame was not CONSULT")
            };
            assert_eq!(consult.content_type(), CONSULT_TYPE);
            assert_eq!(
                serde_json::from_slice::<Consult>(&consult.payload)
                    .unwrap()
                    .max_tokens,
                1
            );
        }
        assert_eq!(
            captured.next_frame().unwrap(),
            framed_stream::Frame::End(framed_stream::EndStatus::Complete)
        );
        let _ = std::fs::remove_file(response_path);
        let _ = std::fs::remove_file(capture_path);
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
    fn startup_work_is_actually_concurrent() {
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let left = barrier.clone();
        let right = barrier.clone();
        let (left, right) = concurrently(
            move || {
                left.wait();
                Ok(0usize)
            },
            move || {
                right.wait();
                Ok(1usize)
            },
        )
        .expect("workers did not panic");
        assert_eq!(left.expect("left"), 0);
        assert_eq!(right.expect("right"), 1);
    }

    #[test]
    fn zero_startup_deadline_is_rejected_before_launch() {
        let error = ServePair::spawn_with_timeout(
            [RankCommand::local("unused"), RankCommand::local("unused")],
            std::time::Duration::ZERO,
        )
        .err()
        .expect("zero deadline must fail");
        assert!(error.to_string().contains("must be nonzero"));
    }

    #[test]
    fn broker_streams_only_fragment_pairs_that_agree() {
        let (send, receive) = std::sync::mpsc::sync_channel(8);
        send.send(RankEvent::text(0, "one")).unwrap();
        send.send(RankEvent::text(0, " two")).unwrap();
        send.send(RankEvent::text(1, "one")).unwrap();
        send.send(RankEvent::text(1, " two")).unwrap();
        let mut rank0 = fake_end(&[41, 42]);
        rank0.first_token_secs = 0.03;
        rank0.turn_secs = 0.07;
        let mut rank1 = fake_end(&[41, 42]);
        rank1.first_token_secs = 0.05;
        rank1.turn_secs = 0.06;
        send.send(RankEvent::end(0, rank0)).unwrap();
        send.send(RankEvent::end(1, rank1)).unwrap();
        drop(send);
        let mut streamed = Vec::new();
        let end = broker_rank_streams(receive, 4, &mut |text| {
            streamed.push(text.to_string());
            Ok(())
        })
        .expect("matching streams");
        assert_eq!(streamed, ["one", " two"]);
        assert_eq!(end.token_ids, [41, 42]);
        assert_eq!(end.first_token_secs, 0.05);
        assert_eq!(end.turn_secs, 0.07);
    }

    #[test]
    fn broker_refuses_text_divergence_before_releasing_it() {
        let (send, receive) = std::sync::mpsc::sync_channel(8);
        send.send(RankEvent::text(0, "same")).unwrap();
        send.send(RankEvent::text(1, "same")).unwrap();
        send.send(RankEvent::text(0, " left")).unwrap();
        send.send(RankEvent::text(1, " right")).unwrap();
        drop(send);
        let mut streamed = String::new();
        let error = broker_rank_streams(receive, 4, &mut |text| {
            streamed.push_str(text);
            Ok(())
        })
        .expect_err("divergence must fail");
        assert_eq!(error.kind(), ServePairFailure::Divergence);
        assert_eq!(streamed, "same");
        assert_eq!(error.confirmed_extent(), 4);
    }

    #[test]
    fn broker_refuses_equal_text_with_different_token_ids() {
        let (send, receive) = std::sync::mpsc::sync_channel(8);
        send.send(RankEvent::text(0, "same")).unwrap();
        send.send(RankEvent::text(1, "same")).unwrap();
        send.send(RankEvent::end(0, fake_end(&[7]))).unwrap();
        send.send(RankEvent::end(1, fake_end(&[8]))).unwrap();
        drop(send);
        let error = broker_rank_streams(receive, 4, &mut |_| Ok(()))
            .expect_err("token-id mismatch must fail");
        assert_eq!(error.kind(), ServePairFailure::Divergence);
        assert_eq!(error.confirmed_extent(), 4);
    }

    #[test]
    fn broker_preserves_confirmed_extent_when_consumer_panics() {
        let (send, receive) = std::sync::mpsc::sync_channel(8);
        send.send(RankEvent::text(0, "same")).unwrap();
        send.send(RankEvent::text(1, "same")).unwrap();
        send.send(RankEvent::text(0, " panic")).unwrap();
        send.send(RankEvent::text(1, " panic")).unwrap();
        drop(send);
        let error = broker_rank_streams(receive, 4, &mut |text| {
            assert_ne!(text, " panic", "fixture consumer panic");
            Ok(())
        })
        .expect_err("consumer panic must become a typed failure");
        assert_eq!(error.kind(), ServePairFailure::Consumer);
        assert_eq!(error.confirmed_extent(), 4);
    }

    #[test]
    fn broker_bounds_unconfirmed_rank_skew() {
        let (send, receive) = std::sync::mpsc::sync_channel(8);
        for text in ["0", "1", "2"] {
            send.send(RankEvent::text(0, text)).unwrap();
        }
        drop(send);
        let error =
            broker_rank_streams(receive, 2, &mut |_| Ok(())).expect_err("unbounded skew must fail");
        assert_eq!(error.kind(), ServePairFailure::Divergence);
        assert!(error.to_string().contains("agreement window"));
    }

    #[test]
    fn failed_child_is_observed_before_waiting_and_kills_its_peer() {
        let failed = std::process::Command::new("sh")
            .args(["-c", "exit 23"])
            .spawn()
            .expect("spawn failing child");
        let sleeping = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleeping child");
        let children = [
            ChildHandle::new(failed, "failing fixture".to_string()),
            ChildHandle::new(sleeping, "sleeping fixture".to_string()),
        ];
        let started = std::time::Instant::now();
        let statuses = wait_children_result(&children).expect("reap both children");
        assert!(!statuses[0].success());
        assert!(!statuses[1].success());
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
    }

    #[test]
    fn child_wait_deadline_kills_both_live_processes() {
        let children = [
            ChildHandle::new(
                std::process::Command::new("sleep")
                    .arg("30")
                    .spawn()
                    .expect("spawn first sleeping child"),
                "first sleeping fixture".to_string(),
            ),
            ChildHandle::new(
                std::process::Command::new("sleep")
                    .arg("30")
                    .spawn()
                    .expect("spawn second sleeping child"),
                "second sleeping fixture".to_string(),
            ),
        ];
        let started = std::time::Instant::now();
        let error =
            wait_children_result_with_timeout(&children, std::time::Duration::from_millis(50))
                .expect_err("live children must hit the deadline");
        assert!(error.to_string().contains("did not exit"));
        wait_children(&children);
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
    }

    #[test]
    fn remote_command_is_structured_and_quotes_each_word() {
        let command = RankCommand::ssh("spark2-zt", "/srv/mary/inkling serve")
            .env("INK_NOTE", "it's exact")
            .arg("--layers")
            .arg("0:42")
            .command()
            .expect("remote command");
        assert_eq!(command.get_program(), "ssh");
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(&args[..5], ["-T", "-o", "BatchMode=yes", "--", "spark2-zt"]);
        assert!(args[5].starts_with("'exec' 'inkling_serve_pair' '__supervise'"));
        assert!(args[5].contains("'--shutdown-timeout-secs' '45'"));
        assert!(args[5].contains("'--program' '/srv/mary/inkling serve'"));
        assert!(args[5].contains("'--env' 'INK_NOTE=it'\"'\"'s exact'"));
        assert!(args[5].contains("'--arg' '--layers' '--arg' '0:42'"));
    }

    #[cfg(feature = "drive-mind")]
    #[test]
    fn a_typed_drive_result_crosses_the_text_only_seam_deliberately() {
        let content = drive::content::Content::text("output\n[exit 7]");
        assert_eq!(
            text_result_delta("do thing", &content, false, Some(7)),
            "\n$ do thing\noutput\n[exit 7]\n[result status: exit_code=7]\n"
        );
        assert_eq!(
            text_result_delta("do thing", &content, true, Some(7)),
            "\n$ do thing\noutput\n[exit 7]\n[result status: isError=true, exit_code=7]\n"
        );

        // The normal historical text shape stays byte-for-byte: typed results
        // do not grow a second routine status rendering on every turn.
        let ok = drive::content::Content::text("output\n[exit 0]");
        assert_eq!(
            text_result_delta("do thing", &ok, false, Some(0)),
            "\n$ do thing\noutput\n[exit 0]\n"
        );
    }
}
