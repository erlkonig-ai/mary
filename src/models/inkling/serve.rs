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
//! CONTENT (context going in, a token coming out), and a record that overrides
//! it is CONTROL. Three overrides exist, and there is no second socket, no
//! length-prefixed sidecar and no JSON-lines mode:
//!
//! | content type | direction | meaning |
//! |---|---|---|
//! | [`READY_TYPE`] | serve → client | the model is loaded; here is what it is |
//! | [`CONSULT_TYPE`] | client → serve | the delta is complete — produce a turn |
//! | [`TURN_TYPE`] | serve → client | the turn is over; here is how it went |
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
//! No tool-calling, no conversation loop, no sampling knobs, no HTTP, no
//! sandbox. A serving process serves turns. The tokenizer is on the SERVE side
//! (text on the wire, never ids) because drive's `Mind` seam explicitly asks for
//! no tokenizer — the loop never teacher-forces tokens.
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
//! ## What the FAN-OUT PROXY still needs, precisely
//!
//! It speaks THIS protocol on both sides — two [`ServeClient`]s upstream, an
//! `inkling_serve`-shaped server downstream — so `drive` cannot tell the
//! difference and [`Ready::partial`] becomes `false` for the first time. Five
//! things are missing, and none of them is in this file:
//!
//! 1. **`Session::load` has to accept a group it did not form.** `Group::form`
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
/// Control record, serve → client: the turn is over, and here is how it went.
pub const TURN_TYPE: &str = "application/vnd.mary.inkling-turn+json";

/// What the serving process is, announced once when the weights are up.
///
/// Sent AFTER the load rather than in the preamble, because the load is minutes
/// and the preamble is the handshake: a client that got this in the preamble
/// could not tell "starting" from "ready", which is the one thing it needs to
/// know before it asks for a turn.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Ready {
    /// The pile the weights came from.
    pub pile: String,
    /// The layer range this rank runs, `[lo, hi)`.
    pub layers: [usize; 2],
    /// How many layers the whole stack has.
    pub stack: usize,
    /// Whether this is a STRICT SUBRANGE — if so its tokens are diagnostic, not
    /// the model's. Stated rather than inferred.
    pub partial: bool,
    /// Effective vocabulary width the head is sliced to.
    pub vocab: usize,
    /// Wall-clock seconds `Session::load` took. The number a serving process
    /// exists to pay ONCE.
    pub load_secs: f64,
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
    /// tokenising the delta, `extend`/`prefill` over it, and one forward. On
    /// turn 0 this is the prompt's prefill; on every turn after it, it is what
    /// the KV cache saves. THE framing rule: seconds per FIRST TOKEN OF A TURN,
    /// on one box, over the layer range in [`Ready::layers`] — not per token and
    /// not per turn.
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

// ── the client half ─────────────────────────────────────────────────────────

/// A running `inkling_serve`, with a framed stream in each direction.
///
/// Deliberately synchronous and strictly alternating: the client writes context
/// and one consult record, then reads until the turn ends. Neither side ever
/// writes while the other is writing, so the two-pipe deadlock `drive::stream`
/// spawns a thread to avoid cannot arise here and no thread is spawned.
pub struct ServeClient {
    child: std::process::Child,
    writer: Option<framed_stream::FramedWriter<std::process::ChildStdin>>,
    reader: framed_stream::FramedReader<std::process::ChildStdout>,
    ready: Ready,
    label: String,
}

impl ServeClient {
    /// Start `command` and wait for it to say it is READY.
    ///
    /// This blocks for the whole model load — minutes on a real range — because
    /// there is nothing useful a caller can do with a half-loaded model and a
    /// client that returned early would only move the wait to the first turn,
    /// where it would look like a slow turn instead of a slow start.
    pub fn spawn(command: &mut std::process::Command) -> Result<Self> {
        let label = command.get_program().to_string_lossy().to_string();
        let mut child = command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            // Inherited on purpose: the serving process's load diagnostics are
            // the caller's business, and they must not be on the protocol's fd.
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .with_context(|| format!("start the serving process {label}"))?;
        let stdin = child.stdin.take().context("serving process has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("serving process has no stdout")?;
        // Open the WRITER first on both sides: each preamble is written before
        // either side reads, so neither blocks waiting for the other's.
        let writer = framed_stream::FramedWriter::open(stdin, CONTENT_TYPE, UNIT)
            .context("write the serving process's input preamble")?;
        let mut reader = framed_stream::FramedReader::open(stdout)
            .context("read the serving process's output preamble")?;
        reader.require_content_type(CONTENT_TYPE)?;
        let ready = match reader.next_frame().context("wait for the model to load")? {
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
        Ok(Self {
            child,
            writer: Some(writer),
            reader,
            ready,
            label,
        })
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
    /// A byte-level BPE token can be a PARTIAL UTF-8 sequence, so the serving
    /// process decodes the turn's ids as a growing prefix and emits the suffix
    /// that is new. When a later token changes bytes that were already emitted
    /// (a replacement character resolving into the character it stood for), the
    /// prefix property is broken and the producer declares a GAP naming it
    /// rather than silently re-emitting. A gap here is surfaced as an error
    /// carrying the reason, because a consumer that has already SPOKEN the
    /// wrong bytes cannot un-speak them and should be told.
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
        self.end_input()?;
        self.child
            .wait()
            .with_context(|| format!("wait for {} to exit", self.label))
    }

    /// Kill the serving process. Its output stream reads as truncated, which is
    /// the honest report: it was killed, it did not finish.
    pub fn kill(&mut self) -> Result<()> {
        self.child.kill().context("kill the serving process")
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
/// `Payload::Result` events are the untrusted output of a command the mind ran.
/// They ARE the delta and they are fed to the model as context. Drive never
/// scans them for intent and neither does this: they go in as text.
///
/// # The decision
///
/// Always [`drive::mind::Disposition::NoAction`], covering the span it was
/// shown. This backend does not derive commands: tool-calling is a lane of its
/// own and half of it is worse than none. What matters for the seam is that the
/// anti-repression invariant holds — every consultation leaves an audited trace
/// bound to the world it saw — and it does.
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
    /// System prompt, fed as the very first delta so the model has somewhere to
    /// start. It is context like any other; nothing here templates a chat.
    system: Option<String>,
    turns: usize,
    label: String,
    /// Per-turn numbers, shared so the caller can report them after the run.
    log: std::sync::Arc<std::sync::Mutex<Vec<TurnEnd>>>,
    proofs: std::sync::Arc<std::sync::Mutex<Vec<StreamProof>>>,
}

#[cfg(feature = "drive-mind")]
impl InklingMind {
    /// Wrap a running serving process as a mind.
    pub fn new(client: ServeClient, max_tokens: usize, system: Option<String>) -> Self {
        let label = match client.ready().partial {
            true => "inkling(partial)".to_string(),
            false => "inkling".to_string(),
        };
        Self {
            client,
            voice: std::sync::Arc::new(std::sync::Mutex::new(None)),
            buffer: drive::mind::MonologueBuffer::with_cap(64 * 1024),
            scanned_abs: 0,
            max_tokens,
            system,
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
        if let Err(error) = self.client.end_input() {
            eprintln!("inkling_serve: could not end the input stream cleanly: {error:#}");
        }
        match self.client.child.wait() {
            Ok(status) => eprintln!("inkling_serve: the serving process exited: {status}"),
            Err(error) => {
                eprintln!("inkling_serve: could not wait for the serving process: {error}")
            }
        }
    }
}

#[cfg(feature = "drive-mind")]
impl drive::mind::Mind for InklingMind {
    fn observe(&mut self, view: drive::world::MergedView<'_>) -> drive::mind::Turn {
        match self.turn(view.events, view.watermark) {
            Ok(turn) => turn,
            Err(error) => {
                // A backend that cannot answer must still leave a trace: the
                // anti-repression invariant is not suspended because the model
                // broke. The span is the coverage it was shown; the rationale
                // carries the failure, so the pile records WHY a turn was empty.
                eprintln!("inkling_serve: {error:#}");
                let span_end = self.buffer.end_offset();
                let span_start = self.scanned_abs.min(span_end);
                self.scanned_abs = span_end;
                drive::mind::Turn::silent(drive::mind::Decision::no_action(
                    span_start,
                    span_end,
                    view.watermark,
                    format!("inkling serving process failed: {error:#}"),
                ))
            }
        }
    }

    fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(feature = "drive-mind")]
impl InklingMind {
    /// One turn, with the failure path lifted out so `observe` can stay total.
    fn turn(
        &mut self,
        events: &[drive::world::Event],
        watermark: drive::world::Coord,
    ) -> Result<drive::mind::Turn> {
        // ── read the world ──────────────────────────────────────────────────
        for event in events {
            match &event.payload {
                // Coordinates only. The model already attended to these tokens:
                // its own `step` fed back all but each turn's last, and the
                // serving process carries that last one into the next turn's
                // delta. See this type's doc for why both halves have to be
                // true, and what happened while only one was.
                drive::world::Payload::Monologue(text) => self.buffer.push_free(text),
                // The delta: untrusted text from a sandbox, in as text.
                drive::world::Payload::Result { command, rendered } => {
                    self.client.feed(&format!("\n$ {command}\n{rendered}\n"))?;
                }
            }
        }
        if let Some(system) = self.system.take() {
            self.client.feed(&system)?;
        }

        // ── consult, streaming every token into the voice as it arrives ─────
        let voice = self.voice.lock().expect("voice slot").clone();
        let mut said = String::new();
        let mut tokens = 0usize;
        let mut tokens_at_first_return = None;
        let turn = self.turns;
        let end = self
            .client
            .consult(&Consult::new(self.max_tokens), |text| {
                said.push_str(text);
                tokens += 1;
                if let Some(voice) = &voice {
                    // One record per token, flushed, into the stream the shell
                    // opened. This is `Shell::claim_voice`'s finer grain: the
                    // faculty starts on the first word of the sentence.
                    voice.say(text)?;
                    if tokens_at_first_return.is_none() && voice.report().records > 0 {
                        // The faculty has already produced output and this turn is
                        // demonstrably still running: that IS the streaming proof,
                        // taken from inside the turn rather than after it.
                        tokens_at_first_return = Some(tokens);
                    }
                }
                Ok(())
            })?;
        let records_at_end = voice.as_ref().map(|v| v.report().records).unwrap_or(0);
        self.proofs.lock().expect("proof log").push(StreamProof {
            turn,
            tokens,
            tokens_at_first_return,
            records_at_end,
        });
        self.log.lock().expect("turn log").push(end);
        self.turns += 1;

        // ── the audited outcome ─────────────────────────────────────────────
        //
        // The span is the coverage this consultation was shown, in the loop's
        // own session-absolute monologue coordinates — which is what makes it
        // verifiable rather than self-attested.
        let span_end = self.buffer.end_offset();
        let span_start = self
            .scanned_abs
            .max(self.buffer.base_offset())
            .min(span_end);
        self.scanned_abs = span_end;
        Ok(drive::mind::Turn::new(
            said,
            drive::mind::Decision::no_action(
                span_start,
                span_end,
                watermark,
                "inkling: consulted over the released world; this backend derives no commands",
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    /// The three control types must be distinct from each other and from the
    /// stream's own type, or a control record would read as content.
    #[test]
    fn the_control_types_are_distinct_from_content() {
        let types = [READY_TYPE, CONSULT_TYPE, TURN_TYPE];
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
}
