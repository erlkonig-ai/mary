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
    /// Exact token ids generated by the rank.
    ///
    /// A tensor-parallel pair compares these before it accepts a turn. Text is
    /// also compared fragment by fragment for streaming, but two different
    /// byte-level tokens can decode to the same text; ids are the unambiguous
    /// agreement signal. The paired downstream proxy clears this field before
    /// forwarding the historical `inkling_serve` turn shape.
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
            let _ = self.child.kill();
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
            let _ = self.child.kill();
        }
    }
}

impl ServeClient {
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
        if let Err(error) = self.end_input() {
            let _ = self.child.kill();
            let _ = self.child.wait();
            return Err(error);
        }
        self.child.wait()
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
                if let Ok(rank) = &rank0 {
                    let _ = rank.child.kill();
                    let _ = rank.child.wait();
                }
                if let Ok(rank) = &rank1 {
                    let _ = rank.child.kill();
                    let _ = rank.child.wait();
                }
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

        let children = [launched0.child.clone(), launched1.child.clone()];
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

        let mut clients: [Option<ServeClient>; 2] = [None, None];
        let mut startup_error = None;
        for _ in 0..2 {
            let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) else {
                startup_error.get_or_insert_with(|| {
                    format!("serving pair did not become ready within {startup_timeout:?}")
                });
                kill_children(&children);
                break;
            };
            match receive.recv_timeout(remaining) {
                Ok((rank, Ok(client))) => clients[rank] = Some(client),
                Ok((rank, Err(error))) => {
                    startup_error.get_or_insert_with(|| {
                        format!("rank {rank} did not become ready: {error:#}")
                    });
                    kill_children(&children);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    startup_error.get_or_insert_with(|| {
                        format!("serving pair did not become ready within {startup_timeout:?}")
                    });
                    kill_children(&children);
                    break;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    startup_error.get_or_insert_with(|| {
                        "rank startup workers disappeared before READY".to_string()
                    });
                    kill_children(&children);
                    break;
                }
            }
        }
        let worker0_panicked = worker0.join().is_err();
        let worker1_panicked = worker1.join().is_err();
        if worker0_panicked || worker1_panicked {
            startup_error.get_or_insert_with(|| "a rank startup worker panicked".to_string());
            kill_children(&children);
        }
        if let Some(error) = startup_error {
            wait_children(&children);
            anyhow::bail!(error);
        }
        let rank0 = clients[0].take().context("rank 0 returned no client")?;
        let rank1 = clients[1].take().context("rank 1 returned no client")?;
        let ready = match compatible_ready(rank0.ready(), rank1.ready()) {
            Ok(ready) => ready,
            Err(error) => {
                kill_children(&children);
                wait_children(&children);
                return Err(error);
            }
        };
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
        for (name, identity) in [
            ("model", ready.model_identity.as_str()),
            ("tokenizer", ready.tokenizer_identity.as_str()),
        ] {
            anyhow::ensure!(
                identity.len() == 64 && identity.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "rank {rank} announced an invalid {name} identity {identity:?}"
            );
        }
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
    Ok(Ready {
        pile: rank0.pile.clone(),
        model_identity: rank0.model_identity.clone(),
        tokenizer_identity: rank0.tokenizer_identity.clone(),
        layers: rank0.layers,
        stack: rank0.stack,
        partial: false,
        vocab: rank0.vocab,
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
                (RankPart::End(left), RankPart::End(right)) => {
                    compare_turn_ends(&left, &right).map_err(|message| {
                        ServePairError::new(ServePairFailure::Divergence, confirmed_extent, message)
                    })?;
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

fn wait_children(children: &[ChildHandle; 2]) {
    let _ = wait_children_result_with_timeout(children, std::time::Duration::from_secs(10));
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
        if let Err(error) = self.client.end_input() {
            eprintln!("inkling_serve: could not end the input stream cleanly: {error:#}");
            if let Err(kill) = self.client.kill() {
                eprintln!("inkling_serve: could not kill the failed serving process: {kill:#}");
            }
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
impl InklingMind {
    /// One turn, with the failure path lifted out so `observe` can stay total.
    fn turn(
        &mut self,
        events: &[drive::world::Event],
        watermark: drive::world::Coord,
    ) -> std::result::Result<drive::mind::Turn, FailedTurn> {
        // The system prompt is position zero of the model's logical sequence.
        // Released command results come after it even when the first view
        // already contains results.
        if let Some(system) = self.system.take() {
            self.client.feed(&system)?;
        }

        // ── read the world ──────────────────────────────────────────────────
        for event in events {
            match &event.payload {
                // Coordinates only. The model already attended to these tokens:
                // its own `step` fed back all but each turn's last, and the
                // serving process carries that last one into the next turn's
                // delta. See this type's doc for why both halves have to be
                // true, and what happened while only one was.
                drive::world::Payload::Monologue(text) => self.buffer.push_free(text),
                // The Session is text-token-only today, so drive's typed result
                // crosses its deliberate text projection seam here. Abnormal
                // structural status is stated rather than silently discarded.
                drive::world::Payload::Result {
                    command,
                    content,
                    is_error,
                    exit_code,
                } => {
                    self.client
                        .feed(&text_result_delta(command, content, *is_error, *exit_code))?;
                }
            }
        }

        // ── consult, streaming every token into the voice as it arrives ─────
        let voice = self.voice.lock().expect("voice slot").clone();
        let mut said = String::new();
        let mut tokens = 0usize;
        let mut tokens_at_first_return = None;
        let turn = self.turns;
        let end = match self.client.consult(&Consult::new(self.max_tokens), |text| {
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
        }) {
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

        // ── the audited outcome ─────────────────────────────────────────────
        //
        // The span is the coverage this consultation was shown, in the loop's
        // own session-absolute monologue coordinates — which is what makes it
        // verifiable rather than self-attested.
        let (span_start, span_end) = self.finish_coverage();
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

    #[cfg(all(feature = "drive-mind", unix))]
    #[derive(Clone, Default)]
    struct SharedSink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    #[cfg(all(feature = "drive-mind", unix))]
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

    #[cfg(all(feature = "drive-mind", unix))]
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

    fn fake_ready(pile: &str) -> Ready {
        Ready {
            pile: pile.to_string(),
            model_identity: "11".repeat(32),
            tokenizer_identity: "22".repeat(32),
            layers: [0, 42],
            stack: 42,
            partial: false,
            vocab: 200_058,
            load_secs: 1.0,
        }
    }

    #[test]
    fn pair_compatibility_is_content_identity_not_pile_path() {
        let left = fake_ready("/models/left.pile");
        let right = fake_ready("/different/host/right.pile");
        let ready = compatible_ready(&left, &right).expect("same runtime content");
        assert_eq!(ready.model_identity, left.model_identity);
        assert_eq!(ready.tokenizer_identity, left.tokenizer_identity);
        assert_eq!(ready.pile, left.pile, "pile is rank-0 diagnostics only");
    }

    #[test]
    fn pair_compatibility_refuses_model_or_tokenizer_mismatch() {
        let left = fake_ready("left");
        let mut right = left.clone();
        right.model_identity = "33".repeat(32);
        let error = compatible_ready(&left, &right).expect_err("different model facts");
        assert!(error.to_string().contains("model identity mismatch"));

        let mut right = left.clone();
        right.tokenizer_identity = "44".repeat(32);
        let error = compatible_ready(&left, &right).expect_err("different tokenizer bytes");
        assert!(error.to_string().contains("tokenizer identity mismatch"));
    }

    #[cfg(all(feature = "drive-mind", unix))]
    #[test]
    fn failed_stream_becomes_one_terminal_turn_after_system_then_results() {
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
        response_writer.text("confirmed ").expect("first fragment");
        response_writer.text("β").expect("second fragment");
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
        input_writer.text(system).expect("system frame");
        input_writer.text(&result_delta).expect("result frame");
        let consult_payload = serde_json::to_vec(&Consult::new(max_tokens)).expect("CONSULT json");
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
cat >/dev/null
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

        assert_eq!(turn.said, "confirmed β");
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
        let status = child.wait().expect("failed fixture was reaped");
        assert!(
            !status.success(),
            "observe must kill the otherwise-live fixture"
        );

        // The exact request proves the model's position zero is the system
        // prompt even when the first released view already carries a result.
        let mut captured = framed_stream::FramedReader::open(
            std::fs::File::open(&capture_path).expect("open captured input"),
        )
        .expect("captured preamble");
        captured
            .require_content_type(CONTENT_TYPE)
            .expect("captured content type");
        let framed_stream::Frame::Record(system_record) =
            captured.next_frame().expect("system frame")
        else {
            panic!("first input frame was not the system record")
        };
        assert_eq!(system_record.content_type(), CONTENT_TYPE);
        assert_eq!(system_record.text().expect("system text"), system);
        let framed_stream::Frame::Record(result_record) =
            captured.next_frame().expect("result frame")
        else {
            panic!("second input frame was not the result record")
        };
        assert_eq!(result_record.content_type(), CONTENT_TYPE);
        assert_eq!(result_record.text().expect("result text"), result_delta);
        let framed_stream::Frame::Record(consult_record) =
            captured.next_frame().expect("CONSULT frame")
        else {
            panic!("third input frame was not CONSULT")
        };
        assert_eq!(consult_record.content_type(), CONSULT_TYPE);
        let consult: Consult =
            serde_json::from_slice(&consult_record.payload).expect("captured CONSULT");
        assert_eq!(consult.max_tokens, max_tokens);

        drop(mind);
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
        send.send(RankEvent::end(0, fake_end(&[41, 42]))).unwrap();
        send.send(RankEvent::end(1, fake_end(&[41, 42]))).unwrap();
        drop(send);
        let mut streamed = Vec::new();
        let end = broker_rank_streams(receive, 4, &mut |text| {
            streamed.push(text.to_string());
            Ok(())
        })
        .expect("matching streams");
        assert_eq!(streamed, ["one", " two"]);
        assert_eq!(end.token_ids, [41, 42]);
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
