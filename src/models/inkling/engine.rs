//! **The model, in this process** — one [`Session`], the checkpoint's
//! tokenizer, the incremental detokenizer, the one-token carry, and the link
//! that keeps the other rank in step.
//!
//! [`Session`]: super::session::Session
//!
//! This is what `inkling_serve` was, minus the process boundary. Its input loop
//! decoded framed-stream records into `Session` calls; there are no records now,
//! so the decode is gone and the calls are methods. Everything that was MODEL
//! logic — the delta, the carry, the detokenizer's cross-turn state, the
//! reinitialization boundary, the execution manifest — is here, unchanged in
//! substance.
//!
//! # What the collapse deleted, and what it did not
//!
//! Deleted: `inkling_serve` (1206 lines), `inkling_serve_pair` (788),
//! `inkling_serve_gate` (770), the nine wire content types, `ServeClient`,
//! `ServePair` and the `ssh` launch path. Also deleted, and worth naming
//! separately because it was load-bearing and is now merely absent:
//! `claim_stdout`. The serving process had to `dup` fd 1 away and point every
//! `println!` at stderr, because *"a single stray line in the middle of a framed
//! stream is not a cosmetic problem: it is a corrupt record"*. Nothing owns
//! stdout as a protocol any more, so the load path's diagnostics are just
//! diagnostics.
//!
//! Not deleted: [`super::tp`] (966 lines) and [`super::tpcomm`] (527) —
//! the within-layer split and the NCCL rendezvous — and
//! [`super::session`] (2052). The collapse removed a wire between the mind and
//! the model; it did not touch the wire between the two halves of the model,
//! which was always the right one.
//!
//! # Two ranks, one binary, and who decides
//!
//! Both boxes run this program with the SAME arguments. `tpcomm::elect_rank` decides
//! which is which by comparing this box's own addresses to the
//! `--tp-rendezvous` host — see its documentation for why that is the right
//! discriminator and what the alternatives cost.
//!
//! * **Rank 0** builds an [`Engine`], hands it to `InklingMind`, and runs Drive.
//!   Every `Session` pass it makes, it first NAMES on the rank link.
//! * **Rank 1** builds a `Session` and calls [`Follower::follow`]. It has no tokenizer, no
//!   context codec, no detokenizer, no pile and no Drive loop; it replays the
//!   passes rank 0 names and discards the tokens, which are identical to rank
//!   0's by construction ([`super::tpcomm::Group::argmax_across`] reads back the
//!   same all-reduced buffer on every rank).
//!
//! The old proxy mirrored INPUT BYTES to two ranks and relied on both deriving
//! the same passes — its own third invariant, and the reason it had to warn that
//! "the proxy must feed both ranks the SAME context bytes and the same
//! `max_tokens`, not merely the same turns". Naming the passes removes that
//! derivation entirely.
//!
//! # One leak, on purpose
//!
//! `tokenizers::DecodeStream` borrows its `Tokenizer`, and the decoder's state
//! belongs to the whole logical sequence rather than to one turn — so an
//! [`Engine`] has to hold both, which a plain struct cannot do. The tokenizer is
//! therefore `Box::leak`ed once at load and the decoder is a `'static` closure
//! over it. That is honest for what this is: a resident process that holds a
//! 121 GiB model until it exits, leaking a few megabytes of tokenizer exactly
//! once. The alternative — `unsafe` self-reference, or re-decoding the whole
//! sequence per token — is worse in both directions.

use anyhow::{Context, Result};

use super::resident::{
    Consult, ContextPlacement, ContextPreflight, ContextPreflighted, ExecutionManifest,
    InklingContext, InklingContextCodec, Model, Ready, Reinitialized, TurnEnd, context_preflight, SenseMedia,
};
use super::session::{Session, SessionConfig};
use super::tp::Tp;
use super::tpcomm::{Group, Pass, transport_note};

/// One decoded text chunk per generated id, or `None` while an incomplete
/// UTF-8 sequence waits for a later token.
///
/// A boxed closure rather than a named `DecodeStream`, because the stream's
/// type carries five tokenizer-component generics and its lifetime; the
/// behaviour is one function and this is the shape every caller already wanted
/// (the deleted `serve_turn` took exactly this).
type Detokenizer = Box<dyn FnMut(u32) -> Result<Option<String>> + Send>;

/// How to load one rank of the model.
pub struct EngineConfig {
    /// The model collection: weights, config.json AND the tokenizer graph.
    pub pile: std::path::PathBuf,
    /// Layers this rank runs. A tensor-parallel rank must run all of them.
    pub layers: Option<std::ops::Range<usize>>,
    /// Maximum token rows one prefill pass processes at once.
    pub prefill_budget: Option<usize>,
    /// Maximum positions the session may retain across all turns.
    pub context_budget: Option<usize>,
    /// Rank, world and rendezvous, once `tpcomm::elect_rank` has decided them.
    pub tensor_parallel: Option<TensorParallel>,
    /// Refuse execution-changing environment overrides and announce
    /// `sealed-v1`.
    pub sealed: bool,
    /// The key that signs a learned version written back into the model
    /// graph (`Model::persist_learned`). `None`: nothing learned is ever
    /// written back.
    pub signing_key: Option<std::path::PathBuf>,
}

/// The tensor-parallel placement of one rank.
pub struct TensorParallel {
    pub tp: Tp,
    /// Rank 0's `HOST:PORT` on the fast fabric. Rank 0 binds it; every other
    /// rank dials it. It is also the thing rank election compares against.
    pub rendezvous: String,
}

/// A loaded rank-0 model: the [`Model`] `InklingMind` consults.
pub struct Engine {
    session: Session,
    codec: InklingContextCodec,
    /// Leaked once at load; see the module header. Held so a reinitialization
    /// can build a fresh decode stream over the same tokenizer.
    tokenizer: &'static tokenizers::Tokenizer,
    decode: Detokenizer,
    ready: Ready,
    context_budget: usize,
    /// Context tokenized and waiting for the next consult to attend to it.
    delta: Vec<usize>,
    /// The dMel levels behind the audio slots in `delta`, in order; staged
    /// into every rank's Session ahead of the pass that consumes them.
    delta_audio: Vec<u8>,
    /// The patches behind the image slots in `delta`, likewise.
    delta_vision: Vec<u8>,
    /// The token the previous turn EMITTED and never fed back, waiting for the
    /// next pass to put it in the cache. `None` is also "no turn has run yet",
    /// which is the same fact as "nothing is prefilled".
    carry: Option<usize>,
    turn: usize,
    /// Running identity of this rank's sequence: every token id every pass has
    /// returned, and the position after it. Compared with the peer's at every
    /// logical-turn boundary — see [`Model::agree_sequence`].
    digest: blake3::Hasher,
    /// Set once the run has ended or failed. A terminated engine must not enter
    /// another collective, because its peer is no longer in one.
    terminated: bool,
    /// The key that signs a learned version written back (`Model::persist_learned`).
    signing_key: Option<std::path::PathBuf>,
    /// Score every delta as it is attended to (see [`TurnEnd::delta_nll`]).
    /// On by default; `INK_SCORE=0` turns it off.
    score: bool,
    /// The pending delta is her own cover and history -- an Initialize
    /// context, at wake or at a replacement -- not new experience: attended
    /// to, never scored, never learned from. Measured 2026-09-05: scoring a
    /// 58k-position wake cost a fifth of the wake, and a learning run would
    /// have learned her own memories back every wake.
    delta_unscored: bool,
}

/// A rank-1 model: a `Session` and nothing else.
///
/// It holds no tokenizer, no codec, no detokenizer and no pile. Everything it
/// needs to stay in lockstep arrives as a [`Pass`].
pub struct Follower {
    session: Session,
    digest: blake3::Hasher,
    ready: Ready,
}

// ── loading ─────────────────────────────────────────────────────────────────

/// Load one rank, forming the tensor-parallel group first if there is one.
///
/// The layer-range rule INVERTS between the two cases and that is enforced
/// inside `Session`, not here: a single-rank load requires a strict subrange
/// (144 GiB of weights do not fit a 121 GiB box), while a tensor-parallel rank
/// requires exactly `0:num_hidden_layers`, because each rank holds half of
/// EVERY tensor rather than all of some layers. Getting it backwards is a
/// refusal at load rather than a wrong answer, which is the good failure.
pub fn load(config: EngineConfig) -> Result<Loaded> {
    if config.sealed {
        reject_sealed_environment()?;
    }
    let execution_profile = match config.sealed {
        true => "sealed-v1",
        false => "observed-v1",
    };
    let mut execution_manifest = begin_execution_manifest(execution_profile)?;

    let mut session_config = SessionConfig::new(&config.pile);
    if let Some(layers) = config.layers.clone() {
        session_config = session_config.layers(layers);
    }
    if let Some(budget) = config.prefill_budget {
        session_config.prefill_budget = budget;
        // There is one bounded-width append path, not an independently tuned
        // second strategy. A caller narrowing prefill chunks narrows later
        // multi-row extends to the same admitted width; single-token decode is
        // unchanged.
        session_config.extend_batch = session_config.extend_batch.min(budget);
    }
    // The window the resident is built for, not the prefill width. Historically
    // there was one length axis and an unset context budget admitted only as
    // many positions as one prefill pass; that stopped a 200-turn run at turn
    // 77, cleanly, and JP's answer was to fix the number the design already
    // fixes: a million positions. Its KV is priced at load (7.9 GiB at NVFP4
    // with 7 of 42 layers global, halved under tensor parallelism), which is
    // what the budget is FOR. A caller may still name a smaller one.
    session_config.context_budget = config.context_budget.unwrap_or(1 << 20);
    let prefill_budget = session_config.prefill_budget;
    let context_budget = session_config.context_budget;
    let extend_batch = session_config.extend_batch;
    // Select once before any CUDA client. Group/Session observe this same
    // process-global value later; sealed-v1 has already refused an ambient
    // CUBECL_MEMORY_CONFIG override, so the effective baseline is fixed.
    let allocator = super::pool::choose_memory_config();
    let (tp_rank, tp_world) = config
        .tensor_parallel
        .as_ref()
        .map(|parallel| (Some(parallel.tp.rank()), parallel.tp.world()))
        .unwrap_or((None, 1));

    let loaded = std::time::Instant::now();
    let mut session = match config.tensor_parallel {
        None => Session::load(session_config).context("load the model")?,
        Some(tensor_parallel) => {
            eprintln!(
                "inkling: forming tensor rank {} of {} at {}",
                tensor_parallel.tp.rank(),
                tensor_parallel.tp.world(),
                tensor_parallel.rendezvous,
            );
            let group = Group::form_default(tensor_parallel.tp, &tensor_parallel.rendezvous)
                .context("form the tensor-parallel group")?;
            group
                .warm()
                .context("warm and verify the tensor-parallel group")?;
            eprintln!("inkling: tensor group paired ({})", transport_note());
            Session::load_with_group(session_config, group)
                .context("load this tensor-parallel rank")?
        }
    };
    let load_secs = loaded.elapsed().as_secs_f64();

    // The tokenizer comes out of the model pile itself: the model collection
    // carries its tokenizer graph (ingested and proven by
    // `inkling_tokenizer_gate`), so the pile is the model, tokenizer included,
    // and no side file can drift from it. Both views are built here -- the
    // whole tokenizer for structure and the detokenizer, the content-only
    // one for everything untrusted -- and the identity is the graph's own.
    // Both ranks build them: rank 1 for the ids and the identity alone.
    let (tokenizer_identity, tokenizer, content) = {
        let source = session.source();
        let facts = source.facts();
        let mut found = crate::tokenizer::find_tokenizers(facts);
        let tok_id = found.next().with_context(|| {
            format!(
                "the model collection in {} carries no tokenizer graph; ingest it with \
                 inkling_tokenizer_gate <tokenizer.json> <pile> --signing-key <key>",
                config.pile.display()
            )
        })?;
        anyhow::ensure!(
            found.next().is_none(),
            "the model collection in {} carries more than one tokenizer",
            config.pile.display()
        );
        let tokenizer = crate::tokenizer::build_tokenizer(facts, source.reader(), tok_id)
            .map_err(|error| anyhow::anyhow!("build the tokenizer from the model graph: {error}"))?;
        let content =
            crate::tokenizer::build_tokenizer_with_added(facts, source.reader(), tok_id, false)
                .map_err(|error| {
                    anyhow::anyhow!("build the content-only tokenizer from the model graph: {error}")
                })?;
        (format!("{tok_id:X}"), tokenizer, content)
    };
    let runtime_facts = RuntimeFacts::observe();

    let range = session.layer_range();
    let model_identity = hex_identity(session.model_identity());
    for (name, value) in [
        ("model-identity", model_identity.as_str()),
        ("tokenizer-identity", tokenizer_identity.as_str()),
        ("tp-role-schema", "rank-normalized-v1"),
        ("allocator", allocator.env_value()),
        // INK_POOL_CLEANUP is rejected in sealed-v1, so this is the effective
        // default selected by CleanupPolicy::choose.
        ("allocator-cleanup", "when-stranded"),
        (
            "burn-timing-autotune",
            if cfg!(feature = "inkling-cuda-autotune") {
                "enabled"
            } else {
                "disabled"
            },
        ),
    ] {
        execution_manifest.field(name, value.as_bytes());
    }
    execution_manifest.usize("tp-world", tp_world);
    execution_manifest.usize("layer-lo", range.start);
    execution_manifest.usize("layer-hi", range.end);
    execution_manifest.usize(
        "stack-layers",
        session.config().text_config.num_hidden_layers,
    );
    execution_manifest.usize(
        "effective-vocab",
        session.config().text_config.effective_vocab(),
    );
    execution_manifest.usize("prefill-budget", prefill_budget);
    execution_manifest.usize("context-budget", context_budget);
    execution_manifest.usize("extend-batch", extend_batch);
    runtime_facts.add_to(&mut execution_manifest);
    let execution_identity = execution_manifest.finish_hex();

    // The codec is rank 0's alone: rank 1 never sees text. Build it anyway
    // before branching, because it is also where the special ids come from and
    // both ranks' READY records must agree on the execution identity that the
    // tokenizer identity feeds. Rank 1 drops it immediately.
    let codec = InklingContextCodec::from_views(&tokenizer, content)
        .context("build the context codec from the model graph's tokenizer")?;
    // Her shell declares no tools, so the template's tool-call block is not
    // part of her grammar: the token that opens it is never chosen, on every
    // rank alike. The parser reads text and thinking only.
    session.forbid([codec.special_ids().content_invoke_tool_json as usize]);

    let ready = Ready {
        pile: config.pile.display().to_string(),
        model_identity,
        tokenizer_identity,
        special_ids: codec.special_ids().clone(),
        execution_profile: execution_profile.to_string(),
        execution_identity,
        execution_unavailable: runtime_facts.unavailable,
        tp_rank,
        tp_world,
        layers: [range.start, range.end],
        stack: session.config().text_config.num_hidden_layers,
        partial: session.is_partial_stack(),
        vocab: session.config().text_config.effective_vocab(),
        prefill_budget,
        context_budget,
        load_secs,
    };
    eprintln!(
        "inkling: ready in {load_secs:.1}s — layers {}..{} of {}{}, {} {}",
        ready.layers[0],
        ready.layers[1],
        ready.stack,
        match ready.partial {
            true => " (PARTIAL STACK: diagnostic tokens, not the model's)",
            false => "",
        },
        ready.execution_profile,
        ready.execution_identity,
    );
    if !ready.execution_unavailable.is_empty() {
        eprintln!(
            "inkling: execution manifest unavailable facts: {}",
            ready.execution_unavailable.join(", ")
        );
    }

    // The one place the two ranks' code diverges, and it is one `if`.
    if tp_rank.is_some_and(|rank| rank != 0) {
        drop(codec);
        return Ok(Loaded::Follower(Follower {
            session,
            digest: blake3::Hasher::new(),
            ready,
        }));
    }

    // Decoder state belongs to the whole logical token sequence, not to one
    // generated turn. Byte-fallback and spacing decoders both need surrounding
    // ids. New world-context ids advance this stream without being spoken;
    // generated ids advance the same stream and their chunks are emitted. A
    // carried token is never advanced twice: it entered this sequence when it
    // was generated, while `carry` only catches the KV cache up to that fact.
    let tokenizer: &'static tokenizers::Tokenizer = Box::leak(Box::new(tokenizer));
    Ok(Loaded::Engine(Engine {
        signing_key: config.signing_key.clone(),
        session,
        codec,
        tokenizer,
        decode: detokenizer(tokenizer),
        ready,
        context_budget,
        delta: Vec::new(),
        delta_audio: Vec::new(),
        delta_vision: Vec::new(),
        carry: None,
        turn: 0,
        digest: blake3::Hasher::new(),
        terminated: false,
        score: std::env::var("INK_SCORE").map(|v| v != "0").unwrap_or(true),
        delta_unscored: false,
    }))
}

/// Which half of the pair this box turned out to be.
///
/// The election happened before the load, but the LOAD is what makes the
/// difference concrete, so the two roles become two types here rather than a
/// boolean the caller has to keep checking.
pub enum Loaded {
    /// This box holds the rendezvous address: it owns Drive and the pile.
    Engine(Engine),
    /// This box does not: it is a pure model rank.
    Follower(Follower),
}

/// A `'static` incremental detokenizer over a leaked tokenizer.
fn detokenizer(tokenizer: &'static tokenizers::Tokenizer) -> Detokenizer {
    let mut stream = tokenizer.decode_stream(false);
    Box::new(move |id| {
        stream
            .step(id)
            .map_err(|error| anyhow::anyhow!("streaming decode: {error}"))
    })
}

// ── the follower ────────────────────────────────────────────────────────────

impl Follower {
    /// What loaded on this rank.
    pub fn ready(&self) -> &Ready {
        &self.ready
    }

    /// Replay rank 0's passes until it says the run is over.
    ///
    /// This is the whole of rank 1's program. It never decides anything: not
    /// which tokens to generate, not when to stop, not what the delta is. That
    /// is the point — every decision made on both boxes is a decision that can
    /// be made differently on the two boxes.
    ///
    /// The tokens it computes are discarded, but they are NOT ignored: each one
    /// is folded into this rank's sequence digest, which rank 0's
    /// [`Pass::Agree`] then compares against its own. That comparison is what
    /// remains of the deleted `ServePair`'s byte-for-byte cross-rank check.
    pub fn follow(&mut self) -> Result<()> {
        loop {
            let pass = {
                let group = self
                    .session
                    .group_mut()
                    .context("a follower with no Group has nobody to follow")?;
                group.follow()?
            };
            match pass {
                Pass::Prefill(ids) => {
                    let token = self
                        .session
                        .prefill(&ids)
                        .context("prefill this rank's first sequence")?;
                    self.fold(token);
                }
                Pass::Extend(ids) => {
                    let token = self
                        .session
                        .extend(&ids)
                        .context("extend this rank's sequence")?;
                    self.fold(token);
                }
                // Scored passes score here too (the scores are dropped; rank 0
                // keeps its own), and that is what runs this rank's learner.
                Pass::PrefillScored(ids) => {
                    let (token, _nll) = self
                        .session
                        .prefill_scored(&ids)
                        .context("prefill and score this rank's first sequence")?;
                    self.fold(token);
                }
                Pass::ExtendScored(ids) => {
                    let (token, _nll) = self
                        .session
                        .extend_scored(&ids)
                        .context("extend and score this rank's sequence")?;
                    self.fold(token);
                }
                Pass::Audio { slot, levels } => {
                    self.session
                        .push_audio(slot, &levels)
                        .context("stage this rank's dMel frames")?;
                }
                Pass::Vision { slot, patches } => {
                    self.session
                        .push_vision(slot, &patches)
                        .context("stage this rank's patches")?;
                }
                Pass::Evict { from, to } => {
                    self.session
                        .evict(from, to)
                        .context("evict this rank's span")?;
                }
                Pass::Step => {
                    let token = self.session.step().context("advance one token")?;
                    self.fold(token);
                }
                Pass::Reset => {
                    self.session.reset();
                    self.digest = blake3::Hasher::new();
                }
                Pass::Agree => {
                    let digest = *self.digest.clone().finalize().as_bytes();
                    let group = self
                        .session
                        .group_mut()
                        .context("a follower with no Group cannot agree")?;
                    group.agree(digest)?;
                }
                Pass::Export => {
                    let cuts = self
                        .session
                        .export_learned()
                        .context("export this rank's learned experts")?;
                    let group = self
                        .session
                        .group_mut()
                        .context("a follower with no Group cannot export")?;
                    group.send_cuts(&cuts)?;
                }
                Pass::Finish => {
                    eprintln!("inkling: rank 0 ended the run; this rank is stopping cleanly");
                    return Ok(());
                }
                Pass::Abort => {
                    anyhow::bail!(
                        "rank 0 failed terminally and released this rank rather than leaving it \
                         in a collective"
                    )
                }
            }
        }
    }

    fn fold(&mut self, token: usize) {
        fold_pass(&mut self.digest, token, self.session.position());
    }
}

/// The one definition of what a rank's sequence identity IS.
///
/// Both ranks call this and neither may have its own version: a digest that
/// disagreed because the two sides folded differently would be a false alarm,
/// and a false alarm on a check like this is worse than no check.
fn fold_pass(digest: &mut blake3::Hasher, token: usize, position: usize) {
    digest.update(&(token as u64).to_be_bytes());
    digest.update(&(position as u64).to_be_bytes());
}

// ── the engine ──────────────────────────────────────────────────────────────

impl Engine {
    /// What loaded, and whether its tokens are the model's.
    pub fn ready(&self) -> &Ready {
        &self.ready
    }

    /// Make one `Session` pass, having first told every other rank to make it.
    ///
    /// The order is deliberate and is the whole lockstep contract: NAME the
    /// pass, then make it. The write does not wait — the collective inside the
    /// pass is the synchronisation, and the kernel's socket buffer absorbs the
    /// skew.
    fn pass(&mut self, pass: Pass) -> Result<usize> {
        self.lead(&pass)?;
        let token = match &pass {
            Pass::Prefill(ids) => self
                .session
                .prefill(ids)
                .context("prefill the first sequence")?,
            Pass::Extend(ids) => self.session.extend(ids).context("extend the sequence")?,
            Pass::Step => self.session.step().context("advance one token")?,
            other => anyhow::bail!("{other:?} does not produce a token"),
        };
        fold_pass(&mut self.digest, token, self.session.position());
        Ok(token)
    }

    /// [`Engine::pass`] for a `Prefill` or `Extend` that also SCORES what it
    /// appends: the second element is the negative log-likelihood, in nats, of
    /// every appended id after the first, under the model that had seen
    /// everything before it. The wire is unchanged — the peer makes the same
    /// pass unscored, because the head is rank-local and scoring changes no
    /// collective — so a scored rank and a plain one stay in step.
    fn pass_scored(&mut self, pass: Pass) -> Result<(usize, super::session::ScoredNll)> {
        // The wire names the SCORED pass, so the other rank scores too: under
        // tensor parallelism the learner runs on every rank, each on its own
        // cut of the experts, and a rank that ran the plain pass would keep
        // yesterday's half of the model.
        let wire = match &pass {
            Pass::Prefill(ids) => Pass::PrefillScored(ids.clone()),
            Pass::Extend(ids) => Pass::ExtendScored(ids.clone()),
            other => anyhow::bail!("{other:?} is not a pass that appends ids to score"),
        };
        self.lead(&wire)?;
        let (token, nll) = match &pass {
            Pass::Prefill(ids) => self
                .session
                .prefill_scored(ids)
                .context("prefill and score the first sequence")?,
            Pass::Extend(ids) => self
                .session
                .extend_scored(ids)
                .context("extend and score the sequence")?,
            other => anyhow::bail!("{other:?} is not a pass that appends ids to score"),
        };
        fold_pass(&mut self.digest, token, self.session.position());
        Ok((token, nll))
    }

    /// Name the pass to every other rank before making it — the lockstep
    /// contract [`Engine::pass`] describes — and refuse if this rank or its
    /// peer can no longer enter a collective.
    fn lead(&mut self, pass: &Pass) -> Result<()> {
        anyhow::ensure!(
            !self.terminated,
            "this engine is terminated and must not enter another collective"
        );
        if let Some(group) = self.session.group_mut() {
            anyhow::ensure!(
                group.peer_alive(),
                "the peer rank's end of the rank link is closed: its process is gone. \
                 Refusing to enter a collective that would block forever. {}",
                transport_note()
            );
            group.lead(pass)?;
        }
        Ok(())
    }

    /// Every expert the learner has moved, whole, joined across the ranks.
    ///
    /// Rank 0 exports its own cut, every other rank sends its cut over the
    /// rank link ([`Pass::Export`]), and the cuts are joined back into the
    /// experts the pile stores. Nothing is published: see
    /// [`super::learned`] for why the identity of the resulting model is
    /// still an open decision.
    pub fn export_learned(&mut self) -> Result<Vec<super::learned::LearnedExpert>> {
        if self.session.group_mut().is_some() {
            self.lead(&Pass::Export)?;
        }
        let mut cuts = self
            .session
            .export_learned()
            .context("export rank 0's learned experts")?;
        if let Some(group) = self.session.group_mut() {
            cuts.extend(group.recv_cuts()?);
        }
        super::learned::assemble(cuts)
    }

    /// Tell every other rank to do something that is not a pass.
    fn announce(&mut self, pass: Pass) -> Result<()> {
        match self.session.group_mut() {
            Some(group) => group.lead(&pass),
            None => Ok(()),
        }
    }

    /// One turn: attend to the delta, then generate, calling `on_token` with
    /// each fragment AS IT IS DECODED.
    ///
    /// The two `Session` calls that matter are here and nowhere else: `prefill`
    /// for the first sequence, `extend` for every turn after it — which attends
    /// ONLY to what is new, because the KV cache is still holding everything
    /// before it. That is the property the whole exercise is for, and it is why
    /// the second turn is three orders of magnitude cheaper than the first.
    ///
    /// # What is NEW is not only what the caller sent
    ///
    /// A turn's last token is emitted and never fed back: the loop below stops
    /// one step short, because generating a successor for a token the caller
    /// will not read costs a whole decode step (~44 ms at layers 0..21) and
    /// produces nothing. That saving is real and it is kept — but it means the
    /// turn ends with one token of the sequence in the consumer's stream and NOT
    /// in the KV cache.
    ///
    /// So `carry` holds it, and the next turn appends it at the HEAD of its
    /// delta. That is the only place it can go and the cheapest place it could
    /// have gone: `Session::extend` batches, so the carried token is one extra
    /// ROW of a pass the turn was making anyway rather than a decode step of its
    /// own.
    ///
    /// **Until 2026-08-27 nothing carried it and every turn lost its own final
    /// word, permanently.** The failure was invisible from inside: `position()`
    /// stayed exactly `prompt + fed`, no length disagreed with any other, and
    /// the cache was perfectly CONSISTENT — one token short of the sequence it
    /// stood for. `inkling_session --carry` is the gate that catches it, and it
    /// catches it by asking the model what comes next rather than by measuring
    /// anything.
    ///
    /// # No stop ids
    ///
    /// The serving process took `--stop-id` and refused it under tensor
    /// parallelism, because "one rank stopping while its peer enters the next
    /// collective would deadlock". There is no such knob here at all: the stop
    /// decision belongs to `InklingMind`, which makes it between one-token
    /// consults, on rank 0, and communicates it by simply not asking for
    /// another token. One place, and it is the place that can see the model's
    /// structural end-of-sampling marker.
    fn generate(
        &mut self,
        want: usize,
        on_token: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<TurnEnd> {
        let delta_ids = std::mem::take(&mut self.delta);
        // The frames go to every rank BEFORE the pass that consumes them, on
        // the same link and in order, so the follower stages them first too.
        let delta_audio = std::mem::take(&mut self.delta_audio);
        if !delta_audio.is_empty() {
            let slot = self.codec.special_ids().audio_slot as usize;
            self.lead(&Pass::Audio {
                slot,
                levels: delta_audio.clone(),
            })?;
            self.session
                .push_audio(slot, &delta_audio)
                .context("stage the delta's dMel frames")?;
        }
        let delta_vision = std::mem::take(&mut self.delta_vision);
        if !delta_vision.is_empty() {
            let slot = self.codec.special_ids().image_slot as usize;
            self.lead(&Pass::Vision {
                slot,
                patches: delta_vision.clone(),
            })?;
            self.session
                .push_vision(slot, &delta_vision)
                .context("stage the delta's patches")?;
        }
        // Context participates in decoding even though it is not speech.
        // Advancing and discarding here makes the next generated id see its
        // real predecessor without ever echoing a tool result. `carry` is
        // excluded because the decoder already saw that id when the previous
        // turn generated it.
        for &id in &delta_ids {
            let _ = (self.decode)(id as u32)?;
        }
        anyhow::ensure!(
            self.carry.is_some() || !delta_ids.is_empty(),
            "the first turn has nothing to attend to: a prefill with no tokens would be vacuous"
        );

        // What this pass appends: the previous turn's unfed last token, then the
        // new context. On turn 0 there is no carry and this IS the delta.
        let ids: Vec<usize> = self
            .carry
            .iter()
            .copied()
            .chain(delta_ids.iter().copied())
            .collect();
        let carried = ids.len() - delta_ids.len();

        let started = std::time::Instant::now();
        let pass = match self.carry.is_some() {
            false => Pass::Prefill(ids),
            // Never empty on a primed session: the carry alone is a token, so a
            // consult with no new context is still a one-row `extend` rather
            // than a bare `step`. Same pass, and it is the pass that closes the
            // gap.
            true => Pass::Extend(ids),
        };
        // The delta is scored as it is attended to: every appended id after the
        // first gets the model's negative log-likelihood for it, BEFORE the
        // model has seen it. On a primed session the first appended id is the
        // carry, so the scores are exactly one per delta token; on turn 0 the
        // first delta token has no predecessor and goes unscored. This is the
        // live loss data the online-learning path exists for, and it is what
        // makes a learning change measurable at all: `INK_SCORE=0` turns it
        // off for a run that wants the head's last row only.
        let unscored = std::mem::take(&mut self.delta_unscored);
        let (first, scored) = match self.score && !unscored {
            true => self.pass_scored(pass)?,
            false => (self.pass(pass)?, super::session::ScoredNll::default()),
        };
        let super::session::ScoredNll {
            nll: delta_nll,
            frozen: delta_nll_frozen,
        } = scored;
        let first_token_secs = started.elapsed().as_secs_f64();

        // ── the incremental detokenizer ─────────────────────────────────────
        //
        // `DecodeStream` owns the prefix needed by byte-fallback and spacing
        // decoders. Each generated id therefore yields either one final text
        // chunk or `None` while an incomplete sequence waits for a later logical
        // token. No replacement character is emitted and no spoken prefix is
        // rewritten.
        let mut generated: Vec<u32> = Vec::with_capacity(want);
        let mut token = first;
        for step in 0..want {
            generated.push(token as u32);
            if let Some(text) = (self.decode)(token as u32)?
                && !text.is_empty()
            {
                // Handed to the consumer HERE, inside the generation loop. This
                // one call is the difference between a stream and a batch, and
                // it used to be a framed record written and flushed to a pipe.
                on_token(&text)?;
            }
            // One step short on purpose: the successor of the last emitted token
            // would cost a full decode step and nobody would read it. The token
            // itself is not lost — it leaves in `carry` below and is appended by
            // the next turn's `extend`. Break that pairing and the model stops
            // hearing its own last word. See this function's doc.
            if step + 1 < want {
                token = self.pass(Pass::Step)?;
            }
        }

        // What this turn emitted and did not feed back. A turn always emits at
        // least one token, so this is always `Some` afterwards — which is also
        // what tells the next turn it is not the first.
        self.carry = generated.last().map(|&t| t as usize);

        let tokens = generated.len();
        // The completed-turn seam check survives; its per-turn `eprintln!` does
        // not. `inkling_serve` printed two lines per consult, and a consult is
        // now ONE TOKEN, so that was two lines of stderr per generated word.
        // The check is the part that was load-bearing.
        self.session
            .validate_cache_completeness()
            .context("validate every attention cache at the completed-turn seam")?;
        let turn = self.turn;
        self.turn += 1;
        Ok(TurnEnd {
            turn,
            tokens,
            token_ids: generated,
            delta_tokens: delta_ids.len(),
            delta_nll,
            delta_nll_frozen,
            carried,
            // The engine generated exactly what it was asked for. Whether the
            // model's response is FINISHED is a structural question about the
            // generated ids, and `InklingMind` owns it.
            stopped: "max_tokens".to_string(),
            first_token_secs,
            turn_secs: started.elapsed().as_secs_f64(),
            position: self.session.position(),
        })
    }

    /// EVICT absolute positions `from..to` from every global layer's cache, on
    /// every rank: a folded span of the moment leaving the context while the
    /// positions around it keep their place. See `Session::evict`.
    pub fn evict_span(&mut self, from: usize, to: usize) -> Result<()> {
        self.lead(&Pass::Evict { from, to })?;
        self.session
            .evict(from, to)
            .context("evict the span from rank 0's caches")
    }
}

impl Model for Engine {
    fn ready(&self) -> &Ready {
        &self.ready
    }

    fn context(&mut self, context: &InklingContext) -> Result<()> {
        let ids = self
            .codec
            .encode(context)
            .context("encode typed Inkling context")?;
        self.delta.extend(ids);
        if matches!(context, InklingContext::Initialize { .. }) {
            self.delta_unscored = true;
        }
        // The payloads behind the slots just emitted, each medium to its own
        // queue, in the order the slots were emitted.
        for record in self.codec.sensed(context) {
            match &record.media {
                SenseMedia::Dmel { levels } => self.delta_audio.extend_from_slice(levels),
                SenseMedia::Text { .. } => {}
                SenseMedia::Patches { patches } => self.delta_vision.extend_from_slice(patches),
            }
        }
        Ok(())
    }

    fn preflight_context(&mut self, request: &ContextPreflight) -> Result<ContextPreflighted> {
        anyhow::ensure!(
            self.delta.is_empty(),
            "context preflight requires an empty pending delta"
        );
        if request.placement == ContextPlacement::Replace {
            anyhow::ensure!(
                matches!(&request.context, InklingContext::Initialize { .. }),
                "replacement preflight requires one complete Initialize payload"
            );
            validate_reinitialize_boundary(self.turn, self.delta.len(), self.carry.is_some())?;
        }
        let encoded = self
            .codec
            .encode(&request.context)
            .context("encode context preflight")?;
        context_preflight(
            request.placement,
            self.session.position(),
            usize::from(self.carry.is_some()),
            encoded.len(),
            request.max_response_tokens,
            self.context_budget,
            self.session.evicted_rows(),
        )
    }

    fn reinitialize(&mut self, initialization: &InklingContext) -> Result<Reinitialized> {
        anyhow::ensure!(
            matches!(initialization, InklingContext::Initialize { .. }),
            "a reinitialization requires one complete Initialize payload"
        );
        validate_reinitialize_boundary(self.turn, self.delta.len(), self.carry.is_some())?;

        // Every fallible operation that can reject the replacement is above the
        // reset. An invalid or over-wide cover leaves the old sequence
        // byte-for-byte alive — including on the OTHER rank, which is why
        // `Pass::Reset` is not announced until after every check has passed.
        let replacement = self
            .codec
            .encode(initialization)
            .context("encode the replacement initialization")?;
        anyhow::ensure!(
            replacement.len() <= self.context_budget,
            "the {}-token replacement initialization exceeds this Session's \
             {}-token context budget",
            replacement.len(),
            self.context_budget,
        );
        let acknowledgement = Reinitialized {
            previous_position: self.session.position(),
            previous_turns: self.turn,
            initialization_tokens: replacement.len(),
        };

        self.announce(Pass::Reset)?;
        self.session.reset();
        self.digest = blake3::Hasher::new();
        // A reset session has no decoder history either. The tokenizer is
        // leaked and `'static`, so a fresh stream costs nothing but its own
        // state.
        self.decode = detokenizer(self.tokenizer);
        self.delta = replacement;
        self.delta_unscored = true;
        self.delta_audio.clear();
        self.delta_vision.clear();
        self.carry = None;
        self.turn = 0;

        eprintln!(
            "inkling: reinitialized after {} turn(s) at position {}; {} replacement token(s) staged",
            acknowledgement.previous_turns,
            acknowledgement.previous_position,
            acknowledgement.initialization_tokens,
        );
        Ok(acknowledgement)
    }

    fn consult(
        &mut self,
        request: &Consult,
        on_token: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<TurnEnd> {
        self.generate(request.max_tokens.max(1), on_token)
    }

    fn agree_sequence(&mut self) -> Result<()> {
        if self.terminated {
            return Ok(());
        }
        let digest = *self.digest.clone().finalize().as_bytes();
        self.announce(Pass::Agree)?;
        match self.session.group_mut() {
            Some(group) => group.agree(digest),
            None => Ok(()),
        }
    }

    fn kill(&mut self) -> Result<()> {
        if self.terminated {
            return Ok(());
        }
        self.terminated = true;
        // Terminal, and it MUST reach the peer: a rank whose partner stops
        // issuing collectives blocks in NCCL with no timeout. This is the
        // fifth invariant of the deleted `ServePair` ("a dead rank must kill
        // its peer"), kept, with the pipe replaced by the rendezvous socket.
        //
        // It cannot help a peer that is already INSIDE a collective — see the
        // note on `Group::peer_alive`. What frees that one is this process
        // exiting, which tears down the communicator and makes the peer's
        // collectives fail.
        self.announce(Pass::Abort)
            .context("release the peer rank after a terminal failure")
    }

    fn position(&self) -> Option<usize> {
        Some(self.session.position())
    }

    fn staged_end(&self) -> Option<usize> {
        Some(self.session.position() + usize::from(self.carry.is_some()) + self.delta.len())
    }

    fn cut_to_tokens(&self, text: &str, max_tokens: usize) -> Result<Option<(String, usize)>> {
        let ids = self
            .codec
            .encode_raw_content(text)
            .context("count a result's tokens")?;
        if ids.len() <= max_tokens {
            return Ok(None);
        }
        let head: Vec<u32> = ids[..max_tokens].iter().map(|&id| id as u32).collect();
        let prefix = self
            .tokenizer
            .decode(&head, true)
            .map_err(|error| anyhow::anyhow!("decode a cut result's prefix: {error}"))?;
        Ok(Some((prefix, ids.len())))
    }

    fn evict(&mut self, from: usize, to: usize) -> Result<()> {
        Engine::evict_span(self, from, to)
    }

    fn persist_learned(
        &mut self,
        recipe: &super::resident::VersionRecipe,
    ) -> Result<Option<super::resident::Persisted>> {
        use triblespace::prelude::*;
        if self.terminated || !self.session.learning() {
            return Ok(None);
        }
        let Some(key_path) = self.signing_key.clone() else {
            return Ok(None);
        };
        let key = triblespace::core::signing_key_file::load_existing(&key_path)
            .with_context(|| format!("load the signing key {}", key_path.display()))?;
        let learned = self.export_learned()?;
        if learned.is_empty() {
            return Ok(None);
        }
        let parent = self.session.model_root();
        let mut store = Pile::open(std::path::Path::new(&self.ready.pile))
            .map_err(|e| anyhow::anyhow!("open {} to write a version: {e:?}", self.ready.pile))?;
        store
            .refresh()
            .map_err(|e| anyhow::anyhow!("refresh {}: {e:?}", self.ready.pile))?;
        let version = super::version::learned_version(&mut store, &learned, parent, recipe)?;
        let persisted = super::resident::Persisted {
            root: version.root,
            parent: version.parent,
            name: version.name.clone(),
            replaced: version.replaced,
            genesis: version.genesis,
        };
        if version.root != version.parent {
            super::version::publish_version(&mut store, &key, version)?;
        }
        store
            .close()
            .map_err(|e| anyhow::anyhow!("close {}: {e:?}", self.ready.pile))?;
        Ok(Some(persisted))
    }

    fn shutdown(&mut self) -> Result<()> {
        if self.terminated {
            return Ok(());
        }
        self.terminated = true;
        self.announce(Pass::Finish)
            .context("tell the peer rank the run is over")
    }
}

/// The last resort for releasing the peer rank.
///
/// `InklingMind`'s own `Drop` calls [`Model::shutdown`], which is the ordinary
/// path and which marks this engine terminated so this does nothing. But an
/// engine can also die BEFORE it ever reaches a mind — `validate_ready` refuses
/// a partial stack, the response cap does not fit the context budget, the shell
/// fails to open a sandbox — and on the other box a `Follower` is already
/// blocked in `Group::follow` holding its whole arena.
///
/// Nothing else would tell it. Dropping the socket would eventually surface as
/// an `UnexpectedEof` in the follower, which is handled and is a fine backstop,
/// but "rank 0 said Finish" and "rank 0 vanished" are different events and only
/// one of them is a clean exit. This makes every early return the first one.
impl Drop for Engine {
    fn drop(&mut self) {
        if self.terminated {
            return;
        }
        if let Err(error) = self.shutdown() {
            eprintln!("inkling: could not release the peer rank on drop: {error:#}");
        }
    }
}

// ── the reinitialization boundary ───────────────────────────────────────────

/// A reinitialization replaces a sequence; it must not become a disguised way
/// to discard context already queued for the next turn.
///
/// The engine is synchronous, so reaching this check proves no consult is in
/// flight. It cannot see whether Drive still has a tool execution outstanding —
/// that is the foreground runner's semantic boundary, and this is only the
/// narrow mechanical one.
fn validate_reinitialize_boundary(
    completed_turns: usize,
    pending_delta_tokens: usize,
    has_carry: bool,
) -> Result<()> {
    anyhow::ensure!(
        completed_turns > 0 && has_carry,
        "reinitialization requires a completed turn"
    );
    anyhow::ensure!(
        pending_delta_tokens == 0,
        "reinitialization cannot discard {pending_delta_tokens} pending context token(s)"
    );
    Ok(())
}

// ── execution identity ──────────────────────────────────────────────────────
//
// Moved verbatim from `inkling_serve`. It is not transport: it is what makes
// two runs comparable, and under a ONE-BINARY deployment it is what makes the
// claim "the same binary on both boxes" checkable rather than asserted —
// `executable-bytes` hashes `/proc/self/exe`, so two ranks announcing the same
// execution identity have literally the same bytes running.

/// Environment is inherited ambient authority. Sealed mode refuses namespaces
/// this runtime uses to alter kernels, numerics, scheduling, or allocation
/// before a CUDA client exists. The exact exceptions below are rank-local
/// placement, transport routing, and diagnostics: they must be able to differ
/// across hosts and do not choose model numerics or kernels. In particular,
/// `CUDA_VISIBLE_DEVICES` maps CUDA's logical device 0, whose effective class
/// and compute capability are witnessed in the shared manifest; the NCCL
/// exceptions only route the two-rank transport or report what it selected.
/// Every other CUDA/NCCL variable remains rejected, including algorithm knobs.
///
/// Explicit serving CLI settings remain allowed because they enter the
/// manifest. Library discovery through `LD_LIBRARY_PATH` is the other narrow
/// phase-1 exception: the exact selected library bytes are hashed after load,
/// while rejecting it would make the CUDA deployment layout unusable.
fn sealed_environment_rejections(names: impl IntoIterator<Item = String>) -> Vec<String> {
    const PREFIXES: &[&str] = &[
        "INK_",
        "CUBECL_",
        "CUDA_",
        "NCCL_",
        "NVRTC_",
        "CUBLAS_",
        "CUDNN_",
        "BURN_",
        "OMP_",
        "MKL_",
        "OPENBLAS_",
        "RAYON_",
        "MALLOC_",
    ];
    const EXACT: &[&str] = &["GLIBC_TUNABLES", "LD_AUDIT", "LD_PRELOAD"];
    const RANK_LOCAL_EXACT: &[&str] = &[
        "CUDA_VISIBLE_DEVICES",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
    ];
    let mut rejected = names
        .into_iter()
        .filter(|name| {
            let rank_local = RANK_LOCAL_EXACT.contains(&name.as_str())
                || name == "NCCL_DEBUG"
                || name.starts_with("NCCL_DEBUG_");
            !rank_local
                && (EXACT.contains(&name.as_str())
                    || PREFIXES.iter().any(|prefix| name.starts_with(prefix)))
        })
        .collect::<Vec<_>>();
    rejected.sort();
    rejected
}

fn reject_sealed_environment() -> Result<()> {
    let rejected = sealed_environment_rejections(
        std::env::vars_os().map(|(name, _)| name.to_string_lossy().into_owned()),
    );
    anyhow::ensure!(
        rejected.is_empty(),
        "sealed-v1 refuses execution-changing environment overrides: {}. Express serving shape \
         through explicit CLI arguments; library selection is witnessed by exact mapped-library \
         hashes",
        rejected.join(", ")
    );
    Ok(())
}

fn hex_identity(bytes: [u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut text = String::with_capacity(64);
    for byte in bytes {
        write!(&mut text, "{byte:02X}").expect("writing into a String is infallible");
    }
    text
}

struct RuntimeFacts {
    gpu_class: String,
    compute_capability: String,
    cuda_version: String,
    cuda_library_hashes: String,
    nvrtc_version: String,
    nvrtc_library_hashes: String,
    nccl_version: String,
    nccl_library_hashes: String,
    unavailable: Vec<String>,
}

impl RuntimeFacts {
    /// Observe only facts supplied by the loaded process and CUDA APIs. Phase 1
    /// does not invoke nvidia-smi, nvcc, the package manager, or filesystem
    /// heuristics that could describe a toolkit different from the one executing
    /// kernels. `/proc/self/maps` is the exact phase-1 boundary for library
    /// identity: it hashes mapped backing-file bytes after confirming the path
    /// still names the mapped device/inode, not relocated in-memory pages. When
    /// that backing file cannot be named/read, the manifest records unavailable.
    fn observe() -> Self {
        let mut unavailable = Vec::new();
        let (gpu_class, compute_capability) = observe_gpu(&mut unavailable);
        let cuda_library_hashes = observe_library("cuda.library", "libcuda.so", &mut unavailable);
        let nvrtc_library_hashes =
            observe_library("nvrtc.library", "libnvrtc.so", &mut unavailable);
        let nccl_library_hashes = observe_library("nccl.library", "libnccl.so", &mut unavailable);

        let cuda_version = observe_version("cuda.version", &mut unavailable, || {
            use cudarc::driver::result::DriverError;

            let mut version = 0;
            let result: std::result::Result<(), DriverError> =
                unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version).result() };
            result.map_err(|error| format!("{error}"))?;
            Ok(version.to_string())
        });
        let nvrtc_version = match nvrtc_library_hashes.as_str() {
            "unavailable" => {
                unavailable.push("nvrtc.version".to_string());
                "unavailable".to_string()
            }
            _ => observe_version("nvrtc.version", &mut unavailable, || {
                let (mut major, mut minor) = (0, 0);
                unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor).result() }
                    .map_err(|error| format!("{error}"))?;
                Ok(format!("{major}.{minor}"))
            }),
        };
        let nccl_version = match nccl_library_hashes.as_str() {
            "unavailable" => {
                if !unavailable.iter().any(|fact| fact == "nccl.version") {
                    unavailable.push("nccl.version".to_string());
                }
                "unavailable".to_string()
            }
            _ => observe_version("nccl.version", &mut unavailable, || {
                cudarc::nccl::result::get_nccl_version()
                    .map(|version| version.to_string())
                    .map_err(|error| format!("{error:?}"))
            }),
        };
        unavailable.sort();
        unavailable.dedup();
        Self {
            gpu_class,
            compute_capability,
            cuda_version,
            cuda_library_hashes,
            nvrtc_version,
            nvrtc_library_hashes,
            nccl_version,
            nccl_library_hashes,
            unavailable,
        }
    }

    fn add_to(&self, manifest: &mut ExecutionManifest) {
        for (name, value) in [
            ("gpu-class", self.gpu_class.as_str()),
            ("gpu-compute-capability", self.compute_capability.as_str()),
            ("cuda-driver-version", self.cuda_version.as_str()),
            ("cuda-library-blake3", self.cuda_library_hashes.as_str()),
            ("nvrtc-version", self.nvrtc_version.as_str()),
            ("nvrtc-library-blake3", self.nvrtc_library_hashes.as_str()),
            ("nccl-version", self.nccl_version.as_str()),
            ("nccl-library-blake3", self.nccl_library_hashes.as_str()),
        ] {
            manifest.field(name, value.as_bytes());
        }
    }
}

fn observe_gpu(unavailable: &mut Vec<String>) -> (String, String) {
    let result = std::panic::catch_unwind(|| -> Result<(String, String)> {
        cudarc::driver::result::init().context("initialize CUDA driver observation")?;
        let device =
            cudarc::driver::result::device::get(0).context("open CUDA device ordinal 0")?;
        let name =
            cudarc::driver::result::device::get_name(device).context("read CUDA device class")?;
        let major = unsafe {
            cudarc::driver::result::device::get_attribute(
                device,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
        }
        .context("read CUDA compute-capability major")?;
        let minor = unsafe {
            cudarc::driver::result::device::get_attribute(
                device,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
        }
        .context("read CUDA compute-capability minor")?;
        Ok((name, format!("{major}.{minor}")))
    });
    match result {
        Ok(Ok(facts)) => facts,
        Ok(Err(error)) => {
            eprintln!("inkling: GPU manifest facts unavailable: {error:#}");
            unavailable.extend([
                "gpu.class".to_string(),
                "gpu.compute-capability".to_string(),
            ]);
            ("unavailable".to_string(), "unavailable".to_string())
        }
        Err(_) => {
            unavailable.extend([
                "gpu.class".to_string(),
                "gpu.compute-capability".to_string(),
            ]);
            ("unavailable".to_string(), "unavailable".to_string())
        }
    }
}

fn observe_version(
    name: &str,
    unavailable: &mut Vec<String>,
    observe: impl FnOnce() -> std::result::Result<String, String>,
) -> String {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(observe)) {
        Ok(Ok(version)) => version,
        Ok(Err(error)) => {
            eprintln!("inkling: {name} unavailable: {error}");
            unavailable.push(name.to_string());
            "unavailable".to_string()
        }
        Err(_) => {
            unavailable.push(name.to_string());
            "unavailable".to_string()
        }
    }
}

fn observe_library(name: &str, prefix: &str, unavailable: &mut Vec<String>) -> String {
    match mapped_library_hashes(prefix) {
        Ok(hashes) if !hashes.is_empty() => hashes.join(","),
        Ok(_) => {
            unavailable.push(name.to_string());
            "unavailable".to_string()
        }
        Err(error) => {
            eprintln!("inkling: {name} unavailable: {error:#}");
            unavailable.push(name.to_string());
            "unavailable".to_string()
        }
    }
}

fn mapped_library_hashes(prefix: &str) -> Result<Vec<String>> {
    let maps = std::fs::read_to_string("/proc/self/maps").context("read /proc/self/maps")?;
    let mut paths = std::collections::BTreeMap::new();
    for line in maps.lines() {
        let mut columns = line.split_whitespace();
        let _range = columns.next();
        let _permissions = columns.next();
        let _offset = columns.next();
        let device = columns.next();
        let inode = columns.next().and_then(|value| value.parse::<u64>().ok());
        let Some(path_start) = line.find('/') else {
            continue;
        };
        let path = line[path_start..].trim_end_matches(" (deleted)");
        let matches = std::path::Path::new(path)
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|file| file.starts_with(prefix));
        if matches {
            let (Some(device), Some(inode)) = (device, inode) else {
                anyhow::bail!("mapped library row has no device/inode witness: {line}");
            };
            let path = std::path::PathBuf::from(path);
            if let Some(previous) = paths.insert(path.clone(), (device.to_string(), inode)) {
                anyhow::ensure!(
                    previous == (device.to_string(), inode),
                    "mapped library {} changed identity within /proc/self/maps",
                    path.display()
                );
            }
        }
    }
    let mut hashes = Vec::with_capacity(paths.len());
    for (path, (_mapped_device, _mapped_inode)) in paths {
        let mut file = std::fs::File::open(&path)
            .with_context(|| format!("open mapped library {}", path.display()))?;
        // Hashing the spelling from /proc/self/maps without this check could
        // hash a replacement installed after dlopen rather than the inode the
        // process mapped.
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::MetadataExt as _;

            let metadata = file.metadata()?;
            let actual_device = format!(
                "{:02x}:{:02x}",
                libc::major(metadata.dev()),
                libc::minor(metadata.dev())
            );
            anyhow::ensure!(
                metadata.ino() == _mapped_inode && actual_device == _mapped_device,
                "mapped library {} is device/inode {_mapped_device}/{_mapped_inode}, but its path \
                 now names {actual_device}/{}",
                path.display(),
                metadata.ino()
            );
        }
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let count = std::io::Read::read(&mut file, &mut buffer)
                .with_context(|| format!("hash mapped library {}", path.display()))?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        hashes.push(hex_identity(*hasher.finalize().as_bytes()));
    }
    hashes.sort();
    hashes.dedup();
    Ok(hashes)
}

fn begin_execution_manifest(profile: &str) -> Result<ExecutionManifest> {
    // Linux exposes the inode this process is actually executing, even if the
    // launch path is replaced later. current_exe() cannot make that claim.
    let file = std::fs::File::open("/proc/self/exe")
        .context("open /proc/self/exe for the execution manifest")?;
    let length = file.metadata()?.len();
    let mut manifest = ExecutionManifest::new(profile);
    manifest.reader("executable-bytes", length, file)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::unicode::NFC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{Tokenizer, TokenizerBuilder};

    use super::{
        reject_sealed_environment, sealed_environment_rejections, validate_reinitialize_boundary,
    };

    fn byte_fallback_tokenizer() -> Tokenizer {
        let vocab = [
            ("<0x20>".to_string(), 0),
            ("<0xC3>".to_string(), 1),
            ("<0xA9>".to_string(), 2),
        ];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .byte_fallback(true)
            .build()
            .unwrap();
        TokenizerBuilder::default()
            .with_model(bpe)
            .with_decoder(Some(ByteFallback::default()))
            .with_normalizer(Some(NFC))
            .with_pre_tokenizer(Some(ByteLevel::default()))
            .with_post_processor(Some(ByteLevel::default()))
            .build()
            .unwrap()
            .into()
    }

    /// What `Engine::generate` does with a delta before it generates: advance
    /// the decoder over every context id and discard the text.
    fn advance_context_decode(
        decode: &mut impl FnMut(u32) -> anyhow::Result<Option<String>>,
        ids: &[usize],
    ) -> anyhow::Result<()> {
        for &id in ids {
            let _ = decode(id as u32)?;
        }
        Ok(())
    }

    #[test]
    fn sealed_tp_child_accepts_deployed_rank_local_environment() {
        const CHILD: &str = "MARY_SEALED_ENVIRONMENT_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            reject_sealed_environment()
                .expect("the deployed rank-local environment passes sealed validation");
            return;
        }

        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("tests::sealed_tp_child_accepts_deployed_rank_local_environment")
            .arg("--exact")
            .arg("--nocapture")
            .env_clear()
            .env(CHILD, "1")
            .env("CUDA_VISIBLE_DEVICES", "0")
            .env("NCCL_IB_DISABLE", "0")
            .env("NCCL_SOCKET_IFNAME", "rocep1s0f0")
            .env("NCCL_IB_HCA", "rocep1s0f0")
            .env("NCCL_DEBUG", "INFO")
            .env("NCCL_DEBUG_SUBSYS", "INIT,NET")
            .env("NCCL_DEBUG_FILE", "/tmp/nccl.%h.%p.log")
            .output()
            .expect("launch the sealed-environment fixture child");
        assert!(
            output.status.success(),
            "sealed TP child did not pass environment validation:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    #[test]
    fn sealed_environment_still_rejects_execution_selection() {
        let rejected = sealed_environment_rejections(
            ["NCCL_ALGO", "CUBECL_MEMORY_CONFIG"]
                .into_iter()
                .map(str::to_owned),
        );
        assert_eq!(rejected, ["CUBECL_MEMORY_CONFIG", "NCCL_ALGO"]);
    }

    #[test]
    fn reinitialization_requires_an_empty_completed_turn_boundary() {
        let before_first = validate_reinitialize_boundary(0, 0, false)
            .expect_err("there is no old sequence to replace");
        assert!(
            before_first.to_string().contains("completed turn"),
            "{before_first:#}"
        );

        let queued = validate_reinitialize_boundary(3, 7, true)
            .expect_err("queued context cannot be discarded");
        assert!(queued.to_string().contains("7 pending"), "{queued:#}");

        validate_reinitialize_boundary(3, 0, true)
            .expect("a completed turn with no pending context is exact");
    }

    #[test]
    fn incomplete_output_can_finish_on_the_next_turn() {
        let tokenizer = byte_fallback_tokenizer();
        let mut stream = tokenizer.decode_stream(false);
        let mut decode = |id| stream.step(id).map_err(|error| anyhow::anyhow!("{error}"));

        advance_context_decode(&mut decode, &[0]).unwrap();
        assert_eq!(decode(1).unwrap(), None, "the first output byte waits");
        assert_eq!(
            decode(2).unwrap().as_deref(),
            Some("é"),
            "a no-delta next turn completes rather than loses the character"
        );
    }

    #[test]
    fn text_completed_by_hidden_context_stays_hidden() {
        let tokenizer = byte_fallback_tokenizer();
        let mut stream = tokenizer.decode_stream(false);
        let mut decode = |id| stream.step(id).map_err(|error| anyhow::anyhow!("{error}"));

        assert_eq!(decode(1).unwrap(), None, "the output byte is incomplete");
        advance_context_decode(&mut decode, &[2]).unwrap();
        assert_eq!(
            decode(0).unwrap().as_deref(),
            Some(" "),
            "bytes completed partly by world input are consumed, not spoken later"
        );
    }
}
