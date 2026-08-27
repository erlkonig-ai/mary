//! A model you can HOLD: weights, KV and position as one value that survives
//! across calls.
//!
//! # What this is for, and what it replaces
//!
//! Every other module here is a component, and [`super::assembly`] is the
//! composition. Neither is a model you can *talk to*: to get a token out of
//! Inkling before this existed you launched `inkling_forward`, which opened the
//! pile, priced admission, copied 80 GiB into an anonymous arena, bound the
//! embedding and the unembedding, ran a prompt, printed a report and exited —
//! and then, for the next turn of the same conversation, did all of it again.
//!
//! At this model's scale that is not a slow serving path, it is *not a serving
//! path*. The startup is minutes and the prompt is re-read from scratch, so the
//! marginal cost of the second token of a conversation is the same as the first.
//! A `Session` is the fix and it is a small one, because the expensive parts
//! were never the problem: the problem was that they lived inside `main`.
//!
//! ```no_run
//! # use mary::models::inkling::session::{Session, SessionConfig};
//! let mut s = Session::load(SessionConfig::new("model.pile"))?;
//! let mut tok = s.prefill(&prompt_ids)?;      // the prompt, once
//! for _ in 0..64 {
//!     tok = s.step()?;                         // and then one token at a time
//! }
//! // …and on the NEXT turn, `s` still holds the KV. `prefill` again with only
//! // what is new: the session never re-reads what it has already attended to.
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! # Stateful is the whole point
//!
//! [`Session::step`] takes `&mut self` and reads nothing from the world. The KV
//! cache, the short-convolution histories and the position counter live in the
//! `Session`, so the second call continues the first. That is the property
//! `drive`'s `Mind` trait is written against — it hands a backend a causally
//! ordered *delta* rather than a re-rendered transcript, precisely because a
//! real backend is expected to still be holding the conversation.
//!
//! [`Session::extend`] is the delta form: give it the tokens that are new since
//! last time and it attends to those alone. [`Session::reset`] drops the caches
//! and starts a fresh sequence against the same warm weights, which is the
//! operation a serving process actually needs between conversations — it costs
//! a deallocation, not a reload.
//!
//! # And the third lever: going BACK
//!
//! `reset` is all-or-nothing, and for a while it was the only way to un-attend
//! to anything. That is fine for a conversation, which only ever grows, and
//! wrong for a prompt whose SETTLED PREFIX can change underneath it — a memory
//! cover, say, which re-refines its recent edge every time a memory is written
//! and can move the first differing byte thousands of characters back.
//!
//! [`Session::checkpoint`] keeps a position and [`Session::rewind`] returns to
//! it, so a caller that checkpoints per chunk pays, on a change, only for the
//! chunks at or after the first one that differs. Everything before it keeps its
//! KV and is never read again. What makes that more than a `pos` assignment is
//! the sliding window: thirty-five of the forty-two layers have already dropped
//! the keys a rewound position needs, so a checkpoint holds those layers'
//! stores — bounded by the window, and therefore the same size at position
//! 500,000 as at position 1,000. See [`Checkpoint`] and
//! [`super::burn::AttnRewind`].
//!
//! # What a Session deliberately is NOT
//!
//! It is not the serving process, and it is not `inkling_forward`.
//!
//! `inkling_forward` is a MEASUREMENT harness that happens to run the model: it
//! carries the pipe between two nodes, the CUDA-graph capture lane, batched
//! slots and cohorts, the MTP drafter, token-tree construction, the router and
//! plan A/B arms and about a hundred `INK_*` reporting switches. Every one of
//! those is a question someone is asking about the model, and none of them is
//! something a conversation needs. They stay in the binary. The narrow
//! exception is [`Session::begin_target_extension`]: a cache transaction on
//! which a future drafter can stand, not a drafter or acceptance policy itself.
//!
//! What is here is the lane that runs when nobody sets anything: the default
//! configuration, which is also the one the frontier benchmark measures. Greedy
//! argmax, cached decode, the W4A16 head, NVFP4 KV pages, the device router and
//! the device row plan — on by default, so on here.
//!
//! # Tensor parallelism: a Session is PER RANK
//!
//! Tensor parallelism runs this model as two processes, one per box, joined by
//! a NCCL all-reduce inside every layer. Under it each rank runs *every* layer
//! on *half* of each tensor, so both ranks hold the embedding, the whole stack
//! and the unembedding, and both produce the same token.
//!
//! A `Session` is therefore one RANK, not one model, and a caller that wants a
//! TP pair holds two of them in two processes. [`Session::load`] remains the
//! single-rank entry point and refuses `INK_TP`; a serving process explicitly
//! forms and warms a [`super::tpcomm::Group`] and hands that one value to
//! [`Session::load_with_group`]. The group owns rank AND client, so the startup
//! slice, the communicator and the stream on which its fences order kernels
//! cannot be configured as three independent facts.
//!
//! # The layer split
//!
//! `INK_LAYERS` is required for the same reason it is required in the binary:
//! 144 GiB of weights do not fit a 120 GiB box, and a process that would run the
//! whole stack pages its experts off the SSD between tokens. A `Session` runs
//! the subrange it is given. Only the range that includes the last layer owns
//! the final norm and the unembedding, so only that one can turn a hidden state
//! into a token — [`Session::step`] says so rather than returning a wrong one.

use std::collections::BTreeMap;

use anyhow::{Context, Result};

use super::assembly::{
    BT, Bk, DeviceDense, LayerCache, LayerDev, MoeState, RouterArm, T2, argmax_row_dev,
    argmax_rows_dev, bind_layer, dense_mlp_bf16, dev_lane, dev_lane_resid, moe_layer,
    quantized_bf16, up1r, up2, w4a16_bind,
};
use super::attn::{AttnDims, LogScaling};
use super::config::{AttnKind, InklingConfig};
use super::pile::Elem;
use super::pool::{CleanupGate, CleanupPolicy};
use super::source::Weights;
use super::stack::embed_and_norm_bf16;
use super::target::{
    Boundary as TargetBoundary, DEFAULT_TARGET_BUDGET, PrefixCache, Settlement as TargetSettlement,
    WidthAdmission as TargetWidthAdmission,
};
use super::tp::Tp;
use super::tpcomm::Group;

/// The next sequence id. Process-wide and monotone, so no two sequences —
/// across sessions or across one session's resets — ever share one, and a
/// [`Checkpoint`] can therefore be refused by the session it does not belong to
/// rather than silently accepted because both counters happened to read 0.
fn next_seq() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// `(base, len)` of the complete cache a layer must hold at `position`.
fn required_cache_span(kind: AttnKind, position: usize, window: usize) -> (usize, usize) {
    match kind {
        AttnKind::Local => {
            let len = position.min(window);
            (position - len, len)
        }
        AttnKind::Global => (0, position),
    }
}

/// A Session exposes a layer boundary, but not the routed lane's internal
/// stage boundaries. Refuse the stronger policy rather than print "stage" and
/// quietly perform only layer cleanup.
fn session_cleanup_policy(policy: CleanupPolicy) -> Result<CleanupPolicy> {
    anyhow::ensure!(
        policy != CleanupPolicy::PerStage,
        "INK_POOL_CLEANUP=stage requires the one-shot forward's internal stage boundaries; a \
         Session can clean between layers. Use INK_POOL_CLEANUP=1 for the strongest policy this \
         execution path implements."
    );
    Ok(policy)
}

/// Validate the two placement modes without touching a GPU.
///
/// A layer split gives one process a strict subrange. Tensor parallelism is the
/// opposite: every rank runs every layer on its own within-layer slice. Keeping
/// this as one arithmetic decision is what prevents startup copy and forward
/// from quietly choosing different geometries.
fn validate_layer_range(
    layers: &std::ops::Range<usize>,
    total: usize,
    tp: Option<Tp>,
) -> Result<bool> {
    let (lo, hi) = (layers.start, layers.end);
    anyhow::ensure!(
        lo < hi,
        "a Session runs LO..HI with LO < HI, got {lo}..{hi}"
    );
    anyhow::ensure!(hi <= total, "{lo}..{hi} runs past the {total}-layer stack");
    match tp {
        Some(tp) => anyhow::ensure!(
            (lo, hi) == (0, total),
            "tensor parallel rank {} of {} is a WITHIN-layer split: every rank runs every layer, \
             so the range must be 0..{total}, not {lo}..{hi}",
            tp.rank(),
            tp.world(),
        ),
        None => anyhow::ensure!(
            hi - lo < total,
            "{lo}..{hi} is the whole {total}-layer stack on one node, which does not fit ({} \
             GiB of weights). Split it: no node may run every layer, and two is the MINIMUM \
             rather than the number.",
            144,
        ),
    }
    Ok(hi < total)
}

/// How to open a model. Everything here has a default that matches what
/// `inkling_forward` does when nothing is set.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// The pile holding the weights AND the config — one source, so a pile that
    /// cannot answer `config.json` is not authoritative, only large.
    pub pile: std::path::PathBuf,
    /// `config.json` from a file instead of from the pile's facts. For a pile
    /// written before the sidecars were ingested; an override you have to type,
    /// never a fallback you never see.
    pub config_override: Option<std::path::PathBuf>,
    /// Which layers THIS process runs, as `lo..hi`. [`Session::load`] requires
    /// a strict subrange; [`Session::load_with_group`] requires the exact full
    /// range because every tensor-parallel rank runs every layer.
    pub layers: std::ops::Range<usize>,
    /// Pay the storage layer's content hash for this range's experts at load
    /// rather than in whichever decode step first routes to each of them. On by
    /// default: a serving process that skips this pays 0.5–274 ms in a random
    /// later token, which is the one variable that used to explain a whole
    /// latency spread.
    pub warm_experts: bool,
    /// The widest prefill chunk admission is priced at.
    ///
    /// Admission reserves activation headroom before the arena is filled, and
    /// the size of a prefill's largest buffer is a function of how many tokens
    /// it takes at once. A session does not know its prompts in advance, so it
    /// prices a fixed budget: big enough that a conversational turn fits under
    /// it, small enough that the reservation does not eat the arena. A prompt
    /// longer than this is processed in consecutive chunks of this width, so
    /// prompt length consumes KV capacity without silently demanding an
    /// equally wide activation allocation.
    pub prefill_budget: usize,

    /// Maximum rows one speculative target pass may verify at once.
    ///
    /// `0` disables target transactions. Unlike ordinary prefill and extend,
    /// target verification projects every row through the full vocabulary
    /// head, so callers must opt into a concrete proposal width rather than
    /// accidentally inheriting the much larger prefill chunk. It may not
    /// exceed [`SessionConfig::prefill_budget`].
    pub target_budget: usize,

    /// Maximum positions this session admits across its whole conversation.
    ///
    /// This is independent of [`SessionConfig::prefill_budget`]: the latter is
    /// transient activation width, while this one prices persistent KV and is
    /// enforced on every pass. Keeping them separate is what lets a long memory
    /// prefix arrive in bounded chunks without pretending its retained state is
    /// only one chunk long.
    pub context_budget: usize,

    /// How many positions one [`Session::extend`] pass appends to an existing
    /// cache at once.
    ///
    /// A delta longer than this is appended in consecutive passes of this
    /// width, each committed whole, which is the same sequence of states a
    /// single pass would leave — a chunk boundary is a commit point and nothing
    /// else. `1` is the WALKED arm: one decode step a token, the shape
    /// [`Session::extend`] had before it batched, kept because the equivalence
    /// gate has to be able to build the same cache both ways on one session.
    ///
    /// It may not exceed [`SessionConfig::prefill_budget`]: a batched append
    /// allocates activations as a function of its width exactly as a prefill
    /// does, and that budget is what admission reserved room for.
    ///
    /// # Why the default is WIDE rather than a tidy small number
    ///
    /// Because the cost of a pass is not linear in its width, and the
    /// non-linearity is at the FIRST pass of each width rather than at any
    /// particular size. cubecl keys its compiled kernels on shape and burn's
    /// matmul autotune keys its choice the same way, so a pass at a width
    /// nothing has run before pays for that width once and never again —
    /// measured on a GB10 at layers 0..21 as 11.66 ms/token against 10.97 for
    /// one new width, and 14.37 against 11.28 when `div_ceil` split the same
    /// delta into 107+107+106 and so needed TWO. A default narrow enough to
    /// split ordinary deltas would therefore pay that twice per turn for
    /// nothing; 4096 makes a conversational delta one pass at one width.
    ///
    /// What it is still for is the case a wide default does not cover: a delta
    /// so long that one pass of it is a resource problem rather than a shape
    /// one. A batch is not trimmed until it commits, so a local layer holds
    /// `window + extend_batch` rows for the duration of the pass, and the
    /// fused attention's partial-output buffer is linear in the row count.
    /// Both are bounded by this and by nothing else.
    pub extend_batch: usize,
}

impl SessionConfig {
    /// A config for `pile`, running the layers `INK_LAYERS` names.
    ///
    /// The layer range comes from the environment here because that is where the
    /// two-box launch scripts put it and a second spelling of the same fact is a
    /// second thing to get wrong. Set [`SessionConfig::layers`] directly to say
    /// it in code instead.
    pub fn new(pile: impl Into<std::path::PathBuf>) -> Self {
        let layers = std::env::var("INK_LAYERS")
            .ok()
            .and_then(|s| {
                let (a, b) = s.split_once(':')?;
                Some(a.parse::<usize>().ok()?..b.parse::<usize>().ok()?)
            })
            .unwrap_or(0..0);
        Self {
            pile: pile.into(),
            config_override: std::env::var("INK_CONFIG").ok().map(Into::into),
            layers,
            warm_experts: true,
            prefill_budget: 4096,
            target_budget: DEFAULT_TARGET_BUDGET,
            context_budget: 4096,
            extend_batch: 4096,
        }
    }

    /// Run `layers` on this rank.
    pub fn layers(mut self, layers: std::ops::Range<usize>) -> Self {
        self.layers = layers;
        self
    }
}

/// One live model: warm weights, a KV cache in flight, and a position.
///
/// Held across calls and across turns. Dropping it releases the arena, which on
/// this model is tens of gibibytes and takes the kernel a few tens of seconds to
/// hand back — so a serving process holds ONE and calls [`Session::reset`]
/// between conversations rather than dropping and reloading.
pub struct Session {
    cfg: InklingConfig,
    src: Weights,
    dev: burn::backend::cuda::CudaDevice,
    client: cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    /// Runtime bound on free pages retained by cubecl's pool. Admission charges
    /// live tensors and relies on this gate for historical pages; it must persist
    /// across passes so a pass that cleaned keeps the next one polling per layer.
    /// Without it, Session memory grows with the sequence of allocation shapes:
    /// the admission arithmetic remains true only under a cleanup assumption the
    /// resident path did not enforce, and a long-lived server eventually fails
    /// even when no individual request violates admission.
    cleanup_gate: CleanupGate,
    /// Present exactly when this Session is one tensor-parallel rank. The
    /// Group owns both [`Tp`] and the client above; the client is cloned FROM
    /// this value at load rather than independently constructed.
    group: Option<Group>,
    aliases: Option<super::fp4gemm::Aliases>,

    /// Which layers this rank runs.
    lo: usize,
    hi: usize,
    /// Whether this Session runs a STRICT SUBRANGE of the stack. It still
    /// unembeds — a single process has to answer — but through layers it did not
    /// all run, so the tokens are diagnostic. `false` is the real model.
    partial: bool,

    /// Per-layer weights, keyed by checkpoint prefix and bound on first use —
    /// the same lazy bind the binary does, so the first pass pays the upload and
    /// no pass after it does.
    layers: BTreeMap<String, LayerDev>,
    /// The two dense layers' MLPs and the shared ("sink") experts.
    dense: DeviceDense,

    /// The embedding table, as the BF16 the pile stores.
    embed: Vec<u8>,
    /// The embedding norm's gain, on the host: the embed is a host gather.
    embed_norm: Vec<f32>,
    /// The final norm's gain, uploaded once.
    final_norm: BT<Bk, 1>,
    /// The unembedding, bound W4A16 — the weight stays four bits and the
    /// activation stays BF16.
    unembed: dev_lane::ProjW,

    /// The MoE half's row-plan invariants and per-layer expert tables, built on
    /// first touch and held for the life of the session.
    moe: MoeState,
    /// Whether the shared experts' fused `w13` is stored halved (contiguous) or
    /// interleaved. A property of the pile, read once.
    shared_halved: bool,

    /// THE STATE. One cache per layer this rank runs, carrying K, V and both
    /// short-convolution histories. Empty until the first prefill; alive from
    /// then until [`Session::reset`].
    caches: Vec<LayerCache>,
    /// How many positions the caches hold — the next token's position.
    pos: usize,
    /// The token the last pass produced, which is what [`Session::step`] feeds.
    last: Option<usize>,
    /// How many positions one [`Session::extend`] pass appends. See
    /// [`SessionConfig::extend_batch`].
    extend_batch: usize,
    /// The widest pass admission reserved activation headroom for, kept so
    /// [`Session::set_extend_batch`] can hold to the same bound `load` did.
    prefill_budget: usize,
    /// Explicit maximum width of a speculative target transaction. Zero keeps
    /// the transaction path disabled for ordinary serving Sessions.
    target_budget: usize,
    /// Maximum number of positions this sequence may retain. Admission prices
    /// its persistent KV before the weight arena is allocated.
    context_budget: usize,
    /// Whether a pass FAILED PART WAY THROUGH the layer stack, leaving the
    /// caches at two different positions.
    ///
    /// A pass advances every layer's cache in turn and only then advances
    /// [`Session::pos`], so an error raised at layer `l` leaves layers below it
    /// holding rows that layers above it do not, and the position counter
    /// agreeing with neither. That is true of a one-row pass and of a `k`-row
    /// one alike — batching changes how MANY positions a tear spans, not
    /// whether one can happen — and there is no cheap undo, because the rows
    /// the lower layers appended are already in their stores.
    ///
    /// So the pass is not all-or-nothing and this says so: the session is
    /// POISONED, every later pass is refused by name, and the two ways out are
    /// [`Session::reset`] (throw the sequence away) and [`Session::rewind`] to
    /// a [`Checkpoint`] taken before the tear — which restores EVERY layer from
    /// one consistent instant and is therefore a real repair rather than a
    /// second guess.
    torn: bool,
    /// Which SEQUENCE this is. Minted per `load` and re-minted by
    /// [`Session::reset`], so a [`Checkpoint`] cannot be handed to a session
    /// that has since started a different conversation — or to a different
    /// session entirely, which holds different device buffers and for which the
    /// checkpoint's positions mean nothing.
    seq: u64,
}

/// A position a [`Session`] can be put back to, and the state it needs to stand
/// there.
///
/// # Why this is a TOKEN and not a number
///
/// The obvious API is `truncate_to(pos)`. It is the wrong one, because on
/// thirty-five of this model's forty-two layers most positions cannot be
/// truncated to at all: a sliding-window layer has released the keys before its
/// window, and a cache cut back to a position it has already run past attends
/// over fewer keys than the sequence has, silently and forever. See
/// [`super::burn::AttnRewind`] for the mechanism.
///
/// So the position a caller may rewind to is exactly a position at which
/// somebody kept the windowed layers' stores — and a value that IS the kept
/// state cannot name a position where none was kept. `truncate_to(9_203)` is a
/// number anyone can produce; a `Checkpoint` is not.
///
/// # What it costs
///
/// One clone of thirty-five bounded stores plus four small tensors a layer.
/// Bounded is the load-bearing word: a windowed layer's store never exceeds its
/// window, so a checkpoint at position 500,000 is the same size as one at
/// position 1,000. The global layers keep nothing at all — a truncation puts
/// them back exactly.
///
/// **Framing rule for the bytes below**: derived from the tensor shapes of the
/// 42-layer release (hidden 4096, 8 KV heads × 128 head_dim = 1024-wide KV
/// rows, sliding window 512, `sconv_kernel_size` 4) on the NVFP4 KV lane
/// ([`super::kvpages::fp4_kv`], 4.5 bits a value), NOT measured, and stated per
/// CHECKPOINT — against the alternative of keeping a whole session's KV.
///
/// At the instant it is taken a checkpoint allocates nothing: every page it
/// names is one the live cache also holds. What it costs is what it keeps ALIVE
/// once the live window has slid past — at most `512 + 128` rows a store (the
/// window plus a page of not-yet-cut dead prefix):
///
/// * 35 local layers × 2 stores × 640 × 1024 × 4.5 bits ≈ **25.8 MB**
/// * 42 layers × (`k_pre`, `v_pre`, and the two convolution histories) ≈ 5.2 MB
/// * 7 global layers: **nothing**
///
/// ≈ **31 MB for the whole stack, at any position** — against a live KV that is
/// ~424 MB at position 50,000 and grows with it, because the global layers do.
/// A single-rank session running half the stack keeps about half that (layers
/// 0..21 hold 18 local and 3 global layers: ≈ 16 MB).
///
/// The TIME lives in the gate rather than here, because it is a comparison and
/// not a constant: `checkpoint` and `rewind` measured 0.1 ms and 0.4 ms at
/// layers 0..21 on a GB10 — free — and what actually decides whether a rewind
/// is worth taking is the cost of re-extending afterwards. `inkling_session
/// --rewind` prints both sides and its doc carries the framing rule.
pub struct Checkpoint {
    /// The position the session stood at.
    pos: usize,
    /// The token the pass that reached `pos` produced.
    last: Option<usize>,
    /// The sequence this was taken from. See [`Session::seq`].
    seq: u64,
    /// One per cache SLOT, in the session's layer order.
    layers: Vec<LayerRewind>,
}

/// One layer's half of a [`Checkpoint`].
struct LayerRewind {
    attn: dev_lane::AttnRewind<Bk>,
    attn_sconv: BT<Bk, 2>,
    mlp_sconv: Option<BT<Bk, 2>>,
}

impl Checkpoint {
    /// The position this checkpoint stands at — what [`Session::position`]
    /// returned when it was taken, and what it will return again after a
    /// [`Session::rewind`] to it.
    pub fn position(&self) -> usize {
        self.pos
    }
}

/// The committed result of one [`TargetExtension`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetCommit {
    accepted: usize,
    position: usize,
    next: usize,
}

impl TargetCommit {
    /// How many leading proposed rows became facts about the sequence.
    pub fn accepted(&self) -> usize {
        self.accepted
    }

    /// The session position after settlement.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The target prediction after the accepted prefix, or the session's prior
    /// prediction when zero rows were accepted.
    pub fn next_token(&self) -> usize {
        self.next
    }
}

/// An uncommitted target-model extension.
///
/// The proposed tokens have run through every target layer, but their K/V and
/// convolution rows remain pending and the Session's position has not moved.
/// [`TargetExtension::base_prediction`] is the target argmax before row zero;
/// [`TargetExtension::post_row_predictions`] gives the target argmax after each
/// proposed row. Thus proposal zero is checked against the base prediction and
/// proposal `i > 0` against `post_row_predictions[i - 1]`. An acceptance policy
/// outside this module chooses a leading count and calls
/// [`TargetExtension::commit`]. Dropping the value or calling
/// [`TargetExtension::abort`] discards the pass.
///
/// This value borrows the Session mutably, so a second pass cannot begin before
/// this one is settled. It is intentionally a linear boundary rather than a
/// mode on ordinary [`Session::extend`], whose commit-all behavior is unchanged.
#[must_use = "a target extension must be committed; dropping it aborts the pass"]
pub struct TargetExtension<'a> {
    session: &'a mut Session,
    boundary: TargetBoundary,
    post_row_predictions: Vec<usize>,
    settled: bool,
}

impl TargetExtension<'_> {
    /// Absolute position of proposed row zero.
    pub fn base_position(&self) -> usize {
        self.boundary.base_position()
    }

    /// Number of rows in this target pass.
    pub fn proposed_rows(&self) -> usize {
        self.boundary.proposed()
    }

    /// Target-model argmax immediately before proposed row zero.
    ///
    /// Linear speculative acceptance compares the first proposed token against
    /// this value.
    pub fn base_prediction(&self) -> usize {
        self.boundary
            .base_last()
            .expect("begin_target_extension requires an existing prediction")
    }

    /// Target-model argmax after each proposed input row, in row order.
    ///
    /// Proposal `i + 1` is checked against `post_row_predictions[i]`; after
    /// accepting `k > 0` rows, `post_row_predictions[k - 1]` is the next target
    /// token.
    pub fn post_row_predictions(&self) -> &[usize] {
        &self.post_row_predictions
    }

    /// Keep exactly `accepted` leading rows and discard the remaining suffix.
    ///
    /// `accepted = 0` is valid and restores the pre-pass state; accepting every
    /// row is the transactional twin of ordinary [`Session::extend`]. An invalid
    /// count returns an error and the value's drop then aborts the pass.
    pub fn commit(mut self, accepted: usize) -> Result<TargetCommit> {
        let settled =
            self.session
                .accept_target(self.boundary, &self.post_row_predictions, accepted)?;
        self.settled = true;
        Ok(TargetCommit {
            accepted: settled.accepted,
            position: settled.position,
            next: settled
                .last
                .expect("a target transaction begins from an existing prediction"),
        })
    }

    /// Discard every proposed row and restore the pre-pass target state.
    pub fn abort(mut self) -> Result<()> {
        self.session
            .abort_target(self.boundary, &self.post_row_predictions)?;
        self.settled = true;
        Ok(())
    }
}

impl Drop for TargetExtension<'_> {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        // Settlement is tensor bookkeeping after the whole stack has already
        // been validated. If an internal invariant nevertheless fails, poison
        // the Session rather than release a value whose layer caches disagree.
        if self
            .session
            .abort_target(self.boundary, &self.post_row_predictions)
            .is_err()
        {
            self.session.torn = true;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassMode {
    Commit,
    Target,
}

enum PassOutput {
    Committed(usize),
    Target(Vec<usize>),
}

/// One device layer viewed through the backend-free target settlement trait.
struct PendingTargetLayer<'a> {
    cache: &'a mut LayerCache,
    window: Option<usize>,
    kernel: usize,
}

impl PrefixCache for PendingTargetLayer<'_> {
    fn pending_rows(&self) -> Option<usize> {
        let rows = self.cache.attn.pending_rows()?;
        let hist = self.kernel.checked_sub(1)?;
        let attn_rows = self.cache.attn_sconv_pending.as_ref()?.dims()[0].checked_sub(hist)?;
        let mlp_rows = self.cache.mlp_sconv_pending.as_ref()?.dims()[0].checked_sub(hist)?;
        (rows == attn_rows && rows == mlp_rows).then_some(rows)
    }

    fn end_position(&self) -> usize {
        self.cache.attn.base() + self.cache.attn.len()
    }

    fn commit_prefix(&mut self, keep: usize) {
        self.cache.attn.commit(keep, self.window);
        let hist = self.kernel - 1;

        let attn = self
            .cache
            .attn_sconv_pending
            .take()
            .expect("target settlement validated the attention convolution window");
        if keep > 0 {
            let dim = attn.dims()[1];
            self.cache.attn_sconv =
                dev_lane::conv_history(attn.slice([0..hist + keep, 0..dim]), self.kernel);
        }

        let mlp = self
            .cache
            .mlp_sconv_pending
            .take()
            .expect("target settlement validated the MLP convolution window");
        if keep > 0 {
            let dim = mlp.dims()[1];
            self.cache.mlp_sconv = Some(dev_lane::conv_history(
                mlp.slice([0..hist + keep, 0..dim]),
                self.kernel,
            ));
        }
    }
}

impl Session {
    /// Open a model and make it ready to answer.
    ///
    /// This is `inkling_forward`'s startup, in a function: read the config out of
    /// the same source as the weights, price admission for this layer range, move
    /// this rank's share into one anonymous arena, register it with the GPU, and
    /// bind the two global tables the range owns. What it does NOT do is bind the
    /// per-layer weights — those bind on the first pass that reaches each layer,
    /// which is what keeps a lane nobody takes from paying for an upload.
    pub fn load(cfg: SessionConfig) -> Result<Self> {
        // `INK_TP` is a WITHIN-layer split across two processes joined by a
        // per-layer collective. A `Session` is one rank of such a pair and the
        // group has to be formed by the serving layer that knows about the other
        // rank, because forming it here would block this call on a peer starting
        // up somewhere else. Refused rather than half-supported.
        anyhow::ensure!(
            std::env::var("INK_TP").is_err(),
            "INK_TP is a within-layer split across two PROCESSES: a Session is one RANK of the \
             pair, and the NCCL rendezvous that joins them is a collective every rank must \
             reach. Forming it inside `Session::load` would block this call until a peer on \
             another box started. Form and warm a `tpcomm::Group`, then pass it to \
             `Session::load_with_group`."
        );
        Self::load_inner(cfg, None)
    }

    /// Open one rank of a tensor-parallel model with an already formed group.
    ///
    /// The serving process owns rendezvous and calls [`Group::warm`] before
    /// entering here. No environment variable is read for rank or world: the
    /// Group is the one fact from which the startup slice, bind shapes and
    /// collectives are all derived. Every rank must run the exact full layer
    /// range; embedding and unembedding remain replicated.
    pub fn load_with_group(cfg: SessionConfig, group: Group) -> Result<Self> {
        anyhow::ensure!(
            group.tp().is_split(),
            "Session::load_with_group needs a split Group; use Session::load for one rank"
        );
        anyhow::ensure!(
            cfg.config_override.is_none(),
            "a tensor-parallel Session requires config.json from the authoritative model \
             collection. A per-rank config override is not covered by the model identity and \
             could make two ranks run different numerical graphs."
        );
        Self::load_inner(cfg, Some(group))
    }

    fn load_inner(cfg: SessionConfig, group: Option<Group>) -> Result<Self> {
        super::fatal::arm();

        // Refuse an arbitrary raw client before opening or copying a byte of
        // the model. `form_default` records the Burn device from which the
        // Group's client came; using that exact witness is what makes the
        // communicator and all later tensors one device fact.
        let dev = match group.as_ref() {
            Some(group) => group.default_device()?,
            None => burn::backend::cuda::CudaDevice::default(),
        };

        anyhow::ensure!(
            std::env::var("INK_PIPE").is_err(),
            "INK_PIPE is the two-node LAYER split's wire, and it belongs to the process that \
             owns the sockets, not to a Session. Give each end its own Session with its own \
             `layers` range and carry the hidden state between them yourself."
        );

        let mut src = Weights::open(&cfg.pile)
            .with_context(|| format!("opening model collection in {}", cfg.pile.display()))?;

        // The config comes from the SAME source as the weights. In a pile it is
        // FACTS, one entity per JSON scalar, so this is a query and not a stored
        // file being read back.
        let text = match &cfg.config_override {
            Some(p) => std::fs::read_to_string(p)
                .with_context(|| format!("config override {}", p.display()))?,
            None => src
                .document("config.json")
                .context(
                    "this pile carries no config.json. Ingest the checkpoint's sidecars as \
                     facts (inkling_meta_gate <ckpt> <pile> --signing-key <existing-key>), \
                     or set SessionConfig::config_override to run without them",
                )?
                .to_string(),
        };
        let conf = InklingConfig::from_json(&text).context("parsing config.json")?;
        let t = &conf.text_config;
        let tp = group.as_ref().map(Group::tp);

        let (lo, hi) = (cfg.layers.start, cfg.layers.end);
        let partial = validate_layer_range(&cfg.layers, t.num_hidden_layers, tp)?;
        // A batched append is a prefill-shaped pass against an existing cache:
        // its activations are a function of its width in exactly the same way,
        // and `prefill_budget` is the width admission reserved headroom for.
        // Refused here rather than at the allocator, where it would arrive as a
        // failed buffer with nothing to say about which knob caused it.
        anyhow::ensure!(
            cfg.extend_batch >= 1,
            "an extend pass appends at least one position"
        );
        anyhow::ensure!(
            cfg.extend_batch <= cfg.prefill_budget,
            "extend_batch {} is wider than the {}-token prefill budget admission reserves \
             activations for: a batched append allocates like a prefill of the same width. \
             Raise prefill_budget or lower extend_batch.",
            cfg.extend_batch,
            cfg.prefill_budget
        );
        anyhow::ensure!(
            cfg.prefill_budget <= cfg.context_budget,
            "the {}-token prefill chunk is wider than the {}-token total context budget. \
             Raise context_budget or lower prefill_budget.",
            cfg.prefill_budget,
            cfg.context_budget
        );
        anyhow::ensure!(
            cfg.target_budget <= cfg.prefill_budget,
            "target_budget {} is wider than the {}-token prefill budget admission reserves \
             activations for. Raise prefill_budget or lower target_budget.",
            cfg.target_budget,
            cfg.prefill_budget
        );

        // A Session is ONE process and has to be able to answer, so it always
        // owns the final norm and the unembedding — exactly as `inkling_forward`
        // does when no `INK_PIPE` is set. What that means on a range that stops
        // short of the last layer is that it unembeds an INCOMPLETE STACK, and
        // the tokens are diagnostic rather than the model's. Said out loud
        // rather than left to be inferred from a fluent-looking wrong answer.
        let owns_embed = lo == 0;

        let h = t.hidden_size;
        // The router arm and the within-layer split decide WHICH weights the
        // BF16 lane binds, so admission has to know them before it prices the
        // arena copy. `from_env` is pure and is the same call the lane itself
        // makes, so reading it twice cannot disagree.
        let allocator = match group.as_ref() {
            Some(group) => group.allocator()?,
            None => super::pool::choose_memory_config(),
        };
        let cleanup = session_cleanup_policy(CleanupPolicy::choose())?;
        let cleanup_gate = CleanupGate::new(cleanup);
        let mut admission = super::budget::AdmissionPolicy::runtime(allocator)
            .with_router_bf16(RouterArm::from_env() == RouterArm::Bf16)
            .with_drafting(false)
            .with_tp_world(tp.map(Tp::world).unwrap_or(1));
        for layer in lo..hi {
            if !t.is_dense(layer) {
                let experts = format!("model.llm.layers.{layer}.mlp.experts.w13_weight");
                if !src.is_nvfp4(&experts) {
                    admission = admission.with_plain_bf16_layer(layer);
                } else {
                    admission = admission.with_wide_routed_layer(layer);
                }
            }
        }

        // Price the prefill's activations before anything is copied: the arena
        // has to leave room for them, and a copy that filled the box would fail
        // later, at a buffer, with nothing to say about why.
        let attention_bytes = super::budget::chunked_prefill_activation_bytes(
            t,
            lo..hi,
            cfg.prefill_budget,
            cfg.context_budget,
            admission,
        );

        // Move this rank's share into ONE anonymous allocation before any GPU
        // handle can alias it. The routed experts are cut here or nowhere: there
        // is no second pass over 80 GiB that could cut them afterwards.
        let mut globals: Vec<&str> = Vec::new();
        if owns_embed {
            globals.push("model.llm.embed.weight");
        }
        // Always: a Session has to be able to answer, so it always binds the
        // unembedding.
        globals.push("model.llm.unembed.weight");
        src.copy_share(lo..hi, &globals, attention_bytes, admission, tp)?;

        // The compute client taken FROM a Burn tensor rather than constructed
        // beside it: `seam::handle_of` hands a Burn allocation to a raw kernel on
        // this client, and two clients would be a wrong answer rather than an
        // error.
        let client = match group.as_ref() {
            Some(group) => group.client(),
            None => super::seam::client_of(&BT::<Bk, 2>::zeros([1, 1], &dev)),
        };

        // One registration for the whole arena -- a pile is one file, so a
        // zero-copy lane registers once and every later bind aliases rather
        // than copies. A target that cannot register
        // host mappings falls back to a copying bind, which is slower but not
        // wrong -- `Aliases::disabled()` counts what it copied instead of
        // aliasing it.
        let aliases = Some(
            super::fp4gemm::Aliases::register(&client, src.mappings()?)
                .unwrap_or_else(super::fp4gemm::Aliases::disabled),
        );

        anyhow::ensure!(
            owns_embed,
            "a Session that does not start at layer 0 has no embedding table and cannot turn \
             token ids into a hidden state. Feed it the previous rank's hidden state instead \
             (not yet exposed), or give it a range starting at 0."
        );
        let embed = {
            let leaf = src.stored("model.llm.embed.weight")?;
            anyhow::ensure!(
                leaf.elem == Elem::Bf16,
                "the embedding table is {:?}; this lane reads the stored BF16",
                leaf.elem
            );
            leaf.bytes.to_vec()
        };
        let embed_norm = src.held("model.llm.embed_norm.weight")?.data.clone();

        let (final_norm, unembed) = {
            let fnorm = src.held("model.llm.norm.weight")?;
            let leaf = src.stored("model.llm.unembed.weight")?;
            anyhow::ensure!(
                leaf.elem == Elem::Bf16,
                "the unembed table is stored as {:?}, and this lane multiplies BF16 by \
                 BF16. Widening it to reuse an f32 path is what a 4-bit model exists to \
                 avoid.",
                leaf.elem
            );
            let (rows, cols) = (leaf.dims[0] as usize, leaf.dims[1] as usize);
            anyhow::ensure!(
                rows == t.vocab_size && cols == h,
                "unembed is {rows}x{cols}, expected {}x{h}",
                t.vocab_size
            );
            // The half of NVFP4 that costs no calibration: the WEIGHT stays
            // four bits, the activation stays BF16 and is dequantised in
            // registers inside the B-fragment load. 1.65 GB of table at a
            // quarter of the bytes, and the per-step floor cut by the same
            // ratio.
            let packed = quantized_bf16(&client, &leaf.bytes, rows, cols);
            (
                up1r::<Bk>(&fnorm.data, h, &dev),
                w4a16_bind(&client, packed, true),
            )
        };

        // Pay the content hash for this range's experts HERE, once, rather than
        // in whichever decode step first routes to each of them. Rule 6 says
        // never touch the SSD once running, and this is the read that was doing
        // it.
        if cfg.warm_experts {
            src.warm_experts(lo..hi, |_, _, _| {})?;
        }

        println!(
            "  session            : layers {lo}..{hi}{}, prefill chunk {} tokens, context budget \
             {} tokens{}",
            match partial {
                true =>
                    " (PARTIAL STACK -- unembeds through layers it did not all run, so \
                          the tokens are diagnostic, not the model's)",
                false => "",
            },
            cfg.prefill_budget,
            cfg.context_budget,
            tp.map(|tp| format!(", tensor rank {} of {}", tp.rank(), tp.world()))
                .unwrap_or_default(),
        );
        println!(
            "    pool cleanup: {}; polling {}",
            cleanup.name(),
            cleanup_gate.schedule()
        );

        Ok(Self {
            cfg: conf,
            src,
            dev,
            client,
            cleanup_gate,
            group,
            aliases,
            lo,
            hi,
            partial,
            layers: BTreeMap::new(),
            dense: DeviceDense::default(),
            moe: MoeState::default(),
            shared_halved: super::load::shared_w13_halved(),
            embed,
            embed_norm,
            final_norm,
            unembed,
            caches: Vec::new(),
            pos: 0,
            last: None,
            extend_batch: cfg.extend_batch,
            prefill_budget: cfg.prefill_budget,
            target_budget: cfg.target_budget,
            context_budget: cfg.context_budget,
            torn: false,
            seq: next_seq(),
        })
    }

    /// How many positions one [`Session::extend`] pass appends at once, after
    /// the session is open. See [`SessionConfig::extend_batch`] for what the
    /// number means and why `1` is a value worth having.
    ///
    /// A setter and not a builder because the one caller that needs to CHANGE
    /// it is the gate that builds the same cache both ways on one session:
    /// re-loading eighty gibibytes to move a number would make the two arms
    /// two different processes, which is exactly what the comparison must not
    /// be.
    pub fn set_extend_batch(&mut self, rows: usize) -> Result<()> {
        anyhow::ensure!(rows >= 1, "an extend pass appends at least one position");
        anyhow::ensure!(
            rows <= self.prefill_budget,
            "extend_batch {rows} is wider than the {}-token prefill budget this session's \
             admission reserved activations for",
            self.prefill_budget
        );
        self.extend_batch = rows;
        Ok(())
    }

    /// The model's configuration, as the source stated it.
    pub fn config(&self) -> &InklingConfig {
        &self.cfg
    }

    /// Canonical identity of the projected model facts backing this session.
    pub fn model_identity(&self) -> [u8; 32] {
        self.src.model_identity()
    }

    /// How many positions the KV cache holds. The next token's position, and the
    /// length of the sequence this session has attended to.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Prove that every layer still holds the complete attention span its kind
    /// requires at [`Self::position`].
    ///
    /// Position monotonicity is not enough: a windowed cache that accidentally
    /// drops one extra row has a smaller `base + len`, so the guard before the
    /// next append becomes easier to satisfy while that layer attends over too
    /// little context forever.  The exact local invariant is therefore
    /// `len = min(position, window)` and `base = position - len`.  Global
    /// layers are the same statement with an unbounded window: `len = position`
    /// and `base = 0`.
    ///
    /// Returns the number of local layers checked, which lets a serving seam
    /// report that the model's windowed majority actually participated in the
    /// proof rather than merely saying that an empty loop succeeded.
    pub fn validate_cache_completeness(&self) -> Result<usize> {
        anyhow::ensure!(
            !self.torn,
            "a torn Session cannot make a claim about cache completeness"
        );
        if self.pos == 0 {
            anyhow::ensure!(
                self.caches.is_empty(),
                "position 0 has {} layer cache(s)",
                self.caches.len()
            );
            return Ok(0);
        }
        anyhow::ensure!(
            self.caches.len() == self.hi - self.lo,
            "{} caches for {} layers at position {}",
            self.caches.len(),
            self.hi - self.lo,
            self.pos
        );

        let t = &self.cfg.text_config;
        let mut local_layers = 0usize;
        for layer in self.lo..self.hi {
            let cache = &self.caches[layer - self.lo].attn;
            let kind = t.attn_kind(layer);
            local_layers += usize::from(kind == AttnKind::Local);
            let (expected_base, expected_len) =
                required_cache_span(kind, self.pos, t.sliding_window_size);
            anyhow::ensure!(
                cache.len() == expected_len && cache.base() == expected_base,
                "layer {layer} {:?} cache is base {} + len {} at position {}, expected base \
                 {expected_base} + len {expected_len}",
                kind,
                cache.base(),
                cache.len(),
                self.pos
            );
        }
        Ok(local_layers)
    }

    /// Which layers this rank runs.
    pub fn layer_range(&self) -> std::ops::Range<usize> {
        self.lo..self.hi
    }

    /// Whether this Session runs a strict subrange of the stack, and its tokens
    /// are therefore diagnostic rather than the model's.
    pub fn is_partial_stack(&self) -> bool {
        self.partial
    }

    /// Run proposed tokens through the target model without committing them yet.
    ///
    /// This is the transaction boundary a speculative caller needs and no more:
    /// it does not draft, choose candidates, or decide acceptance.
    /// [`TargetExtension::base_prediction`] is the prediction for proposed row
    /// zero, and row `i` of [`TargetExtension::post_row_predictions`] is the
    /// prediction after proposed input row `i` (therefore the comparison for
    /// proposal `i + 1`). The caller applies its own acceptance policy and
    /// commits one leading count, or aborts.
    ///
    /// The Session must already hold a prefix. Target transactions are disabled
    /// unless [`SessionConfig::target_budget`] explicitly admits a width, and
    /// that width remains bounded by prefill admission because it allocates the
    /// same widened activations. No ordinary path calls this method, so enabling
    /// a drafter above the Session remains an explicit deployment decision.
    pub fn begin_target_extension<'a>(
        &'a mut self,
        proposed: &[usize],
    ) -> Result<TargetExtension<'a>> {
        // Width is a pure, backend-free preflight and deliberately comes before
        // every cache check or pending-row operation. A disabled or over-width
        // target request cannot begin a transaction and therefore has nothing
        // to roll back.
        let width =
            TargetWidthAdmission::new(proposed.len(), self.target_budget, self.prefill_budget)?;
        anyhow::ensure!(
            !self.torn,
            "a previous pass tore this Session; reset or rewind before beginning a target extension"
        );
        anyhow::ensure!(
            !proposed.is_empty(),
            "a target extension proposes at least one token"
        );
        let end = self
            .pos
            .checked_add(width.rows())
            .context("target sequence position overflow")?;
        anyhow::ensure!(
            end <= self.context_budget,
            "this target extension would advance the sequence from {} to {end} positions, past \
             the admitted {}-token context budget. Start a new session or raise context_budget \
             so its persistent KV is priced before the model loads.",
            self.pos,
            self.context_budget
        );
        anyhow::ensure!(
            !self.caches.is_empty(),
            "a target extension needs an existing prefix; prefill the Session first"
        );
        anyhow::ensure!(
            self.last.is_some(),
            "the Session has caches but no target prediction; reset this inconsistent sequence"
        );
        anyhow::ensure!(
            self.caches.len() == self.hi - self.lo,
            "{} caches for {} target layers",
            self.caches.len(),
            self.hi - self.lo
        );
        anyhow::ensure!(
            self.caches.iter().all(|cache| {
                cache.attn.pending_rows().is_none()
                    && cache.attn_sconv_pending.is_none()
                    && cache.mlp_sconv_pending.is_none()
            }),
            "a target extension is already pending"
        );
        if let Err(error) = self.validate_cache_completeness() {
            self.torn = true;
            return Err(error.context("the target extension found an incomplete committed cache"));
        }

        let boundary = TargetBoundary::new(self.pos, self.last, width.rows())?;
        let predictions = match self.forward_pass(proposed, PassMode::Target) {
            Ok(PassOutput::Target(predictions)) => predictions,
            Ok(PassOutput::Committed(_)) => {
                unreachable!("a target pass returned a committed prediction")
            }
            Err(source) => {
                if let Err(rollback) = self.abort_partial_target(boundary) {
                    self.torn = true;
                    return Err(source.context(format!(
                        "the target pass failed and its partial cache could not be aborted: \
                         {rollback:#}"
                    )));
                }
                return Err(source.context("the target pass failed; its pending rows were aborted"));
            }
        };
        debug_assert_eq!(predictions.len(), proposed.len());
        Ok(TargetExtension {
            session: self,
            boundary,
            post_row_predictions: predictions,
            settled: false,
        })
    }

    fn accept_target(
        &mut self,
        boundary: TargetBoundary,
        predictions: &[usize],
        accepted: usize,
    ) -> Result<TargetSettlement> {
        self.settle_target(boundary, predictions, Some(accepted))
    }

    fn abort_target(
        &mut self,
        boundary: TargetBoundary,
        predictions: &[usize],
    ) -> Result<TargetSettlement> {
        self.settle_target(boundary, predictions, None)
    }

    /// Settle a complete target pass across every layer. `Some(k)` is an
    /// acceptance decision (including accept-zero); `None` is an explicit abort.
    fn settle_target(
        &mut self,
        boundary: TargetBoundary,
        predictions: &[usize],
        accepted: Option<usize>,
    ) -> Result<TargetSettlement> {
        let layer_count = self.hi - self.lo;
        let kernel = self.cfg.text_config.sconv_kernel_size;
        let windows: Vec<Option<usize>> = (self.lo..self.hi)
            .map(|layer| {
                (self.cfg.text_config.attn_kind(layer) == AttnKind::Local)
                    .then_some(self.cfg.text_config.sliding_window_size)
            })
            .collect();
        let settled = {
            let Session {
                caches, pos, last, ..
            } = self;
            let mut layers: Vec<PendingTargetLayer<'_>> = caches
                .iter_mut()
                .zip(windows)
                .map(|(cache, window)| PendingTargetLayer {
                    cache,
                    window,
                    kernel,
                })
                .collect();
            debug_assert_eq!(layers.len(), layer_count);
            match accepted {
                Some(keep) => boundary.accept(&mut layers, pos, last, predictions, keep)?,
                None => boundary.abort(&mut layers, pos, last, predictions)?,
            }
        };
        if let Err(error) = self.validate_cache_completeness() {
            self.torn = true;
            return Err(error.context("target settlement left an incomplete committed cache"));
        }
        Ok(settled)
    }

    /// Undo whatever prefix of a target pass ran before an ordinary `Result`
    /// error. Unlike normal settlement, not every layer is required to have
    /// reached pending state: completed layers are aborted and untouched layers
    /// are checked at the original position.
    fn abort_partial_target(&mut self, boundary: TargetBoundary) -> Result<()> {
        anyhow::ensure!(
            self.pos == boundary.base_position(),
            "the Session position moved during an uncommitted target pass"
        );
        let t = &self.cfg.text_config;
        for (slot, cache) in self.caches.iter_mut().enumerate() {
            let layer = self.lo + slot;
            let window = (t.attn_kind(layer) == AttnKind::Local).then_some(t.sliding_window_size);
            // Idempotent when this layer had not been reached. If attention had
            // appended before a later operation failed, keep-zero restores its
            // K/V projections and both attention-convolution histories.
            cache.attn.commit(0, window);
            cache.attn_sconv_pending = None;
            cache.mlp_sconv_pending = None;
            anyhow::ensure!(
                cache.attn.base() + cache.attn.len() == boundary.base_position(),
                "target layer {layer} did not roll back to position {}",
                boundary.base_position()
            );
        }
        self.validate_cache_completeness()
            .context("the partial target rollback left an incomplete committed cache")?;
        Ok(())
    }

    /// Drop the sequence and start a fresh one against the SAME warm weights.
    ///
    /// This is what a serving process does between conversations. It releases
    /// the KV pages and the convolution histories and keeps everything that took
    /// minutes to load — which is the entire reason a session is worth holding.
    pub fn reset(&mut self) {
        self.caches.clear();
        self.pos = 0;
        self.last = None;
        // Throwing the caches away is what un-tears a torn session: there is
        // nothing left for the layers to disagree about.
        self.torn = false;
        // A new sequence, so every [`Checkpoint`] taken from the old one stops
        // being a rewind target. Their device buffers are still alive (a
        // checkpoint holds handles), so without this a rewind after a reset
        // would succeed and put the session back into the PREVIOUS
        // conversation's cache while the caller believed it had started a fresh
        // one.
        self.seq = next_seq();
    }

    /// Keep where this session stands, so it can be put back here later.
    ///
    /// # The operation this is for
    ///
    /// A caller whose prompt is a chain of settled chunks — a memory cover
    /// decomposed into recall pairs, say — takes one of these after each chunk.
    /// When the chain changes it finds the first chunk that differs, rewinds to
    /// the checkpoint before it, and extends with the new tail: every chunk
    /// before the change keeps its KV and is never read again.
    ///
    /// That is a rewind and not a re-prefill, and the difference is the whole
    /// point. It is also why the granularity is the caller's: a checkpoint per
    /// chunk makes the chunk the unit at which the cache can be recovered.
    ///
    /// Refuses on a session with no sequence in flight — position 0 is what
    /// [`Session::reset`] gets you, and it costs nothing to keep.
    pub fn checkpoint(&self) -> Result<Checkpoint> {
        anyhow::ensure!(
            !self.torn,
            "a pass failed part way through the layer stack, so this session's caches stand at \
             two different positions and a checkpoint of them would be a rewind target that \
             restores the inconsistency. `reset`, or `rewind` to a checkpoint taken before it."
        );
        anyhow::ensure!(
            !self.caches.is_empty(),
            "this Session holds no sequence, so there is nothing to come back to. `reset` is \
             the way back to position 0."
        );
        anyhow::ensure!(
            self.caches.len() == self.hi - self.lo,
            "{} caches for {} layers",
            self.caches.len(),
            self.hi - self.lo
        );
        let t = &self.cfg.text_config;
        let mut layers = Vec::with_capacity(self.caches.len());
        for layer in self.lo..self.hi {
            let slot = layer - self.lo;
            // The SAME window `forward` hands the layer, derived the same way:
            // a rewind point that disagreed with the layer about whether it
            // forgets is the one mistake this whole type exists to prevent.
            let window = match t.attn_kind(layer) == AttnKind::Local {
                true => Some(t.sliding_window_size),
                false => None,
            };
            let c = &self.caches[slot];
            anyhow::ensure!(
                c.attn_sconv_pending.is_none() && c.mlp_sconv_pending.is_none(),
                "layer {layer} holds an uncommitted speculative convolution window; a Session \
                 never drafts, so this cache did not come from one"
            );
            layers.push(LayerRewind {
                attn: c.attn.rewind_point(window),
                attn_sconv: c.attn_sconv.clone(),
                mlp_sconv: c.mlp_sconv.clone(),
            });
        }
        Ok(Checkpoint {
            pos: self.pos,
            last: self.last,
            seq: self.seq,
            layers,
        })
    }

    /// Put this session back where `cp` was taken and drop everything after it.
    ///
    /// [`Session::position`] returns `cp.position()` afterwards, and
    /// [`Session::extend`] continues from there — with DIFFERENT tokens if that
    /// is what changed, which is the operation this exists for.
    ///
    /// # It fails loudly, and the failures are the design
    ///
    /// * A checkpoint from another session, or from this one before a
    ///   [`Session::reset`], is refused by sequence id. Its handles are still
    ///   alive, so the alternative to refusing is a session that silently
    ///   becomes a different conversation.
    /// * A checkpoint from AHEAD of where the session now stands is refused:
    ///   the rows between are not in the cache to be restored, and a "rewind"
    ///   forward would leave the position counter ahead of the keys.
    /// * A checkpoint whose layer count is not this session's is refused.
    ///
    /// There is deliberately no `truncate_to(pos)` beside this. See
    /// [`Checkpoint`] for why a position is not a thing a caller may name.
    pub fn rewind(&mut self, cp: &Checkpoint) -> Result<()> {
        anyhow::ensure!(
            cp.seq == self.seq,
            "this checkpoint was taken from a different sequence (checkpoint {}, session {}) -- \
             either from another Session or from this one before a `reset`. Its device buffers \
             are still alive, so rewinding to it would put this session into a conversation it \
             is no longer having.",
            cp.seq,
            self.seq
        );
        anyhow::ensure!(
            !self.caches.is_empty(),
            "this Session holds no sequence to rewind"
        );
        anyhow::ensure!(
            cp.pos <= self.pos,
            "a checkpoint at position {} against a session at {}: a rewind goes BACK, and the \
             positions between are not in this cache to be restored",
            cp.pos,
            self.pos
        );
        anyhow::ensure!(
            cp.layers.len() == self.caches.len(),
            "a {}-layer checkpoint against a {}-layer session",
            cp.layers.len(),
            self.caches.len()
        );
        for (slot, l) in cp.layers.iter().enumerate() {
            let c = &mut self.caches[slot];
            c.attn.rewind_to(&l.attn);
            c.attn_sconv = l.attn_sconv.clone();
            c.mlp_sconv = l.mlp_sconv.clone();
            c.attn_sconv_pending = None;
            c.mlp_sconv_pending = None;
        }
        self.pos = cp.pos;
        self.last = cp.last;
        // A rewind repairs a TORN session, and it is the only thing besides
        // `reset` that can. The loop above restored every layer from one
        // instant, so whatever disagreement a half-finished pass left is gone —
        // and it is gone even for the layers that were AHEAD, because a
        // checkpoint is a position the whole stack stood at and `rewind_to`
        // truncates or replaces each store to reach it.
        self.torn = false;
        Ok(())
    }

    /// Attend to `ids` as the start of a sequence and return the token that
    /// follows them.
    ///
    /// Refuses on a session that already has a sequence in flight: continuing one
    /// is [`Session::extend`], and silently treating a second prefill as a
    /// continuation is the kind of state confusion that produces fluent wrong
    /// text rather than an error.
    ///
    /// Prompts wider than [`SessionConfig::prefill_budget`] are appended in
    /// consecutive passes of that width. The first pass establishes the cache;
    /// later passes use the same committed batched-append path as [`Self::extend`].
    /// Every pass therefore stays inside the activation width admission priced,
    /// while the KV store alone grows with the complete logical prefix. A chunk
    /// boundary is a commit point, never a missing token or a reset.
    pub fn prefill(&mut self, ids: &[usize]) -> Result<usize> {
        anyhow::ensure!(
            self.pos == 0,
            "this Session already holds {} positions. Use `extend` to continue the sequence, \
             or `reset` to start a new one.",
            self.pos
        );
        anyhow::ensure!(!ids.is_empty(), "a prefill with no tokens would be vacuous");
        let mut out = 0;
        for chunk in ids.chunks(self.prefill_budget) {
            out = self.forward(chunk)?;
        }
        Ok(out)
    }

    /// Attend to tokens that are NEW since the last call, continuing the
    /// sequence, and return the token that follows them.
    ///
    /// This is the delta form, and it is the one a conversation uses: the caller
    /// hands over what has happened since the last turn — never a re-rendered
    /// transcript — because the KV cache is still holding everything before it.
    /// An empty delta re-answers from the current state, which is
    /// [`Session::step`].
    ///
    /// # It appends the delta in ONE BATCHED PASS
    ///
    /// This walked the delta one position at a time until 2026-08-27, on the
    /// reasoning that a cached pass over `k > 1` rows is the SPECULATIVE batch
    /// and reaching for it means getting a commit path right that nothing here
    /// would exercise. The reasoning was sound and the conclusion was
    /// expensive: measured at layers 0..21 on a GB10, a walked delta costs
    /// **47.4 ms a token** against **11.32 ms** for the same tokens through a
    /// batched prefill — 4.19x, paid by every turn in which a faculty returns
    /// output, because a command result, a recalled memory and a tool response
    /// are all known multi-token deltas.
    ///
    /// Batched, the same 320-token delta costs **10.97 ms a token** at layers
    /// 0..21 and **3.42 ms** at 0..6, against 46.92 and 19.42 walked — 4.28x
    /// and 5.68x. The number that says it is DONE rather than merely improved
    /// is the one beside it: a warm batched `prefill` of the same prompt on the
    /// same box costs 10.88 and 3.38, so an appended token now costs a
    /// prefilled token, which is the floor. `inkling_session --batched` carries
    /// the framing rule and the rest of the table.
    ///
    /// So the commit path is spelled out instead of avoided, and it is short,
    /// because **a conversational delta has no verifier**: every token in it is
    /// already a fact about the sequence. `attention_steps` leaves its rows
    /// PENDING for a verifier to accept or roll back; this pass commits all `k`
    /// of them in the same layer iteration that produced them, so no
    /// uncommitted speculative state ever survives a call, and
    /// [`Session::checkpoint`]'s refusal to stand on one is a statement about
    /// misuse rather than about this path.
    ///
    /// Four questions that a batched append has and a walked one does not, and
    /// where each is answered:
    ///
    /// * **The sliding window.** A batch is NOT trimmed row by row as it is
    ///   appended, so a batch of `k` may carry a local layer past its window in
    ///   one go. That is correct and not an oversight:
    ///   `attention_steps` masks every (row, key) pair by the ABSOLUTE distance
    ///   `pos - abs`, so a key outside row `i`'s window contributes `-inf`
    ///   whether or not it is still stored, and the trim that follows the
    ///   commit is what makes the store bounded again. Trimming DURING the
    ///   batch would be the bug — it is why `AttnCache::commit` defers the trim
    ///   at all — because a row that is dropped mid-batch is one a later row of
    ///   the same batch may still need. Gated by
    ///   `batched_and_walked_caches_answer_the_same` over batches that straddle
    ///   the window boundary and batches LONGER than the whole window.
    /// * **What it costs while it is uncommitted.** The flip side of the same
    ///   fact: a local layer holds `window + k` rows for the duration of the
    ///   pass instead of `window`, and reads all of them. That is bounded by
    ///   `k`, which is what [`SessionConfig::extend_batch`] bounds — a delta
    ///   longer than it is appended in consecutive passes of that width rather
    ///   than one enormous one.
    /// * **Partial failure.** A pass is not all-or-nothing and cannot cheaply
    ///   be made so; see [`Session::torn`], and note that this is a property of
    ///   the layer loop rather than of batching — a one-row pass tears the same
    ///   way over one position instead of `k`.
    /// * **`last`.** The pass unembeds its LAST row only, exactly as a prefill
    ///   does, so [`Session::step`] afterwards feeds the token that follows the
    ///   whole delta. The walked arm computed an argmax per position and threw
    ///   away all but the last; this one does not compute them. Same answer,
    ///   `k - 1` fewer unembeddings.
    ///
    /// `extend_batch = 1` puts the walked arm back, and the equivalence gate
    /// uses it to build the same cache both ways on one session.
    pub fn extend(&mut self, ids: &[usize]) -> Result<usize> {
        if ids.is_empty() {
            return self.step();
        }
        if self.caches.is_empty() {
            // Nothing cached yet: this IS a prefill, and a prefill batches.
            return self.forward(ids);
        }
        let mut out = 0;
        // A chunk boundary is a commit point and nothing else, so a delta split
        // into two passes leaves the same cache one pass would — which is the
        // property that lets `extend_batch` be a resource knob rather than a
        // semantic one.
        for chunk in ids.chunks(self.extend_batch) {
            out = self.forward(chunk)?;
        }
        Ok(out)
    }

    /// Advance one token: feed back what the last pass produced, and return the
    /// next.
    ///
    /// One position, against a cache that already holds the whole sequence. This
    /// is the call a generation loop makes, and the reason it is cheap is that
    /// nothing before it is recomputed.
    pub fn step(&mut self) -> Result<usize> {
        let last = self
            .last
            .context("nothing to step from: prefill a prompt first")?;
        self.forward(&[last])
    }

    /// The one forward. `ids` are the positions being added; everything before
    /// them is in the caches.
    ///
    /// Three shapes: a BATCHED pass that establishes the cache (the prefill), a
    /// ONE-POSITION pass against a cache that already exists (the decode step),
    /// and a BATCHED pass against a cache that already exists (the
    /// conversational delta). The third used to be refused as "the speculative
    /// batch"; it is the same rows through the same functions, distinguished
    /// only by having no verifier — so it commits unconditionally, in the layer
    /// iteration that produced it. See [`Session::extend`].
    ///
    /// Refuses a session a previous pass tore. See [`Session::torn`].
    fn forward(&mut self, ids: &[usize]) -> Result<usize> {
        anyhow::ensure!(!ids.is_empty(), "a pass with no tokens would be vacuous");
        let end = self
            .pos
            .checked_add(ids.len())
            .context("sequence position overflow")?;
        anyhow::ensure!(
            end <= self.context_budget,
            "this pass would advance the sequence from {} to {end} positions, past the \
             admitted {}-token context budget. Start a new session or raise context_budget so \
             its persistent KV is priced before the model loads.",
            self.pos,
            self.context_budget
        );
        anyhow::ensure!(
            !self.torn,
            "a previous pass failed part way through the layer stack, so this session's caches \
             stand at two different positions and nothing this one computed would mean \
             anything. `reset` to start over, or `rewind` to a checkpoint taken before the \
             failure -- which restores every layer from one instant and is a real repair."
        );
        // Everything from here MUTATES, and the mutation is per layer. An error
        // out of the loop below leaves the stack half advanced, which no
        // caller can detect and none should have to; poison the session and
        // name the two ways out.
        let out = match self.forward_pass(ids, PassMode::Commit) {
            Ok(PassOutput::Committed(best)) => self.validate_cache_completeness().map(|_| best),
            Ok(PassOutput::Target(_)) => {
                unreachable!("an ordinary pass returned target predictions")
            }
            Err(error) => Err(error),
        };
        if out.is_err() {
            self.torn = true;
        }
        out
    }

    /// [`Session::forward`]'s mutating half, split out so that every failure
    /// path through it lands in one place: the caller poisons the session on
    /// any error, and it can only do that if the errors have somewhere to
    /// return to.
    fn forward_pass(&mut self, ids: &[usize], mode: PassMode) -> Result<PassOutput> {
        let t = &self.cfg.text_config;
        let h = t.hidden_size;
        let n = ids.len();
        let pos0 = self.pos;
        let tp = self.group.as_ref().map(Group::tp);
        let group = self.group.as_ref();
        let dev = &self.dev;
        let tp_reduce = |x: T2, calls: &mut usize| -> T2 {
            match group {
                Some(group) => {
                    *calls += 1;
                    super::tpcomm::reduce_activation(group, dev, x)
                }
                None => x,
            }
        };
        // A pass with a cache behind it is a decode step; the first one is the
        // prefill that establishes the cache.
        let cached = !self.caches.is_empty();
        debug_assert!(cached || mode == PassMode::Commit);
        super::fatal::note_pass(n, pos0 + n);
        self.cleanup_gate.begin_pass();

        // The embedding is a host gather over the stored BF16, normed on the
        // host, and uploaded once. It is the only host arithmetic left in a pass.
        let x_in = embed_and_norm_bf16(
            ids,
            &self.embed,
            &self.embed_norm,
            t.rms_norm_eps,
            t.vocab_size,
            h,
        );
        let mut xd: T2 = dev_lane_resid::as_resid(up2::<Bk>(x_in, n, h, &self.dev));

        // Read once per pass, exactly as the binary does: `RouterArm::from_env`
        // is a `OnceLock` behind an env var and the arm cannot change under a
        // running session.
        let router_arm = RouterArm::from_env();
        let t_read = std::cell::Cell::new(0f64);

        for layer in self.lo..self.hi {
            let mut tp_calls = 0usize;
            // Cache SLOT, not layer number. A rank running 20..42 keeps 22
            // caches and its first layer is slot 0 — indexing by the absolute
            // layer would walk off the end of a Vec that only holds this rank's
            // half.
            let slot = layer - self.lo;
            let kind = t.attn_kind(layer);
            let (global_heads, global_kv_heads, head_dim) = t.heads(kind);
            let (heads, kv_heads) = match tp {
                Some(tp) => (
                    tp.share("q_heads", global_heads)
                        .map_err(|e| anyhow::anyhow!("layer {layer}: {e}"))?,
                    tp.share("kv_heads", global_kv_heads)
                        .map_err(|e| anyhow::anyhow!("layer {layer}: {e}"))?,
                ),
                None => (global_heads, global_kv_heads),
            };
            let p = format!("model.llm.layers.{layer}.");

            if !self.layers.contains_key(&p) {
                let b = bind_layer(
                    &self.src,
                    &self.dev,
                    &self.client,
                    self.aliases.as_ref(),
                    &p,
                    layer,
                    t,
                    tp,
                    router_arm,
                    false,
                    &t_read,
                )?;
                self.layers.insert(p.clone(), b.layer);
            }
            let ld = self.layers.get(&p).expect("inserted directly above");
            let shared_halved = self.shared_halved;

            // ---- attention ------------------------------------------------
            let hn = dev_lane_resid::rms_norm(xd.clone(), ld.attn_norm.clone(), t.rms_norm_eps);
            let dims = AttnDims {
                hidden: h,
                heads,
                kv_heads,
                head_dim,
                d_rel: t.d_rel,
                rel_extent: t.rel_span(kind),
                kernel: t.sconv_kernel_size,
                rms_eps: t.rms_norm_eps,
                kind,
            };
            // The same distinction the causal mask carries, in the form the
            // cache needs: how far back a query may look, and therefore how much
            // of the cache can never be read again.
            let window = match kind == AttnKind::Local {
                true => Some(t.sliding_window_size),
                false => None,
            };
            let ls = LogScaling {
                n_floor: t.log_scaling_n_floor as f32,
                alpha: t.log_scaling_alpha as f32,
            };

            let a = match (cached, mode, n > 1) {
                // The explicit target transaction. Even one proposed row takes
                // the widened path: unlike `attention_step`, it can be kept or
                // discarded after the target predictions are known. Neither
                // attention nor the block convolution advances its committed
                // history here.
                (true, PassMode::Target, _) => {
                    let y = dev_lane::attention_steps(
                        hn,
                        &ld.attn,
                        &dims,
                        Some(ls),
                        pos0,
                        window,
                        &mut self.caches[slot].attn,
                    );
                    let y = tp_reduce(y, &mut tp_calls);
                    let (out, all) = dev_lane::short_conv_steps(
                        self.caches[slot].attn_sconv.clone(),
                        y,
                        ld.attn_sconv.clone(),
                    );
                    self.caches[slot].attn_sconv_pending = Some(all);
                    out
                }
                // THE CONVERSATIONAL DELTA: `n` known positions against a cache
                // that already holds the prefix, in one pass.
                //
                // `attention_steps` appends all `n` rows and leaves them
                // PENDING, trimming nothing — which is exactly right here, and
                // for the same reason it is right for a speculative batch: a
                // row dropped mid-batch is one a later row of the same batch
                // may still be inside the window of. The per-(row, key) mask it
                // builds from absolute positions is what makes the untrimmed
                // rows harmless, and `commit` below is what makes the store
                // bounded again.
                (true, PassMode::Commit, true) => {
                    let y = dev_lane::attention_steps(
                        hn,
                        &ld.attn,
                        &dims,
                        Some(ls),
                        pos0,
                        window,
                        &mut self.caches[slot].attn,
                    );
                    // This rank computed only its heads. Reduce the partial
                    // hidden vector BEFORE mixing it with the already-whole
                    // convolution history; moving this below the convolution
                    // stays finite and is wrong.
                    let y = tp_reduce(y, &mut tp_calls);
                    let (out, all) = dev_lane::short_conv_steps(
                        self.caches[slot].attn_sconv.clone(),
                        y,
                        ld.attn_sconv.clone(),
                    );
                    // THE COMMIT, unconditional and in the same iteration.
                    // `keep = n` because every token of a delta is a fact: there
                    // is no verifier to wait for, and waiting for one would be
                    // the only way to leave uncommitted rows behind. The trim
                    // the window needs happens inside it.
                    self.caches[slot].attn.commit(n, window);
                    // The block's own convolution memory, the same slice a
                    // verifier that accepted everything would take: `all` is
                    // `kernel - 1` history rows followed by this batch's `n`,
                    // and the history after keeping all of them is its tail.
                    self.caches[slot].attn_sconv = dev_lane::conv_history(all, t.sconv_kernel_size);
                    out
                }
                (true, PassMode::Commit, false) => {
                    let y = dev_lane::attention_step(
                        hn,
                        &ld.attn,
                        &dims,
                        Some(ls),
                        pos0,
                        window,
                        &mut self.caches[slot].attn,
                    );
                    let y = tp_reduce(y, &mut tp_calls);
                    let (out, hist) = dev_lane::short_conv_step(
                        self.caches[slot].attn_sconv.clone(),
                        y,
                        ld.attn_sconv.clone(),
                    );
                    self.caches[slot].attn_sconv = hist;
                    out
                }
                (false, PassMode::Commit, _) => {
                    let (y, attn) =
                        dev_lane::attention_prefill(hn, &ld.attn, &dims, Some(ls), window, window);
                    let y = tp_reduce(y, &mut tp_calls);
                    let hist = dev_lane::conv_history(y.clone(), t.sconv_kernel_size);
                    let out = dev_lane::short_conv(y, ld.attn_sconv.clone());
                    self.caches.push(LayerCache {
                        attn,
                        attn_sconv: hist,
                        mlp_sconv: None,
                        attn_sconv_pending: None,
                        mlp_sconv_pending: None,
                    });
                    out
                }
                (false, PassMode::Target, _) => {
                    unreachable!("a target extension requires an existing prefix")
                }
            };
            xd = dev_lane_resid::add_resid(xd, a);

            // ---- MLP ------------------------------------------------------
            let hn = dev_lane_resid::rms_norm(xd.clone(), ld.mlp_norm.clone(), t.rms_norm_eps);
            let y = match t.is_dense(layer) {
                true => {
                    let w = self.dense.dense_for(
                        &self.src,
                        &self.client,
                        self.aliases.as_ref(),
                        &p,
                        h,
                        tp,
                    )?;
                    dense_mlp_bf16(hn, w)
                }
                false => {
                    let r = ld.router.as_ref().expect("a MoE layer has a router");
                    // The router's PROJECTION is a matmul and runs on the
                    // device; its DECISION is control plane. On the default lane
                    // the decision runs on the device too (`INK_DEV_PLAN`), so
                    // nothing in the layer reads anything back and nothing in
                    // the layer blocks.
                    moe_layer(
                        &self.src,
                        &self.client,
                        self.aliases.as_ref(),
                        &mut self.dense,
                        &mut self.moe,
                        &self.dev,
                        &p,
                        layer,
                        t,
                        r,
                        hn,
                        n,
                        shared_halved,
                        tp,
                    )?
                }
            };
            // Dense and routed MLPs are both split on their intermediate axis.
            // As with attention, the partial sum must become whole before the
            // stateful short convolution consumes it.
            let y = tp_reduce(y, &mut tp_calls);
            let out = match (cached, mode, n > 1) {
                // The target half of the fourth convolution. Keep the entire
                // window beside the committed history until one prefix count is
                // applied to every layer.
                (true, PassMode::Target, _) => {
                    let h0 = self.caches[slot]
                        .mlp_sconv
                        .clone()
                        .expect("a prefill seeds the MLP convolution");
                    let (out, all) = dev_lane::short_conv_steps(h0, y, ld.mlp_sconv.clone());
                    self.caches[slot].mlp_sconv_pending = Some(all);
                    out
                }
                // The fourth of the four short convolutions a widened pass runs
                // per layer -- two are inside `attention_steps`, one is the
                // block's `attn_sconv` above, and this is the MLP's. All four
                // need the batched form, and a widened pass that left one of
                // them on the single-row kernel would convolve `n` positions
                // out of one position's history with no error at all.
                (true, PassMode::Commit, true) => {
                    let h0 = self.caches[slot]
                        .mlp_sconv
                        .clone()
                        .expect("a prefill seeds the MLP convolution");
                    let (o, all) = dev_lane::short_conv_steps(h0, y, ld.mlp_sconv.clone());
                    self.caches[slot].mlp_sconv =
                        Some(dev_lane::conv_history(all, t.sconv_kernel_size));
                    o
                }
                (true, PassMode::Commit, false) => {
                    let h0 = self.caches[slot]
                        .mlp_sconv
                        .clone()
                        .expect("a prefill seeds the MLP convolution");
                    let (o, hi) = dev_lane::short_conv_step(h0, y, ld.mlp_sconv.clone());
                    self.caches[slot].mlp_sconv = Some(hi);
                    o
                }
                (false, PassMode::Commit, _) => {
                    let hist = dev_lane::conv_history(y.clone(), t.sconv_kernel_size);
                    self.caches[slot].mlp_sconv = Some(hist);
                    dev_lane::short_conv(y, ld.mlp_sconv.clone())
                }
                (false, PassMode::Target, _) => {
                    unreachable!("a target extension requires an existing prefix")
                }
            };
            xd = dev_lane_resid::add_resid(xd, out);

            let expected = if group.is_some() { 2 } else { 0 };
            assert_eq!(
                tp_calls, expected,
                "layer {layer} issued {tp_calls} tensor-parallel collectives, not {expected} \
                 (one after attention and one after the MLP, both before their short convolution)"
            );

            // Admission deliberately charges only LIVE tensors. Free pages from
            // earlier, differently-sized KV runs are bounded here, after this
            // layer's temporaries have been consumed and before the next layer
            // asks the pool for another shape. This is the same gate the one-shot
            // forward uses; without it a chunked Session grows a history of every
            // global-cache size it has visited.
            let last_layer = layer + 1 == self.hi;
            let client = &self.client;
            let want_cleanup = self.cleanup_gate.at_layer(last_layer, || {
                client
                    .memory_usage()
                    .map(|usage| {
                        super::pool::stranded_bytes(
                            usage.bytes_reserved,
                            usage.bytes_in_use,
                            usage.bytes_padding,
                        )
                    })
                    .unwrap_or(0)
            });
            if want_cleanup {
                <Bk as burn::tensor::backend::Backend>::sync(&self.dev)
                    .expect("sync before Session pool cleanup");
                self.client.memory_cleanup();
            }
        }

        // ---- the head -----------------------------------------------------
        //
        // Only the LAST row matters to an ordinary pass: the token that follows
        // the sequence. A target verifier instead needs one prediction per row
        // to find its accepted prefix. The explicit target path pays that wider
        // head; the default path keeps slicing before the projection, where the
        // difference is a 16 KB GEMM versus an `n x 200058` one.
        let (hx, head_rows) = match mode {
            PassMode::Commit => (xd.slice([n - 1..n, 0..h]), 1),
            PassMode::Target => (xd, n),
        };
        let hs = dev_lane_resid::rms_norm(hx, self.final_norm.clone(), t.rms_norm_eps)
            .div_scalar(t.logits_mup_width_multiplier as f32);
        let uw = &self.unembed;
        let logits = dev_lane::linear_w(hs, uw).slice([0..head_rows, 0..t.effective_vocab()]);
        match mode {
            PassMode::Commit => {
                let best = argmax_row_dev(logits);
                self.pos += n;
                self.last = Some(best);
                Ok(PassOutput::Committed(best))
            }
            PassMode::Target => Ok(PassOutput::Target(argmax_rows_dev(logits))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_cache_span_is_exact_on_both_sides_of_the_window() {
        assert_eq!(required_cache_span(AttnKind::Local, 0, 512), (0, 0));
        assert_eq!(required_cache_span(AttnKind::Local, 511, 512), (0, 511));
        assert_eq!(required_cache_span(AttnKind::Local, 512, 512), (0, 512));
        assert_eq!(required_cache_span(AttnKind::Local, 513, 512), (1, 512));
        assert_eq!(
            required_cache_span(AttnKind::Local, 19_175, 512),
            (18_663, 512)
        );
        assert_eq!(
            required_cache_span(AttnKind::Global, 19_175, 512),
            (0, 19_175)
        );
    }

    #[test]
    fn session_cleanup_policy_refuses_a_stage_precision_it_cannot_deliver() {
        assert_eq!(
            session_cleanup_policy(CleanupPolicy::PerLayer).unwrap(),
            CleanupPolicy::PerLayer
        );
        let err = session_cleanup_policy(CleanupPolicy::PerStage)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Use INK_POOL_CLEANUP=1"), "{err}");
    }

    /// The layer-range rules are arithmetic and testable without a GPU: they are
    /// the difference between a run and a disk benchmark.
    #[test]
    fn a_session_config_reads_the_layer_range_it_is_given() {
        let c = SessionConfig::new("/nowhere.pile").layers(0..21);
        assert_eq!(c.layers, 0..21);
    }

    #[test]
    fn warming_experts_is_on_by_default() {
        assert!(SessionConfig::new("/nowhere.pile").warm_experts);
    }

    #[test]
    fn target_transactions_are_disabled_by_default() {
        assert_eq!(
            SessionConfig::new("/nowhere.pile").target_budget,
            DEFAULT_TARGET_BUDGET
        );
    }

    /// The default config must satisfy the rule [`Session::load`] enforces: a
    /// batched append allocates like a prefill of the same width, so it may not
    /// be wider than the width admission reserved for. A default that refused
    /// itself would only be discovered on a machine with a GPU.
    #[test]
    fn the_default_extend_batch_fits_the_default_prefill_budget() {
        let c = SessionConfig::new("/nowhere.pile");
        assert!(c.extend_batch >= 1);
        assert!(
            c.extend_batch <= c.prefill_budget,
            "extend_batch {} against a prefill budget of {}",
            c.extend_batch,
            c.prefill_budget
        );
        assert!(
            c.prefill_budget <= c.context_budget,
            "prefill budget {} against a context budget of {}",
            c.prefill_budget,
            c.context_budget
        );
    }

    #[test]
    fn a_single_rank_keeps_the_strict_subrange_rule() {
        assert!(validate_layer_range(&(0..21), 42, None).unwrap());
        assert!(validate_layer_range(&(21..42), 42, None).is_ok());
        let err = validate_layer_range(&(0..42), 42, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("whole 42-layer stack"), "{err}");
    }

    #[test]
    fn tensor_parallel_ranks_require_the_exact_full_stack() {
        let tp = Tp::new(1, 2).unwrap();
        assert!(!validate_layer_range(&(0..42), 42, Some(tp)).unwrap());
        for range in [0..21, 21..42, 1..42] {
            let err = validate_layer_range(&range, 42, Some(tp))
                .unwrap_err()
                .to_string();
            assert!(err.contains("every rank runs every layer"), "{err}");
        }
    }
}
