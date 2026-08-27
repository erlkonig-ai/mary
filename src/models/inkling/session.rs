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
//! # What a Session deliberately is NOT
//!
//! It is not the serving process, and it is not `inkling_forward`.
//!
//! `inkling_forward` is a MEASUREMENT harness that happens to run the model: it
//! carries the pipe between two nodes, the CUDA-graph capture lane, batched
//! slots and cohorts, token-tree speculation, the MTP drafting experiment, the
//! router and plan A/B arms and about a hundred `INK_*` reporting switches.
//! Every one of those is a question someone is asking about the model, and none
//! of them is something a conversation needs. They stay in the binary.
//!
//! What is here is the lane that runs when nobody sets anything: the default
//! configuration, which is also the one the frontier benchmark measures. Greedy
//! argmax, cached decode, the W4A16 head, NVFP4 KV pages, the device router and
//! the device row plan — on by default, so on here.
//!
//! # Tensor parallelism: a Session is PER RANK
//!
//! `INK_TP=rank:2` runs this model as two processes, one per box, joined by a
//! NCCL all-reduce inside every layer. Under it each rank runs *every* layer on
//! *half* of each tensor, so both ranks hold the embedding, the whole stack and
//! the unembedding, and both produce the same token.
//!
//! A `Session` is therefore one RANK, not one model, and a caller that wants a
//! TP pair holds two of them in two processes. [`Session::load`] refuses
//! `INK_TP` for exactly that reason: the rendezvous
//! ([`super::tpcomm::Group`]) is a collective, every rank must reach it, and a
//! library call that blocks until a peer on another box starts up is not a thing
//! a serving process can be handed without knowing it. Single rank works today;
//! the pair needs an addressing decision that belongs to the serving layer and
//! is written up in the commit that introduced this file.
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
    BT, Bk, DeviceDense, LayerCache, LayerDev, MoeState, RouterArm, T2, argmax_row_dev, bind_layer,
    dense_mlp_bf16, dev_lane, dev_lane_resid, moe_layer, quantized_bf16, up1r, up2, w4a16_bind,
};
use super::attn::{AttnDims, LogScaling};
use super::config::{AttnKind, InklingConfig};
use super::pile::Elem;
use super::source::Weights;
use super::stack::embed_and_norm_bf16;

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
    /// Which layers THIS process runs, as `lo..hi`. Required, and a strict
    /// subrange: no node may run the whole stack.
    pub layers: std::ops::Range<usize>,
    /// Pay the storage layer's content hash for this range's experts at load
    /// rather than in whichever decode step first routes to each of them. On by
    /// default: a serving process that skips this pays 0.5–274 ms in a random
    /// later token, which is the one variable that used to explain a whole
    /// latency spread.
    pub warm_experts: bool,
    /// The prefill length admission is priced at.
    ///
    /// Admission reserves activation headroom before the arena is filled, and
    /// the size of a prefill's largest buffer is a function of how many tokens
    /// it takes at once. A session does not know its prompts in advance, so it
    /// prices a fixed budget: big enough that a conversational turn fits under
    /// it, small enough that the reservation does not eat the arena. A prompt
    /// longer than this is not refused here — it is refused by the allocator,
    /// later and less legibly, which is why raising it is the fix rather than
    /// catching the failure.
    pub prefill_budget: usize,
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
        super::fatal::arm();

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
             another box started. Run one Session per rank and form the group above them."
        );
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
                     facts (inkling_meta_gate <ckpt> <pile>), or set SessionConfig::\
                     config_override to run without them",
                )?
                .to_string(),
        };
        let conf = InklingConfig::from_json(&text).context("parsing config.json")?;
        let t = &conf.text_config;

        let (lo, hi) = (cfg.layers.start, cfg.layers.end);
        anyhow::ensure!(
            lo < hi,
            "a Session runs LO..HI with LO < HI, got {lo}..{hi}"
        );
        anyhow::ensure!(
            hi <= t.num_hidden_layers,
            "{lo}..{hi} runs past the {}-layer stack",
            t.num_hidden_layers
        );
        // The same rule the binary enforces, and for the same measured reason:
        // one box cannot hold this model, and a process that would run the whole
        // stack re-reads its experts off the SSD between tokens. What that
        // measures is a disk.
        anyhow::ensure!(
            hi - lo < t.num_hidden_layers,
            "{lo}..{hi} is the whole {}-layer stack on one node, which does not fit ({} GiB of \
             weights). Split it: no node may run every layer, and two is the MINIMUM rather \
             than the number.",
            t.num_hidden_layers,
            144,
        );

        // A Session is ONE process and has to be able to answer, so it always
        // owns the final norm and the unembedding — exactly as `inkling_forward`
        // does when no `INK_PIPE` is set. What that means on a range that stops
        // short of the last layer is that it unembeds an INCOMPLETE STACK, and
        // the tokens are diagnostic rather than the model's. Said out loud
        // rather than left to be inferred from a fluent-looking wrong answer.
        let owns_embed = lo == 0;
        let partial = hi < t.num_hidden_layers;

        let h = t.hidden_size;
        // The router arm and the within-layer split decide WHICH weights the
        // BF16 lane binds, so admission has to know them before it prices the
        // arena copy. `from_env` is pure and is the same call the lane itself
        // makes, so reading it twice cannot disagree.
        let allocator = super::pool::choose_memory_config();
        let mut admission = super::budget::AdmissionPolicy::runtime(allocator)
            .with_router_bf16(RouterArm::from_env() == RouterArm::Bf16)
            .with_drafting(false);
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
        let attention_bytes =
            super::budget::prefill_activation_bytes(t, lo..hi, cfg.prefill_budget, admission);

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
        src.copy_share(lo..hi, &globals, attention_bytes, admission, None)?;

        let dev = burn::backend::cuda::CudaDevice::default();
        // The compute client taken FROM a Burn tensor rather than constructed
        // beside it: `seam::handle_of` hands a Burn allocation to a raw kernel on
        // this client, and two clients would be a wrong answer rather than an
        // error.
        let client = super::seam::client_of(&BT::<Bk, 2>::zeros([1, 1], &dev));

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
            "  session            : layers {lo}..{hi}{}, prefill budget {} tokens",
            match partial {
                true =>
                    " (PARTIAL STACK -- unembeds through layers it did not all run, so \
                          the tokens are diagnostic, not the model's)",
                false => "",
            },
            cfg.prefill_budget,
        );

        Ok(Self {
            cfg: conf,
            src,
            dev,
            client,
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
        })
    }

    /// The model's configuration, as the source stated it.
    pub fn config(&self) -> &InklingConfig {
        &self.cfg
    }

    /// How many positions the KV cache holds. The next token's position, and the
    /// length of the sequence this session has attended to.
    pub fn position(&self) -> usize {
        self.pos
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

    /// Drop the sequence and start a fresh one against the SAME warm weights.
    ///
    /// This is what a serving process does between conversations. It releases
    /// the KV pages and the convolution histories and keeps everything that took
    /// minutes to load — which is the entire reason a session is worth holding.
    pub fn reset(&mut self) {
        self.caches.clear();
        self.pos = 0;
        self.last = None;
    }

    /// Attend to `ids` as the start of a sequence and return the token that
    /// follows them.
    ///
    /// Refuses on a session that already has a sequence in flight: continuing one
    /// is [`Session::extend`], and silently treating a second prefill as a
    /// continuation is the kind of state confusion that produces fluent wrong
    /// text rather than an error.
    pub fn prefill(&mut self, ids: &[usize]) -> Result<usize> {
        anyhow::ensure!(
            self.pos == 0,
            "this Session already holds {} positions. Use `extend` to continue the sequence, \
             or `reset` to start a new one.",
            self.pos
        );
        self.forward(ids)
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
    /// # It walks the delta one position at a time, and that is deliberate
    ///
    /// A cached pass over `k > 1` rows is not a small generalisation of a cached
    /// pass over one. It is the SPECULATIVE batch: `attention_steps` leaves the
    /// rows PENDING in the cache, because the caller is expected to come back
    /// and say how many of them a verifier kept — rows computed from tokens the
    /// model did not choose have to be rolled back, and leaving them behind does
    /// not error, it shows up later as an acceptance rate that drifts down. A
    /// conversational delta has nothing to verify (every token in it is a fact),
    /// so it would always commit all `k`, but reaching that through the
    /// speculation machinery means getting a commit path right that nothing
    /// here would exercise.
    ///
    /// So this walks the delta through the SAME single-position path
    /// [`Session::step`] uses, which is the one that is checked. It costs `k`
    /// passes where a batched one would cost one, and it still never re-reads a
    /// token the cache already holds — which is the property the session exists
    /// for, and the one that is worth two orders of magnitude. Batching the
    /// delta is an optimisation on top, and it wants the commit semantics
    /// spelled out rather than inherited.
    pub fn extend(&mut self, ids: &[usize]) -> Result<usize> {
        if ids.is_empty() {
            return self.step();
        }
        if self.caches.is_empty() {
            // Nothing cached yet: this IS a prefill, and a prefill batches.
            return self.forward(ids);
        }
        let mut out = 0;
        for &id in ids {
            out = self.forward(&[id])?;
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
    /// Two shapes only, and the refusal below is what keeps it to two: a BATCHED
    /// pass that establishes the cache, and a ONE-POSITION pass against a cache
    /// that already exists. The third shape — several rows against an existing
    /// cache — is the speculative batch and it is not this; see
    /// [`Session::extend`], which walks a delta rather than reaching for it.
    fn forward(&mut self, ids: &[usize]) -> Result<usize> {
        anyhow::ensure!(!ids.is_empty(), "a pass with no tokens would be vacuous");
        anyhow::ensure!(
            self.caches.is_empty() || ids.len() == 1,
            "a pass of {} rows against an EXISTING cache is the speculative batch -- \
             `attention_steps` leaves those rows pending for a verifier to accept or roll \
             back, and a conversational delta has no verifier. Use `extend`, which walks the \
             delta one position at a time through the checked path.",
            ids.len()
        );
        let t = &self.cfg.text_config;
        let h = t.hidden_size;
        let n = ids.len();
        let pos0 = self.pos;
        // A pass with a cache behind it is a decode step; the first one is the
        // prefill that establishes the cache.
        let cached = !self.caches.is_empty();
        super::fatal::note_tokens(n);

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
            // Cache SLOT, not layer number. A rank running 20..42 keeps 22
            // caches and its first layer is slot 0 — indexing by the absolute
            // layer would walk off the end of a Vec that only holds this rank's
            // half.
            let slot = layer - self.lo;
            let kind = t.attn_kind(layer);
            let (heads, kv_heads, head_dim) = t.heads(kind);
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
                    None,
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

            let a = match cached {
                true => {
                    let y = dev_lane::attention_step(
                        hn,
                        &ld.attn,
                        &dims,
                        Some(ls),
                        pos0,
                        window,
                        &mut self.caches[slot].attn,
                    );
                    let (out, hist) = dev_lane::short_conv_step(
                        self.caches[slot].attn_sconv.clone(),
                        y,
                        ld.attn_sconv.clone(),
                    );
                    self.caches[slot].attn_sconv = hist;
                    out
                }
                false => {
                    let (y, attn) =
                        dev_lane::attention_prefill(hn, &ld.attn, &dims, Some(ls), window, window);
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
                        None,
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
                    )?
                }
            };
            let (out, hist) = match cached {
                true => {
                    let h0 = self.caches[slot]
                        .mlp_sconv
                        .clone()
                        .expect("a prefill seeds the MLP convolution");
                    let (o, hi) = dev_lane::short_conv_step(h0, y, ld.mlp_sconv.clone());
                    (o, hi)
                }
                false => {
                    let hist = dev_lane::conv_history(y.clone(), t.sconv_kernel_size);
                    (dev_lane::short_conv(y, ld.mlp_sconv.clone()), hist)
                }
            };
            self.caches[slot].mlp_sconv = Some(hist);
            xd = dev_lane_resid::add_resid(xd, out);
        }

        // ---- the head -----------------------------------------------------
        //
        // Only the LAST row matters: the token that follows the sequence. A
        // prefill computes `n` rows of residual stream and unembeds one of them,
        // and slicing on the INPUT rather than on the logits is the difference
        // between a 16 KB GEMM and an `n x 200058` one.
        let hx = xd.slice([n - 1..n, 0..h]);
        let hs = dev_lane_resid::rms_norm(hx, self.final_norm.clone(), t.rms_norm_eps)
            .div_scalar(t.logits_mup_width_multiplier as f32);
        let uw = &self.unembed;
        let row = dev_lane::linear_w(hs, uw).slice([0..1, 0..t.effective_vocab()]);
        let best = argmax_row_dev(row);

        self.pos += n;
        self.last = Some(best);
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
