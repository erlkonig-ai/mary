//! Token-TREE speculative decoding over Inkling's CHAINED MTP heads.
//!
//! # STATUS: NOT SHIPPED, AND THE LOGIC IS UNVERIFIED
//!
//! The measurement this module exists to make came back NEGATIVE — see the
//! VERDICT section below — so nothing here is on any decode path and the
//! tensor lane it drives (`INK_TREE`) is not a production configuration.
//!
//! What was actually established, stated exactly:
//!
//! * 35 host tests pass (topology, top-b, the drafting plan, the ancestor
//!   masks, the accept walk, the rollback indices);
//! * 5 device tests pass on the GB10, including a tree row measuring
//!   bit-identical (`0e0`) to the row it would have been in a linear batch
//!   holding only its own path;
//! * three real bugs were found and fixed by that testing.
//!
//! What was NOT established: **the same-width numerics gate was never run.**
//! An end-to-end run of `INK_TREE=b` against an `INK_WIDTH=n` reference on the
//! same forced sequence — the comparison that holds the arithmetic fixed and
//! varies only the tree — has not happened. It was skipped deliberately, on
//! the grounds that verifying a path nobody is shipping is not worth contended
//! GPU time, and that is a defensible trade only while it is written down.
//!
//! So the honest summary is **"showed no evidence of a bug"**, which is not
//! the same as "verified", and the distinction is the whole reason this
//! section exists. An unverified path that LOOKS tested — 40 green tests, a
//! `0e0`, and a confident module header — is worse than one that looks
//! untested, because the next reader trusts it.
//!
//! **Anyone reviving this must run that gate FIRST**, before trusting a line
//! of it. `INK_FORCE_IDS` is the fixed reference it needs; the invocation is
//! recorded in `inkling_forward.rs`'s header next to the note explaining why a
//! one-row baseline cannot serve.
//!
//! Everything in this module is pure `std` arithmetic over indices and tokens:
//! no backend, no tensors, no CUDA. That is deliberate. The parts of tree
//! speculation that are easy to get subtly wrong — which head reads which row,
//! which row is a fact about which path, and which of the verifier's argmaxes
//! may be believed — are all index algebra, and index algebra can be tested on
//! a laptop. The tensor work is then a transcription of a plan this module
//! produced and a mask this module built.
//!
//! # The heads are CHAINED, not parallel
//!
//! Read `inkling_forward.rs` around the MTP draft loop before trusting anything
//! here. Inkling's heads are DeepSeek-style: head `0` reads the main trunk's
//! hidden state, head `d` reads head `d - 1`'s output, and each head also
//! consumes the EMBEDDING of a token one position further ahead than the head
//! below it. Concretely, with `n = ids.len()` and `best` the token the trunk
//! just produced (so `best` sits at position `n`):
//!
//! * head `d`'s row at index `p` is fed `(row p of head d-1, embed(tok[p+d+1]))`
//!   and predicts the token at `p + d + 2`;
//! * head `d`'s newest STABLE row is therefore `p = n - d - 1` (its embedding
//!   input is `best`), and it predicts position `n + 1`;
//! * the rows past it are functions of DRAFTS, and head `d`'s speculative row
//!   `i` (`i = 0..d-1`) sits at `pos = n - d + i`, is fed the draft at depth
//!   `i + 1`, and predicts depth `i + 2`.
//!
//! The heads are NOT interchangeable — head `d` is trained for offset `d`.
//! So BREADTH cannot come from recruiting other heads; it comes from taking
//! the top-`b` of one head's own distribution ([`top_b`]).
//!
//! # The algebra that makes a tree cheap-ish
//!
//! Substituting `i = j - 1` (a speculative row is indexed by the DEPTH `j` of
//! the drafted token whose embedding it eats) gives the two facts the whole
//! module is built on:
//!
//! 1. head `d`'s row for a node `v` at depth `j` sits at `pos = n - d + j - 1`
//!    and depends only on `v`'s ancestor path — never on `d` beyond the choice
//!    of weights. Two heads reading the same path prefix read the SAME prefix.
//! 2. that row's `hin` is head `d - 1`'s row for `parent(v)`, at the SAME
//!    position; and when `j == 1` the parent row is head `d - 1`'s newest
//!    STABLE row, index `n - d`, which is branch-independent.
//!
//! Consequence 2 is the load-bearing one. Setting `d = j` in it: the row that
//! draws node `v`'s CHILDREN (head `j`, `pos = n - 1`) reads the row that drew
//! `v` itself (head `j - 1`, `pos = n - 1`). So the residual chain is a plain
//! linked list along the path — one step per node — and the only reason the
//! cost is quadratic at all is that head `d`'s attention at `pos = n - 1` must
//! find positions `n - d .. n - 2` already in its cache. Those are the FILL
//! rows.
//!
//! Counting, with `K` heads available and `n_j` nodes at depth `j`:
//!
//! ```text
//! exact fills:   ops = sum_j  n_j * (K - j)          [chain: K(K-1)/2, today's cost]
//! shared fills:  ops = (K-1)(K-2)/2 + sum_{j<K} n_j
//! ```
//!
//! [`CacheFill::SharedGreedy`] is the second line: fill each head's cache once
//! from the greedy path and reuse it for every branch at that depth. It is an
//! APPROXIMATION of the drafts — and it cannot hurt correctness, because
//! speculative decoding's correctness lives entirely in the accept step
//! ([`accept_tree`]). Which candidates get proposed is free.
//!
//! # Where the tensor side stands
//!
//! The layer is done: `dev_lane::attention_steps_tree` takes a [`TreeAttn`]
//! and handles all three of the things a tree changes at once (visibility,
//! position, and the two convolutions inside the attention);
//! `dev_lane::short_conv_tree` is the gathered convolution; `commit_rows` and
//! `conv_history_rows` are the rollback, which for a tree is a GATHER out of
//! the accepted path rather than a truncation of a prefix. A tree row measures
//! bit-identical to the row it would have been in a linear batch holding only
//! its own path (`a_tree_row_is_its_own_branch_local`, GB10, 0e0).
//!
//! # THE VERDICT: breadth does not pay on this architecture
//!
//! Measured 2026-08-26, the two-machine pipe (head 0:21 on zgx-0d6e, tail
//! 21:42 on zgx-16ec, so the drafter sees the REAL final hidden state),
//! `INK_MTP_TOPK=4`, teacher-forced over each corpus, full vocab. "vs STACK"
//! is head 0's hit rate against the main stack's OWN argmax at that position,
//! which is what a verifier accepts; Wilson 95% intervals:
//!
//! ```text
//! corpus                rows   hit@1            hit@2            breadth gain
//! technical prose       3584   0.2969 ±.015     0.3895 ±.016     +0.0926
//! Rust source           4096   0.6956 ±.014     0.7915 ±.012     +0.0959
//! counting (1 2 3 ...)  3328   0.9892 ±.004     0.9961 ±.002     +0.0069
//! ```
//!
//! Acceptance is enormously corpus-dependent — 30% to 99% — and that is not
//! the finding. **The finding is the last column.** Breadth buys about +9.5
//! points wherever there is room, and nothing at all where there is not, and
//! it always costs ONE EXTRA ROW.
//!
//! Priced against `inkling_forward`'s own published `c(2) = 1.492` (a two-row
//! pass) and a measured three-row width probe at 1.60x, with the identity this
//! project already uses, `speedup = (1 + p) / cost`:
//!
//! ```text
//! corpus            chain k=1        tree b=2        tree b=2
//!                   (2 rows, 1.492)  (3 rows, 1.60)  (3 rows, measured 2.11)
//! technical prose   0.869x           0.868x          0.659x
//! Rust source       1.136x           1.120x          0.849x
//! counting          1.333x           1.248x          0.946x
//! ```
//!
//! **The tree never beats the chain — on any corpus, at the most generous
//! tree cost that is even arguable.** +9.5 points of acceptance and one extra
//! row are the same size, so the trade is a wash where it is available and a
//! loss where it is not. Where acceptance is low, +9.5 points is not enough to
//! cover a row; where acceptance is high there is nothing left to win (+0.7
//! points) and the row is still paid. There is no corpus in between where the
//! two separate.
//!
//! Nor do wider trees escape it: hit@3 and hit@4 add +5.0 and +4.2 points on
//! prose, +3.4 and +2.3 on code, for a row each. The marginal acceptance
//! falls while the marginal cost does not.
//!
//! So: this module is a NEGATIVE RESULT, and the thing it rules out is worth
//! the ruling. What the same table says positively is that the CHAIN pays
//! handsomely on predictable text (1.14x on code, 1.33x on counting) and loses
//! on prose — which is an argument for gating the existing `INK_SPEC` lane on
//! the draft head's confidence, not for widening it.
//!
//! CAVEATS, because these numbers are a product of two runs and the framing
//! rule applies: acceptance is from the FULL stack over the pipe; the cost
//! ratios are `c(2) = 1.492` from this project's earlier pipe measurement and
//! a 1.60x three-row width probe measured on a HALF stack (layers 0:21,
//! ctx512) that was not idle-gated. The verdict does not rest on the fragile
//! half: at the TOP of the hit@2 confidence interval on prose (0.4056), and
//! assuming a three-row tree somehow cost no more than a two-row pass
//! (1.492x), b = 2 still returns 0.942x.
//!
//! # WHAT WOULD REVIVE THE TREE, stated as a threshold
//!
//! The verdict above is a MARGINAL identity — breadth buys about +0.095
//! tokens per pass and costs one row — and that form is robust. The NUMBERS
//! in it are not equally robust, and one of them is about to move.
//!
//! What a row costs today: `c(2) = 1.492` and a measured three-row probe at
//! 1.60x, so the marginal row is about **0.11 of a step**. Breadth buys
//! **0.095**. Those are the two quantities, and the gap between them is 15%
//! — which is why breadth is a wash rather than an obvious loss.
//!
//! So the verdict has an explicit revival threshold:
//!
//! ```text
//! breadth pays when   marginal row cost < 0.095 of a step
//! today               marginal row cost ~ 0.11
//! ```
//!
//! It is closer than "dead" suggests, and the step it is denominated in has
//! just been re-measured as **78% host enqueue** (73.6 ms of 93.9, both nodes
//! serialised). If CUDA graphs collapse that region — the unmerged capture
//! measured 35.4 ms of enqueue going to ~112 µs at production layer count —
//! then `c2`, the marginal row, and `d` are all quantities measured against a
//! step that no longer exists, and **every speedup in this module must be
//! re-derived before it is quoted again.**
//!
//! Which way it moves is not obvious and should not be guessed. If the
//! marginal row is mostly host enqueue it becomes nearly free and breadth
//! revives; if enqueue is per-KERNEL rather than per-ROW then a widened pass
//! never paid much enqueue in the first place, the marginal row is already
//! device-bound, and the verdict stands unchanged. Today's numbers cannot
//! distinguish those cases — `c(2)` and the 78% were measured in different
//! runs at different configs, and multiplying them would be exactly the
//! framing error this file has already made once.
//!
//! **The verdict is therefore: dead at today's row cost, with a stated price
//! at which it is not.** File it that way rather than as "dead", because the
//! second form invites re-deriving it badly from stale constants and the
//! first says precisely which constant to re-measure.
//!
//! # THE GATE, and why it is a smaller lever than it looks
//!
//! The verdict table says the chain wins on code and counting and LOSES on
//! prose, so an always-on lane ships the mean of a win and a loss. Gating
//! should recover the difference. Measured (prose, 3584 rows, same pipe run),
//! head 0's own top-1 probability as the signal:
//!
//! ```text
//! T      kept    p_kept   p_dropped   gated
//! 0.00   1.000   0.2969   -           0.843x   (= always)
//! 0.20   0.361   0.5379   0.1607      0.975x
//! 0.40   0.202   0.6841   0.1987      0.993x   <- best
//! 0.60   0.136   0.7684   0.2225      0.992x
//! 0.90   0.068   0.8689   0.2551      0.980x
//! ```
//!
//! **The signal works and the gate still loses.** p_kept 0.684 against
//! p_dropped 0.199 is enormous separation — the drafter emphatically knows
//! when it is right. But never-speculate is 1.000x and pays NO DRAFT AT ALL,
//! while a gate reading the DRAFT's confidence must draft first and therefore
//! pays `d` on every pass including the ones it declines. The criterion is
//!
//! ```text
//! f * (p_kept - (c2 - 1)) > d        [c2 - 1 = 0.492, d = 0.047]
//! ```
//!
//! and its maximum over the whole sweep is 0.0388 — **short by 0.008, which
//! is 17% of the required margin.** The gate converts a 0.843x loss into a
//! 0.993x near-miss and never crosses.
//!
//! What blocks it is the always-paid draft, not the acceptance. So the design
//! that could work is a gate on a signal the pass ALREADY HAS — the main
//! stack's own top-1 probability at the current row, available BEFORE the
//! draft is made, which moves the draft inside the gate and turns `d` into
//! `f * d`. Then the criterion is just `p_kept > d + c2 - 1 = 0.539` with no
//! constant to overcome.
//!
//! Substituting the measured head-confidence buckets into that model bounds
//! what such a gate could pay on prose:
//!
//! ```text
//! T=0.40  f=0.202  p=0.684 -> 1.026x
//! T=0.60  f=0.136  p=0.768 -> 1.029x   <- ceiling
//! T=0.70  f=0.112  p=0.800 -> 1.028x
//! ```
//!
//! **About 1.03x on prose.** That is a real win where the always-on lane
//! loses 16%, and it is not a large one. Note what it is NOT: a proof. It
//! substitutes head-0's confidence for the free signal, on the assumption
//! that no cheaper signal predicts head 0's correctness BETTER than head 0's
//! own probability does. That is plausible and unproven, and it is the one
//! measurement still owed — `INK_MTP_TOPK` now emits the `stackp1` sweep that
//! settles it.
//!
//! On a mixed workload the gate's value is not that it beats the best arm
//! anywhere; it is that it removes the worst arm's downside. Equal token
//! counts across the three corpora, aggregating as time (`3 / sum(1/s)`):
//! never 1.000x, always 1.046x, gate about 1.13x — and essentially all of the
//! difference is prose.
//!
//! SIZE THIS HONESTLY BEFORE BUILDING IT. ~1.03x on prose and ~1.13x mixed is
//! a real lever and a small one. The same decode step is 78% host enqueue, and
//! an unmerged graph capture measured that region collapsing 316x. Speculation
//! attacks acceptance economics and graphs attack host enqueue, so the two
//! compose rather than compete — but they are not the same size, and the
//! gate's threshold is denominated in a step graphs would redefine. Measure,
//! do not build, until that lands.
//!
//! # What a SINGLE BOX can and cannot measure
//!
//! `INK_TREE` runs on one box, and measured there (GB10, layers 0:21, ctx512,
//! `INK_GEN=48`, 2026-08-26) a `b = 2` tree accepted **0 of 48** passes. That
//! is not a defect in the tree. A half stack unembeds a MID-STACK hidden state,
//! while an MTP head is trained to predict the FULL model's next token from the
//! FULL model's hidden state — so the drafter and the verifier are two
//! different models and agreement is accidental. The 22% depth-1 acceptance in
//! `inkling_forward`'s header was measured on the TAIL (layers 20:42), which
//! owns the real final hidden state.
//!
//! The full 42 layers do not fit one GB10 (21 layers is already 84 GiB of
//! weights on a 121 GiB box), which is why speculation is a two-machine
//! arrangement in the first place. So: a single box prices the tree's COST and
//! gates its arithmetic, and cannot measure its ACCEPTANCE at all. Both halves
//! are needed to say whether breadth pays, and they come from different runs.
//!
//! What is NOT here is the DECODE LOOP. `inkling_forward.rs` still builds a
//! chain, and three things stand between it and a tree:
//!
//! 1. `INK_SPEC` is gated on `pipe_spec.is_some()` — today's speculation is a
//!    two-machine arrangement where only the tail can draft and only the head
//!    can embed. A single-box tree run needs its own entry, not a widened one.
//! 2. the drafting side needs a device top-`b`, where `draft_pick` is a device
//!    unembed-and-argmax. [`top_b`] is the host twin and the specification.
//! 3. the block's own `attn_sconv` and `mlp_sconv` are the caller's — they
//!    take the same [`TreeAttn::taps`] through `short_conv_tree_steps` and the
//!    same rollback through `conv_history_rows`, but the call sites are in the
//!    decode loop, not in the layer.

use std::fmt;

/// The root of a [`TreeSpec`] — the token the trunk already confirmed — has no
/// parent.
pub const NO_PARENT: usize = usize::MAX;

/// The widest depth-1 tree the expert-union measurement supports.
///
/// Not a hard limit — nothing enforces it — but the number to argue with
/// before exceeding it. Past `b = 3` the marginal candidate costs more distinct
/// experts than an extra SEQUENTIAL token does, and by `b = 6` the pass costs
/// more than the 2.0 tokens a depth-1 tree can possibly accept. See
/// [`TreeSpec::breadth`] for the table.
pub const MAX_MEASURED_BREADTH: usize = 3;

/// What can be wrong with a topology or a plan request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeError {
    /// A path was given whose depth exceeds the number of trained heads.
    TooDeep { depth: usize, heads: usize },
    /// More than 64 nodes, asked for as a 64-bit mask.
    TooWide { nodes: usize },
    /// An empty path list, or a topology with no draft nodes at all.
    Empty,
    /// Two children of one node were assigned the same token, so a verifier
    /// prediction would not name a unique branch.
    DuplicateSibling { parent: usize, token: usize },
    /// `node_tokens` did not have one entry per node.
    BadArity { got: usize, want: usize },
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::TooDeep { depth, heads } => write!(
                f,
                "a tree of depth {depth} needs {depth} trained MTP heads, the checkpoint has \
                 {heads}"
            ),
            TreeError::TooWide { nodes } => {
                write!(f, "{nodes} nodes does not fit a 64-bit ancestor mask")
            }
            TreeError::Empty => write!(f, "a speculation tree needs at least one draft node"),
            TreeError::DuplicateSibling { parent, token } => write!(
                f,
                "node {parent} has two children holding token {token}; a verifier argmax would \
                 not name one branch"
            ),
            TreeError::BadArity { got, want } => {
                write!(f, "{got} tokens for a {want}-node tree")
            }
        }
    }
}

impl std::error::Error for TreeError {}

/// One node of the speculation tree.
///
/// Node `0` is the ROOT and is not a draft: it holds the token the trunk
/// already produced, which is the token the whole verify batch's row 0 is fed.
/// Its `rank` is meaningless and its `parent` is [`NO_PARENT`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub parent: usize,
    /// Which slot of the parent's top-`b` this node took. 0 is the argmax.
    pub rank: usize,
    pub depth: usize,
    pub children: Vec<usize>,
}

/// A fixed sparse token tree, in Medusa's "choices" form.
///
/// Nodes are ordered by `(depth, path)` lexicographically, which guarantees
/// `parent < child` — so a verify batch laid out in node order is already
/// causal in the ordinary sense, and the ancestor mask is strictly lower
/// triangular.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeSpec {
    nodes: Vec<TreeNode>,
    depth_counts: Vec<usize>,
    width: usize,
}

impl TreeSpec {
    /// Build from a list of rank-paths, e.g. `[[0], [1], [0,0], [0,1], [1,0]]`.
    ///
    /// The list is PREFIX-CLOSED automatically: naming `[0, 1]` also creates
    /// `[0]`. That is not politeness, it is required — a node whose parent is
    /// absent has nothing to attend to and nothing to draft it.
    pub fn from_paths(paths: &[Vec<usize>]) -> Result<TreeSpec, TreeError> {
        let mut all: Vec<Vec<usize>> = Vec::new();
        for p in paths {
            if p.is_empty() {
                continue;
            }
            for take in 1..=p.len() {
                all.push(p[..take].to_vec());
            }
        }
        if all.is_empty() {
            return Err(TreeError::Empty);
        }
        all.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        all.dedup();

        // `all` is depth-ordered, so a parent is always placed before its
        // children and this single pass can resolve parents by lookup.
        let mut nodes = vec![TreeNode {
            parent: NO_PARENT,
            rank: 0,
            depth: 0,
            children: Vec::new(),
        }];
        let mut index: Vec<(Vec<usize>, usize)> = Vec::with_capacity(all.len());
        let mut width = 0usize;
        for path in all {
            let depth = path.len();
            let rank = path[depth - 1];
            width = width.max(rank + 1);
            let parent = if depth == 1 {
                0
            } else {
                let head = &path[..depth - 1];
                index
                    .iter()
                    .find(|(p, _)| p.as_slice() == head)
                    .map(|(_, id)| *id)
                    .expect("prefix-closed and depth-ordered, so the parent is already placed")
            };
            let id = nodes.len();
            nodes.push(TreeNode {
                parent,
                rank,
                depth,
                children: Vec::new(),
            });
            nodes[parent].children.push(id);
            index.push((path, id));
        }

        let max_depth = nodes.iter().map(|nd| nd.depth).max().unwrap_or(0);
        let mut depth_counts = vec![0usize; max_depth + 1];
        for nd in &nodes {
            depth_counts[nd.depth] += 1;
        }
        Ok(TreeSpec {
            nodes,
            depth_counts,
            width,
        })
    }

    /// The degenerate tree: one chain of `depth` argmax drafts.
    ///
    /// This is exactly what `inkling_forward.rs` drafts today, and the plans
    /// and masks it produces are the regression anchor for everything else.
    pub fn chain(depth: usize) -> Result<TreeSpec, TreeError> {
        if depth == 0 {
            return Err(TreeError::Empty);
        }
        TreeSpec::from_paths(&[vec![0; depth]])
    }

    /// The measured shape: `b` candidates for the NEXT token and nothing else.
    ///
    /// # Why this is narrow, and must stay narrow
    ///
    /// Measured on this checkpoint (layers 21:31, ctx 512, 8 warm passes), the
    /// DISTINCT experts a verify pass gathers grow very differently for
    /// same-position candidates than for sequential ones:
    ///
    /// ```text
    /// width           2       3       4       6
    /// linear      11.55   15.25   19.80   26.55
    /// tree         8.55   11.00   16.70   21.05
    /// marginal, per ADDED token:
    /// linear      +3.95   +4.05   +3.65   +1.80
    /// tree        +0.55   +2.45   +5.70   +2.18
    /// ```
    ///
    /// The tree's whole advantage is the FIRST added candidate: +0.55 experts
    /// against +3.95, a 7.2x saving. The second still wins (+2.45); the THIRD
    /// costs +5.70, which is worse than adding a sequential token. The
    /// cheapness lives in the top of head 0's distribution and is exhausted
    /// almost immediately as you walk down it.
    ///
    /// There is also a ceiling, and it is the one-line argument against ever
    /// widening this "as an optimisation". Breadth at depth 1 accepts AT MOST
    /// one drafted token plus the free bonus, so the expected tokens per pass
    /// cannot exceed 2.0 no matter how good the drafts are. Against that
    /// ceiling the expert-union measurement's own cost model (MoE = 72.7% of
    /// step bytes) prices `b = 6` at 2.286x a plain step — ABOVE it, so width 6
    /// cannot pay even with perfect acceptance — with `b = 4` needing 95% of
    /// the ceiling to break even and `b = 2` needing 57%.
    ///
    /// **Those three percentages are DERIVED, and they are not re-derivable
    /// from the table above.** The table counts distinct EXPERTS; turning that
    /// into a step cost needs the other 27.3% of the bytes as well, and how
    /// that part scales with ROW COUNT is not in the table. Assume it is
    /// row-invariant and `b = 2` prices at 1.050x (break-even at 5% of the
    /// ceiling); take the file header's measured linear `c(2) = 1.492` as
    /// implying it scales nearly linearly and the same `b = 2` prices at about
    /// 1.39x (break-even near 39%). A factor of eight in the conclusion sits
    /// between two readings of one unstated assumption, which is exactly why a
    /// number carries its framing or is not evidence.
    ///
    /// So do not quote 57% as a measurement. It has since been replaced by
    /// one, and the answer is that breadth does not pay: the marginal
    /// candidate buys about +9.5 points of acceptance and costs a whole extra
    /// row, everywhere, on every corpus. See the module header's VERDICT
    /// section for the table and the arithmetic. Neither my 57% nor the 1.050x
    /// that the row-invariant reading gives was close; both were cost models
    /// standing in for a measurement.
    ///
    /// The corollary is the reason breadth is worth building at all: breadth
    /// and depth price two DIFFERENT axes. Breadth raises the probability of
    /// reaching the ceiling; only depth raises the ceiling. So breadth exists
    /// to PROTECT depth — to hedge the chained heads' inability to reconsider
    /// a token they have already conditioned on — not to replace it.
    ///
    /// See [`MAX_MEASURED_BREADTH`].
    pub fn breadth(b: usize) -> Result<TreeSpec, TreeError> {
        TreeSpec::balanced(&[b])
    }

    /// `breadths[j]` children for every node at depth `j`.
    ///
    /// `balanced(&[4, 2, 2, 1])` is the shape people reach for first: wide at
    /// the top, narrowing. Note from the cost formula in the module docs that
    /// this is also the WORST place to put breadth on chained heads — a node
    /// at depth `j` costs `K - j` fill rows, so depth-1 breadth is the most
    /// expensive breadth there is. It is also the breadth most likely to pay.
    pub fn balanced(breadths: &[usize]) -> Result<TreeSpec, TreeError> {
        let mut paths: Vec<Vec<usize>> = vec![Vec::new()];
        let mut out: Vec<Vec<usize>> = Vec::new();
        for &b in breadths {
            let mut next = Vec::new();
            for p in &paths {
                for r in 0..b {
                    let mut q = p.clone();
                    q.push(r);
                    next.push(q);
                }
            }
            out.extend(next.iter().cloned());
            paths = next;
            if paths.is_empty() {
                break;
            }
        }
        TreeSpec::from_paths(&out)
    }

    /// Total nodes INCLUDING the root, which is the number of verify rows.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    /// Draft nodes only — the number of tokens actually being gambled on.
    pub fn drafts(&self) -> usize {
        self.nodes.len() - 1
    }

    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    pub fn node(&self, id: usize) -> &TreeNode {
        &self.nodes[id]
    }

    pub fn max_depth(&self) -> usize {
        self.depth_counts.len() - 1
    }

    /// The `b` that [`top_b`] must supply: one more than the largest rank used.
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn depth_counts(&self) -> &[usize] {
        &self.depth_counts
    }

    /// Node ids at a given depth, ascending.
    pub fn at_depth(&self, depth: usize) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&i| self.nodes[i].depth == depth)
            .collect()
    }

    /// Ancestors of `id`, root FIRST, `id` LAST.
    pub fn path_to(&self, id: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.nodes[id].depth + 1);
        let mut cur = id;
        loop {
            out.push(cur);
            let p = self.nodes[cur].parent;
            if p == NO_PARENT {
                break;
            }
            cur = p;
        }
        out.reverse();
        out
    }

    /// The rank-path, which is what [`TreeSpec::from_paths`] was given.
    pub fn ranks_of(&self, id: usize) -> Vec<usize> {
        self.path_to(id)
            .into_iter()
            .skip(1)
            .map(|i| self.nodes[i].rank)
            .collect()
    }

    /// The greedy path: follow rank 0 as far as the tree goes.
    ///
    /// [`CacheFill::SharedGreedy`] fills every head's cache from this path.
    pub fn greedy_path(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut cur = 0usize;
        loop {
            match self.nodes[cur]
                .children
                .iter()
                .copied()
                .min_by_key(|&c| self.nodes[c].rank)
            {
                Some(c) => {
                    out.push(c);
                    cur = c;
                }
                None => break,
            }
        }
        out
    }

    /// `Err` if this tree is deeper than the checkpoint has trained heads.
    pub fn check_heads(&self, heads: usize) -> Result<(), TreeError> {
        if self.max_depth() > heads {
            return Err(TreeError::TooDeep {
                depth: self.max_depth(),
                heads,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// top-b
// ---------------------------------------------------------------------------

/// One candidate continuation: a token and the logit it was picked on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cand {
    pub token: usize,
    pub logit: f32,
}

/// The `b` largest logits, descending, ties broken by LOWER token id.
///
/// Deterministic top-`b`, not temperature sampling. Sampling `b` draws from a
/// draft distribution draws DUPLICATES and can miss a high-probability
/// candidate entirely; the head's distribution over the vocabulary already IS
/// the ensemble of alternatives, so reading it directly is both cheaper and
/// strictly more informative. Temperature belongs in the TARGET's sampling,
/// where it changes the output distribution — putting it in the drafter only
/// changes which gambles are placed.
///
/// `NaN` logits are SKIPPED outright rather than compared. Ordering against a
/// `NaN` is not a partial order — every comparison is false — so letting one
/// into the running set silently corrupts the ones after it, which is how a
/// single bad logit turns into a draft set that quietly stops containing the
/// argmax. `-inf` is left in: a masked token is a real, orderable choice, and
/// it can only be picked when there is nothing better.
pub fn top_b(logits: &[f32], b: usize) -> Vec<Cand> {
    let b = b.min(logits.len());
    let mut out: Vec<Cand> = Vec::with_capacity(b);
    if b == 0 {
        return out;
    }
    for (token, &logit) in logits.iter().enumerate() {
        if logit.is_nan() {
            continue;
        }
        // Cheap rejection first: the common case is a logit that beats nothing.
        if out.len() == b && !(logit > out[b - 1].logit) {
            continue;
        }
        let mut at = out.len();
        while at > 0 && logit > out[at - 1].logit {
            at -= 1;
        }
        if at < b {
            if out.len() == b {
                out.pop();
            }
            out.insert(at, Cand { token, logit });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// the draft plan
// ---------------------------------------------------------------------------

/// How much of each head's speculative K/V is recomputed per branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheFill {
    /// Every branch gets its own fill rows. Bit-identical to today's drafting
    /// on a chain; `sum_j n_j * (K - j)` steps.
    Exact,
    /// Fill each head once along the greedy path and let every branch at that
    /// depth attend to it. `(K-1)(K-2)/2 + sum_{j<K} n_j` steps.
    ///
    /// The fill rows are the only thing shared — each node's `hin` is still
    /// its own ancestry, so a branch is never confused with another branch.
    /// The rows it attends to are simply a slightly wrong context, which makes
    /// the DRAFTS slightly worse and the output not at all different: the
    /// accept step is what carries correctness.
    SharedGreedy,
}

/// Where a step's `hin` comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hin {
    /// Head `head`'s newest STABLE row, index `row`. Branch-independent, so it
    /// is read straight out of `mtp_stage_dev[head]` (or, for `head == 0`, out
    /// of the trunk's entry states).
    Stable { head: usize, row: usize },
    /// The output of an earlier op in this same plan.
    Op(usize),
}

/// One `mtp_block_step_dev` call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepOp {
    /// Which MTP head's weights.
    pub head: usize,
    /// The position this row occupies in that head's K/V. Always
    /// `seq - head + depth(embed_node) - 1`.
    pub pos: usize,
    pub hin: Hin,
    /// Whose token is embedded and fed alongside `hin`.
    pub embed_node: usize,
    /// The working cache lane this row is appended to.
    pub lane: usize,
    /// `Some(v)` when this row's output, unembedded, is the distribution that
    /// proposes `v`'s CHILDREN. Exactly the rows with `pos == seq - 1`.
    pub drafts_children_of: Option<usize>,
}

/// A cache lane operation, interleaved with the steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanOp {
    /// Clone head `head`'s committed cache into a fresh working lane.
    OpenLane {
        lane: usize,
        head: usize,
    },
    /// Clone an existing working lane — a branch point.
    ForkLane {
        lane: usize,
        from: usize,
    },
    /// The lane will not be written again.
    CloseLane {
        lane: usize,
    },
    Step(StepOp),
}

/// The whole drafting schedule for one tree at one decode step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftPlan {
    pub ops: Vec<PlanOp>,
    pub lanes: usize,
    pub seq: usize,
    pub heads: usize,
    pub fill: CacheFill,
    /// `draft_from[v]` is the op index whose output proposes `v`'s children,
    /// or `None` for `v == 0` — the ROOT's children come off head 0's newest
    /// stable row, which no step in this plan computes.
    pub draft_from: Vec<Option<usize>>,
}

impl DraftPlan {
    /// How many `mtp_block_step_dev` calls this plan makes.
    pub fn steps(&self) -> usize {
        self.ops
            .iter()
            .filter(|o| matches!(o, PlanOp::Step(_)))
            .count()
    }

    /// How many cache clones. `OpenLane` and `ForkLane` both copy.
    pub fn clones(&self) -> usize {
        self.ops
            .iter()
            .filter(|o| matches!(o, PlanOp::OpenLane { .. } | PlanOp::ForkLane { .. }))
            .count()
    }

    pub fn step_ops(&self) -> impl Iterator<Item = &StepOp> {
        self.ops.iter().filter_map(|o| match o {
            PlanOp::Step(s) => Some(s),
            _ => None,
        })
    }

    /// Where the ROOT's children come from: head 0's newest stable row.
    pub fn root_source(&self) -> Hin {
        Hin::Stable {
            head: 0,
            row: self.seq - 1,
        }
    }
}

/// Head `d`'s row for node `v` is NEEDED only if something reads it: either
/// `v` sits at depth `d` and has children to draft, or head `d + 1` will chain
/// off it for one of `v`'s descendants. Both cases collapse to one rule.
///
/// A LEAF at depth `d` is therefore not stepped on head `d` at all — there is
/// nothing past it to propose, and computing the row anyway is a full head
/// step spent on a distribution nobody reads.
pub fn needed_nodes(tree: &TreeSpec, d: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for t in tree.at_depth(d) {
        if tree.node(t).children.is_empty() {
            continue;
        }
        for v in tree.path_to(t).into_iter().skip(1) {
            if !out.contains(&v) {
                out.push(v);
            }
        }
    }
    out.sort_unstable();
    out
}

/// The one path every head's shared fill rows are taken from: the leftmost
/// (highest-ranked at every branch) path that reaches the tree's maximum
/// depth. Root excluded, so `fill_path()[j - 1]` is the node at depth `j`.
///
/// It has to be ONE path across all heads, not a per-head choice: head `d`'s
/// fill row at depth `j` chains off head `d - 1`'s row for the node at depth
/// `j - 1`, so the two heads must agree on which node that is.
pub fn fill_path(tree: &TreeSpec) -> Vec<usize> {
    let deepest = tree
        .at_depth(tree.max_depth())
        .into_iter()
        .min()
        .expect("a tree always has a node at its own maximum depth");
    tree.path_to(deepest).into_iter().skip(1).collect()
}

/// How many `mtp_block_step_dev` calls a plan will make, without building it.
///
/// Computed from the node sets rather than from the plan, so a test comparing
/// it against [`DraftPlan::steps`] is comparing two independent derivations.
pub fn predicted_steps(tree: &TreeSpec, heads: usize, fill: CacheFill) -> usize {
    (1..heads)
        .map(|d| match fill {
            CacheFill::Exact => needed_nodes(tree, d).len(),
            CacheFill::SharedGreedy => {
                let targets = tree
                    .at_depth(d)
                    .into_iter()
                    .filter(|&t| !tree.node(t).children.is_empty())
                    .count();
                if targets == 0 { 0 } else { (d - 1) + targets }
            }
        })
        .sum()
}

/// Build the drafting schedule.
///
/// `heads` is `mtp_config.num_nextn_predict_layers` (or whatever `INK_MTP`
/// narrowed it to) and `seq` is `ids.len()` at the moment the trunk produced
/// its token — the same `n` the module docs use.
///
/// A tree of maximum depth 1 produces an EMPTY plan, which is not a
/// degenerate case to be tolerated but the cheapest useful tree there is: the
/// root's `b` children all come off head 0's newest stable row, which the
/// decode step has already computed. Breadth at depth 1 costs one top-`b`
/// instead of one argmax and not a single extra head step.
pub fn draft_plan(
    tree: &TreeSpec,
    heads: usize,
    seq: usize,
    fill: CacheFill,
) -> Result<DraftPlan, TreeError> {
    tree.check_heads(heads)?;
    if tree.drafts() == 0 {
        return Err(TreeError::Empty);
    }
    assert!(
        seq > heads,
        "head {} trails the sequence by {} rows and a {seq}-token prompt has none to spare",
        heads.saturating_sub(1),
        heads.saturating_sub(1)
    );

    let mut ops: Vec<PlanOp> = Vec::new();
    let mut lanes = 0usize;
    let mut draft_from: Vec<Option<usize>> = vec![None; tree.len()];
    // `row_of[d][v]` — the op index of head `d`'s row for node `v`, which is
    // the `hin` of head `d + 1`'s row for each of `v`'s children.
    let mut row_of: Vec<Vec<Option<usize>>> = vec![vec![None; tree.len()]; heads.max(1)];
    let path = fill_path(tree);

    for d in 1..heads {
        match fill {
            CacheFill::Exact => {
                let wanted = needed_nodes(tree, d);
                if wanted.is_empty() {
                    continue;
                }
                for v in tree.node(0).children.iter().copied() {
                    if !wanted.contains(&v) {
                        continue;
                    }
                    let lane = lanes;
                    lanes += 1;
                    ops.push(PlanOp::OpenLane { lane, head: d });
                    emit_exact(
                        tree,
                        d,
                        seq,
                        v,
                        lane,
                        &wanted,
                        &mut ops,
                        &mut lanes,
                        &mut row_of,
                        &mut draft_from,
                    );
                    ops.push(PlanOp::CloseLane { lane });
                }
            }
            CacheFill::SharedGreedy => {
                let targets: Vec<usize> = tree
                    .at_depth(d)
                    .into_iter()
                    .filter(|&t| !tree.node(t).children.is_empty())
                    .collect();
                if targets.is_empty() {
                    continue;
                }
                emit_shared(
                    tree,
                    d,
                    seq,
                    &targets,
                    &path,
                    &mut ops,
                    &mut lanes,
                    &mut row_of,
                    &mut draft_from,
                );
            }
        }
    }

    Ok(DraftPlan {
        ops,
        lanes,
        seq,
        heads,
        fill,
        draft_from,
    })
}

/// `hin` for head `d`'s row for node `v`: head `d-1`'s row for `parent(v)`, at
/// the SAME position — which for a depth-1 node is head `d-1`'s newest STABLE
/// row, index `seq - d`, and is therefore branch-independent.
fn hin_for(tree: &TreeSpec, d: usize, seq: usize, v: usize, row_of: &[Vec<Option<usize>>]) -> Hin {
    if tree.node(v).depth == 1 {
        Hin::Stable {
            head: d - 1,
            row: seq - d,
        }
    } else {
        let p = tree.node(v).parent;
        Hin::Op(
            row_of[d - 1][p]
                .expect("heads are emitted in increasing d, so the parent's row already exists"),
        )
    }
}

/// Depth-first over one subtree of head `d`'s needed nodes.
///
/// `lane` already holds `v`'s ancestors' rows and nothing else. `v`'s row goes
/// in, then each child gets a lane: every child but the LAST forks a copy
/// taken at this instant, and the last inherits `lane` itself. The fork has to
/// happen BEFORE any sibling writes — two siblings append DIFFERENT rows at
/// the SAME position, and `attention_step` asserts `pos >= base + len`, so a
/// lane that already holds one sibling can never take the other.
#[allow(clippy::too_many_arguments)]
fn emit_exact(
    tree: &TreeSpec,
    d: usize,
    seq: usize,
    v: usize,
    lane: usize,
    wanted: &[usize],
    ops: &mut Vec<PlanOp>,
    lanes: &mut usize,
    row_of: &mut [Vec<Option<usize>>],
    draft_from: &mut [Option<usize>],
) {
    let j = tree.node(v).depth;
    let hin = hin_for(tree, d, seq, v, row_of);
    let drafts = if j == d { Some(v) } else { None };
    let idx = ops.len();
    ops.push(PlanOp::Step(StepOp {
        head: d,
        pos: seq - d + j - 1,
        hin,
        embed_node: v,
        lane,
        drafts_children_of: drafts,
    }));
    row_of[d][v] = Some(idx);
    if let Some(v) = drafts {
        draft_from[v] = Some(idx);
    }

    let kids: Vec<usize> = tree
        .node(v)
        .children
        .iter()
        .copied()
        .filter(|c| wanted.contains(c))
        .collect();
    if kids.is_empty() {
        return;
    }
    // Take every fork first, off the lane as it stands right now.
    let mut child_lanes: Vec<usize> = Vec::with_capacity(kids.len());
    for _ in 0..kids.len() - 1 {
        let l = *lanes;
        *lanes += 1;
        ops.push(PlanOp::ForkLane {
            lane: l,
            from: lane,
        });
        child_lanes.push(l);
    }
    child_lanes.push(lane);
    for (c, cl) in kids.iter().copied().zip(child_lanes) {
        emit_exact(tree, d, seq, c, cl, wanted, ops, lanes, row_of, draft_from);
        if cl != lane {
            ops.push(PlanOp::CloseLane { lane: cl });
        }
    }
}

/// Head `d` with ONE set of fill rows, taken from [`fill_path`], reused by
/// every branch at depth `d`.
#[allow(clippy::too_many_arguments)]
fn emit_shared(
    tree: &TreeSpec,
    d: usize,
    seq: usize,
    targets: &[usize],
    path: &[usize],
    ops: &mut Vec<PlanOp>,
    lanes: &mut usize,
    row_of: &mut [Vec<Option<usize>>],
    draft_from: &mut [Option<usize>],
) {
    let base = *lanes;
    *lanes += 1;
    ops.push(PlanOp::OpenLane {
        lane: base,
        head: d,
    });
    for j in 1..d {
        let v = path[j - 1];
        let hin = hin_for(tree, d, seq, v, row_of);
        let idx = ops.len();
        ops.push(PlanOp::Step(StepOp {
            head: d,
            pos: seq - d + j - 1,
            hin,
            embed_node: v,
            lane: base,
            drafts_children_of: None,
        }));
        // A shared fill row is a fact about the fill path only; recording it
        // under that node is exactly what lets head d+1 chain off it there.
        row_of[d][v] = Some(idx);
    }
    // One row per target at pos == seq - 1. They all write the same position,
    // so they cannot share a lane: all but the last fork off `base` before any
    // of them writes.
    for (nth, &v) in targets.iter().enumerate() {
        let lane = if nth + 1 == targets.len() {
            base
        } else {
            let l = *lanes;
            *lanes += 1;
            ops.push(PlanOp::ForkLane {
                lane: l,
                from: base,
            });
            l
        };
        let hin = hin_for(tree, d, seq, v, row_of);
        let idx = ops.len();
        ops.push(PlanOp::Step(StepOp {
            head: d,
            pos: seq - 1,
            hin,
            embed_node: v,
            lane,
            drafts_children_of: Some(v),
        }));
        row_of[d][v] = Some(idx);
        draft_from[v] = Some(idx);
        if lane != base {
            ops.push(PlanOp::CloseLane { lane });
        }
    }
    ops.push(PlanOp::CloseLane { lane: base });
}

// ---------------------------------------------------------------------------
// the verify batch
// ---------------------------------------------------------------------------

/// The rows the target model is asked to score, and the mask they need.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyBatch {
    /// One token per node, node order. `tokens[0]` is the CONFIRMED token, so
    /// row 0's prediction is always kept.
    pub tokens: Vec<usize>,
    /// `seq + depth(v)`. Siblings SHARE a position, which is the whole point.
    pub positions: Vec<usize>,
    /// `mask[i][j]` — may row `i` attend to row `j`? True exactly when `j` is
    /// an ancestor-or-self of `i`.
    pub mask: Vec<Vec<bool>>,
}

/// `mask[i][j] = j is an ancestor-or-self of i`. Strictly lower triangular
/// plus the diagonal, because node order is `(depth, path)`.
pub fn ancestor_mask(tree: &TreeSpec) -> Vec<Vec<bool>> {
    let n = tree.len();
    let mut m = vec![vec![false; n]; n];
    for i in 0..n {
        for a in tree.path_to(i) {
            m[i][a] = true;
        }
    }
    m
}

/// The same thing as bits, for a kernel that wants one word per row.
pub fn ancestor_bitmask(tree: &TreeSpec) -> Result<Vec<u64>, TreeError> {
    let n = tree.len();
    if n > 64 {
        return Err(TreeError::TooWide { nodes: n });
    }
    let mut out = vec![0u64; n];
    for i in 0..n {
        for a in tree.path_to(i) {
            out[i] |= 1u64 << a;
        }
    }
    Ok(out)
}

/// Assemble the batch. `node_tokens[0]` must be the trunk's confirmed token.
pub fn verify_batch(
    tree: &TreeSpec,
    node_tokens: &[usize],
    seq: usize,
) -> Result<VerifyBatch, TreeError> {
    if node_tokens.len() != tree.len() {
        return Err(TreeError::BadArity {
            got: node_tokens.len(),
            want: tree.len(),
        });
    }
    check_siblings(tree, node_tokens)?;
    Ok(VerifyBatch {
        tokens: node_tokens.to_vec(),
        positions: tree.nodes().iter().map(|nd| seq + nd.depth).collect(),
        mask: ancestor_mask(tree),
    })
}

/// Two children of one node holding the same token would make the accept walk
/// ambiguous, and the ambiguity would show up as a quietly falling acceptance
/// rate rather than as an error. [`top_b`] cannot produce it; a hand-built
/// candidate set can.
pub fn check_siblings(tree: &TreeSpec, node_tokens: &[usize]) -> Result<(), TreeError> {
    for (id, nd) in tree.nodes().iter().enumerate() {
        for (a, &x) in nd.children.iter().enumerate() {
            for &y in &nd.children[a + 1..] {
                if node_tokens[x] == node_tokens[y] {
                    return Err(TreeError::DuplicateSibling {
                        parent: id,
                        token: node_tokens[x],
                    });
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// accept
// ---------------------------------------------------------------------------

/// What a verify pass confirmed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeAccept {
    /// The tokens the model actually chose, in order. NEVER empty: row 0 was
    /// fed a confirmed token, so its prediction is a fact regardless.
    pub new_toks: Vec<usize>,
    /// Verify-batch rows whose K, V and convolution memory are facts about the
    /// sequence: the root followed by the accepted path. Ascending, but NOT
    /// contiguous — a tree rollback is a GATHER, not a truncation.
    pub kept_rows: Vec<usize>,
    /// Drafts accepted, i.e. `kept_rows.len() - 1`.
    pub accepted: usize,
}

/// Greedy tree verification.
///
/// `preds[i]` is the target model's argmax for verify row `i`. Walk from the
/// root: whatever the target predicted at the current node is a confirmed
/// token; if some child of that node HOLDS that token, the child's own row was
/// scored in a context the model has now committed to, so its prediction is a
/// fact too and the walk continues. Otherwise the walk stops, and the last
/// prediction is the one bonus token every speculative pass gets for free.
///
/// This reduces EXACTLY to the linear rule
/// `while accepted < drafts.len() && drafts[accepted] == preds[accepted]` when
/// the tree is a chain — see the tests.
pub fn accept_tree(tree: &TreeSpec, node_tokens: &[usize], preds: &[usize]) -> TreeAccept {
    assert_eq!(node_tokens.len(), tree.len(), "one token per verify row");
    assert_eq!(preds.len(), tree.len(), "one prediction per verify row");
    debug_assert!(
        check_siblings(tree, node_tokens).is_ok(),
        "siblings must hold distinct tokens for the walk to be unambiguous"
    );
    let mut kept_rows = vec![0usize];
    let mut new_toks = Vec::new();
    let mut cur = 0usize;
    loop {
        let want = preds[cur];
        new_toks.push(want);
        match tree
            .node(cur)
            .children
            .iter()
            .copied()
            .find(|&c| node_tokens[c] == want)
        {
            Some(c) => {
                kept_rows.push(c);
                cur = c;
            }
            None => break,
        }
    }
    let accepted = kept_rows.len() - 1;
    TreeAccept {
        new_toks,
        kept_rows,
        accepted,
    }
}

/// The linear rule, written out, so the equivalence test has something to
/// compare against that is not `accept_tree` again.
pub fn accept_linear(drafts: &[usize], preds: &[usize]) -> (usize, Vec<usize>) {
    assert_eq!(preds.len(), drafts.len() + 1, "one row per token plus root");
    let mut accepted = 0usize;
    while accepted < drafts.len() && drafts[accepted] == preds[accepted] {
        accepted += 1;
    }
    (accepted, preds[..=accepted].to_vec())
}

// ---------------------------------------------------------------------------
// what the VERIFY pass needs
// ---------------------------------------------------------------------------

/// Everything the device-side verify pass has to be told about a tree batch.
///
/// # Why this is not just a mask
///
/// A tree's rows are not consecutive positions of one sequence, and Inkling's
/// decoder layer depends on that in FOUR places per layer, not one:
///
/// 1. attention's own visibility — the mask, the obvious one;
/// 2. the relative-position bias and the log-scaling `tau`, which read a row's
///    POSITION, and siblings share a position rather than occupying two;
/// 3. the depthwise SHORT CONVOLUTIONS. There are four of them in a widened
///    pass (`k_sconv` and `v_sconv` inside the attention, then `attn_sconv`
///    and `mlp_sconv` around it), every one with `sconv_kernel_size = 4`, and
///    every one reads `all[i ..= i + kernel - 1]` — the three rows physically
///    preceding row `i` in the batch. For a chain those ARE row `i`'s
///    ancestors. For a tree they are whatever the layout happens to put there,
///    which for the very simplest tree (`b = 2` at depth 1) means the second
///    candidate is convolved out of the first one's projections.
///
/// That third item is the one to be careful about, because it does not look
/// like an error: masked attention still runs, the numbers stay finite, and
/// the text stays fluent. It shows up as a verify pass that quietly scores a
/// candidate the model was never asked about.
///
/// [`TreeAttn::taps`] fixes it without a new kernel or a new mask: the taps of
/// row `i` are gathered along `i`'s ANCESTRY instead of along the batch
/// layout, and for a chain they reduce to exactly the contiguous window the
/// existing kernel already reads ([`TreeAttn::is_linear`] says so, and the
/// device path can keep today's arithmetic untouched when it holds).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeAttn {
    pub rows: usize,
    pub kernel: usize,
    /// Row `i`'s depth. Its absolute position is `pos0 + depth[i]`, NOT
    /// `pos0 + i`, and its cache slot is still `i`.
    pub depth: Vec<usize>,
    /// `visible[i][j]` — may batch row `i` attend to batch row `j`? The cached
    /// PREFIX is visible to every row and is not described here.
    pub visible: Vec<Vec<bool>>,
    /// `taps[i][t]` indexes the `kernel - 1 + rows` window the short
    /// convolutions consume: `0 .. kernel-1` is the carried history and
    /// `kernel-1 + j` is batch row `j`.
    pub taps: Vec<Vec<usize>>,
}

impl TreeAttn {
    /// The degenerate descriptor: `rows` consecutive positions of one
    /// sequence. Exactly what the verify pass does today.
    pub fn linear(rows: usize, kernel: usize) -> TreeAttn {
        let depth: Vec<usize> = (0..rows).collect();
        let visible = (0..rows)
            .map(|i| (0..rows).map(|j| j <= i).collect())
            .collect();
        let taps = (0..rows)
            .map(|i| (0..kernel).map(|t| i + t).collect())
            .collect();
        TreeAttn {
            rows,
            kernel,
            depth,
            visible,
            taps,
        }
    }

    /// True when this descriptor asks for nothing the existing contiguous
    /// path does not already do — one chain, consecutive positions, taps in a
    /// row. The device lane may then take its fast path unchanged, which is
    /// what keeps a non-tree run bit-identical.
    pub fn is_linear(&self) -> bool {
        *self == TreeAttn::linear(self.rows, self.kernel)
    }

    /// Absolute positions, for the relative-bias table and the log scaling.
    pub fn positions(&self, pos0: usize) -> Vec<usize> {
        self.depth.iter().map(|&d| pos0 + d).collect()
    }
}

/// Derive the verify-pass descriptor from a topology.
///
/// `kernel` is `sconv_kernel_size`. The tap rule is one line: row `i` at depth
/// `j` wants, for tap `t`, the sequence element `j - (kernel - 1 - t)` ALONG
/// ITS OWN PATH — an ancestor when that is non-negative, and the carried
/// history when it is not. On a chain, `j == i` and the rule collapses to
/// `i + t`, which is what the batched convolution kernel already reads.
pub fn tree_attn(tree: &TreeSpec, kernel: usize) -> TreeAttn {
    assert!(
        kernel >= 2,
        "a short convolution with kernel {kernel} has no history"
    );
    let rows = tree.len();
    let depth: Vec<usize> = tree.nodes().iter().map(|nd| nd.depth).collect();
    let visible = ancestor_mask(tree);
    let mut taps = Vec::with_capacity(rows);
    for i in 0..rows {
        // `path[d]` is row `i`'s ancestor at depth `d`.
        let path = tree.path_to(i);
        let j = depth[i] as isize;
        let mut row = Vec::with_capacity(kernel);
        for t in 0..kernel {
            let off = j - (kernel as isize - 1 - t as isize);
            let idx = if off >= 0 {
                kernel - 1 + path[off as usize]
            } else {
                // The carried history: `all[kernel-1]` is batch row 0, so
                // `all[kernel-1 + off]` is the committed row `off` places
                // before it. `off > -(kernel-1)` always, since `t < kernel`.
                (kernel as isize - 1 + off) as usize
            };
            row.push(idx);
        }
        taps.push(row);
    }
    TreeAttn {
        rows,
        kernel,
        depth,
        visible,
        taps,
    }
}

/// The `kernel - 1` window rows the NEXT position's convolution must carry,
/// after a verify pass kept `kept` of its rows.
///
/// Indices into the same `kernel - 1 + rows` window [`TreeAttn::taps`] uses,
/// so a rollback is a gather out of the window the batch already built —
/// which is what lets the verifier decide late.
///
/// On a chain with `kept = 0..keep` this returns `keep .. keep + kernel - 1`,
/// exactly the slice `AttnCache::commit` takes today. On a tree it is the tail
/// of `history ++ accepted path`, which is the same sentence and a different
/// slice, because the accepted rows are not contiguous.
pub fn conv_next_history(kernel: usize, kept: &[usize]) -> Vec<usize> {
    let hist = kernel - 1;
    let mut seq: Vec<usize> = (0..hist).collect();
    seq.extend(kept.iter().map(|&r| hist + r));
    seq[seq.len() - hist..].to_vec()
}

/// The absolute position of every slot a tree verify pass can attend to.
///
/// `base` is the cache's first absolute position and `len_before` the rows it
/// held before this batch, so slot `s < len_before` sits at `base + s` and
/// batch row `j` sits at `pos0 + depth[j]` — which is NOT `base + len_before +
/// j`, and is the whole reason this helper exists rather than the arithmetic
/// being inlined at the mask.
pub fn slot_positions(attn: &TreeAttn, base: usize, len_before: usize, pos0: usize) -> Vec<usize> {
    let mut out: Vec<usize> = (0..len_before).map(|s| base + s).collect();
    out.extend(attn.positions(pos0));
    out
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ---- topology -------------------------------------------------------

    #[test]
    fn paths_are_prefix_closed_and_parents_precede_children() {
        let t = TreeSpec::from_paths(&[vec![0, 1, 2], vec![1]]).unwrap();
        // [0,1,2] pulled in [0] and [0,1]; [1] is a fourth.
        assert_eq!(t.drafts(), 4);
        for (id, nd) in t.nodes().iter().enumerate().skip(1) {
            assert!(
                nd.parent < id,
                "node {id}'s parent {} is not earlier",
                nd.parent
            );
            assert_eq!(nd.depth, t.node(nd.parent).depth + 1);
        }
        assert_eq!(t.node(0).parent, NO_PARENT);
        assert_eq!(t.max_depth(), 3);
        assert_eq!(t.width(), 3, "rank 2 was used, so top-3 is required");
    }

    #[test]
    fn nodes_are_ordered_by_depth_then_path() {
        let t = TreeSpec::balanced(&[2, 2]).unwrap();
        let depths: Vec<usize> = t.nodes().iter().map(|n| n.depth).collect();
        assert_eq!(depths, vec![0, 1, 1, 2, 2, 2, 2]);
        assert_eq!(t.ranks_of(3), vec![0, 0]);
        assert_eq!(t.ranks_of(6), vec![1, 1]);
        assert_eq!(t.depth_counts(), &[1, 2, 4]);
    }

    #[test]
    fn chain_is_one_path() {
        let t = TreeSpec::chain(4).unwrap();
        assert_eq!(t.drafts(), 4);
        assert_eq!(t.max_depth(), 4);
        assert_eq!(t.width(), 1);
        assert_eq!(t.greedy_path(), vec![1, 2, 3, 4]);
        assert_eq!(t.path_to(4), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn greedy_and_fill_path_follow_rank_zero() {
        let t = TreeSpec::from_paths(&[vec![1, 0], vec![0]]).unwrap();
        // [0] is a leaf; the only depth-2 node hangs off [1].
        assert_eq!(t.greedy_path().len(), 1, "rank 0 at depth 1 is a leaf");
        assert_eq!(fill_path(&t).len(), 2, "the fill path must reach max depth");
    }

    #[test]
    fn too_deep_is_refused() {
        let t = TreeSpec::chain(9).unwrap();
        assert_eq!(
            t.check_heads(8),
            Err(TreeError::TooDeep { depth: 9, heads: 8 })
        );
        assert!(t.check_heads(9).is_ok());
    }

    // ---- top-b ----------------------------------------------------------

    fn brute_top_b(logits: &[f32], b: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &c| {
            logits[c]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&c))
        });
        idx.truncate(b);
        idx
    }

    #[test]
    fn top_b_matches_a_full_sort() {
        // A deterministic LCG, so a failure is reproducible.
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        for trial in 0..64 {
            let n = 1 + (trial * 7) % 97;
            let logits: Vec<f32> = (0..n)
                .map(|_| {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect();
            for b in [1usize, 2, 3, 5, 8] {
                let got: Vec<usize> = top_b(&logits, b).into_iter().map(|c| c.token).collect();
                assert_eq!(got, brute_top_b(&logits, b.min(n)), "n={n} b={b}");
            }
        }
    }

    #[test]
    fn top_b_breaks_ties_by_lower_token_and_is_descending() {
        let logits = [1.0f32, 3.0, 3.0, 2.0, 3.0];
        let got = top_b(&logits, 3);
        assert_eq!(
            got.iter().map(|c| c.token).collect::<Vec<_>>(),
            vec![1, 2, 4]
        );
        assert!(got.windows(2).all(|w| w[0].logit >= w[1].logit));
    }

    #[test]
    fn top_b_clamps_and_survives_nan() {
        assert!(top_b(&[1.0, 2.0], 0).is_empty());
        assert_eq!(top_b(&[1.0, 2.0], 9).len(), 2);
        let got = top_b(&[f32::NAN, 1.0, 2.0], 2);
        assert_eq!(
            got.iter().map(|c| c.token).collect::<Vec<_>>(),
            vec![2, 1],
            "a NaN never outranks a finite logit"
        );
    }

    // ---- the plan -------------------------------------------------------

    /// Replay a plan the way the device would, so the invariants
    /// `attention_step` asserts at runtime are asserted here instead.
    ///
    /// Returns, for every step that drafts, the sequence of embedded nodes its
    /// lane held at that moment — which is the branch the row was computed on.
    fn replay(plan: &DraftPlan) -> HashMap<usize, Vec<usize>> {
        let mut lanes: HashMap<usize, (usize, Vec<(usize, usize)>)> = HashMap::new();
        let mut branch_of: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut outputs: Vec<bool> = Vec::new();
        for op in &plan.ops {
            match op {
                PlanOp::OpenLane { lane, head } => {
                    assert!(
                        lanes.insert(*lane, (*head, Vec::new())).is_none(),
                        "lane {lane} opened twice"
                    );
                    outputs.push(false);
                }
                PlanOp::ForkLane { lane, from } => {
                    let src = lanes.get(from).expect("fork of a live lane").clone();
                    assert!(
                        lanes.insert(*lane, src).is_none(),
                        "lane {lane} opened twice"
                    );
                    outputs.push(false);
                }
                PlanOp::CloseLane { lane } => {
                    assert!(lanes.remove(lane).is_some(), "close of a dead lane");
                    outputs.push(false);
                }
                PlanOp::Step(s) => {
                    let idx = outputs.len();
                    outputs.push(true);
                    let (head, rows) = lanes.get_mut(&s.lane).expect("step on a live lane");
                    assert_eq!(*head, s.head, "lane {} belongs to head {head}", s.lane);
                    // `attention_step`: pos >= base + len, and a head's
                    // committed cache already covers 0..seq-head.
                    let want = (plan.seq - s.head) + rows.len();
                    assert_eq!(
                        s.pos,
                        want,
                        "head {} lane {} would append at {} with {} rows cached",
                        s.head,
                        s.lane,
                        s.pos,
                        rows.len()
                    );
                    match s.hin {
                        Hin::Stable { head, row } => {
                            assert_eq!(head + 1, s.head, "hin comes from the head below");
                            assert!(row < plan.seq - head, "a stable row is not speculative");
                        }
                        Hin::Op(o) => {
                            assert!(o < idx, "hin reads an op that has not run");
                            assert!(outputs[o], "hin reads a lane op, not a row");
                        }
                    }
                    rows.push((s.pos, s.embed_node));
                    if let Some(v) = s.drafts_children_of {
                        assert_eq!(s.pos, plan.seq - 1, "a drafting row sits at seq-1");
                        branch_of.insert(v, rows.iter().map(|&(_, n)| n).collect());
                    }
                }
            }
        }
        assert!(lanes.is_empty(), "every lane must be closed");
        branch_of
    }

    /// The op list today's `inkling_forward.rs` draft loop makes, written out
    /// independently: head `d`, rows `i = 0..d-1`, at `pos = seq - d + i`,
    /// eating `drafts[i]` and chaining off head `d-1`'s row at the same
    /// position (or its newest stable row, `seq - d`, when `i == 0`).
    fn todays_loop(k: usize, seq: usize) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for d in 1..k {
            for i in 0..d {
                out.push((d, seq - d + i, i + 1));
            }
        }
        out
    }

    #[test]
    fn a_chain_plan_is_todays_draft_loop() {
        let seq = 512;
        for k in 2..=8 {
            let t = TreeSpec::chain(k).unwrap();
            for fill in [CacheFill::Exact, CacheFill::SharedGreedy] {
                let plan = draft_plan(&t, k, seq, fill).unwrap();
                let got: Vec<(usize, usize, usize)> = plan
                    .step_ops()
                    .map(|s| (s.head, s.pos, t.node(s.embed_node).depth))
                    .collect();
                assert_eq!(got, todays_loop(k, seq), "k={k} fill={fill:?}");
                assert_eq!(plan.steps(), k * (k - 1) / 2);
                replay(&plan);
            }
        }
    }

    #[test]
    fn a_chain_needs_no_forks() {
        let plan = draft_plan(&TreeSpec::chain(6).unwrap(), 6, 256, CacheFill::Exact).unwrap();
        assert_eq!(plan.clones(), 5, "one lane per head, no branch to fork");
        assert!(
            !plan
                .ops
                .iter()
                .any(|o| matches!(o, PlanOp::ForkLane { .. }))
        );
    }

    #[test]
    fn depth_one_breadth_is_free() {
        // The shape arm C says to build: b candidates at t+1 and nothing else.
        for b in 2..=6 {
            let t = TreeSpec::balanced(&[b]).unwrap();
            assert_eq!(t.len(), b + 1);
            assert_eq!(t.width(), b);
            let plan = draft_plan(&t, 8, 256, CacheFill::Exact).unwrap();
            assert_eq!(
                plan.steps(),
                0,
                "every candidate comes off head 0's stable row, which the step already has"
            );
            assert_eq!(plan.clones(), 0);
            assert_eq!(plan.draft_from, vec![None; b + 1]);
            assert_eq!(plan.root_source(), Hin::Stable { head: 0, row: 255 });
        }
    }

    #[test]
    fn predicted_steps_agrees_with_the_plan() {
        let trees = [
            TreeSpec::chain(4).unwrap(),
            TreeSpec::balanced(&[2]).unwrap(),
            TreeSpec::balanced(&[2, 2]).unwrap(),
            TreeSpec::balanced(&[4, 2, 2]).unwrap(),
            TreeSpec::from_paths(&[vec![0, 0, 0], vec![1], vec![0, 1]]).unwrap(),
        ];
        for t in &trees {
            for fill in [CacheFill::Exact, CacheFill::SharedGreedy] {
                let plan = draft_plan(t, 8, 1024, fill).unwrap();
                assert_eq!(
                    plan.steps(),
                    predicted_steps(t, 8, fill),
                    "tree {:?} fill {fill:?}",
                    t.depth_counts()
                );
                replay(&plan);
            }
        }
    }

    #[test]
    fn shared_fills_are_never_more_expensive_than_exact() {
        for t in [
            TreeSpec::balanced(&[2, 2, 2]).unwrap(),
            TreeSpec::balanced(&[4, 2, 2, 1]).unwrap(),
            TreeSpec::chain(8).unwrap(),
        ] {
            let e = predicted_steps(&t, 8, CacheFill::Exact);
            let s = predicted_steps(&t, 8, CacheFill::SharedGreedy);
            assert!(s <= e, "shared {s} > exact {e}");
        }
    }

    #[test]
    fn every_exact_draft_row_sits_on_its_own_branch() {
        let t = TreeSpec::balanced(&[2, 2, 2]).unwrap();
        let plan = draft_plan(&t, 8, 300, CacheFill::Exact).unwrap();
        let branches = replay(&plan);
        for v in 0..t.len() {
            if t.node(v).children.is_empty() || v == 0 {
                continue;
            }
            let want: Vec<usize> = t.path_to(v).into_iter().skip(1).collect();
            assert_eq!(
                branches.get(&v),
                Some(&want),
                "node {v}'s drafting row must have eaten exactly its own ancestry"
            );
        }
    }

    #[test]
    fn shared_draft_rows_keep_their_own_ancestry_in_hin() {
        // Only the FILL rows are shared; the last embed is still the node's
        // own token and the hin chain is still its own.
        let t = TreeSpec::balanced(&[2, 2, 2]).unwrap();
        let plan = draft_plan(&t, 8, 300, CacheFill::SharedGreedy).unwrap();
        let branches = replay(&plan);
        let path = fill_path(&t);
        for v in 0..t.len() {
            if t.node(v).children.is_empty() || v == 0 {
                continue;
            }
            let j = t.node(v).depth;
            let mut want: Vec<usize> = path[..j - 1].to_vec();
            want.push(v);
            assert_eq!(branches.get(&v), Some(&want), "node {v}");
        }
        // ...and the hin of each drafting row is the drafting row of its parent.
        for s in plan.step_ops() {
            if let Some(v) = s.drafts_children_of {
                if t.node(v).depth >= 2 {
                    let p = t.node(v).parent;
                    assert_eq!(s.hin, Hin::Op(plan.draft_from[p].unwrap()), "node {v}");
                }
            }
        }
    }

    #[test]
    fn leaves_are_not_stepped() {
        // [0] is a leaf at depth 1; [1,0] goes deeper. Head 1 should step only
        // the node that has children.
        let t = TreeSpec::from_paths(&[vec![0], vec![1, 0]]).unwrap();
        let plan = draft_plan(&t, 4, 64, CacheFill::Exact).unwrap();
        let stepped: Vec<usize> = plan.step_ops().map(|s| s.embed_node).collect();
        assert!(!stepped.contains(&1), "node 1 is a leaf and drafts nothing");
        assert!(stepped.contains(&2));
    }

    // ---- masks ----------------------------------------------------------

    #[test]
    fn ancestor_mask_is_lower_triangular_with_depth_plus_one_set() {
        let t = TreeSpec::balanced(&[3, 2]).unwrap();
        let m = ancestor_mask(&t);
        for i in 0..t.len() {
            for j in 0..t.len() {
                if m[i][j] {
                    assert!(j <= i, "row {i} attends forward to {j}");
                }
            }
            assert!(m[i][i], "a row always attends to itself");
            assert_eq!(
                m[i].iter().filter(|&&b| b).count(),
                t.node(i).depth + 1,
                "row {i} sees its ancestors and nothing else"
            );
        }
        let bits = ancestor_bitmask(&t).unwrap();
        for i in 0..t.len() {
            for j in 0..t.len() {
                assert_eq!(bits[i] >> j & 1 == 1, m[i][j]);
            }
        }
    }

    #[test]
    fn a_chain_mask_is_plain_causal() {
        let t = TreeSpec::chain(5).unwrap();
        let m = ancestor_mask(&t);
        for i in 0..t.len() {
            for j in 0..t.len() {
                assert_eq!(m[i][j], j <= i, "chain masks must be exactly causal");
            }
        }
    }

    #[test]
    fn siblings_cannot_see_each_other() {
        let t = TreeSpec::balanced(&[2]).unwrap();
        let m = ancestor_mask(&t);
        assert!(!m[2][1] && !m[1][2], "the b=2 tree's whole point");
        assert!(m[1][0] && m[2][0], "both see the confirmed token");
    }

    #[test]
    fn a_wide_tree_refuses_a_64_bit_mask() {
        let t = TreeSpec::balanced(&[8, 8]).unwrap();
        assert_eq!(t.len(), 73);
        assert_eq!(ancestor_bitmask(&t), Err(TreeError::TooWide { nodes: 73 }));
    }

    #[test]
    fn verify_batch_positions_are_shared_between_siblings() {
        let t = TreeSpec::balanced(&[2, 2]).unwrap();
        let toks = vec![100, 200, 201, 300, 301, 302, 303];
        let vb = verify_batch(&t, &toks, 64).unwrap();
        assert_eq!(vb.positions, vec![64, 65, 65, 66, 66, 66, 66]);
        assert_eq!(vb.tokens, toks);
    }

    #[test]
    fn duplicate_siblings_are_refused() {
        let t = TreeSpec::balanced(&[2]).unwrap();
        assert_eq!(
            verify_batch(&t, &[9, 7, 7], 64),
            Err(TreeError::DuplicateSibling {
                parent: 0,
                token: 7
            })
        );
        assert_eq!(
            verify_batch(&t, &[9, 7], 64),
            Err(TreeError::BadArity { got: 2, want: 3 })
        );
    }

    // ---- the verify-pass descriptor -------------------------------------

    #[test]
    fn a_chain_descriptor_is_exactly_todays_contiguous_window() {
        for k in 1..=6usize {
            let t = TreeSpec::chain(k).unwrap();
            let a = tree_attn(&t, 4);
            assert!(
                a.is_linear(),
                "a chain must ask the device for nothing new (k={k})"
            );
            assert_eq!(a, TreeAttn::linear(k + 1, 4));
            // The batched convolution kernel reads all[i ..= i + kernel - 1].
            for i in 0..a.rows {
                assert_eq!(a.taps[i], vec![i, i + 1, i + 2, i + 3]);
            }
            assert_eq!(a.positions(100), (100..=100 + k).collect::<Vec<_>>());
        }
    }

    #[test]
    fn breadth_two_taps_never_cross_the_branch() {
        let t = TreeSpec::breadth(2).unwrap();
        let a = tree_attn(&t, 4);
        assert!(!a.is_linear(), "a tree is not the contiguous case");
        // Window layout: 0,1,2 are the carried history, 3/4/5 are rows 0/1/2.
        assert_eq!(a.taps[0], vec![0, 1, 2, 3], "the root is a normal row");
        assert_eq!(a.taps[1], vec![1, 2, 3, 4]);
        assert_eq!(
            a.taps[2],
            vec![1, 2, 3, 5],
            "candidate 2 must convolve out of the ROOT, never out of candidate 1"
        );
        for i in 0..a.rows {
            assert!(
                !a.taps[i].contains(&4) || i == 1,
                "row {i} reached into row 1"
            );
        }
        assert_eq!(
            a.positions(64),
            vec![64, 65, 65],
            "siblings share a position"
        );
    }

    #[test]
    fn taps_only_ever_name_ancestors_or_history() {
        let t = TreeSpec::balanced(&[3, 2]).unwrap();
        let kernel = 4;
        let a = tree_attn(&t, kernel);
        for i in 0..a.rows {
            for (t_i, &idx) in a.taps[i].iter().enumerate() {
                assert!(idx < kernel - 1 + a.rows, "tap out of the window");
                if idx >= kernel - 1 {
                    let j = idx - (kernel - 1);
                    assert!(
                        a.visible[i][j],
                        "row {i} tap {t_i} reads row {j}, which it may not even attend to"
                    );
                    assert_eq!(
                        a.depth[j] as isize,
                        a.depth[i] as isize - (kernel as isize - 1 - t_i as isize),
                        "a tap must sit at its own sequence offset"
                    );
                }
            }
            assert_eq!(
                *a.taps[i].last().unwrap(),
                kernel - 1 + i,
                "the last tap is always the row itself"
            );
        }
    }

    #[test]
    fn a_shallow_tree_reaches_into_the_carried_history() {
        // Depth 1 with kernel 4: the root's taps are three history rows, and a
        // candidate's are two history rows plus the root.
        let a = tree_attn(&TreeSpec::breadth(3).unwrap(), 4);
        assert_eq!(a.taps[0][..3], [0, 1, 2]);
        for c in 1..=3 {
            assert_eq!(a.taps[c][..2], [1, 2], "row {c}");
            assert_eq!(a.taps[c][2], 3, "row {c} taps the confirmed token");
        }
    }

    #[test]
    fn next_history_on_a_chain_is_the_slice_commit_takes_today() {
        for kernel in [2usize, 3, 4, 5] {
            for keep in 0..=6usize {
                let kept: Vec<usize> = (0..keep).collect();
                assert_eq!(
                    conv_next_history(kernel, &kept),
                    (keep..keep + kernel - 1).collect::<Vec<_>>(),
                    "kernel={kernel} keep={keep}"
                );
            }
        }
    }

    #[test]
    fn next_history_gathers_a_scattered_accepted_path() {
        // b=2 at depth 1, and the SECOND candidate was accepted: rows 0 and 2.
        let got = conv_next_history(4, &[0, 2]);
        assert_eq!(
            got,
            vec![2, 3, 5],
            "history row 2, then the root, then row 2"
        );
        // ...and it is exactly the taps a row appended after that path wants.
        let a = tree_attn(&TreeSpec::breadth(2).unwrap(), 4);
        assert_eq!(a.taps[2][1..], got[..], "the accepted row's own taps agree");
    }

    #[test]
    fn slot_positions_place_siblings_together() {
        let a = tree_attn(&TreeSpec::breadth(2).unwrap(), 4);
        // A cache based at 0 holding 10 rows, this batch starting at 10.
        let p = slot_positions(&a, 0, 10, 10);
        assert_eq!(p.len(), 13);
        assert_eq!(&p[..10], &(0..10).collect::<Vec<_>>()[..]);
        assert_eq!(&p[10..], &[10, 11, 11]);
    }

    #[test]
    fn the_descriptor_and_the_mask_agree() {
        let t = TreeSpec::balanced(&[2, 2]).unwrap();
        let a = tree_attn(&t, 4);
        assert_eq!(a.visible, ancestor_mask(&t));
        assert_eq!(
            a.depth,
            t.nodes().iter().map(|n| n.depth).collect::<Vec<_>>()
        );
    }

    // ---- accept ---------------------------------------------------------

    #[test]
    fn accept_on_a_chain_is_the_linear_rule() {
        // Exhaustive over every draft/prediction combination in a small
        // alphabet, for chains of every length up to 4.
        for k in 1..=4usize {
            let t = TreeSpec::chain(k).unwrap();
            let combos = 3usize.pow((2 * k + 1) as u32);
            for c in 0..combos {
                let mut x = c;
                let mut toks = vec![0usize; k + 1];
                let mut preds = vec![0usize; k + 1];
                for slot in toks.iter_mut().skip(1) {
                    *slot = x % 3;
                    x /= 3;
                }
                for slot in preds.iter_mut() {
                    *slot = x % 3;
                    x /= 3;
                }
                toks[0] = 99; // the confirmed token, never compared
                let drafts: Vec<usize> = toks[1..].to_vec();
                let (accepted, new_toks) = accept_linear(&drafts, &preds);
                let got = accept_tree(&t, &toks, &preds);
                assert_eq!(got.accepted, accepted, "k={k} c={c}");
                assert_eq!(got.new_toks, new_toks, "k={k} c={c}");
                assert_eq!(got.kept_rows, (0..=accepted).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn accept_walks_the_branch_the_target_chose() {
        // root 10; children 20 (rank 0) and 21 (rank 1); each with children.
        let t = TreeSpec::balanced(&[2, 2]).unwrap();
        let toks = vec![10, 20, 21, 30, 31, 32, 33];
        // The target says t+1 is 21 (the SECOND candidate), then 33.
        let preds = vec![21, 77, 33, 0, 0, 0, 88];
        let got = accept_tree(&t, &toks, &preds);
        assert_eq!(
            got.kept_rows,
            vec![0, 2, 6],
            "a scattered, non-contiguous set"
        );
        assert_eq!(got.new_toks, vec![21, 33, 88]);
        assert_eq!(got.accepted, 2);
    }

    #[test]
    fn a_rejected_first_token_still_confirms_one() {
        let t = TreeSpec::balanced(&[2]).unwrap();
        let toks = vec![10, 20, 21];
        let got = accept_tree(&t, &toks, &[55, 0, 0]);
        assert_eq!(got.accepted, 0);
        assert_eq!(got.new_toks, vec![55], "the bonus token is never lost");
        assert_eq!(got.kept_rows, vec![0]);
    }

    #[test]
    fn breadth_two_accepts_where_breadth_one_would_not() {
        // The concrete reason a b=2 tree beats linear k=1: the target's choice
        // is the SECOND candidate, which a single-draft pass never proposed.
        let wide = TreeSpec::balanced(&[2]).unwrap();
        let narrow = TreeSpec::chain(1).unwrap();
        let preds_wide = vec![21, 0, 41];
        assert_eq!(accept_tree(&wide, &[10, 20, 21], &preds_wide).accepted, 1);
        assert_eq!(accept_tree(&narrow, &[10, 20], &[21, 0]).accepted, 0);
    }

    #[test]
    fn kept_rows_are_a_path_and_stay_ascending() {
        let t = TreeSpec::balanced(&[3, 2, 2]).unwrap();
        let toks: Vec<usize> = (0..t.len()).map(|i| 1000 + i).collect();
        // Follow rank 1 at depth 1, then rank 0 twice.
        let mut preds = vec![0usize; t.len()];
        preds[0] = toks[2];
        let a = t.node(2).children[0];
        preds[2] = toks[a];
        let b = t.node(a).children[0];
        preds[a] = toks[b];
        preds[b] = 4242;
        let got = accept_tree(&t, &toks, &preds);
        assert_eq!(got.kept_rows, vec![0, 2, a, b]);
        assert!(got.kept_rows.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*got.new_toks.last().unwrap(), 4242);
        // Every kept row is an ancestor of the last one — the mask agrees.
        let m = ancestor_mask(&t);
        for &r in &got.kept_rows {
            assert!(m[b][r]);
        }
    }
}
