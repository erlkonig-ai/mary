//! The WITHIN-LAYER split: which slice of each tensor a rank owns, and the
//! collective that puts the pieces back together.
//!
//! `INK_LAYERS` splits the stack BETWEEN layers, so the two nodes run strictly
//! in sequence and each is idle for about half of every token. This module is
//! the other axis: every rank runs EVERY layer, on half of each tensor, at the
//! same time. It divides how many bytes a token has to traverse rather than
//! where those bytes live, and at batch one, where a decode step is very nearly
//! a memory-bandwidth measurement, that is the difference between `M / B` and
//! `M / (2B)`.
//!
//! # The arithmetic that decides it, with its framing rule
//!
//! The standing objection to a within-layer split is that it needs an
//! all-reduce per layer, on the critical path, and a pipeline split needs one
//! activation handoff per token. That trade is settled HERE, on this hardware,
//! and it is not close.
//!
//! **Measured**, on the direct-attach ConnectX pair between the two GB10 boxes
//! (`scripts/interconnect_probe.sh`, GPU-resident buffers, 2 ranks, one
//! `cudaStreamSynchronize` per iteration so the figure includes the sync the
//! decode loop would pay):
//!
//! ```text
//!   ib_write_lat, 8 KB            3.68 us one way   (p99 3.78, stdev 0.01)
//!   ib_write_bw, peak            13.02 GB/s         (saturates by 16 KB)
//!   iperf3 TCP, 8 streams       111 Gbit/s
//!   NCCL all-reduce,  4096 f32   29.56 us    0.55 GB/s
//!   NCCL all-reduce, 65536 f32   76.49 us    3.43 GB/s
//!   NCCL all-reduce, large                  13.78 GB/s
//!   kernel socket path, 8 KB    ~185 us round trip  (ICMP agrees: 171 us min)
//! ```
//!
//! A within-layer split wants two all-reduces per layer — after the attention
//! out-projection and after the MoE down-projection — so **84 per token**, each
//! `[1, 4096]`, 16 KB at f32. Two more are POSSIBLE at the ends (an embedding
//! broadcast and a sharded-unembed reduce) for 86, and this build issues
//! **neither**: both tables are replicated instead, because neither sits under
//! a reduce, so replicating them is correct and costs only bytes — 1.53 GiB of
//! embedding residency and 0.103 GiB a token of head reads at the default
//! `HeadLane::W4a16` + `INK_ANN_HEAD` shortlist, under 1% of a step.
//!
//! ```text
//!   COMMS   84 x 29.56 us                             =    2.48 ms / token
//!           84 x 16 KB                                =    1.31 MB / token
//!           1.31 MB at the measured 13.02 GB/s        =    0.10 ms of WIRE
//! ```
//!
//! For the record on the link's headline rate, because a wrong version of it
//! circulates: 13.02 GB/s is **52% of the 200 Gbit/s the ConnectX pair
//! negotiates**, and the other two instruments agree — NCCL at large sizes
//! 13.78 GB/s (55%) and `iperf3` over 8 TCP streams 111 Gbit/s (56%). There is
//! no ~57% figure and no second-PCIe-path finding in this tree; the real
//! caveat, from `1183fd6`, is that **the box has more than one NIC and the fast
//! one is not the default**, which is why an earlier version of this argument
//! measured the management NIC and concluded the opposite.
//!
//! None of that matters to the decision. The collective is latency-bound, so
//! recovering the missing 45% of the link would move 0.10 ms of a 2.48 ms cost.
//!
//! So the collective is **latency, not bandwidth** — 96% of those 2.48 ms is
//! per-message overhead and only 4% is the wire. Two consequences that a
//! bandwidth framing gets backwards:
//!
//! * **Do not quantise the all-reduce payload.** Sending BF16 instead of f32
//!   halves 1.31 MB to 0.66 MB and saves ~0.05 ms of a 2.48 ms cost. It is not
//!   worth the numerical argument.
//! * **The only lever is FEWER messages**, and RMSNorm is nonlinear, so the two
//!   per layer cannot be fused into one. 86 is the floor for this block shape.
//!
//! Against that, what the split saves. Per token at batch one, every byte of
//! weight the 42-layer model reads (derived from the config, and each line
//! reproduces the corresponding counter in a live run's per-pass report):
//!
//! ```text
//!   term                    GiB/token   check against a run's own counters
//!   attention (BF16)             3.45   "1.73 GiB in 21 attention layers"
//!   dense MLP, layers 0-1        0.75
//!   shared experts (BF16)        3.75   "1.97 GiB in 21 shared ... layers"
//!   routed top-6 (NVFP4)         3.37   "bind ALIAS 126 calls 1701 MiB"
//!   unembed (BF16)               1.53   "1.53 GiB of BF16, read whole"
//!   TOTAL                       12.85
//! ```
//!
//! 12.85 GiB at the ~140 GB/s this lane achieves end to end is **98.5 ms of
//! weight streaming per token**, against a measured two-node round trip of
//! **105.2 ms** (`/tmp/pipe-abl` `base.rep1`, spark2 driving spark, layers 0:21
//! and 21:42, `INK_KV=1`, batch one, 3738-token context, warm p50 over 4 steps,
//! 9.505 tok/s). So **~94% of a batch-one step is weight bytes**, the pipeline
//! split pays all of them in sequence, and 2.48 ms of collectives is **2.4% of
//! the step** — the trade wins by roughly forty to one.
//!
//! Said as the comparison the objection asks for: a within-layer split buys
//! ~49 ms a token of halved streaming and spends 2.48 ms to get it. **The
//! answer to "does the per-layer all-reduce eat the bandwidth saving" is no,
//! by a factor of nineteen.**
//!
//! # What the split is projected to be worth, and what bounds it
//!
//! Not 2x, and the reason is the one term that does NOT halve. The same run's
//! stage report, both ends, batch one:
//!
//! ```text
//!                                    head (21 L)   tail (21 L)
//!   layer loop, host bracket            39.7 ms       36.2 ms
//!     of which pool hand-back (DRAIN)   13.8           9.7
//!     of which true enqueue             25.9          26.5     ~1.25 ms/layer
//!   DEVICE still owed at the sync        5.8           7.0
//!   head / unembed (device)               —            5.0
//! ```
//!
//! Under a pipeline split each node issues 21 layers of kernels; under a
//! within-layer split each node issues **42**, at half the width. The kernel
//! COUNT per layer is unchanged, so host enqueue does not halve — it doubles
//! per node, to ~52 ms. Device streaming halves, to ~49 ms. The two overlap
//! (nothing in the layer loop synchronises), so the pass becomes bounded by
//! whichever is larger:
//!
//! ```text
//!   PP2, measured                                        105.2 ms/token
//!   TP2, projected  max(device 49.2 + comms 2.5,
//!                       host enqueue 52.5) + drain ~12    ~65    ms/token   1.62x
//!   TP2 with the pool hand-back amortised                 ~55    ms/token   1.91x
//!   TP2 with enqueue collapsed (graph capture)            ~52    ms/token   2.02x
//! ```
//!
//! One term inside those projections was believed to be hardware and
//! un-engineerable. **It is not there.** This said the two boxes are not equally
//! fast, on the evidence that `stream_packed` — the same kernel over the same
//! 0.431 GiB of codes and scales — reads 218.4 GB/s on spark2-zt against 248 on
//! spark-zt. **Neither reading reproduces.** Re-run 2026-08-27 with
//! `scripts/gb10-lock.sh` held on both boxes and both verified idle, at that
//! arm's own framing (min of four warm launches, per launch, head shape):
//!
//! ```text
//!   spark  (zgx-0d6e)   240.7, 241.2, 241.3 GB/s   median 241.2
//!   spark2 (zgx-16ec)   240.8 .. 244.2 (n=7)       median 242.9
//! ```
//!
//! The two boxes are **1.1% apart on this kernel**, not 12%. A pipeline split
//! ADDS the two halves so it pays the average; a within-layer split runs them at
//! once and pays the SLOWER box — the structure of the argument is right and the
//! penalty is nearly gone:
//!
//! ```text
//!   PP2 bandwidth term   (1/241.2 + 1/242.9)   =  0.00826 per D/2
//!   TP2 bandwidth term    2 x (1/241.2)        =  0.00829 per D/2   1.993x
//! ```
//!
//! So ~0.35% of the ideal doubling is lost to the boxes disagreeing, not ~6%.
//! The TP2 projection loses a headwind it never actually had; the reason it
//! still does not reach 2.00x is the host enqueue path below, which is the
//! honest binding constraint. NOTE THE SCOPE: this refutes the box asymmetry
//! **for this kernel**, which is the only evidence the claim ever rested on.
//! Whether the two boxes differ end to end is a separate measurement and was not
//! made. The full reconciliation of this ceiling is in `w4a16gemm`, above
//! `w4a16_linear_wide`.
//!
//! **MEASURED, 2026-08-27**, and the projection held at its low end. Same
//! binary bytes on both boxes, `INK_FORCE_IDS`-pinned 3732-token context,
//! `INK_GEN=64 INK_KV=1`, no speculation, arms INTERLEAVED (TP, PP, TP, PP,
//! TP, PP), both boxes locked with `scripts/gb10-lock.sh` and verified idle, a
//! memory settle and >=60 s between reps, free-running clocks. Per DECODE
//! STEP, warm only (the first two of 64 discarded), from each run's own
//! `WARM steps only` line:
//!
//! ```text
//!   arm  rep   WARM ms/step   p50     min
//!   PP2   A         86.6      86.5   85.4
//!   PP2   B         86.3      86.2   84.6
//!   PP2   C         86.0      86.1   84.4
//!   TP2   A        239.6     204.5   56.5
//!   TP2   B         68.7      58.5   56.2
//!   TP2   C         57.0      57.0   55.8
//!
//!   last interleaved pair, C against C     86.0 -> 57.0    1.51x
//!   min of warm launches, either arm       84.4 -> 55.8    1.51x
//! ```
//!
//! Read the TP2 column honestly: **its step is not yet as reproducible as the
//! pipeline's.** The pipeline lands 86.6/86.3/86.0 -- a 0.7% spread -- and the
//! split lands 239.6/68.7/57.0 while its MINIMUM sits at 55.8-56.5 in all
//! three. The clean step is the same every time; what varies is how often the
//! box stalls, and it falls monotonically as the machine settles between runs.
//! That is an allocator-under-pressure story, not a split story, and the
//! residency section below prices it.
//!
//! and the breakdown says it is the projected mechanism and not a coincidence:
//! `layer loop 50.2 ms` of HOST bracket against `DEVICE still owed 5.4 ms` at
//! the sync. **The pass is host-enqueue bound after this change**, exactly as
//! the projection above said it would be -- so the next millisecond comes from
//! the enqueue path (graph capture) and not from the wire.
//!
//! The collective term, measured through the real path rather than budgeted:
//! `scripts/tp-probe.sh` at 84 collectives of `[1, 4096]` f32, median of 7 warm
//! reps, **2.377 ms a token** against the 2.483 ms this header budgets -- 28.30
//! us apiece pipelined, of which 0.810 ms is host. That is **4.2% of a 57.0 ms
//! TP2 step**, and the trade stands as costed.
//!
//! The 1.9x end of the range was NOT reached and the reason is named in the
//! residency section below: it is not the interconnect. That is worth saying twice: after this change the binding
//! constraint moves from LPDDR to the CPU issuing kernels, and every
//! millisecond taken out of the 1.25 ms/layer enqueue cost is then worth double
//! what it is worth today.
//!
//! # Why the routed experts are split WITHIN an expert and not BY expert
//!
//! Two ways to halve a MoE layer, and they differ at batch one:
//!
//! * **Intra-expert** (this module): every rank owns half of every expert's
//!   intermediate axis -- `w13`'s rows and `w2`'s columns. Both ranks run the
//!   router (it is replicated and deterministic), both compute every pick at
//!   half width, and the one all-reduce the layer already needs sums the two
//!   partials. **No input communication at all** -- the residual is already
//!   replicated -- and the routed term halves EXACTLY.
//! * **Expert-parallel**: rank `r` owns experts `r * 128 .. (r+1) * 128` and a
//!   token's six picks fall wherever they fall.
//!
//! Expert-parallel does **not** halve the routed term at batch one, and the
//! amount by which it misses is exactly computable. With top-6 split
//! binomially over two ranks the step waits on the slower rank, and
//! `E[max(k, 6-k)] = 3.9375`, not 3:
//!
//! ```text
//!   k on a rank    3        4        5        6
//!   P(max = k)     0.3125   0.46875  0.1875   0.03125     E = 3.9375
//! ```
//!
//! So the routed term would shrink by 6 / 3.9375 = **1.52x** where this gives a
//! flat 2x. The gap is `(3.9375 - 3) * 13.5 MiB * 39 layers` = **0.48
//! GiB/token, about 3.5 ms**, which is ~5% of a projected TP2 step.
//!
//! ## The layout objection, which was WRONG about this tree
//!
//! This section used to argue the other way, and the argument was: every
//! expert slab is a byte offset into one registered mapping, `w13` is two
//! disjoint ranges and `w2`'s K-dim half is a stride, so intra-expert
//! sharding means **rewriting the checkpoint** -- 72 GiB a node -- to buy 3.5
//! ms. Three things in that are false here.
//!
//! 1. **The rewrite already happens.** [`super::pile::PileSource::copy_share`]
//!    copies this node's whole share out of the mapping into one anonymous
//!    arena at startup, and the offset table is built against the ARENA. The
//!    cut is a change of source index inside a memcpy that already runs.
//! 2. **A permutation inside that memcpy is already known to be free.** The
//!    `INK_SWZ` MMA-fragment swizzle re-indexes 256-byte blocks in the same
//!    pass and was measured at 14.6/14.7/14.8 s against 15.0/15.1/15.9 s NOT
//!    permuting, on 66.50 GiB with 20 threads -- the permuting arm is faster
//!    by the spread. The pass is bound by the memory system, not the address
//!    arithmetic. A cut is strictly cheaper still: it moves HALF the bytes.
//! 3. **`w13` is not two ranges.** `mlp.experts.w13_weight` is converted with
//!    `Interleave(dim=1)` and left fused, so its output rows alternate
//!    `g0, u0, g1, u1, ...` -- which `super::fp4gemm::gate_up_silu` reads
//!    directly. A rank's share of the INTERMEDIATE axis is therefore one
//!    CONTIGUOUS run of `2 * inter / world` rows, gate and up already paired.
//!    See [`Tp::w13_interleaved_rows`]. (`w2` is still a column cut, and that
//!    is still a gather -- but a gather into a buffer the copy was going to
//!    write anyway costs nothing extra.)
//!
//! So the arena SHRINKS with the cut, the copy gets faster, and the routed
//! term halves flat instead of by 1.52x. The remaining honest cost of
//! intra-expert is that both ranks run all six picks, so the LAUNCH count per
//! layer does not fall -- which is the same host-enqueue term the projection
//! above already names as the binding constraint, and expert-parallel does not
//! escape it either (it launches `E[max] = 3.94` picks, not 3).
//!
//! ## What expert-parallel would additionally have cost, which is not layout
//!
//! [`super::devplan`]'s hoist. Six of [`super::moegroup::RowPlan`]'s seven
//! fields are a function of `(n, top_k)` ALONE, uploaded once for the whole
//! run, and that is what deleted an 8.1 ms/step readback (`INK_ROUTE_STALE=1`
//! against base, 72.2 -> 64.1 ms/step). Expert-parallel makes the number of
//! blocks a rank runs DATA-DEPENDENT -- between 0 and 6 -- so `blk_slot`,
//! `tok_rows` and `tok_cnt` change per layer per pass and have to come back
//! from the device. Restoring an 8.1 ms readback to buy 3.5 ms is a loss
//! before any of the layout question is reached.
//!
//! Note also that the imbalance is a batch-one artifact: at `INK_SLOTS=32` the
//! active-expert set is ~113 of 256 and the two halves differ by a few percent,
//! so expert-parallel is much closer to 2x on a wide pass. The lane this
//! module exists for is batch one.
//!
//! # Where the reduces are, and the three places that need none
//!
//! Per layer, exactly two, and both are unavoidable because an RMSNorm sits
//! between them:
//!
//! 1. after `wo` (attention out-projection) — `burn.rs`'s `linear_bf16(out, &w.wo)`
//! 2. after the MoE down-projection and the shared-expert add, and **before**
//!    the MLP short convolution, because that convolution is followed by the
//!    residual add and both ranks must carry the same residual.
//!
//! Three things need no collective at all, and each is a real saving:
//!
//! * **Attention needs none until `wo`.** GQA 32/8 splits 16 query heads and 4
//!   KV heads to a rank. A head's scores, softmax and `p @ v` touch only that
//!   head's K and V, so nothing crosses until the out-projection reduces over
//!   `heads * head_dim`. The KV cache halves with it — 4 KV heads a rank —
//!   which is the term that grows with context and with slot count.
//! * **The unembedding is column-parallel over the VOCABULARY.** Rank `r`
//!   holds rows `r * 100512 .. (r+1) * 100512` and computes its half of the
//!   logits. Greedy decoding then reduces **8 bytes** — one `(value, index)`
//!   pair — instead of all-gathering 201024 floats. That halves a 1.53 GiB
//!   read (0.43 GiB at `HeadLane::W4a16`) for the cost of one more 26 us
//!   collective. Sampling needs a two-float log-sum-exp reduce on top; a
//!   `top_k` for MTP needs `k` pairs. All are latency-sized.
//! * **The shared experts split by INSTANCE.** There are exactly two and there
//!   are exactly two ranks, so rank `r` owns shared expert `r`. Perfectly
//!   balanced, no duplicated bytes, and the sum folds into the MoE reduce that
//!   the layer was already paying for.
//!
//! # What stays replicated, and why that is not a leak
//!
//! The residual stream, both RMSNorm weights, both short convolutions, the
//! router projection and its bias, and the embedding norm. Together these are
//! ~2.1 MB a layer of router plus a few kilobytes of norms and convolutions —
//! under 1% of a layer's bytes — and replicating them is what makes the design
//! work: with an identical residual on both ranks the router reaches an
//! identical decision without a collective, and the norms need no reduce
//! because each rank already holds the whole vector.
//!
//! The **embedding table** is the one deliberate exception. It is 2.40 GiB of
//! residency to read one 8 KB row per token, so replicating it would cost each
//! box 2.4 GiB of page cache for no bandwidth. Rank 0 keeps it and broadcasts
//! the embedded row — one extra collective per token, 85 rather than 84.
//!
//! # What this competes with, which is not what it looks like
//!
//! The standing argument for a within-layer split is "the head is blocked on
//! the tail for half the loop". That is true and measured, but the specific
//! **50.9% blocked** figure that gets quoted is the head's accounting on the
//! `1 x 64` **single-cohort** arm — and `INK_COHORTS=2` already takes that arm
//! to **16.6% blocked**, measured, at 1.45-1.49x aggregate throughput, without
//! a collective or a reshard. So on THROUGHPUT the idle half is largely
//! claimed already, and a within-layer split is not competing for it: the two
//! do not even compose, because after this change there is no idle half for a
//! second cohort to fill.
//!
//! What a within-layer split buys that nothing else here does:
//!
//! * **Single-stream latency.** `INK_COHORTS` interleaves independent
//!   sequences; it moves aggregate tok/s and leaves the time-to-next-token of
//!   one stream exactly where it was. Halving the bytes on the critical path
//!   is the only lever that moves that number, and for one user waiting on one
//!   answer, latency IS throughput.
//! * **The KV and activation working set**, halved with the heads — the term
//!   that put `MemAvailable` at 1.38 GiB and swapped 5.50 GiB of the head at 96
//!   slots.
//!
//! # Is the split's output the SAME output? No, and it cannot be
//!
//! Gated 2026-08-27 against the pipeline arm on the same binary, same pinned
//! 3732-token context, `INK_GEN=64 INK_KV=1`, no speculation. The verdict, in
//! order of what each measurement rules out:
//!
//! * **Both configurations are bit-REPRODUCIBLE.** Two independent PP2
//!   processes agreed on 65/65 decoded tokens, on the whole top-5 table
//!   (identical sha256), and on every printed prefill layer-RMS. Three
//!   independent TP2 processes agreed with each other on all of the same. So
//!   the runtime is not the noise source here, and any difference BETWEEN the
//!   two arms is real and systematic. (Note against the claim that circulates
//!   in `inkling_forward`'s header: with `INK_FORCE_IDS` pinning the sequence,
//!   the cached decode lane reproduced exactly, twice, on both arms.)
//! * **The two TP2 ranks agree bit for bit with each other**, which is what an
//!   all-reduce guarantees and what makes "both ranks decode the same token
//!   without a collective" sound.
//! * **The two ARMS do not agree**: 10 of 65 argmax positions differ, and 60
//!   of 65 positions differ somewhere in their top FIVE.
//!
//! That last line is expected and is not evidence of a wrong shard, because
//! the split RE-ASSOCIATES every reduced sum: `wo` accumulates over 16 heads
//! on each rank instead of 32 in one accumulator, and the MoE down projection
//! over `k = 1024` twice instead of `k = 2048` once. f32 addition is not
//! associative, so bit-identity is not available at any level of correctness.
//!
//! What DOES separate "re-associated" from "wrong" is where the divergence
//! starts. Prefill residual RMS, layer by layer, PP2 against TP2:
//!
//! ```text
//!   layer  0  dense MLP, sharded + reduced      1.5674 vs 1.5674   rel 0.0e+00
//!   layer  1  dense MLP, sharded + reduced      2.1222 vs 2.1222   rel 0.0e+00
//!   layer  2  BF16 MoE + shared, both sharded   2.6913 vs 2.6913   rel 0.0e+00
//!   layer  3  first NVFP4 MoE layer             3.8883 vs 3.8882   rel 2.6e-05
//!   layer 16                                  224.85 vs 225.02     rel 7.2e-04
//!   layer 32                                 4598.97 vs 4606.06    rel 1.5e-03
//!   layer 41                                10041.02 vs 9905.73    rel 1.3e-02
//! ```
//!
//! The first three layers agree to the printed precision -- five significant
//! figures -- and those three exercise the sharded attention, the sharded
//! dense MLP, the sharded shared experts, the sharded BF16 routed experts and
//! four all-reduces. A DOUBLED MLP half would put layer 2 out by about 100%,
//! not by less than 1e-4. Divergence starts at 2.6e-05 at the first NVFP4
//! layer -- the size f32 re-association gives over a 2048-term sum -- and
//! grows to 1.3e-02 by layer 41 as the residual norm itself grows 6400-fold.
//!
//! Both continuations are the same document continued the same way: 55 of the
//! 65 tokens are identical, in the same slots, and the ten that differ are
//! near-ties inside numeric fields. Neither degenerates. That is the bar
//! `inkling_forward`'s header sets -- capability, not identity -- and it is
//! met; what is NOT available, at any correctness, is bit-identity.
//!
//! # What is left REPLICATED in the arena, which is the binding residency term
//!
//! Measured 2026-08-27 at `INK_LAYERS=0:42 INK_TP=r:2`, from the run's own
//! admission line: **87.07 GiB of weights** a node, against 84-85 GiB for a
//! pipeline half-stack. The routed experts are cut in the startup copy, so
//! they are ~74 GiB of that instead of ~147; what keeps the total from
//! falling further is the set of tensors cut at BIND time rather than at COPY
//! time, which the arena therefore still holds whole:
//!
//! ```text
//!   shared experts, 40 layers          7.85 GiB whole   3.93 if cut in the copy
//!   attention projections, 42 layers   1.85            0.93
//!   dense MLP, 2 layers                0.79            0.40
//!   embedding table (replicated)       1.53            --   (one row a token)
//!   unembedding (replicated)           1.53            0.77 if cut by vocab
//!                                      ----            ----
//!                                     13.55            6.03      about 5.2 GiB
//! ```
//!
//! None of it is a BANDWIDTH term -- every one of those is already cut before
//! it is bound, so the device reads half either way. It is residency only, and
//! residency is what decides whether a step is 57 ms or 600. Measured on the
//! 119.63 GiB box: with 87.07 GiB of arena plus 5.21 GiB of device pool the
//! admission gate passes with 1.95 GiB of headroom, and a decode step then
//! costs 56.5 ms when the allocator is not fighting for pages and 200-670 ms
//! when it is -- the SAME binary, the same tokens, twice in one hour.
//!
//! The first three rows of that table are now cut IN THE COPY:
//! [`super::tpshard::dense_cut`] gives `copy_share` the binders' own rule per
//! tensor, the arena is laid out at the cut sizes, and the binders read each
//! leaf as a whole weight of the local width (`Weights::precut` is what tells
//! them to). The bytes that reach `bind_bf16` are the same bytes in the same
//! order -- pinned per tensor in `tpshard`'s tests -- so this is a copy-order
//! change, not an arithmetic one. The unembedding stays replicated: it is not
//! cut at bind time either (no reduce follows it), and cutting it by vocab
//! would be the `argmax_across` design, which is a different change.
//!
//! # Residency, which this does NOT improve over the pipeline split
//!
//! Worth stating because the module doc of `inkling_forward` has claimed
//! otherwise: a pipeline split ALREADY halves per-node weight residency, and
//! expert-parallel lands in the same place (~79 GiB a node against a 119.6 GiB
//! box). What the within-layer split halves that the pipeline split does not is
//! the **activation and KV** working set — half the heads, half the experts —
//! which is the term that ran the host out of memory at 96 slots.

use std::ops::Range;

/// Which rank this process is and how many there are.
///
/// Parsed from `INK_TP=rank:world`. `world == 1` is the degenerate identity
/// case and every method below returns the unsharded answer, so the sharded
/// code path can be exercised on one box without a second process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tp {
    rank: usize,
    world: usize,
}

impl Default for Tp {
    fn default() -> Self {
        Self { rank: 0, world: 1 }
    }
}

/// A tensor axis that does not divide by the world size.
///
/// Returned rather than panicked because the useful message names the axis and
/// the two numbers, and a panic three frames inside a binder does not.
#[derive(Debug, PartialEq, Eq)]
pub struct Indivisible {
    pub axis: &'static str,
    pub extent: usize,
    pub world: usize,
}

impl std::fmt::Display for Indivisible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} is {} and does not divide by a world of {}; a within-layer split \
             cannot cut this axis evenly",
            self.axis, self.extent, self.world
        )
    }
}

impl std::error::Error for Indivisible {}

impl Tp {
    /// `INK_TP=rank:world`, or the identity when it is unset.
    ///
    /// Refuses the contradictions rather than producing a run that is silently
    /// wrong: a rank past the world, a world of zero. `world == 1` is allowed
    /// and means "no split", which is what makes a single-process correctness
    /// gate possible.
    pub fn from_env() -> anyhow::Result<Self> {
        let Ok(spec) = std::env::var("INK_TP") else {
            return Ok(Self::default());
        };
        let (r, w) = spec
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("INK_TP wants RANK:WORLD, got {spec:?}"))?;
        let rank: usize = r
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("INK_TP rank {r:?} is not a number"))?;
        let world: usize = w
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("INK_TP world {w:?} is not a number"))?;
        Self::new(rank, world)
    }

    pub fn new(rank: usize, world: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(world >= 1, "INK_TP world must be at least 1, got {world}");
        anyhow::ensure!(
            rank < world,
            "INK_TP rank {rank} is past a world of {world}; ranks are 0..world"
        );
        Ok(Self { rank, world })
    }

    pub fn rank(self) -> usize {
        self.rank
    }

    pub fn world(self) -> usize {
        self.world
    }

    /// Whether anything is actually split. Every caller that would otherwise
    /// branch on `world == 1` should ask this instead, so the identity case
    /// reads as a fact about the configuration and not as an optimisation.
    pub fn is_split(self) -> bool {
        self.world > 1
    }

    /// This rank's half-open share of `extent`, which must divide.
    ///
    /// Every axis this module cuts divides evenly on the 42-layer release (32
    /// heads, 8 KV heads, 256 experts, 2 shared experts, 201024 vocabulary
    /// rows, 16384 dense intermediate), so an uneven cut is a configuration
    /// error rather than a case to handle. The 66-layer sibling has
    /// `swa_num_key_value_heads = 16`, which also divides.
    pub fn shard(self, axis: &'static str, extent: usize) -> Result<Range<usize>, Indivisible> {
        if extent % self.world != 0 {
            return Err(Indivisible {
                axis,
                extent,
                world: self.world,
            });
        }
        let per = extent / self.world;
        Ok(self.rank * per..(self.rank + 1) * per)
    }

    /// The size of this rank's share, without building the range.
    pub fn share(self, axis: &'static str, extent: usize) -> Result<usize, Indivisible> {
        self.shard(axis, extent).map(|r| r.len())
    }

    // ---- the named cuts -------------------------------------------------
    //
    // Each of these is one call to `shard` with the axis named, and they exist
    // so that a binder reads as "this rank's query heads" rather than as index
    // arithmetic. Naming the axis is also what makes `Indivisible`'s message
    // useful, which is the whole reason it carries a `&'static str`.

    /// Query heads. `wq` is `[heads * head_dim, hidden]` and heads are
    /// contiguous rows, so this is a **span of the pile's mapping** and the
    /// zero-copy bind still aliases.
    pub fn q_heads(self, heads: usize) -> Result<Range<usize>, Indivisible> {
        self.shard("num_attention_heads", heads)
    }

    /// KV heads. `wk`/`wv` are `[kv_heads * head_dim, hidden]`, contiguous rows
    /// again, and the KV **cache** follows the same cut — which is why a rank
    /// holds half the pages rather than all of them.
    pub fn kv_heads(self, kv_heads: usize) -> Result<Range<usize>, Indivisible> {
        self.shard("num_key_value_heads", kv_heads)
    }

    /// The relative-position path, `wr` is `[heads * d_rel, hidden]`. It cuts
    /// on the same head axis as `wq`, so a rank's `wr` rows are
    /// `q_heads * d_rel`.
    pub fn rel_rows(self, heads: usize, d_rel: usize) -> Result<Range<usize>, Indivisible> {
        let h = self.q_heads(heads)?;
        Ok(h.start * d_rel..h.end * d_rel)
    }

    /// The ROUTED experts' intermediate axis -- the intra-expert cut.
    ///
    /// Every expert of every MoE layer is cut the same way and by the same
    /// range, so a rank's `w13` rows and its `w2` columns name the same
    /// intermediate units and the two partials add to the whole. This is the
    /// axis, not the weight: [`Tp::w13_interleaved_rows`] turns it into `w13`'s
    /// row range and the range itself is `w2`'s column range.
    ///
    /// It replaced an expert-parallel cut (`rank * 128 .. (rank+1) * 128` of
    /// `n_routed_experts`) on 2026-08-27; the module header says why, and the
    /// short version is that expert-parallel buys 1.52x rather than 2x at batch
    /// one AND makes `devplan`'s block count data-dependent.
    pub fn routed_inter(self, inter: usize) -> Result<Range<usize>, Indivisible> {
        self.shard("intermediate_size", inter)
    }

    /// The SHARED experts' intermediate axis.
    ///
    /// The same cut as [`Tp::routed_inter`] and deliberately not a cut by
    /// INSTANCE. By instance is exact on this hardware (two experts, two
    /// ranks) and stops being exact on any other, and it would need the shared
    /// gamma column lifted by the rank at every call site -- an off-by-one that
    /// produces finite numbers. Cutting the intermediate axis leaves
    /// `n_shared` at 2 on both ranks, so nothing downstream of the bind moves.
    pub fn shared_inter(self, inter: usize) -> Result<Range<usize>, Indivisible> {
        self.shard("intermediate_size", inter)
    }

    /// Unembed rows. Column-parallel over the vocabulary, reduced to one
    /// `(value, index)` pair for greedy decoding.
    ///
    /// Takes the PADDED `vocab_size` because that is what is bound — the MMA
    /// tiles `n` by 8 and `unpadded_vocab_size` (200058) does not divide. The
    /// padded columns are sliced off after the GEMM exactly as they are today,
    /// and [`Tp::unembed_offset`] is what turns a local argmax back into a
    /// global token id.
    pub fn unembed_rows(self, vocab_size: usize) -> Result<Range<usize>, Indivisible> {
        self.shard("vocab_size", vocab_size)
    }

    /// What to add to a local argmax to get a global token id.
    pub fn unembed_offset(self, vocab_size: usize) -> Result<usize, Indivisible> {
        self.unembed_rows(vocab_size).map(|r| r.start)
    }

    /// The dense MLP's intermediate axis, for layers below `dense_mlp_idx`.
    ///
    /// Column-parallel on `w13` and row-parallel on `w2`, which means the same
    /// single all-reduce the MoE layers pay. `mlp.w13_dn.weight` is stored
    /// INTERLEAVED and `super::load::split_gate_up_bytes` de-interleaves it into
    /// two `[dense_inter, hidden]` buffers before anything binds, so a rank's
    /// share is one plain row range of EACH of those -- not of the fused
    /// weight. Two layers, so the copy is 384 MiB each and is the same
    /// non-aliased host concatenation `INK_FUSE_QKVR` already performs.
    pub fn dense_inter(self, dense_inter: usize) -> Result<Range<usize>, Indivisible> {
        self.shard("dense_intermediate_size", dense_inter)
    }

    /// A rank's `w13` rows when the fused weight is stored INTERLEAVED --
    /// `g0, u0, g1, u1, ...` -- which is how `mlp.experts.w13_weight` is
    /// stored and how [`super::fp4gemm::gate_up_silu`] reads it.
    ///
    /// One CONTIGUOUS range of `2 * inter / world` rows, because interleaving
    /// keeps each intermediate unit's gate row and up row adjacent. This is the
    /// reason the routed cut is a span of the arena rather than a gather.
    ///
    /// Split out and named for the LAYOUT rather than for the weight, because
    /// the two layouts are shape-identical and the wrong one produces finite
    /// numbers: read as HALVED, this range is the first half of the gates and
    /// none of the ups.
    pub fn w13_interleaved_rows(self, inter: usize) -> Result<Range<usize>, Indivisible> {
        let s = self.shard("intermediate_size", inter)?;
        Ok(2 * s.start..2 * s.end)
    }

    /// A rank's `w13` rows when the fused weight is stored HALVED -- every gate
    /// row, then every up row -- which is what `INK_SHARED_W13_HALVED=1`
    /// selects for `shared_w13_weight`. TWO disjoint ranges: the gate block's
    /// share, then the up block's.
    ///
    /// Split out because getting this wrong is silent -- a contiguous
    /// `[0 .. inter]` slice of `[2 * inter, hidden]` is all of the gate and
    /// none of the up, which produces finite numbers and fluent text.
    pub fn w13_halved_rows(
        self,
        inter: usize,
    ) -> Result<(Range<usize>, Range<usize>), Indivisible> {
        let g = self.shard("intermediate_size", inter)?;
        let u = g.start + inter..g.end + inter;
        Ok((g, u))
    }
}

/// How many collectives one token costs at a given layer count, so the number
/// in the header can be recomputed rather than trusted.
///
/// Two per layer -- after `wo` and after the MoE down-projection -- and in this
/// build that is ALL of them, so `ends` is 0. The embedding table and the
/// unembedding are REPLICATED rather than cut: neither sits under a reduce (the
/// final hidden state is already whole when it reaches the head, and nothing
/// reduces the logits), so both ranks compute the same logits and take the same
/// argmax without a collective.
///
/// `ends` is what cutting them WOULD add -- 1 for the embedding broadcast, 1
/// for the `(value, index)` reduce that closes a sharded unembedding -- and it
/// is a parameter rather than a constant `2` because a caller that assumes the
/// ends are cut when they are not is measuring 2.4% of a collective budget that
/// does not exist.
pub const fn collectives_per_token(layers: usize, ends: usize) -> usize {
    2 * layers + ends
}

/// Microseconds one token spends in collectives, at a measured per-collective
/// latency.
///
/// The default `us_each` a caller should pass is **29.56**, from
/// `scripts/interconnect_probe.sh` at 4096 f32 on the ConnectX pair. It is a
/// LATENCY: at this size the wire is 4% of it, so a caller that halves the
/// payload should not expect this number to move.
pub fn collective_ms_per_token(layers: usize, ends: usize, us_each: f64) -> f64 {
    collectives_per_token(layers, ends) as f64 * us_each / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 42-layer release's axes, so a config change that breaks the split
    /// fails here rather than in a binder.
    const HEADS: usize = 32;
    const KV_HEADS: usize = 8;
    const D_REL: usize = 16;
    const N_ROUTED: usize = 256;
    const N_SHARED: usize = 2;
    const VOCAB: usize = 201024;
    const INTER: usize = 2048;
    const DENSE_INTER: usize = 16384;

    fn pair() -> (Tp, Tp) {
        (Tp::new(0, 2).unwrap(), Tp::new(1, 2).unwrap())
    }

    #[test]
    fn the_identity_world_shards_nothing() {
        let t = Tp::default();
        assert!(!t.is_split());
        assert_eq!(t.q_heads(HEADS).unwrap(), 0..HEADS);
        assert_eq!(t.routed_inter(INTER).unwrap(), 0..INTER);
        assert_eq!(t.unembed_offset(VOCAB).unwrap(), 0);
    }

    #[test]
    fn every_axis_of_the_42_layer_release_divides_by_two() {
        let (a, _) = pair();
        for (name, extent) in [
            ("heads", HEADS),
            ("kv_heads", KV_HEADS),
            ("routed", N_ROUTED),
            ("shared", N_SHARED),
            ("vocab", VOCAB),
            ("inter", INTER),
            ("dense_inter", DENSE_INTER),
        ] {
            assert!(
                a.shard("axis", extent).is_ok(),
                "{name} = {extent} does not divide by 2"
            );
        }
    }

    /// The property that matters: the two ranks' shares TILE the axis — they
    /// touch every index once and no index twice. A split that overlaps
    /// double-counts under an all-reduce sum and a split that gaps drops
    /// weight, and both produce plausible text.
    #[test]
    fn the_shares_tile_the_axis_exactly() {
        for world in [1usize, 2, 4, 8] {
            for extent in [32usize, 8, 256, 201024, 16384] {
                if extent % world != 0 {
                    continue;
                }
                let mut seen = vec![0u8; extent];
                for rank in 0..world {
                    let t = Tp::new(rank, world).unwrap();
                    for i in t.shard("axis", extent).unwrap() {
                        seen[i] += 1;
                    }
                }
                assert!(
                    seen.iter().all(|&c| c == 1),
                    "world {world} over extent {extent} does not tile"
                );
            }
        }
    }

    #[test]
    fn gqa_splits_16_query_heads_and_4_kv_heads_per_rank() {
        let (a, b) = pair();
        assert_eq!(a.q_heads(HEADS).unwrap(), 0..16);
        assert_eq!(b.q_heads(HEADS).unwrap(), 16..32);
        // No KV duplication: 8 KV heads split 4/4, which is what makes the
        // cache halve rather than replicate.
        assert_eq!(a.kv_heads(KV_HEADS).unwrap(), 0..4);
        assert_eq!(b.kv_heads(KV_HEADS).unwrap(), 4..8);
    }

    #[test]
    fn the_relative_path_cuts_on_the_same_head_axis_as_wq() {
        let (a, b) = pair();
        assert_eq!(a.rel_rows(HEADS, D_REL).unwrap(), 0..256);
        assert_eq!(b.rel_rows(HEADS, D_REL).unwrap(), 256..512);
        // and it is exactly q_heads * d_rel, not its own independent cut
        for t in [a, b] {
            let q = t.q_heads(HEADS).unwrap();
            let r = t.rel_rows(HEADS, D_REL).unwrap();
            assert_eq!(r.len(), q.len() * D_REL);
        }
    }

    /// Every MLP-family weight is cut on the SAME axis by the SAME range, which
    /// is what lets one all-reduce close all of them.
    #[test]
    fn routed_shared_and_dense_all_cut_the_intermediate_axis() {
        let (a, b) = pair();
        assert_eq!(a.routed_inter(INTER).unwrap(), 0..1024);
        assert_eq!(b.routed_inter(INTER).unwrap(), 1024..2048);
        assert_eq!(
            a.shared_inter(INTER).unwrap(),
            a.routed_inter(INTER).unwrap()
        );
        assert_eq!(
            b.shared_inter(INTER).unwrap(),
            b.routed_inter(INTER).unwrap()
        );
        assert_eq!(a.dense_inter(DENSE_INTER).unwrap(), 0..8192);
        assert_eq!(b.dense_inter(DENSE_INTER).unwrap(), 8192..16384);
        // `n_shared` is NOT cut: both ranks still hold both shared experts, at
        // half width each. A cut by instance would leave `n_shared == 1` and
        // silently shift every shared gamma column by the rank.
        for t in [a, b] {
            assert_eq!(t.shared_inter(INTER).unwrap().len(), INTER / 2);
        }
        let _ = N_SHARED;
    }

    /// The interleaved layout's whole point: one contiguous run, gate and up
    /// already paired, and it is NOT the halved layout's first range.
    #[test]
    fn interleaved_w13_rows_are_one_run_and_halved_rows_are_two() {
        let (a, b) = pair();
        assert_eq!(a.w13_interleaved_rows(INTER).unwrap(), 0..2048);
        assert_eq!(b.w13_interleaved_rows(INTER).unwrap(), 2048..4096);
        // Together they tile `[0, 2 * inter)` exactly once.
        let mut seen = vec![0u8; 2 * INTER];
        for r in [
            a.w13_interleaved_rows(INTER).unwrap(),
            b.w13_interleaved_rows(INTER).unwrap(),
        ] {
            for i in r {
                seen[i] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        // The wrong reading, pinned: rank 0's interleaved run and rank 0's
        // halved GATE range are both row ranges of the same weight and they are
        // not the same rows. Reading one as the other is finite and fluent.
        let (g, u) = a.w13_halved_rows(INTER).unwrap();
        assert_eq!((g.clone(), u), (0..1024, 2048..3072));
        assert_ne!(a.w13_interleaved_rows(INTER).unwrap(), g);
    }

    #[test]
    fn a_local_argmax_maps_back_to_a_global_token_id() {
        let (a, b) = pair();
        assert_eq!(a.unembed_offset(VOCAB).unwrap(), 0);
        assert_eq!(b.unembed_offset(VOCAB).unwrap(), 100512);
        // The winner is the larger of the two locals, and rank 1's index has to
        // be LIFTED by its offset before the comparison means anything. Token
        // 126500 lives on rank 1 at local row 25988; forgetting the lift would
        // return 25988, which is a real token on rank 0 and therefore fluent
        // nonsense rather than an error.
        let lift = |t: Tp, local: usize| local + t.unembed_offset(VOCAB).unwrap();
        assert_eq!(lift(b, 25988), 126500);

        let (val_a, idx_a) = (18.50f32, 306usize);
        let (val_b, idx_b) = (18.59f32, 25988usize);
        let winner = if val_a >= val_b {
            lift(a, idx_a)
        } else {
            lift(b, idx_b)
        };
        assert_eq!(winner, 126500);

        // Ties go to the lower rank, which reproduces cubek's ArgMax rule
        // ("the smallest coordinate in case of equality") across the shard
        // boundary rather than only within a shard.
        let tie = if val_a >= val_a {
            lift(a, idx_a)
        } else {
            lift(b, idx_b)
        };
        assert_eq!(tie, 306);
    }

    /// The failure this exists to make loud: a contiguous half of
    /// `[2 * inter, hidden]` is all gate and no up.
    #[test]
    fn w13_halved_rows_are_two_disjoint_ranges_not_one() {
        let (a, b) = pair();
        let (ga, ua) = a.w13_halved_rows(INTER).unwrap();
        let (gb, ub) = b.w13_halved_rows(INTER).unwrap();
        assert_eq!(ga, 0..1024);
        assert_eq!(ua, 2048..3072);
        assert_eq!(gb, 1024..2048);
        assert_eq!(ub, 3072..4096);
        // Together the four ranges tile [0, 2 * inter) exactly once.
        let mut seen = vec![0u8; 2 * INTER];
        for r in [ga, ua, gb, ub] {
            for i in r {
                seen[i] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
    }

    #[test]
    fn an_axis_that_does_not_divide_names_itself() {
        let t = Tp::new(0, 2).unwrap();
        // The 66-layer sibling's 7 global layers, as a stand-in for any odd
        // axis someone tries to cut.
        let e = t.shard("num_global_layers", 7).unwrap_err();
        assert_eq!(e.axis, "num_global_layers");
        assert!(e.to_string().contains("num_global_layers"));
        assert!(e.to_string().contains('7'));
    }

    #[test]
    fn a_rank_past_the_world_is_refused() {
        assert!(Tp::new(2, 2).is_err());
        assert!(Tp::new(0, 0).is_err());
        assert!(Tp::new(0, 1).is_ok());
    }

    /// The header's headline number, recomputed rather than quoted.
    #[test]
    fn the_collective_budget_is_two_and_a_half_milliseconds_a_token() {
        // What this build issues: two a layer and nothing at the ends.
        assert_eq!(collectives_per_token(42, 0), 84);
        let ms = collective_ms_per_token(42, 0, 29.56);
        assert!(
            (ms - 2.483).abs() < 1e-3,
            "84 layer collectives at 29.56 us is {ms} ms"
        );
        // Against the measured 105.2 ms/token two-node round trip, that is
        // under 3% -- the claim the whole design rests on.
        assert!(ms / 105.2 < 0.03);
        // What cutting the two ends WOULD add, so the 86 that circulates is
        // recomputable rather than merely remembered.
        assert_eq!(collectives_per_token(42, 2), 86);
    }

    /// Expert-parallel does not balance at batch one, and this is by how much.
    /// If someone later changes `num_experts_per_tok`, this recomputes.
    #[test]
    fn top_6_over_two_ranks_waits_on_3_9375_experts_not_3() {
        fn comb(n: u64, k: u64) -> f64 {
            (0..k).map(|i| (n - i) as f64 / (i + 1) as f64).product()
        }
        let top_k = 6u64;
        let e: f64 = (0..=top_k)
            .map(|k| {
                let p = comb(top_k, k) / 2f64.powi(top_k as i32);
                p * std::cmp::max(k, top_k - k) as f64
            })
            .sum();
        assert!((e - 3.9375).abs() < 1e-9, "E[max] = {e}");
        // 1.52x on the routed term, not 2x.
        assert!((top_k as f64 / e - 1.5238).abs() < 1e-3);
    }
}
