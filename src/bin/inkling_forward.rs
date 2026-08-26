//! A real forward pass of Inkling-Small across a CLUSTER, on the device.
//!
//! Every gate ends with the same disclaimer: the checkpoint-name to module
//! mapping is authored on both sides, so a shared misreading would pass. This
//! is the check that can settle it. Coherent continuations cannot come out of
//! a wrong mapping — a transposed projection or a swapped gate/up half
//! produces noise, not English.
//!
//! # What a bare run measures, and the rule that keeps it honest
//!
//! Every switch below started life default-OFF, because a switch is written the
//! day its lane is newer than its measurement and the arm that has run for
//! months is the one a run that says nothing should get. Left alone, that rule
//! has a failure mode with a name: the DEFAULT CONFIGURATION BECOMES THE
//! UN-IMPROVED BASELINE, so every run measures the thing we did not build, and
//! the features accumulate untested beside it.
//!
//! So the rule has a second half. **A lane whose output is bit-identical or
//! better, at no admission cost, goes ON by default, and the switch survives
//! only as its ABLATION -- same name, inverted default.** A lane that changes
//! what the model SAYS stays off until someone has measured what it changes,
//! and no argument from bandwidth or grid occupancy substitutes for that
//! measurement.
//!
//! Which side of the line each switch is on:
//!
//! | switch | default | why |
//! |---|---|---|
//! | `INK_FUSE_QKVR` | **ON** | output measured bit-identical; 52 MiB/layer, charged at admission |
//! | `INK_DEV_ROUTE` | **ON** | same decision, computed where the logits already are |
//! | `INK_ACT_BF16` | **ON** | the reference's own operand dtype |
//! | `INK_DEV_PLAN` | **ON** | +8.33%, 5 of 5 interleaved pairs, and it halves the spread |
//! | `INK_SWZ` | **ON** | the routed experts are written in MMA-fragment order by the startup copy; output identical, startup cost nil |
//! | *(the head lane)* | **gone** | was `INK_W4A16_HEAD`. No switch: the lane is W4A16 |
//! | *(the sink lane)* | **gone** | was `INK_W4A16_SINKS`. No switch: the sinks are W4A16 |
//! | *(the KV lane)* | **gone** | was `INK_FP4_KV`. No switch: the pages are NVFP4 |
//! | `INK_ANN_HEAD` | **8192** | the approximate head, on. `0` is the exact-lane ablation |
//! | `INK_ANN_ROT` | on | the sketch's random rotation. `0` is the raw-coordinate ablation |
//! | `INK_TEMP` | 0.0 | sampling, as noise on the query. `0.0` is today's greedy decode exactly |
//! | `INK_DRAFT_TOPK` | **512** | pruned by default; `0` disables, for the sweep and for `INK_MTP_PROB` |
//! | `INK_GEMM_AUTOTUNE` | off | times a GEMM that had the whole device, which four overlapping projections do not |
//! | `INK_DENSE_WEIGHTS=device` | off | faster, but costs 3.42 GiB at 0:15 and REFUSES ranges that run today |
//! | `INK_ZEROCOPY=0` | off | diagnostic; its 60+ GiB of expert duplication is priced nowhere |
//!
//! ## The bar is capability, and it is NOT bit-identicality
//!
//! Four of the rows above used to read "off -- model-quality change", meaning
//! their output was not bit-identical to the arm they replaced. That bar is
//! retired, for a reason worth writing down rather than re-deriving:
//!
//! **This runtime is not bit-identical to itself IN THE CACHED DECODE LANE.**
//! `devplan_verify_layer` records it disagreeing on 8.55% of argmax positions
//! between two runs of the same binary. A bar the baseline fails is not a bar.
//!
//! CORRECTION, 2026-08-25, measured after this was written: that is lane-
//! specific and was over-generalised here. In the UNCACHED PREFILL lane the
//! runtime is perfectly reproducible -- 3988/3988 identical argmax positions
//! and identical logits across two genuinely independent runs. So a change's
//! output difference in that lane is REAL and reproducible, not noise, and it
//! cannot be waved away by citing nondeterminism. The conclusion survives on
//! its own merits (capability, not identity) -- but the premise as stated is
//! only true of one lane.
//!
//! And it rejected things that work. The `INK_FP4_KV` gate refused NVFP4 KV
//! because it perturbs 91% of dense RMS -- while the reference implementation
//! ships the same `fp4_mx_block16` and retrieves a needle EXACTLY from a
//! 307,581-token prompt. Nobody wants an unperturbed RMS. They want retrieval.
//!
//! So: coherent text, retrieval, acceptance. A different-but-fluent
//! continuation is not a regression -- the W4A16 head changed one token in 24,
//! at a 0.08-logit gap, and both continuations read as English.
//!
//! Bit-identicality is still the right instrument for a DIFFERENT claim: when a
//! change is supposed to be pure scheduling, a matching digest proves it. That
//! is what `gemm_grid_parity` is for, and why it was committed BEFORE the
//! kernel change it verifies. Evidence for a claim, not a licence to ship.
//!
//! # There is no single-node mode, and that is deliberate
//!
//! One box cannot hold this model: 144 GiB of weights against 119 GiB of RAM.
//! It USED to run anyway, by streaming — re-reading each token's expert slabs
//! off the SSD because the page cache had evicted them — and every measurement
//! taken in that mode was a measurement of a disk, not of a model. So
//! `INK_LAYERS` is now REQUIRED and must name a strict subrange: a process that
//! would run the whole stack refuses to start.
//!
//! Not "exactly two". Two is what THIS model needs (byte-balanced at layer 20:
//! layer 2 carries BF16 experts, 12.7 GiB against 3.55 GiB for an NVFP4 layer,
//! so an even 21/21 split is a lopsided 82.73/76.14 GiB one). The 66-layer
//! sibling wants five to seven. The rule is that no node runs the whole stack.
//!
//! ## And the split really is 20, which was said here and not done
//!
//! Every perf run used 0:21 / 21:42 while this paragraph named layer 20. Four
//! runs at 130,000 tokens, one binary, everything else equal.
//!
//! Read DEMAND -- `used + swap` -- and not either column alone. The kernel
//! decides how much of a working set to keep resident and how much to page, and
//! that decision moves several gibibytes between the two columns from run to
//! run: the two 20/22 rows below differ by 12.6 GiB of swap on the tail and by
//! 1.3 GiB of demand. Demand is the quantity that reproduces; swap is the
//! quantity that is easy to quote and does not.
//!
//! Capacity is 121.63 + 16 GiB of swap on the head's box, 119.63 + 16 on the
//! tail's. `margin` is capacity less demand on whichever box is worse -- which
//! is the only number that says how much further this pair could go.
//!
//! | split | head demand | tail demand | margin on the worse box |
//! |---|---|---|---|
//! | 21/21 | 131.72 GiB | 117.88 GiB | **5.91 GiB** (head) |
//! | 20/22 | 127.51 | 122.18 | 10.12 (head) |
//! | 20/22, repeat | 128.04 | 123.50 | 9.59 (head) |
//! | 19/23 | 124.48 | 125.81 | 9.82 (tail) |
//!
//! So 21/21 is measurably wrong -- it leaves under six gibibytes on the head
//! while the tail sits on nineteen -- and 20/22 and 19/23 are both about four
//! gibibytes better and are NOT distinguishable from each other: 9.59, 9.82 and
//! 10.12 span less than the repeat spans. Wall time is 1820, 1828, 1832 and
//! 1855 s, a 1.9% spread that orders nothing either.
//!
//! 20 is the one to take, and the reason is not that it measured highest. It is
//! byte-balanced, so it is the split that does not depend on `n`; 19 wins its
//! half a gibibyte by over-correcting for a term that grows.
//!
//! That term is why the head is the heavier end at every split. Layers 0 and 1
//! are the DENSE ones and their `[n, 16384]` gate and up are 3.97 GiB EACH at
//! 130,000 tokens -- about 8 GiB the tail never carries, proportional to `n`,
//! and the head's by definition. So the true balance point sits slightly left
//! of byte-balance and drifts further left as the sequence grows. At 130,000 it
//! has not drifted a whole layer yet.
//!
//! `INK_LAYERS=LO:HI` runs that half-open range; `INK_PIPE=head:HOST:PORT`
//! sends the residual stream on when the range ends, and `INK_PIPE=tail:ADDR`
//! receives it, finishes the stack and returns the argmax. Only `[n, 4096]` f32
//! crosses — 16 KB per token per boundary, once.
//!
//! # The split is by layer for now, and the reason it used to give is wrong
//!
//! This doc used to say a within-layer split "needs an all-reduce per layer and
//! 1 GbE cannot carry it". That measured the management NIC. The boxes are also
//! joined by a direct-attach ConnectX pair, and on that link, measured:
//!
//!   ib_write_lat, 8 KB          3.68 us one way   (p99 3.78, stdev 0.01)
//!   ib_write_bw, peak          13.02 GB/s         (saturates by 16 KB)
//!   iperf3 TCP, 8 streams     111 Gbit/s
//!   NCCL all-reduce, 16 KB     26.84 us           (RoCE, GPU-resident, 2 ranks)
//!   NCCL all-reduce, large     13.78 GB/s
//!
//! A within-layer split wants two all-reduces per layer — one after the
//! attention out-projection, one after the MoE down-projection — so 84 per
//! token, each `[1, 4096]`. At the measured 26.84 us that is 2.25 ms per token
//! against a per-token budget in the tens of ms. Bandwidth is not close to
//! being the constraint: 84 x 16 KB is 1.3 MB per token, well under a percent
//! of the link.
//!
//! The transport does matter, and it is the whole finding. The kernel network
//! path on these boxes costs ~185 us round trip for an 8 KB ping-pong (ICMP
//! agrees: 171 us minimum), which is 25x the RDMA number and would put the same
//! 84 collectives at 15.5 ms per token. So a within-layer split is affordable
//! over RDMA or NCCL and is NOT affordable over ordinary sockets — which is
//! also why `burn-collective`, whose transport is WebSocket plus MessagePack,
//! is the wrong tool here despite exposing exactly the right API.
//!
//! What a within-layer split would buy is the idle half. The layer split runs
//! the two nodes strictly in sequence; the pipe-utilisation block below reports
//! each end computing near 50% and blocked on the other for the rest. Splitting
//! within a layer runs both ends on every layer at once. That is the case for
//! doing it, and the bandwidth objection above is not a reason against it.
//!
//!   # tail, on the second box
//!   INK_LAYERS=20:42 INK_PIPE=tail:0.0.0.0:7654 inkling_forward <pile> <ids> <out>
//!   # head, on the first
//!   INK_LAYERS=0:20  INK_PIPE=head:<tail-host>:7654 inkling_forward <pile> <ids> <out>
//!
//! # Drafting happens on the tail
//!
//! `INK_MTP=k` used to refuse a pipe, on the reasoning that "the head owns no
//! unembedding and the tail owns no embedding table, so neither end can draft
//! alone". Half of that is right. An MTP head takes the stack's FINAL hidden
//! state and the embedding of the token one step ahead, and the tail computes
//! the last layer, owns the final norm, owns the unembedding, and follows the
//! sequence by recomputing the argmax rather than being told. The embedding
//! table was the only missing piece, and a tail with `INK_MTP` set now loads it
//! — deliberately, for this configuration and no other, since the point of the
//! split is that neither box pins a table it never reads.
//!
//! Set `INK_MTP` on the TAIL process only. It is refused on a head, which holds
//! neither the final hidden state nor a way to turn one into a token.
//!
//! ## What a draft costs, and `INK_DRAFT_TOPK`
//!
//! A draft is one MTP block and one unembedding, and the second of those is the
//! expensive half: the table is `201024 x 4096` BF16, 1.65 GiB, and every draft
//! DEPTH streams all of it to keep one token. That is the same weight stream the
//! stage timer charges to `head / unembed`, paid again per depth, and it is why
//! speculation on this stack has to be argued rather than assumed.
//!
//! Two of the three costs there were pure overhead and are gone unconditionally:
//! the full logits row is no longer read back to the host (only the winning index
//! is, via [`argmax_row_dev`]), and the readback that `INK_MTP_PROB` needs now
//! happens only when `INK_MTP_PROB` is set. Neither changes a token.
//!
//! The weight stream itself is not overhead, and shrinking it means drafting
//! against fewer tokens. `INK_DRAFT_TOPK=N` restricts the DRAFT's unembedding to
//! the N tokens the main model just ranked highest at this position — gathered
//! once per step, shared by every depth, 4 MiB at N = 512 against 1.65 GiB. A
//! token outside that set can no longer be drafted, so drafts and acceptance
//! both move.
//!
//! **This paragraph said "default-off" until 2026-08-25 and the switch table at
//! the top said 512, and the table was right.** The default flipped without the
//! sweep the code's own comment says would settle it, and the two halves of this
//! file then disagreed about what a bare run does — which matters because it is
//! exactly the kind of number that gets quoted across runs. Worth knowing when
//! reading older acceptance figures: `mary-measure` (160848d) and everything
//! before the flip measured the UNPRUNED head, so those figures do not transfer
//! to a bare run of current main.
//!
//! And the pruning had a defect that only the SPECULATIVE lane could reach: the
//! candidate rows were gathered from the LAST row of the pass rather than from
//! the row the answer came off. A verify pass keeps the leading run of drafts
//! that agreed, so the answer is row `new_toks.len() - 1`, and on the 31.9% of
//! `INK_SPEC=1` passes that accept nothing the last row is a distribution over a
//! token the model rejected -- one that need not even contain `best`, which the
//! comment there asserted it always would. Fixed 2026-08-25. Exact-argmax
//! verification means it could never emit a wrong token and never raise a flag;
//! it could only lower acceptance, silently, on the configuration that had just
//! become the default. It is refused together with
//! `INK_MTP_PROB`, which scores the draft's distribution by full-vocabulary token
//! index and would otherwise be handed 512 numbers about a different index space.
//!
//! # The decode baseline, and the exact configuration that produced it
//!
//! Two rates get quoted for this model and they are not the same number, so
//! both are here with the run that measured them. Current main, two nodes over
//! the direct link, layers 0..21 and 21..42, `INK_KV=1`, `INK_GEN=100`,
//! `INK_MTP` unset, a five-token English prompt, context 5 -> 105:
//!
//!   round trip, one token                126.8 ms
//!   tail node, layers 21..42 + unembed    69.3 ms   p50 of 96 warm passes
//!   head node, layers 0..21 + embed      ~55.5 ms   derived, see below
//!   the head's own view of one step      125.8 ms   p50, compute AND blocked
//!
//! The round trip is the wall clock between two consecutive tokens, taken
//! between step 3 and step 100 so that the two cold passes -- 4.54 s and 0.55 s
//! of kernel compilation -- fall outside the interval instead of being averaged
//! into it. Averaged in, the same run reports 135.2 ms/step, which is why the
//! summary block below now prints a WARM line beside the pooled one. A separate
//! 40-step run reads 125.1 ms/step warm against 144.8 ms/step pooled, so the
//! correction is worth more than the run-to-run spread, not less.
//!
//! A NODE rate is not a token rate. The two halves run strictly in sequence, so
//! 69.3 and ~55.5 add; neither of them, and no fraction of the round trip, is a
//! per-token figure on its own. The head does not report its compute per step
//! (only pooled, where the cold pass is still in the average), so its ~55.5 ms
//! is the round trip less the tail's p50 and two 16 KB wire crossings.
//!
//! # What the idle half is worth, and why drafting does not claim it
//!
//! Each end spends about half the loop waiting for the other -- head computing
//! 46.9% and blocked 53.1%, tail computing 55.2% and blocked 44.8%. That idle
//! half is the standing argument for speculation, and the same run prices it.
//!
//! `INK_MTP=4` on the tail, same prompt, same 100 steps:
//!
//!   round trip                           151.5 ms   against 126.8 undrafted
//!   tail compute                         151.0 ms/step, drafting 110.5 of it
//!   tail BLOCKED on the head               0.0 ms/step
//!
//! Four device MTP heads do not fit in the idle half. They fill it and then
//! push 25 ms into the critical path: the tail goes from 44.8% idle to 0.0%.
//!
//! Nor are the drafts worth that. Over the 100 steps, depth-1 acceptance is
//! 22.0% (95% CI 15.0-31.1%), and a whole four-deep draft set yields a mean
//! 0.268 accepted tokens per verify pass. Against the measured width cost
//! c(2) = 1.492, a k=1 loop that always speculates is (1 + 0.220) / 1.492 =
//! 0.818x -- it LOSES 18%. Gating on the draft head's own confidence at
//! tau = 0.2 turns it into 1.059x, a 6% gain taken on 27% of steps.
//!
//! So the idle half is idle, and this file does not consume a draft at all:
//! `INK_MTP` measures what an accept-and-skip loop WOULD have kept, and no such
//! loop is here. A change that wants that half of the machine -- a within-layer
//! split, most obviously -- is not competing with speculation for it.
//!
//! # How a speculative token rate gets to be three times too good
//!
//! A verify pass `w + 1` rows wide can yield at most `w + 1` tokens. It does
//! not: the leading accepted run is what a loop keeps, and that run measures
//! 1.6 tokens per pass here and saturates -- widening `w` past 2 buys nothing,
//! because the acceptance rate falls faster than the row count rises. Dividing
//! a pass's cost by `w + 1` rather than by the measured mean is where a
//! 40 ms/token speculative rate comes from, and at w = 5 that is a factor of
//! 3.7 optimistic. A 240 ms pass yielding 1.6 tokens is 150 ms a token, which
//! is SLOWER than the 126.8 ms one row costs.
//!
//! This is not hypothetical arithmetic: the same conclusion falls out of a
//! wired accept-and-skip loop, which reads 0.916x at w = 1, 0.866x at w = 2 and
//! 0.742x at w = 3 against an unspeculated 127.1 ms baseline -- a baseline that
//! is also an independent confirmation of the 126.8 ms above.
//! # `INK_SPEC=k`: the accept-and-skip loop, and what decides whether it pays
//!
//! The loop the MTP acceptance measurement was for, wired end to end. Set it on
//! BOTH processes and to the same value. Every answer carries `k` drafts back
//! with it; the next pass feeds the confirmed token FOLLOWED BY those drafts,
//! so a verify pass is `k + 1` rows wide, its rows' argmaxes are compared to
//! the drafts they were fed, and the leading run that agrees is kept. Both ends
//! roll back to that prefix through [`dev_lane::AttnCache::commit`] and the two
//! short convolutions' kept windows. Acceptance is exact argmax match, which is
//! not a concession: measured on this model it accepts MORE than a stochastic
//! rule (49.5% against 45.6% sampled and 40.6% under 1-TV).
//!
//! **It is off by default, and the reason is no longer "it loses" -- it is that
//! whether it wins is a property of the TEXT and not of the machine.** Two
//! corpora, the same binary, the same pipe (layers 0:21 and 21:42 over the
//! direct link, `INK_KV=1`, default GEMM lane), warm p50 of the whole cycle,
//! two runs per arm and both quoted:
//!
//!   A five-token English prompt, context 5 -> 105, `INK_GEN=100`:
//!
//!     arm          p50 ms   tok/pass   tok/s at p50   vs base
//!     INK_SPEC=0    127.2      1.000       7.862       1.000
//!     INK_SPEC=1    199.9      1.712       8.563       1.089
//!     INK_SPEC=2    236.8      2.082       8.792       1.118
//!     INK_SPEC=3    263.0      2.082       7.920       1.007
//!
//!   A 3732-token document, context 3732 -> 3792, `INK_GEN=60`:
//!
//!     arm          p50 ms   tok/pass   tok/s at p50   vs base
//!     INK_SPEC=0    151.6      1.000       6.599       1.000
//!     INK_SPEC=1    221.8      1.429       6.445       0.977
//!     INK_SPEC=2    264.9      1.395       5.268       0.798
//!
//! The width cost is nearly the SAME on both -- c(2) is 1.571 and 1.463 -- and
//! the acceptance is not: 71.2% of depth-1 drafts on the short prompt against
//! 42.9% on the document. The five-token prompt's continuation is a repeating
//! list template, and a template is easy to draft; a document's continuation is
//! not. So the honest reading of the short-prompt column is not "speculation
//! pays 1.1x", it is "speculation pays on text this predictable", and the
//! corpus that is not a five-token prompt says 0.98x.
//!
//! Widening past `k = 2` buys nothing on either corpus: the depth-3 draft was
//! accepted 0 times in 49 verify passes, so `k = 3` pays a wider pass for the
//! same 2.082 tokens and lands back at 1.007x.
//!
//! ## `INK_MTP_TEACH`: acceptance measured where there is enough of it to measure
//!
//! Every acceptance figure above is taken over tens of GENERATED steps, and
//! the section below is about how badly that number moves with the sequence.
//! `INK_MTP_TEACH=1` takes the same measurement a different way: head `d`'s row
//! `j` is fed `(stage[d-1][j], embed(ids[j + d + 1]))` and predicts
//! `ids[j + d + 2]`, which for every row but the last few is a token the PROMPT
//! ALREADY CONTAINS. So the drafting prefill scores two thousand positions in
//! one pass, against real text rather than against the model's own
//! continuation.
//!
//! It is the exact conditional rate the expected-prefix arithmetic wants:
//! `E = 1 + p1 + p1 p2 + ...` with `p_d = P(draft d right | drafts 1..d-1
//! right)`, and "drafts 1..d-1 right" is precisely the state teacher forcing
//! puts head `d` in -- the token head `d-1` was fed IS the token the verifier
//! would have accepted.
//!
//! Two warnings come with it, and both are load-bearing:
//!
//!  - It scores the PREFILL lane. `INK_MTP_STEPCHECK=1` is what says whether
//!    the cached STEP lane -- the one a decode loop drafts on -- computes the
//!    same rows.
//!  - A rate on a human document is a LOWER BOUND on the decode rate. On a
//!    document the head is asked for a token the MAIN STACK ITSELF only gets
//!    right 30% of the time; at decode the target IS the main stack's argmax.
//!    `INK_MTP_TEACH_FROM=<n>` points the same instrument at a seed followed by
//!    this model's own greedy continuation, which is the sequence a decode loop
//!    actually verifies.
//!
//! ## The ceiling, and why it changes what a low rate means
//!
//! `INK_MTP_TEACH` also scores DEPTH 0 -- the main stack's own next-token
//! argmax, on the same rows, through the same unembedding. That row is the one
//! that makes the rest readable, and it had never been measured:
//!
//! 2048 scored positions per depth, concat hidden-first, both corpora:
//!
//!                          corpus A (3732)   corpus B (3988)
//!     depth 0 (MAIN STACK)      0.2988           0.3369  [0.3168, 0.3577]
//!     depth 1                   0.2266           0.2593  [0.2408, 0.2787]
//!     depth 2                   0.1929           0.2358
//!     depth 3                   0.2085           0.2383
//!     depth 4                   0.2319           0.2495
//!
//!     depth 1 / depth 0            76%              77%
//!
//! Corpus A's depth-0 row was first taken before the double-norm fix; the
//! corrected instrument reads the SAME 612/2048 on the same rows, for the reason
//! given under "why three of these readings could never have mattered".
//!
//! **And corpus B's ceiling is independently confirmed.** A sibling ran the same
//! 3988 tokens teacher-forced through the model's ORDINARY logits path with a
//! different binary and dumped top-5 per position: 1309/3987 = **0.3283** top-1,
//! 0.5179 top-5. This instrument reads 0.3369 on the same file, whose interval
//! [0.3168, 0.3577] contains it. Two implementations, two paths, one number.
//!
//! So ~0.33 is what this checkpoint scores on real text, and the ratio the
//! draft head holds against it is 76-77% on BOTH corpora.
//!
//! A defect in the instrument, fixed, and worth keeping because the fix turned
//! out to change nothing and the reason it changed nothing is the interesting
//! part. `teach_rows` scored `main_dev`, which holds `entry` -- already
//! final-normed -- and normed it again. That is not idempotent:
//! `rms_norm(normalise(x) * g)` is `normalise(x) * g^2 / c`, so the ceiling was
//! being read through the learned gain SQUARED. The `norm` argument is the fix,
//! and the corrected run returns the identical 612/2048, because that gain is a
//! near-scalar. The bug was real; the bias it could introduce was not.
//!
//! Which means the draft head reaches **roughly three quarters of the main
//! stack's own accuracy on the equivalent task** -- 76% and 77% on the two
//! corpora, which is a steadier number than either rate alone --, and the per-depth table being FLAT is not a symptom:
//! as `d` grows the head gets more true-token context through the chained
//! embeddings and a staler hidden state, and the two roughly cancel.
//!
//! So a reference quoting "MTP1 acceptance ~0.85" is not quoting this quantity.
//! No drafter can score 0.85 against real text when the model it drafts for
//! scores 0.33; that figure has to be agreement with the target's ARGMAX on the
//! model's own decode stream, and the two must never be compared directly.
//!
//! ## What the wrapper composition turned out to be
//!
//! `transformers` defines no MTP composition, so [`MtpConcat`] kept both concat
//! orders reachable and said the acceptance rate would decide. With 2048
//! positions instead of twenty generated steps, it decides unambiguously:
//!
//!     concat hidden-first    depth 1  0.2266
//!     concat embed-first     depth 1  0.0010    <- 226x worse, below chance
//!
//! `mtp_hidden_states_first` is ALSO ABSENT from this checkpoint's
//! `mtp_config` -- the flag the default was named for is not there, only
//! `num_nextn_predict_layers`, `chain_hidden_post_norm` and `local_layer_ids`.
//! The default was right; it was right on a guess, and now it is measured.
//!
//! Every other reading of the wrapper, priced the same way on the same corpus
//! (depth 1, 2048 positions each):
//!
//!     THE DEFAULT                   0.2266
//!     concat embed-first            0.0010     226x worse -- settled
//!     swap the two norms            0.1650     -27%, so the NAMES are right
//!     backbone embed_norm OFF       0.1675     -26%, so the 08-24 fix is real
//!     entry RAW (INK_MTP_RAW=1)     0.2310     indistinguishable
//!     no output norm (OUTNORM=0)    0.2329     indistinguishable
//!     ablate the hidden operand     0.0869     both operands carry the rate
//!     ablate the embed operand      0.0347
//!
//! Which is the whole space of readings this wrapper has: which operand goes in
//! which half of `input_proj`, which gain norms which operand, whether the
//! backbone's embed norm precedes the depth head's, whether the entry and the
//! output take the final norm, and whether either operand matters at all. The
//! default wins or ties every one of them. **The composition is not the reason
//! acceptance is what it is.**
//!
//! The entry row corrects this file. The comment on `entry` says feeding the
//! final-normed hidden "measured twice as well (25% -> 50% on a matched
//! 20-token run)"; on 2048 positions the two are 0.2310 and 0.2266, which is
//! inside the interval. A twenty-step difference was a twenty-step difference.
//!
//! ### Why three of these readings could never have mattered
//!
//! Entry raw vs final-normed, the ceiling normed once vs twice, and
//! `INK_MTP_OUTNORM=0` vs on all came back inside their intervals. That is one
//! fact, and it is a property of the checkpoint rather than a coincidence:
//!
//!     model.llm.norm.weight       mean 5.6379  sd 0.3184   sd/mean 0.056
//!     model.mtp.0.embed_norm      mean 0.8388  sd 0.4408   sd/mean 0.526
//!
//! **The backbone's final norm is a near-SCALAR** -- a 4096-vector whose gain
//! varies by 5.6% about 5.64. RMS norm is scale-invariant, so applying it, not
//! applying it, or applying it twice moves the vector by something very close to
//! a global factor, and a global factor cannot move an argmax. The reasoning in
//! the `entry` comment -- "the two differ ONLY by the final norm's learned
//! weight vector" -- is right, and the weight vector turns out to be nearly
//! constant.
//!
//! The MTP heads' own `embed_norm` is NOT: sd/mean 0.526, a real per-channel
//! gain. Which is exactly the one whose ablation moves the rate, by 26%. The
//! norms that are shaped like a scalar do nothing; the norm that is shaped like
//! a function does something. Worth knowing before the next reading of the
//! wrapper gets measured on twenty steps.
//!
//! ## The cached step lane is not the bug (`INK_MTP_STEPCHECK`)
//!
//! The teacher-forced rate is measured in the drafting PREFILL and the decode
//! rate in the cached STEP, and nothing had ever checked that the two compute
//! the same rows -- `INK_MTP_CHECK` asserts only for the HOST cached lane and
//! merely reports for the device one, which is the lane every speculative run
//! drafts on. So a step path that quietly diverged would have looked exactly
//! like a model that drafts badly.
//!
//! `INK_MTP_STEPCHECK=1` reruns the head's whole-sequence prefill beside the
//! step and compares the row they share. Ten checks, two depths, a 256-token
//! seed:
//!
//!     max|prefill - step| 0.82% to 3.95% of |prefill|max
//!     drafted token: agree, 10 of 10
//!
//! The residual is two different attention kernels in BF16 -- the prefill takes
//! the fused/banded lane and the step the decode one -- and it never moved an
//! argmax. So the step lane is sound, and the prefill and decode measurements
//! are measurements of the same head.
//!
//! ## The mid-run stall, and where the evidence points
//!
//! Every `INK_SPEC>0` run over a long context has two or three passes that take
//! **8 to 24 seconds** while its neighbours take 0.21. The unspeculated arm on
//! the same prompt has none over one second after the two cold passes. It drags
//! a 212 ms/step arm to 541, and the harness prints the honest verdict:
//! `spread 87.4% <- SMALLER THAN THE SPREAD. Not a result.`
//!
//! What the logs already settle:
//!
//!  - **Both ends stall on the SAME pass, by different amounts.** Pass 30: the
//!    head's own layer loop 7.88 s, the tail's whole pass 7.78 s. Pass 106:
//!    12.24 s and 10.79 s. So it is not one box and it is not the wire; it is
//!    the same trigger arriving at two processes running the same shapes.
//!  - **It is not the pool.** `pool[after stack]` reads 12.44 GiB reserved /
//!    3.11 live / 9.34 GiB stranded over 503 slices on the passes either side of
//!    a stall and on the stall itself, and `pool cleanups: 0 of 21 layers`.
//!    Nothing was reserved, freed or drained.
//!  - **It is not extra work.** Both reps of `spec1` accept an identical
//!    81/119 and differ by 2.5x in wall time.
//!  - **It shows up in a HOST-ONLY bracket** -- 7.77 s of the head's
//!    "attention half", which the report labels "enqueue only (nothing in the
//!    loop synchronises)". Something inside an enqueue blocked.
//!
//! And what the autotune cache says. `~/.cache/cubecl/autotune/0.10.0/` keys
//! matmuls by shape ANCHORED TO POWERS OF TWO, and the live cache holds 247
//! entries whose `m` histogram is `{1: 28, 2: 19, 4: 36, 8: 33, 16: 30, ...}`.
//! So **`m = 2` and `m = 4` are their own keys**: every shape a verify pass
//! makes is a shape the `m == 1` lane never tunes, and the first process to see
//! one pays a timing race over every candidate kernel. `n` and `k` anchor too,
//! which is why a context growing 3732 -> 3930 does not itself trigger anything
//! -- it stays in the 4096 bucket.
//!
//! That leaves the question of what new shape appears at pass 30 and not at
//! pass 3, and the routed experts are the obvious candidate: an expert's GEMM
//! is `[rows on this expert, ...]`, a verify pass has two or three rows that can
//! CO-ROUTE, and how many land together is a property of the token, not of the
//! pass number. `CUBECL_DEBUG_LOG=<file>` makes cubecl name what it tunes and
//! when, which is the instrument that would close this.
//!
//! ## The binding constraint is c(2), not acceptance
//!
//! It is worth writing the break-even down, because it decides what work is
//! worth doing and it is not the work the acceptance rate points at.
//!
//! Speculation at width `k` pays iff `E(k) > c(k+1)`, where `c(m)` is what an
//! `m`-row verify pass costs against a one-row pass. On the REAL 42-layer
//! configuration -- two nodes, layers 0:21 and 21:42 over the direct link,
//! `INK_KV=1`, a 3732-token document, warm p50, both reps of a two-rep run:
//!
//!     spec0   117.7 ms/step   8.499 tok/s   E 1.000
//!     spec1   212.0 ms/step   7.942 tok/s   E 1.681   ->  0.934x
//!
//! Read `E 1.681` with its caveat: `INK_GEN=200` from this prompt, and the
//! continuation REPEATS near the end (`... 185338 314 17284 185338 314 ...`).
//! Repetition is easy to draft, so a degenerate tail inflates acceptance, and
//! the same file's `INK_MTP` section documents the opposite failure on the
//! five-token prompt -- a stack stuck on one id while the head proposed sensible
//! English, which DEFLATES it. Neither is a fact about the drafter. The
//! teacher-forced rate is the one that carries no such tail.
//!
//!   c(2) = 212.0 / 117.7 = **1.801**
//!
//! So the loop needs `p1 > 0.801` merely to break even, and it has 0.681. That
//! is a 7% shortfall -- and the number that makes it a shortfall is c(2), not
//! p1. At the reference's own quoted 0.85 the same machine would return
//! 1.85 / 1.801 = **1.03x**: a three-percent win. There is no acceptance rate
//! this configuration can be handed that makes speculation worth the risk while
//! the second row of a verify pass costs 80% of a whole pass.
//!
//! The same arithmetic run the other way: at c(2) = 1.2 -- what a second row
//! ought to cost when the lane is a real batched GEMM -- today's 1.681 would be
//! **1.40x**. The drafter is not what is between this stack and a speculative
//! win; the `m > 1` lane is. See "The width cost is a STEP" below, which found
//! half of it and named the rest.
//!
//! ### The draft is NOT free, and that is what the prune can buy
//!
//! Worth separating carefully, because both this file and the tail's own report
//! say the opposite. The `ANSWERED the head at` line reads "everything after
//! that (report, drafting) overlaps the head's next pass and the head never
//! waits for it". Without speculation that is true. **With `INK_SPEC` it is
//! false**, and the reason is one line further down: the drafts are sent
//! separately, AFTER drafting -- and the head cannot build its next feed without
//! them, because that feed IS `[best, draft0, ...]`.
//!
//! The pooled per-pass split says so directly:
//!
//!             head compute   head BLOCKED    tail to answer   tail drafting
//!     spec0      66.7 ms        77.2 ms          75.9 ms          0.0 ms
//!     spec1     124.7 ms       152.7 ms         123.0 ms         35.6 ms
//!
//! On spec0 the head's blocked time is the tail's answer time, 77.2 against
//! 75.9. On spec1 it is the tail's answer time PLUS its drafting, 152.7 against
//! 123.0 + 35.6 = 158.6. Were the draft hidden, the head would block for ~123.
//!
//! So of the 1.80x, roughly 0.30 of a pass is the drafter sitting on the
//! critical path, and it is the part that is cheap to attack:
//!
//!  - `INK_DRAFT_TOPK` cuts the draft's unembedding, which this file measures as
//!    about two thirds of a depth's cost. 35.6 -> ~13 ms takes the round trip to
//!    ~189 ms, `c_eff(2)` to ~1.61, and E = 1.681 from **0.93x to ~1.04x** --
//!    from losing to winning. That is a prediction; the sweep is what settles it,
//!    and it has to read tok/s, because acceptance can only fall.
//!  - Or take the draft off the critical path altogether: the head could start
//!    its own row 0 (`best`, which arrives first) while the drafts are still
//!    being computed. That keeps the full-vocabulary head AND the 35 ms.
//!
//! The other 1.50 is the verify pass itself, split across both nodes -- the
//! head's compute goes 66.7 -> 124.7 and the tail's answer 75.9 -> 123.0 for ONE
//! extra row -- and that is the `m > 1` lane, which is a bigger piece of work.
//!
//! ## Three acceptance rates that are all correct
//!
//! This file has quoted 22.0%, 50.0% and 71.2% for depth-1 acceptance and they
//! do not contradict each other -- they are the same measurement on three
//! different SEQUENCES, which is the whole finding.
//!
//! `INK_MTP=1` and `INK_MTP=4` with no speculation both report exactly
//! **22.0%** (95% CI 15.0-31.1) over 100 steps of the five-token prompt, so the
//! draft depth is not what moves it. What moves it is that the unspeculated
//! run's own continuation collapses into a single repeated token around step 38
//! and stays there: the log reads `drafted 410, actual 58189 -- miss` for sixty
//! consecutive steps, the draft head proposing sensible English while the stack
//! emits one id forever. A speculative run does not go there -- a verify pass
//! is `m > 1` and takes a different GEMM lane, and the two lanes diverge -- so
//! its 71.2% is measured on a sequence that never degenerates.
//!
//! Which is to say: none of the three is a corpus-independent acceptance rate
//! for this draft head, and the 41.9% on the document is the only one of them
//! measured on text worth quoting.
//!
//! ## The width cost is a STEP, and half of it was never the GEMM lane
//!
//! c(2) = 1.57 on the short prompt, against 1.332 that the uncached lane
//! predicted. The tail's own half is FLAT once the second row exists, which is
//! what a lost lane looks like rather than what extra work looks like, and
//! `gemv plane par` -- which requires `m == 1` and is the only lane that
//! reaches this part's memory roofline -- is the one a verify pass cannot use.
//!
//! That diagnosis was right and it was not the whole thing. Timed per stage on
//! the tail, the one-row-to-two-row step used to be +52 ms, of which the MLP's
//! short convolution alone was +31 and the attention half (which contains two
//! more of them) +18, against 2.5 ms for the MoE and dense GEMMs. The
//! convolutions were slice-built for any width above one;
//! [`crate::models::inkling::sconv::short_conv_batch`] is the kernel that
//! replaced them, and it took c(2) from 1.613 to 1.571, a two-row width probe
//! from 197.5 ms to 172.8, and the document's k=1 arm from 0.932x to 0.977x.
//! The residue is the GEMM lane, and a narrow lane at the gemv's bandwidth is
//! still the thing to build -- it is just no longer the whole bill.
//!
//! ## The same lane decides what the model SAYS
//!
//! A speculative run's text diverges from an `INK_SPEC=0` run's and the loop is
//! not why: `INK_WIDTH=2`, `4` and `8` -- a cost probe with no speculation
//! anywhere, whose extra rows carry different filler at every width -- produce
//! text IDENTICAL to each other and diverging from `INK_WIDTH=1` at the same
//! token. Every `m > 1` arm agrees with every other; only `m == 1` differs.
//! Against a host `f64` reference on identical BF16 bits, `gemv plane par` is
//! off by 1.2e-7 and the narrow tile lane by 1.36e-5, a factor of 113; over 42
//! layers and 40 cached positions that is enough to take the stack apart. Held
//! to one lane the batched cached attention is BIT-IDENTICAL to the single-row
//! lane at every position (`drift_table_at_real_width` in `burn.rs`, run with
//! `INK_GEMM='double cyclic mma'`), so the loop preserves the text exactly and
//! the runtime does not.
//!
//! # `INK_WIDTH=b`: what a b-row decode step COSTS
//!
//! An instrument, not a feature. Batched decode -- `b` independent sequences
//! sharing one weight stream -- is the largest untaken lever on this model, and
//! the question that decides whether it is worth building is whether a `b`-row
//! cached step costs `b` times a one-row one or barely more than one. A decode
//! step is bound on streaming BF16 weights; `b` rows stream them once. Against
//! that, every multi-row pass loses `gemv plane par`, which requires `m == 1`.
//!
//! `INK_WIDTH=b` prices the trade without any of the machinery a real batch
//! needs, because the COST of a `b`-row pass barely depends on whether the rows
//! belong to one sequence or to `b` of them: same projections, same MoE gather
//! over `b` independent routings, same unembedding of `b` rows, same weight
//! stream. Row 0 carries the real token, rows 1..b carry filler drawn fresh
//! every pass so the router picks `b` independent expert sets, only row 0's
//! argmax is taken and only row 0 is committed. Set it on the HEAD; the tail
//! reads the width off the wire.
//!
//! Warm p50 of the whole cycle, two runs per arm, `INK_GEN=60`:
//!
//!   five-token prompt, ctx 5 -> 65        3732-token document, ctx 3732 -> 3792
//!    b   p50 ms   c(b)   agg tok/s         b   p50 ms   c(b)   agg tok/s
//!    1    124.7   1.000     8.02           1    151.6   1.000     6.60
//!    2    172.8   1.386    11.57
//!    4    204.8   1.642    19.53           4    231.8   1.529    17.26
//!    8    253.7   2.035    31.53           8    280.1   1.848    28.56
//!
//! **Strongly sublinear, and that is the answer.** Eight rows cost 2.04 times
//! one row, so eight sequences decode at 31.5 tokens a second against 8.02 for
//! one -- 3.9x -- with no fabric work, no second box and no draft head. Nearly
//! the whole penalty is the STEP at the second row: from b = 2 to b = 8, four
//! times the rows cost 1.47 times the pass.
//!
//! ## What the probe does not charge for
//!
//! Its `b` rows share ONE KV cache, so attention reads `L` keys once where `b`
//! real sequences would read `L` keys `b` times. At ctx 65 that is nothing. At
//! ctx 3792 it is not, and the same table bounds it: one sequence's extra 3727
//! keys cost 151.6 - 124.7 = 26.9 ms a pass, so charging `(b - 1) x 26.9` ms is
//! an upper bound on the correction -- upper, because it charges the whole
//! context-length delta to attention and some of it is the wider relative-bias
//! table. Corrected, ctx 3792 reads 313 ms at b = 4 (12.8 tok/s aggregate,
//! 1.94x) and 468 ms at b = 8 (17.1 tok/s, 2.59x). The lever survives the
//! correction; it is 3.9x at short context and 2.6x at four thousand tokens.
//!
//! What it does NOT understate is the MoE: the router runs per row and the
//! expert gather follows it, and the probe's filler is drawn fresh every pass
//! precisely so that eight rows select eight independent expert sets. The log's
//! own counter agrees -- 126 expert slabs decoded at b = 1 against 544-642 at
//! b = 8 -- so expert streaming is priced at its real width and it is not what
//! makes the curve bend.
//!
//! ## The probe checks itself
//!
//! Rows 1..b are causally invisible to row 0, so `INK_WIDTH=b` must produce the
//! same text as `INK_WIDTH=1`. It produces the same text as every OTHER width:
//! b = 2, 4 and 8, whose filler rows are entirely different from each other,
//! emit token-for-token identical continuations on both corpora, and all three
//! diverge from b = 1 at the same position. That is the `m == 1` GEMM lane and
//! nothing else -- the same divergence a speculative run shows, reproduced with
//! no speculation anywhere in the process.
//!
//! ## What batching needed, and what it turned out to cost
//!
//! [`dev_lane::attention_steps`] takes ONE cache and `pos0`, puts row `i` at
//! `pos0 + i`, and masks row `i` against every earlier row of the batch -- it
//! is contiguous and causal by construction, which is what a speculative batch
//! is and what `b` independent sequences are not. That is what `INK_SLOTS`
//! below builds, and the answer is in the next section: a real b-slot pass came
//! out CHEAPER than this probe's, because the layout the slots needed was
//! cheaper than the one the single-row lane had.
//!
//! # `INK_SLOTS=b`: b independent sequences, one pass, b tokens
//!
//! The feature the probe above was an instrument for. `b` slots, each with its
//! own KV cache and its own token stream; one pass advances all of them by one
//! token. [`dev_lane::SlotCache`] and [`dev_lane::attention_slots`] are the
//! whole of it, and the "block-diagonal mask" the probe said this would need
//! does not exist: K and V are held `[slots * kv_heads, cap, head_dim]` and the
//! scores are a batched matmul over that leading axis, so slot `s`'s query
//! multiplies slot `s`'s keys and there is no mask element whose sign could be
//! wrong.
//!
//! The `b` prompts are `b` DISJOINT chunks of one token file, `INK_SLOT_LEN`
//! apart, and `INK_SLOT_OFFSETS` names the chunk per slot. Slot 0's text
//! therefore does not move when `b` changes, which is what makes the arms
//! comparable and what makes the contamination test possible. A prefill is
//! compute-bound and gains nothing from a batch, so the `b` of them run one at
//! a time and the batch begins where the decoding does.
//!
//! ## What it costs
//!
//! Warm p50 of the whole cycle, at least two runs per arm, two-node pipe,
//! layers 0:21 and 21:42, `INK_KV=1`, `/tmp/cover_150000.ids` (35,845 tokens of
//! prose; slot `s` takes `[s*L, (s+1)*L)`). "one-row" is the same binary with
//! `INK_SLOTS` unset -- [`dev_lane::attention_step`], which is what every
//! measurement before this one was taken on. `c(b)` is against the mean of the
//! `b = 1` SLOT-lane runs, not against the one-row lane: those are two
//! implementations of the same function, and mixing them into one ratio would
//! charge batching for a layout change.
//!
//!   L = 512, ctx 512 -> 612           L = 3732, ctx 3732 -> 3852
//!    b   p50 ms       c(b)   tok/s     b   p50 ms             c(b)   tok/s
//!    -   133.3 135.2    -    7.5 7.4   -   150.5 150.1          -    6.65 6.66
//!    1   125.5 134.1  1.000  8.0 7.5   1   135.6 130.5        1.000  7.38 7.67
//!    2   178.5 168.6  1.337  11.2 11.9 2   184.1 183.7        1.382  10.87 10.89
//!    4   207.6 212.0  1.617  19.3 18.9 4   219.5 219.8 217.9  1.650  18.2 18.2 18.4
//!    8   277.8 273.6  2.124  28.8 29.2 6   254.9 248.5 255.6  1.902  23.5 24.1 23.5
//!                                      8   282.7 286.9 289.9  2.153  28.3 27.9 27.6
//!
//! The `b = 6` and `b = 8` rows of the 3732 column are three runs each and are
//! post-fix; see "Where the b = 8 irreproducibility actually was" below for the
//! numbers they replace (283-657 and 311-1465) and what was wrong with them.
//!
//! **Eight independent sequences decode at 3.7x one**, and at a 3.7k context
//! each of them still carries its own 3792-key cache and reads it in full on
//! every pass. Nearly the whole penalty is the step at the SECOND row, exactly
//! as the probe found: from `b = 2` to `b = 8`, four times the sequences cost
//! 1.56 times the pass.
//!
//! **The layout change is worth its own line.** At `b = 1` the slot lane is
//! 130-136 ms where the one-row lane is 150 ms at a 3.7k context, and 125-134
//! against 133-135 at 512. [`dev_lane::attention_step`] holds K as
//! `[len, kv_heads * head_dim]` and expands it per step to
//! `[heads, len, head_dim]`, repeating each KV head `groups` times so the score
//! matmul is one GEMM over `heads`: 63 MB a layer at 3.8k keys, twice, on every
//! layer of every step. The slot cache is head-major and moves the repetition to
//! the QUERY side, where the `groups` queries sharing a KV head become the `m`
//! rows of that head's GEMM and cost nothing. One sequence gets that for free.
//!
//! ## What the probe overcharged, and by how much
//!
//! The probe bounded the b-th cache read at `(b - 1) x 26.9` ms, on the
//! reasoning that one sequence's extra 3727 keys cost the one-row lane
//! 151.6 - 124.7 ms a pass. Measured with `b` real caches each read in full, the
//! whole 512 -> 3732 context step costs **3.3 ms at b = 1, 10.3 ms at b = 2 and
//! 9.9 ms at b = 4 -- for the entire batch**, not per sequence. The bound was
//! five to ten times too pessimistic, and the reason is the expansion above:
//! most of the one-row lane's context delta was a copy that grows with the
//! context, not the attention it was attributed to.
//!
//! So the probe's corrected figures are wrong in the safe direction. It
//! predicted 313 ms and 12.8 aggregate tok/s at `b = 4` and ctx 3792; measured,
//! with four independent caches, it is **220 ms and 18.2 tok/s** -- better than
//! the probe's own UNCORRECTED number (231.8 ms, 17.26 tok/s), which is the
//! shape of a correction that was measuring the wrong thing.
//!
//! ## Where the b = 8 irreproducibility actually was
//!
//! `b` up to 4 was reproducible to a millisecond and 6 and 8 were not: the p50
//! over 120 passes came out anywhere from 283 to 657 ms at `b = 6` and 311 to
//! 1465 at `b = 8` across runs of the same binary on the same prompts, with
//! multi-SECOND maxima. Read as a decode problem that looks bimodal, and the
//! page ladder below was a plausible account of it. It is neither bimodal nor
//! a decode problem.
//!
//! The per-pass sequence says so on its own. Every one of those runs settles at
//! ~250 ms (`b = 6`) or ~285 ms (`b = 8`) and stays there; what varies is how
//! many passes it takes to GET there -- one run reached the floor on its second
//! decode pass and another was still at 1.5 s a hundred passes in. A p50 that
//! ranges 5x over a floor that does not move is not a bimodal steady state. It
//! is one transient of variable length, and the transient is the node climbing
//! back out of the swap that the `b` PREFILLS put it into.
//!
//! ## The prefill sequence held 30.49 GiB of a 25.77 GiB headroom
//!
//! The `b` prefills run one after another, and every finished one used to be
//! KEPT until all `b` were in, because the batch was assembled on the first
//! decode pass. That is the whole of it. `seam::pool_line`
//! (`memory_usage().bytes_in_use`) across the eight prefills of a 3732-token
//! run on the 21-layer head, 25.77 GiB of headroom by the admission gate:
//!
//!   after prefill   1     2     3     4     5     6     7     8   | batch
//!   pool live GiB   5.37  8.96 12.55 16.14 19.73 23.31 26.90 30.49|  3.24
//!   that pass, s    20.9   7.0   7.1   6.8   6.9 118.4 246.4  47.2|
//!
//! **+3.59 GiB per slot, against 0.16 GiB of keys and values.** A prefilled
//! [`dev_lane::AttnCache`] costs twenty times the keys it holds, and the `cat`
//! in the batch assembly is what collapsed it -- 30.49 GiB of per-slot caches
//! became a 3.24 GiB batch in one step. The sixth prefill is where the headroom
//! runs out and it is exactly where the pass times explode; the
//! reserved-minus-live column stayed under 1.1 GiB throughout, so page
//! stranding -- the thing `memory_cleanup` was added for, and the thing the
//! ladder was blamed for -- was never more than 4% of it.
//!
//! Slots are seated as they are prefilled now ([`dev_lane::SlotCache::seeded`],
//! [`dev_lane::SlotCache::seat`]), so the batch is the only long-lived
//! allocation and it is made once, from slot 0, before any of the churn:
//!
//!   after prefill   1     2     3     4     5     6     7     8
//!   pool live GiB   3.29  3.29  3.29  3.29  3.29  3.29  3.29  3.29
//!   that pass, s    20.9   6.8   6.8   6.9   6.8   6.9   6.9   6.9
//!
//! The whole prefill phase goes from 460 s to 69 s, the head's floor on
//! `MemAvailable` from 0.6 GiB to 21.4 and the tail's from 6.6 to 25.7, neither
//! node takes a page of swap where the tail took 5.2 GiB, and the decode is at
//! its floor on the second pass. Three independent runs of the same binary at
//! `b = 8` and a 3.7k context: **p50 282.7, 286.9, 289.9 ms** -- a 2.5% spread
//! where it was 4.7x.
//!
//! Three things that were proposed for the bimodality and are not the fix, said
//! plainly because each of them costs real work: rebalancing the layer split
//! moves ~3.5 GiB between two nodes that were both ~25 GiB short; a BF16 KV
//! cache halves 1.3 GiB of a 28 GiB overshoot; a paged KV cache addresses the
//! stranding term, and the stranding term measured 1.0 GiB.
//!
//! ## What the wall was actually capping, which was not eight
//!
//! Eight slots was where the old lane stopped being reproducible, so eight is
//! where the measurements stopped. It was never a property of the batch: the
//! batch costs 0.19 GiB a slot and the node has 25.77 GiB, and what ran out was
//! the b UNASSEMBLED prefills. With them gone the same `INK_SLOTS` goes
//! straight past it. Same pipe, 3732-token prompts, `/tmp/cover_400000.ids`
//! (100,623 tokens, so sixteen disjoint chunks fit), warm p50 over 40 passes:
//!
//!    b    p50 ms   aggregate tok/s   pool live (head)   MemAvailable floor
//!    8    289.0        27.7              3.29 GiB          21.7 / 26.1 GiB
//!   12    351.3        34.2              4.05             20.0 / 24.3
//!   16    399.7        40.0              4.80             19.6 / 24.0
//!
//! `b = 16` holds 388-415 ms over its forty passes and neither node takes a
//! page of swap. **40.0 aggregate tok/s**, and the pool is 4.80 GiB of a
//! 25.77 GiB headroom, so this is not the ceiling either -- it is where the
//! corpus ran out of disjoint chunks.
//!
//! ## Where the ceiling actually is, on a corpus that does not run out
//!
//! `/tmp/mix_891k.ids` is 891,510 tokens: the 100,623-token file above followed
//! by 790,887 tokens encoded from 3.2 MB of ordinary English prose -- 320 KB
//! from each of ten public-domain novels, front matter stripped. Slot `s` still
//! takes `[s * L, (s + 1) * L)`, so slots 0..26 carry exactly the tokens the
//! shorter file gave them and the arms remain comparable to every row above:
//! `b = 16` reads 404.8 ms here against 399.7 there. It admits 238 disjoint
//! 3732-token chunks.
//!
//! Real prose and not synthesis, because a 256-expert MoE routes pathologically
//! on a low-vocabulary prompt and a decode that degenerates into one repeated
//! token measures the text rather than the machine. Checked rather than
//! asserted: each 3732-token chunk carries 1106-1632 DISTINCT ids, and slot 0's
//! continuation is English at every `b`.
//!
//! Warm p50 over 40 passes, both columns on the same corpus and the same pipe;
//! "before" is the schedule this file shipped with and "after" is the fill gate
//! in [`crate::models::inkling::moegroup::grouped_nrep`]:
//!
//! ```text
//!    b   before p50   after p50   tok/s before -> after  pool live  MemAvail  swap
//!    8      289.0          -            27.7              3.29 GiB  21.7 GiB    0
//!   12      351.3          -            34.2              4.05      20.0        0
//!   16      399.7          -            40.0              4.80      19.2        0
//!          404.8
//!   24      496.0          -            48.4              6.31      16.6        0
//!   32      585.3        585.7          54.7 -> 54.6      7.82      15.8        0
//!          586.7
//!          586.4
//!   40      927.3          -            43.1              9.32      10.7        0
//!   48     1113.3        744.6          43.1 -> 64.5     10.83      10.1        0
//!                        741.5
//!   64     1459.5        866.3          43.9 -> 73.9     13.85       8.3        0
//!                        865.5
//!   96         -        1081.4                  88.8     19.88       1.4   5.50 GiB
//!                       1091.0
//!  128         -            -                      -     25.91         -      -
//! ```
//!
//! The `pool live` and `MemAvail` columns are the head's and are read off the
//! post-fix arm where there is one. `b = 40`'s p50 is not a description of that
//! run and the next section says why.
//!
//! **The old peak was 32 and it was a bug, not a limit.** Everything from 40 up
//! was paying for a grouped-GEMM schedule that reads the padded tile count as a
//! measure of work; `b = 40` is bimodal for the same reason, 648-691 ms or
//! 900-993 ms and nothing between. That is its own commit and its own header.
//!
//! ## Which resource binds, which is none of the three that were expected
//!
//! Not the device pool as a device: 19.88 GiB live at `b = 96` against the
//! 25.77 GiB the admission gate allows. Not the per-pass arithmetic: it is
//! still sublinear at 96, where 3x the slots of `b = 32` cost 1.85x the pass.
//! It is HOST memory, and the pool is what spends it, because on this part the
//! pool IS host memory. At `b = 96` the pool RESERVES 27.94 GiB to hold 19.88 --
//! 8.06 GiB stranded over 440 slices -- and `MemAvailable` bottoms at 1.38 GiB
//! with 5.50 GiB of the head swapped out. It still runs, at **88.8 aggregate
//! tok/s**, because what got swapped is the reservation nothing reads. The
//! DECODE is what survives, not the whole run: a second  reads 1091.0 ms
//! against 1081.4 while its prefill phase goes from 7.7 s a slot to 13.8, so at
//! ninety-six the thing memory pressure moves first is the part that runs once.
//!
//! `b = 128` is over it: 25.91 GiB live, and a prefill goes from 6.8 s a slot to
//! 56, 139, 72 and 59, so that arm was stopped after five of its hundred and
//! twenty-eight rather than left two hours to say the same thing a sixth time.
//! So the wall is between 96 and 128, it is the host's RAM, and
//! a third of what the batch spends there at 96 is allocator reservation rather
//! than cache. A pool that returned its stranded slices would be worth more of
//! this ceiling than anything in the model.
//!
//! For scale rather than for a claim: an independent vLLM TP2 setup on two of
//! these boxes reports 61.97 aggregate tok/s at five concurrent streams. This is
//! 88.8 at ninety-six, which is a different point on a different curve -- more
//! sequences, a 3.7k context each -- and the two are not a like-for-like
//! comparison in either direction.
//!
//! # Where a decode pass goes, measured on the device rather than at the host
//!
//! `nsys -t cuda` on the HEAD at `INK_SLOTS=32`, a 20 s window covering 34 warm
//! passes. The profiled run's own p50 is 586.7 ms against 585.3 unprofiled, so
//! nothing below is paying for the instrument.
//!
//! GPU kernel time totals **9538 ms of the 20 000 ms window, 47.7%**. The head's
//! own half of the pass is 51.3-51.6% of the wall, so **the head's GPU is busy
//! 93% of the time the head is not blocked on the tail**, and launch gaps are
//! the other 7%. This lane is not enqueue-bound, which is what the host-side
//! table above cannot tell you: with nothing synchronising in the loop, a
//! stage's device cost surfaces at whichever later call blocks, and `router +
//! group`'s BLOCKING read and `mlp short_conv` are mostly where it lands.
//!
//! Per pass, with the bytes each kernel reads and the rate that implies against
//! a 273 GB/s bus (236 measured by `inkling_bf16_gemm_bench`):
//!
//! ```text
//!   kernel                 ms/pass  share   reads    achieved
//!   fp4_linear_grouped      147.9   52.7%  25.4 GB   171 GB/s  18 NVFP4 MoE layers
//!   matmul_entry f32         39.7   14.1%   6.0 GB   152 GB/s  attention scores + values
//!   matmul_entry bf16        37.8   13.5%   3.8 GB   100 GB/s  q/k/v/r/o, router, shared
//!   bf16_linear_grouped      34.9   12.4%   5.7 GB   163 GB/s  layer 2, BF16 experts
//!   the other eighteen       20.2    7.3%
//! ```
//!
//! 41 GB in 302 ms is 138 GB/s, 58% of the bus, and the shape of the shortfall
//! is 187 small BF16 GEMMs at 100 GB/s rather than one lane in the wrong gear.
//!
//! Two things the table is honest about. The kernel times are a SUM and a sum
//! over-counts if anything overlaps; it does not here, and the check is that
//! 9538 ms fits inside the 10 260 ms the head was not blocked, which a lane
//! running two streams would not have done. And the `reads` column for the two
//! grouped rows is DERIVED, not measured per kernel: the pass's own
//! `stored bytes` counter gives 28.91 GiB for all nineteen routed layers
//! together, and it is split between them by expert size -- 48 MiB where they
//! are BF16 and 13.5 MiB where they are NVFP4, which the startup line's
//! 72.8 GiB over 9728 planes fixes. That puts ~113 active experts at layer 2,
//! and 113 is also what the next section's threshold story needs, so the two
//! agree without having been fitted to each other.
//!
//! ## What that rules out, said explicitly because each was a standing suspect
//!
//! * **The narrow GEMM lane.** `gemv plane par` requires `m == 1` and every
//!   wide pass loses it, and that was the largest named suspect for a batched
//!   decode. It is 13.5% of the head's GPU time. A lane that recovered ALL of
//!   it -- not the shortfall, the whole 37.8 ms -- is 6% of a 586 ms pass.
//! * **Attention at a long context.** 14.1%, at 152 GB/s, with 35 of 42 layers
//!   banded to a 512 window. It is already near the roofline and it is not the
//!   term that grows.
//! * **The routed-expert lane itself.** 52.7% of the time and 73% of the bus.
//!   There is nothing in it to win except fewer bytes, and fewer bytes per TOKEN
//!   is exactly what raising `b` already buys: 1.39 GiB a token at 8 slots,
//!   0.90 at 32, 0.48 at 96, because the expert set saturates while the tokens
//!   do not.
//!
//! ## What is left, and what it is worth
//!
//! The largest number in the profile is not a kernel. It is the **48% of the
//! wall on which this node's GPU has nothing to do**, and batching did not fill
//! it and cannot: head computing 51.3% / blocked 48.7% at `b = 32` and
//! 49.6 / 50.4 at 48, against 46.9 / 53.1 at one row. The two halves of one
//! token are strictly ordered, so widening the batch scales both halves and
//! leaves the ratio where it was. Two ways to take it, priced rather than
//! assumed:
//!
//! * **A within-layer split.** 84 all-reduces a token at the measured 26.84 us
//!   is 2.25 ms, **0.26% of an 866 ms pass** at `b = 64`; bandwidth is four
//!   orders of magnitude clear. It halves the bytes each node reads AND runs
//!   both GPUs on the same token, so the bound is 2x -- ~148 aggregate tok/s at
//!   64 -- and it halves per-node weight residency, which is the thing that ran
//!   out at 96. It needs every weight resharded, NCCL inside the loop, and the
//!   KV cache split by head.
//! * **A two-cohort pipeline interleave.** BUILT -- `INK_COHORTS`, the section
//!   after next. It predicted `p50(2c) / p50(c)` at a fixed slot budget:
//!   866.3 / 585.7 = 1.48x at 64 slots, 1.45x at 96. Measured, **1.49x at 64
//!   and 1.46x at 32**, and the head's idle half went from 48.7% to 16.6%.
//!
//! The interleave was the one to build first, and it stays the cheaper half of
//! the answer: it is a `VecDeque` and an index against every weight resharded,
//! and the two COMPOSE rather than competing -- the split halves what each node
//! reads, the interleave stops each node waiting, and neither takes the other's
//! win. What the interleave does NOT do is reduce per-node weight residency,
//! which is the thing that ran out at 96 slots, so the split is still the lever
//! for the ceiling and the interleave is the lever for the idle half.
//!
//! ## Batching the prefills, priced rather than built
//!
//! The `b` prefills are still sequential and are now the startup cost: eight of
//! them at 6.86 s is 55 s before the first decoded token. Sharing the batch
//! machinery would amortise the per-prefill FIXED part over `b`, and that part
//! is measurable -- a 512-token prefill is 1.42 s and a 3732-token one 6.86 s,
//! which is 1.69 ms a token plus **0.56 s** that does not depend on the length.
//! Eight of those is 4.5 s of the 55, so a batched prefill is worth at most 8%
//! of the phase it would restructure. The rest is arithmetic over `b * n`
//! tokens and a batch does not make it cheaper.
//!
//! ## The contamination test, which is the one this can fail
//!
//! Re-run on the fill gate, whole stack, 512-token prompts, 20 decode passes:
//! slot 0 beside seven different chunks and slot 0 beside seven copies of
//! itself emit the identical 20-token stream; the heterogeneous run's eight
//! slots emit eight DISTINCT continuations; the homogeneous run emits eight
//! identical tokens on every step. `slots_stay_independent_on_a_global_layer`
//! reads 0e0 and `..._on_a_local_layer` 1.15356445e-2, both unmoved. And the
//! schedule change is visible nowhere in the output: at `INK_SLOTS=32` and 48,
//! the old lane, `INK_MOE_NREP=1` and the gate emit token-for-token identical
//! streams over every pass and every slot.
//!
//! Batch contamination -- slot `s` reading slot `s'`'s keys -- produces fluent
//! text and no error, so it has to be looked for deliberately, and it cannot be
//! looked for by comparing `b = 8` against `b = 1`: those already disagree on
//! this model because `gemv plane par` requires `m == 1` and a wider pass loses
//! it. So the test holds `b` FIXED at 8 and changes only the neighbours.
//!
//! Whole stack, 512-token prompts, 20 decode passes:
//!
//! * slot 0 beside seven DIFFERENT chunks and slot 0 beside seven copies of
//!   itself emit the **identical** 20-token stream, token for token;
//! * the eight slots in the first of those emit eight distinct continuations,
//!   so the agreement is not a batch that collapsed into one sequence;
//! * the second emits eight identical tokens on every step, which is the
//!   symmetry the batch has to have and does not get for free;
//! * held to one GEMM lane (`INK_GEMM='double cyclic mma'`), `b = 8` with eight
//!   identical slots reproduces `b = 1` on the layer-RMS ladder -- 0.0000 on ten
//!   of twelve passes and one unit in the last printed decimal on the other two
//!   -- and the two token streams are identical.
//!
//! At the layer level the same question is asked against the UNCACHED lane:
//! eight slots carrying eight different sequences, each held to its own
//! whole-sequence run, is BIT-IDENTICAL on a global layer and 1.15e-2 on a local
//! one, which is the TF32-against-banded-prefill gap the existing cached-local
//! test already sits at (`slots_stay_independent_on_a_*_layer` in `burn.rs`).
//!
//! # `INK_COHORTS=c`: c cohorts, offset so neither node is half idle
//!
//! The largest number in the profile above is not a kernel. It is the 48% of
//! the wall on which a node's GPU has nothing to do, and the section before it
//! is the argument that batching cannot fill that: the two halves of one token
//! are strictly ordered, so widening scales both and leaves the ratio where it
//! was. Two cohorts can fill it. The slots are already independent -- that is
//! what `INK_SLOTS` establishes -- so nothing stops cohort B's layers 0:21
//! running on the head while cohort A's 21:42 run on the tail. A round is `c`
//! passes, advances every cohort by one token, and costs `c * max(H, T)` where
//! the same total width in one cohort costs `H' + T'` at `c` times the per-node
//! batch.
//!
//! ## It is a queue and an index
//!
//! The head pushes the cohort it just sent onto a `VecDeque` and then reads the
//! answer to the OLDEST outstanding one rather than to the one it just sent,
//! leaving `c - 1` in flight. That is the whole interleave. Around it:
//! `slots_dev` grows a leading cohort axis so `slots_dev[c][layer]` is one
//! cohort's [`dev_lane::SlotCache`] and the two indices are never mixed; the
//! answer is committed to the cohort it belongs to and not to the one this pass
//! computed; and the wire grows one `u64`.
//!
//! That `u64` is the cohort id and it is not redundant. Both ends derive the
//! cohort from the same step counter and would normally agree by arithmetic; a
//! pair that ever disagreed would run one cohort's residual against another
//! cohort's keys, which is fluent text and no error at all. Eight bytes a pass
//! buys a loud failure instead of a silent one.
//!
//! No collective, no resharding, no new kernel, and no change to any lane's
//! arithmetic: a cohort's pass is byte-for-byte the pass it would have run
//! alone, which the gate below checks rather than assumes.
//!
//! The PREFILLS stay strict whatever `c` is. A prefill's answer seeds the slot
//! it just filled and there is nothing to overlap it with, so the only thing
//! `INK_COHORTS` changes before the first decoded token is that there are
//! `c * b` prefills instead of `b`.
//!
//! ## What it costs and what it buys
//!
//! Warm p50 over 40 passes, `/tmp/mix_891k.ids` (891,510 tokens of ordinary
//! English prose), 3732-token prompts, two-node pipe over the direct link,
//! layers 0:21 and 21:42, `INK_KV=1`. Every row is compared against the
//! SINGLE-cohort arm of the same TOTAL slot count, because that is the arm it
//! replaces -- `2 x 32` and `1 x 64` hold sixty-four sequences either way.
//!
//! ```text
//!   total   ONE cohort           TWO cohorts                        gain
//!   slots   p50 ms  agg tok/s    b    p50 ms       agg tok/s
//!     32     585.7     54.6      16   200.8 203.3  79.7  78.7      1.46x
//!     64     866.3     73.9      32   289.9 290.3  110.4 110.2     1.49x
//!     96    1081.4     88.8      48   371.8        129.1           1.45x
//! ```
//!
//! **1.46x, 1.49x and 1.45x, against predictions of 1.46, 1.48 and 1.45.** The
//! prediction was not fitted to anything: it is `p50(2c) / p50(c)` read off the
//! single-cohort table above, which is the batch's own sublinearity, and that
//! is precisely what the interleave converts into throughput. It follows that
//! the gain is bounded by how sublinear the batch is at that width and by
//! nothing else -- the ratio sits between 1.38 and 1.50 everywhere the
//! single-cohort table has been measured, so there is no width at which this is
//! worth much more or much less than a half.
//!
//! **129.1 aggregate tok/s** is where this lane peaks on the wide cache, at
//! ninety-six sequences of a 3732-token context each. The narrow cache carries
//! it to 162.1 at a hundred and twenty-eight; see the section after next.
//!
//! Reproducibility below the memory wall: the two `2 x 32` runs read 289.9 and
//! 290.3 ms, a 0.1% spread; the two `2 x 16` runs 200.8 and 203.3, 1.2%.
//!
//! ## The idle half is two thirds gone, and what is left is not idle
//!
//! The head's own accounting, `2 x 32` against `1 x 64`: **computing 83.4%,
//! blocked 16.6%**, against 49.1% and 50.9%. At `2 x 48` against `1 x 96` it is
//! 82.2 / 17.8 against 47.2 / 52.8. Writing to the wire is 6.7 ms over forty
//! passes at 32 and 10.0 ms at 48 -- a fifth of a millisecond a pass -- so
//! nothing here is the socket, and the head never blocks in `send_stream`: a
//! 48-row residual is 768 KB and the kernel buffers it while the tail is busy.
//! That was the one way this could have failed quietly, by serialising the
//! pipeline on a socket write and still producing correct text.
//!
//! The residual 16.6% is not structural idle. It is JITTER between two stages
//! that are within 1% of each other: the head's pass p50 is 289.9 ms and the
//! tail's 287.7, so whichever end happens to be slower on a given pass makes
//! the other wait, and over forty passes that averages to a real 60-odd ms. A
//! deeper queue would absorb it -- with `c - 1` answers outstanding the head can
//! run ahead -- and that is the one thing a third cohort would buy.
//!
//! ## Depth past two, and why it is not measured here
//!
//! It cannot pay, and the argument is short enough to make instead of running
//! it. A round costs `max(H, T)` a pass at ANY depth above one -- the second
//! cohort is what stops a node waiting, and a third finds nothing left to fill.
//! What a third cohort does take is WIDTH: at a fixed slot budget it makes each
//! cohort smaller, and width is where the sublinearity that the interleave
//! converts actually lives. `3 x 16` and `2 x 24` are both forty-eight
//! sequences; the first emits 16 tokens on a pass the measured table puts near
//! 201 ms and the second 24 on one near 248, which is the same machine and a
//! fifth less throughput, plus one more cohort of caches to hold on the
//! resource that is already the ceiling.
//!
//! The one thing a deeper queue WOULD buy is the jitter above -- with two
//! answers outstanding the head never waits for the slower end. That is worth
//! at most the 17% and it costs width, so it is a trade to make only if the two
//! halves are ever left badly unbalanced.
//!
//! ## Memory: the same caches, and the same wall
//!
//! Two cohorts of `b` hold exactly what one cohort of `2b` holds -- `2b` caches
//! -- and the measurement says so to the hundredth of a gibibyte: `pool live`
//! is 13.85 GiB at `2 x 32` and 13.85 GiB at `1 x 64`, and 19.88 GiB at both
//! `2 x 48` and `1 x 96`. **The interleave buys throughput and it does not buy
//! headroom**, which matters because host RAM is what this part runs out of.
//!
//! Below the wall the reservation differs slightly in the interleave's favour:
//! at 64 slots 15.98 GiB reserved against 17.40, so 2.13 GiB stranded over 608
//! slices against 3.55 over 440, and `MemAvailable` on the head bottoms at
//! 10.42 GiB against 8.27.
//!
//! At 96 it does not. Both arms meet the SAME bound, and it is the host's RAM
//! and not the device pool or the arithmetic: `2 x 48` reserves 28.68 GiB to
//! hold 19.88, `MemAvailable` bottoms at **0.43 GiB** and the head swaps
//! 4.25 GiB, against 1.38 GiB and 5.50 for `1 x 96`. What it costs is the
//! PREFILL and not the decode: prefills that run at 7 s a slot while only
//! cohort 0's batch exists go to 45 s a slot from the moment cohort 1's batch
//! is allocated -- forty-two minutes for the phase -- while the decode that
//! follows holds 354-372 ms over its forty passes and reads its predicted rate.
//! That is the same shape the single-cohort arm shows at 96 and for the same
//! reason: what gets swapped is reservation nothing reads.
//!
//! The 96-slot row is ONE run, not two, and it is quoted that way on purpose:
//! its arm costs forty-two minutes of prefill under swap, and repeating it
//! would price the allocator rather than the interleave. The two arms below the
//! wall are two runs each and agree to 0.1% and 1.2%.
//!
//! ## The gate, and the one thing this could have got wrong
//!
//! Two cohorts sharing one pipe is exactly the shape that could cross-
//! contaminate, and cross-contamination is fluent text and no error. So the
//! check holds the cohort's own contents FIXED and changes only whether a
//! second cohort is in the pipe beside it. Whole stack, eight slots of
//! 512-token prompts, 20 decode passes:
//!
//! * cohort 0 of a `2 x 8` run and the same eight slots as a `1 x 8` run emit
//!   the **identical** 20-token stream, token for token, on every slot;
//! * the eight slots of that cohort emit eight DISTINCT continuations, so the
//!   agreement is not a batch that collapsed into one sequence;
//! * with cohort 0 homogeneous -- slot 0 beside seven copies of itself, and a
//!   heterogeneous cohort 1 in the pipe beside it -- all eight slots emit the
//!   same token on every step, and it is the same token slot 0 emits in the
//!   heterogeneous run;
//! * `slots_stay_independent_on_a_global_layer` and `..._on_a_local_layer` are
//!   unmoved at 0e0 and 1.15e-2.
//!
//! And at `INK_COHORTS=1` the transcript is character for character the one
//! every arm above was measured on: the cohort tag is only printed when there
//! is more than one, and the same eight slots on the pre-interleave binary and
//! on this one emit identical streams.
//!
//! # The KV cache is BF16, and `INK_ATTN_BF16=0` is the wide arm
//!
//! The attention matmuls are 14.1% of a decode pass at 152 GB/s against a
//! measured 236 GB/s bus, so they are bandwidth-bound, and what they read is
//! almost entirely the KV cache: 6.0 GB a pass at 32 slots against 5.4 GB of
//! slot KV. Halving the width of the cache halves what the lane moves. It also
//! halves what the run HOLDS, and on this part that is the ceiling.
//!
//! **Every table above this section was measured on the WIDE cache**, which is
//! `INK_ATTN_BF16=0` now and was the only lane when they were taken. They are
//! left as they were rather than re-measured: what they are about is the batch
//! and the pipe, and re-running forty arms to move each of them a few percent
//! would price the cache twice and the interleave not at all. The row-by-row
//! effect of the narrow cache is the table at the end of this section.
//!
//! Held narrow, not cast per pass -- casting would ADD a full-width read to
//! save a half-width one. So [`dev_lane::SlotCache`] and [`dev_lane::AttnCache`]
//! allocate, append, merge and trim in BF16, and `q @ k^T` and `p @ v` run
//! there; the scores come back to f32 before the relative bias, the mask and
//! the softmax, which are untouched. `[slots * kv_heads, groups, cap]` is 16 MB
//! at 32 slots against gigabytes of cache, so keeping the reduction's OUTPUT
//! wide costs nothing. A PREFILL computes its scores from keys it has just
//! projected and never round-trips them, so it has nothing to save and is left
//! alone.
//!
//! ## The precision argument is not the one it looks like
//!
//! Burn's f32 matmul on this runtime is TF32 -- about ten mantissa bits, not
//! twenty-three, pinned by `f32_matmul_is_tf32_on_this_runtime` at 9.3e-4
//! relative on a 128-deep product. The four-byte load was paying for
//! twenty-three bits that the tensor cores round away before they multiply.
//! BF16 carries eight, which is genuinely fewer.
//!
//! **How much fewer is not a question an f64 reference can answer**, and this
//! file no longer contains one to ask. The weights are four bits; an f64 sum of
//! the same values is a more expensive computation of the model, not the
//! model's ground truth, and the reference implementation this port is held to
//! does not compute attention that way either. So the acceptance criterion is
//! `golden/paired/` -- the same prompts through the BF16 reference and through
//! each arm, reported as agreement rather than as distance.
//!
//! A layer-RMS ladder is still worth a glance, as a smoke check and not as a
//! bound: on the FIRST decode pass, where both arms are on identical inputs,
//! the ladder moves 5.3e-3 across the head's 21 layers and 2.5e-2 across the
//! tail's -- the same order as `CACHE_TOLERANCE_LOCAL`, which is what the f32
//! cached lane already sits at against the uncached one. Later passes move far
//! more (30% by the twentieth) and that number means nothing: by then the two
//! arms are decoding different text.
//!
//! Token agreement means little here for the same reason. Held to eight slots
//! of 512-token prompts, the two arms agree on 51% of 160 slot-steps and
//! diverge from the first cached step, which is what greedy decoding does the
//! moment any position flips -- the same lane already shows `INK_WIDTH=1` and
//! `INK_WIDTH=2` taking the stack apart over forty positions with no
//! speculation anywhere. That is why the gate is capability and not agreement.
//!
//! ## The gate: `golden/paired/`, and it does not move
//!
//! Eight generation prompts, 48 tokens each, decoded through the pipe with the
//! cache on; then the prompt AND that continuation through the unquantised BF16
//! reference as one sequence, which is asked at every position what IT would
//! have emitted. 384 scored positions an arm. `INK_ATTN_BF16=0` and `=1` on the
//! same binary, same prompts, same reference procedure.
//!
//! ```text
//!                                                      wide cache   narrow cache
//!   reference vs the runtime's own cached generation   362/384 94.3%  363/384 94.5%
//!   reference vs the runtime, both teacher-forced      359/384 93.5%  361/384 94.0%
//!   the runtime UNCACHED vs its own cached generation  365/384 95.1%  363/384 94.5%
//! ```
//!
//! Every 95% interval on those six numbers spans four points and every pair
//! overlaps almost entirely; the largest gap between the arms is TWO positions
//! out of 384, and it points in a different direction on the second line than
//! on the third. **The narrow cache is not distinguishable from the wide one
//! here**, and the third line is the one that isolates the cache: it compares
//! each arm's cached lane against its OWN uncached forward on the same ids, so
//! the model and the reference cancel and what is left is the cache.
//!
//! Said with its limit rather than without: eight sequences is what this
//! detects, and the harness's own `--flip` calibration on the multiple-choice
//! set says a small regression is not detectable at that size. What this run
//! rules out is a regression large enough to matter at the width the change was
//! made for, and it does not rule out a small one.
//!
//! The 40-item multiple-choice set is NOT the instrument here and was not run
//! as one: those items answer with the FIRST token, which is a prefill, and a
//! prefill reads no cache -- it would have scored a run in which this change
//! never executed. That property is checked rather than asserted: the same
//! uncached forward over the same eight sequences, with the flag on and with it
//! off, writes byte-identical top-5 tables, all eight for all eight.
//!
//! ## What the unit tests can and cannot say about it
//!
//! The cached-lane tests in `burn.rs` assert that feeding one token at a time
//! reproduces feeding all of them, at a tolerance sized for two implementations
//! of the SAME arithmetic. A narrow cache is not that -- it stores less -- so
//! those tests take the wide lane explicitly rather than being loosened until
//! the narrow one fits, which would be inventing a bound to match the answer.
//! What the narrow lane is held to instead is the property that holds exactly
//! at any dtype and is the one batching can get wrong:
//! `narrow_slots_are_still_independent` runs slot 0 beside three different
//! sequences and beside three copies of itself and requires the two answers to
//! be **bit-identical** -- 0e0, measured -- while the four slots of the
//! heterogeneous batch stay 4.1e-1 apart.
//!
//! ## What it costs and what it buys
//!
//! Warm p50 over 40 passes, `/tmp/mix_891k.ids`, 3732-token prompts, two-node
//! pipe, layers 0:21 and 21:42, `INK_KV=1`. `pool live`, `slot KV` and the
//! `MemAvailable` floor are the head's.
//!
//! ```text
//!   slots  cohorts  lane   p50 ms        agg tok/s     pool live  slot KV  MemAvail  swap
//!     32      1     f32    585.7 585.3   54.6          7.82 GiB   5.01 GiB   15.8     0
//!     32      1     bf16   556.4 555.0   57.5  57.7    4.84       2.51       16.8     0
//!     96      1     f32   1081.4         88.8         19.88      15.04        1.4  5.50 GiB
//!     96      1     bf16   968.7         99.1         10.95       7.52       10.2     0
//!     96      2     f32    371.8        129.1         19.88      15.04        0.4  4.25 GiB
//!     96      2     bf16   344.8        139.2         10.95       7.52       12.4     0
//!    128      1     f32       --  did not run: 25.91 GiB live, prefills 56/139/72/59 s
//!    128      1     bf16  1091.5        117.3         14.00      10.03        4.4  4.63 GiB
//!    128      2     bf16   394.7        162.1         14.00      10.03        6.7     0
//! ```
//!
//! **5.4% on the pass at 32 slots and 11.6% at 96**, and the two are different
//! numbers for a reason. At 32 the saving is exactly what the bandwidth share
//! predicts: the attention products are ~13.6% of the round trip across the two
//! nodes and halving their bytes takes about half of that. At 96 the f32 arm is
//! ALSO paying for memory pressure -- 19.88 GiB live, `MemAvailable` at
//! 1.4 GiB, 5.50 GiB of the head swapped -- and the BF16 arm is not, because
//! 7.52 GiB of slot KV against 15.04 is what takes the run off the wall.
//!
//! The cache is 50% narrower and the POOL is 38-45% smaller, which is more than
//! the cache alone: the pass's transient score and probability tensors follow
//! the dtype of what they are built from.
//!
//! **With the interleave, ninety-six sequences decode at 139.2 aggregate tok/s
//! against 88.8 before either change** -- 1.57x, at a 3732-token context each,
//! on a pair of nodes where the single-cohort f32 arm was swapping.
//!
//! ## What this does to the ceiling, which was the point
//!
//! The last section's ceiling was host RAM and nothing else. `b = 128` was over
//! it at 25.91 GiB live, and that arm was stopped after five of its hundred and
//! twenty-eight prefills -- 56, 139, 72 and 59 seconds each -- rather than left
//! two hours to say so a sixth time. `b = 96` ran but swapped 5.50 GiB, and the
//! two-cohort arm at the same width bottomed at 0.43 GiB of `MemAvailable`.
//!
//! Narrow, **`b = 128` runs**: 14.00 GiB live against 25.91, prefills at their
//! floor, and 117.3 aggregate tok/s single-cohort. Interleaved it is
//! **162.1 aggregate tok/s** -- and the interleaved arm is the one that stays
//! OFF swap at that width, because two 64-slot batches reserve 15.65 GiB to hold
//! 14.00 where one 128-slot batch reserves 17.61, and 1.96 GiB of stranded
//! reservation is the whole margin at this point on the curve.
//!
//! So the two changes compose the way they were argued to: **88.8 aggregate
//! tok/s before either, 162.1 with both**, at a 3732-token context on every one
//! of a hundred and twenty-eight sequences. The interleave stops each node
//! waiting; the narrow cache is what buys the width to wait for.
//!
//! # One lane, all the way down
//!
//! There is nothing to select. Attention, the dense and shared MLPs, the head
//! and the routed experts are all on the device, always; residency is
//! unconditional; the routed experts go through the NVFP4 tensor cores as the
//! NVFP4 they are stored as. `INK_GPU`, `INK_ATTN`, `INK_DENSE`, `INK_HEAD`,
//! `INK_EXPERTS`, `INK_RESIDENT`, `INK_DEQUANT`, `INK_ZEROCOPY`, `INK_WARM` and
//! `INK_HOST_SUM` all named a second, slower, host-or-widening implementation
//! of something, and each of them is gone with it. A lane nobody selects is a
//! lane nobody tests; a host lane you CAN select is one you will run by
//! accident, which is how a 401 s forward happened.
//!
//! The host lane is now gone from the LIBRARY as well, not just from the
//! switches here. `mlp::routed_experts`, `mlp::shared_experts`, `mlp::moe` and
//! `mlp::expert_ffn_one` were unreachable from this file and still cost real
//! work: they were the readable transcription of the routed algorithm, so an
//! agent asked to make the experts faster found THEM, and twice in one day
//! someone analysed how to parallelise a function no forward calls. Legibility
//! attracts effort, which means the legible version has to be the live one.
//! [`routed_experts_fp4`]'s doc comment is where that statement of the
//! algorithm lives now. What survives on the host is `mlp::dense_mlp` and
//! `layer::decoder_layer`, and neither is a fallback: they are the only
//! implementation the MTP heads have.
//!
//! Mixed precision is not a second lane either. Layer 2's experts are BF16 and
//! the other 41 layers' are NVFP4, so the routed block picks the instruction the
//! stored format calls for — `mma.sync…bf16` or the block-scaled
//! `…mxf4nvf4.block_scale` — and nothing else changes. Both accumulate in f32,
//! which is the MMA's own output type and not a widening; widening the BF16
//! weight to f32 to reuse the f32 path is the exact thing this file does not do.
//!
//! # Where the weights come from
//!
//! The pile, positionally, through its sole native model collection. All of it:
//! the weights AND `config.json` AND the chat template, so a run reads nothing
//! off a checkpoint directory and there is no path it could.
//!
//! It used to be the other way round — a checkpoint directory positionally,
//! with `INK_PILE=<path>` as an opt-in swap — and by the end that argument was
//! vestigial theatre: the last runs passed `--` for it because the pile
//! answered for everything. An opt-in that every real invocation opts into is
//! a default written backwards, and a source that CAN be a directory of shards
//! is one that will be by accident. `mary` already made this move for qwen3tts
//! (8475167) and for the runtime at large (906cbc9, "safetensors gated behind
//! `import`, runtime is pile-only"); Inkling is the last holdout.
//!
//!   cargo run --release --features inkling-cuda,cuda-backend,import \
//!       --bin inkling_forward -- <pile> <ids.bin> <out.bin>

use std::collections::BTreeMap;
use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use mary::models::inkling::attn::{AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::bf16gemm::Bf16W;
use mary::models::inkling::block::{Routing, rms_norm, route_from_logits};
use mary::models::inkling::budget;
use mary::models::inkling::config::{AttnKind, InklingConfig};
use mary::models::inkling::fatal;
use mary::models::inkling::layer::{LayerMlp, LayerWeights};
use mary::models::inkling::load::{Held, split_gate_up};
use mary::models::inkling::mtp::{
    Concat as MtpConcat, MtpCache, MtpHead, mtp_block, mtp_block_prefill, mtp_block_step,
};
use mary::models::inkling::pile::Elem;
use mary::models::inkling::source::Weights;
use mary::models::inkling::spectree::{self, TreeSpec};
use mary::models::inkling::stack::{embed_and_norm_bf16, embed_row_bf16};
use mary::models::inkling::stepstat;

/// One gibibyte, as the divisor every byte count here is printed against.
const GIB: f64 = (1u64 << 30) as f64;

/// How many decode steps are COLD, and excluded from the warm rate.
///
/// Every kernel shape a decode step reaches is compiled on first use, and the
/// first two steps of a pipe run pay 4.54 s and 0.55 s of that against a 127 ms
/// median. Two is not a fit: it is how many steps report a pass an order of
/// magnitude off the median, and the third is already within 6% of it.
const COLD_DECODE_STEPS: usize = 2;

/// How long each end of an `INK_PIPE` waits for the other at the rendezvous.
///
/// The wire is opened AFTER the weights, and building the index takes about a
/// minute, so whichever end arrives first is talking to a process that is not
/// listening yet. Without a wait that is a RANK-ORDER HAZARD: the head's
/// `connect` got `ECONNREFUSED` and the run died, so the two commands had to be
/// started in one specific order and far enough apart. The launch scripts hide
/// it by polling for the tail's `pipe: listening` line; a hand launch does not,
/// and the reference implementation carries the same rule in prose ("worker
/// rank 1 must start BEFORE head rank 0, or rank 0 exits before the
/// rendezvous"). A minute of slack costs nothing and removes the ordering
/// constraint entirely.
///
/// It is a BOUND and not an infinite wait. A wrong host, a wrong port or a
/// firewall are misconfigurations and must still fail — the fix here is that
/// they fail legibly, naming which end was waiting and for how long, instead of
/// a bare "Connection refused" on one side and a process that never returns on
/// the other. `INK_PIPE_WAIT=<seconds>` overrides it.
fn pipe_wait() -> Duration {
    Duration::from_secs(
        std::env::var("INK_PIPE_WAIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(180),
    )
}

/// Connect to the tail, retrying while it is still loading its weights.
///
/// Backs off 50 ms doubling to 2 s, so an immediate rendezvous costs one
/// syscall and a slow one costs a handful of attempts a minute rather than a
/// spin. Only the errors that mean "nobody is listening THERE YET" are retried;
/// a name that does not resolve or an address that cannot be used fails on the
/// first try, because waiting three minutes to repeat a typo helps nobody.
fn pipe_connect(addr: &str, wait: Duration) -> Result<TcpStream> {
    let t0 = Instant::now();
    let mut backoff = Duration::from_millis(50);
    let mut tries = 0usize;
    loop {
        tries += 1;
        let err = match TcpStream::connect(addr) {
            Ok(sock) => return Ok(sock),
            Err(e) => e,
        };
        let retry = matches!(
            err.kind(),
            std::io::ErrorKind::ConnectionRefused
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::TimedOut
        );
        let waited = t0.elapsed();
        if !retry {
            return Err(err).with_context(|| format!("connecting to the tail at {addr}"));
        }
        if waited >= wait {
            anyhow::bail!(
                "the head waited {:.0}s ({tries} attempts) for the tail to listen on {addr} \
                 and it never did: {err}. Start the tail too \
                 (INK_LAYERS=<lo>:<hi> INK_PIPE=tail:0.0.0.0:<port>), check the host and port \
                 match, or raise INK_PIPE_WAIT (currently {}s).",
                waited.as_secs_f32(),
                wait.as_secs()
            );
        }
        std::thread::sleep(backoff.min(wait - waited));
        backoff = (backoff * 2).min(Duration::from_secs(2));
    }
}

/// Accept the head, giving up rather than wedging if it never arrives.
///
/// `TcpListener::accept` has no deadline of its own, so the listener is put in
/// non-blocking mode and polled. Without this a head that crashed while loading
/// its weights left the tail sitting on a GPU forever, which on a shared box is
/// worse than the failure it was waiting through.
fn pipe_accept(
    l: &TcpListener,
    addr: &str,
    wait: Duration,
) -> Result<(TcpStream, std::net::SocketAddr)> {
    let t0 = Instant::now();
    l.set_nonblocking(true)
        .with_context(|| format!("polling for the head on {addr}"))?;
    loop {
        match l.accept() {
            Ok((sock, peer)) => {
                // Explicitly, and not by trusting inheritance: the accepted
                // socket's blocking mode is a platform detail and every read
                // and write after this assumes it blocks.
                sock.set_nonblocking(false)
                    .with_context(|| format!("accepting the head on {addr}"))?;
                l.set_nonblocking(false)
                    .with_context(|| format!("accepting the head on {addr}"))?;
                return Ok((sock, peer));
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                let waited = t0.elapsed();
                if waited >= wait {
                    anyhow::bail!(
                        "the tail listened on {addr} for {:.0}s and the head never connected. \
                         Start the head too \
                         (INK_LAYERS=<lo>:<hi> INK_PIPE=head:<this-host>:<port>), check it is \
                         pointed at this host and port, or raise INK_PIPE_WAIT (currently {}s).",
                        waited.as_secs_f32(),
                        wait.as_secs()
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(e).with_context(|| format!("accepting the head on {addr}")),
        }
    }
}

/// Bytes this process has actually pulled off the block device.
///
/// NOT the same as bytes read: a page-cache hit costs nothing here. That is
/// precisely what makes it the right instrument for residency — a working set
/// that is genuinely held reads ~0 on the second pass, and one that is merely
/// "probably still in the page cache" does not.
fn io_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("read_bytes:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

/// A 95% Wilson score interval for `hits` of `n`.
///
/// Wilson rather than the textbook normal interval, because this measurement
/// produces rates at both ends — depth 1 near 3/4, depth 4 at or near 0 — and
/// there the normal interval runs off the end of [0, 1] and reports a bound that
/// is not a bound. It is printed beside every rate so the counts can be read as
/// evidence: 15/20 and 150/200 are the same percentage and not the same claim.
fn wilson95(hits: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.959_963_984_540_054_f64;
    let (h, n) = (hits as f64, n as f64);
    let p = h / n;
    let d = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / d;
    let half = z * ((p * (1.0 - p) / n) + z * z / (4.0 * n * n)).sqrt() / d;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// One logit row as a probability distribution.
///
/// The exponentials are accumulated in f64 over the max-shifted row: the row is
/// 200 058 wide and the summands span the whole dynamic range, so an f32 sum
/// loses the tail — and the tail is exactly what an acceptance rule reads.
fn softmax_row(row: &[f32]) -> Vec<f32> {
    let mut m = f32::NEG_INFINITY;
    for &x in row {
        if x > m {
            m = x;
        }
    }
    let mut out = Vec::with_capacity(row.len());
    let mut sum = 0f64;
    for &x in row {
        let e = ((x - m) as f64).exp();
        sum += e;
        out.push(e);
    }
    out.iter().map(|e| (e / sum) as f32).collect()
}

/// Which end of a two-machine split this process is, if it is one.
///
/// The head owns the embedding and the token loop; the tail owns the final norm,
/// the unembedding and the argmax. Neither loads the other's tables — on a box
/// chosen because the working set only just fits, pinning 3.3 GB of unembedding
/// on the machine that will never read it is the whole problem in miniature.
enum Pipe {
    Head(TcpStream),
    Tail(TcpStream),
}

/// The residual stream, on the wire.
///
/// `pos0` travels with it because the tail's half of the stack needs it and
/// cannot derive it: with a KV cache a decode step feeds ONE token whose
/// absolute position is what log scaling and the relative bias are functions of,
/// and the tail has never seen the sequence it belongs to.
fn send_stream(s: &mut TcpStream, n: usize, pos0: usize, coh: usize, x: &[f32]) -> Result<()> {
    let mut b = Vec::with_capacity(24 + x.len() * 4);
    b.extend_from_slice(&(n as u64).to_le_bytes());
    b.extend_from_slice(&(pos0 as u64).to_le_bytes());
    b.extend_from_slice(&(coh as u64).to_le_bytes());
    for v in x {
        b.extend_from_slice(&v.to_le_bytes());
    }
    s.write_all(&b)?;
    s.flush()?;
    Ok(())
}

/// The other side of [`send_stream`]. `None` when the peer is done.
fn recv_stream(s: &mut TcpStream, h: usize) -> Result<Option<(usize, usize, usize, Vec<f32>)>> {
    let mut hdr = [0u8; 24];
    if s.read_exact(&mut hdr).is_err() {
        return Ok(None);
    }
    let n = u64::from_le_bytes(hdr[..8].try_into().unwrap()) as usize;
    let pos0 = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
    let coh = u64::from_le_bytes(hdr[16..].try_into().unwrap()) as usize;
    if n == 0 {
        return Ok(None);
    }
    let mut buf = vec![0u8; n * h * 4];
    s.read_exact(&mut buf)?;
    let x = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(Some((n, pos0, coh, x)))
}

/// The tail's answer: a length and that many token ids.
///
/// One shape for BOTH of the tail's messages, because they are the same kind of
/// thing — a short list of token ids — and a second wire format would be a
/// second thing to get out of step. Message one is what the verify pass
/// CONFIRMED (the accepted prefix plus the token past it, so never empty);
/// message two is what the MTP heads drafted for the next pass (empty when the
/// run is not speculating). They are sent separately on purpose: the head can
/// commit its caches on the first while the tail is still computing the second.
fn send_toks(s: &mut TcpStream, toks: &[usize]) -> Result<()> {
    let mut b = Vec::with_capacity(8 + toks.len() * 8);
    b.extend_from_slice(&(toks.len() as u64).to_le_bytes());
    for &tk in toks {
        b.extend_from_slice(&(tk as i64).to_le_bytes());
    }
    s.write_all(&b)?;
    s.flush()?;
    Ok(())
}

/// The other side of [`send_toks`].
fn recv_toks(s: &mut TcpStream) -> Result<Vec<usize>> {
    let mut hdr = [0u8; 8];
    s.read_exact(&mut hdr).context("the tail closed mid-step")?;
    let n = u64::from_le_bytes(hdr) as usize;
    let mut buf = vec![0u8; n * 8];
    s.read_exact(&mut buf)
        .context("the tail closed mid-answer")?;
    Ok(buf
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect())
}

/// The backend the device lane runs on.
type Bk = burn::backend::Cuda<f32>;
use burn::prelude::Backend;
use burn::tensor::{Tensor as BT, TensorData as BTD};
use mary::models::inkling::burn as dev_lane;
use mary::models::inkling::resid as dev_lane_resid;

/// Move a host `[rows, cols]` matrix to the device, consuming it.
///
/// Takes the `Vec` by value on purpose: the dense `w13` is 537 MB at f32 and a
/// borrowing helper would hold two copies of it at once.
fn up2<B: Backend>(v: Vec<f32>, rows: usize, cols: usize, dev: &B::Device) -> BT<B, 2> {
    assert_eq!(
        v.len(),
        rows * cols,
        "{} values are not [{rows}, {cols}]",
        v.len()
    );
    BT::from_data(BTD::new(v, [rows, cols]), dev)
}

fn up1r<B: Backend>(v: &[f32], len: usize, dev: &B::Device) -> BT<B, 1> {
    assert_eq!(v.len(), len, "{} values are not [{len}]", v.len());
    BT::from_data(BTD::new(v.to_vec(), [len]), dev)
}

fn up1<B: Backend>(v: Vec<f32>, len: usize, dev: &B::Device) -> BT<B, 1> {
    assert_eq!(v.len(), len, "{} values are not [{len}]", v.len());
    BT::from_data(BTD::new(v, [len]), dev)
}

/// Read a `[rows, cols]` device tensor back to the host. This is also the sync,
/// so a timer around the call measures work rather than enqueueing.
fn down<B: Backend>(t: BT<B, 2>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("device readback")
}

/// A device tensor of this run's backend, named once so the residency types
/// below do not have to repeat it.
type T2 = burn::tensor::Tensor<Bk, 2>;

/// The index of the largest element of a `[1, cols]` device tensor, reduced
/// where the data already is.
///
/// The twin of [`down`] for the one case that wants a single integer and not a
/// row. A draft head's unembedding produces 201024 f32 and the caller keeps
/// exactly one index off it; reading the row back to find that index cost 804
/// KB over the bus per DRAFT DEPTH, plus a 201k-iteration host loop, to deliver
/// eight bytes.
///
/// Bit-identical to the loop it replaces, ties included: `cubek`'s `ArgMax`
/// documents "the smallest coordinate in case of equality", which is what
/// `val > dl[b]` selected as well.
fn argmax_row_dev(row: T2) -> usize {
    let [rows, _] = row.dims();
    debug_assert_eq!(rows, 1, "argmax_row_dev reads exactly one row");
    row.argmax(1)
        .into_data()
        .iter::<i64>()
        .next()
        .expect("device argmax readback") as usize
}

/// The five largest elements of a logit row, by value, ties broken by the
/// SMALLER index.
///
/// # Why this is not a sort
///
/// It was one: `(0..v).collect()` then `sort_unstable_by` over the whole row.
/// That is a 1.6 MB index allocation and ~3.5 M comparisons across a 200058-wide
/// f32 row — per CONFIRMED ROW, per pass — to read five numbers, and it is the
/// REPORT's, not the model's. Measured on the two-node tail (spark, layers
/// 21..42, `INK_SPEC=0`, `INK_KV=1`, ctx ~3732, 58 warm passes, one row per
/// pass): the gap between `whole_pass` — sampled before the report — and the
/// `pass_ms` the harness records was 3.20 ms p50 (min 3.10, max 3.30) on a 51.5
/// ms p50 pass, 6.2% of it. On the HEAD the same gap is 0.10 ms, and the head
/// is the only difference: it has no logits, so it never ranks a row. The sort
/// was the whole of it.
///
/// A five-slot insertion sweep is one pass over the row, allocates nothing but
/// the five-element answer, and skips on a single failed compare for all but a
/// handful of elements.
///
/// # What it costs the pipeline right now, which is nothing
///
/// Said plainly so nobody quotes this as a step-time win: at the config above
/// the tail replies to its peer BEFORE it prints, and the head then spends ~40
/// ms in its own layer loop, so the tail's 3.20 ms was already hidden. This
/// removes a real per-pass cost from the tail's budget; it does not move the
/// two-node ms/step, and a measurement that claims it did is measuring drift.
///
/// # Ties
///
/// `sort_unstable` does not say where equal keys land, so a flat row could print
/// a different top-5 ORDER on two runs of the same binary — on a model this file
/// already documents as disagreeing with itself on 8.55% of argmax positions,
/// that is one more source nobody needs. The rule here is the argmax's own:
/// strictly greater wins, so an equal value never displaces an earlier index.
/// Non-finite values are skipped by the same `>` that the host argmax skips them
/// with, rather than panicking in a `partial_cmp().unwrap()` inside a report.
fn top5_desc(row: &[f32]) -> Vec<usize> {
    let k = 5.min(row.len());
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, &val) in row.iter().enumerate() {
        // A NaN is not ranked, and it is dropped HERE rather than compared
        // away. `partition_point` below needs `!(val > v)` to hold on a prefix
        // and fail after it; one NaN admitted into `best` makes that predicate
        // true again past the split and the binary search returns a position
        // that means nothing. Caught by `a_nan_does_not_kill_the_report`, which
        // is the whole reason that test exists.
        if val.is_nan() {
            continue;
        }
        // The overwhelmingly common case: a value that cannot displace the
        // fifth. One comparison and nothing else.
        if best.len() == k && !(val > best[k - 1].0) {
            continue;
        }
        // `best` is sorted descending, so `!(val > v)` holds on a prefix and
        // `at` is the first slot `val` outranks. Inserting there leaves an
        // equal value already present ahead of this one, which is the
        // smaller-index rule.
        let at = best.partition_point(|&(v, _)| !(val > v));
        best.insert(at, (val, i));
        best.truncate(k);
    }
    best.into_iter().map(|(_, i)| i).collect()
}

/// Host seconds inside the routed-expert lane, split by WHAT THE HOST DID.
///
/// One bucket used to cover binding, quantising, four enqueues and the layer's
/// blocking read, and it was called "upload"; a profiling session went looking
/// for a transfer that turned out to be 4% of it. So each field names one kind
/// of work and the sync has a field of its own, because a bucket that mixes
/// "issued a kernel" with "waited for the GPU" cannot answer whether the host
/// is the bottleneck.
#[derive(Default, Clone, Copy)]
struct HostT {
    /// `expert_packed` / `expert_bf16`: a hash lookup and a view of the pile.
    slice: f64,
    /// Copying this expert's tokens out of the residual stream, on the host.
    gather: f64,
    /// Binding the weight, quantising the activation, issuing the kernels.
    /// Non-blocking by construction.
    enqueue: f64,
    /// `read_one`: BLOCKING, so this is the layer's device time as much as the
    /// host's.
    drain: f64,
    /// Scattering each expert's rows back into the accumulator, weighted.
    accum: f64,
    /// Layers the GROUPED lane took: one launch per stage for the whole layer.
    grouped: usize,
    /// Layers that fell back to the per-expert loop, because their weights are
    /// not offsets into one registered mapping.
    per_expert: usize,
    /// Summed over MoE layers: the number of DISTINCT experts this pass had to
    /// gather. One token needs `top_k` routed plus the shared ones. A WIDER pass
    /// needs the UNION of its tokens' expert sets, and that union is why
    /// speculation costs real bytes here rather than being nearly free as it is
    /// on a dense model, where verifying k+1 tokens re-reads exactly the same
    /// weights. Divided by the layer count it reads as "distinct experts per MoE
    /// layer", which is the quantity that decides whether a wider verify pass
    /// pays for itself.
    expert_slots: usize,
    /// The grouped lane's small plan uploads that DEPEND on the routing
    /// decision: the two offset tables, the two second-level scale vectors and
    /// the per-row weights. A device-resident router deletes exactly these --
    /// they become a gather out of a per-layer `[n_routed]` table by ids the
    /// device already holds -- so they are timed apart from the ones it does
    /// not.
    plan_up_routed: f64,
    /// The grouped lane's small plan uploads that are a function of `n` and
    /// `top_k` ALONE: the row->token map, the three block-plan vectors and the
    /// token->rows table. At `n == 1` every one of them is the same bytes on
    /// every layer of every pass, so this is what a hoist out of the loop is
    /// worth, independently of where the routing decision lives.
    plan_up_static: f64,
    /// Layer-passes whose row plan was built ON THE DEVICE, from a decision
    /// that was never read back. Counted because every other counter in this
    /// struct is the same on both arms -- `grouped` and `expert_loads` cannot
    /// tell them apart, and a lane you cannot count is a lane you cannot claim.
    plan_dev: usize,
    /// Layer-passes whose row plan came off a blocking read. The complement,
    /// and printed beside it: a run where this is not zero has a sync left in
    /// the loop, and WHICH layer it is matters more than how many.
    plan_host: usize,
}

/// One layer's shared experts, on the device, as the BF16 the pile stores.
///
/// `gate_up` is ONE weight, `[2 * n_shared * inter, hidden]`, holding every
/// shared expert's gate block followed by every up block. It used to be four
/// `Bf16W` — a gate and an up per expert — and four separate GEMMs against the
/// same activation, which is four launches and, more to the point, four grids
/// of 256 cubes.
///
/// The `m16n8k16` kernel gives one warp each `(m_tile, n_tile)`, so with M
/// padded to one tile the grid IS `n / 8` and nothing else. 256 cubes of one
/// warp cannot cover DRAM latency on this part: measured, those calls ran at
/// 79 GB/s while the unembed — same kernel, same instruction, 25128 cubes —
/// ran at 175. Concatenating along `n` is the one axis that adds cubes without
/// touching the arithmetic: each warp still computes the same output tile by
/// the same k-loop, so this is a scheduling change and not a numerical one.
struct SharedOnDevice {
    gate_up: dev_lane::ProjW,
    down: Vec<dev_lane::ProjW>,
}

/// Shortlist rows the approximate head rescores exactly when `INK_ANN_HEAD` is
/// not set.
///
/// # Why 8192, and what was measured to pick it
///
/// Recall against the EXACT head, on the hidden states a real prompt produced,
/// 65 decode steps per arm, `INK_ANN_VERIFY=1`, layers 0:4, spark-zt:
///
/// ```text
///   budget    recall@1   mean |exact - approx| at the winner
///       64      0.6615   0.5198 logits
///      256      0.8923   0.2612
///     1024      0.9231   0.2719
///     4096      0.9846   0.0616
///     8192      1.0000   0.0000
///    16384      1.0000   0.0000
/// ```
///
/// A second truncated stack agrees: layers 0:3, one prompt, 91 steps, 26
/// distinct winners, budget 1024 gives **0.9890** against 0:4's 0.9231.
///
/// The mean exact top1-top2 gap over those steps was 0.37-0.43 logits, so these
/// are genuinely tight decisions and not a distribution where anything would
/// win.
///
/// Every arm's "exact winner shortlisted" rate EQUALS its recall to four places,
/// and that equality is doing more work than it looks. This lane has exactly two
/// ways to pick the wrong token, and the equality measures BOTH at zero:
///
///  1. the winner never cleared the floor, so it was never rescored. That is
///     `1 - shortlisted`, and it accounts for every miss in the table.
///  2. the winner WAS rescored, and an unrescored row's ESTIMATE still beat its
///     exact score. The returned row is a blend, so this is possible in
///     principle. It would show as recall BELOW the shortlisted rate, and it
///     never does -- in any arm, at any budget, including the ones where a
///     third of the shortlists missed.
///
/// So the budget is the only lever, and raising it is the only fix.
///
/// And raising it is FREE, which is what makes 8192 an easy choice rather than
/// a brave one. At the head's own shape (`n = 201024`, `k = 4096`, 24 queries,
/// min of the warm launches in one process, launch + sync, GPU idle) the whole
/// lane costs 0.779 ms at budget 64 and 0.808 ms at budget 8192 — a 0.03 ms
/// spread across a 128x range, against a 0.54 ms spread on the exact arm over
/// the same four runs. The scan is the entire cost; 8192 rows of NVFP4 is 20 MB
/// of contiguous 2 KiB rows against the sketch's 103 MB, and it vanishes into
/// it. So the recall curve above can simply be bought outright.
///
/// # What is NOT measured yet
///
/// Those 65 steps ran on a FOUR-LAYER stack, because the full 42 needs both
/// Sparks and the second one's copy of the checkpoint predates the
/// branch-to-collection migration and cannot be opened. A truncated stack is a
/// real model on a real prompt, but its final hidden states are not the ones
/// the shipped model produces, and the logit geometry — how crowded the top of
/// the vocabulary is — is exactly what the budget is sized against.
///
/// **How much that matters is itself measured, and it is a lot.** The same
/// budget of 1024 that scores 0.9231 at layers 0:4 scores 0.0370 at layers 0:2.
/// Most of that collapse is the SAMPLE and not the lane — the two-layer stump
/// loops on a single token, so all 81 steps are one hidden state and the "rate"
/// is one query's luck (the report now says so out loud; see
/// [`mary::models::inkling::annhead::VERIFY_WINNERS`]). But it settles the
/// direction of the caveat: recall here is a property of the hidden-state
/// DISTRIBUTION, the 0:4 table is 65 correlated steps from one prompt, and
/// extrapolating it to a 42-layer stack is an extrapolation.
///
/// So this number is evidence and not proof. Re-running `INK_ANN_VERIFY=1` on
/// the two-node stack, on more than one prompt, is owed before anyone quotes it
/// as a property of the model.
///
/// It is nevertheless ON, because the cost of being wrong here is bounded and
/// visible: a shortlist miss produces a different-but-fluent token, the way the
/// W4A16 head this sits on already does (one token in 24, at a 0.08-logit gap),
/// and this runtime disagrees with ITSELF on 8.55% of argmax positions between
/// two runs of the same binary. `INK_ANN_HEAD=0` is the exact lane, unchanged,
/// one environment variable away.
const ANN_BUDGET_DEFAULT: usize = 8192;

/// The seed of the sketch's random rotation.
///
/// Fixed rather than drawn, because the rotation has to be the same one on
/// every process that reads the same table: it is a property of the SKETCH, and
/// a sketch built under a different rotation than the query is rotated by is not
/// a worse estimate, it is noise. It is a constant and not a switch for the same
/// reason -- there is nothing a caller could usefully vary it to.
const ANN_SKETCH_SEED: u64 = 0x414E_4E5F_5545_3031;

/// Which lane the unembed table is bound to.
///
/// The head is the single largest term in the per-step INTERCEPT, and it is
/// physics rather than overhead: `[vocab_size, hidden]` BF16 is 1.53 GiB read
/// WHOLE on every pass, measured at 10.3 ms = 159 GB/s = 65% of this box's
/// measured 242.9 GB/s. It does not scale with context or with how many layers
/// this node holds, and no launch-side change moves a byte of it -- only fewer
/// bytes do. At NVFP4 the same table is 0.43 GiB.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HeadLane {
    /// The stored BF16, aliased out of the pile's own mapping.
    Bf16,
    /// NVFP4 codes against a BF16 activation.
    W4a16,
    /// The same NVFP4 codes against a quantised activation -- the routed
    /// experts' native lane, here only so the two can be compared.
    W4a4,
}

/// The unembed table is held as NVFP4 and multiplied against a
/// BF16 activation. `INK_W4A16_HEAD=4` binds the identical codes to the W4A4
/// lane instead, which is a comparison arm and not a recommendation.
///
/// W4A16 is the default of the two: the checkpoint quantised the ROUTED
/// EXPERTS and nothing else, so there is no calibrated input quantiser for
/// this tensor, and these are the logits that top-k and sampling read
/// directly. The whole switch is off by default because quantising the head is
/// a model-quality decision, not an engineering one.
///
/// ## First measurement, 2026-08-25
///
/// Two Spark boxes, `INK_SPLIT=21`, `INK_KV=1`, `INK_GEN=24`, a 14-token
/// English prompt, tail node, p50 of 24 warm passes of the `head / unembed`
/// stage timer (device time, one sync). ALL THREE FIGURES ARE PER TAIL PASS:
///
/// | lane  | table    | kernel        | head stage | achieved  |
/// |-------|----------|---------------|-----------:|----------:|
/// | BF16  | 1.53 GiB | cubek, TUNED  |    10.1 ms | 163 GB/s  |
/// | W4A16 | 0.43 GiB | hand mma      |     6.2 ms |  74 GB/s  |
/// | W4A4  | 0.43 GiB | hand mma      |     6.1 ms |  76 GB/s  |
///
/// The bytes fell by 3.56x and the TIME by 1.63x, and THE ARMS DO NOT SHARE A
/// KERNEL. `hand BF16 lane: 0 launches` in both logs: every plain-BF16 GEMM in
/// this run went to `cubek::matmul::launch_ref`, because the unembed table
/// aliases the pile at `align >= MIN_TUNED_ALIGN`. The four-bit lanes have no
/// tuned equivalent and run `w4a16gemm` / `fp4gemm`, one warp per
/// `(m_tile, n_tile)`. So the 163-against-74 GB/s gap is a TUNED-VERSUS-HAND
/// comparison wearing a bits-per-weight label, and this measurement cannot
/// separate the two. What it does establish is the ceiling: a four-bit head
/// that reached the BF16 lane's 163 GB/s would be 2.84 ms, so ~3.4 ms a pass
/// is unclaimed inside the four-bit lane -- more than the switch has won so
/// far.
///
/// One piece of it was measured directly. The B-fragment load used to fetch
/// the packed `u32` and the E4M3 scale once per ELEMENT rather than once per
/// pair; hoisting both out of the `j` loop (same bytes, quarter the memory
/// instructions, bit-identical output -- the lane-parity test's rel RMS is
/// 0.0091 either way) moved the stage 6.7 -> 6.2 ms. Worth having, and small
/// enough to say that instruction count is not where the rest of the gap is.
///
/// Output: 24 greedy tokens, one differed from the BF16 arm, and it differed
/// at a 0.08-logit gap (`26500` at 18.59 against `306` at 18.51, which W4A16
/// scored 18.32 against 18.32). Both continuations read as English. W4A4
/// produced the SAME 24 tokens as W4A16 on this prompt, so the activation
/// quantiser cost nothing here that the weight quantiser had not already cost.
///
/// **There is no switch.** It was `INK_W4A16_HEAD`, default off, on the
/// grounds that quantising the head is a model-quality change. That bar was
/// bit-identicality, and bit-identicality is not a property this runtime has:
/// `devplan_verify_layer` records it disagreeing with ITSELF on 8.55% of
/// argmax positions between two runs of the same binary. A gate the baseline
/// fails is not a gate. The bar is capability -- coherent text, retrieval,
/// acceptance -- and this lane meets it: one token differed in 24, at a
/// 0.08-logit gap, both continuations English.
fn head_lane() -> HeadLane {
    HeadLane::W4a16
}

/// The shortlist budget for the approximate head, and the switch that selects
/// it. `0` runs the exact `w4a16` lane.
///
/// The head is an exhaustive maximum-inner-product search: 0.43 GiB of NVFP4
/// read whole to produce one integer. `INK_ANN_HEAD=N` replaces the exhaustive
/// scan with a 1-bit sign sketch over the same rows (0.103 GiB, a quarter of
/// the bytes) and an EXACT rescore of the `N` rows whose estimates come out on
/// top. See [`mary::models::inkling::annhead`] for why narrower codes and not
/// fewer rows, and for the rotation and the unbiasing scalar that make a
/// one-bit estimate rank correctly at all.
///
/// `N` is a budget rather than a threshold because the lane's own question is a
/// threshold — every row that could still win — and a budget is how a caller
/// says what it will pay to answer it.
fn ann_budget() -> usize {
    static CHOSEN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| {
        std::env::var("INK_ANN_HEAD")
            .ok()
            .map(|v| {
                v.parse::<usize>()
                    .unwrap_or_else(|_| panic!("INK_ANN_HEAD={v:?} wants a shortlist size, or 0"))
            })
            .unwrap_or(ANN_BUDGET_DEFAULT)
    })
}

/// The logit window the shortlist floor is chosen inside.
///
/// The floor comes from a histogram over `[max - range, max]`; a row further
/// than `range` below the best estimate is not a candidate under any budget.
/// Twelve logits is far wider than any near-tie this model produces (the W4A16
/// head's worst measured disagreement was a 0.08-logit gap) and narrow enough
/// that 1024 bins resolve to 0.012 logits, which is finer than that gap.
fn ann_range() -> f32 {
    static CHOSEN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| {
        std::env::var("INK_ANN_RANGE")
            .ok()
            .map(|v| {
                v.parse::<f32>()
                    .unwrap_or_else(|_| panic!("INK_ANN_RANGE={v:?} wants a logit window"))
            })
            .unwrap_or(12.0)
    })
}

/// Build the sketch on RAW coordinates instead of in the rotated basis.
///
/// The ablation for the claim that the rotation is not optional, and it has to
/// exist on the real table rather than only in `inkling_ann_gate`: the whole
/// argument for rotating is about the structure of a real embedding matrix —
/// rogue dimensions carrying disproportionate mass, error pooling on whichever
/// tokens load them — and a synthetic table can only show that the mechanism
/// works on a structure someone planted there on purpose.
fn ann_rotated() -> bool {
    static CHOSEN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| {
        std::env::var("INK_ANN_ROT")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Run BOTH head lanes on the same hidden state and count how often they agree.
///
/// The paired-run instrument, and the only honest one: recall is a property of
/// the pair, so measuring it on synthetic queries would answer a question about
/// synthetic queries. This runs the exact `w4a16` GEMM beside the approximate
/// lane on the SAME normed, perturbed, muP-divided hidden state a real prompt
/// produced, and folds the comparison into
/// [`mary::models::inkling::annhead::VERIFY`].
///
/// It roughly doubles the head stage, so it is a gate and not a default.
fn ann_verify() -> bool {
    static CHOSEN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| {
        std::env::var("INK_ANN_VERIFY")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Sampling temperature, applied as noise on the QUERY.
///
/// `0.0` is greedy decode and is bit-identical to a run with no sampling path at
/// all: the perturbation is not built, not uploaded and not added. That is the
/// property the switch is written around — this is the first sampling mechanism
/// in this codebase, and it has to reduce EXACTLY to what shipped before it.
///
/// # Why the noise goes on the hidden state and not on the logits
///
/// The textbook mechanism is Gumbel-max: add `Gumbel(0, T)` to every logit and
/// take the argmax. It is exact, and it presupposes a SCAN — it is only defined
/// if you visit every row — so it silently constrains the head to be linear in
/// the vocabulary, which is the thing an approximate head exists to escape.
/// Query noise perturbs the INPUT, so it composes with whatever retrieval
/// structure the head grows.
///
/// It also does a second job that score noise cannot do at all. An approximate
/// head's error is DETERMINISTIC given the hidden state: a row this sketch
/// under-estimates is under-estimated every time, excluded from the shortlist
/// every time, and never rescored — so it can never be emitted, ever. Adding
/// `g_i` to a biased estimate does not fix that, because `argmax(est_i + g_i)`
/// still under-selects a row whose estimate is biased low; the noise rides on
/// top of the bias. Perturbing the query changes `sign(Rq)`, which changes which
/// bits agree, which RE-ROLLS the error for every row at once.
///
/// # What it costs, stated rather than discovered later
///
/// This does not reproduce a softmax exactly, and it is not meant to.
/// `<h + eps, w_i> = <h, w_i> + <eps, w_i>`: symmetric `eps` induces symmetric
/// logit noise where Gumbel is skewed, and `Var = sigma^2 ||w_i||^2` makes the
/// effective temperature PER TOKEN, scaled by that token's embedding-row norm.
/// The bar here is capability — coherent text, retrieval, acceptance — and not
/// numerical identity with a softmax; this runtime disagrees with ITSELF on
/// 8.55% of argmax positions between two runs of the same binary, so a bar of
/// distributional identity is one nothing on this stack meets.
///
/// The value is a temperature in LOGIT units. It is divided by the mean
/// embedding-row norm to become a hidden-state sigma, and multiplied by
/// `pi/sqrt(6)` so that the induced logit noise has the standard deviation a
/// Gumbel of the same temperature would have. Both conversions are exact
/// statements about the first two moments and neither claims the shapes match.
fn head_temp() -> f32 {
    static CHOSEN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| {
        std::env::var("INK_TEMP")
            .ok()
            .map(|v| {
                v.parse::<f32>()
                    .unwrap_or_else(|_| panic!("INK_TEMP={v:?} wants a temperature"))
            })
            .unwrap_or(0.0)
    })
}

/// Seed for the query perturbation. Fixed so a temperature run reproduces.
fn head_temp_seed() -> u64 {
    static CHOSEN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| {
        std::env::var("INK_TEMP_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0x5EED_1107)
    })
}

/// `count` standard normals from a counter-based generator.
///
/// Counter-based rather than stateful because the noise has to be a function of
/// `(seed, step)` and nothing else: a stateful RNG makes the perturbation depend
/// on how many times the process happened to draw, which is exactly the kind of
/// hidden dependence that makes a sampling run irreproducible for reasons nobody
/// can find. Splitmix64 for the bits, Box-Muller for the shape.
fn normals(seed: u64, step: u64, count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    let mut i = 0u64;
    while out.len() < count {
        let mix = |mut z: u64| -> u64 {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let a = mix(seed
            ^ step
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(i.wrapping_mul(0xD1B5_4A32_D192_ED03)));
        // Open on both ends: `ln(0)` is the one input Box-Muller cannot take.
        let u1 = ((a >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let u2 = ((mix(a) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        out.push((r * th.cos()) as f32);
        if out.len() < count {
            out.push((r * th.sin()) as f32);
        }
        i += 1;
    }
    out
}

/// The shared/sink experts are held as NVFP4 codes and
/// multiply them against a BF16 activation.
///
/// # Why W4A16 and not the routed experts' lane
///
/// This switch used to be `INK_SHARED_FP4` and it bound
/// [`dev_lane::ProjW::Fp4`] -- the W4A4 lane, which quantises the ACTIVATION as
/// well. That was wrong for these tensors specifically, and wrong for the same
/// reason it was wrong on the unembedding. The checkpoint's
/// `hf_quant_config.json` enables an input quantiser only for the layers it
/// quantised, which is the ROUTED experts; the sinks have no calibrated
/// activation quantiser at all, so W4A4 was inventing one. The reference
/// implementation this project is chasing calls `flashinfer.mm_bf16_fp4` for
/// exactly this tensor: four-bit weights, BF16 activations. That is
/// [`dev_lane::ProjW::W4a16`], and it is measurably the closer lane -- the
/// weight-only parity test
/// (`linear_w4a16_tracks_linear_bf16_on_the_same_weight`) puts W4A16 at 0.0091
/// rel RMS against the BF16 reference where W4A4 is at 0.0155.
///
/// The W4A4 arm is GONE rather than kept beside it. Two arms on one switch is
/// how the wrong one got bound in the first place.
///
/// # What it buys, and what it costs
///
/// Per decode layer-step this model streams two sink experts at **100.7 MB of
/// BF16** against six routed experts at 84.9 MB of NVFP4: the sinks cost more
/// bandwidth than the six experts that do the routing, purely because the
/// publisher quantised the routed experts and left these alone. At four bits
/// the same two tensors are ~28 MB. On the operation itself, the same shapes
/// run 65.08 ms on the BF16 grouped lane against 18.44 ms on the NVFP4 one.
///
/// Still off by default, and NOT for the reason the fused-QKVR switch is off.
/// This one is a MODEL-QUALITY change: it overrides the publisher's choice with
/// our own quantiser, and unlike the fused concatenation its output is not
/// bit-identical to the arm it replaces. It defaults on when someone has
/// measured what it does to the tokens, the way [`head_lane`] documents its
/// own.
///
/// **There is no switch.** It was `INK_W4A16_SINKS`, default off, waiting for
/// someone to "measure what it does to the tokens". Two BF16 sinks move
/// 1.407 GB a pass -- exactly as many bytes as all six routed NVFP4 experts
/// combined, from two experts instead of six -- so this is the largest single
/// byte win in the step, and it was switched off by a bar the baseline itself
/// fails. See [`head_lane`] for why that bar is gone.
fn sink_w4a16() -> bool {
    true
}

/// The dense weights that live in DEVICE memory for the whole run, unwidened.
///
/// Distinct from the host-resident set, and the difference is the point. Host
/// residency stops the re-read; device residency moves the weight into the
/// memory the GPU reads fastest and lets the matmul run there.
///
/// What changed with rule 3: these used to be `Tensor<Bk, 2>` — Burn f32 — so
/// every BF16 leaf on the way here was doubled, once into a host `Vec<f32>` and
/// again into a device buffer. 4.88 GiB of f32 on the 20-layer head to hold
/// 2.44 GiB of stored weight, and twice the bytes for the GEMM to read on every
/// token. They are [`Bf16W`] now: a handle over BF16, multiplied by the
/// `mma.sync…bf16` instruction whose f32 accumulator is its own output type and
/// not a widening.
#[derive(Default)]
struct DeviceDense {
    shared: std::collections::BTreeMap<String, SharedOnDevice>,
    dense: std::collections::BTreeMap<String, (Bf16W, Bf16W, Bf16W, f32)>,
    bytes: u64,
}

/// One MTP head's weights, OWNED, so the borrowed [`MtpHead`] handed to
/// `mtp_block` can be rebuilt per draft without re-reading anything.
///
/// The split exists because `MtpHead` borrows and the loop needs the owner to
/// outlive every borrow. `gate`/`up` are materialised rather than held because
/// `split_gate_up` de-interleaves the fused matrix into two, and doing that once
/// per head at load beats doing it once per draft.
struct MtpOwned {
    embed_norm: Held,
    hidden_norm: Held,
    input_proj: Held,
    attn_norm: Held,
    mlp_norm: Held,
    attn_sconv: Held,
    mlp_sconv: Held,
    wq: Held,
    wk: Held,
    wv: Held,
    wr: Held,
    wo: Held,
    q_norm: Held,
    k_norm: Held,
    k_sconv: Held,
    v_sconv: Held,
    rel_proj: Held,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Held,
    global_scale: f32,
    dims: AttnDims,
    local: bool,
}

impl MtpOwned {
    /// The sliding window this head attends within, or `None` on a global
    /// one.
    ///
    /// One fact in one place: the causal mask and the cache's idea of which
    /// keys can never be read again are the same distinction. A head that
    /// masked as local while caching as global would still answer correctly
    /// and grow its cache without bound, which is the kind of bug that only
    /// ever shows up as a memory graph.
    fn window(&self, sliding: usize) -> Option<usize> {
        if self.local { Some(sliding) } else { None }
    }

    /// Borrow this head in the shape `mtp_block` wants.
    fn borrow(&self, inter: usize) -> MtpHead<'_> {
        MtpHead {
            embed_norm: &self.embed_norm.data,
            hidden_norm: &self.hidden_norm.data,
            input_proj: &self.input_proj.data,
            lw: LayerWeights {
                attn_norm: &self.attn_norm.data,
                mlp_norm: &self.mlp_norm.data,
                attn_sconv: &self.attn_sconv.data,
                mlp_sconv: &self.mlp_sconv.data,
            },
            aw: AttnWeights {
                wq: &self.wq.data,
                wk: &self.wk.data,
                wv: &self.wv.data,
                wr: &self.wr.data,
                wo: &self.wo.data,
                k_sconv: &self.k_sconv.data,
                v_sconv: &self.v_sconv.data,
                q_norm: &self.q_norm.data,
                k_norm: &self.k_norm.data,
                rel_proj: &self.rel_proj.data,
            },
            mlp: LayerMlp {
                gate: &self.gate,
                up: &self.up,
                down: &self.down.data,
                global_scale: self.global_scale,
                inter,
            },
        }
    }
}

/// Bind a `[rows, cols]` BF16 byte block to the device, aliasing where it can.
///
/// `bytes` is either a view of the pile's mapping — in which case this is a
/// pointer, not a transfer — or a de-interleaved copy, which cannot be aliased
/// because it is not IN the mapping. `Aliases::slice_or_copy` decides by
/// pointer containment and counts which happened, so the report can say.
/// Upload BF16 weight bytes, quantise them to NVFP4, and keep only the codes.
///
/// The BF16 upload is a scratch buffer whose handle dies with this call, so
/// what the run holds afterwards is the packed form: `[rows, cols]` at four
/// bits plus one E4M3 scale per sixteen, against two bytes an element. The
/// publisher did not quantise these tensors, so there is no `scale2` to carry
/// and the quantiser folds the range into the block scales -- the same
/// arrangement the activation quantiser has always used on this lane.
///
/// Returns the bare [`dev_lane::PackedW`] and NOT a [`dev_lane::ProjW`], which
/// is the whole point of the signature. It used to hand back
/// `ProjW::Fp4` -- the W4A4 lane -- and every caller that wanted W4A16 got W4A4
/// instead and COMPILED, because the two variants carry the same payload. That
/// mistake shipped twice: once on the unembedding and once on the sink experts.
/// A caller that must now NAME the lane cannot make it a third time.
fn quantized_bf16(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> dev_lane::PackedW {
    assert_eq!(
        bytes.len(),
        rows * cols * 2,
        "{rows}x{cols} BF16 is not {} bytes",
        bytes.len()
    );
    let src = client.create_from_slice(bytes);
    let (codes, scales) =
        mary::models::inkling::fp4quant::quantize_nvfp4_bf16(client, &src, rows, cols);
    dev_lane::PackedW {
        codes,
        scales,
        n: rows,
        k: cols,
        scale2: 1.0,
        swizzled: false,
    }
}

/// Bind a quantised weight to the W4A16 lane, in MMA-fragment order where the
/// shape allows it.
///
/// The routed experts get their permutation free, inside the startup memcpy out
/// of the pile. These weights have no such memcpy -- they are QUANTISED on the
/// device by [`quantized_bf16`] -- so it is its own pass: one linear write with
/// a gathered read, 0.43 GiB at the head's shape, once per process. The gather
/// is the scattered side, and moving it here is the entire trade: it is what
/// the GEMM stops doing on every step, of every pass, forever.
///
/// It is applied HERE and not inside the quantiser because the quantiser is
/// lane-agnostic: `HeadLane::W4a4` binds the identical `PackedW` to
/// [`dev_lane::linear_fp4`], which is `m16n8k64` and would read k16-permuted
/// bytes as if they were row-major. `linear_fp4` refuses such a weight, so that
/// mistake is an error rather than a plausible-looking logit -- but the right
/// place to not make it is here.
///
/// `swizzled` reports whether the pass RAN, never whether it was requested.
fn w4a16_bind(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    mut p: dev_lane::PackedW,
) -> dev_lane::ProjW {
    use mary::models::inkling::w4a16gemm as k16;
    // THE APPROXIMATE HEAD OWNS m=1, AND IT CANNOT READ FRAGMENT ORDER.
    // `linear_ann` -> `ann_logits` reads codes/scales as row-major [n, k/8] and
    // has no fragment path, so permuting the head while the aNN lane is on makes
    // every exact rescore read permuted bytes as row-major. Neither branch is
    // wrong alone; the seam between them is.
    //
    // Skipping the permutation costs almost nothing here: with the aNN lane on,
    // m=1 never reaches `linear_w4a16` at all, so the head swizzle only ever
    // benefited verify passes (m>=2). Teaching `ann_logits` the fragment layout
    // would recover that and is the better fix; this is the correct one.
    let ann_owns_m1 = ann_budget() > 0;
    if !ann_owns_m1 && k16::swizzle_w4a16() && k16::swizzleable(p.n, p.k) {
        let (c, s) = k16::swizzle_w4a16_device(client, &p.codes, &p.scales, p.n, p.k);
        p.codes = c;
        p.scales = s;
        p.swizzled = true;
    }
    // Once a process, not once a weight: the sink experts come through here
    // several times a layer and the line is about the LAYOUT, which is one
    // decision.
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        println!(
            "  W4A16 weights written in {} order",
            if p.swizzled {
                "MMA-FRAGMENT (m16n8k16)"
            } else if ann_owns_m1 {
                "row-major [n, k/8] (the approximate head owns m=1)"
            } else {
                "row-major [n, k/8]"
            }
        );
    }
    dev_lane::ProjW::W4a16(p)
}

fn bind_bf16(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Bf16W {
    assert_eq!(
        bytes.len(),
        rows * cols * 2,
        "{rows}x{cols} BF16 is not {} bytes",
        bytes.len()
    );
    assert!(
        Bf16W::tileable(rows, cols),
        "{rows}x{cols} does not tile as m16n8k16; this lane multiplies BF16 by BF16 and \
         widening to reach a shape is what rule 3 forbids"
    );
    let align = note_align(bytes, rows, cols);
    // `INK_ALIGN_COPY=1`: pay a COPY for a weight the alias seam can only place
    // at 4 or 8, and buy the tuned GEMM lane for it. The trade is device bytes
    // against kernel throughput and it is measured both ways rather than
    // assumed; the default is to alias and take the hand kernel on those.
    let copy_to_align = align < 16
        && std::env::var("INK_ALIGN_COPY")
            .map(|v| v == "1")
            .unwrap_or(false);
    // `INK_DENSE_WEIGHTS=device`: take the pool copy and read the weight at
    // device rate rather than over the host page tables. 179.0 -> 226.8 GB/s
    // achieved and 18.4 -> 20.2 tok/s, for a second residency that
    // `budget::dense_weights` makes admission charge -- 3.53 GiB at 0:16, which
    // is why it is not yet the default. See `budget::dense_weights`.
    let device = budget::dense_weights() == budget::DenseWeights::DevicePool;
    let h = match aliases {
        Some(al) if !copy_to_align && !device => al.slice_or_copy(client, bytes),
        _ => client.create_from_slice(bytes),
    };
    Bf16W {
        h,
        n: rows,
        k: cols,
        align: if copy_to_align { 16 } else { align },
    }
}

/// How every plain-BF16 weight is aligned, and how much of the model that is.
///
/// The tuned matmul picks its load width from the SHAPE and never from the
/// pointer, so a `[4096, 4096]` operand gets 16-byte loads whatever address it
/// sits at. The aliasing seam only promises 4 (the checkpoint packs tensors
/// back to back and puts the expert slabs at 4 mod 16), so this counts what the
/// dense lane actually gets before anything is decided on the strength of it.
static ALIGN: [core::sync::atomic::AtomicU64; 8] =
    [const { core::sync::atomic::AtomicU64::new(0) }; 8];

fn note_align(bytes: &[u8], rows: usize, cols: usize) -> usize {
    use core::sync::atomic::Ordering::Relaxed;
    let p = bytes.as_ptr() as usize;
    let (slot, align) = if p % 16 == 0 {
        (0, 16)
    } else if p % 8 == 0 {
        (1, 8)
    } else if p % 4 == 0 {
        (2, 4)
    } else {
        (3, 1)
    };
    ALIGN[slot].fetch_add(1, Relaxed);
    ALIGN[4 + slot].fetch_add((rows * cols * 2) as u64, Relaxed);
    align
}

/// Print the alignment census gathered by [`note_align`].
fn report_align(charged: u64) {
    use core::sync::atomic::Ordering::Relaxed;
    let a: Vec<u64> = ALIGN.iter().map(|c| c.load(Relaxed)).collect();
    let n = a[0] + a[1] + a[2] + a[3];
    if n == 0 {
        return;
    }
    // The ledger for `pile::device_weight_bytes`. That function enumerates the
    // weights this lane binds from a list of names; this counter is what the
    // lane ACTUALLY bound. They are two faces of one interface and the whole
    // point of printing them together is that reading either alone proves
    // nothing -- a lane added later that admission does not know about is
    // invisible in the estimate and visible here, as a gap with a size.
    let bound = a[4] + a[5] + a[6] + a[7];
    if budget::dense_weights() == budget::DenseWeights::DevicePool {
        let gap = bound as i64 - charged as i64;
        println!(
            "  device-pool weights: admission charged {:.2} GiB, the stack bound {n} weights \
             = {:.2} GiB{}",
            charged as f64 / GIB,
            bound as f64 / GIB,
            if charged == 0 {
                "  (nothing charged: no startup copy ran)".to_string()
            } else if gap.unsigned_abs() > bound / 100 {
                format!(
                    "  -- UNPRICED {:+.2} GiB, more than 1% of the bound total. \
                     A binding lane admission does not know about.",
                    gap as f64 / GIB
                )
            } else {
                String::new()
            },
        );
    }
    println!(
        "  plain-BF16 weight binds: {n} weights, {:.2} GiB -- 16B {} ({:.2} GiB), 8B {} ({:.2} GiB), \
         4B {} ({:.2} GiB), worse {} ({:.2} GiB)",
        (a[4] + a[5] + a[6] + a[7]) as f64 / GIB,
        a[0],
        a[4] as f64 / GIB,
        a[1],
        a[5] as f64 / GIB,
        a[2],
        a[6] as f64 / GIB,
        a[3],
        a[7] as f64 / GIB
    );
}

impl DeviceDense {
    /// One layer's shared experts, bound on first use.
    #[allow(clippy::too_many_arguments)]
    fn shared_for(
        &mut self,
        cp: &Weights,
        client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
        aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
        p: &str,
        n_shared: usize,
        inter: usize,
        h: usize,
        halved: bool,
    ) -> Result<&SharedOnDevice> {
        if !self.shared.contains_key(p) {
            let fused = cp.stored(&format!("{p}mlp.shared_experts.shared_w13_weight"))?;
            anyhow::ensure!(fused.elem == Elem::Bf16, "shared_w13 is {:?}", fused.elem);
            let (g, u) = mary::models::inkling::load::split_shared_w13_bytes(
                &fused.bytes,
                n_shared,
                inter,
                h,
                halved,
                2,
            );
            let d = cp.stored(&format!("{p}mlp.shared_experts.shared_w2_weight"))?;
            anyhow::ensure!(d.elem == Elem::Bf16, "shared_w2 is {:?}", d.elem);
            let per_d = h * inter * 2;
            // Gate blocks then up blocks, one buffer. `split_shared_w13_bytes`
            // already returns each side with every expert contiguous, so this
            // is a concatenation and not a second de-interleave; the row order
            // is what `shared_experts_bf16` slices the result by.
            let mut gu = g;
            gu.extend_from_slice(&u);
            // One bind or the other, never both: holding the BF16 twin as
            // well would keep the 100.7 MB a layer this exists to stop
            // streaming, and the admission gate prices what is held.
            let mut sd = SharedOnDevice {
                gate_up: if sink_w4a16() {
                    // W4A16 and NOT `Fp4`: four-bit weights against a BF16
                    // activation, because nothing calibrated an input
                    // quantiser for this tensor. See [`sink_w4a16`].
                    w4a16_bind(client, quantized_bf16(client, &gu, 2 * n_shared * inter, h))
                } else {
                    dev_lane::ProjW::Bf16(bind_bf16(client, aliases, &gu, 2 * n_shared * inter, h))
                },
                down: Vec::new(),
            };
            for e in 0..n_shared {
                let raw = &d.bytes[e * per_d..(e + 1) * per_d];
                sd.down.push(if sink_w4a16() {
                    w4a16_bind(client, quantized_bf16(client, raw, h, inter))
                } else {
                    // `w2` is NOT de-interleaved, so this one is a view of the
                    // pile and aliases outright.
                    dev_lane::ProjW::Bf16(bind_bf16(client, aliases, raw, h, inter))
                });
            }
            self.bytes += (gu.len() + n_shared * per_d) as u64;
            self.shared.insert(p.to_string(), sd);
        }
        Ok(&self.shared[p])
    }

    /// One dense layer's MLP, bound on first use.
    fn dense_for(
        &mut self,
        cp: &Weights,
        client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
        aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
        p: &str,
        h: usize,
    ) -> Result<&(Bf16W, Bf16W, Bf16W, f32)> {
        if !self.dense.contains_key(p) {
            let fused = cp.stored(&format!("{p}mlp.w13_dn.weight"))?;
            anyhow::ensure!(fused.elem == Elem::Bf16, "dense w13 is {:?}", fused.elem);
            let (g, u) = mary::models::inkling::load::split_gate_up_bytes(&fused.bytes, h, 2);
            let down = cp.stored(&format!("{p}mlp.w2_md.weight"))?;
            anyhow::ensure!(down.elem == Elem::Bf16, "dense w2 is {:?}", down.elem);
            let (drows, dcols) = (down.dims[0] as usize, down.dims[1] as usize);
            let inter = g.len() / (h * 2);
            // The global scale is one f32 and is a SCALAR the product is
            // multiplied by, not a weight, so it comes through the widening
            // accessor and costs four bytes.
            let gs = cp.tensor(&format!("{p}mlp.global_scale"))?.data[0];
            self.bytes += (g.len() + u.len() + down.bytes.len()) as u64;
            let trip = (
                bind_bf16(client, aliases, &g, inter, h),
                bind_bf16(client, aliases, &u, inter, h),
                bind_bf16(client, aliases, &down.bytes, drows, dcols),
                gs,
            );
            self.dense.insert(p.to_string(), trip);
        }
        Ok(&self.dense[p])
    }
}

/// The dense MLP, with every weight the BF16 it is stored as.
///
/// `dev_lane::dense_mlp`'s twin: `down(silu(gate(x)) * up(x)) * global_scale`,
/// with the elementwise half still in Burn.
fn dense_mlp_bf16(x: T2, w: &(Bf16W, Bf16W, Bf16W, f32)) -> T2 {
    // `dense_intermediate_size` is 16384 against the residual stream's 4096, so
    // each of these is FOUR TIMES the stream per token -- 64 KiB a token at f32
    // -- and the wide form holds four of them at once: the gate, the up, the
    // gated gate and their product. That is 256 KiB a token in the two dense
    // layers, which on a 21-layer head is the same order as the whole routed
    // MoE, from two layers out of forty-two.
    //
    // Narrowed the moment each is produced. `linear_bf16` accepts the product
    // in that same BF16 storage, so there is no f32 copy between the
    // elementwise half and the down projection.
    let g = dev_lane::as_act(dev_lane::linear_bf16(x.clone(), &w.0));
    let u = dev_lane::as_act(dev_lane::linear_bf16(x, &w.1));
    dev_lane::linear_bf16(dev_lane::silu(g) * u, &w.2).mul_scalar(w.3)
}

/// One MTP head's weights, on the DEVICE.
///
/// The draft path was scalar HOST arithmetic while every other multiply in this
/// file was on the device, and that is not a detail: measured on the two-node
/// pipe, a warm decode round trip is 131 ms and `INK_MTP=4` drafting is 1470 ms.
/// A draft that costs eleven forward passes is not a cheap approximation of one,
/// so speculation could not pay at ANY acceptance rate — the arithmetic said so
/// before the model got a vote. Nothing about the shape justified it: an MTP
/// block's `transformer_block.*` is shape-identical to an ordinary decoder
/// layer, so it multiplies by exactly the operators the stack already runs on
/// the device, and the wrapper around it is two RMS norms, a concatenation and
/// one `[hidden, 2 * hidden]` projection.
///
/// The host lane stays as the CONTROL and not as a fallback: `INK_MTP_DEV=0`
/// selects it, and it is what says whether this transcription drafts the same
/// tokens.
struct MtpDev {
    attn: dev_lane::AttnWeightsDev,
    attn_sconv: T2,
    mlp_sconv: T2,
    attn_norm: BT<Bk, 1>,
    mlp_norm: BT<Bk, 1>,
    embed_norm: BT<Bk, 1>,
    hidden_norm: BT<Bk, 1>,
    /// `[hidden, 2 * hidden]`, which is the whole of what
    /// `mtp_hidden_states_first` is a claim about.
    input_proj: Bf16W,
    dense: (Bf16W, Bf16W, Bf16W, f32),
}

/// What one device MTP head retains between draft steps.
///
/// The same three things a main-stack layer keeps, and for the same reasons —
/// see [`dev_lane::AttnCache`] on why the pre-convolution projections are not
/// optional. `Clone` is load-bearing: a SPECULATIVE row is run against a clone
/// that is then dropped, so a rejected draft leaves nothing to undo.
#[derive(Clone)]
struct MtpDevCache {
    attn: dev_lane::AttnCache<Bk>,
    attn_sconv: T2,
    mlp_sconv: T2,
}

/// The MTP wrapper's input: each operand normed by its OWN weight, concatenated
/// in the order under test, projected back down to `hidden`.
///
/// Two norms is the tell that the operands are joined rather than summed; a
/// residual add would want one. The concat order is the whole open question the
/// acceptance rate answers, so it is a parameter here exactly as it is on the
/// host.
/// `INK_MTP_BACKBONE_NORM=0` restores the pre-2026-08-24 behaviour: a drafted
/// token's embedding goes to the depth head RAW, without the backbone
/// `embed_norm`. It exists so the change can be A/B'd in one binary against the
/// arm it replaces, which is the only honest way to price it -- not as a
/// fallback. Default on.
fn backbone_embed_norm() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_MTP_BACKBONE_NORM")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// One embedding row as an MTP depth layer must see it.
///
/// The depth layers were trained on the embeddings the BACKBONE consumes, so
/// the chain is `depth_embed_norm(backbone_embed_norm(embed(id)))`. Their own
/// `embed_norm` weights are near-identity trims; the backbone's is a whitening
/// norm, and skipping it hands the head a differently-scaled vector. vLLM's
/// implementation states the cost outright: "feeding raw embeddings drops MTP1
/// acceptance from ~0.85 to ~0.70".
///
/// `backbone_norm` is `None` when the model declares no `use_embed_norm`, or
/// under `INK_MTP_BACKBONE_NORM=0`; then this is exactly [`embed_row_bf16`].
///
/// Speculation is self-verifying, so getting this wrong never produced wrong
/// text -- only a low acceptance rate, which is why it survived so long.
/// Whether a draft's hidden state takes the BACKBONE's final norm on its way to
/// the unembedding. Default on, and `INK_MTP_OUTNORM=0` is the ablation.
///
/// The reason it is a question: DeepSeek-style MTP gives every depth module its
/// OWN `shared_head.norm` before the shared unembedding, and this checkpoint
/// ships no such tensor -- only `embed_norm`, `hidden_norm` and `input_proj`
/// per depth. So either the backbone's `model.llm.norm` is reused (what this
/// does) or there is no norm at all, and the absent tensor is equally
/// consistent with both. An RMS norm is not a scale, so the choice moves the
/// argmax.
fn mtp_out_norm() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_MTP_OUTNORM")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

fn mtp_embed_row(
    table: &[u8],
    backbone_norm: Option<&[f32]>,
    id: usize,
    eps: f64,
    vocab: usize,
    hidden: usize,
) -> Vec<f32> {
    match backbone_norm {
        Some(gain) => embed_and_norm_bf16(&[id], table, gain, eps, vocab, hidden),
        None => embed_row_bf16(table, id, vocab, hidden),
    }
}

/// Which operand of the MTP wrapper to ZERO, for the ablation that says whether
/// a head is using it at all.
///
/// The teacher-forced rate came out FLAT across depths -- 0.227, 0.193, 0.209,
/// 0.232 for depths 1..4 on a document -- and flat is the shape of a predictor
/// whose accuracy does not depend on how far ahead it is asked to see. A head
/// reading its hidden state cannot behave that way; a head reading only the
/// embedding it was handed can, because "the token after this token" is the
/// same task at every depth. So: zero one operand and see which one the rate
/// was living on. `INK_MTP_ABLATE=hidden` or `=embed`.
fn mtp_ablate() -> Option<&'static str> {
    use std::sync::OnceLock;
    static A: OnceLock<Option<String>> = OnceLock::new();
    A.get_or_init(|| std::env::var("INK_MTP_ABLATE").ok())
        .as_deref()
        .map(|v| match v {
            "hidden" => "hidden",
            "embed" => "embed",
            other => panic!("INK_MTP_ABLATE wants hidden|embed, got {other:?}"),
        })
}

fn mtp_input_dev(hidden: T2, embeds: T2, w: &MtpDev, eps: f64, order: MtpConcat) -> T2 {
    let (hidden, embeds) = match mtp_ablate() {
        Some("hidden") => (hidden.zeros_like(), embeds),
        Some("embed") => (hidden, embeds.zeros_like()),
        _ => (hidden, embeds),
    };
    // `INK_MTP_SWAPNORM=1`: apply `embed_norm` to the hidden operand and
    // `hidden_norm` to the embedding. The names say otherwise and nothing else
    // does -- `transformers` discards these tensors and no reference consumes
    // them -- so which gain belongs to which operand is a READING, in the same
    // class as the concat order that turned out to matter by a factor of 200.
    let (gh, ge) = if std::env::var("INK_MTP_SWAPNORM")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        (w.embed_norm.clone(), w.hidden_norm.clone())
    } else {
        (w.hidden_norm.clone(), w.embed_norm.clone())
    };
    let hn = dev_lane::rms_norm(hidden, gh, eps);
    let en = dev_lane::rms_norm(embeds, ge, eps);
    let cat = match order {
        MtpConcat::HiddenFirst => BT::cat(vec![hn, en], 1),
        MtpConcat::EmbedFirst => BT::cat(vec![en, hn], 1),
    };
    dev_lane::linear_bf16(cat, &w.input_proj)
}

/// One MTP head over a whole sequence, on the device, keeping the cache.
///
/// The device twin of `mtp::mtp_block_prefill`, and the same decoder layer the
/// main stack runs: norm, attention, short convolution, residual; norm, DENSE
/// MLP, short convolution, residual. Dense regardless of `dense_mlp_idx` —
/// every MTP block is, and the two intermediate sizes differ by 8x.
#[allow(clippy::too_many_arguments)]
fn mtp_block_prefill_dev(
    hidden: T2,
    embeds: T2,
    w: &MtpDev,
    dims: &AttnDims,
    ls: Option<LogScaling>,
    window: Option<usize>,
    kernel: usize,
    eps: f64,
    order: MtpConcat,
) -> (T2, MtpDevCache) {
    let x = mtp_input_dev(hidden, embeds, w, eps, order);
    let hn = dev_lane::rms_norm(x.clone(), w.attn_norm.clone(), eps);
    let (y, attn) = dev_lane::attention_prefill(hn, &w.attn, dims, ls, window, window);
    let ahist = dev_lane::conv_history(y.clone(), kernel);
    let x1 = x + dev_lane::short_conv(y, w.attn_sconv.clone());
    let hn = dev_lane::rms_norm(x1.clone(), w.mlp_norm.clone(), eps);
    let y = dense_mlp_bf16(hn, &w.dense);
    let mhist = dev_lane::conv_history(y.clone(), kernel);
    let out = x1 + dev_lane::short_conv(y, w.mlp_sconv.clone());
    (
        out,
        MtpDevCache {
            attn,
            attn_sconv: ahist,
            mlp_sconv: mhist,
        },
    )
}

/// One position of one MTP head on the device, reading the cache.
#[allow(clippy::too_many_arguments)]
fn mtp_block_step_dev(
    hidden: T2,
    embeds: T2,
    w: &MtpDev,
    dims: &AttnDims,
    ls: Option<LogScaling>,
    pos: usize,
    window: Option<usize>,
    cache: &mut MtpDevCache,
    eps: f64,
    order: MtpConcat,
) -> T2 {
    let x = mtp_input_dev(hidden, embeds, w, eps, order);
    let hn = dev_lane::rms_norm(x.clone(), w.attn_norm.clone(), eps);
    let y = dev_lane::attention_step(hn, &w.attn, dims, ls, pos, window, &mut cache.attn);
    let (a, ah) = dev_lane::short_conv_step(cache.attn_sconv.clone(), y, w.attn_sconv.clone());
    cache.attn_sconv = ah;
    let x1 = x + a;
    let hn = dev_lane::rms_norm(x1.clone(), w.mlp_norm.clone(), eps);
    let y = dense_mlp_bf16(hn, &w.dense);
    let (m, mh) = dev_lane::short_conv_step(cache.mlp_sconv.clone(), y, w.mlp_sconv.clone());
    cache.mlp_sconv = mh;
    x1 + m
}

/// The shared experts, with every weight the BF16 it is stored as.
///
/// `dev_lane::shared_experts_dev`'s twin, same reason. The gamma multiplies the
/// ACTIVATION, before the down projection — not the block's output, which is
/// algebraically the same only because `down` is linear and is a different
/// function the moment anything else is inserted.
fn shared_experts_bf16(
    dev: &burn::backend::cuda::CudaDevice,
    x: T2,
    sw: &SharedOnDevice,
    gammas: &[f32],
    n_shared: usize,
) -> T2 {
    let [n, _] = x.dims();
    assert_eq!(
        gammas.len(),
        n * n_shared,
        "{} gammas for {n} tokens",
        gammas.len()
    );
    // ONE projection for every gate and every up in the layer. See
    // [`SharedOnDevice`] for why: four GEMMs against the same activation are
    // four grids of 256 cubes, and this is one of 1024.
    let inter = sw.gate_up.n() / (2 * n_shared);
    let gu = dev_lane::linear_w(x, &sw.gate_up);
    let mut out: Option<T2> = None;
    for s in 0..n_shared {
        let g = gu.clone().slice([0..n, s * inter..(s + 1) * inter]);
        let u = gu
            .clone()
            .slice([0..n, (n_shared + s) * inter..(n_shared + s + 1) * inter]);
        let col: Vec<f32> = (0..n).map(|tk| gammas[tk * n_shared + s]).collect();
        let gam = BT::<Bk, 2>::from_data(BTD::new(col, [n, 1]), dev);
        let c = dev_lane::linear_w(dev_lane::silu(g) * u * gam, &sw.down[s]);
        out = Some(match out {
            Some(o) => o + c,
            None => c,
        });
    }
    out.expect("a MoE layer with no shared experts")
}

// `lin_bf16` was here. It is `dev_lane::linear_bf16` now: once the attention
// projections needed the same bridge, keeping a second copy of it in the binary
// meant two places to get the M padding right. `dev_lane` can host it because
// nothing about it was generic -- `Bf16W` is a raw cubecl handle and the seam
// that produces one is concrete on `Bk`, which is exactly what `short_conv_step`
// already established.

/// One layer's router, on the device except for the part that is a decision.
///
/// `proj` is the projection, `[n_routed + n_shared, hidden]` however it is
/// oriented, and it is a matmul so it lives on the device. `bias` and
/// `global_scale` never multiply an
/// activation: the bias shifts the 256 scores used to PICK the top-k and takes
/// no part in the weights, and the global scale scales the weights themselves.
/// They are control plane, so they stay host, where the decision is made.
struct RouterDev {
    proj: RouterProj,
    /// `INK_ROUTER_DIFF=1` only: the f32 `[rows, hidden]` lane, held BESIDE the
    /// active one so the two selections can be compared on the same activation.
    /// It is `None` otherwise, so the ordinary run carries neither the weight
    /// nor the second matmul.
    reference: Option<T2>,
    bias: Vec<f32>,
    global_scale: f32,
}

/// What `INK_ROUTER_DIFF=1` counted for one layer, comparing the arm the run
/// ACTED ON against the f32 `[rows, hidden]` lane every earlier run shipped.
///
/// Top-k is discrete, so a changed logit CAN flip an expert, and that is a
/// behavioural change rather than a rounding one. Neither assuming it happens
/// nor assuming it does not is a measurement, so this counts it: per layer, per
/// token, with the examined count printed beside every other number so a zero
/// says how much was looked at.
///
/// The reference is computed, compared and DISCARDED. It never reaches
/// `by_expert`, so the run this instruments is the arm's own run and the
/// counters describe it rather than some third thing.
#[derive(Default, Clone, Copy)]
struct RouteDiff {
    /// Token-positions whose selection was compared.
    examined: usize,
    /// ...where the SET of chosen experts differs.
    set_differs: usize,
    /// ...where the set agrees but the order the top-k came out in does not.
    /// Harmless by itself -- the weights follow the experts -- but it is the
    /// near miss that says how close the ordering is to flipping.
    order_differs: usize,
    /// Individual (position, slot) pairs naming a different expert.
    slots_differ: usize,
    /// Largest `|active - reference|` over every logit compared.
    max_abs_logit: f32,
    /// Largest `|active - reference|` over the weights the chosen experts got.
    /// Only defined where the sets agree, since otherwise the weights are not
    /// the same quantity.
    max_abs_weight: f32,
}

impl RouteDiff {
    /// One token-position, both selections in hand.
    fn note(&mut self, a: &Routing, b: &Routing, la: &[f32], lb: &[f32]) {
        self.examined += 1;
        for (x, y) in la.iter().zip(lb) {
            self.max_abs_logit = self.max_abs_logit.max((x - y).abs());
        }
        let differ = a
            .experts
            .iter()
            .zip(&b.experts)
            .filter(|(x, y)| x != y)
            .count();
        self.slots_differ += differ;
        let mut sa = a.experts.clone();
        let mut sb = b.experts.clone();
        sa.sort_unstable();
        sb.sort_unstable();
        if sa != sb {
            self.set_differs += 1;
        } else if differ != 0 {
            self.order_differs += 1;
        } else {
            // Slot for slot the same experts, so slot for slot the same
            // quantity. Anywhere else the two `weights` vectors are indexed by
            // different experts and subtracting them compares nothing.
            for (x, y) in a.weights.iter().zip(&b.weights) {
                self.max_abs_weight = self.max_abs_weight.max((x - y).abs());
            }
        }
    }
}

/// Drop a `[n, cols]` row-major block down to its first `keep` columns.
///
/// The BF16 arm's weight is padded to the `mma` instruction's n tile, so the
/// logits come back six columns wide of what the router asked for. Sliced HERE,
/// on the host, rather than with a device `slice`: on a decode step this is one
/// row of 264 floats and a device slice would be another kernel launch in the
/// one stage of the layer that is already launch-bound.
///
/// Returns its argument untouched when there is no pad, so the f32 arms pay
/// nothing for the BF16 arm's shape.
fn drop_pad_cols(v: Vec<f32>, n: usize, cols: usize, keep: usize) -> Vec<f32> {
    assert!(keep <= cols, "cannot keep {keep} of {cols} columns");
    assert_eq!(
        v.len(),
        n * cols,
        "{} values are not [{n}, {cols}]",
        v.len()
    );
    if cols == keep {
        return v;
    }
    let mut out = Vec::with_capacity(n * keep);
    for t in 0..n {
        out.extend_from_slice(&v[t * cols..t * cols + keep]);
    }
    out
}

/// `[rows, cols]` row-major to `[cols, rows]` row-major.
///
/// The router's projection is uploaded once per layer and multiplied once per
/// token per layer for the whole run, so the orientation the matmul wants is
/// the orientation to store it in. This is that permutation, paid once on the
/// host at upload, where the device lane paid it on every call.
fn transpose_rows(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        v.len(),
        rows * cols,
        "{} values are not [{rows}, {cols}]",
        v.len()
    );
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = v[r * cols + c];
        }
    }
    out
}

/// Which router lane `INK_ROUTER` selected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RouterArm {
    /// `[rows, hidden]` f32, `.transpose()` on the device per call.
    Transpose,
    /// `[hidden, rows]` f32, transposed once on the host at upload.
    Pre,
    /// The BF16 the pile stores, into `mma.sync…bf16`. THE DEFAULT: it is the
    /// precision the model is in and the precision the reference computes in.
    Bf16,
}

impl RouterArm {
    /// `INK_ROUTER=transpose|pre|bf16`.
    ///
    /// `bf16` is the default, and the reason is precision rather than speed.
    /// Inkling is a bfloat16 model with NVFP4 experts, and the official
    /// implementation multiplies the router in bf16 on the tensor cores. The
    /// f32 the other two arms multiply in was never the model's: `gv` widened
    /// the pile's stored BF16 on the way out of the mapping, and that widening
    /// was OUR addition. So the arm that agrees with the reference is this one,
    /// and the f32 arms are the deviation.
    ///
    /// The previous round gated this arm on bitwise agreement with the widened
    /// f32 arm and the gate did not pass: on all eight `fp4_rep2` prompts the
    /// greedy continuation diverges, and 0.46% of 5048 router selections chose
    /// a different SET of six experts. Those numbers still stand and are still
    /// worth having -- they are a measured property of NVFP4 routing, which is
    /// how close the 6th and 7th expert scores sit -- but they are not evidence
    /// against this arm. Demanding that a bf16 model reproduce an f32 lane's
    /// exact top-k demands a precision the model does not have, which is the
    /// same error the deleted f32 CPU reference made.
    ///
    /// Both f32 arms stay reachable so the comparison stays runnable.
    fn from_env() -> Self {
        match std::env::var("INK_ROUTER").as_deref() {
            Ok("transpose") => RouterArm::Transpose,
            Ok("pre") => RouterArm::Pre,
            Ok("bf16") | Err(_) => RouterArm::Bf16,
            Ok(other) => panic!("INK_ROUTER={other:?} is not one of: transpose, pre, bf16"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            RouterArm::Transpose => "f32 [rows,hidden], transposed on the device PER CALL",
            RouterArm::Pre => "f32 [hidden,rows], transposed ONCE on the host at upload",
            RouterArm::Bf16 => "the STORED BF16, into mma.sync...bf16, nothing widened",
        }
    }
}

/// The router projection, in the orientation this run multiplies it in.
///
/// One enum rather than a flag because the two arms hold DIFFERENT TENSORS, not
/// the same tensor used differently: only the arm the run selected is uploaded,
/// so switching arms cannot leave a second copy of every layer's projection on
/// the device unnoticed.
enum RouterProj {
    /// `[rows, hidden]`, the checkpoint's own orientation, transposed on the
    /// device on EVERY call. `INK_ROUTER=transpose`, and what every run before
    /// this one did.
    PerCall(T2),
    /// `[hidden, rows]`, transposed ONCE on the host at upload.
    Pre(T2),
    /// The pile's own BF16 bytes, `[rows, hidden]`, into `mma.sync…bf16` — the
    /// same lane 71b837b put the attention projections on, and no f32 copy of
    /// the weight exists anywhere. The default.
    ///
    /// Inkling is a bfloat16 model with NVFP4 experts. The f32 the two arms
    /// above multiply in is not the model's precision, it is ours: the widen
    /// happened on the host on the way out of the mapping, doubled the bytes,
    /// and bought a precision the checkpoint does not have. This arm also casts
    /// the ACTIVATION to BF16, by the hardware's round-to-nearest-even, which is
    /// what the official implementation's `bfloat16` linear does. So the logits
    /// differ from the f32 arms' and are not meant not to; what has to hold is
    /// the SELECTION, and `INK_ROUTER_DIFF=1` below counts whether it does.
    ///
    /// `n` is the row count PADDED to the instruction's n tile: 258 is not a
    /// multiple of 8, so six zero rows are appended and their logits sliced off
    /// on the host. That pad is why this arm copies rather than aliases -- a
    /// padded buffer is not inside the pile's mapping -- but the copy is at the
    /// STORED width. Nothing here widens; 2.16 MB a layer moves once at upload
    /// where the f32 arms materialised 4.23 MB a layer and uploaded that. The
    /// bind counters see it as exactly eight more `UNMAPPED` copies and 16 MiB.
    Bf16 { w: Bf16W },
}

/// Everything one layer multiplies by, in DEVICE memory, for the whole run.
///
/// This used to be two attention tensors in a map and everything else read out
/// of the host residency cache per token: `attn_norm`, `mlp_norm` and
/// `mlp_sconv` were `Vec<f32>` because the operations consuming them were host
/// operations. They are not any more. A weight the device multiplies by lives
/// on the device, and the host copy is dropped after the upload rather than
/// held beside it -- holding both doubles a budget for no gain, since after the
/// upload nothing on the host reads them again.
struct LayerDev {
    attn: dev_lane::AttnWeightsDev,
    attn_sconv: T2,
    mlp_sconv: T2,
    attn_norm: BT<Bk, 1>,
    mlp_norm: BT<Bk, 1>,
    /// `None` on a dense layer, which has no experts to route to.
    router: Option<RouterDev>,
}

/// Every routed expert for one layer, on the NATIVE NVFP4 tensor-core path.
///
/// # What a routed MoE block computes
///
/// This is written out because it used to be written out somewhere else. A
/// scalar f32 host transcription (`mlp::routed_experts`, over
/// `mlp::expert_ffn_one`) served as the readable statement of the algorithm and
/// was deleted; it had no caller in the data plane, and being the legible
/// version of a hot function is exactly what made people optimise it instead of
/// this. So the legibility moves here, to the code that runs.
///
/// The router has already chosen, per token, `top_k` experts and a weight for
/// each. `by_expert` is that decision INVERTED — keyed by expert, listing
/// `(token row, routing weight)` — because a weight is 12.6 MB and a token row
/// is 16 KB, so the loop that must not be repeated is the one over weights.
/// Each expert then computes, for every token routed to it:
///
/// ```text
/// both = x  · w13ᵀ            w13 is [2 * intermediate, hidden], GATE rows first
/// act  = silu(both[..I]) * both[I..]                       I = intermediate
/// y    = act · w2ᵀ            w2  is [hidden, intermediate]
/// out[token] += y * weight
/// ```
///
/// Four things in that are easy to get wrong and are each pinned somewhere:
///
/// * the gate half comes FIRST in `w13`, and the checkpoint stores the two
///   halves INTERLEAVED (`g0, u0, g1, u1, …`). `2 * intermediate == hidden` in
///   both releases, so the fused matrix is square and a transposed or
///   un-deinterleaved reading loads without complaint and computes nonsense.
///   The de-interleave happens once, at import, and `inkling_fp4_expert_gate`
///   holds one whole real expert to an f64 arbiter over the same operands.
/// * the routing weight multiplies the expert's OUTPUT, once, after `w2` — not
///   the activation, and not the input.
/// * a token's contributions from its `top_k` experts are SUMMED, so the
///   scatter-add below is an add and not a write.
/// * the shared experts do NOT read this function's output. They run beside it
///   on the same normed input — see [`shared_experts_bf16`].
///
/// # This lane, and the two it replaced
///
/// The only routed lane there is. The packed bytes go straight into
/// `mma.sync…kind::mxf4nvf4…ue4m3`. The device lane this replaced
/// (`INK_EXPERTS=gpu`) decoded each expert into a 67.1 + 33.6 MB f32 pair,
/// multiplied THAT, and dropped it — 100 MB of device memory materialised per
/// expert to hold a weight the pile stores in 12.6, four times a token per
/// layer. It is gone rather than kept as a control, because the control was a
/// widening and the whole point of an NVFP4 model is that the weight is never
/// widened. The host lane is gone for the sibling reason: an f32 reference
/// demands agreement at a precision a bfloat16 model with 4-bit weights does not
/// have, so "the device matches the host" would have been a statement about
/// which of the two was written first.
///
/// Activations are quantised to E2M1 in dynamic per-16 blocks with E4M3
/// scales, which the instruction requires and which is what the checkpoint's
/// own `hf_quant_config.json` specifies for `*input_quantizer`.
///
/// # Device in, device out
///
/// `hn` is the normed residual stream ON THE DEVICE and the return is the
/// accumulated expert output ON THE DEVICE. It used to be `&[f32]` and
/// `Vec<f32>`, and the two conversions that implied were not free bookkeeping:
/// the gather copied each expert's rows out of a host buffer, and the drain
/// BLOCKED on `read_one` per expert to get 16 KB back so the host could do a
/// weighted add. Now `select` gathers, `select_assign` scatter-adds, and
/// nothing in the layer waits.
///
/// The accumulation order is unchanged and deliberately so: `by_expert` is a
/// `BTreeMap`, the scatter-adds are issued in its order, and each token appears
/// at most once per expert, so `acc` is the same sum in the same order the host
/// loop made. That is what keeps this a plumbing change rather than a numerics
/// one.
#[allow(clippy::too_many_arguments)]
fn routed_experts_fp4(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    admitted_narrow: bool,
    host: &mut HostT,
) -> Result<T2> {
    // Which of the two lanes runs the layer. The grouped one computes the same
    // thing with one launch per STAGE instead of one sequence per EXPERT, and
    // it returns `None` exactly when its premise -- every active expert's
    // weight is an offset into one registered mapping -- does not hold.
    //
    // `INK_GROUPED=0` takes the per-expert loop, which is how the two are held
    // to the same bits over a whole run; `INK_GROUPED=2` runs BOTH per layer
    // and prints where they part company, which is how a disagreement gets
    // located instead of argued about.
    //
    // Mode 2 REFUSES the narrow activation lane, and the refusal is the whole
    // point of it. `3614c11` gave [`grouped_experts_fp4`] a BF16 staging arm
    // under [`act_bf16`] and defaulted it ON; `per_expert_fp4` never grew one
    // and still widens to f32. So at the default the A/B compared a BF16 lane
    // against an f32 lane and printed 20480 of 20480 elements differing at rel
    // ~1.99 -- the FP4 re-quantization of BF16-rounded activations, not a
    // defect in either lane. Diagnosed 2026-08-24: forced wide, the two arms
    // are ulp-equal at all 29 routed NVFP4 layers measured.
    //
    // An instrument that reads as "checked" while comparing two different
    // precisions is worse than no instrument, and this is the only one left
    // that can see a wrong expert selection or accumulation order -- the
    // output-level gates were retired 2026-08-18 and this binary disagrees
    // with ITSELF on 8.55% of argmax positions run to run. So it fails loud
    // rather than printing a number nobody can interpret.
    //
    // NOTE the BF16-expert lane needs no such guard: `grouped_experts_bf16`
    // has no `narrow` branch, so both its arms stay in one precision. That is
    // why layer 2 was the only clean one, and it is a consequence of this same
    // cause rather than a coincidence.
    let mode = std::env::var("INK_GROUPED").unwrap_or_else(|_| "1".to_string());
    anyhow::ensure!(
        mode != "2" || !mary::models::inkling::burn::act_bf16(),
        "INK_GROUPED=2 compares the grouped lane against per_expert_fp4, but \
         INK_ACT_BF16 is on, so the grouped arm stages activations in BF16 while \
         the reference widens to f32. The comparison would be between two \
         precisions and its output is meaningless. Re-run with INK_ACT_BF16=0 to \
         put both arms on the wide lane. Note this certifies the WIDE lane only; \
         the shipped narrow kernels still have no bit-exact reference arm."
    );
    if mode != "0" {
        if let Some(al) = aliases {
            if let Some(acc) = grouped_experts_fp4(
                src, al, client, dev, prefix, by_expert, hn, n, h, inter, host,
            )? {
                if mode == "2" {
                    let reference = per_expert_fp4(
                        src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host,
                    )?;
                    report_ab(prefix, &acc, &reference, h);
                }
                host.grouped += 1;
                host.expert_slots += by_expert.len();
                return Ok(acc);
            }
        }
        // Narrow admission is a contract with this implementation, not a
        // preference. A missing/unaligned mapping or any other grouped-lane
        // premise failure would otherwise fall through to the f32 per-expert
        // buffers after the startup-copy gate had priced BF16. Diagnostic and
        // explicit copying modes are admitted wide and may still fall back.
        anyhow::ensure!(
            !admitted_narrow,
            "the packed grouped expert lane could not bind its source after this layer was \
             admitted with BF16 staging buffers. Refusing instead of silently falling back to \
             the f32 per-expert lane; set INK_GROUPED=0 or INK_ZEROCOPY=0 before launch so \
             admission prices that lane."
        );
    }
    host.per_expert += 1;
    host.expert_slots += by_expert.len();
    per_expert_fp4(
        src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host,
    )
}

/// Where the two lanes' accumulators differ, and by how much.
///
/// A MEASUREMENT, not a verdict, and that distinction is the whole point. The
/// grouped lane fuses the routing-weight multiply into the accumulating add and
/// the per-expert lane does not, so the two round in different PLACES and a gap
/// of order an ulp is EXPECTED here -- see
/// [`mary::models::inkling::moegroup`], and `91f81b4` for the time this tree
/// mistook that expectation for a defect and paid a launch to hide it.
///
/// What the number is for is the SIZE and SHAPE of the gap. An ulp-scale
/// difference on a fraction of the elements is the fused multiply. Anything
/// larger, anything structured, or any difference at all in the routing, the
/// gather, the operand order or the accumulation order is a real defect --
/// those are bit-exact and stay that way.
///
/// Compared as BITS, so a `-0.0` against a `0.0` counts and a NaN is not
/// silently equal to itself.
fn report_ab(prefix: &str, a: &T2, b: &T2, h: usize) {
    let av = down(a.clone());
    let bv = down(b.clone());
    let mut differ = 0usize;
    let mut worst = 0.0f32;
    let mut worst_rel = 0.0f32;
    let mut first: Option<(usize, usize, f32, f32)> = None;
    for i in 0..av.len() {
        if av[i].to_bits() == bv[i].to_bits() {
            continue;
        }
        differ += 1;
        let d = (av[i] - bv[i]).abs();
        worst = worst.max(d);
        let scale = av[i].abs().max(bv[i].abs()).max(f32::MIN_POSITIVE);
        worst_rel = worst_rel.max(d / scale);
        if first.is_none() {
            first = Some((i / h, i % h, av[i], bv[i]));
        }
    }
    let ln = prefix.trim_end_matches('.');
    match first {
        None => println!("  [A/B] {ln}: {} elements, ALL BITS EQUAL", av.len()),
        Some((r, c, x, y)) => println!(
            "  [A/B] {ln}: rounding gap on {differ} of {}, max abs {worst:.3e} rel {worst_rel:.3e}, first row {r} col {c}: grouped {x:.9e} per-expert {y:.9e}",
            av.len()
        ),
    }
}

/// Every routed expert for one layer in a handful of launches, or `None` if
/// this lane cannot take the layer.
///
/// Same weights, same kernels, same arithmetic and -- by construction rather
/// than by hope -- the same accumulation order as [`per_expert_fp4`]. See
/// [`mary::models::inkling::moegroup`] for why the last of those had to be
/// designed for. What differs is who walks the expert list: the host did, and
/// now `CUBE_POS_Y` does.
///
/// # What makes it possible, and when it is not
///
/// The kernel reaches every active expert's weight through ONE bound buffer
/// plus a table of byte offsets, so the whole layer's weights have to live in a
/// single registered mapping. A pile is one file and the zero-copy seam
/// registers it once, so that is the ordinary case. `INK_ZEROCOPY=0`, a device
/// that cannot address host memory, or a slab whose offset does not land on the
/// 4-byte vector the packed plane is read in -- each of those is `None`, and
/// the per-expert loop runs the layer instead. That is not a comfort blanket
/// kept beside a better lane: it is the only lane there is when this one's
/// premise fails, and the per-pass report counts which ran.
#[allow(clippy::too_many_arguments)]
fn grouped_experts_fp4(
    src: &Weights,
    al: &mary::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<Option<T2>> {
    use mary::models::inkling::moegroup::{BlockPlanDev, RowPlan};
    use mary::models::inkling::seam::handle_of_any;

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let slots = by_expert.len();
    if slots == 0 {
        return Ok(None);
    }

    // Where every plane of every active expert lives, as a byte offset into the
    // one mapping they must share. This IS the bind, and it is arithmetic.
    let t_s = Instant::now();
    let mut off13: Vec<u64> = Vec::with_capacity(2 * slots);
    let mut off2: Vec<u64> = Vec::with_capacity(2 * slots);
    let mut sc13: Vec<f32> = Vec::with_capacity(slots);
    let mut sc2: Vec<f32> = Vec::with_capacity(slots);
    let mut plane_bytes: Vec<usize> = Vec::with_capacity(4 * slots);
    let mut which: Option<usize> = None;
    for &e in by_expert.keys() {
        let w13 = src.expert_packed(&n13, e)?;
        let w2 = src.expert_packed(&n2, e)?;
        let planes: [&[u8]; 4] = [&w13.codes, &w13.scales, &w2.codes, &w2.scales];
        let mut o = [0u64; 4];
        for (i, plane) in planes.into_iter().enumerate() {
            match al.locate(plane) {
                Some((m, byte)) if which.map_or(true, |w| w == m) => {
                    which = Some(m);
                    o[i] = byte;
                }
                _ => return Ok(None),
            }
            plane_bytes.push(plane.len());
        }
        // The packed planes are read as 4-byte vectors out of the mapping, so
        // an offset that does not land on one cannot be expressed as an index.
        // The SCALE planes are read the same way now -- the instruction takes
        // its four E4M3 block scales as one 32-bit register, so the kernel
        // fetches them as one -- which puts `o[1]` and `o[3]` under the same
        // rule. The startup copy packs every view to a 4-byte boundary, so this
        // refuses nothing it did not already refuse; it is here because the
        // kernel is launched unchecked and an unaligned scale plane would be
        // read one vector to the left of where it starts.
        if o.iter().any(|v| v % 4 != 0) {
            return Ok(None);
        }
        off13.push(o[0]);
        off13.push(o[1]);
        off2.push(o[2]);
        off2.push(o[3]);
        sc13.push(w13.scale2);
        sc2.push(w2.scale2);
    }
    let (wmap, wmap_bytes) = match which.and_then(|i| al.map(i)) {
        Some(m) => m,
        None => return Ok(None),
    };
    // Charged only now that the lane is committed: a probe that turned back
    // moved nothing and must not report that it did.
    for b in plane_bytes {
        al.note_alias(b);
    }
    host.slice += t_s.elapsed().as_secs_f64();

    // The row plan, and the only host->device traffic in the layer: `[M]`
    // indices and weights, `[M/16]` slots, `[2*slots]` offsets, `[slots]`
    // second-level scales and the token->rows table. Nine small uploads for the
    // whole layer, against two per expert before.
    let t_g = Instant::now();
    let plan = RowPlan::build(by_expert.values(), n, RowPlan::planes());
    if std::env::var("INK_MOE_DEBUG").is_ok() {
        eprintln!(
            "MOEPLAN {prefix} slots={} rows={} tiles={} blocks={}",
            by_expert.len(),
            plan.m_total(),
            plan.m_total() / 16,
            plan.blk_slot.len()
        );
    }
    if plan_check() {
        plan_check_note(prefix, n, by_expert, &plan);
    }
    let m_total = plan.m_total();
    let (hn_h, hn_dt) = handle_of_any(hn.clone());
    // Split in two on purpose: see `HostT::plan_up_routed`. The `_static` half
    // is the same bytes every layer of every decode pass; the `_routed` half is
    // the only part a device-resident routing decision would have to produce.
    let t_us = Instant::now();
    let h_rowtok = client.create_from_slice(bytes_of(&plan.row_tok));
    let blk = BlockPlanDev {
        slot: client.create_from_slice(bytes_of(&plan.blk_slot)),
        tile0: client.create_from_slice(bytes_of(&plan.blk_tile0)),
        cnt: client.create_from_slice(bytes_of(&plan.blk_cnt)),
        blocks: plan.blk_slot.len(),
        planes: RowPlan::planes(),
        rows_real: plan.rows_real(),
    };
    let h_tokrows = client.create_from_slice(bytes_of(&plan.tok_rows));
    let h_tokcnt = client.create_from_slice(bytes_of(&plan.tok_cnt));
    host.plan_up_static += t_us.elapsed().as_secs_f64();
    let t_ur = Instant::now();
    let h_rowwgt = client.create_from_slice(bytes_of(&plan.row_wgt));
    let h_off13 = client.create_from_slice(bytes_of(&off13));
    let h_off2 = client.create_from_slice(bytes_of(&off2));
    let h_sc13 = client.create_from_slice(bytes_of(&sc13));
    let h_sc2 = client.create_from_slice(bytes_of(&sc2));
    host.plan_up_routed += t_ur.elapsed().as_secs_f64();
    Ok(Some(grouped_experts_core(
        client,
        dev,
        prefix,
        &wmap,
        wmap_bytes,
        &blk,
        &hn_h,
        hn_dt,
        &h_rowtok,
        &h_rowwgt,
        &h_tokrows,
        &h_tokcnt,
        &h_off13,
        &h_off2,
        &h_sc13,
        &h_sc2,
        slots,
        m_total,
        plan.kmax,
        n,
        h,
        inter,
        src.experts_swizzled(),
        t_g,
        host,
    )))
}

/// The grouped lane once its plan exists, whoever built it.
///
/// Everything from the gather to the scatter-add: two NVFP4 quantisations, two
/// grouped GEMMs, the fused SiLU between them, and the weighted accumulate.
/// None of it knows where the plan came from, which is the point — the host
/// lane and the device one differ only in who filled these eleven buffers, so
/// they cannot drift in the arithmetic.
///
/// `t_g` is the caller's gather timer, started before it began building the
/// plan: the plan uploads are charged inside the gather bucket and the report
/// says so, and moving the boundary would silently move the number.
#[allow(clippy::too_many_arguments)]
fn grouped_experts_core(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    wmap: &cubecl::server::Handle,
    wmap_bytes: usize,
    blk: &mary::models::inkling::moegroup::BlockPlanDev,
    hn_h: &cubecl::server::Handle,
    hn_dt: burn::tensor::DType,
    h_rowtok: &cubecl::server::Handle,
    h_rowwgt: &cubecl::server::Handle,
    h_tokrows: &cubecl::server::Handle,
    h_tokcnt: &cubecl::server::Handle,
    h_off13: &cubecl::server::Handle,
    h_off2: &cubecl::server::Handle,
    h_sc13: &cubecl::server::Handle,
    h_sc2: &cubecl::server::Handle,
    slots: usize,
    m_total: usize,
    kmax: usize,
    n: usize,
    h: usize,
    inter: usize,
    // Whether the bound expert planes are in MMA-fragment order; see
    // `Weights::experts_swizzled`. The two layouts are the same bytes, so this
    // cannot be inferred here and a wrong answer is silently wrong numbers
    // rather than a failure.
    swz: bool,
    t_g: Instant,
    host: &mut HostT,
) -> T2 {
    use mary::models::inkling::burn::act_bf16;
    use mary::models::inkling::fp4gemm::{gate_up_silu_launch, gate_up_silu_narrow_launch};
    use mary::models::inkling::fp4quant::{quantize_nvfp4, quantize_nvfp4_bf16};
    use mary::models::inkling::moegroup::{
        fp4_linear_grouped_bf16_launch, fp4_linear_grouped_launch, gather_grouped,
        gather_grouped_bf16, gather_grouped_bf16_from_bf16, scatter_weighted,
        scatter_weighted_bf16,
    };
    use mary::models::inkling::seam::tensor_of;

    // The two staging buffers this lane owns, at the dtype
    // [`act_bf16`] names. Both are `[m_total, _]` and `m_total` is about
    // `experts_per_token * n`, so at prefill they are the largest allocations
    // in the layer -- larger than anything attention holds -- and each is read
    // exactly once, by a quantizer that turns it into four bits.
    let narrow = act_bf16();
    // Three combinations and not four: the gather's OUTPUT is what `act_bf16`
    // decides, its SOURCE is whatever dtype the normed residual stream arrived
    // in, and a narrow stream feeding a wide gather is not a lane anybody asks
    // for -- `resid_bf16` defaults to `act_bf16`, so the stream is only BF16
    // when the gather is too.
    let x_h = match (narrow, hn_dt) {
        (true, burn::tensor::DType::BF16) => {
            gather_grouped_bf16_from_bf16(client, hn_h, h_rowtok, n, m_total, h)
        }
        (true, _) => gather_grouped_bf16(client, hn_h, h_rowtok, n, m_total, h),
        (false, burn::tensor::DType::BF16) => {
            panic!(
                "a BF16 residual stream with INK_ACT_BF16=0: set INK_RESID_BF16=0 for the wide lane"
            )
        }
        (false, _) => gather_grouped(client, hn_h, h_rowtok, n, m_total, h),
    };
    host.gather += t_g.elapsed().as_secs_f64();

    let t_w = Instant::now();
    let (a, asc) = if narrow {
        quantize_nvfp4_bf16(client, &x_h, m_total, h)
    } else {
        quantize_nvfp4(client, &x_h, m_total, h)
    };
    let both = if narrow {
        fp4_linear_grouped_bf16_launch(
            client,
            &a,
            &asc,
            wmap,
            wmap_bytes,
            blk,
            h_off13,
            h_sc13,
            slots,
            m_total,
            h,
            2 * inter,
            swz,
        )
    } else {
        fp4_linear_grouped_launch(
            client,
            &a,
            &asc,
            wmap,
            wmap_bytes,
            blk,
            h_off13,
            h_sc13,
            slots,
            m_total,
            h,
            2 * inter,
            swz,
        )
    };
    // Retain the activation handle until both quantization launches have been
    // enqueued; its name starts with `_` because lifetime, not a Rust read, is
    // the reason it remains in scope.
    let (_act_h, a2, asc2) = if narrow {
        let act_h = gate_up_silu_narrow_launch(client, &both, m_total, inter);
        let (a2, asc2) = quantize_nvfp4_bf16(client, &act_h, m_total, inter);
        (act_h, a2, asc2)
    } else {
        let act_h = gate_up_silu_launch(client, &both, m_total, inter);
        let (a2, asc2) = quantize_nvfp4(client, &act_h, m_total, inter);
        (act_h, a2, asc2)
    };
    let y_h = if narrow {
        fp4_linear_grouped_bf16_launch(
            client, &a2, &asc2, wmap, wmap_bytes, blk, h_off2, h_sc2, slots, m_total, inter, h, swz,
        )
    } else {
        fp4_linear_grouped_launch(
            client, &a2, &asc2, wmap, wmap_bytes, blk, h_off2, h_sc2, slots, m_total, inter, h, swz,
        )
    };
    host.enqueue += t_w.elapsed().as_secs_f64();

    // Every buffer this lane allocated is still live on this line -- the gather,
    // its NVFP4 codes, the `[m_total, 2 * inter]` gate-and-up, the activation,
    // its codes and the `[m_total, h]` result. That is the point of reading the
    // pool HERE and not after the return: `m_total` is about `k * n`, so these
    // are the largest buffers a prefill layer holds and they are gone by the
    // time the caller could ask.
    if mem_trace() {
        <Bk as burn::tensor::backend::Backend>::sync(dev).expect("sync before pool trace");
        println!(
            "{}",
            mary::models::inkling::seam::pool_line(client, &format!("{prefix}experts m={m_total}"))
        );
    }

    let t_c = Instant::now();
    let acc_h = if narrow {
        scatter_weighted_bf16(
            client, &y_h, h_rowwgt, h_tokrows, h_tokcnt, m_total, n, h, kmax,
        )
    } else {
        scatter_weighted(
            client, &y_h, h_rowwgt, h_tokrows, h_tokcnt, m_total, n, h, kmax,
        )
    };
    let acc = tensor_of(client.clone(), dev.clone(), acc_h, n, h);
    host.accum += t_c.elapsed().as_secs_f64();

    acc
}

/// `INK_PLAN_CHECK=1`: does the row plan actually DEPEND on the routing?
///
/// The claim this exists to refute or confirm: at `n == 1` six of
/// [`RowPlan`]'s seven fields are a function of `n` and `top_k` alone, and only
/// `row_wgt` carries the decision. If that holds, the six can be uploaded once
/// for a whole run and the per-layer readback that produces them can go.
///
/// It is a CHECK and not an assumption: the first plan seen at a given `n` is
/// kept, every later one is compared against it field by field, and the count
/// of where they differ is printed. `row_wgt` is expected to differ and is
/// counted too, because a check whose control never fires is not a check.
fn plan_check_note(
    prefix: &str,
    n: usize,
    by_expert: &std::collections::BTreeMap<usize, Vec<(usize, f32)>>,
    plan: &mary::models::inkling::moegroup::RowPlan,
) {
    use std::sync::Mutex;
    struct Snap {
        row_tok: Vec<i32>,
        row_wgt: Vec<f32>,
        blk_slot: Vec<u32>,
        blk_tile0: Vec<u32>,
        blk_cnt: Vec<u32>,
        tok_rows: Vec<u32>,
        tok_cnt: Vec<u32>,
        kmax: usize,
        seen: usize,
        /// row_tok, row_wgt, blk_slot, blk_tile0, blk_cnt, tok_rows, tok_cnt, kmax
        diff: [usize; 8],
    }
    static S: Mutex<Option<std::collections::HashMap<usize, Snap>>> = Mutex::new(None);
    let mut g = S.lock().expect("plan check");
    let map = g.get_or_insert_with(std::collections::HashMap::new);
    let first = !map.contains_key(&n);
    let s = map.entry(n).or_insert_with(|| Snap {
        row_tok: plan.row_tok.clone(),
        row_wgt: plan.row_wgt.clone(),
        blk_slot: plan.blk_slot.clone(),
        blk_tile0: plan.blk_tile0.clone(),
        blk_cnt: plan.blk_cnt.clone(),
        tok_rows: plan.tok_rows.clone(),
        tok_cnt: plan.tok_cnt.clone(),
        kmax: plan.kmax,
        seen: 0,
        diff: [0; 8],
    });
    s.seen += 1;
    if first {
        println!(
            "PLANCHECK first plan at n={n}  ({prefix}) slots={} m_total={} kmax={}",
            by_expert.len(),
            plan.m_total(),
            plan.kmax
        );
        println!("  experts   {:?}", by_expert.keys().collect::<Vec<_>>());
        println!("  row_tok   {:?}", plan.row_tok);
        println!("  blk_slot  {:?}", plan.blk_slot);
        println!("  blk_tile0 {:?}", plan.blk_tile0);
        println!("  blk_cnt   {:?}", plan.blk_cnt);
        println!("  tok_rows  {:?}", plan.tok_rows);
        println!("  tok_cnt   {:?}", plan.tok_cnt);
        println!("  row_wgt   {:?}", plan.row_wgt);
        return;
    }
    // Compared as bits for `row_wgt` so a -0.0 counts, and by value for the
    // integer fields. A LENGTH difference is a difference: a layer that routed
    // to five distinct experts instead of six refutes the claim just as loudly
    // as one that reordered them.
    let names = [
        "row_tok",
        "row_wgt",
        "blk_slot",
        "blk_tile0",
        "blk_cnt",
        "tok_rows",
        "tok_cnt",
        "kmax",
    ];
    let hit = [
        s.row_tok != plan.row_tok,
        s.row_wgt.len() != plan.row_wgt.len()
            || s.row_wgt
                .iter()
                .zip(plan.row_wgt.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
        s.blk_slot != plan.blk_slot,
        s.blk_tile0 != plan.blk_tile0,
        s.blk_cnt != plan.blk_cnt,
        s.tok_rows != plan.tok_rows,
        s.tok_cnt != plan.tok_cnt,
        s.kmax != plan.kmax,
    ];
    for (i, &h) in hit.iter().enumerate() {
        if !h {
            continue;
        }
        s.diff[i] += 1;
        // The first four of each field, in full, because "it varied" without
        // the value is a finding nobody can act on.
        if s.diff[i] <= 4 && i != 1 {
            println!(
                "PLANCHECK VARIES n={n} {} ({prefix}) obs {}: first {:?} now {:?}",
                names[i],
                s.seen,
                match i {
                    0 => format!("{:?}", s.row_tok),
                    2 => format!("{:?}", s.blk_slot),
                    3 => format!("{:?}", s.blk_tile0),
                    4 => format!("{:?}", s.blk_cnt),
                    5 => format!("{:?}", s.tok_rows),
                    6 => format!("{:?}", s.tok_cnt),
                    _ => format!("{}", s.kmax),
                },
                match i {
                    0 => format!("{:?}", plan.row_tok),
                    2 => format!("{:?}", plan.blk_slot),
                    3 => format!("{:?}", plan.blk_tile0),
                    4 => format!("{:?}", plan.blk_cnt),
                    5 => format!("{:?}", plan.tok_rows),
                    6 => format!("{:?}", plan.tok_cnt),
                    _ => format!("{}", plan.kmax),
                },
            );
        }
    }
    if s.seen % 256 == 0 {
        println!(
            "PLANCHECK n={n}: {} plans, differ from the first -- row_tok {} row_wgt {} \
             blk_slot {} blk_tile0 {} blk_cnt {} tok_rows {} tok_cnt {} kmax {}",
            s.seen,
            s.diff[0],
            s.diff[1],
            s.diff[2],
            s.diff[3],
            s.diff[4],
            s.diff[5],
            s.diff[6],
            s.diff[7]
        );
    }
}

/// Whether the row plan is checked against the first one seen. `INK_PLAN_CHECK=1`.
fn plan_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_PLAN_CHECK")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// The run-scoped half of the device row plan.
///
/// Six of [`RowPlan`]'s seven fields are the same bytes on every layer of every
/// decode pass at `n == 1` — measured, not assumed; see
/// [`mary::models::inkling::devplan`] and `INK_PLAN_CHECK=1`. So they are built
/// once, uploaded once, and held here for the run, and the per-layer kernel
/// produces only the seventh.
struct DevRoute {
    /// `[k * MTILE]` i32, `0` at every tile start and `-1` in the padding.
    row_tok: cubecl::server::Handle,
    /// `[k]` u32, the identity: one expert per block.
    blk_slot: cubecl::server::Handle,
    /// `[k]` u32, the identity.
    blk_tile0: cubecl::server::Handle,
    /// `[k]` u32, all ones: one M tile per expert, so `INK_MOE_PLANES` has
    /// nothing to group at this width.
    blk_cnt: cubecl::server::Handle,
    /// `[1, k]` u32, `[0, 16, 32, …]`.
    tok_rows: cubecl::server::Handle,
    /// `[1]` u32, `[k]`.
    tok_cnt: cubecl::server::Handle,
    /// One u32 for the WHOLE RUN. The kernel raises it; the host reads it once,
    /// after the last pass. A per-layer read of it would be the read this
    /// entire lane exists to delete.
    fault: cubecl::server::Handle,
    /// Per absolute layer: its weight table, or `None` for a layer this lane
    /// cannot take (BF16 experts, no registered mapping, a misaligned plane).
    /// The `None` is cached too — a layer that refused once refuses every pass,
    /// and re-deriving that costs 1024 lookups.
    tabs: std::collections::HashMap<usize, Option<mary::models::inkling::devplan::ExpertTable>>,
    /// Expert SLOTS in the plan, which is also the block count: `n * top_k`.
    ///
    /// It was called `k` while it could only be `top_k`, and the rename is the
    /// whole `n > 1` change stated once. At `n == 1` it is still `top_k` and the
    /// lane is byte-identical to what it was.
    k: usize,
    /// `top_k`: the number of experts ONE token routes to, which is `RowPlan`'s
    /// `kmax` and is NOT the slot count once `n > 1`.
    kmax: usize,
    /// The row count these invariants were derived at. They are a function of
    /// `n` and `top_k` alone (see [`devroute_new`]), so a pass at a different
    /// width needs a different set and this is what notices.
    n: usize,
    /// `k * MTILE`.
    m_total: usize,
    /// What `RowPlan::planes()` said, carried so the launch shape matches the
    /// host lane's exactly.
    planes: usize,
}

/// Build [`DevRoute`]'s invariants — from [`RowPlan::build`] itself, at the
/// shape a one-token pass produces.
///
/// Derived rather than transcribed on purpose: if the stacking rule ever
/// changes, this follows it instead of disagreeing with it silently.
fn devroute_new(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    k: usize,
    n: usize,
) -> DevRoute {
    use mary::models::inkling::moegroup::RowPlan;
    // `n * k` SLOTS of one token each: slot `s` carries row `s / k`'s
    // `s % k`-th pick. Token index, not weight -- the weights are exactly the
    // field this does NOT hoist.
    //
    // ## Why one slot per (token, pick) and not one per DISTINCT expert
    //
    // The host lane dedups: two tokens that pick the same expert share a tile,
    // and the expert's plane is read once. That saves bandwidth and it makes
    // the plan's SHAPE data-dependent -- `blk_slot` changes length with the
    // number of distinct experts, and the module doc records exactly that
    // happening at `n == 5`, where it moved between 17 and 24 from one layer to
    // the next. A shape that changes per layer cannot be hoisted, and hoisting
    // is the entire lane: it is what lets six of `RowPlan`'s seven fields be
    // uploaded once instead of read back every layer.
    //
    // Not deduping restores the property at every `n`: `n * k` slots, one token
    // each, every field but `row_wgt` and the offsets a function of `n` and `k`.
    // It costs the re-read of a shared expert's plane. MEASURED on this box at
    // `n = 2`, the log's own counter: 114 slabs a pass deduped, ~200 as the
    // router actually routes, 228 without dedup -- so the charge is the 28-slab
    // difference, about 0.40 GiB a pass and ~2 ms at this part's achievable
    // bandwidth, against a readback whose removal is worth 8.1 ms at the same
    // width (`INK_ROUTE_STALE=1` against base, 72.2 -> 64.1 ms/step, two
    // interleaved reps, layers 0:21, ctx 3784).
    //
    // The per-token ACCUMULATION ORDER is unchanged, which is the part that is
    // not a trade: a token's picks land in its own `k` slots in ascending
    // expert id, so `tok_rows` walks them in the same order the host's
    // `BTreeMap` did.
    let each: Vec<Vec<(usize, f32)>> = (0..n * k).map(|s| vec![(s / k, 0.0f32)]).collect();
    let plan = RowPlan::build(each.iter(), n, RowPlan::planes());
    assert_eq!(
        plan.kmax, k,
        "a token routed to {k} picks has kmax {k}, whatever {n} is"
    );
    assert_eq!(
        plan.blk_slot.len(),
        n * k,
        "one tile a slot is one block a slot"
    );
    DevRoute {
        row_tok: client.create_from_slice(bytes_of(&plan.row_tok)),
        blk_slot: client.create_from_slice(bytes_of(&plan.blk_slot)),
        blk_tile0: client.create_from_slice(bytes_of(&plan.blk_tile0)),
        blk_cnt: client.create_from_slice(bytes_of(&plan.blk_cnt)),
        tok_rows: client.create_from_slice(bytes_of(&plan.tok_rows)),
        tok_cnt: client.create_from_slice(bytes_of(&plan.tok_cnt)),
        fault: client.create_from_slice(&0u32.to_le_bytes()),
        tabs: std::collections::HashMap::new(),
        k: n * k,
        kmax: k,
        n,
        m_total: plan.m_total(),
        planes: RowPlan::planes(),
    }
}

/// One layer's weight table over EVERY routed expert, or `None` if this lane
/// cannot take the layer.
///
/// The same four numbers per expert the host lane derived per PASS, derived
/// once instead — and for all `n_routed` of them rather than for the six that
/// happened to be active, because nothing on the host knows which six any more.
/// The refusals are the grouped lane's own, unchanged and applied to the whole
/// layer instead of to the active slice: one mapping for every plane, and every
/// offset on the 4-byte vector the packed planes are read in.
fn build_expert_table(
    src: &Weights,
    al: &mary::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    prefix: &str,
    n_routed: usize,
) -> Result<Option<mary::models::inkling::devplan::ExpertTable>> {
    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut off13: Vec<u64> = Vec::with_capacity(2 * n_routed);
    let mut off2: Vec<u64> = Vec::with_capacity(2 * n_routed);
    let mut sc13: Vec<f32> = Vec::with_capacity(n_routed);
    let mut sc2: Vec<f32> = Vec::with_capacity(n_routed);
    let mut which: Option<usize> = None;
    let mut expert_bytes = 0usize;
    for e in 0..n_routed {
        let w13 = src.expert_packed(&n13, e)?;
        let w2 = src.expert_packed(&n2, e)?;
        let planes: [&[u8]; 4] = [&w13.codes, &w13.scales, &w2.codes, &w2.scales];
        let mut o = [0u64; 4];
        let mut bytes = 0usize;
        for (i, plane) in planes.into_iter().enumerate() {
            match al.locate(plane) {
                Some((m, byte)) if which.map_or(true, |w| w == m) => {
                    which = Some(m);
                    o[i] = byte;
                }
                _ => return Ok(None),
            }
            bytes += plane.len();
        }
        if o.iter().any(|v| v % 4 != 0) {
            return Ok(None);
        }
        // Every expert of a layer has the same shape, so this is the exact
        // per-expert figure the alias accounting needs and not an average. A
        // layer where it is not constant is a layer this lane has no business
        // charging for, so it refuses instead.
        if e == 0 {
            expert_bytes = bytes;
        } else if bytes != expert_bytes {
            return Ok(None);
        }
        off13.push(o[0]);
        off13.push(o[1]);
        off2.push(o[2]);
        off2.push(o[3]);
        sc13.push(w13.scale2);
        sc2.push(w2.scale2);
    }
    let (wmap, wmap_bytes) = match which.and_then(|i| al.map(i)) {
        Some(m) => m,
        None => return Ok(None),
    };
    Ok(Some(mary::models::inkling::devplan::ExpertTable {
        off13: client.create_from_slice(bytes_of(&off13)),
        off2: client.create_from_slice(bytes_of(&off2)),
        sc13: client.create_from_slice(bytes_of(&sc13)),
        sc2: client.create_from_slice(bytes_of(&sc2)),
        wmap,
        wmap_bytes,
        expert_bytes,
        n_routed,
        stride: 2,
        scaled: true,
    }))
}

/// [`build_expert_table`] for a layer nothing quantised.
///
/// One plane a matrix instead of two, an offset in BF16 ELEMENTS rather than
/// bytes — the unit `b` is indexed in, converted here exactly as
/// `grouped_experts_bf16` converts it — and no second-level scale at all. The
/// scale vectors are still allocated, filled with zeros, so the plan kernel
/// needs no second form; the GEMM this feeds never reads them.
fn build_expert_table_bf16(
    src: &Weights,
    al: &mary::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    prefix: &str,
    n_routed: usize,
) -> Result<Option<mary::models::inkling::devplan::ExpertTable>> {
    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut off13: Vec<u64> = Vec::with_capacity(n_routed);
    let mut off2: Vec<u64> = Vec::with_capacity(n_routed);
    let mut which: Option<usize> = None;
    let mut expert_bytes = 0usize;
    for e in 0..n_routed {
        let w13 = src.expert_bf16(&n13, e)?;
        let w2 = src.expert_bf16(&n2, e)?;
        let planes: [&[u8]; 2] = [&w13.bytes, &w2.bytes];
        let mut o = [0u64; 2];
        let mut bytes = 0usize;
        for (i, plane) in planes.into_iter().enumerate() {
            match al.locate(plane) {
                Some((m, byte)) if which.map_or(true, |w| w == m) => {
                    which = Some(m);
                    o[i] = byte;
                }
                _ => return Ok(None),
            }
            bytes += plane.len();
        }
        if o[0] % 4 != 0 || o[1] % 4 != 0 {
            return Ok(None);
        }
        if e == 0 {
            expert_bytes = bytes;
        } else if bytes != expert_bytes {
            return Ok(None);
        }
        off13.push(o[0] / 2);
        off2.push(o[1] / 2);
    }
    let (wmap, wmap_bytes) = match which.and_then(|i| al.map(i)) {
        Some(m) => m,
        None => return Ok(None),
    };
    let zeros = vec![0f32; n_routed];
    Ok(Some(mary::models::inkling::devplan::ExpertTable {
        off13: client.create_from_slice(bytes_of(&off13)),
        off2: client.create_from_slice(bytes_of(&off2)),
        sc13: client.create_from_slice(bytes_of(&zeros)),
        sc2: client.create_from_slice(bytes_of(&zeros)),
        wmap,
        wmap_bytes,
        expert_bytes,
        n_routed,
        stride: 1,
        scaled: false,
    }))
}

/// The routed experts of one layer with the plan already on the device.
///
/// The sibling of [`grouped_experts_fp4`] whose whole difference is upstream:
/// eleven buffers instead of nine uploads, and no `by_expert` at all. Both end
/// in [`grouped_experts_core`], so the arithmetic is one implementation.
#[allow(clippy::too_many_arguments)]
fn routed_experts_fp4_dev(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    tab: &mary::models::inkling::devplan::ExpertTable,
    dp: &mary::models::inkling::devplan::DevRowPlan,
    dr: &DevRoute,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    swz: bool,
    t_g: Instant,
    host: &mut HostT,
) -> T2 {
    use mary::models::inkling::moegroup::BlockPlanDev;
    use mary::models::inkling::seam::handle_of_any;
    let blk = BlockPlanDev {
        slot: dr.blk_slot.clone(),
        tile0: dr.blk_tile0.clone(),
        cnt: dr.blk_cnt.clone(),
        blocks: dr.k,
        planes: dr.planes,
        // One token an expert, so every stacked row past the first of a tile is
        // padding. This is what picks the schedule, and getting it wrong would
        // pick the prefill's.
        rows_real: dr.k,
    };
    let (hn_h, hn_dt) = handle_of_any(hn.clone());
    grouped_experts_core(
        client,
        dev,
        prefix,
        &tab.wmap,
        tab.wmap_bytes,
        &blk,
        &hn_h,
        hn_dt,
        &dr.row_tok,
        &dp.row_wgt,
        &dr.tok_rows,
        &dr.tok_cnt,
        &dp.off13,
        &dp.off2,
        &dp.sc13,
        &dp.sc2,
        dr.k,
        dr.m_total,
        dr.kmax,
        n,
        h,
        inter,
        swz,
        t_g,
        host,
    )
}

/// The routed experts of one BF16 layer with the plan already on the device.
#[allow(clippy::too_many_arguments)]
fn routed_experts_bf16_dev(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    tab: &mary::models::inkling::devplan::ExpertTable,
    dp: &mary::models::inkling::devplan::DevRowPlan,
    dr: &DevRoute,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    t_g: Instant,
    host: &mut HostT,
) -> T2 {
    use mary::models::inkling::moegroup::BlockPlanDev;
    use mary::models::inkling::seam::handle_of;
    let blk = BlockPlanDev {
        slot: dr.blk_slot.clone(),
        tile0: dr.blk_tile0.clone(),
        cnt: dr.blk_cnt.clone(),
        blocks: dr.k,
        planes: dr.planes,
        rows_real: dr.k,
    };
    // Widened for the same reason the host twin widens it: this lane's gather
    // and scatter index f32 bytes.
    let hn_h = handle_of(mary::models::inkling::resid::from_resid(hn.clone()));
    grouped_experts_bf16_core(
        client,
        dev,
        &tab.wmap,
        tab.wmap_bytes,
        &blk,
        &hn_h,
        &dr.row_tok,
        &dp.row_wgt,
        &dr.tok_rows,
        &dr.tok_cnt,
        &dp.off13,
        &dp.off2,
        dr.k,
        dr.m_total,
        dr.kmax,
        n,
        h,
        inter,
        t_g,
        host,
    )
}

/// [`shared_experts_bf16`] with the gammas left where the router put them.
///
/// The host twin takes `&[f32]` and builds an `[n, 1]` tensor per shared
/// expert; this slices the same column out of `routetopk`'s own output. Same
/// f32 values, same multiply, same order — the readback was the only thing
/// between them.
fn shared_experts_dev(x: T2, sw: &SharedOnDevice, topk: T2, top_k: usize, n_shared: usize) -> T2 {
    let [n, _] = x.dims();
    let inter = sw.gate_up.n() / (2 * n_shared);
    let gu = dev_lane::linear_w(x, &sw.gate_up);
    let mut out: Option<T2> = None;
    for s in 0..n_shared {
        let g = gu.clone().slice([0..n, s * inter..(s + 1) * inter]);
        let u = gu
            .clone()
            .slice([0..n, (n_shared + s) * inter..(n_shared + s + 1) * inter]);
        let gam = topk.clone().slice([0..n, 2 * top_k + s..2 * top_k + s + 1]);
        let c = dev_lane::linear_w(dev_lane::silu(g) * u * gam, &sw.down[s]);
        out = Some(match out {
            Some(o) => o + c,
            None => c,
        });
    }
    out.expect("a MoE layer with no shared experts")
}

/// `INK_DEVPLAN_CHECK=1`: the device plan against the host plan, as BITS.
///
/// The sharpest instrument this lane has, and the one the ascending sort needs.
/// A mis-sorted expert list changes the ORDER the scatter accumulates a token's
/// six contributions in, which changes the sum in the last few ulps and nowhere
/// else — and this runtime disagrees with itself on 8.55% of argmax positions
/// between two runs of the same binary, so no output-level comparison could
/// ever see it. This one compares the plan itself, where the difference is
/// exact.
///
/// Reads five device buffers, so it is far slower than either lane and is a
/// diagnostic rather than an arm.
#[allow(clippy::too_many_arguments)]
fn devplan_verify_layer(
    src: &Weights,
    al: &mary::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    prefix: &str,
    routing: &[Routing],
    dp: &mary::models::inkling::devplan::DevRowPlan,
    dr: &DevRoute,
    scaled: bool,
) -> Result<()> {
    use mary::models::inkling::fp4gemm::MTILE;
    // The expected plan, from the ROUTING rather than from `by_expert`.
    //
    // It used to be built from the host's `BTreeMap`, which is the deduplicated
    // expert set, and that is the same thing as "each row's picks, ascending"
    // only while there is one row. `devroute_new` no longer dedups, so the two
    // part company at `n > 1` and the routing is the one that still says what
    // the plan must be. At `n == 1` this builds exactly what the `BTreeMap`
    // form built, so the check did not get weaker where it already worked.
    let k = dr.kmax;
    anyhow::ensure!(
        routing.len() == dr.n,
        "{prefix}: the plan was derived at {} rows and the pass routed {}",
        dr.n,
        routing.len()
    );
    let mut want_ids: Vec<u32> = Vec::with_capacity(dr.n * k);
    let mut want_wgt: Vec<f32> = vec![0.0f32; dr.n * k * MTILE];
    let mut picks: Vec<usize> = Vec::with_capacity(dr.n * k);
    for (t, rt) in routing.iter().enumerate() {
        anyhow::ensure!(
            rt.experts.len() == k,
            "{prefix}: row {t} routed to {} experts and the plan holds {k} a row",
            rt.experts.len()
        );
        let mut row: Vec<(usize, f32)> = rt
            .experts
            .iter()
            .copied()
            .zip(rt.weights.iter().copied())
            .collect();
        // ASCENDING EXPERT ID, which is the order the scatter accumulates this
        // token's contributions in and therefore the order the sum is defined
        // by. `sort_by_key` is stable and the ids within a row are distinct, so
        // there is no tie to break.
        row.sort_by_key(|&(e, _)| e);
        for (j, &(e, w)) in row.iter().enumerate() {
            want_ids.push(e as u32);
            want_wgt[(t * k + j) * MTILE] = w;
            picks.push(e);
        }
    }
    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut off13: Vec<u64> = Vec::new();
    let mut off2: Vec<u64> = Vec::new();
    let mut sc13: Vec<f32> = Vec::new();
    let mut sc2: Vec<f32> = Vec::new();
    for &e in picks.iter() {
        if scaled {
            let w13 = src.expert_packed(&n13, e)?;
            let w2 = src.expert_packed(&n2, e)?;
            let planes: [&[u8]; 4] = [&w13.codes, &w13.scales, &w2.codes, &w2.scales];
            for (i, plane) in planes.into_iter().enumerate() {
                let (_, byte) = al
                    .locate(plane)
                    .context("the host plan cannot locate a plane")?;
                match i {
                    0 | 1 => off13.push(byte),
                    _ => off2.push(byte),
                }
            }
            sc13.push(w13.scale2);
            sc2.push(w2.scale2);
        } else {
            let w13 = src.expert_bf16(&n13, e)?;
            let w2 = src.expert_bf16(&n2, e)?;
            let (_, b13) = al
                .locate(&w13.bytes)
                .context("the host plan cannot locate a plane")?;
            let (_, b2) = al
                .locate(&w2.bytes)
                .context("the host plan cannot locate a plane")?;
            off13.push(b13 / 2);
            off2.push(b2 / 2);
        }
    }
    let rd = |hnd: &cubecl::server::Handle| -> Vec<u8> {
        client
            .read_one(hnd.clone())
            .expect("read a device plan buffer")
            .to_vec()
    };
    let d_wgt = rd(&dp.row_wgt);
    let d_o13 = rd(&dp.off13);
    let d_o2 = rd(&dp.off2);
    let d_s13 = rd(&dp.sc13);
    let d_s2 = rd(&dp.sc2);
    let bad = |what: &str, host: &[u8], got: &[u8]| -> Result<()> {
        anyhow::ensure!(
            host == got,
            "DEVPLAN MISMATCH at {prefix}: {what} differs between the host plan and the device \
             plan. host {host:02x?} device {got:02x?}. The picks the routing made, row by row \
             and ascending within a row, were {picks:?}."
        );
        Ok(())
    };
    anyhow::ensure!(
        picks.len() == dr.k,
        "{prefix} routed {} (row, pick) pairs, and the device plan is built for exactly {}",
        picks.len(),
        dr.k
    );
    // THE ORDER, as integers, before anything derived from it. A mis-sorted
    // plan perturbs the accumulated sum at exactly the magnitude the fused
    // multiply does, so no floating-point comparison downstream can tell the
    // two apart by size -- only by how MANY elements moved, which is a
    // statistic and not an answer. This is the answer: six ids against six
    // keys, and a failure prints the permutation.
    let d_ids = rd(&dp.ids);
    let got_ids: Vec<u32> = d_ids
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("four bytes")))
        .collect();
    anyhow::ensure!(
        got_ids == want_ids,
        "DEVPLAN ORDER MISMATCH at {prefix}: the device plan stacked the experts as {got_ids:?} \
         and the routing, row by row and ascending within a row, is {want_ids:?}. These must be \
         identical -- the order is the order the scatter accumulates a token's contributions in, \
         and floating-point addition is not associative."
    );
    bad("row_wgt", bytes_of(&want_wgt), &d_wgt)?;
    bad("off13", bytes_of(&off13), &d_o13)?;
    bad("off2", bytes_of(&off2), &d_o2)?;
    if scaled {
        bad("sc13", bytes_of(&sc13), &d_s13)?;
        bad("sc2", bytes_of(&sc2), &d_s2)?;
    }
    Ok(())
}

/// Where the grouped lane's row plan is built. `INK_DEV_PLAN`.
///
/// `Host` is the lane that reads the router's decision back every layer;
/// `Dev` never reads it; `Ab(r)` alternates every `r` decode passes so the two
/// can be priced inside ONE process. The last is the only honest form of the
/// comparison: pass-to-pass drift on this box is 2-3 ms against a difference of
/// about four, so two separate runs cannot resolve it.
#[derive(Clone, Copy, PartialEq)]
enum PlanArm {
    Host,
    Dev,
    Ab(usize),
}

impl PlanArm {
    /// `INK_DEV_PLAN`: unset or `0` is the host lane, `1`/`on` the device one,
    /// `ab:<r>` the interleave.
    ///
    /// **Default ON as of 2026-08-25.** It was off on the rule that "this lane
    /// is newer than its measurement, and the arm that has run for months is
    /// the one a run that says nothing should get". It now HAS its
    /// measurement, so the rule points the other way.
    ///
    /// spark-zt (GB10), cached decode lane (one row a step), `INK_LAYERS=0:16`,
    /// 3732-token cover, `INK_GEN=12`, first two passes of each rep discarded,
    /// arms INTERLEAVED, median over reps:
    ///
    /// * **5 of 5** interleaved pairs favour it, by 2.8-5.5 ms
    /// * 59.8 -> 55.2 ms a step, **+8.33%**, against base's own 4.3% spread
    /// * it HALVES the spread, to 1.6%, because the sync it removes is the
    ///   jittery part -- corroboration independent of the median
    /// * token stream IDENTICAL: one md5 over all twelve steps, eight runs of
    ///   each arm across two rounds, no fault flag in any run
    ///
    /// What it removes is 3.5 ms of true serialisation, and that figure is
    /// bounded rather than inferred: `INK_ROUTE_STALE=1` (a probe) and this
    /// lane remove the read by different means and land on the SAME 54.5 ms a
    /// step. Their agreeing is the result -- there is nothing further in the
    /// router. The `router + group` bracket is 84% genuine GPU work waited on,
    /// not stall, which is why a bracket that reads as 41% of the wall step is
    /// worth 3.5 ms and not 24.
    fn from_env() -> Result<PlanArm> {
        match std::env::var("INK_DEV_PLAN") {
            Err(_) => Ok(PlanArm::Dev),
            Ok(v) if v == "0" => Ok(PlanArm::Host),
            Ok(v) if v == "1" || v == "on" => Ok(PlanArm::Dev),
            Ok(v) => match v.strip_prefix("ab:").and_then(|r| r.parse::<usize>().ok()) {
                Some(r) if r >= 1 => Ok(PlanArm::Ab(r)),
                _ => anyhow::bail!(
                    "INK_DEV_PLAN={v}: expected 0, 1, or ab:<round length in decode passes>"
                ),
            },
        }
    }

    /// Whether THIS decode pass builds its plan on the device.
    fn on(&self, decode_step: usize) -> bool {
        match self {
            PlanArm::Host => false,
            PlanArm::Dev => true,
            // The host arm goes first, so the cold passes -- kernel
            // compilation, the first touch of every weight table -- land on the
            // lane that has to pay them anyway.
            PlanArm::Ab(r) => (decode_step / r) % 2 == 1,
        }
    }
}

/// Whether the routed-expert lane reports what it is holding. `INK_MEM_TRACE=1`.
///
/// Read once: this sits inside the per-layer path.
fn mem_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_MEM_TRACE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// A slice of POD as the bytes `create_from_slice` uploads.
fn bytes_of<T: cubecl::prelude::CubeElement>(v: &[T]) -> &[u8] {
    T::as_bytes(v)
}

/// The per-expert lane: one launch sequence per active expert, in `BTreeMap`
/// order, which is the order the accumulation is defined by.
#[allow(clippy::too_many_arguments)]
fn per_expert_fp4(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<T2> {
    use mary::models::inkling::fp4gemm::{
        MTILE, fp4_linear_launch, fp4_linear_swz_launch, gate_up_silu_launch,
    };
    use mary::models::inkling::fp4quant::quantize_nvfp4;
    use mary::models::inkling::pad::gather_rows_pad;
    use mary::models::inkling::seam::{handle_of, int_handle_of, tensor_of};

    // Which layout the arena holds. Read ONCE: it is a fact about the startup
    // copy, and a per-expert question would suggest it could differ per expert.
    let swz = src.experts_swizzled();

    // Zero copy where the hardware allows it: the GPU reads the source's own
    // mapped pages in place. The mappings were registered ONCE at startup, so
    // this is offset arithmetic on a pointer, not a device round trip.
    let bind = |data: &[u8]| match aliases {
        Some(al) => al.slice_or_copy(client, data),
        None => client.create_from_slice(data),
    };

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut acc: T2 = burn::tensor::Tensor::zeros([n, h], dev);
    // The residual stream's buffer, taken once: every expert gathers out of the
    // same rows and re-deriving the handle per expert would be six clones of a
    // refcount for nothing.
    // A lane that is not the default one, reading through kernels that index
    // f32 bytes: widen the normed stream here rather than teach four fallback
    // gathers a dtype they will never see on the lane that runs.
    let hn_h = handle_of(mary::models::inkling::resid::from_resid(hn.clone()));

    for (&e, toks) in by_expert {
        let t_s = Instant::now();
        let w13 = src.expert_packed(&n13, e)?;
        let w2 = src.expert_packed(&n2, e)?;
        host.slice += t_s.elapsed().as_secs_f64();

        let t_g = Instant::now();
        let m = toks.len();
        let m_pad = m.div_ceil(MTILE) * MTILE;
        let (idx, wgt) = expert_rows::<Bk>(toks, dev);
        // One kernel reads this expert's rows out of the residual stream where
        // they already lie and writes them into the `[m_pad, hidden]` buffer the
        // MMA wants, zeros and all. It was a `select`, a `zeros` and a `cat` —
        // four launches to move one row on a decode step.
        let x_h = gather_rows_pad(client, &hn_h, &int_handle_of(idx.clone()), n, m, m_pad, h);
        host.gather += t_g.elapsed().as_secs_f64();

        let t_w = Instant::now();
        // Quantise on the DEVICE, both times. The host lane this replaces had
        // to bring the intermediate activation back across the bus between the
        // two GEMMs purely to requantise it; `act_h` never leaves the device.
        let (a, asc) = quantize_nvfp4(client, &x_h, m_pad, h);

        // The FALLBACK lane reads the same arena the grouped one does, so it
        // reads whatever layout the startup copy wrote. `fp4_linear_swz` is
        // `fp4_linear` with one index expression changed and is bit-identical
        // given the permuted operand -- which is what makes this a choice of
        // READER rather than a second implementation to keep in step.
        let (b, bsc) = (bind(&w13.codes), bind(&w13.scales));
        let both = if swz {
            fp4_linear_swz_launch(
                client,
                &a,
                &asc,
                &b,
                &bsc,
                m_pad,
                h,
                2 * inter,
                w13.scale2,
                true,
            )
        } else {
            fp4_linear_launch(client, &a, &asc, &b, &bsc, m_pad, h, 2 * inter, w13.scale2)
        };

        let act_h = gate_up_silu_launch(client, &both, m_pad, inter);
        let (a2, asc2) = quantize_nvfp4(client, &act_h, m_pad, inter);

        let (b2, bsc2) = (bind(&w2.codes), bind(&w2.scales));
        let y_h = if swz {
            fp4_linear_swz_launch(
                client, &a2, &asc2, &b2, &bsc2, m_pad, inter, h, w2.scale2, true,
            )
        } else {
            fp4_linear_launch(client, &a2, &asc2, &b2, &bsc2, m_pad, inter, h, w2.scale2)
        };
        host.enqueue += t_w.elapsed().as_secs_f64();

        let t_c = Instant::now();
        let y = tensor_of(client.clone(), dev.clone(), y_h, m_pad, h);
        let y = y.slice([0..m, 0..h]) * wgt;
        acc = acc.select_assign(0, idx, y, burn::tensor::IndexingUpdateOp::Add);
        host.accum += t_c.elapsed().as_secs_f64();
    }
    Ok(acc)
}

/// The same lane for layer 2, whose experts are BF16.
///
/// Deliberately the same shape as [`routed_experts_fp4`], line for line — and
/// that doc comment is where the algorithm is written down, since the host
/// transcription that used to hold it is deleted. The
/// same `select` gather, the same pointer-containment binding, the same
/// `select_assign` in `BTreeMap` order. What the format takes away is all that
/// differs — no block scales to bind, no `scale2` to fold in, no activation
/// quantiser. What it puts back is one cast: the MMA takes the same type on
/// both operands, so the f32 residual stream is rounded to BF16 on the device
/// before it enters. That is not a liberty — the reference implementation runs
/// this layer in BF16 throughout, so a BF16 activation is what `transformers`
/// multiplies too.
#[allow(clippy::too_many_arguments)]
fn routed_experts_bf16(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<T2> {
    // Same dispatch as the packed lane, and it earns its keep on the same
    // measurement: eight grouped NVFP4 layers cost the host 0.8 ms a pass and
    // this ONE layer, still looping, cost about eight.
    let mode = std::env::var("INK_GROUPED").unwrap_or_else(|_| "1".to_string());
    if mode != "0" {
        if let Some(al) = aliases {
            if let Some(acc) = grouped_experts_bf16(
                src, al, client, dev, prefix, by_expert, hn, n, h, inter, host,
            )? {
                if mode == "2" {
                    let reference = per_expert_bf16(
                        src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host,
                    )?;
                    report_ab(prefix, &acc, &reference, h);
                }
                host.grouped += 1;
                host.expert_slots += by_expert.len();
                return Ok(acc);
            }
        }
    }
    host.per_expert += 1;
    host.expert_slots += by_expert.len();
    per_expert_bf16(
        src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host,
    )
}

/// Layer 2's routed experts in a handful of launches, or `None` if this lane
/// cannot take the layer.
///
/// [`grouped_experts_fp4`] with the format's differences and no others: one
/// weight plane per expert instead of two, no second-level scale to carry, and
/// a cast in place of the activation quantiser. The offsets are BF16 elements
/// because that is the unit the unscaled MMA indexes its B operand in.
#[allow(clippy::too_many_arguments)]
fn grouped_experts_bf16(
    src: &Weights,
    al: &mary::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<Option<T2>> {
    use mary::models::inkling::moegroup::{BlockPlanDev, RowPlan};
    use mary::models::inkling::seam::handle_of;

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let slots = by_expert.len();
    if slots == 0 {
        return Ok(None);
    }

    let t_s = Instant::now();
    let mut off13: Vec<u64> = Vec::with_capacity(slots);
    let mut off2: Vec<u64> = Vec::with_capacity(slots);
    let mut plane_bytes: Vec<usize> = Vec::with_capacity(2 * slots);
    let mut which: Option<usize> = None;
    for &e in by_expert.keys() {
        let w13 = src.expert_bf16(&n13, e)?;
        let w2 = src.expert_bf16(&n2, e)?;
        let planes: [&[u8]; 2] = [&w13.bytes, &w2.bytes];
        let mut o = [0u64; 2];
        for (i, plane) in planes.into_iter().enumerate() {
            match al.locate(plane) {
                Some((m, byte)) if which.map_or(true, |w| w == m) => {
                    which = Some(m);
                    o[i] = byte;
                }
                _ => return Ok(None),
            }
            plane_bytes.push(plane.len());
        }
        // Read as two BF16 per 32-bit vector, so the offset has to be a whole
        // number of them -- which is the same 4-byte rule the alias predicate
        // already applies to the pointer.
        if o[0] % 4 != 0 || o[1] % 4 != 0 {
            return Ok(None);
        }
        off13.push(o[0] / 2);
        off2.push(o[1] / 2);
    }
    let (wmap, wmap_bytes) = match which.and_then(|i| al.map(i)) {
        Some(m) => m,
        None => return Ok(None),
    };
    for b in plane_bytes {
        al.note_alias(b);
    }
    host.slice += t_s.elapsed().as_secs_f64();

    let t_g = Instant::now();
    let plan = RowPlan::build(by_expert.values(), n, RowPlan::planes());
    if std::env::var("INK_MOE_DEBUG").is_ok() {
        eprintln!(
            "MOEPLAN {prefix} slots={} rows={} tiles={} blocks={}",
            by_expert.len(),
            plan.m_total(),
            plan.m_total() / 16,
            plan.blk_slot.len()
        );
    }
    let m_total = plan.m_total();
    // A lane that is not the default one, reading through kernels that index
    // f32 bytes: widen the normed stream here rather than teach four fallback
    // gathers a dtype they will never see on the lane that runs.
    let hn_h = handle_of(mary::models::inkling::resid::from_resid(hn.clone()));
    let h_rowtok = client.create_from_slice(bytes_of(&plan.row_tok));
    let h_rowwgt = client.create_from_slice(bytes_of(&plan.row_wgt));
    let blk = BlockPlanDev {
        slot: client.create_from_slice(bytes_of(&plan.blk_slot)),
        tile0: client.create_from_slice(bytes_of(&plan.blk_tile0)),
        cnt: client.create_from_slice(bytes_of(&plan.blk_cnt)),
        blocks: plan.blk_slot.len(),
        planes: RowPlan::planes(),
        rows_real: plan.rows_real(),
    };
    let h_off13 = client.create_from_slice(bytes_of(&off13));
    let h_off2 = client.create_from_slice(bytes_of(&off2));
    let h_tokrows = client.create_from_slice(bytes_of(&plan.tok_rows));
    let h_tokcnt = client.create_from_slice(bytes_of(&plan.tok_cnt));
    Ok(Some(grouped_experts_bf16_core(
        client, dev, &wmap, wmap_bytes, &blk, &hn_h, &h_rowtok, &h_rowwgt, &h_tokrows, &h_tokcnt,
        &h_off13, &h_off2, slots, m_total, plan.kmax, n, h, inter, t_g, host,
    )))
}
/// The grouped BF16 lane once its plan exists, whoever built it.
///
/// [`grouped_experts_core`]'s unquantised sibling — layer 2 and nothing else.
/// One offset an expert instead of two, no second-level scale, and an f32
/// stream throughout, because nothing here reads four-bit codes.
#[allow(clippy::too_many_arguments)]
fn grouped_experts_bf16_core(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    wmap: &cubecl::server::Handle,
    wmap_bytes: usize,
    blk: &mary::models::inkling::moegroup::BlockPlanDev,
    hn_h: &cubecl::server::Handle,
    h_rowtok: &cubecl::server::Handle,
    h_rowwgt: &cubecl::server::Handle,
    h_tokrows: &cubecl::server::Handle,
    h_tokcnt: &cubecl::server::Handle,
    h_off13: &cubecl::server::Handle,
    h_off2: &cubecl::server::Handle,
    slots: usize,
    m_total: usize,
    kmax: usize,
    n: usize,
    h: usize,
    inter: usize,
    t_g: Instant,
    host: &mut HostT,
) -> T2 {
    use mary::models::inkling::bf16gemm::to_bf16_launch;
    use mary::models::inkling::fp4gemm::gate_up_silu_bf16_launch;
    use mary::models::inkling::moegroup::{
        bf16_linear_grouped_launch, gather_grouped, scatter_weighted,
    };
    use mary::models::inkling::seam::tensor_of;

    let x_h = gather_grouped(client, hn_h, h_rowtok, n, m_total, h);
    host.gather += t_g.elapsed().as_secs_f64();

    let t_w = Instant::now();
    let a = to_bf16_launch(client, &x_h, m_total * h, m_total * h);
    let both = bf16_linear_grouped_launch(
        client,
        &a,
        wmap,
        wmap_bytes,
        blk,
        h_off13,
        slots,
        m_total,
        h,
        2 * inter,
    );
    let act = gate_up_silu_bf16_launch(client, &both, m_total, inter);
    let y_h = bf16_linear_grouped_launch(
        client, &act, wmap, wmap_bytes, blk, h_off2, slots, m_total, inter, h,
    );
    host.enqueue += t_w.elapsed().as_secs_f64();

    let t_c = Instant::now();
    let acc_h = scatter_weighted(
        client, &y_h, h_rowwgt, h_tokrows, h_tokcnt, m_total, n, h, kmax,
    );
    let acc = tensor_of(client.clone(), dev.clone(), acc_h, n, h);
    host.accum += t_c.elapsed().as_secs_f64();

    acc
}

/// The per-expert BF16 lane: one launch sequence per active expert, in
/// `BTreeMap` order.
#[allow(clippy::too_many_arguments)]
fn per_expert_bf16(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<T2> {
    use mary::models::inkling::bf16gemm::{MTILE, bf16_linear_launch, to_bf16_launch};
    use mary::models::inkling::fp4gemm::gate_up_silu_bf16_launch;
    use mary::models::inkling::pad::gather_rows_pad;
    use mary::models::inkling::seam::{handle_of, int_handle_of, tensor_of};

    let bind = |data: &[u8]| match aliases {
        Some(al) => al.slice_or_copy(client, data),
        None => client.create_from_slice(data),
    };

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut acc: T2 = burn::tensor::Tensor::zeros([n, h], dev);
    // A lane that is not the default one, reading through kernels that index
    // f32 bytes: widen the normed stream here rather than teach four fallback
    // gathers a dtype they will never see on the lane that runs.
    let hn_h = handle_of(mary::models::inkling::resid::from_resid(hn.clone()));

    for (&e, toks) in by_expert {
        let t_s = Instant::now();
        let w13 = src.expert_bf16(&n13, e)?;
        let w2 = src.expert_bf16(&n2, e)?;
        host.slice += t_s.elapsed().as_secs_f64();

        let t_g = Instant::now();
        let m = toks.len();
        let m_pad = m.div_ceil(MTILE) * MTILE;
        let (idx, wgt) = expert_rows::<Bk>(toks, dev);
        let x_h = gather_rows_pad(client, &hn_h, &int_handle_of(idx.clone()), n, m, m_pad, h);
        host.gather += t_g.elapsed().as_secs_f64();

        let t_w = Instant::now();
        let a = to_bf16_launch(client, &x_h, m_pad * h, m_pad * h);
        let b = bind(&w13.bytes);
        let both = bf16_linear_launch(client, &a, &b, m_pad, h, 2 * inter);

        // The intermediate never leaves the device and never becomes f32 on the
        // host: `gate_up_silu` writes BF16 straight into the second MMA's A
        // operand.
        let act = gate_up_silu_bf16_launch(client, &both, m_pad, inter);
        let b2 = bind(&w2.bytes);
        let y_h = bf16_linear_launch(client, &act, &b2, m_pad, inter, h);
        host.enqueue += t_w.elapsed().as_secs_f64();

        let t_c = Instant::now();
        let y = tensor_of(client.clone(), dev.clone(), y_h, m_pad, h);
        let y = y.slice([0..m, 0..h]) * wgt;
        acc = acc.select_assign(0, idx, y, burn::tensor::IndexingUpdateOp::Add);
        host.accum += t_c.elapsed().as_secs_f64();
    }
    Ok(acc)
}

/// One expert's `(token rows, routing weights)` as device tensors.
///
/// The only host->device traffic left in the routed lane, and it is `m` indices
/// and `m` floats — 8 bytes a token against the 14.2 MB of weight the same
/// expert is about to read. This is control plane crossing into the data plane,
/// which is the direction that is allowed.
fn expert_rows<B: Backend>(
    toks: &[(usize, f32)],
    dev: &B::Device,
) -> (BT<B, 1, burn::tensor::Int>, BT<B, 2>) {
    let idx: Vec<i32> = toks.iter().map(|&(ti, _)| ti as i32).collect();
    let w: Vec<f32> = toks.iter().map(|&(_, wgt)| wgt).collect();
    let m = toks.len();
    (
        BT::<B, 1, burn::tensor::Int>::from_data(BTD::new(idx, [m]), dev),
        BT::from_data(BTD::new(w, [m, 1]), dev),
    )
}

fn main() -> Result<()> {
    // FIRST, before anything can allocate. A refused device buffer panics a
    // cubecl worker thread, and a worker-thread panic leaves this one running:
    // the forward then read a buffer nothing had written, exited 0, and printed
    // a layer-RMS ladder of 36.7..80.3 where the coherent one is 1.5..14.7.
    fatal::arm();
    let allocator = mary::models::inkling::pool::choose_memory_config();
    let cleanup = mary::models::inkling::pool::CleanupPolicy::choose();
    // How often the policy is ASKED, which is a separate question from what it
    // answers and was costing more than the answer. See `pool::CleanupGate`.
    let mut cleanup_gate = mary::models::inkling::pool::CleanupGate::new(cleanup);
    anyhow::ensure!(
        dev_lane::act_bf16() || !dev_lane_resid::resid_bf16(),
        "INK_RESID_BF16=1 requires the narrow activation lane; either leave \
         INK_ACT_BF16 enabled or set INK_RESID_BF16=0 for the fully wide lane"
    );

    let pile_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: <pile> <ids> <out>")?;
    let ids_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .context("usage: <pile> <ids> <out>")?;
    let out_path = std::env::args()
        .nth(3)
        .map(PathBuf::from)
        .context("usage: <pile> <ids> <out>")?;

    // The weights, and everything else the run needs to know about the model.
    // There is no second arm: no `INK_PILE` to opt into, and no checkpoint
    // directory to fall back to.
    let t_open = Instant::now();
    let mut cp = Weights::open(&pile_path)
        .with_context(|| format!("opening model collection in {}", pile_path.display()))?;
    let open_secs = t_open.elapsed().as_secs_f64();

    // …and the config comes from the SAME source. It used to come from a
    // checkpoint directory unconditionally, which meant the pile could hold 159
    // GiB and the run still depended on the 40 KB beside it: a pile that cannot
    // answer this is not authoritative, only large. In a pile the config is
    // FACTS (one entity per JSON scalar, `mary::jsonfacts`), so this is a query,
    // not a stored file being read back.
    //
    // `INK_CONFIG=<file>` overrides it, LOUDLY, for one case: a pile written
    // before the sidecars were facts. That pile still holds every weight and is
    // still worth running, and the alternative — falling back to the checkpoint
    // directory when the pile has no config — is exactly the silent dependency
    // this change removes. An override you have to type is a different thing
    // from a fallback you never see.
    let cfg_source = std::env::var("INK_CONFIG").ok();
    let cfg_text = match &cfg_source {
        Some(p) => std::fs::read_to_string(p).with_context(|| format!("INK_CONFIG={p}"))?,
        None => cp
            .document("config.json")
            .context(
                "this pile carries no config.json. Ingest the checkpoint's \
                 sidecars as facts (inkling_meta_gate <ckpt> <pile>), or point \
                 INK_CONFIG at the file to run without them",
            )?
            .to_string(),
    };
    let cfg = InklingConfig::from_json(&cfg_text).context("parsing config.json")?;
    let t = &cfg.text_config;

    // Which layers THIS process runs. REQUIRED, and a strict subrange.
    //
    // There is no default any more. Unset used to mean "the whole stack here",
    // which on a 119 GiB box against 144 GiB of weights means the page cache
    // evicts what the next token needs and every token pays real block-device
    // I/O — and a run that reads the SSD between tokens is not this model
    // running, it is a disk benchmark wearing its name.
    //
    // The bound is `hi - lo < num_hidden_layers`, NOT `nodes == 2`. Two is what
    // this model happens to need; the 66-layer sibling wants five to seven.
    // What is actually required is that no node carries the whole stack, and
    // that is what this says.
    let spec = std::env::var("INK_LAYERS").map_err(|_| {
        anyhow::anyhow!(
            "INK_LAYERS is required: this model does not fit one node ({} GiB of weights), so \
             every process runs a strict subrange and at least two nodes are needed.\n  \
             tail: INK_LAYERS=20:{}  INK_PIPE=tail:0.0.0.0:7654\n  \
             head: INK_LAYERS=0:20   INK_PIPE=head:<tail-host>:7654",
            144,
            t.num_hidden_layers,
        )
    })?;
    let (a, b) = spec.split_once(':').context("INK_LAYERS wants LO:HI")?;
    let (lo, hi) = (a.parse::<usize>()?, b.parse::<usize>()?);
    anyhow::ensure!(lo < hi, "INK_LAYERS wants LO < HI, got {lo}:{hi}");
    anyhow::ensure!(
        hi <= t.num_hidden_layers,
        "INK_LAYERS {lo}:{hi} runs past the {}-layer stack",
        t.num_hidden_layers
    );
    anyhow::ensure!(
        hi - lo < t.num_hidden_layers,
        "INK_LAYERS {lo}:{hi} is the whole {}-layer stack on one node, which does not fit. \
         Split it: no node may run every layer, and two is the MINIMUM rather than the number.",
        t.num_hidden_layers
    );
    // A pipe end is only a pipe end if it is not the whole stack; refusing the
    // contradiction here is cheaper than debugging a head that also unembeds.
    let pipe_spec = std::env::var("INK_PIPE").ok();
    let is_head = pipe_spec
        .as_deref()
        .map(|s| s.starts_with("head:"))
        .unwrap_or(false);
    let is_tail = pipe_spec
        .as_deref()
        .map(|s| s.starts_with("tail:"))
        .unwrap_or(false);
    anyhow::ensure!(
        pipe_spec.is_none() || is_head || is_tail,
        "INK_PIPE wants head:HOST:PORT or tail:ADDR:PORT"
    );
    anyhow::ensure!(
        !is_head || hi < t.num_hidden_layers,
        "a head that runs to the last layer has nothing to send; set INK_LAYERS"
    );
    anyhow::ensure!(
        !is_tail || lo > 0,
        "a tail that starts at layer 0 has nothing to receive"
    );

    let corpus: Vec<usize> = std::fs::read(&ids_path)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    anyhow::ensure!(
        !corpus.is_empty(),
        "no tokens — the forward would be vacuous"
    );

    // ---- INK_SLOTS=b: b INDEPENDENT sequences, decoded together ------------
    //
    // Not `INK_WIDTH=b`, which is the cost probe this is the thing it priced:
    // that one widens the pass with filler and commits row 0, so its rows are
    // one sequence's and its cache is one cache. These rows are b sequences
    // with b caches, and the only thing they share is the weight stream -- see
    // [`dev_lane::SlotCache`], which is where the sharing actually pays.
    //
    // The b prompts are b DISJOINT chunks of one token file, `slot_len` apart:
    // one corpus, b different pieces of real prose, and slot 0's piece does not
    // move when b changes. That last property is what makes the arms
    // comparable and what makes the contamination test possible, because
    // `INK_SLOT_OFFSETS` can then put slot 0 beside seven different neighbours
    // at the SAME b -- holding the GEMM lane fixed while the neighbours change,
    // which is the only way to tell contamination from the m == 1 width effect
    // that already makes b = 1 and b > 1 disagree.
    let slot_lane = std::env::var("INK_SLOTS").is_ok();
    let nslots: usize = std::env::var("INK_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    anyhow::ensure!(nslots >= 1, "INK_SLOTS counts sequences and starts at 1");
    // ---- INK_COHORTS=c: c cohorts of b slots, offset by half a round -------
    //
    // The layer split runs the two nodes strictly in sequence: the head's half
    // of a token has to finish before the tail's can start, so each end is idle
    // for about half of every pass and widening the batch does not change that
    // -- it scales both halves. `INK_SLOTS` proved the slots are independent of
    // each other; `INK_COHORTS` uses that a second time, at the granularity of
    // the pipe rather than of the batch. Cohort B's layers 0:21 run on the head
    // while cohort A's 21:42 run on the tail, so a round advances BOTH cohorts
    // and costs `2 * max(H, T)` rather than `2 * (H + T)`.
    //
    // Nothing about the arithmetic changes. Two cohorts of b slots hold the
    // same caches as one cohort of 2b and read the same weights; what changes
    // is the send/receive ORDER, which is the whole of the interleave and is
    // the `want` line below.
    let ncohorts: usize = std::env::var("INK_COHORTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    anyhow::ensure!(ncohorts >= 1, "INK_COHORTS counts cohorts and starts at 1");
    anyhow::ensure!(
        ncohorts == 1 || slot_lane,
        "INK_COHORTS interleaves cohorts of SLOTS -- set INK_SLOTS as well"
    );
    anyhow::ensure!(
        ncohorts == 1 || pipe_spec.is_some(),
        "INK_COHORTS fills the half of a PIPE that is blocked on its peer, and a run on one \
         node has no such half"
    );
    // Every cohort is `nslots` sequences with their own caches, so this is how
    // many prompts the corpus has to supply and how many caches the node holds.
    let total_slots = nslots * ncohorts;
    let slot_len: usize = std::env::var("INK_SLOT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if slot_lane {
            corpus.len() / total_slots
        } else {
            corpus.len()
        });
    anyhow::ensure!(
        slot_len > 0,
        "INK_SLOT_LEN is a prompt length and starts at 1"
    );
    let slot_at: Vec<usize> = match std::env::var("INK_SLOT_OFFSETS") {
        Ok(v) => v
            .split(',')
            .map(|c| c.trim().parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("INK_SLOT_OFFSETS is a comma-separated list of chunk indices")?,
        Err(_) => (0..total_slots).collect(),
    };
    anyhow::ensure!(
        slot_at.len() == total_slots,
        "INK_SLOT_OFFSETS names {} chunks against INK_SLOTS={nslots} x INK_COHORTS={ncohorts}",
        slot_at.len()
    );
    for &c in &slot_at {
        anyhow::ensure!(
            (c + 1) * slot_len <= corpus.len(),
            "chunk {c} of {slot_len} tokens runs past a {}-token file",
            corpus.len()
        );
    }
    let mut slot_ids: Vec<Vec<usize>> = slot_at
        .iter()
        .map(|&c| corpus[c * slot_len..(c + 1) * slot_len].to_vec())
        .collect();
    // `ids` is slot 0's stream. Every report that indexes a position -- the
    // per-position top-5, the MTP scoring, the final dump -- reads it, and one
    // sequence is what those were written about; the other slots keep their own
    // streams and are reported as a batch.
    let mut ids: Vec<usize> = slot_ids[0].clone();
    let n = ids.len();
    // So a crash can name what caused it. Set again per pass below, because a
    // pipe tail takes its length from the wire and a cached step feeds one
    // token, and both are the number that would appear in a refused size.
    fatal::note_tokens(n);

    let h = t.hidden_size;
    println!("=== forward ===");
    println!(
        "  config     : {}",
        match &cfg_source {
            Some(p) => format!("INK_CONFIG={p}  (OVERRIDE -- the pile was not asked)"),
            None => format!("config.json from the pile ({})", pile_path.display()),
        }
    );
    println!(
        "  weights    : pile {}  (index built in {open_secs:.1}s)",
        pile_path.display(),
    );
    println!("  tokens     : {n}  {ids:?}");
    if slot_lane {
        println!(
            "  slots      : INK_SLOTS={nslots} x INK_COHORTS={ncohorts} -- {total_slots} \
             independent sequences of {slot_len} tokens, chunks {slot_at:?} of a {}-token file",
            corpus.len()
        );
    }
    println!(
        "  layers     : {}  hidden {h}  experts {}+{} shared",
        t.num_hidden_layers, t.n_routed_experts, t.n_shared_experts
    );
    println!(
        "  this process: layers {lo}..{hi} ({} of {}){}",
        hi - lo,
        t.num_hidden_layers,
        match (is_head, is_tail) {
            (true, _) => "  PIPE HEAD -- embeds, then sends the stream on",
            (_, true) => "  PIPE TAIL -- receives the stream, unembeds, argmax",
            _ => "  whole stack on one machine",
        }
    );
    // 968 dense leaves + 20 480 expert leaves. The safetensors index named
    // 1 360 tensors for the same model, because its expert entries are STACKS
    // of 256; the pile names each expert on its own, and that is the
    // granularity a layer split can partition and a deduplicating store can
    // address.
    println!("  {}", cp.inventory());
    println!("{}", mary::models::inkling::pile::mem_line("index built"));

    // How many tokens to generate past the prompt. 0 reproduces the original
    // single-forward behaviour exactly.
    let gen_steps: usize = std::env::var("INK_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // Drafting runs on the TAIL, and only there. This used to refuse a pipe
    // outright -- "neither end can draft alone" -- which was true of the head
    // and false of the tail, and once INK_LAYERS became required it left NO
    // configuration in which acceptance could be measured at all. The tail
    // computes the last layer, so `x` at the draft site IS the whole stack's
    // final hidden state; it owns the final norm and the unembedding because it
    // takes the argmax; and it follows the sequence by recomputing that argmax
    // rather than being told, so its `ids` is the head's. The embedding table
    // was the only missing piece, and it is loaded below.
    //
    // `INK_MTP=k` drafts k tokens ahead with the MTP heads and scores them
    // against what the stack actually generates. It measures ACCEPTANCE, which
    // is the only oracle the composition has -- see mary::models::inkling::mtp.
    // Read here rather than beside the other lane switches because it decides
    // WHICH TABLES THIS PROCESS LOADS: drafting is the one configuration in
    // which a tail needs the embedding table, and the switch has to precede its
    // use.
    let mtp_k: usize = std::env::var("INK_MTP")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(0);
    let mtp_order = match std::env::var("INK_MTP_ORDER") {
        Ok(v) => MtpConcat::parse(&v)
            .with_context(|| format!("INK_MTP_ORDER wants hidden|embed, got {v:?}"))?,
        Err(_) => MtpConcat::HiddenFirst,
    };
    // A head is still refused, and not out of caution: it holds neither the
    // final hidden state nor any way to turn one into a token.
    anyhow::ensure!(
        mtp_k == 0 || !is_head,
        "INK_MTP drafts on the TAIL, not the head: a head owns neither the stack's final hidden \
         state nor the unembedding, so it cannot turn a draft into a token. Set INK_MTP on the \
         tail process -- it loads the embedding table for exactly this -- and leave it unset here."
    );

    let started = Instant::now();
    // Hoisted: re-reading 4.8 GB of embedding tables per generated token would
    // dwarf everything else in the loop.
    //
    // Split by role, and NOT loaded by the end that will never read them. That
    // is not tidiness: the whole reason for two machines is that this model's
    // working set does not fit in one page cache, and 4.8 GB of embedding
    // pinned on the box that only unembeds is 4.8 GB of expert slabs evicted.
    //
    // Drafting is the ONE exception, and only on the tail. An MTP head takes the
    // stack's final hidden state and the embedding of the token one step ahead;
    // the tail already has the first of those (it computes the last layer), the
    // final norm and the unembedding, so the embedding table is the single
    // missing piece. `INK_MTP` on a tail buys exactly that piece back, and
    // nothing else changes about which end holds what. The embedding NORM stays
    // head-only: it belongs to the main stack's input, and every MTP head norms
    // its own embeddings with its own `embed_norm`.
    // What prefill will hold in ACTIVATIONS at this sequence length: the
    // residual stream, every layer's kept keys and values, and the widest
    // layer's own working set. The gate charged a flat per-layer figure for
    // this, which is how it admitted a run that peaked at 119.5 GiB of a 119.6
    // GiB node; it then charged the score blocks, which stopped growing as soon
    // as the dense lane blocked its queries and left the estimate flat again --
    // 13.84 GiB at 16,384 tokens and 13.52 at 100,623. What actually scales is
    // the routed-expert lane, six rows a token through hidden-width buffers.
    //
    // The source decides which routed implementation each layer takes. Packed
    // NVFP4 experts can use the grouped BF16-storage lane; plain-BF16 experts
    // and the diagnostic per-expert arms retain their f32 gather and outputs.
    // Price that exact split before the startup copy rather than assuming the
    // model is uniformly packed. Whether this GPU can register the anonymous
    // mapping is learned only after the CUDA context exists; unsupported
    // registration is therefore a target-hardware gate, while the two explicit
    // fallback switches are conservatively priced here.
    let grouped_narrow = std::env::var("INK_GROUPED").unwrap_or_else(|_| "1".into()) == "1"
        && !std::env::var("INK_ZEROCOPY")
            .map(|v| v == "0")
            .unwrap_or(false);
    // The router arm and drafting decide WHICH weights the BF16 lane binds, so
    // admission has to know both before it prices the pool copy. `from_env` is
    // pure and is the same call the lane itself makes further down; reading it
    // twice cannot disagree, whereas a second parse of INK_ROUTER could.
    let mut admission = budget::AdmissionPolicy::runtime(allocator)
        .with_router_bf16(RouterArm::from_env() == RouterArm::Bf16)
        .with_drafting(mtp_k > 0);
    for layer in lo..hi {
        if !t.is_dense(layer) {
            let experts = format!("model.llm.layers.{layer}.mlp.experts.w13_weight");
            if !cp.is_nvfp4(&experts) {
                admission = admission.with_plain_bf16_layer(layer);
            } else if !grouped_narrow {
                admission = admission.with_wide_routed_layer(layer);
            }
        }
    }
    let attn_heads = t.heads(AttnKind::Global).0.max(t.heads(AttnKind::Local).0);
    let attn_head_dim = t.heads(AttnKind::Global).2.max(t.heads(AttnKind::Local).2);
    let attention_bytes = budget::prefill_activation_bytes(t, lo..hi, n, admission);
    // What b slots hold in K and V once they are all prefilled. A global
    // layer keeps the whole context, a local one keeps its window, and both
    // keep K and V -- so this is the term that multiplies with the batch and
    // the one an admission gate written for a single sequence does not have.
    let slot_kv_bytes: u64 = if slot_lane {
        (lo..hi)
            .map(|layer| {
                let kind = t.attn_kind(layer);
                let (_, kv_heads, head_dim) = t.heads(kind);
                let keep = match kind {
                    AttnKind::Local => t.sliding_window_size.min(n + gen_steps),
                    AttnKind::Global => n + gen_steps,
                };
                // Four bytes a value, or two where the cache is held narrow.
                // The admission gate reads this, and a gate that priced a BF16
                // cache at f32 would refuse ranges that fit -- which is the
                // safe direction and still the wrong number.
                let w = admission.cache.bytes();
                2 * total_slots as u64 * keep as u64 * (kv_heads * head_dim) as u64 * w
            })
            .sum()
    } else {
        0
    };
    if slot_lane {
        println!(
            "  slot KV            : {:.2} GiB for {total_slots} slots over layers {lo}..{hi}{}",
            slot_kv_bytes as f64 / GIB,
            if admission.cache == budget::StorageDType::Bf16 {
                "  (BF16)"
            } else {
                ""
            }
        );
    }

    // Every row's logits, when a reader has asked for them. `INK_ALL_LOGITS=1`
    // and `INK_DUMP_DIR` both mean `[n, effective_vocab]` f32 instead of the one
    // row a forward needs, which on this model is 764 KiB a TOKEN -- 11.34 GiB
    // at a 14,169-token prefill, larger than the whole modelled activation
    // working set. Only the node that owns the unembedding pays it.
    //
    // It went unpriced for as long as the allocator floor was a third of the
    // machine and covered it by accident. It is not covered by accident any
    // more, and a term that big cannot be left to luck.
    let all_logits = std::env::var("INK_DUMP_DIR").is_ok()
        || std::env::var("INK_ALL_LOGITS")
            .map(|val| val == "1")
            .unwrap_or(false);
    let logits_bytes: u64 = if all_logits && !is_head {
        n as u64 * t.effective_vocab() as u64 * core::mem::size_of::<f32>() as u64
    } else {
        0
    };
    if logits_bytes > 0 {
        println!(
            "  all logits         : {:.2} GiB for {n} rows of {} (INK_ALL_LOGITS)",
            logits_bytes as f64 / GIB,
            t.effective_vocab(),
        );
    }

    let want_embed = !is_tail || mtp_k > 0;
    // A drafting tail needs it too: the MTP depth layers consume the
    // BACKBONE-normed embedding, not the raw one. See `e_bn` at the draft site.
    let want_embed_norm = !is_tail || mtp_k > 0;
    let want_head = !is_head;
    // The supported lane copies every weight this process can alias into one
    // anonymous allocation before registration. `INK_STARTUP_COPY=0` exists
    // only for the pressure reproducer: it deliberately restores the unsafe
    // file-backed alias so the gate can prove that pressure was present.
    //
    // `charged_device_weights` is what admission charged the device pool, so
    // the bind census at the end of the run can be read against it rather than
    // on its own. Zero means nothing was charged, which the report says in
    // those words.
    let mut charged_device_weights = 0u64;
    if std::env::var("INK_STARTUP_COPY")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        println!("  startup weight copy: DISABLED (INK_STARTUP_COPY=0) -- UNSAFE diagnostic arm");
    } else {
        let mut globals = Vec::new();
        if want_embed {
            globals.push("model.llm.embed.weight");
        }
        if want_head {
            globals.push("model.llm.unembed.weight");
        }
        let t0 = Instant::now();
        let (experts, dense, bytes, device_weights) =
            cp.copy_share(lo..hi, &globals, attention_bytes + slot_kv_bytes, admission)?;
        println!(
            "  startup weight copy: {experts} expert + {dense} dense views, {:.2} GiB anonymous in {:.1}s",
            bytes as f64 / GIB,
            t0.elapsed().as_secs_f64(),
        );
        charged_device_weights = device_weights;
    }
    // The embedding table, as the BF16 the pile stores. `cp.held` turned 2.40
    // GiB of stored weight into 4.81 GB of host f32 and pinned it for the run,
    // on a box chosen because the working set only just fits -- and every token
    // read one row of it. `stored` hands back a view of the mapping; the
    // widening is now per LOOKUP, 16 KB a token, which is where it belongs.
    let embed_w = if want_embed {
        let leaf = cp.stored("model.llm.embed.weight")?;
        anyhow::ensure!(
            leaf.elem == Elem::Bf16,
            "the embedding table is {:?}",
            leaf.elem
        );
        Some(leaf.bytes.clone())
    } else {
        None
    };
    // A drafting tail takes BOTH the table and the norm. It used to take only
    // the table, on the reasoning that "every MTP head norms its own embeddings
    // with its own `embed_norm`" -- which is true and not sufficient. vLLM's
    // implementation (vllm/models/inkling/.../mtp.py), the oracle this lane did
    // not have when it was written, says the depth layers consume
    // `embed_norm(embed(ids))`: the BACKBONE norm first, the depth norm second,
    // "mtp embed_norm weights are near-identity (trained on already-normalized
    // inputs), and feeding raw embeddings drops MTP1 acceptance from ~0.85 to
    // ~0.70".
    let embed_n = if want_embed_norm {
        Some(cp.held("model.llm.embed_norm.weight")?)
    } else {
        None
    };
    let fnorm = if want_head {
        Some(cp.held("model.llm.norm.weight")?)
    } else {
        None
    };
    // The unembed table is NOT read here any more, and not widened anywhere.
    // It used to be `cp.held(...)`, which turned 1.6 GiB of stored BF16 into
    // 3.3 GB of host f32, uploaded THAT, and dropped the host copy. Now the
    // stored bytes are bound where they lie (see `unembed_w` below) and the
    // 3.3 GB never exists in either place.
    println!(
        "  embedding tables loaded in {:.1}s{}",
        started.elapsed().as_secs_f32(),
        if is_tail && mtp_k > 0 {
            "  (tail + INK_MTP: the embedding table is loaded here too, for drafting)"
        } else {
            ""
        }
    );

    // The MTP heads, only when asked for: eight dense blocks is 4.16 GiB, and a
    // run that is not drafting should not pay for them.
    let mtp_heads: Vec<MtpOwned> = if mtp_k > 0 {
        let t0 = Instant::now();
        let mut v = Vec::with_capacity(mtp_k);
        for i in 0..mtp_k {
            let p = format!("model.mtp.layers.{i}.transformer_block.");
            let g = |n: &str| cp.held(&format!("{p}{n}"));
            let fused = cp.held(&format!("{p}mlp.w13_dn.weight"))?;
            let (gate, up) = split_gate_up(&fused.data, h);
            let local = cfg.mtp_config.attn_kind(i) == AttnKind::Local;
            v.push(MtpOwned {
                embed_norm: cp.held(&format!("model.mtp.layers.{i}.embed_norm.weight"))?,
                hidden_norm: cp.held(&format!("model.mtp.layers.{i}.hidden_norm.weight"))?,
                input_proj: cp.held(&format!("model.mtp.layers.{i}.input_proj.weight"))?,
                attn_norm: g("attn_norm.weight")?,
                mlp_norm: g("mlp_norm.weight")?,
                attn_sconv: g("attn_sconv.weight")?,
                mlp_sconv: g("mlp_sconv.weight")?,
                wq: g("attn.wq_du.weight")?,
                wk: g("attn.wk_dv.weight")?,
                wv: g("attn.wv_dv.weight")?,
                wr: g("attn.wr_du.weight")?,
                wo: g("attn.wo_ud.weight")?,
                q_norm: g("attn.q_norm.weight")?,
                k_norm: g("attn.k_norm.weight")?,
                k_sconv: g("attn.k_sconv.weight")?,
                v_sconv: g("attn.v_sconv.weight")?,
                rel_proj: g("attn.rel_logits_proj.proj")?,
                gate,
                up,
                down: g("mlp.w2_md.weight")?,
                global_scale: g("mlp.global_scale")?.data[0],
                dims: AttnDims {
                    hidden: h,
                    heads: t.num_attention_heads,
                    kv_heads: t.num_key_value_heads,
                    head_dim: t.head_dim,
                    d_rel: t.d_rel,
                    // The same span rule the LLM's layers use, asked of the
                    // config rather than restated -- an MTP block's kind comes
                    // from mtp_config, but what a kind REACHES is one fact.
                    rel_extent: t.rel_span(if local {
                        AttnKind::Local
                    } else {
                        AttnKind::Global
                    }),
                    kernel: t.sconv_kernel_size,
                    rms_eps: t.rms_norm_eps,
                    kind: if local {
                        AttnKind::Local
                    } else {
                        AttnKind::Global
                    },
                },
                local,
            });
        }
        println!(
            "  MTP heads          : {mtp_k} loaded in {:.1}s, concat {} ({})",
            t0.elapsed().as_secs_f32(),
            mtp_order.name(),
            if mtp_order == MtpConcat::HiddenFirst {
                "what mtp_hidden_states_first reads as"
            } else {
                "the alternative reading"
            }
        );
        v
    } else {
        Vec::new()
    };
    // Drafts waiting to be scored: (the step whose token they predict, how many
    // heads deep, the token). Scored by DRAINING on the step they name, so a
    // draft is compared against what the stack actually produced rather than
    // against another draft.
    let mut mtp_pending: Vec<(usize, usize, usize)> = Vec::new();
    let mut mtp_hits = vec![0usize; mtp_k];
    let mut mtp_seen = vec![0usize; mtp_k];
    // Per ISSUING step, which depths turned out right. The per-depth rates are
    // MARGINALS — head d's draft against the true token at step s+d+1 — and an
    // accept-and-skip loop does not get to keep a correct depth-2 draft that sits
    // behind a wrong depth-1 one. What it keeps is the LEADING RUN of correct
    // drafts, so that is measured here rather than inferred from the marginals:
    // inferring it would need the depths to be independent, and they are not,
    // because head d is fed draft d-1.
    let mut mtp_issued: BTreeMap<usize, Vec<Option<bool>>> = BTreeMap::new();
    // `INK_MTP_PROB=1` scores THREE acceptance rules from one run, because
    // "is the draft the argmax" is not the question a speculation loop asks.
    //
    //   A  exact argmax match. What greedy speculation keeps, and the strictest
    //      rule available: a draft that was the target's second choice at
    //      p = 0.30 is thrown away.
    //   B  greedy draft, SAMPLED target. The draft is a point mass, so
    //      min(1, p_t(x)/p_d(x)) is p_t(x): accept the drafted token with the
    //      probability the target itself assigns it. Resampling from the
    //      normalised residual on rejection preserves the target's distribution
    //      EXACTLY (Leviathan et al. / Chen et al. 2023) -- this is not an
    //      approximation, it is a different exact algorithm.
    //   C  sampled draft, sampled target. Averaged over a draft sampled from
    //      p_d, the acceptance probability is
    //      sum_x p_d(x) min(1, p_t(x)/p_d(x)) = sum_x min(p_d(x), p_t(x)),
    //      which is 1 - TV(p_d, p_t) and does not depend on which token was
    //      drawn. The cleanest single number for "how close are these two
    //      distributions, in the currency a speculation loop spends".
    //
    // A is a lower bound on B and C by construction (it is B at temperature
    // zero), so a run where they coincide says the heads are weak and a run
    // where they part says the RULE was the constraint.
    let mtp_prob = std::env::var("INK_MTP_PROB")
        .map(|val| val == "1")
        .unwrap_or(false);
    // `INK_DRAFT_TOPK=N` prunes the DRAFT's unembedding to the N tokens the
    // main model just ranked highest at this position; 0, the default, is the
    // full vocabulary.
    //
    // The full table is 201024 x 4096 BF16 = 1.65 GiB and every draft depth
    // streams all of it to keep one index. N = 512 streams 4 MiB, gathered once
    // per step and shared by every depth.
    //
    // It is a bet on a CANDIDATE SET, not a cheaper way to compute the same
    // thing: a token outside the target's top N cannot be drafted at all, so
    // the drafts change and the acceptance rate with them. Off by default for
    // that reason -- the flag exists so the trade can be ablated, and the name
    // says what it does to the model rather than what it does to the clock.
    //
    // ## It cannot change the OUTPUT, and that is not the reason it is off
    //
    // `INK_SPEC` accepts on EXACT ARGMAX MATCH, so a draft the target does not
    // agree with is discarded and the target's own token is kept. Pruning the
    // candidate set can therefore only lower the ACCEPTANCE RATE; it cannot
    // produce a token the unspeculated run would not have produced. (The one
    // rule where that would be false is `INK_MTP_PROB`'s sampled acceptance,
    // and the two flags already refuse each other, three paragraphs down.)
    //
    // So the risk is not correctness. It is that acceptance is precisely the
    // variable this file measures speculation's whole verdict against, and that
    // verdict is ALREADY negative on text that is not a template: 0.977x at
    // `INK_SPEC=1` and 0.798x at `INK_SPEC=2` on the 3732-token document, at
    // 42.9% depth-1 acceptance against the short prompt's 71.2%. Turning on a
    // switch whose only direction is DOWN on that variable, unmeasured, would
    // make the losing arm lose more.
    //
    // ## Why it is nevertheless the switch most worth sweeping
    //
    // The other half of the trade is large enough to move the verdict rather
    // than the margin. Two thirds of a ~15.5 ms draft depth is the unembed
    // matmul alone, and every depth pays it again: 1.65 GiB streamed to keep
    // ONE index. At N = 512 the gather is 4 MiB, done once per step and shared
    // by every depth. For scale, the reference implementation's own best
    // remaining draft-path idea is quantising the draft LM head to W4A16, which
    // takes it from 785 MiB to about 200 MiB PER DEPTH -- this is 4 MiB, once.
    //
    // That is the shape of a trade that can flip a losing loop: the width cost
    // c(k) falls by most of the draft's cost while acceptance falls by however
    // much the top-N bet costs. Which is why the sweep has to measure END-TO-END
    // tok/s on the DOCUMENT corpus and not acceptance alone -- acceptance alone
    // can only make this look bad, and tok/s is the number the trade is about.
    // N in {256, 512, 1024, 2048, 0} against `INK_SPEC=1` and `=2`, both
    // corpora, is the run that would settle the default.
    let draft_topk: usize = std::env::var("INK_DRAFT_TOPK")
        .ok()
        .map(|v| v.parse())
        .transpose()
        .context("INK_DRAFT_TOPK wants a token count")?
        // Pruned by DEFAULT. Every draft depth otherwise streams the whole
        // 201024 x 4096 BF16 unembed -- 1.65 GiB to keep ONE token, ~66% of a
        // depth's cost -- and a draft outside the top-N is simply REJECTED by
        // the verifier, so this cannot change an output, only an acceptance
        // rate. It stays a tunable because N is a real knob; 0 disables the
        // pruning for the sweep and for `INK_MTP_PROB`.
        .unwrap_or_else(|| {
            // `INK_MTP_PROB` scores the draft head's distribution BY TOKEN INDEX
            // over the whole vocabulary, which a pruned head cannot emit -- the
            // two are refused together below. Before the prune became the
            // default that refusal was unreachable; now an `INK_MTP_PROB=1`
            // command line that worked yesterday hard-errors. An explicit
            // `INK_MTP_PROB` therefore IMPLIES the unpruned head.
            if std::env::var("INK_MTP_PROB").is_ok() {
                0
            } else {
                512
            }
        });
    // The two are exclusive rather than silently one-sided. `INK_MTP_PROB`
    // scores the draft head's distribution against the target's BY TOKEN INDEX
    // over the whole vocabulary, and a pruned head emits a distribution over
    // candidates -- 512 numbers about a different index space. Ignoring one
    // flag would hand back numbers that look like the usual ones.
    anyhow::ensure!(
        !(mtp_prob && draft_topk > 0),
        "INK_MTP_PROB scores the draft's FULL-vocabulary distribution and INK_DRAFT_TOPK={} \
         removes it. Run the probability scoring against the unpruned head, or drop \
         INK_MTP_PROB and read acceptance.",
        draft_topk
    );
    // And the same refusal against the approximate head, for the same reason
    // one step removed.
    //
    // The aNN lane returns a full-width row, so nothing here would SHAPE-fail --
    // which is exactly why it needs saying. Only the shortlist carries exact
    // scores; every other entry is a sketch estimate, so a softmax over that row
    // has a denominator built partly out of estimates. The argmax does not care
    // (the top is exact and the measurement says so), and a per-token
    // PROBABILITY does. `INK_MTP_PROB` is the one consumer that reads the row as
    // a distribution rather than as a ranking.
    //
    // Nothing else in the pass has this exposure. The per-position top-5 report
    // and `INK_DRAFT_TOPK`'s gather both read the row as a RANKING and both stop
    // far inside the shortlist -- 5 and 512 against 8192 -- so every row they
    // touch was rescored exactly.
    anyhow::ensure!(
        !(mtp_prob && ann_budget() > 0),
        "INK_MTP_PROB reads the head's row as a DISTRIBUTION, and the approximate head \
         (INK_ANN_HEAD={}) returns exact scores only for the {} rows it shortlisted -- \
         the rest of the row is the sketch's estimate, so the softmax denominator would \
         be part estimate and the probabilities would look ordinary while being wrong. \
         Set INK_ANN_HEAD=0 for probability scoring.",
        ann_budget(),
        ann_budget()
    );
    // The draft head's distribution, kept from the step that issued it until
    // the step it names arrives with the target's. 800 KB a draft at f32 and
    // at most k(k+1)/2 alive at once.
    let mut mtp_pd: BTreeMap<(usize, usize), Vec<f32>> = BTreeMap::new();
    let mut mtp_b_sum = vec![0f64; mtp_k];
    let mut mtp_c_sum = vec![0f64; mtp_k];
    let mut mtp_prob_n = vec![0usize; mtp_k];
    // Per issuing step, the two acceptance PROBABILITIES per depth -- the
    // stochastic twin of `mtp_issued`, and needed for the same reason: an
    // accept-and-skip loop keeps a leading RUN, so the expected prefix is
    // sum_j prod_{i<=j} q_i and not any function of the marginals alone.
    let mut mtp_issued_q: BTreeMap<usize, Vec<Option<(f64, f64)>>> = BTreeMap::new();
    // Per depth, every scored draft as (the DRAFT head's own top-1 probability,
    // did it hit). The marginal acceptance rate is an average over drafts the
    // head was sure of and drafts it was guessing at, and a loop does not have
    // to speculate on the guesses: it can read its own confidence first and
    // pay c(1) when it is low. Whether that is worth doing is a fact about the
    // JOINT distribution of confidence and correctness, which no aggregate rate
    // can answer, so the pairs are kept.
    let mut mtp_conf: Vec<Vec<(f32, bool)>> = vec![Vec::new(); mtp_k];
    // The MTP entry hidden state for every position, retained. 16 KB a token
    // at f32, and the reason the cached lane can draft at all -- see the draft
    // block for why a row, once produced, never changes.
    let mut mtp_main: Vec<f32> = Vec::new();
    // Whether the pruned-unembed line has been printed. Once per process, not
    // once per step: the shape does not change and 100 identical lines would
    // bury the report the run exists to produce.
    let mut drafted_pruned = false;
    // Per head: its STABLE hidden rows, which are what the NEXT head reads,
    // and its own K/V cache. Ragged by one row per depth, because head d's
    // stable rows stop at position seq-1-d.
    let mut mtp_stage: Vec<Vec<f32>> = vec![Vec::new(); mtp_k];
    let mut mtp_caches: Vec<Option<MtpCache>> = (0..mtp_k).map(|_| None).collect();
    // The draft lane. ON by default, because the host one is not an
    // implementation of drafting so much as a proof that drafting composes: at
    // 1470 ms for four heads against a 131 ms two-node round trip it made
    // speculation arithmetically impossible before the model was consulted.
    // `INK_MTP_DEV=0` is the control, and `INK_MTP_CHECK=1` runs it as one.
    let mtp_dev_on = std::env::var("INK_MTP_DEV")
        .map(|val| val != "0")
        .unwrap_or(true);
    // Built on the first draft rather than at load, exactly as `layers_dev` is:
    // a lane nobody takes should not pay for the upload.
    let mut mtp_devs: Vec<MtpDev> = Vec::new();
    // The device twins of `mtp_main` / `mtp_stage` / `mtp_caches`. The entry
    // states are UPLOADED from the host `entry` rather than recomputed on the
    // device, so a device/host draft comparison isolates the block and does not
    // fold in a second rms_norm's rounding.
    let mut mtp_main_dev: Option<T2> = None;
    let mut mtp_stage_dev: Vec<Option<T2>> = (0..mtp_k).map(|_| None).collect();
    let mut mtp_dev_caches: Vec<Option<MtpDevCache>> = (0..mtp_k).map(|_| None).collect();

    // Experts are read, applied and dropped. There is no decoded-expert cache:
    // it measured as no speedup, and it existed to paper over a capacity
    // shortfall (160 GB of checkpoint against 119 GB of box) that a second
    // Spark closes. See this file's header.
    //
    // There are no lane switches left. `INK_GPU`, `INK_ATTN`, `INK_DENSE`,
    // `INK_HEAD` and `INK_EXPERTS` each chose between a device implementation
    // and a host one; `INK_RESIDENT` chose between holding a weight and
    // re-reading it off the SSD every token; `INK_DEQUANT=chain` chose the
    // 46-launch Burn decode over the fused kernel. Every one of those choices
    // has a right answer on this hardware and a wrong one that still runs, and
    // a wrong one that still runs is a wrong one you will ship. The right
    // answers are now the only answers.
    //
    // The KV cache is the one thing still switched, and it is not a host/device
    // choice: `INK_KV=0` is the uncached lane, which is the ONLY oracle the
    // cached lane has (`INK_MTP_CHECK` compares the two in one process), so
    // deleting it would delete the check that the cache is right.
    let kv = std::env::var("INK_KV")
        .map(|v| v == "1" || v == "on")
        .unwrap_or(false);
    // `INK_REPEAT=1` keeps the token loop from GROWING the sequence: every step
    // re-runs the identical pass over the identical prompt. It exists because
    // the only pass of a given width a process can otherwise produce is its
    // FIRST one, which is also the one that pays for uploading every resident
    // weight -- 34 s against 1.4 s for the next. Comparing the cost of a 1-row
    // pass with a 2-row one therefore compared two different warm-up states,
    // not two widths. With this set the second and later steps are warm and
    // identical, so the widths can be compared by running the binary twice with
    // two prompts. Measurement only: the generated token is still printed and
    // still thrown away, so the run produces no continuation.
    let repeat = std::env::var("INK_REPEAT")
        .map(|v| v == "1" || v == "on")
        .unwrap_or(false);
    anyhow::ensure!(
        !repeat || !kv,
        "INK_REPEAT wants the uncached lane: with a KV cache a \
         repeated pass would append the same position to the cache again"
    );
    // ---- INK_SPEC=k: accept-and-skip ------------------------------------
    //
    // The loop the MTP measurement was for. `k` drafts ride back with every
    // answer; the next pass feeds the confirmed token FOLLOWED BY those drafts,
    // so a verify pass is `k + 1` rows wide and confirms between one and `k + 1`
    // tokens. Both ends roll their caches back to the accepted prefix, which is
    // why [`dev_lane::AttnCache::commit`] exists and why the drafts travel on
    // the wire rather than being re-derived: only the tail can draft (it owns
    // the final hidden state and the unembedding) and only the head can embed.
    //
    // Set it on BOTH processes and to the same value -- the protocol is
    // symmetric and a mismatch shows up as a width assertion, not as bad text.
    let spec_k: usize = std::env::var("INK_SPEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    anyhow::ensure!(
        spec_k == 0 || kv,
        "INK_SPEC={spec_k} wants INK_KV=1: speculation is about skipping sequential steps, and \
         the uncached lane has no cache to roll back"
    );
    anyhow::ensure!(
        !repeat || spec_k == 0,
        "INK_REPEAT and INK_SPEC measure different things"
    );
    anyhow::ensure!(
        spec_k == 0 || pipe_spec.is_some(),
        "INK_SPEC needs the pipe: the drafts are made on the tail and fed by the head"
    );
    anyhow::ensure!(
        !is_tail || spec_k == 0 || mtp_k == spec_k,
        "INK_SPEC={spec_k} on the tail wants INK_MTP={spec_k}: the drafts it sends ARE the MTP \
         heads' output, and mtp_k={mtp_k} would send a different number of them"
    );
    anyhow::ensure!(
        !is_tail || spec_k == 0 || mtp_dev_on,
        "INK_SPEC on the tail wants the device draft lane (INK_MTP_DEV unset or 1); the host \
         lane drafts in 1.5 s and there is no loop that pays for that"
    );
    if spec_k > 0 {
        println!(
            "  speculation        : INK_SPEC={spec_k} -- verify pass is {} rows",
            spec_k + 1
        );
    }

    // ---- INK_WIDTH=b: what a b-row decode step COSTS ----------------------
    //
    // Not a feature: an instrument. Batched decode -- b independent sequences
    // sharing one weight stream -- is the largest untaken lever on this model,
    // and the question that decides whether it is worth building is whether a
    // b-row cached step costs b times a one-row one or barely more than one.
    // This prices exactly that without any of the machinery a real batch needs
    // (b caches, b positions, a block-diagonal mask), because the COST of a
    // b-row pass does not depend on whether the rows belong to one sequence or
    // to b of them: the same projections, the same MoE gather over b routings,
    // the same unembedding, the same weight stream.
    //
    // Row 0 carries the real token; rows 1..b carry filler drawn fresh every
    // pass, so the router picks b independent expert sets rather than gathering
    // the same eight slabs b times. Only row 0's argmax is taken and only row 0
    // is committed, so the run produces one token per pass and the SAME text as
    // INK_WIDTH=1 -- which is the check that the extra rows are not changing
    // the answer, and it is a check the run makes on itself.
    let width: usize = std::env::var("INK_WIDTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    anyhow::ensure!(width >= 1, "INK_WIDTH counts rows and starts at 1");
    anyhow::ensure!(
        width == 1 || kv,
        "INK_WIDTH wants INK_KV=1: the uncached lane feeds the whole prefix and has no one-row \
         step to widen"
    );
    anyhow::ensure!(
        width == 1 || spec_k == 0,
        "INK_WIDTH and INK_SPEC both widen the pass"
    );
    anyhow::ensure!(
        width == 1 || mtp_k == 0,
        "INK_WIDTH is a cost probe and drafting is not part of what it prices"
    );
    if width > 1 {
        println!(
            "  width probe        : INK_WIDTH={width} -- every cached step is {width} rows, \
                  one of them real"
        );
    }
    anyhow::ensure!(
        !slot_lane || kv,
        "INK_SLOTS wants INK_KV=1: a slot IS a cache"
    );
    anyhow::ensure!(
        !slot_lane || width == 1,
        "INK_WIDTH prices a b-row pass with filler; INK_SLOTS runs one with b sequences in it, \
         and setting both would put filler rows in a batch that already has real ones"
    );
    anyhow::ensure!(
        !slot_lane || spec_k == 0,
        "INK_SLOTS and INK_SPEC both widen the pass"
    );
    anyhow::ensure!(
        !slot_lane || mtp_k == 0,
        "INK_SLOTS and INK_MTP: drafting follows one sequence and there are {nslots} here"
    );
    anyhow::ensure!(
        !slot_lane || !repeat,
        "INK_REPEAT and INK_SLOTS measure different things"
    );
    if slot_lane {
        println!(
            "  batched decode     : INK_SLOTS={nslots} -- {nslots} caches, one pass, {nslots} \
             tokens a pass"
        );
        if ncohorts > 1 {
            println!(
                "  pipeline interleave: INK_COHORTS={ncohorts} -- {ncohorts} cohorts of \
                 {nslots}, offset so this node computes one cohort while its peer computes \
                 another. A round is {ncohorts} passes and advances every cohort by a token."
            );
        }
    }
    // Head d's first STABLE row sits at position seq-1-d, so a prompt shorter
    // than the number of heads leaves a depth with no stable row at all: every
    // position of that head would be a function of drafts and the cache would
    // have nothing to hold. Refuse rather than quietly fall back to recomputing
    // the prefix, which is the failure this whole change exists to remove.
    anyhow::ensure!(
        mtp_k == 0 || !kv || ids.len() >= mtp_k,
        "INK_MTP={mtp_k} with INK_KV needs a prompt of at least {mtp_k} tokens, this one is {}",
        ids.len()
    );
    anyhow::ensure!(
        mtp_k <= cfg.mtp_config.num_nextn_predict_layers,
        "INK_MTP={mtp_k} but the checkpoint ships {} MTP heads",
        cfg.mtp_config.num_nextn_predict_layers
    );

    // ---- INK_TREE: single-box TOKEN-TREE speculation ------------------------
    //
    // A separate entry from `INK_SPEC` rather than a widening of it, and that is
    // a decision rather than an accident. `INK_SPEC` is a TWO-MACHINE
    // arrangement -- only the tail can draft, only the head can embed, and the
    // drafts travel on the wire -- because today's parallelism is pipeline
    // parallel. Three reasons not to teach that wire a topology:
    //
    //  * the tree exists to answer ONE question (does breadth pay at b = 2),
    //    and running it over a pipe adds the wire, the pipeline fill and the
    //    head/tail split to a number that should be about the drafter alone;
    //  * TP2 replaces PP2, and under TP2 both ranks run all 42 layers, so
    //    "only the tail can draft" dissolves. A wire protocol for the PP2
    //    arrangement is an investment in a path being removed;
    //  * the phase-3 gate is token-for-token agreement with non-speculative
    //    greedy, which is a clean comparison on one box and a messy one across
    //    a pipe.
    //
    // `b` candidates for the NEXT token and nothing else. The draft side of
    // that is FREE: every candidate comes off head 0's newest stable row, which
    // the decode step has already computed, so it is one top-b instead of one
    // argmax and not a single extra head step. See
    // [`mary::models::inkling::spectree::TreeSpec::breadth`] for the measured
    // cost of the verify side and the 2.0-tokens-per-pass ceiling that bounds
    // what breadth at depth 1 can ever be worth.
    let tree_b: usize = std::env::var("INK_TREE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    anyhow::ensure!(
        tree_b == 0 || tree_b >= 2,
        "INK_TREE={tree_b} is not a tree: breadth 1 is the chain this already runs"
    );
    anyhow::ensure!(
        tree_b == 0 || kv,
        "INK_TREE wants INK_KV=1: speculation is about skipping sequential steps, and the \
         uncached lane has no cache to roll back"
    );
    anyhow::ensure!(
        tree_b == 0 || pipe_spec.is_none(),
        "INK_TREE is the SINGLE-BOX lane; INK_SPEC is the pipe's. Setting both asks one \
         process to be two arrangements at once"
    );
    anyhow::ensure!(
        tree_b == 0 || spec_k == 0,
        "INK_TREE and INK_SPEC are two speculation lanes and both widen the pass"
    );
    anyhow::ensure!(
        tree_b == 0 || !slot_lane,
        "INK_TREE follows ONE sequence and INK_SLOTS runs several"
    );
    anyhow::ensure!(
        tree_b == 0 || width == 1,
        "INK_WIDTH fills a widened pass with filler rows; INK_TREE fills it with candidates"
    );
    anyhow::ensure!(
        tree_b == 0 || !repeat,
        "INK_REPEAT and INK_TREE measure different things"
    );
    anyhow::ensure!(
        tree_b == 0 || mtp_dev_on,
        "INK_TREE wants the device draft lane (INK_MTP_DEV unset or 1)"
    );
    // Depth 1 needs head 0 and no other. Refusing the rest is not tidiness:
    // heads 1..k would each run their triangle of speculative rows, and every
    // one of those drafts a token this tree has no node for.
    anyhow::ensure!(
        tree_b == 0 || mtp_k == 1,
        "INK_TREE={tree_b} is a DEPTH-1 tree and wants INK_MTP=1; INK_MTP={mtp_k} would draft \
         {} deeper token(s) the tree has no node for and pay for them",
        mtp_k.saturating_sub(1)
    );
    let spec_tree: Option<TreeSpec> = if tree_b > 0 {
        Some(TreeSpec::breadth(tree_b)?)
    } else {
        None
    };
    // Built once: the taps, the visibility and the depths are facts about the
    // TOPOLOGY, and the topology is fixed for the run.
    let tree_attn = spec_tree
        .as_ref()
        .map(|tr| spectree::tree_attn(tr, t.sconv_kernel_size));
    if let Some(tr) = spec_tree.as_ref() {
        println!(
            "  tree speculation   : INK_TREE={tree_b} -- verify pass is {} rows, {} candidate(s) \
             for t+1, draft side costs {} extra head step(s)",
            tr.len(),
            tr.drafts(),
            spectree::predicted_steps(tr, mtp_k, spectree::CacheFill::Exact),
        );
        if tree_b > spectree::MAX_MEASURED_BREADTH {
            println!(
                "  NOTE: b={tree_b} is past MAX_MEASURED_BREADTH={}; the marginal candidate \
                 costs more distinct experts than an extra SEQUENTIAL token does",
                spectree::MAX_MEASURED_BREADTH
            );
        }
    }
    println!("  attention          : device, weights DEVICE-RESIDENT");
    let routed_layers = (lo..hi).filter(|&layer| !t.is_dense(layer)).count();
    let routed_f32 = (lo..hi)
        .filter(|&layer| !t.is_dense(layer) && admission.routed(layer) == budget::StorageDType::F32)
        .count();
    println!(
        "  activations        : {} operands, {} residual stream; {routed_f32} of \
         {routed_layers} routed layers priced f32",
        admission.activation.name(),
        admission.residual.name(),
    );
    println!("  shared + dense MLP : device, uploaded once and held");
    println!(
        "  routed experts     : device, NATIVE tensor cores -- NVFP4 where packed, BF16 at layer 2"
    );
    println!("  head (unembed)     : device");
    let router_arm = RouterArm::from_env();
    println!("  router projection  : {}", router_arm.label());
    // OPT-IN, and this run is slower for it: it uploads a second projection per
    // layer and issues a second matmul and a second BLOCKING read per MoE layer
    // per token. A timing run must not have it on, and the report says which
    // runs did.
    let router_diff = std::env::var("INK_ROUTER_DIFF")
        .map(|v| v == "1")
        .unwrap_or(false);
    if router_diff {
        println!(
            "  router diff        : ON -- selection compared against the f32 [rows,hidden] lane, this pass IS slower"
        );
    }
    println!(
        "  kv cache           : {}",
        if kv {
            "on"
        } else {
            "off (prefix recomputed each step)"
        }
    );
    // The SHARED experts' w13 is square, so nothing but a forward can tell the
    // two readings apart. INK_SHARED_W13_HALVED=1 selects the other one.
    let shared_halved = mary::models::inkling::load::shared_w13_halved();
    println!(
        "  shared w13 split   : {}",
        if shared_halved {
            "HALVED (contiguous)"
        } else {
            "INTERLEAVED"
        }
    );
    let dev = burn::backend::cuda::CudaDevice::default();
    let mut ddense = DeviceDense::default();
    // Every attention layer's projections, on the device, for the whole run.
    //
    // Attention was the last lane that still STREAMED, and it streamed because
    // of a reading of `INK_RESIDENT` that stopped one step short: the host copy
    // is indeed worthless after the upload, so the lane declined to hold it --
    // and then held nothing at all, and paid the read, the widen AND the
    // transfer again on every token. Measured on the pile, decode step, 42
    // layers: 2.2 s widening 6.9 GiB of BF16 into f32 and 6.0 s pushing it
    // across, against 0.2 s of attention. The weights do not change between
    // tokens; only where they are held was ever in question, and on a device
    // lane the answer is the device -- which is why there is no longer a flag
    // that can answer it the other way.
    //
    // Keyed by the layer prefix, exactly as [`DeviceDense`] is, and populated
    // on first use rather than eagerly: a lane that is never taken should not
    // pay for the upload.
    let mut layers_dev: std::collections::BTreeMap<String, LayerDev> =
        std::collections::BTreeMap::new();
    let mut dattn_bytes = 0u64;
    // The compute client, taken FROM a Burn tensor rather than constructed
    // beside it. `CudaRuntime::client(&Default::default())` is meant to return
    // the same client Burn is using, and "meant to" is not a thing to bet a
    // device pointer on: `seam::handle_of` hands a Burn allocation to a raw
    // kernel launched on this client, and if they were two clients that would
    // be a wrong answer rather than an error.
    let fp4_client = mary::models::inkling::seam::client_of(&BT::<Bk, 2>::zeros([1, 1], &dev));
    println!(
        "{}",
        mary::models::inkling::pile::mem_line("after CUDA context")
    );
    println!(
        "{}",
        mary::models::inkling::seam::pool_line(&fp4_client, "cold")
    );
    println!(
        "  memory pool        : {} pages, cleanup {}",
        allocator.env_value(),
        cleanup.name(),
    );
    println!("  pool polled        : {}", cleanup_gate.schedule());
    // The one number that decides whether this sequence length is runnable at
    // all, asked of the device rather than modelled. It is checked HERE and not
    // beside the admission gate because this is the first client in the
    // process, and taking `max_page_size` from a client Burn did not make would
    // be reading a different device's answer. Nothing has run a layer yet.
    let qblock = budget::query_block(attn_heads, n);
    println!(
        "  attention budget   : queries in blocks of {qblock}, so [{attn_heads}, {qblock}, {n}] \
         f32 scores = {:.2} GiB per layer (the whole square would be {:.2} GiB) beside \
         [{attn_heads}, {n}, {attn_head_dim}] {} activations = {:.2} GiB; the widest single \
         buffer this range asks for is {:.2} GiB and this device allows {:.2} GiB \
         (up to {} tokens)",
        budget::score_block_bytes(attn_heads, qblock, n) as f64 / GIB,
        budget::score_matrix_bytes(attn_heads, n) as f64 / GIB,
        admission.activation.name(),
        budget::activation_bytes(attn_heads, attn_head_dim, n, admission.activation) as f64 / GIB,
        budget::largest_buffer(t, lo..hi, n, admission) as f64 / GIB,
        budget::largest_allocation(&fp4_client) as f64 / GIB,
        budget::longest_sequence(
            t,
            lo..hi,
            budget::largest_allocation(&fp4_client),
            admission,
        ),
    );
    budget::check(&fp4_client, t, lo..hi, n, admission)?;
    // Nine blocking device round trips for the whole run, instead of four per
    // expert. Every later slab is an offset view of one of these.
    //
    // Always an `Aliases`, even when nothing can be aliased. In the supported
    // lane its one registered mapping is the anonymous startup allocation, not
    // the pile file. The unsafe reproducer arm deliberately registers the pile.
    //
    // `Aliases::disabled()` copies exactly as the old `None` arm did but COUNTS
    // it, so a source whose bytes cannot be aliased reports that rather than
    // going quiet. The registration is unconditional — on a unified-memory part
    // a copy of a weight the device can read where it lies is a copy for
    // nothing, and there is no configuration in which we want one.
    #[cfg(feature = "inkling-cuda")]
    let fp4_aliases = {
        let c = &fp4_client;
        // `INK_ZEROCOPY=0` sends every weight through `create_from_slice`
        // instead of aliasing the startup allocation. It remains a diagnostic
        // arm, not a lane: the supported path aliases anonymous memory and
        // does not need a per-bind device copy.
        if std::env::var("INK_ZEROCOPY")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            println!("  zero-copy mappings : DISABLED (INK_ZEROCOPY=0) -- every bind copies");
            Some(mary::models::inkling::fp4gemm::Aliases::disabled())
        } else {
            let alias_started = Instant::now();
            let maps = cp.mappings()?;
            let n = maps.len();
            let a = mary::models::inkling::fp4gemm::Aliases::register(c, maps);
            println!(
                "  zero-copy mappings : {} {n} in {:.1} ms",
                if a.is_some() {
                    "registered"
                } else {
                    "UNSUPPORTED"
                },
                alias_started.elapsed().as_secs_f64() * 1e3
            );
            // Packed grouped layers were admitted at their BF16 activation
            // width only on the premise that the mapping-backed grouped lane
            // can run. Falling through to the per-expert copying lane here
            // would silently switch those layers to f32 after the startup-copy
            // gate has already used the narrow figure. Make the documented
            // target-hardware gate real. Explicit diagnostic/copying modes are
            // priced wide before the copy and therefore remain runnable.
            anyhow::ensure!(
                a.is_some()
                    || (lo..hi).all(|layer| {
                        t.is_dense(layer) || admission.routed(layer) == budget::StorageDType::F32
                    }),
                "this CUDA target cannot register host mappings, but pre-copy admission priced \
                 one or more packed routed layers for the BF16 grouped lane. Refusing instead \
                 of silently falling back to its f32 per-expert buffers. Set INK_GROUPED=0 or \
                 INK_ZEROCOPY=0 before launch so admission prices the copying lane."
            );
            Some(a.unwrap_or_else(mary::models::inkling::fp4gemm::Aliases::disabled))
        }
    };

    // The unembed table, as the BF16 the pile stores. Not uploaded: BOUND.
    //
    // This is the single largest weight in the process and it was the single
    // largest widening: `[201024, 4096]` BF16 is 1.61 GiB, and reading it as
    // f32 made it 3.22 GB — on the host first, then again on the device,
    // costing 1.6 GiB of RAM the box does not have to spare and doubling the
    // bytes the biggest matmul in the forward has to read. Measured at 22.5 ms
    // a token, which is 146 GB/s against a 273 GB/s bus: bandwidth-bound, so
    // the bytes ARE the time.
    //
    // Bound through the same `Aliases` seam the experts use, so on this
    // unified-memory part the GPU reads the pile's own mapped pages and the
    // table is never copied at all. `slice_or_copy` reports which happened.
    //
    // The FULL padded vocabulary is bound, not the effective 200058 rows: the
    // MMA tiles n by 8 and 200058 does not divide, while the padded 201024
    // does. The extra rows are the checkpoint's own padding and the argmax
    // slices them off, exactly as the f32 path sliced them off after uploading
    // them.
    // Filled inside the bind below, where the codes exist. Declared out here
    // because the bind is an expression and the sketch is the third thing it
    // produces.
    let mut head_sketch: Option<mary::models::inkling::annhead::Sketch> = None;
    let (unembed_w, unembed_bytes) = if want_head {
        use mary::models::inkling::bf16gemm::Bf16W;
        use mary::models::inkling::pile::Elem;
        let leaf = cp.stored("model.llm.unembed.weight")?;
        anyhow::ensure!(
            leaf.elem == Elem::Bf16,
            "the unembed table is stored as {:?}, and this lane multiplies BF16 by BF16. \
             Widening it to reuse an f32 path is the thing rule 3 forbids; a pile holding \
             f32 here needs an f32 MMA, not a cast.",
            leaf.elem
        );
        let (rows, cols) = (leaf.dims[0] as usize, leaf.dims[1] as usize);
        anyhow::ensure!(
            rows == t.vocab_size && cols == h,
            "unembed is {rows}x{cols}"
        );
        anyhow::ensure!(
            Bf16W::tileable(rows, cols),
            "unembed {rows}x{cols} does not tile as m16n8k16"
        );
        let align = note_align(&leaf.bytes, rows, cols);
        let copy_to_align = align < 16
            && std::env::var("INK_ALIGN_COPY")
                .map(|v| v == "1")
                .unwrap_or(false);
        // The same placement `bind_bf16` takes, on the largest single weight in
        // the process -- 1.53 GiB of the 4.94 the device arm holds.
        let device = budget::dense_weights() == budget::DenseWeights::DevicePool;
        let hnd = match fp4_aliases.as_ref() {
            Some(al) if !copy_to_align && !device => al.slice_or_copy(&fp4_client, &leaf.bytes),
            _ => fp4_client.create_from_slice(&leaf.bytes),
        };
        println!(
            "  unembed BOUND as BF16, {rows} x {cols} = {:.2} GiB stored (the f32 lane it \
             replaces materialised {:.2} GiB)",
            leaf.bytes.len() as f64 / GIB,
            2.0 * leaf.bytes.len() as f64 / GIB
        );
        (
            Some(if head_lane() != HeadLane::Bf16 {
                // Quantise once, here, and let the BF16 upload die with `hnd`:
                // what the run then holds is 0.43 GiB, not 1.53.
                //
                // The SAME `PackedW` either way -- identical codes, identical
                // scales, nothing moves on the device. The variant chooses only
                // whether the ACTIVATION is quantised on the way in, and W4A4
                // is here as a comparison arm: the publisher calibrated an
                // input quantiser for the routed experts and for nothing else,
                // so this tensor has none.
                let p = quantized_bf16(&fp4_client, &leaf.bytes, rows, cols);
                // The sign sketch, from the codes that are already here.
                //
                // Built on the DEVICE and from the NVFP4, not from `leaf.bytes`,
                // for the reason the rebind above exists: the BF16 is 1.53 GiB
                // and reading it a second time to derive 0.103 GiB of signs
                // would cost more than the sketch saves in its first three
                // hundred tokens. What the sketch approximates is therefore what
                // the exact lane computes -- the same four-bit codes -- so the
                // rescore agrees with the lane it shortlists for rather than
                // with a BF16 reference neither of them runs.
                // Built for the TEMPERATURE too, not only for the aNN lane. The
                // knob is a temperature in logit units and it becomes a
                // hidden-state sigma by dividing by the mean embedding-row norm,
                // which is computed here and nowhere else -- so without this the
                // same `INK_TEMP` would mean two different temperatures
                // depending on an unrelated switch, and the run would not say so.
                if ann_budget() > 0 || head_temp() > 0.0 {
                    let t0 = Instant::now();
                    let sk = mary::models::inkling::annhead::build_sketch(
                        &fp4_client,
                        &p.codes,
                        &p.scales,
                        p.n,
                        p.k,
                        p.scale2,
                        ANN_SKETCH_SEED,
                        ann_rotated(),
                    );
                    println!(
                        "  unembed SIGN SKETCH built in {:.2} s, basis {}: {} x {} bits = \
                         {:.3} GiB ({:.2}x under the NVFP4 codes), {} live rows of {}, \
                         mean row norm {:.4}. NOT in the admission accounting: this \
                         buffer is a client allocation, not a charged weight",
                        t0.elapsed().as_secs_f64(),
                        if sk.rotated {
                            "ROTATED"
                        } else {
                            "RAW (ablation)"
                        },
                        sk.n,
                        sk.k,
                        sk.bytes() as f64 / GIB,
                        (leaf.bytes.len() as f64 * 4.5 / 16.0) / sk.bytes() as f64,
                        sk.live_rows,
                        sk.n,
                        sk.mean_norm,
                    );
                    head_sketch = Some(sk);
                }
                let w = match head_lane() {
                    HeadLane::W4a16 => w4a16_bind(&fp4_client, p),
                    HeadLane::W4a4 => dev_lane::ProjW::Fp4(p),
                    HeadLane::Bf16 => unreachable!("guarded by head_lane() != Bf16"),
                };
                println!(
                    "  unembed RE-BOUND as NVFP4 / {:?}: {:.2} GiB -> {:.2} GiB, \
                     a per-step FLOOR cut by the same ratio",
                    head_lane(),
                    leaf.bytes.len() as f64 / GIB,
                    leaf.bytes.len() as f64 * 4.5 / 16.0 / GIB
                );
                drop(hnd);
                w
            } else {
                dev_lane::ProjW::Bf16(Bf16W {
                    h: hnd,
                    n: rows,
                    k: cols,
                    align: if copy_to_align { 16 } else { align },
                })
            }),
            // The same stored bytes, kept reachable from the HOST as well, for
            // `INK_DRAFT_TOPK`. Pruning the draft's unembedding is a row gather
            // and the rows to gather are not known until a step has run, so the
            // gather reads the pile's mapping rather than the device buffer.
            // `Bytes` is a handle over that mapping: this clone is a refcount,
            // not 1.65 GiB.
            Some(leaf.bytes.clone()),
        )
    } else {
        (None, None)
    };
    let head_sketch = head_sketch;
    // The final norm's gain, uploaded once for the same reason -- it used to be
    // re-uploaded from the host copy on every pass, and on every MTP draft.
    let fnorm_dev = fnorm.as_ref().map(|f| up1r::<Bk>(&f.data, h, &dev));

    // Pay the storage layer's BLAKE3 for this node's experts HERE, once, rather
    // than in whichever decode step first routes to each of them.
    //
    // Rule 6 says never touch the SSD once running, and this is the read that
    // was doing it -- 9.4 to 33.6 MB an expert, hashed on first touch, landing
    // wherever the router sent a token first. On the 20-layer head at step 41
    // to 80 of an 80-token generation it was still costing between 0.5 and 274
    // ms a pass, and that one variable explained the entire spread.
    //
    // It also makes an A/B possible at all. Two builds that produce different
    // tokens route to different experts and therefore pay different amounts of
    // a cost that belongs to neither of them; with the warm done, the
    // comparison is between the models rather than between their luck.
    {
        let t0 = Instant::now();
        let mut last = Instant::now();
        let (count, bytes) = cp.warm_experts(lo..hi, |i, total, b| {
            if last.elapsed().as_secs_f64() > 10.0 || i == total {
                last = Instant::now();
                println!(
                    "  warming experts    : {i}/{total}  {:.1} GiB  {:.0}s",
                    b as f64 / GIB,
                    t0.elapsed().as_secs_f64()
                );
            }
        })?;
        println!(
            "  warmed             : {count} expert planes, {:.2} GiB validated in {:.1}s ({:.2} GiB/s)",
            bytes as f64 / GIB,
            t0.elapsed().as_secs_f64(),
            bytes as f64 / GIB / t0.elapsed().as_secs_f64()
        );
    }

    // Everything one layer carries between generated tokens. The attention
    // cache is the headline, but the two layer-level short convolutions have
    // state too: they reach `kernel - 1` positions back, and a cache that
    // remembers K and V while forgetting those is wrong in a way that still
    // produces fluent-looking text.
    struct LayerCache {
        attn: dev_lane::AttnCache<Bk>,
        attn_sconv: BT<Bk, 2>,
        /// `None` until the prefill seeds it, and a device tensor rather than a
        /// `Vec<f32>` for the same reason everything else here is: the
        /// convolution that reads it runs on the device, and a history that
        /// lived on the host would drag the whole MLP half back across.
        mlp_sconv: Option<BT<Bk, 2>>,
        /// What a speculative batch convolved, kept until the verifier says how
        /// many of its rows survived. `kernel - 1` history rows followed by the
        /// batch's own inputs, so the history after keeping `keep` of them is
        /// the window starting at `keep` -- the same shape, and the same
        /// argument, as [`dev_lane::AttnCache`]'s pending K/V projections.
        attn_sconv_pending: Option<BT<Bk, 2>>,
        mlp_sconv_pending: Option<BT<Bk, 2>>,
    }
    let mut caches: Vec<LayerCache> = Vec::new();

    // The same, for b slots. Every field is its single-sequence twin with a
    // leading slot dimension, and the reason there is a second struct rather
    // than a generalised one is that they are not two settings of a thing:
    // `LayerCache` can be rolled back to an accepted prefix (that is what the
    // `_pending` fields are for) and a slot batch has nothing to roll back,
    // because every row of it is a token the model chose for its own sequence.
    struct SlotLayerCache {
        attn: dev_lane::SlotCache<Bk>,
        /// `[slots, kernel - 1, hidden]`.
        attn_sconv: BT<Bk, 3>,
        mlp_sconv: Option<BT<Bk, 3>>,
    }
    // The slot batch, built one slot at a time as the b prefills land. A
    // prefill is compute-bound and gains nothing from a batch, so the b of them
    // run one at a time; each one is SEATED the moment it finishes rather than
    // kept until all b are in. See `SlotCache::seeded` for what keeping them
    // cost.
    // One batch per COHORT. `slots_dev[c][l]` is cohort `c`'s layer-`l` state,
    // and the two indices are never mixed: a pass names its cohort once, at the
    // top, and every cache access below goes through that name.
    let mut slots_dev: Vec<Vec<SlotLayerCache>> = (0..ncohorts).map(|_| Vec::new()).collect();

    // The wire, opened AFTER the weights so a connection is never left hanging
    // while the other end spends a minute building its index. The tail binds and
    // waits; the head connects and retries while it waits. Both ends are
    // bounded by `INK_PIPE_WAIT` — see [`pipe_wait`] for why the order the two
    // commands are started in used to matter and no longer does.
    let wait = pipe_wait();
    let mut pipe = match pipe_spec.as_deref() {
        Some(s) if is_head => {
            let addr = &s["head:".len()..];
            let t0 = Instant::now();
            let sock = pipe_connect(addr, wait)?;
            sock.set_nodelay(true)?;
            println!(
                "  pipe: connected to the tail at {addr} in {:.1}s",
                t0.elapsed().as_secs_f32()
            );
            Some(Pipe::Head(sock))
        }
        Some(s) if is_tail => {
            let addr = &s["tail:".len()..];
            let l = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
            println!("  pipe: listening on {addr}");
            let t0 = Instant::now();
            let (sock, peer) = pipe_accept(&l, addr, wait)?;
            sock.set_nodelay(true)?;
            println!(
                "  pipe: head connected from {peer} in {:.1}s",
                t0.elapsed().as_secs_f32()
            );
            Some(Pipe::Tail(sock))
        }
        _ => None,
    };

    // One per layer of the whole stack, indexed by absolute layer number, so
    // the summary can name the layer rather than a position in this node's
    // slice. Declared here and not in the pass because the question is what the
    // RUN did, not what one token did.
    let mut route_diff = vec![RouteDiff::default(); t.num_hidden_layers];

    // `INK_DEV_ROUTE=0` puts the router's DECISION back on the host. It is on
    // by default; the flag exists so the two lanes can be interleaved from one
    // binary, which is the only honest way to price a 15% change against a
    // 2-3 ms pass-to-pass drift.
    let dev_route = std::env::var("INK_DEV_ROUTE")
        .map(|v| v != "0")
        .unwrap_or(true);
    // `INK_ROUTE_STALE=1` is a TIMING PROBE and NOT A LANE: it reuses the
    // PREVIOUS pass's routing decision for a layer instead of reading this
    // pass's back, so the router projection is still enqueued, the device still
    // computes the logits, and the only thing that goes away is the BLOCKING
    // read. Every kernel, every shape and every other piece of host work is
    // unchanged -- `top_k` distinct experts, one token each at decode -- so the
    // wall clock difference against an interleaved control IS the price of the
    // per-layer sync. The generated TEXT is wrong under it, deliberately and
    // always: the decision it acts on belongs to a different activation.
    let route_stale = std::env::var("INK_ROUTE_STALE")
        .map(|v| v == "1")
        .unwrap_or(false);
    // The gate bias is `[n_routed]` f32 and does not change during a run, so it
    // is uploaded once per layer rather than once per pass. One KiB either way;
    // it is here because a per-pass upload in a lane whose whole subject is
    // host->device round trips would be embarrassing.
    let mut bias_dev: std::collections::HashMap<usize, cubecl::server::Handle> =
        std::collections::HashMap::new();
    // `INK_ROUTE_STALE=1` only: last pass's decision per layer.
    let mut route_cache: std::collections::HashMap<usize, Vec<Routing>> =
        std::collections::HashMap::new();

    // ---- INK_DEV_PLAN: the row plan on the device ---------------------------
    //
    // `INK_DEV_ROUTE` moved the DECISION to the device and still read it back;
    // this is what makes the readback unnecessary. See
    // [`mary::models::inkling::devplan`] for what is invariant at `n == 1` and
    // why none of it transfers to a prefill.
    let plan_arm = PlanArm::from_env()?;
    // Five diagnostics want the decision on the host, and every one of them
    // silently stops working if this lane takes the layer instead. They are
    // read once, here, rather than per layer per pass: a `getenv` inside the
    // loop this lane exists to shorten would be a joke at its own expense.
    let route_log_on = std::env::var("INK_ROUTE_LOG").is_ok();
    let route_dbg_on = std::env::var("INK_ROUTE_DBG").is_ok();
    let grouped_mode = std::env::var("INK_GROUPED").unwrap_or_else(|_| "1".to_string());
    // `INK_GROUPED=2` is the surviving EQUIVALENCE instrument -- the grouped
    // lane against the per-expert loop, on the same input, compared as bits.
    // It needs `by_expert`, so under it this lane builds the plan on the device
    // AND the host still reads the decision back. That makes it slower than
    // either arm and is exactly right: it is measuring agreement, not time.
    let grouped_ab = grouped_mode == "2";
    let devplan_verify = std::env::var("INK_DEVPLAN_CHECK")
        .map(|v| v == "1")
        .unwrap_or(false);
    // `INK_DEV_PLAN_MAXN`: the widest pass the device plan will take.
    //
    // **It defaults to 1, and that is a MEASURED default rather than the old
    // restriction left in place.** The plan can now be built at any width up to
    // `MTILE` (see `devplan.rs`), and doing so removes the per-layer blocking
    // readback -- but only by giving up the deduplication of experts across
    // rows, and on this model those two are worth about the same. One node,
    // spark2-zt (GB10), layers 0:21, a 3772-token prompt, 12 cached decode
    // steps, two INTERLEAVED reps, medians, one binary:
    //
    //     arm                                    w = 2     w = 3
    //     MAXN=1   (dedup, readback)              69.0      75.6
    //     MAXN=16  (no dedup, no readback)        67.9      78.2
    //     INK_ROUTE_STALE=1 (dedup, no readback)  64.0        --
    //
    // At w = 2 the two cancel to within the spread; at w = 3 dedup wins by
    // 2.6 ms, because a third row duplicates more experts while the readback is
    // worth less. The log's own counter says the same thing in slabs: 193 a
    // pass deduplicated at w = 2 against 228 without.
    //
    // The probe row is the one that matters for whoever picks this up. A plan
    // that dedups AND stays on the device is worth 5.0 ms at w = 2 against
    // MAXN=1, and none of that is reachable without moving `row_tok`,
    // `blk_slot`, `blk_tile0`, `blk_cnt`, `tok_rows` and `tok_cnt` onto the
    // device too -- because dedup is exactly what turns them into data.
    // `MAXN=16` stays because it is the half of that which is already built and
    // checked (`INK_DEVPLAN_CHECK=1` passes at width 2 over 19 layers x 4
    // steps), not because it wins.
    let dev_plan_maxn: usize = std::env::var("INK_DEV_PLAN_MAXN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .min(mary::models::inkling::fp4gemm::MTILE);
    let mut devroute: Option<DevRoute> = None;
    // Per decode pass: which arm it ran and what it cost. The arms are
    // interleaved inside one process, so this is the only pairing that is not
    // confounded by warm-up, by clocks, or by whatever else the box is doing.
    let mut pass_ms_arm: Vec<(bool, f64)> = Vec::new();

    // ---- what the pipe costs, and what it wastes --------------------------
    //
    // A two-node split has one node blocked on the other for most of every
    // token, and that idle half is the whole argument for speculation: it is
    // already paid-for hardware doing nothing, so a draft that gets rejected
    // costs work the machine was not going to do anyway. None of it was
    // measured, because the head charges the ENTIRE round trip -- wire out,
    // the tail's whole half of the stack, wire back -- to the one slot the
    // unembed occupies on a whole-stack run. That slot cannot separate wire
    // from peer compute, so both are timed here instead, on both ends, and the
    // wire falls out as the difference between the head's wait and the tail's
    // own pass.
    let mut acc_send = 0f64;
    let mut acc_wait_peer = 0f64;
    // What the tail drafted last pass. On the head these are the rows it will
    // FEED; on the tail they are what its next verify pass will be judged
    // against. Same list, one machine apart.
    let mut drafts_in: Vec<usize> = Vec::new();
    // The cohorts this head has sent and not yet read an answer for, oldest
    // first. A single-cohort head empties it on the same pass it fills it and
    // the queue is a formality; an interleaved head keeps `ncohorts - 1` in it,
    // and THAT is the pipeline -- the tail is working on those while this pass
    // runs. A tail never uses it: it answers the pass it is on.
    let mut in_flight: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut last_drafts: Vec<usize> = Vec::new();
    // The tree's candidate tokens for t+1, node order (so rank order at depth
    // 1), root EXCLUDED -- the root is the token `ids` already ends with. Empty
    // on the pass that has not drafted yet, which is exactly the prefill.
    let mut tree_drafts: Vec<usize> = Vec::new();
    // Tokens, not passes: a speculative pass confirms between 1 and k+1 of
    // them, so this is what the run's length and its tok/s are counted in.
    let mut gen_tokens = 0usize;
    // A depth-1 tree confirms at most 2 tokens a pass, whatever `b` is: breadth
    // raises the PROBABILITY of reaching that ceiling and only depth raises the
    // ceiling itself.
    let tree_depth = spec_tree.as_ref().map(|tr| tr.max_depth()).unwrap_or(0);
    let mut spec_hist = vec![0usize; spec_k.max(tree_depth) + 2];
    // WHICH candidate the verifier took, by rank, with the last bucket meaning
    // it took none. This is the number the whole tree exists to produce: a
    // rank-0 acceptance is one a plain chain would also have got, and only the
    // rank>0 column is a token breadth WON. Without it an acceptance rate says
    // the tree is working and not whether it is worth anything.
    let mut tree_rank_hist = vec![0usize; tree_b + 1];
    let mut pass_ms: Vec<f64> = Vec::new();
    // The machine under the pass, sampled per pass, for the intermittent
    // multi-second stall. Off unless `INK_STEPSTAT=1`; see
    // [`mary::models::inkling::stepstat`] for what each field decides and what
    // the sample costs.
    let stepstat_on = stepstat::enabled();
    let mut stepstat_prev = if stepstat_on {
        stepstat::StepStat::sample()
    } else {
        stepstat::StepStat::default()
    };
    let mut acc_recv = 0f64;
    let mut acc_pass = 0f64;
    let mut acc_draft = 0f64;
    // The tail's pass up to the MOMENT IT ANSWERS, which is the only part of it
    // the head is waiting for. Its printed pass includes the report and the
    // drafting that follow the reply, so subtracting THAT from the head's
    // blocked time prices the wire as negative. This is the number that makes
    // the subtraction honest.
    let mut acc_to_reply = 0f64;
    let mut acc_steps = 0usize;
    // The same wall, over the WARM steps only. The first two decode steps of a
    // pipe run cost 4.54 s and 0.55 s of kernel compilation, and averaging them
    // into a hundred 127 ms ones reports 135 ms/step -- an 6% error that lands
    // squarely on the number people quote. Two is not fitted: it is how many
    // steps print a pass an order of magnitude off the median.
    let mut warm_wall = 0f64;
    let mut warm_steps = 0usize;
    let mut warm_tokens = 0usize;
    let loop_started = Instant::now();
    let mut top_all: Vec<i64> = Vec::new();
    // A `for step in 0..=gen_steps` used to bound this, and it cannot any more:
    // a speculative pass confirms a variable number of tokens, so counting
    // passes would make the run's LENGTH a function of how well the drafts did.
    // The break at the bottom counts tokens and reproduces the old count
    // exactly when nothing is speculated.
    let mut step = 0usize;
    loop {
        // A tail's step BEGINS on the wire, and it waits before its own timers
        // start: a tail that charged itself for the head's half would report the
        // pipeline's latency as its own cost, and the per-machine split is the
        // entire question here.
        let t_rv = Instant::now();
        let incoming = match pipe.as_mut() {
            Some(Pipe::Tail(s)) => match recv_stream(s, h)? {
                Some(v) => Some(v),
                // The head closed. Not an error -- it is how a finished run ends.
                None => break,
            },
            _ => None,
        };
        // Charged whether or not this process is a tail: on anything else it is
        // zero, and a zero that is measured beats a branch that hides it.
        let t_recv = t_rv.elapsed().as_secs_f64();

        // A prefill per slot, then one decode pass per token per slot. With no
        // slot lane `prefill_passes` is 1 and `is_decode` is the `step > 0` this
        // replaces, character for character.
        let prefill_passes = if slot_lane { total_slots } else { 1 };
        let is_decode = step >= prefill_passes;
        // Which cohort this pass advances. Over the prefills it is the cohort the
        // slot being seated belongs to -- they are filled cohort by cohort -- and
        // over the decode the cohorts take turns, which IS the interleave. Both
        // ends derive it from the same step counter, and the head puts it on the
        // wire as well so a pair that ever disagreed says so.
        let coh = if !slot_lane {
            0
        } else if is_decode {
            (step - prefill_passes) % ncohorts
        } else {
            step / nslots
        };
        if let Some((_, _, wc, _)) = incoming.as_ref() {
            anyhow::ensure!(
                *wc == coh,
                "the head sent cohort {wc} where this tail was about to read cohort {coh}'s keys"
            );
        }

        let pass = Instant::now();
        // Which arm THIS pass runs, fixed before the layer loop so a pass is never
        // half of each. The prefill is always the host lane: its plan is a function
        // of the routing and nothing hoists.
        let dev_plan_now = is_decode && plan_arm.on(step - prefill_passes);
        let io0 = io_read_bytes();
        cp.io_reset();
        // Same scope as the loader counters, for the same reason: the report below
        // says "this ONE pass", and a bind total that accumulated across passes
        // would silently make it say something else.
        #[cfg(feature = "inkling-cuda")]
        if let Some(al) = fp4_aliases.as_ref() {
            al.stats_reset();
        }
        // With a cache, every pass past the prefill feeds exactly the token the
        // previous pass produced; without one, the whole prefix goes through again.
        // `pos0` is that token's ABSOLUTE position, which is what log scaling and
        // the relative bias are functions of -- it is only equal to zero on a pass
        // that starts from the beginning.
        let (feed, pos0): (Vec<usize>, usize) = if slot_lane && is_decode {
            // One row per slot: the token that slot's own previous pass produced.
            // Every slot stands at the same absolute position because they were
            // prefilled with prompts of the same length and have advanced together
            // ever since -- see [`dev_lane::SlotCache`] for what that buys.
            // This cohort's slots and no other's. Every slot of a cohort stands at
            // the same absolute position; two cohorts do NOT, because they take
            // turns, so `pos0` is read off this cohort's own slot 0.
            let base = coh * nslots;
            (
                slot_ids[base..base + nslots]
                    .iter()
                    .map(|q| *q.last().expect("every slot's prefill produced a token"))
                    .collect(),
                slot_ids[base].len() - 1,
            )
        } else if slot_lane {
            (slot_ids[step].clone(), 0)
        } else if kv && step > 0 {
            // The verify batch: the token the last pass confirmed, then the tail's
            // drafts for the positions after it. `drafts_in` is empty unless this
            // is a speculating head, so the non-speculative shape is the same one
            // row it always was.
            let mut f = vec![
                *ids.last()
                    .expect("a step past the prefill has produced a token"),
            ];
            f.extend(drafts_in.iter().copied());
            // The tree lane's candidates. Same shape as `drafts_in` and made on
            // this machine rather than read off a wire, which is the whole
            // difference between the two lanes. `f` is now the verify batch in
            // NODE order, `f[0]` being the root, which is what `accept_tree`
            // and the ancestor mask were both built to index.
            f.extend(tree_drafts.iter().copied());
            // The width probe's filler. Drawn from a counter rather than from the
            // sequence: a batch of the same token routes to the same eight experts
            // and would price the expert stream once for the whole batch, which is
            // the one thing the probe exists to find out.
            let mut lcg = 0x9E3779B97F4A7C15u64 ^ (step as u64).wrapping_mul(0x100000001B3);
            for _ in 1..width {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                f.push(((lcg >> 33) as usize) % t.vocab_size);
            }
            (f, ids.len() - 1)
        } else {
            (ids.clone(), 0)
        };
        // The tail is handed the stream the head already embedded and ran; it takes
        // `n` and `pos0` from the wire rather than from `ids`, because those are
        // facts about the pass and only the head owns the token loop.
        // Everything this pass does BEFORE the embedding -- the feed
        // construction above. Named so that "it is small" is a number in the log
        // rather than a belief about the log.
        let t_prep = pass.elapsed().as_secs_f64();
        let t_emb = Instant::now();
        // Whether THIS pass is carrying a tree. A tree lane still runs ordinary
        // one-row passes (the prefill, and any pass before the first draft), and
        // on those the descriptor must be absent rather than merely unused --
        // `attention_steps_tree` asserts its arity against the batch.
        let pass_tree_ready = tree_b > 0 && !tree_drafts.is_empty();
        let (n, pos0, x_in) = match incoming {
            Some((n, p, _c, x)) => (n, p, x),
            None => {
                let n = feed.len();
                let e_w = embed_w.as_ref().expect("the head owns the embedding table");
                let e_n = embed_n.as_ref().expect("the head owns the embedding norm");
                (
                    n,
                    pos0,
                    embed_and_norm_bf16(&feed, e_w, &e_n.data, t.rms_norm_eps, t.vocab_size, h),
                )
            }
        };
        fatal::note_tokens(n);
        // The tree descriptor for THIS pass, or `None` on a pass that is a plain
        // chain. Both conditions matter: `pass_tree_ready` says the previous pass
        // left candidates, and `n > 1` says they are actually in this batch --
        // the prefill has neither.
        let pass_tree: Option<&spectree::TreeAttn> = if pass_tree_ready && kv && is_decode && n > 1
        {
            let tr = tree_attn
                .as_ref()
                .expect("the descriptor is built whenever INK_TREE is set");
            assert_eq!(
                tr.rows, n,
                "the tree describes {} rows and the pass feeds {n}",
                tr.rows
            );
            Some(tr)
        } else {
            None
        };

        let dump_dir = std::env::var("INK_DUMP_DIR").ok();
        if let Some(dir) = dump_dir.as_ref() {
            std::fs::create_dir_all(dir)?;
            let mut bytes = Vec::with_capacity(x_in.len() * 4);
            for v in &x_in {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(format!("{dir}/h_embed.bin"), &bytes)?;
        }

        // THE upload. One per pass -- 16 KB a token -- and from here to the end of
        // this node's layers the residual stream does not touch the host again.
        //
        // A head embeds and uploads; a tail uploads what came off the wire. Both
        // are the same 16 KB, which is why the pipe crosses BETWEEN layers and not
        // inside one.
        // At the dtype the residual stream is held in. On the narrow lane the
        // upload is still f32 -- the embedding is host work and the wire is f32 --
        // and this is the one cast that pays for the whole stack staying narrow.
        let mut xd: T2 = dev_lane_resid::as_resid(up2::<Bk>(x_in, n, h, &dev));
        let t_embed = t_emb.elapsed().as_secs_f64();
        // The layer loop AND its setup, bracketed WHOLE.
        //
        // `t_attn`, `t_other`, the first-touch uploads and the pool hand-back all
        // sit inside this bracket, so `t_layers` minus their sum is in-loop work
        // that no bucket names -- which is precisely what UNATTRIBUTED used to
        // swallow without saying so. A partition needs an outer edge; this is it.
        let t_ly = Instant::now();

        // A new pass: re-arm per-layer polling if the last pass handed anything
        // back, and drop to one poll a pass if it did not.
        cleanup_gate.begin_pass();

        let ls = LogScaling {
            n_floor: t.log_scaling_n_floor as f32,
            alpha: t.log_scaling_alpha as f32,
        };
        #[allow(unused_assignments)]
        let mut expert_loads = 0usize;
        let (mut t_attn, mut t_expert, mut t_other, mut t_shared) = (0f64, 0f64, 0f64, 0f64);
        // The attention half at its two host-side seams: reading+widening the ten
        // projections out of the source, and getting them onto the device. What is
        // left over is device work. The sync after the uploads is what makes the
        // second number a TRANSFER rather than an enqueue.
        #[allow(unused_mut)]
        let (mut t_attn_read, mut t_attn_up) = (0f64, 0f64);
        // Reading a tensor out of the mapping and widening BF16 to f32 is host work
        // no lane can move, so it is counted once, separately, rather than being
        // smeared across whichever bucket happened to ask for the weight.
        let t_read = std::cell::Cell::new(0f64);
        // Host-side and therefore honestly attributable, unlike anything downstream
        // of an enqueued device call.
        let mut host_t = HostT::default();
        // The scalar host arithmetic in the block itself, which no timer above
        // covers because it is not "the attention half" or "the expert lane" -- it
        // is the connective tissue between them, and connective tissue that runs on
        // a CPU is a host path in the data plane.
        // `t_h_norm` and `t_h_resid` have no writer any more and that is the point:
        // they are printed, they read zero, and a zero that is measured is worth
        // more than a claim that is asserted.
        let (t_h_norm, mut t_h_route, mut t_h_sconv, t_h_resid) = (0f64, 0f64, 0f64, 0f64);
        // The router bucket, split three ways. It used to be one number, and one
        // number cannot distinguish "the matmul was described" from "the host waited
        // for every kernel this layer had already issued" from "the top-k ran on a
        // CPU". At decode the whole bucket is the middle one, and that is only
        // visible once the other two are subtracted out.
        let (mut t_rt_mm, mut t_rt_read, mut t_rt_host) = (0f64, 0f64, 0f64);
        // `INK_STAGE_SYNC=1` inserts an explicit device sync at each stage boundary
        // in the block and charges the wait to that stage. It is OPT-IN and off by
        // default because it SERIALISES the lane: with it on the pass is slower, and
        // the comparison that matters is probe-on-total against probe-off-total.
        // It changes no arithmetic -- a sync is a wait, not an operation -- so both
        // arms emit the same tokens, which is itself checked below.
        let stage_sync = std::env::var("INK_STAGE_SYNC")
            .map(|v| v == "1")
            .unwrap_or(false);
        // `INK_MEM_TRACE=1`: the device pool's live bytes after each stage of each
        // layer, which is the only way to find out WHICH stage holds the peak
        // rather than to model it.
        //
        // It exists because the model was wrong. A prefill's largest buffers were
        // taken to be attention's `[heads, n, head_dim]` activations, and the
        // routed-expert lane -- which gathers `k * n` rows of the residual stream
        // and runs them through two `[k * n, 4096]` intermediates -- is several
        // times larger at every length. A number that has to be inferred from the
        // source is a number nobody checks; this one is read off the allocator.
        //
        // It SYNCS, so a traced pass is slower than an untraced one and the timings
        // beside it are not the pass's. It reports memory, not time.
        let mem_trace = mem_trace();
        // Between-layer cleanup is a POLICY now, not an off switch. It was chosen
        // once before any context or worker existed, and the same value is printed
        // in the header above. See
        // `super::pool` for why the argument that kept it off -- a decode step pays
        // an allocation per layer per token for a reservation that is already small
        // -- is an argument for asking what the pass is, which the pass knows.
        // How many of this pass's layers actually handed pages back. Printed, so
        // that "the policy took the cheap branch" is a number in the log rather
        // than an inference from the absence of one.
        let mut cleanups = 0usize;
        // How many times the pass ASKED the pool what it was stranding. Printed
        // beside the cleanup count because the two used to be the same number by
        // construction and are not any more: the question is what cost, not the
        // answer. See `pool::CleanupGate`.
        let mut pool_polls = 0usize;
        // The `memory_usage` barrier alone, separated from the `/proc/meminfo`
        // read and from the cleanup's own sync, because the three have different
        // fixes and were being reported as one number.
        let mut t_pool_poll = 0f64;
        // What the hand-back COSTS, beside how often it happened. It lands between
        // `t_other` stopping and the next layer starting, so until it was timed it
        // charged its own cost to nothing.
        //
        // The naming was wrong for four months and the wrong name is what hid it:
        // on the DEFAULT policy with room to spare this line never reaches a
        // device drain or a free, because the cleanup is inside the `if` and the
        // `if` is false. Everything it measured was the QUESTION -- the
        // `memory_usage` round trip and a `/proc/meminfo` read -- charged to a
        // bracket named after the answer.
        let mut t_cleanup = 0f64;
        let (mut d_attn, mut d_router, mut d_expert, mut d_shared, mut d_tail) =
            (0f64, 0f64, 0f64, 0f64, 0f64);
        let mut stage_syncs = 0usize;

        // Per-layer diagnostics, COLLECTED rather than printed inside the loop.
        //
        // The RMS of the residual stream after each layer used to be computed on
        // the host, from a `Vec<f32>` the loop had anyway. It does not have one any
        // more, and reading `x` back per layer to print a number would reintroduce
        // exactly the round trip this change removes -- forty of them a token, to
        // produce a log line. So each layer enqueues its own reduction and the
        // whole column is read once, after the stack, in a single `cat`.
        //
        // The per-layer WALL TIME went with it, and its absence is the honest
        // report: with nothing synchronising inside the loop, "how long did layer
        // 17 take" is not a question the host can answer. It would have measured
        // enqueueing.
        let mut layer_rms: Vec<T2> = Vec::with_capacity(hi - lo);
        let mut layer_kind: Vec<(usize, bool)> = Vec::with_capacity(hi - lo);

        // One sync, charged to one stage. Written as a macro rather than a closure
        // because every call site names a different accumulator and a closure would
        // have to borrow all five of them mutably at once.
        macro_rules! stage_sync {
            ($acc:expr, $lay:expr, $tag:expr) => {
                if stage_sync {
                    let s = Instant::now();
                    <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("stage sync");
                    $acc += s.elapsed().as_secs_f64();
                    stage_syncs += 1;
                }
                // `INK_POOL_CLEANUP=stage`: four internal hand-backs here, then
                // the tail boundary below after its always-on RMS diagnostic has
                // released its full-width temporary. That is five syncs on a
                // routed layer, not five here plus a duplicate sixth at the end.
                if cleanup.at_stage() && $tag != "tail" {
                    <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("stage cleanup sync");
                    fp4_client.memory_cleanup();
                }
                if mem_trace {
                    <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("mem trace sync");
                    println!(
                        "{}",
                        mary::models::inkling::seam::pool_line(
                            &fp4_client,
                            &format!("L{} {}", $lay, $tag)
                        )
                    );
                }
            };
        }

        for layer in lo..hi {
            // Cache slot, not layer number. A tail running 20..42 keeps 22 caches
            // and its first layer is its slot 0 — indexing by the absolute layer
            // would walk off the end of a Vec that only ever holds this node's half.
            let slot = layer - lo;
            let kind = t.attn_kind(layer);
            let is_local = kind == AttnKind::Local;
            let (heads, kv_heads, head_dim) = t.heads(kind);
            let p = format!("model.llm.layers.{layer}.");
            // ONE accessor now. `g` used to exist beside it, holding an f32 copy on
            // the host for the host lanes to read; there are no host lanes left in
            // this loop, so every weight is read once, uploaded, and the host copy
            // dropped.
            let gv = |nm: &str| -> Result<Vec<f32>> {
                let s = Instant::now();
                let r = cp.tensor(&format!("{p}{nm}"))?.data;
                t_read.set(t_read.get() + s.elapsed().as_secs_f64());
                Ok(r)
            };

            // ---- this layer's weights, on the device, built once ---------------
            //
            // Everything the block multiplies by: the ten attention projections,
            // BOTH short convolutions, BOTH norm gains and the router matrix. The
            // norms and the mlp short convolution used to be read per token as
            // host `Vec<f32>` because the operations that consumed them were host
            // operations. They are not any more, so they join the rest.
            if !layers_dev.contains_key(&p) {
                let r0 = t_read.get();
                let t_w0 = Instant::now();
                // The five projections bind as the BF16 the pile stores. `gv`
                // widens to f32 on the way out of the mapping and would double
                // every one of them on the device for nothing: `mma.sync…bf16`
                // takes the stored bytes, and where those bytes are inside a
                // registered mapping `bind_bf16` aliases them instead of copying.
                let pw = |nm: &str, rows: usize, cols: usize| -> Result<Bf16W> {
                    let s = Instant::now();
                    let leaf = cp.stored(&format!("{p}{nm}"))?;
                    anyhow::ensure!(
                        leaf.elem == Elem::Bf16,
                        "{p}{nm} is {:?}; this lane multiplies BF16 by BF16",
                        leaf.elem
                    );
                    t_read.set(t_read.get() + s.elapsed().as_secs_f64());
                    Ok(bind_bf16(
                        &fp4_client,
                        fp4_aliases.as_ref(),
                        &leaf.bytes,
                        rows,
                        cols,
                    ))
                };
                // The concatenation reads the same four leaves `pw` binds, in
                // the output order [`dev_lane::project_qkvr`] slices back.
                let fused_qkvr = if dev_lane::fuse_qkvr() {
                    let mut b: Vec<u8> = Vec::new();
                    for nm in [
                        "attn.wq_du.weight",
                        "attn.wk_dv.weight",
                        "attn.wv_dv.weight",
                        "attn.wr_du.weight",
                    ] {
                        b.extend_from_slice(&cp.stored(&format!("{p}{nm}"))?.bytes);
                    }
                    let rows = (heads * head_dim) + 2 * (kv_heads * head_dim) + heads * t.d_rel;
                    Some(bind_bf16(&fp4_client, fp4_aliases.as_ref(), &b, rows, h))
                } else {
                    None
                };
                let attn = dev_lane::AttnWeightsDev {
                    wq: pw("attn.wq_du.weight", heads * head_dim, h)?,
                    wk: pw("attn.wk_dv.weight", kv_heads * head_dim, h)?,
                    wv: pw("attn.wv_dv.weight", kv_heads * head_dim, h)?,
                    wr: pw("attn.wr_du.weight", heads * t.d_rel, h)?,
                    wqkvr: fused_qkvr,
                    wo: pw("attn.wo_ud.weight", h, heads * head_dim)?,
                    k_sconv: up2(
                        gv("attn.k_sconv.weight")?,
                        kv_heads * head_dim,
                        t.sconv_kernel_size,
                        &dev,
                    ),
                    v_sconv: up2(
                        gv("attn.v_sconv.weight")?,
                        kv_heads * head_dim,
                        t.sconv_kernel_size,
                        &dev,
                    ),
                    q_norm: up1(gv("attn.q_norm.weight")?, head_dim, &dev),
                    k_norm: up1(gv("attn.k_norm.weight")?, head_dim, &dev),
                    rel_proj: up2(
                        gv("attn.rel_logits_proj.proj")?,
                        t.d_rel,
                        t.rel_span(kind),
                        &dev,
                    ),
                };
                let router = if t.is_dense(layer) {
                    None
                } else {
                    let rows = t.n_routed_experts + t.n_shared_experts;
                    // `gv` widens on the way out of the mapping, so the BF16 arm
                    // must not call it for the projection at all -- that read IS the
                    // widening. It takes `stored` instead, like the five attention
                    // projections beside it.
                    let proj = match router_arm {
                        RouterArm::Transpose => {
                            RouterProj::PerCall(up2(gv("mlp.gate.weight")?, rows, h, &dev))
                        }
                        // Transposed HERE, on the host, once per layer for the run,
                        // instead of on the device once per token per layer.
                        RouterArm::Pre => RouterProj::Pre(up2(
                            transpose_rows(&gv("mlp.gate.weight")?, rows, h),
                            h,
                            rows,
                            &dev,
                        )),
                        RouterArm::Bf16 => {
                            use mary::models::inkling::bf16gemm::NTILE;
                            let s = Instant::now();
                            let leaf = cp.stored(&format!("{p}mlp.gate.weight"))?;
                            anyhow::ensure!(
                                leaf.elem == Elem::Bf16,
                                "{p}mlp.gate.weight is {:?}; INK_ROUTER=bf16 multiplies the STORED \
                             BF16 and will not widen to reach an element type",
                                leaf.elem
                            );
                            // Six zero rows so 258 tiles as n8. They produce logits
                            // the host slices off; they are a pad, not a widening.
                            let pad = rows.div_ceil(NTILE) * NTILE;
                            let mut bytes = vec![0u8; pad * h * 2];
                            bytes[..rows * h * 2].copy_from_slice(&leaf.bytes);
                            t_read.set(t_read.get() + s.elapsed().as_secs_f64());
                            RouterProj::Bf16 {
                                w: bind_bf16(&fp4_client, fp4_aliases.as_ref(), &bytes, pad, h),
                            }
                        }
                    };
                    // Held only under the diff probe, and it is the arm every run
                    // before 969bf6f shipped -- the thing the new arm has to be
                    // compared AGAINST, not a second opinion invented for the
                    // occasion.
                    let reference = if router_diff {
                        Some(up2(gv("mlp.gate.weight")?, rows, h, &dev))
                    } else {
                        None
                    };
                    Some(RouterDev {
                        proj,
                        reference,
                        bias: gv("mlp.gate.bias")?,
                        global_scale: gv("mlp.gate.global_scale")?[0],
                    })
                };
                let built = LayerDev {
                    attn,
                    attn_sconv: up2(gv("attn_sconv.weight")?, h, t.sconv_kernel_size, &dev),
                    mlp_sconv: up2(gv("mlp_sconv.weight")?, h, t.sconv_kernel_size, &dev),
                    attn_norm: up1(gv("attn_norm.weight")?, h, &dev),
                    mlp_norm: up1(gv("mlp_norm.weight")?, h, &dev),
                    router,
                };
                <Bk as burn::tensor::backend::Backend>::sync(&dev)
                    .expect("sync after the layer uploads");
                let span = t_w0.elapsed().as_secs_f64();
                let rd = t_read.get() - r0;
                t_attn_read += rd;
                t_attn_up += span - rd;
                // Two bytes for the projections and four for the rest, because that
                // is now what is on the device. Counting the projections at four
                // would report the widening this commit removed.
                dattn_bytes += (2
                    * (heads * head_dim * h
                        + 2 * kv_heads * head_dim * h
                        + heads * t.d_rel * h
                        + h * heads * head_dim)
                    + 4 * (2 * kv_heads * head_dim * t.sconv_kernel_size
                        + 2 * head_dim
                        + t.d_rel * t.rel_span(kind)
                        + 2 * h * t.sconv_kernel_size
                        + 2 * h)) as u64;
                layers_dev.insert(p.clone(), built);
            }
            let ld = layers_dev.get(&p).expect("inserted directly above");

            // ---- attention ----------------------------------------------------
            let t_a = Instant::now();
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
            // The same distinction the mask carries, in the form the cache needs:
            // how far back a query may look, and therefore how much of the cache
            // can never be read again.
            let window = if is_local {
                Some(t.sliding_window_size)
            } else {
                None
            };
            let a = if slot_lane && is_decode {
                // b sequences, one position each, one pass. `attention_steps` is
                // the wrong function here and not by a little: its rows are
                // consecutive positions of ONE sequence and its mask admits every
                // earlier row of the batch, which for independent slots is exactly
                // the contamination this lane exists to make impossible.
                let y = dev_lane::attention_slots(
                    hn,
                    &ld.attn,
                    &dims,
                    Some(ls),
                    pos0,
                    window,
                    &mut slots_dev[coh][slot].attn,
                );
                let (out, hist) = dev_lane::short_conv_slot_step(
                    slots_dev[coh][slot].attn_sconv.clone(),
                    y,
                    ld.attn_sconv.clone(),
                );
                slots_dev[coh][slot].attn_sconv = hist;
                out
            } else if kv && is_decode && n > 1 {
                // The speculative width. `attention_steps` leaves the batch PENDING
                // and the convolution keeps its whole window, so neither is final
                // until the verifier below says how many rows survived. Nothing
                // here knows that yet -- the answer is a machine away.
                let y = dev_lane::attention_steps_tree(
                    hn,
                    &ld.attn,
                    &dims,
                    Some(ls),
                    pos0,
                    window,
                    &mut caches[slot].attn,
                    pass_tree,
                );
                // Two of the FOUR short convolutions a widened pass runs are
                // inside the attention above; these are the block's own, and
                // they need the same taps for the same reason. The batched
                // kernel reads the rows physically preceding row `i`, which for
                // a chain are its ancestors and for a tree are whatever the
                // layout put there -- and the failure is silent, because masked
                // attention does not reach a convolution and the numbers stay
                // finite.
                let (out, all) = match pass_tree {
                    None => dev_lane::short_conv_steps(
                        caches[slot].attn_sconv.clone(),
                        y,
                        ld.attn_sconv.clone(),
                    ),
                    Some(tr) => dev_lane::short_conv_tree_steps(
                        caches[slot].attn_sconv.clone(),
                        y,
                        ld.attn_sconv.clone(),
                        &tr.taps,
                    ),
                };
                caches[slot].attn_sconv_pending = Some(all);
                out
            } else if kv && is_decode {
                let y = dev_lane::attention_step(
                    hn,
                    &ld.attn,
                    &dims,
                    Some(ls),
                    pos0,
                    window,
                    &mut caches[slot].attn,
                );
                let (out, hist) = dev_lane::short_conv_step(
                    caches[slot].attn_sconv.clone(),
                    y,
                    ld.attn_sconv.clone(),
                );
                caches[slot].attn_sconv = hist;
                out
            } else if kv {
                let (y, attn) =
                    dev_lane::attention_prefill(hn, &ld.attn, &dims, Some(ls), window, window);
                let hist = dev_lane::conv_history(y.clone(), t.sconv_kernel_size);
                let out = dev_lane::short_conv(y, ld.attn_sconv.clone());
                caches.push(LayerCache {
                    attn,
                    attn_sconv: hist,
                    mlp_sconv: None,
                    attn_sconv_pending: None,
                    mlp_sconv_pending: None,
                });
                out
            } else {
                let y = dev_lane::attention(hn, &ld.attn, &dims, Some(ls), window);
                dev_lane::short_conv(y, ld.attn_sconv.clone())
            };
            xd = dev_lane_resid::add_resid(xd, a);

            stage_sync!(d_attn, layer, "attn");
            // ---- MLP ----------------------------------------------------------
            t_attn += t_a.elapsed().as_secs_f64();
            let t_o = Instant::now();
            let hn = dev_lane_resid::rms_norm(xd.clone(), ld.mlp_norm.clone(), t.rms_norm_eps);

            let y = if t.is_dense(layer) {
                // Device-resident: uploaded on the first token that reaches this
                // layer and held for the run. The host reference that used to sit
                // beside it (`host_dense`, selected by leaving `INK_DENSE` unset)
                // was a scalar f32 lane over a 537 MB weight; it is not a lane a
                // 276 B model has any use for, and being selectable is how it got
                // run by accident.
                let w = ddense.dense_for(&cp, &fp4_client, fp4_aliases.as_ref(), &p, h)?;
                dense_mlp_bf16(hn, w)
            } else {
                let inter = t.intermediate_size;
                let r = ld.router.as_ref().expect("a MoE layer has a router");
                // The router's PROJECTION is a matmul and runs on the device; its
                // DECISION is control plane and runs here. What crosses is one row
                // of the chosen ids and weights on the default lane and the whole
                // [n, 258] f32 on `INK_DEV_ROUTE=0` -- 60 bytes against 1 KB, and
                // neither is the cost. The cost is that this read BLOCKS, and it is
                // the only place in the layer that does.
                //
                // What it is worth, measured 2026-08-23 on spark-zt at
                // `INK_KV=1 INK_LAYERS=0:16`, nine interleaved rounds of the
                // `INK_ROUTE_STALE=1` probe against the same binary with the probe
                // off: 55.7 -> 51.2 ms p50, i.e. 4.5 ms a token and 17.95 -> 19.53
                // tok/s. The per-layer `BLOCKING read` bucket is 19.3 ms of that
                // pass and only 4.5 of it is serialisation -- the rest is device
                // time that resurfaces at the one sync (1.3 -> 5.1 ms) and host
                // work that gets dearer once the queue runs deep. `nsys -t cuda`
                // over the same two arms puts GPU kernel time at 41.2 ms of the
                // pass, so the device is busy 74% of a blocking pass and 80% of a
                // probe pass, and 41 ms is the floor either way.
                //
                // Reproduced 2026-08-25 at commit 56c1ebbcdff6, same box, same
                // lane, `bench-decode.sh -n 3 --gen 12 --layers 0:16` on the
                // 3732-token cover, four arms interleaved, medians over 33 warm
                // passes. It also settles what `INK_DEV_PLAN=1` is worth, which
                // the probe alone could not say -- the probe is not a lane:
                //
                //   arm                 ms/step   `router + group`   BLOCKING read
                //   base                   58.0             23.8            23.5
                //   INK_ROUTE_STALE=1      54.5              0.4             0.0
                //   INK_DEV_PLAN=1         54.5              0.3             0.0
                //
                // The two non-base arms land on the SAME 54.5, which is the
                // result: `INK_DEV_PLAN` captures all of what removing the read
                // can capture, and there is nothing further in the router. So the
                // 23.8 ms bucket is 3.5 ms of serialisation, 0.2 ms of describing
                // the router matmul, 0.0 ms of host top-k, and ~20 ms of GPU work
                // the host would have waited for somewhere. Under `INK_DEV_PLAN`
                // that 20 ms reappears as attention-half ENQUEUE (11.6 -> 22.8 ms,
                // back-pressure, not work) and at the one sync (1.3 -> 5.4).
                //
                // A second run the same day, base against `INK_DEV_PLAN=1` alone
                // and five reps interleaved, is the decision-grade one: 59.8 ->
                // 55.2 ms/step median, +8.33%, and all FIVE pairs favour the
                // device plan by 2.8-5.5 ms, which is the part a median cannot
                // say. Base's spread is 4.3%, the device plan's 1.6% -- the
                // removed sync is the jittery part.
                //
                // `INK_DEV_PLAN=1` emitted a token stream identical to base in
                // every one of those runs -- one md5 over all twelve steps, eight
                // runs of each arm across the two rounds -- and raised no fault
                // flag in any of them.
                let rows = t.n_routed_experts + t.n_shared_experts;
                let t_rt = Instant::now();
                // `cols` is what comes BACK, which is `rows` except on the BF16 arm,
                // whose weight carries the instruction's n padding.
                let (lg, cols) = match &r.proj {
                    // Two A/B arms whose weight is an f32 Burn tensor, so their
                    // matmul wants an f32 activation. `Bf16` is the lane that runs
                    // and it takes the narrow stream as it lies; these widen, and
                    // the temporary is the price of keeping an arm comparable
                    // rather than of anything on the path.
                    RouterProj::PerCall(w) => (
                        dev_lane::linear(dev_lane_resid::from_resid(hn.clone()), w.clone()),
                        rows,
                    ),
                    RouterProj::Pre(wt) => (
                        dev_lane::linear_pre_t(dev_lane_resid::from_resid(hn.clone()), wt.clone()),
                        rows,
                    ),
                    RouterProj::Bf16 { w, .. } => (dev_lane::linear_bf16(hn.clone(), w), w.n),
                };
                t_rt_mm += t_rt.elapsed().as_secs_f64();
                stage_sync!(d_router, layer, "router");
                // Two lanes, and the difference is WHERE the top-k runs, not what
                // it decides. The host lane reads `[n, rows]` f32 back and sorts;
                // the device lane runs `routetopk` on the logits where they already
                // are and reads back `[n, 2k + shared + 1]`, which at 512 tokens is
                // 30 KB against 528 KB and, more to the point, is 512 rows of
                // 256-wide selection the host no longer walks.
                //
                // The reference arm below wants the full host logits, so it selects
                // the host lane: a diagnostic that changed the lane it measures
                // would be measuring itself.
                let host_route = !dev_route || r.reference.is_some();
                // Can this layer's plan stay on the device this pass? Every clause
                // is a separate reason and none of them is a preference:
                //
                // * `n <= MTILE` is the shape the invariants hold at. It was
                //   `n == 1`, which is where they hold when the plan DEDUPS
                //   experts; `devroute_new` no longer does, so they hold for
                //   every width whose tokens still fit one tile an expert --
                //   i.e. every `n` up to `MTILE`. Above it a slot would need
                //   more than one tile and the count would be data-dependent
                //   again, which is a prefill and keeps the host lane.
                //   `INK_DEV_PLAN_MAXN` narrows the band without a rebuild, and
                //   `=1` restores the old behaviour exactly.
                // * the diagnostics below read `routing`, which this lane does not
                //   produce -- so they select the host lane rather than being
                //   silently wrong.
                // * `INK_GROUPED=0` is the per-expert loop, which has no plan.
                // * a BF16-expert layer goes through a different lane entirely.
                let plan_dev_ok = dev_plan_now
                    && n <= dev_plan_maxn
                    && dev_route
                    && !route_stale
                    && grouped_mode != "0"
                    && !route_log_on
                    && !route_dbg_on
                    && r.reference.is_none();
                // The layer's weight table, derived on its first device-lane pass
                // and held for the run. A layer that cannot be tabled caches its
                // `None` and takes the host lane every pass thereafter.
                let plan_dev = if plan_dev_ok {
                    // Keyed by `n`: the invariants are derived at a width and
                    // are wrong at any other. A decode run holds one width, so
                    // this fires once -- and it carries the weight TABLES over,
                    // because those are per-layer facts about the checkpoint and
                    // have nothing to do with how many rows the pass feeds.
                    if devroute.as_ref().is_some_and(|d| d.n != n) {
                        let old = devroute.take().expect("checked");
                        let mut fresh = devroute_new(&fp4_client, t.num_experts_per_tok, n);
                        fresh.tabs = old.tabs;
                        devroute = Some(fresh);
                    }
                    let dr = devroute
                        .get_or_insert_with(|| devroute_new(&fp4_client, t.num_experts_per_tok, n));
                    if !dr.tabs.contains_key(&layer) {
                        let t_s = Instant::now();
                        // Which table the layer's own bytes ask for. Both lanes
                        // are grouped and both take a plan; only the shape of the
                        // weight differs, so this is the same branch the expert
                        // dispatch below makes and it is made from the same fact.
                        let nvfp4 = cp.is_nvfp4(&format!("{p}mlp.experts.w13_weight"));
                        let tb = match fp4_aliases.as_ref() {
                            Some(al) if nvfp4 => {
                                build_expert_table(&cp, al, &fp4_client, &p, t.n_routed_experts)?
                            }
                            Some(al) => build_expert_table_bf16(
                                &cp,
                                al,
                                &fp4_client,
                                &p,
                                t.n_routed_experts,
                            )?,
                            None => None,
                        };
                        host_t.slice += t_s.elapsed().as_secs_f64();
                        if tb.is_none() {
                            println!(
                                "  INK_DEV_PLAN: {p} keeps the host lane (no single aligned mapping \
                             for all {} routed experts)",
                                t.n_routed_experts
                            );
                        }
                        dr.tabs.insert(layer, tb);
                    }
                    dr.tabs[&layer].is_some()
                } else {
                    false
                };
                // Two independent questions, and conflating them is how a
                // diagnostic ends up measuring a lane that was not running: does
                // the PLAN come from the device, and does the host still read the
                // decision back? The second is true whenever something downstream
                // needs `routing`, whether or not the first is.
                let need_routing = !plan_dev || grouped_ab || devplan_verify;
                let routing: Vec<Routing>;
                let mut topk_h: Option<cubecl::server::Handle> = None;
                let mut topk_width = 0usize;
                let mut logits: Vec<f32> = Vec::new();
                // The probe only stands in for a decision of the same SHAPE, and it
                // never stands in for the reference arm, which wants this pass's
                // logits and would otherwise be handed an empty vector.
                let stale_hit = route_stale
                    && r.reference.is_none()
                    && route_cache.get(&layer).is_some_and(|v| v.len() == n);
                if stale_hit {
                    routing = route_cache[&layer].clone();
                } else if host_route {
                    let t_rr = Instant::now();
                    logits = drop_pad_cols(down(lg), n, cols, rows);
                    t_rt_read += t_rr.elapsed().as_secs_f64();
                    let t_rh = Instant::now();
                    routing = route_from_logits(
                        &logits,
                        &r.bias,
                        r.global_scale,
                        t.route_scale as f32,
                        n,
                        t.n_routed_experts,
                        t.n_shared_experts,
                        t.num_experts_per_tok,
                    );
                    t_rt_host += t_rh.elapsed().as_secs_f64();
                } else {
                    use mary::models::inkling::routetopk::router_topk_launch;
                    use mary::models::inkling::seam::{handle_of, tensor_of};
                    let k = t.num_experts_per_tok;
                    let ns = t.n_shared_experts;
                    let width = 2 * k + ns + 1;
                    let bias_h = bias_dev
                        .entry(layer)
                        .or_insert_with(|| fp4_client.create_from_slice(bytes_of(&r.bias)))
                        .clone();
                    let t_rt2 = Instant::now();
                    let lg_h = handle_of(lg);
                    let out_h = router_topk_launch(
                        &fp4_client,
                        &lg_h,
                        &bias_h,
                        n,
                        cols,
                        t.n_routed_experts,
                        ns,
                        k,
                        t.route_scale as f32 * r.global_scale,
                    );
                    t_rt_mm += t_rt2.elapsed().as_secs_f64();
                    topk_width = width;
                    topk_h = Some(out_h.clone());
                    if !need_routing {
                        // THE POINT OF THE WHOLE LANE. The answer stays where it
                        // was computed; `plan_from_topk_launch` below reads it with
                        // a kernel. Nothing on the host waits for the device in
                        // this layer, so the queue runs as deep as the stack.
                        routing = Vec::new();
                    } else {
                        let t_rr = Instant::now();
                        let flat =
                            down(tensor_of(fp4_client.clone(), dev.clone(), out_h, n, width));
                        t_rt_read += t_rr.elapsed().as_secs_f64();
                        let t_rh = Instant::now();
                        let mut rs = Vec::with_capacity(n);
                        for ti in 0..n {
                            let row = &flat[ti * width..(ti + 1) * width];
                            let bad = row[width - 1] as u32;
                            assert!(
                                bad == 0,
                                "router logit is non-finite at token {ti}, row {}",
                                bad - 1
                            );
                            rs.push(Routing {
                                experts: row[..k].iter().map(|&v| v as usize).collect(),
                                weights: row[k..2 * k].to_vec(),
                                shared_gammas: row[2 * k..2 * k + ns].to_vec(),
                            });
                        }
                        routing = rs;
                        // `INK_ROUTE_DBG=1`: the SAME logits through the host rule, and
                        // a count of where the two lanes disagree. It reads the logits
                        // back and routes twice, so it is slower than either lane and
                        // is not a lane -- but it compares the two on ONE input, which
                        // is the thing two separate runs cannot do. A device router and
                        // a host router started from the same prompt part company after
                        // a few layers no matter how right they both are: the routing
                        // weights differ in the eighth decimal, the residual stream
                        // carries that forward, and a later layer has a near-tie at the
                        // top-k boundary that falls the other way. That is chaos, not
                        // error, and only a same-input comparison can tell them apart.
                        if std::env::var("INK_ROUTE_DBG").is_ok() {
                            let hl = drop_pad_cols(
                                down(tensor_of(
                                    fp4_client.clone(),
                                    dev.clone(),
                                    lg_h.clone(),
                                    n,
                                    cols,
                                )),
                                n,
                                cols,
                                rows,
                            );
                            let hr = route_from_logits(
                                &hl,
                                &r.bias,
                                r.global_scale,
                                t.route_scale as f32,
                                n,
                                t.n_routed_experts,
                                ns,
                                k,
                            );
                            let mut bad = 0usize;
                            let mut worst_w = 0f32;
                            for ti in 0..n {
                                for j in 0..k {
                                    let d = (routing[ti].weights[j] - hr[ti].weights[j]).abs();
                                    if d > worst_w {
                                        worst_w = d;
                                    }
                                }
                                for j in 0..ns {
                                    let d = (routing[ti].shared_gammas[j]
                                        - hr[ti].shared_gammas[j])
                                        .abs();
                                    if d > worst_w {
                                        worst_w = d;
                                    }
                                }
                                if hr[ti].experts != routing[ti].experts {
                                    bad += 1;
                                    if bad <= 4 {
                                        println!(
                                            "ROUTEDBG layer {layer} t {ti} host {:?} dev {:?}",
                                            hr[ti].experts, routing[ti].experts
                                        );
                                    }
                                }
                            }
                            println!(
                                "ROUTEGATE layer {layer}: {n} rows examined, {bad} selections differ, \
                         max |dev-host| weight {worst_w:.3e}"
                            );
                        }
                        t_rt_host += t_rh.elapsed().as_secs_f64();
                    }
                }
                if route_stale && !stale_hit {
                    route_cache.insert(layer, routing.clone());
                }
                let t_rh = Instant::now();

                // `INK_ROUTER_DIFF=1`: the same activation through the f32 lane,
                // the same selection rule on the result, and a count of where the
                // two disagree. Nothing below reads `ref_routing` -- the run acts on
                // `routing`, whichever arm produced it -- so this measures the arm
                // rather than replacing it.
                if let Some(rw) = r.reference.as_ref() {
                    // `INK_ROUTER_DIFF=1` only, and widened for the same reason
                    // the two f32 router arms above are: this lane's weight is an
                    // f32 Burn tensor.
                    let ref_logits = down(dev_lane::linear(
                        dev_lane_resid::from_resid(hn.clone()),
                        rw.clone(),
                    ));
                    let ref_routing = route_from_logits(
                        &ref_logits,
                        &r.bias,
                        r.global_scale,
                        t.route_scale as f32,
                        n,
                        t.n_routed_experts,
                        t.n_shared_experts,
                        t.num_experts_per_tok,
                    );
                    let d = &mut route_diff[layer];
                    for ti in 0..n {
                        d.note(
                            &routing[ti],
                            &ref_routing[ti],
                            &logits[ti * rows..(ti + 1) * rows],
                            &ref_logits[ti * rows..(ti + 1) * rows],
                        );
                    }
                }

                // One launch, one unit, and the layer's whole plan. Enqueued, so
                // the timer is the host's description of the work and not the work.
                let dev_plan_out = if plan_dev {
                    use mary::models::inkling::fp4gemm::MTILE;
                    let th = topk_h
                        .clone()
                        .expect("the device route lane produced a top-k buffer");
                    let dr = devroute
                        .as_ref()
                        .expect("a device plan implies the run state");
                    let tb = dr.tabs[&layer]
                        .as_ref()
                        .expect("a device plan implies a table");
                    Some(mary::models::inkling::devplan::plan_from_topk_launch(
                        &fp4_client,
                        &th,
                        tb,
                        &dr.fault,
                        dr.kmax,
                        MTILE,
                        topk_width,
                        n,
                    ))
                } else {
                    None
                };

                // Group tokens by expert, so each slab is read once. Empty on the
                // device lane -- `routing` is empty there and nothing below reads
                // this -- which is the one-line statement of what moved.
                let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
                for (ti, rt) in routing.iter().enumerate() {
                    for (slot, &e) in rt.experts.iter().enumerate() {
                        by_expert.entry(e).or_default().push((ti, rt.weights[slot]));
                    }
                }
                t_rt_host += t_rh.elapsed().as_secs_f64();
                t_h_route += t_rt.elapsed().as_secs_f64();

                // `INK_ROUTE_LOG=<path>` appends this layer's routing, one line per
                // position, plus the layer's DISTINCT-expert count. The count is
                // written so the file can be checked against `expert slabs
                // decoded` instead of trusted: the sum of the `distinct=` lines of
                // a pass must equal that counter, and a log that disagrees with the
                // number the run acted on is describing a different run.
                //
                // Written from `routing`, the same vector the block then feeds to
                // the experts -- not recomputed -- so the log cannot drift from
                // what was decoded. Positions are ABSOLUTE (`pos0 + ti`), because a
                // cached step's `ti` is always 0 and the sequence position is the
                // whole point of an adjacency measurement.
                if let Ok(rl) = std::env::var("INK_ROUTE_LOG") {
                    use std::io::Write as _;
                    let mut f = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&rl)
                        .with_context(|| format!("INK_ROUTE_LOG={rl}"))?;
                    for (ti, r) in routing.iter().enumerate() {
                        writeln!(f, "R {step} {layer} {} {:?}", pos0 + ti, r.experts)?;
                    }
                    writeln!(f, "D {step} {layer} {}", by_expert.len())?;
                }

                // `INK_DEVPLAN_CHECK=1`: the device plan against the host plan as
                // BITS, before either is used. This is where a wrong sort is
                // visible; downstream it is four ulps inside a runtime that
                // disagrees with itself by far more than that.
                if devplan_verify {
                    if let (Some(dp), Some(al)) = (dev_plan_out.as_ref(), fp4_aliases.as_ref()) {
                        let dr = devroute
                            .as_ref()
                            .expect("a device plan implies the run state");
                        let tb = dr.tabs[&layer]
                            .as_ref()
                            .expect("a device plan implies a table");
                        devplan_verify_layer(
                            &cp,
                            al,
                            &fp4_client,
                            &p,
                            &routing,
                            dp,
                            dr,
                            tb.scaled,
                        )?;
                    }
                }

                let t_d = Instant::now();
                // Two formats, two instructions, one lane. 41 of the 42 layers
                // carry NVFP4 experts and go through the block-scaled MMA; layer 2
                // carries plain BF16 ones -- no `.scale` sidecar, because nothing
                // quantised them -- and goes through the unscaled BF16 MMA
                // (`mma.sync...bf16`, f32 accumulator, which is the instruction's
                // own output type and not a widening).
                //
                // Which one is decided by the pile, not by a flag.
                let acc = if let Some(dp) = dev_plan_out.as_ref() {
                    let t_g = Instant::now();
                    let dr = devroute
                        .as_ref()
                        .expect("a device plan implies the run state");
                    let tb = dr.tabs[&layer]
                        .as_ref()
                        .expect("a device plan implies a table");
                    let a = if tb.scaled {
                        routed_experts_fp4_dev(
                            &fp4_client,
                            &dev,
                            &p,
                            tb,
                            dp,
                            dr,
                            &hn,
                            n,
                            h,
                            inter,
                            cp.experts_swizzled(),
                            t_g,
                            &mut host_t,
                        )
                    } else {
                        routed_experts_bf16_dev(
                            &fp4_client,
                            &dev,
                            tb,
                            dp,
                            dr,
                            &hn,
                            n,
                            h,
                            inter,
                            t_g,
                            &mut host_t,
                        )
                    };
                    // The accounting the host lane did one bind at a time. Nothing
                    // on the host sees these binds any more, and a lane that
                    // quietly stopped reporting would look like a lane that stopped
                    // moving bytes -- see `Aliases::note_alias`.
                    if let Some(al) = fp4_aliases.as_ref() {
                        for _ in 0..dr.k {
                            al.note_alias(tb.expert_bytes);
                        }
                    }
                    if grouped_ab {
                        let reference = if tb.scaled {
                            per_expert_fp4(
                                &cp,
                                fp4_aliases.as_ref(),
                                &fp4_client,
                                &dev,
                                &p,
                                &by_expert,
                                &hn,
                                n,
                                h,
                                inter,
                                &mut host_t,
                            )?
                        } else {
                            per_expert_bf16(
                                &cp,
                                fp4_aliases.as_ref(),
                                &fp4_client,
                                &dev,
                                &p,
                                &by_expert,
                                &hn,
                                n,
                                h,
                                inter,
                                &mut host_t,
                            )?
                        };
                        report_ab(&p, &a, &reference, h);
                    }
                    host_t.grouped += 1;
                    host_t.plan_dev += 1;
                    expert_loads += dr.k;
                    a
                } else {
                    let a = if cp.is_nvfp4(&format!("{p}mlp.experts.w13_weight")) {
                        routed_experts_fp4(
                            &cp,
                            fp4_aliases.as_ref(),
                            &fp4_client,
                            &dev,
                            &p,
                            &by_expert,
                            &hn,
                            n,
                            h,
                            inter,
                            admission.routed(layer) == budget::StorageDType::Bf16,
                            &mut host_t,
                        )?
                    } else {
                        routed_experts_bf16(
                            &cp,
                            fp4_aliases.as_ref(),
                            &fp4_client,
                            &dev,
                            &p,
                            &by_expert,
                            &hn,
                            n,
                            h,
                            inter,
                            &mut host_t,
                        )?
                    };
                    expert_loads += by_expert.len();
                    host_t.plan_host += 1;
                    a
                };
                // ENQUEUE time, not work: nothing in this lane synchronises any
                // more. The layer's device time shows up in the one sync after the
                // stack, which is where it belongs and where it cannot be
                // misattributed to whichever bucket happened to hold the readback.
                t_expert += t_d.elapsed().as_secs_f64();
                stage_sync!(d_expert, layer, "expert");

                let ns = t.n_shared_experts;
                let t_s = Instant::now();
                // Device-resident, uploaded once. `split_shared_w13` is the
                // settled reading — this used to be an open `deinterleave_rows`
                // here and a halved split in the gate, which is the contradiction
                // the INTERLEAVED result closed.
                let sh = {
                    let sw = ddense.shared_for(
                        &cp,
                        &fp4_client,
                        fp4_aliases.as_ref(),
                        &p,
                        ns,
                        inter,
                        h,
                        shared_halved,
                    )?;
                    match (dev_plan_out.as_ref(), topk_h.as_ref()) {
                        // The shared gammas rode back in the same readback the
                        // routed weights did, so they are the other half of what
                        // this lane deletes. Sliced out of `routetopk`'s own
                        // output, they are the same f32 values in the same order.
                        (Some(_), Some(th)) => {
                            let g = mary::models::inkling::seam::tensor_of(
                                fp4_client.clone(),
                                dev.clone(),
                                th.clone(),
                                n,
                                topk_width,
                            );
                            shared_experts_dev(hn, sw, g, t.num_experts_per_tok, ns)
                        }
                        _ => {
                            let gammas: Vec<f32> = routing
                                .iter()
                                .flat_map(|rt| rt.shared_gammas.clone())
                                .collect();
                            shared_experts_bf16(&dev, hn, sw, &gammas, ns)
                        }
                    }
                };
                stage_sync!(d_shared, layer, "shared");
                t_shared += t_s.elapsed().as_secs_f64();
                acc + sh
            };

            // The MLP half's own short convolution carries state across generated
            // tokens exactly as attention's do.
            let t_sc = Instant::now();
            let y = if slot_lane && is_decode {
                let hist = slots_dev[coh][slot]
                    .mlp_sconv
                    .clone()
                    .expect("a slot batch carries its own convolution memory");
                let (out, next) = dev_lane::short_conv_slot_step(hist, y, ld.mlp_sconv.clone());
                slots_dev[coh][slot].mlp_sconv = Some(next);
                out
            } else if kv {
                if is_decode {
                    let hist = caches[slot]
                        .mlp_sconv
                        .clone()
                        .expect("a step past the prefill has a history");
                    if n > 1 {
                        // The fourth convolution, and the same taps again.
                        let (out, all) = match pass_tree {
                            None => dev_lane::short_conv_steps(hist, y, ld.mlp_sconv.clone()),
                            Some(tr) => dev_lane::short_conv_tree_steps(
                                hist,
                                y,
                                ld.mlp_sconv.clone(),
                                &tr.taps,
                            ),
                        };
                        caches[slot].mlp_sconv_pending = Some(all);
                        out
                    } else {
                        let (out, next) = dev_lane::short_conv_step(hist, y, ld.mlp_sconv.clone());
                        caches[slot].mlp_sconv = Some(next);
                        out
                    }
                } else {
                    caches[slot].mlp_sconv =
                        Some(dev_lane::conv_history(y.clone(), t.sconv_kernel_size));
                    dev_lane::short_conv(y, ld.mlp_sconv.clone())
                }
            } else {
                dev_lane::short_conv(y, ld.mlp_sconv.clone())
            };
            t_h_sconv += t_sc.elapsed().as_secs_f64();
            xd = dev_lane_resid::add_resid(xd, y);
            stage_sync!(d_tail, layer, "tail");

            // A debug dump is a SYNC, and it is the one place left in the loop that
            // costs one. That is the trade: this path exists to compare against a
            // Python capture layer by layer, and it cannot do that without the
            // numbers.
            if let Some(dir) = dump_dir.as_ref() {
                let hx = down(xd.clone());
                let mut bytes = Vec::with_capacity(hx.len() * 4);
                for v in &hx {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(format!("{dir}/h_after_{layer:02}.bin"), &bytes)?;
            }
            t_other += t_o.elapsed().as_secs_f64();
            // The same reduction the host loop did, enqueued rather than run:
            // sqrt(mean(x^2)) over the whole [n, h] stream. Read after the stack.
            // Widened first, and deliberately: `mean` over `n * 4096` terms is the
            // one reduction in this loop whose accumulator is not already f32, and
            // a diagnostic that reads wrong because it was measured narrow is worse
            // than no diagnostic. It is one cast of a buffer that is about to be
            // reduced to a scalar, and the fusion collapses it into the reduce.
            layer_rms.push(
                dev_lane_resid::from_resid(xd.clone())
                    .powf_scalar(2.0)
                    .mean()
                    .reshape([1, 1]),
            );
            layer_kind.push((layer, is_local));
            // Hand the pool's unused pages back between layers, when
            // `CleanupPolicy` says this pass is the kind that wants it.
            //
            // Every layer's activations are freed before the next one allocates its
            // own, so between two layers the pool holds almost nothing and RESERVES
            // almost everything: 12.56 GiB reserved against 1.19 live at the end of
            // a layer, on eight layers at 20,000 tokens with `INK_MEM_TRACE=1`. On
            // this part the pool is host memory, so what the node runs out of is
            // the reserved figure, not the live one.
            //
            // What it trades is ALLOCATION: the next layer asks for the same
            // pages again, and a page this runtime hands back is a `cudaFree` it
            // will pay a `cudaMalloc` for. That cost is per layer, so a prefill --
            // few layers, enormous reservation -- pays it once a layer and a decode
            // step would pay it once a layer PER TOKEN for nothing. Which is why
            // the policy asks how many positions the pass computes rather than
            // asking an operator to remember a variable.
            // `memory_usage` was documented here as the pool's own host-side
            // bookkeeping, free to read. It is not, and that sentence is what hid
            // 18% of a decode step. On this cubecl lineage the compute server has
            // its own thread and `memory_usage` is a `submit_blocking`: it queues
            // a closure behind everything this layer just enqueued, wakes the
            // runner, and blocks until the runner has drained down to it -- a HOST
            // launch-queue barrier -- and then walks every page of twenty-four
            // pools and every slice of the persistent one. Which is why it is now
            // behind a closure the gate calls only when the answer needs it.
            //
            // Timed from BEFORE the gate, so the policy's own question -- the
            // `/proc/meminfo` and two cgroup reads -- is inside the number and not
            // beside it. `t_pool_poll` is the barrier alone, so the two halves can
            // be read apart.
            let t_cl = Instant::now();
            let last_layer = layer + 1 == hi;
            let want_cleanup = cleanup_gate.at_layer(last_layer, || {
                pool_polls += 1;
                let t_pl = Instant::now();
                let usage = fp4_client.memory_usage();
                t_pool_poll += t_pl.elapsed().as_secs_f64();
                match usage {
                    Ok(u) => mary::models::inkling::pool::stranded_bytes(
                        u.bytes_reserved,
                        u.bytes_in_use,
                        u.bytes_padding,
                    ),
                    Err(_) => 0,
                }
            });
            if want_cleanup {
                <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("sync before cleanup");
                fp4_client.memory_cleanup();
                cleanups += 1;
            }
            t_cleanup += t_cl.elapsed().as_secs_f64();
        }
        let t_layers = t_ly.elapsed().as_secs_f64();

        // This slot is prefilled; seat it in the batch and let go of it. The next
        // slot starts from an empty `caches`.
        //
        // Seated HERE and not once all b are in, which is what this used to do, and
        // the difference is the largest memory event in the lane. A prefilled
        // `AttnCache` costs far more than the keys it holds -- 3.59 GiB against
        // 0.16 GiB on the 21-layer head at a 3732-token prompt -- and the batch
        // assembly is what collapsed it. Keeping b of them meant the sixth prefill
        // ran a 25.8 GiB headroom out of memory. See `dev_lane::SlotCache::seeded`.
        //
        // The pool is handed back afterwards, which is not tidiness either: a
        // prefill's peak is a [heads, query_block, tokens] score matrix, 1.68 GiB
        // at a 3.7k context, and cubecl keeps the page it was served from because
        // keeping it is the right policy inside a loop. With nothing of this slot
        // left alive there is a whole page to hand back, which is exactly what
        // `memory_cleanup` could not do while the b caches were pinning it.
        // The last gap in the partition. Zero on a decode pass -- the block below
        // is prefill-only -- but it holds a `Backend::sync` and a
        // `memory_cleanup`, so on a slot prefill it is not small, and a bracket
        // that is only correct on the pass you happened to look at is not a
        // partition.
        let t_st = Instant::now();
        if slot_lane && !is_decode {
            // Which row of THIS cohort's batch is being seated. Cohort `coh` was
            // prefilled by steps `coh * nslots .. (coh + 1) * nslots`, so row zero
            // is the one that seeds the batch and the rest seat into it.
            let row = step % nslots;
            for (l, c) in caches.drain(..).enumerate() {
                let kind = t.attn_kind(lo + l);
                let (_, kv_heads, head_dim) = t.heads(kind);
                let asc = c.attn_sconv.reshape([1, t.sconv_kernel_size - 1, h]);
                let msc = c
                    .mlp_sconv
                    .expect("a prefill seeds the MLP convolution")
                    .reshape([1, t.sconv_kernel_size - 1, h]);
                if row == 0 {
                    slots_dev[coh].push(SlotLayerCache {
                        attn: dev_lane::SlotCache::seeded(nslots, c.attn, kv_heads, head_dim),
                        attn_sconv: dev_lane::seat_first3(nslots, asc),
                        mlp_sconv: Some(dev_lane::seat_first3(nslots, msc)),
                    });
                } else {
                    slots_dev[coh][l].attn.seat(row, c.attn);
                    dev_lane::seat_row3(&mut slots_dev[coh][l].attn_sconv, row, asc);
                    let mut m = slots_dev[coh][l]
                        .mlp_sconv
                        .take()
                        .expect("seeded by slot 0");
                    dev_lane::seat_row3(&mut m, row, msc);
                    slots_dev[coh][l].mlp_sconv = Some(m);
                }
            }
            assert_eq!(
                slots_dev[coh].len(),
                hi - lo,
                "the slot batch is missing layers"
            );
            <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("sync after a slot prefill");
            fp4_client.memory_cleanup();
            println!(
                "{}",
                mary::models::inkling::seam::pool_line(
                    &fp4_client,
                    &format!("prefill {}/{total_slots}", step + 1)
                )
            );
        }
        let t_seat = t_st.elapsed().as_secs_f64();

        // ---- the one sync for this node's whole stack --------------------------
        //
        // Everything above is enqueued. THIS is where the device time is, and
        // putting the timer here rather than around each stage is the only honest
        // place for it: with no readbacks inside the loop, a per-stage number would
        // measure how long the host took to describe the work, not how long the
        // work took.
        //
        // The forty-two per-layer RMS reductions come back in one read rather than
        // forty-two, which is the same trick one level up.
        let t_sy = Instant::now();
        // An EXPLICIT sync, not an implicit one. Reading the RMS column would drain
        // the queue anyway -- it depends on every layer's output -- but "would
        // anyway" is how a timer ends up measuring something other than its label.
        // This one costs nothing (the head cannot start before the stack finishes)
        // and it makes `t_stack_sync` the stack's device time and `t_head` the
        // head's, rather than one number smeared over both.
        <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("sync after the stack");
        // What the device pool has RESERVED once the whole stack has run, which is
        // the quantity the admission gate exists to predict. On a unified-memory
        // part the pool is node memory, so everything in it beyond the weight arena
        // IS the activation working set at this sequence length -- and unlike
        // `MemAvailable` it is not polluted by whatever else the box is doing.
        println!(
            "{}",
            mary::models::inkling::seam::pool_line(&fp4_client, "after stack")
        );
        // The admission gate's prediction beside what the run actually reserved, on
        // the same line, every pass. The gate is the only thing standing between a
        // long input and a node in swap, and the way it failed before was not that
        // it was noisy -- it was FLAT, charging 13.5 GiB whether the input was
        // 16,384 tokens or 100,623, and nothing printed by the run said otherwise.
        // Printing the outcome next to the estimate makes every run a measurement
        // of its own gate: a reserved figure above the charge is a run that was
        // admitted and should not have been.
        let reserved = mary::models::inkling::seam::pool_reserved(&fp4_client);
        if reserved > 0 {
            println!(
                "    activations: {:.2} GiB charged at admission, {:.2} GiB reserved by the pool \
             ({:+.0}%)",
                attention_bytes as f64 / GIB,
                reserved as f64 / GIB,
                100.0 * (reserved as f64 / attention_bytes.max(1) as f64 - 1.0),
            );
        }
        let rms_col: Vec<f32> = if layer_rms.is_empty() {
            Vec::new()
        } else {
            down(BT::cat(std::mem::take(&mut layer_rms), 0))
        };
        let t_stack_sync = t_sy.elapsed().as_secs_f64();
        // Everything from the stack sync to the report: the per-layer RMS lines,
        // the residual readback, the head, the argmax, the KV commit, the MTP
        // draft and the peer wait. `t_x_down`, `t_head`, `t_draft` and
        // `t_wait_peer` are NESTED inside this bracket and are printed as parts
        // of it -- never added to `named` a second time.
        let t_tl = Instant::now();
        for (i, &(layer, is_local)) in layer_kind.iter().enumerate() {
            println!(
                "  layer {layer:2} [{}] rms {:.4}",
                if is_local { "local " } else { "global" },
                rms_col[i].sqrt()
            );
        }

        // The residual stream on the HOST, and only for the readers that genuinely
        // need it there: the wire (16 KB to the tail), the MTP draft path (all
        // scalar host arithmetic), and a debug dump. A whole-stack or tail process
        // that is not drafting never materialises it -- the head reads `xd`.
        let want_host_x = is_head || mtp_k > 0 || dump_dir.is_some();
        // Timed, and it is the head's alone: a tail never materialises the stream.
        // That asymmetry is why it was the first suspect when the head's report
        // summed to two thirds of its pass at 48 slots and the tail's summed to
        // 95% -- and it reads 0.3 ms, so it is a suspect that has been RULED OUT
        // rather than the untimed thing anything could be blamed on.
        let t_dn = Instant::now();
        // Widened on the DEVICE before the readback, not on the host after it. The
        // wire is f32 and `down` would happily convert a BF16 `TensorData` for it --
        // in a host loop over `n * 4096` elements, which at 100,000 tokens is four
        // hundred million of them on one core. The cast is a kernel; let it be one.
        let x: Vec<f32> = if want_host_x {
            down(dev_lane_resid::from_resid(xd.clone()))
        } else {
            Vec::new()
        };
        let t_x_down = t_dn.elapsed().as_secs_f64();

        // Does anything downstream need a logit row that is NOT the last one?
        //
        // The argmax reads exactly one row -- the last -- and that is the whole of
        // what a forward produces. Every other row of the head exists for the
        // REPORT: the per-position top-5 table and the `INK_DUMP_DIR` capture. On a
        // 512-token prefill those rows are 512 x 200058 f32 = 410 MB read back
        // across the bus, on top of a 4096-wide GEMM over 512 rows instead of 16.
        // So they are computed when a reader has asked for them and not otherwise.
        // `INK_ALL_LOGITS=1` is that ask; a dump implies it.
        // The same predicate admission priced, above; see `logits_bytes`.
        let all_logits = dump_dir.is_some() || all_logits;

        // ---- head, or the wire in its place ------------------------------------
        let v = t.effective_vocab();
        let t_h = Instant::now();
        // A head has no logits and never will: the rest of the stack and the
        // unembedding both live on the other machine. So it hands the stream over
        // and takes the argmax back, and that blocking call is charged to the same
        // slot the head/unembed occupies on a whole-stack run — which is what makes
        // the two reports read against each other line for line.
        let mut best_wire = None;
        // What the approximate head did this pass, if it ran. `None` is the
        // exact lane, and the report says so rather than printing zeros.
        let mut ann_stat: Option<mary::models::inkling::annhead::AnnStat> = None;
        // Which position `logits[0]` is. The head computes `logit_row0..n`, so this
        // is 0 when everything was asked for and `n - 1` when only the argmax's row
        // was. A head computes nothing and the value is unread there.
        // How many of this pass's rows the verifier has to read an argmax off. One,
        // normally -- a forward produces one token. A speculative pass produces one
        // PER ROW, and every one of them is needed: the accepted prefix is the
        // leading run where the draft and the argmax agree, so a rule that only
        // looked at the last row could not find where the agreement stopped.
        let verify_rows = if (spec_k > 0 || pass_tree.is_some()) && kv && step > 0 {
            n
        } else {
            1
        };
        // The width probe's rows, derived from the batch the head actually sent
        // rather than from this process's environment -- so INK_WIDTH is set on the
        // head alone and the two ends cannot disagree about it.
        //
        // Every row is unembedded, and that is deliberate: b independent sequences
        // each need their own logits, so a probe that unembedded one row would
        // leave the widest matmul in the stack out of the price.
        let probe_rows = if spec_k == 0 && tree_b == 0 && kv && is_decode && n > 1 && !slot_lane {
            n
        } else {
            1
        };
        // The rows a slot batch reads an argmax off: all of them, because each one
        // is a different sequence and each one's next token is a fact about it.
        // This is the widest matmul in the stack run at its real width, which is
        // the half of batched decode the width probe was already honest about.
        let slot_rows = if slot_lane && is_decode { n } else { 1 };
        let logit_row0 = if all_logits || probe_rows > 1 || slot_rows > 1 {
            0
        } else {
            n - verify_rows
        };
        let (mut t_send, mut t_wait_peer) = (0f64, 0f64);
        // THIS pass's drafting. `acc_draft` is the run's, and a per-pass report
        // cannot be a partition using a run-scoped total.
        let mut t_draft = 0f64;
        let mut wire_toks: Vec<usize> = Vec::new();
        // Which cohort the answer read on THIS pass belongs to. The pass's own
        // cohort everywhere except an interleaved head, where it is the cohort sent
        // a pass earlier -- so the tokens land in that cohort's streams and not in
        // the one this pass just computed.
        let mut answer_coh = coh;
        // False on exactly the passes an interleaved head starts a cohort it has
        // not yet heard back about: the first `ncohorts - 1` decode passes. Nothing
        // was confirmed, so nothing is committed.
        let mut answered = true;
        let logits = if let Some(Pipe::Head(s)) = pipe.as_mut() {
            let t_s = Instant::now();
            send_stream(s, n, pos0, coh, &x)?;
            t_send = t_s.elapsed().as_secs_f64();
            in_flight.push_back(coh);
            // THE interleave, and it is this line. `want` is how many answers the
            // head is content to leave outstanding once this pass is over: zero
            // with one cohort, which is the strict send-then-block loop this file
            // has always run, and `ncohorts - 1` while decoding with more, which
            // leaves the tail exactly one round of work to do while the head starts
            // the next cohort. The prefills stay strict whatever `ncohorts` is --
            // a prefill's answer seeds the slot it just filled, and there is
            // nothing to overlap it with.
            let want = if is_decode { ncohorts - 1 } else { 0 };
            let t_w = Instant::now();
            if in_flight.len() > want {
                // The tail's FIRST message: the tokens its verify pass confirmed.
                // Never empty -- the row fed the last confirmed token always
                // produces one -- and longer than one exactly when drafts were
                // accepted. The drafts for the NEXT pass are a second message, read
                // further down, so this process gets to commit its caches in
                // between.
                answer_coh = in_flight.pop_front().expect("just pushed one");
                wire_toks = recv_toks(s)?;
                anyhow::ensure!(!wire_toks.is_empty(), "the tail confirmed no token at all");
                best_wire = Some(*wire_toks.last().expect("checked non-empty"));
            } else {
                answered = false;
            }
            t_wait_peer = t_w.elapsed().as_secs_f64();
            Vec::new()
        } else {
            // 109 x 4096 x 200058 is 89 G multiply-adds — the single largest
            // matmul in the forward, and the one left standing once attention and
            // the MLPs move. There is no host twin: 89 G scalar multiply-adds is
            // not a reference, it is an afternoon. The muP divisor divides BEFORE
            // the projection, matching the reference: doing it after is
            // algebraically equal and numerically not.
            //
            // The rows: `logit_row0..n`, which is the last row alone unless a
            // reporter asked for all of them. The unembedding is the widest matmul
            // in the stack (n x 4096 x 200058) and the only consumer of all but its
            // final row is a print, so the slice happens on the INPUT -- before the
            // GEMM and before the readback -- rather than after both.
            let hx = if all_logits {
                xd.clone()
            } else {
                xd.clone().slice([logit_row0..n, 0..h])
            };
            let hs = dev_lane_resid::rms_norm(
                hx,
                fnorm_dev.clone().expect("the tail owns the final norm"),
                t.rms_norm_eps,
            )
            .div_scalar(t.logits_mup_width_multiplier as f32);
            // Query-space noise: the sampling temperature, and -- when the
            // approximate head is on -- the thing that stops any row being
            // PERMANENTLY invisible to the shortlist. Applied to the normed,
            // muP-divided hidden state, so the induced logit noise is
            // `sigma * ||w_i||` in the same units the argmax below compares.
            //
            // At `INK_TEMP=0` nothing is built, uploaded or added: greedy decode
            // is the same arithmetic it was before this existed.
            let hs = if head_temp() > 0.0 {
                let rows = n - logit_row0;
                // A REFUSAL and not a fallback. `unwrap_or(1.0)` here would make
                // `INK_TEMP=0.8` a different temperature on a table whose rows
                // average 5.6 -- seven times colder -- and nothing in the run
                // would say so. A knob that silently means something else is
                // worse than a knob that is not there.
                let sk = head_sketch.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "INK_TEMP needs the mean embedding-row norm to turn a temperature \
                         in logit units into a hidden-state sigma, and only the sign \
                         sketch computes it. This build bound the head to {:?}, which \
                         builds no sketch.",
                        head_lane()
                    )
                })?;
                // A temperature in logit units becomes a hidden-state sigma by
                // dividing by the row norm the logit noise gets multiplied by.
                // `pi/sqrt(6)` is Gumbel's standard deviation at unit
                // temperature, so the two mechanisms agree on the second moment
                // -- which is the most that can be claimed, since one is skewed
                // and the other is not.
                let sigma = head_temp() * std::f32::consts::PI / 6f32.sqrt() / sk.mean_norm;
                let e = normals(head_temp_seed(), step as u64, rows * h);
                let noise = burn::tensor::Tensor::<Bk, 2>::from_data(
                    burn::tensor::TensorData::new(
                        e.iter().map(|v| v * sigma).collect::<Vec<f32>>(),
                        [rows, h],
                    ),
                    &dev,
                );
                let dt = mary::models::inkling::seam::dtype_of(&hs);
                let noise = if dt == burn::tensor::DType::BF16 {
                    noise.cast(burn::tensor::FloatDType::BF16)
                } else {
                    noise
                };
                hs + noise
            } else {
                hs
            };
            let uw = unembed_w
                .as_ref()
                .expect("the tail binds the unembed table");
            // The approximate lane, when it is on and the pass is one row wide.
            //
            // The `m > 1` fallback is not a gap. The exact GEMM amortises ONE
            // read of the table over all `m` rows, so a verify pass already gets
            // most of what narrower codes would buy it; the decode step is
            // `m = 1` and is where the 4.6 ms lives.
            match (head_sketch.as_ref(), uw) {
                (Some(sk), dev_lane::ProjW::W4a16(p))
                    if ann_budget() > 0 && n - logit_row0 == 1 =>
                {
                    let exact = if ann_verify() {
                        Some(down(dev_lane::linear_w(hs.clone(), uw).slice([0..1, 0..v])))
                    } else {
                        None
                    };
                    let (lg, st) = dev_lane::linear_ann(hs, p, sk, ann_budget(), ann_range());
                    ann_stat = Some(st);
                    let approx = down(lg.slice([0..1, 0..v]));
                    match exact {
                        // Verify mode FOLLOWS THE EXACT HEAD. The approximate
                        // row is scored and then discarded, and the sequence
                        // continues from the token the exact lane picked.
                        //
                        // Without this the instrument cannot compare anything.
                        // One disagreement rewrites every later hidden state, so
                        // two arms of an ablation walk into different
                        // continuations and their recall rates are computed over
                        // DIFFERENT queries -- measured: at layers 0:2 the
                        // rotated arm looped on one token and scored 0.0370
                        // while the raw arm found a 13-token loop and scored
                        // 0.2346, and neither number was about the other's
                        // queries. Pinning the trajectory to the exact head
                        // makes every arm of every ablation see the same states.
                        //
                        // What this run therefore does NOT measure is error
                        // COMPOUNDING over a generation. That is what a plain
                        // run without `INK_ANN_VERIFY` shows, and the two
                        // questions want different instruments.
                        Some(e) => {
                            mary::models::inkling::annhead::verify_row(&e, &approx, st.floor);
                            e
                        }
                        None => approx,
                    }
                }
                _ => down(dev_lane::linear_w(hs, uw).slice([0..n - logit_row0, 0..v])),
            }
        };
        let t_head = t_h.elapsed().as_secs_f64();

        // Greedy: the last position's argmax is the next token. A head took it off
        // the wire instead of computing it, and either way it is decided HERE --
        // before the reporting -- so a tail can answer its peer immediately rather
        // than making the head wait on a page of printing.
        // How many DRAFTS this pass kept, and the tokens it confirmed. Acceptance
        // is exact argmax match and deliberately not a stochastic rule: measured on
        // this model the exact rule accepts MORE (49.5% against 45.6% sampled and
        // 40.6% under 1-TV), because when the draft is the argmax the target agrees
        // strongly and when it is not the target puts little mass there either.
        let new_toks: Vec<usize>;
        // Which VERIFY ROWS this pass's accept walk kept, when the pass carried a
        // tree. The root followed by the accepted path: ascending, but NOT
        // contiguous, which is the whole difference from a linear rollback. Empty
        // on every other kind of pass, and that emptiness is what selects the
        // truncating rollback below.
        let mut tree_kept: Vec<usize> = Vec::new();
        // The host argmax, which is a scalar loop over a 200058-wide f32 row and
        // is run once per confirmed row. It is the largest piece of pure host
        // arithmetic left in a decode pass and nothing has ever timed it, so it
        // has been inside the unexplained bucket the whole time.
        let t_am = std::cell::Cell::new(0f64);
        let best = if !answered {
            // An interleaved head, on the pass that opened a cohort. The answer is
            // still on the tail; it is read one pass later and committed there.
            new_toks = Vec::new();
            0
        } else if let Some(b) = best_wire {
            new_toks = wire_toks.clone();
            b
        } else {
            let mut accepted = 0usize;
            let rows = n - logit_row0;
            let argmax_of = |i: usize| -> usize {
                let t_a0 = Instant::now();
                let row = &logits[i * v..(i + 1) * v];
                let mut b = 0usize;
                for (j, &val) in row.iter().enumerate() {
                    if val > row[b] {
                        b = j;
                    }
                }
                t_am.set(t_am.get() + t_a0.elapsed().as_secs_f64());
                b
            };
            if let (true, Some(tr)) = (verify_rows > 1, spec_tree.as_ref()) {
                debug_assert_eq!(logit_row0, 0, "a verify pass reads from row 0");
                anyhow::ensure!(
                    n == tr.len(),
                    "the pass fed {n} rows against a {}-node tree",
                    tr.len()
                );
                // Row `i` was fed node `i`'s token and scored in a context that
                // is exactly node `i`'s ancestry -- which is what the mask, the
                // positions and the four gathered convolutions were all for. So
                // walk from the root: whatever the target predicted at the
                // current node is a confirmed token, and if some CHILD holds
                // that token then that child's own row was scored in a context
                // the model has now committed to and its prediction is a fact
                // too. Reduces exactly to the linear rule on a chain.
                let preds: Vec<usize> = (0..rows).map(argmax_of).collect();
                let acc = spectree::accept_tree(tr, &feed, &preds);
                tree_kept = acc.kept_rows;
                new_toks = acc.new_toks;
            } else if verify_rows > 1 {
                debug_assert_eq!(logit_row0, 0, "a verify pass reads from row 0");
                anyhow::ensure!(
                    n == 1 + last_drafts.len(),
                    "the head fed {n} rows against {} drafts -- the two ends disagree on the \
                 speculation width",
                    last_drafts.len()
                );
                let preds: Vec<usize> = (0..rows).map(argmax_of).collect();
                // Row i was fed the token at pos0+i and predicts pos0+i+1. Row 0
                // was fed a CONFIRMED token, so its prediction is always kept;
                // row i>0 was fed draft i-1, so its prediction is only a fact
                // about the sequence if every draft before it was right.
                while accepted < last_drafts.len() && last_drafts[accepted] == preds[accepted] {
                    accepted += 1;
                }
                new_toks = preds[..=accepted].to_vec();
            } else if slot_rows > 1 {
                // One argmax per SLOT. Nothing is accepted or rejected here --
                // a slot batch has no drafts, every row was fed a token its own
                // sequence confirmed, and every row's prediction is kept.
                new_toks = (0..rows).map(argmax_of).collect();
            } else if probe_rows > 1 {
                // Row 0 is the sequence; rows 1.. are the probe's filler and
                // their argmaxes are about nothing. Reading row 0 and not the
                // last row is what keeps the text identical to INK_WIDTH=1.
                new_toks = vec![argmax_of(0)];
            } else {
                new_toks = vec![argmax_of(rows - 1)];
            }
            *new_toks
                .last()
                .expect("at least one row is always confirmed")
        };
        let t_argmax = t_am.get();
        // `ids`, the MTP scoring and the per-position report were all written about
        // ONE sequence, and slot 0 is the one they follow. `best` is therefore slot
        // 0's token and not the last row's, which is what it means everywhere else.
        let best = if slot_lane && is_decode && answered {
            new_toks[0]
        } else {
            best
        };
        let mut t_to_reply = 0f64;
        if let Some(Pipe::Tail(s)) = pipe.as_mut() {
            send_toks(s, &new_toks)?;
            t_to_reply = pass.elapsed().as_secs_f64();
        }
        // Everything but the LAST confirmed token goes into `ids` now; the last one
        // is `best` and is pushed where it has always been pushed, so the MTP block
        // below sees exactly the sequence-and-a-held-back-argmax it was written
        // against.
        // The tree lane joins the tail here rather than at the report below, and
        // the ORDER is the reason: the MTP block reads `ids` and requires that
        // it hold every confirmed token EXCEPT the last, which is `best`. Left
        // to the report, a pass that accepted a draft would hand the drafter a
        // sequence short by exactly the tokens it just confirmed, and the
        // symptom would be an acceptance rate rather than an error.
        if (is_tail || tree_b > 0) && gen_steps > 0 && !repeat && new_toks.len() > 1 && !slot_lane {
            ids.extend_from_slice(&new_toks[..new_toks.len() - 1]);
        }
        // Each slot's own stream. A prefill pass produced the first generated token
        // of the slot it prefilled; a decode pass produces one for every slot. Both
        // ends run this -- the head off the wire, the tail off its own argmax --
        // for the same reason `ids` is recomputed rather than sent.
        if slot_lane && gen_steps > 0 && !repeat {
            if is_decode {
                if answered {
                    let base = answer_coh * nslots;
                    for (q, tok) in slot_ids[base..base + nslots]
                        .iter_mut()
                        .zip(new_toks.iter())
                    {
                        q.push(*tok);
                    }
                }
            } else {
                slot_ids[step].push(best);
            }
        }
        // ---- both ends roll back to the accepted prefix ------------------------
        //
        // `keep` is 1 + accepted: row 0 fed a confirmed token and rows 1..=accepted
        // fed drafts the verifier kept, so their K, V and convolution memory are
        // facts about the sequence the model actually chose. The rows past them
        // were computed from tokens it did not choose, and leaving them behind does
        // not error -- it shows up later as an acceptance rate that drifts down.
        if verify_rows > 1 || probe_rows > 1 {
            // One row for the probe -- its filler rows are not facts about any
            // sequence and their K, V and convolution memory go away with them.
            let keep = if verify_rows > 1 { new_toks.len() } else { 1 };
            let hist = t.sconv_kernel_size - 1;
            for (slot, c) in caches.iter_mut().enumerate() {
                let window = if t.attn_kind(lo + slot) == AttnKind::Local {
                    Some(t.sliding_window_size)
                } else {
                    None
                };
                // A linear pass keeps a PREFIX of the batch and rolls back with
                // a truncation. A tree keeps a PATH through it, whose rows are
                // not contiguous, so there is no slice of the cache that is
                // those rows and the rollback is a GATHER -- of K and V, and of
                // all three convolution windows, by the same indices.
                if tree_kept.is_empty() {
                    c.attn.commit(keep, window);
                    if let Some(all) = c.attn_sconv_pending.take() {
                        c.attn_sconv = dev_lane::conv_history(
                            all.slice([0..hist + keep, 0..h]),
                            t.sconv_kernel_size,
                        );
                    }
                    if let Some(all) = c.mlp_sconv_pending.take() {
                        c.mlp_sconv = Some(dev_lane::conv_history(
                            all.slice([0..hist + keep, 0..h]),
                            t.sconv_kernel_size,
                        ));
                    }
                } else {
                    debug_assert_eq!(tree_kept.len(), keep, "one kept row per confirmed token");
                    c.attn.commit_rows(&tree_kept, window);
                    if let Some(all) = c.attn_sconv_pending.take() {
                        c.attn_sconv =
                            dev_lane::conv_history_rows(all, t.sconv_kernel_size, &tree_kept);
                    }
                    if let Some(all) = c.mlp_sconv_pending.take() {
                        c.mlp_sconv = Some(dev_lane::conv_history_rows(
                            all,
                            t.sconv_kernel_size,
                            &tree_kept,
                        ));
                    }
                }
            }
        }
        // The tail's SECOND message, and the head's second wait: the drafts to feed
        // next pass. Read after the commit on purpose -- that is device work this
        // process can enqueue while the other machine is still drafting.
        if let Some(Pipe::Head(s)) = pipe.as_mut() {
            if spec_k > 0 {
                let t_w = Instant::now();
                drafts_in = recv_toks(s)?;
                t_wait_peer += t_w.elapsed().as_secs_f64();
                anyhow::ensure!(
                    drafts_in.len() == spec_k,
                    "the tail sent {} drafts against INK_SPEC={spec_k}",
                    drafts_in.len()
                );
            }
        }
        if is_decode && !slot_lane {
            let bucket = new_toks.len().min(spec_hist.len() - 1);
            spec_hist[bucket] += 1;
        }
        if is_decode && tree_b > 0 && verify_rows > 1 {
            let tr = spec_tree.as_ref().expect("the tree lane has a topology");
            match tree_kept.get(1) {
                Some(&node) => tree_rank_hist[tr.node(node).rank] += 1,
                None => tree_rank_hist[tree_b] += 1,
            }
        }

        // ---- MTP: score the drafts that named this step, then draft afresh -----
        if mtp_k > 0 {
            // SCORE FIRST, against `best` -- the token the full stack just produced.
            // This is the whole experiment: a draft is right or it is not, and the
            // rate over many steps is the only oracle the composition has.
            // The target's OWN distribution at this position, once per pass rather
            // than once per pending draft. `logits` holds the argmax's row and only
            // that row unless a reporter asked for more, which is exactly the row
            // every rule below reads.
            // WHICH ROW OF `logits` the step's answer came off, and it is not
            // always the last one. A verify pass computes a row per DRAFT it was
            // fed and keeps the leading run that agreed; `best` is the argmax of
            // row `new_toks.len() - 1`, and every row past that is a function of
            // tokens the model did not choose.
            //
            // This used to read `n - 1 - logit_row0` -- the LAST row -- and the
            // comment under `draft_cand` asserted the invariant it breaks:
            // "`best` is always in it (it is the top-1)". Under INK_SPEC=1 that
            // is false on every pass that accepts nothing, which is 31.9% of
            // them on the document corpus, and `INK_DRAFT_TOPK` now DEFAULTS to
            // 512 -- so the default speculative configuration was gathering its
            // draft candidates from the distribution of a rejected row, and
            // could gather one that does not even contain the token just
            // confirmed. It cannot produce a wrong token (the verifier is exact
            // argmax) and it cannot raise a flag. It can only lower acceptance,
            // silently, which is this file's recurring failure mode.
            //
            // `logits` starts at row `logit_row0`, so the index is relative.
            // Written as the three cases rather than as one expression, because
            // the probe lane's answer is row 0 and an arithmetic identity that
            // happens to be right for two of the three is how this went wrong the
            // first time.
            let answer_abs = if verify_rows > 1 {
                new_toks.len() - 1
            } else if probe_rows > 1 {
                0
            } else {
                n - 1
            };
            let answer_row = answer_abs - logit_row0;
            let p_t: Vec<f32> = if mtp_prob && !logits.is_empty() {
                softmax_row(&logits[answer_row * v..(answer_row + 1) * v])
            } else {
                Vec::new()
            };
            mtp_pending.retain(|&(target, depth, tok)| {
                // Off when the loop is speculating: `step` no longer names one
                // token, so a draft issued at step s cannot be matched to "the
                // token step s+d+1 produced". The accept-and-skip loop verifies its
                // own drafts against the rows they were fed into and reports the
                // prefix histogram instead, which is the same measurement taken
                // where it is now decidable.
                if spec_k > 0 || target != step {
                    return true;
                }
                mtp_seen[depth] += 1;
                if mtp_prob {
                    if let Some(pd) = mtp_pd.remove(&(target, depth)) {
                        // B: the target's own probability on the token that was
                        // drafted. C: the overlap of the two distributions.
                        let b = p_t.get(tok).copied().unwrap_or(0.0) as f64;
                        let c: f64 = pd
                            .iter()
                            .zip(p_t.iter())
                            .map(|(a, t)| a.min(*t) as f64)
                            .sum();
                        mtp_b_sum[depth] += b;
                        mtp_c_sum[depth] += c;
                        mtp_prob_n[depth] += 1;
                        // `tok` IS the draft's argmax, so this is the head's own
                        // top-1 mass -- its confidence, not the target's.
                        mtp_conf[depth].push((pd.get(tok).copied().unwrap_or(0.0), tok == best));
                        if let Some(slots) = mtp_issued_q.get_mut(&(target - depth - 1)) {
                            slots[depth] = Some((b, c));
                        }
                    }
                }
                if tok == best {
                    mtp_hits[depth] += 1;
                }
                // The step this draft was ISSUED at: head d drafted the token d+1
                // steps ahead, so the issuer is target - depth - 1.
                if let Some(slots) = mtp_issued.get_mut(&(target - depth - 1)) {
                    slots[depth] = Some(tok == best);
                }
                println!(
                    "  MTP depth {}: drafted {tok}, actual {best} -- {}",
                    depth + 1,
                    if tok == best { "HIT" } else { "miss" }
                );
                false
            });

            // DRAFT. Head d is fed the previous stage's hidden states and the
            // embeddings of the tokens shifted one further along, so head 0 sees
            // the token the stack just chose and head d sees draft d-1.
            let e_w = embed_w
                .as_ref()
                .expect("drafting needs the embedding table");
            // The BACKBONE embed_norm, applied to a drafted token's embedding
            // BEFORE the depth head's own near-identity `embed_norm`. Gated on
            // `use_embed_norm` exactly as vLLM gates its `backbone_embed_norm`;
            // `None` reproduces the old raw-embedding behaviour exactly, which
            // is what `INK_MTP_BACKBONE_NORM=0` is for.
            let e_bn: Option<&[f32]> = if t.use_embed_norm && backbone_embed_norm() {
                Some(
                    embed_n
                        .as_ref()
                        .expect("a drafting tail loads the backbone embed_norm")
                        .data
                        .as_slice(),
                )
            } else {
                None
            };
            let fnorm_d = fnorm.as_ref().expect("drafting needs the final norm");
            // Which hidden state head 0 is fed. `x` is the stack's RAW output, which
            // `head` norms on its way to logits. Feeding the FINAL-NORMED one
            // measured twice as well (25% -> 50% on a matched 20-token run), so it
            // is the default; `INK_MTP_RAW=1` is the control that established it.
            //
            // Why it is not merely a scale fix: RMS norm is scale-invariant, so the
            // two differ ONLY by the final norm's learned weight vector applied in
            // between. That the weights help says the heads were trained on
            // post-final-norm hidden states.
            //
            // It also makes `chain_hidden_post_norm: false` coherent, which the raw
            // reading did not. That flag governs the CHAIN — whether a head's output
            // is normed before the next head sees it — and it is off, so stages 1..k
            // stay raw. The ENTRY from the main stack is a separate question and the
            // flag never spoke to it.
            let entry = if std::env::var("INK_MTP_RAW")
                .map(|val| val == "1")
                .unwrap_or(false)
            {
                x.clone()
            } else {
                rms_norm(&x, &fnorm_d.data, t.rms_norm_eps, n, h)
            };
            // ...for the rows the verifier KEPT, and no further. A speculative pass
            // computes a hidden state for every row it fed, and the ones past the
            // accepted prefix are functions of tokens the model did not choose. An
            // MTP head drafting from one of those would be drafting off a state
            // that never happened, and nothing downstream would say so.
            let entry = if !tree_kept.is_empty() {
                // ...and for a TREE the kept rows are a PATH, not a prefix. The
                // slice below is right for a linear speculation, where the
                // accepted rows are rows 0..m by construction, and silently
                // wrong here: accepting the SECOND candidate keeps rows 0 and 2,
                // and taking rows 0 and 1 instead would feed head 0 the hidden
                // state of a branch the model rejected. It would draft off a
                // state that never happened and nothing downstream would say so.
                let mut kept = Vec::with_capacity(tree_kept.len() * h);
                for &r in &tree_kept {
                    kept.extend_from_slice(&entry[r * h..(r + 1) * h]);
                }
                kept
            } else if verify_rows > 1 && new_toks.len() < n {
                entry[..new_toks.len() * h].to_vec()
            } else {
                entry
            };
            // RETAIN it, and that is the whole enabling change. An MTP head's input
            // at position j is (main_hidden[j], embed(token[j+1])), and
            // main_hidden[j] never changes once produced -- attention is causal, so
            // nothing later reaches back and alters it. What stopped the cached lane
            // from drafting was never that the values move; it was that the loop
            // DISCARDED them. 16 KB a token at f32, against a 144 GiB working set.
            //
            // Uncached, every pass recomputes the whole prefix, so this is an
            // assignment there and an append here, and both lanes end holding the
            // same table over the same sequence.
            // `kv` as well as the switch: the device draft lane only runs cached,
            // and uncached this would upload the whole recomputed prefix once a
            // pass for a reader that does not exist.
            if mtp_dev_on && kv {
                // Uploaded BEFORE the host table takes ownership, and appended on
                // the same rule: an append with a cache, a replacement without one,
                // so both tables end holding the same rows over the same sequence.
                let rows = entry.len() / h;
                let e_dev = up2::<Bk>(entry.clone(), rows, h, &dev);
                mtp_main_dev = Some(match (kv, mtp_main_dev.take()) {
                    (true, Some(prev)) => BT::cat(vec![prev, e_dev], 0),
                    _ => e_dev,
                });
            }
            if kv {
                mtp_main.extend_from_slice(&entry);
            } else {
                mtp_main = entry;
            }
            let seq = ids.len();
            debug_assert_eq!(mtp_main.len(), seq * h, "one retained hidden row per token");

            // One row of logits, argmaxed. The device head returns the FULL vocab
            // width, so index by what came back rather than by whichever constant
            // happens to match, or the argmax silently reads the wrong row.
            //
            // ONE row and not `n` of them: a draft is read off the last position and
            // the head is per-row, so unembedding the prefix was 89 G multiply-adds
            // per position thrown away.
            // Every draft head's distribution, in the order the heads are read, for
            // the pass that is drafting now. A side channel rather than a return
            // value because `draft_argmax` is called from two lanes and one of them
            // (`draft_whole`) is also the INK_MTP_CHECK control, which must not
            // contribute.
            let draft_probs: std::cell::RefCell<Vec<Vec<f32>>> =
                std::cell::RefCell::new(Vec::new());

            // The rows of the unembedding this step's drafts are allowed to pick
            // from, gathered ONCE and shared by every depth, or `None` for the
            // whole table.
            //
            // The candidate set is the main model's own top-`INK_DRAFT_TOPK` at
            // this position -- the distribution the drafts are trying to
            // anticipate, read off the row the argmax above was taken from. It is
            // a defensible source and not a cheap one to fake: it costs a partial
            // selection over a vector the step already had on the host.
            //
            // `best` is always in it (it is the top-1), so the set is never empty
            // of the token the step just confirmed.
            let draft_cand: Option<(Vec<usize>, mary::models::inkling::bf16gemm::Bf16W)> =
                if draft_topk > 0 && !logits.is_empty() {
                    use mary::models::inkling::bf16gemm::NTILE;
                    let row = &logits[answer_row * v..(answer_row + 1) * v];
                    // Rounded DOWN to the MMA's n-tile: the gemm tiles its output
                    // by `NTILE` and a remainder is not a shape it has. Down rather
                    // than up so the result is a width the vocabulary can supply.
                    let want = (draft_topk.min(v) / NTILE).max(1) * NTILE;
                    // Partial selection, not a sort: the ORDER of the candidates is
                    // never read, only their membership and their row index.
                    let mut idx: Vec<u32> = (0..v as u32).collect();
                    idx.select_nth_unstable_by(want - 1, |&a, &b| {
                        row[b as usize]
                            .partial_cmp(&row[a as usize])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    idx.truncate(want);
                    // The invariant the comment above states, CHECKED. `best` is
                    // the argmax of the row this set was selected from, so it is
                    // in the top-`want` by construction -- and that is exactly
                    // what stopped being true when the row was the wrong one.
                    // 512 comparisons a step against a 3.7 GiB weight stream:
                    // the reason this was not caught is that nothing looked, not
                    // that looking was expensive.
                    anyhow::ensure!(
                        idx.contains(&(best as u32)),
                        "the draft candidate set was gathered from a row whose argmax is not \
                         `best` ({best}): the pruned head cannot draft the token the step just \
                         confirmed, which shows up only as acceptance"
                    );
                    // Once, on the first step that prunes. The unembed BIND prints
                    // what it bound and how, and a table that silently becomes 512
                    // rows wide for half the matmuls in the process is a bigger
                    // change than that -- a run whose log does not say it happened
                    // cannot be told apart from one where the flag was ignored.
                    if !drafted_pruned {
                        drafted_pruned = true;
                        println!(
                            "  draft unembed PRUNED to {want} of {v} rows ({:.2} MiB gathered per \
                             step against {:.2} GiB streamed per depth); INK_DRAFT_TOPK={draft_topk} \
                             -- DRAFTS AND ACCEPTANCE CHANGE, this is not the unpruned model",
                            (want * h * 2) as f64 / (1024.0 * 1024.0),
                            (t.vocab_size * h * 2) as f64 / GIB,
                        );
                    }
                    let ub = unembed_bytes
                        .as_ref()
                        .expect("drafting needs the unembed table");
                    let mut buf = Vec::with_capacity(want * h * 2);
                    for &tok in &idx {
                        let o = tok as usize * h * 2;
                        buf.extend_from_slice(&ub[o..o + h * 2]);
                    }
                    Some((
                        idx.iter().map(|&i| i as usize).collect(),
                        mary::models::inkling::bf16gemm::Bf16W {
                            h: fp4_client.create_from_slice(&buf),
                            n: want,
                            k: h,
                            // A buffer this process just created, not a view into
                            // the pile's 4-aligned arena -- the same thing
                            // `INK_ALIGN_COPY` copies for.
                            align: 16,
                        },
                    ))
                } else {
                    None
                };

            // One draft head's unembedding, from a row that is already on the
            // device. Both draft lanes end here, so the two wastes below were paid
            // once per draft DEPTH on every pass that drafted.
            //
            // The readback: `down` used to pull the whole 201024-wide logits row
            // (804 KB) to the host so a `for` loop could find one index in it.
            // `argmax_row_dev` reduces where the data is and returns the index.
            //
            // The readback's only real consumer: `INK_MTP_PROB`, which is off by
            // default. Nothing else ever looked at the row, so nothing else has to
            // pay for it.
            //
            // Factored out of [`draft_pick`] because the tree lane needs the
            // same row's TOP-B and the unembedding is the widest matmul in the
            // stack: 4096 x 200058. Running it twice a step to get an argmax and
            // then the b candidates that already contain it would be the most
            // expensive way imaginable to learn something the first call knew.
            let draft_width = match draft_cand.as_ref() {
                Some((_, w)) => w.n,
                None => v,
            };
            let draft_logits = |row: T2| -> T2 {
                let hs = if mtp_out_norm() {
                    dev_lane::rms_norm(
                        row,
                        fnorm_dev.clone().expect("drafting needs the final norm"),
                        t.rms_norm_eps,
                    )
                } else {
                    row
                }
                .div_scalar(t.logits_mup_width_multiplier as f32);
                // The pruned table is a gathered BF16 slab; the full one takes
                // the head lane, which is W4A16.
                match draft_cand.as_ref() {
                    Some((_, w)) => dev_lane::linear_bf16(hs, w),
                    None => dev_lane::linear_w(
                        hs,
                        unembed_w
                            .as_ref()
                            .expect("drafting needs the unembed table"),
                    ),
                }
                .slice([0..1, 0..draft_width])
            };
            // The `b` best continuations of ONE row, which for a depth-1 tree is
            // the whole draft. Deterministic top-b and not sampling: `b` draws
            // from a draft distribution draw duplicates and can miss a
            // high-probability candidate, while the row already IS the ensemble
            // of alternatives. Temperature belongs in the target's sampling.
            //
            // The readback is one `draft_width` f32 row -- 800 KB, about 8 us on
            // this part -- against a step that is tens of milliseconds. It is the
            // same readback `INK_MTP_PROB` already makes.
            let draft_topb = |row: T2, b: usize| -> Vec<usize> {
                let dl = down(draft_logits(row));
                spectree::top_b(&dl[..draft_width], b)
                    .into_iter()
                    .map(|c| match draft_cand.as_ref() {
                        // A pruned table's outputs are candidates, not tokens.
                        Some((ids, _)) => ids[c.token],
                        None => c.token,
                    })
                    .collect()
            };
            let draft_pick = |row: T2| -> usize {
                let width = draft_width;
                let lg = draft_logits(row);
                let b = if mtp_prob {
                    // The one caller that wants the row itself. Startup refuses
                    // this together with a pruned table, so `width` is `v` here.
                    let dl = down(lg);
                    let mut b = 0usize;
                    for (i, &val) in dl.iter().take(width).enumerate() {
                        if val > dl[b] {
                            b = i;
                        }
                    }
                    draft_probs.borrow_mut().push(softmax_row(&dl[..width]));
                    b
                } else {
                    argmax_row_dev(lg)
                };
                // A pruned table's outputs are candidates, not tokens.
                match draft_cand.as_ref() {
                    Some((ids, _)) => ids[b],
                    None => b,
                }
            };

            // The host draft lane's entry: a `&[f32]` row, uploaded and handed to
            // the same unembedding.
            let draft_argmax = |row: &[f32]| -> usize {
                debug_assert_eq!(row.len(), h, "the draft head unembeds exactly one position");
                draft_pick(up2::<Bk>(row.to_vec(), 1, h, &dev))
            };

            // The whole-sequence draft: every head over every position. This is what
            // the uncached lane runs, and what `INK_MTP_CHECK` gates the cached lane
            // against. `hidden` is the ENTRY state for the WHOLE sequence, which is
            // precisely the thing the cache exists so as not to need.
            let draft_whole =
                |hidden: &[f32], seq: usize, ids: &[usize], best: usize| -> Vec<usize> {
                    let mut stage = hidden.to_vec();
                    // The token at each position, one step ahead: position j predicts
                    // ids[j+1], and the LAST position predicts `best`.
                    let mut ahead: Vec<usize> = ids[1..].to_vec();
                    ahead.push(best);
                    let mut out = Vec::with_capacity(mtp_heads.len());
                    for headw in mtp_heads.iter() {
                        debug_assert_eq!(ahead.len(), seq, "one shifted token per position");
                        let mut embeds = vec![0f32; seq * h];
                        for (j, &tok) in ahead.iter().enumerate() {
                            embeds[j * h..(j + 1) * h].copy_from_slice(&mtp_embed_row(
                                e_w,
                                e_bn,
                                tok,
                                t.rms_norm_eps,
                                t.vocab_size,
                                h,
                            ));
                        }
                        stage = mtp_block(
                            &stage,
                            &embeds,
                            // DENSE intermediate, not the routed experts' -- every MTP
                            // block is dense regardless of dense_mlp_idx, and the two
                            // sizes differ by 8x (16384 against 2048).
                            &headw.borrow(t.dense_intermediate_size),
                            &headw.dims,
                            Some(ls),
                            headw.window(t.sliding_window_size),
                            seq,
                            mtp_order,
                        );
                        let b = draft_argmax(&stage[(seq - 1) * h..seq * h]);
                        out.push(b);
                        ahead.remove(0);
                        ahead.push(b);
                    }
                    out
                };

            // The same unembedding, fed a row that is already on the device. The
            // host twin above exists for the host draft lane and reads a `&[f32]`;
            // this one would otherwise pay a 16 KB readback and a 16 KB upload per
            // draft for the privilege of handing the value straight back.
            let draft_argmax_dev = |row: T2| -> usize { draft_pick(row) };

            // `INK_MTP_TEACH=1`: the TEACHER-FORCED per-depth acceptance, taken
            // over the whole PROMPT in the drafting prefill rather than over a
            // generated continuation.
            //
            // Why this and not the generated measurement: head `d`'s row `j` is
            // fed `(stage[d-1][j], embed(ids[j + d + 1]))` and predicts the token
            // at `j + d + 2`, which for every row but the last `d + 2` is a token
            // the prompt ALREADY CONTAINS. So one prefill scores ~3700 positions
            // at once, where a 40-step generation scores 40 -- and it scores them
            // against real text rather than against the model's own continuation,
            // which is the confound that makes the generated rate corpus-shaped
            // (this file has quoted 22.0%, 50.0% and 71.2% for the same head).
            //
            // It is exactly the conditional rate the expected-prefix arithmetic
            // wants. `E = 1 + p1 + p1 p2 + ...` where `p_d = P(draft d right |
            // drafts 1..d-1 right)`, and "drafts 1..d-1 right" is precisely the
            // state teacher forcing puts head `d` in: the token head `d-1` was
            // fed IS the token the verifier would have accepted.
            //
            // FULL vocabulary, never the pruned candidate table -- the pruning is
            // a property of the drafting step's candidate gather and not of the
            // head, and mixing the two would price two changes with one number.
            let teach = std::env::var("INK_MTP_TEACH")
                .map(|v| v == "1")
                .unwrap_or(false);
            // The first row to score. Zero for a plain corpus; set it to the
            // PROMPT length when the ids file is a seed followed by this model's
            // own greedy continuation, so the rate is measured on the sequence a
            // decode loop actually verifies.
            //
            // That distinction is not a detail. Teacher forcing on a human
            // document asks the draft head to predict a token the MAIN STACK
            // itself only gets right 30% of the time; at decode the target IS the
            // main stack's argmax, which the head was trained to anticipate. So a
            // corpus rate is a LOWER BOUND on the decode rate, and the two should
            // never be quoted as the same number.
            let teach_from = std::env::var("INK_MTP_TEACH_FROM")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            // `norm` is FALSE for a row that has already taken the final norm.
            // `mtp_main` holds `entry`, which IS final-normed unless INK_MTP_RAW
            // is set, and norming it again is not idempotent: rms_norm of
            // (normalise(x) * g) is normalise(x) * g^2 / c, so the ceiling would
            // be measured through the final norm's learned gain SQUARED.
            let teach_rows = |rows: T2, norm: bool| -> Vec<usize> {
                let rows_n = rows.dims()[0];
                let hs = if norm {
                    dev_lane::rms_norm(
                        rows,
                        fnorm_dev
                            .clone()
                            .expect("teacher forcing needs the final norm"),
                        t.rms_norm_eps,
                    )
                } else {
                    rows
                }
                .div_scalar(t.logits_mup_width_multiplier as f32);
                // SLICED to the effective vocabulary before the argmax, exactly
                // as the tail's own logits path and `draft_pick` are. The bound
                // table is `vocab_size` rows wide and the model only uses
                // `effective_vocab()` of them; the rest are padding, and an argmax
                // that can land in the padding is not measuring the model.
                dev_lane::linear_w(
                    hs,
                    unembed_w
                        .as_ref()
                        .expect("teacher forcing needs the unembed table"),
                )
                .slice([0..rows_n, 0..v])
                .argmax(1)
                .into_data()
                .iter::<i64>()
                .map(|x| x as usize)
                .collect()
            };

            draft_probs.borrow_mut().clear();
            let t_mtp = Instant::now();
            let drafts: Vec<usize> = if kv && mtp_dev_on {
                // The device lane, structurally identical to the host one below --
                // ragged stable rows, a speculative tail run against a CLONE of the
                // cache -- so the two can be read against each other line for line.
                if mtp_devs.is_empty() {
                    let t_up = Instant::now();
                    let mut bytes = 0u64;
                    for i in 0..mtp_k {
                        let pre = format!("model.mtp.layers.{i}.");
                        let p = format!("{pre}transformer_block.");
                        let hd = &mtp_heads[i].dims;
                        let pw = |nm: &str, rows: usize, cols: usize| -> Result<Bf16W> {
                            let leaf = cp.stored(&format!("{p}{nm}"))?;
                            anyhow::ensure!(
                                leaf.elem == Elem::Bf16,
                                "{p}{nm} is {:?}; this lane multiplies BF16 by BF16",
                                leaf.elem
                            );
                            Ok(bind_bf16(
                                &fp4_client,
                                fp4_aliases.as_ref(),
                                &leaf.bytes,
                                rows,
                                cols,
                            ))
                        };
                        let ip = cp.stored(&format!("{pre}input_proj.weight"))?;
                        anyhow::ensure!(ip.elem == Elem::Bf16, "input_proj is {:?}", ip.elem);
                        let gv = |nm: &str| -> Result<Vec<f32>> { Ok(cp.tensor(nm)?.data) };
                        // Bound here rather than through [`DeviceDense`], whose map
                        // is keyed by LAYER prefix and whose byte counter feeds the
                        // per-layer report: an MTP head is not one of this node's
                        // layers and counting it there would misattribute 1 GiB.
                        let dense = {
                            let fused = cp.stored(&format!("{p}mlp.w13_dn.weight"))?;
                            anyhow::ensure!(
                                fused.elem == Elem::Bf16,
                                "mtp w13 is {:?}",
                                fused.elem
                            );
                            let (g, u) = mary::models::inkling::load::split_gate_up_bytes(
                                &fused.bytes,
                                h,
                                2,
                            );
                            let dw = cp.stored(&format!("{p}mlp.w2_md.weight"))?;
                            anyhow::ensure!(dw.elem == Elem::Bf16, "mtp w2 is {:?}", dw.elem);
                            let (drows, dcols) = (dw.dims[0] as usize, dw.dims[1] as usize);
                            let inter = g.len() / (h * 2);
                            let gs = cp.tensor(&format!("{p}mlp.global_scale"))?.data[0];
                            bytes += (g.len() + u.len() + dw.bytes.len()) as u64;
                            (
                                bind_bf16(&fp4_client, fp4_aliases.as_ref(), &g, inter, h),
                                bind_bf16(&fp4_client, fp4_aliases.as_ref(), &u, inter, h),
                                bind_bf16(
                                    &fp4_client,
                                    fp4_aliases.as_ref(),
                                    &dw.bytes,
                                    drows,
                                    dcols,
                                ),
                                gs,
                            )
                        };
                        let built = MtpDev {
                            attn: dev_lane::AttnWeightsDev {
                                wq: pw("attn.wq_du.weight", hd.heads * hd.head_dim, h)?,
                                wk: pw("attn.wk_dv.weight", hd.kv_heads * hd.head_dim, h)?,
                                wv: pw("attn.wv_dv.weight", hd.kv_heads * hd.head_dim, h)?,
                                wr: pw("attn.wr_du.weight", hd.heads * hd.d_rel, h)?,
                                wqkvr: None,
                                wo: pw("attn.wo_ud.weight", h, hd.heads * hd.head_dim)?,
                                k_sconv: up2(
                                    gv(&format!("{p}attn.k_sconv.weight"))?,
                                    hd.kv_heads * hd.head_dim,
                                    t.sconv_kernel_size,
                                    &dev,
                                ),
                                v_sconv: up2(
                                    gv(&format!("{p}attn.v_sconv.weight"))?,
                                    hd.kv_heads * hd.head_dim,
                                    t.sconv_kernel_size,
                                    &dev,
                                ),
                                q_norm: up1(
                                    gv(&format!("{p}attn.q_norm.weight"))?,
                                    hd.head_dim,
                                    &dev,
                                ),
                                k_norm: up1(
                                    gv(&format!("{p}attn.k_norm.weight"))?,
                                    hd.head_dim,
                                    &dev,
                                ),
                                rel_proj: up2(
                                    gv(&format!("{p}attn.rel_logits_proj.proj"))?,
                                    hd.d_rel,
                                    hd.rel_extent,
                                    &dev,
                                ),
                            },
                            attn_sconv: up2(
                                gv(&format!("{p}attn_sconv.weight"))?,
                                h,
                                t.sconv_kernel_size,
                                &dev,
                            ),
                            mlp_sconv: up2(
                                gv(&format!("{p}mlp_sconv.weight"))?,
                                h,
                                t.sconv_kernel_size,
                                &dev,
                            ),
                            attn_norm: up1(gv(&format!("{p}attn_norm.weight"))?, h, &dev),
                            mlp_norm: up1(gv(&format!("{p}mlp_norm.weight"))?, h, &dev),
                            embed_norm: up1(gv(&format!("{pre}embed_norm.weight"))?, h, &dev),
                            hidden_norm: up1(gv(&format!("{pre}hidden_norm.weight"))?, h, &dev),
                            input_proj: bind_bf16(
                                &fp4_client,
                                fp4_aliases.as_ref(),
                                &ip.bytes,
                                h,
                                2 * h,
                            ),
                            dense,
                        };
                        bytes += ip.bytes.len() as u64;
                        mtp_devs.push(built);
                    }
                    <Bk as burn::tensor::backend::Backend>::sync(&dev)
                        .expect("sync after MTP upload");
                    println!(
                        "  MTP heads on the device: {mtp_k} in {:.2}s, {:.2} GiB bound",
                        t_up.elapsed().as_secs_f32(),
                        bytes as f64 / GIB
                    );
                }
                let main_dev = mtp_main_dev
                    .clone()
                    .expect("the entry states were uploaded above");
                // The tree's candidates, filled at d == 0 and read after the
                // loop. Head 0's newest STABLE row is the only row a depth-1
                // tree reads, and the decode step has already computed it.
                let mut tree_next: Vec<usize> = Vec::new();
                let mut prev_rows: Vec<T2> = Vec::new();
                let mut drafts: Vec<usize> = Vec::with_capacity(mtp_k);
                for d in 0..mtp_k {
                    let hd = mtp_heads[d].dims;
                    let window = mtp_heads[d].window(t.sliding_window_size);
                    let want = seq - d;
                    let have = mtp_stage_dev[d].as_ref().map(|x| x.dims()[0]).unwrap_or(0);
                    let row_of = |src: &Option<T2>, lo: usize, hi: usize| -> T2 {
                        match src {
                            Some(x) => x.clone().slice([lo..hi, 0..h]),
                            None => {
                                unreachable!("head d-1 always has more stable rows than head d")
                            }
                        }
                    };
                    let stable: T2 = if have == 0 {
                        let mut embeds = vec![0f32; want * h];
                        for j in 0..want {
                            let tok = if j + d + 1 < seq {
                                ids[j + d + 1]
                            } else {
                                best
                            };
                            embeds[j * h..(j + 1) * h].copy_from_slice(&mtp_embed_row(
                                e_w,
                                e_bn,
                                tok,
                                t.rms_norm_eps,
                                t.vocab_size,
                                h,
                            ));
                        }
                        let ed = up2::<Bk>(embeds, want, h, &dev);
                        let hin = if d == 0 {
                            main_dev.clone().slice([0..want, 0..h])
                        } else {
                            row_of(&mtp_stage_dev[d - 1], 0, want)
                        };
                        let (y, c) = mtp_block_prefill_dev(
                            hin,
                            ed,
                            &mtp_devs[d],
                            &hd,
                            Some(ls),
                            window,
                            t.sconv_kernel_size,
                            t.rms_norm_eps,
                            mtp_order,
                        );
                        mtp_dev_caches[d] = Some(c);
                        if teach && d == 0 {
                            // The CEILING, and the number every depth below is
                            // meaningless without: the main stack's own
                            // teacher-forced next-token accuracy on these same
                            // rows. A draft head at 0.227 is a catastrophe against
                            // a stack at 0.9 and unremarkable against a stack at
                            // 0.3, and nothing in a per-depth table says which.
                            let last = seq.saturating_sub(1);
                            let first = teach_from.min(last);
                            let mut hits = 0usize;
                            let mut scored = 0usize;
                            let nblk = ((last - first) / 256).max(1);
                            let take = nblk.min(8);
                            for b in 0..take {
                                let lo = first
                                    + if take == 1 {
                                        0
                                    } else {
                                        (b * (nblk - 1) / (take - 1).max(1)) * 256
                                    };
                                let hi = (lo + 256).min(last);
                                if hi <= lo {
                                    continue;
                                }
                                // `main_dev` is `entry`: normed already, unless
                                // INK_MTP_RAW asked for the stack's raw output.
                                let picks = teach_rows(
                                    main_dev.clone().slice([lo..hi, 0..h]),
                                    std::env::var("INK_MTP_RAW")
                                        .map(|val| val == "1")
                                        .unwrap_or(false),
                                );
                                for (i, &pick) in picks.iter().enumerate() {
                                    if pick == ids[lo + i + 1] {
                                        hits += 1;
                                    }
                                    scored += 1;
                                }
                            }
                            let (clo, chi) = wilson95(hits, scored);
                            println!(
                                "  MTP TEACH depth 0 (THE MAIN STACK): {hits}/{scored} = {:.4} \
                                 (95% CI {:.4}-{:.4}) -- the same rows, the same unembedding, \
                                 argmax against the token the prompt actually has next",
                                hits as f64 / scored.max(1) as f64,
                                clo,
                                chi,
                            );
                        }
                        if teach {
                            // Row `j` predicts `ids[j + d + 2]`, so the last row
                            // with an answer in the prompt is `seq - d - 3`.
                            let last = seq.saturating_sub(d + 2);
                            // Scored in contiguous blocks of `TEACH_BLOCK` rows,
                            // spread EVENLY over the prompt rather than taken from
                            // one end: acceptance is a property of the text, and a
                            // document's first 2048 tokens are not its last 2048.
                            // The cap exists because the cost is per CALL -- the
                            // unembedding streams its whole table however many rows
                            // it is handed -- so 8 blocks of 256 is 8 streams, and
                            // scoring every row would be 15.
                            const TEACH_BLOCK: usize = 256;
                            let cap = std::env::var("INK_MTP_TEACH_MAX")
                                .ok()
                                .and_then(|v| v.parse::<usize>().ok())
                                .unwrap_or(2048);
                            let first = teach_from.min(last);
                            let nblk = ((last - first) / TEACH_BLOCK).max(1);
                            let take = nblk.min((cap / TEACH_BLOCK).max(1));
                            let mut hits = 0usize;
                            let mut scored = 0usize;
                            for b in 0..take {
                                let lo = first
                                    + if take == 1 {
                                        0
                                    } else {
                                        (b * (nblk - 1) / (take - 1).max(1)) * TEACH_BLOCK
                                    };
                                let hi = (lo + TEACH_BLOCK).min(last);
                                if hi <= lo {
                                    continue;
                                }
                                let picks =
                                    teach_rows(y.clone().slice([lo..hi, 0..h]), mtp_out_norm());
                                for (i, &pick) in picks.iter().enumerate() {
                                    if pick == ids[lo + i + d + 2] {
                                        hits += 1;
                                    }
                                    scored += 1;
                                }
                            }
                            let (clo, chi) = wilson95(hits, scored);
                            println!(
                                "  MTP TEACH depth {}: {hits}/{scored} = {:.4} (95% CI \
                                 {:.4}-{:.4}) -- teacher-forced over the prompt, FULL vocab, \
                                 concat {}, entry {}, backbone embed_norm {}",
                                d + 1,
                                hits as f64 / scored.max(1) as f64,
                                clo,
                                chi,
                                mtp_order.name(),
                                if std::env::var("INK_MTP_RAW")
                                    .map(|v| v == "1")
                                    .unwrap_or(false)
                                {
                                    "raw"
                                } else {
                                    "final-normed"
                                },
                                if t.use_embed_norm && backbone_embed_norm() {
                                    "on"
                                } else {
                                    "off"
                                },
                            );
                        }
                        y
                    } else {
                        // ONE row per CONFIRMED TOKEN, not one per pass. That used
                        // to be the same number and an `assert_eq!(have + 1, want)`
                        // said so; a speculative pass confirms 1 + accepted tokens
                        // at once and every one of them makes a row of every head
                        // stable. Head d's row at `pos` is fed the token at
                        // pos + d + 1, which is in `ids` for all of them but the
                        // last, where it is the argmax still being held back.
                        let adv = want - have;
                        assert!(adv >= 1, "a pass makes at least one row stable");
                        let mut made: Vec<T2> = Vec::with_capacity(adv);
                        for i in 0..adv {
                            let pos = have + i;
                            let hin = if d == 0 {
                                main_dev.clone().slice([pos..pos + 1, 0..h])
                            } else {
                                row_of(&mtp_stage_dev[d - 1], pos, pos + 1)
                            };
                            let ahead = pos + d + 1;
                            let tok = if ahead < seq { ids[ahead] } else { best };
                            let ed = up2::<Bk>(
                                mtp_embed_row(e_w, e_bn, tok, t.rms_norm_eps, t.vocab_size, h),
                                1,
                                h,
                                &dev,
                            );
                            made.push(mtp_block_step_dev(
                                hin,
                                ed,
                                &mtp_devs[d],
                                &hd,
                                Some(ls),
                                pos,
                                window,
                                mtp_dev_caches[d]
                                    .as_mut()
                                    .expect("prefilled on the first pass"),
                                t.rms_norm_eps,
                                mtp_order,
                            ));
                        }
                        let stepped = if made.len() == 1 {
                            made.pop().expect("one row")
                        } else {
                            BT::cat(made, 0)
                        };
                        // `INK_MTP_STEPCHECK=1`: the cached STEP against a fresh
                        // whole-sequence PREFILL of the same head over the same
                        // rows. Nothing else checks this pair.
                        //
                        // `INK_MTP_CHECK` compares the cached lane to the host
                        // whole-sequence one, and is ASSERTED only for the host
                        // cached lane -- the device one it merely reports. So the
                        // lane every speculative run actually drafts on has never
                        // had its step path checked against its own prefill, and a
                        // step path that diverges does not error: it shows up as an
                        // acceptance rate, months later, indistinguishable from a
                        // model that simply drafts badly.
                        //
                        // Expensive by construction (a full prefill per head per
                        // pass), so: short prompt, few steps.
                        if std::env::var("INK_MTP_STEPCHECK")
                            .map(|val| val == "1")
                            .unwrap_or(false)
                        {
                            let mut embeds = vec![0f32; want * h];
                            for j in 0..want {
                                let tok = if j + d + 1 < seq {
                                    ids[j + d + 1]
                                } else {
                                    best
                                };
                                embeds[j * h..(j + 1) * h].copy_from_slice(&mtp_embed_row(
                                    e_w,
                                    e_bn,
                                    tok,
                                    t.rms_norm_eps,
                                    t.vocab_size,
                                    h,
                                ));
                            }
                            let ed = up2::<Bk>(embeds, want, h, &dev);
                            let hin = if d == 0 {
                                main_dev.clone().slice([0..want, 0..h])
                            } else {
                                row_of(&mtp_stage_dev[d - 1], 0, want)
                            };
                            let (fresh, _) = mtp_block_prefill_dev(
                                hin,
                                ed,
                                &mtp_devs[d],
                                &hd,
                                Some(ls),
                                window,
                                t.sconv_kernel_size,
                                t.rms_norm_eps,
                                mtp_order,
                            );
                            let a = fresh.slice([want - 1..want, 0..h]);
                            let b = stepped.clone().slice([adv - 1..adv, 0..h]);
                            let diff = (a.clone() - b.clone()).abs().max().into_scalar();
                            let scale = a.clone().abs().max().into_scalar();
                            let (ta, tb) = (draft_argmax_dev(a), draft_argmax_dev(b));
                            println!(
                                "  MTP STEPCHECK depth {} pos {}: max|prefill-step| {:.4e}                                  against |prefill|max {:.4e} ({:.2}%); token {} vs {} -- {}",
                                d + 1,
                                want - 1,
                                diff,
                                scale,
                                100.0 * diff as f64 / (scale as f64).max(1e-30),
                                ta,
                                tb,
                                if ta == tb { "agree" } else { "DISAGREE" }
                            );
                        }
                        stepped
                    };
                    mtp_stage_dev[d] = Some(match mtp_stage_dev[d].take() {
                        None => stable,
                        Some(prev) => BT::cat(vec![prev, stable], 0),
                    });
                    let mut rows: Vec<T2> = vec![row_of(&mtp_stage_dev[d], want - 1, want)];
                    let mut last = rows[0].clone();
                    if d > 0 {
                        let mut scratch = mtp_dev_caches[d].as_ref().expect("prefilled").clone();
                        for i in 0..d {
                            let ed = up2::<Bk>(
                                mtp_embed_row(
                                    e_w,
                                    e_bn,
                                    drafts[i],
                                    t.rms_norm_eps,
                                    t.vocab_size,
                                    h,
                                ),
                                1,
                                h,
                                &dev,
                            );
                            last = mtp_block_step_dev(
                                prev_rows[i].clone(),
                                ed,
                                &mtp_devs[d],
                                &hd,
                                Some(ls),
                                want + i,
                                window,
                                &mut scratch,
                                t.rms_norm_eps,
                                mtp_order,
                            );
                            rows.push(last.clone());
                        }
                    }
                    if tree_b > 0 && d == 0 {
                        // ONE unembedding, read twice. The tree's candidates for
                        // t+1 and the chain's argmax are the same row's top-b and
                        // its top-1, and `cands[0]` IS that argmax -- so the
                        // depth-1 tree's whole draft side is a wider read of a
                        // matmul the step was already going to run. Not a single
                        // extra head step.
                        let cands = draft_topb(last.clone(), tree_b);
                        anyhow::ensure!(
                            cands.len() == tree_b,
                            "the drafter offered {} candidates against INK_TREE={tree_b}",
                            cands.len()
                        );
                        for (i, &a) in cands.iter().enumerate() {
                            anyhow::ensure!(
                                !cands[..i].contains(&a),
                                "two candidates hold token {a}; a verifier argmax would not \
                                 name one branch"
                            );
                        }
                        drafts.push(cands[0]);
                        tree_next = cands;
                    } else {
                        drafts.push(draft_argmax_dev(last));
                    }
                    prev_rows = rows;
                }
                if tree_b > 0 {
                    tree_drafts = tree_next;
                }
                drafts
            } else if kv {
                // Head d's newest STABLE row is at position seq-1-d: past that, its
                // input embedding is a token the stack has not produced yet. So a
                // step appends exactly ONE row per head, and the d rows after it --
                // the ones the draft is actually read off -- are functions of drafts
                // and run against a CLONE of the cache that is then dropped.
                // Rollback is the default and there is no commit path to forget,
                // which matters because a speculative K/V left behind does not
                // error: it shows up months later as an acceptance rate that drifts
                // down.
                let mut prev_rows: Vec<Vec<f32>> = Vec::new();
                let mut drafts: Vec<usize> = Vec::with_capacity(mtp_heads.len());
                for (d, headw) in mtp_heads.iter().enumerate() {
                    let window = headw.window(t.sliding_window_size);
                    let hw = headw.borrow(t.dense_intermediate_size);
                    let want = seq - d;
                    let have = mtp_stage[d].len() / h;
                    let stable: Vec<f32> = if have == 0 {
                        // The prefill: every stable row this head has, in one
                        // whole-sequence pass -- the same call the uncached lane
                        // makes, so the same arithmetic seeds the cache.
                        let mut embeds = vec![0f32; want * h];
                        for j in 0..want {
                            let tok = if j + d + 1 < seq {
                                ids[j + d + 1]
                            } else {
                                best
                            };
                            embeds[j * h..(j + 1) * h].copy_from_slice(&mtp_embed_row(
                                e_w,
                                e_bn,
                                tok,
                                t.rms_norm_eps,
                                t.vocab_size,
                                h,
                            ));
                        }
                        let hin: &[f32] = if d == 0 {
                            &mtp_main[..want * h]
                        } else {
                            &mtp_stage[d - 1][..want * h]
                        };
                        let (y, cache) = mtp_block_prefill(
                            hin,
                            &embeds,
                            &hw,
                            &headw.dims,
                            Some(ls),
                            window,
                            want,
                            mtp_order,
                        );
                        mtp_caches[d] = Some(cache);
                        y
                    } else {
                        // One row, at the position the token just produced made
                        // stable. Its embedding is `best` for EVERY head: head d's
                        // row seq-1-d wants the token at seq, whatever d is.
                        assert_eq!(have + 1, want, "a step makes exactly one row stable");
                        let p = want - 1;
                        let hin: &[f32] = if d == 0 {
                            &mtp_main[p * h..(p + 1) * h]
                        } else {
                            &mtp_stage[d - 1][p * h..(p + 1) * h]
                        };
                        mtp_block_step(
                            hin,
                            &mtp_embed_row(e_w, e_bn, best, t.rms_norm_eps, t.vocab_size, h),
                            &hw,
                            &headw.dims,
                            Some(ls),
                            p,
                            window,
                            mtp_caches[d].as_mut().expect("prefilled on the first pass"),
                            mtp_order,
                        )
                    };
                    mtp_stage[d].extend_from_slice(&stable);
                    debug_assert_eq!(mtp_stage[d].len(), want * h);

                    // The speculative tail: positions want..seq-1, one per draft
                    // already made, each attending to the ones before it. `rows` is
                    // what head d+1 reads -- the input to its own stable row, then
                    // the inputs to its speculative ones.
                    let mut rows: Vec<Vec<f32>> =
                        vec![mtp_stage[d][(want - 1) * h..want * h].to_vec()];
                    let mut last = rows[0].clone();
                    if d > 0 {
                        let mut scratch = mtp_caches[d].as_ref().expect("prefilled").clone();
                        for i in 0..d {
                            last = mtp_block_step(
                                &prev_rows[i],
                                &mtp_embed_row(
                                    e_w,
                                    e_bn,
                                    drafts[i],
                                    t.rms_norm_eps,
                                    t.vocab_size,
                                    h,
                                ),
                                &hw,
                                &headw.dims,
                                Some(ls),
                                want + i,
                                window,
                                &mut scratch,
                                mtp_order,
                            );
                            rows.push(last.clone());
                        }
                    }
                    // `scratch` is gone with the block: the speculative K and V
                    // never reached the cache, so a wrong draft leaves nothing to
                    // undo.
                    drafts.push(draft_argmax(&last));
                    prev_rows = rows;
                }
                drafts
            } else {
                draft_whole(&mtp_main, seq, &ids, best)
            };
            // `tree_b == 0` for the same reason `spec_k == 0` is here: this
            // bookkeeping keys a draft by the STEP it will be scored at, which
            // is only a token index while a pass confirms exactly one token. A
            // tree pass confirms one or two, so every key past the first
            // acceptance would name the wrong position -- and it would read as a
            // per-depth acceptance rate, not as an error.
            if spec_k == 0 && tree_b == 0 {
                mtp_issued.insert(step, vec![None; drafts.len()]);
                if mtp_prob {
                    mtp_issued_q.insert(step, vec![None; drafts.len()]);
                    for (d, p) in draft_probs.borrow_mut().drain(..).enumerate() {
                        mtp_pd.insert((step + d + 1, d), p);
                    }
                }
                for (d, &b) in drafts.iter().enumerate() {
                    // Head d predicts the token d+1 steps past the one just chosen.
                    mtp_pending.push((step + d + 1, d, b));
                }
            }
            let d_mtp = t_mtp.elapsed().as_secs_f64();
            acc_draft += d_mtp;
            t_draft += d_mtp;
            // The drafts go to the head, which is the only process that can embed
            // them. Sent HERE and not with the answer, so the head was blocked on
            // the verify pass alone and this drafting overlaps its commit.
            if spec_k > 0 {
                last_drafts = drafts.clone();
                if let Some(Pipe::Tail(s)) = pipe.as_mut() {
                    send_toks(s, &drafts)?;
                }
            }
            println!(
                "  MTP drafted {} token(s) in {:.3}s: {drafts:?}",
                mtp_heads.len(),
                t_mtp.elapsed().as_secs_f32(),
            );
            // The cached lane against the whole-sequence one, on the SAME retained
            // hidden states -- so a disagreement is about the CACHE and nothing
            // else. Both lanes are scalar host arithmetic summed in the same order,
            // so the drafts should agree EXACTLY; anything less is a bug and not
            // rounding, which is what makes this worth asserting rather than
            // reporting.
            if kv
                && std::env::var("INK_MTP_CHECK")
                    .map(|val| val == "1")
                    .unwrap_or(false)
            {
                let t_c = Instant::now();
                let whole = draft_whole(&mtp_main, seq, &ids, best);
                println!(
                    "  MTP cache check ({:.2}s): cached {drafts:?} vs whole-sequence {whole:?} -- {}",
                    t_c.elapsed().as_secs_f32(),
                    if whole == drafts { "agree" } else { "DISAGREE" }
                );
                // Asserted for the HOST cached lane, which is the same arithmetic
                // summed in the same order and therefore has no licence to differ;
                // REPORTED for the device one, where an argmax over a 200k row can
                // legitimately flip on a near-tie. Asserting there would be a
                // determinism gate wearing a correctness costume, and the honest
                // instrument for a transcription is the acceptance rate it goes on
                // to produce.
                anyhow::ensure!(
                    mtp_dev_on || whole == drafts,
                    "the MTP cache drafted {drafts:?} where the whole sequence drafts {whole:?}"
                );
            }
        }
        // A tail follows the sequence by RECOMPUTING it, not by being told: it owns
        // the argmax, so pushing it here keeps its `ids` identical to the head's
        // without a second thing on the wire to get out of step.
        if is_tail && gen_steps > 0 && !repeat {
            ids.push(best);
        }

        let t_tail_host = t_tl.elapsed().as_secs_f64();
        // The pass as it stands BEFORE the report prints a line. Sampling it
        // inside the report -- which is what this used to do -- charged the
        // report's own hundred-odd `println!`s to the pass it was describing,
        // and stdout to a pipe is not free.
        let whole_pass = pass.elapsed().as_secs_f64();
        println!("\n=== predictions ===");
        println!("  expert slabs decoded: {expert_loads}");
        // Which branch the cleanup policy took, as a count and not as an inference
        // from the absence of a line. Zero on a pass the node had room for.
        println!("  pool cleanups: {cleanups} of {} layers", hi - lo);
        // t_other covers the whole MLP half, so the expert buckets are inside it.
        // MILLISECONDS. At 0.1 s resolution a decode pass of this stack prints as
        // "0.4 0.4 0.0" and every question worth asking of it is unanswerable.
        let ms = |v: f64| v * 1e3;
        // Two kinds of number, kept apart, because with the residual stream on the
        // device they no longer measure the same thing. Everything above the sync
        // line is the HOST describing work; the device time is the sync line. A
        // per-stage "seconds" column would now report how fast the CPU can enqueue,
        // and it would look wonderful.
        println!("  where the time went, ms:");
        println!("    HOST, enqueue only (nothing in the loop synchronises):");
        println!(
            "      embed + upload  {:9.1}   (the one host->device crossing per pass)",
            ms(t_embed)
        );
        println!("      attention half  {:9.1}", ms(t_attn));
        println!("      mlp half        {:9.1}   of which:", ms(t_other));
        println!(
            "        routed experts{:9.1}   (slice + bind + issue)",
            ms(t_expert)
        );
        println!("        shared experts{:9.1}", ms(t_shared));
        println!(
            "        rest of half  {:9.1}   (dense layers, sconv)",
            ms(t_other - t_expert - t_shared - t_h_route - t_h_sconv)
        );
        // WHICH lane blocked, not which one used to. `INK_DEV_ROUTE` has defaulted
        // on for a while and the device lane reads back the DECISION -- 2k + shared
        // + 1 f32, 60 bytes at decode -- not the logits. The line said `[n, 258]`
        // and "top-k on the host" either way, and a reader who priced the sync as a
        // 1 KB transfer was reading a label for a lane that had stopped running.
        // The cost is the SYNC and it is the same either way; the bytes are not.
        println!(
            "      router + group  {:9.1}   BLOCKS: [n,{}] {} back",
            ms(t_h_route),
            if dev_route {
                2 * t.num_experts_per_tok + t.n_shared_experts + 1
            } else {
                t.n_routed_experts + t.n_shared_experts
            },
            if dev_route {
                "top-k DECISION (device top-k)"
            } else {
                "logits, then top-k on the host"
            }
        );
        println!(
            "        of which: matmul enqueue {:7.1}, BLOCKING read {:7.1}, top-k + group {:7.1}",
            ms(t_rt_mm),
            ms(t_rt_read),
            ms(t_rt_host)
        );
        println!("      mlp short_conv  {:9.1}", ms(t_h_sconv));
        println!(
            "      first-touch uploads: read+widen {:9.1}, transfer {:9.1}   (once per layer, not per token)",
            ms(t_attn_read),
            ms(t_attn_up)
        );
        println!(
            "    DEVICE, one sync for this node's whole stack: {:9.1}",
            ms(t_stack_sync)
        );
        if stage_sync {
            println!(
                "    DEVICE per stage (INK_STAGE_SYNC=1 -- {stage_syncs} extra syncs, this pass IS slower for them):"
            );
            println!("      attention half  {:9.1}", ms(d_attn));
            println!("      router matmul   {:9.1}", ms(d_router));
            println!("      routed experts  {:9.1}", ms(d_expert));
            println!("      shared experts  {:9.1}", ms(d_shared));
            println!("      sconv + resid   {:9.1}", ms(d_tail));
            println!(
                "      staged total    {:9.1}",
                ms(d_attn + d_router + d_expert + d_shared + d_tail)
            );
        }
        println!(
            "    {:17} {:9.1}   ({})",
            if best_wire.is_some() {
                "tail + wire"
            } else {
                "head / unembed"
            },
            ms(t_head),
            if best_wire.is_some() {
                "BLOCKING: the other machine's layers, its head, and the round trip"
            } else {
                "device"
            }
        );
        if best_wire.is_none() && unembed_w.is_some() {
            // Why the head is worth reading as PHYSICS and not as overhead: the
            // unembed table is read WHOLE on every pass. `[vocab_size, hidden]`
            // BF16 is a fixed number of bytes per step, independent of context
            // length and of how many layers this node holds, so it lands in the
            // per-step INTERCEPT rather than the per-layer slope -- and no
            // launch-side change (fusion, graph capture, a deeper queue) can move
            // a byte of it. Printed as bytes rather than as a predicted
            // millisecond because the millisecond needs a bandwidth figure, and a
            // bandwidth figure quoted without its own measurement is not evidence.
            println!(
                "      unembed table {:9.2} GiB of BF16, read whole every pass (a per-step FLOOR: bytes / achievable bandwidth)",
                (t.vocab_size * h * 2) as f64 / GIB
            );
        }
        println!(
            "    residual to the host{:9.1}   (the [n, hidden] the wire and the draft path read)",
            ms(t_x_down)
        );
        // The report is a partition or it is decoration.
        //
        // It was decoration. `named` summed six stage timers that between them
        // covered maybe two thirds of a pass: the first-touch uploads were
        // MEASURED and then left out of the sum, the per-layer pool hand-back was
        // never timed at all, and everything from the head to here -- the argmax
        // over a 200k-wide row, the KV commit, the MTP draft -- was in no bucket.
        // Worse, `whole` was read INSIDE the report, so the report's own hundred
        // `println!`s were charged to UNATTRIBUTED as well.
        //
        // So the partition is now cut at four OUTER brackets that tile the pass
        // by construction -- prologue, layer loop, stack sync, everything after
        // it -- and every stage above is a part of one of them. A number that
        // grows here is now genuinely work no line names, which is the only
        // reading that was ever worth printing.
        let t_named_in_loop = t_attn + t_other + t_attn_read + t_attn_up + t_cleanup;
        let t_named_after = t_x_down + t_head + t_draft + t_wait_peer;
        println!(
            "    pass prologue   {:9.1}   (feed construction, before the embedding)",
            ms(t_prep)
        );
        println!(
            "    layer loop      {:9.1}   (outer bracket; the HOST lines above are its parts)",
            ms(t_layers)
        );
        println!(
            "      pool hand-back{:9.1}   ({} of {} layers cleaned; {} pool polls costing {:.1} ms -- \
             each poll is a blocking round trip to the compute-server thread, not free bookkeeping)",
            ms(t_cleanup),
            cleanups,
            hi - lo,
            pool_polls,
            ms(t_pool_poll)
        );
        println!(
            "      unnamed in-loop{:8.1}   (loop total less attention + mlp + first-touch + hand-back)",
            ms(t_layers - t_named_in_loop)
        );
        if t_seat > 0.0005 {
            println!(
                "    slot seating    {:9.1}   (prefill only: seat + sync + pool hand-back)",
                ms(t_seat)
            );
        }
        println!(
            "    after the sync  {:9.1}   (outer bracket: RMS lines, residual, head, argmax, commit, draft)",
            ms(t_tail_host)
        );
        println!(
            "      MTP draft     {:9.1}, peer wait {:9.1}, sampling + commit {:9.1}",
            ms(t_draft),
            ms(t_wait_peer),
            ms(t_tail_host - t_named_after)
        );
        println!(
            "        of which host argmax {:9.1}   ({} row(s) x {v} f32 on one core)",
            ms(t_argmax),
            new_toks.len().max(1)
        );
        if let Some(a) = ann_stat {
            // The shortlist is the lane's own account of what it did, and it is
            // printed every pass rather than summarised: a step whose top is flat
            // enough to overflow the budget is exactly the step whose token is
            // worth doubting, and an average would hide it.
            println!(
                "        aNN head: {} above the floor of {} (floor {:.3}, best estimate \
                 {:.3}, budget {}){}",
                a.shortlist,
                t.vocab_size,
                a.floor,
                a.est_max,
                ann_budget(),
                // The counter is uncapped and the rescore is not, so this is the
                // one line that can say the shortlist did not fit. It is not a
                // wrong answer -- the rows that were rescored are exact -- it is
                // a step whose top was flat enough that the budget did not cover
                // it, and it is worth seeing rather than averaging away.
                if a.shortlist > ann_budget() + ann_budget() / 4 + 1024 {
                    "  OVERFLOW: only the first budget*4 were rescored"
                } else {
                    ""
                }
            );
        }
        {
            let named = t_prep + t_embed + t_layers + t_seat + t_stack_sync + t_tail_host;
            println!(
                "    UNATTRIBUTED    {:9.1}   (this pass, {:.1} ms, less the four outer brackets)",
                ms(whole_pass - named),
                ms(whole_pass)
            );
        }
        println!(
            "    of the above, host-only tensor reads (mmap + BF16 widening): {:9.1}",
            ms(t_read.get())
        );
        {
            // What the HOST did in the routed-expert lane, one bucket per kind of
            // work. `read (BLOCKS)` is gone from this list because the read is
            // gone: the accumulator is a device tensor and nothing waits for it.
            println!("    of the routed-expert total, what the host did ({expert_loads} loads):");
            println!(
                "      slice from pile {:9.1}   ({:.3} ms/load)   BLAKE3 on first touch",
                ms(host_t.slice),
                ms(host_t.slice) / expert_loads.max(1) as f64
            );
            println!(
                "      gather (select) {:9.1}   ({:.3} ms/load)   enqueue",
                ms(host_t.gather),
                ms(host_t.gather) / expert_loads.max(1) as f64
            );
            // Sub-buckets of the line above, not extra time: the plan uploads happen
            // inside the gather's timer. Split because only one of the two halves
            // is a consequence of the routing decision living on the host.
            println!(
                "        of which plan uploads: routing-dependent {:6.1}, static at n=1 {:6.1}",
                ms(host_t.plan_up_routed),
                ms(host_t.plan_up_static)
            );
            // Which arm each MoE layer of this pass actually took. The two counters
            // above cannot say -- both lanes are "grouped" and both load six slabs
            // -- and a `BLOCKING read` that is not zero on the device arm is a
            // layer this lane refused, not noise.
            println!(
                "        row plan: {} layer(s) on the DEVICE, {} off a blocking read",
                host_t.plan_dev, host_t.plan_host
            );
            println!(
                "      bind + enqueue  {:9.1}   ({:.3} ms/load)",
                ms(host_t.enqueue),
                ms(host_t.enqueue) / expert_loads.max(1) as f64
            );
            println!(
                "      scatter-add     {:9.1}   ({:.3} ms/load)   enqueue",
                ms(host_t.accum),
                ms(host_t.accum) / expert_loads.max(1) as f64
            );
            // WHICH lane ran, counted rather than asserted. The grouped one is a
            // claim about launches per LAYER and the per-expert one about launches
            // per EXPERT; a per-load average over a mixture of the two is a number
            // about neither, so the split has to be visible beside it.
            println!(
                "      lanes: {} layer(s) GROUPED (one launch per stage), {} per-expert",
                host_t.grouped, host_t.per_expert
            );
            // The union, which is the number speculation lives or dies by. A
            // batch-1 step gathers `top_k + shared`; anything wider gathers the
            // UNION over its rows, and MoE is the large majority of the bytes a
            // step reads, so this ratio lands almost undiluted on the step time.
            let moe_layers = host_t.grouped + host_t.per_expert;
            if moe_layers > 0 {
                println!(
                    "      experts gathered: {} distinct over {} MoE layer(s) = {:.2} per layer",
                    host_t.expert_slots,
                    moe_layers,
                    host_t.expert_slots as f64 / moe_layers as f64
                );
            }
            let named = host_t.slice + host_t.gather + host_t.enqueue + host_t.drain + host_t.accum;
            println!(
                "      remainder       {:9.1}   (whatever the four above did not cover)",
                ms(t_expert - named)
            );
        }
        // Rule 2, as a number that should be zero and is. Every one of these was
        // scalar f32 arithmetic over the residual stream, on a CPU, between device
        // calls; none of it was control plane. The line stays in the report BECAUSE
        // it reads zero -- a claim of "no host path in the data plane" that nothing
        // measures is a claim that rots.
        println!("    HOST DATA PLANE in the block itself, ms (want: zero):");
        println!("      rms_norm        {:9.1}", ms(t_h_norm));
        println!("      residual adds   {:9.1}", ms(t_h_resid));
        println!(
            "      expert gather   {:9.1}   (a `select` on the device now)",
            0.0
        );
        println!(
            "      expert accum    {:9.1}   (a `select_assign` on the device now)",
            0.0
        );
        println!("      TOTAL           {:9.1}", ms(t_h_norm + t_h_resid));
        let (calls, hits, fileb, hostb, loader_ns) = cp.io_totals();
        let (rb, rn) = cp.resident_bytes();
        println!("  what this ONE pass moved:");
        println!("    loader reads        {calls:8}   answered from RAM {hits:8}");
        println!(
            "    stored bytes        {:8.2} GiB   (what the reads touched, stored precision)",
            fileb as f64 / GIB
        );
        println!(
            "    host f32 bytes      {:8.2} GiB   (what they became after widening)",
            hostb as f64 / GIB
        );
        println!("    seconds in loader   {:8.1}", loader_ns as f64 / 1e9);
        println!(
            "    disk read_bytes     {:8.2} GiB   (/proc/self/io -- page-cache hits are free)",
            (io_read_bytes() - io0) as f64 / GIB
        );
        println!(
            "    resident set        {:8.2} GiB in {rn} weights  (host)",
            rb as f64 / GIB
        );
        println!(
            "    device-resident     {:8.2} GiB in {} shared + {} dense layers",
            ddense.bytes as f64 / GIB,
            ddense.shared.len(),
            ddense.dense.len()
        );
        println!(
            "    device-resident     {:8.2} GiB in {} attention layers",
            dattn_bytes as f64 / GIB,
            layers_dev.len()
        );
        // What the zero-copy seam actually achieved on THIS run, at the seam every
        // expert weight passes through. Printed unconditionally: a seam whose hit
        // rate is only visible behind a flag is a seam nobody checks.
        #[cfg(feature = "inkling-cuda")]
        if let Some(al) = fp4_aliases.as_ref() {
            print!("{}", al.stats().report());
        }
        #[cfg(feature = "inkling-cuda")]
        {
            println!(
                "    device-resident     {:8.2} GiB in {} shared + {} dense layers",
                ddense.bytes as f64 / GIB,
                ddense.shared.len(),
                ddense.dense.len()
            );
        }
        #[cfg(feature = "inkling-cuda")]
        if !layers_dev.is_empty() {
            println!(
                "    device-resident     {:8.2} GiB in {} attention layers",
                dattn_bytes as f64 / GIB,
                layers_dev.len()
            );
        }
        if std::env::var("INK_IOSTATS").is_ok() {
            print!("{}", cp.io_table(28));
        }
        report_align(charged_device_weights);
        mary::models::inkling::bf16gemm::report_hand();
        println!("  elapsed: {:.1}s", started.elapsed().as_secs_f32());

        // Per-position top-5. Uncached, the final pass has recomputed every
        // position, so it reports all of them and earlier passes report nothing.
        // Cached, each pass computes only the positions it was handed and they
        // accumulate -- prefill contributes the prompt, every step one more -- so
        // the two lanes end with the same table over the same sequence, which is
        // what makes the outputs comparable.
        if !kv {
            top_all.clear();
        }
        // A head has no logits to rank -- the tail owns the table, and writes it.
        // Not in the slot lane: this table is indexed by POSITION in one sequence,
        // and a slot batch's rows are b sequences at the same position. Reporting
        // them here would print b top-5 rows against b consecutive positions of
        // slot 0, which is not a thing that happened.
        if (kv || gen_tokens + new_toks.len() > gen_steps) && best_wire.is_none() && !slot_lane {
            // Only the rows the verifier kept. A speculative pass computes logits
            // for rows it then throws away, and ranking those would print a top-5
            // for a position the run never visited -- and index `ids` past its end.
            let valid_to = if verify_rows > 1 || probe_rows > 1 {
                logit_row0 + new_toks.len()
            } else {
                n
            };
            for ti in logit_row0..valid_to {
                let pos = pos0 + ti;
                let row = &logits[(ti - logit_row0) * v..(ti - logit_row0 + 1) * v];
                let top = top5_desc(row);
                println!(
                    "  after token {pos} (id {}): top5 {:?}  logits {:?}",
                    ids[pos],
                    top,
                    top.iter()
                        .map(|&i| (row[i] * 100.0).round() / 100.0)
                        .collect::<Vec<_>>()
                );
                for &i in &top {
                    top_all.push(i as i64);
                }
            }
        }

        if gen_steps > 0 {
            // The cohort is named only when there is more than one, so a
            // single-cohort run's transcript is character for character the one
            // every arm above was compared on.
            let coh_tag = if ncohorts > 1 {
                format!(" cohort {answer_coh}")
            } else {
                String::new()
            };
            println!(
                "  step {step}{coh_tag}: +{new_toks:?}   [pass {:.1}s, total {:.1}s, ctx {}, pass_ms {:.1}]",
                pass.elapsed().as_secs_f32(),
                started.elapsed().as_secs_f32(),
                ids.len(),
                pass.elapsed().as_secs_f64() * 1e3
            );
            // The tail already pushed all but the last, when it answered its peer.
            if !is_tail && !repeat {
                // Not in the slot lane: `new_toks` is one token per SLOT there, not
                // an accepted prefix of one sequence, and extending `ids` with all
                // of them puts seven other sequences into slot 0's stream. It is
                // only a report -- nothing computes off `ids` in that lane -- which
                // is exactly why it read as a plausible context length (4052
                // against 3780) instead of as a failure.
                // The tree lane already extended, above the MTP block, because
                // the drafter reads `ids` and cannot wait for the report.
                if new_toks.len() > 1 && !slot_lane && tree_b == 0 {
                    ids.extend_from_slice(&new_toks[..new_toks.len() - 1]);
                }
                // `ids` is ONE sequence's -- cohort 0's slot 0 -- because every
                // report that indexes a position reads it. Appending a second
                // cohort's token would interleave two sequences into one stream and
                // read as a plausible context length rather than as a failure,
                // which is the same trap the slot batch already sprang once.
                if !slot_lane || (answered && (ncohorts == 1 || answer_coh == 0)) {
                    ids.push(best);
                }
            }
        }
        // The prefill is a different animal -- a 512-row pass against a 1-row one,
        // and the only pass that pays for uploading every resident weight -- so it
        // is excluded from the utilisation summary rather than averaged into it.
        if is_decode {
            acc_send += t_send;
            acc_wait_peer += t_wait_peer;
            acc_recv += t_recv;
            acc_pass += pass.elapsed().as_secs_f64();
            acc_to_reply += t_to_reply;
            acc_steps += 1;
            pass_ms.push((pass.elapsed().as_secs_f64() + t_recv) * 1e3);
            pass_ms_arm.push((dev_plan_now, (pass.elapsed().as_secs_f64() + t_recv) * 1e3));
        }
        // On its OWN line, never folded into the `step N:` line above, because
        // `bench-decode.sh` and `pipe-bench.sh` both parse that line and an
        // instrument that changes what it is measuring is not an instrument.
        // The prefill gets a line too: it is where the anonymous arena's huge
        // pages are decided, and the first decode pass's deltas are read
        // against it. `pass_ms` here is the same quantity the `step N:` line
        // prints, so the two can be joined on it.
        if stepstat_on {
            let now = stepstat::StepStat::sample();
            println!(
                "{}",
                now.line(&stepstat_prev, step, pass.elapsed().as_secs_f64() * 1e3)
            );
            stepstat_prev = now;
        }
        if is_decode && step - prefill_passes >= COLD_DECODE_STEPS {
            warm_wall += pass.elapsed().as_secs_f64() + t_recv;
            warm_steps += 1;
            warm_tokens += if slot_lane && is_decode {
                n
            } else {
                new_toks.len()
            };
        }
        // A slot lane's prefill passes produce a token each and they are not decode
        // tokens: counting them would put b of them in the numerator of a rate
        // whose denominator is decode passes only.
        // By the pass this node COMPUTED, not by the answer it happened to read.
        // The two are the same pass on a tail and on a single-cohort head; on an
        // interleaved head the answer is a pass behind, and counting answers would
        // make the two ends disagree about how many tokens one run produced.
        gen_tokens += if slot_lane && !is_decode {
            0
        } else if slot_lane {
            n
        } else {
            new_toks.len()
        };
        // Tokens, not passes -- and with nothing speculated this fires on exactly
        // the pass the old `for step in 0..=gen_steps` fired on.
        // Passes, in the slot lane. Every arm has to run the same number of decode
        // passes for the per-pass cost to be comparable across b, and a token count
        // would make the run b times shorter at b times the width.
        let done = if slot_lane {
            is_decode && step + 1 - prefill_passes >= gen_steps
        } else {
            gen_tokens > gen_steps
        };
        if done {
            break;
        }
        step += 1;
    }

    // ---- the answers still on the wire ------------------------------------
    //
    // An interleaved head finishes `ncohorts - 1` passes ahead of the tail, so
    // that many answers are still coming. They are read rather than dropped:
    // each is one token per slot of a real cohort, and a run that discarded
    // them would report every cohort but the last a token short -- which is
    // exactly the sort of missing row a contamination check reads as a
    // difference between neighbours.
    if let Some(Pipe::Head(s)) = pipe.as_mut() {
        while let Some(c) = in_flight.pop_front() {
            let toks = recv_toks(s)?;
            if slot_lane && gen_steps > 0 && !repeat {
                let base = c * nslots;
                for (q, tok) in slot_ids[base..base + nslots].iter_mut().zip(toks.iter()) {
                    q.push(*tok);
                }
            }
            println!("  drain cohort {c}: +{toks:?}");
        }
    }

    // ---- how much of the wall clock this node spent waiting for the other --
    if acc_steps > 0 && pipe.is_some() {
        let ms = |v: f64| v * 1e3;
        let wall = acc_pass + acc_recv;
        println!("\n=== pipe utilisation over {acc_steps} decode steps (prefill excluded) ===");
        println!(
            "  role                 : {}",
            if is_head { "head" } else { "tail" }
        );
        println!(
            "  wall in the loop     : {:9.1} ms   ({:.1} ms/step)",
            ms(wall),
            ms(wall) / acc_steps as f64
        );
        if warm_steps > 0 {
            println!(
                "  WARM steps only      : {:9.1} ms   ({:.1} ms/step over {warm_steps} steps, \
                 the first {COLD_DECODE_STEPS} excluded)",
                ms(warm_wall),
                ms(warm_wall) / warm_steps as f64
            );
            if warm_tokens > 0 {
                println!(
                    "  WARM per TOKEN       : {:9.1} ms   ({:.3} tok/s over {warm_tokens} tokens)",
                    ms(warm_wall) / warm_tokens as f64,
                    warm_tokens as f64 / warm_wall
                );
            }
        }
        if is_head {
            // `pass` contains the wait, so compute is what is left of it.
            let compute = acc_pass - acc_wait_peer - acc_send;
            println!(
                "  computing            : {:9.1} ms   {:5.1}%",
                ms(compute),
                100.0 * compute / wall
            );
            println!(
                "  writing to the wire  : {:9.1} ms   {:5.1}%",
                ms(acc_send),
                100.0 * acc_send / wall
            );
            println!(
                "  BLOCKED on the tail  : {:9.1} ms   {:5.1}%",
                ms(acc_wait_peer),
                100.0 * acc_wait_peer / wall
            );
            println!(
                "  per step: compute {:.1} ms, blocked {:.1} ms",
                ms(compute) / acc_steps as f64,
                ms(acc_wait_peer) / acc_steps as f64
            );
        } else {
            println!(
                "  computing            : {:9.1} ms   {:5.1}%",
                ms(acc_pass),
                100.0 * acc_pass / wall
            );
            println!(
                "    of which drafting  : {:9.1} ms   {:5.1}%",
                ms(acc_draft),
                100.0 * acc_draft / wall
            );
            println!(
                "  ANSWERED the head at : {:9.1} ms into its pass ({:.1} ms/step) -- everything\n  \
                 after that (report, drafting) overlaps the head's next pass and the head\n  \
                 never waits for it. Subtract THIS from the head's blocked figure for the wire.",
                ms(acc_to_reply),
                ms(acc_to_reply) / acc_steps as f64
            );
            println!(
                "  BLOCKED on the head  : {:9.1} ms   {:5.1}%",
                ms(acc_recv),
                100.0 * acc_recv / wall
            );
            println!(
                "  per step: compute {:.1} ms, blocked {:.1} ms",
                ms(acc_pass) / acc_steps as f64,
                ms(acc_recv) / acc_steps as f64
            );
        }
        // ---- THE gate ------------------------------------------------------
        //
        // Tokens per second, not milliseconds per pass. Speculation trades more
        // compute for fewer sequential steps, so a per-pass figure is SUPPOSED
        // to get worse; the only number that says whether the trade paid is how
        // much text came out per second of wall clock.
        // The prefill's own token is in `gen_tokens` on the single-sequence
        // lane and is not a decode token; the slot lane never counted it.
        let decode_toks = if slot_lane {
            gen_tokens
        } else {
            gen_tokens.saturating_sub(1)
        };
        let mut sorted = pass_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration"));
        let p50 = if sorted.is_empty() {
            0.0
        } else {
            sorted[sorted.len() / 2]
        };
        println!(
            "  TOKENS/SEC           : {:.3}   ({decode_toks} tokens past the prefill in {:.1} ms \
             of decode wall)",
            decode_toks as f64 / wall,
            ms(wall)
        );
        println!(
            "  per pass, ms         : p50 {p50:.1}, min {:.1}, max {:.1}, mean {:.1}   over {} passes",
            sorted.first().copied().unwrap_or(0.0),
            sorted.last().copied().unwrap_or(0.0),
            ms(wall) / acc_steps as f64,
            sorted.len()
        );
        let tpp = decode_toks as f64 / acc_steps as f64;
        println!("  tokens per pass      : {tpp:.3}");
        if tree_b > 0 {
            let passes: usize = tree_rank_hist.iter().sum();
            let won: usize = tree_rank_hist[1..tree_b].iter().sum();
            println!("  which candidate the verifier took, over {passes} tree passes:");
            for (r, &c) in tree_rank_hist.iter().enumerate() {
                let what = if r == tree_b {
                    "none  ".to_string()
                } else {
                    format!("rank {r}")
                };
                println!(
                    "    {what}: {c:5}   ({:5.1}%){}",
                    100.0 * c as f64 / passes.max(1) as f64,
                    if r > 0 && r < tree_b {
                        "   <- a token the chain would have missed"
                    } else {
                        ""
                    }
                );
            }
            println!(
                "    breadth WON {won} of {passes} passes ({:.1}%); the rest a chain would have                  got too, or not at all",
                100.0 * won as f64 / passes.max(1) as f64
            );
        }
        if spec_k > 0 || tree_b > 0 {
            let sets: usize = spec_hist.iter().sum();
            let lane = if tree_b > 0 {
                format!("INK_TREE={tree_b}")
            } else {
                format!("INK_SPEC={spec_k}")
            };
            println!("  accepted prefix over {sets} verify passes ({lane}):");
            for (l, &c) in spec_hist.iter().enumerate().skip(1) {
                println!(
                    "    {} accepted: {c:5}   ({:5.1}%)",
                    l - 1,
                    100.0 * c as f64 / sets.max(1) as f64
                );
            }
            let mean: f64 = spec_hist
                .iter()
                .enumerate()
                .skip(1)
                .map(|(l, &c)| (l - 1) as f64 * c as f64)
                .sum::<f64>()
                / sets.max(1) as f64;
            println!("    mean {mean:.3} draft tokens accepted per verify pass");
        }
        println!(
            "  loop wall (both ends see the same clock): {:.1} ms",
            ms(loop_started.elapsed().as_secs_f64())
        );
        println!(
            "  the wire itself is the head's BLOCKED figure minus the tail's per-step compute;"
        );
        println!(
            "  neither process can subtract that on its own, so the two reports are read together."
        );
    }

    // ---- what the router arm changed, if anything -------------------------
    //
    // Printed whether or not it found something, with the examined count on
    // every line. A zero that says how many selections it looked at is a
    // measurement; a zero on its own is a claim.
    if router_diff {
        println!(
            "\n=== router selection: {} vs the f32 [rows,hidden] lane ===",
            router_arm.label()
        );
        println!("  layer   examined   set!=   order!=   slots!=   max|dlogit|   max|dweight|");
        let (mut ex, mut sd, mut od, mut sl) = (0usize, 0usize, 0usize, 0usize);
        let (mut ml, mut mw) = (0f32, 0f32);
        for (layer, d) in route_diff.iter().enumerate() {
            if d.examined == 0 {
                continue;
            }
            println!(
                "  {layer:5}   {:8}   {:5}   {:7}   {:7}   {:11.3e}   {:12.3e}",
                d.examined,
                d.set_differs,
                d.order_differs,
                d.slots_differ,
                d.max_abs_logit,
                d.max_abs_weight
            );
            ex += d.examined;
            sd += d.set_differs;
            od += d.order_differs;
            sl += d.slots_differ;
            ml = ml.max(d.max_abs_logit);
            mw = mw.max(d.max_abs_weight);
        }
        if ex == 0 {
            println!(
                "  nothing examined: this node's slice has no MoE layer, so there was no router to compare."
            );
        } else {
            println!("  TOTAL   {ex:8}   {sd:5}   {od:7}   {sl:7}   {ml:11.3e}   {mw:12.3e}");
            println!(
                "  {:.4}% of {ex} selections chose a different SET of experts; {:.4}% reordered one.",
                100.0 * sd as f64 / ex as f64,
                100.0 * od as f64 / ex as f64,
            );
            println!(
                "  {sl} of {} expert slots named a different expert.",
                ex * t.num_experts_per_tok
            );
        }
    }

    // ---- the MTP experiment's result --------------------------------------
    // Reported whether or not it looks good. A composition that drafts badly is
    // the measurement working, not the run failing, and burying a 0% would make
    // the next window repeat the experiment.
    if mtp_k > 0 {
        println!("\n=== MTP acceptance, concat {} ===", mtp_order.name());
        let scored: usize = mtp_seen.iter().sum();
        if scored == 0 {
            println!("  nothing scored -- every draft named a step past the end of the run.");
            println!("  raise INK_GEN above INK_MTP so drafts have a token to be judged against.");
        } else {
            for d in 0..mtp_k {
                if mtp_seen[d] == 0 {
                    println!("  depth {}: never scored", d + 1);
                    continue;
                }
                let (c_lo, c_hi) = wilson95(mtp_hits[d], mtp_seen[d]);
                println!(
                    "  depth {}: {:5}/{:<5} = {:5.1}%   95% CI [{:5.1}%, {:5.1}%]",
                    d + 1,
                    mtp_hits[d],
                    mtp_seen[d],
                    100.0 * mtp_hits[d] as f64 / mtp_seen[d] as f64,
                    100.0 * c_lo,
                    100.0 * c_hi
                );
            }
            let hits: usize = mtp_hits.iter().sum();
            let (c_lo, c_hi) = wilson95(hits, scored);
            println!(
                "  pooled : {hits:5}/{scored:<5} = {:5.1}%   95% CI [{:5.1}%, {:5.1}%]",
                100.0 * hits as f64 / scored as f64,
                100.0 * c_lo,
                100.0 * c_hi
            );
            // Pooling across depths is a number to read carefully: it averages a
            // depth-1 rate with a depth-4 one, so it moves when INK_MTP moves and
            // is a fact about the configuration as much as about the model. It is
            // kept because a pooled ZERO settles the concat question at a glance;
            // the per-depth rows are what a speculation decision reads.

            // What an accept-and-skip loop would actually keep. Only draft sets
            // every depth of which got scored count — a set whose deeper heads
            // named steps past the end of the run has no prefix length yet, and
            // counting it as "prefix ended here" would bias the mean down by
            // exactly the runs that ran out of tokens.
            let complete: Vec<&Vec<Option<bool>>> = mtp_issued
                .values()
                .filter(|v| v.iter().all(|s| s.is_some()))
                .collect();
            if !complete.is_empty() {
                let sets = complete.len();
                let mut hist = vec![0usize; mtp_k + 1];
                for v in &complete {
                    let mut run = 0usize;
                    while run < v.len() && v[run] == Some(true) {
                        run += 1;
                    }
                    hist[run] += 1;
                }
                let mean: f64 = hist
                    .iter()
                    .enumerate()
                    .map(|(l, &c)| l as f64 * c as f64)
                    .sum::<f64>()
                    / sets as f64;
                println!("  accepted prefix, over {sets} fully-scored draft sets:");
                for (l, &c) in hist.iter().enumerate() {
                    println!(
                        "    {l} accepted: {c:5}   ({:5.1}%)",
                        100.0 * c as f64 / sets as f64
                    );
                }
                println!("    mean {mean:.3} draft tokens accepted per verify pass");
            }
            // What the number MEANS, said here rather than left to the reader,
            // because the whole point of this experiment is that a low rate is
            // evidence about the composition and not about the model.
            println!(
                "  {}",
                if hits * 4 >= scored * 3 {
                    "high -- this reading of mtp_hidden_states_first computes something the \
                     stack agrees with"
                } else if hits == 0 {
                    "zero -- either this concat order is wrong or the wrapper composes some \
                     other way; run INK_MTP_ORDER with the other order before concluding"
                } else {
                    "partial -- better than chance over a 200k vocabulary, so the composition is \
                     close to right; the shortfall is what to explain next"
                }
            );
        }
        // ---- the same drafts, under two rules that are not equality --------
        //
        // Reported beside the exact-match rate rather than instead of it,
        // because they answer different questions and the gap between them IS
        // the finding: A is "is the draft the argmax", B and C are "is the
        // draft acceptable". A greedy loop can only spend A; a loop that
        // samples can spend B or C, and sampling is not a quality concession --
        // the residual-resampling step makes the output distribution exactly
        // the target's.
        if mtp_prob && mtp_prob_n.iter().any(|&n| n > 0) {
            println!("\n=== MTP acceptance, three rules, same drafts ===");
            println!("  A  exact argmax match                 -- greedy speculation");
            println!("  B  greedy draft, sampled target       -- E[p_target(draft)]");
            println!("  C  sampled draft, sampled target      -- E[1 - TV(p_draft, p_target)]");
            println!("  depth      n        A        B        C");
            for d in 0..mtp_k {
                if mtp_prob_n[d] == 0 {
                    println!("  {:5}      0   never scored", d + 1);
                    continue;
                }
                let nn = mtp_prob_n[d] as f64;
                println!(
                    "  {:5}  {:5}   {:5.1}%   {:5.1}%   {:5.1}%",
                    d + 1,
                    mtp_prob_n[d],
                    100.0 * mtp_hits[d] as f64 / mtp_seen[d].max(1) as f64,
                    100.0 * mtp_b_sum[d] / nn,
                    100.0 * mtp_c_sum[d] / nn,
                );
            }
            // The expected LEADING RUN, which is what converts to tok/s. Under
            // a stochastic rule a draft set has no realised prefix -- it has a
            // distribution over prefixes -- so the expectation is taken
            // directly: E[prefix] = sum_j prod_{i<=j} q_i.
            let complete_q: Vec<&Vec<Option<(f64, f64)>>> = mtp_issued_q
                .values()
                .filter(|v| v.iter().all(|s| s.is_some()))
                .collect();
            if !complete_q.is_empty() {
                let sets = complete_q.len();
                let (mut eb, mut ec) = (0f64, 0f64);
                for v in &complete_q {
                    let (mut pb, mut pc) = (1f64, 1f64);
                    for slot in v.iter() {
                        let (b, c) = slot.expect("filtered to fully-scored sets");
                        pb *= b;
                        pc *= c;
                        eb += pb;
                        ec += pc;
                    }
                }
                println!("  expected accepted prefix over {sets} fully-scored draft sets:");
                println!(
                    "    rule B: {:.3} draft tokens per verify pass",
                    eb / sets as f64
                );
                println!(
                    "    rule C: {:.3} draft tokens per verify pass",
                    ec / sets as f64
                );
            }
            println!(
                "  depths past 1 are CONDITIONAL on the greedy chain: head d was fed the ARGMAX\n  \
                 draft of head d-1, which a sampled loop would not always have drawn."
            );

            // ---- acceptance against the draft head's OWN confidence --------
            //
            // A loop that speculates unconditionally pays the width premium on
            // every step, including the ones where the head was guessing. It
            // does not have to: the head's top-1 mass is available BEFORE the
            // verify pass is issued, for free, so the loop can decline. What
            // that is worth is `(1 + P*a) / (P*c + (1 - P))` against the
            // unconditional `(1 + a_all) / c`, and both need the measured
            // width cost, which is a property of the machine and not of this
            // run -- `INK_SPEC_C2` carries it in.
            //
            // **1.571, and it is now measured where it is spent.** The old
            // default was 1.492, taken on the UNCACHED lane with `INK_REPEAT`,
            // because until `INK_SPEC` existed there was no cached multi-row
            // pass to time. There is one now, and the cached lane is worse:
            // 100-token runs on the two-node pipe, warm p50 of the whole cycle,
            // a five-token prompt, two runs per arm,
            //
            //     INK_SPEC=0  w=1  127.2 ms      c = 1.000
            //     INK_SPEC=1  w=2  199.9 ms      c = 1.571
            //     INK_SPEC=2  w=3  236.8 ms      c = 1.862
            //     INK_SPEC=3  w=4  263.0 ms      c = 2.068
            //
            // against 1.000 / 1.332 / 1.434 / ~1.52 uncached, and against
            // 1.000 / 1.463 / 1.747 on a 3732-token document, where a wider
            // pass is a smaller fraction of a slower one.
            //
            // A STEP, not a slope: the penalty is nearly all at the second row
            // and flat after it. Two things make it. `gemv plane par` requires
            // m == 1, a cached decode step is weight-streaming-bound, and that
            // lane is the only one reaching the roofline -- losing it costs
            // 1.33x at m == 1 alone (70.1/73.0 ms against 96.2/93.3, measured
            // in `bf16gemm`). The four short convolutions a layer ran
            // slice-built above one row -- 49 of the 52 ms the second row used
            // to add -- and they now run as one kernel each, which is where
            // 1.613 became 1.571.
            let c2: f64 = std::env::var("INK_SPEC_C2")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.571);
            if !mtp_conf[0].is_empty() {
                println!("\n=== depth-1 acceptance against the draft head's own confidence ===");
                println!("  c(2) = {c2:.3} (INK_SPEC_C2); a k=1 loop that always speculates:");
                let all_n = mtp_conf[0].len();
                let all_hit = mtp_conf[0].iter().filter(|(_, h)| *h).count();
                let a_all = all_hit as f64 / all_n as f64;
                println!(
                    "    a = {:.3} over {all_n} events  ->  (1 + a) / c(2) = {:.3}x",
                    a_all,
                    (1.0 + a_all) / c2
                );
                println!("  gated on p_draft(top1) >= tau, paying c(1) when it declines:");
                println!("    tau     P(spec)   a|spec    (1 + P*a) / (P*c2 + 1 - P)");
                for tau in [0.0f32, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95] {
                    let sel: Vec<&(f32, bool)> =
                        mtp_conf[0].iter().filter(|(p, _)| *p >= tau).collect();
                    if sel.is_empty() {
                        println!("    {tau:4.2}    0        --        --");
                        continue;
                    }
                    let pp = sel.len() as f64 / all_n as f64;
                    let a = sel.iter().filter(|(_, h)| *h).count() as f64 / sel.len() as f64;
                    println!(
                        "    {tau:4.2}    {:.3}    {:.3}     {:.3}x",
                        pp,
                        a,
                        (1.0 + pp * a) / (pp * c2 + 1.0 - pp)
                    );
                }
                println!("  a threshold that beats the tau=0 row is a loop worth gating; one that");
                println!("  does not means confidence and correctness are not linked here.");
            }
        }
    }

    // A zero-length batch is the head saying it is done, so the tail's loop
    // ends on a read rather than blocking forever on a peer that has exited.
    if let Some(Pipe::Head(s)) = pipe.as_mut() {
        let _ = send_stream(s, 0, 0, 0, &[]);
    }

    // The head has no top-5 table -- the tail computed the logits and wrote it.
    // Writing an empty file here would look like a run that produced nothing.
    if is_head {
        println!("  head done; the tail wrote the top-5 table. ids {ids:?}");
        return Ok(());
    }
    let mut bytes = Vec::new();
    for i in top_all {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    std::fs::write(&out_path, &bytes)?;
    // ---- the approximate head, against the exact one -----------------------
    //
    // Printed at the end and not per pass because recall is a rate: a single
    // step either agreed or did not, and the interesting number is over a
    // generation. The mean top1-top2 gap prints beside it so a disagreement
    // rate can be read at all -- losing a 0.001-logit tie and losing a
    // 2-logit one are not the same event, and the rate alone cannot tell them
    // apart.
    if let Some(r) = mary::models::inkling::annhead::verify_report() {
        println!("\n=== approximate head ===");
        println!("  {r}");
    }

    // ---- the device row plan: the fault flag and the two arms --------------
    //
    // Outside the pipe report on purpose. Both of these are statements about
    // the passes THIS process ran, and a single-node decode -- which is what a
    // measurement of this lane looks like -- has no pipe at all.
    if let Some(dr) = devroute.as_ref() {
        // The whole run's fault flag, in one read. It is raised by the kernel
        // and never by the host, so a run that ends with it clear is a run in
        // which no layer saw a non-finite router logit and no layer picked one
        // expert twice. The host lane panics at the layer instead; this trades
        // that for not reading anything per layer.
        let f = fp4_client
            .read_one(dr.fault.clone())
            .expect("read the device plan fault");
        let v = u32::from_le_bytes(f[..4].try_into().expect("a u32 came back"));
        anyhow::ensure!(
            v == 0,
            "INK_DEV_PLAN raised its fault flag ({v:#x}) during the run: {}. The flag is \
             run-scoped and names the LAST fault, not the first.",
            if v == 0xd {
                "two of a token's top-k picks were the same expert"
            } else {
                "a router logit was NaN or infinite"
            }
        );
    }
    if pass_ms_arm.iter().any(|(d, _)| *d) && pass_ms_arm.iter().any(|(d, _)| !*d) {
        let tpp = if acc_steps > 0 {
            (if slot_lane {
                gen_tokens
            } else {
                gen_tokens.saturating_sub(1)
            }) as f64
                / acc_steps as f64
        } else {
            0.0
        };
        // p10 and p90 beside the median and not instead of it: if the arms'
        // spreads overlap, the medians are not a difference, and a mean over a
        // bimodal set is not anything.
        let q = |v: &mut Vec<f64>, f: f64| -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration"));
            v[((v.len() as f64 - 1.0) * f).round() as usize]
        };
        let mut host_p: Vec<f64> = pass_ms_arm
            .iter()
            .filter(|(d, _)| !*d)
            .map(|(_, m)| *m)
            .collect();
        let mut dev_p: Vec<f64> = pass_ms_arm
            .iter()
            .filter(|(d, _)| *d)
            .map(|(_, m)| *m)
            .collect();
        let (hn_, dn_) = (host_p.len(), dev_p.len());
        let (h50, h10, h90) = (
            q(&mut host_p, 0.5),
            q(&mut host_p, 0.1),
            q(&mut host_p, 0.9),
        );
        let (d50, d10, d90) = (q(&mut dev_p, 0.5), q(&mut dev_p, 0.1), q(&mut dev_p, 0.9));
        println!("\n=== INK_DEV_PLAN, both arms interleaved in this ONE process ===");
        println!(
            "  plan on the HOST   : p50 {h50:7.1} ms   p10 {h10:7.1}  p90 {h90:7.1}   \
             {:.3} tok/s at p50   over {hn_} decode passes",
            if h50 > 0.0 { tpp / (h50 / 1e3) } else { 0.0 }
        );
        println!(
            "  plan on the DEVICE : p50 {d50:7.1} ms   p10 {d10:7.1}  p90 {d90:7.1}   \
             {:.3} tok/s at p50   over {dn_} decode passes",
            if d50 > 0.0 { tpp / (d50 / 1e3) } else { 0.0 }
        );
        println!(
            "  device against host: {:+.1} ms a pass at p50, {:+.1}% tok/s   (the arms {})",
            d50 - h50,
            100.0 * (h50 / d50 - 1.0),
            if d90 < h10 || h90 < d10 {
                "do not overlap between p10 and p90"
            } else {
                "OVERLAP between p10 and p90 -- read the medians with that in mind"
            }
        );
    }
    println!("  wrote top-5 ids per position to {}", out_path.display());
    Ok(())
}

#[cfg(test)]
mod pipe_tests {
    use super::{pipe_accept, pipe_connect};
    use std::net::TcpListener;
    use std::time::{Duration, Instant};

    /// A port nothing is bound to, taken by binding one and dropping it.
    ///
    /// Not a hardcoded number: these run in parallel with whatever else the
    /// test binary is doing, and a fixed port is a test that fails on somebody
    /// else's machine for a reason that has nothing to do with the code.
    fn free_port() -> u16 {
        let l = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        l.local_addr().expect("bound").port()
    }

    /// The rank-order hazard itself: the head reaches the rendezvous while the
    /// tail is still loading, and connects anyway once the tail arrives.
    ///
    /// This is the whole fix. Before it, the `connect` below happened once,
    /// got `ECONNREFUSED` from a port with no listener, and killed the run —
    /// so the two commands had to be started tail-first and far enough apart,
    /// a rule that lived only in the launch scripts and in the reference
    /// implementation's prose.
    #[test]
    fn the_head_waits_for_a_tail_that_is_not_listening_yet() {
        let port = free_port();
        let addr = format!("127.0.0.1:{port}");
        let late = addr.clone();
        let bind_at = Instant::now() + Duration::from_millis(400);
        let tail = std::thread::spawn(move || {
            std::thread::sleep(bind_at.saturating_duration_since(Instant::now()));
            let l = TcpListener::bind(&late).expect("binding late");
            pipe_accept(&l, &late, Duration::from_secs(10)).expect("the head should arrive")
        });
        let t0 = Instant::now();
        let sock = pipe_connect(&addr, Duration::from_secs(10))
            .expect("the head should have waited for the late tail");
        assert!(
            t0.elapsed() >= Duration::from_millis(300),
            "the connect did not actually wait: {:?}",
            t0.elapsed()
        );
        let (peer, _) = tail.join().expect("the tail thread");
        drop((sock, peer));
    }

    /// And it is a BOUND, not an infinite wait: a port nobody will ever listen
    /// on still fails, naming the end that was waiting and how to change it.
    #[test]
    fn a_head_with_no_tail_fails_legibly_rather_than_forever() {
        let addr = format!("127.0.0.1:{}", free_port());
        let t0 = Instant::now();
        let err = pipe_connect(&addr, Duration::from_millis(400))
            .expect_err("nothing is listening there");
        let msg = format!("{err:#}");
        assert!(t0.elapsed() >= Duration::from_millis(400), "gave up early");
        assert!(t0.elapsed() < Duration::from_secs(10), "overshot the bound");
        assert!(msg.contains("the head waited"), "unhelpful: {msg}");
        assert!(msg.contains(&addr), "does not name the address: {msg}");
        assert!(msg.contains("INK_PIPE_WAIT"), "no way out named: {msg}");
    }

    /// The other end of the same hazard: a head that died while loading its
    /// weights used to leave the tail holding a GPU forever.
    #[test]
    fn a_tail_with_no_head_fails_legibly_rather_than_forever() {
        let l = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let addr = l.local_addr().expect("bound").to_string();
        let t0 = Instant::now();
        let err =
            pipe_accept(&l, &addr, Duration::from_millis(400)).expect_err("nobody will connect");
        let msg = format!("{err:#}");
        assert!(t0.elapsed() >= Duration::from_millis(400), "gave up early");
        assert!(t0.elapsed() < Duration::from_secs(10), "overshot the bound");
        assert!(msg.contains("the tail listened"), "unhelpful: {msg}");
        assert!(msg.contains(&addr), "does not name the address: {msg}");
        assert!(msg.contains("INK_PIPE_WAIT"), "no way out named: {msg}");
    }
}

#[cfg(test)]
mod ann_temp_tests {
    use super::*;

    /// [`normals`] really is standard normal, and really does depend on the step.
    ///
    /// Worth a test because `INK_TEMP` claims a PRECISE calibration — a
    /// temperature in logit units times `pi/sqrt(6)` divided by the mean row
    /// norm — and every term of that is exact arithmetic on the assumption that
    /// what comes out of here has unit variance. A Box-Muller with a dropped
    /// factor of two would still look like noise, still produce fluent text, and
    /// silently make every stated temperature wrong by 1.41x. Nothing downstream
    /// could catch that; this can.
    #[test]
    fn the_query_noise_is_standard_normal_and_moves_with_the_step() {
        let n = 200_000;
        let v = normals(0x5EED_1107, 7, n);
        assert_eq!(v.len(), n);
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
        let var = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / n as f64 - mean * mean;
        // 4 standard errors at n = 200k: the mean's is 1/sqrt(n) = 0.0022 and
        // the variance's is sqrt(2/n) = 0.0032. Loose enough not to flake,
        // tight enough that a factor of sqrt(2) is nowhere near it.
        assert!(mean.abs() < 0.01, "mean {mean} is not zero");
        assert!((var - 1.0).abs() < 0.02, "variance {var} is not one");
        // Both halves of each Box-Muller pair must be used and used correctly;
        // a bug that returned the cosine twice would pass the moments above.
        let odd: f64 = v
            .iter()
            .skip(1)
            .step_by(2)
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            / (n / 2) as f64;
        assert!((odd - 1.0).abs() < 0.03, "the sine half has variance {odd}");

        // Counter-based: a different step is a different draw, and the SAME
        // step is the same draw. Both directions, because a generator that
        // ignored `step` would give a reproducible run that never re-rolls the
        // sketch's error -- which is the entire reason the noise is on the
        // query.
        let a = normals(0x5EED_1107, 7, 64);
        let b = normals(0x5EED_1107, 8, 64);
        let c = normals(0x5EED_1107, 7, 64);
        assert_eq!(a, c, "the same (seed, step) drew differently");
        assert_ne!(a, b, "consecutive steps drew the same noise");
    }
}

#[cfg(test)]
mod top5_tests {
    use super::top5_desc;

    /// The reference this replaced, spelled out: rank every index by value,
    /// descending, ties by the smaller index. Written here rather than reused
    /// from the file because a test that shares an implementation with the
    /// thing it checks checks nothing.
    fn reference(row: &[f32]) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(5);
        idx
    }

    #[test]
    fn the_five_slot_sweep_agrees_with_a_full_sort_on_random_rows() {
        // A real vocabulary width, because the failure mode worth catching is a
        // truncation bug that a ten-element row cannot express.
        let v = 200_058usize;
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut row = vec![0f32; v];
        for x in row.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *x = ((state >> 40) as f32 / 1024.0) - 12.0;
        }
        assert_eq!(top5_desc(&row), reference(&row));
    }

    /// The case the old `sort_unstable` left undefined and this one pins: a row
    /// flat enough that the top five are all ties.
    #[test]
    fn ties_go_to_the_smaller_index() {
        let mut row = vec![-1.0f32; 100];
        for i in [7, 3, 40, 11, 90, 2] {
            row[i] = 5.0;
        }
        assert_eq!(top5_desc(&row), vec![2, 3, 7, 11, 40]);
        // Every element equal: the answer is the first five positions, and it
        // is the same on every run.
        assert_eq!(top5_desc(&vec![0.5f32; 50]), vec![0, 1, 2, 3, 4]);
    }

    /// A row narrower than five, and one exactly five wide. The old code
    /// indexed `idx[..5]` and would have panicked on the first.
    #[test]
    fn a_short_row_returns_what_it_has() {
        assert_eq!(top5_desc(&[1.0, 3.0, 2.0]), vec![1, 2, 0]);
        assert_eq!(top5_desc(&[]), Vec::<usize>::new());
        assert_eq!(top5_desc(&[1.0, 2.0, 3.0, 4.0, 5.0]), vec![4, 3, 2, 1, 0]);
    }

    /// A NaN in the row is skipped rather than panicking. The host argmax
    /// beside this already skips them -- `val > row[b]` is false for NaN -- so
    /// a report that killed the run where the token pick did not was the odd
    /// one out.
    #[test]
    fn a_nan_does_not_kill_the_report() {
        let row = [1.0f32, f32::NAN, 9.0, 4.0, f32::NAN, 7.0, 2.0];
        assert_eq!(top5_desc(&row), vec![2, 5, 3, 6, 0]);
    }
}
