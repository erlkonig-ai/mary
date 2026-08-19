//! A real forward pass of Inkling-Small across a CLUSTER, on the device.
//!
//! Every gate ends with the same disclaimer: the checkpoint-name to module
//! mapping is authored on both sides, so a shared misreading would pass. This
//! is the check that can settle it. Coherent continuations cannot come out of
//! a wrong mapping — a transposed projection or a swapped gate/up half
//! produces noise, not English.
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
//! so an even 21/21 split is a lopsided 85/71 GiB one). The 66-layer sibling
//! wants five to seven. The rule is that no node runs the whole stack.
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
//! ## What batching would still need
//!
//! [`dev_lane::attention_steps`] takes ONE cache and `pos0`, puts row `i` at
//! `pos0 + i`, and masks row `i` against every earlier row of the batch -- it
//! is contiguous and causal by construction, which is what a speculative batch
//! is and what `b` independent sequences are not. Real batching wants `b`
//! caches, `b` positions and a block-diagonal mask instead of a causal one. The
//! rest of the layer -- every projection, the router, the expert gather, the
//! unembedding -- takes `[b, hidden]` unchanged, which is why the cost above is
//! the right price for a thing that does not exist yet.
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
//! The pile, positionally, on branch `INK_PILE_BRANCH` (default `inkling`). All
//! of it: the weights AND `config.json` AND the chat template, so a run reads
//! nothing off a checkpoint directory and there is no path it could.
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
use std::time::Instant;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::budget;
use mary::models::inkling::fatal;
use mary::models::inkling::block::{rms_norm, route_from_logits, Routing};
use mary::models::inkling::config::{AttnKind, InklingConfig};
use mary::models::inkling::load::{split_gate_up, Held};
use mary::models::inkling::bf16gemm::Bf16W;
use mary::models::inkling::pile::Elem;
use mary::models::inkling::source::Weights;
use mary::models::inkling::mtp::{
    mtp_block, mtp_block_prefill, mtp_block_step, Concat as MtpConcat, MtpCache, MtpHead,
};
use mary::models::inkling::layer::{LayerMlp, LayerWeights};
use mary::models::inkling::stack::{embed_and_norm_bf16, embed_row_bf16};

/// One gibibyte, as the divisor every byte count here is printed against.
const GIB: f64 = (1u64 << 30) as f64;

/// How many decode steps are COLD, and excluded from the warm rate.
///
/// Every kernel shape a decode step reaches is compiled on first use, and the
/// first two steps of a pipe run pay 4.54 s and 0.55 s of that against a 127 ms
/// median. Two is not a fit: it is how many steps report a pass an order of
/// magnitude off the median, and the third is already within 6% of it.
const COLD_DECODE_STEPS: usize = 2;

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
fn send_stream(s: &mut TcpStream, n: usize, pos0: usize, x: &[f32]) -> Result<()> {
    let mut b = Vec::with_capacity(16 + x.len() * 4);
    b.extend_from_slice(&(n as u64).to_le_bytes());
    b.extend_from_slice(&(pos0 as u64).to_le_bytes());
    for v in x {
        b.extend_from_slice(&v.to_le_bytes());
    }
    s.write_all(&b)?;
    s.flush()?;
    Ok(())
}

/// The other side of [`send_stream`]. `None` when the peer is done.
fn recv_stream(s: &mut TcpStream, h: usize) -> Result<Option<(usize, usize, Vec<f32>)>> {
    let mut hdr = [0u8; 16];
    if s.read_exact(&mut hdr).is_err() {
        return Ok(None);
    }
    let n = u64::from_le_bytes(hdr[..8].try_into().unwrap()) as usize;
    let pos0 = u64::from_le_bytes(hdr[8..].try_into().unwrap()) as usize;
    if n == 0 {
        return Ok(None);
    }
    let mut buf = vec![0u8; n * h * 4];
    s.read_exact(&mut buf)?;
    let x = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(Some((n, pos0, x)))
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
    s.read_exact(&mut buf).context("the tail closed mid-answer")?;
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

/// Move a host `[rows, cols]` matrix to the device, consuming it.
///
/// Takes the `Vec` by value on purpose: the dense `w13` is 537 MB at f32 and a
/// borrowing helper would hold two copies of it at once.
fn up2<B: Backend>(v: Vec<f32>, rows: usize, cols: usize, dev: &B::Device) -> BT<B, 2> {
    assert_eq!(v.len(), rows * cols, "{} values are not [{rows}, {cols}]", v.len());
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
    t.into_data().convert::<f32>().to_vec::<f32>().expect("device readback")
}

/// A device tensor of this run's backend, named once so the residency types
/// below do not have to repeat it.
type T2 = burn::tensor::Tensor<Bk, 2>;

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
    gate_up: Bf16W,
    down: Vec<Bf16W>,
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
        if self.local {
            Some(sliding)
        } else {
            None
        }
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
fn bind_bf16(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Bf16W {
    assert_eq!(bytes.len(), rows * cols * 2, "{rows}x{cols} BF16 is not {} bytes", bytes.len());
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
        && std::env::var("INK_ALIGN_COPY").map(|v| v == "1").unwrap_or(false);
    let h = match aliases {
        Some(al) if !copy_to_align => al.slice_or_copy(client, bytes),
        _ => client.create_from_slice(bytes),
    };
    Bf16W { h, n: rows, k: cols, align: if copy_to_align { 16 } else { align } }
}

/// How every plain-BF16 weight is aligned, and how much of the model that is.
///
/// The tuned matmul picks its load width from the SHAPE and never from the
/// pointer, so a `[4096, 4096]` operand gets 16-byte loads whatever address it
/// sits at. The aliasing seam only promises 4 (the checkpoint packs tensors
/// back to back and puts the expert slabs at 4 mod 16), so this counts what the
/// dense lane actually gets before anything is decided on the strength of it.
static ALIGN: [core::sync::atomic::AtomicU64; 8] = [const { core::sync::atomic::AtomicU64::new(0) }; 8];

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
fn report_align() {
    use core::sync::atomic::Ordering::Relaxed;
    let a: Vec<u64> = ALIGN.iter().map(|c| c.load(Relaxed)).collect();
    let n = a[0] + a[1] + a[2] + a[3];
    if n == 0 {
        return;
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
                &fused.bytes, n_shared, inter, h, halved, 2,
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
            let mut sd = SharedOnDevice {
                gate_up: bind_bf16(client, aliases, &gu, 2 * n_shared * inter, h),
                down: Vec::new(),
            };
            for e in 0..n_shared {
                // `w2` is NOT de-interleaved, so this one is a view of the pile
                // and aliases outright.
                sd.down.push(bind_bf16(
                    client, aliases, &d.bytes[e * per_d..(e + 1) * per_d], h, inter,
                ));
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
    let g = dev_lane::linear_bf16(x.clone(), &w.0);
    let u = dev_lane::linear_bf16(x, &w.1);
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
fn mtp_input_dev(hidden: T2, embeds: T2, w: &MtpDev, eps: f64, order: MtpConcat) -> T2 {
    let hn = dev_lane::rms_norm(hidden, w.hidden_norm.clone(), eps);
    let en = dev_lane::rms_norm(embeds, w.embed_norm.clone(), eps);
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
    (out, MtpDevCache { attn, attn_sconv: ahist, mlp_sconv: mhist })
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
    assert_eq!(gammas.len(), n * n_shared, "{} gammas for {n} tokens", gammas.len());
    // ONE projection for every gate and every up in the layer. See
    // [`SharedOnDevice`] for why: four GEMMs against the same activation are
    // four grids of 256 cubes, and this is one of 1024.
    let inter = sw.gate_up.n / (2 * n_shared);
    let gu = dev_lane::linear_bf16(x, &sw.gate_up);
    let mut out: Option<T2> = None;
    for s in 0..n_shared {
        let g = gu.clone().slice([0..n, s * inter..(s + 1) * inter]);
        let u = gu.clone().slice([0..n, (n_shared + s) * inter..(n_shared + s + 1) * inter]);
        let col: Vec<f32> = (0..n).map(|tk| gammas[tk * n_shared + s]).collect();
        let gam = BT::<Bk, 2>::from_data(BTD::new(col, [n, 1]), dev);
        let c = dev_lane::linear_bf16(dev_lane::silu(g) * u * gam, &sw.down[s]);
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
        let differ = a.experts.iter().zip(&b.experts).filter(|(x, y)| x != y).count();
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
    assert_eq!(v.len(), n * cols, "{} values are not [{n}, {cols}]", v.len());
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
    assert_eq!(v.len(), rows * cols, "{} values are not [{rows}, {cols}]", v.len());
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
    Bf16 {
        w: Bf16W,
        /// Real rows, `n_routed + n_shared`, before the pad.
        rows: usize,
    },
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
    let mode = std::env::var("INK_GROUPED").unwrap_or_else(|_| "1".to_string());
    if mode != "0" {
        if let Some(al) = aliases {
            if let Some(acc) =
                grouped_experts_fp4(src, al, client, dev, prefix, by_expert, hn, n, h, inter, host)?
            {
                if mode == "2" {
                    let reference = per_expert_fp4(
                        src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host,
                    )?;
                    report_ab(prefix, &acc, &reference, h);
                }
                host.grouped += 1;
                return Ok(acc);
            }
        }
    }
    host.per_expert += 1;
    per_expert_fp4(src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host)
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
    use mary::models::inkling::fp4gemm::gate_up_silu_launch;
    use mary::models::inkling::fp4quant::quantize_nvfp4;
    use mary::models::inkling::moegroup::{
        fp4_linear_grouped_launch, gather_grouped, scatter_weighted, BlockPlanDev, RowPlan,
    };
    use mary::models::inkling::seam::{handle_of, tensor_of};

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
    let m_total = plan.m_total();
    let hn_h = handle_of(hn.clone());
    let h_rowtok = client.create_from_slice(bytes_of(&plan.row_tok));
    let h_rowwgt = client.create_from_slice(bytes_of(&plan.row_wgt));
    let blk = BlockPlanDev {
        slot: client.create_from_slice(bytes_of(&plan.blk_slot)),
        tile0: client.create_from_slice(bytes_of(&plan.blk_tile0)),
        cnt: client.create_from_slice(bytes_of(&plan.blk_cnt)),
        blocks: plan.blk_slot.len(),
        planes: RowPlan::planes(),
    };
    let h_off13 = client.create_from_slice(bytes_of(&off13));
    let h_off2 = client.create_from_slice(bytes_of(&off2));
    let h_sc13 = client.create_from_slice(bytes_of(&sc13));
    let h_sc2 = client.create_from_slice(bytes_of(&sc2));
    let h_tokrows = client.create_from_slice(bytes_of(&plan.tok_rows));
    let h_tokcnt = client.create_from_slice(bytes_of(&plan.tok_cnt));
    let x_h = gather_grouped(client, &hn_h, &h_rowtok, n, m_total, h);
    host.gather += t_g.elapsed().as_secs_f64();

    let t_w = Instant::now();
    let (a, asc) = quantize_nvfp4(client, &x_h, m_total, h);
    let both = fp4_linear_grouped_launch(
        client, &a, &asc, &wmap, wmap_bytes, &blk, &h_off13, &h_sc13, slots, m_total, h,
        2 * inter,
    );
    let act_h = gate_up_silu_launch(client, &both, m_total, inter);
    let (a2, asc2) = quantize_nvfp4(client, &act_h, m_total, inter);
    let y_h = fp4_linear_grouped_launch(
        client, &a2, &asc2, &wmap, wmap_bytes, &blk, &h_off2, &h_sc2, slots, m_total, inter, h,
    );
    host.enqueue += t_w.elapsed().as_secs_f64();

    let t_c = Instant::now();
    let acc_h = scatter_weighted(
        client, &y_h, &h_rowwgt, &h_tokrows, &h_tokcnt, &h_rowtok, m_total, n, h, plan.kmax,
    );
    let acc = tensor_of(client.clone(), dev.clone(), acc_h, n, h);
    host.accum += t_c.elapsed().as_secs_f64();

    Ok(Some(acc))
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
    use mary::models::inkling::fp4gemm::{fp4_linear_launch, gate_up_silu_launch, MTILE};
    use mary::models::inkling::fp4quant::quantize_nvfp4;
    use mary::models::inkling::pad::gather_rows_pad;
    use mary::models::inkling::seam::{handle_of, int_handle_of, tensor_of};

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
    let hn_h = handle_of(hn.clone());

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

        let (b, bsc) = (bind(&w13.codes), bind(&w13.scales));
        let both = fp4_linear_launch(client, &a, &asc, &b, &bsc, m_pad, h, 2 * inter, w13.scale2);

        let act_h = gate_up_silu_launch(client, &both, m_pad, inter);
        let (a2, asc2) = quantize_nvfp4(client, &act_h, m_pad, inter);

        let (b2, bsc2) = (bind(&w2.codes), bind(&w2.scales));
        let y_h = fp4_linear_launch(client, &a2, &asc2, &b2, &bsc2, m_pad, inter, h, w2.scale2);
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
                return Ok(acc);
            }
        }
    }
    host.per_expert += 1;
    per_expert_bf16(src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host)
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
    use mary::models::inkling::bf16gemm::to_bf16_launch;
    use mary::models::inkling::fp4gemm::gate_up_silu_bf16_launch;
    use mary::models::inkling::moegroup::{
        bf16_linear_grouped_launch, gather_grouped, scatter_weighted, BlockPlanDev, RowPlan,
    };
    use mary::models::inkling::seam::{handle_of, tensor_of};

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
    let hn_h = handle_of(hn.clone());
    let h_rowtok = client.create_from_slice(bytes_of(&plan.row_tok));
    let h_rowwgt = client.create_from_slice(bytes_of(&plan.row_wgt));
    let blk = BlockPlanDev {
        slot: client.create_from_slice(bytes_of(&plan.blk_slot)),
        tile0: client.create_from_slice(bytes_of(&plan.blk_tile0)),
        cnt: client.create_from_slice(bytes_of(&plan.blk_cnt)),
        blocks: plan.blk_slot.len(),
        planes: RowPlan::planes(),
    };
    let h_off13 = client.create_from_slice(bytes_of(&off13));
    let h_off2 = client.create_from_slice(bytes_of(&off2));
    let h_tokrows = client.create_from_slice(bytes_of(&plan.tok_rows));
    let h_tokcnt = client.create_from_slice(bytes_of(&plan.tok_cnt));
    let x_h = gather_grouped(client, &hn_h, &h_rowtok, n, m_total, h);
    host.gather += t_g.elapsed().as_secs_f64();

    let t_w = Instant::now();
    let a = to_bf16_launch(client, &x_h, m_total * h, m_total * h);
    let both = bf16_linear_grouped_launch(
        client, &a, &wmap, wmap_bytes, &blk, &h_off13, slots, m_total, h, 2 * inter,
    );
    let act = gate_up_silu_bf16_launch(client, &both, m_total, inter);
    let y_h = bf16_linear_grouped_launch(
        client, &act, &wmap, wmap_bytes, &blk, &h_off2, slots, m_total, inter, h,
    );
    host.enqueue += t_w.elapsed().as_secs_f64();

    let t_c = Instant::now();
    let acc_h = scatter_weighted(
        client, &y_h, &h_rowwgt, &h_tokrows, &h_tokcnt, &h_rowtok, m_total, n, h, plan.kmax,
    );
    let acc = tensor_of(client.clone(), dev.clone(), acc_h, n, h);
    host.accum += t_c.elapsed().as_secs_f64();

    Ok(Some(acc))
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
    use mary::models::inkling::bf16gemm::{bf16_linear_launch, to_bf16_launch, MTILE};
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
    let hn_h = handle_of(hn.clone());

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

    let pile_path = std::env::args().nth(1).map(PathBuf::from).context("usage: <pile> <ids> <out>")?;
    let ids_path = std::env::args().nth(2).map(PathBuf::from).context("usage: <pile> <ids> <out>")?;
    let out_path = std::env::args().nth(3).map(PathBuf::from).context("usage: <pile> <ids> <out>")?;

    // The weights, and everything else the run needs to know about the model.
    // There is no second arm: no `INK_PILE` to opt into, and no checkpoint
    // directory to fall back to.
    let pile_branch = std::env::var("INK_PILE_BRANCH").unwrap_or_else(|_| "inkling".to_string());
    let t_open = Instant::now();
    let mut cp = Weights::open(&pile_path, &pile_branch)
        .with_context(|| format!("opening {} on branch {pile_branch}", pile_path.display()))?;
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
        Some(p) => std::fs::read_to_string(p)
            .with_context(|| format!("INK_CONFIG={p}"))?,
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
    let is_head = pipe_spec.as_deref().map(|s| s.starts_with("head:")).unwrap_or(false);
    let is_tail = pipe_spec.as_deref().map(|s| s.starts_with("tail:")).unwrap_or(false);
    anyhow::ensure!(
        pipe_spec.is_none() || is_head || is_tail,
        "INK_PIPE wants head:HOST:PORT or tail:ADDR:PORT"
    );
    anyhow::ensure!(
        !is_head || hi < t.num_hidden_layers,
        "a head that runs to the last layer has nothing to send; set INK_LAYERS"
    );
    anyhow::ensure!(!is_tail || lo > 0, "a tail that starts at layer 0 has nothing to receive");

    let corpus: Vec<usize> = std::fs::read(&ids_path)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    anyhow::ensure!(!corpus.is_empty(), "no tokens — the forward would be vacuous");

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
    let nslots: usize =
        std::env::var("INK_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    anyhow::ensure!(nslots >= 1, "INK_SLOTS counts sequences and starts at 1");
    let slot_len: usize = std::env::var("INK_SLOT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if slot_lane { corpus.len() / nslots } else { corpus.len() });
    anyhow::ensure!(slot_len > 0, "INK_SLOT_LEN is a prompt length and starts at 1");
    let slot_at: Vec<usize> = match std::env::var("INK_SLOT_OFFSETS") {
        Ok(v) => v
            .split(',')
            .map(|c| c.trim().parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("INK_SLOT_OFFSETS is a comma-separated list of chunk indices")?,
        Err(_) => (0..nslots).collect(),
    };
    anyhow::ensure!(
        slot_at.len() == nslots,
        "INK_SLOT_OFFSETS names {} chunks against INK_SLOTS={nslots}",
        slot_at.len()
    );
    for &c in &slot_at {
        anyhow::ensure!(
            (c + 1) * slot_len <= corpus.len(),
            "chunk {c} of {slot_len} tokens runs past a {}-token file",
            corpus.len()
        );
    }
    let mut slot_ids: Vec<Vec<usize>> =
        slot_at.iter().map(|&c| corpus[c * slot_len..(c + 1) * slot_len].to_vec()).collect();
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
            "  slots      : INK_SLOTS={nslots} -- {nslots} independent sequences of {slot_len} \
             tokens, chunks {slot_at:?} of a {}-token file",
            corpus.len()
        );
    }
    println!("  layers     : {}  hidden {h}  experts {}+{} shared",
             t.num_hidden_layers, t.n_routed_experts, t.n_shared_experts);
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
    let mtp_k: usize = std::env::var("INK_MTP").ok().map(|v| v.parse()).transpose()?.unwrap_or(0);
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
    // What prefill will hold in `[heads, n, n]` score matrices, which is the
    // largest thing this run allocates and the only term that grows as `n^2`.
    // The admission gate charged a flat per-layer figure for "activations"
    // regardless of sequence length, which is how it admitted a run that peaked
    // at 119.5 GiB of a 119.6 GiB node.
    let attn_heads = t.heads(AttnKind::Global).0.max(t.heads(AttnKind::Local).0);
    let attn_head_dim = t.heads(AttnKind::Global).2.max(t.heads(AttnKind::Local).2);
    let attention_bytes = budget::prefill_peak_bytes(attn_heads, n);
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
                2 * nslots as u64 * keep as u64 * (kv_heads * head_dim) as u64 * 4
            })
            .sum()
    } else {
        0
    };
    if slot_lane {
        println!(
            "  slot KV            : {:.2} GiB for {nslots} slots over layers {lo}..{hi}",
            slot_kv_bytes as f64 / GIB
        );
    }

    let want_embed = !is_tail || mtp_k > 0;
    let want_embed_norm = !is_tail;
    let want_head = !is_head;
    // The supported lane copies every weight this process can alias into one
    // anonymous allocation before registration. `INK_STARTUP_COPY=0` exists
    // only for the pressure reproducer: it deliberately restores the unsafe
    // file-backed alias so the gate can prove that pressure was present.
    if std::env::var("INK_STARTUP_COPY").map(|v| v == "0").unwrap_or(false) {
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
        let (experts, dense, bytes) =
            cp.copy_share(lo..hi, &globals, attention_bytes + slot_kv_bytes)?;
        println!(
            "  startup weight copy: {experts} expert + {dense} dense views, {:.2} GiB anonymous in {:.1}s",
            bytes as f64 / GIB,
            t0.elapsed().as_secs_f64(),
        );
    }
    // The embedding table, as the BF16 the pile stores. `cp.held` turned 2.40
    // GiB of stored weight into 4.81 GB of host f32 and pinned it for the run,
    // on a box chosen because the working set only just fits -- and every token
    // read one row of it. `stored` hands back a view of the mapping; the
    // widening is now per LOOKUP, 16 KB a token, which is where it belongs.
    let embed_w = if want_embed {
        let leaf = cp.stored("model.llm.embed.weight")?;
        anyhow::ensure!(leaf.elem == Elem::Bf16, "the embedding table is {:?}", leaf.elem);
        Some(leaf.bytes.clone())
    } else {
        None
    };
    // `want_embed_norm`, NOT `want_embed`: a drafting tail takes the embedding
    // table and leaves the norm behind, because every MTP head norms its own
    // embeddings with its own `embed_norm`.
    let embed_n = if want_embed_norm { Some(cp.held("model.llm.embed_norm.weight")?) } else { None };
    let fnorm = if want_head { Some(cp.held("model.llm.norm.weight")?) } else { None };
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
                    rel_extent: t.rel_span(if local { AttnKind::Local } else { AttnKind::Global }),
                    kernel: t.sconv_kernel_size,
                    rms_eps: t.rms_norm_eps,
                    kind: if local { AttnKind::Local } else { AttnKind::Global },
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
    let mtp_prob = std::env::var("INK_MTP_PROB").map(|val| val == "1").unwrap_or(false);
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
    let mtp_dev_on = std::env::var("INK_MTP_DEV").map(|val| val != "0").unwrap_or(true);
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
    let kv = std::env::var("INK_KV").map(|v| v == "1" || v == "on").unwrap_or(false);
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
    let repeat = std::env::var("INK_REPEAT").map(|v| v == "1" || v == "on").unwrap_or(false);
    anyhow::ensure!(!repeat || !kv, "INK_REPEAT wants the uncached lane: with a KV cache a \
         repeated pass would append the same position to the cache again");
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
    let spec_k: usize = std::env::var("INK_SPEC").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    anyhow::ensure!(
        spec_k == 0 || kv,
        "INK_SPEC={spec_k} wants INK_KV=1: speculation is about skipping sequential steps, and \
         the uncached lane has no cache to roll back"
    );
    anyhow::ensure!(!repeat || spec_k == 0, "INK_REPEAT and INK_SPEC measure different things");
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
        println!("  speculation        : INK_SPEC={spec_k} -- verify pass is {} rows", spec_k + 1);
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
    let width: usize = std::env::var("INK_WIDTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    anyhow::ensure!(width >= 1, "INK_WIDTH counts rows and starts at 1");
    anyhow::ensure!(
        width == 1 || kv,
        "INK_WIDTH wants INK_KV=1: the uncached lane feeds the whole prefix and has no one-row \
         step to widen"
    );
    anyhow::ensure!(width == 1 || spec_k == 0, "INK_WIDTH and INK_SPEC both widen the pass");
    anyhow::ensure!(
        width == 1 || mtp_k == 0,
        "INK_WIDTH is a cost probe and drafting is not part of what it prices"
    );
    if width > 1 {
        println!("  width probe        : INK_WIDTH={width} -- every cached step is {width} rows, \
                  one of them real");
    }
    anyhow::ensure!(!slot_lane || kv, "INK_SLOTS wants INK_KV=1: a slot IS a cache");
    anyhow::ensure!(
        !slot_lane || width == 1,
        "INK_WIDTH prices a b-row pass with filler; INK_SLOTS runs one with b sequences in it, \
         and setting both would put filler rows in a batch that already has real ones"
    );
    anyhow::ensure!(!slot_lane || spec_k == 0, "INK_SLOTS and INK_SPEC both widen the pass");
    anyhow::ensure!(
        !slot_lane || mtp_k == 0,
        "INK_SLOTS and INK_MTP: drafting follows one sequence and there are {nslots} here"
    );
    anyhow::ensure!(!slot_lane || !repeat, "INK_REPEAT and INK_SLOTS measure different things");
    if slot_lane {
        println!(
            "  batched decode     : INK_SLOTS={nslots} -- {nslots} caches, one pass, {nslots} \
             tokens a pass"
        );
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
    println!("  attention          : device, weights DEVICE-RESIDENT");
    println!("  shared + dense MLP : device, uploaded once and held");
    println!("  routed experts     : device, NATIVE tensor cores -- NVFP4 where packed, BF16 at layer 2");
    println!("  head (unembed)     : device");
    let router_arm = RouterArm::from_env();
    println!("  router projection  : {}", router_arm.label());
    // OPT-IN, and this run is slower for it: it uploads a second projection per
    // layer and issues a second matmul and a second BLOCKING read per MoE layer
    // per token. A timing run must not have it on, and the report says which
    // runs did.
    let router_diff = std::env::var("INK_ROUTER_DIFF").map(|v| v == "1").unwrap_or(false);
    if router_diff {
        println!("  router diff        : ON -- selection compared against the f32 [rows,hidden] lane, this pass IS slower");
    }
    println!("  kv cache           : {}", if kv { "on" } else { "off (prefix recomputed each step)" });
    // The SHARED experts' w13 is square, so nothing but a forward can tell the
    // two readings apart. INK_SHARED_W13_HALVED=1 selects the other one.
    let shared_halved = mary::models::inkling::load::shared_w13_halved();
    println!(
        "  shared w13 split   : {}",
        if shared_halved { "HALVED (contiguous)" } else { "INTERLEAVED" }
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
    let fp4_client = mary::models::inkling::seam::client_of(
        &BT::<Bk, 2>::zeros([1, 1], &dev),
    );
    println!("{}", mary::models::inkling::pile::mem_line("after CUDA context"));
    // The one number that decides whether this sequence length is runnable at
    // all, asked of the device rather than modelled. It is checked HERE and not
    // beside the admission gate because this is the first client in the
    // process, and taking `max_page_size` from a client Burn did not make would
    // be reading a different device's answer. Nothing has run a layer yet.
    let qblock = budget::query_block(attn_heads, n);
    println!(
        "  attention budget   : queries in blocks of {qblock}, so [{attn_heads}, {qblock}, {n}] \
         f32 scores = {:.2} GiB per layer (the whole square would be {:.2} GiB) beside \
         [{attn_heads}, {n}, {attn_head_dim}] f32 activations = {:.2} GiB; largest single \
         allocation this device allows {:.2} GiB (up to {} tokens)",
        budget::score_block_bytes(attn_heads, qblock, n) as f64 / GIB,
        budget::score_matrix_bytes(attn_heads, n) as f64 / GIB,
        budget::activation_bytes(attn_heads, attn_head_dim, n) as f64 / GIB,
        budget::largest_allocation(&fp4_client) as f64 / GIB,
        budget::longest_sequence(
            attn_heads,
            attn_head_dim,
            budget::largest_allocation(&fp4_client)
        ),
    );
    budget::check(&fp4_client, attn_heads, attn_head_dim, n)?;
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
        if std::env::var("INK_ZEROCOPY").map(|v| v == "0").unwrap_or(false) {
            println!("  zero-copy mappings : DISABLED (INK_ZEROCOPY=0) -- every bind copies");
            Some(mary::models::inkling::fp4gemm::Aliases::disabled())
        } else {
            let t = Instant::now();
            let maps = cp.mappings()?;
            let n = maps.len();
            let a = mary::models::inkling::fp4gemm::Aliases::register(c, maps);
            println!(
                "  zero-copy mappings : {} {n} in {:.1} ms",
                if a.is_some() { "registered" } else { "UNSUPPORTED, copying" },
                t.elapsed().as_secs_f64() * 1e3
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
    let unembed_w = if want_head {
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
        anyhow::ensure!(rows == t.vocab_size && cols == h, "unembed is {rows}x{cols}");
        anyhow::ensure!(
            Bf16W::tileable(rows, cols),
            "unembed {rows}x{cols} does not tile as m16n8k16"
        );
        let align = note_align(&leaf.bytes, rows, cols);
        let copy_to_align = align < 16
            && std::env::var("INK_ALIGN_COPY").map(|v| v == "1").unwrap_or(false);
        let hnd = match fp4_aliases.as_ref() {
            Some(al) if !copy_to_align => al.slice_or_copy(&fp4_client, &leaf.bytes),
            _ => fp4_client.create_from_slice(&leaf.bytes),
        };
        println!(
            "  unembed BOUND as BF16, {rows} x {cols} = {:.2} GiB stored (the f32 lane it \
             replaces materialised {:.2} GiB)",
            leaf.bytes.len() as f64 / GIB,
            2.0 * leaf.bytes.len() as f64 / GIB
        );
        Some(Bf16W { h: hnd, n: rows, k: cols, align: if copy_to_align { 16 } else { align } })
    } else {
        None
    };
    // The final norm's gain, uploaded once for the same reason -- it used to be
    // re-uploaded from the host copy on every pass, and on every MTP draft.
    let fnorm_dev = fnorm
        .as_ref()
        .map(|f| up1r::<Bk>(&f.data, h, &dev));

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
    // One `Vec<LayerCache>` per slot, filled by that slot's prefill pass, and
    // consumed once when the first decode pass stacks them. A prefill is
    // compute-bound and gains nothing from a batch, so the b of them run one at
    // a time and the batch begins where the decoding does.
    let mut slot_prefills: Vec<Vec<LayerCache>> = Vec::new();
    let mut slots_dev: Vec<SlotLayerCache> = Vec::new();

    // The wire, opened AFTER the weights so a connection is never left hanging
    // while the other end spends a minute building its index. The tail binds and
    // waits; the head connects, so the tail must be started first.
    let mut pipe = match pipe_spec.as_deref() {
        Some(s) if is_head => {
            let addr = &s["head:".len()..];
            let t0 = Instant::now();
            let sock = TcpStream::connect(addr)
                .with_context(|| format!("connecting to the tail at {addr}"))?;
            sock.set_nodelay(true)?;
            println!("  pipe: connected to the tail at {addr} in {:.1}s", t0.elapsed().as_secs_f32());
            Some(Pipe::Head(sock))
        }
        Some(s) if is_tail => {
            let addr = &s["tail:".len()..];
            let l = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
            println!("  pipe: listening on {addr}");
            let (sock, peer) = l.accept()?;
            sock.set_nodelay(true)?;
            println!("  pipe: head connected from {peer}");
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
    let dev_route = std::env::var("INK_DEV_ROUTE").map(|v| v != "0").unwrap_or(true);
    // The gate bias is `[n_routed]` f32 and does not change during a run, so it
    // is uploaded once per layer rather than once per pass. One KiB either way;
    // it is here because a per-pass upload in a lane whose whole subject is
    // host->device round trips would be embarrassing.
    let mut bias_dev: std::collections::HashMap<usize, cubecl::server::Handle> =
        std::collections::HashMap::new();

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
    let mut last_drafts: Vec<usize> = Vec::new();
    // Tokens, not passes: a speculative pass confirms between 1 and k+1 of
    // them, so this is what the run's length and its tok/s are counted in.
    let mut gen_tokens = 0usize;
    let mut spec_hist = vec![0usize; spec_k + 2];
    let mut pass_ms: Vec<f64> = Vec::new();
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
    let prefill_passes = if slot_lane { nslots } else { 1 };
    let is_decode = step >= prefill_passes;

    let pass = Instant::now();
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
        (
            slot_ids
                .iter()
                .map(|q| *q.last().expect("every slot's prefill produced a token"))
                .collect(),
            slot_ids[0].len() - 1,
        )
    } else if slot_lane {
        (slot_ids[step].clone(), 0)
    } else if kv && step > 0 {
        // The verify batch: the token the last pass confirmed, then the tail's
        // drafts for the positions after it. `drafts_in` is empty unless this
        // is a speculating head, so the non-speculative shape is the same one
        // row it always was.
        let mut f = vec![*ids.last().expect("a step past the prefill has produced a token")];
        f.extend(drafts_in.iter().copied());
        // The width probe's filler. Drawn from a counter rather than from the
        // sequence: a batch of the same token routes to the same eight experts
        // and would price the expert stream once for the whole batch, which is
        // the one thing the probe exists to find out.
        let mut lcg = 0x9E3779B97F4A7C15u64 ^ (step as u64).wrapping_mul(0x100000001B3);
        for _ in 1..width {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            f.push(((lcg >> 33) as usize) % t.vocab_size);
        }
        (f, ids.len() - 1)
    } else {
        (ids.clone(), 0)
    };
    // The b prefilled caches become one slot batch, once, on the first pass
    // that decodes. Layer by layer, because a layer's KV width is a function of
    // its kind and a local layer's is not a global one's.
    if slot_lane && is_decode && slots_dev.is_empty() {
        assert_eq!(slot_prefills.len(), nslots, "a slot did not prefill");
        for l in 0..(hi - lo) {
            let kind = t.attn_kind(lo + l);
            let (_, kv_heads, head_dim) = t.heads(kind);
            let mut attn = Vec::with_capacity(nslots);
            let mut asc = Vec::with_capacity(nslots);
            let mut msc = Vec::with_capacity(nslots);
            for per_slot in slot_prefills.iter() {
                let c = &per_slot[l];
                attn.push(c.attn.clone());
                asc.push(c.attn_sconv.clone().reshape([1, t.sconv_kernel_size - 1, h]));
                msc.push(
                    c.mlp_sconv
                        .clone()
                        .expect("a prefill seeds the MLP convolution")
                        .reshape([1, t.sconv_kernel_size - 1, h]),
                );
            }
            slots_dev.push(SlotLayerCache {
                attn: dev_lane::SlotCache::from_prefills(attn, kv_heads, head_dim),
                attn_sconv: BT::cat(asc, 0),
                mlp_sconv: Some(BT::cat(msc, 0)),
            });
        }
        slot_prefills.clear();
    }
    // The tail is handed the stream the head already embedded and ran; it takes
    // `n` and `pos0` from the wire rather than from `ids`, because those are
    // facts about the pass and only the head owns the token loop.
    let t_emb = Instant::now();
    let (n, pos0, x_in) = match incoming {
        Some((n, p, x)) => (n, p, x),
        None => {
            let n = feed.len();
            let e_w = embed_w.as_ref().expect("the head owns the embedding table");
            let e_n = embed_n.as_ref().expect("the head owns the embedding norm");
            (n, pos0, embed_and_norm_bf16(&feed, e_w, &e_n.data, t.rms_norm_eps, t.vocab_size, h))
        }
    };
    fatal::note_tokens(n);

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
    let mut xd: T2 = up2::<Bk>(x_in, n, h, &dev);
    let t_embed = t_emb.elapsed().as_secs_f64();

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
    let stage_sync = std::env::var("INK_STAGE_SYNC").map(|v| v == "1").unwrap_or(false);
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
        ($acc:expr) => {
            if stage_sync {
                let s = Instant::now();
                <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("stage sync");
                $acc += s.elapsed().as_secs_f64();
                stage_syncs += 1;
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
                Ok(bind_bf16(&fp4_client, fp4_aliases.as_ref(), &leaf.bytes, rows, cols))
            };
            let attn = dev_lane::AttnWeightsDev {
                wq: pw("attn.wq_du.weight", heads * head_dim, h)?,
                wk: pw("attn.wk_dv.weight", kv_heads * head_dim, h)?,
                wv: pw("attn.wv_dv.weight", kv_heads * head_dim, h)?,
                wr: pw("attn.wr_du.weight", heads * t.d_rel, h)?,
                wo: pw("attn.wo_ud.weight", h, heads * head_dim)?,
                k_sconv: up2(gv("attn.k_sconv.weight")?, kv_heads * head_dim, t.sconv_kernel_size, &dev),
                v_sconv: up2(gv("attn.v_sconv.weight")?, kv_heads * head_dim, t.sconv_kernel_size, &dev),
                q_norm: up1(gv("attn.q_norm.weight")?, head_dim, &dev),
                k_norm: up1(gv("attn.k_norm.weight")?, head_dim, &dev),
                rel_proj: up2(gv("attn.rel_logits_proj.proj")?, t.d_rel, t.rel_span(kind), &dev),
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
                            rows,
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
            dattn_bytes += (2 * (heads * head_dim * h
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
        let hn = dev_lane::rms_norm(xd.clone(), ld.attn_norm.clone(), t.rms_norm_eps);
        let dims = AttnDims {
            hidden: h, heads, kv_heads, head_dim,
            d_rel: t.d_rel,
            rel_extent: t.rel_span(kind),
            kernel: t.sconv_kernel_size,
            rms_eps: t.rms_norm_eps,
            kind,
        };
        // The same distinction the mask carries, in the form the cache needs:
        // how far back a query may look, and therefore how much of the cache
        // can never be read again.
        let window = if is_local { Some(t.sliding_window_size) } else { None };
        let a = if slot_lane && is_decode {
            // b sequences, one position each, one pass. `attention_steps` is
            // the wrong function here and not by a little: its rows are
            // consecutive positions of ONE sequence and its mask admits every
            // earlier row of the batch, which for independent slots is exactly
            // the contamination this lane exists to make impossible.
            let y = dev_lane::attention_slots(
                hn, &ld.attn, &dims, Some(ls), pos0, window, &mut slots_dev[slot].attn,
            );
            let (out, hist) = dev_lane::short_conv_slot_step(
                slots_dev[slot].attn_sconv.clone(), y, ld.attn_sconv.clone(),
            );
            slots_dev[slot].attn_sconv = hist;
            out
        } else if kv && is_decode && n > 1 {
            // The speculative width. `attention_steps` leaves the batch PENDING
            // and the convolution keeps its whole window, so neither is final
            // until the verifier below says how many rows survived. Nothing
            // here knows that yet -- the answer is a machine away.
            let y = dev_lane::attention_steps(
                hn, &ld.attn, &dims, Some(ls), pos0, window, &mut caches[slot].attn,
            );
            let (out, all) = dev_lane::short_conv_steps(
                caches[slot].attn_sconv.clone(), y, ld.attn_sconv.clone(),
            );
            caches[slot].attn_sconv_pending = Some(all);
            out
        } else if kv && is_decode {
            let y = dev_lane::attention_step(
                hn, &ld.attn, &dims, Some(ls), pos0, window, &mut caches[slot].attn,
            );
            let (out, hist) =
                dev_lane::short_conv_step(caches[slot].attn_sconv.clone(), y, ld.attn_sconv.clone());
            caches[slot].attn_sconv = hist;
            out
        } else if kv {
            let (y, attn) = dev_lane::attention_prefill(
                hn, &ld.attn, &dims, Some(ls), window, window,
            );
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
        xd = xd + a;

        stage_sync!(d_attn);
        // ---- MLP ----------------------------------------------------------
        t_attn += t_a.elapsed().as_secs_f64();
        let t_o = Instant::now();
        let hn = dev_lane::rms_norm(xd.clone(), ld.mlp_norm.clone(), t.rms_norm_eps);

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
            // DECISION is control plane and runs here. What crosses is
            // [n, 258] f32 -- 1 KB on a decode step -- against the 14.2 MB of
            // expert weight the decision selects. This read BLOCKS, and it is
            // the only place in the layer that does; it cannot be avoided by
            // scheduling, because which weights to read next is a function of
            // the number that just came back.
            let rows = t.n_routed_experts + t.n_shared_experts;
            let t_rt = Instant::now();
            // `cols` is what comes BACK, which is `rows` except on the BF16 arm,
            // whose weight carries the instruction's n padding.
            let (lg, cols) = match &r.proj {
                RouterProj::PerCall(w) => (dev_lane::linear(hn.clone(), w.clone()), rows),
                RouterProj::Pre(wt) => (dev_lane::linear_pre_t(hn.clone(), wt.clone()), rows),
                RouterProj::Bf16 { w, .. } => (dev_lane::linear_bf16(hn.clone(), w), w.n),
            };
            t_rt_mm += t_rt.elapsed().as_secs_f64();
            stage_sync!(d_router);
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
            let routing: Vec<Routing>;
            let mut logits: Vec<f32> = Vec::new();
            if host_route {
                let t_rr = Instant::now();
                logits = drop_pad_cols(down(lg), n, cols, rows);
                t_rt_read += t_rr.elapsed().as_secs_f64();
                let t_rh = Instant::now();
                routing = route_from_logits(
                    &logits, &r.bias, r.global_scale, t.route_scale as f32,
                    n, t.n_routed_experts, t.n_shared_experts, t.num_experts_per_tok,
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
                    &fp4_client, &lg_h, &bias_h, n, cols, t.n_routed_experts, ns, k,
                    t.route_scale as f32 * r.global_scale,
                );
                t_rt_mm += t_rt2.elapsed().as_secs_f64();
                let t_rr = Instant::now();
                let flat = down(tensor_of(fp4_client.clone(), dev.clone(), out_h, n, width));
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
                        down(tensor_of(fp4_client.clone(), dev.clone(), lg_h.clone(), n, cols)),
                        n, cols, rows,
                    );
                    let hr = route_from_logits(
                        &hl, &r.bias, r.global_scale, t.route_scale as f32,
                        n, t.n_routed_experts, ns, k,
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
                            let d = (routing[ti].shared_gammas[j] - hr[ti].shared_gammas[j]).abs();
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
            let t_rh = Instant::now();

            // `INK_ROUTER_DIFF=1`: the same activation through the f32 lane,
            // the same selection rule on the result, and a count of where the
            // two disagree. Nothing below reads `ref_routing` -- the run acts on
            // `routing`, whichever arm produced it -- so this measures the arm
            // rather than replacing it.
            if let Some(rw) = r.reference.as_ref() {
                let ref_logits = down(dev_lane::linear(hn.clone(), rw.clone()));
                let ref_routing = route_from_logits(
                    &ref_logits, &r.bias, r.global_scale, t.route_scale as f32,
                    n, t.n_routed_experts, t.n_shared_experts, t.num_experts_per_tok,
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

            // Group tokens by expert, so each slab is read once.
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

            let t_d = Instant::now();
            // Two formats, two instructions, one lane. 41 of the 42 layers
            // carry NVFP4 experts and go through the block-scaled MMA; layer 2
            // carries plain BF16 ones -- no `.scale` sidecar, because nothing
            // quantised them -- and goes through the unscaled BF16 MMA
            // (`mma.sync...bf16`, f32 accumulator, which is the instruction's
            // own output type and not a widening).
            //
            // Which one is decided by the pile, not by a flag.
            let acc = {
                let a = if cp.is_nvfp4(&format!("{p}mlp.experts.w13_weight")) {
                    routed_experts_fp4(
                        &cp, fp4_aliases.as_ref(), &fp4_client, &dev,
                        &p, &by_expert, &hn, n, h, inter, &mut host_t,
                    )?
                } else {
                    routed_experts_bf16(
                        &cp, fp4_aliases.as_ref(), &fp4_client, &dev,
                        &p, &by_expert, &hn, n, h, inter, &mut host_t,
                    )?
                };
                expert_loads += by_expert.len();
                a
            };
            // ENQUEUE time, not work: nothing in this lane synchronises any
            // more. The layer's device time shows up in the one sync after the
            // stack, which is where it belongs and where it cannot be
            // misattributed to whichever bucket happened to hold the readback.
            t_expert += t_d.elapsed().as_secs_f64();
            stage_sync!(d_expert);

            let ns = t.n_shared_experts;
            let gammas: Vec<f32> = routing.iter().flat_map(|rt| rt.shared_gammas.clone()).collect();
            let t_s = Instant::now();
            // Device-resident, uploaded once. `split_shared_w13` is the
            // settled reading — this used to be an open `deinterleave_rows`
            // here and a halved split in the gate, which is the contradiction
            // the INTERLEAVED result closed.
            let sh = {
                let sw = ddense.shared_for(
                    &cp, &fp4_client, fp4_aliases.as_ref(), &p, ns, inter, h, shared_halved,
                )?;
                shared_experts_bf16(&dev, hn, sw, &gammas, ns)
            };
            stage_sync!(d_shared);
            t_shared += t_s.elapsed().as_secs_f64();
            acc + sh
        };

        // The MLP half's own short convolution carries state across generated
        // tokens exactly as attention's do.
        let t_sc = Instant::now();
        let y = if slot_lane && is_decode {
            let hist = slots_dev[slot]
                .mlp_sconv
                .clone()
                .expect("a slot batch carries its own convolution memory");
            let (out, next) = dev_lane::short_conv_slot_step(hist, y, ld.mlp_sconv.clone());
            slots_dev[slot].mlp_sconv = Some(next);
            out
        } else if kv {
            if is_decode {
                let hist = caches[slot]
                    .mlp_sconv
                    .clone()
                    .expect("a step past the prefill has a history");
                if n > 1 {
                    let (out, all) = dev_lane::short_conv_steps(hist, y, ld.mlp_sconv.clone());
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
        xd = xd + y;
        stage_sync!(d_tail);

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
        layer_rms.push(xd.clone().powf_scalar(2.0).mean().reshape([1, 1]));
        layer_kind.push((layer, is_local));
    }

    // This slot is prefilled; the next one starts from an empty `caches` and
    // the batch is assembled once all b are in.
    if slot_lane && !is_decode {
        slot_prefills.push(std::mem::take(&mut caches));
    }

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
    let rms_col: Vec<f32> = if layer_rms.is_empty() {
        Vec::new()
    } else {
        down(BT::cat(std::mem::take(&mut layer_rms), 0))
    };
    let t_stack_sync = t_sy.elapsed().as_secs_f64();
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
    let x: Vec<f32> = if want_host_x { down(xd.clone()) } else { Vec::new() };

    // Does anything downstream need a logit row that is NOT the last one?
    //
    // The argmax reads exactly one row -- the last -- and that is the whole of
    // what a forward produces. Every other row of the head exists for the
    // REPORT: the per-position top-5 table and the `INK_DUMP_DIR` capture. On a
    // 512-token prefill those rows are 512 x 200058 f32 = 410 MB read back
    // across the bus, on top of a 4096-wide GEMM over 512 rows instead of 16.
    // So they are computed when a reader has asked for them and not otherwise.
    // `INK_ALL_LOGITS=1` is that ask; a dump implies it.
    let all_logits =
        dump_dir.is_some() || std::env::var("INK_ALL_LOGITS").map(|val| val == "1").unwrap_or(false);

    // ---- head, or the wire in its place ------------------------------------
    let v = t.effective_vocab();
    let t_h = Instant::now();
    // A head has no logits and never will: the rest of the stack and the
    // unembedding both live on the other machine. So it hands the stream over
    // and takes the argmax back, and that blocking call is charged to the same
    // slot the head/unembed occupies on a whole-stack run — which is what makes
    // the two reports read against each other line for line.
    let mut best_wire = None;
    // Which position `logits[0]` is. The head computes `logit_row0..n`, so this
    // is 0 when everything was asked for and `n - 1` when only the argmax's row
    // was. A head computes nothing and the value is unread there.
    // How many of this pass's rows the verifier has to read an argmax off. One,
    // normally -- a forward produces one token. A speculative pass produces one
    // PER ROW, and every one of them is needed: the accepted prefix is the
    // leading run where the draft and the argmax agree, so a rule that only
    // looked at the last row could not find where the agreement stopped.
    let verify_rows = if spec_k > 0 && kv && step > 0 { n } else { 1 };
    // The width probe's rows, derived from the batch the head actually sent
    // rather than from this process's environment -- so INK_WIDTH is set on the
    // head alone and the two ends cannot disagree about it.
    //
    // Every row is unembedded, and that is deliberate: b independent sequences
    // each need their own logits, so a probe that unembedded one row would
    // leave the widest matmul in the stack out of the price.
    let probe_rows = if spec_k == 0 && kv && is_decode && n > 1 && !slot_lane { n } else { 1 };
    // The rows a slot batch reads an argmax off: all of them, because each one
    // is a different sequence and each one's next token is a fact about it.
    // This is the widest matmul in the stack run at its real width, which is
    // the half of batched decode the width probe was already honest about.
    let slot_rows = if slot_lane && is_decode { n } else { 1 };
    let logit_row0 =
        if all_logits || probe_rows > 1 || slot_rows > 1 { 0 } else { n - verify_rows };
    let (mut t_send, mut t_wait_peer) = (0f64, 0f64);
    let mut wire_toks: Vec<usize> = Vec::new();
    let logits = if let Some(Pipe::Head(s)) = pipe.as_mut() {
        let t_s = Instant::now();
        send_stream(s, n, pos0, &x)?;
        t_send = t_s.elapsed().as_secs_f64();
        let t_w = Instant::now();
        // The tail's FIRST message: the tokens its verify pass confirmed. Never
        // empty -- the row fed the last confirmed token always produces one --
        // and longer than one exactly when drafts were accepted. The drafts for
        // the NEXT pass are a second message, read further down, so this
        // process gets to commit its caches in between.
        wire_toks = recv_toks(s)?;
        t_wait_peer = t_w.elapsed().as_secs_f64();
        anyhow::ensure!(!wire_toks.is_empty(), "the tail confirmed no token at all");
        best_wire = Some(*wire_toks.last().expect("checked non-empty"));
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
        let hs = dev_lane::rms_norm(
            hx,
            fnorm_dev.clone().expect("the tail owns the final norm"),
            t.rms_norm_eps,
        )
        .div_scalar(t.logits_mup_width_multiplier as f32);
        let uw = unembed_w.as_ref().expect("the tail binds the unembed table");
        down(dev_lane::linear_bf16(hs, uw).slice([0..n - logit_row0, 0..v]))
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
    let best = match best_wire {
        Some(b) => {
            new_toks = wire_toks.clone();
            b
        }
        None => {
            let mut accepted = 0usize;
            let rows = n - logit_row0;
            let argmax_of = |i: usize| -> usize {
                let row = &logits[i * v..(i + 1) * v];
                let mut b = 0usize;
                for (j, &val) in row.iter().enumerate() {
                    if val > row[b] {
                        b = j;
                    }
                }
                b
            };
            if verify_rows > 1 {
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
            *new_toks.last().expect("at least one row is always confirmed")
        }
    };
    // `ids`, the MTP scoring and the per-position report were all written about
    // ONE sequence, and slot 0 is the one they follow. `best` is therefore slot
    // 0's token and not the last row's, which is what it means everywhere else.
    let best = if slot_lane && is_decode { new_toks[0] } else { best };
    let mut t_to_reply = 0f64;
    if let Some(Pipe::Tail(s)) = pipe.as_mut() {
        send_toks(s, &new_toks)?;
        t_to_reply = pass.elapsed().as_secs_f64();
    }
    // Everything but the LAST confirmed token goes into `ids` now; the last one
    // is `best` and is pushed where it has always been pushed, so the MTP block
    // below sees exactly the sequence-and-a-held-back-argmax it was written
    // against.
    if is_tail && gen_steps > 0 && !repeat && new_toks.len() > 1 && !slot_lane {
        ids.extend_from_slice(&new_toks[..new_toks.len() - 1]);
    }
    // Each slot's own stream. A prefill pass produced the first generated token
    // of the slot it prefilled; a decode pass produces one for every slot. Both
    // ends run this -- the head off the wire, the tail off its own argmax --
    // for the same reason `ids` is recomputed rather than sent.
    if slot_lane && gen_steps > 0 && !repeat {
        if is_decode {
            for (q, tok) in slot_ids.iter_mut().zip(new_toks.iter()) {
                q.push(*tok);
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

    // ---- MTP: score the drafts that named this step, then draft afresh -----
    if mtp_k > 0 {
        // SCORE FIRST, against `best` -- the token the full stack just produced.
        // This is the whole experiment: a draft is right or it is not, and the
        // rate over many steps is the only oracle the composition has.
        // The target's OWN distribution at this position, once per pass rather
        // than once per pending draft. `logits` holds the argmax's row and only
        // that row unless a reporter asked for more, which is exactly the row
        // every rule below reads.
        let p_t: Vec<f32> = if mtp_prob && !logits.is_empty() {
            softmax_row(&logits[(n - 1 - logit_row0) * v..(n - logit_row0) * v])
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
        let e_w = embed_w.as_ref().expect("drafting needs the embedding table");
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
        let entry = if std::env::var("INK_MTP_RAW").map(|val| val == "1").unwrap_or(false) {
            x.clone()
        } else {
            rms_norm(&x, &fnorm_d.data, t.rms_norm_eps, n, h)
        };
        // ...for the rows the verifier KEPT, and no further. A speculative pass
        // computes a hidden state for every row it fed, and the ones past the
        // accepted prefix are functions of tokens the model did not choose. An
        // MTP head drafting from one of those would be drafting off a state
        // that never happened, and nothing downstream would say so.
        let entry = if verify_rows > 1 && new_toks.len() < n {
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
        let draft_probs: std::cell::RefCell<Vec<Vec<f32>>> = std::cell::RefCell::new(Vec::new());
        let draft_argmax = |row: &[f32]| -> usize {
            debug_assert_eq!(row.len(), h, "the draft head unembeds exactly one position");
            let dl = {
                let ud = unembed_w.as_ref().expect("drafting needs the unembed table");
                let hs = dev_lane::rms_norm(
                    up2::<Bk>(row.to_vec(), 1, h, &dev),
                    fnorm_dev.clone().expect("drafting needs the final norm"),
                    t.rms_norm_eps,
                )
                .div_scalar(t.logits_mup_width_multiplier as f32);
                down(dev_lane::linear_bf16(hs, ud).slice([0..1, 0..v]))
            };
            let mut b = 0usize;
            for (i, &val) in dl.iter().take(v).enumerate() {
                if val > dl[b] {
                    b = i;
                }
            }
            if mtp_prob {
                draft_probs.borrow_mut().push(softmax_row(&dl[..v]));
            }
            b
        };

        // The whole-sequence draft: every head over every position. This is what
        // the uncached lane runs, and what `INK_MTP_CHECK` gates the cached lane
        // against. `hidden` is the ENTRY state for the WHOLE sequence, which is
        // precisely the thing the cache exists so as not to need.
        let draft_whole = |hidden: &[f32], seq: usize, ids: &[usize], best: usize| -> Vec<usize> {
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
                    embeds[j * h..(j + 1) * h]
                        .copy_from_slice(&embed_row_bf16(e_w, tok, t.vocab_size, h));
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
        let draft_argmax_dev = |row: T2| -> usize {
            let dl = {
                let ud = unembed_w.as_ref().expect("drafting needs the unembed table");
                let hs = dev_lane::rms_norm(
                    row,
                    fnorm_dev.clone().expect("drafting needs the final norm"),
                    t.rms_norm_eps,
                )
                .div_scalar(t.logits_mup_width_multiplier as f32);
                down(dev_lane::linear_bf16(hs, ud).slice([0..1, 0..v]))
            };
            let mut b = 0usize;
            for (i, &val) in dl.iter().take(v).enumerate() {
                if val > dl[b] {
                    b = i;
                }
            }
            if mtp_prob {
                draft_probs.borrow_mut().push(softmax_row(&dl[..v]));
            }
            b
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
                        Ok(bind_bf16(&fp4_client, fp4_aliases.as_ref(), &leaf.bytes, rows, cols))
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
                        anyhow::ensure!(fused.elem == Elem::Bf16, "mtp w13 is {:?}", fused.elem);
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
                            bind_bf16(&fp4_client, fp4_aliases.as_ref(), &dw.bytes, drows, dcols),
                            gs,
                        )
                    };
                    let built = MtpDev {
                        attn: dev_lane::AttnWeightsDev {
                            wq: pw("attn.wq_du.weight", hd.heads * hd.head_dim, h)?,
                            wk: pw("attn.wk_dv.weight", hd.kv_heads * hd.head_dim, h)?,
                            wv: pw("attn.wv_dv.weight", hd.kv_heads * hd.head_dim, h)?,
                            wr: pw("attn.wr_du.weight", hd.heads * hd.d_rel, h)?,
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
                            q_norm: up1(gv(&format!("{p}attn.q_norm.weight"))?, hd.head_dim, &dev),
                            k_norm: up1(gv(&format!("{p}attn.k_norm.weight"))?, hd.head_dim, &dev),
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
                <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("sync after MTP upload");
                println!(
                    "  MTP heads on the device: {mtp_k} in {:.2}s, {:.2} GiB bound",
                    t_up.elapsed().as_secs_f32(),
                    bytes as f64 / GIB
                );
            }
            let main_dev = mtp_main_dev.clone().expect("the entry states were uploaded above");
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
                        None => unreachable!("head d-1 always has more stable rows than head d"),
                    }
                };
                let stable: T2 = if have == 0 {
                    let mut embeds = vec![0f32; want * h];
                    for j in 0..want {
                        let tok = if j + d + 1 < seq { ids[j + d + 1] } else { best };
                        embeds[j * h..(j + 1) * h]
                            .copy_from_slice(&embed_row_bf16(e_w, tok, t.vocab_size, h));
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
                            embed_row_bf16(e_w, tok, t.vocab_size, h),
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
                            mtp_dev_caches[d].as_mut().expect("prefilled on the first pass"),
                            t.rms_norm_eps,
                            mtp_order,
                        ));
                    }
                    if made.len() == 1 {
                        made.pop().expect("one row")
                    } else {
                        BT::cat(made, 0)
                    }
                };
                mtp_stage_dev[d] = Some(match mtp_stage_dev[d].take() {
                    None => stable,
                    Some(prev) => BT::cat(vec![prev, stable], 0),
                });
                let mut rows: Vec<T2> = vec![row_of(&mtp_stage_dev[d], want - 1, want)];
                let mut last = rows[0].clone();
                if d > 0 {
                    let mut scratch =
                        mtp_dev_caches[d].as_ref().expect("prefilled").clone();
                    for i in 0..d {
                        let ed = up2::<Bk>(
                            embed_row_bf16(e_w, drafts[i], t.vocab_size, h),
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
                drafts.push(draft_argmax_dev(last));
                prev_rows = rows;
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
                        let tok = if j + d + 1 < seq { ids[j + d + 1] } else { best };
                        embeds[j * h..(j + 1) * h]
                            .copy_from_slice(&embed_row_bf16(e_w, tok, t.vocab_size, h));
                    }
                    let hin: &[f32] = if d == 0 {
                        &mtp_main[..want * h]
                    } else {
                        &mtp_stage[d - 1][..want * h]
                    };
                    let (y, cache) = mtp_block_prefill(
                        hin, &embeds, &hw, &headw.dims, Some(ls), window, want, mtp_order,
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
                        &embed_row_bf16(e_w, best, t.vocab_size, h),
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
                let mut rows: Vec<Vec<f32>> = vec![mtp_stage[d][(want - 1) * h..want * h].to_vec()];
                let mut last = rows[0].clone();
                if d > 0 {
                    let mut scratch = mtp_caches[d].as_ref().expect("prefilled").clone();
                    for i in 0..d {
                        last = mtp_block_step(
                            &prev_rows[i],
                            &embed_row_bf16(e_w, drafts[i], t.vocab_size, h),
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
        if spec_k == 0 {
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
        acc_draft += t_mtp.elapsed().as_secs_f64();
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
        if kv && std::env::var("INK_MTP_CHECK").map(|val| val == "1").unwrap_or(false) {
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

    println!("\n=== predictions ===");
    println!("  expert slabs decoded: {expert_loads}");
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
    println!("      embed + upload  {:9.1}   (the one host->device crossing per pass)", ms(t_embed));
    println!("      attention half  {:9.1}", ms(t_attn));
    println!("      mlp half        {:9.1}   of which:", ms(t_other));
    println!("        routed experts{:9.1}   (slice + bind + issue)", ms(t_expert));
    println!("        shared experts{:9.1}", ms(t_shared));
    println!("        rest of half  {:9.1}   (dense layers, sconv)",
             ms(t_other - t_expert - t_shared - t_h_route - t_h_sconv));
    println!("      router + group  {:9.1}   BLOCKS: [n,{}] logits back, then top-k on the host",
             ms(t_h_route), t.n_routed_experts + t.n_shared_experts);
    println!("        of which: matmul enqueue {:7.1}, BLOCKING read {:7.1}, top-k + group {:7.1}",
             ms(t_rt_mm), ms(t_rt_read), ms(t_rt_host));
    println!("      mlp short_conv  {:9.1}", ms(t_h_sconv));
    println!("      first-touch uploads: read+widen {:9.1}, transfer {:9.1}   (once per layer, not per token)",
             ms(t_attn_read), ms(t_attn_up));
    println!("    DEVICE, one sync for this node's whole stack: {:9.1}", ms(t_stack_sync));
    if stage_sync {
        println!("    DEVICE per stage (INK_STAGE_SYNC=1 -- {stage_syncs} extra syncs, this pass IS slower for them):");
        println!("      attention half  {:9.1}", ms(d_attn));
        println!("      router matmul   {:9.1}", ms(d_router));
        println!("      routed experts  {:9.1}", ms(d_expert));
        println!("      shared experts  {:9.1}", ms(d_shared));
        println!("      sconv + resid   {:9.1}", ms(d_tail));
        println!("      staged total    {:9.1}", ms(d_attn + d_router + d_expert + d_shared + d_tail));
    }
    println!(
        "    {:17} {:9.1}   ({})",
        if best_wire.is_some() { "tail + wire" } else { "head / unembed" },
        ms(t_head),
        if best_wire.is_some() {
            "BLOCKING: the other machine's layers, its head, and the round trip"
        } else {
            "device"
        }
    );
    println!("    of the above, host-only tensor reads (mmap + BF16 widening): {:9.1}", ms(t_read.get()));
    {
        // What the HOST did in the routed-expert lane, one bucket per kind of
        // work. `read (BLOCKS)` is gone from this list because the read is
        // gone: the accumulator is a device tensor and nothing waits for it.
        println!("    of the routed-expert total, what the host did ({expert_loads} loads):");
        println!("      slice from pile {:9.1}   ({:.3} ms/load)   BLAKE3 on first touch",
                 ms(host_t.slice), ms(host_t.slice) / expert_loads.max(1) as f64);
        println!("      gather (select) {:9.1}   ({:.3} ms/load)   enqueue",
                 ms(host_t.gather), ms(host_t.gather) / expert_loads.max(1) as f64);
        println!("      bind + enqueue  {:9.1}   ({:.3} ms/load)",
                 ms(host_t.enqueue), ms(host_t.enqueue) / expert_loads.max(1) as f64);
        println!("      scatter-add     {:9.1}   ({:.3} ms/load)   enqueue",
                 ms(host_t.accum), ms(host_t.accum) / expert_loads.max(1) as f64);
        // WHICH lane ran, counted rather than asserted. The grouped one is a
        // claim about launches per LAYER and the per-expert one about launches
        // per EXPERT; a per-load average over a mixture of the two is a number
        // about neither, so the split has to be visible beside it.
        println!("      lanes: {} layer(s) GROUPED (one launch per stage), {} per-expert",
                 host_t.grouped, host_t.per_expert);
        let named = host_t.slice + host_t.gather + host_t.enqueue + host_t.drain + host_t.accum;
        println!("      remainder       {:9.1}   (whatever the four above did not cover)",
                 ms(t_expert - named));
    }
    // Rule 2, as a number that should be zero and is. Every one of these was
    // scalar f32 arithmetic over the residual stream, on a CPU, between device
    // calls; none of it was control plane. The line stays in the report BECAUSE
    // it reads zero -- a claim of "no host path in the data plane" that nothing
    // measures is a claim that rots.
    println!("    HOST DATA PLANE in the block itself, ms (want: zero):");
    println!("      rms_norm        {:9.1}", ms(t_h_norm));
    println!("      residual adds   {:9.1}", ms(t_h_resid));
    println!("      expert gather   {:9.1}   (a `select` on the device now)", 0.0);
    println!("      expert accum    {:9.1}   (a `select_assign` on the device now)", 0.0);
    println!("      TOTAL           {:9.1}", ms(t_h_norm + t_h_resid));
    let (calls, hits, fileb, hostb, loader_ns) = cp.io_totals();
    let (rb, rn) = cp.resident_bytes();
    println!("  what this ONE pass moved:");
    println!("    loader reads        {calls:8}   answered from RAM {hits:8}");
    println!("    stored bytes        {:8.2} GiB   (what the reads touched, stored precision)",
             fileb as f64 / GIB);
    println!("    host f32 bytes      {:8.2} GiB   (what they became after widening)",
             hostb as f64 / GIB);
    println!("    seconds in loader   {:8.1}", loader_ns as f64 / 1e9);
    println!("    disk read_bytes     {:8.2} GiB   (/proc/self/io -- page-cache hits are free)",
             (io_read_bytes() - io0) as f64 / GIB);
    println!("    resident set        {:8.2} GiB in {rn} weights  (host)", rb as f64 / GIB);
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
    report_align();
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
        let valid_to =
            if verify_rows > 1 || probe_rows > 1 { logit_row0 + new_toks.len() } else { n };
        for ti in logit_row0..valid_to {
            let pos = pos0 + ti;
            let row = &logits[(ti - logit_row0) * v..(ti - logit_row0 + 1) * v];
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            let top: Vec<usize> = idx[..5].to_vec();
            println!("  after token {pos} (id {}): top5 {:?}  logits {:?}",
                     ids[pos], top,
                     top.iter().map(|&i| (row[i] * 100.0).round() / 100.0).collect::<Vec<_>>());
            for &i in &top {
                top_all.push(i as i64);
            }
        }
    }

    if gen_steps > 0 {
        println!("  step {step}: +{new_toks:?}   [pass {:.1}s, total {:.1}s, ctx {}, pass_ms {:.1}]",
                 pass.elapsed().as_secs_f32(), started.elapsed().as_secs_f32(), ids.len(),
                 pass.elapsed().as_secs_f64() * 1e3);
        // The tail already pushed all but the last, when it answered its peer.
        if !is_tail && !repeat {
            // Not in the slot lane: `new_toks` is one token per SLOT there, not
            // an accepted prefix of one sequence, and extending `ids` with all
            // of them puts seven other sequences into slot 0's stream. It is
            // only a report -- nothing computes off `ids` in that lane -- which
            // is exactly why it read as a plausible context length (4052
            // against 3780) instead of as a failure.
            if new_toks.len() > 1 && !slot_lane {
                ids.extend_from_slice(&new_toks[..new_toks.len() - 1]);
            }
            ids.push(best);
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
    }
    if is_decode && step - prefill_passes >= COLD_DECODE_STEPS {
        warm_wall += pass.elapsed().as_secs_f64() + t_recv;
        warm_steps += 1;
        warm_tokens += new_toks.len();
    }
    // A slot lane's prefill passes produce a token each and they are not decode
    // tokens: counting them would put b of them in the numerator of a rate
    // whose denominator is decode passes only.
    gen_tokens += if slot_lane && !is_decode { 0 } else { new_toks.len() };
    // Tokens, not passes -- and with nothing speculated this fires on exactly
    // the pass the old `for step in 0..=gen_steps` fired on.
    // Passes, in the slot lane. Every arm has to run the same number of decode
    // passes for the per-pass cost to be comparable across b, and a token count
    // would make the run b times shorter at b times the width.
    let done =
        if slot_lane { is_decode && step + 1 - prefill_passes >= gen_steps } else { gen_tokens > gen_steps };
    if done {
        break;
    }
    step += 1;
    }

    // ---- how much of the wall clock this node spent waiting for the other --
    if acc_steps > 0 && pipe.is_some() {
        let ms = |v: f64| v * 1e3;
        let wall = acc_pass + acc_recv;
        println!("\n=== pipe utilisation over {acc_steps} decode steps (prefill excluded) ===");
        println!("  role                 : {}", if is_head { "head" } else { "tail" });
        println!("  wall in the loop     : {:9.1} ms   ({:.1} ms/step)", ms(wall), ms(wall) / acc_steps as f64);
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
            println!("  computing            : {:9.1} ms   {:5.1}%", ms(compute), 100.0 * compute / wall);
            println!("  writing to the wire  : {:9.1} ms   {:5.1}%", ms(acc_send), 100.0 * acc_send / wall);
            println!("  BLOCKED on the tail  : {:9.1} ms   {:5.1}%", ms(acc_wait_peer), 100.0 * acc_wait_peer / wall);
            println!("  per step: compute {:.1} ms, blocked {:.1} ms",
                     ms(compute) / acc_steps as f64, ms(acc_wait_peer) / acc_steps as f64);
        } else {
            println!("  computing            : {:9.1} ms   {:5.1}%", ms(acc_pass), 100.0 * acc_pass / wall);
            println!("    of which drafting  : {:9.1} ms   {:5.1}%", ms(acc_draft), 100.0 * acc_draft / wall);
            println!(
                "  ANSWERED the head at : {:9.1} ms into its pass ({:.1} ms/step) -- everything\n  \
                 after that (report, drafting) overlaps the head's next pass and the head\n  \
                 never waits for it. Subtract THIS from the head's blocked figure for the wire.",
                ms(acc_to_reply),
                ms(acc_to_reply) / acc_steps as f64
            );
            println!("  BLOCKED on the head  : {:9.1} ms   {:5.1}%", ms(acc_recv), 100.0 * acc_recv / wall);
            println!("  per step: compute {:.1} ms, blocked {:.1} ms",
                     ms(acc_pass) / acc_steps as f64, ms(acc_recv) / acc_steps as f64);
        }
        // ---- THE gate ------------------------------------------------------
        //
        // Tokens per second, not milliseconds per pass. Speculation trades more
        // compute for fewer sequential steps, so a per-pass figure is SUPPOSED
        // to get worse; the only number that says whether the trade paid is how
        // much text came out per second of wall clock.
        // The prefill's own token is in `gen_tokens` on the single-sequence
        // lane and is not a decode token; the slot lane never counted it.
        let decode_toks = if slot_lane { gen_tokens } else { gen_tokens.saturating_sub(1) };
        let mut sorted = pass_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration"));
        let p50 = if sorted.is_empty() { 0.0 } else { sorted[sorted.len() / 2] };
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
        // The same gate, taken at the MEDIAN pass instead of the mean. A decode
        // loop pays for kernel compilation in its first few passes -- one 1.8 s
        // outlier in a 40-pass run moves the mean by 20% and the median by
        // nothing -- so this is the figure a short run can be compared on and
        // the mean above is the figure a long one can.
        println!(
            "  TOKENS/SEC at p50    : {:.3}   (tokens/pass over the median pass)",
            if p50 > 0.0 { tpp / (p50 / 1e3) } else { 0.0 }
        );
        if spec_k > 0 {
            let sets: usize = spec_hist.iter().sum();
            println!("  accepted prefix over {sets} verify passes (INK_SPEC={spec_k}):");
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
        println!("  loop wall (both ends see the same clock): {:.1} ms", ms(loop_started.elapsed().as_secs_f64()));
        println!("  the wire itself is the head's BLOCKED figure minus the tail's per-step compute;");
        println!("  neither process can subtract that on its own, so the two reports are read together.");
    }

    // ---- what the router arm changed, if anything -------------------------
    //
    // Printed whether or not it found something, with the examined count on
    // every line. A zero that says how many selections it looked at is a
    // measurement; a zero on its own is a claim.
    if router_diff {
        println!("\n=== router selection: {} vs the f32 [rows,hidden] lane ===", router_arm.label());
        println!("  layer   examined   set!=   order!=   slots!=   max|dlogit|   max|dweight|");
        let (mut ex, mut sd, mut od, mut sl) = (0usize, 0usize, 0usize, 0usize);
        let (mut ml, mut mw) = (0f32, 0f32);
        for (layer, d) in route_diff.iter().enumerate() {
            if d.examined == 0 {
                continue;
            }
            println!(
                "  {layer:5}   {:8}   {:5}   {:7}   {:7}   {:11.3e}   {:12.3e}",
                d.examined, d.set_differs, d.order_differs, d.slots_differ,
                d.max_abs_logit, d.max_abs_weight
            );
            ex += d.examined;
            sd += d.set_differs;
            od += d.order_differs;
            sl += d.slots_differ;
            ml = ml.max(d.max_abs_logit);
            mw = mw.max(d.max_abs_weight);
        }
        if ex == 0 {
            println!("  nothing examined: this node's slice has no MoE layer, so there was no router to compare.");
        } else {
            println!(
                "  TOTAL   {ex:8}   {sd:5}   {od:7}   {sl:7}   {ml:11.3e}   {mw:12.3e}"
            );
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
            let complete: Vec<&Vec<Option<bool>>> =
                mtp_issued.values().filter(|v| v.iter().all(|s| s.is_some())).collect();
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
                    println!("    {l} accepted: {c:5}   ({:5.1}%)", 100.0 * c as f64 / sets as f64);
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
            let complete_q: Vec<&Vec<Option<(f64, f64)>>> =
                mtp_issued_q.values().filter(|v| v.iter().all(|s| s.is_some())).collect();
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
                println!("    rule B: {:.3} draft tokens per verify pass", eb / sets as f64);
                println!("    rule C: {:.3} draft tokens per verify pass", ec / sets as f64);
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
        let _ = send_stream(s, 0, 0, &[]);
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
    println!("  wrote top-5 ids per position to {}", out_path.display());
    Ok(())
}
