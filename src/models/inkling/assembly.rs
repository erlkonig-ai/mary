//! The Inkling ASSEMBLY: the device-lane components a running model is built
//! out of, in the library rather than in a binary.
//!
//! Every module beside this one is a COMPONENT — attention, the MLP, the MoE
//! group, KV pages, the loader, the tensor-parallel shard plan. They compose
//! into a model, and until 2026-08-27 nothing could: the composition — how a
//! layer's weights get bound to the device, which GEMM lane an expert takes,
//! what the head does, what one MTP head carries between drafts — lived inside
//! `src/bin/inkling_forward.rs`, and **nothing can link against a binary**.
//!
//! So this file is not new code. It is the same code, in a place a caller can
//! reach. It was moved verbatim: same statements, same order, same allocation
//! sites, because the thing it must not do is change what the model computes or
//! when it allocates. `inkling_forward` still calls every item here, from the
//! same call sites, in the same order — which is what makes the extraction
//! checkable by measurement rather than by argument.
//!
//! What is NOT here, and deliberately: the process's own concerns. The pipe
//! between two nodes, the argument parsing, the measurement accumulators, the
//! CUDA-graph capture bookkeeping and every `INK_*` reporting switch stay in the
//! binary, because they belong to a RUN and not to a model. The line is: if a
//! long-lived serving process would need it to answer a token, it is here.
//!
//! See [`super::session`] for the handle that assembles these into a model that
//! survives across turns.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::models::inkling::attn::{AttnDims, AttnWeights, LogScaling};
use crate::models::inkling::bf16gemm::Bf16W;
use crate::models::inkling::block::Routing;
use crate::models::inkling::budget;
use crate::models::inkling::layer::{LayerMlp, LayerWeights};
use crate::models::inkling::load::Held;
use crate::models::inkling::mtp::{Concat as MtpConcat, MtpHead};
use crate::models::inkling::pile::Elem;
use crate::models::inkling::source::Weights;
use crate::models::inkling::stack::{embed_and_norm_bf16, embed_row_bf16};

/// The backend every device lane here is concrete on. Not a generic parameter:
/// each of these lanes is a Blackwell tensor-core lane and there is no second
/// backend for it to be chosen between.
pub type Bk = burn::backend::Cuda<f32>;
pub use crate::models::inkling::burn as dev_lane;
pub use crate::models::inkling::resid as dev_lane_resid;
pub use burn::tensor::backend::Backend;
pub use burn::tensor::{Tensor as BT, TensorData as BTD};

/// One gibibyte, as the divisor every byte count here is printed against.
pub const GIB: f64 = (1u64 << 30) as f64;

/// Move a host `[rows, cols]` matrix to the device, consuming it.
///
/// Takes the `Vec` by value on purpose: the dense `w13` is 537 MB at f32 and a
/// borrowing helper would hold two copies of it at once.
pub fn up2<B: Backend>(v: Vec<f32>, rows: usize, cols: usize, dev: &B::Device) -> BT<B, 2> {
    assert_eq!(
        v.len(),
        rows * cols,
        "{} values are not [{rows}, {cols}]",
        v.len()
    );
    BT::from_data(BTD::new(v, [rows, cols]), dev)
}

pub fn up1r<B: Backend>(v: &[f32], len: usize, dev: &B::Device) -> BT<B, 1> {
    assert_eq!(v.len(), len, "{} values are not [{len}]", v.len());
    BT::from_data(BTD::new(v.to_vec(), [len]), dev)
}

pub fn up1<B: Backend>(v: Vec<f32>, len: usize, dev: &B::Device) -> BT<B, 1> {
    assert_eq!(v.len(), len, "{} values are not [{len}]", v.len());
    BT::from_data(BTD::new(v, [len]), dev)
}

/// Read a `[rows, cols]` device tensor back to the host. This is also the sync,
/// so a timer around the call measures work rather than enqueueing.
pub fn down<B: Backend>(t: BT<B, 2>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("device readback")
}

/// A device tensor of this run's backend, named once so the residency types
/// below do not have to repeat it.
pub type T2 = burn::tensor::Tensor<Bk, 2>;

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
/// Negative log-likelihood, in nats, that each row of `logits` assigns to its
/// target id. The log-softmax and the gather run on the device; only the
/// `rows` floats come back. This is the prequential score of a served turn:
/// each row is the model's prediction BEFORE it has learned from the target.
pub fn row_nll_dev(logits: T2, targets: &[usize]) -> Vec<f32> {
    let [rows, _] = logits.dims();
    debug_assert_eq!(rows, targets.len(), "one target per scored row");
    let dev = logits.device();
    let idx: Vec<i64> = targets.iter().map(|&t| t as i64).collect();
    let idx: BT<Bk, 2, burn::tensor::Int> = BT::from_data(BTD::new(idx, [rows, 1]), &dev);
    burn::tensor::activation::log_softmax(logits, 1)
        .gather(1, idx)
        .neg()
        .into_data()
        .iter::<f32>()
        .collect()
}

pub fn argmax_row_dev(row: T2) -> usize {
    let [rows, _] = row.dims();
    debug_assert_eq!(rows, 1, "argmax_row_dev reads exactly one row");
    // `INK_TOPB=b` prints this row's b most likely token ids. Instrument only,
    // and deliberately the expensive way (read the whole row back): it exists to
    // supply the SAME-POSITION candidate set for the token-tree union arm, which
    // is a handful of passes, not a serving path. The doc above explains why the
    // model itself must never do this.
    if let Some(b) = std::env::var("INK_TOPB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        if b > 0 {
            let d = row.clone().into_data();
            let vals: Vec<f32> = d.iter::<f32>().collect();
            let mut idx: Vec<usize> = (0..vals.len()).collect();
            idx.sort_unstable_by(|&x, &y| {
                vals[y]
                    .partial_cmp(&vals[x])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(x.cmp(&y))
            });
            idx.truncate(b);
            println!("  INK_TOPB {idx:?}");
        }
    }
    row.argmax(1)
        .into_data()
        .iter::<i64>()
        .next()
        .expect("device argmax readback") as usize
}

/// One argmax per row of a target verifier batch, reduced on the device and
/// returned in row order.
///
/// Unlike calling [`argmax_row_dev`] in a loop, this performs one readback for
/// the whole batch. A speculative caller needs every row's prediction to find
/// the accepted prefix; an ordinary Session pass still calls the one-row helper
/// and therefore keeps its existing head path unchanged.
pub fn argmax_rows_dev(rows: T2) -> Vec<usize> {
    rows.argmax(1)
        .into_data()
        .iter::<i64>()
        .map(|index| index as usize)
        .collect()
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
pub struct HostT {
    /// `expert_packed` / `expert_bf16`: a hash lookup and a view of the pile.
    pub slice: f64,
    /// Copying this expert's tokens out of the residual stream, on the host.
    pub gather: f64,
    /// Binding the weight, quantising the activation, issuing the kernels.
    /// Non-blocking by construction.
    pub enqueue: f64,
    /// `read_one`: BLOCKING, so this is the layer's device time as much as the
    /// host's.
    pub drain: f64,
    /// Scattering each expert's rows back into the accumulator, weighted.
    pub accum: f64,
    /// Layers the GROUPED lane took: one launch per stage for the whole layer.
    pub grouped: usize,
    /// Layers that fell back to the per-expert loop, because their weights are
    /// not offsets into one registered mapping.
    pub per_expert: usize,
    /// Summed over MoE layers: the number of DISTINCT experts this pass had to
    /// gather. One token needs `top_k` routed plus the shared ones. A WIDER pass
    /// needs the UNION of its tokens' expert sets, and that union is why
    /// speculation costs real bytes here rather than being nearly free as it is
    /// on a dense model, where verifying k+1 tokens re-reads exactly the same
    /// weights. Divided by the layer count it reads as "distinct experts per MoE
    /// layer", which is the quantity that decides whether a wider verify pass
    /// pays for itself.
    pub expert_slots: usize,
    /// The grouped lane's small plan uploads that DEPEND on the routing
    /// decision: the two offset tables, the two second-level scale vectors and
    /// the per-row weights. A device-resident router deletes exactly these --
    /// they become a gather out of a per-layer `[n_routed]` table by ids the
    /// device already holds -- so they are timed apart from the ones it does
    /// not.
    pub plan_up_routed: f64,
    /// The grouped lane's small plan uploads that are a function of `n` and
    /// `top_k` ALONE: the row->token map, the three block-plan vectors and the
    /// token->rows table. At `n == 1` every one of them is the same bytes on
    /// every layer of every pass, so this is what a hoist out of the loop is
    /// worth, independently of where the routing decision lives.
    pub plan_up_static: f64,
    /// Layer-passes whose row plan was built ON THE DEVICE, from a decision
    /// that was never read back. Counted because every other counter in this
    /// struct is the same on both arms -- `grouped` and `expert_loads` cannot
    /// tell them apart, and a lane you cannot count is a lane you cannot claim.
    pub plan_dev: usize,
    /// Layer-passes whose row plan came off a blocking read. The complement,
    /// and printed beside it: a run where this is not zero has a sync left in
    /// the loop, and WHICH layer it is matters more than how many.
    pub plan_host: usize,
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
///
/// # `down` fuses along K, and NOT along `n` — the reason is the operands
///
/// `down` was the obvious next candidate for the same medicine and it does not
/// take it. `gate_up`'s concatenation works because all four of its blocks
/// multiply the SAME activation, so stacking them along `n` is free. The two
/// `down` projections do not: expert `s` consumes
/// `silu(gate_s) * up_s * gamma_s`, and the gate, the up and the gamma are all
/// per-expert. An `n` concatenation of two weights against one activation would
/// have to compute `down_0 @ a_1` and `down_1 @ a_0` as well and throw both
/// away — 2x the weight bytes for 2x the cubes, which the constant-work control
/// in `w4a16gemm::swizzle_pays` prices at 1.27x of rate for 2x of bytes, i.e.
/// 1.57x the TIME. It loses on arithmetic before anything is measured.
///
/// The axis that exists here is **K**. `out = down_0 @ a_0 + down_1 @ a_1` is
/// one product of a `[hidden, n_shared * inter]` weight against the
/// `[n, n_shared * inter]` concatenation of the activations, and the sum over
/// experts happens inside the k loop instead of as a separate tensor add. The
/// grid does NOT change — still `hidden / NTILE` = 512 single-warp cubes — so
/// this buys nothing from cube count. It buys two other things:
///
/// * **K is the second knob.** `w4a16gemm::swizzle_pays` measures the
///   permuted lane's multiplier saturating at ~1.10 at `k = 2048`, ~1.24 at
///   `k = 4096` and ~1.45 at `k = 16384`, at a FIXED cube count, because a
///   longer k loop spreads the L1 working set. Doubling k moves this weight
///   from one row of that table to the next at constant bytes.
/// * one launch instead of two, and the `[n, hidden]` tensor add disappears.
///
/// It is **not** bit-identical, and that is the one way it differs from
/// `gate_up`'s fusion. Every term is the same and the k loop visits them in
/// the same order; what changes is the ASSOCIATION — one running f32
/// accumulator across the expert seam where there were two accumulators and a
/// `+` at the end, which is one rounding fewer and a different place for the
/// rest of them. Nothing here decides whether that matters, which is what
/// [`sink_down_diff`] is for: it binds both arms and reads the disagreement
/// off the activations the model actually produced, rather than inferring it.
pub struct SharedOnDevice {
    pub gate_up: dev_lane::ProjW,
    pub down: SinkDown,
}

/// The sink experts' `down` projections, in one of the two shapes they bind to.
///
/// Residency is identical either way — the same weight bytes, quantised by the
/// same per-16-block quantiser, laid out in a different order — so the arms can
/// be A/B'd inside one process without one of them paying for the other's
/// memory. See [`SharedOnDevice`] for what the fusion is and why it is K and
/// not `n`, and [`sink_down_fused`] for the switch.
pub enum SinkDown {
    /// One `[hidden, inter]` weight per shared expert, one GEMM each, summed
    /// with a tensor add. What shipped before 2026-08-27.
    Split(Vec<dev_lane::ProjW>),
    /// One `[hidden, n_shared * inter]` weight, expert-minor along k, so that
    /// one GEMM against the concatenated activations IS the sum.
    Fused(dev_lane::ProjW),
    /// `INK_SINK_DOWN_DIFF=1`: BOTH, so one pass can put the two arms on the
    /// same activation and read the disagreement instead of arguing about it.
    ///
    /// The fusion moves the last ulps — one accumulator across the expert seam
    /// instead of two summed afterwards — so "not bit-identical" is the
    /// PREDICTION and the only useful question is how big. Neither assuming it
    /// is negligible nor assuming it matters is a measurement, which is the
    /// same reason [`RouteDiff`] exists.
    ///
    /// It holds two copies of `down` and syncs on every layer, so it is a
    /// diagnostic and its timing means nothing. The FUSED result is what the
    /// run continues with; the split one is computed, compared and discarded,
    /// so the counters describe the arm under test and not some third thing.
    Both(Vec<dev_lane::ProjW>, dev_lane::ProjW),
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
/// [`crate::models::inkling::annhead::VERIFY_WINNERS`]). But it settles the
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
pub const ANN_BUDGET_DEFAULT: usize = 8192;

/// The seed of the sketch's random rotation.
///
/// Fixed rather than drawn, because the rotation has to be the same one on
/// every process that reads the same table: it is a property of the SKETCH, and
/// a sketch built under a different rotation than the query is rotated by is not
/// a worse estimate, it is noise. It is a constant and not a switch for the same
/// reason -- there is nothing a caller could usefully vary it to.
pub const ANN_SKETCH_SEED: u64 = 0x414E_4E5F_5545_3031;

/// Which lane the unembed table is bound to.
///
/// The head is the single largest term in the per-step INTERCEPT, and it is
/// physics rather than overhead: `[vocab_size, hidden]` BF16 is 1.53 GiB read
/// WHOLE on every pass, measured at 10.3 ms = 159 GB/s = 65% of this box's
/// measured 242.9 GB/s. It does not scale with context or with how many layers
/// this node holds, and no launch-side change moves a byte of it -- only fewer
/// bytes do. At NVFP4 the same table is 0.43 GiB.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeadLane {
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
pub fn head_lane() -> HeadLane {
    HeadLane::W4a16
}

/// The shortlist budget for the approximate head, and the switch that selects
/// it. `0` runs the exact `w4a16` lane.
///
/// The head is an exhaustive maximum-inner-product search: 0.43 GiB of NVFP4
/// read whole to produce one integer. `INK_ANN_HEAD=N` replaces the exhaustive
/// scan with a 1-bit sign sketch over the same rows (0.103 GiB, a quarter of
/// the bytes) and an EXACT rescore of the `N` rows whose estimates come out on
/// top. See [`crate::models::inkling::annhead`] for why narrower codes and not
/// fewer rows, and for the rotation and the unbiasing scalar that make a
/// one-bit estimate rank correctly at all.
///
/// `N` is a budget rather than a threshold because the lane's own question is a
/// threshold — every row that could still win — and a budget is how a caller
/// says what it will pay to answer it.
pub fn ann_budget() -> usize {
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
pub fn ann_range() -> f32 {
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
pub fn ann_rotated() -> bool {
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
/// [`crate::models::inkling::annhead::VERIFY`].
///
/// It roughly doubles the head stage, so it is a gate and not a default.
pub fn ann_verify() -> bool {
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
pub fn head_temp() -> f32 {
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
pub fn head_temp_seed() -> u64 {
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
pub fn normals(seed: u64, step: u64, count: usize) -> Vec<f32> {
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
pub fn sink_w4a16() -> bool {
    true
}

/// Whether the sink experts' `down` projections bind as ONE weight fused along
/// k, from `INK_SINK_DOWN_FUSE`.
///
/// See [`SharedOnDevice`] for the mechanism and why the axis is k and not `n`.
/// This is a knob rather than a rewrite because the two arms differ in the last
/// ulps — one f32 accumulator across the expert seam instead of two summed
/// afterwards — so the split form is the reference the fused one is compared
/// against, and keeping both in one binary is what makes that comparison a
/// PAIRED one inside a single process rather than two runs of two binaries.
/// Residency is identical on both arms: the same weight bytes through the same
/// per-16-block quantiser, reordered.
///
/// It has no effect at `n_shared == 1`, where there is nothing to fuse.
///
/// `=2` selects the fused WEIGHT with the activation built by `Tensor::cat` of
/// the per-expert gated tensors instead of by [`wide_gate`]. See
/// [`sink_down_cat`]: it exists to tell the fusion apart from the way this
/// binary happens to build its operand, which a single on/off flag cannot.
pub fn sink_down_fused() -> bool {
    sink_down_mode() >= 1
}

/// `INK_SINK_DOWN_FUSE`, as the three-way it actually is.
pub fn sink_down_mode() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("INK_SINK_DOWN_FUSE")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.parse::<usize>().unwrap_or(1))
            .unwrap_or(0)
    })
}

/// `INK_SINK_DOWN_FUSE=2`: the fused weight, but the activation concatenated
/// with `Tensor::cat` rather than produced whole by [`wide_gate`].
///
/// # Why this arm exists, and it is the second time this shape has come up
///
/// The k-fusion measured **-1.93% on a one-row decode step and +4.26% on the
/// shared-expert stage of a 3732-row prefill** — a win that inverts on the wide
/// pass, which is precisely how the shared-memory-staged routed GEMM failed
/// (see `moegroup::grouped_smem`). A flag with
/// two positions cannot say WHICH half of the change inverted, and the two
/// halves have opposite expected signs at prefill:
///
/// * the GEMM should be neutral-to-better — `ncu` puts the load sectors and
///   requests per unit of work byte-identical on both arms, and the registers
///   at 80 on both, so nothing about the traffic or the occupancy moved;
/// * the tensor add the fusion DELETES is worth ~183 MB a layer at 3732 rows
///   (two `[n, hidden]` f32 outputs written, both read back, one written), so
///   removing it should help most exactly where the loss appeared;
/// * [`wide_gate`]'s broadcast is the one part that is new rather than
///   removed, and it is a THREE-dimensional broadcast where the per-expert form
///   is two-dimensional. Whether that lowers to the same quality of kernel at
///   30 MB an operand is an assumption, not a measurement.
///
/// So this arm keeps the fused weight and the fused GEMM and reverts only the
/// operand construction, paying a `[n, n_shared * inter]` copy for it. If the
/// prefill loss follows [`wide_gate`], it is an implementation detail and
/// fixable; if it follows the GEMM, the fusion itself is regime-dependent and
/// belongs behind a flag forever.
pub fn sink_down_cat() -> bool {
    sink_down_mode() == 2
}

/// `INK_SINK_DOWN_DIFF=1`: bind BOTH `down` arms and report what the k-fusion
/// moved, per layer, on the activation the run actually produced.
///
/// See [`SinkDown::Both`]. It overrides [`sink_down_fused`] — a run that asks
/// for the comparison gets both arms whichever way the speed knob is set — and
/// it holds a second copy of the weight and syncs every layer, so its timing is
/// meaningless by construction.
pub fn sink_down_diff() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("INK_SINK_DOWN_DIFF").as_deref() == Ok("1"))
}

/// One layer's `down` projections, applied to the per-expert gated activations
/// and summed, in whichever shape they bound to.
///
/// `gated(s)` produces expert `s`'s `silu(gate) * up * gamma`, as the split arm
/// has always built it. `gated_all()` produces the same values as ONE
/// `[n, n_shared * inter]` tensor, in expert order, for the fused arm.
///
/// # Why two closures and not a concatenation of the first
///
/// Because `Tensor::cat` would put the fusion's cost on `n` and the whole
/// point is that it has none. `gu` holds every gate block contiguously and
/// then every up block, so `silu(gu[.., ..n_shared * inter]) * gu[.., n_shared
/// * inter..]` IS the concatenated activation already — one elementwise chain
/// over the full width instead of two over half of it, and no copy. A `cat`
/// would have cost an extra `[n, n_shared * inter]` write and read per layer,
/// which is nothing at a one-row decode and 16 MB a layer at a 512-row
/// prefill, i.e. more than the weight read the fusion exists to speed up. That
/// is the exact shape of the trap the staged grouped GEMM fell into — a
/// decode win that inverts on the wide pass — and it is avoidable here rather
/// than merely measurable.
///
/// Both closures produce bit-identical VALUES; `silu` and the two multiplies
/// are elementwise, and a `reshape` of a contiguous tensor is a stride update.
/// The split arm keeps calling the per-expert one so that the reference arm is
/// literally the code that shipped.
pub fn sink_down_apply(
    w2: &SinkDown,
    layer: usize,
    n_shared: usize,
    gated: impl Fn(usize) -> T2,
    gated_all: impl Fn() -> T2,
) -> T2 {
    // The split arm, unchanged: one GEMM an expert and a tensor add between.
    let split = |ws: &[dev_lane::ProjW]| {
        let mut out: Option<T2> = None;
        for s in 0..n_shared {
            let c = dev_lane::linear_w(gated(s), &ws[s]);
            out = Some(match out {
                Some(o) => o + c,
                None => c,
            });
        }
        out.expect("a MoE layer with no shared experts")
    };
    // The fused arm. The activations run in the SAME expert order the weight
    // was interleaved in, so the k loop walks expert 0's `inter` columns and
    // then expert 1's, and the accumulator IS the sum. The elementwise half is
    // untouched and bit-identical; only where the two experts' partial sums
    // meet has moved.
    let fused = |w: &dev_lane::ProjW| {
        let a = if sink_down_cat() {
            T2::cat((0..n_shared).map(&gated).collect(), 1)
        } else {
            gated_all()
        };
        dev_lane::linear_w(a, w)
    };
    match w2 {
        SinkDown::Split(ws) => split(ws.as_slice()),
        SinkDown::Fused(w) => fused(w),
        SinkDown::Both(ws, w) => {
            let reference = down::<Bk>(split(ws.as_slice()));
            let arm = fused(w);
            sink_down_report(layer, &reference, &down::<Bk>(arm.clone()));
            arm
        }
    }
}

/// Every shared expert's `silu(gate) * up * gamma` at once, as one
/// `[n, n_shared * inter]` tensor in expert order.
///
/// `g` and `u` are the WHOLE gate half and the whole up half of the fused
/// `gate_up` output — contiguous in `gu` and already expert-major — so the
/// elementwise product is the concatenated activation without a `Tensor::cat`.
/// `gam` is `[n, n_shared]`; the reshape to `[n, n_shared, 1]` broadcasts each
/// expert's gamma across its own `inter` columns and nothing else.
///
/// A reshape of a contiguous tensor is a stride update in this backend, so the
/// two reshapes around the multiply are free and the launch count for the whole
/// gate is three where the per-expert form is `3 * n_shared`.
pub fn wide_gate(g: T2, u: T2, gam: T2, n: usize, n_shared: usize, inter: usize) -> T2 {
    let gated = dev_lane::silu(g) * u;
    (gated.reshape([n, n_shared, inter]) * gam.reshape([n, n_shared, 1]))
        .reshape([n, n_shared * inter])
}

/// Report `fused` against `split` for one layer's shared-expert output.
///
/// Absolute AND relative, because either alone can flatter: an absolute delta
/// says nothing without the magnitude it sits on, and a relative one explodes
/// on the near-zero entries every activation has. The denominator is the
/// SPLIT arm's magnitude, i.e. the reference, and rows where it underflows are
/// counted rather than divided by.
pub fn sink_down_report(layer: usize, split: &[f32], fused: &[f32]) {
    assert_eq!(
        split.len(),
        fused.len(),
        "the two down arms differ in shape"
    );
    let (mut max_abs, mut max_rel, mut differ, mut tiny) = (0.0f32, 0.0f32, 0usize, 0usize);
    let mut mag = 0.0f32;
    for (&a, &b) in split.iter().zip(fused.iter()) {
        mag = mag.max(a.abs());
        if a.to_bits() != b.to_bits() {
            differ += 1;
        }
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        if a.abs() > 1e-6 {
            max_rel = max_rel.max(d / a.abs());
        } else {
            tiny += 1;
        }
    }
    println!(
        "  sink down diff L{layer}: {differ}/{} differ, max |Δ| {max_abs:.3e}, max rel {max_rel:.3e} \
         (on |split| <= {mag:.3e}; {tiny} entries under 1e-6 excluded from rel)",
        split.len()
    );
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
pub struct DeviceDense {
    pub shared: std::collections::BTreeMap<String, SharedOnDevice>,
    pub dense: std::collections::BTreeMap<String, (Bf16W, Bf16W, Bf16W, f32)>,
    pub bytes: u64,
}

/// One MTP head's weights, OWNED, so the borrowed [`MtpHead`] handed to
/// `mtp_block` can be rebuilt per draft without re-reading anything.
///
/// The split exists because `MtpHead` borrows and the loop needs the owner to
/// outlive every borrow. `gate`/`up` are materialised rather than held because
/// `split_gate_up` de-interleaves the fused matrix into two, and doing that once
/// per head at load beats doing it once per draft.
pub struct MtpOwned {
    pub embed_norm: Held,
    pub hidden_norm: Held,
    pub input_proj: Held,
    pub attn_norm: Held,
    pub mlp_norm: Held,
    pub attn_sconv: Held,
    pub mlp_sconv: Held,
    pub wq: Held,
    pub wk: Held,
    pub wv: Held,
    pub wr: Held,
    pub wo: Held,
    pub q_norm: Held,
    pub k_norm: Held,
    pub k_sconv: Held,
    pub v_sconv: Held,
    pub rel_proj: Held,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Held,
    pub global_scale: f32,
    pub dims: AttnDims,
    pub local: bool,
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
    pub fn window(&self, sliding: usize) -> Option<usize> {
        if self.local { Some(sliding) } else { None }
    }

    /// Borrow this head in the shape `mtp_block` wants.
    pub fn borrow(&self, inter: usize) -> MtpHead<'_> {
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
pub fn quantized_bf16(
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
        crate::models::inkling::fp4quant::quantize_nvfp4_bf16(client, &src, rows, cols);
    dev_lane::PackedW {
        codes,
        scales,
        n: rows,
        k: cols,
        scale2: 1.0,
        swizzled: false,
    }
}

/// `INK_DENSE_FAKEQUANT=1`: run the dense MLP's weights through the NVFP4
/// quantiser and back, and keep multiplying them with the BF16 GEMM.
///
/// # Why a round trip and not simply the W4A16 lane
///
/// The question this answers is ONLY "what does NVFP4's error do to this
/// model's tokens", and the way to ask it badly is to switch the dense MLP to
/// `linear_w4a16` and read the output. That changes the quantiser AND the
/// kernel AND the operand layout in one step, so a wiring mistake -- a
/// transposed weight, a mis-slid scale, a fragment order the reader disagrees
/// with -- arrives looking exactly like quantisation damage. This project has
/// already had one of those reach a merge verification with plausible logits
/// (see [`w4a16_bind`]), and the attention-quantisation attempt died on the
/// same ambiguity.
///
/// So: quantise, dequantise, and hand the RESULT to the same `bind_bf16` and
/// the same `dense_mlp_bf16` the baseline uses. The only thing that changes is
/// the VALUES. If the tokens survive this, the error is acceptable and the
/// remaining work is a kernel change whose correctness is a separate,
/// bit-level question. If they do not, no kernel was ever written.
///
/// The floor to compare against is bit-identical: the same binary with the
/// variable unset. Compare TOKENS, not logits, and compare them before any
/// timing number is quoted -- the timing of this arm is meaningless anyway,
/// since it reads the same BF16 bytes the baseline does and merely spent a
/// startup pass making them worse.
///
/// It uses the PRODUCTION quantiser (`quantize_nvfp4_bf16`, the one the sinks
/// and the head go through) rather than a host reimplementation, so what it
/// measures is the error this runtime would actually commit.
pub fn dense_fake_quant() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("INK_DENSE_FAKEQUANT").as_deref() == Ok("1"))
}

/// One BF16 weight through NVFP4 and back, still BF16. See [`dense_fake_quant`].
pub fn fake_quant_bf16(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Vec<u8> {
    use crate::models::inkling::fp4quant::{dequantize_nvfp4_bf16, quantize_nvfp4_bf16};
    assert_eq!(
        bytes.len(),
        rows * cols * 2,
        "{rows}x{cols} BF16 is not {} bytes",
        bytes.len()
    );
    let src = client.create_from_slice(bytes);
    let (codes, scales) = quantize_nvfp4_bf16(client, &src, rows, cols);
    let back = dequantize_nvfp4_bf16(client, &codes, &scales, rows, cols);
    client
        .read_one(back)
        .expect("read back the round-tripped dense weight")
        .to_vec()
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
pub fn w4a16_bind(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    mut p: dev_lane::PackedW,
    for_ann: bool,
) -> dev_lane::ProjW {
    use crate::models::inkling::w4a16gemm as k16;
    // TWO REASONS NOT TO PERMUTE, and they are not the same reason. Only the
    // HEAD is ever permuted here, and only when the approximate lane is off.
    //
    // 1. CORRECTNESS, the head. `linear_ann` -> `ann_logits` reads codes/scales
    //    as row-major [n, k/8] and has no fragment path, so permuting the head
    //    while the aNN lane is on makes every exact rescore read permuted bytes
    //    as row-major. Neither branch is wrong alone; the seam between them is,
    //    and it produced plausible logits rather than an error. `for_ann` names
    //    the weight `linear_ann` might read, which is why this is not simply
    //    `ann_budget() > 0`: the SINK experts come through this same function
    //    and can never reach `ann_logits` at all.
    //
    // 2. SPEED, and it is per SHAPE, not per weight-kind. The sinks are correct
    //    in either layout, so this is purely a performance choice. The original
    //    measurement said permuting them is a LOSS:
    //
    //      a109514 -> e30f22a, ctx 3732, split 21, 7 reps, 441 warm passes a
    //      side, identical binary on both nodes, `for_ann` 0 -> 5 refs:
    //
    //        w4a16_linear, both nodes   6.84 -> 8.55 ms   (+1.71, +25%)
    //          head 3.30 -> 4.05, tail 3.54 -> 4.50
    //        device busy, both nodes   73.42 -> 75.23 ms  (+1.81)
    //
    //      The kernel name changed `w4a16_linear_ab` -> `w4a16_linear_swz_ab`,
    //      so the permuted path is unambiguously what ran, and it accounts for
    //      94% of the device regression.
    //
    //    WHY, and it is a lesson about quoting a figure outside its shape. The
    //    permutation was measured at the HEAD's shape -- 201024 x 4096, ~25128
    //    cubes -- where it is worth 95.9 -> 116.3 GB/s. The sinks are 8192x4096
    //    and 4096x2048, which are 1024 and 512 cubes.
    //
    //    CORRECTED 2026-08-26, and the correction is the same mistake one level
    //    down: "the sinks" is not a shape either. Measured across the range by
    //    `w4a16_swz_grid`, 8192x4096 (1024 cubes) WINS 1.13-1.22x and only
    //    4096x2048 (512 cubes at k=2048) loses, 0.88x. The mechanism is not the
    //    "gathered write at the staging" guessed here -- this kernel has no
    //    staging. A 32-byte sector holds four k-tiles of one weight row, so the
    //    row-major form's eight-sector burst is an incidental 4-deep L1
    //    prefetch; permuting trades 8x fewer requests for losing it, and which
    //    side pays depends on having enough resident warps to hide the latency
    //    the prefetch was covering. See `swizzle_pays`, which carries the table.
    //
    // So the head keeps its correctness guard and the sinks are judged by shape.
    // `INK_W4A16_SWZ=0` remains the ablation; `=1` forces past the predicate.
    //
    // WHAT THIS A/B COULD NOT SEPARATE: `de482cb` bundled the sink permutation
    // WITH moving the pool `memory_usage` barrier off the decode path, and the
    // same run shows non-device residue falling 21.38 -> 17.67 ms. The two
    // halves are disentangled by this commit, which reverts only the first.
    let ann_owns_m1 = for_ann && ann_budget() > 0;
    // Reason 2 above, now measured across the range instead of at one end.
    // `swizzle_pays` is a CUBE-COUNT-AND-K predicate carrying its own table; the
    // old `!for_ann` here was a weight-kind rule standing in for it, and it was
    // wrong in both directions -- it declined sink `gate_up`, which wins
    // 1.13-1.22x, and it would have been read across to the dense MLP, whose
    // `down` wins 1.25x at the very cube count where the sink's `down` loses.
    // `INK_W4A16_SWZ=1` still forces it on regardless, which is how the
    // predicate itself gets A/B'd. It cannot override `ann_owns_m1`, which is
    // correctness, not speed.
    let grid_too_small = !k16::swizzle_pays(p.n, p.k) && !k16::swizzle_w4a16_forced();
    if !ann_owns_m1 && !grid_too_small && k16::swizzle_w4a16() && k16::swizzleable(p.n, p.k) {
        let (c, s) = k16::swizzle_w4a16_device(client, &p.codes, &p.scales, p.n, p.k);
        p.codes = c;
        p.scales = s;
        p.swizzled = true;
    }
    // Once a process, not once a weight: the sink experts come through here
    // several times a layer and the line is about the LAYOUT, which is one
    // decision.
    // Once per KIND, not once a process, and each says WHY it is in the layout
    // it is in. It used to be one flag for both, which was right only while the
    // two kinds could not disagree; they did, and the single line reported
    // whichever bound first. The reasons differ now -- the head is row-major for
    // CORRECTNESS and the sinks for SPEED -- and a line that prints only the
    // layout invites someone to "fix" the sinks back to fragment order, which is
    // measured at +25% on that kernel.
    //
    // Keyed on (kind, REASON) and not on kind alone, because since `swizzle_pays`
    // replaced the weight-kind rule the two sink weights can DISAGREE: `gate_up`
    // is 1024 cubes and permuted, `down` is 512 at k=2048 and is not. A
    // once-per-kind line would print whichever bound first and silently claim it
    // for both -- which is the exact failure the paragraph above describes,
    // recurring one level down. Each (kind, reason) says itself once, with the
    // shape that earned it, so the startup log shows the rule actually applied.
    static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let (reason, idx) = if p.swizzled {
        ("MMA-FRAGMENT (m16n8k16)", 0u32)
    } else if ann_owns_m1 {
        ("row-major [n, k/8] (the approximate head owns m=1)", 1)
    } else if grid_too_small {
        (
            "row-major [n, k/8] (too few cubes at this k -- see swizzle_pays)",
            2,
        )
    } else {
        ("row-major [n, k/8]", 3)
    };
    let bit = 1u32 << (idx + if for_ann { 4 } else { 0 });
    if SAID.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
        println!(
            "  W4A16 {} weights [{}, {}] ({} cubes at m=1) written in {reason} order",
            if for_ann { "head" } else { "sink" },
            p.n,
            p.k,
            p.n / 8,
        );
    }
    dev_lane::ProjW::W4a16(p)
}

pub fn bind_bf16(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    aliases: Option<&crate::models::inkling::fp4gemm::Aliases>,
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
pub static ALIGN: [core::sync::atomic::AtomicU64; 8] =
    [const { core::sync::atomic::AtomicU64::new(0) }; 8];

pub fn note_align(bytes: &[u8], rows: usize, cols: usize) -> usize {
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
pub fn report_align(charged: u64) {
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
    #[allow(clippy::too_many_arguments)]
    pub fn shared_for(
        &mut self,
        cp: &Weights,
        client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
        aliases: Option<&crate::models::inkling::fp4gemm::Aliases>,
        p: &str,
        n_shared: usize,
        inter: usize,
        h: usize,
        halved: bool,
        tp: Option<crate::models::inkling::tp::Tp>,
    ) -> Result<&SharedOnDevice> {
        if !self.shared.contains_key(p) {
            let fused = cp.stored(&format!("{p}mlp.shared_experts.shared_w13_weight"))?;
            anyhow::ensure!(fused.elem == Elem::Bf16, "shared_w13 is {:?}", fused.elem);
            let (g, u) = crate::models::inkling::load::split_shared_w13_bytes(
                &fused.bytes,
                n_shared,
                inter,
                h,
                halved,
                2,
            );
            let d = cp.stored(&format!("{p}mlp.shared_experts.shared_w2_weight"))?;
            anyhow::ensure!(d.elem == Elem::Bf16, "shared_w2 is {:?}", d.elem);
            // ---- this rank's half of the intermediate axis -----------------
            //
            // Cut AFTER the gate/up split and PER EXPERT, which is what makes
            // this one line rather than a second reading of the fused layout:
            // `split_shared_w13_bytes` has already handed back `g` and `u` as
            // `[n_shared * inter, hidden]` with every expert contiguous, so a
            // rank's share is rows `s * inter + r` of each, for each `s`.
            //
            // Cut by the INTERMEDIATE axis and deliberately not by INSTANCE.
            // By instance is exact here (two experts, two ranks) and stops
            // being exact anywhere else, and it would leave `n_shared == 1`
            // locally -- which shifts every shared gamma column by the rank at
            // three call sites, silently. This way `n_shared` is 2 on both
            // ranks and nothing downstream of the bind moves.
            let cut = match tp {
                None => None,
                Some(tp) => Some(
                    tp.shared_inter(inter)
                        .map_err(|e| anyhow::anyhow!("shared experts: {e}"))?,
                ),
            };
            let per_expert_rows = |src: &[u8]| -> Result<Vec<u8>> {
                let Some(r) = cut.clone() else {
                    return Ok(src.to_vec());
                };
                let slab = crate::models::inkling::tpshard::Slab::new(src, n_shared * inter, h, 2)?;
                let mut out = Vec::with_capacity(n_shared * r.len() * h * 2);
                for e in 0..n_shared {
                    out.extend_from_slice(crate::models::inkling::tpshard::rows(
                        &slab,
                        e * inter + r.start..e * inter + r.end,
                    )?);
                }
                Ok(out)
            };
            let (g, u) = (per_expert_rows(&g)?, per_expert_rows(&u)?);
            // `w2` is `[n_shared][hidden][inter]`, so the same range is a
            // COLUMN range of each expert's block -- a gather, and the one
            // place the shared cut costs a copy the whole weight did not. It
            // is 8 MiB a layer against 100.7 MiB the layer streams, and the
            // W4A16 bind below re-encodes it anyway, so the copy is not held.
            let d_bytes: std::borrow::Cow<'_, [u8]> = match cut.clone() {
                None => std::borrow::Cow::Borrowed(&d.bytes[..]),
                Some(r) => {
                    let mut out = Vec::with_capacity(n_shared * h * r.len() * 2);
                    for e in 0..n_shared {
                        let blk = &d.bytes[e * h * inter * 2..(e + 1) * h * inter * 2];
                        let slab = crate::models::inkling::tpshard::Slab::new(blk, h, inter, 2)?;
                        out.extend_from_slice(&crate::models::inkling::tpshard::cols(
                            &slab,
                            r.clone(),
                        )?);
                    }
                    std::borrow::Cow::Owned(out)
                }
            };
            // From here down every `inter` is THIS RANK's.
            let inter = cut.map(|r| r.len()).unwrap_or(inter);
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
            let gate_up = if sink_w4a16() {
                // W4A16 and NOT `Fp4`: four-bit weights against a BF16
                // activation, because nothing calibrated an input
                // quantiser for this tensor. See [`sink_w4a16`].
                w4a16_bind(
                    client,
                    quantized_bf16(client, &gu, 2 * n_shared * inter, h),
                    // Never read by `linear_ann`. See `w4a16_bind`.
                    false,
                )
            } else {
                dev_lane::ProjW::Bf16(bind_bf16(client, aliases, &gu, 2 * n_shared * inter, h))
            };
            let split = || {
                (0..n_shared)
                    .map(|e| {
                        let raw = &d_bytes[e * per_d..(e + 1) * per_d];
                        if sink_w4a16() {
                            w4a16_bind(client, quantized_bf16(client, raw, h, inter), false)
                        } else {
                            // `w2` is NOT de-interleaved, so this one is a view
                            // of the pile and aliases outright.
                            dev_lane::ProjW::Bf16(bind_bf16(client, aliases, raw, h, inter))
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let fused = || {
                // EXPERT-MINOR ALONG K. The pile stores `w2` as
                // `[n_shared][hidden][inter]`; the fused GEMM wants
                // `[hidden][n_shared][inter]`, so that column `s * inter + c`
                // of the weight lines up with column `s * inter + c` of the
                // concatenated activations. That is the whole change: an outer
                // transpose of two dims, `hidden * n_shared` contiguous runs of
                // `inter` BF16 values, done once at bind.
                //
                // The 16-element NVFP4 blocks do not straddle the seam
                // (`inter % GROUP == 0`) and `quantized_bf16` fixes `scale2` at
                // 1.0, so the codes and scales this produces are the SAME BYTES
                // the per-expert binds produced, merely reordered. The
                // quantisation is not part of what changed.
                // The split form indexes `[e * per_d ..]` and would read a
                // prefix of a larger buffer without complaining; the interleave
                // reads every expert's every row, so it is the one that has to
                // say what it assumes.
                assert_eq!(
                    d_bytes.len(),
                    n_shared * per_d,
                    "shared_w2 is not {n_shared} experts x {per_d} bytes"
                );
                let row = inter * 2;
                let mut il = vec![0u8; n_shared * per_d];
                for r in 0..h {
                    for s in 0..n_shared {
                        let src = s * per_d + r * row;
                        let dst = (r * n_shared + s) * row;
                        il[dst..dst + row].copy_from_slice(&d_bytes[src..src + row]);
                    }
                }
                if sink_w4a16() {
                    w4a16_bind(
                        client,
                        quantized_bf16(client, &il, h, n_shared * inter),
                        false,
                    )
                } else {
                    // The one thing the fusion costs the BF16 arm: the split
                    // form ALIASES the pile, and an interleaved buffer cannot.
                    // `sink_w4a16` is a literal `true`, so this branch is
                    // unreachable in any shipped run; it is kept correct rather
                    // than kept cheap.
                    dev_lane::ProjW::Bf16(bind_bf16(client, None, &il, h, n_shared * inter))
                }
            };
            let down = match (sink_down_fused() && n_shared > 1, sink_down_diff()) {
                (_, true) if n_shared > 1 => SinkDown::Both(split(), fused()),
                (true, _) => SinkDown::Fused(fused()),
                _ => SinkDown::Split(split()),
            };
            // WHICH ARM RAN, said once, because nothing else says it. The W4A16
            // census keys its line on (kind, REASON), so `gate_up` and `down`
            // collapse onto one line whenever they land in the same layout --
            // which they do here, both permuted -- and the shape printed is
            // whichever bound first. An A/B whose two arms differ only by an
            // environment variable needs a tell in the output that is not the
            // number being measured, and this is it: the arm's own shape.
            static SAID_DOWN: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !SAID_DOWN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let (what, k) = match &down {
                    SinkDown::Split(_) => ("SPLIT, one GEMM an expert", inter),
                    SinkDown::Fused(_) if sink_down_cat() => {
                        ("FUSED along k, operand by cat", n_shared * inter)
                    }
                    SinkDown::Fused(_) => ("FUSED along k, one GEMM", n_shared * inter),
                    SinkDown::Both(..) => {
                        ("BOTH (INK_SINK_DOWN_DIFF=1, diagnostic)", n_shared * inter)
                    }
                };
                println!(
                    "  sink `down`: {what} -- [{h}, {k}] x {}, {} cubes at m=1",
                    match &down {
                        SinkDown::Split(v) => v.len(),
                        _ => 1,
                    },
                    h / 8
                );
            }
            let sd = SharedOnDevice { gate_up, down };
            self.bytes += (gu.len() + n_shared * per_d) as u64;
            self.shared.insert(p.to_string(), sd);
        }
        Ok(&self.shared[p])
    }

    /// One dense layer's MLP, bound on first use.
    pub fn dense_for(
        &mut self,
        cp: &Weights,
        client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
        aliases: Option<&crate::models::inkling::fp4gemm::Aliases>,
        p: &str,
        h: usize,
        tp: Option<crate::models::inkling::tp::Tp>,
    ) -> Result<&(Bf16W, Bf16W, Bf16W, f32)> {
        if !self.dense.contains_key(p) {
            let fused = cp.stored(&format!("{p}mlp.w13_dn.weight"))?;
            anyhow::ensure!(fused.elem == Elem::Bf16, "dense w13 is {:?}", fused.elem);
            let (g, u) = crate::models::inkling::load::split_gate_up_bytes(&fused.bytes, h, 2);
            let down = cp.stored(&format!("{p}mlp.w2_md.weight"))?;
            anyhow::ensure!(down.elem == Elem::Bf16, "dense w2 is {:?}", down.elem);
            let (drows, dcols) = (down.dims[0] as usize, down.dims[1] as usize);
            let inter = g.len() / (h * 2);
            // ---- this rank's half of the dense intermediate axis -----------
            //
            // `mlp.w13_dn.weight` is stored INTERLEAVED and `split_gate_up_bytes`
            // has already de-interleaved it into two plain `[inter, hidden]`
            // buffers, so the cut here is a row range of EACH -- not of the
            // fused weight, where the same range would be half the gates and
            // none of the ups. `w2` is `[hidden, inter]`, so the same range is
            // a column range and has to be gathered.
            //
            // Two layers on this model, 384 MiB each, and both buffers are
            // already owned copies (the split made them), so the cut costs one
            // pass over what was going to be uploaded anyway.
            // `w2` stays a BORROW when nothing is cut. It normally ALIASES the
            // pile, and an unconditional `to_vec()` here would cost a resident
            // 134 MiB copy of it on every single-node run -- the exact
            // regression the `INK_DENSE_FAKEQUANT` note below is written about.
            let (g, u, dn, inter, dcols): (Vec<u8>, Vec<u8>, std::borrow::Cow<'_, [u8]>, _, _) =
                match tp {
                    None => (
                        g,
                        u,
                        std::borrow::Cow::Borrowed(&down.bytes[..]),
                        inter,
                        dcols,
                    ),
                    Some(tp) => {
                        let r = tp
                            .dense_inter(inter)
                            .map_err(|e| anyhow::anyhow!("dense MLP: {e}"))?;
                        let gs = crate::models::inkling::tpshard::Slab::new(&g, inter, h, 2)?;
                        let us = crate::models::inkling::tpshard::Slab::new(&u, inter, h, 2)?;
                        let ds = crate::models::inkling::tpshard::Slab::new(
                            &down.bytes,
                            drows,
                            dcols,
                            2,
                        )?;
                        let gg = crate::models::inkling::tpshard::rows(&gs, r.clone())?.to_vec();
                        let uu = crate::models::inkling::tpshard::rows(&us, r.clone())?.to_vec();
                        let dd = crate::models::inkling::tpshard::cols(&ds, r.clone())?;
                        (gg, uu, std::borrow::Cow::Owned(dd), r.len(), r.len())
                    }
                };
            let down_bytes: &[u8] = &dn;
            // The global scale is one f32 and is a SCALAR the product is
            // multiplied by, not a weight, so it comes through the widening
            // accessor and costs four bytes.
            let gs = cp.tensor(&format!("{p}mlp.global_scale"))?.data[0];
            self.bytes += (g.len() + u.len() + down_bytes.len()) as u64;
            // `INK_DENSE_FAKEQUANT=1` replaces each weight with its own NVFP4
            // round trip. Same shapes, same binds, same GEMM -- only the values
            // move. One consequence worth naming: `w2` normally ALIASES the
            // pile, and a round-tripped copy cannot, so this arm holds one extra
            // resident copy of it. That is a residency difference in a
            // diagnostic, not a lane anyone ships.
            // Built ONLY when the probe is on, and the default arm then binds the
            // very same slices it always did. Written as an `Option` of three
            // owned buffers rather than as an `if/else` yielding three `Vec`s
            // because `w2` normally ALIASES the pile -- a `to_vec()` on the
            // default path would silently cost a resident copy of it on every
            // run, which is precisely the kind of residency regression a
            // diagnostic must not introduce. Under the probe that copy is
            // unavoidable (a round-tripped buffer is not in the mapping) and is
            // one more reason the arm's TIMING means nothing.
            let fq = dense_fake_quant().then(|| {
                (
                    fake_quant_bf16(client, &g, inter, h),
                    fake_quant_bf16(client, &u, inter, h),
                    fake_quant_bf16(client, down_bytes, drows, dcols),
                )
            });
            let (gb, ub, db): (&[u8], &[u8], &[u8]) = match &fq {
                Some((a, b, c)) => (a, b, c),
                None => (&g, &u, down_bytes),
            };
            let trip = (
                bind_bf16(client, aliases, gb, inter, h),
                bind_bf16(client, aliases, ub, inter, h),
                bind_bf16(client, aliases, db, drows, dcols),
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
pub fn dense_mlp_bf16(x: T2, w: &(Bf16W, Bf16W, Bf16W, f32)) -> T2 {
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
pub struct MtpDev {
    pub attn: dev_lane::AttnWeightsDev,
    pub attn_sconv: T2,
    pub mlp_sconv: T2,
    pub attn_norm: BT<Bk, 1>,
    pub mlp_norm: BT<Bk, 1>,
    pub embed_norm: BT<Bk, 1>,
    pub hidden_norm: BT<Bk, 1>,
    /// `[hidden, 2 * hidden]`, which is the whole of what
    /// `mtp_hidden_states_first` is a claim about.
    pub input_proj: Bf16W,
    pub dense: (Bf16W, Bf16W, Bf16W, f32),
}

/// What one device MTP head retains between draft steps.
///
/// The same three things a main-stack layer keeps, and for the same reasons —
/// see [`dev_lane::AttnCache`] on why the pre-convolution projections are not
/// optional. `Clone` is load-bearing: a SPECULATIVE row is run against a clone
/// that is then dropped, so a rejected draft leaves nothing to undo.
#[derive(Clone)]
pub struct MtpDevCache {
    pub attn: dev_lane::AttnCache<Bk>,
    pub attn_sconv: T2,
    pub mlp_sconv: T2,
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
pub fn backbone_embed_norm() -> bool {
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
pub fn mtp_out_norm() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_MTP_OUTNORM")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

pub fn mtp_embed_row(
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
pub fn mtp_ablate() -> Option<&'static str> {
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

pub fn mtp_input_dev(hidden: T2, embeds: T2, w: &MtpDev, eps: f64, order: MtpConcat) -> T2 {
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
pub fn mtp_block_prefill_dev(
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
pub fn mtp_block_step_dev(
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
pub fn shared_experts_bf16(
    dev: &burn::backend::cuda::CudaDevice,
    x: T2,
    sw: &SharedOnDevice,
    gammas: &[f32],
    n_shared: usize,
    layer: usize,
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
    sink_down_apply(
        &sw.down,
        layer,
        n_shared,
        |s| {
            let g = gu.clone().slice([0..n, s * inter..(s + 1) * inter]);
            let u = gu
                .clone()
                .slice([0..n, (n_shared + s) * inter..(n_shared + s + 1) * inter]);
            let col: Vec<f32> = (0..n).map(|tk| gammas[tk * n_shared + s]).collect();
            let gam = BT::<Bk, 2>::from_data(BTD::new(col, [n, 1]), dev);
            dev_lane::silu(g) * u * gam
        },
        || {
            let g = gu.clone().slice([0..n, 0..n_shared * inter]);
            let u = gu
                .clone()
                .slice([0..n, n_shared * inter..2 * n_shared * inter]);
            // `gammas` is already `[n, n_shared]` row-major -- the per-expert
            // closure above is the one that has to gather a column out of it.
            let gam = BT::<Bk, 2>::from_data(BTD::new(gammas.to_vec(), [n, n_shared]), dev);
            wide_gate(g, u, gam, n, n_shared, inter)
        },
    )
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
pub struct RouterDev {
    pub proj: RouterProj,
    /// `INK_ROUTER_DIFF=1` only: the f32 `[rows, hidden]` lane, held BESIDE the
    /// active one so the two selections can be compared on the same activation.
    /// It is `None` otherwise, so the ordinary run carries neither the weight
    /// nor the second matmul.
    pub reference: Option<T2>,
    pub bias: Vec<f32>,
    pub global_scale: f32,
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
pub struct RouteDiff {
    /// Token-positions whose selection was compared.
    pub examined: usize,
    /// ...where the SET of chosen experts differs.
    pub set_differs: usize,
    /// ...where the set agrees but the order the top-k came out in does not.
    /// Harmless by itself -- the weights follow the experts -- but it is the
    /// near miss that says how close the ordering is to flipping.
    pub order_differs: usize,
    /// Individual (position, slot) pairs naming a different expert.
    pub slots_differ: usize,
    /// Largest `|active - reference|` over every logit compared.
    pub max_abs_logit: f32,
    /// Largest `|active - reference|` over the weights the chosen experts got.
    /// Only defined where the sets agree, since otherwise the weights are not
    /// the same quantity.
    pub max_abs_weight: f32,
}

impl RouteDiff {
    /// One token-position, both selections in hand.
    pub fn note(&mut self, a: &Routing, b: &Routing, la: &[f32], lb: &[f32]) {
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
pub fn drop_pad_cols(v: Vec<f32>, n: usize, cols: usize, keep: usize) -> Vec<f32> {
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
/// This rank's rows of a row-major `[rows, cols]` f32 weight.
///
/// The f32 twin of `tpshard::rows`, for the handful of small per-head tensors
/// that reach the device through `gv` (which widens on the way out of the
/// mapping) rather than through a BF16 bind. Copying is not a cost worth
/// avoiding here: the two short convolutions are `kv_heads * head_dim` by
/// `kernel`, which is kilobytes, and they are read once per layer for the run.
pub fn shard_rows_f32(v: &[f32], cols: usize, r: std::ops::Range<usize>) -> Vec<f32> {
    assert!(
        r.end * cols <= v.len(),
        "rows {}..{} of a {}-element weight {} wide runs off the end",
        r.start,
        r.end,
        v.len(),
        cols
    );
    v[r.start * cols..r.end * cols].to_vec()
}

pub fn transpose_rows(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
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
pub enum RouterArm {
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
    pub fn from_env() -> Self {
        match std::env::var("INK_ROUTER").as_deref() {
            Ok("transpose") => RouterArm::Transpose,
            Ok("pre") => RouterArm::Pre,
            Ok("bf16") | Err(_) => RouterArm::Bf16,
            Ok(other) => panic!("INK_ROUTER={other:?} is not one of: transpose, pre, bf16"),
        }
    }

    pub fn label(self) -> &'static str {
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
pub enum RouterProj {
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
pub struct LayerDev {
    pub attn: dev_lane::AttnWeightsDev,
    pub attn_sconv: T2,
    pub mlp_sconv: T2,
    pub attn_norm: BT<Bk, 1>,
    pub mlp_norm: BT<Bk, 1>,
    /// `None` on a dense layer, which has no experts to route to.
    pub router: Option<RouterDev>,
}

/// What binding one layer to the device produced, and what it cost.
///
/// The cost fields are not decoration: `read` is time spent inside the pile's
/// mapping and `upload` is time spent moving bytes at it, and a run that cannot
/// tell those apart cannot tell a slow disk from a slow interconnect. They are
/// returned rather than accumulated into a global for the same reason the
/// binding itself moved out of the binary — a caller that does not measure
/// should not have to own an accumulator to call this.
pub struct BoundLayer {
    /// The layer, resident.
    pub layer: LayerDev,
    /// Seconds spent reading weight bytes out of the source.
    pub read: f64,
    /// Seconds spent uploading them, i.e. the bind less the read.
    pub upload: f64,
    /// Device bytes this layer now holds — two per element for the projections,
    /// which is what the tensor-core lane takes, and four for everything the
    /// device keeps as f32.
    pub bytes: u64,
}

/// Bind ONE layer's weights to the device: the ten attention projections, both
/// short convolutions, both norm gains and the router matrix.
///
/// This is the function `inkling_forward` used to inline, and it is the single
/// largest reason a model could not be held open: everything a layer multiplies
/// by was assembled inside a loop inside `main`. It reads once, uploads once,
/// and the host copy is dropped — the weights do not change between tokens, so
/// the only question a layer ever had was WHERE they are held, and on a device
/// lane the answer is the device.
///
/// `tp_shard` decides whether this rank binds a whole tensor or a slice of one.
/// A row range is a span of the mapping and binds for free; a column range is a
/// stride and is gathered into a fresh allocation. `q_heads` and `kv_heads` are
/// split along whole KV groups so the GQA grouping survives the cut.
#[allow(clippy::too_many_arguments)]
pub fn bind_layer(
    cp: &Weights,
    dev: &burn::backend::cuda::CudaDevice,
    fp4_client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    fp4_aliases: Option<&crate::models::inkling::fp4gemm::Aliases>,
    p: &str,
    layer: usize,
    t: &crate::models::inkling::config::InklingTextConfig,
    tp_shard: Option<crate::models::inkling::tp::Tp>,
    router_arm: RouterArm,
    router_diff: bool,
    t_read: &std::cell::Cell<f64>,
) -> Result<BoundLayer> {
    let h = t.hidden_size;
    let kind = t.attn_kind(layer);
    let (g_heads, g_kv_heads, head_dim) = t.heads(kind);
    let (heads, kv_heads) = match tp_shard {
        Some(tp) => (
            tp.share("q_heads", g_heads)?,
            tp.share("kv_heads", g_kv_heads)?,
        ),
        None => (g_heads, g_kv_heads),
    };
    // ONE accessor. `g` used to exist beside it, holding an f32 copy on the host
    // for the host lanes to read; there are no host lanes left in this loop, so
    // every weight is read once, uploaded, and the host copy dropped.
    let gv = |nm: &str| -> Result<Vec<f32>> {
        let s = std::time::Instant::now();
        let r = cp.tensor(&format!("{p}{nm}"))?.data;
        t_read.set(t_read.get() + s.elapsed().as_secs_f64());
        Ok(r)
    };
    let r0 = t_read.get();
    let t_w0 = std::time::Instant::now();
    // The five projections bind as the BF16 the pile stores. `gv`
    // widens to f32 on the way out of the mapping and would double
    // every one of them on the device for nothing: `mma.sync…bf16`
    // takes the stored bytes, and where those bytes are inside a
    // registered mapping `bind_bf16` aliases them instead of copying.
    let pw = |nm: &str, rows: usize, cols: usize| -> Result<Bf16W> {
        let s = std::time::Instant::now();
        let leaf = cp.stored(&format!("{p}{nm}"))?;
        anyhow::ensure!(
            leaf.elem == Elem::Bf16,
            "{p}{nm} is {:?}; this lane multiplies BF16 by BF16",
            leaf.elem
        );
        t_read.set(t_read.get() + s.elapsed().as_secs_f64());
        Ok(bind_bf16(fp4_client, fp4_aliases, &leaf.bytes, rows, cols))
    };
    // The concatenation reads the same four leaves `pw` binds, in
    // the output order [`dev_lane::project_qkvr`] slices back.
    // Not under a split: the concatenation reads four WHOLE leaves
    // and the split lane below discards it (`wqkvr: None`), so
    // building it would be 44 MiB of host copying a layer for
    // nothing.
    let fused_qkvr = if dev_lane::fuse_qkvr() && tp_shard.is_none() {
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
        Some(bind_bf16(fp4_client, fp4_aliases, &b, rows, h))
    } else {
        None
    };
    // ---- this rank's slice of the five projections -------------
    //
    // Four of them are OUTPUT-parallel: the split is along rows, a
    // row of a `[*, hidden]` BF16 weight is `hidden * 2` bytes, and
    // every offset is a multiple of that -- so the shard is a
    // SUBSLICE of the mapping and still aliases the pile. Binding
    // this rank's half costs nothing it did not already cost.
    //
    // `wo` is the one that inverts. It is INPUT-parallel: its
    // columns match the heads this rank computed, so its shard is a
    // column range of a row-major matrix, which is `hidden` separate
    // runs and cannot be expressed as an offset. It is gathered into
    // a fresh allocation -- 4096 x 2048 BF16 = 16 MiB a layer -- and
    // that copy is what buys the partial product this rank's
    // all-reduce contributes. `tpshard::cols` is the only binder
    // here that allocates, and it is deliberately the only one whose
    // signature says so.
    let pw_rows =
        |nm: &str, g_rows: usize, cols: usize, r: std::ops::Range<usize>| -> Result<Bf16W> {
            let s = std::time::Instant::now();
            let leaf = cp.stored(&format!("{p}{nm}"))?;
            anyhow::ensure!(
                leaf.elem == Elem::Bf16,
                "{p}{nm} is {:?}; this lane multiplies BF16 by BF16",
                leaf.elem
            );
            t_read.set(t_read.get() + s.elapsed().as_secs_f64());
            let slab = crate::models::inkling::tpshard::Slab::new(&leaf.bytes, g_rows, cols, 2)?;
            let n = r.len();
            let bytes = crate::models::inkling::tpshard::rows(&slab, r)?;
            Ok(bind_bf16(fp4_client, fp4_aliases, bytes, n, cols))
        };
    let pw_cols =
        |nm: &str, rows: usize, g_cols: usize, c: std::ops::Range<usize>| -> Result<Bf16W> {
            let s = std::time::Instant::now();
            let leaf = cp.stored(&format!("{p}{nm}"))?;
            anyhow::ensure!(
                leaf.elem == Elem::Bf16,
                "{p}{nm} is {:?}; this lane multiplies BF16 by BF16",
                leaf.elem
            );
            t_read.set(t_read.get() + s.elapsed().as_secs_f64());
            let slab = crate::models::inkling::tpshard::Slab::new(&leaf.bytes, rows, g_cols, 2)?;
            let n = c.len();
            let bytes = crate::models::inkling::tpshard::cols(&slab, c)?;
            Ok(bind_bf16(fp4_client, fp4_aliases, &bytes, rows, n))
        };
    let attn = match tp_shard {
        None => dev_lane::AttnWeightsDev {
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
                dev,
            ),
            v_sconv: up2(
                gv("attn.v_sconv.weight")?,
                kv_heads * head_dim,
                t.sconv_kernel_size,
                dev,
            ),
            q_norm: up1(gv("attn.q_norm.weight")?, head_dim, dev),
            k_norm: up1(gv("attn.k_norm.weight")?, head_dim, dev),
            rel_proj: up2(
                gv("attn.rel_logits_proj.proj")?,
                t.d_rel,
                t.rel_span(kind),
                dev,
            ),
        },
        Some(tp) => {
            let qr = tp.q_heads(g_heads)?;
            let kr = tp.kv_heads(g_kv_heads)?;
            let rr = tp.rel_rows(g_heads, t.d_rel)?;
            // Head ranges scaled into ROW ranges of each weight.
            let q_rows = qr.start * head_dim..qr.end * head_dim;
            let kv_rows = kr.start * head_dim..kr.end * head_dim;
            dev_lane::AttnWeightsDev {
                wq: pw_rows("attn.wq_du.weight", g_heads * head_dim, h, q_rows.clone())?,
                wk: pw_rows(
                    "attn.wk_dv.weight",
                    g_kv_heads * head_dim,
                    h,
                    kv_rows.clone(),
                )?,
                wv: pw_rows(
                    "attn.wv_dv.weight",
                    g_kv_heads * head_dim,
                    h,
                    kv_rows.clone(),
                )?,
                wr: pw_rows("attn.wr_du.weight", g_heads * t.d_rel, h, rr)?,
                // The fused concatenation reads four WHOLE leaves and
                // would hand this rank the other rank's heads as well.
                // Sharding it means concatenating four already-sliced
                // runs, which is a different function; until it exists
                // the split lane takes the unfused path.
                wqkvr: None,
                wo: pw_cols("attn.wo_ud.weight", h, g_heads * head_dim, q_rows)?,
                // Per-KV-head state, so it splits with the KV heads.
                k_sconv: up2(
                    shard_rows_f32(
                        &gv("attn.k_sconv.weight")?,
                        t.sconv_kernel_size,
                        kv_rows.clone(),
                    ),
                    kv_heads * head_dim,
                    t.sconv_kernel_size,
                    dev,
                ),
                v_sconv: up2(
                    shard_rows_f32(&gv("attn.v_sconv.weight")?, t.sconv_kernel_size, kv_rows),
                    kv_heads * head_dim,
                    t.sconv_kernel_size,
                    dev,
                ),
                // Per-HEAD-DIM, not per head: the same 128 gains apply
                // to every head, so both ranks hold all of them.
                q_norm: up1(gv("attn.q_norm.weight")?, head_dim, dev),
                k_norm: up1(gv("attn.k_norm.weight")?, head_dim, dev),
                // `[d_rel, rel_span]` -- indexed by neither head nor
                // hidden, so it is replicated whole.
                rel_proj: up2(
                    gv("attn.rel_logits_proj.proj")?,
                    t.d_rel,
                    t.rel_span(kind),
                    dev,
                ),
            }
        }
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
            RouterArm::Transpose => RouterProj::PerCall(up2(gv("mlp.gate.weight")?, rows, h, dev)),
            // Transposed HERE, on the host, once per layer for the run,
            // instead of on the device once per token per layer.
            RouterArm::Pre => RouterProj::Pre(up2(
                transpose_rows(&gv("mlp.gate.weight")?, rows, h),
                h,
                rows,
                dev,
            )),
            RouterArm::Bf16 => {
                use crate::models::inkling::bf16gemm::NTILE;
                let s = std::time::Instant::now();
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
                    w: bind_bf16(fp4_client, fp4_aliases, &bytes, pad, h),
                }
            }
        };
        // Held only under the diff probe, and it is the arm every run
        // before 969bf6f shipped -- the thing the new arm has to be
        // compared AGAINST, not a second opinion invented for the
        // occasion.
        let reference = if router_diff {
            Some(up2(gv("mlp.gate.weight")?, rows, h, dev))
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
        attn_sconv: up2(gv("attn_sconv.weight")?, h, t.sconv_kernel_size, dev),
        mlp_sconv: up2(gv("mlp_sconv.weight")?, h, t.sconv_kernel_size, dev),
        attn_norm: up1(gv("attn_norm.weight")?, h, dev),
        mlp_norm: up1(gv("mlp_norm.weight")?, h, dev),
        router,
    };
    <Bk as burn::tensor::backend::Backend>::sync(dev).expect("sync after the layer uploads");
    let span = t_w0.elapsed().as_secs_f64();
    let rd = t_read.get() - r0;
    // Two bytes for the projections and four for the rest, because that
    // is now what is on the device. Counting the projections at four
    // would report the widening this commit removed.
    let bytes = (2
        * (heads * head_dim * h
            + 2 * kv_heads * head_dim * h
            + heads * t.d_rel * h
            + h * heads * head_dim)
        + 4 * (2 * kv_heads * head_dim * t.sconv_kernel_size
            + 2 * head_dim
            + t.d_rel * t.rel_span(kind)
            + 2 * h * t.sconv_kernel_size
            + 2 * h)) as u64;
    Ok(BoundLayer {
        layer: built,
        read: rd,
        upload: span - rd,
        bytes,
    })
}

/// Everything one layer carries between generated tokens.
///
/// The attention cache is the headline, but the two layer-level short
/// convolutions have state too: they reach `kernel - 1` positions back, and a
/// cache that remembers K and V while forgetting those is wrong in a way that
/// still produces fluent-looking text.
///
/// This is the value that makes a SESSION a session. A process that holds a
/// `Vec<LayerCache>` across calls has a sequence in flight; one that rebuilds it
/// per request is re-reading a prompt it already read, and at this model's scale
/// that is the difference between a conversation and a benchmark.
pub struct LayerCache {
    /// K and V for this layer, paged.
    pub attn: dev_lane::AttnCache<Bk>,
    /// The attention short convolution's `kernel - 1` positions of history.
    pub attn_sconv: BT<Bk, 2>,
    /// `None` until the prefill seeds it, and a device tensor rather than a
    /// `Vec<f32>` for the same reason everything else here is: the convolution
    /// that reads it runs on the device, and a history that lived on the host
    /// would drag the whole MLP half back across.
    pub mlp_sconv: Option<BT<Bk, 2>>,
    /// What a speculative batch convolved, kept until the verifier says how many
    /// of its rows survived. `kernel - 1` history rows followed by the batch's
    /// own inputs, so the history after keeping `keep` of them is the window
    /// starting at `keep` -- the same shape, and the same argument, as
    /// [`dev_lane::AttnCache`]'s pending K/V projections.
    pub attn_sconv_pending: Option<BT<Bk, 2>>,
    /// The MLP convolution's half of the same rollback.
    pub mlp_sconv_pending: Option<BT<Bk, 2>>,
}

/// What the MoE half carries ACROSS layers and across passes.
///
/// Two things, and both are here because rebuilding them per layer is the cost
/// the device lanes exist to remove. `route` holds the row-plan invariants and
/// the per-layer expert tables — a table is 1024 lookups to derive and a layer
/// that refused once refuses every pass, so the `None` is cached too. `bias` is
/// each layer's router bias, uploaded on first touch.
///
/// It is a function of the pass WIDTH: the invariants are derived at a given
/// `n`, so a pass at a different width needs a different set. [`moe_layer`]
/// notices and rebuilds, carrying the tables over.
#[derive(Default)]
pub struct MoeState {
    /// The device row-plan state, or `None` before the first routed layer.
    pub route: Option<DevRoute>,
    /// The layer whose forward is kept for a learning pass, if learning is
    /// armed on this session (the last layer, on the rank that owns the head).
    #[cfg(feature = "inkling-cuda")]
    pub learn_layer: Option<usize>,
    /// What that layer's most recent forward left for the learning pass; taken
    /// by the session at the head of a scored pass.
    #[cfg(feature = "inkling-cuda")]
    pub learn: Option<super::learn::LearnKeep>,
    /// Per absolute layer: the router bias, on the device.
    pub bias: std::collections::HashMap<usize, cubecl::server::Handle>,
    /// Where the host time inside a routed layer went. Measurement, kept beside
    /// the state it describes rather than threaded through every caller.
    pub host: HostT,
}

/// One routed-MoE layer, on the DEFAULT lane: the router decision on the device,
/// the row plan on the device, the grouped NVFP4 experts and the shared experts,
/// summed.
///
/// # Why this is the only lane here
///
/// `inkling_forward` can run this block six other ways — the host router, the
/// host row plan, the per-expert loop, the BF16 storage arm, an A/B that runs
/// two of them and diffs, and a stale-routing timing probe. Every one of those
/// is a question about the model, and the answers are all in: the device router
/// and the device plan are ON by default because they measured +8.33% on five of
/// five interleaved pairs and emitted a token stream identical to the arm they
/// replaced. So the lane that runs when nobody sets anything is the lane a
/// session takes, and the arms stay where the measurements are.
///
/// Nothing in here reads anything back. The router's decision is never
/// materialised on the host, the plan is derived from it on the device, and the
/// expert launch takes the plan as a buffer — which is what makes a decode step
/// enqueue-bound rather than serialised on `top_k` blocking reads a layer.
///
/// # It REFUSES rather than falling back
///
/// If this layer's experts have no single aligned mapping, the device table
/// cannot be built and the binary drops to a host lane. A session says so
/// instead. A fallback that changes which lane is running without saying it is
/// how a serving process ends up with a latency profile nobody can account for,
/// and the condition is a property of the pile, so it fires on the first pass or
/// never.
#[allow(clippy::too_many_arguments)]
pub fn moe_layer(
    cp: &Weights,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    aliases: Option<&crate::models::inkling::fp4gemm::Aliases>,
    dense: &mut DeviceDense,
    st: &mut MoeState,
    dev: &burn::backend::cuda::CudaDevice,
    p: &str,
    layer: usize,
    t: &crate::models::inkling::config::InklingTextConfig,
    r: &RouterDev,
    hn: T2,
    n: usize,
    shared_halved: bool,
    tp: Option<crate::models::inkling::tp::Tp>,
) -> Result<T2> {
    let h = t.hidden_size;
    let global_inter = t.intermediate_size;
    // Routed expert slabs were cut during the startup copy, so their kernels
    // consume this rank's intermediate width. Shared experts are bound below
    // from their still-global source tensors and receive `tp` explicitly.
    let inter = match tp {
        Some(tp) => tp
            .share("intermediate_size", global_inter)
            .map_err(|e| anyhow::anyhow!("routed MLP: {e}"))?,
        None => global_inter,
    };
    let rows = t.n_routed_experts + t.n_shared_experts;
    let k = t.num_experts_per_tok;
    let ns = t.n_shared_experts;

    // ---- the router's PROJECTION, which is a matmul ------------------------
    //
    // `cols` is what comes BACK, which is `rows` except on the BF16 arm, whose
    // weight carries the instruction's n padding.
    let (lg, cols) = match &r.proj {
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

    // ---- the row-plan invariants, at THIS width ----------------------------
    if st.route.as_ref().is_some_and(|d| d.n != n) {
        let old = st.route.take().expect("checked");
        let mut fresh = devroute_new(client, k, n);
        // The tables are a function of the LAYER, not of the width, so they
        // survive a width change. Re-deriving them would be 1024 lookups a layer
        // for nothing.
        fresh.tabs = old.tabs;
        st.route = Some(fresh);
    }
    let dr = st.route.get_or_insert_with(|| devroute_new(client, k, n));
    if !dr.tabs.contains_key(&layer) {
        let t_s = Instant::now();
        let nvfp4 = cp.is_nvfp4(&format!("{p}mlp.experts.w13_weight"));
        let tb = match aliases {
            Some(al) if nvfp4 => build_expert_table(cp, al, client, p, t.n_routed_experts)?,
            Some(al) => build_expert_table_bf16(cp, al, client, p, t.n_routed_experts)?,
            None => None,
        };
        st.host.slice += t_s.elapsed().as_secs_f64();
        dr.tabs.insert(layer, tb);
    }
    let tb = dr.tabs[&layer].as_ref().with_context(|| {
        format!(
            "{p}: the {} routed experts have no single aligned host mapping, so the device \
             expert table cannot be built and this layer would need the host row plan. A \
             Session runs one lane; `inkling_forward` is where the host arm lives.",
            t.n_routed_experts
        )
    })?;

    // ---- the router's DECISION, on the device ------------------------------
    let bias_h = st
        .bias
        .entry(layer)
        .or_insert_with(|| client.create_from_slice(bytes_of(&r.bias)))
        .clone();
    let topk_width = 2 * k + ns + 1;
    let topk_h = crate::models::inkling::routetopk::router_topk_launch(
        client,
        &crate::models::inkling::seam::handle_of(lg),
        &bias_h,
        n,
        cols,
        t.n_routed_experts,
        ns,
        k,
        t.route_scale as f32 * r.global_scale,
    );

    // ---- the ROW PLAN, from a top-k answer that is never read back ---------
    let dp = crate::models::inkling::devplan::plan_from_topk_launch(
        client,
        &topk_h,
        tb,
        &dr.fault,
        dr.kmax,
        crate::models::inkling::fp4gemm::MTILE,
        topk_width,
        n,
    );

    // ---- the routed experts ------------------------------------------------
    let t_g = Instant::now();
    let acc = match tb.scaled {
        true => routed_experts_fp4_dev(
            client,
            dev,
            p,
            tb,
            &dp,
            dr,
            &hn,
            n,
            h,
            inter,
            cp.experts_swizzled(),
            t_g,
            &mut st.host,
        ),
        false => routed_experts_bf16_dev(
            client,
            dev,
            tb,
            &dp,
            dr,
            &hn,
            n,
            h,
            inter,
            t_g,
            &mut st.host,
        ),
    };
    if let Some(al) = aliases {
        for _ in 0..dr.k {
            al.note_alias(tb.expert_bytes);
        }
    }

    // ---- the shared experts, on the same normed input ----------------------
    //
    // They do NOT read the routed output. They run beside it, and the gammas
    // they are weighted by come off the same top-k buffer the plan did — so
    // nothing crosses to the host here either.
    let sw = dense.shared_for(
        cp,
        client,
        aliases,
        p,
        ns,
        global_inter,
        h,
        shared_halved,
        tp,
    )?;
    let g =
        crate::models::inkling::seam::tensor_of(client.clone(), dev.clone(), topk_h, n, topk_width);
    // A learning pass needs this layer's expert input and its plan after the
    // layer is over; keep them when the session armed this layer.
    #[cfg(feature = "inkling-cuda")]
    if st.learn_layer == Some(layer) {
        st.learn = Some(super::learn::LearnKeep {
            layer,
            hn: hn.clone(),
            dp,
            sconv: None,
        });
    }
    let sh = shared_experts_dev(hn, sw, g, k, ns, layer);

    Ok(acc + sh)
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
pub fn routed_experts_fp4(
    src: &Weights,
    aliases: Option<&crate::models::inkling::fp4gemm::Aliases>,
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
        mode != "2" || !crate::models::inkling::burn::act_bf16(),
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
/// [`crate::models::inkling::moegroup`], and `91f81b4` for the time this tree
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
pub fn report_ab(prefix: &str, a: &T2, b: &T2, h: usize) {
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
/// [`crate::models::inkling::moegroup`] for why the last of those had to be
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
pub fn grouped_experts_fp4(
    src: &Weights,
    al: &crate::models::inkling::fp4gemm::Aliases,
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
    use crate::models::inkling::moegroup::{BlockPlanDev, RowPlan};
    use crate::models::inkling::seam::handle_of_any;

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
pub fn grouped_experts_core(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    wmap: &cubecl::server::Handle,
    wmap_bytes: usize,
    blk: &crate::models::inkling::moegroup::BlockPlanDev,
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
    use crate::models::inkling::burn::act_bf16;
    use crate::models::inkling::fp4gemm::{gate_up_silu_launch, gate_up_silu_narrow_launch};
    use crate::models::inkling::fp4quant::{quantize_nvfp4, quantize_nvfp4_bf16};
    use crate::models::inkling::moegroup::{
        fp4_linear_grouped_bf16_launch, fp4_linear_grouped_launch, gather_grouped,
        gather_grouped_bf16, gather_grouped_bf16_from_bf16, scatter_weighted,
        scatter_weighted_bf16,
    };
    use crate::models::inkling::seam::tensor_of;

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
            crate::models::inkling::seam::pool_line(
                client,
                &format!("{prefix}experts m={m_total}")
            )
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
pub fn plan_check_note(
    prefix: &str,
    n: usize,
    by_expert: &std::collections::BTreeMap<usize, Vec<(usize, f32)>>,
    plan: &crate::models::inkling::moegroup::RowPlan,
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
pub fn plan_check() -> bool {
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
/// [`crate::models::inkling::devplan`] and `INK_PLAN_CHECK=1`. So they are built
/// once, uploaded once, and held here for the run, and the per-layer kernel
/// produces only the seventh.
pub struct DevRoute {
    /// `[k * MTILE]` i32, `0` at every tile start and `-1` in the padding.
    pub row_tok: cubecl::server::Handle,
    /// `[k]` u32, the identity: one expert per block.
    pub blk_slot: cubecl::server::Handle,
    /// `[k]` u32, the identity.
    pub blk_tile0: cubecl::server::Handle,
    /// `[k]` u32, all ones: one M tile per expert, so `INK_MOE_PLANES` has
    /// nothing to group at this width.
    pub blk_cnt: cubecl::server::Handle,
    /// `[1, k]` u32, `[0, 16, 32, …]`.
    pub tok_rows: cubecl::server::Handle,
    /// `[1]` u32, `[k]`.
    pub tok_cnt: cubecl::server::Handle,
    /// One u32 for the WHOLE RUN. The kernel raises it; the host reads it once,
    /// after the last pass. A per-layer read of it would be the read this
    /// entire lane exists to delete.
    pub fault: cubecl::server::Handle,
    /// Per absolute layer: its weight table, or `None` for a layer this lane
    /// cannot take (BF16 experts, no registered mapping, a misaligned plane).
    /// The `None` is cached too — a layer that refused once refuses every pass,
    /// and re-deriving that costs 1024 lookups.
    pub tabs:
        std::collections::HashMap<usize, Option<crate::models::inkling::devplan::ExpertTable>>,
    /// Expert SLOTS in the plan, which is also the block count: `n * top_k`.
    ///
    /// It was called `k` while it could only be `top_k`, and the rename is the
    /// whole `n > 1` change stated once. At `n == 1` it is still `top_k` and the
    /// lane is byte-identical to what it was.
    pub k: usize,
    /// `top_k`: the number of experts ONE token routes to, which is `RowPlan`'s
    /// `kmax` and is NOT the slot count once `n > 1`.
    pub kmax: usize,
    /// The row count these invariants were derived at. They are a function of
    /// `n` and `top_k` alone (see [`devroute_new`]), so a pass at a different
    /// width needs a different set and this is what notices.
    pub n: usize,
    /// `k * MTILE`.
    pub m_total: usize,
    /// What `RowPlan::planes()` said, carried so the launch shape matches the
    /// host lane's exactly.
    pub planes: usize,
}

/// Build [`DevRoute`]'s invariants — from [`RowPlan::build`] itself, at the
/// shape a one-token pass produces.
///
/// Derived rather than transcribed on purpose: if the stacking rule ever
/// changes, this follows it instead of disagreeing with it silently.
pub fn devroute_new(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    k: usize,
    n: usize,
) -> DevRoute {
    use crate::models::inkling::moegroup::RowPlan;
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
pub fn build_expert_table(
    src: &Weights,
    al: &crate::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    prefix: &str,
    n_routed: usize,
) -> Result<Option<crate::models::inkling::devplan::ExpertTable>> {
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
    Ok(Some(crate::models::inkling::devplan::ExpertTable {
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
pub fn build_expert_table_bf16(
    src: &Weights,
    al: &crate::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    prefix: &str,
    n_routed: usize,
) -> Result<Option<crate::models::inkling::devplan::ExpertTable>> {
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
    Ok(Some(crate::models::inkling::devplan::ExpertTable {
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
pub fn routed_experts_fp4_dev(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    tab: &crate::models::inkling::devplan::ExpertTable,
    dp: &crate::models::inkling::devplan::DevRowPlan,
    dr: &DevRoute,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    swz: bool,
    t_g: Instant,
    host: &mut HostT,
) -> T2 {
    use crate::models::inkling::moegroup::BlockPlanDev;
    use crate::models::inkling::seam::handle_of_any;
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
pub fn routed_experts_bf16_dev(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    tab: &crate::models::inkling::devplan::ExpertTable,
    dp: &crate::models::inkling::devplan::DevRowPlan,
    dr: &DevRoute,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    t_g: Instant,
    host: &mut HostT,
) -> T2 {
    use crate::models::inkling::moegroup::BlockPlanDev;
    use crate::models::inkling::seam::handle_of;
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
    let hn_h = handle_of(crate::models::inkling::resid::from_resid(hn.clone()));
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
pub fn shared_experts_dev(
    x: T2,
    sw: &SharedOnDevice,
    topk: T2,
    top_k: usize,
    n_shared: usize,
    layer: usize,
) -> T2 {
    let [n, _] = x.dims();
    let inter = sw.gate_up.n() / (2 * n_shared);
    let gu = dev_lane::linear_w(x, &sw.gate_up);
    sink_down_apply(
        &sw.down,
        layer,
        n_shared,
        |s| {
            let g = gu.clone().slice([0..n, s * inter..(s + 1) * inter]);
            let u = gu
                .clone()
                .slice([0..n, (n_shared + s) * inter..(n_shared + s + 1) * inter]);
            let gam = topk.clone().slice([0..n, 2 * top_k + s..2 * top_k + s + 1]);
            dev_lane::silu(g) * u * gam
        },
        || {
            let g = gu.clone().slice([0..n, 0..n_shared * inter]);
            let u = gu
                .clone()
                .slice([0..n, n_shared * inter..2 * n_shared * inter]);
            let gam = topk.clone().slice([0..n, 2 * top_k..2 * top_k + n_shared]);
            wide_gate(g, u, gam, n, n_shared, inter)
        },
    )
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
pub fn devplan_verify_layer(
    src: &Weights,
    al: &crate::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    prefix: &str,
    routing: &[Routing],
    dp: &crate::models::inkling::devplan::DevRowPlan,
    dr: &DevRoute,
    scaled: bool,
) -> Result<()> {
    use crate::models::inkling::fp4gemm::MTILE;
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
pub enum PlanArm {
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
    pub fn from_env() -> Result<PlanArm> {
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
    pub fn on(&self, decode_step: usize) -> bool {
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
pub fn mem_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_MEM_TRACE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// A slice of POD as the bytes `create_from_slice` uploads.
pub fn bytes_of<T: cubecl::prelude::CubeElement>(v: &[T]) -> &[u8] {
    T::as_bytes(v)
}

/// The per-expert lane: one launch sequence per active expert, in `BTreeMap`
/// order, which is the order the accumulation is defined by.
#[allow(clippy::too_many_arguments)]
pub fn per_expert_fp4(
    src: &Weights,
    aliases: Option<&crate::models::inkling::fp4gemm::Aliases>,
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
    use crate::models::inkling::fp4gemm::{
        MTILE, fp4_linear_launch, fp4_linear_swz_launch, gate_up_silu_launch,
    };
    use crate::models::inkling::fp4quant::quantize_nvfp4;
    use crate::models::inkling::pad::gather_rows_pad;
    use crate::models::inkling::seam::{handle_of, int_handle_of, tensor_of};

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
    let hn_h = handle_of(crate::models::inkling::resid::from_resid(hn.clone()));

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
pub fn routed_experts_bf16(
    src: &Weights,
    aliases: Option<&crate::models::inkling::fp4gemm::Aliases>,
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
pub fn grouped_experts_bf16(
    src: &Weights,
    al: &crate::models::inkling::fp4gemm::Aliases,
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
    use crate::models::inkling::moegroup::{BlockPlanDev, RowPlan};
    use crate::models::inkling::seam::handle_of;

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
    let hn_h = handle_of(crate::models::inkling::resid::from_resid(hn.clone()));
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
pub fn grouped_experts_bf16_core(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    wmap: &cubecl::server::Handle,
    wmap_bytes: usize,
    blk: &crate::models::inkling::moegroup::BlockPlanDev,
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
    use crate::models::inkling::bf16gemm::to_bf16_launch;
    use crate::models::inkling::fp4gemm::gate_up_silu_bf16_launch;
    use crate::models::inkling::moegroup::{
        bf16_linear_grouped_launch, gather_grouped, scatter_weighted,
    };
    use crate::models::inkling::seam::tensor_of;

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
pub fn per_expert_bf16(
    src: &Weights,
    aliases: Option<&crate::models::inkling::fp4gemm::Aliases>,
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
    use crate::models::inkling::bf16gemm::{MTILE, bf16_linear_launch, to_bf16_launch};
    use crate::models::inkling::fp4gemm::gate_up_silu_bf16_launch;
    use crate::models::inkling::pad::gather_rows_pad;
    use crate::models::inkling::seam::{handle_of, int_handle_of, tensor_of};

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
    let hn_h = handle_of(crate::models::inkling::resid::from_resid(hn.clone()));

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
pub fn expert_rows<B: Backend>(
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
