//! Additional tensor storage for one typed SDFT learner and its scored cohort.
//!
//! This is ADDED to ordinary inference/context admission. The captured final
//! layer uses a device plan with one 16-row tile per (token, expert), not the
//! host plan's deduplicated padding. Backward also keeps buffers that inference
//! never allocates. Neither an oversized prefill budget nor unused anchor
//! reservations are a substitute for pricing these allocations.
//!
//! The ledger deliberately sums forward, recomputation and backward storage,
//! even where stream ordering permits reuse. It bounds the named tensor/array
//! storage in the current implementation; it is not a measured peak or a
//! throughput estimate. Pool pages, alignment/fragmentation and runtime overhead
//! remain the enclosing admission policy's responsibility. The frozen trunk,
//! KV contexts, EMA bank and transposed-head weight/binding reserve are separate.

use anyhow::{Context, Result, ensure};

use super::{config::InklingTextConfig, fp4gemm::MTILE, sdft};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceBytes {
    pub captured_rows: usize,
    pub device_rows: usize,
    pub routed_forward: u64,
    pub routed_recompute: u64,
    pub routed_backward: u64,
    pub route_metadata: u64,
    pub captured_residuals: u64,
    pub head: u64,
    pub normalization: u64,
    pub convolution: u64,
    pub retained_targets: u64,
    pub total: u64,
}

/// Maximum captured rows allowed by PreparedExample's prompt-plus-rollout
/// bound. The prompt is nonempty; its last token is already a response input,
/// leaving at most `context - rollout - 1` preceding prompt rows to capture.
pub fn captured_rows(rollout: usize, context: usize, kernel: usize) -> Result<usize> {
    ensure!(rollout > 0 && context > rollout, "SDFT needs response rows and a nonempty prompt");
    ensure!(kernel > 0, "SDFT short convolution needs a positive kernel");
    let antecedents = (kernel - 1).min(context - rollout - 1);
    rollout.checked_add(antecedents).context("SDFT captured row count overflow")
}

/// Price one rank's learning workspace before allocating or binding weights.
/// `local_intermediate` must be the already-validated TP share, not the global
/// width. Hidden/head widths and target distributions do NOT divide by TP.
pub fn workspace_bytes(
    t: &InklingTextConfig, config: &sdft::Config, local_intermediate: usize,
) -> Result<WorkspaceBytes> {
    config.validate()?;
    ensure!(t.intermediate_size > 0 && local_intermediate > 0
        && t.intermediate_size % local_intermediate == 0,
        "SDFT local intermediate width must divide the model's intermediate width");
    calculate(Shape {
        rollout: config.max_rollout, context: config.context_budget,
        sequences: config.sequences, kernel: if t.use_sconv { t.sconv_kernel_size } else { 1 },
        hidden: t.hidden_size, inter: local_intermediate,
        experts: t.n_routed_experts, shared: t.n_shared_experts,
        top_k: t.num_experts_per_tok,
        vocab: t.effective_vocab(), vocab_pad: t.vocab_size,
    })
}

#[derive(Clone, Copy)]
struct Shape {
    rollout: usize,
    context: usize,
    sequences: usize,
    kernel: usize,
    hidden: usize,
    inter: usize,
    experts: usize,
    shared: usize,
    top_k: usize,
    vocab: usize,
    vocab_pad: usize,
}

fn product(factors: &[usize]) -> Result<u64> {
    factors.iter().try_fold(1u64, |acc, &factor| {
        let factor = u64::try_from(factor).context("SDFT dimension exceeds u64")?;
        acc.checked_mul(factor).context("SDFT workspace byte count overflow")
    })
}

fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |acc, &value| {
        acc.checked_add(value).context("SDFT workspace byte count overflow")
    })
}

fn padded(value: usize, alignment: usize) -> Result<usize> {
    let groups = value.checked_add(alignment - 1).context("SDFT tensor padding overflow")?
        / alignment;
    groups.checked_mul(alignment).context("SDFT tensor padding overflow")
}

fn packed(rows: usize, columns: usize) -> Result<u64> {
    // The validated k64 geometry makes both divisions exact: four-bit codes
    // plus one E4M3 scale byte per sixteen values (fp4quant::quantize_nvfp4*).
    let values = product(&[rows, columns])?;
    sum(&[values / 2, values / 16])
}

fn calculate(s: Shape) -> Result<WorkspaceBytes> {
    ensure!((1..64).contains(&s.sequences) && (1..=64).contains(&s.rollout),
        "SDFT workspace requires 1..=63 students and 1..=64 response rows");
    ensure!(s.hidden > 0 && s.hidden % 64 == 0 && s.inter > 0 && s.inter % 64 == 0,
        "SDFT learning projections require positive k64 hidden/intermediate widths");
    ensure!(s.experts > 0 && s.top_k > 0 && s.top_k <= s.experts,
        "SDFT routing needs a positive top-k no larger than the expert bank");
    ensure!(s.vocab > 0 && s.vocab <= s.vocab_pad && s.vocab_pad % 64 == 0,
        "SDFT effective vocabulary must fit a nonempty k64 padded head");
    let n = captured_rows(s.rollout, s.context, s.kernel)?;
    let slots = n.checked_mul(s.top_k).context("SDFT routed slot count overflow")?;
    let m = slots.checked_mul(MTILE).context("SDFT device row count overflow")?;
    let h = s.hidden;
    let i = s.inter;
    let r = s.rollout;
    let head_rows = padded(r, MTILE)?;
    let router_width = s.experts.checked_add(s.shared).context("SDFT router width overflow")?;
    // F32 Burn matmul may round output extents to 64; the explicit BF16
    // router's m16/n8 padding fits inside the same storage bound.
    let router_rows = padded(n, 64)?;
    let router_columns = padded(router_width, 64)?;
    let topk_width = s.top_k.checked_mul(2).and_then(|v| v.checked_add(s.shared))
        .and_then(|v| v.checked_add(1)).context("SDFT top-k width overflow")?;

    // assembly::grouped_experts_core: gathered x, gate/up, gated activation,
    // down output, both packed activation pairs, and scattered output. Price
    // every staging/output at F32 so INK_ACT_BF16=0 cannot widen this estimate.
    let routed_forward = sum(&[
        product(&[m, h, 4])?, product(&[m, 2, i, 4])?,
        product(&[m, i, 4])?, product(&[m, h, 4])?,
        packed(m, h)?, packed(m, i)?, product(&[n, h, 4])?,
    ])?;
    // learn_last_layer recomputes x/gate/up explicitly in BF16, irrespective
    // of the forward staging setting. No reuse credit against forward above.
    let routed_recompute = sum(&[
        product(&[m, h, 2])?, packed(m, h)?, product(&[m, 2, i, 2])?,
    ])?;
    // Actual named backward arrays: g_rows F32 [m,h], g_act F32 [m,i],
    // g_both BF16 [m,2*i], and act BF16 [m,i]. Expert SGD itself accumulates
    // its gradient in registers, not in another full-sized weight tensor.
    let routed_backward = sum(&[
        product(&[m, h, 4])?, product(&[m, i, 4])?,
        product(&[m, 2, i, 2])?, product(&[m, i, 2])?,
    ])?;
    // DevRoute invariants + captured DevRowPlan + backward expert grouping.
    // Include router logits/top-k output without credit for an earlier route.
    let route_metadata = sum(&[
        product(&[m, 4])?,                   // row_tok
        product(&[slots, 3, 4])?,            // blk_slot/tile0/cnt
        product(&[slots, 4])?, product(&[n, 4])?, 4, // tok_rows/cnt/fault
        product(&[m, 4])?,                   // row_wgt
        product(&[slots, 2, 2, 8])?,         // off13/off2, stride=2
        product(&[slots, 2, 4])?, product(&[slots, 4])?, // scale2 pairs/ids
        product(&[s.experts, 2, 4])?, product(&[slots, 4])?, // group start/cnt/rows
        product(&[router_rows, router_columns, 4])?,
        product(&[router_rows, h, 2])?,       // padded BF16 router input
        product(&[n, topk_width, 4])?,
    ])?;
    let history = s.kernel - 1;
    // LearnKeep.hn/x_pre, the returned residual xd, and the preceding history.
    let captured_residuals = sum(&[product(&[n, h, 3, 4])?, product(&[history, h, 4])?])?;

    // Sum teacher and student head work even though they run at different
    // times. W4A16 pads M to 16 and produces F32 at vocab_pad, before slicing.
    // The 16 effective-vocab arrays allow: two slices, four old/new masked
    // logits, six stable-softmax elementwise outputs, teacher cast, target
    // slice, probability difference, and weighted gradient. Reduction work
    // additionally gets four full-vocabulary arrays plus scalar row chains;
    // charging only the three final target-validation scalars would miss it.
    let head = sum(&[
        product(&[head_rows, s.vocab_pad, 2, 4])?, // both head GEMM outputs
        product(&[r, s.vocab, 16, 4])?,
        product(&[r, s.vocab_pad, 2, 4])?,    // g_full zero + slice_assign
        product(&[head_rows, h, 2, 2])?,     // both forward head BF16 inputs
        product(&[head_rows, s.vocab_pad, 2])?, // backward head BF16 input
        product(&[head_rows, h, 4])?,        // backward head GEMM output
        product(&[r, s.vocab, 4, 4])?,       // reduction intermediates
        product(&[r, 32, 4])?,              // softmax/target-validation reductions
    ])?;
    // Both head RMS forward expressions: square, divide, gain and muP divide
    // plus mean/add/sqrt reductions. Backward charges g_hs and replacement,
    // g_rms, x cast, u, square, u*x, u*r, x*r^3, its product with dot, final
    // subtraction and g_slice assignment, all [n,h] F32. Scalar row chains
    // (r and dot, including powers/means) are charged separately.
    let normalization = sum(&[
        product(&[r, h, 2, 4, 4])?, product(&[r, 2, 3, 4])?,
        product(&[n, h, 12, 4])?, product(&[n, 12, 4])?,
    ])?;
    // Bound both the uncached first-pass convolution and cached window path.
    // Per tap, allow three forward arrays (slice/product/sum) and five
    // backward arrays (slice/zero/concatenation/product/sum), each no larger
    // than [n,h]. Charge all taps even though only a few coexist. History,
    // joined input and the final residual/output are additional F32 storage.
    let convolution = sum(&[
        product(&[s.kernel, n, h, 8, 4])?, product(&[s.kernel, h, 4])?,
        product(&[history, h, 2, 4])?, product(&[n, h, 3, 4])?,
    ])?;
    // Every slot's targets remain alive until the cohort's EMA boundary, not
    // only until that slot learns. This is explicit even if capsule admission
    // independently over-reserves now-disabled anchors. Never divide by TP.
    let retained_targets = product(&[s.sequences, r, s.vocab, 4])?;
    let total = sum(&[routed_forward, routed_recompute, routed_backward, route_metadata,
        captured_residuals, head, normalization, convolution, retained_targets])?;
    Ok(WorkspaceBytes { captured_rows: n, device_rows: m, routed_forward, routed_recompute,
        routed_backward, route_metadata, captured_residuals, head, normalization,
        convolution, retained_targets, total })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        Shape { rollout: 4, context: 32, sequences: 2, kernel: 3,
            hidden: 64, inter: 128, experts: 4, shared: 1, top_k: 2,
            vocab: 63, vocab_pad: 64 }
    }

    #[test]
    fn convolution_capture_is_bounded_by_available_prompt_antecedents() {
        assert_eq!(captured_rows(64, 4096, 4).unwrap(), 67);
        assert_eq!(captured_rows(4, 5, 100).unwrap(), 4);
        assert_eq!(captured_rows(4, 7, 100).unwrap(), 6);
        assert_eq!(captured_rows(4, 32, 1).unwrap(), 4);
        assert!(captured_rows(0, 32, 4).is_err());
        assert!(captured_rows(4, 4, 4).is_err());
        assert!(captured_rows(4, 32, 0).is_err());
    }

    #[test]
    fn every_token_expert_pair_gets_its_own_device_tile() {
        let s = shape();
        let b = calculate(s).unwrap();
        assert_eq!(b.captured_rows, 6);
        assert_eq!(b.device_rows, 6 * 2 * MTILE);
        let host_rows = 6 * 2 + (MTILE - 1) * s.experts;
        assert!(b.device_rows > host_rows);
        assert_eq!(b.routed_recompute, (192 * 64 * 2 + 192 * 64 * 9 / 16
            + 192 * 2 * 128 * 2) as u64);
        assert_eq!(b.routed_backward, (192 * 64 * 4 + 192 * 128 * 4
            + 192 * 2 * 128 * 2 + 192 * 128 * 2) as u64);
    }

    #[test]
    fn all_cohort_targets_are_charged_until_ema_not_just_one_slot() {
        let one = calculate(Shape { sequences: 1, ..shape() }).unwrap();
        let three = calculate(Shape { sequences: 3, ..shape() }).unwrap();
        assert_eq!(three.retained_targets, 3 * 4 * 63 * 4);
        assert_eq!(three.total - one.total, 2 * 4 * 63 * 4);
    }

    #[test]
    fn tp_local_intermediate_only_shrinks_intermediate_buffers() {
        let full = calculate(shape()).unwrap();
        let half = calculate(Shape { inter: 64, ..shape() }).unwrap();
        assert!(half.routed_forward < full.routed_forward);
        assert!(half.routed_recompute < full.routed_recompute);
        assert!(half.routed_backward < full.routed_backward);
        assert_eq!(half.head, full.head);
        assert_eq!(half.retained_targets, full.retained_targets);
        assert_eq!(half.normalization, full.normalization);
        assert_eq!(half.device_rows, full.device_rows);
    }

    #[test]
    fn accounting_is_explicitly_additive_without_reuse_credit() {
        let b = calculate(shape()).unwrap();
        assert_eq!(b.total, b.routed_forward + b.routed_recompute + b.routed_backward
            + b.route_metadata + b.captured_residuals + b.head + b.normalization
            + b.convolution + b.retained_targets);
        let wider = calculate(Shape { rollout: 17, context: 32, ..shape() }).unwrap();
        assert!(wider.total > b.total);
        assert!(wider.head > b.head);
    }

    #[test]
    fn invalid_geometry_and_checked_overflow_are_refused() {
        for invalid in [
            Shape { hidden: 0, ..shape() }, Shape { inter: 63, ..shape() },
            Shape { top_k: 0, ..shape() }, Shape { top_k: 5, ..shape() },
            Shape { vocab: 65, ..shape() }, Shape { vocab_pad: 63, ..shape() },
            Shape { rollout: 65, context: 100, ..shape() },
            Shape { sequences: 64, ..shape() },
            Shape { context: usize::MAX, kernel: usize::MAX, ..shape() },
            Shape { hidden: usize::MAX / 64 * 64, ..shape() },
        ] {
            assert!(calculate(invalid).is_err());
        }
        assert!(product(&[usize::MAX, usize::MAX, 16]).is_err());
        assert!(sum(&[u64::MAX, 1]).is_err());
        assert!(padded(usize::MAX, 16).is_err());
    }
}
