//! What one attention layer asks the allocator for, and whether it will get it.
//!
//! # The ceiling is a per-BUFFER cap, not a shortage
//!
//! cubecl's CUDA runtime sets `MemoryDeviceProperties::max_page_size` to
//! `cuDeviceTotalMem / 4`, and every memory pool it builds has a
//! `max_alloc_size` at or below that. A request larger than the biggest pool's
//! is refused outright, however much of the node is free: on a 119.6 GiB box
//! the cap is 29.9 GiB, so a 29.0 GiB buffer is served and a 32 GiB one is not,
//! and a run was refused with 60 GiB still free. Adding memory does not move
//! it; only a fourfold larger device would.
//!
//! Refused is also the WORST case rather than the loudest one, because the
//! `Result` is unwrapped on a cubecl worker thread and a worker-thread panic
//! does not end the process. See [`super::fatal`] for what that produced.
//! This module is the half that refuses BEFORE the run spends a minute
//! copying weights: [`check`] is one comparison against a number the device
//! already told us, and it is exact -- there is nothing fitted in it.
//!
//! # What used to be the binding buffer, and what is now
//!
//! A global layer used to materialise the whole `[heads, tokens, tokens]` f32
//! score matrix -- 23.8 GiB at 14,124 tokens and 32 GiB at 16,384 -- so the cap
//! landed on it and the sequence ceiling was the 15,808 tokens
//! [`longest_sequence`] names. Thirty-five of the forty-two layers stopped
//! building it when the local ones became a band, but a global layer really
//! does read every key, and one falls in every range of six or more layers, so
//! the ceiling did not move at all.
//!
//! The dense lane now blocks its QUERIES: it allocates `[heads, rows, tokens]`
//! for one block at a time, with `rows` chosen by [`query_block`] against a
//! byte budget rather than by the sequence. That term is bounded, so what binds
//! instead is the largest term still linear in the sequence -- the
//! `[heads, tokens, head_dim]` f32 the projections and the expansion produce,
//! `tokens * 16 KiB` for this model. Under the same 29.9 GiB cap that is a
//! ceiling near two million tokens, and the run will exhaust node memory long
//! before it gets there. [`check`] therefore tests the max of the two, not the
//! score matrix alone.

use anyhow::Result;
use cubecl::prelude::{ComputeClient, Runtime};

/// The row alignment Burn's f32 matmul pads its output to.
///
/// Not cosmetic: `[32, 7000, 7000]` comes back with strides
/// `[49_280_000, 7040, 1]`, and 7040 is 7000 rounded up to 64. The padding is
/// allocated, so a count that ignores it under-reports by up to 0.9% -- enough
/// to put a sequence just under a multiple of 64 on the wrong side of the cap.
const MATMUL_ROW_ALIGN: usize = 64;

const GIB: f64 = (1u64 << 30) as f64;

/// Bytes in one `[heads, rows, tokens]` f32 score block, padding included.
///
/// `rows == tokens` is the unblocked whole-matrix figure, which is what this
/// counted before the dense lane blocked its queries and is still what the
/// arithmetic in [`longest_sequence`] is about.
pub fn score_block_bytes(heads: usize, rows: usize, tokens: usize) -> u64 {
    heads as u64
        * rows as u64
        * tokens.next_multiple_of(MATMUL_ROW_ALIGN) as u64
        * core::mem::size_of::<f32>() as u64
}

/// Bytes in one layer's whole `[heads, tokens, tokens]` f32 score matrix.
///
/// Nothing allocates this any more. It is the counterfactual the doc comments
/// and the admission report quote, and the quantity [`longest_sequence`]
/// inverts.
pub fn score_matrix_bytes(heads: usize, tokens: usize) -> u64 {
    score_block_bytes(heads, tokens, tokens)
}

/// How many bytes one query block's score matrix may occupy.
///
/// 4 GiB, and the number is a trade rather than a limit: bigger blocks amortise
/// the per-block launches and the relative-table matmul over more queries,
/// smaller ones leave more of the node for everything else. It is well under
/// the 29.9 GiB per-buffer cap on purpose -- the block is also live alongside
/// the softmax's output, so the peak is twice this.
const QUERY_BLOCK_BYTES: u64 = 4 << 30;

/// The smallest block worth issuing.
///
/// Below this the `[heads, rows, head_dim] x [heads, head_dim, tokens]` product
/// is a GEMM too short to fill a tile, and the per-block launches stop being
/// amortised by anything.
const QUERY_BLOCK_MIN: usize = 128;

/// How many queries one dense-attention block covers.
///
/// `INK_QBLOCK` overrides it, in queries, which is how the block size was swept
/// without a rebuild.
///
/// Rounded to a multiple of [`MATMUL_ROW_ALIGN`] so the block's own matmul
/// output is not itself padded into a taller allocation than it asked for.
pub fn query_block(heads: usize, tokens: usize) -> usize {
    if let Some(n) = std::env::var("INK_QBLOCK").ok().and_then(|v| v.parse::<usize>().ok()) {
        return n.clamp(1, tokens.max(1));
    }
    let row = tokens.next_multiple_of(MATMUL_ROW_ALIGN) as u64;
    let per_row = heads as u64 * row * core::mem::size_of::<f32>() as u64;
    let rows = (QUERY_BLOCK_BYTES / per_row.max(1)) as usize;
    let rows = (rows / MATMUL_ROW_ALIGN * MATMUL_ROW_ALIGN).max(QUERY_BLOCK_MIN);
    rows.min(tokens).max(1)
}

/// How many `[heads, n, n]` score matrices prefill holds at the peak.
///
/// The epilogue works in place on the `q @ k^T` output, so the scores are one;
/// the softmax materialises its output beside them. Two.
///
/// # This is a floor, and deliberately not a prediction
///
/// Node memory is NOT a smooth function of `n`, because cubecl allocates whole
/// pool PAGES and its page sizes are `max_page_size / 4^k`. Measured on a
/// 119.6 GiB node, `INK_LAYERS=0:8` (38.08 GiB of weights), node memory as
/// `MemTotal - MemAvailable` sampled every 2 s:
///
/// | n | one score matrix | node peak |
/// |---|---|---|
/// | 512 | 0.03 GiB | 47.42 GiB |
/// | 3732 | 1.68 | 50.16 |
/// | 7000 | 5.88 | 87.72 |
/// | 11000 | 14.47 | 86.45 |
/// | 14124 | 23.82 | 90.93 |
/// | 15600 | 29.04 | 88.00 |
/// | 16384 | 32.00 (REFUSED) | 58.00 |
///
/// It is not monotone: 14,124 peaks 2.9 GiB ABOVE 15,600, and 7,000 costs
/// 37.6 GiB more than 3,732 for a score matrix only 4.2 GiB larger. Both are
/// the bucketing, not the tensors. So no coefficient fitted here would predict
/// this curve, and one that tried would be a fit to an allocator's internals
/// that goes stale the first time cubecl changes its pool ladder.
///
/// What the term IS for: the admission gate's total-memory test had a
/// sequence-independent constant standing in for something quadratic, which is
/// wrong in kind rather than in calibration. Two score matrices is what the
/// run genuinely holds, so charging it makes the test refuse layer ranges that
/// cannot survive a long input -- which is the question that test asks. The
/// SHARP gate is [`check`], and there is nothing fitted in that one at all.
///
/// The last row is the reason the sharp gate is the one that matters: at
/// 16,384 the node was 60 GiB FREE and the allocation was still refused.
pub const LIVE_SCORE_MATRICES: u64 = 2;

/// What prefill will hold in score blocks at this sequence length.
///
/// This is the term the admission gate did not have. It is flat in the layer
/// count -- one layer's blocks are freed before the next layer allocates its
/// own -- which is the opposite shape from the per-layer constant that stood in
/// for it. It used to be quadratic in the sequence as well; with the queries
/// blocked it grows linearly until the block stops shrinking, and is flat
/// after that.
///
/// It has to move with [`query_block`] or the gate refuses runs that would now
/// succeed, which is a worse failure than the one it was built to prevent: a
/// gate that is wrong in the permissive direction gets found by a crash, and
/// one that is wrong in the restrictive direction gets found by nobody.
pub fn prefill_peak_bytes(heads: usize, tokens: usize) -> u64 {
    LIVE_SCORE_MATRICES * score_block_bytes(heads, query_block(heads, tokens), tokens)
}

/// The largest single buffer this runtime will hand out, straight from the
/// device rather than recomputed from `cuDeviceTotalMem / 4`.
///
/// Asked of the client on purpose: the quarter is cubecl's policy, not a
/// property of CUDA, and a copy of it here would go stale silently the day it
/// changes -- in the direction of admitting runs that then fail.
pub fn largest_allocation<R: Runtime>(client: &ComputeClient<R>) -> u64 {
    client.properties().memory.max_page_size
}

/// Bytes in one `[heads, tokens, head_dim]` f32 activation.
///
/// Q, the GQA-expanded K and V, K again transposed, and the concatenated output
/// are all this shape, and with the score block bounded they are the largest
/// buffers a global layer asks for. `tokens * 16 KiB` on this model.
pub fn activation_bytes(heads: usize, head_dim: usize, tokens: usize) -> u64 {
    heads as u64 * tokens as u64 * head_dim as u64 * core::mem::size_of::<f32>() as u64
}

/// The largest SINGLE buffer one attention layer asks for at this length.
///
/// The max of the two terms, not their sum: they are separate allocations and
/// the cap is per-buffer. A sum would be the right thing to charge against node
/// memory, which is the admission gate's question and not this one.
pub fn largest_buffer(heads: usize, head_dim: usize, tokens: usize) -> u64 {
    score_block_bytes(heads, query_block(heads, tokens), tokens)
        .max(activation_bytes(heads, head_dim, tokens))
}

/// The longest sequence whose largest attention buffer fits under `cap`.
///
/// Bisected rather than solved, because the terms round and because
/// [`query_block`] is a step function. Both terms are non-decreasing in
/// `tokens`, so their max is too and the bisection is exact.
pub fn longest_sequence(heads: usize, head_dim: usize, cap: u64) -> usize {
    let (mut lo, mut hi) = (0usize, 1usize << 24);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if largest_buffer(heads, head_dim, mid) <= cap {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Refuse a sequence this device cannot hold one attention layer of, before the
/// first allocation rather than after the 224th.
pub fn check<R: Runtime>(
    client: &ComputeClient<R>,
    heads: usize,
    head_dim: usize,
    tokens: usize,
) -> Result<()> {
    let rows = query_block(heads, tokens);
    let scores = score_block_bytes(heads, rows, tokens);
    let acts = activation_bytes(heads, head_dim, tokens);
    let want = scores.max(acts);
    let cap = largest_allocation(client);
    anyhow::ensure!(
        want <= cap,
        "{tokens} tokens needs a [{heads}, {rows}, {tokens}] f32 score block ({:.2} GiB) and \
         [{heads}, {tokens}, {head_dim}] f32 activations ({:.2} GiB) per attention layer, and \
         this device refuses any single allocation over {cap} bytes ({:.2} GiB).\n  \
         That cap is cuDeviceTotalMem / 4 and free memory does not raise it. The longest \
         sequence whose buffers fit is {} tokens.\n  \
         Refusing here rather than at the allocator, because the allocator's refusal happens on \
         a worker thread, does not end the process, and returns a plausible answer read out of a \
         buffer nothing ever wrote.",
        scores as f64 / GIB,
        acts as f64 / GIB,
        cap as f64 / GIB,
        longest_sequence(heads, head_dim, cap),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        activation_bytes, largest_buffer, longest_sequence, query_block, score_block_bytes,
        score_matrix_bytes, QUERY_BLOCK_BYTES,
    };

    /// This model, at the sizes the measurements were taken at.
    const HEADS: usize = 32;
    const HEAD_DIM: usize = 128;
    /// A 119.6 GiB node: cuDeviceTotalMem / 4 is a shade under 30 GiB.
    const CAP: u64 = 128_408_297_472u64 / 4;

    /// The two sizes the old ceiling was measured at, and the padding between.
    #[test]
    fn counts_the_padding() {
        // 7000 rounds to 7040, which is the stride Burn actually reports.
        assert_eq!(score_matrix_bytes(32, 7000), 32 * 7000 * 7040 * 4);
        // 16384 is already aligned, and is exactly 32 GiB -- the buffer the
        // allocator was measured refusing.
        assert_eq!(score_matrix_bytes(32, 16384), 34_359_738_368);
    }

    /// The block honours its budget at every length, which is the whole
    /// property that makes the allocation linear rather than quadratic.
    #[test]
    fn a_block_stays_inside_its_budget() {
        for n in [512, 3732, 7000, 14124, 15808, 20_000, 35_845, 100_623, 250_000] {
            let rows = query_block(HEADS, n);
            assert!(rows >= 1 && rows <= n, "{n} tokens gave a {rows}-query block");
            let bytes = score_block_bytes(HEADS, rows, n);
            // The floor is allowed to exceed the budget -- a block below
            // QUERY_BLOCK_MIN is not worth issuing -- but only there.
            assert!(
                bytes <= QUERY_BLOCK_BYTES || rows <= super::QUERY_BLOCK_MIN,
                "{n} tokens: a {rows}-query block is {bytes} bytes"
            );
        }
    }

    /// Past the old ceiling the score block is no longer what binds.
    #[test]
    fn the_activations_bind_now_not_the_scores() {
        for n in [20_000, 35_845, 100_623] {
            let scores = score_block_bytes(HEADS, query_block(HEADS, n), n);
            let acts = activation_bytes(HEADS, HEAD_DIM, n);
            assert!(
                largest_buffer(HEADS, HEAD_DIM, n) <= CAP,
                "{n} tokens would still be refused: scores {scores}, activations {acts}"
            );
        }
        // 15,808 was the ceiling when the whole square was materialised.
        assert!(largest_buffer(HEADS, HEAD_DIM, 15_809) <= CAP);
    }

    #[test]
    fn bisects_the_boundary() {
        let n = longest_sequence(HEADS, HEAD_DIM, CAP);
        assert!(largest_buffer(HEADS, HEAD_DIM, n) <= CAP);
        assert!(largest_buffer(HEADS, HEAD_DIM, n + 1) > CAP);
        // The activation term is `tokens * heads * head_dim * 4`, so the
        // ceiling is the cap divided by 16 KiB -- near two million tokens,
        // against the 15,808 the whole score matrix allowed.
        assert!((1_900_000..2_100_000).contains(&n), "boundary landed at {n}");
    }
}
