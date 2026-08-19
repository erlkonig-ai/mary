//! What one attention layer asks the allocator for, and whether it will get it.
//!
//! Prefill attention materialises `[heads, tokens, tokens]` f32 per layer. That
//! is 33.5 MB at 512 tokens, 1.68 GiB at 3,732, 23.8 GiB at 14,124 and 32 GiB
//! at 16,384 -- it grows as `n^2`, so every 41% more tokens doubles it -- and
//! it is by a wide margin the largest single thing the run allocates.
//!
//! # The ceiling is a per-BUFFER cap, not a shortage
//!
//! cubecl's CUDA runtime sets `MemoryDeviceProperties::max_page_size` to
//! `cuDeviceTotalMem / 4`, and every memory pool it builds has a
//! `max_alloc_size` at or below that. A request larger than the biggest pool's
//! is refused outright, however much of the node is free: on a 119.6 GiB box
//! the cap is 29.9 GiB, so `[32, 15600, 15616]` f32 (29.0 GB) is served and
//! `[32, 16384, 16384]` f32 (32 GiB) is not. Adding memory does not move it;
//! only a fourfold larger device would.
//!
//! Refused is also the WORST case rather than the loudest one, because the
//! `Result` is unwrapped on a cubecl worker thread and a worker-thread panic
//! does not end the process. See [`super::fatal`] for what that produced.
//! This module is the half that refuses BEFORE the run spends a minute
//! copying weights: [`check`] is one comparison against a number the device
//! already told us, and it is exact -- there is nothing fitted in it.

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

/// Bytes in one layer's `[heads, tokens, tokens]` f32 score matrix, padding
/// included.
pub fn score_matrix_bytes(heads: usize, tokens: usize) -> u64 {
    heads as u64
        * tokens as u64
        * tokens.next_multiple_of(MATMUL_ROW_ALIGN) as u64
        * core::mem::size_of::<f32>() as u64
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

/// What prefill will hold in score matrices at this sequence length.
///
/// This is the term the admission gate did not have. It is quadratic in the
/// sequence and flat in the layer count -- one layer's matrices are freed
/// before the next layer allocates its own -- which is the opposite shape from
/// the per-layer constant that stood in for it.
pub fn prefill_peak_bytes(heads: usize, tokens: usize) -> u64 {
    LIVE_SCORE_MATRICES * score_matrix_bytes(heads, tokens)
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

/// The longest sequence whose score matrix fits under `cap`.
///
/// Bisected rather than solved, because `score_matrix_bytes` rounds and the
/// closed form would have to round with it. The predicate is monotone in
/// `tokens`, so the bisection is exact.
pub fn longest_sequence(heads: usize, cap: u64) -> usize {
    let (mut lo, mut hi) = (0usize, 1usize << 24);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if score_matrix_bytes(heads, mid) <= cap {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Refuse a sequence this device cannot hold the scores for, before the first
/// allocation rather than after the 224th.
pub fn check<R: Runtime>(client: &ComputeClient<R>, heads: usize, tokens: usize) -> Result<()> {
    let want = score_matrix_bytes(heads, tokens);
    let cap = largest_allocation(client);
    anyhow::ensure!(
        want <= cap,
        "{tokens} tokens needs a [{heads}, {tokens}, {tokens}] f32 score matrix per attention \
         layer -- {want} bytes, {:.2} GiB -- and this device refuses any single allocation over \
         {cap} bytes ({:.2} GiB).\n  \
         That cap is cuDeviceTotalMem / 4 and free memory does not raise it. The longest \
         sequence whose scores fit is {} tokens.\n  \
         Refusing here rather than at the allocator, because the allocator's refusal happens on \
         a worker thread, does not end the process, and returns a plausible answer read out of a \
         buffer nothing ever wrote.",
        want as f64 / GIB,
        cap as f64 / GIB,
        longest_sequence(heads, cap),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{longest_sequence, score_matrix_bytes};

    /// The two sizes the ceiling was measured at, and the padding in between.
    #[test]
    fn counts_the_padding() {
        // 7000 rounds to 7040, which is the stride Burn actually reports.
        assert_eq!(score_matrix_bytes(32, 7000), 32 * 7000 * 7040 * 4);
        // 16384 is already aligned, and is exactly 32 GiB -- the buffer the
        // allocator was measured refusing.
        assert_eq!(score_matrix_bytes(32, 16384), 34_359_738_368);
    }

    #[test]
    fn bisects_the_boundary() {
        // A 119.6 GiB node: cuDeviceTotalMem / 4 is a shade under 30 GiB, and
        // the measured boundary is 15,600 served / 16,384 refused.
        let cap = 128_408_297_472u64 / 4;
        let n = longest_sequence(32, cap);
        assert!(score_matrix_bytes(32, n) <= cap);
        assert!(score_matrix_bytes(32, n + 1) > cap);
        assert!((15_600..16_384).contains(&n), "boundary landed at {n}");
    }
}
