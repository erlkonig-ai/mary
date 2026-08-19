//! The attention score epilogue as ONE kernel: scale, relative-position bias,
//! causal mask and sliding window, in a single pass over `[heads, n, n]`.
//!
//! # What it replaces, and why that mattered
//!
//! [`super::burn::attention_prefill`] built the biased, masked score matrix out
//! of Burn ops, and every one of them is a full pass over an `[heads, n, n]`
//! f32 tensor:
//!
//! ```text
//! idx    = [n, n] i32 built on the HOST, one scalar loop over n^2
//! valid  = [n, n] f32 built on the HOST, the same loop again
//! idx   -> repeat_dim(0, heads)          heads * n^2 i32 WRITTEN
//! bias   = rel.gather(2, idx)            heads * n^2 read + written
//! bias   = bias * valid                  heads * n^2 read + written
//! scores = qk * scaling                  heads * n^2 read + written
//! scores = scores + bias                 heads * n^2 read + written
//! scores = scores + mask                 heads * n^2 read + written, and the
//!                                        mask is itself an [n, n] host Vec
//!                                        cloned and uploaded once per pass
//! ```
//!
//! That is five materialised `heads * n^2` tensors and about seven full passes
//! over one, for an epilogue whose entire content is a function of `(q, k)`
//! that fits in a register. At 512 tokens the waste is a few milliseconds. At
//! 14124 tokens one `[32, n, n]` f32 tensor is **25.5 GiB**, the host loops run
//! `2 * n^2 = 400 M` scalar iterations PER LAYER PER PASS, and the allocator
//! runs the node out of memory: the measured ceiling before the CUDA allocator
//! starts refusing 32-GiB buffers -- silently, with the process still exiting 0
//! and printing a garbage answer -- is between 14 k and 16 k tokens.
//!
//! Here the same epilogue is one kernel, one pass, zero extra tensors:
//!
//! ```text
//! dist = q - k
//! vis  = dist >= 0 && (window == 0 || dist < window)
//! b    = (0 <= dist < eff) ? rel[q, h, dist] : 0
//! s    = vis ? qk * scaling + b : -3.4028235e38
//! ```
//!
//! `rel` is `[n, heads, eff]` with `eff = min(rel_extent, n) <= 1024`, so it is
//! linear in `n` and stays small. Nothing else is materialised. That layout is
//! the one the projection already produces -- swapping it to `[heads, n, eff]`
//! would read better and cost a full copy, because Burn reshapes a permuted
//! view by copying it.
//!
//! # Why there is one launch per head
//!
//! Every device-side index here is 32 bits -- `ABSOLUTE_POS_X` is a `u32`, and
//! so, on this runtime, is a cubecl `usize` inside a kernel. `heads * n^2`
//! passes `u32::MAX` at `n = 11586`, so a single grid over the whole score
//! matrix wraps for the high heads. That failure is SILENT and it is not
//! hypothetical: the first version of this file did exactly that, agreed with
//! the Burn lane to four decimals at 512, 3732 and 7000 tokens, and produced a
//! visibly different layer-RMS ladder at 14124 (2.12 against 1.57 at layer 0).
//! Three sizes agreeing is what a 32-bit overflow looks like from below.
//!
//! So the head is a launch, not a grid dimension: `Handle::offset_start` moves
//! the base pointer by `head * n^2 * 4` bytes and the kernel indexes `0..n^2`.
//! `rel` stays whole because `n * heads * eff` is 4.6e8 at the memory ceiling,
//! and that is asserted rather than assumed. The extra launches are 32 per
//! layer against a kernel that runs for tens of milliseconds.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// Threads per cube. One thread per score element.
const CUBE_SIZE: u32 = 256;

/// One score element: scale it, add its relative-position bias, mask it.
///
/// `scores` is modified in place -- it is the raw `q @ k^T` on the way in and
/// the softmax input on the way out. In place because the alternative is a
/// second `heads * n^2` allocation, which is the thing this kernel exists to
/// avoid.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn score_epilogue_kernel(
    scores: &mut Array<f32>,
    rel: &Array<f32>,
    scaling: f32,
    tokens: u32,
    heads: u32,
    head: u32,
    eff: u32,
    row: u32,
    window: u32,
) {
    let flat = ABSOLUTE_POS_X;
    if flat < tokens * tokens {
        let q = flat / tokens;
        let k = flat % tokens;
        // The score matrix is PADDED, not contiguous: `row` is the matmul's
        // own row stride, 7040 where `tokens` is 7000. Honouring it here is
        // what lets the caller skip a 6.3 GiB contiguity copy per layer.
        let i = q * row + k;

        // Causality and the window, as the predicate the additive mask encoded.
        if k <= q {
            let dist = q - k;
            if window == 0u32 || dist < window {
                let mut b = f32::new(0.0);
                if dist < eff {
                    b = rel[(q * heads * eff + head * eff + dist) as usize];
                }
                scores[i as usize] = scores[i as usize] * scaling + b;
            } else {
                scores[i as usize] = f32::new(-3.4028235e38);
            }
        } else {
            scores[i as usize] = f32::new(-3.4028235e38);
        }
    }
}

/// Launch the epilogue over `[heads, tokens, tokens]`, in place on `scores`.
///
/// `rel` is `[tokens, heads, eff]`. `window` is `Some(w)` on a local layer and
/// `None` on a global one -- the same distinction
/// [`super::attn::causal_mask`] took, expressed as a predicate instead of as an
/// `n^2` tensor of zeros and negative infinities.
#[allow(clippy::too_many_arguments)]
pub fn score_epilogue_launch<R: Runtime>(
    client: &ComputeClient<R>,
    scores: &Handle,
    rel: &Handle,
    heads: usize,
    tokens: usize,
    eff: usize,
    strides: [usize; 3],
    scaling: f32,
    window: Option<usize>,
) {
    assert!(tokens > 0 && heads > 0, "an empty attention has no epilogue");
    assert!(eff > 0, "the relative table must reach at least one distance");
    let per_head = tokens * tokens;
    assert!(
        per_head <= u32::MAX as usize,
        "{tokens} tokens is {per_head} score elements, past the 32-bit launch index"
    );
    assert!(
        strides[0] <= u32::MAX as usize,
        "a head is {} elements, past the 32-bit index",
        strides[0]
    );
    assert!(
        tokens * heads * eff <= u32::MAX as usize,
        "the relative table is {} elements, past the 32-bit index",
        tokens * heads * eff
    );
    assert_eq!(strides[2], 1, "the innermost stride is {}, not 1", strides[2]);
    assert!(strides[1] >= tokens, "a row stride of {} cannot hold {tokens}", strides[1]);
    let w = window.unwrap_or(0);
    let cubes = (per_head as u32).div_ceil(CUBE_SIZE);
    let f32b = core::mem::size_of::<f32>();
    for head in 0..heads {
        // The head's slice of the score matrix, as its own array. One launch
        // per head rather than a `[cubes, heads]` grid because the flat index
        // `head * n^2 + i` is 6.4e9 at 14124 tokens and every device-side index
        // here is 32-bit: a single grid produced a silently WRONG answer above
        // n = 11586 -- heads 22..31 wrapped -- while agreeing to four decimals
        // at 512, 3732 and 7000. `offset_start` moves the base pointer instead,
        // so the largest index a kernel forms is n^2.
        let slice = scores.clone().offset_start((head * strides[0] * f32b) as u64);
        unsafe {
            score_epilogue_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(CUBE_SIZE),
                ArrayArg::from_raw_parts(slice, strides[0]),
                ArrayArg::from_raw_parts(rel.clone(), heads * tokens * eff),
                scaling,
                tokens as u32,
                heads as u32,
                head as u32,
                eff as u32,
                strides[1] as u32,
                w as u32,
            );
        }
    }
}
