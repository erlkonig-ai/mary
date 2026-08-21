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

use crate::models::inkling::config::{AttnKind, InklingTextConfig};
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

/// The floor cubecl's page ladder puts under any run that allocates at all.
///
/// # A pool hands out slices of a WHOLE page, and keeps the page
///
/// `MemoryConfiguration::SubSlices` builds its pools by quartering
/// `max_page_size` down to 32 MB, so on a 119.6 GiB device the page sizes are
/// 29.91 GiB, 7.48, 1.87, 0.47 and 0.12, and each pool takes only slices up to
/// `page_size / 2^k` -- 3.74 GiB for the 7.48 GiB pool, 0.47 for the 1.87 one.
/// A request larger than a pool's slice limit falls through to the next pool
/// up, and the last one has no limit below its own 29.91 GiB page.
///
/// The pool allocates that page whole, and `SlicedPool::cleanup` returns a page
/// only when EVERY slice of it is free. So one allocation over 3.74 GiB takes a
/// 29.91 GiB page and holds it for as long as anything on it is live.
///
/// This model always has one. The score blocks are 3.75 to 4.00 GiB at every
/// sequence [`query_block`] admits -- that is what the 4 GiB budget means --
/// and past about 40,000 tokens the routed lane's gather is larger still. So
/// the largest page is always out, and the 7.48 GiB one under it goes the same
/// way to the next size class down.
///
/// # Measured
///
/// `INK_LAYERS=0:8`, `INK_REPEAT=1`, node peak as `MemTotal - MemAvailable`
/// sampled every second, against the 38.08 GiB weight share plus the terms
/// [`super::pile`] already charged plus [`prefill_activation_bytes`]:
///
/// | tokens | activations charged | pool reserved | node peak | swap |
/// |---|---|---|---|---|
/// | 16,384 | 10.61 GiB | 41.74 | 89.02 | 0.34 |
/// | 32,768 | 13.20 | 42.20 | 89.68 | 0.34 |
/// | 65,536 | 25.53 | 50.15 | 104.62 | 0.34 |
/// | 81,920 | 31.91 | 50.50 | 100.29 | 0.34 |
/// | 100,623 | 39.19 | 68.02 | 113.26 | 6.65 |
///
/// The node peak is not monotone -- 81,920 tokens peak 4.3 GiB BELOW 65,536 --
/// which is the page ladder and not the tensors, and is why nothing fitted to
/// this curve would stay true. The pool reserves 41.74 GiB to hold 1.14 GiB of
/// live tensors at 16,384 tokens, 40.60 GiB of it STRANDED over 104 slices.
///
/// Charging the two largest pages is what closes the gap. With it the estimate
/// is 91.91 GiB against a measured 89.02 at 16,384, 106.83 against 104.62 at
/// 65,536, and 113.21 against 100.29 at 81,920 -- above the truth everywhere,
/// which is what a gate has to be -- and 120.49 at 100,623, over the 119.63 GiB
/// machine. So 81,920 is admitted with 6.4 GiB of nominal headroom and runs at
/// 100.29 GiB touching no swap, and 100,623 is refused: it runs at 113.26 GiB
/// with 6.7 GiB of swap and takes 296.7 s for a pass that costs 137.9 s when
/// the same input is not thrashing.
///
/// It is NOT a fitted coefficient. It is two page sizes off a ladder the
/// runtime derives from the device, which is why it is a fraction of the
/// machine and not a constant: the two nodes this runs on differ by 2 GiB and
/// their page ladders differ with them.
pub fn pool_page_floor(machine: u64) -> u64 {
    machine / 4 + machine / 16
}

/// One element of a KV cache, in bytes.
///
/// Read from [`super::burn::attn_bf16`] rather than assumed, because the cache
/// dtype is an env switch and a gate that priced a BF16 cache at f32 would
/// refuse ranges that fit.
fn cache_elem_bytes() -> u64 {
    if super::burn::attn_bf16() {
        2
    } else {
        4
    }
}

/// What ONE attention layer holds in activations at this sequence length.
///
/// A SUM and not a max: every buffer counted here is `let`-bound in
/// [`super::burn::attention_prefill_lane`] and lives to the end of it, so they
/// are all resident together at the layer's peak.
///
/// The two arms are different functions, not two settings of one, and their
/// largest terms are in different places:
///
/// * a LOCAL layer builds its relative-position table for the WHOLE sequence in
///   one launch -- `[tokens, heads, sliding_window_size]` f32, which is 64 KiB
///   a token on this model, four times Q and the largest single activation
///   outside the MoE lane. It is what the band reads instead of a square.
/// * a GLOBAL layer materialises Q, K^T and V contiguously ahead of its block
///   loop -- three `[heads, tokens, head_dim]` f32 -- and accumulates the
///   blocks' outputs before concatenating them. Its score blocks are
///   [`prefill_peak_bytes`], counted here because they are live inside this
///   layer and nowhere else.
pub fn attention_activation_bytes(
    t: &InklingTextConfig,
    kind: AttnKind,
    tokens: usize,
) -> u64 {
    let (heads, kv_heads, head_dim) = t.heads(kind);
    let n = tokens as u64;
    let f32b = core::mem::size_of::<f32>() as u64;
    let q = n * (heads * head_dim) as u64 * f32b;
    let kv = n * (kv_heads * head_dim) as u64 * f32b;
    let rel_proj = n * (heads * t.d_rel) as u64 * f32b;
    let hidden = n * t.hidden_size as u64 * f32b;
    // Q, the two pre-convolution projections, the two convolved ones (the
    // pre-convolution pair is KEPT -- it is the convolution's memory) and the
    // rank-`d_rel` relative projection. Every arm computes these.
    let projections = q + 4 * kv + rel_proj;
    match kind {
        AttnKind::Local => {
            let eff = t.rel_span(kind).min(tokens) as u64;
            let rel = n * heads as u64 * eff * f32b;
            // The relative table, the dimension-major copy of K the band's
            // score phase reads, the band's own output and what `w_o` makes
            // of it.
            projections + rel + kv + q + hidden
        }
        AttnKind::Global => {
            // Q, K^T and V made contiguous once ahead of the loop; the blocks'
            // outputs, which together are one whole sequence; their
            // concatenation; and `w_o`'s output.
            projections + 3 * q + 2 * q + hidden + prefill_peak_bytes(heads, tokens)
        }
    }
}

/// What ONE MLP holds in activations at this sequence length.
///
/// The MoE arm is the peak of the whole model and it is not close. The routed
/// lane stacks every token's `num_experts_per_tok` rows into ONE buffer per
/// stage ([`super::moegroup`] -- that is the whole point of it, one launch per
/// stage per layer instead of one per expert), so every stage is `top_k` times
/// as tall as the sequence: six times, on this model, against the one time an
/// attention buffer is. The gather, both GEMM outputs and the gated activation
/// are all `let`-bound in the same scope and all live at the second GEMM.
///
/// The shared experts run afterwards on the same normed input, while only the
/// scattered sum is still live, and their widest buffer is
/// `2 * n_shared * intermediate_size` against the routed lane's
/// `top_k * 2 * intermediate_size` -- so the routed lane is the peak and this
/// does not count them twice.
pub fn mlp_activation_bytes(t: &InklingTextConfig, layer: usize, tokens: usize) -> u64 {
    let n = tokens as u64;
    let f32b = core::mem::size_of::<f32>() as u64;
    let hidden = t.hidden_size as u64;
    if t.is_dense(layer) {
        let i = t.dense_intermediate_size as u64;
        // The gate and the up projection, the activation and the gated
        // product, then the down projection's output and its scaling.
        n * (4 * i + 2 * hidden) * f32b
    } else {
        let i = t.intermediate_size as u64;
        let m = n * t.num_experts_per_tok as u64;
        let gathered = m * hidden * f32b;
        // Gate and up arrive in one buffer, so `2 * i` wide.
        let w13 = m * 2 * i * f32b;
        let act = m * i * f32b;
        let down = m * hidden * f32b;
        // NVFP4 is a nibble a value plus one E4M3 scale per group of sixteen,
        // and both GEMM inputs are quantised.
        let packed = (m * hidden + m * i) / 2 + (m * hidden + m * i) / 16;
        let scattered = n * hidden * f32b;
        gathered + w13 + act + down + packed + scattered
    }
}

/// What a prefill of `layers` holds in activations at this sequence length.
///
/// Three shapes, and the gate had none of them:
///
/// * what every layer reads and no layer frees -- the residual stream and the
///   normalized copy beside it. Linear in the sequence, flat in the range.
/// * what each layer KEEPS: its K and V, for the whole stack. A global layer
///   keeps the sequence and a local one keeps its window, so this is the one
///   term that grows with the range as well as the sequence.
/// * what one layer holds while it runs. Layers run one at a time and each
///   frees its own working set before the next allocates, so what binds is the
///   WIDEST layer in the range and not the sum over it.
///
/// It replaces a per-layer CONSTANT. That constant made the estimate flat in
/// the sequence -- 13.84 GiB at 16,384 tokens, 13.34 at 81,920 and 13.52 at
/// 100,623, a spread of half a GiB across a sixfold range -- because the only
/// sequence-dependent term it had was [`prefill_peak_bytes`], which stops
/// growing as soon as [`query_block`] starts shrinking. Everything that
/// actually scales was missing, and being wrong in the PERMISSIVE direction is
/// how a gate reports 68 GiB of headroom for a run that then swaps.
pub fn prefill_activation_bytes(
    t: &InklingTextConfig,
    layers: core::ops::Range<usize>,
    tokens: usize,
) -> u64 {
    let n = tokens as u64;
    let f32b = core::mem::size_of::<f32>() as u64;
    let carried = 2 * n * t.hidden_size as u64 * f32b;
    let elem = cache_elem_bytes();
    let caches: u64 = layers
        .clone()
        .map(|l| {
            let kind = t.attn_kind(l);
            let (_, kv_heads, head_dim) = t.heads(kind);
            let keep = match kind {
                AttnKind::Local => t.sliding_window_size.min(tokens),
                AttnKind::Global => tokens,
            } as u64;
            2 * keep * (kv_heads * head_dim) as u64 * elem
        })
        .sum();
    let widest = layers
        .map(|l| {
            attention_activation_bytes(t, t.attn_kind(l), tokens)
                .max(mlp_activation_bytes(t, l, tokens))
        })
        .max()
        .unwrap_or(0);
    carried + caches + widest
}

/// The largest SINGLE buffer this range asks for at this length.
///
/// A max and never a sum: these are separate allocations and the cap is
/// per-buffer. The sum is what belongs against node memory, which is
/// [`prefill_activation_bytes`]'s question and not this one.
///
/// # It is not an attention buffer any more
///
/// This used to read only the score block and the `[heads, tokens, head_dim]`
/// activation, which made the ceiling `cap / 16 KiB` -- near two million
/// tokens, a number no run will ever reach. Two buffers are far larger and
/// neither is in attention:
///
/// * the routed-expert lane stacks every token's `num_experts_per_tok` rows
///   into ONE buffer per stage, so its gather and both its GEMM outputs are
///   `top_k * hidden` f32 a token: 96 KiB here, six times the activation this
///   used to call the largest;
/// * a local layer builds its relative-position table for the whole sequence,
///   `[tokens, heads, sliding_window_size]` f32, 64 KiB a token.
///
/// So the honest ceiling is `cap / 96 KiB`, about 326,000 tokens on this
/// device, and the two-million figure was the same permissive error the
/// admission gate had: a term left out because it was not in the file this
/// module was written about.
pub fn largest_buffer(
    t: &InklingTextConfig,
    layers: core::ops::Range<usize>,
    tokens: usize,
) -> u64 {
    let n = tokens as u64;
    let f32b = core::mem::size_of::<f32>() as u64;
    layers
        .map(|l| {
            let kind = t.attn_kind(l);
            let (heads, _, head_dim) = t.heads(kind);
            let act = activation_bytes(heads, head_dim, tokens);
            let attn = match kind {
                AttnKind::Local => {
                    let eff = t.rel_span(kind).min(tokens) as u64;
                    act.max(n * heads as u64 * eff * f32b)
                }
                AttnKind::Global => {
                    act.max(score_block_bytes(heads, query_block(heads, tokens), tokens))
                }
            };
            let mlp = if t.is_dense(l) {
                // The gate and the up projection are separate linears.
                n * t.dense_intermediate_size as u64 * f32b
            } else {
                let m = n * t.num_experts_per_tok as u64;
                (m * t.hidden_size as u64 * f32b)
                    .max(m * 2 * t.intermediate_size as u64 * f32b)
            };
            attn.max(mlp)
        })
        .max()
        .unwrap_or(0)
}

/// The longest sequence whose largest single buffer fits under `cap`.
///
/// Bisected rather than solved, because the terms round and because
/// [`query_block`] is a step function. Every term is non-decreasing in
/// `tokens`, so their max is too and the bisection is exact.
pub fn longest_sequence(
    t: &InklingTextConfig,
    layers: core::ops::Range<usize>,
    cap: u64,
) -> usize {
    let (mut lo, mut hi) = (0usize, 1usize << 24);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if largest_buffer(t, layers.clone(), mid) <= cap {
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
    t: &InklingTextConfig,
    layers: core::ops::Range<usize>,
    tokens: usize,
) -> Result<()> {
    let (first, last) = (layers.start, layers.end);
    let heads = t.heads(AttnKind::Global).0.max(t.heads(AttnKind::Local).0);
    let rows = query_block(heads, tokens);
    let scores = score_block_bytes(heads, rows, tokens);
    let want = largest_buffer(t, layers.clone(), tokens);
    let cap = largest_allocation(client);
    anyhow::ensure!(
        want <= cap,
        "{tokens} tokens over layers {}..{} needs a single {:.2} GiB buffer -- the widest of a \
         [{heads}, {rows}, {tokens}] f32 score block ({:.2} GiB), a local layer's \
         [{tokens}, {heads}, {}] f32 relative table and the routed lane's \
         [{tokens} x {}, {}] f32 stack -- and this device refuses any single allocation over \
         {cap} bytes ({:.2} GiB).\n  \
         That cap is cuDeviceTotalMem / 4 and free memory does not raise it. The longest \
         sequence whose buffers fit is {} tokens.\n  \
         Refusing here rather than at the allocator, because the allocator's refusal happens on \
         a worker thread, does not end the process, and returns a plausible answer read out of a \
         buffer nothing ever wrote.",
        first,
        last,
        want as f64 / GIB,
        scores as f64 / GIB,
        t.sliding_window_size,
        t.num_experts_per_tok,
        t.hidden_size,
        cap as f64 / GIB,
        longest_sequence(t, layers, cap),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        activation_bytes, attention_activation_bytes, largest_buffer, longest_sequence,
        mlp_activation_bytes, prefill_activation_bytes, prefill_peak_bytes, query_block,
        score_block_bytes, score_matrix_bytes, QUERY_BLOCK_BYTES,
    };
    use crate::models::inkling::config::{AttnKind, InklingTextConfig};

    /// The 42-layer release, from its own `config.json`.
    ///
    /// Parsed rather than built field by field, so a field this module reads
    /// that the checkpoint renames fails HERE instead of silently reverting to
    /// a default and making the charge small again.
    fn small() -> InklingTextConfig {
        serde_json::from_str(
            r#"{
              "hidden_size": 4096, "num_hidden_layers": 42,
              "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
              "swa_num_attention_heads": 32, "swa_num_key_value_heads": 8,
              "swa_head_dim": 128,
              "vocab_size": 201024, "unpadded_vocab_size": 200058,
              "d_rel": 16, "rel_extent": 1024, "rms_norm_eps": 1e-6,
              "sconv_kernel_size": 4, "use_sconv": true,
              "sliding_window_size": 512,
              "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22,
                                  24,25,26,27,28,30,31,32,33,34,36,37,38,39,40],
              "dense_mlp_idx": 2, "dense_intermediate_size": 16384,
              "intermediate_size": 2048,
              "n_routed_experts": 256, "num_experts_per_tok": 6, "n_shared_experts": 2,
              "shared_expert_sink": true, "route_scale": 2.5,
              "gate_activation": "sigmoid"
            }"#,
        )
        .expect("the fixture is this model config.json")
    }

    /// This model, at the sizes the measurements were taken at.
    const HEADS: usize = 32;
    const HEAD_DIM: usize = 128;
    /// A 119.6 GiB node: cuDeviceTotalMem / 4 is a shade under 30 GiB.
    const CAP: u64 = 128_408_297_472u64 / 4;

    /// The two sizes the old ceiling was measured at, and the padding between.
    /// The bug this module was carrying: the charge did not move with the
    /// sequence.
    ///
    /// The old term was [`prefill_peak_bytes`] alone, and it is bounded by
    /// construction -- [`query_block`] shrinks as the sequence grows, so their
    /// product stops growing. Measured from the gate's own report at
    /// `INK_LAYERS=0:8`: 13.84 GiB at 16,384 tokens, 13.34 at 81,920 and 13.52
    /// at 100,623. A sixfold sequence moved the estimate by 2%, DOWNWARD, while
    /// the run's own working set went past 110 GiB of a 119.63 GiB node.
    #[test]
    fn the_old_term_was_flat_and_the_new_one_is_not() {
        let t = small();
        // The counterfactual, so the regression names what it defends against
        // rather than trusting a comment about it.
        let (a, b) = (prefill_peak_bytes(32, 16_384), prefill_peak_bytes(32, 100_623));
        let spread = (a.max(b) - a.min(b)) as f64 / a as f64;
        assert!(spread < 0.05, "the old term moved {spread:.3} across the ladder");

        let new = |n| prefill_activation_bytes(&t, 0..8, n);
        let ratio = new(100_623) as f64 / new(16_384) as f64;
        // Not 6.14x, and the shortfall is not sloppiness: at 16,384 tokens the
        // widest layer is still the GLOBAL one, whose two score blocks are 8
        // GiB and do not grow, and by 100,623 it is the MoE lane, which is all
        // sequence. A max over two terms with different shapes is sublinear
        // exactly where they cross. What matters is that it moves at all.
        assert!(
            (3.0..4.5).contains(&ratio),
            "16,384 -> 100,623 tokens multiplied the charge by {ratio:.2}, want ~3.7"
        );
        // Past the crossing it IS linear, which is the half of the range a long
        // input lives in.
        let slope = |a: usize, b: usize| {
            (new(b) - new(a)) as f64 / (b - a) as f64
        };
        let far = slope(65_536, 100_623);
        assert!(
            (380_000.0..420_000.0).contains(&far),
            "the long-sequence slope is {far:.0} bytes a token, want the MoE lane's ~400k"
        );
        assert!(new(81_920) > new(65_536) && new(65_536) > new(32_768));
    }

    /// The routed-expert lane is the peak, and by a wide margin.
    ///
    /// It is the reason the charge is the size it is: every stage of it is
    /// `num_experts_per_tok` rows per token rather than one, so six sequences'
    /// worth of rows pass through buffers as wide as the residual stream.
    /// Anything that made attention the peak would mean this had been mis-read.
    #[test]
    fn the_moe_lane_is_the_widest_layer() {
        let t = small();
        let n = 81_920;
        let moe = mlp_activation_bytes(&t, 5, n);
        let dense = mlp_activation_bytes(&t, 0, n);
        let local = attention_activation_bytes(&t, AttnKind::Local, n);
        let global = attention_activation_bytes(&t, AttnKind::Global, n);
        assert!(moe > dense, "MoE {moe} vs dense MLP {dense}");
        assert!(moe > global, "MoE {moe} vs global attention {global}");
        assert!(moe > local, "MoE {moe} vs local attention {local}");
        // Six rows a token through a 4096-wide gather is the unit of it, and
        // the lane holds the gather, both GEMM outputs, the gated activation
        // and two quantised copies at once -- close to four of them.
        let gather = n as u64 * 6 * 4096 * 4;
        assert!(moe > 3 * gather, "the MoE charge {moe} is under three gathers");
        assert!(moe < 5 * gather, "the MoE charge {moe} is over five gathers");
    }

    /// A local layer's relative table is built for the WHOLE sequence.
    ///
    /// `[tokens, heads, sliding_window_size]` f32 is 64 KiB a token here --
    /// four times Q -- and it is what makes a local layer as expensive as a
    /// global one despite its never materialising a square.
    #[test]
    fn a_local_layer_pays_for_its_whole_relative_table() {
        let t = small();
        let n = 65_536;
        let local = attention_activation_bytes(&t, AttnKind::Local, n);
        let rel = n as u64 * 32 * 512 * 4;
        assert!(local > rel, "a local layer charges {local}, under its {rel}-byte table");
        // A sequence shorter than the window cannot reach past it, so the table
        // is `tokens` wide and not `sliding_window_size` wide -- which shows up
        // as a lower charge PER TOKEN, the total being linear either way.
        let rate = |n: usize| attention_activation_bytes(&t, AttnKind::Local, n) as f64 / n as f64;
        assert!(
            rate(128) < rate(n) * 0.7,
            "128 tokens cost {:.0} a token against {:.0} at {n}; the table did not shrink",
            rate(128),
            rate(n)
        );
    }

    /// A longer range costs more, but not by a layer's width: layers run one at
    /// a time and each frees its working set before the next allocates.
    #[test]
    fn the_range_moves_the_caches_and_not_the_working_set() {
        let t = small();
        let n = 32_768;
        let eight = prefill_activation_bytes(&t, 0..8, n);
        let twenty = prefill_activation_bytes(&t, 0..20, n);
        assert!(twenty > eight, "a longer range charged less");
        // Twelve more layers, two of them global, so what grows is the caches
        // and nothing else -- far under twelve working sets.
        let widest = mlp_activation_bytes(&t, 5, n);
        assert!(
            twenty - eight < widest,
            "twelve more layers added {} bytes, more than one whole layer's working set",
            twenty - eight
        );
    }

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
        let t = small();
        for n in [20_000, 35_845, 100_623] {
            let scores = score_block_bytes(HEADS, query_block(HEADS, n), n);
            let acts = activation_bytes(HEADS, HEAD_DIM, n);
            assert!(
                largest_buffer(&t, 0..8, n) <= CAP,
                "{n} tokens would still be refused: scores {scores}, activations {acts}"
            );
        }
        // 15,808 was the ceiling when the whole square was materialised.
        assert!(largest_buffer(&t, 0..8, 15_809) <= CAP);
    }

    /// The per-buffer ceiling is the MoE lane's, not attention's.
    ///
    /// It read `cap / 16 KiB` -- near two million tokens -- while the routed
    /// lane was asking for `top_k * hidden` f32 a token, six times that. The
    /// honest ceiling is six times lower, and it is a number a run can actually
    /// reach.
    #[test]
    fn bisects_the_boundary() {
        let t = small();
        let n = longest_sequence(&t, 0..8, CAP);
        assert!(largest_buffer(&t, 0..8, n) <= CAP);
        assert!(largest_buffer(&t, 0..8, n + 1) > CAP);
        assert!((300_000..350_000).contains(&n), "boundary landed at {n}");
        // A range with no MoE layer in it is bound by something else and its
        // ceiling is genuinely higher, so this is a property of the RANGE.
        let dense_only = longest_sequence(&t, 0..2, CAP);
        assert!(dense_only > n, "a dense-only range bisected to {dense_only}, under {n}");
    }
}
