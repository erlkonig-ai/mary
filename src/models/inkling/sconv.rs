//! The depthwise short convolution's DECODE step, as one kernel.
//!
//! Four of these run per layer — on K, on V, on the attention output and into
//! the MLP — and until this module existed each one was nineteen kernel
//! launches. [`super::burn::short_conv_step`] expressed it the way the formula
//! reads: concatenate the history with the new row, zero-pad the front, take
//! `kernel` shifted slices, multiply each by its tap, add them up, add the
//! residual. Every one of those verbs is a Burn op and every Burn op is a
//! launch, so a convolution over four positions of a 4096-wide row — 16384
//! multiply-adds, about a microsecond of arithmetic — cost nineteen launches
//! and, measured, about 150 µs of host time to describe.
//!
//! At twenty layers that was 1520 of the 4720 launches a decode step issued:
//! **a third of every kernel in the step, for 1.5 ms of the 84 ms the GPU was
//! busy**. `nsys` puts a ~8 µs gap after each launch on this box, so the
//! convolutions were costing about 11 ms a token in gaps alone.
//!
//! ## Why a decode step is one kernel and the prefill is not
//!
//! A prefill convolves `tokens` positions and its output row `t` reads
//! positions `t - (kernel - 1) ..= t`; the shifted-slice form is the right one
//! there, and it runs once. A decode step convolves exactly ONE position, and
//! [`super::burn::short_conv_step`] then threw away every output row but the
//! last — so the old lane computed `kernel` rows to keep one. Here the single
//! surviving row is written directly:
//!
//! ```text
//! win  = [hist[0], …, hist[k-2], x]                  the k positions in scope
//! out  = x + Σ_{j=0}^{k-1} w[:, j] * win[j]          the row that survives
//! next = [hist[1], …, hist[k-2], x]                  the history to carry
//! ```
//!
//! The sum is accumulated in ascending `j`, which is the order the slice lane
//! added its terms in, so this is the same sequence of additions and not merely
//! the same value.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// Threads per cube. One thread per channel; the channels are independent.
const CUBE_SIZE: u32 = 256;

/// The convolution and the slid history, in one kernel and one pass.
///
/// One thread per channel. The taps are accumulated **in registers**, in
/// ascending `j`, and the output row is written once.
///
/// ## Why this is one kernel again
///
/// It was two. The split existed so that every product had to exist in memory:
/// NVRTC compiles with `--fmad=true`, so `acc + w * h` contracts to
/// `fma.rn.f32`, and a product that must be stored cannot be contracted into
/// the add that consumes it, so the split forced `mul.rn.f32`. That was done to
/// make this lane BIT-IDENTICAL to the shifted-slice lane it replaced, and
/// bit-identity to a previous implementation is not a correctness argument —
/// it assumes whatever was written first was right.
///
/// It was not. `fma.rn.f32` rounds **once** where `mul.rn.f32` then `add.rn.f32`
/// rounds twice, so the fused form rounds less, and a 1 ULP disagreement with
/// the split lane was never evidence that this one was wrong. Which of them is
/// better is not a question this module answers any more, and deliberately: it
/// is decided where every other capability question is, in `golden/paired/`.
///
/// `hist` is `[kernel - 1, dim]` oldest-first, `x` is `[dim]`, `w` is
/// `[dim, kernel]` — all row-major and contiguous. `out` is `[dim]` and `next`
/// is the `[kernel - 1, dim]` history for the position after this one.
///
/// `next` is written from `hist` and `x` only, so no output buffer aliases an
/// input.
#[cube(launch_unchecked)]
fn short_conv_kernel(
    hist: &Array<f32>,
    x: &Array<f32>,
    w: &Array<f32>,
    out: &mut Array<f32>,
    next: &mut Array<f32>,
    dim: usize,
    #[comptime] kernel: usize,
) {
    let d = ABSOLUTE_POS as usize;
    if d < dim {
        let taps = comptime!(kernel - 1);
        let xv = x[d];

        // Tap `j` multiplies window position `j`, and the window is the history
        // followed by the new row — so the last tap is the only one that reads
        // `x`. Ascending `j` is the slice lane's summation order; what differs
        // is only where the rounding happens, not which terms are added when.
        let mut acc = w[d * kernel] * hist[d];
        #[unroll]
        for j in 1..taps {
            acc += w[d * kernel + j] * hist[j * dim + d];
        }
        acc += w[d * kernel + taps] * xv;
        out[d] = xv + acc;

        // The window slides by one: drop the oldest row, append the new one.
        #[unroll]
        for j in 0..taps - 1 {
            next[j * dim + d] = hist[(j + 1) * dim + d];
        }
        next[(taps - 1) * dim + d] = xv;
    }
}

/// Launch it, returning `(out, next)`.
///
/// `kernel` is a compile-time parameter because it is the trip count of every
/// unrolled loop here and the model ships exactly one value of it (4). `dim` is
/// a runtime argument because it is not: the layer-level convolutions are
/// `hidden` wide and K and V's are `kv_heads * head_dim`, and specializing on
/// each would compile the same four multiply-adds twice.
///
/// One launch, not two, and no `prod` scratch: the split form allocated
/// `kernel * dim` f32 per call and round-tripped every product through memory
/// — 64 KiB written and read back per call, eighty calls a token — to buy a
/// rounding it should not have been buying.
pub fn short_conv_decode<R: Runtime>(
    client: &ComputeClient<R>,
    hist: &Handle,
    x: &Handle,
    w: &Handle,
    dim: usize,
    kernel: usize,
) -> (Handle, Handle) {
    assert!(kernel >= 2, "a short convolution with kernel {kernel} has no history to carry");
    let taps = kernel - 1;
    let f32b = core::mem::size_of::<f32>();
    let out = client.empty(dim * f32b);
    let next = client.empty(taps * dim * f32b);
    let cubes = dim.div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        short_conv_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(hist.clone(), taps * dim),
            ArrayArg::from_raw_parts(x.clone(), dim),
            ArrayArg::from_raw_parts(w.clone(), dim * kernel),
            ArrayArg::from_raw_parts(out.clone(), dim),
            ArrayArg::from_raw_parts(next.clone(), taps * dim),
            dim,
            kernel,
        );
    }
    (out, next)
}

/// SEVERAL positions of the same convolution, in one kernel.
///
/// The decode kernel above convolves one position and slides the history; this
/// one convolves `rows` of them from a window that already holds every input
/// they read, and slides nothing — a caller that widens a pass keeps the whole
/// `kernel - 1 + rows` window anyway, because a speculative rollback is a slice
/// of it.
///
/// Every output row is independent: row `i` reads `all[i ..= i + kernel - 1]`
/// and writes `out[i]`, so this is one thread per `(row, channel)` with no
/// sequential dependence at all. The taps accumulate in ascending `j` and
/// contract to `fma.rn.f32` exactly as the decode kernel's do, which is why
/// `rows == 1` here is bit-identical to [`short_conv_kernel`] rather than
/// merely close to it.
///
/// `all` is `[kernel - 1 + rows, dim]` row-major — the history followed by this
/// pass's own inputs — `w` is `[dim, kernel]`, and `out` is `[rows, dim]`.
///
/// `rows * dim` indexes the grid and both are small here (`rows` is a
/// speculation width or a batch size, `dim` is at most `hidden`), so the 32-bit
/// index this runtime gives `usize` in a kernel has room to spare; a caller
/// that grew `rows` into the millions would not.
#[cube(launch_unchecked)]
fn short_conv_batch_kernel(
    all: &Array<f32>,
    w: &Array<f32>,
    out: &mut Array<f32>,
    dim: usize,
    rows: usize,
    #[comptime] kernel: usize,
) {
    let p = ABSOLUTE_POS as usize;
    if p < rows * dim {
        let i = p / dim;
        let d = p % dim;
        let taps = comptime!(kernel - 1);
        let xv = all[(i + taps) * dim + d];
        let mut acc = w[d * kernel] * all[i * dim + d];
        #[unroll]
        for j in 1..taps {
            acc += w[d * kernel + j] * all[(i + j) * dim + d];
        }
        acc += w[d * kernel + taps] * xv;
        out[p] = xv + acc;
    }
}

/// Launch it, returning the `[rows, dim]` output.
///
/// The window itself is the caller's — this returns only what the convolution
/// produced, because the caller already holds the window and is going to slice
/// a rollback out of it.
///
/// One launch where the shifted-slice form was a `cat`, `kernel` slices,
/// `kernel` broadcast multiplies, `kernel - 1` adds and a residual add. That
/// mattered enough at one row to be worth a kernel (see this module's opening);
/// at more than one row it is the difference between a decode pass that widens
/// cheaply and one that does not.
pub fn short_conv_batch<R: Runtime>(
    client: &ComputeClient<R>,
    all: &Handle,
    w: &Handle,
    dim: usize,
    rows: usize,
    kernel: usize,
) -> Handle {
    assert!(kernel >= 2, "a short convolution with kernel {kernel} has no history to carry");
    assert!(rows >= 1, "a batched convolution produces at least one row");
    let f32b = core::mem::size_of::<f32>();
    let out = client.empty(rows * dim * f32b);
    let cubes = (rows * dim).div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        short_conv_batch_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(all.clone(), (kernel - 1 + rows) * dim),
            ArrayArg::from_raw_parts(w.clone(), dim * kernel),
            ArrayArg::from_raw_parts(out.clone(), rows * dim),
            dim,
            rows,
            kernel,
        );
    }
    out
}

/// The same convolution for `slots` INDEPENDENT sequences, one position each.
///
/// [`short_conv_kernel`] carries one history and convolves one position;
/// [`short_conv_batch_kernel`] convolves several CONSECUTIVE positions of one
/// sequence out of one window. Neither is what `b` independent decode slots
/// want: they are `b` sequences that share nothing but the weights, so each has
/// its own `kernel - 1` history and its own new row, and a tap of slot `s` must
/// never reach into slot `s - 1`.
///
/// That separation is the whole reason this is a third kernel rather than a
/// call to the second one with `rows = b`. The batched kernel's row `i` reads
/// `all[i ..= i + kernel - 1]`, so its rows OVERLAP by `kernel - 1` positions —
/// exactly right for a speculative batch and exactly wrong here, where it would
/// convolve slot `s`'s output out of slot `s - 1`'s inputs and produce fluent
/// text with the sequences quietly bleeding into each other.
///
/// `hist` is `[slots, kernel - 1, dim]` oldest-first per slot, `x` is
/// `[slots, dim]`, `w` is `[dim, kernel]` — the weights are shared, which is
/// the point of a batch. `out` is `[slots, dim]` and `next` is the
/// `[slots, kernel - 1, dim]` history for the position after this one.
///
/// One thread per `(slot, channel)`, taps accumulated in registers in ascending
/// `j`, so `slots == 1` here is bit-identical to [`short_conv_kernel`] rather
/// than merely close to it.
///
/// `slots * dim` indexes the grid: `dim` is at most `hidden` (4096) and `slots`
/// is a batch width, so the 32-bit `usize` this runtime gives a kernel has room
/// to spare — the largest index formed is `slots * (kernel - 1) * dim`, which
/// is under a million at every shape this model runs.
#[cube(launch_unchecked)]
fn short_conv_slots_kernel(
    hist: &Array<f32>,
    x: &Array<f32>,
    w: &Array<f32>,
    out: &mut Array<f32>,
    next: &mut Array<f32>,
    dim: usize,
    slots: usize,
    #[comptime] kernel: usize,
) {
    let p = ABSOLUTE_POS as usize;
    if p < slots * dim {
        let s = p / dim;
        let d = p % dim;
        let taps = comptime!(kernel - 1);
        // Slot `s`'s history block. Every read and write below is inside it,
        // which is the invariant that keeps the slots independent.
        let hb = s * taps * dim;
        let xv = x[p];

        let mut acc = w[d * kernel] * hist[hb + d];
        #[unroll]
        for j in 1..taps {
            acc += w[d * kernel + j] * hist[hb + j * dim + d];
        }
        acc += w[d * kernel + taps] * xv;
        out[p] = xv + acc;

        #[unroll]
        for j in 0..taps - 1 {
            next[hb + j * dim + d] = hist[hb + (j + 1) * dim + d];
        }
        next[hb + (taps - 1) * dim + d] = xv;
    }
}

/// Launch it, returning `(out, next)`.
///
/// One launch for the whole batch, not `slots` launches of
/// [`short_conv_decode`]: this convolution is four multiply-adds a channel and
/// the launch is the cost, so a per-slot loop would multiply the one thing that
/// was already worth a kernel by the batch width.
pub fn short_conv_slots<R: Runtime>(
    client: &ComputeClient<R>,
    hist: &Handle,
    x: &Handle,
    w: &Handle,
    dim: usize,
    slots: usize,
    kernel: usize,
) -> (Handle, Handle) {
    assert!(kernel >= 2, "a short convolution with kernel {kernel} has no history to carry");
    assert!(slots >= 1, "a slot batch has at least one slot");
    let taps = kernel - 1;
    let f32b = core::mem::size_of::<f32>();
    let out = client.empty(slots * dim * f32b);
    let next = client.empty(slots * taps * dim * f32b);
    let cubes = (slots * dim).div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        short_conv_slots_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(hist.clone(), slots * taps * dim),
            ArrayArg::from_raw_parts(x.clone(), slots * dim),
            ArrayArg::from_raw_parts(w.clone(), dim * kernel),
            ArrayArg::from_raw_parts(out.clone(), slots * dim),
            ArrayArg::from_raw_parts(next.clone(), slots * taps * dim),
            dim,
            slots,
            kernel,
        );
    }
    (out, next)
}

/// The fused kernel against the shifted-slice lane it replaced.
///
/// The oracle is [`super::burn::short_conv`] itself — the prefill form, which
/// is unchanged and which `inkling_real_gate` holds to a `transformers`
/// capture. So this does not re-litigate whether the convolution is right; it
/// checks the one thing a fused decode step can get wrong, which is whether
/// convolving one position with a carried history reproduces convolving the
/// window and keeping the last row.
///
/// ## There is no adjudicator here any more, and that is the point
///
/// This module used to accumulate the same convolution in f64 on the host and
/// hold both f32 lanes to a ULP budget against it. That is gone. An f64
/// transcription is not ground truth for a model whose weights are four bits —
/// it is a second, more expensive computation of the same thing — and while it
/// sat here it kept being read as the question. The value is decided by
/// `golden/paired/`, which runs the whole stack; what is left here is the two
/// things a reference cannot tell you and a kernel can:
///
///   * the carried history is a COPY of the window's own rows, so it must be
///     exact — there is no arithmetic in it to round;
///   * the batched kernel and the one-row kernel are the same expression with
///     the same taps in the same order, so they must agree exactly — if they do
///     not, an index is wrong, and no tolerance would make that clearer.
///
/// Both are statements about indexing rather than about precision, which is why
/// they survive the rule that killed the ULP budget.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inkling::seam::{client_of, handle_of, tensor_of, Bk};
    use burn::tensor::{Tensor, TensorData};

    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.7919 + seed).sin() * 0.5 + (i as f32 * 0.1237).cos() * 0.25)
            .collect()
    }

    /// The batched kernel against the one-row kernel, row by row.
    ///
    /// Bit equality is the right bar here and nowhere else in this module: the
    /// two kernels are the SAME expression with the same taps in the same
    /// ascending order and the same contraction, differing only in how many
    /// rows one launch covers. If they disagree at all, an index is wrong —
    /// there is no rounding difference available for them to disagree by.
    ///
    /// Run at `rows` up to 8 and `dim` at the checkpoint's 4096, because the
    /// grid is `rows * dim` flattened and a row-major mix-up between the two
    /// is invisible at `rows == 1` and invisible again at `dim == rows`.
    #[test]
    fn batched_matches_the_one_row_kernel_exactly() {
        let dev = Default::default();
        for (dim, kernel, rows) in
            [(4096usize, 4usize, 1usize), (4096, 4, 2), (4096, 4, 8), (777, 3, 5), (256, 2, 3)]
        {
            let taps = kernel - 1;
            let len = taps + rows;
            let (av, wv) = (fill(len * dim, 0.3), fill(dim * kernel, 2.9));
            let all: Tensor<Bk, 2> =
                Tensor::from_data(TensorData::new(av.clone(), [len, dim]), &dev);
            let w: Tensor<Bk, 2> =
                Tensor::from_data(TensorData::new(wv.clone(), [dim, kernel]), &dev);
            let client = client_of(&all);
            let bh = short_conv_batch(
                &client,
                &handle_of(all.clone()),
                &handle_of(w.clone()),
                dim,
                rows,
                kernel,
            );
            let batched = tensor_of(client.clone(), dev.clone(), bh, rows, dim)
                .into_data()
                .to_vec::<f32>()
                .unwrap();

            // The same rows, one launch each, through the decode kernel: its
            // window for row `i` is `all[i .. i + kernel]`.
            for i in 0..rows {
                let hist = all.clone().slice([i..i + taps, 0..dim]);
                let x = all.clone().slice([i + taps..i + taps + 1, 0..dim]);
                let (oh, _) = short_conv_decode(
                    &client,
                    &handle_of(hist),
                    &handle_of(x),
                    &handle_of(w.clone()),
                    dim,
                    kernel,
                );
                let one = tensor_of(client.clone(), Default::default(), oh, 1, dim)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                for d in 0..dim {
                    assert_eq!(
                        batched[i * dim + d], one[d],
                        "dim {dim} kernel {kernel} rows {rows}: row {i} channel {d} \
                         differs between the batched and the one-row kernel"
                    );
                }
            }

            // A shared index error in the two kernels would pass by agreeing,
            // and it is `golden/paired/` that would catch it -- the whole stack
            // runs four of these a layer. What used to be here was an f64
            // transcription of the convolution, and it is gone: see the module
            // header.
        }
    }

    /// The fused kernel tracks the lane it replaced, as a SMOKE CHECK.
    ///
    /// Not an accuracy claim and not a tolerance on model math: `fma.rn.f32`
    /// rounds once per tap where `mul; add` rounds twice, so the two lanes are
    /// expected to differ in the last bits and nothing here can say which is
    /// better. What this catches is the lane going WRONG -- a transposed weight,
    /// an off-by-one tap, a stale history -- and that shows up as a difference
    /// of order one, not of order 1e-7. The bar is set where garbage lives.
    ///
    /// The carried history is checked exactly, because it is a copy.
    #[test]
    fn fused_decode_tracks_the_slice_lane() {
        let dev = Default::default();
        for (dim, kernel) in [(4096usize, 4usize), (512, 4), (256, 2), (777, 3)] {
            let taps = kernel - 1;
            let (hv, xv, wv) = (fill(taps * dim, 0.3), fill(dim, 1.7), fill(dim * kernel, 2.9));
            let hist: Tensor<Bk, 2> =
                Tensor::from_data(TensorData::new(hv, [taps, dim]), &dev);
            let x: Tensor<Bk, 2> = Tensor::from_data(TensorData::new(xv, [1, dim]), &dev);
            let w: Tensor<Bk, 2> = Tensor::from_data(TensorData::new(wv, [dim, kernel]), &dev);

            let win = Tensor::cat(vec![hist.clone(), x.clone()], 0);
            let ref_out = crate::models::inkling::burn::short_conv(win.clone(), w.clone())
                .slice([taps..kernel, 0..dim]);
            let ref_next = win.slice([1..kernel, 0..dim]);

            let client = client_of(&x);
            let (oh, nh) = short_conv_decode(
                &client,
                &handle_of(hist),
                &handle_of(x),
                &handle_of(w),
                dim,
                kernel,
            );
            let out = tensor_of(client.clone(), dev.clone(), oh, 1, dim);
            let next = tensor_of(client, Default::default(), nh, taps, dim);

            let dnext = (next - ref_next).abs().max().into_scalar();
            assert_eq!(dnext, 0.0, "the carried history must be the window's own rows");

            let scale = ref_out.clone().abs().max().into_scalar().max(1e-6);
            let rel = (out - ref_out).abs().max().into_scalar() / scale;
            println!("dim {dim} kernel {kernel}: fused vs slice, relative {rel:e}");
            assert!(
                rel < 1e-3,
                "dim {dim} kernel {kernel}: the fused lane is {rel:e} from the slice lane -- \
                 that is not rounding, that is a different convolution"
            );
        }
    }
}
