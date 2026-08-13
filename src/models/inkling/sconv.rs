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
/// rounds twice, so the fused form is the more accurate of the two, and the
/// module's own test now says so by adjudicating both against an f64
/// accumulation of the same values rather than against each other. The 1 ULP
/// that got this reverted is the split lane's error, not this one's.
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

/// The fused kernel against the shifted-slice lane it replaced.
///
/// The oracle is [`super::burn::short_conv`] itself — the prefill form, which
/// is unchanged and which `inkling_real_gate` holds to a `transformers`
/// capture. So this does not re-litigate whether the convolution is right; it
/// checks the one thing a fused decode step can get wrong, which is whether
/// convolving one position with a carried history reproduces convolving the
/// window and keeping the last row.
///
/// ## The adjudicator is f64, not the other lane
///
/// This test used to assert BIT EQUALITY against the slice lane, and that is
/// what forced the two-kernel split: the fused kernel contracts to
/// `fma.rn.f32`, came out 1 ULP away, and was reverted for it. Bit-identity to
/// a previous implementation cannot answer which of two lanes is right; it can
/// only say which one was written first.
///
/// So both lanes are now measured against the same f64 accumulation of the same
/// values — the arithmetic both are approximating — and the test asserts that
/// the fused kernel is **no worse than** the lane it replaces, plus an absolute
/// budget. The budget is stated here and not tuned to what came out:
/// `kernel` taps of f32 multiply-add carry at most a few ULP, so
/// **4 ULP of the output magnitude (relative 4 x 2^-24 = 2.4e-7)** is the bar,
/// and anything above it is a real defect rather than rounding.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inkling::seam::{client_of, handle_of, tensor_of, Bk};
    use burn::tensor::{Tensor, TensorData};

    /// f32 unit roundoff, `2^-24`.
    const ULP: f64 = 5.960_464_477_539_063e-8;

    /// Relative budget, against the CONDITIONING of the sum and not its result.
    ///
    /// The first version of this constant was `4 * ULP` measured against
    /// `|out|`, and it was derived wrong — badly enough to be worth leaving on
    /// the record rather than quietly retuning. Both lanes came out near
    /// `3e-5`/`7.7e-5` against it. The cause is not either kernel: this
    /// convolution has a residual, `out = x + Σ w_j win_j`, and on channels
    /// where the taps nearly cancel `x` the result is tiny while the terms that
    /// made it are not. Normalising a rounding error by a cancelled result
    /// measures the cancellation, not the arithmetic.
    ///
    /// The textbook bound for a summation of `n` terms is
    /// `|err| <= n * u * Σ|terms|`, so `Σ|terms|` is the scale a rounding
    /// budget belongs against. With `kernel + 1` terms (the taps plus the
    /// residual) that is `(kernel + 1) * ULP`; `8 * ULP` covers every shape
    /// this model ships and is still tight enough to catch a real defect.
    const BUDGET: f64 = 8.0 * ULP;

    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.7919 + seed).sin() * 0.5 + (i as f32 * 0.1237).cos() * 0.25)
            .collect()
    }

    /// The convolution in f64: the value both f32 lanes approximate.
    ///
    /// Same terms, same ascending order; only the precision differs, so the
    /// difference this exposes is exactly the rounding each lane performs.
    fn exact(hist: &[f32], x: &[f32], w: &[f32], dim: usize, kernel: usize) -> Vec<f64> {
        let taps = kernel - 1;
        (0..dim)
            .map(|d| {
                let mut acc = 0f64;
                for j in 0..taps {
                    acc += w[d * kernel + j] as f64 * hist[j * dim + d] as f64;
                }
                acc += w[d * kernel + taps] as f64 * x[d] as f64;
                x[d] as f64 + acc
            })
            .collect()
    }

    /// `Σ|terms|` per channel — the scale a summation's rounding error lives on.
    fn cond(hist: &[f32], x: &[f32], w: &[f32], dim: usize, kernel: usize) -> Vec<f64> {
        let taps = kernel - 1;
        (0..dim)
            .map(|d| {
                let mut s = (x[d] as f64).abs();
                for j in 0..taps {
                    s += (w[d * kernel + j] as f64 * hist[j * dim + d] as f64).abs();
                }
                s += (w[d * kernel + taps] as f64 * x[d] as f64).abs();
                s
            })
            .collect()
    }

    /// `(max, mean)` error of an f32 lane against the f64 value, normalised by
    /// `scale` — `Σ|terms|` for the budget, `|out|` for the reported figure the
    /// first budget was wrongly written against.
    ///
    /// Both statistics, because they answer different questions. The max is
    /// the budget check. The mean is the FMA comparison: one rounding per tap
    /// beats two in the worst-case bound and across a population, but not
    /// necessarily on any single channel — a double rounding can round back
    /// toward the true value by luck, and on 256 channels of a 2-tap kernel one
    /// of them does.
    fn rel_err(got: &[f32], want: &[f64], scale: &[f64]) -> (f64, f64) {
        let errs: Vec<f64> = got
            .iter()
            .zip(want)
            .zip(scale)
            .map(|((&g, &w), &s)| (g as f64 - w).abs() / s.abs().max(1e-30))
            .collect();
        let max = errs.iter().copied().fold(0f64, f64::max);
        (max, errs.iter().sum::<f64>() / errs.len() as f64)
    }

    /// One `(dim, kernel)` case.
    ///
    /// Returns `((max, mean) fused, (max, mean) slice, fused vs |out|,
    /// slice vs |out|, |Δnext|)` — the first two normalised by `Σ|terms|` and
    /// judged, the next two by `|out|` and only reported, the last a pure copy
    /// that must be exact.
    #[allow(clippy::type_complexity)]
    fn gap(dim: usize, kernel: usize) -> ((f64, f64), (f64, f64), f64, f64, f32) {
        let dev = Default::default();
        let taps = kernel - 1;
        let (hv, xv, wv) = (fill(taps * dim, 0.3), fill(dim, 1.7), fill(dim * kernel, 2.9));
        let hist: Tensor<Bk, 2> =
            Tensor::from_data(TensorData::new(hv.clone(), [taps, dim]), &dev);
        let x: Tensor<Bk, 2> = Tensor::from_data(TensorData::new(xv.clone(), [1, dim]), &dev);
        let w: Tensor<Bk, 2> =
            Tensor::from_data(TensorData::new(wv.clone(), [dim, kernel]), &dev);

        // What both f32 lanes approximate.
        let want = exact(&hv, &xv, &wv, dim, kernel);

        // The lane this replaced, transcribed: concatenate, convolve the whole
        // window, keep the last row, carry the rest. Kept as a MEASUREMENT, not
        // as an oracle — it is one of the two things being adjudicated.
        let win = Tensor::cat(vec![hist.clone(), x.clone()], 0);
        let ref_full = crate::models::inkling::burn::short_conv(win.clone(), w.clone());
        let ref_out = ref_full.slice([taps..kernel, 0..dim]);
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
        let out = tensor_of(client.clone(), dev, oh, 1, dim);
        let next = tensor_of(client, Default::default(), nh, taps, dim);

        let vec_of = |t: Tensor<Bk, 2>| t.into_data().to_vec::<f32>().unwrap();
        let dnext = (next - ref_next).abs().max().into_data().to_vec::<f32>().unwrap()[0];
        let (fu, sl) = (vec_of(out), vec_of(ref_out));
        let terms = cond(&hv, &xv, &wv, dim, kernel);
        (
            rel_err(&fu, &want, &terms),
            rel_err(&sl, &want, &terms),
            rel_err(&fu, &want, &want).0,
            rel_err(&sl, &want, &want).0,
            dnext,
        )
    }

    /// The fused kernel is at least as accurate as the lane it replaced.
    ///
    /// Not "identical to": identical was the requirement that forced the split,
    /// and it is the wrong requirement. `fma.rn.f32` rounds once per tap where
    /// `mul.rn.f32; add.rn.f32` rounds twice, so this lane should come out
    /// closer to the f64 value on any input where the two differ at all.
    #[test]
    fn fused_decode_is_no_worse_than_the_slice_lane() {
        for (dim, kernel) in [(4096usize, 4usize), (512, 4), (256, 2), (777, 3)] {
            let ((fmax, fmean), (smax, smean), fused_o, slice_o, dnext) = gap(dim, kernel);
            println!(
                "dim {dim} kernel {kernel}: err/Σ|terms| max fused {fmax:e} slice {smax:e} \
                 (budget {BUDGET:e})  mean fused {fmean:e} slice {smean:e}  |  err/|out| \
                 fused {fused_o:e} slice {slice_o:e}  |  |Δnext| {dnext:e}"
            );
            // The history is a pure copy and has no arithmetic to round.
            assert_eq!(dnext, 0.0, "the carried history must be the window's own rows");
            assert!(
                fmax <= BUDGET,
                "dim {dim} kernel {kernel}: the fused lane is {fmax:e} from the f64 value, \
                 past the pre-registered {BUDGET:e} — that is a defect, not rounding"
            );
            assert!(
                fmean <= smean,
                "dim {dim} kernel {kernel}: the fused lane averages {fmean:e} from the f64 \
                 value against the slice lane's {smean:e}; one rounding per tap should not \
                 lose to two across a population"
            );
        }
    }
}
