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

/// The taps: every product the convolution sums, plus the slid history.
///
/// One thread per channel. The products are **written to memory** rather than
/// summed here, and that is the whole reason this is two kernels and not one.
/// NVRTC compiles with `--fmad=true`, so `acc + w * h` becomes `fma.rn.f32` —
/// one rounding where the lane this replaces had two. Measured end to end that
/// is 1 ULP at layer 0 and 0.6% of the residual stream by layer 19, and a
/// different token. A product that has to exist in memory cannot be folded into
/// the add that consumes it, so `mul.rn.f32` is what the compiler must emit.
///
/// `hist` is `[kernel - 1, dim]` oldest-first, `x` is `[dim]`, `w` is
/// `[dim, kernel]` — all row-major and contiguous. `prod` comes out
/// `[kernel, dim]`, tap-major, and `next` is the `[kernel - 1, dim]` history for
/// the position after this one.
///
/// `next` is written from `hist` and `x` only, never from `prod`, so no output
/// buffer aliases an input.
#[cube(launch_unchecked)]
fn short_conv_taps_kernel(
    hist: &Array<f32>,
    x: &Array<f32>,
    w: &Array<f32>,
    prod: &mut Array<f32>,
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
        // `x`.
        #[unroll]
        for j in 0..taps {
            prod[j * dim + d] = w[d * kernel + j] * hist[j * dim + d];
        }
        prod[taps * dim + d] = w[d * kernel + taps] * xv;

        // The window slides by one: drop the oldest row, append the new one.
        #[unroll]
        for j in 0..taps - 1 {
            next[j * dim + d] = hist[(j + 1) * dim + d];
        }
        next[(taps - 1) * dim + d] = xv;
    }
}

/// The sum, in ascending tap order, plus the convolution's internal residual.
///
/// Nothing here multiplies, so there is nothing for a fused multiply-add to
/// contract with and the additions round exactly where the slice lane's did.
#[cube(launch_unchecked)]
fn short_conv_sum_kernel(
    prod: &Array<f32>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    dim: usize,
    #[comptime] kernel: usize,
) {
    let d = ABSOLUTE_POS as usize;
    if d < dim {
        let mut acc = prod[d];
        #[unroll]
        for j in 1..kernel {
            acc += prod[j * dim + d];
        }
        out[d] = x[d] + acc;
    }
}

/// Launch the pair, returning `(out, next)`.
///
/// `kernel` is a compile-time parameter because it is the trip count of every
/// unrolled loop here and the model ships exactly one value of it (4). `dim` is
/// a runtime argument because it is not: the layer-level convolutions are
/// `hidden` wide and K and V's are `kv_heads * head_dim`, and specializing on
/// each would compile the same four multiply-adds twice.
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
    let prod = client.empty(kernel * dim * f32b);
    let out = client.empty(dim * f32b);
    let next = client.empty(taps * dim * f32b);
    let cubes = dim.div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        short_conv_taps_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(hist.clone(), taps * dim),
            ArrayArg::from_raw_parts(x.clone(), dim),
            ArrayArg::from_raw_parts(w.clone(), dim * kernel),
            ArrayArg::from_raw_parts(prod.clone(), kernel * dim),
            ArrayArg::from_raw_parts(next.clone(), taps * dim),
            dim,
            kernel,
        );
        short_conv_sum_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(prod.clone(), kernel * dim),
            ArrayArg::from_raw_parts(x.clone(), dim),
            ArrayArg::from_raw_parts(out.clone(), dim),
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
/// The tolerance is ZERO, and it has to be. The first version of this module
/// summed the taps in registers in a single kernel and was 1 ULP off; twenty
/// layers of residual stream turned that into 0.6% and a different token by
/// layer 19, measured against a dump of the arm it replaced. So the assertion
/// is bit equality, which is a check that fails for a reason the tolerance
/// version would have passed for.
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

    /// `(max |Δout|, max |Δnext|, max |out|)` between the fused kernel and the
    /// slice lane, over `dim` channels and a `kernel`-tap convolution.
    fn gap(dim: usize, kernel: usize) -> (f32, f32, f32) {
        let dev = Default::default();
        let taps = kernel - 1;
        let hist: Tensor<Bk, 2> =
            Tensor::from_data(TensorData::new(fill(taps * dim, 0.3), [taps, dim]), &dev);
        let x: Tensor<Bk, 2> =
            Tensor::from_data(TensorData::new(fill(dim, 1.7), [1, dim]), &dev);
        let w: Tensor<Bk, 2> =
            Tensor::from_data(TensorData::new(fill(dim * kernel, 2.9), [dim, kernel]), &dev);

        // The lane this replaced, transcribed: concatenate, convolve the whole
        // window, keep the last row, carry the rest.
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

        let d = |a: Tensor<Bk, 2>, b: Tensor<Bk, 2>| -> f32 {
            (a - b).abs().max().into_data().to_vec::<f32>().unwrap()[0]
        };
        let mag = ref_out.clone().abs().max().into_data().to_vec::<f32>().unwrap()[0];
        (d(out, ref_out), d(next, ref_next), mag)
    }

    #[test]
    fn fused_decode_matches_the_slice_lane() {
        for (dim, kernel) in [(4096usize, 4usize), (512, 4), (256, 2), (777, 3)] {
            let (dout, dnext, mag) = gap(dim, kernel);
            println!("dim {dim} kernel {kernel}: |Δout| {dout:e}  |Δnext| {dnext:e}  |out| {mag:e}");
            // The history is a pure copy and has no arithmetic to round.
            assert_eq!(dnext, 0.0, "the carried history must be the window's own rows");
            assert_eq!(
                dout, 0.0,
                "dim {dim} kernel {kernel}: |Δout| {dout:e} on values of size {mag:e} — \
                 this lane is supposed to be the slice lane's arithmetic, not an \
                 approximation of it"
            );
        }
    }
}
