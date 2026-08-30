//! The QK-norm as ONE kernel per operand, with `head_dim` a compile-time
//! constant.
//!
//! # What the Burn lane costs
//!
//! [`super::burn::rms_norm`] is six Burn ops -- `powf_scalar(2)`, `mean_dim`,
//! `add_scalar(eps)`, `sqrt`, the broadcast divide, the gain multiply -- and
//! [`super::seam::Bk`] is the UNFUSED `CubeBackend`, so each of the six is a
//! launch and each launch binds its operands as `Tensor`s, i.e. carries a
//! shape/stride upload. A decode step runs the norm twice a layer, on `q` and
//! on the convolved `k`: twelve launches a layer, 252 on a 21-layer node-step,
//! to normalise 40 rows of 128 floats. The `mean_dim` in the middle is
//! [`cubek`]'s rank-generic reduce, which is built for a `[.., 4096]` residual
//! stream and not for 128 elements a row.
//!
//! This module is one launch per operand -- two a layer, 42 a node-step --
//! with the operands bound as `Array<T>` (no shape upload; the previous
//! change made that the rule for every inkling kernel), the reduction width a
//! `#[comptime]` parameter, and both passes over the row `#[unroll]`ed. One
//! thread per `[head_dim]` row, the row held in registers between the two
//! passes, so each element is read from memory once.
//!
//! # Numerics: where this can differ from the Burn lane, and where it cannot
//!
//! Everything after the sum is the SAME sequence of f32 operations in the
//! same order -- `sum / width`, `+ eps`, `sqrt`, `x / denom`, `* gain` --
//! compiled by the same cubecl codegen with the same (default, non-fast-math)
//! instruction modes, so given the same sum the two lanes round identically.
//! In particular it DIVIDES by `sqrt(mean + eps)` rather than multiplying by
//! a reciprocal, for the reason [`super::burn::rms_norm`] gives. Two things
//! reach the sum differently:
//!
//! * **The square.** Burn's `powf_scalar(2.0)` lowers to `Vector::powf`, which
//!   on the CUDA dialect is libm `powf(x, 2.0f)`; this kernel computes `x * x`,
//!   one IEEE multiply. `powf` is not guaranteed to equal `x * x` bit for bit,
//!   so the squared terms can differ in their last f32 bit.
//! * **The order.** `mean_dim` is cubek-reduce's unit routine, which walks the
//!   row as vectorised lines -- several interleaved partial sums folded at the
//!   end; this kernel accumulates the 128 squares sequentially. Same f32
//!   accumulation (cubek's `Mean` accumulates in the INPUT dtype, f32 here;
//!   nothing is widened or narrowed), different association, so the sum can
//!   differ by a few f32 ulp.
//!
//! Both are at the 1e-7 relative level on a 128-term sum and both vanish under
//! the bf16 rounding every consumer of this output performs: `k` goes into the
//! BF16/NVFP4 KV cache and `q` becomes a BF16 GEMM operand. The test below
//! asserts exactly that bar and says why.
//!
//! **There is no bf16 rounding inside either lane at the call site.** `q`
//! comes out of [`super::burn::project_qkvr`] as f32 (`tensor_of` builds the
//! GEMM's output at `DType::F32`) and `k_new` out of the short convolutions as
//! f32, so the f32 -> f32 arm is the one that runs. The kernel is generic over
//! its element types anyway, like [`super::resid::rms_norm_kernel`], and a
//! BF16 operand would be widened on the read and rounded ONCE on the store --
//! which is NOT what Burn's `rms_norm` on a BF16 tensor does (it would square
//! and round per element in BF16 and accumulate the mean in BF16), so if the
//! call site ever narrows, the two lanes stop being comparable there and this
//! one is the better of the two. That is a difference to know about, not one
//! that exists today.
//!
//! # Why not [`super::resid::rms_norm_kernel`]
//!
//! It is the same arithmetic, and it was the first thing tried. But it is
//! built for the residual stream: one CUBE per row, 256 units in a
//! shared-memory tree, the width a RUNTIME argument. At `head_dim = 128` that
//! is a cube of 128 units, a tree reduction and seven barriers to sum 128
//! numbers, and a `while i < h` loop the compiler cannot unroll because `h`
//! arrives at launch. Making `h` comptime there would change the residual
//! kernel's specialisation for a caller that does not need it; the design rule
//! (dimensions at compile time, `#[unroll]` inner loops) wanted a kernel whose
//! shape IS the head, so this is that kernel and the residual one is left as
//! the residual one.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::seam::{Bk, client_of, dtype_of, handle_of_any, tensor_of_dt};
use burn::tensor::{DType, Tensor};

/// Threads per cube. One thread per `[head_dim]` row; a decode step's `q` is
/// `heads` rows and its `k` is `kv_heads`, so a step is one cube per operand.
const HEAD_UNITS: u32 = 128;

/// Whether the QK-norm runs through [`head_rms_norm`] rather than Burn.
///
/// `INK_HEAD_RMS_NATIVE=1` turns it on. **Off by default** until it has been
/// measured on the frontier row: the output is not bit-identical to the Burn
/// lane (see the module doc for the two f32-ulp-level reasons), and the rule
/// in `inkling_forward` is that a lane whose output differs stays off until
/// someone has measured what it changes.
pub fn head_rms_native() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_HEAD_RMS_NATIVE")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// One thread per row of `[rows, hd]`. `x` is `[rows * hd]`, `gain` is `[hd]`,
/// `out` is `[rows * hd]`.
///
/// The row is widened to f32 on the read and held in registers across both
/// passes, so the value that is squared is the value that is normalised and
/// each element is loaded once. `hd` is comptime: both loops unroll to `hd`
/// straight-line steps, `v` is a register array, and the kernel is
/// specialised per head width.
#[cube(launch_unchecked)]
fn head_rms_norm_kernel<I: Scalar + Cast, O: Scalar + Cast>(
    x: &Array<I>,
    gain: &Array<f32>,
    out: &mut Array<O>,
    eps: f32,
    rows: usize,
    #[comptime] hd: usize,
) {
    let r = ABSOLUTE_POS as usize;
    if r < rows {
        let base = r * hd;
        let mut v = Array::<f32>::new(hd);
        let mut acc = f32::new(0.0_f32);
        #[unroll]
        for i in 0..hd {
            let e = f32::cast_from(x[base + i]);
            v[i] = e;
            acc += e * e;
        }
        // `mean` then `sqrt(mean + eps)` then a DIVIDE, in that order: the
        // order the Burn lane performs them in, and the one place a reciprocal
        // would cost bits.
        let width = f32::new(comptime!(hd as f32));
        let denom = Sqrt::sqrt(acc / width + eps);
        #[unroll]
        for i in 0..hd {
            out[base + i] = O::cast_from(v[i] / denom * gain[i]);
        }
    }
}

/// Launch [`head_rms_norm_kernel`] over `rows` rows of `hd`.
fn head_rms_norm_as<I: Scalar + Cast, O: Scalar + Cast, R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    gain: &Handle,
    rows: usize,
    hd: usize,
    eps: f32,
) -> Handle {
    let total = rows * hd;
    let out = client.empty(total * core::mem::size_of::<O>());
    let cubes = rows.div_ceil(HEAD_UNITS as usize) as u32;
    unsafe {
        head_rms_norm_kernel::launch_unchecked::<I, O, R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(HEAD_UNITS),
            ArrayArg::from_raw_parts(x.clone(), total),
            ArrayArg::from_raw_parts(gain.clone(), hd),
            ArrayArg::from_raw_parts(out.clone(), total),
            eps,
            rows,
            hd,
        );
    }
    out
}

/// RMS-normalise each head slice of `[tokens, heads * head_dim]`, one launch.
///
/// The native twin of the Burn expression in [`super::burn`]'s `head_rms_norm`,
/// which is what dispatches here when [`head_rms_native`] is on. No reshape on
/// either side: a contiguous `[tokens, heads * head_dim]` buffer IS the
/// `[tokens * heads, head_dim]` buffer, and the kernel indexes it as the
/// latter. (`handle_of_any` makes a sliced `q` contiguous first -- the same
/// copy Burn's `reshape` of that slice performs on the other lane.)
///
/// `gain` is `[head_dim]` f32, as the checkpoint's `q_norm` / `k_norm` are held
/// in `AttnWeightsDev`; `eps` is the config's `rms_norm_eps`, narrowed to f32
/// exactly as Burn's `add_scalar` narrows it.
pub fn head_rms_norm(
    v: Tensor<Bk, 2>,
    gain: Tensor<Bk, 1>,
    heads: usize,
    head_dim: usize,
    eps: f64,
) -> Tensor<Bk, 2> {
    let [tokens, width] = v.dims();
    assert_eq!(
        width,
        heads * head_dim,
        "{width} is not {heads} x {head_dim}"
    );
    assert_eq!(
        gain.dims()[0],
        head_dim,
        "head_rms_norm: gain is {} wide, the head is {head_dim}",
        gain.dims()[0]
    );
    let dt = dtype_of(&v);
    let client = client_of(&v);
    let dev = v.device();
    let (xh, _) = handle_of_any(v);
    let (gh, gdt) = handle_of_any(gain);
    assert_eq!(gdt, DType::F32, "head_rms_norm: the gain is f32");
    let rows = tokens * heads;
    let out = match dt {
        DType::F32 => {
            head_rms_norm_as::<f32, f32, _>(&client, &xh, &gh, rows, head_dim, eps as f32)
        }
        DType::BF16 => head_rms_norm_as::<half::bf16, half::bf16, _>(
            &client, &xh, &gh, rows, head_dim, eps as f32,
        ),
        _ => panic!("head_rms_norm: no lane for a {dt:?} operand"),
    };
    tensor_of_dt(client, dev, out, tokens, width, dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{FloatDType, Tensor, TensorData};

    /// Deterministic pseudo-random values in `[-1, 1)`: an LCG rather than
    /// `rand`, so the input is the same bytes on every machine that runs this.
    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// A `[64, 128]` input whose rows span four decades of magnitude.
    ///
    /// At scale `1e-3` a row's mean square is `1e-6`, the size of `eps`, so a
    /// kernel that dropped, mis-narrowed or mis-placed `eps` fails on those
    /// rows by O(1) instead of hiding under it at unit scale.
    fn input() -> (Vec<f32>, Vec<f32>) {
        let (rows, hd) = (64usize, 128usize);
        let mut xv = lcg(rows * hd, 1);
        for r in 0..rows {
            let s = [1.0f32, 1e-1, 1e-2, 1e-3][r % 4];
            for e in &mut xv[r * hd..(r + 1) * hd] {
                *e *= s;
            }
        }
        let gv: Vec<f32> = lcg(hd, 2).into_iter().map(|g| 1.0 + 0.5 * g).collect();
        (xv, gv)
    }

    /// The native kernel against the Burn expression it replaces, on a random
    /// `[64, 128]` f32 input -- the dtype the call site holds -- read as
    /// 8 tokens of 8 heads so that the `tokens * heads` row folding is
    /// exercised and not just `heads == rows`.
    ///
    /// **Tolerance: an absolute `2^-9` per element, and why.** A normalised row
    /// has unit RMS by construction, times a gain in `[0.5, 1.5]`, so the
    /// output's own magnitude is ~1 and `2^-9` is HALF A BF16 ULP at that
    /// magnitude -- the rounding every consumer of this output performs (`k`
    /// into the BF16/NVFP4 KV cache, `q` into a BF16 GEMM operand). The two
    /// lanes are expected to differ only by f32 summation order and
    /// `powf(x, 2)` against `x * x` (module doc), which is ~1e-7 relative:
    /// three orders of magnitude under the bar. Anything a BF16 consumer could
    /// see would be a wrong eps, gain, row or width, and those are O(1).
    ///
    /// NOT EXECUTED on the machine this was written on (no CUDA device; `Bk`
    /// is `burn::backend::Cuda`). On a Spark:
    /// `cargo test --release --features inkling-cuda --lib models::inkling::headnorm -- --nocapture`
    #[test]
    fn native_tracks_burn_within_half_a_bf16_ulp() {
        let dev = Default::default();
        let (rows, hd) = (64usize, 128usize);
        let (tokens, heads) = (8usize, 8usize);
        let (xv, gv) = input();
        let eps = 1e-6f64;
        let x: Tensor<Bk, 2> = Tensor::from_data(TensorData::new(xv, [rows, hd]), &dev);
        let gain: Tensor<Bk, 1> = Tensor::from_data(TensorData::new(gv, [hd]), &dev);

        let want = crate::models::inkling::burn::rms_norm(x.clone(), gain.clone(), eps)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let got = head_rms_norm(x.reshape([tokens, heads * hd]), gain, heads, hd, eps)
            .reshape([rows, hd])
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), want.len());

        let bar = 2f32.powi(-9);
        let mut worst = 0f32;
        let mut exact = 0usize;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g.is_finite(), "row {} element {} is {g}", i / hd, i % hd);
            let d = (g - w).abs();
            worst = worst.max(d);
            exact += (g == w) as usize;
            assert!(
                d <= bar,
                "row {} element {}: native {g} vs burn {w}, off by {d} (bar {bar})",
                i / hd,
                i % hd
            );
        }
        eprintln!(
            "head_rms_norm: worst |native - burn| = {worst:.3e} (bar {bar:.3e}), \
             {exact}/{} elements bit-identical",
            got.len()
        );
    }

    /// The BF16 arm against the f32 arm's result: the same kernel at
    /// `I = O = bf16`, which is the arm that runs if the call site ever
    /// narrows. Not against Burn, whose BF16 `rms_norm` accumulates the mean in
    /// BF16 and is the wrong control (module doc).
    ///
    /// Tolerance `2^-7` absolute at unit magnitude: one bf16 ulp from rounding
    /// the INPUT (which moves each element by up to half an ulp and the
    /// denominator by less) plus one from rounding the OUTPUT. Same execution
    /// note as above.
    #[test]
    fn bf16_arm_is_the_f32_arm_rounded() {
        let dev = Default::default();
        let (rows, hd) = (64usize, 128usize);
        let (xv, gv) = input();
        let eps = 1e-6f64;
        let x: Tensor<Bk, 2> = Tensor::from_data(TensorData::new(xv, [rows, hd]), &dev);
        let gain: Tensor<Bk, 1> = Tensor::from_data(TensorData::new(gv, [hd]), &dev);

        let wide = head_rms_norm(x.clone(), gain.clone(), 1, hd, eps)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let narrow = head_rms_norm(x.cast(FloatDType::BF16), gain, 1, hd, eps)
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let bar = 2f32.powi(-7);
        let mut worst = 0f32;
        for (i, (n, w)) in narrow.iter().zip(&wide).enumerate() {
            let d = (n - w).abs();
            worst = worst.max(d);
            assert!(
                d <= bar,
                "row {} element {}: bf16 arm {n} vs f32 arm {w}, off by {d} (bar {bar})",
                i / hd,
                i % hd
            );
        }
        eprintln!("head_rms_norm bf16 arm: worst |bf16 - f32| = {worst:.3e} (bar {bar:.3e})");
    }
}
