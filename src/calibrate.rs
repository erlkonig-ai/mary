//! Activation-aware NVFP4 packing of one linear layer.
//!
//! Round-to-nearest NVFP4 of every linear in nomic-embed-text-v1.5 costs
//! nineteen points of neighbour recall (80.9% recall@10 against the f32 model
//! over our own prose). Two changes to the rounding, neither of which trains
//! anything, win eleven of them back (92.3%):
//!
//! * **AWQ channel scales.** Scale input channel `j` by `mean|x_j|^alpha`
//!   (geometric mean one over the tensor) before rounding and divide it back
//!   after, so the rounding grid follows the channels the inputs actually
//!   exercise. `alpha` is chosen per tensor from a grid by least output error
//!   `sum |(W - Wq) x|^2` over captured input rows.
//! * **GPTQ error feedback.** Round the columns left to right and push each
//!   column's rounding error onto the columns still to come through the
//!   inverse Hessian `(X^T X / R + damp)^-1` of the inputs, so the layer's
//!   output error is what gets minimised rather than the weight error. Block
//!   scales are fixed per row when a block of sixteen columns is reached, from
//!   the values the feedback has left there. Kept only where it beats nearest
//!   rounding on the same rows.
//!
//! The channel scale cannot live inside the codes (it is per input column, the
//! codes are per element under a per-block scale), so a packed linear is the
//! codes of `W * diag(s)` plus the vector `1 / s`, applied to the input. A
//! consumer that wants a dense f32 matrix multiplies column `j` by
//! `input_scale[j]`, which [`Calibrated::decode`] does and
//! `selection::fold_input_scales` does for a whole keymap.
//!
//! Numbers and framing: wiki 88b76d5f, Compass goal dcf9dbaf. Host code by
//! design: this is weight preprocessing done once per model, not model math.

use anyhow::{Result, ensure};

use crate::nvfp4::{BLOCK, Packed, block_unit, code_of, decode_code, global_scale, pack_nearest};

/// The alpha grid: 0.0 is plain rounding, the rest bend the grid toward the
/// busy channels by that power of their mean magnitude.
pub const AWQ_ALPHAS: [f32; 10] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// What to try.
#[derive(Clone, Copy, Debug)]
pub struct Options<'a> {
    /// Channel-scale exponents to search; include 0.0 for the plain baseline.
    pub alphas: &'a [f32],
    /// Error-feedback rounding after the scale search.
    pub feedback: bool,
}

impl Default for Options<'static> {
    fn default() -> Self {
        Self {
            alphas: &AWQ_ALPHAS,
            feedback: true,
        }
    }
}

/// One packed linear and how it was made.
#[derive(Clone, Debug)]
pub struct Calibrated {
    /// Codes of `W * diag(s)`.
    pub packed: Packed,
    /// `1 / s`, one per input column; all ones when no scale was applied.
    pub input_scale: Vec<f32>,
    /// The chosen channel-scale exponent (0.0: none).
    pub alpha: f32,
    /// Output error over the calibration rows relative to plain rounding.
    pub error_ratio: f64,
    /// Whether error feedback made the final rounding.
    pub feedback_used: bool,
}

impl Calibrated {
    /// Plain nearest rounding, no scales, no feedback.
    pub fn plain(packed: Packed) -> Self {
        let cols = packed.cols;
        Self {
            packed,
            input_scale: vec![1.0; cols],
            alpha: 0.0,
            error_ratio: 1.0,
            feedback_used: false,
        }
    }

    /// The dense f32 matrix a consumer of `W` should use: the decoded codes
    /// with the input scale folded into the columns.
    pub fn decode(&self) -> Vec<f32> {
        let mut w = self.packed.decode();
        let cols = self.packed.cols;
        for row in w.chunks_mut(cols) {
            for (v, s) in row.iter_mut().zip(&self.input_scale) {
                *v *= s;
            }
        }
        w
    }
}

/// Output error of a quantised weight on real inputs: the sum over rows of
/// `|(W - Wq) x|^2`, the quantity AWQ minimises.
pub fn output_error(w: &[f32], wq: &[f32], cols: usize, rows: &[Vec<f32>]) -> f64 {
    let diff: Vec<f32> = w.iter().zip(wq).map(|(a, b)| a - b).collect();
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(32).max(1);
    let chunk = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|part| {
                let diff = &diff;
                scope.spawn(move || {
                    let mut err = 0f64;
                    for x in part {
                        for row in diff.chunks(cols) {
                            let acc: f32 = row.iter().zip(x).map(|(d, xi)| d * xi).sum();
                            err += (acc as f64) * (acc as f64);
                        }
                    }
                    err
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("error worker")).sum()
    })
}

/// Cholesky factor L (lower, row-major n x n) of a symmetric positive definite
/// matrix; panics on a non-positive pivot, which the damping prevents.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                assert!(sum > 0.0, "cholesky: non-positive pivot at {i}: {sum}");
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    l
}

/// Inverse of a symmetric positive definite matrix from its Cholesky factor,
/// one column of the identity per solve, columns spread over threads.
fn spd_inverse(l: &[f64], n: usize) -> Vec<f64> {
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get()).min(32).max(1);
    let chunk = n.div_ceil(threads).max(1);
    let mut inv = vec![0f64; n * n];
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|start| {
                let end = (start + chunk).min(n);
                scope.spawn(move || {
                    let mut cols = Vec::with_capacity((end - start) * n);
                    let mut y = vec![0f64; n];
                    for c in start..end {
                        for i in 0..n {
                            let mut sum = if i == c { 1.0 } else { 0.0 };
                            for k in 0..i {
                                sum -= l[i * n + k] * y[k];
                            }
                            y[i] = sum / l[i * n + i];
                        }
                        let mut x = vec![0f64; n];
                        for i in (0..n).rev() {
                            let mut sum = y[i];
                            for k in i + 1..n {
                                sum -= l[k * n + i] * x[k];
                            }
                            x[i] = sum / l[i * n + i];
                        }
                        cols.extend_from_slice(&x);
                    }
                    (start, cols)
                })
            })
            .collect();
        for h in handles {
            let (start, cols) = h.join().expect("inverse worker");
            for (c, col) in cols.chunks(n).enumerate() {
                for i in 0..n {
                    inv[i * n + start + c] = col[i];
                }
            }
        }
    });
    inv
}

/// Error-feedback (GPTQ) NVFP4 of `w` (`[rows, cols]`) against input rows `x`
/// (each of length `cols`), under one global scale from `w`'s own maximum.
pub fn pack_gptq(w: &[f32], cols: usize, x: &[Vec<f32>]) -> Result<Packed> {
    ensure!(cols > 0 && cols % BLOCK == 0, "cols {cols} is not a positive multiple of {BLOCK}");
    ensure!(w.len() % cols == 0, "{} values do not fill rows of {cols}", w.len());
    ensure!(!x.is_empty(), "error feedback needs input rows");
    ensure!(x.iter().all(|r| r.len() == cols), "an input row is not {cols} wide");
    let rows = w.len() / cols;
    let n = cols;

    // H = X^T X / R, damped by one percent of its mean diagonal.
    let mut h = vec![0f64; n * n];
    for r in x {
        for i in 0..n {
            let xi = r[i] as f64;
            if xi == 0.0 {
                continue;
            }
            let row = &mut h[i * n..(i + 1) * n];
            for (hij, &xj) in row.iter_mut().zip(r.iter()) {
                *hij += xi * xj as f64;
            }
        }
    }
    let scale = 1.0 / x.len() as f64;
    for v in h.iter_mut() {
        *v *= scale;
    }
    let mean_diag = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let damp = 0.01 * mean_diag.max(1e-12);
    for i in 0..n {
        h[i * n + i] += damp;
    }
    let l = cholesky(&h, n);
    let hinv = spd_inverse(&l, n);
    // U = chol(Hinv)^T, upper; the feedback uses its rows.
    let lu = cholesky(&hinv, n);
    let mut u = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            u[j * n + i] = lu[i * n + j];
        }
    }

    let scale2 = global_scale(w);
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get()).min(32).max(1);
    let chunk = rows.div_ceil(threads).max(1);
    let mut codes = vec![0u8; w.len() / 2];
    let mut scales = vec![0u8; w.len() / BLOCK];
    std::thread::scope(|scope| {
        let handles: Vec<_> = w
            .chunks(chunk * cols)
            .enumerate()
            .map(|(ci, part)| {
                let u = &u;
                scope.spawn(move || {
                    let mut work: Vec<f32> = part.to_vec();
                    let nrows = part.len() / cols;
                    let mut codes = vec![0u8; part.len() / 2];
                    let mut scales = vec![0u8; part.len() / BLOCK];
                    let mut units = vec![0f32; nrows];
                    for j in 0..cols {
                        if j % BLOCK == 0 {
                            for r in 0..nrows {
                                let block = &work[r * cols + j..r * cols + j + BLOCK];
                                let (byte, unit) = block_unit(block, scale2);
                                scales[(r * cols + j) / BLOCK] = byte;
                                units[r] = unit;
                            }
                        }
                        let ujj = u[j * n + j];
                        for r in 0..nrows {
                            let e = r * cols + j;
                            let v = work[e];
                            let code = code_of(v, units[r]);
                            codes[e / 2] |= code << (4 * (e % 2));
                            let qv = decode_code(code) * units[r];
                            let err = ((v - qv) as f64 / ujj) as f32;
                            if err != 0.0 {
                                let urow = &u[j * n + j + 1..(j + 1) * n];
                                let wrow = &mut work[e + 1..(r + 1) * cols];
                                for (wk, &ujk) in wrow.iter_mut().zip(urow) {
                                    *wk -= err * ujk as f32;
                                }
                            }
                        }
                    }
                    (ci, codes, scales)
                })
            })
            .collect();
        for h in handles {
            let (ci, c, s) = h.join().expect("gptq worker");
            let c0 = ci * chunk * cols / 2;
            codes[c0..c0 + c.len()].copy_from_slice(&c);
            let s0 = ci * chunk * cols / BLOCK;
            scales[s0..s0 + s.len()].copy_from_slice(&s);
        }
    });
    Ok(Packed {
        rows,
        cols,
        codes,
        scales,
        scale2,
    })
}

/// `mean|x|^alpha`, geometric mean one, or all ones at alpha zero.
fn channel_scales(mean_abs: &[f32], alpha: f32) -> Vec<f32> {
    if alpha == 0.0 {
        return vec![1.0; mean_abs.len()];
    }
    let mut s: Vec<f32> = mean_abs.iter().map(|m| (m + 1e-8).powf(alpha)).collect();
    let log_mean = s.iter().map(|v| v.ln() as f64).sum::<f64>() / s.len() as f64;
    let norm = log_mean.exp() as f32;
    for v in &mut s {
        *v /= norm;
    }
    s
}

fn scale_columns(w: &[f32], cols: usize, s: &[f32]) -> Vec<f32> {
    let mut out = w.to_vec();
    for row in out.chunks_mut(cols) {
        for (v, si) in row.iter_mut().zip(s) {
            *v *= si;
        }
    }
    out
}

/// Calibrated NVFP4 of one linear `w` (`[rows, cols]`): input rows `rows`
/// (each `cols` wide) and per-channel `mean_abs` of the inputs drive the
/// channel-scale search and the error feedback.
pub fn pack_linear(
    w: &[f32],
    cols: usize,
    rows: &[Vec<f32>],
    mean_abs: &[f32],
    opts: &Options<'_>,
) -> Result<Calibrated> {
    ensure!(cols > 0 && cols % BLOCK == 0, "cols {cols} is not a positive multiple of {BLOCK}");
    ensure!(w.len() % cols == 0, "{} values do not fill rows of {cols}", w.len());
    ensure!(!rows.is_empty(), "calibration needs input rows");
    ensure!(rows.iter().all(|r| r.len() == cols), "an input row is not {cols} wide");
    ensure!(mean_abs.len() == cols, "mean_abs has {} entries for {cols} columns", mean_abs.len());

    let plain = pack_nearest(w, cols)?;
    let plain_err = output_error(w, &plain.decode(), cols, rows);
    let mut best = Calibrated::plain(plain);
    let mut best_err = plain_err;

    for &alpha in opts.alphas.iter().filter(|&&a| a != 0.0) {
        let s = channel_scales(mean_abs, alpha);
        let packed = pack_nearest(&scale_columns(w, cols, &s), cols)?;
        let cal = Calibrated {
            packed,
            input_scale: s.iter().map(|v| 1.0 / v).collect(),
            alpha,
            error_ratio: 1.0,
            feedback_used: false,
        };
        let err = output_error(w, &cal.decode(), cols, rows);
        if err < best_err {
            best = cal;
            best_err = err;
        }
    }

    if opts.feedback {
        let s = channel_scales(mean_abs, best.alpha);
        let xs: Vec<Vec<f32>> = rows
            .iter()
            .map(|r| r.iter().zip(&s).map(|(xi, si)| xi / si).collect())
            .collect();
        let packed = pack_gptq(&scale_columns(w, cols, &s), cols, &xs)?;
        let cal = Calibrated {
            packed,
            input_scale: s.iter().map(|v| 1.0 / v).collect(),
            alpha: best.alpha,
            error_ratio: 1.0,
            feedback_used: true,
        };
        let err = output_error(w, &cal.decode(), cols, rows);
        if err < best_err {
            best = cal;
            best_err = err;
        }
    }

    best.error_ratio = if plain_err > 0.0 { best_err / plain_err } else { 1.0 };
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let x = ((i * 7919) % 1000) as f32 / 1000.0 - 0.5;
                // Column j carries a magnitude that grows with j, so channel
                // scales have something to follow.
                x * (1.0 + (i % cols) as f32 / cols as f32)
            })
            .collect()
    }

    fn inputs(n: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|r| {
                (0..cols)
                    .map(|j| {
                        let x = (((r * 131 + j * 17) % 997) as f32 / 997.0) - 0.5;
                        // The last quarter of the channels is busy.
                        if j >= 3 * cols / 4 { x * 8.0 } else { x }
                    })
                    .collect()
            })
            .collect()
    }

    fn mean_abs(rows: &[Vec<f32>], cols: usize) -> Vec<f32> {
        let mut m = vec![0f32; cols];
        for r in rows {
            for (mj, x) in m.iter_mut().zip(r) {
                *mj += x.abs();
            }
        }
        for v in &mut m {
            *v /= rows.len() as f32;
        }
        m
    }

    #[test]
    fn calibration_never_loses_to_plain_rounding() {
        let (rows, cols) = (8, 64);
        let w = matrix(rows, cols);
        let x = inputs(64, cols);
        let m = mean_abs(&x, cols);
        let cal = pack_linear(&w, cols, &x, &m, &Options::default()).expect("calibrates");
        assert!(cal.error_ratio <= 1.0, "ratio {}", cal.error_ratio);
        assert_eq!(cal.input_scale.len(), cols);
        assert_eq!(cal.decode().len(), rows * cols);
    }

    #[test]
    fn feedback_alone_beats_nearest_on_correlated_inputs() {
        let (rows, cols) = (4, 32);
        let w = matrix(rows, cols);
        let x = inputs(48, cols);
        let plain = pack_nearest(&w, cols).expect("packs");
        let fed = pack_gptq(&w, cols, &x).expect("feeds back");
        let e_plain = output_error(&w, &plain.decode(), cols, &x);
        let e_fed = output_error(&w, &fed.decode(), cols, &x);
        assert!(e_fed <= e_plain, "feedback {e_fed} vs plain {e_plain}");
        assert_eq!(fed.scale2, plain.scale2, "the global scale is the tensor's, not the feedback's");
    }

    #[test]
    fn decode_folds_the_input_scale_into_the_columns() {
        let cols = 16;
        let w = vec![1.0f32; cols];
        let packed = pack_nearest(&w, cols).expect("packs");
        let cal = Calibrated {
            packed,
            input_scale: (0..cols).map(|j| j as f32).collect(),
            alpha: 0.5,
            error_ratio: 1.0,
            feedback_used: false,
        };
        let dec = cal.decode();
        for (j, v) in dec.iter().enumerate() {
            assert_eq!(*v, j as f32);
        }
    }
}
