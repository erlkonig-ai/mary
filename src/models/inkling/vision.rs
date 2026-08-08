//! Inkling's vision (HMLP) and audio (dMel) towers — the f32 reference lane.
//!
//! Together they are 132 MiB of a 159 GB checkpoint. Neither is an encoder
//! stack: vision is a hierarchy of MLPs over folded patches, audio is a sum of
//! per-bin codebook lookups. Both embed straight into the text model.
//!
//! The layer widths are **derived**, not tabulated. `plan_out_scales` in the
//! reference computes them from `patch_size`, `temporal_patch_size`,
//! `n_layers` and `n_channels`; hardcoding the released model's
//! `[128, 320, 4800]` happens to be right for a 40px patch and silently wrong
//! for anything else.

use crate::models::inkling::block::rms_norm;

/// Prime factors in ascending order.
fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut f = Vec::new();
    while n % 2 == 0 {
        f.push(2);
        n /= 2;
    }
    let mut p = 3;
    while p * p <= n {
        while n % p == 0 {
            f.push(p);
            n /= p;
        }
        p += 2;
    }
    if n > 1 {
        f.push(n);
    }
    f
}

/// One `(t, h, w, c)` grid in the pyramid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scale {
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

/// The `n_layers + 1` grids the HMLP passes through.
///
/// Mirrors `plan_out_scales`: build the candidate spatial and temporal grids,
/// then pick `n_layers + 1` of them whose log size-reductions sit closest to an
/// even geometric spacing, with the first and last pinned.
///
/// The reference uses `scipy.linear_sum_assignment` for the under-determined
/// case. With at most a handful of candidates the same optimum is reachable by
/// enumerating injective assignments, which avoids carrying a Hungarian solver
/// for a problem this size; `MAX_ENUM` bounds it.
pub fn plan_out_scales(
    temporal_patch_size: usize,
    patch_size: usize,
    n_layers: usize,
    n_channels: usize,
) -> Vec<Scale> {
    const MAX_ENUM: usize = 9;

    let cumprod = |v: Vec<usize>| -> Vec<usize> {
        let mut out = Vec::with_capacity(v.len());
        let mut acc = 1usize;
        for x in v {
            acc *= x;
            out.push(acc);
        }
        out
    };
    let mut hf = prime_factors(patch_size);
    hf.reverse();
    let h = cumprod(hf);
    let mut tf = prime_factors(temporal_patch_size);
    tf.reverse();
    let t = cumprod(tf);

    let round64 = |x: f64| -> usize { (x / 64.0).ceil() as usize * 64 };
    let hlast = *h.last().expect("patch_size must have a factor");

    let mut scales = vec![Scale { t: 1, h: 1, w: 1, c: n_channels }];
    for &hi in &h {
        scales.push(Scale {
            t: 1,
            h: hi,
            w: hi,
            c: round64((hi * hi * n_channels) as f64),
        });
    }
    for &ti in &t {
        scales.push(Scale {
            t: ti,
            h: hlast,
            w: hlast,
            c: (hlast * hlast * n_channels * ti) * 64,
        });
    }

    let size_reduction: Vec<f64> = scales.iter().map(|s| (s.t * s.h * s.w) as f64).collect();
    let total = (patch_size * patch_size * temporal_patch_size * n_channels) as f64;
    let want = n_layers + 1;
    let ideal: Vec<f64> = (0..want)
        .map(|i| total.ln() * i as f64 / (want - 1) as f64)
        .collect();

    let cost = |row: usize, col: usize| (ideal[row] - size_reduction[col].ln()).abs();

    let idxs: Vec<usize> = if n_layers >= scales.len() {
        (0..want)
            .map(|r| {
                (0..scales.len())
                    .min_by(|&a, &b| cost(r, a).partial_cmp(&cost(r, b)).unwrap())
                    .unwrap()
            })
            .collect()
    } else {
        assert!(
            want <= MAX_ENUM && scales.len() <= MAX_ENUM,
            "assignment too large to enumerate: {want} of {}",
            scales.len()
        );
        // Minimum-cost injective assignment, by enumeration.
        let mut best: Option<(f64, Vec<usize>)> = None;
        let mut chosen = vec![0usize; want];
        let mut used = vec![false; scales.len()];
        fn rec(
            r: usize,
            want: usize,
            n: usize,
            used: &mut Vec<bool>,
            chosen: &mut Vec<usize>,
            acc: f64,
            cost: &dyn Fn(usize, usize) -> f64,
            best: &mut Option<(f64, Vec<usize>)>,
        ) {
            if let Some((b, _)) = best {
                if acc >= *b {
                    return;
                }
            }
            if r == want {
                *best = Some((acc, chosen.clone()));
                return;
            }
            for c in 0..n {
                if used[c] {
                    continue;
                }
                used[c] = true;
                chosen[r] = c;
                rec(r + 1, want, n, used, chosen, acc + cost(r, c), cost, best);
                used[c] = false;
            }
        }
        rec(0, want, scales.len(), &mut used, &mut chosen, 0.0, &cost, &mut best);
        best.expect("no assignment found").1
    };

    let mut idxs = idxs;
    idxs[0] = 0;
    let last = idxs.len() - 1;
    idxs[last] = scales.len() - 1;
    idxs.into_iter().map(|i| scales[i]).collect()
}

/// One HMLP stage's shape: how much it folds, and what it projects.
#[derive(Debug, Clone, Copy)]
pub struct VisionStage {
    pub t_fold: usize,
    pub hw_fold: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub add_norm: bool,
}

/// The stages implied by a plan, given the text width the last one targets.
pub fn vision_stages(scales: &[Scale], n_layers: usize, text_hidden: usize) -> Vec<VisionStage> {
    scales
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            let (s, e) = (w[0], w[1]);
            let shuffle = (e.t / s.t) * (e.h / s.h) * (e.w / s.w);
            VisionStage {
                t_fold: e.t / s.t,
                hw_fold: e.h / s.h,
                input_dim: s.c * shuffle,
                output_dim: if i == n_layers - 1 { text_hidden } else { e.c },
                add_norm: i != n_layers - 1,
            }
        })
        .collect()
}

/// `gelu` as torch's default (the exact erf form, not the tanh approximation).
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// Abramowitz & Stegun 7.1.26 is not accurate enough here; use the standard
/// high-precision rational approximation instead.
fn erf(x: f32) -> f32 {
    let z = x as f64;
    let t = 1.0 / (1.0 + 0.5 * z.abs());
    let y = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587
                                        + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    let v = 1.0 - y;
    (if z >= 0.0 { v } else { -v }) as f32
}

/// `(T, H, W, C) -> (T/t, H/hw, W/hw, C * t * hw^2)` for one patch.
///
/// The reference permutes to `(t_new, h_new, w_new, t_fold, hw_fold, hw_fold, C)`
/// before flattening, so the folded axes end up **outside** the channel axis —
/// getting that order wrong permutes the projection's input and is invisible to
/// any shape check.
pub fn fold_timespace_to_depth(
    x: &[f32],
    t: usize,
    hh: usize,
    ww: usize,
    c: usize,
    t_fold: usize,
    hw_fold: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), t * hh * ww * c);
    assert!(t % t_fold == 0 && hh % hw_fold == 0 && ww % hw_fold == 0);
    let (tn, hn, wn) = (t / t_fold, hh / hw_fold, ww / hw_fold);
    let cn = t_fold * hw_fold * hw_fold * c;
    let mut out = vec![0f32; tn * hn * wn * cn];
    for tn_i in 0..tn {
        for hn_i in 0..hn {
            for wn_i in 0..wn {
                let dst = ((tn_i * hn + hn_i) * wn + wn_i) * cn;
                let mut k = 0usize;
                for tf in 0..t_fold {
                    for hf in 0..hw_fold {
                        for wf in 0..hw_fold {
                            let ti = tn_i * t_fold + tf;
                            let hi = hn_i * hw_fold + hf;
                            let wi = wn_i * hw_fold + wf;
                            let src = ((ti * hh + hi) * ww + wi) * c;
                            out[dst + k..dst + k + c].copy_from_slice(&x[src..src + c]);
                            k += c;
                        }
                    }
                }
            }
        }
    }
    out
}

/// `y = x W^T` over `rows` vectors, `W` stored `[out, in]`.
fn linear(x: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    let mut out = vec![0f32; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            out[r * out_dim + o] = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    }
    out
}

/// One HMLP stage: fold, project, and — except on the last — norm then GELU.
pub fn vision_stage(
    x: &[f32],
    stage: &VisionStage,
    proj: &[f32],
    norm: Option<&[f32]>,
    grid: (usize, usize, usize, usize),
    eps: f64,
) -> (Vec<f32>, (usize, usize, usize, usize)) {
    let (t, hh, ww, c) = grid;
    let folded = if stage.t_fold > 1 || stage.hw_fold > 1 {
        fold_timespace_to_depth(x, t, hh, ww, c, stage.t_fold, stage.hw_fold)
    } else {
        x.to_vec()
    };
    let (tn, hn, wn) = (t / stage.t_fold, hh / stage.hw_fold, ww / stage.hw_fold);
    let rows = tn * hn * wn;
    let mut y = linear(&folded, proj, rows, stage.input_dim, stage.output_dim);
    if let Some(g) = norm {
        y = rms_norm(&y, g, eps, rows, stage.output_dim);
        for v in y.iter_mut() {
            *v = gelu(*v);
        }
    }
    (y, (tn, hn, wn, stage.output_dim))
}

/// dMel audio: one codebook per mel bin, looked up and **summed**, then normed.
///
/// `ids` is `[frames, n_bins]`, each entry a level in `0..levels`. Bin `b`'s
/// codebook occupies rows `b * levels .. (b + 1) * levels` of `table`, which is
/// what the reference's `audio_tokens_offsets` encodes.
pub fn audio_embed(
    ids: &[usize],
    table: &[f32],
    norm_gain: &[f32],
    eps: f64,
    frames: usize,
    n_bins: usize,
    levels: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(ids.len(), frames * n_bins);
    assert_eq!(table.len(), n_bins * levels * hidden);
    let mut out = vec![0f32; frames * hidden];
    for f in 0..frames {
        for b in 0..n_bins {
            let lvl = ids[f * n_bins + b];
            assert!(lvl < levels, "mel level {lvl} is past {levels}");
            let row = (b * levels + lvl) * hidden;
            for (o, &v) in out[f * hidden..(f + 1) * hidden]
                .iter_mut()
                .zip(&table[row..row + hidden])
            {
                *o += v;
            }
        }
    }
    rms_norm(&out, norm_gain, eps, frames, hidden)
}
