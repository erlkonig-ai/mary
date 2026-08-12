//! Inkling attention — the f32 reference lane.
//!
//! GQA with four things bolted on, taken from
//! `transformers.models.inkling.modeling_inkling.InklingAttention` and gated
//! against it by `inkling_attn_gate`:
//!
//! * a depthwise short convolution on K and V (not on Q), each carrying its own
//!   residual — see [`crate::models::inkling::block::short_conv`];
//! * QK-norm at `head_dim` width, which is why the softmax scaling is
//!   `1 / head_dim` and *not* `1 / sqrt(head_dim)`;
//! * an additive learned relative-position bias of rank `d_rel` per head,
//!   gathered by backward distance and zero outside `0 <= d < rel_extent`;
//! * log scaling past `log_scaling_n_floor`, on global layers only, applied to
//!   the query *and* to the position bias.
//!
//! A local and a global layer are different functions here, not two settings of
//! one: they differ in head counts (`swa_*` against the global fields), in how
//! far the relative table reaches (`sliding_window_size` against `rel_extent`),
//! in whether a window mask applies, and in whether log scaling runs at all.

use crate::models::inkling::block::short_conv;
use crate::models::inkling::config::AttnKind;

/// Shapes for one attention layer, already resolved for its [`AttnKind`].
#[derive(Debug, Clone, Copy)]
pub struct AttnDims {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    /// How far the relative-position table reaches: the sliding window on a
    /// local layer, `rel_extent` on a global one.
    pub rel_extent: usize,
    pub kernel: usize,
    pub rms_eps: f64,
    pub kind: AttnKind,
}

impl AttnDims {
    /// `1 / head_dim`. Q and K are RMS-normalized per head before the product,
    /// so the usual `1 / sqrt(head_dim)` is not what this model uses.
    pub fn scaling(&self) -> f32 {
        1.0 / self.head_dim as f32
    }

    /// GQA repetition factor.
    pub fn groups(&self) -> usize {
        self.heads / self.kv_heads
    }
}

/// Long-context query scaling: `1 + alpha * ln(max(1, (n + 1) / n_floor))`.
#[derive(Debug, Clone, Copy)]
pub struct LogScaling {
    pub n_floor: f32,
    pub alpha: f32,
}

impl LogScaling {
    /// The multiplier for a query at `pos`. Public because the device lane in
    /// [`crate::models::inkling::burn`] builds the same vector and must not
    /// transcribe the formula a second time.
    pub fn tau(&self, pos: usize) -> f32 {
        let ratio = (pos as f32 + 1.0) / self.n_floor;
        1.0 + self.alpha * ratio.max(1.0).ln()
    }
}

/// Every weight one attention layer needs. `w*` are `[out, in]`, the way
/// `nn.Linear` stores them.
pub struct AttnWeights<'a> {
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wr: &'a [f32],
    pub wo: &'a [f32],
    /// `[kv_heads * head_dim, kernel]`.
    pub k_sconv: &'a [f32],
    pub v_sconv: &'a [f32],
    /// `[head_dim]` each.
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    /// `[d_rel, rel_extent]`.
    pub rel_proj: &'a [f32],
}

/// `y = x W^T` for `W` stored `[out, in]`.
fn linear(x: &[f32], w: &[f32], tokens: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), tokens * in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    let mut out = vec![0f32; tokens * out_dim];
    for t in 0..tokens {
        let xt = &x[t * in_dim..(t + 1) * in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            out[t * out_dim + o] = xt.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    }
    out
}

/// RMS-normalize each head slice of `[tokens, heads * head_dim]` in place.
fn head_rms_norm(v: &mut [f32], gain: &[f32], tokens: usize, heads: usize, head_dim: usize, eps: f64) {
    assert_eq!(gain.len(), head_dim);
    for t in 0..tokens {
        for h in 0..heads {
            let base = t * heads * head_dim + h * head_dim;
            let slice = &mut v[base..base + head_dim];
            let mean_sq = slice.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let scale = (mean_sq + eps as f32).sqrt().recip();
            for (i, x) in slice.iter_mut().enumerate() {
                *x = gain[i] * (*x * scale);
            }
        }
    }
}

/// One attention layer over a whole sequence, no cache.
///
/// `mask` is the additive attention mask, `[tokens * tokens]` — zero where a
/// key is visible and a large negative where it is not. It is a parameter
/// rather than something built here because a local layer's mask carries the
/// sliding window and a global layer's does not, and
/// [`causal_mask`] builds either.
pub fn attention(
    x: &[f32],
    w: &AttnWeights<'_>,
    d: &AttnDims,
    log_scaling: Option<LogScaling>,
    mask: &[f32],
    tokens: usize,
) -> Vec<f32> {
    let q_width = d.heads * d.head_dim;
    let kv_width = d.kv_heads * d.head_dim;
    assert_eq!(x.len(), tokens * d.hidden);
    assert_eq!(mask.len(), tokens * tokens);

    let mut q = linear(x, w.wq, tokens, d.hidden, q_width);
    // K and V pass through their short convolutions; Q does not.
    let k = linear(x, w.wk, tokens, d.hidden, kv_width);
    let mut k = short_conv(&k, w.k_sconv, tokens, kv_width, d.kernel);
    let v = linear(x, w.wv, tokens, d.hidden, kv_width);
    let v = short_conv(&v, w.v_sconv, tokens, kv_width, d.kernel);
    let r = linear(x, w.wr, tokens, d.hidden, d.heads * d.d_rel);

    head_rms_norm(&mut q, w.q_norm, tokens, d.heads, d.head_dim, d.rms_eps);
    head_rms_norm(&mut k, w.k_norm, tokens, d.kv_heads, d.head_dim, d.rms_eps);

    // rel_logits[t][h][e] = sum_c r[t][h][c] * rel_proj[c][e]
    let mut rel = vec![0f32; tokens * d.heads * d.rel_extent];
    for t in 0..tokens {
        for h in 0..d.heads {
            for e in 0..d.rel_extent {
                let mut acc = 0f32;
                for c in 0..d.d_rel {
                    acc += r[t * d.heads * d.d_rel + h * d.d_rel + c]
                        * w.rel_proj[c * d.rel_extent + e];
                }
                rel[(t * d.heads + h) * d.rel_extent + e] = acc;
            }
        }
    }

    // Log scaling multiplies the query AND the position bias, and only on
    // global layers. Applying it to one but not the other is a silent error at
    // short context, where tau is exactly 1 and neither shows.
    let taus: Vec<f32> = (0..tokens)
        .map(|t| match (d.kind, log_scaling) {
            (AttnKind::Global, Some(ls)) => ls.tau(t),
            _ => 1.0,
        })
        .collect();
    if taus.iter().any(|&t| t != 1.0) {
        for t in 0..tokens {
            for i in 0..q_width {
                q[t * q_width + i] *= taus[t];
            }
        }
    }

    let scaling = d.scaling();
    let groups = d.groups();
    let mut out = vec![0f32; tokens * q_width];
    let mut scores = vec![0f32; tokens];

    for h in 0..d.heads {
        let kv_h = h / groups;
        for qi in 0..tokens {
            let qv = &q[qi * q_width + h * d.head_dim..qi * q_width + (h + 1) * d.head_dim];
            for ki in 0..tokens {
                let kv = &k[ki * kv_width + kv_h * d.head_dim
                    ..ki * kv_width + (kv_h + 1) * d.head_dim];
                let dot: f32 = qv.iter().zip(kv).map(|(a, b)| a * b).sum();
                // The bias is zero outside [0, rel_extent); causality lives in
                // the mask, not here.
                let dist = qi as isize - ki as isize;
                let bias = if dist >= 0 && (dist as usize) < d.rel_extent {
                    rel[(qi * d.heads + h) * d.rel_extent + dist as usize] * taus[qi]
                } else {
                    0.0
                };
                scores[ki] = dot * scaling + bias + mask[qi * tokens + ki];
            }

            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                denom += *s;
            }
            for s in scores.iter_mut() {
                *s /= denom;
            }

            for ki in 0..tokens {
                let p = scores[ki];
                if p == 0.0 {
                    continue;
                }
                let vv = &v[ki * kv_width + kv_h * d.head_dim
                    ..ki * kv_width + (kv_h + 1) * d.head_dim];
                let o = &mut out[qi * q_width + h * d.head_dim..qi * q_width + (h + 1) * d.head_dim];
                for (acc, &val) in o.iter_mut().zip(vv) {
                    *acc += p * val;
                }
            }
        }
    }

    linear(&out, w.wo, tokens, q_width, d.hidden)
}

/// Additive causal mask, optionally restricted to a sliding window.
///
/// Zero where a key is visible, `f32::NEG_INFINITY` where it is not. A local
/// layer sees the `window` most recent keys inclusive of itself; a global layer
/// sees everything up to the query.
pub fn causal_mask(tokens: usize, window: Option<usize>) -> Vec<f32> {
    let mut m = vec![f32::NEG_INFINITY; tokens * tokens];
    for q in 0..tokens {
        for k in 0..=q {
            let visible = match window {
                Some(wnd) => q - k < wnd,
                None => true,
            };
            if visible {
                m[q * tokens + k] = 0.0;
            }
        }
    }
    m
}
