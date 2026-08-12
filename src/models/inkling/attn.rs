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

use crate::models::inkling::block::{conv_history, short_conv, short_conv_step};
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
///
/// This is [`attention_prefill`] with the cache dropped, exactly as the device
/// lane's [`crate::models::inkling::burn::attention`] is: there is one
/// transcription of the layer, so the cached and uncached lanes cannot drift.
pub fn attention(
    x: &[f32],
    w: &AttnWeights<'_>,
    d: &AttnDims,
    log_scaling: Option<LogScaling>,
    mask: &[f32],
    tokens: usize,
) -> Vec<f32> {
    attention_prefill(x, w, d, log_scaling, mask, tokens, None).0
}

/// Everything one attention layer must retain between generated tokens — the
/// host twin of [`crate::models::inkling::burn::AttnCache`], holding the same
/// two kinds of state for the same reason.
///
/// The keys and values themselves, and the `kernel - 1` **pre-convolution** K
/// and V projections that the *next* position's short convolution reaches back
/// into. Caching only the post-convolution K/V reads as complete and silently
/// truncates every short convolution at the prefill boundary — the taps see
/// zeros where three real positions should be.
///
/// `k` is post-convolution **and** post-QK-norm, `v` post-convolution; both are
/// functions of the prefix alone, which is the property that makes them
/// cacheable at all. Row 0 is absolute position [`AttnCache::base`], not 0: a
/// local layer drops keys that have left its window, so the row index is not
/// the position and every distance must be computed through `base`.
///
/// `Clone` on purpose, and it is what makes a rejected draft harmless: a
/// speculative position appends to a CLONE that is then dropped, so rollback is
/// the default and committing is the deliberate act. The other way round — mutate
/// and undo on rejection — fails silently, as a slowly falling acceptance rate
/// rather than an error.
#[derive(Clone)]
pub struct AttnCache {
    k: Vec<f32>,
    v: Vec<f32>,
    k_pre: Vec<f32>,
    v_pre: Vec<f32>,
    kv_width: usize,
    base: usize,
}

impl AttnCache {
    /// Keys retained — *not* the sequence length, because a windowed layer
    /// forgets.
    pub fn len(&self) -> usize {
        self.k.len() / self.kv_width
    }

    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }

    /// Absolute position of row 0.
    pub fn base(&self) -> usize {
        self.base
    }

    /// Drop the keys no future query can reach.
    ///
    /// A query at `pos` sees `pos - p < window`, and every later query sees a
    /// strictly older cut-off, so the last `window` rows are exactly enough and
    /// never more. Without this a local layer's cache grows without bound over
    /// a long generation while the extra rows are masked to `-inf` on every
    /// step — correct, and quadratic in a layer that was chosen to be linear.
    fn trim(&mut self, window: Option<usize>) {
        let Some(w) = window else { return };
        let len = self.len();
        if len <= w {
            return;
        }
        let drop = len - w;
        self.k.drain(..drop * self.kv_width);
        self.v.drain(..drop * self.kv_width);
        self.base += drop;
    }
}

/// The same layer, keeping what a decode step will need.
///
/// Identical arithmetic to [`attention`] — that function is this one with the
/// cache dropped — so whatever gates one gates both.
///
/// `window` is the sliding window on a local layer and `None` on a global one,
/// the same distinction [`causal_mask`] takes; it decides how much of the cache
/// survives, and passing `None` for a local layer would grow the cache past the
/// window rather than give a wrong answer.
pub fn attention_prefill(
    x: &[f32],
    w: &AttnWeights<'_>,
    d: &AttnDims,
    log_scaling: Option<LogScaling>,
    mask: &[f32],
    tokens: usize,
    window: Option<usize>,
) -> (Vec<f32>, AttnCache) {
    let q_width = d.heads * d.head_dim;
    let kv_width = d.kv_heads * d.head_dim;
    assert_eq!(x.len(), tokens * d.hidden);
    assert_eq!(mask.len(), tokens * tokens);

    let mut q = linear(x, w.wq, tokens, d.hidden, q_width);
    // K and V pass through their short convolutions; Q does not. The
    // pre-convolution projections are kept: they are the convolution's memory,
    // and a decode step cannot reconstruct them from the cached K and V.
    let k_pre = linear(x, w.wk, tokens, d.hidden, kv_width);
    let mut k = short_conv(&k_pre, w.k_sconv, tokens, kv_width, d.kernel);
    let v_pre = linear(x, w.wv, tokens, d.hidden, kv_width);
    let v = short_conv(&v_pre, w.v_sconv, tokens, kv_width, d.kernel);
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

    let mut cache = AttnCache {
        k,
        v,
        k_pre: conv_history(&k_pre, tokens, kv_width, d.kernel),
        v_pre: conv_history(&v_pre, tokens, kv_width, d.kernel),
        kv_width,
        base: 0,
    };
    cache.trim(window);
    (linear(&out, w.wo, tokens, q_width, d.hidden), cache)
}

/// One position through one attention layer, reading the cache.
///
/// The whole point of the cache: `x` is the single new position, `pos` is its
/// absolute index in the sequence, and the prefix is never recomputed. The
/// cache is advanced in place — the new K and V are appended, the short
/// convolution histories roll forward, and a windowed layer forgets its oldest
/// key. Give it a CLONE when the position is speculative.
///
/// `pos` is a parameter and not `cache.len()` because those are different
/// numbers the moment a window drops a key, and log scaling and the relative
/// bias are both functions of the **absolute** position. Deriving one from the
/// other works for exactly as long as the sequence is shorter than the window.
///
/// No `mask` argument: causality is structural here — every cached key precedes
/// `pos` — and the window is applied against the retained distances directly.
pub fn attention_step(
    x: &[f32],
    w: &AttnWeights<'_>,
    d: &AttnDims,
    log_scaling: Option<LogScaling>,
    pos: usize,
    window: Option<usize>,
    cache: &mut AttnCache,
) -> Vec<f32> {
    let q_width = d.heads * d.head_dim;
    let kv_width = d.kv_heads * d.head_dim;
    assert_eq!(x.len(), d.hidden, "a decode step feeds exactly one token");
    assert_eq!(cache.kv_width, kv_width, "this cache was built at a different layer shape");
    assert!(pos >= cache.base + cache.len(), "position {pos} is already cached");

    let mut q = linear(x, w.wq, 1, d.hidden, q_width);
    let mut k_new = short_conv_step(
        &mut cache.k_pre,
        &linear(x, w.wk, 1, d.hidden, kv_width),
        w.k_sconv,
        kv_width,
        d.kernel,
    );
    let v_new = short_conv_step(
        &mut cache.v_pre,
        &linear(x, w.wv, 1, d.hidden, kv_width),
        w.v_sconv,
        kv_width,
        d.kernel,
    );
    let r = linear(x, w.wr, 1, d.hidden, d.heads * d.d_rel);

    head_rms_norm(&mut q, w.q_norm, 1, d.heads, d.head_dim, d.rms_eps);
    head_rms_norm(&mut k_new, w.k_norm, 1, d.kv_heads, d.head_dim, d.rms_eps);

    cache.k.extend_from_slice(&k_new);
    cache.v.extend_from_slice(&v_new);
    cache.trim(window);
    let len = cache.len();
    let base = cache.base;

    // The same pairing the whole-sequence lane makes — the query AND the
    // position bias, global layers only — at the one absolute position this
    // step is at.
    let tau = match (d.kind, log_scaling) {
        (AttnKind::Global, Some(ls)) => ls.tau(pos),
        _ => 1.0,
    };
    if tau != 1.0 {
        for val in q.iter_mut() {
            *val *= tau;
        }
    }

    // Only the distances that can occur are worth projecting: the oldest
    // retained key is `pos - base` back and the table stops at `rel_extent`.
    let eff = d.rel_extent.min(pos - base + 1);
    let mut rel = vec![0f32; d.heads * eff];
    for h in 0..d.heads {
        for e in 0..eff {
            let mut acc = 0f32;
            for c in 0..d.d_rel {
                acc += r[h * d.d_rel + c] * w.rel_proj[c * d.rel_extent + e];
            }
            rel[h * eff + e] = acc;
        }
    }

    let scaling = d.scaling();
    let groups = d.groups();
    let mut out = vec![0f32; q_width];
    let mut scores = vec![0f32; len];

    for h in 0..d.heads {
        let kv_h = h / groups;
        let qv = &q[h * d.head_dim..(h + 1) * d.head_dim];
        for j in 0..len {
            let kv =
                &cache.k[j * kv_width + kv_h * d.head_dim..j * kv_width + (kv_h + 1) * d.head_dim];
            let dot: f32 = qv.iter().zip(kv).map(|(a, b)| a * b).sum();
            // Every retained key is at or before `pos`, so this cannot go
            // negative — which is the whole reason no causal mask is needed.
            let dist = pos - (base + j);
            let bias = if dist < d.rel_extent { rel[h * eff + dist] * tau } else { 0.0 };
            // `trim` has usually applied the window already; keeping it here
            // makes this function right without depending on that.
            let wmask = if window.is_some_and(|wnd| dist >= wnd) {
                f32::NEG_INFINITY
            } else {
                0.0
            };
            scores[j] = dot * scaling + bias + wmask;
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

        for j in 0..len {
            let p = scores[j];
            if p == 0.0 {
                continue;
            }
            let vv = &cache.v[j * kv_width + kv_h * d.head_dim
                ..j * kv_width + (kv_h + 1) * d.head_dim];
            let o = &mut out[h * d.head_dim..(h + 1) * d.head_dim];
            for (acc, &val) in o.iter_mut().zip(vv) {
                *acc += p * val;
            }
        }
    }

    linear(&out, w.wo, 1, q_width, d.hidden)
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
