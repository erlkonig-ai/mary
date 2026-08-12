//! Inkling's Burn lane — the same arithmetic as the f32 slice lane, on a backend.
//!
//! Mirrors how `k3` keeps `kda.rs` beside its Burn path: the slice lane in
//! [`crate::models::inkling::mlp`] is the reference, gated against
//! `transformers`, and this is checked against *it*. So the Burn lane gets a
//! real oracle without needing torch in the loop.
//!
//! Scope was the cost-dominant part first — the routed experts — and is now the
//! whole arithmetic of a decoder layer: attention with its two short
//! convolutions, the shared experts, the dense MLP, RMSNorm, and the NVFP4
//! decode. What drove the second half was a measured 401 s forward of 109
//! tokens in which attention was 108 s and the shared plus dense MLPs 145 s, all
//! of it scalar host code left behind as a correctness reference. Moving both
//! took them to 8.9 s and 15.0 s.
//!
//! Every one of these is gated against `transformers` by `inkling_burn_gate`,
//! not against the slice lane: the two lanes were written by the same hand and
//! agreeing with each other proves only that.
//!
//! Everything stays f32. The slice lane is f32 and the checkpoint's dense
//! weights are BF16 widened to f32, so a rounding policy like K3's `ActRound`
//! would be describing a lane that does not exist yet; when a bf16 lane is
//! added it should get that treatment explicitly rather than by default.

use burn::prelude::*;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

/// `x * sigmoid(x)`, elementwise.
pub fn silu<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let s = burn::tensor::activation::sigmoid(x.clone());
    x * s
}

/// `nn.Linear(bias=False)`: `x @ Wᵀ` for a `[out, in]` weight.
///
/// The weight keeps its checkpoint orientation, so a transposition mistake is
/// a shape error rather than a plausible wrong answer.
pub fn linear<B: Backend>(x: Tensor<B, 2>, w: Tensor<B, 2>) -> Tensor<B, 2> {
    let [_, k] = x.dims();
    let [_, kw] = w.dims();
    assert_eq!(k, kw, "linear: x is [_, {k}] but the weight is [_, {kw}]");
    x.matmul(w.transpose())
}

/// RMS normalization with a per-feature gain.
///
/// Divides by `sqrt(var + eps)` rather than multiplying by its reciprocal: on
/// some backends `recip` dispatches to an approximate SIMD reciprocal, which
/// cost K3 about fourteen bits of accuracy before it was caught. Same hazard
/// here, same avoidance.
pub fn rms_norm<B: Backend>(x: Tensor<B, 2>, gain: Tensor<B, 1>, eps: f64) -> Tensor<B, 2> {
    let [_, w] = x.dims();
    assert_eq!(gain.dims()[0], w, "rms_norm: gain is {} wide, input {w}", gain.dims()[0]);
    let mean_sq = x.clone().powf_scalar(2.0).mean_dim(1);
    let denom = mean_sq.add_scalar(eps).sqrt();
    let normed = x / denom;
    normed * gain.unsqueeze::<2>()
}

/// One expert's feed-forward: `down(silu(gate) * up)`.
///
/// `gate_up` is `[2 * intermediate, hidden]` with the gate rows FIRST — the
/// checkpoint stores them interleaved and
/// [`crate::models::inkling::load::deinterleave_fused`] puts them in this order
/// at load. Passing a raw checkpoint tensor here is shape-identical and wrong,
/// which is exactly the bug that made the whole model emit noise while every
/// parity gate passed.
pub fn expert_ffn<B: Backend>(
    x: Tensor<B, 2>,
    gate_up: Tensor<B, 2>,
    down: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [two_inter, _] = gate_up.dims();
    assert!(two_inter % 2 == 0, "gate_up must have an even row count");
    let inter = two_inter / 2;
    let both = linear(x, gate_up);
    let [rows, _] = both.dims();
    let gate = both.clone().slice([0..rows, 0..inter]);
    let up = both.slice([0..rows, inter..2 * inter]);
    linear(silu(gate) * up, down)
}

/// The dense MLP: `down(silu(gate(x)) * up(x)) * global_scale`.
pub fn dense_mlp<B: Backend>(
    x: Tensor<B, 2>,
    gate: Tensor<B, 2>,
    up: Tensor<B, 2>,
    down: Tensor<B, 2>,
    global_scale: f32,
) -> Tensor<B, 2> {
    let g = linear(x.clone(), gate);
    let u = linear(x, up);
    linear(silu(g) * u, down).mul_scalar(global_scale)
}

/// The shared experts, on device — every token visits all of them.
///
/// `gate` and `up` are `[n_shared * intermediate, hidden]`, `down` is
/// `[n_shared * hidden, intermediate]`, and `gammas` is `[tokens, n_shared]`.
/// The gamma multiplies the **activation**, before the down projection — not
/// the block's output, which is algebraically the same only because `down` is
/// linear and is a different function the moment anything else is inserted.
///
/// Gate and up arrive already split rather than fused, so this can be gated
/// straight against `transformers`' own `shared_experts.gate_proj` /
/// `up_proj` with nothing transcribed in between; the checkpoint's interleaved
/// `shared_w13_weight` is turned into these by [`split_shared_fused`].
pub fn shared_experts<B: Backend>(
    x: Tensor<B, 2>,
    gate: Tensor<B, 2>,
    up: Tensor<B, 2>,
    down: Tensor<B, 2>,
    gammas: Tensor<B, 2>,
    n_shared: usize,
) -> Tensor<B, 2> {
    let [tokens, hidden] = x.dims();
    let [drows, inter] = down.dims();
    assert_eq!(drows, n_shared * hidden, "shared w2 has {drows} rows, want {}", n_shared * hidden);
    assert_eq!(gate.dims(), [n_shared * inter, hidden], "shared gate is {:?}", gate.dims());
    assert_eq!(up.dims(), [n_shared * inter, hidden], "shared up is {:?}", up.dims());
    assert_eq!(gammas.dims(), [tokens, n_shared], "gammas must be [tokens, n_shared]");

    let mut acc: Option<Tensor<B, 2>> = None;
    for s in 0..n_shared {
        let g = gate.clone().slice([s * inter..(s + 1) * inter, 0..hidden]);
        let u = up.clone().slice([s * inter..(s + 1) * inter, 0..hidden]);
        let dn = down.clone().slice([s * hidden..(s + 1) * hidden, 0..inter]);
        let gamma = gammas.clone().slice([0..tokens, s..s + 1]);
        let act = silu(linear(x.clone(), g)) * linear(x.clone(), u) * gamma;
        let contrib = linear(act, dn);
        acc = Some(match acc {
            None => contrib,
            Some(a) => a + contrib,
        });
    }
    acc.expect("a MoE layer has at least one shared expert")
}

/// Split the checkpoint's fused `shared_w13_weight` into gate and up blocks.
///
/// `fused` is `[n_shared * 2 * intermediate, hidden]` in **checkpoint
/// interleave** — gate on the even rows, up on the odd ones, per shared expert.
/// Returns `[n_shared * intermediate, hidden]` twice, which is what
/// [`shared_experts`] wants.
///
/// Splitting on device keeps the host copy in raw checkpoint order, so the
/// 33 M-element shuffle per layer never happens on a scalar loop.
pub fn split_shared_fused<B: Backend>(
    fused: Tensor<B, 2>,
    n_shared: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [frows, hidden] = fused.dims();
    assert_eq!(frows % (2 * n_shared), 0, "{frows} rows do not split into {n_shared} experts");
    let inter = frows / (2 * n_shared);
    let mut gates = Vec::with_capacity(n_shared);
    let mut ups = Vec::with_capacity(n_shared);
    for s in 0..n_shared {
        let gu = deinterleave_rows_device(
            fused.clone().slice([s * 2 * inter..(s + 1) * 2 * inter, 0..hidden]),
        );
        gates.push(gu.clone().slice([0..inter, 0..hidden]));
        ups.push(gu.slice([inter..2 * inter, 0..hidden]));
    }
    (Tensor::cat(gates, 0), Tensor::cat(ups, 0))
}

/// The last `kernel - 1` rows of `x`, left-padded with zeros when `x` is short.
///
/// This is the short convolution's whole memory: the taps reach `kernel - 1`
/// positions back, and a sequence that has not got that far yet is padded with
/// the same zeros [`short_conv`] assumes for `x[<0]`. Seeding the history from a
/// prefill shorter than the kernel and *not* padding would silently shift every
/// subsequent tap by one position.
pub fn conv_history<B: Backend>(x: Tensor<B, 2>, kernel: usize) -> Tensor<B, 2> {
    let [tokens, dim] = x.dims();
    let want = kernel - 1;
    if tokens >= want {
        x.slice([tokens - want..tokens, 0..dim])
    } else {
        let dev = x.device();
        let pad: Tensor<B, 2> = Tensor::zeros([want - tokens, dim], &dev);
        Tensor::cat(vec![pad, x], 0)
    }
}

/// One position of the short convolution, given the `kernel - 1` inputs before
/// it. Returns the output row and the history to carry to the next position.
///
/// `hist` is oldest-first, so `cat(hist, x)` is exactly the window
/// `x[pos - (kernel - 1) ..= pos]` that [`short_conv`]'s last row reads —
/// which is why this delegates to that function rather than restating the tap
/// arithmetic. The two lanes cannot drift because there is only one.
pub fn short_conv_step<B: Backend>(
    hist: Tensor<B, 2>,
    x: Tensor<B, 2>,
    weight: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [rows, dim] = x.dims();
    assert_eq!(rows, 1, "a decode step convolves exactly one position");
    let [_, kernel] = weight.dims();
    assert_eq!(
        hist.dims(),
        [kernel - 1, dim],
        "the history must be the {} rows before this one",
        kernel - 1
    );
    let win = Tensor::cat(vec![hist, x], 0);
    let out = short_conv(win.clone(), weight);
    (out.slice([kernel - 1..kernel, 0..dim]), conv_history(win, kernel))
}

/// Depthwise causal short convolution **plus its internal residual**, on device.
///
/// The device twin of [`crate::models::inkling::block::short_conv`]:
///
/// ```text
/// conv[t] = sum_{j=0}^{k-1} w[j] * x[t + j - (k - 1)]        x[<0] = 0
/// out[t]  = x[t] + conv[t]
/// ```
///
/// Written as `k` shifted slices of a front-zero-padded input rather than as a
/// convolution kernel, because `k` is 4 and the shift is exactly what the
/// formula says. Returning only `conv` — dropping the module's own residual —
/// is the mistake this shape makes hard to hide.
pub fn short_conv<B: Backend>(x: Tensor<B, 2>, weight: Tensor<B, 2>) -> Tensor<B, 2> {
    let [tokens, dim] = x.dims();
    let [wdim, kernel] = weight.dims();
    assert_eq!(dim, wdim, "short_conv: x is [_, {dim}] but the weight is [{wdim}, _]");
    assert!(kernel > 0, "a short convolution needs a kernel");
    let dev = x.device();
    let pad: Tensor<B, 2> = Tensor::zeros([kernel - 1, dim], &dev);
    let padded = Tensor::cat(vec![pad, x.clone()], 0);

    let mut conv: Option<Tensor<B, 2>> = None;
    for j in 0..kernel {
        // t + j - (kernel - 1) in x is t + j in the padded tensor.
        let seg = padded.clone().slice([j..j + tokens, 0..dim]);
        let wj = weight.clone().slice([0..dim, j..j + 1]).reshape([1, dim]);
        let term = seg * wj;
        conv = Some(match conv {
            None => term,
            Some(c) => c + term,
        });
    }
    x + conv.expect("kernel > 0")
}

/// RMS-normalize each head slice of `[tokens, heads * head_dim]`.
fn head_rms_norm<B: Backend>(
    v: Tensor<B, 2>,
    gain: Tensor<B, 1>,
    heads: usize,
    head_dim: usize,
    eps: f64,
) -> Tensor<B, 2> {
    let [tokens, width] = v.dims();
    assert_eq!(width, heads * head_dim, "{width} is not {heads} x {head_dim}");
    rms_norm(v.reshape([tokens * heads, head_dim]), gain, eps).reshape([tokens, width])
}

/// Every weight one attention layer needs, already on the device.
///
/// Orientations are the checkpoint's: `w*` are `[out, in]` the way `nn.Linear`
/// stores them, the short convolutions are `[dim, kernel]`, and `rel_proj` is
/// `[d_rel, rel_extent]`.
pub struct AttnWeightsDev<B: Backend> {
    pub wq: Tensor<B, 2>,
    pub wk: Tensor<B, 2>,
    pub wv: Tensor<B, 2>,
    pub wr: Tensor<B, 2>,
    pub wo: Tensor<B, 2>,
    pub k_sconv: Tensor<B, 2>,
    pub v_sconv: Tensor<B, 2>,
    pub q_norm: Tensor<B, 1>,
    pub k_norm: Tensor<B, 1>,
    pub rel_proj: Tensor<B, 2>,
}

/// Everything one attention layer must retain between generated tokens.
///
/// Two kinds of state, and forgetting the second is the bug this type exists to
/// make hard: the keys and values themselves, and the `kernel - 1`
/// **pre-convolution** K and V projections that the *next* position's short
/// convolution reaches back into. Caching only the post-convolution K/V reads
/// as complete and silently truncates every short convolution at the prefill
/// boundary — the taps see zeros where three real positions should be.
///
/// `k` is post-convolution **and** post-QK-norm, `v` post-convolution; both are
/// functions of the prefix alone, which is the property that makes them
/// cacheable. Row 0 is absolute position [`AttnCache::base`], not 0: a local
/// layer drops keys that have left its window, so the row index is not the
/// position and every distance must be computed through `base`.
pub struct AttnCache<B: Backend> {
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    k_pre: Tensor<B, 2>,
    v_pre: Tensor<B, 2>,
    base: usize,
}

impl<B: Backend> AttnCache<B> {
    /// Keys retained — *not* the sequence length, because a windowed layer
    /// forgets.
    pub fn len(&self) -> usize {
        self.k.dims()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Absolute position of row 0.
    pub fn base(&self) -> usize {
        self.base
    }
}

/// Drop the keys no future query can reach.
///
/// A query at `pos` sees `pos - p < window`, and every later query sees a
/// strictly older cut-off, so the last `window` rows are exactly enough and
/// never more. Without this a local layer's cache grows without bound over a
/// long generation while the extra rows are masked to `-inf` on every step —
/// correct, and quadratic in a layer that was chosen to be linear.
fn trim<B: Backend>(c: &mut AttnCache<B>, window: Option<usize>) {
    let Some(w) = window else { return };
    let [len, dim] = c.k.dims();
    if len <= w {
        return;
    }
    let drop = len - w;
    c.k = c.k.clone().slice([drop..len, 0..dim]);
    c.v = c.v.clone().slice([drop..len, 0..dim]);
    c.base += drop;
}

/// One attention layer over a whole sequence, on the device, no cache.
///
/// The device twin of [`crate::models::inkling::attn::attention`] and gated
/// against the same `transformers` capture, not against it: matching the slice
/// lane would only prove the two agree, and they were written by the same hand.
///
/// `mask` is the additive `[tokens, tokens]` mask — zero where a key is visible
/// and `-inf` where it is not — because a local layer's mask carries the sliding
/// window and a global layer's does not.
///
/// Two things are folded together here that a careless reading separates:
/// log scaling multiplies the query **and** the relative-position bias, and only
/// on global layers; and the bias is zero outside `0 <= q - k < rel_extent`,
/// while causality lives in the mask.
pub fn attention<B: Backend>(
    x: Tensor<B, 2>,
    w: &AttnWeightsDev<B>,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 2> {
    attention_prefill(x, w, d, log_scaling, mask, None).0
}

/// The same layer, keeping what a decode step will need.
///
/// Identical arithmetic to [`attention`] — that function is this one with the
/// cache dropped, so the `transformers` gate covers both and there is no second
/// transcription of the layer to drift.
///
/// `window` is the sliding window on a local layer and `None` on a global one,
/// the same distinction [`crate::models::inkling::attn::causal_mask`] takes;
/// it decides how much of the cache survives, and passing `None` for a local
/// layer would grow the cache past the window rather than give a wrong answer.
pub fn attention_prefill<B: Backend>(
    x: Tensor<B, 2>,
    w: &AttnWeightsDev<B>,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    mask: Tensor<B, 2>,
    window: Option<usize>,
) -> (Tensor<B, 2>, AttnCache<B>) {
    use crate::models::inkling::config::AttnKind;

    let [tokens, hidden] = x.dims();
    assert_eq!(hidden, d.hidden, "x is [_, {hidden}] but the config says {}", d.hidden);
    assert_eq!(mask.dims(), [tokens, tokens], "the mask must be [tokens, tokens]");
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(groups * kv_heads, heads, "{heads} heads do not divide into {kv_heads} kv heads");

    // K and V pass through their short convolutions; Q does not. The
    // pre-convolution projections are kept: they are the convolution's memory,
    // and a decode step cannot reconstruct them from the cached K and V.
    let q = linear(x.clone(), w.wq.clone());
    let k_pre = linear(x.clone(), w.wk.clone());
    let v_pre = linear(x.clone(), w.wv.clone());
    let k = short_conv(k_pre.clone(), w.k_sconv.clone());
    let v = short_conv(v_pre.clone(), w.v_sconv.clone());
    let r = linear(x, w.wr.clone());

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k = head_rms_norm(k, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    // Log scaling: the same vector the slice lane builds, from the same method.
    let taus: Vec<f32> = (0..tokens)
        .map(|t| match (d.kind, log_scaling) {
            (AttnKind::Global, Some(ls)) => ls.tau(t),
            _ => 1.0,
        })
        .collect();
    let tau: Tensor<B, 1> = Tensor::from_data(TensorData::new(taus, [tokens]), &dev);
    let q = q * tau.clone().reshape([tokens, 1]);

    // Only distances that can occur are worth projecting: a distance is at most
    // `tokens - 1` and the table stops at `rel_extent`.
    let eff = d.rel_extent.min(tokens);
    let mut idx = vec![0i32; tokens * tokens];
    let mut valid = vec![0f32; tokens * tokens];
    for qi in 0..tokens {
        for ki in 0..tokens {
            let dist = qi as isize - ki as isize;
            if dist >= 0 && (dist as usize) < d.rel_extent {
                idx[qi * tokens + ki] = dist as i32;
                valid[qi * tokens + ki] = 1.0;
            }
        }
    }
    let idx: Tensor<B, 3, Int> =
        Tensor::from_data(TensorData::new(idx, [1, tokens, tokens]), &dev).repeat_dim(0, heads);
    let valid: Tensor<B, 3> = Tensor::from_data(TensorData::new(valid, [1, tokens, tokens]), &dev);

    let rel = r
        .reshape([tokens * heads, d.d_rel])
        .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
        .reshape([tokens, heads, eff])
        .swap_dims(0, 1)
        * tau.reshape([1, tokens, 1]);
    let bias = rel.gather(2, idx) * valid;

    // [heads, tokens, head_dim]; the KV heads are repeated in place, so head h
    // reads kv head h / groups exactly as the slice lane indexes it.
    let qh = q.reshape([tokens, heads, head_dim]).swap_dims(0, 1);
    let expand = |t: Tensor<B, 2>| -> Tensor<B, 3> {
        t.reshape([tokens, kv_heads, head_dim])
            .swap_dims(0, 1)
            .reshape([kv_heads, 1, tokens, head_dim])
            .repeat_dim(1, groups)
            .reshape([heads, tokens, head_dim])
    };
    let kh = expand(k.clone());
    let vh = expand(v.clone());

    let scores = qh.matmul(kh.swap_dims(1, 2)).mul_scalar(d.scaling()) + bias
        + mask.reshape([1, tokens, tokens]);
    let probs = burn::tensor::activation::softmax(scores, 2);
    let out = probs
        .matmul(vh)
        .swap_dims(0, 1)
        .reshape([tokens, heads * head_dim]);

    let mut cache = AttnCache {
        k,
        v,
        k_pre: conv_history(k_pre, d.kernel),
        v_pre: conv_history(v_pre, d.kernel),
        base: 0,
    };
    trim(&mut cache, window);
    (linear(out, w.wo.clone()), cache)
}

/// One generated token through one attention layer, reading the cache.
///
/// The whole point of the cache: `x` is the single new position, `pos` is its
/// absolute index in the sequence, and the prefix is never recomputed. The
/// cache is advanced in place — the new K and V are appended, the short
/// convolution histories roll forward, and a windowed layer forgets its oldest
/// key.
///
/// `pos` is a parameter and not `cache.len()` because those are different
/// numbers the moment a window drops a key, and log scaling and the relative
/// bias are both functions of the **absolute** position. Deriving one from the
/// other works for exactly as long as the sequence is shorter than the window.
///
/// No `mask` argument: causality is structural here — every cached key precedes
/// `pos` — and the window is applied against the retained distances directly.
pub fn attention_step<B: Backend>(
    x: Tensor<B, 2>,
    w: &AttnWeightsDev<B>,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    pos: usize,
    window: Option<usize>,
    cache: &mut AttnCache<B>,
) -> Tensor<B, 2> {
    use crate::models::inkling::config::AttnKind;

    let [rows, hidden] = x.dims();
    assert_eq!(rows, 1, "a decode step feeds exactly one token, got {rows}");
    assert_eq!(hidden, d.hidden, "x is [_, {hidden}] but the config says {}", d.hidden);
    assert!(pos >= cache.base + cache.len(), "position {pos} is already cached");
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(groups * kv_heads, heads, "{heads} heads do not divide into {kv_heads} kv heads");

    let q = linear(x.clone(), w.wq.clone());
    let (k_new, k_hist) =
        short_conv_step(cache.k_pre.clone(), linear(x.clone(), w.wk.clone()), w.k_sconv.clone());
    let (v_new, v_hist) =
        short_conv_step(cache.v_pre.clone(), linear(x.clone(), w.wv.clone()), w.v_sconv.clone());
    cache.k_pre = k_hist;
    cache.v_pre = v_hist;
    let r = linear(x, w.wr.clone());

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k_new = head_rms_norm(k_new, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    cache.k = Tensor::cat(vec![cache.k.clone(), k_new], 0);
    cache.v = Tensor::cat(vec![cache.v.clone(), v_new], 0);
    trim(cache, window);
    let len = cache.len();
    let base = cache.base;

    let tau = match (d.kind, log_scaling) {
        (AttnKind::Global, Some(ls)) => ls.tau(pos),
        _ => 1.0,
    };
    let q = q.mul_scalar(tau);

    // One row of what the full lane builds as a [tokens, tokens] table: the
    // backward distance to each retained key, whether the relative table
    // reaches that far, and whether the window admits it at all. Built on the
    // host because `len` is the context length, not a matrix.
    let mut idx = vec![0i32; len];
    let mut valid = vec![0f32; len];
    let mut wmask = vec![0f32; len];
    let mut max_dist = 0usize;
    for j in 0..len {
        // Every retained key is at or before `pos`, so this cannot go negative.
        let dist = pos - (base + j);
        if dist < d.rel_extent {
            idx[j] = dist as i32;
            valid[j] = 1.0;
        }
        if window.is_some_and(|wnd| dist >= wnd) {
            wmask[j] = f32::NEG_INFINITY;
        }
        max_dist = max_dist.max(dist);
    }
    let eff = d.rel_extent.min(max_dist + 1);
    let idx: Tensor<B, 3, Int> =
        Tensor::from_data(TensorData::new(idx, [1, 1, len]), &dev).repeat_dim(0, heads);
    let valid: Tensor<B, 3> = Tensor::from_data(TensorData::new(valid, [1, 1, len]), &dev);
    let wmask: Tensor<B, 3> = Tensor::from_data(TensorData::new(wmask, [1, 1, len]), &dev);

    let rel = r
        .reshape([heads, d.d_rel])
        .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
        .reshape([heads, 1, eff])
        .mul_scalar(tau);
    let bias = rel.gather(2, idx) * valid;

    let qh = q.reshape([1, heads, head_dim]).swap_dims(0, 1);
    let expand = |t: Tensor<B, 2>| -> Tensor<B, 3> {
        t.reshape([len, kv_heads, head_dim])
            .swap_dims(0, 1)
            .reshape([kv_heads, 1, len, head_dim])
            .repeat_dim(1, groups)
            .reshape([heads, len, head_dim])
    };
    let kh = expand(cache.k.clone());
    let vh = expand(cache.v.clone());

    let scores = qh.matmul(kh.swap_dims(1, 2)).mul_scalar(d.scaling()) + bias + wmask;
    let probs = burn::tensor::activation::softmax(scores, 2);
    let out = probs.matmul(vh).swap_dims(0, 1).reshape([1, heads * head_dim]);
    linear(out, w.wo.clone())
}

/// The SHARED experts, on the device, from weights that are ALREADY on it.
///
/// The host lane in [`crate::models::inkling::mlp::shared_experts`] is the
/// reference and this has to agree with it. Two details are easy to lose in
/// translation and each one changes every token:
///
/// * the gamma multiplies the ACTIVATION — between the SwiGLU and the down
///   projection — not the block's output. Applying it after `down` is a
///   different function whenever `down` is not the identity, which is always.
/// * gammas arrive token-major, `[token, shared]`, so shared expert `s` takes a
///   stride-`n_shared` column and not a contiguous block. A contiguous read is
///   correct for `n_shared == 1` and silently wrong for 2, which is what this
///   checkpoint has.
///
/// `gate` and `up` are `[inter, hidden]` per shared expert and `down` is
/// `[hidden, inter]` — one tensor each rather than a stacked rank 3, because
/// these are uploaded once for the whole run and never sliced again. That is
/// the entire point: the caller holds the handles, and a token costs a matmul
/// against memory the device already owns rather than a fresh upload.
pub fn shared_experts_dev<B: Backend>(
    x: Tensor<B, 2>,
    gate: &[Tensor<B, 2>],
    up: &[Tensor<B, 2>],
    down: &[Tensor<B, 2>],
    gammas: &[f32],
    n_shared: usize,
) -> Tensor<B, 2> {
    assert_eq!(gate.len(), n_shared, "{} gate weights for {n_shared} shared experts", gate.len());
    assert_eq!(up.len(), n_shared, "{} up weights for {n_shared} shared experts", up.len());
    assert_eq!(down.len(), n_shared, "{} down weights for {n_shared} shared experts", down.len());
    let [n, _] = x.dims();
    assert_eq!(gammas.len(), n * n_shared, "{} gammas for {n} tokens", gammas.len());

    let dev = x.device();
    let mut out: Option<Tensor<B, 2>> = None;
    for s in 0..n_shared {
        let g = linear(x.clone(), gate[s].clone());
        let u = linear(x.clone(), up[s].clone());
        let col: Vec<f32> = (0..n).map(|t| gammas[t * n_shared + s]).collect();
        let gam = Tensor::<B, 2>::from_data(TensorData::new(col, [n, 1]), &dev);
        let c = linear(silu(g) * u * gam, down[s].clone());
        out = Some(match out {
            Some(o) => o + c,
            None => c,
        });
    }
    out.expect("a MoE layer with no shared experts")
}

/// FP4 (E2M1) values by 4-bit code, and the E4M3 table, as device tensors.
///
/// Built on the host from the scalar decoders in
/// [`crate::models::inkling::nvfp4`], which are gated bit-exactly against
/// `compressed_tensors` and against torch over all 256 E4M3 patterns. A gather
/// through those tables cannot drift from the CPU lane; a reimplemented
/// bit-twiddle could.
fn luts<B: Backend>(dev: &B::Device) -> (Tensor<B, 1>, Tensor<B, 1>) {
    use crate::models::inkling::nvfp4::{e4m3_to_f32, FP4_E2M1};
    let fp4 = Tensor::from_data(TensorData::new(FP4_E2M1.to_vec(), [16]), dev);
    // NaN would poison a gather, and only 0x7F/0xFF are NaN in E4M3-fn; they
    // never appear as a block scale, so map them to zero rather than carrying
    // NaN into every product in the row.
    let e4m3: Vec<f32> = (0..256u16)
        .map(|b| {
            let v = e4m3_to_f32(b as u8);
            if v.is_nan() { 0.0 } else { v }
        })
        .collect();
    let e4m3 = Tensor::from_data(TensorData::new(e4m3, [256]), dev);
    (fp4, e4m3)
}

/// Look up `idx` in a 1-D table, preserving the index tensor's shape.
fn gather2<B: Backend>(table: Tensor<B, 1>, idx: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    let [r, c] = idx.dims();
    table.select(0, idx.reshape([r * c])).reshape([r, c])
}

/// Dequantise NVFP4 on device.
///
/// `codes` is `[rows, bytes]` holding the packed byte values, `scales` is
/// `[rows, bytes * 2 / GROUP]` holding raw E4M3 byte values, and `scale2` is one
/// factor per row. Returns `[rows, bytes * 2]`.
///
/// Nibble order is low-first, settled against
/// `compressed_tensors.compressors.unpack_fp4_from_uint8`; the association is
/// `(fp4 * block_scale) * scale2`, matching the reference, because float
/// multiplication does not associate and the CPU lane was gated on that order.
pub fn dequant_nvfp4<B: Backend>(
    codes: Tensor<B, 2, Int>,
    scales: Tensor<B, 2, Int>,
    scale2: Tensor<B, 1>,
) -> Tensor<B, 2> {
    use crate::models::inkling::nvfp4::GROUP;
    let dev = codes.device();
    let (fp4_lut, e4m3_lut) = luts::<B>(&dev);
    let [rows, bytes] = codes.dims();
    let logical = bytes * 2;

    // Two 4-bit codes per byte, low nibble FIRST.
    let lo = codes.clone().bitwise_and_scalar(0x0Fi32.elem());
    let hi = codes
        .bitwise_right_shift_scalar(4i32.elem())
        .bitwise_and_scalar(0x0Fi32.elem());
    // Interleave: [rows, bytes, 2] -> [rows, 2 * bytes] gives lo, hi, lo, hi...
    let pairs = Tensor::cat(
        vec![lo.reshape([rows, bytes, 1]), hi.reshape([rows, bytes, 1])],
        2,
    )
    .reshape([rows, logical]);
    let vals = gather2(fp4_lut, pairs);

    // One E4M3 scale per GROUP logical elements, widened to match.
    let n_scales = logical / GROUP;
    let s = gather2(e4m3_lut, scales)
        .reshape([rows, n_scales, 1])
        .repeat_dim(2, GROUP)
        .reshape([rows, logical]);

    // Block scale first, then the per-row factor -- the reference's order.
    vals.mul(s).mul(scale2.reshape([rows, 1]).repeat_dim(1, logical))
}


/// Reorder a fused `[2 * intermediate, hidden]` from the checkpoint's
/// interleave to gate-rows-first, on device.
///
/// The checkpoint stores gate and up **alternating by row**: gate is the even
/// output rows, up the odd ones. It does NOT store them as contiguous halves.
/// Splitting down the middle loads without complaint and scrambles every
/// SwiGLU in every layer -- shape-identical, catastrophically wrong, and
/// invisible to any check that compares two lanes which share the assumption.
/// Authority is `transformers`' `conversion_mapping.py`, key `inkling_mm_model`.
///
/// This is the device twin of
/// [`crate::models::inkling::load::deinterleave_fused`]; the two must agree.
pub fn deinterleave_rows_device<B: Backend>(fused: Tensor<B, 2>) -> Tensor<B, 2> {
    let [rows, _] = fused.dims();
    assert!(rows % 2 == 0, "fused row count {rows} is odd; gate/up cannot interleave");
    let half = rows / 2;
    let mut order: Vec<i32> = Vec::with_capacity(rows);
    order.extend((0..half).map(|r| (2 * r) as i32)); // gate: even rows
    order.extend((0..half).map(|r| (2 * r + 1) as i32)); // up: odd rows
    let dev = fused.device();
    let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(order, [rows]), &dev);
    fused.select(0, idx)
}

/// Upload one expert's packed NVFP4 bytes and dequantise them on the device.
///
/// Takes exactly what [`crate::models::inkling::load::Checkpoint::expert_slice_packed`]
/// returns, so the host never materialises the f32 weight. Returns
/// `[rows, cols * 2]`.
///
/// `scale2` is one factor for the whole expert; it is broadcast to every row
/// because [`dequant_nvfp4`] takes a per-row vector, which is the shape the
/// stacked layout would need if scale2 ever became per-row.
pub fn expert_weight_from_packed<B: Backend>(
    codes: &[u8],
    scales: &[u8],
    scale2: f32,
    rows: usize,
    cols: usize,
    dev: &B::Device,
) -> Tensor<B, 2> {
    assert_eq!(codes.len(), rows * cols, "codes is {} bytes, want {rows}x{cols}", codes.len());
    assert_eq!(scales.len() % rows, 0, "{} scales do not divide {rows} rows", scales.len());
    let n_scales = scales.len() / rows;

    // Bitcast four bytes into a word rather than widening each to an i32:
    // same bytes, a quarter of the elements, and no host-side expansion of the
    // very data the packed path exists to keep packed.
    assert_eq!(cols % 4, 0, "{cols} bytes per row does not pack into i32 words");
    assert_eq!(n_scales % 4, 0, "{n_scales} scales per row does not pack into i32 words");
    let word = |b: &[u8]| -> Vec<i32> {
        b.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let codes_t =
        Tensor::<B, 2, Int>::from_data(TensorData::new(word(codes), [rows, cols / 4]), dev);
    let scales_t =
        Tensor::<B, 2, Int>::from_data(TensorData::new(word(scales), [rows, n_scales / 4]), dev);
    let s2 = Tensor::<B, 1>::from_data(TensorData::new(vec![scale2; rows], [rows]), dev);
    dequant_nvfp4_words(codes_t, scales_t, s2)
}


/// Dequantise NVFP4 from **word-packed** codes, so the host never widens them.
///
/// `code_words` is `[rows, cols / 4]`: four consecutive packed bytes bitcast
/// into one little-endian `i32`. `scale_words` is `[rows, n_scales / 4]`, the
/// same treatment for the raw E4M3 scale bytes. Returns `[rows, cols * 2]`.
///
/// Why words: uploading one `i32` per byte expands the packed weight 4x on the
/// host, which measured at 27.6s of a 38.8s expert lane against 4.5s of actual
/// device work. A bitcast moves the same bytes and a quarter as many elements.
///
/// The nibble arithmetic is simpler here than in the byte form, not more
/// complex. Little-endian byte j occupies bits `8j..8j+7`, and NVFP4 stores the
/// low nibble first, so logical element k of a word is exactly `(w >> 4k) & 0xF`
/// for k in 0..8 — the byte stage has no reason to exist. Sign extension from
/// the arithmetic shift is discarded by the mask.
///
/// Gated against [`dequant_nvfp4`], which is itself bit-exact against
/// `compressed_tensors`, so this inherits that oracle rather than asserting its
/// own correctness.
pub fn dequant_nvfp4_words<B: Backend>(
    code_words: Tensor<B, 2, Int>,
    scale_words: Tensor<B, 2, Int>,
    scale2: Tensor<B, 1>,
) -> Tensor<B, 2> {
    use crate::models::inkling::nvfp4::GROUP;
    let dev = code_words.device();
    let (fp4_lut, e4m3_lut) = luts::<B>(&dev);
    let [rows, cwords] = code_words.dims();
    let [rows_s, swords] = scale_words.dims();
    assert_eq!(rows, rows_s, "codes have {rows} rows, scales {rows_s}");
    let logical = cwords * 8;

    // Eight 4-bit codes per word, low nibble first, in logical order.
    let mut nib = Vec::with_capacity(8);
    for k in 0..8u32 {
        nib.push(
            code_words
                .clone()
                .bitwise_right_shift_scalar(((4 * k) as i32).elem())
                .bitwise_and_scalar(0x0Fi32.elem())
                .reshape([rows, cwords, 1]),
        );
    }
    let codes = Tensor::cat(nib, 2).reshape([rows, logical]);
    let vals = gather2(fp4_lut, codes);

    // Four E4M3 scale bytes per word, likewise in order.
    let mut by = Vec::with_capacity(4);
    for j in 0..4u32 {
        by.push(
            scale_words
                .clone()
                .bitwise_right_shift_scalar(((8 * j) as i32).elem())
                .bitwise_and_scalar(0xFFi32.elem())
                .reshape([rows, swords, 1]),
        );
    }
    let n_scales = swords * 4;
    assert_eq!(n_scales * GROUP, logical, "{n_scales} scales cannot cover {logical} values");
    let s = gather2(e4m3_lut, Tensor::cat(by, 2).reshape([rows, n_scales]))
        .reshape([rows, n_scales, 1])
        .repeat_dim(2, GROUP)
        .reshape([rows, logical]);

    // Block scale first, then the per-row factor -- the reference's order,
    // because float multiplication does not associate.
    vals.mul(s).mul(scale2.reshape([rows, 1]).repeat_dim(1, logical))
}

/// The KV cache against the lane it is supposed to be an optimization of.
///
/// The oracle here is [`attention`] itself — the function `inkling_burn_gate`
/// checks against `transformers`. So these tests do not re-litigate whether the
/// layer is right; they check the only thing a cache can get wrong, which is
/// whether feeding one token at a time reproduces feeding all of them.
///
/// Deliberately at dimensions the real checkpoint never reaches: a window of 5
/// against 11 tokens, so the cache actually forgets, and a log-scaling floor of
/// 4 so `tau` varies per position. On the real config the window is 512 and the
/// floor 128000, which means an end-to-end run of a dozen tokens exercises
/// neither path — a passing generation would prove nothing about either.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inkling::attn::{causal_mask, AttnDims, LogScaling};
    use crate::models::inkling::config::AttnKind;

    type B = burn_ndarray::NdArray<f32>;

    /// Deterministic filler. A fixed pattern rather than a seeded RNG so a
    /// failure is reproducible from the source alone.
    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.7919 + seed).sin() * 0.5 + (i as f32 * 0.1237).cos() * 0.25)
            .collect()
    }

    fn dims(kind: AttnKind, rel_extent: usize) -> AttnDims {
        AttnDims {
            hidden: 8,
            heads: 4,
            kv_heads: 2,
            head_dim: 2,
            d_rel: 3,
            rel_extent,
            kernel: 4,
            rms_eps: 1e-6,
            kind,
        }
    }

    fn weights(d: &AttnDims, dev: &burn_ndarray::NdArrayDevice) -> AttnWeightsDev<B> {
        let m = |rows: usize, cols: usize, seed: f32| -> Tensor<B, 2> {
            Tensor::from_data(TensorData::new(fill(rows * cols, seed), [rows, cols]), dev)
        };
        let v = |n: usize, seed: f32| -> Tensor<B, 1> {
            Tensor::from_data(TensorData::new(fill(n, seed), [n]), dev)
        };
        let (q_w, kv_w) = (d.heads * d.head_dim, d.kv_heads * d.head_dim);
        AttnWeightsDev {
            wq: m(q_w, d.hidden, 0.1),
            wk: m(kv_w, d.hidden, 0.2),
            wv: m(kv_w, d.hidden, 0.3),
            wr: m(d.heads * d.d_rel, d.hidden, 0.4),
            wo: m(d.hidden, q_w, 0.5),
            k_sconv: m(kv_w, d.kernel, 0.6),
            v_sconv: m(kv_w, d.kernel, 0.7),
            q_norm: v(d.head_dim, 0.8),
            k_norm: v(d.head_dim, 0.9),
            rel_proj: m(d.d_rel, d.rel_extent, 1.0),
        }
    }

    /// Run `tokens` positions two ways and return the largest absolute
    /// disagreement over the decoded rows, plus the decoded rows themselves.
    fn compare(
        kind: AttnKind,
        rel_extent: usize,
        window: Option<usize>,
        ls: Option<LogScaling>,
        tokens: usize,
        prefill: usize,
        sabotage_conv_history: bool,
    ) -> f32 {
        let dev = burn_ndarray::NdArrayDevice::default();
        let d = dims(kind, rel_extent);
        let w = weights(&d, &dev);
        let xs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(fill(tokens * d.hidden, 2.5), [tokens, d.hidden]),
            &dev,
        );

        let full = attention(
            xs.clone(),
            &w,
            &d,
            ls,
            Tensor::from_data(
                TensorData::new(causal_mask(tokens, window), [tokens, tokens]),
                &dev,
            ),
        );

        let head = xs.clone().slice([0..prefill, 0..d.hidden]);
        let (_, mut cache) = attention_prefill(
            head,
            &w,
            &d,
            ls,
            Tensor::from_data(
                TensorData::new(causal_mask(prefill, window), [prefill, prefill]),
                &dev,
            ),
            window,
        );
        if sabotage_conv_history {
            // The mutation this cache exists to avoid: keep K and V, forget
            // what the short convolution still needs to see.
            cache.k_pre = cache.k_pre.clone().zeros_like();
            cache.v_pre = cache.v_pre.clone().zeros_like();
        }

        let mut worst = 0f32;
        for pos in prefill..tokens {
            let step = attention_step(
                xs.clone().slice([pos..pos + 1, 0..d.hidden]),
                &w,
                &d,
                ls,
                pos,
                window,
                &mut cache,
            );
            let want = full.clone().slice([pos..pos + 1, 0..d.hidden]);
            let diff = (step - want).abs().max().into_scalar();
            worst = worst.max(diff);
        }
        worst
    }

    /// A global layer with log scaling that actually varies, and a relative
    /// table that runs out before the sequence does — so distances past
    /// `rel_extent` must contribute a zero bias rather than a gathered one.
    #[test]
    fn cached_global_matches_full() {
        let ls = Some(LogScaling { n_floor: 4.0, alpha: 0.5 });
        let worst = compare(AttnKind::Global, 5, None, ls, 11, 4, false);
        assert!(worst < 2e-5, "cached global attention drifts by {worst}");
    }

    /// A local layer whose window is shorter than the sequence, so the cache
    /// must forget: 11 tokens through a window of 5 drops six keys.
    #[test]
    fn cached_local_matches_full_across_the_window() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 4, false);
        assert!(worst < 2e-5, "cached windowed attention drifts by {worst}");
    }

    /// The cache must survive a prefill shorter than the convolution kernel,
    /// where the history is mostly the zero padding `short_conv` assumes.
    #[test]
    fn cached_matches_full_from_a_two_token_prefill() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 2, false);
        assert!(worst < 2e-5, "cached attention from a short prefill drifts by {worst}");
    }

    /// A gate that cannot fail and a gate that has never failed look identical
    /// from outside. Forgetting the pre-convolution history is the plausible
    /// version of this bug — it leaves K and V cached and correct-looking — so
    /// prove the comparison above would have caught it.
    #[test]
    fn dropping_the_conv_history_is_caught() {
        let worst = compare(AttnKind::Local, 5, Some(5), None, 11, 4, true);
        assert!(worst > 1e-2, "sabotaged conv history only moved the answer by {worst}");
    }
}
