//! Inkling block primitives — the f32 reference lane.
//!
//! Plain slices, no backend, the way `k3::kda` keeps a reference lane beside
//! the Burn one: this is the arithmetic the GPU lane gets checked against, and
//! it is easier to be sure of when nothing is hidden behind a kernel.
//!
//! Semantics are taken from `transformers.models.inkling.modeling_inkling`,
//! which is authoritative, and gated against it by `inkling_block_gate`. Two
//! of them are easy to get wrong from the checkpoint alone:
//!
//! * [`short_conv`] returns `x + conv(x)`. The reference's
//!   `InklingShortConvolution.forward` keeps its own residual, so an
//!   implementation that returns just the convolution is wrong by an identity
//!   term — worth up to 4.57 absolute on the captured oracle, so it fails
//!   loudly rather than subtly.
//! * [`route`]'s score-correction bias decides *which* experts are chosen and
//!   never *how much* they weigh. The weights come from the raw logits.

/// `log(sigmoid(x))`, computed without overflowing either tail.
fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// RMS normalization: variance in f32, then the elementwise gain.
///
/// `x` is `[tokens, width]` row-major; `weight` is `[width]`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f64, tokens: usize, width: usize) -> Vec<f32> {
    assert_eq!(x.len(), tokens * width);
    assert_eq!(weight.len(), width);
    let mut out = vec![0f32; tokens * width];
    for t in 0..tokens {
        let row = &x[t * width..(t + 1) * width];
        // The reference accumulates in f32 after an explicit .float(); f64 here
        // would be *more* accurate and therefore a different function.
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let scale = (mean_sq + eps as f32).sqrt().recip();
        for (i, &v) in row.iter().enumerate() {
            out[t * width + i] = weight[i] * (v * scale);
        }
    }
    out
}

/// Depthwise causal short convolution, **plus its internal residual**.
///
/// `x` is `[tokens, dim]` row-major, `weight` is `[dim, kernel]` (the
/// checkpoint stores `[dim, 1, kernel]`; the singleton is dropped).
///
/// `F.conv1d` is a cross-correlation, and the reference pads by `kernel - 1`
/// then truncates to `tokens`, so tap `kernel - 1` multiplies the current
/// token and tap 0 multiplies the oldest one:
///
/// ```text
/// conv[t] = sum_{j=0}^{k-1} w[j] * x[t + j - (k - 1)]        x[<0] = 0
/// out[t]  = x[t] + conv[t]
/// ```
pub fn short_conv(x: &[f32], weight: &[f32], tokens: usize, dim: usize, kernel: usize) -> Vec<f32> {
    assert_eq!(x.len(), tokens * dim);
    assert_eq!(weight.len(), dim * kernel);
    let mut out = vec![0f32; tokens * dim];
    for t in 0..tokens {
        for d in 0..dim {
            let mut acc = 0f32;
            for j in 0..kernel {
                // t + j - (kernel - 1); anything before the sequence is zero.
                let src = t + j;
                if src < kernel - 1 {
                    continue;
                }
                let src = src - (kernel - 1);
                if src < tokens {
                    acc += weight[d * kernel + j] * x[src * dim + d];
                }
            }
            out[t * dim + d] = x[t * dim + d] + acc;
        }
    }
    out
}

/// The last `kernel - 1` rows of `x`, front-padded with zeros when the sequence
/// is shorter than that — the state [`short_conv_step`] reads.
///
/// A tap reaches `kernel - 1` positions back and a sequence that has not got
/// that far yet is padded with the same zeros [`short_conv`] assumes for
/// positions before the sequence. Seeding the history from a prefill shorter
/// than the kernel and *not* padding would silently shift every subsequent tap
/// by one position.
pub fn conv_history(x: &[f32], tokens: usize, dim: usize, kernel: usize) -> Vec<f32> {
    assert_eq!(x.len(), tokens * dim);
    let want = kernel - 1;
    let mut h = vec![0f32; want * dim];
    let take = want.min(tokens);
    h[(want - take) * dim..].copy_from_slice(&x[(tokens - take) * dim..tokens * dim]);
    h
}

/// One position of the short convolution, advancing `hist` in place.
///
/// `cat(hist, x)` is exactly the window the last row of [`short_conv`] reads,
/// so the tap arithmetic is not restated here — there is one implementation and
/// the cached lane cannot drift from the uncached one.
pub fn short_conv_step(
    hist: &mut Vec<f32>,
    x: &[f32],
    weight: &[f32],
    dim: usize,
    kernel: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), dim, "a decode step convolves exactly one position");
    assert_eq!(
        hist.len(),
        (kernel - 1) * dim,
        "history must be the {} rows before it",
        kernel - 1
    );
    let mut win = std::mem::take(hist);
    win.extend_from_slice(x);
    let out = short_conv(&win, weight, kernel, dim, kernel);
    *hist = win[dim..].to_vec();
    out[(kernel - 1) * dim..].to_vec()
}

/// What the router decided for one token.
#[derive(Debug, Clone)]
pub struct Routing {
    /// The `top_k` routed experts, in the order the weights are given.
    pub experts: Vec<usize>,
    /// One weight per chosen routed expert.
    pub weights: Vec<f32>,
    /// One weight per shared expert.
    pub shared_gammas: Vec<f32>,
}

/// Sigmoid router with shared-expert sinks.
///
/// `weight` is `[n_routed + n_shared, hidden]` — the shared experts occupy
/// their own rows, which is what `shared_expert_sink` means. `bias` is
/// `[n_routed]`: it shifts the scores used to *pick* the top `k` and takes no
/// part in the weights.
///
/// The chosen routed logits and *all* the shared logits are then normalized
/// together, so the shared experts compete in the same distribution rather
/// than being added on afterwards. The result is scaled by
/// `route_scale * global_scale`, which is why a token's weights sum to
/// `route_scale` and not to one.
#[allow(clippy::too_many_arguments)]
pub fn route(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    global_scale: f32,
    route_scale: f32,
    tokens: usize,
    hidden: usize,
    n_routed: usize,
    n_shared: usize,
    top_k: usize,
) -> Vec<Routing> {
    let rows = n_routed + n_shared;
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(weight.len(), rows * hidden);

    let mut logits = vec![0f32; tokens * rows];
    for t in 0..tokens {
        let xt = &x[t * hidden..(t + 1) * hidden];
        for e in 0..rows {
            let w = &weight[e * hidden..(e + 1) * hidden];
            logits[t * rows + e] = xt.iter().zip(w).map(|(a, b)| a * b).sum::<f32>();
        }
    }
    route_from_logits(
        &logits,
        bias,
        global_scale,
        route_scale,
        tokens,
        n_routed,
        n_shared,
        top_k,
    )
}

/// [`route`] with the projection already done, wherever it was done.
///
/// The split is not a refactor for tidiness: the projection is
/// `[tokens, hidden] x [n_routed + n_shared, hidden]ᵀ`, a real matmul that
/// belongs on the device, and everything below it — a sigmoid, a sort of 256
/// scores, a log-softmax over `top_k + n_shared` of them — is CONTROL PLANE. It
/// decides which weights to read. It touches 258 numbers a token and it decides
/// something the host has to know anyway, so it stays here.
///
/// One implementation of the selection rule, called from both, because the rule
/// is where the subtlety is: the bias shifts the scores used to PICK and takes
/// no part in the weights, and a second transcription would eventually forget
/// that.
///
/// `logits` is `[tokens, n_routed + n_shared]` row-major.
#[allow(clippy::too_many_arguments)]
pub fn route_from_logits(
    logits: &[f32],
    bias: &[f32],
    global_scale: f32,
    route_scale: f32,
    tokens: usize,
    n_routed: usize,
    n_shared: usize,
    top_k: usize,
) -> Vec<Routing> {
    let rows = n_routed + n_shared;
    assert_eq!(logits.len(), tokens * rows);
    assert_eq!(bias.len(), n_routed);

    let mut out = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let logits = &logits[t * rows..(t + 1) * rows];
        if let Some((row, value)) = logits
            .iter()
            .copied()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            panic!("router logit is non-finite at token {t}, row {row}: {value}");
        }

        // Selection uses sigmoid(logit) + bias; the weights below do not.
        let mut order: Vec<usize> = (0..n_routed).collect();
        order.sort_by(|&a, &b| {
            let sa = sigmoid(logits[a]) + bias[a];
            let sb = sigmoid(logits[b]) + bias[b];
            sb.partial_cmp(&sa).unwrap().then(a.cmp(&b))
        });
        let experts: Vec<usize> = order[..top_k].to_vec();

        // Chosen routed logits, then every shared logit.
        let mut selected: Vec<f32> = experts.iter().map(|&e| logits[e]).collect();
        selected.extend_from_slice(&logits[n_routed..]);

        let lp: Vec<f32> = selected.iter().map(|&v| log_sigmoid(v)).collect();
        let m = lp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + lp.iter().map(|&v| (v - m).exp()).sum::<f32>().ln();
        let w: Vec<f32> = lp
            .iter()
            .map(|&v| (v - lse).exp() * route_scale * global_scale)
            .collect();

        out.push(Routing {
            experts,
            weights: w[..top_k].to_vec(),
            shared_gammas: w[top_k..].to_vec(),
        });
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod route_finite_tests {
    use super::route_from_logits;

    #[test]
    #[should_panic(expected = "router logit is non-finite at token 0, row 3")]
    fn shared_nan_is_loud_before_softmax() {
        let logits = [0.25, -0.5, 1.0, f32::NAN];
        let _ = route_from_logits(&logits, &[0.0, 0.0], 1.0, 1.0, 1, 2, 2, 1);
    }

    #[test]
    fn finite_logits_produce_only_finite_weights() {
        let logits = [0.25, -0.5, 1.0, -2.0, 0.75, 0.125, -0.25, 0.5];
        let routes = route_from_logits(&logits, &[0.0, 0.0], 1.0, 1.0, 2, 2, 2, 1);
        let examined = routes
            .iter()
            .map(|r| r.weights.len() + r.shared_gammas.len())
            .sum::<usize>();
        println!("examined {examined} routing weights");
        assert_eq!(examined, 6);
        assert!(
            routes
                .iter()
                .flat_map(|r| r.weights.iter().chain(&r.shared_gammas))
                .all(|v| v.is_finite())
        );
    }
}
