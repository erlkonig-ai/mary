//! Inkling's Burn lane — the same arithmetic as the f32 slice lane, on a backend.
//!
//! Mirrors how `k3` keeps `kda.rs` beside its Burn path: the slice lane in
//! [`crate::models::inkling::mlp`] is the reference, gated against
//! `transformers`, and this is checked against *it*. So the Burn lane gets a
//! real oracle without needing torch in the loop.
//!
//! Scope is the cost-dominant part first. A 5-token forward decoded 929 expert
//! slabs; each one is a `[2 * intermediate, hidden]` and a
//! `[hidden, intermediate]` matmul, which is where the time goes. RMSNorm is
//! here too because every block runs two of them.
//!
//! Everything stays f32. The slice lane is f32 and the checkpoint's dense
//! weights are BF16 widened to f32, so a rounding policy like K3's `ActRound`
//! would be describing a lane that does not exist yet; when a bf16 lane is
//! added it should get that treatment explicitly rather than by default.

use burn::prelude::*;
use burn::tensor::Tensor;

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
