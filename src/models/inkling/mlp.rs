//! Inkling MLPs — dense, routed experts, shared experts — the f32 reference lane.
//!
//! Semantics from `transformers.models.inkling.modeling_inkling`, gated by
//! `inkling_layer_gate`. Three things here are easy to get wrong:
//!
//! * The stacked expert matrix is `[experts, 2 * intermediate, hidden]`, laid
//!   out the way `nn.Linear` stores a weight. In BOTH released checkpoints
//!   `2 * intermediate == hidden`, so it is square and a transposed reading
//!   loads without complaint and computes nonsense. The oracle config makes it
//!   non-square on purpose so the gate can see the difference.
//! * **`shared_w13_weight` is square for the same reason, and had the same
//!   hazard with no warning attached.** `[n_shared, 2 * intermediate, hidden]`
//!   = `[2, 4096, 4096]`, so neither its shape nor any total sum distinguishes
//!   the INTERLEAVED reading (`g0, u0, g1, u1, …`) from the HALVED one (all
//!   gates, then all ups) — the two splits are permutations of each other and
//!   have the identical total. It is INTERLEAVED, settled by running both on a
//!   real forward: ' Paris' at top-1 logit 18.69 against '<|begin_of_text|>' at
//!   8.94, the latter emitting no English at all. `load::split_shared_w13` is
//!   the one place that does the split, `inkling_real_gate` asserts the
//!   orientation against the oracle, and `INK_SHARED_W13_HALVED=1` re-runs the
//!   experiment.
//! * The shared experts consume the MoE block's **original input**, not the
//!   routed output — they run beside the routed path, not after it.
//! * The dense MLP has its own `global_scale` scalar, applied to the block's
//!   output. It is one multiply and trivially forgotten.

use crate::models::inkling::block::Routing;

/// `x * sigmoid(x)` — `hidden_act` is silu in both releases.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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

/// Dense MLP: `down(silu(gate(x)) * up(x)) * global_scale`.
pub fn dense_mlp(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    global_scale: f32,
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let g = linear(x, gate_w, tokens, hidden, inter);
    let u = linear(x, up_w, tokens, hidden, inter);
    let act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
    let mut out = linear(&act, down_w, tokens, inter, hidden);
    for v in out.iter_mut() {
        *v *= global_scale;
    }
    out
}

/// One expert's feed-forward on one token: `down(silu(gate) * up)`.
///
/// `gate_up` is `[2 * intermediate, hidden]` with the gate rows FIRST; the
/// checkpoint interleaves them and `load::deinterleave_fused` puts them in this
/// order. This is the unit the Burn lane implements, so both lanes call the
/// same arithmetic rather than two transcriptions of it.
pub fn expert_ffn_one(x: &[f32], gate_up: &[f32], down: &[f32], hidden: usize, inter: usize) -> Vec<f32> {
    assert_eq!(x.len(), hidden);
    let both = linear(x, gate_up, 1, hidden, 2 * inter);
    let act: Vec<f32> = (0..inter).map(|i| silu(both[i]) * both[inter + i]).collect();
    linear(&act, down, 1, inter, hidden)
}

/// Routed experts over a stacked `[experts, 2 * intermediate, hidden]` matrix.
///
/// Each token goes to its `top_k` experts, is weighted by that expert's routing
/// weight, and the contributions are summed.
pub fn routed_experts(
    x: &[f32],
    gate_up: &[f32],
    down: &[f32],
    routing: &[Routing],
    experts: usize,
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    assert_eq!(gate_up.len(), experts * 2 * inter * hidden);
    assert_eq!(down.len(), experts * hidden * inter);
    assert_eq!(routing.len(), tokens);

    let mut out = vec![0f32; tokens * hidden];
    for (t, r) in routing.iter().enumerate() {
        let xt = &x[t * hidden..(t + 1) * hidden];
        for (slot, &e) in r.experts.iter().enumerate() {
            let gu = &gate_up[e * 2 * inter * hidden..(e + 1) * 2 * inter * hidden];
            let dn = &down[e * hidden * inter..(e + 1) * hidden * inter];
            let contrib = expert_ffn_one(xt, gu, dn, hidden, inter);
            let wgt = r.weights[slot];
            for (o, c) in out[t * hidden..(t + 1) * hidden].iter_mut().zip(&contrib) {
                *o += c * wgt;
            }
        }
    }
    out
}

/// Shared experts: every token visits all of them, weighted by its gammas.
///
/// `gate` and `up` are `[shared, intermediate, hidden]`, `down` is
/// `[shared, hidden, intermediate]`. The checkpoint concatenates gate and up
/// into one `shared_w13_weight`; splitting it is the layout's job.
pub fn shared_experts(
    x: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    gammas: &[f32],
    n_shared: usize,
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    assert_eq!(gate.len(), n_shared * inter * hidden);
    assert_eq!(up.len(), n_shared * inter * hidden);
    assert_eq!(down.len(), n_shared * hidden * inter);
    assert_eq!(gammas.len(), tokens * n_shared);

    let mut out = vec![0f32; tokens * hidden];
    for s in 0..n_shared {
        let g = &gate[s * inter * hidden..(s + 1) * inter * hidden];
        let u = &up[s * inter * hidden..(s + 1) * inter * hidden];
        let d = &down[s * hidden * inter..(s + 1) * hidden * inter];
        let gs = linear(x, g, tokens, hidden, inter);
        let us = linear(x, u, tokens, hidden, inter);
        // The gamma multiplies the activation, before the down projection.
        let act: Vec<f32> = (0..tokens * inter)
            .map(|i| silu(gs[i]) * us[i] * gammas[(i / inter) * n_shared + s])
            .collect();
        let contrib = linear(&act, d, tokens, inter, hidden);
        for (o, c) in out.iter_mut().zip(&contrib) {
            *o += c;
        }
    }
    out
}

/// The whole MoE block: routed experts plus shared experts.
///
/// The shared experts see `x`, the block's input — not the routed result.
#[allow(clippy::too_many_arguments)]
pub fn moe(
    x: &[f32],
    routing: &[Routing],
    gate_up: &[f32],
    down: &[f32],
    shared_gate: &[f32],
    shared_up: &[f32],
    shared_down: &[f32],
    experts: usize,
    n_shared: usize,
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let gammas: Vec<f32> = routing.iter().flat_map(|r| r.shared_gammas.clone()).collect();
    let routed = routed_experts(x, gate_up, down, routing, experts, tokens, hidden, inter);
    let shared = shared_experts(x, shared_gate, shared_up, shared_down, &gammas, n_shared, tokens, hidden, inter);
    routed.iter().zip(&shared).map(|(a, b)| a + b).collect()
}
