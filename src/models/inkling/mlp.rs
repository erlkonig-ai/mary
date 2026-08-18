//! Inkling's DENSE MLP on the host — what the MTP heads run, and nothing else.
//!
//! Semantics from `transformers.models.inkling.modeling_inkling`, gated by
//! `inkling_layer_gate`.
//!
//! # What was here, and why it is gone
//!
//! An f32 host reference for the WHOLE MoE block — `expert_ffn_one`,
//! `routed_experts`, `shared_experts`, `moe` — sat beside these three functions
//! and was the readable transcription of the routed lane. It is deleted, and not
//! for tidiness. It was a lane with no caller in the data plane and a shape that
//! invited work: it read as the algorithm, so anyone asked to make the routed
//! experts faster found IT first and optimised a function the forward never
//! calls. That happened twice in one day. The live routed lane is
//! `inkling_forward::routed_experts_fp4` (NVFP4) and `routed_experts_bf16`
//! (layer 2), and the algorithm those two implement is now documented there,
//! where the code is.
//!
//! The gates that ran against it went with it: an f32 reference demands a
//! precision this model does not have. Inkling is bfloat16 with NVFP4 experts,
//! so "the device agrees with the f32 host lane" is a claim about which
//! implementation was written first. `inkling_bf16_expert_gate` and
//! `inkling_fp4_expert_gate` are what replaced it, and both hold the device
//! kernels to something Python wrote.
//!
//! # Why the dense MLP stays
//!
//! It is not a second implementation of anything. The main stack's dense layers
//! are on the device (`inkling_forward::dense_mlp_bf16`); this is what the MTP
//! heads run, and the MTP heads have no device lane at all. Deleting it would
//! delete multi-token prediction, not a fallback.
//!
//! One hazard survives the deletion and is recorded here because the file it was
//! written in is the one that shrank. `shared_w13_weight` is
//! `[n_shared, 2 * intermediate, hidden]` = `[2, 4096, 4096]` in both released
//! checkpoints, so neither its shape nor any total sum distinguishes the
//! INTERLEAVED reading (`g0, u0, g1, u1, …`) from the HALVED one (all gates,
//! then all ups) — the two splits are permutations of each other and have the
//! identical total. It is INTERLEAVED, settled by running both on a real
//! forward: ' Paris' at top-1 logit 18.69 against '<|begin_of_text|>' at 8.94,
//! the latter emitting no English at all. `load::split_shared_w13` is the one
//! place that does the split and `INK_SHARED_W13_HALVED=1` re-runs the
//! experiment.
//!
//! The dense MLP has its own `global_scale` scalar, applied to the block's
//! output. It is one multiply and trivially forgotten.

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
