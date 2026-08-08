//! One whole Inkling decoder layer — the f32 reference lane.
//!
//! From `InklingDecoderLayer.forward`:
//!
//! ```text
//! r = x;  h = attn_norm(x);  h = attn(h);  h = attn_sconv(h);  x = r + h
//! r = x;  h = mlp_norm(x);   h = mlp(h);   h = mlp_sconv(h);   x = r + h
//! ```
//!
//! The two layer-level short convolutions follow attention and the MLP rather
//! than feeding them, and each carries its own internal residual, so the block
//! has *four* additive paths where a careless reading sees two.

use crate::models::inkling::attn::{attention, AttnDims, AttnWeights, LogScaling};
use crate::models::inkling::block::{rms_norm, route, short_conv, Routing};
use crate::models::inkling::mlp::{dense_mlp, moe};

/// Which MLP a layer carries. `dense_mlp_idx` picks between them, and they use
/// different checkpoint names as well as different arithmetic.
pub enum LayerMlp<'a> {
    Dense {
        gate: &'a [f32],
        up: &'a [f32],
        down: &'a [f32],
        global_scale: f32,
        inter: usize,
    },
    Sparse {
        router_weight: &'a [f32],
        router_bias: &'a [f32],
        router_global_scale: f32,
        route_scale: f32,
        top_k: usize,
        gate_up: &'a [f32],
        down: &'a [f32],
        shared_gate: &'a [f32],
        shared_up: &'a [f32],
        shared_down: &'a [f32],
        experts: usize,
        n_shared: usize,
        inter: usize,
    },
}

impl LayerMlp<'_> {
    /// Run the MLP, returning its output and — when sparse — what the router
    /// decided, which the gate inspects separately.
    pub fn forward(&self, x: &[f32], tokens: usize, hidden: usize) -> (Vec<f32>, Option<Vec<Routing>>) {
        match self {
            LayerMlp::Dense { gate, up, down, global_scale, inter } => (
                dense_mlp(x, gate, up, down, *global_scale, tokens, hidden, *inter),
                None,
            ),
            LayerMlp::Sparse {
                router_weight, router_bias, router_global_scale, route_scale, top_k,
                gate_up, down, shared_gate, shared_up, shared_down, experts, n_shared, inter,
            } => {
                let routing = route(
                    x, router_weight, router_bias, *router_global_scale, *route_scale,
                    tokens, hidden, *experts, *n_shared, *top_k,
                );
                let y = moe(
                    x, &routing, gate_up, down, shared_gate, shared_up, shared_down,
                    *experts, *n_shared, tokens, hidden, *inter,
                );
                (y, Some(routing))
            }
        }
    }
}

/// Everything a decoder layer needs beyond its attention weights.
pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub mlp_norm: &'a [f32],
    /// `[hidden, kernel]` each — applied AFTER attention and after the MLP.
    pub attn_sconv: &'a [f32],
    pub mlp_sconv: &'a [f32],
}

/// One decoder layer over a whole sequence, no cache.
pub fn decoder_layer(
    x: &[f32],
    lw: &LayerWeights<'_>,
    aw: &AttnWeights<'_>,
    dims: &AttnDims,
    log_scaling: Option<LogScaling>,
    mlp: &LayerMlp<'_>,
    mask: &[f32],
    tokens: usize,
) -> (Vec<f32>, Option<Vec<Routing>>) {
    let hidden = dims.hidden;
    let kernel = dims.kernel;

    let h = rms_norm(x, lw.attn_norm, dims.rms_eps, tokens, hidden);
    let h = attention(&h, aw, dims, log_scaling, mask, tokens);
    let h = short_conv(&h, lw.attn_sconv, tokens, hidden, kernel);
    let x1: Vec<f32> = x.iter().zip(&h).map(|(a, b)| a + b).collect();

    let h = rms_norm(&x1, lw.mlp_norm, dims.rms_eps, tokens, hidden);
    let (h, routing) = mlp.forward(&h, tokens, hidden);
    let h = short_conv(&h, lw.mlp_sconv, tokens, hidden, kernel);
    let x2: Vec<f32> = x1.iter().zip(&h).map(|(a, b)| a + b).collect();

    (x2, routing)
}
