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
//!
//! Three of those four paths carry state across generated tokens — attention's
//! K/V and both short convolutions — which is why [`LayerCache`] holds three
//! things and not one. A cache that remembered only attention would read as
//! complete and be wrong from the first decode step.

use crate::models::inkling::attn::{
    attention_prefill, attention_step, AttnCache, AttnDims, AttnWeights, LogScaling,
};
use crate::models::inkling::block::{conv_history, rms_norm, route, short_conv, short_conv_step, Routing};
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

/// Everything one decoder layer must retain between generated tokens.
///
/// Attention's cache, and the `kernel - 1` inputs each of the two LAYER-level
/// short convolutions reaches back into. The second pair is the part a reading
/// of [`decoder_layer`] skips: those convolutions sit on the residual paths and
/// carry state across tokens exactly as attention does, so a cache holding only
/// `attn` is wrong at the first decode step and wrong quietly — the taps read
/// zeros where three real positions should be.
///
/// `Clone` so a speculative position can run against a copy that is then
/// dropped. See [`AttnCache`].
#[derive(Clone)]
pub struct LayerCache {
    pub attn: AttnCache,
    attn_sconv: Vec<f32>,
    mlp_sconv: Vec<f32>,
}

/// One decoder layer over a whole sequence, no cache.
///
/// [`decoder_layer_prefill`] with the cache dropped, so the gates that cover
/// this cover that.
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
    let (y, routing, _) =
        decoder_layer_prefill(x, lw, aw, dims, log_scaling, mlp, mask, tokens, None);
    (y, routing)
}

/// The same layer, keeping what a decode step will need.
///
/// `window` is the sliding window on a local layer and `None` on a global one —
/// the same distinction the `mask` already carries, in the form the cache needs:
/// how far back a query may look, and therefore how much of the cache can never
/// be read again.
pub fn decoder_layer_prefill(
    x: &[f32],
    lw: &LayerWeights<'_>,
    aw: &AttnWeights<'_>,
    dims: &AttnDims,
    log_scaling: Option<LogScaling>,
    mlp: &LayerMlp<'_>,
    mask: &[f32],
    tokens: usize,
    window: Option<usize>,
) -> (Vec<f32>, Option<Vec<Routing>>, LayerCache) {
    let hidden = dims.hidden;
    let kernel = dims.kernel;

    let h = rms_norm(x, lw.attn_norm, dims.rms_eps, tokens, hidden);
    let (h, attn) = attention_prefill(&h, aw, dims, log_scaling, mask, tokens, window);
    // The history is of the convolution's INPUT, not its output: the taps read
    // `x`, and seeding from the output would be a different function that looks
    // the same in every shape check.
    let attn_sconv = conv_history(&h, tokens, hidden, kernel);
    let h = short_conv(&h, lw.attn_sconv, tokens, hidden, kernel);
    let x1: Vec<f32> = x.iter().zip(&h).map(|(a, b)| a + b).collect();

    let h = rms_norm(&x1, lw.mlp_norm, dims.rms_eps, tokens, hidden);
    let (h, routing) = mlp.forward(&h, tokens, hidden);
    let mlp_sconv = conv_history(&h, tokens, hidden, kernel);
    let h = short_conv(&h, lw.mlp_sconv, tokens, hidden, kernel);
    let x2: Vec<f32> = x1.iter().zip(&h).map(|(a, b)| a + b).collect();

    (x2, routing, LayerCache { attn, attn_sconv, mlp_sconv })
}

/// One position through one decoder layer, reading the cache.
///
/// `x` is the single new position and `pos` is its absolute index. The cache is
/// advanced in place; give it a CLONE when the position is speculative and drop
/// the clone, which is what makes a rejected draft leave nothing behind.
pub fn decoder_layer_step(
    x: &[f32],
    lw: &LayerWeights<'_>,
    aw: &AttnWeights<'_>,
    dims: &AttnDims,
    log_scaling: Option<LogScaling>,
    mlp: &LayerMlp<'_>,
    pos: usize,
    window: Option<usize>,
    cache: &mut LayerCache,
) -> (Vec<f32>, Option<Vec<Routing>>) {
    let hidden = dims.hidden;
    let kernel = dims.kernel;
    assert_eq!(x.len(), hidden, "a decode step feeds exactly one token");

    let h = rms_norm(x, lw.attn_norm, dims.rms_eps, 1, hidden);
    let h = attention_step(&h, aw, dims, log_scaling, pos, window, &mut cache.attn);
    let h = short_conv_step(&mut cache.attn_sconv, &h, lw.attn_sconv, hidden, kernel);
    let x1: Vec<f32> = x.iter().zip(&h).map(|(a, b)| a + b).collect();

    let h = rms_norm(&x1, lw.mlp_norm, dims.rms_eps, 1, hidden);
    let (h, routing) = mlp.forward(&h, 1, hidden);
    let h = short_conv_step(&mut cache.mlp_sconv, &h, lw.mlp_sconv, hidden, kernel);
    let x2: Vec<f32> = x1.iter().zip(&h).map(|(a, b)| a + b).collect();

    (x2, routing)
}
