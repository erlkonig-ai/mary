//! SmolVLA's timestep embedding for flow-matching. Unlike `f5`'s scale-1000
//! sinusoid, SmolVLA spaces the periods **geometrically** from `min_period`
//! (4e-3) to `max_period` (4.0), giving sensitivity across time ∈ [0,1].
//!
//! Reference (`create_sinusoidal_pos_embedding`):
//!   fraction = linspace(0, 1, dim/2)
//!   period   = min_period * (max_period/min_period)^fraction
//!   scaling  = (1/period) * 2π
//!   emb      = cat([sin(scaling·t), cos(scaling·t)])

use burn::prelude::*;

/// `time: [B]` → `[B, dim]`. `dim` must be even.
pub fn sinusoidal_time_embedding<B: Backend>(
    time: Tensor<B, 1>,
    dim: usize,
    min_period: f64,
    max_period: f64,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(dim % 2 == 0, "time embedding dim must be even");
    let half = dim / 2;
    let b = time.dims()[0];

    // fraction = linspace(0,1,half) = arange(half) / (half-1)
    let fraction = Tensor::<B, 1, Int>::arange(0..half as i64, device)
        .float()
        .div_scalar(half as f64 - 1.0);
    // period = min_period * (max_period/min_period)^fraction
    let ratio_ln = (max_period / min_period).ln();
    let period = fraction.mul_scalar(ratio_ln).exp().mul_scalar(min_period);
    // scaling = (1/period) * 2π                                 [half]
    let scaling = period.recip().mul_scalar(2.0 * std::f64::consts::PI);

    // sin_input = scaling[None,:] * time[:,None]                [B, half]
    let sin_input = time.reshape([b, 1]).mul(scaling.reshape([1, half]));
    Tensor::cat(vec![sin_input.clone().sin(), sin_input.cos()], 1)
}
