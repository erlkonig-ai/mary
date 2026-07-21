//! RoPE for the SmolVLA towers — **half-split** (GPT-NeoX) convention, distinct
//! from `f5`'s interleaved pairs. Applied per-head to `[B, L, H, D]` given
//! integer positions `[B, L]`.
//!
//! Reference (`apply_rope`, max_wavelength=10000):
//!   freq_exponents = (2/D)·arange(D/2)
//!   timescale      = max_wavelength^freq_exponents
//!   radians        = positions / timescale            (broadcast over heads)
//!   x1, x2         = x.split(D/2, dim=-1)
//!   out[:D/2]      = x1·cos − x2·sin
//!   out[D/2:]      = x2·cos + x1·sin

use burn::prelude::*;

/// `x: [B, L, H, D]`, `positions: [B, L]` (float) → `[B, L, H, D]`.
pub fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    positions: Tensor<B, 2>,
    max_wavelength: f64,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [b, l, _h, d] = x.dims();
    let d_half = d / 2;

    // timescale = max_wavelength^((2/D)·arange(d_half))            [d_half]
    let freq = Tensor::<B, 1, Int>::arange(0..d_half as i64, device)
        .float()
        .mul_scalar(2.0 / d as f64);
    let timescale = freq.mul_scalar(max_wavelength.ln()).exp();

    // radians = positions[...,None] / timescale[None,None,:]       [B, L, 1, d_half]
    let radians = positions
        .reshape([b, l, 1])
        .div(timescale.reshape([1, 1, d_half]))
        .reshape([b, l, 1, d_half]);
    let sin = radians.clone().sin();
    let cos = radians.cos();

    // x1, x2 = split(d_half, -1)                                   [B, L, H, d_half]
    let x1 = x.clone().narrow(3, 0, d_half);
    let x2 = x.narrow(3, d_half, d_half);

    let out1 = x1.clone().mul(cos.clone()).sub(x2.clone().mul(sin.clone()));
    let out2 = x2.mul(cos).add(x1.mul(sin));
    Tensor::cat(vec![out1, out2], 3)
}
