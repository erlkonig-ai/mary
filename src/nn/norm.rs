//! Normalization primitives shared across ports.

use burn::prelude::*;

/// LayerNorm without affine parameters (`elementwise_affine=False`):
/// `(x − mean) / sqrt(var + eps)` over the last dim. x: [B, S, D] → [B, S, D].
pub fn layer_norm_no_affine<B: Backend>(x: Tensor<B, 3>, eps: f64) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
    let inv_std = (variance + eps).sqrt().recip();
    centered * inv_std
}
