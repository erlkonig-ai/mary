use burn::prelude::*;

/// 4D Rotary Position Embedding for Flux2.
///
/// Unlike the text encoder's 1D RoPE, this operates on position IDs with shape [S, 4]
/// where each column corresponds to an axis (t, h, w, l).
///
/// For each axis i with dimension axes_dims[i], compute 1D rotary frequencies
/// then concatenate cos/sin along the feature dimension.
///
/// Output: (cos: [S, D_total], sin: [S, D_total]) where D_total = sum(axes_dims).
/// Uses repeat_interleave_real=True: each frequency is repeated twice [f0,f0,f1,f1,...].

/// Compute 4D RoPE embeddings from position IDs.
///
/// ids: [S, 4] position IDs (float)
/// axes_dims: [32, 32, 32, 32] dimension per axis
/// theta: base frequency (2000 for Flux2)
///
/// Returns: (cos, sin) each [S, sum(axes_dims)]
pub fn flux2_rope<B: Backend>(
    ids: Tensor<B, 2>, // [S, 4]
    axes_dims: &[usize],
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [seq_len, _num_axes] = ids.dims();

    let mut cos_parts: Vec<Tensor<B, 2>> = Vec::with_capacity(axes_dims.len());
    let mut sin_parts: Vec<Tensor<B, 2>> = Vec::with_capacity(axes_dims.len());

    for (axis_idx, &dim) in axes_dims.iter().enumerate() {
        // Extract positions for this axis: [S, 1] -> [S]
        let pos_2d = ids.clone().slice([0..seq_len, axis_idx..axis_idx + 1]); // [S, 1]
        // Squeeze from [S, 1] to [S] by reshaping
        let pos = pos_2d.reshape([seq_len]); // [S]

        // Compute 1D rotary embedding for this axis
        let (cos, sin) = get_1d_rotary_pos_embed(dim, pos, theta, device);
        cos_parts.push(cos);
        sin_parts.push(sin);
    }

    // Concatenate along feature dimension
    let freqs_cos = Tensor::cat(cos_parts, 1); // [S, sum(axes_dims)]
    let freqs_sin = Tensor::cat(sin_parts, 1); // [S, sum(axes_dims)]

    (freqs_cos, freqs_sin)
}

/// Compute 1D rotary position embedding for a single axis.
///
/// dim: embedding dimension for this axis (e.g. 32)
/// pos: [S] position values (float)
/// theta: base frequency
///
/// Returns: (cos, sin) each [S, dim] with repeat_interleave_real=True
fn get_1d_rotary_pos_embed<B: Backend>(
    dim: usize,
    pos: Tensor<B, 1>, // [S]
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half_dim = dim / 2;

    // inv_freq = 1.0 / (theta ^ (arange(0, dim, 2) / dim))
    let mut inv_freq_data = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        inv_freq_data.push(1.0 / theta.powf(i as f64 * 2.0 / dim as f64));
    }
    let inv_freq =
        Tensor::<B, 1>::from_floats(inv_freq_data.as_slice(), device).unsqueeze_dim::<2>(0);
    // inv_freq: [1, half_dim]

    // pos: [S] -> [S, 1]
    let pos_2d = pos.unsqueeze_dim::<2>(1);

    // freqs = pos * inv_freq -> [S, half_dim]
    let freqs = pos_2d.matmul(inv_freq);

    // cos/sin of freqs
    let cos_freqs = freqs.clone().cos(); // [S, half_dim]
    let sin_freqs = freqs.sin(); // [S, half_dim]

    // repeat_interleave: [a, b, c] -> [a, a, b, b, c, c]
    let cos = repeat_interleave(cos_freqs); // [S, dim]
    let sin = repeat_interleave(sin_freqs); // [S, dim]

    (cos, sin)
}

/// Repeat-interleave along last dimension: [S, D] -> [S, 2*D]
/// Each element is doubled: [a, b, c] -> [a, a, b, b, c, c]
fn repeat_interleave<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let [seq_len, half_dim] = x.dims();
    // [S, half_dim] -> [S, half_dim, 1] -> [S, half_dim, 2] -> [S, half_dim * 2]
    let expanded = x.unsqueeze_dim::<3>(2).repeat_dim(2, 2);
    expanded.reshape([seq_len, half_dim * 2])
}

/// Apply rotary embeddings to a tensor.
///
/// x: [B, S, H, D] (sequence_dim=1 format from Flux2)
/// cos, sin: [S, D]
///
/// This uses the same rotate_half pattern as the text encoder RoPE, but adapted for
/// the [B, S, H, D] layout used by Flux2 attention (sequence_dim=1).
pub fn apply_rotary_emb<B: Backend>(
    x: Tensor<B, 4>,   // [B, S, H, D]
    cos: Tensor<B, 2>, // [S, D]
    sin: Tensor<B, 2>, // [S, D]
) -> Tensor<B, 4> {
    // cos: [S, D] -> [1, S, 1, D]
    let cos = cos
        .unsqueeze_dim::<3>(0) // [1, S, D]
        .unsqueeze_dim::<4>(2); // [1, S, 1, D]
    let sin = sin
        .unsqueeze_dim::<3>(0) // [1, S, D]
        .unsqueeze_dim::<4>(2); // [1, S, 1, D]

    // rotate_half for [B, S, H, D]: reshape to [B, S, H, D/2, 2], swap and negate
    let x_rotated = rotate_half_bshd(x.clone());

    x * cos + x_rotated * sin
}

/// Rotate half for [B, S, H, D] layout.
/// For each pair (x0, x1) in the last dim, produce (-x1, x0).
fn rotate_half_bshd<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, s, h, d] = x.dims();
    let half = d / 2;
    // Reshape to [B, S, H, D/2, 2]
    let reshaped = x.reshape([b, s, h, half, 2]);
    // Split into pairs
    let x0 = reshaped.clone().slice([0..b, 0..s, 0..h, 0..half, 0..1]);
    let x1 = reshaped.slice([0..b, 0..s, 0..h, 0..half, 1..2]);
    // [-x1, x0]
    let neg_x1 = x1.neg();
    let rotated = Tensor::cat(vec![neg_x1, x0], 4); // [B, S, H, D/2, 2]
    rotated.reshape([b, s, h, d])
}
