use burn::prelude::*;

/// Precomputed rotary position embeddings for 1D sequences.
pub struct RotaryEmbedding<B: Backend> {
    pub cos: Tensor<B, 2>, // [max_seq_len, head_dim]
    pub sin: Tensor<B, 2>, // [max_seq_len, head_dim]
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Precompute cos/sin tables for RoPE.
    /// `dim` is head_dim, `max_len` is max sequence length, `theta` is the base frequency.
    pub fn new(dim: usize, max_len: usize, theta: f64, device: &B::Device) -> Self {
        // freqs = 1.0 / (theta ^ (arange(0, dim, 2) / dim))
        let half_dim = dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freq.push(1.0 / theta.powf(i as f64 * 2.0 / dim as f64));
        }
        let inv_freq_tensor =
            Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device).unsqueeze::<2>(); // [1, half_dim]

        // positions = arange(0, max_len)
        let positions: Vec<f32> = (0..max_len).map(|i| i as f32).collect();
        let pos_tensor =
            Tensor::<B, 1>::from_floats(positions.as_slice(), device).unsqueeze_dim::<2>(1); // [max_len, 1]

        // freqs = positions * inv_freq -> [max_len, half_dim]
        let freqs = pos_tensor.matmul(inv_freq_tensor);

        // Duplicate freqs: [max_len, half_dim] -> [max_len, dim]
        // Python uses torch.cat((freqs, freqs), dim=-1): [f0, f1, ..., f0, f1, ...]
        let cos_freqs = freqs.clone().cos(); // [max_len, half_dim]
        let sin_freqs = freqs.sin(); // [max_len, half_dim]

        // Match Python's halved convention: cat(freqs, freqs)
        let cos = Tensor::cat(vec![cos_freqs.clone(), cos_freqs], 1); // [max_len, dim]
        let sin = Tensor::cat(vec![sin_freqs.clone(), sin_freqs], 1); // [max_len, dim]

        Self { cos, sin }
    }

    /// Apply rotary embeddings to a tensor of shape [B, heads, S, head_dim].
    /// `start_pos` is the starting position for this chunk of sequence.
    pub fn apply(&self, x: Tensor<B, 4>, start_pos: usize) -> Tensor<B, 4> {
        let [_b, _h, s, _d] = x.dims();

        // Slice cos/sin to [S, D], then broadcast to [1, 1, S, D]
        let cos = self
            .cos
            .clone()
            .slice([start_pos..start_pos + s])
            .unsqueeze::<3>() // [1, S, D]
            .unsqueeze::<4>(); // [1, 1, S, D] — broadcasts with [B, H, S, D]
        let sin = self
            .sin
            .clone()
            .slice([start_pos..start_pos + s])
            .unsqueeze::<3>()
            .unsqueeze::<4>();

        // Halved rotate_half: split into first/second half, produce (-x2, x1)
        let x_rotated = Self::rotate_half(x.clone());

        x * cos + x_rotated * sin
    }

    /// Rotate half (halved convention matching Python's HuggingFace):
    /// Split x into first half and second half along last dim.
    /// x1 = x[..., :D/2], x2 = x[..., D/2:]
    /// result = cat(-x2, x1)
    fn rotate_half(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, s, d] = x.dims();
        let half = d / 2;
        let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
        let x2 = x.slice([0..b, 0..h, 0..s, half..d]);
        Tensor::cat(vec![x2.neg(), x1], 3)
    }
}
