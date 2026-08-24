//! GPU-native int8 quantized KV cache.
//!
//! Keeps quantized data entirely on GPU as packed i32 tensors (4 int8 values per i32).
//! Quantization and dequantization are burn tensor operations — no CPU involvement.
//!
//! This eliminates the GPU→CPU→GPU transfer that makes the CPU-based `QuantizedKvCache`
//! catastrophically slow at long contexts (0.2 tok/s vs 8.2 tok/s at 4K).
//!
//! Layout:
//! - Packed data: `Tensor<B, 2, Int>` shaped `[n_groups, group_size/4]`
//!   where n_groups = batch * n_kv_heads * seq_len, group_size = head_dim.
//!   Each i32 packs 4 uint8 values: `[v0 | v1<<8 | v2<<16 | v3<<24]`.
//! - Scale: `Tensor<B, 1>` shaped `[n_groups]` — per-group scale factor.
//! - Zero: `Tensor<B, 1>` shaped `[n_groups]` — per-group zero point (min value).
//!
//! Compression: 4x vs f32 for the data, plus tiny scale/zero overhead.

use burn::prelude::*;

// ---------------------------------------------------------------------------
// GPU quantize: f32 → packed i32 + scale + zero
// ---------------------------------------------------------------------------

/// Quantized tensor stored entirely on GPU.
///
/// `packed` holds `[n_groups, packed_dim]` where `packed_dim = head_dim / 4`.
/// Each i32 packs 4 uint8 quantized values.
pub struct GpuQuantTensor<B: Backend> {
    /// Packed quantized data: [n_groups, packed_dim] as Int tensor.
    pub packed: Tensor<B, 2, Int>,
    /// Per-group scale: [n_groups]. Dequantized value = quantized * scale + zero.
    pub scale: Tensor<B, 1>,
    /// Per-group zero point: [n_groups]. The minimum of the original range.
    pub zero: Tensor<B, 1>,
    /// Original shape: [batch, n_kv_heads, seq_len, head_dim].
    pub shape: [usize; 4],
}

/// Quantize a 4D f32 tensor to GPU-packed int8 representation.
///
/// Input: `[batch, n_kv_heads, seq_len, head_dim]` — must have head_dim divisible by 4.
/// The quantization is per-group where each group is one head_dim vector.
///
/// All operations happen on GPU via burn tensor ops.
pub fn gpu_quantize<B: Backend<IntElem = i32>>(input: Tensor<B, 4>) -> GpuQuantTensor<B> {
    let [batch, n_kv_heads, seq_len, head_dim] = input.dims();
    assert!(
        head_dim % 4 == 0,
        "head_dim must be divisible by 4, got {}",
        head_dim
    );

    let n_groups = batch * n_kv_heads * seq_len;
    let packed_dim = head_dim / 4;

    // Flatten to [n_groups, head_dim] for per-group operations
    let flat = input.reshape([n_groups, head_dim]);

    // Per-group min and max along dim 1 → [n_groups, 1]
    let group_min = flat.clone().min_dim(1);
    let group_max = flat.clone().max_dim(1);

    // Scale = (max - min) / 255, clamped to avoid division by zero
    // [n_groups, 1]
    let range = group_max.clone() - group_min.clone();
    let scale = range.clone() / 255.0;
    // Where range is 0, use scale=1 to avoid NaN (values will all quantize to 0)
    let scale = scale.clone().mask_fill(range.lower_equal_elem(0.0), 1.0f32);

    // Quantize: q = round(clamp((x - min) / scale, 0, 255))
    let normalized = (flat - group_min.clone()) / scale.clone();
    let quantized = normalized.clamp(0.0, 255.0);
    // Round to nearest integer, convert to int
    // burn doesn't have a direct round() on float tensors that gives Int,
    // so we add 0.5 and use int() which truncates.
    let q_int = quantized.add_scalar(0.5).int(); // [n_groups, head_dim] as Int

    // Pack 4 int8 values into each i32: v0 | (v1 << 8) | (v2 << 16) | (v3 << 24)
    // Reshape to [n_groups, packed_dim, 4]
    let q_grouped = q_int.reshape([n_groups, packed_dim, 4]);

    // Extract each of the 4 lanes
    let v0 = q_grouped
        .clone()
        .slice([0..n_groups, 0..packed_dim, 0..1])
        .reshape([n_groups, packed_dim]);
    let v1 = q_grouped
        .clone()
        .slice([0..n_groups, 0..packed_dim, 1..2])
        .reshape([n_groups, packed_dim]);
    let v2 = q_grouped
        .clone()
        .slice([0..n_groups, 0..packed_dim, 2..3])
        .reshape([n_groups, packed_dim]);
    let v3 = q_grouped
        .slice([0..n_groups, 0..packed_dim, 3..4])
        .reshape([n_groups, packed_dim]);

    let packed = v0
        .bitwise_or(v1.bitwise_left_shift_scalar(8))
        .bitwise_or(v2.bitwise_left_shift_scalar(16))
        .bitwise_or(v3.bitwise_left_shift_scalar(24));

    // Squeeze scale/zero to [n_groups]
    let scale_1d = scale.reshape([n_groups]);
    let zero_1d = group_min.reshape([n_groups]);

    GpuQuantTensor {
        packed,
        scale: scale_1d,
        zero: zero_1d,
        shape: [batch, n_kv_heads, seq_len, head_dim],
    }
}

/// Dequantize GPU-packed int8 tensor back to f32.
///
/// Output: `[batch, n_kv_heads, seq_len, head_dim]` as f32 tensor.
/// All operations happen on GPU via burn tensor ops.
pub fn gpu_dequantize<B: Backend<IntElem = i32>>(qt: &GpuQuantTensor<B>) -> Tensor<B, 4> {
    let [batch, n_kv_heads, seq_len, head_dim] = qt.shape;
    let n_groups = batch * n_kv_heads * seq_len;
    let packed_dim = head_dim / 4;

    // Unpack: extract 4 uint8 values from each i32
    let v0 = qt.packed.clone().bitwise_and_scalar(0xFF);
    let v1 = qt
        .packed
        .clone()
        .bitwise_right_shift_scalar(8)
        .bitwise_and_scalar(0xFF);
    let v2 = qt
        .packed
        .clone()
        .bitwise_right_shift_scalar(16)
        .bitwise_and_scalar(0xFF);
    let v3 = qt
        .packed
        .clone()
        .bitwise_right_shift_scalar(24)
        .bitwise_and_scalar(0xFF);

    // Interleave back to [n_groups, head_dim]
    // Stack along last dim: [n_groups, packed_dim, 4]
    let v0 = v0.reshape([n_groups, packed_dim, 1]);
    let v1 = v1.reshape([n_groups, packed_dim, 1]);
    let v2 = v2.reshape([n_groups, packed_dim, 1]);
    let v3 = v3.reshape([n_groups, packed_dim, 1]);
    let unpacked = Tensor::cat(vec![v0, v1, v2, v3], 2).reshape([n_groups, head_dim]);

    // Convert to float: dequantized = q * scale + zero
    let q_float = unpacked.float();
    let scale = qt.scale.clone().reshape([n_groups, 1]); // broadcast over head_dim
    let zero = qt.zero.clone().reshape([n_groups, 1]);

    let dequantized = q_float * scale + zero;

    dequantized.reshape([batch, n_kv_heads, seq_len, head_dim])
}

// ---------------------------------------------------------------------------
// GPU Quantized KV Cache
// ---------------------------------------------------------------------------

/// Per-layer KV cache stored as GPU-native packed int8 tensors.
///
/// Unlike `QuantizedKvCache` which moves data to CPU for quantization,
/// this keeps everything on GPU. Quantization and dequantization are
/// burn tensor ops executed as GPU shaders.
pub struct GpuQuantKvCache<B: Backend> {
    k: Option<GpuQuantTensor<B>>,
    v: Option<GpuQuantTensor<B>>,
}

impl<B: Backend> GpuQuantKvCache<B> {
    /// Create an empty GPU quantized cache.
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Current sequence length stored in cache (0 if empty).
    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(qt) => qt.shape[2],
            None => 0,
        }
    }

    /// Approximate memory usage in bytes for the cached K+V data.
    ///
    /// Packed data: n_groups * packed_dim * 4 bytes (i32 per packed group of 4 values)
    /// Scale: n_groups * 4 bytes (f32)
    /// Zero: n_groups * 4 bytes (f32)
    /// Total per tensor: n_groups * (packed_dim + 2) * 4
    pub fn memory_bytes(&self) -> usize {
        let per_tensor = |qt: &GpuQuantTensor<B>| {
            let [batch, n_kv_heads, seq_len, head_dim] = qt.shape;
            let n_groups = batch * n_kv_heads * seq_len;
            let packed_dim = head_dim / 4;
            (n_groups * packed_dim + 2 * n_groups) * 4
        };
        self.k.as_ref().map_or(0, per_tensor) + self.v.as_ref().map_or(0, per_tensor)
    }

    /// What the equivalent uncompressed f32 KV cache would use in bytes.
    pub fn uncompressed_bytes(&self) -> usize {
        match &self.k {
            Some(qt) => {
                let [batch, n_kv_heads, seq_len, head_dim] = qt.shape;
                2 * batch * n_kv_heads * seq_len * head_dim * 4
            }
            None => 0,
        }
    }
}

impl<B: Backend<IntElem = i32>> GpuQuantKvCache<B> {
    /// Append new K/V to the cache and return the full (dequantized) K/V.
    ///
    /// new_k, new_v: `[batch, n_kv_heads, new_len, head_dim]` in f32 on GPU.
    /// Returns `(full_k, full_v)` dequantized to f32 for attention.
    ///
    /// The hot path (new_len=1 during generation): quantize 1 new position,
    /// concatenate packed tensors on GPU, dequantize entire cache on GPU.
    /// No CPU involvement at any step.
    pub fn update(
        &mut self,
        new_k: Tensor<B, 4>,
        new_v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // Quantize the new K/V on GPU
        let new_k_q = gpu_quantize(new_k);
        let new_v_q = gpu_quantize(new_v);

        // Concatenate with existing cache (packed tensor cat on GPU)
        let full_k_q = match self.k.take() {
            Some(prev) => concat_gpu_quant(prev, new_k_q),
            None => new_k_q,
        };
        let full_v_q = match self.v.take() {
            Some(prev) => concat_gpu_quant(prev, new_v_q),
            None => new_v_q,
        };

        // Dequantize on GPU for attention
        let full_k = gpu_dequantize(&full_k_q);
        let full_v = gpu_dequantize(&full_v_q);

        // Store the quantized tensors
        self.k = Some(full_k_q);
        self.v = Some(full_v_q);

        (full_k, full_v)
    }
}

/// Concatenate two GPU quantized tensors along the sequence dimension.
///
/// Both must have the same [batch, n_kv_heads, *, head_dim] shape except seq_len.
/// The packed i32 data, scale, and zero tensors are concatenated on GPU.
fn concat_gpu_quant<B: Backend<IntElem = i32>>(
    a: GpuQuantTensor<B>,
    b: GpuQuantTensor<B>,
) -> GpuQuantTensor<B> {
    let [batch, n_kv_heads, seq_a, head_dim] = a.shape;
    let [_, _, seq_b, _] = b.shape;
    let total_seq = seq_a + seq_b;
    let packed_dim = head_dim / 4;

    // Packed data: reshape to [batch * n_kv_heads, seq, packed_dim], cat on seq dim,
    // then flatten back to [n_groups, packed_dim].
    let n_bh = batch * n_kv_heads;
    let a_packed = a.packed.reshape([n_bh, seq_a, packed_dim]);
    let b_packed = b.packed.reshape([n_bh, seq_b, packed_dim]);
    let cat_packed =
        Tensor::cat(vec![a_packed, b_packed], 1).reshape([n_bh * total_seq, packed_dim]);

    // Scale and zero: reshape to [batch * n_kv_heads, seq], cat, flatten.
    let a_scale = a.scale.reshape([n_bh, seq_a]);
    let b_scale = b.scale.reshape([n_bh, seq_b]);
    let cat_scale = Tensor::cat(vec![a_scale, b_scale], 1).reshape([n_bh * total_seq]);

    let a_zero = a.zero.reshape([n_bh, seq_a]);
    let b_zero = b.zero.reshape([n_bh, seq_b]);
    let cat_zero = Tensor::cat(vec![a_zero, b_zero], 1).reshape([n_bh * total_seq]);

    GpuQuantTensor {
        packed: cat_packed,
        scale: cat_scale,
        zero: cat_zero,
        shape: [batch, n_kv_heads, total_seq, head_dim],
    }
}

// ---------------------------------------------------------------------------
// Layer caches collection
// ---------------------------------------------------------------------------

/// Collection of GPU quantized KV caches for all layers.
pub struct GpuQuantLayerCaches<B: Backend> {
    pub caches: Vec<GpuQuantKvCache<B>>,
}

impl<B: Backend> GpuQuantLayerCaches<B> {
    /// Create empty GPU quant caches for `n_layers` transformer layers.
    pub fn new(n_layers: usize) -> Self {
        Self {
            caches: (0..n_layers).map(|_| GpuQuantKvCache::new()).collect(),
        }
    }

    /// Total memory usage across all layers in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.caches.iter().map(|c| c.memory_bytes()).sum()
    }

    /// Total uncompressed equivalent in bytes.
    pub fn uncompressed_bytes(&self) -> usize {
        self.caches.iter().map(|c| c.uncompressed_bytes()).sum()
    }
}

// ===========================================================================
// GPU TurboQuant: near-optimal vector quantization entirely on GPU
// ===========================================================================
//
// Ports the three TurboQuant components from CPU (turbo_quant.rs) to burn
// tensor operations that compile to GPU shaders:
//
//   1. Randomized Hadamard Transform (RHT) — decorrelates coordinates
//   2. Lloyd-Max optimal scalar quantization — exploits Gaussian statistics
//   3. QJL binary residual sketch (optional) — unbiased inner product fix
//
// Storage: packed i32 tensors on GPU (same approach as GpuQuantTensor above).
// Per-group sigma and norm stored as float tensors on GPU.

use crate::models::gemma::turbo_quant::TurboQuantConfig;

// ---------------------------------------------------------------------------
// Precomputed context for GPU TurboQuant operations
// ---------------------------------------------------------------------------

/// Precomputed GPU tensors for TurboQuant: Rademacher signs, Lloyd-Max tables,
/// and optional QJL projection matrix. Created once, reused across all
/// quantizations for a given (head_dim, seed, device).
#[allow(dead_code)]
pub struct GpuTurboQuantCtx<B: Backend> {
    /// Rademacher +/-1 signs for the diagonal D matrix, shape [padded_dim].
    signs: Tensor<B, 1>,
    /// Original (unpadded) dimension.
    dim: usize,
    /// Padded to next power of 2 (needed for Hadamard butterfly).
    padded_dim: usize,
    /// 1/sqrt(padded_dim) normalization factor.
    norm: f64,
    /// Lloyd-Max decision boundaries for the MSE stage, shape [n_boundaries].
    boundaries: Tensor<B, 1>,
    /// Lloyd-Max reconstruction centroids for the MSE stage, shape [n_levels].
    centroids: Tensor<B, 1>,
    /// Number of MSE quantization bits.
    mse_bits: usize,
    /// Number of quantization levels = 2^mse_bits.
    n_levels: usize,
    /// Optional QJL projection matrix: random +/-1 values, shape [m, dim].
    qjl_signs: Option<Tensor<B, 2>>,
    /// Number of QJL sketch bits (0 if disabled).
    residual_bits: usize,
    /// Full config.
    config: TurboQuantConfig,
}

/// Simple xorshift64 PRNG (matches turbo_quant.rs for reproducibility).
fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Lloyd-Max centroids for N(0,1) — same tables as turbo_quant.rs.
fn lloyd_max_centroids_f32(bits: usize) -> Vec<f32> {
    match bits {
        1 => vec![-0.7978846, 0.7978846],
        2 => vec![-1.510, -0.4528, 0.4528, 1.510],
        3 => vec![
            -2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519,
        ],
        4 => vec![
            -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881,
            0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
        ],
        _ => panic!("Lloyd-Max tables only for 1..=4 bits, got {}", bits),
    }
}

fn lloyd_max_boundaries_f32(bits: usize) -> Vec<f32> {
    match bits {
        1 => vec![0.0],
        2 => vec![-0.9816, 0.0, 0.9816],
        3 => vec![-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480],
        4 => vec![
            -2.4008, -1.8435, -1.4372, -1.0993, -0.7996, -0.5224, -0.2582, 0.0, 0.2582, 0.5224,
            0.7996, 1.0993, 1.4372, 1.8435, 2.4008,
        ],
        _ => panic!("Lloyd-Max tables only for 1..=4 bits, got {}", bits),
    }
}

/// Next power of 2 >= n.
fn next_pow2(n: usize) -> usize {
    if n.is_power_of_two() {
        return n;
    }
    let mut v = n;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v + 1
}

impl<B: Backend<IntElem = i32>> GpuTurboQuantCtx<B> {
    /// Create a new GPU TurboQuant context. Precomputes all needed tensors on device.
    pub fn new(config: &TurboQuantConfig, device: &B::Device) -> Self {
        let dim = config.head_dim;
        let padded_dim = next_pow2(dim);
        let norm = 1.0 / (padded_dim as f64).sqrt();
        let mse_bits = if config.residual_bits > 0 {
            config.bits - 1
        } else {
            config.bits
        };
        let n_levels = 1 << mse_bits;

        // Generate Rademacher signs using same xorshift64 as CPU version
        let mut state = config.seed;
        let signs_data: Vec<f32> = (0..padded_dim)
            .map(|_| {
                state = xorshift64(state);
                if state & 1 == 0 { 1.0f32 } else { -1.0f32 }
            })
            .collect();
        let signs = Tensor::<B, 1>::from_floats(&signs_data[..], device);

        // Lloyd-Max tables
        let boundaries_data = lloyd_max_boundaries_f32(mse_bits);
        let centroids_data = lloyd_max_centroids_f32(mse_bits);
        let boundaries = Tensor::<B, 1>::from_floats(&boundaries_data[..], device);
        let centroids = Tensor::<B, 1>::from_floats(&centroids_data[..], device);

        // Optional QJL projection matrix
        let qjl_signs = if config.residual_bits > 0 {
            let m = config.residual_bits;
            let mut qjl_state = config.seed.wrapping_add(1);
            let qjl_data: Vec<f32> = (0..m * dim)
                .map(|_| {
                    qjl_state = xorshift64(qjl_state);
                    if qjl_state & 1 == 0 { 1.0f32 } else { -1.0f32 }
                })
                .collect();
            Some(Tensor::<B, 1>::from_floats(&qjl_data[..], device).reshape([m, dim]))
        } else {
            None
        };

        Self {
            signs,
            dim,
            padded_dim,
            norm,
            boundaries,
            centroids,
            mse_bits,
            n_levels,
            qjl_signs,
            residual_bits: config.residual_bits,
            config: config.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Hadamard rotation via tensor ops
    // -----------------------------------------------------------------------
    //
    // The Walsh-Hadamard butterfly is: for each stage s=0..log2(n)-1,
    //   for pairs (i, i+half) in strided groups:
    //     a' = a + b
    //     b' = a - b
    //
    // On a GPU tensor of shape [n_groups, padded_dim]:
    //   1. Reshape to [n_groups, n_blocks, 2] where n_blocks = padded_dim/(2*1)
    //   2. Compute sum and diff along the last dimension
    //   3. Reshape back
    //   Repeat with increasing block sizes.
    //
    // More precisely, at stage s (half = 2^s):
    //   Reshape [n_groups, padded_dim] -> [n_groups, padded_dim/(2*half), 2, half]
    //   The dim=2 axis separates the "a" block and "b" block.
    //   a = slice(:, :, 0, :), b = slice(:, :, 1, :)
    //   result(:, :, 0, :) = a + b, result(:, :, 1, :) = a - b
    //   Reshape back to [n_groups, padded_dim].

    /// Apply randomized Hadamard rotation to vectors.
    /// Input: [n_groups, dim] (unpadded). Output: [n_groups, dim] (unpadded).
    ///
    /// Steps: sign flip -> zero-pad to padded_dim -> butterfly -> normalize -> truncate.
    fn rotate(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n_groups, d] = x.dims();
        debug_assert_eq!(d, self.dim);

        // 1. Sign flip: x * signs[:dim]
        let signs_d = self.signs.clone().slice([0..self.dim]);
        let flipped = x * signs_d.unsqueeze::<2>(); // broadcast [1, dim] over [n_groups, dim]

        // 2. Zero-pad to padded_dim if needed
        let buf = if self.padded_dim > self.dim {
            let pad_size = self.padded_dim - self.dim;
            let zeros = Tensor::<B, 2>::zeros([n_groups, pad_size], &flipped.device());
            Tensor::cat(vec![flipped, zeros], 1) // [n_groups, padded_dim]
        } else {
            flipped
        };

        // 3. Walsh-Hadamard butterfly: log2(padded_dim) stages
        let buf = self.hadamard_butterfly(buf, n_groups);

        // 4. Normalize by 1/sqrt(padded_dim)
        let buf = buf * self.norm;

        // 5. Truncate back to original dim
        if self.padded_dim > self.dim {
            buf.slice([0..n_groups, 0..self.dim])
        } else {
            buf
        }
    }

    /// Apply inverse randomized Hadamard rotation.
    /// Input: [n_groups, dim] (unpadded). Output: [n_groups, dim] (unpadded).
    ///
    /// Since R = (1/sqrt(d)) * H * D and R is orthogonal:
    ///   R^{-1} = R^T = D * (1/sqrt(d)) * H
    /// So: inverse = D * normalize * H * zero_pad(y)
    fn rotate_inv(&self, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n_groups, d] = y.dims();
        debug_assert_eq!(d, self.dim);

        // 1. Zero-pad to padded_dim
        let buf = if self.padded_dim > self.dim {
            let pad_size = self.padded_dim - self.dim;
            let zeros = Tensor::<B, 2>::zeros([n_groups, pad_size], &y.device());
            Tensor::cat(vec![y, zeros], 1)
        } else {
            y
        };

        // 2. Hadamard butterfly
        let buf = self.hadamard_butterfly(buf, n_groups);

        // 3. Normalize by 1/sqrt(padded_dim) and multiply by signs (D)
        let signs_trunc = if self.padded_dim > self.dim {
            self.signs.clone().slice([0..self.dim])
        } else {
            self.signs.clone()
        };

        // Truncate first, then apply D and normalize
        let buf = if self.padded_dim > self.dim {
            buf.slice([0..n_groups, 0..self.dim])
        } else {
            buf
        };

        buf * self.norm * signs_trunc.unsqueeze::<2>()
    }

    /// Walsh-Hadamard butterfly on tensor of shape [n_groups, padded_dim].
    /// Performs log2(padded_dim) stages in place using tensor reshapes and arithmetic.
    fn hadamard_butterfly(&self, mut buf: Tensor<B, 2>, n_groups: usize) -> Tensor<B, 2> {
        let n = self.padded_dim;
        let n_stages = (n as f64).log2() as usize;

        for s in 0..n_stages {
            let half = 1 << s;
            let n_blocks = n / (2 * half);

            // Reshape to [n_groups, n_blocks, 2, half]
            let reshaped = buf.reshape([n_groups, n_blocks, 2, half]);

            // Extract a and b: the two halves along dim 2
            let a = reshaped
                .clone()
                .slice([0..n_groups, 0..n_blocks, 0..1, 0..half])
                .reshape([n_groups, n_blocks, half]);
            let b = reshaped
                .slice([0..n_groups, 0..n_blocks, 1..2, 0..half])
                .reshape([n_groups, n_blocks, half]);

            // Butterfly: a' = a + b, b' = a - b
            let sum = a.clone() + b.clone();
            let diff = a - b;

            // Interleave back: [n_groups, n_blocks, 2, half] -> [n_groups, padded_dim]
            let sum = sum.reshape([n_groups, n_blocks, 1, half]);
            let diff = diff.reshape([n_groups, n_blocks, 1, half]);
            buf = Tensor::cat(vec![sum, diff], 2).reshape([n_groups, n]);
        }

        buf
    }

    // -----------------------------------------------------------------------
    // Lloyd-Max quantization via tensor ops
    // -----------------------------------------------------------------------
    //
    // For each element x (normalized to N(0,1) by dividing by per-group sigma):
    //   bin_index = sum(x > boundary[i] for i in 0..n_boundaries)
    //
    // This is a broadcast comparison:
    //   x_expanded: [n_groups, dim, 1]
    //   boundaries: [1, 1, n_boundaries]
    //   comparison: [n_groups, dim, n_boundaries] of bool
    //   bin_index = sum(comparison, dim=2) -> [n_groups, dim] of int
    //
    // Dequantize: gather centroids by index -> multiply by sigma.

    /// Quantize: returns (indices as int tensor [n_groups, dim], sigma [n_groups]).
    fn lloyd_max_quantize(&self, rotated: Tensor<B, 2>) -> (Tensor<B, 2, Int>, Tensor<B, 1>) {
        let [n_groups, dim] = rotated.dims();

        // Estimate per-group sigma = sqrt(var)
        // var = mean(x^2) - mean(x)^2  (but for rotated unit vectors, mean ~ 0)
        let sq = rotated.clone() * rotated.clone();
        let var = sq.mean_dim(1); // [n_groups, 1]
        let sigma = var.clone().sqrt().clamp(1e-10, f32::MAX); // [n_groups, 1]

        // Normalize: x / sigma -> approximately N(0, 1)
        let x_norm = rotated / sigma.clone(); // broadcast [n_groups, dim] / [n_groups, 1]

        // Quantize: for each element, count how many boundaries it exceeds
        // x_expanded: [n_groups, dim, 1]
        // boundaries: [1, 1, n_boundaries]
        let n_boundaries = self.n_levels - 1;
        let x_expanded = x_norm.reshape([n_groups, dim, 1]);
        let boundaries = self.boundaries.clone().reshape([1, 1, n_boundaries]);

        // Compare: result is float 0.0 or 1.0 (burn doesn't have bool->int directly,
        // but greater(x, b) returns a bool tensor; we need to convert)
        // Use: (x > boundary) as float, then sum -> int
        // Approach: sign(x - boundary + epsilon) converted to 0/1, sum along boundaries dim
        //
        // Actually, burn has Tensor::greater_elem and similar, but for broadcast comparison
        // we compute it via arithmetic:
        //   mask = (x_expanded - boundaries) as sign, clamp to 0..1
        let diff = x_expanded - boundaries; // [n_groups, dim, n_boundaries]
        // step function: 1 where diff > 0, 0 otherwise
        // Use: (sign(diff) + 1) / 2, clamped. sign gives -1, 0, 1.
        // For diff=0, we want bin index to round up (match CPU: boundary hit -> upper bin)
        // So use >= 0: (sign(diff + epsilon) + 1) / 2
        let step = diff.add_scalar(1e-7).sign().add_scalar(1.0).div_scalar(2.0);
        // Sum along boundaries dimension to get bin index
        let bin_index = step.sum_dim(2).reshape([n_groups, dim]); // float in [0, n_boundaries]
        let bin_index = bin_index.clamp(0.0, (self.n_levels - 1) as f32);
        // Convert to int (add 0.5 and truncate to handle float rounding)
        let bin_index_int = bin_index.add_scalar(0.5).int(); // [n_groups, dim]

        let sigma_1d = sigma.reshape([n_groups]);
        (bin_index_int, sigma_1d)
    }

    /// Dequantize: bin indices + sigma -> reconstructed rotated values.
    /// indices: [n_groups, dim] int tensor, sigma: [n_groups] float tensor.
    /// Returns [n_groups, dim] float tensor of reconstructed rotated values.
    fn lloyd_max_dequantize(
        &self,
        indices: Tensor<B, 2, Int>,
        sigma: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let [n_groups, dim] = indices.dims();

        // Gather centroids: indices -> centroid values
        // centroids: [n_levels]. We need to index with [n_groups, dim] int tensor.
        //
        // Approach: one-hot encode indices, matmul with centroids vector.
        // one_hot: [n_groups * dim, n_levels], centroids: [n_levels, 1]
        // result = one_hot @ centroids -> [n_groups * dim, 1] -> reshape
        //
        // Alternative: use the indices as float, build a lookup via broadcast.
        // For small n_levels (2-16), broadcast is fine:
        //   For each level l, mask = (indices == l), centroid_l * mask, sum all levels.
        //
        // Most efficient for small n_levels: direct broadcast gather.
        let flat_idx = indices.reshape([n_groups * dim]); // [n_groups * dim]
        let flat_float = flat_idx.float(); // [n_groups * dim]

        // Build lookup: for each possible level, check if index matches
        let mut result = Tensor::<B, 1>::zeros([n_groups * dim], &sigma.device());
        let centroids_data = lloyd_max_centroids_f32(self.mse_bits);
        for l in 0..self.n_levels {
            let centroid_val = centroids_data[l];
            // mask: 1 where index == l, 0 elsewhere
            // Compute as: 1 - min(|idx - l|, 1)  (exact for integer indices)
            let diff = flat_float.clone().sub_scalar(l as f32).abs();
            let mask = diff.neg().add_scalar(1.0).clamp(0.0, 1.0); // 1 if diff==0, 0 if diff>=1
            result = result + mask.mul_scalar(centroid_val);
        }

        let values = result.reshape([n_groups, dim]);

        // Scale by sigma: values * sigma
        let sigma_2d = sigma.reshape([n_groups, 1]); // broadcast over dim
        values * sigma_2d
    }

    // -----------------------------------------------------------------------
    // Packing: sub-byte indices into i32 tensors
    // -----------------------------------------------------------------------
    //
    // For mse_bits in {1, 2, 3, 4}:
    //   - 1-bit: 32 values per i32
    //   - 2-bit: 16 values per i32
    //   - 3-bit: 10 values per i32 (with 2 wasted bits)
    //   - 4-bit: 8 values per i32
    //
    // For simplicity and GPU efficiency, we pack to a fixed number of values
    // per i32 (rounding down), potentially wasting a few bits. This avoids
    // complex cross-word bit shifting on GPU.
    //
    // For 3-bit: we pack 10 values per i32 (30 bits used, 2 wasted).
    // For dim=128, that's 13 i32s per group (130 slots, 128 used).

    /// Pack quantization indices into i32 tensors.
    /// indices: [n_groups, dim] int tensor (values in 0..n_levels).
    /// Returns packed: [n_groups, packed_dim] int tensor.
    fn pack_indices(&self, indices: Tensor<B, 2, Int>) -> (Tensor<B, 2, Int>, usize) {
        let [n_groups, dim] = indices.dims();
        let bits = self.mse_bits;

        // Values per i32 word
        let vals_per_word = 32 / bits; // 1->32, 2->16, 3->10, 4->8
        let packed_dim = (dim + vals_per_word - 1) / vals_per_word;

        // Pad dim to multiple of vals_per_word if needed
        let padded_dim = packed_dim * vals_per_word;
        let indices = if padded_dim > dim {
            let padding =
                Tensor::<B, 2, Int>::zeros([n_groups, padded_dim - dim], &indices.device());
            Tensor::cat(vec![indices, padding], 1)
        } else {
            indices
        };

        // Reshape to [n_groups, packed_dim, vals_per_word]
        let grouped = indices.reshape([n_groups, packed_dim, vals_per_word]);

        // Pack: shift each lane by (lane_index * bits) and OR together
        let mut packed = grouped
            .clone()
            .slice([0..n_groups, 0..packed_dim, 0..1])
            .reshape([n_groups, packed_dim]);

        for lane in 1..vals_per_word {
            let lane_vals = grouped
                .clone()
                .slice([0..n_groups, 0..packed_dim, lane..(lane + 1)])
                .reshape([n_groups, packed_dim]);
            let shift = (lane * bits) as i32;
            packed = packed.bitwise_or(lane_vals.bitwise_left_shift_scalar(shift));
        }

        (packed, packed_dim)
    }

    /// Unpack i32 tensor back to per-element indices.
    /// packed: [n_groups, packed_dim] int tensor.
    /// Returns [n_groups, dim] int tensor.
    fn unpack_indices(&self, packed: Tensor<B, 2, Int>, dim: usize) -> Tensor<B, 2, Int> {
        let [n_groups, packed_dim] = packed.dims();
        let bits = self.mse_bits;
        let vals_per_word = 32 / bits;
        let mask = (1i32 << bits) - 1;

        // Extract each lane
        let mut lanes: Vec<Tensor<B, 3, Int>> = Vec::with_capacity(vals_per_word);
        for lane in 0..vals_per_word {
            let shift = (lane * bits) as i32;
            let lane_vals = packed
                .clone()
                .bitwise_right_shift_scalar(shift)
                .bitwise_and_scalar(mask);
            lanes.push(lane_vals.reshape([n_groups, packed_dim, 1]));
        }

        let interleaved = Tensor::cat(lanes, 2).reshape([n_groups, packed_dim * vals_per_word]);

        // Truncate to original dim (remove padding)
        if packed_dim * vals_per_word > dim {
            interleaved.slice([0..n_groups, 0..dim])
        } else {
            interleaved
        }
    }

    // -----------------------------------------------------------------------
    // Full quantize / dequantize
    // -----------------------------------------------------------------------

    /// Quantize a 4D f32 tensor using GPU TurboQuant.
    ///
    /// Input: [batch, n_kv_heads, seq_len, head_dim].
    /// Returns a GpuTurboQuantTensor with all data on GPU.
    pub fn quantize(&self, input: Tensor<B, 4>) -> GpuTurboQuantTensor<B> {
        let [batch, n_kv_heads, seq_len, head_dim] = input.dims();
        assert_eq!(
            head_dim, self.dim,
            "head_dim mismatch: expected {}, got {}",
            self.dim, head_dim
        );
        let n_groups = batch * n_kv_heads * seq_len;

        // Flatten to [n_groups, dim]
        let flat = input.reshape([n_groups, head_dim]);

        // 1. Compute per-group L2 norm
        let norms = (flat.clone() * flat.clone()).sum_dim(1).sqrt(); // [n_groups, 1]

        // 2. Normalize to unit sphere
        let inv_norms = norms.clone().clamp(1e-12, f32::MAX).recip(); // [n_groups, 1]
        let normalized = flat * inv_norms; // [n_groups, dim]

        // 3. Rotate (RHT)
        let rotated = self.rotate(normalized.clone());

        // 4. Lloyd-Max quantize
        let (indices, sigma) = self.lloyd_max_quantize(rotated);

        // 5. Pack indices
        let (packed, packed_dim) = self.pack_indices(indices.clone());

        // 6. Optional QJL residual sketch
        let (qjl_packed, qjl_residual_norms) = if let Some(ref qjl_matrix) = self.qjl_signs {
            // Dequantize the MSE approximation in rotated space
            let deq_rotated = self.lloyd_max_dequantize(indices, sigma.clone());
            // Inverse rotate to get back to normalized space
            let deq_normalized = self.rotate_inv(deq_rotated);
            // Compute residual
            let residual = normalized - deq_normalized; // [n_groups, dim]
            // Compute residual norms
            let rnorms = (residual.clone() * residual.clone())
                .sum_dim(1)
                .sqrt()
                .reshape([n_groups]); // [n_groups]

            // Project: projected = residual @ qjl_matrix^T -> [n_groups, m]
            let projected = residual.matmul(qjl_matrix.clone().transpose()); // [n_groups, m]
            // Store sign bits: 1 where projected >= 0, 0 elsewhere
            // Pack into i32: 32 sign bits per i32
            let sign_float = projected.sign().add_scalar(1.0).div_scalar(2.0); // 0 or 1
            let sign_int = sign_float.add_scalar(0.5).int(); // [n_groups, m]

            let m = self.residual_bits;
            let qjl_packed_dim = (m + 31) / 32;
            let padded_m = qjl_packed_dim * 32;

            let sign_padded = if padded_m > m {
                let padding =
                    Tensor::<B, 2, Int>::zeros([n_groups, padded_m - m], &packed.device());
                Tensor::cat(vec![sign_int, padding], 1)
            } else {
                sign_int
            };

            let sign_grouped = sign_padded.reshape([n_groups, qjl_packed_dim, 32]);
            let mut qjl_packed = sign_grouped
                .clone()
                .slice([0..n_groups, 0..qjl_packed_dim, 0..1])
                .reshape([n_groups, qjl_packed_dim]);
            for bit in 1..32 {
                let lane = sign_grouped
                    .clone()
                    .slice([0..n_groups, 0..qjl_packed_dim, bit..(bit + 1)])
                    .reshape([n_groups, qjl_packed_dim]);
                qjl_packed = qjl_packed.bitwise_or(lane.bitwise_left_shift_scalar(bit as i32));
            }

            (Some(qjl_packed), Some(rnorms))
        } else {
            (None, None)
        };

        let norms_1d = norms.reshape([n_groups]);

        GpuTurboQuantTensor {
            packed,
            packed_dim,
            sigma,
            norms: norms_1d,
            qjl_packed,
            qjl_residual_norms,
            shape: [batch, n_kv_heads, seq_len, head_dim],
        }
    }

    /// Dequantize a GpuTurboQuantTensor back to f32.
    ///
    /// Output: [batch, n_kv_heads, seq_len, head_dim] as f32 tensor on GPU.
    pub fn dequantize(&self, qt: &GpuTurboQuantTensor<B>) -> Tensor<B, 4> {
        let [batch, n_kv_heads, seq_len, head_dim] = qt.shape;
        let n_groups = batch * n_kv_heads * seq_len;

        // 1. Unpack indices
        let indices = self.unpack_indices(qt.packed.clone(), head_dim);

        // 2. Lloyd-Max dequantize (centroid lookup * sigma)
        let deq_rotated = self.lloyd_max_dequantize(indices, qt.sigma.clone());

        // 3. Inverse rotate
        let mut deq_normalized = self.rotate_inv(deq_rotated);

        // 4. Optional QJL correction
        if let (Some(qjl_matrix), Some(qjl_packed), Some(rnorms)) =
            (&self.qjl_signs, &qt.qjl_packed, &qt.qjl_residual_norms)
        {
            let m = self.residual_bits;
            let qjl_packed_dim = (m + 31) / 32;

            // Unpack sign bits from i32
            let mut sign_lanes: Vec<Tensor<B, 3, Int>> = Vec::with_capacity(32);
            for bit in 0..32 {
                let lane = qjl_packed
                    .clone()
                    .bitwise_right_shift_scalar(bit as i32)
                    .bitwise_and_scalar(1);
                sign_lanes.push(lane.reshape([n_groups, qjl_packed_dim, 1]));
            }
            let sign_unpacked = Tensor::cat(sign_lanes, 2).reshape([n_groups, qjl_packed_dim * 32]);
            // Truncate to m
            let sign_bits = if qjl_packed_dim * 32 > m {
                sign_unpacked.slice([0..n_groups, 0..m])
            } else {
                sign_unpacked
            };
            // Convert 0/1 to -1/+1
            let sign_vals = sign_bits.float().mul_scalar(2.0).sub_scalar(1.0); // [n_groups, m]

            // Correction: coeff * sign_vals @ qjl_matrix -> [n_groups, dim]
            // coeff = sqrt(pi/2) / m * gamma (per group)
            let pi_over_2_sqrt = (std::f64::consts::PI / 2.0).sqrt() as f32;
            let coeff = rnorms.clone().mul_scalar(pi_over_2_sqrt / m as f32); // [n_groups]
            let correction = sign_vals.matmul(qjl_matrix.clone()); // [n_groups, dim]
            let correction = correction * coeff.reshape([n_groups, 1]); // broadcast

            deq_normalized = deq_normalized + correction;
        }

        // 5. Rescale by original norm
        let norms_2d = qt.norms.clone().reshape([n_groups, 1]); // broadcast over dim
        let result = deq_normalized * norms_2d;

        result.reshape([batch, n_kv_heads, seq_len, head_dim])
    }
}

// ---------------------------------------------------------------------------
// GPU TurboQuant tensor storage
// ---------------------------------------------------------------------------

/// Quantized tensor stored entirely on GPU using TurboQuant algorithm.
///
/// `packed` holds [n_groups, packed_dim] where each i32 packs multiple
/// sub-byte quantization indices (e.g., 10x 3-bit values per i32).
pub struct GpuTurboQuantTensor<B: Backend> {
    /// Packed quantization indices: [n_groups, packed_dim] as Int tensor.
    pub packed: Tensor<B, 2, Int>,
    /// Number of i32 words per group.
    pub packed_dim: usize,
    /// Per-group sigma (std of rotated coordinates): [n_groups].
    pub sigma: Tensor<B, 1>,
    /// Per-group L2 norm of original vector: [n_groups].
    pub norms: Tensor<B, 1>,
    /// Optional QJL packed sign bits: [n_groups, qjl_packed_dim].
    pub qjl_packed: Option<Tensor<B, 2, Int>>,
    /// Optional per-group residual L2 norm for QJL: [n_groups].
    pub qjl_residual_norms: Option<Tensor<B, 1>>,
    /// Original shape: [batch, n_kv_heads, seq_len, head_dim].
    pub shape: [usize; 4],
}

// ---------------------------------------------------------------------------
// GPU TurboQuant KV Cache
// ---------------------------------------------------------------------------

/// Per-layer KV cache stored as GPU-native TurboQuant tensors.
///
/// Unlike `TurboQuantKvCache` (CPU), this keeps everything on GPU.
/// Rotation, quantization, packing, and their inverses are all burn
/// tensor operations executed as GPU shaders.
pub struct GpuTurboQuantKvCache<B: Backend> {
    k: Option<GpuTurboQuantTensor<B>>,
    v: Option<GpuTurboQuantTensor<B>>,
    ctx: GpuTurboQuantCtx<B>,
}

impl<B: Backend<IntElem = i32>> GpuTurboQuantKvCache<B> {
    /// Create an empty GPU TurboQuant cache.
    pub fn new(config: &TurboQuantConfig, device: &B::Device) -> Self {
        let ctx = GpuTurboQuantCtx::new(config, device);
        Self {
            k: None,
            v: None,
            ctx,
        }
    }

    /// Current sequence length stored in cache (0 if empty).
    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(qt) => qt.shape[2],
            None => 0,
        }
    }

    /// Approximate memory usage in bytes for cached K+V data.
    pub fn memory_bytes(&self) -> usize {
        let per_tensor = |qt: &GpuTurboQuantTensor<B>| {
            let n_groups = qt.sigma.dims()[0];
            let packed_bytes = n_groups * qt.packed_dim * 4; // i32 per packed word
            let sigma_bytes = n_groups * 4; // f32
            let norm_bytes = n_groups * 4; // f32
            let qjl_bytes = qt
                .qjl_packed
                .as_ref()
                .map_or(0, |p| p.dims()[0] * p.dims()[1] * 4);
            let rnorm_bytes = qt
                .qjl_residual_norms
                .as_ref()
                .map_or(0, |r| r.dims()[0] * 4);
            packed_bytes + sigma_bytes + norm_bytes + qjl_bytes + rnorm_bytes
        };
        self.k.as_ref().map_or(0, per_tensor) + self.v.as_ref().map_or(0, per_tensor)
    }

    /// What the equivalent uncompressed f32 KV cache would use in bytes.
    pub fn uncompressed_bytes(&self) -> usize {
        match &self.k {
            Some(qt) => {
                let [batch, n_kv_heads, seq_len, head_dim] = qt.shape;
                2 * batch * n_kv_heads * seq_len * head_dim * 4
            }
            None => 0,
        }
    }

    /// Append new K/V to the cache and return the full (dequantized) K/V.
    ///
    /// new_k, new_v: [batch, n_kv_heads, new_len, head_dim] in f32 on GPU.
    /// Returns (full_k, full_v) dequantized to f32 for attention.
    ///
    /// Strategy: quantize new tokens, concatenate packed tensors on GPU,
    /// dequantize everything. No CPU involvement.
    pub fn update(
        &mut self,
        new_k: Tensor<B, 4>,
        new_v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let new_k_q = self.ctx.quantize(new_k);
        let new_v_q = self.ctx.quantize(new_v);

        let full_k_q = match self.k.take() {
            Some(prev) => concat_gpu_turbo_quant(prev, new_k_q),
            None => new_k_q,
        };
        let full_v_q = match self.v.take() {
            Some(prev) => concat_gpu_turbo_quant(prev, new_v_q),
            None => new_v_q,
        };

        let full_k = self.ctx.dequantize(&full_k_q);
        let full_v = self.ctx.dequantize(&full_v_q);

        self.k = Some(full_k_q);
        self.v = Some(full_v_q);

        (full_k, full_v)
    }
}

/// Concatenate two GPU TurboQuant tensors along the sequence dimension.
fn concat_gpu_turbo_quant<B: Backend<IntElem = i32>>(
    a: GpuTurboQuantTensor<B>,
    b: GpuTurboQuantTensor<B>,
) -> GpuTurboQuantTensor<B> {
    let [batch, n_kv_heads, seq_a, head_dim] = a.shape;
    let [_, _, seq_b, _] = b.shape;
    let total_seq = seq_a + seq_b;

    let n_bh = batch * n_kv_heads;
    let packed_dim = a.packed_dim;

    // Concatenate packed data along seq dimension
    let a_packed = a.packed.reshape([n_bh, seq_a, packed_dim]);
    let b_packed = b.packed.reshape([n_bh, seq_b, packed_dim]);
    let cat_packed =
        Tensor::cat(vec![a_packed, b_packed], 1).reshape([n_bh * total_seq, packed_dim]);

    // Concatenate sigma
    let a_sigma = a.sigma.reshape([n_bh, seq_a]);
    let b_sigma = b.sigma.reshape([n_bh, seq_b]);
    let cat_sigma = Tensor::cat(vec![a_sigma, b_sigma], 1).reshape([n_bh * total_seq]);

    // Concatenate norms
    let a_norms = a.norms.reshape([n_bh, seq_a]);
    let b_norms = b.norms.reshape([n_bh, seq_b]);
    let cat_norms = Tensor::cat(vec![a_norms, b_norms], 1).reshape([n_bh * total_seq]);

    // Concatenate optional QJL data
    let cat_qjl_packed = match (a.qjl_packed, b.qjl_packed) {
        (Some(a_qjl), Some(b_qjl)) => {
            let qjl_packed_dim = a_qjl.dims()[1];
            let a_qjl = a_qjl.reshape([n_bh, seq_a, qjl_packed_dim]);
            let b_qjl = b_qjl.reshape([n_bh, seq_b, qjl_packed_dim]);
            Some(Tensor::cat(vec![a_qjl, b_qjl], 1).reshape([n_bh * total_seq, qjl_packed_dim]))
        }
        _ => None,
    };

    let cat_qjl_rnorms = match (a.qjl_residual_norms, b.qjl_residual_norms) {
        (Some(a_rn), Some(b_rn)) => {
            let a_rn = a_rn.reshape([n_bh, seq_a]);
            let b_rn = b_rn.reshape([n_bh, seq_b]);
            Some(Tensor::cat(vec![a_rn, b_rn], 1).reshape([n_bh * total_seq]))
        }
        _ => None,
    };

    GpuTurboQuantTensor {
        packed: cat_packed,
        packed_dim,
        sigma: cat_sigma,
        norms: cat_norms,
        qjl_packed: cat_qjl_packed,
        qjl_residual_norms: cat_qjl_rnorms,
        shape: [batch, n_kv_heads, total_seq, head_dim],
    }
}

// ---------------------------------------------------------------------------
// Layer caches collection for GPU TurboQuant
// ---------------------------------------------------------------------------

/// Collection of GPU TurboQuant KV caches for all layers.
pub struct GpuTurboQuantLayerCaches<B: Backend> {
    pub caches: Vec<GpuTurboQuantKvCache<B>>,
}

impl<B: Backend<IntElem = i32>> GpuTurboQuantLayerCaches<B> {
    /// Create empty GPU TurboQuant caches for `n_layers` transformer layers.
    pub fn new(n_layers: usize, config: &TurboQuantConfig, device: &B::Device) -> Self {
        Self {
            caches: (0..n_layers)
                .map(|_| GpuTurboQuantKvCache::new(config, device))
                .collect(),
        }
    }

    /// Total memory usage across all layers in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.caches.iter().map(|c| c.memory_bytes()).sum()
    }

    /// Total uncompressed equivalent in bytes.
    pub fn uncompressed_bytes(&self) -> usize {
        self.caches.iter().map(|c| c.uncompressed_bytes()).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    // Wgpu uses i32 for IntElem — matches the packing requirement.
    type B = Wgpu;

    #[test]
    fn test_gpu_quant_roundtrip() {
        let device = WgpuDevice::default();

        // [1, 2, 3, 4] — batch=1, 2 heads, 3 positions, head_dim=4
        let data: Vec<f32> = vec![
            // head 0
            -1.0, 0.5, 2.0, -0.3, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // head 1
            -10.0, 10.0, 5.0, -5.0, 0.1, 0.2, 0.3, 0.4, 100.0, -100.0, 0.0, 50.0,
        ];

        let input = Tensor::<B, 1>::from_floats(&data[..], &device).reshape([1, 2, 3, 4]);

        let qt = gpu_quantize(input.clone());
        let output = gpu_dequantize(&qt);

        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

        assert_eq!(input_data.len(), output_data.len());

        for (orig, recon) in input_data.iter().zip(output_data.iter()) {
            let err = (orig - recon).abs();
            // Int8: max error per group is range/255. Widest range is [-100, 100] → step ~0.78
            assert!(
                err < 1.0,
                "Roundtrip error too large: orig={}, recon={}, err={}",
                orig,
                recon,
                err
            );
        }

        // All-zeros group should roundtrip exactly (or near-exactly)
        for i in 4..8 {
            assert!(
                output_data[i].abs() < 0.01,
                "Zero group should roundtrip near-exactly, got {}",
                output_data[i]
            );
        }
    }

    #[test]
    fn test_gpu_quant_cache_update() {
        let device = WgpuDevice::default();

        let mut cache = GpuQuantKvCache::<B>::new();

        // First update: batch=1, 1 head, 2 positions, head_dim=4
        let k1 =
            Tensor::<B, 1>::from_floats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][..], &device)
                .reshape([1, 1, 2, 4]);
        let v1 = k1.clone();

        let (full_k, _full_v) = cache.update(k1, v1);
        assert_eq!(full_k.dims(), [1, 1, 2, 4]);
        assert_eq!(cache.seq_len(), 2);

        // Second update: 1 new position
        let k2 = Tensor::<B, 1>::from_floats(&[9.0, 10.0, 11.0, 12.0][..], &device)
            .reshape([1, 1, 1, 4]);
        let v2 = k2.clone();

        let (full_k, _full_v) = cache.update(k2, v2);
        assert_eq!(full_k.dims(), [1, 1, 3, 4]);
        assert_eq!(cache.seq_len(), 3);

        // Check that the concatenated result has reasonable values
        let data: Vec<f32> = full_k.to_data().to_vec().unwrap();
        // First two positions should be close to original values
        assert!(
            (data[0] - 1.0).abs() < 0.5,
            "Position 0 value drift: {}",
            data[0]
        );
        assert!(
            (data[4] - 5.0).abs() < 0.5,
            "Position 1 value drift: {}",
            data[4]
        );
        assert!(
            (data[8] - 9.0).abs() < 0.5,
            "Position 2 value drift: {}",
            data[8]
        );
    }

    #[test]
    fn test_concat_preserves_ordering() {
        let device = WgpuDevice::default();

        // batch=1, 2 heads, 2 positions, head_dim=4
        let a_data: Vec<f32> = vec![
            // head 0, pos 0-1
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // head 1, pos 0-1
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        ];
        let a = Tensor::<B, 1>::from_floats(&a_data[..], &device).reshape([1, 2, 2, 4]);
        let qa = gpu_quantize(a);

        // 1 new position
        let b_data: Vec<f32> = vec![
            // head 0, pos 2
            9.0, 10.0, 11.0, 12.0, // head 1, pos 2
            90.0, 100.0, 110.0, 120.0,
        ];
        let b = Tensor::<B, 1>::from_floats(&b_data[..], &device).reshape([1, 2, 1, 4]);
        let qb = gpu_quantize(b);

        let cat = concat_gpu_quant(qa, qb);
        assert_eq!(cat.shape, [1, 2, 3, 4]);

        let result = gpu_dequantize(&cat);
        let data: Vec<f32> = result.to_data().to_vec().unwrap();

        // Check head 0 ordering
        assert!((data[0] - 1.0).abs() < 0.5); // head 0, pos 0, dim 0
        assert!((data[4] - 5.0).abs() < 0.5); // head 0, pos 1, dim 0
        assert!((data[8] - 9.0).abs() < 0.5); // head 0, pos 2, dim 0

        // Check head 1 ordering
        assert!((data[12] - 10.0).abs() < 1.0); // head 1, pos 0, dim 0
        assert!((data[16] - 50.0).abs() < 1.0); // head 1, pos 1, dim 0
        assert!((data[20] - 90.0).abs() < 1.0); // head 1, pos 2, dim 0
    }

    // ===================================================================
    // GPU TurboQuant tests
    // ===================================================================

    #[test]
    fn test_gpu_turbo_hadamard_roundtrip() {
        let device = WgpuDevice::default();
        let head_dim = 128;
        let config = TurboQuantConfig::mse_only(3, head_dim);
        let ctx = GpuTurboQuantCtx::<B>::new(&config, &device);

        // Create a batch of vectors
        let n_groups = 4;
        let mut data = vec![0.0f32; n_groups * head_dim];
        let mut state = 42u64;
        for v in data.iter_mut() {
            state = xorshift64(state);
            *v = (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }

        let input = Tensor::<B, 1>::from_floats(&data[..], &device).reshape([n_groups, head_dim]);

        // Rotate then inverse-rotate
        let rotated = ctx.rotate(input.clone());
        let recovered = ctx.rotate_inv(rotated);

        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();
        let recovered_data: Vec<f32> = recovered.to_data().to_vec().unwrap();

        let mse: f32 = input_data
            .iter()
            .zip(recovered_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / (n_groups * head_dim) as f32;

        assert!(mse < 1e-4, "GPU Hadamard roundtrip MSE too large: {}", mse);
    }

    #[test]
    fn test_gpu_turbo_hadamard_preserves_norm() {
        let device = WgpuDevice::default();
        let head_dim = 128;
        let config = TurboQuantConfig::mse_only(3, head_dim);
        let ctx = GpuTurboQuantCtx::<B>::new(&config, &device);

        let n_groups = 4;
        let mut data = vec![0.0f32; n_groups * head_dim];
        let mut state = 99u64;
        for v in data.iter_mut() {
            state = xorshift64(state);
            *v = (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }

        let input = Tensor::<B, 1>::from_floats(&data[..], &device).reshape([n_groups, head_dim]);

        let input_norms = (input.clone() * input.clone()).sum_dim(1).sqrt();
        let rotated = ctx.rotate(input);
        let rotated_norms = (rotated.clone() * rotated.clone()).sum_dim(1).sqrt();

        let in_data: Vec<f32> = input_norms.to_data().to_vec().unwrap();
        let rot_data: Vec<f32> = rotated_norms.to_data().to_vec().unwrap();

        for (a, b) in in_data.iter().zip(rot_data.iter()) {
            let rel_err = ((a - b) / a).abs();
            assert!(
                rel_err < 0.02,
                "Rotation changed norm by {:.2}%",
                rel_err * 100.0
            );
        }
    }

    #[test]
    fn test_gpu_turbo_pack_unpack_roundtrip() {
        let device = WgpuDevice::default();

        for bits in [1, 2, 3, 4] {
            let head_dim = 128;
            let config = TurboQuantConfig::mse_only(bits, head_dim);
            let ctx = GpuTurboQuantCtx::<B>::new(&config, &device);

            let n_groups = 4;
            let n_levels = 1 << bits;
            // Create indices in 0..n_levels
            let data: Vec<i32> = (0..n_groups * head_dim)
                .map(|i| (i % n_levels) as i32)
                .collect();
            let indices =
                Tensor::<B, 1, Int>::from_ints(&data[..], &device).reshape([n_groups, head_dim]);

            let (packed, _packed_dim) = ctx.pack_indices(indices.clone());
            let unpacked = ctx.unpack_indices(packed, head_dim);

            let orig: Vec<i32> = indices.to_data().to_vec().unwrap();
            let recovered: Vec<i32> = unpacked.to_data().to_vec().unwrap();
            assert_eq!(
                orig, recovered,
                "Pack/unpack roundtrip failed for {}-bit",
                bits
            );
        }
    }

    #[test]
    fn test_gpu_turbo_quant_mse_roundtrip() {
        let device = WgpuDevice::default();
        let head_dim = 128;
        let config = TurboQuantConfig::mse_only(3, head_dim);
        let ctx = GpuTurboQuantCtx::<B>::new(&config, &device);

        // Create test data: [1, 2, 4, 128] = 8 groups of 128
        let n_groups = 1 * 2 * 4;
        let mut data = vec![0.0f32; n_groups * head_dim];
        let mut state = 12345u64;
        for v in data.iter_mut() {
            state = xorshift64(state);
            *v = (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }

        let input = Tensor::<B, 1>::from_floats(&data[..], &device).reshape([1, 2, 4, head_dim]);

        let qt = ctx.quantize(input.clone());
        let output = ctx.dequantize(&qt);

        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

        assert_eq!(input_data.len(), output_data.len());

        let mse: f32 = input_data
            .iter()
            .zip(output_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / input_data.len() as f32;

        // For 3-bit TurboQuant, MSE should be reasonable
        assert!(mse < 0.1, "GPU TurboQuant 3-bit MSE too large: {}", mse);
    }

    #[test]
    fn test_gpu_turbo_quant_cache_update() {
        let device = WgpuDevice::default();
        let head_dim = 128;
        let config = TurboQuantConfig::mse_only(3, head_dim);
        let mut cache = GpuTurboQuantKvCache::<B>::new(&config, &device);

        // First update: batch=1, 2 heads, 2 positions
        let mut data = vec![0.0f32; 1 * 2 * 2 * head_dim];
        let mut state = 777u64;
        for v in data.iter_mut() {
            state = xorshift64(state);
            *v = (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }
        let k1 = Tensor::<B, 1>::from_floats(&data[..], &device).reshape([1, 2, 2, head_dim]);
        let v1 = k1.clone();

        let (full_k, _) = cache.update(k1, v1);
        assert_eq!(full_k.dims(), [1, 2, 2, head_dim]);
        assert_eq!(cache.seq_len(), 2);

        // Second update: 1 new position
        let mut data2 = vec![0.0f32; 1 * 2 * 1 * head_dim];
        for v in data2.iter_mut() {
            state = xorshift64(state);
            *v = (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }
        let k2 = Tensor::<B, 1>::from_floats(&data2[..], &device).reshape([1, 2, 1, head_dim]);
        let v2 = k2.clone();

        let (full_k, _) = cache.update(k2, v2);
        assert_eq!(full_k.dims(), [1, 2, 3, head_dim]);
        assert_eq!(cache.seq_len(), 3);

        // Memory should be compressed
        let mem = cache.memory_bytes();
        let uncomp = cache.uncompressed_bytes();
        assert!(
            mem < uncomp,
            "GPU TurboQuant should compress: {} vs {}",
            mem,
            uncomp
        );
    }

    #[test]
    fn test_gpu_turbo_quant_2bit() {
        let device = WgpuDevice::default();
        let head_dim = 128;
        let config = TurboQuantConfig::mse_only(2, head_dim);
        let ctx = GpuTurboQuantCtx::<B>::new(&config, &device);

        let n_groups = 4;
        let mut data = vec![0.0f32; n_groups * head_dim];
        let mut state = 54321u64;
        for v in data.iter_mut() {
            state = xorshift64(state);
            *v = (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }

        let input =
            Tensor::<B, 1>::from_floats(&data[..], &device).reshape([1, 1, n_groups, head_dim]);

        let qt = ctx.quantize(input.clone());
        let output = ctx.dequantize(&qt);

        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

        let mse: f32 = input_data
            .iter()
            .zip(output_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / input_data.len() as f32;

        // 2-bit has higher MSE, but should still be bounded
        assert!(mse < 0.5, "GPU TurboQuant 2-bit MSE too large: {}", mse);
    }

    #[test]
    fn test_gpu_turbo_memory_accounting() {
        let device = WgpuDevice::default();
        let head_dim = 128;
        let config = TurboQuantConfig::mse_only(3, head_dim);
        let ctx = GpuTurboQuantCtx::<B>::new(&config, &device);

        let data = vec![0.5f32; 1 * 8 * 10 * head_dim];
        let input = Tensor::<B, 1>::from_floats(&data[..], &device).reshape([1, 8, 10, head_dim]);

        let qt = ctx.quantize(input);
        let n_groups = 80; // 1*8*10

        // For 3-bit, 128 values: packed_dim = ceil(128/10) = 13 words per group
        // packed: 80 * 13 * 4 = 4160 bytes, sigma: 80 * 4, norm: 80 * 4
        // Total: ~4800 bytes. f32: 80 * 128 * 4 = 40960 bytes.
        let packed_bytes = n_groups * qt.packed_dim * 4;
        let sigma_bytes = n_groups * 4;
        let norm_bytes = n_groups * 4;
        let expected = packed_bytes + sigma_bytes + norm_bytes;
        let uncompressed = 80 * 128 * 4;

        assert!(
            expected < uncompressed,
            "GPU TurboQuant 3-bit should compress: {} vs {} ({:.1}x)",
            expected,
            uncompressed,
            uncompressed as f64 / expected as f64
        );
    }
}
