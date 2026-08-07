//! TurboQuant: Near-optimal vector quantization for KV cache compression.
//!
//! Implements the full TurboQuant algorithm from Zandieh et al. (2025):
//!   1. Random rotation via randomized Hadamard transform (RHT) to decorrelate coordinates
//!   2. Lloyd-Max optimal scalar quantization for the resulting Gaussian-distributed coordinates
//!   3. QJL (Quantized Johnson-Lindenstrauss) binary sketch of the residual for unbiased
//!      inner product estimation
//!
//! Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//!            arXiv:2504.19874v1

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Lloyd-Max codebook tables for N(0, 1) Gaussian
// ---------------------------------------------------------------------------
//
// These are the optimal reconstruction centroids and decision boundaries
// for scalar quantization of a standard Gaussian random variable, computed
// via the iterative Lloyd-Max algorithm. After rotation, each coordinate
// of a unit-norm vector follows approximately N(0, 1/d), so we scale
// these tables by sigma = 1/sqrt(d) at quantization time.
//
// For b bits, we have 2^b reconstruction levels (centroids) and 2^b - 1
// internal decision boundaries.

/// Centroids for 1-bit (2-level) Lloyd-Max quantizer on N(0,1).
/// Boundaries: {0}. Centroids: {-sqrt(2/pi), +sqrt(2/pi)}.
const LLOYD_MAX_1BIT_CENTROIDS: [f64; 2] = [-0.7978845608, 0.7978845608];

/// Centroids for 2-bit (4-level) Lloyd-Max quantizer on N(0,1).
/// Computed numerically. Boundaries at {-0.9816, 0, +0.9816}.
const LLOYD_MAX_2BIT_CENTROIDS: [f64; 4] = [-1.510, -0.4528, 0.4528, 1.510];
const LLOYD_MAX_2BIT_BOUNDARIES: [f64; 3] = [-0.9816, 0.0, 0.9816];

/// Centroids for 3-bit (8-level) Lloyd-Max quantizer on N(0,1).
const LLOYD_MAX_3BIT_CENTROIDS: [f64; 8] = [
    -2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519,
];
const LLOYD_MAX_3BIT_BOUNDARIES: [f64; 7] =
    [-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480];

/// Centroids for 4-bit (16-level) Lloyd-Max quantizer on N(0,1).
const LLOYD_MAX_4BIT_CENTROIDS: [f64; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881, 0.6568,
    0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
];
const LLOYD_MAX_4BIT_BOUNDARIES: [f64; 15] = [
    -2.4008, -1.8435, -1.4372, -1.0993, -0.7996, -0.5224, -0.2582, 0.0, 0.2582, 0.5224, 0.7996,
    1.0993, 1.4372, 1.8435, 2.4008,
];

/// Retrieve the Lloyd-Max centroids for the given bit width (1..=4).
fn lloyd_max_centroids(bits: usize) -> &'static [f64] {
    match bits {
        1 => &LLOYD_MAX_1BIT_CENTROIDS,
        2 => &LLOYD_MAX_2BIT_CENTROIDS,
        3 => &LLOYD_MAX_3BIT_CENTROIDS,
        4 => &LLOYD_MAX_4BIT_CENTROIDS,
        _ => panic!(
            "Lloyd-Max tables only precomputed for 1..=4 bits, got {}",
            bits
        ),
    }
}

/// Centroids for 1-bit quantizer boundary.
const LLOYD_MAX_1BIT_BOUNDARIES: [f64; 1] = [0.0];

/// Retrieve the Lloyd-Max decision boundaries for the given bit width (1..=4).
fn lloyd_max_boundaries(bits: usize) -> &'static [f64] {
    match bits {
        1 => &LLOYD_MAX_1BIT_BOUNDARIES,
        2 => &LLOYD_MAX_2BIT_BOUNDARIES,
        3 => &LLOYD_MAX_3BIT_BOUNDARIES,
        4 => &LLOYD_MAX_4BIT_BOUNDARIES,
        _ => panic!(
            "Lloyd-Max tables only precomputed for 1..=4 bits, got {}",
            bits
        ),
    }
}

/// Find the nearest centroid index for a scalar value using the decision boundaries.
/// `boundaries` must be sorted ascending. Returns index in 0..centroids.len().
fn quantize_scalar(val: f64, boundaries: &[f64]) -> usize {
    // Binary search: find the first boundary >= val
    match boundaries.binary_search_by(|b| b.partial_cmp(&val).unwrap()) {
        Ok(i) => i + 1, // val exactly equals boundary[i] -> goes to upper bin
        Err(i) => i,    // val < boundary[i] for all j>=i, so belongs to bin i
    }
}

// ---------------------------------------------------------------------------
// Randomized Hadamard Transform (RHT)
// ---------------------------------------------------------------------------
//
// R = H * D where H is the normalized Walsh-Hadamard matrix and D is a
// diagonal of random +/-1 entries (Rademacher). This gives O(d log d) rotation
// instead of O(d^2) for a dense random orthogonal matrix.

/// Seeded pseudo-random +/-1 diagonal. Uses a simple xorshift64 for speed and
/// reproducibility.
struct Rademacher {
    signs: Vec<f64>,
}

impl Rademacher {
    fn new(dim: usize, seed: u64) -> Self {
        let mut state = seed;
        let signs: Vec<f64> = (0..dim)
            .map(|_| {
                state = xorshift64(state);
                if state & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        Self { signs }
    }
}

fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Pad dimension up to the next power of 2 (needed for Hadamard).
fn next_pow2(n: usize) -> usize {
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

/// In-place unnormalized Walsh-Hadamard transform on a buffer of length 2^k.
fn hadamard_inplace(buf: &mut [f64]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    let mut half = 1;
    while half < n {
        for i in (0..n).step_by(half * 2) {
            for j in i..i + half {
                let a = buf[j];
                let b = buf[j + half];
                buf[j] = a + b;
                buf[j + half] = a - b;
            }
        }
        half *= 2;
    }
}

/// The rotation context: precomputed Rademacher signs plus the padded dimension.
/// Reused across all quantizations for a given (head_dim, seed).
pub struct RotationCtx {
    /// Rademacher +/-1 signs, length = padded_dim.
    signs: Vec<f64>,
    /// Original dimension (unpadded).
    dim: usize,
    /// Padded to next power of 2.
    padded_dim: usize,
    /// Normalization factor: 1/sqrt(padded_dim).
    norm: f64,
}

impl RotationCtx {
    pub fn new(dim: usize, seed: u64) -> Self {
        let padded_dim = next_pow2(dim);
        let rad = Rademacher::new(padded_dim, seed);
        Self {
            signs: rad.signs,
            dim,
            padded_dim,
            norm: 1.0 / (padded_dim as f64).sqrt(),
        }
    }

    /// Apply the randomized Hadamard rotation: y = (1/sqrt(d)) * H * D * x.
    /// Input `x` has length `self.dim`. Output has length `self.dim` (truncated from padded).
    pub fn rotate(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim);
        // Work in f64 for numerical stability
        let mut buf = vec![0.0f64; self.padded_dim];
        for i in 0..self.dim {
            buf[i] = x[i] as f64 * self.signs[i];
        }
        // Padding entries stay 0 (zero-padded, then multiplied by signs, still 0 contribution)
        hadamard_inplace(&mut buf);
        // Normalize and truncate back to original dim
        buf.iter()
            .take(self.dim)
            .map(|&v| (v * self.norm) as f32)
            .collect()
    }

    /// Apply the inverse rotation: x = D * H * (1/sqrt(d)) * y = (R^T) * y.
    /// Since R = (1/sqrt(d)) * H * D and H is symmetric, D is its own inverse:
    ///   R^T = D * H * (1/sqrt(d))
    /// The inverse of the normalized Hadamard on the padded space is
    /// itself (H^{-1} = (1/d)*H for unnormalized H). So:
    ///   rotate:  y = (1/sqrt(d)) * H * D * x
    ///   inverse: x = D * (1/sqrt(d)) * H * y  (since D^{-1}=D and (1/sqrt(d)*H)^{-1} = (1/sqrt(d))*H)
    pub fn rotate_inv(&self, y: &[f32]) -> Vec<f32> {
        debug_assert_eq!(y.len(), self.dim);
        let mut buf = vec![0.0f64; self.padded_dim];
        for i in 0..self.dim {
            buf[i] = y[i] as f64;
        }
        hadamard_inplace(&mut buf);
        // Multiply by D and normalize
        let mut out = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            out.push((buf[i] * self.norm * self.signs[i]) as f32);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// QJL (Quantized Johnson-Lindenstrauss) projection
// ---------------------------------------------------------------------------
//
// Projects the residual vector through a random matrix and stores only the
// signs. For inner product correction:
//   <x, y> approx= <Q(x), Q(y)> + sqrt(pi/2)/m * sum(sign_x[i]*sign_y[i]) * ||r_x|| * ||r_y||

/// QJL projection context: a random +/-1 matrix of shape [m, d] where m is
/// the number of sketch bits and d is the vector dimension.
///
/// We use a seeded PRNG to generate a random +/-1 matrix (Rademacher entries).
/// This is the standard approach for QJL (see Definition 1 in the paper, except
/// we use +/-1 instead of Gaussian for efficiency — the sign is the same).
pub struct QjlCtx {
    /// Random +/-1 matrix stored row-major: [m, d].
    matrix: Vec<i8>,
    /// Number of sketch bits (projection dimension).
    m: usize,
    /// Original vector dimension.
    d: usize,
}

impl QjlCtx {
    pub fn new(m: usize, d: usize, seed: u64) -> Self {
        let total = m * d;
        let mut state = seed;
        let matrix: Vec<i8> = (0..total)
            .map(|_| {
                state = xorshift64(state);
                if state & 1 == 0 {
                    1i8
                } else {
                    -1i8
                }
            })
            .collect();
        Self { matrix, m, d }
    }

    /// Project a vector and return the sign bits (packed, 8 per byte) plus the L2 norm.
    /// Returns (sign_bits, ||r||_2).
    pub fn project(&self, r: &[f32]) -> (Vec<u8>, f32) {
        debug_assert_eq!(r.len(), self.d);

        let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        let packed_len = (self.m + 7) / 8;
        let mut signs = vec![0u8; packed_len];

        for i in 0..self.m {
            let row = &self.matrix[i * self.d..(i + 1) * self.d];
            let dot: f32 = row.iter().zip(r.iter()).map(|(&s, &v)| s as f32 * v).sum();
            if dot >= 0.0 {
                signs[i / 8] |= 1 << (i % 8);
            }
        }

        (signs, norm)
    }

    /// Dequantize: reconstruct the QJL correction vector.
    /// Returns sqrt(pi/2)/m * gamma * S^T * qjl_signs.
    ///
    /// This is the contribution to add to the MSE-dequantized vector.
    /// `gamma` = ||residual||_2, `signs` = packed sign bits from `project`.
    /// m = number of sketch bits (projection dimension).
    pub fn dequantize_correction(&self, signs: &[u8], gamma: f32) -> Vec<f32> {
        let coeff = (PI as f32 / 2.0).sqrt() / self.m as f32 * gamma;
        let mut out = vec![0.0f32; self.d];

        for i in 0..self.m {
            let sign_bit = (signs[i / 8] >> (i % 8)) & 1;
            let sign_val: f32 = if sign_bit == 1 { 1.0 } else { -1.0 };
            let row = &self.matrix[i * self.d..(i + 1) * self.d];
            for j in 0..self.d {
                out[j] += coeff * sign_val * row[j] as f32;
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// TurboQuant configuration and storage
// ---------------------------------------------------------------------------

/// Configuration for TurboQuant KV cache compression.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Scalar quantization bit width for the MSE-optimal stage (1, 2, 3, or 4).
    pub bits: usize,
    /// Number of QJL sketch bits for residual inner product correction.
    /// 0 = MSE-only mode (no QJL). Otherwise the MSE stage uses `bits - 1` bits
    /// and 1 bit goes to the QJL sketch per coordinate (so `residual_bits = d`),
    /// matching the paper's Algorithm 2. You can also set a custom value.
    pub residual_bits: usize,
    /// Dimension of each attention head (e.g. 128 for Ministral).
    pub head_dim: usize,
    /// Seed for reproducible rotation and QJL matrices.
    pub seed: u64,
}

impl TurboQuantConfig {
    /// MSE-optimal only: no QJL residual. All `bits` go to Lloyd-Max quantization.
    pub fn mse_only(bits: usize, head_dim: usize) -> Self {
        Self {
            bits,
            residual_bits: 0,
            head_dim,
            seed: 0x5A3D_7E1F_C8B2_A406,
        }
    }

    /// Inner-product optimal (Algorithm 2): (bits-1) bits for MSE quantization,
    /// then QJL on the residual with `head_dim` sketch bits (1 bit per coordinate).
    pub fn inner_product(bits: usize, head_dim: usize) -> Self {
        assert!(
            bits >= 2,
            "Inner-product TurboQuant requires bits >= 2 (need at least 1 for MSE + 1 for QJL)"
        );
        Self {
            bits,
            residual_bits: head_dim,
            head_dim,
            seed: 0x5A3D_7E1F_C8B2_A406,
        }
    }

    /// Effective MSE quantization bits (bits if no QJL, bits-1 if QJL is used).
    fn mse_bits(&self) -> usize {
        if self.residual_bits > 0 {
            self.bits - 1
        } else {
            self.bits
        }
    }

    /// Total bits per coordinate (approximate, not counting per-group overhead).
    pub fn total_bits_per_value(&self) -> f32 {
        if self.residual_bits > 0 {
            // (bits - 1) for MSE quantization + residual_bits/head_dim for QJL
            self.mse_bits() as f32 + self.residual_bits as f32 / self.head_dim as f32
        } else {
            self.bits as f32
        }
    }
}

/// Quantized storage for a single tensor under TurboQuant.
///
/// Per-group (one group = one head_dim vector at a specific [batch, head, seq_pos]):
/// - `indices`: packed quantization indices, `mse_bits` bits per coordinate
/// - `sigma`: estimated standard deviation of the rotated coordinates (1 f32)
///
/// If QJL is enabled, additionally per group:
/// - `qjl_signs`: packed sign bits of the QJL projection (residual_bits / 8 bytes)
/// - `residual_norm`: L2 norm of the residual vector (1 f32)
pub struct TurboQuantTensor {
    /// Packed quantization indices. For b-bit quantization of a group of `d` values,
    /// this stores `ceil(b * d / 8)` bytes per group, all groups concatenated.
    pub(crate) indices: Vec<u8>,
    /// Per-group standard deviation sigma of the rotated coordinates.
    pub(crate) sigma: Vec<f32>,
    /// Per-group norm of the original vector (for rescaling on dequant).
    pub(crate) norm: Vec<f32>,
    /// QJL sign bits (present only if residual_bits > 0).
    pub(crate) qjl_signs: Option<Vec<u8>>,
    /// Per-group L2 norm of the residual (present only if residual_bits > 0).
    pub(crate) residual_norm: Option<Vec<f32>>,
    /// Shape: [batch, n_kv_heads, seq_len, head_dim].
    pub shape: [usize; 4],
    /// Config used.
    pub(crate) config: TurboQuantConfig,
}

/// Shared precomputed context for TurboQuant operations on a fixed (head_dim, seed).
pub struct TurboQuantCtx {
    pub rotation: RotationCtx,
    pub qjl: Option<QjlCtx>,
    pub config: TurboQuantConfig,
}

impl TurboQuantCtx {
    pub fn new(config: &TurboQuantConfig) -> Self {
        let rotation = RotationCtx::new(config.head_dim, config.seed);
        let qjl = if config.residual_bits > 0 {
            // Use a different seed for the QJL matrix
            Some(QjlCtx::new(
                config.residual_bits,
                config.head_dim,
                config.seed.wrapping_add(1),
            ))
        } else {
            None
        };
        Self {
            rotation,
            qjl,
            config: config.clone(),
        }
    }

    /// Quantize a flat f32 buffer with shape [batch, n_kv_heads, seq_len, head_dim].
    pub fn quantize(&self, data: &[f32], shape: [usize; 4]) -> TurboQuantTensor {
        let [batch, n_heads, seq_len, head_dim] = shape;
        assert_eq!(head_dim, self.config.head_dim);
        let n_groups = batch * n_heads * seq_len;
        let mse_bits = self.config.mse_bits();
        let centroids = lloyd_max_centroids(mse_bits);
        let boundaries = lloyd_max_boundaries(mse_bits);
        let n_levels = 1 << mse_bits;

        // Bytes per group for packed indices
        let bits_per_group = mse_bits * head_dim;
        let bytes_per_group = (bits_per_group + 7) / 8;

        let mut indices = Vec::with_capacity(n_groups * bytes_per_group);
        let mut sigma_vec = Vec::with_capacity(n_groups);
        let mut norm_vec = Vec::with_capacity(n_groups);

        // QJL storage
        let qjl_bytes_per_group = if self.config.residual_bits > 0 {
            (self.config.residual_bits + 7) / 8
        } else {
            0
        };
        let mut qjl_signs_all: Vec<u8> = if self.config.residual_bits > 0 {
            Vec::with_capacity(n_groups * qjl_bytes_per_group)
        } else {
            Vec::new()
        };
        let mut residual_norms: Vec<f32> = if self.config.residual_bits > 0 {
            Vec::with_capacity(n_groups)
        } else {
            Vec::new()
        };

        // Temporary buffers
        let mut group_indices = vec![0u8; head_dim];

        for g in 0..n_groups {
            let start = g * head_dim;
            let group = &data[start..start + head_dim];

            // 1. Compute the L2 norm of the original vector
            let vec_norm = group.iter().map(|v| v * v).sum::<f32>().sqrt();
            norm_vec.push(vec_norm);

            // 2. Normalize to unit sphere, then rotate
            //    (The paper assumes ||x||=1; we store the norm separately and rescale)
            let inv_norm = if vec_norm > 1e-12 {
                1.0 / vec_norm
            } else {
                0.0
            };
            let normalized: Vec<f32> = group.iter().map(|v| v * inv_norm).collect();
            let rot = self.rotation.rotate(&normalized);

            // 3. Estimate sigma = std of rotated coordinates
            //    Theoretical: sigma = 1/sqrt(d), but we estimate from the data for robustness.
            let mean: f64 = rot.iter().map(|&v| v as f64).sum::<f64>() / head_dim as f64;
            let var: f64 = rot
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / head_dim as f64;
            let sigma = var.sqrt().max(1e-10);
            sigma_vec.push(sigma as f32);

            // 4. Lloyd-Max quantization: scale coordinates by 1/sigma, quantize against
            //    the standard N(0,1) tables, then store indices.
            let inv_sigma = 1.0 / sigma;
            for d in 0..head_dim {
                let normalized_val = rot[d] as f64 * inv_sigma;
                let idx = quantize_scalar(normalized_val, boundaries);
                debug_assert!(idx < n_levels);
                group_indices[d] = idx as u8;
            }

            // 5. Pack indices into bytes (mse_bits bits per index)
            pack_indices(&group_indices[..head_dim], mse_bits, &mut indices);

            // 6. If QJL is enabled, compute residual and project
            if let Some(ref qjl) = self.qjl {
                // Dequantize the MSE approximation (in rotated, normalized space)
                let mut deq_rotated = vec![0.0f32; head_dim];
                for d in 0..head_dim {
                    deq_rotated[d] = (centroids[group_indices[d] as usize] * sigma) as f32;
                }
                // Inverse rotate to get back to original normalized space
                let deq_normalized = self.rotation.rotate_inv(&deq_rotated);
                // Residual in the original normalized space
                let residual: Vec<f32> = normalized
                    .iter()
                    .zip(deq_normalized.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                // QJL project the residual
                let (signs, rnorm) = qjl.project(&residual);
                qjl_signs_all.extend_from_slice(&signs);
                residual_norms.push(rnorm);
            }
        }

        TurboQuantTensor {
            indices,
            sigma: sigma_vec,
            norm: norm_vec,
            qjl_signs: if self.config.residual_bits > 0 {
                Some(qjl_signs_all)
            } else {
                None
            },
            residual_norm: if self.config.residual_bits > 0 {
                Some(residual_norms)
            } else {
                None
            },
            shape,
            config: self.config.clone(),
        }
    }

    /// Dequantize a TurboQuantTensor back to f32 values in original shape order.
    pub fn dequantize(&self, qt: &TurboQuantTensor) -> Vec<f32> {
        let head_dim = qt.shape[3];
        let n_groups = qt.sigma.len();
        let mse_bits = qt.config.mse_bits();
        let centroids = lloyd_max_centroids(mse_bits);

        let bits_per_group = mse_bits * head_dim;
        let bytes_per_group = (bits_per_group + 7) / 8;
        let qjl_bytes_per_group = if qt.config.residual_bits > 0 {
            (qt.config.residual_bits + 7) / 8
        } else {
            0
        };

        let total = n_groups * head_dim;
        let mut out = Vec::with_capacity(total);
        let mut group_indices = vec![0u8; head_dim];

        for g in 0..n_groups {
            let sigma = qt.sigma[g] as f64;
            let vec_norm = qt.norm[g];

            // 1. Unpack indices
            let byte_offset = g * bytes_per_group;
            unpack_indices(
                &qt.indices[byte_offset..byte_offset + bytes_per_group],
                mse_bits,
                head_dim,
                &mut group_indices,
            );

            // 2. Reconstruct in rotated space: centroid * sigma
            let mut deq_rotated = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                deq_rotated[d] = (centroids[group_indices[d] as usize] * sigma) as f32;
            }

            // 3. Inverse rotate back to original normalized space
            let mut deq_normalized = self.rotation.rotate_inv(&deq_rotated);

            // 4. If QJL, add the correction
            if let (Some(ref qjl), Some(ref qjl_signs), Some(ref rnorms)) =
                (&self.qjl, &qt.qjl_signs, &qt.residual_norm)
            {
                let sign_offset = g * qjl_bytes_per_group;
                let gamma = rnorms[g];
                let correction = qjl.dequantize_correction(
                    &qjl_signs[sign_offset..sign_offset + qjl_bytes_per_group],
                    gamma,
                );
                for d in 0..head_dim {
                    deq_normalized[d] += correction[d];
                }
            }

            // 5. Rescale by the original vector norm
            for d in 0..head_dim {
                out.push(deq_normalized[d] * vec_norm);
            }
        }

        out
    }

    /// Approximate memory usage in bytes for a TurboQuantTensor.
    pub fn memory_bytes(qt: &TurboQuantTensor) -> usize {
        let [_batch, _n_heads, _seq_len, _head_dim] = qt.shape;
        let n_groups = qt.sigma.len();

        let index_bytes = qt.indices.len();
        let sigma_bytes = n_groups * 4; // f32 per group
        let norm_bytes = n_groups * 4; // f32 per group
        let qjl_bytes = qt.qjl_signs.as_ref().map_or(0, |s| s.len());
        let rnorm_bytes = qt.residual_norm.as_ref().map_or(0, |r| r.len() * 4);

        index_bytes + sigma_bytes + norm_bytes + qjl_bytes + rnorm_bytes
    }
}

// ---------------------------------------------------------------------------
// Bit packing utilities
// ---------------------------------------------------------------------------

/// Pack a slice of indices (each in 0..2^bits) into bytes, `bits` bits per index.
fn pack_indices(indices: &[u8], bits: usize, out: &mut Vec<u8>) {
    let total_bits = indices.len() * bits;
    let total_bytes = (total_bits + 7) / 8;
    let start = out.len();
    out.resize(start + total_bytes, 0);

    let mut bit_pos = 0usize;
    for &idx in indices {
        let byte_idx = start + bit_pos / 8;
        let bit_offset = bit_pos % 8;

        // Write `bits` bits of `idx` starting at bit_offset within byte_idx
        // May span two bytes if bit_offset + bits > 8
        let val = idx as u16;
        let merged = (val << bit_offset) as u16;
        out[byte_idx] |= merged as u8;
        if bit_offset + bits > 8 && byte_idx + 1 < out.len() {
            out[byte_idx + 1] |= (merged >> 8) as u8;
        }

        bit_pos += bits;
    }
}

/// Unpack indices from packed bytes. Reads `count` indices of `bits` bits each.
fn unpack_indices(data: &[u8], bits: usize, count: usize, out: &mut [u8]) {
    let mask = (1u16 << bits) - 1;
    let mut bit_pos = 0usize;

    for i in 0..count {
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        let mut val = data[byte_idx] as u16 >> bit_offset;
        if bit_offset + bits > 8 && byte_idx + 1 < data.len() {
            val |= (data[byte_idx + 1] as u16) << (8 - bit_offset);
        }
        out[i] = (val & mask) as u8;

        bit_pos += bits;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_basic() {
        // H_2 * [1, 0] = [1, 1]
        let mut buf = vec![1.0, 0.0];
        hadamard_inplace(&mut buf);
        assert_eq!(buf, vec![1.0, 1.0]);

        // H_2 * [1, 1] = [2, 0]
        let mut buf = vec![1.0, 1.0];
        hadamard_inplace(&mut buf);
        assert_eq!(buf, vec![2.0, 0.0]);
    }

    #[test]
    fn test_rotation_roundtrip() {
        let dim = 128;
        let ctx = RotationCtx::new(dim, 42);

        // Create a unit vector
        let mut x = vec![0.0f32; dim];
        for i in 0..dim {
            x[i] = ((i as f32 + 1.0) * 0.1).sin();
        }
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in x.iter_mut() {
            *v /= norm;
        }

        // Rotate and inverse-rotate
        let rotated = ctx.rotate(&x);
        let recovered = ctx.rotate_inv(&rotated);

        // Check roundtrip error
        let mse: f32 = x
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / dim as f32;
        assert!(mse < 1e-6, "Rotation roundtrip MSE too large: {}", mse);
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let dim = 128;
        let ctx = RotationCtx::new(dim, 42);

        let mut x = vec![0.0f32; dim];
        for i in 0..dim {
            x[i] = (i as f32 * 0.07).cos();
        }
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let rotated = ctx.rotate(&x);
        let norm_after: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();

        let rel_err = ((norm_before - norm_after) / norm_before).abs();
        assert!(
            rel_err < 0.01,
            "Rotation changed norm by {:.2}%",
            rel_err * 100.0
        );
    }

    #[test]
    fn test_quantize_scalar_2bit() {
        let boundaries = lloyd_max_boundaries(2);
        // Values very negative -> bin 0
        assert_eq!(quantize_scalar(-5.0, boundaries), 0);
        // Between -0.9816 and 0 -> bin 1
        assert_eq!(quantize_scalar(-0.3, boundaries), 1);
        // Between 0 and 0.9816 -> bin 2
        assert_eq!(quantize_scalar(0.3, boundaries), 2);
        // Very positive -> bin 3
        assert_eq!(quantize_scalar(5.0, boundaries), 3);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        // 2-bit indices: values 0..3
        let indices: Vec<u8> = (0..16).map(|i| (i % 4) as u8).collect();
        let mut packed = Vec::new();
        pack_indices(&indices, 2, &mut packed);

        let mut unpacked = vec![0u8; 16];
        unpack_indices(&packed, 2, 16, &mut unpacked);
        assert_eq!(indices, unpacked);

        // 3-bit indices: values 0..7
        let indices: Vec<u8> = (0..16).map(|i| (i % 8) as u8).collect();
        let mut packed = Vec::new();
        pack_indices(&indices, 3, &mut packed);

        let mut unpacked = vec![0u8; 16];
        unpack_indices(&packed, 3, 16, &mut unpacked);
        assert_eq!(indices, unpacked);

        // 4-bit indices
        let indices: Vec<u8> = (0..16).map(|i| i as u8).collect();
        let mut packed = Vec::new();
        pack_indices(&indices, 4, &mut packed);

        let mut unpacked = vec![0u8; 16];
        unpack_indices(&packed, 4, 16, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_turbo_quant_mse_roundtrip() {
        let head_dim = 128;
        let config = TurboQuantConfig::mse_only(3, head_dim);
        let ctx = TurboQuantCtx::new(&config);

        // Create test data: [1, 2, 4, 128] = 8 groups of 128
        let n_groups = 1 * 2 * 4;
        let mut data = vec![0.0f32; n_groups * head_dim];
        let mut state = 12345u64;
        for v in data.iter_mut() {
            state = xorshift64(state);
            *v = (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }

        let shape = [1, 2, 4, head_dim];
        let qt = ctx.quantize(&data, shape);
        let deq = ctx.dequantize(&qt);

        assert_eq!(deq.len(), data.len());

        // Compute MSE
        let mse: f32 = data
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        // For 3-bit quantization, the paper claims D_mse ~ 0.03 for unit vectors.
        // Our vectors aren't unit, but MSE should still be reasonable.
        println!("TurboQuant 3-bit MSE roundtrip: {:.6}", mse);
        assert!(mse < 0.1, "TurboQuant 3-bit MSE too large: {}", mse);
    }

    #[test]
    fn test_turbo_quant_inner_product() {
        let head_dim = 128;
        let config = TurboQuantConfig::inner_product(3, head_dim);
        let ctx = TurboQuantCtx::new(&config);

        // Create two random vectors
        let mut x = vec![0.0f32; head_dim];
        let mut y = vec![0.0f32; head_dim];
        let mut state = 99999u64;
        for v in x.iter_mut() {
            state = xorshift64(state);
            *v = (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }
        for v in y.iter_mut() {
            state = xorshift64(state);
            *v = (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }

        // True inner product
        let true_ip: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        // Quantize both
        let shape = [1, 1, 1, head_dim];
        let qt_x = ctx.quantize(&x, shape);
        let qt_y = ctx.quantize(&y, shape);

        let deq_x = ctx.dequantize(&qt_x);
        let deq_y = ctx.dequantize(&qt_y);

        // Reconstructed inner product
        let recon_ip: f32 = deq_x.iter().zip(deq_y.iter()).map(|(a, b)| a * b).sum();

        let ip_error = (true_ip - recon_ip).abs();
        let rel_error = ip_error / true_ip.abs().max(1e-6);
        println!(
            "True IP: {:.6}, Reconstructed IP: {:.6}, Error: {:.6}, Rel: {:.4}",
            true_ip, recon_ip, ip_error, rel_error
        );

        // Inner product error should be modest
        assert!(
            rel_error < 0.5,
            "Inner product relative error too large: {}",
            rel_error
        );
    }

    #[test]
    fn test_qjl_project_dimensions() {
        let m = 64;
        let d = 128;
        let qjl = QjlCtx::new(m, d, 42);

        let r: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
        let (signs, norm) = qjl.project(&r);

        assert_eq!(signs.len(), (m + 7) / 8);
        assert!(norm > 0.0);

        let correction = qjl.dequantize_correction(&signs, norm);
        assert_eq!(correction.len(), d);
    }

    #[test]
    fn test_memory_accounting_turbo() {
        let head_dim = 128;
        let config = TurboQuantConfig::mse_only(2, head_dim);
        let ctx = TurboQuantCtx::new(&config);

        let data = vec![0.5f32; 1 * 8 * 10 * head_dim];
        let shape = [1, 8, 10, head_dim];
        let qt = ctx.quantize(&data, shape);

        let mem = TurboQuantCtx::memory_bytes(&qt);
        let uncompressed = 1 * 8 * 10 * head_dim * 4; // f32

        // 2-bit: 2*128/8 = 32 bytes per group for indices, + 4(sigma) + 4(norm) = 40 bytes
        // vs f32: 128*4 = 512 bytes per group. ~12.8x compression on data alone.
        assert!(
            mem < uncompressed,
            "TurboQuant should use less memory than f32: {} vs {}",
            mem,
            uncompressed
        );
        println!(
            "TurboQuant 2-bit memory: {} bytes vs {} f32 bytes ({:.1}x compression)",
            mem,
            uncompressed,
            uncompressed as f64 / mem as f64
        );
    }
}
