//! Rotary Position Embedding (RoPE) with YaRN scaling.
//!
//! Standard RoPE for short contexts (up to original_max_seq_len).
//! YaRN (Yet another RoPE extensioN) for extended contexts beyond training length.
//!
//! YaRN splits frequency dimensions into three bands:
//! - Low frequency (wavelength > original_ctx): scale by factor (NTK-aware interpolation)
//! - High frequency (wavelength < original_ctx / factor): no scaling (these are fine)
//! - Medium frequency: smooth ramp between the two

use burn::prelude::*;

/// YaRN RoPE configuration.
#[derive(Debug, Clone)]
pub struct YarnConfig {
    /// Scaling factor (e.g., 16.0 for 16x context extension)
    pub factor: f64,
    /// Original training context length
    pub original_max_pos: usize,
    /// Beta parameter for the fast (high-freq) boundary
    pub beta_fast: f64,
    /// Beta parameter for the slow (low-freq) boundary
    pub beta_slow: f64,
    /// Magnitude scaling factor
    pub mscale: f64,
    /// Magnitude scaling for all dimensions
    pub mscale_all_dim: f64,
}

impl YarnConfig {
    /// Default YaRN config matching Ministral models.
    pub fn ministral() -> Self {
        Self {
            factor: 16.0,
            original_max_pos: 16384,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 1.0,
        }
    }

    /// YaRN config for DeepSeek-R1-0528-Qwen3-8B (4x extension: 32K → 131K).
    pub fn qwen3_8b() -> Self {
        Self {
            factor: 4.0,
            original_max_pos: 32768,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 1.0,
        }
    }
}

/// Precomputed RoPE frequency table with optional YaRN scaling.
pub struct RopeTable<B: Backend> {
    /// Cosine components [max_len, head_dim/2]
    pub cos: Tensor<B, 2>,
    /// Sine components [max_len, head_dim/2]
    pub sin: Tensor<B, 2>,
}

impl<B: Backend> RopeTable<B> {
    /// Build RoPE table with partial rotation (Gemma 4 global attention).
    /// Only `partial_factor` fraction of dimensions get rotated.
    /// Non-rotated frequencies are set to 0, so cos=1 and sin=0 (pass-through).
    pub fn with_partial_rotation(
        head_dim: usize,
        max_len: usize,
        theta: f64,
        partial_factor: f64,
        device: &B::Device,
    ) -> Self {
        let half_dim = head_dim / 2;
        let n_rotated = (half_dim as f64 * partial_factor) as usize;

        // Build inv_freq: first n_rotated frequencies are real, rest are 0
        let mut inv_freq: Vec<f64> = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            if i < n_rotated {
                // Frequency spacing based on FULL head_dim, not rotated portion
                inv_freq.push(1.0 / theta.powf(2.0 * i as f64 / head_dim as f64));
            } else {
                inv_freq.push(0.0); // cos=1, sin=0 → pass-through
            }
        }

        Self::from_inv_freq(&inv_freq, max_len, 1.0, device)
    }

    /// Build standard RoPE frequency table (no YaRN).
    pub fn new(head_dim: usize, max_len: usize, theta: f64, device: &B::Device) -> Self {
        let inv_freq: Vec<f64> = (0..head_dim / 2)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();

        Self::from_inv_freq(&inv_freq, max_len, 1.0, device)
    }

    /// Build YaRN-scaled RoPE frequency table for extended context.
    pub fn with_yarn(
        head_dim: usize,
        max_len: usize,
        theta: f64,
        yarn: &YarnConfig,
        device: &B::Device,
    ) -> Self {
        let half_dim = head_dim / 2;

        // Base inverse frequencies
        let base_inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();

        // Compute wavelengths for each frequency dimension
        // wavelength = 2π / freq = 2π * theta^(2i/d)
        let wavelengths: Vec<f64> = base_inv_freq.iter()
            .map(|&inv_f| 2.0 * std::f64::consts::PI / inv_f)
            .collect();

        // Find the frequency band boundaries
        // low_freq_wavelen: wavelengths above this get full scaling
        // high_freq_wavelen: wavelengths below this get no scaling
        let low_freq_wavelen = yarn.original_max_pos as f64 / yarn.beta_slow;
        let high_freq_wavelen = yarn.original_max_pos as f64 / yarn.beta_fast;

        // Apply YaRN scaling per dimension
        let scaled_inv_freq: Vec<f64> = base_inv_freq.iter().zip(wavelengths.iter())
            .map(|(&inv_f, &wavelen)| {
                if wavelen < high_freq_wavelen {
                    // High frequency: no scaling needed
                    inv_f
                } else if wavelen > low_freq_wavelen {
                    // Low frequency: full NTK-aware scaling
                    inv_f / yarn.factor
                } else {
                    // Medium frequency: smooth interpolation
                    let ramp = (wavelen - high_freq_wavelen)
                        / (low_freq_wavelen - high_freq_wavelen);
                    let smooth = 1.0 - ramp; // 1 at high_freq boundary, 0 at low_freq
                    let scaled = inv_f / yarn.factor;
                    // Interpolate between unscaled and scaled
                    inv_f * smooth + scaled * (1.0 - smooth)
                }
            })
            .collect();

        // Compute magnitude scaling (mscale)
        // This compensates for the amplitude change due to context extension
        let mscale = if yarn.mscale_all_dim > 0.0 {
            let m = yarn.mscale * (1.0 + yarn.factor.ln() / (yarn.mscale_all_dim * 10.0));
            m.max(1.0)
        } else {
            1.0
        };

        Self::from_inv_freq(&scaled_inv_freq, max_len, mscale, device)
    }

    /// Build cos/sin tables from inverse frequencies.
    fn from_inv_freq(inv_freq: &[f64], max_len: usize, mscale: f64, device: &B::Device) -> Self {
        let half_dim = inv_freq.len();

        // Position indices
        let positions: Vec<f32> = (0..max_len).map(|p| p as f32).collect();
        let pos_tensor = Tensor::<B, 1>::from_floats(&positions[..], device);

        // Convert inv_freq to f32
        let inv_freq_f32: Vec<f32> = inv_freq.iter().map(|&f| f as f32).collect();
        let inv_freq_tensor = Tensor::<B, 1>::from_floats(&inv_freq_f32[..], device);

        // Outer product: [max_len, half_dim]
        let freqs = pos_tensor
            .reshape([max_len, 1])
            .matmul(inv_freq_tensor.reshape([1, half_dim]));

        // Apply magnitude scaling
        let cos = (freqs.clone().cos()) * (mscale as f32);
        let sin = (freqs.sin()) * (mscale as f32);

        Self { cos, sin }
    }

    /// Apply RoPE to a tensor of shape [batch, n_heads, seq_len, head_dim].
    /// `offset` is the position offset for KV cache continuation.
    pub fn apply(
        &self,
        x: Tensor<B, 4>,
        offset: usize,
    ) -> Tensor<B, 4> {
        let [batch, n_heads, seq_len, head_dim] = x.dims();
        let half_dim = head_dim / 2;

        // Slice cos/sin for the current positions
        let cos = self.cos.clone()
            .slice([offset..offset + seq_len, 0..half_dim])
            .reshape([1, 1, seq_len, half_dim]);
        let sin = self.sin.clone()
            .slice([offset..offset + seq_len, 0..half_dim])
            .reshape([1, 1, seq_len, half_dim]);

        // Split into halves and apply rotation
        let x1 = x.clone().narrow(3, 0, half_dim);
        let x2 = x.narrow(3, half_dim, half_dim);

        let cos = cos.expand([batch, n_heads, seq_len, half_dim]);
        let sin = sin.expand([batch, n_heads, seq_len, half_dim]);

        let out1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let out2 = x1 * sin + x2 * cos;

        Tensor::cat(vec![out1, out2], 3)
    }
}
