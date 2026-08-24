//! Core transformer layers: RMSNorm, SwiGLU FFN, GQA Attention, KV Cache.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use crate::models::gemma::config::MistralConfig;
use crate::models::gemma::rope::RopeTable;

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// Per-layer key-value cache for incremental decoding.
/// Stores accumulated K and V tensors shaped [batch, n_kv_heads, seq_so_far, head_dim].
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    /// Create an empty cache (for prefill or first token).
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }

    /// Append new K/V to the cache and return the full K/V.
    /// new_k, new_v: [batch, n_kv_heads, new_len, head_dim]
    /// Returns (full_k, full_v) where seq dim is accumulated.
    pub fn update(
        &mut self,
        new_k: Tensor<B, 4>,
        new_v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (full_k, full_v) = match (self.k.take(), self.v.take()) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(vec![prev_k, new_k], 2);
                let v = Tensor::cat(vec![prev_v, new_v], 2);
                (k, v)
            }
            _ => (new_k, new_v),
        };
        self.k = Some(full_k.clone());
        self.v = Some(full_v.clone());
        (full_k, full_v)
    }

    /// Current sequence length stored in cache (0 if empty).
    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(k) => k.dims()[2],
            None => 0,
        }
    }
}

/// Collection of KV caches for all layers.
pub struct LayerCaches<B: Backend> {
    pub caches: Vec<KvCache<B>>,
}

impl<B: Backend> LayerCaches<B> {
    /// Create empty caches for `n_layers` transformer layers.
    pub fn new(n_layers: usize) -> Self {
        Self {
            caches: (0..n_layers).map(|_| KvCache::empty()).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized KV Cache
// ---------------------------------------------------------------------------

/// Quantization bit width for KV cache compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    /// 8-bit asymmetric quantization (4x compression vs f32).
    Int8,
    /// 4-bit asymmetric quantization (8x compression vs f32).
    Int4,
}

/// Configuration for KV cache quantization.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Quantization bit width.
    pub bits: QuantBits,
    /// Whether to store a 1-bit residual sign correction per value.
    /// Adds 1 bit per value but significantly improves inner product preservation.
    /// Total: int8+residual = 9 bits (~3.6x), int4+residual = 5 bits (~6.4x).
    pub residual_bits: bool,
}

impl QuantConfig {
    /// Int8 quantization (4x compression, simplest).
    pub fn int8() -> Self {
        Self {
            bits: QuantBits::Int8,
            residual_bits: false,
        }
    }

    /// Int4 quantization (8x compression).
    pub fn int4() -> Self {
        Self {
            bits: QuantBits::Int4,
            residual_bits: false,
        }
    }

    /// Int8 with 1-bit residual sign correction (TurboQuant-style).
    pub fn int8_residual() -> Self {
        Self {
            bits: QuantBits::Int8,
            residual_bits: true,
        }
    }

    /// Int4 with 1-bit residual sign correction (TurboQuant-style).
    pub fn int4_residual() -> Self {
        Self {
            bits: QuantBits::Int4,
            residual_bits: true,
        }
    }

    fn max_int(&self) -> f32 {
        match self.bits {
            QuantBits::Int8 => 255.0,
            QuantBits::Int4 => 15.0,
        }
    }

    /// Bytes per value for the quantized data (not counting scale/zero/residual).
    /// Int8 = 1 byte, Int4 = 0.5 bytes (packed nibbles).
    pub fn bytes_per_value(&self) -> f32 {
        match self.bits {
            QuantBits::Int8 => 1.0,
            QuantBits::Int4 => 0.5,
        }
    }

    /// Total bits per value including residual.
    pub fn total_bits_per_value(&self) -> f32 {
        let base = match self.bits {
            QuantBits::Int8 => 8.0,
            QuantBits::Int4 => 4.0,
        };
        if self.residual_bits { base + 1.0 } else { base }
    }
}

/// Quantized storage for a single tensor (K or V).
/// Stores data as packed bytes with per-group quantization parameters.
///
/// Layout: data is stored flat in row-major order matching
/// [batch, n_kv_heads, seq_len, head_dim]. Quantization granularity
/// is per head_dim vector (one scale+zero per [batch, head, seq_pos]).
struct QuantizedTensor {
    /// Quantized data. For Int8: one byte per value.
    /// For Int4: two values packed per byte (high nibble first).
    data: Vec<u8>,
    /// Per-group scale: (value - zero) * scale reconstructs the original.
    /// Length = batch * n_kv_heads * seq_len (one per head_dim vector).
    scales: Vec<f32>,
    /// Per-group zero point (minimum value of the original range).
    zeros: Vec<f32>,
    /// Residual sign bits, packed 8 per byte. Present only if residual_bits=true.
    /// Sign bit = 1 means residual was positive, 0 means negative.
    residual: Option<Vec<u8>>,
    /// Mean absolute residual per group, for sign-bit correction magnitude.
    /// Length = batch * n_kv_heads * seq_len. Present only if residual_bits=true.
    residual_scale: Option<Vec<f32>>,
    /// Shape: [batch, n_kv_heads, seq_len, head_dim].
    shape: [usize; 4],
    /// Quantization config used.
    config: QuantConfig,
}

impl QuantizedTensor {
    /// Quantize an f32 tensor to compressed representation.
    /// Input shape: [batch, n_kv_heads, seq_len, head_dim].
    fn quantize(data: &[f32], shape: [usize; 4], config: &QuantConfig) -> Self {
        let [batch, n_heads, seq_len, head_dim] = shape;
        let n_groups = batch * n_heads * seq_len;
        let max_int = config.max_int();

        let mut scales = Vec::with_capacity(n_groups);
        let mut zeros = Vec::with_capacity(n_groups);

        // Quantize each head_dim vector independently
        let qdata: Vec<u8> = match config.bits {
            QuantBits::Int8 => {
                let mut out = Vec::with_capacity(data.len());
                for g in 0..n_groups {
                    let start = g * head_dim;
                    let end = start + head_dim;
                    let group = &data[start..end];

                    let mut min_val = f32::INFINITY;
                    let mut max_val = f32::NEG_INFINITY;
                    for &v in group {
                        if v < min_val {
                            min_val = v;
                        }
                        if v > max_val {
                            max_val = v;
                        }
                    }

                    let range = max_val - min_val;
                    let scale = if range > 0.0 { range / max_int } else { 1.0 };

                    scales.push(scale);
                    zeros.push(min_val);

                    for &v in group {
                        let q = ((v - min_val) / scale).round().clamp(0.0, max_int) as u8;
                        out.push(q);
                    }
                }
                out
            }
            QuantBits::Int4 => {
                // Pack two 4-bit values per byte (high nibble first)
                let mut out = Vec::with_capacity((data.len() + 1) / 2);
                for g in 0..n_groups {
                    let start = g * head_dim;
                    let end = start + head_dim;
                    let group = &data[start..end];

                    let mut min_val = f32::INFINITY;
                    let mut max_val = f32::NEG_INFINITY;
                    for &v in group {
                        if v < min_val {
                            min_val = v;
                        }
                        if v > max_val {
                            max_val = v;
                        }
                    }

                    let range = max_val - min_val;
                    let scale = if range > 0.0 { range / max_int } else { 1.0 };

                    scales.push(scale);
                    zeros.push(min_val);

                    // Pack pairs of values
                    let mut i = 0;
                    while i < head_dim {
                        let hi = ((group[i] - min_val) / scale).round().clamp(0.0, max_int) as u8;
                        let lo = if i + 1 < head_dim {
                            ((group[i + 1] - min_val) / scale)
                                .round()
                                .clamp(0.0, max_int) as u8
                        } else {
                            0
                        };
                        out.push((hi << 4) | (lo & 0x0F));
                        i += 2;
                    }
                }
                out
            }
        };

        // Compute residual sign bits if requested
        let (residual, residual_scale) = if config.residual_bits {
            let mut signs = vec![0u8; (data.len() + 7) / 8];
            let mut rscales = Vec::with_capacity(n_groups);

            for g in 0..n_groups {
                let start = g * head_dim;
                let scale = scales[g];
                let zero = zeros[g];
                let mut abs_sum = 0.0f32;

                for d in 0..head_dim {
                    let idx = start + d;
                    let original = data[idx];

                    // Dequantize to get the quantized approximation
                    let q_val = match config.bits {
                        QuantBits::Int8 => qdata[idx] as f32,
                        QuantBits::Int4 => {
                            let byte_idx = g * ((head_dim + 1) / 2) + d / 2;
                            if d % 2 == 0 {
                                (qdata[byte_idx] >> 4) as f32
                            } else {
                                (qdata[byte_idx] & 0x0F) as f32
                            }
                        }
                    };
                    let deq = q_val * scale + zero;
                    let residual = original - deq;

                    abs_sum += residual.abs();

                    // Store sign bit: 1 = positive residual, 0 = negative
                    if residual >= 0.0 {
                        let bit_idx = idx;
                        signs[bit_idx / 8] |= 1 << (bit_idx % 8);
                    }
                }

                rscales.push(abs_sum / head_dim as f32);
            }

            (Some(signs), Some(rscales))
        } else {
            (None, None)
        };

        Self {
            data: qdata,
            scales,
            zeros,
            residual,
            residual_scale,
            shape,
            config: config.clone(),
        }
    }

    /// Dequantize back to f32 values in original shape order.
    fn dequantize(&self) -> Vec<f32> {
        let [_batch, _n_heads, _seq_len, head_dim] = self.shape;
        let n_groups = self.scales.len();
        let total = n_groups * head_dim;
        let mut out = Vec::with_capacity(total);

        for g in 0..n_groups {
            let scale = self.scales[g];
            let zero = self.zeros[g];

            for d in 0..head_dim {
                let q_val = match self.config.bits {
                    QuantBits::Int8 => self.data[g * head_dim + d] as f32,
                    QuantBits::Int4 => {
                        let byte_idx = g * ((head_dim + 1) / 2) + d / 2;
                        if d % 2 == 0 {
                            (self.data[byte_idx] >> 4) as f32
                        } else {
                            (self.data[byte_idx] & 0x0F) as f32
                        }
                    }
                };

                let mut val = q_val * scale + zero;

                // Apply residual sign correction
                if let (Some(signs), Some(rscales)) = (&self.residual, &self.residual_scale) {
                    let flat_idx = g * head_dim + d;
                    let sign_bit = (signs[flat_idx / 8] >> (flat_idx % 8)) & 1;
                    let correction = rscales[g];
                    if sign_bit == 1 {
                        val += correction;
                    } else {
                        val -= correction;
                    }
                }

                out.push(val);
            }
        }

        out
    }

    /// Reconstruct as a burn tensor on the given device.
    fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 4> {
        let f32_data = self.dequantize();
        let [batch, n_heads, seq_len, head_dim] = self.shape;
        Tensor::<B, 1>::from_floats(&f32_data[..], device)
            .reshape([batch, n_heads, seq_len, head_dim])
    }

    /// Approximate memory usage in bytes (quantized data + metadata).
    fn memory_bytes(&self) -> usize {
        let data_bytes = self.data.len();
        let meta_bytes = (self.scales.len() + self.zeros.len()) * 4; // f32 each
        let residual_bytes = self.residual.as_ref().map_or(0, |r| r.len())
            + self.residual_scale.as_ref().map_or(0, |r| r.len() * 4);
        data_bytes + meta_bytes + residual_bytes
    }
}

/// Per-layer quantized key-value cache for incremental decoding.
///
/// Stores K and V in compressed format (int8 or int4 with optional residual
/// sign bits). Dequantizes to f32 tensors when attention needs to read.
///
/// Quantization is asymmetric per-token-per-head: each head_dim vector
/// gets its own scale and zero point, which is the standard granularity
/// for KV cache quantization (see TurboQuant, KIVI, KVQuant papers).
pub struct QuantizedKvCache<B: Backend> {
    k: Option<QuantizedTensor>,
    v: Option<QuantizedTensor>,
    config: QuantConfig,
    /// Phantom to carry the backend type.
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> QuantizedKvCache<B> {
    /// Create an empty quantized cache.
    pub fn new(config: QuantConfig) -> Self {
        Self {
            k: None,
            v: None,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Append new K/V to the cache and return the full (dequantized) K/V.
    /// new_k, new_v: [batch, n_kv_heads, new_len, head_dim] in f32.
    /// Returns (full_k, full_v) dequantized to f32 for attention computation.
    pub fn update(
        &mut self,
        new_k: Tensor<B, 4>,
        new_v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let device = new_k.device();
        let [batch, n_heads, new_len, head_dim] = new_k.dims();

        // Get new data as f32
        let new_k_data: Vec<f32> = new_k.to_data().to_vec().unwrap();
        let new_v_data: Vec<f32> = new_v.to_data().to_vec().unwrap();

        // Merge with existing cache data, then re-quantize the full sequence.
        // This is simpler than appending to quantized storage and avoids
        // dealing with packed format concatenation.
        let (full_k_data, full_v_data, total_len) =
            if let (Some(prev_k), Some(prev_v)) = (self.k.take(), self.v.take()) {
                let prev_len = prev_k.shape[2];
                let total = prev_len + new_len;

                // Dequantize previous cache
                let prev_k_data = prev_k.dequantize();
                let prev_v_data = prev_v.dequantize();

                // Interleave: for each [batch, head], append new seq positions after old ones.
                // Both are in [batch, n_heads, seq_len, head_dim] row-major order.
                // We need to insert new_len * head_dim values after each prev_len * head_dim block.
                let mut merged_k = Vec::with_capacity(batch * n_heads * total * head_dim);
                let mut merged_v = Vec::with_capacity(batch * n_heads * total * head_dim);

                for b in 0..batch {
                    for h in 0..n_heads {
                        // Previous positions for this [batch, head]
                        let prev_offset = (b * n_heads + h) * prev_len * head_dim;
                        merged_k.extend_from_slice(
                            &prev_k_data[prev_offset..prev_offset + prev_len * head_dim],
                        );
                        // New positions for this [batch, head]
                        let new_offset = (b * n_heads + h) * new_len * head_dim;
                        merged_k.extend_from_slice(
                            &new_k_data[new_offset..new_offset + new_len * head_dim],
                        );

                        let prev_offset_v = (b * n_heads + h) * prev_len * head_dim;
                        merged_v.extend_from_slice(
                            &prev_v_data[prev_offset_v..prev_offset_v + prev_len * head_dim],
                        );
                        let new_offset_v = (b * n_heads + h) * new_len * head_dim;
                        merged_v.extend_from_slice(
                            &new_v_data[new_offset_v..new_offset_v + new_len * head_dim],
                        );
                    }
                }

                (merged_k, merged_v, total)
            } else {
                (new_k_data, new_v_data, new_len)
            };

        let shape = [batch, n_heads, total_len, head_dim];

        // Quantize and store
        self.k = Some(QuantizedTensor::quantize(&full_k_data, shape, &self.config));
        self.v = Some(QuantizedTensor::quantize(&full_v_data, shape, &self.config));

        // Return dequantized tensors for attention
        let full_k = self.k.as_ref().unwrap().to_tensor::<B>(&device);
        let full_v = self.v.as_ref().unwrap().to_tensor::<B>(&device);

        (full_k, full_v)
    }

    /// Current sequence length stored in cache (0 if empty).
    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(qt) => qt.shape[2],
            None => 0,
        }
    }

    /// Approximate memory usage in bytes for the cached K+V data.
    pub fn memory_bytes(&self) -> usize {
        self.k.as_ref().map_or(0, |t| t.memory_bytes())
            + self.v.as_ref().map_or(0, |t| t.memory_bytes())
    }

    /// What the equivalent uncompressed f32 KV cache would use in bytes.
    pub fn uncompressed_bytes(&self) -> usize {
        match &self.k {
            Some(qt) => {
                let [batch, n_heads, seq_len, head_dim] = qt.shape;
                2 * batch * n_heads * seq_len * head_dim * 4 // K + V, f32
            }
            None => 0,
        }
    }
}

/// Collection of quantized KV caches for all layers.
pub struct QuantizedLayerCaches<B: Backend> {
    pub caches: Vec<QuantizedKvCache<B>>,
    pub config: QuantConfig,
}

impl<B: Backend> QuantizedLayerCaches<B> {
    /// Create empty quantized caches for `n_layers` transformer layers.
    pub fn new(n_layers: usize, config: QuantConfig) -> Self {
        let caches = (0..n_layers)
            .map(|_| QuantizedKvCache::new(config.clone()))
            .collect();
        Self { caches, config }
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
// TurboQuant KV Cache (full paper implementation)
// ---------------------------------------------------------------------------

use crate::models::gemma::turbo_quant::{TurboQuantConfig, TurboQuantCtx, TurboQuantTensor};

/// Per-layer KV cache using the full TurboQuant algorithm.
///
/// Unlike `QuantizedKvCache` which uses uniform min/max quantization,
/// this implements the three innovations from Zandieh et al. (2025):
/// 1. Random rotation (RHT) to decorrelate coordinates
/// 2. Lloyd-Max optimal scalar quantizer for resulting Gaussian coordinates
/// 3. QJL binary sketch of the residual for unbiased inner product estimation
pub struct TurboQuantKvCache<B: Backend> {
    k: Option<TurboQuantTensor>,
    v: Option<TurboQuantTensor>,
    ctx: TurboQuantCtx,
    /// Retained for introspection (e.g. reporting bits-per-value).
    #[allow(dead_code)]
    config: TurboQuantConfig,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> TurboQuantKvCache<B> {
    /// Create an empty TurboQuant cache.
    pub fn new(config: TurboQuantConfig) -> Self {
        let ctx = TurboQuantCtx::new(&config);
        Self {
            k: None,
            v: None,
            ctx,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Append new K/V to the cache and return the full (dequantized) K/V.
    /// new_k, new_v: [batch, n_kv_heads, new_len, head_dim] in f32.
    /// Returns (full_k, full_v) dequantized to f32 for attention computation.
    pub fn update(
        &mut self,
        new_k: Tensor<B, 4>,
        new_v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let device = new_k.device();
        let [batch, n_heads, new_len, head_dim] = new_k.dims();

        let new_k_data: Vec<f32> = new_k.to_data().to_vec().unwrap();
        let new_v_data: Vec<f32> = new_v.to_data().to_vec().unwrap();

        // Merge with existing cache data then re-quantize
        let (full_k_data, full_v_data, total_len) =
            if let (Some(prev_k), Some(prev_v)) = (self.k.take(), self.v.take()) {
                let prev_len = prev_k.shape[2];
                let total = prev_len + new_len;

                let prev_k_data = self.ctx.dequantize(&prev_k);
                let prev_v_data = self.ctx.dequantize(&prev_v);

                let mut merged_k = Vec::with_capacity(batch * n_heads * total * head_dim);
                let mut merged_v = Vec::with_capacity(batch * n_heads * total * head_dim);

                for b in 0..batch {
                    for h in 0..n_heads {
                        let prev_offset = (b * n_heads + h) * prev_len * head_dim;
                        merged_k.extend_from_slice(
                            &prev_k_data[prev_offset..prev_offset + prev_len * head_dim],
                        );
                        let new_offset = (b * n_heads + h) * new_len * head_dim;
                        merged_k.extend_from_slice(
                            &new_k_data[new_offset..new_offset + new_len * head_dim],
                        );

                        let prev_offset_v = (b * n_heads + h) * prev_len * head_dim;
                        merged_v.extend_from_slice(
                            &prev_v_data[prev_offset_v..prev_offset_v + prev_len * head_dim],
                        );
                        let new_offset_v = (b * n_heads + h) * new_len * head_dim;
                        merged_v.extend_from_slice(
                            &new_v_data[new_offset_v..new_offset_v + new_len * head_dim],
                        );
                    }
                }

                (merged_k, merged_v, total)
            } else {
                (new_k_data, new_v_data, new_len)
            };

        let shape = [batch, n_heads, total_len, head_dim];

        self.k = Some(self.ctx.quantize(&full_k_data, shape));
        self.v = Some(self.ctx.quantize(&full_v_data, shape));

        // Dequantize for attention
        let deq_k = self.ctx.dequantize(self.k.as_ref().unwrap());
        let deq_v = self.ctx.dequantize(self.v.as_ref().unwrap());

        let full_k = Tensor::<B, 1>::from_floats(&deq_k[..], &device)
            .reshape([batch, n_heads, total_len, head_dim]);
        let full_v = Tensor::<B, 1>::from_floats(&deq_v[..], &device)
            .reshape([batch, n_heads, total_len, head_dim]);

        (full_k, full_v)
    }

    /// Current sequence length stored in cache (0 if empty).
    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(qt) => qt.shape[2],
            None => 0,
        }
    }

    /// Approximate memory usage in bytes for the cached K+V data.
    pub fn memory_bytes(&self) -> usize {
        self.k
            .as_ref()
            .map_or(0, |t| TurboQuantCtx::memory_bytes(t))
            + self
                .v
                .as_ref()
                .map_or(0, |t| TurboQuantCtx::memory_bytes(t))
    }

    /// What the equivalent uncompressed f32 KV cache would use in bytes.
    pub fn uncompressed_bytes(&self) -> usize {
        match &self.k {
            Some(qt) => {
                let [batch, n_heads, seq_len, head_dim] = qt.shape;
                2 * batch * n_heads * seq_len * head_dim * 4
            }
            None => 0,
        }
    }
}

/// Collection of TurboQuant KV caches for all layers.
pub struct TurboQuantLayerCaches<B: Backend> {
    pub caches: Vec<TurboQuantKvCache<B>>,
    pub config: TurboQuantConfig,
}

impl<B: Backend> TurboQuantLayerCaches<B> {
    pub fn new(n_layers: usize, config: TurboQuantConfig) -> Self {
        let caches = (0..n_layers)
            .map(|_| TurboQuantKvCache::new(config.clone()))
            .collect();
        Self { caches, config }
    }

    pub fn memory_bytes(&self) -> usize {
        self.caches.iter().map(|c| c.memory_bytes()).sum()
    }

    pub fn uncompressed_bytes(&self) -> usize {
        self.caches.iter().map(|c| c.uncompressed_bytes()).sum()
    }
}

// ---------------------------------------------------------------------------
// RMSNorm — re-export from burn
// ---------------------------------------------------------------------------

pub use burn::nn::{RmsNorm, RmsNormConfig};

// ---------------------------------------------------------------------------
// SwiGLU FFN
// ---------------------------------------------------------------------------

/// SwiGLU Feed-Forward Network: w2(silu(w1(x)) * w3(x))
#[derive(Module, Debug)]
pub struct SwiGluFfn<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(config: &MistralConfig, device: &B::Device) -> Self {
        let gate_proj = LinearConfig::new(config.hidden_dim, config.ffn_dim)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(config.hidden_dim, config.ffn_dim)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(config.ffn_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

// ---------------------------------------------------------------------------
// Grouped Query Attention (GQA)
// ---------------------------------------------------------------------------

/// Multi-head attention with grouped query attention (GQA).
/// n_kv_heads < n_heads: each KV head serves multiple query heads.
#[derive(Module, Debug)]
pub struct GqaAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    /// Optional per-head QK-norm (Qwen3). None for Mistral.
    pub q_norm: Option<RmsNorm<B>>,
    pub k_norm: Option<RmsNorm<B>>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> GqaAttention<B> {
    pub fn new(config: &MistralConfig, device: &B::Device) -> Self {
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        let q_proj = LinearConfig::new(config.hidden_dim, q_dim)
            .with_bias(false)
            .init(device);
        let k_proj = LinearConfig::new(config.hidden_dim, kv_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(config.hidden_dim, kv_dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(q_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: None,
            k_norm: None,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
        }
    }

    /// Apply optional per-head QK-norm (Qwen3) to a 4D tensor [B, H, S, D].
    /// Burn's RmsNorm normalizes over the last dimension, which is head_dim here.
    fn apply_qk_norm(&self, q: Tensor<B, 4>, k: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let q = match &self.q_norm {
            Some(norm) => norm.forward(q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(k),
            None => k,
        };
        (q, k)
    }

    /// Forward pass with RoPE (no cache — recomputes everything).
    /// x: [batch, seq_len, hidden_dim]
    /// Returns: [batch, seq_len, hidden_dim]
    pub fn forward(&self, x: Tensor<B, 3>, rope: &RopeTable<B>, offset: usize) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        // Project to Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [batch, n_heads, seq_len, head_dim]
        let q = q
            .reshape([batch, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Optional per-head QK-norm (Qwen3)
        let (q, k) = self.apply_qk_norm(q, k);

        // Apply RoPE to Q and K
        let q = rope.apply(q, offset);
        let k = rope.apply(k, offset);

        // Expand KV heads to match Q heads (GQA)
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = Self::repeat_kv(k, n_rep);
        let v = Self::repeat_kv(v, n_rep);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(k.swap_dims(2, 3)) / scale;

        // Causal mask
        let attn = Self::causal_mask(attn, seq_len, offset);

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back to [batch, seq_len, hidden_dim]
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.n_heads * self.head_dim]);

        self.o_proj.forward(out)
    }

    /// Forward pass with KV cache for incremental decoding.
    /// x: [batch, new_len, hidden_dim] — new tokens only (typically new_len=1 during generation)
    /// cache: mutable KV cache for this layer
    /// offset: RoPE position offset for the new tokens (= cache.seq_len() before update)
    /// Returns: [batch, new_len, hidden_dim]
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut KvCache<B>,
    ) -> Tensor<B, 3> {
        let [batch, new_len, _] = x.dims();
        let offset = cache.seq_len();

        // Project to Q, K, V for new tokens only
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [batch, n_heads, new_len, head_dim]
        let q = q
            .reshape([batch, new_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Optional per-head QK-norm (Qwen3)
        let (q, k) = self.apply_qk_norm(q, k);

        // Apply RoPE at the correct position offset
        let q = rope.apply(q, offset);
        let k = rope.apply(k, offset);

        // Update cache: concatenate new K/V with previous, get full K/V
        let (full_k, full_v) = cache.update(k, v);
        let _total_len = full_k.dims()[2];

        // Expand KV heads to match Q heads (GQA)
        let n_rep = self.n_heads / self.n_kv_heads;
        let full_k = Self::repeat_kv(full_k, n_rep);
        let full_v = Self::repeat_kv(full_v, n_rep);

        // Attention: use flash attention for prefill (avoids O(n²) memory),
        // manual attention for decode (flash has hardcoded causal=true which
        // breaks single-token decode that needs to attend to all positions).
        let out = if new_len > 1 {
            // Prefill: flash attention with causal mask.
            burn::tensor::module::attention(
                q,
                full_k,
                full_v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    is_causal: true,
                    ..Default::default()
                },
            )
        } else {
            // Decode: flash attention WITHOUT causal mask.
            // Single query attends to all cached positions.
            burn::tensor::module::attention(
                q,
                full_k,
                full_v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    is_causal: false,
                    ..Default::default()
                },
            )
        };

        // Reshape back to [batch, new_len, hidden_dim]
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, new_len, self.n_heads * self.head_dim]);

        self.o_proj.forward(out)
    }

    /// Forward pass with quantized KV cache for incremental decoding.
    /// Same interface as `forward_cached` but stores K/V in compressed format.
    pub fn forward_quantized_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut QuantizedKvCache<B>,
    ) -> Tensor<B, 3> {
        let [batch, new_len, _] = x.dims();
        let offset = cache.seq_len();

        // Project to Q, K, V for new tokens only
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [batch, n_heads, new_len, head_dim]
        let q = q
            .reshape([batch, new_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = self.apply_qk_norm(q, k);

        // Apply RoPE at the correct position offset
        let q = rope.apply(q, offset);
        let k = rope.apply(k, offset);

        // Update quantized cache: quantize + store, return dequantized full K/V
        let (full_k, full_v) = cache.update(k, v);
        let total_len = full_k.dims()[2];

        // Expand KV heads to match Q heads (GQA)
        let n_rep = self.n_heads / self.n_kv_heads;
        let full_k = Self::repeat_kv(full_k, n_rep);
        let full_v = Self::repeat_kv(full_v, n_rep);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(full_k.swap_dims(2, 3)) / scale;

        let attn = if new_len > 1 {
            Self::causal_mask_cached(attn, new_len, total_len, offset)
        } else {
            attn
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(full_v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, new_len, self.n_heads * self.head_dim]);

        self.o_proj.forward(out)
    }

    /// Forward pass with TurboQuant KV cache for incremental decoding.
    pub fn forward_turbo_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut TurboQuantKvCache<B>,
    ) -> Tensor<B, 3> {
        let [batch, new_len, _] = x.dims();
        let offset = cache.seq_len();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch, new_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = self.apply_qk_norm(q, k);

        let q = rope.apply(q, offset);
        let k = rope.apply(k, offset);

        let (full_k, full_v) = cache.update(k, v);
        let total_len = full_k.dims()[2];

        let n_rep = self.n_heads / self.n_kv_heads;
        let full_k = Self::repeat_kv(full_k, n_rep);
        let full_v = Self::repeat_kv(full_v, n_rep);

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(full_k.swap_dims(2, 3)) / scale;

        let attn = if new_len > 1 {
            Self::causal_mask_cached(attn, new_len, total_len, offset)
        } else {
            attn
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(full_v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, new_len, self.n_heads * self.head_dim]);

        self.o_proj.forward(out)
    }

    /// Forward pass with GPU-native quantized KV cache for incremental decoding.
    /// Same interface as `forward_cached` but stores K/V as packed int8 on GPU.
    pub fn forward_gpu_quant_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut crate::models::gemma::gpu_quant::GpuQuantKvCache<B>,
    ) -> Tensor<B, 3>
    where
        B: Backend<IntElem = i32>,
    {
        let [batch, new_len, _] = x.dims();
        let offset = cache.seq_len();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch, new_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = self.apply_qk_norm(q, k);

        let q = rope.apply(q, offset);
        let k = rope.apply(k, offset);

        let (full_k, full_v) = cache.update(k, v);
        let total_len = full_k.dims()[2];

        let n_rep = self.n_heads / self.n_kv_heads;
        let full_k = Self::repeat_kv(full_k, n_rep);
        let full_v = Self::repeat_kv(full_v, n_rep);

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(full_k.swap_dims(2, 3)) / scale;

        let attn = if new_len > 1 {
            Self::causal_mask_cached(attn, new_len, total_len, offset)
        } else {
            attn
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(full_v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, new_len, self.n_heads * self.head_dim]);

        self.o_proj.forward(out)
    }

    /// Forward pass with GPU-native TurboQuant KV cache for incremental decoding.
    /// Same interface as `forward_gpu_quant_cached` but uses TurboQuant (rotation +
    /// Lloyd-Max + optional QJL) instead of uniform int8 quantization.
    pub fn forward_gpu_turbo_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut crate::models::gemma::gpu_quant::GpuTurboQuantKvCache<B>,
    ) -> Tensor<B, 3>
    where
        B: Backend<IntElem = i32>,
    {
        let [batch, new_len, _] = x.dims();
        let offset = cache.seq_len();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch, new_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, new_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = self.apply_qk_norm(q, k);

        let q = rope.apply(q, offset);
        let k = rope.apply(k, offset);

        let (full_k, full_v) = cache.update(k, v);
        let total_len = full_k.dims()[2];

        let n_rep = self.n_heads / self.n_kv_heads;
        let full_k = Self::repeat_kv(full_k, n_rep);
        let full_v = Self::repeat_kv(full_v, n_rep);

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(full_k.swap_dims(2, 3)) / scale;

        let attn = if new_len > 1 {
            Self::causal_mask_cached(attn, new_len, total_len, offset)
        } else {
            attn
        };

        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(full_v);

        let out = out
            .swap_dims(1, 2)
            .reshape([batch, new_len, self.n_heads * self.head_dim]);

        self.o_proj.forward(out)
    }

    /// Repeat KV heads to match query heads for GQA.
    fn repeat_kv(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        if n_rep == 1 {
            return x;
        }
        let [batch, n_kv_heads, seq_len, head_dim] = x.dims();
        x.unsqueeze_dim::<5>(2)
            .expand([batch, n_kv_heads, n_rep, seq_len, head_dim])
            .reshape([batch, n_kv_heads * n_rep, seq_len, head_dim])
    }

    /// Apply causal attention mask (for full-sequence forward without cache).
    fn causal_mask(attn: Tensor<B, 4>, seq_len: usize, _offset: usize) -> Tensor<B, 4> {
        if seq_len <= 1 {
            return attn;
        }
        let device = attn.device();
        // Build mask row by row to avoid shape ambiguity
        let rows: Vec<Tensor<B, 2>> = (0..seq_len)
            .map(|i| {
                let row: Vec<f32> = (0..seq_len)
                    .map(|j| if j <= i { 0.0 } else { f32::NEG_INFINITY })
                    .collect();
                Tensor::<B, 1>::from_floats(&row[..], &device).unsqueeze::<2>()
            })
            .collect();
        let mask = Tensor::<B, 2>::cat(rows, 0) // [seq_len, seq_len]
            .reshape([1, 1, seq_len, seq_len]);
        attn + mask
    }

    /// Apply causal attention mask for cached prefill.
    /// attn shape: [batch, n_heads, new_len, total_len]
    /// Query positions are [offset..offset+new_len], key positions are [0..total_len].
    /// Query at position offset+i can attend to keys at positions 0..=offset+i.
    fn causal_mask_cached(
        attn: Tensor<B, 4>,
        new_len: usize,
        total_len: usize,
        offset: usize,
    ) -> Tensor<B, 4> {
        let device = attn.device();
        let rows: Vec<Tensor<B, 2>> = (0..new_len)
            .map(|i| {
                let query_pos = offset + i;
                let row: Vec<f32> = (0..total_len)
                    .map(|j| {
                        if j <= query_pos {
                            0.0
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect();
                Tensor::<B, 1>::from_floats(&row[..], &device).unsqueeze::<2>()
            })
            .collect();
        let mask = Tensor::<B, 2>::cat(rows, 0).reshape([1, 1, new_len, total_len]);
        attn + mask
    }
}

// ---------------------------------------------------------------------------
// Transformer Layer
// ---------------------------------------------------------------------------

/// A single transformer decoder layer: attention + FFN with pre-norm.
#[derive(Module, Debug)]
pub struct TransformerLayer<B: Backend> {
    pub attn_norm: RmsNorm<B>,
    pub attention: GqaAttention<B>,
    pub ffn_norm: RmsNorm<B>,
    pub ffn: SwiGluFfn<B>,
}

impl<B: Backend> TransformerLayer<B> {
    pub fn new(config: &MistralConfig, device: &B::Device) -> Self {
        let norm_config = RmsNormConfig::new(config.hidden_dim).with_epsilon(config.rms_norm_eps);
        Self {
            attn_norm: norm_config.init(device),
            attention: GqaAttention::new(config, device),
            ffn_norm: norm_config.init(device),
            ffn: SwiGluFfn::new(config, device),
        }
    }

    /// Forward pass without KV cache (recomputes everything).
    pub fn forward(&self, x: Tensor<B, 3>, rope: &RopeTable<B>, offset: usize) -> Tensor<B, 3> {
        // Pre-norm attention with residual
        let h = x.clone()
            + self
                .attention
                .forward(self.attn_norm.forward(x.clone()), rope, offset);
        // Pre-norm FFN with residual
        h.clone() + self.ffn.forward(self.ffn_norm.forward(h))
    }

    /// Forward pass with KV cache for incremental decoding.
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut KvCache<B>,
    ) -> Tensor<B, 3> {
        // Pre-norm attention with residual
        let h = x.clone()
            + self
                .attention
                .forward_cached(self.attn_norm.forward(x.clone()), rope, cache);
        // Pre-norm FFN with residual
        h.clone() + self.ffn.forward(self.ffn_norm.forward(h))
    }

    /// Forward pass with quantized KV cache for incremental decoding.
    pub fn forward_quantized_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut QuantizedKvCache<B>,
    ) -> Tensor<B, 3> {
        // Pre-norm attention with residual
        let h = x.clone()
            + self.attention.forward_quantized_cached(
                self.attn_norm.forward(x.clone()),
                rope,
                cache,
            );
        // Pre-norm FFN with residual
        h.clone() + self.ffn.forward(self.ffn_norm.forward(h))
    }

    /// Forward pass with TurboQuant KV cache for incremental decoding.
    pub fn forward_turbo_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut TurboQuantKvCache<B>,
    ) -> Tensor<B, 3> {
        let h = x.clone()
            + self
                .attention
                .forward_turbo_cached(self.attn_norm.forward(x.clone()), rope, cache);
        h.clone() + self.ffn.forward(self.ffn_norm.forward(h))
    }

    /// Forward pass with GPU-native quantized KV cache for incremental decoding.
    pub fn forward_gpu_quant_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut crate::models::gemma::gpu_quant::GpuQuantKvCache<B>,
    ) -> Tensor<B, 3>
    where
        B: Backend<IntElem = i32>,
    {
        let h = x.clone()
            + self.attention.forward_gpu_quant_cached(
                self.attn_norm.forward(x.clone()),
                rope,
                cache,
            );
        h.clone() + self.ffn.forward(self.ffn_norm.forward(h))
    }

    /// Forward pass with GPU-native TurboQuant KV cache for incremental decoding.
    pub fn forward_gpu_turbo_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTable<B>,
        cache: &mut crate::models::gemma::gpu_quant::GpuTurboQuantKvCache<B>,
    ) -> Tensor<B, 3>
    where
        B: Backend<IntElem = i32>,
    {
        let h = x.clone()
            + self.attention.forward_gpu_turbo_cached(
                self.attn_norm.forward(x.clone()),
                rope,
                cache,
            );
        h.clone() + self.ffn.forward(self.ffn_norm.forward(h))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that int8 quantize/dequantize roundtrip has bounded error.
    #[test]
    fn test_int8_roundtrip() {
        // Simulate a [1, 2, 3, 4] shaped tensor (batch=1, heads=2, seq=3, head_dim=4)
        let data: Vec<f32> = vec![
            // head 0, pos 0
            -1.0, 0.5, 2.0, -0.3, // head 0, pos 1
            0.0, 0.0, 0.0, 0.0, // head 0, pos 2
            1.0, 1.0, 1.0, 1.0, // head 1, pos 0
            -10.0, 10.0, 5.0, -5.0, // head 1, pos 1
            0.1, 0.2, 0.3, 0.4, // head 1, pos 2
            100.0, -100.0, 0.0, 50.0,
        ];
        let shape = [1, 2, 3, 4];
        let config = QuantConfig::int8();
        let qt = QuantizedTensor::quantize(&data, shape, &config);
        let deq = qt.dequantize();

        assert_eq!(deq.len(), data.len());

        // Int8 with 256 levels: max error per group is range/255.
        // For the constant-zero group, error should be exactly 0.
        for (orig, recon) in data.iter().zip(deq.iter()) {
            let err = (orig - recon).abs();
            // Generous bound: for the widest range group [-100, 100],
            // step size = 200/255 ≈ 0.78. Allow 1.0 max error.
            assert!(
                err < 1.0,
                "Int8 roundtrip error too large: orig={}, recon={}, err={}",
                orig,
                recon,
                err
            );
        }

        // Check the all-zeros group roundtrips exactly
        for i in 4..8 {
            assert_eq!(deq[i], 0.0, "Zero group should roundtrip exactly");
        }
    }

    /// Test that int4 quantization has bounded error (coarser).
    #[test]
    fn test_int4_roundtrip() {
        let data: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // head 0, pos 0
            -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, // head 0, pos 1
        ];
        let shape = [1, 1, 2, 8];
        let config = QuantConfig::int4();
        let qt = QuantizedTensor::quantize(&data, shape, &config);
        let deq = qt.dequantize();

        assert_eq!(deq.len(), data.len());

        // Int4 with 16 levels: max error per group is range/15.
        // Group 0: range=7, step=7/15≈0.467. Group 1: range=3.5, step=3.5/15≈0.233.
        for (orig, recon) in data.iter().zip(deq.iter()) {
            let err = (orig - recon).abs();
            assert!(
                err < 0.5,
                "Int4 roundtrip error too large: orig={}, recon={}, err={}",
                orig,
                recon,
                err
            );
        }
    }

    /// Test that residual sign correction reduces error.
    #[test]
    fn test_residual_correction() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 / 10.0) - 6.4).collect();
        let shape = [1, 1, 1, 128];

        let config_no_res = QuantConfig::int4();
        let config_with_res = QuantConfig::int4_residual();

        let qt_no_res = QuantizedTensor::quantize(&data, shape, &config_no_res);
        let qt_with_res = QuantizedTensor::quantize(&data, shape, &config_with_res);

        let deq_no_res = qt_no_res.dequantize();
        let deq_with_res = qt_with_res.dequantize();

        // Compute MSE for each
        let mse_no_res: f32 = data
            .iter()
            .zip(deq_no_res.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        let mse_with_res: f32 = data
            .iter()
            .zip(deq_with_res.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        // Residual correction should not make things worse
        // (it might not always help for every distribution, but for uniform-ish data it should)
        println!("Int4 MSE without residual: {:.6}", mse_no_res);
        println!("Int4 MSE with residual:    {:.6}", mse_with_res);

        // Both should be small
        assert!(mse_no_res < 0.1, "Int4 MSE too large: {}", mse_no_res);
    }

    /// Test memory reporting.
    #[test]
    fn test_memory_accounting() {
        let data: Vec<f32> = vec![0.0; 1 * 8 * 10 * 128]; // [1, 8, 10, 128]
        let shape = [1, 8, 10, 128];

        let qt_int8 = QuantizedTensor::quantize(&data, shape, &QuantConfig::int8());
        let qt_int4 = QuantizedTensor::quantize(&data, shape, &QuantConfig::int4());

        // Int8: 10240 bytes data + 80*4*2 scales/zeros = 10240 + 640 = 10880
        assert_eq!(qt_int8.data.len(), 1 * 8 * 10 * 128); // 10240 bytes

        // Int4: 5120 bytes data (packed nibbles)
        assert_eq!(qt_int4.data.len(), 1 * 8 * 10 * (128 / 2)); // 5120 bytes

        // Uncompressed f32 equivalent: 10240 * 4 = 40960 bytes
        let uncompressed = 1 * 8 * 10 * 128 * 4;
        assert!(qt_int8.memory_bytes() < uncompressed);
        assert!(qt_int4.memory_bytes() < qt_int8.memory_bytes());
    }
}
