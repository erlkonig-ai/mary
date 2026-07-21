//! Mimi neural audio codec (Kyutai Moshi `Config::v0_1`) — dimensions measured
//! from the ungated checkpoint `kyutai/moshiko-pytorch-bf16` /
//! `tokenizer-e351c8d8-checkpoint125.safetensors` (moshi state_dict), not the
//! paper.
//!
//! SEANet: dimension 512, n_filters 64, ratios [8,6,5,4] (outermost-first),
//! kernel 7, residual_kernel 3, n_residual_layers 1, dilation_base 2,
//! compress 2, NO LSTM, all causal. Enc + dec transformer bottleneck: d_model
//! 512, 8 heads × 64, 8 layers, ffn 2048, sliding window 250, RoPE θ=10000,
//! GELU, LayerScale. Quantizer: proj 512→256→512, codebook 2048, split-RVQ =
//! 1 semantic + 7 acoustic (dep_q=8), summed. 24 kHz, 12.5 Hz frames, 1920×.

/// Full number of codebooks Mimi emits per 12.5 Hz frame (1 semantic + 7
/// acoustic; the LM's `dep_q`).
pub const NUM_CODEBOOKS: usize = 8;
/// Acoustic quantizers actually used in the residual chain (rvq_rest has 31
/// trained layers; only the first `NUM_CODEBOOKS-1` are active).
pub const N_ACOUSTIC: usize = NUM_CODEBOOKS - 1;

pub const CODEBOOK_SIZE: usize = 2048;
/// Per-quantizer embedding dim (post input_proj, pre output_proj).
pub const CODE_DIM: usize = 256;
pub const HIDDEN: usize = 512; // SEANet dimension == transformer width

/// SEANet strided ratios in **config order** (`ratios=[8,6,5,4]`). The encoder
/// applies them REVERSED (innermost first): its downsample strides run
/// [4,5,6,8] (k=2r → k8 s4, k10 s5, k12 s6, k16 s8). The decoder applies them in
/// config order [8,6,5,4] (upsample k16 s8 first). Use [`ENC_RATIOS`] /
/// [`DEC_RATIOS`] rather than this directly.
pub const RATIOS: [usize; 4] = [8, 6, 5, 4];
/// Encoder downsample strides, outermost-first (config order reversed).
pub const ENC_RATIOS: [usize; 4] = [4, 5, 6, 8];
/// Decoder upsample strides, outermost-first (config order).
pub const DEC_RATIOS: [usize; 4] = [8, 6, 5, 4];

// Transformer bottleneck (identical geometry in encoder and decoder).
pub const TR_LAYERS: usize = 8;
pub const TR_HEADS: usize = 8;
pub const TR_HEAD_DIM: usize = 64;
pub const TR_INTER: usize = 2048;
pub const TR_EPS: f64 = 1e-5;
pub const TR_ROPE_THETA: f64 = 10_000.0;
/// Causal sliding-window context (moshi applies it — `causal=True context=250`).
pub const TR_WINDOW: usize = 250;

/// Samples per 12.5 Hz frame at 24 kHz: 8·6·5·4 (=960) · 2 (down/upsample) = 1920.
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const SAMPLE_RATE: u32 = 24_000;
