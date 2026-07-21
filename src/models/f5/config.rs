//! F5-TTS v1 Base architecture config. Values are the well-known F5TTS_v1_Base
//! settings; codex is dumping the safetensors header to confirm exact dims +
//! key names (see /tmp/codex_outputs/f5_arch.md), and this will be pinned to
//! that ground truth.

/// The flow-matching DiT backbone (single-stream, AdaLN-zero, RoPE).
#[derive(Debug, Clone)]
pub struct F5Config {
    pub dim: usize,          // hidden width
    pub depth: usize,        // transformer blocks
    pub heads: usize,        // attention heads
    pub ff_mult: usize,      // ff_dim = dim * ff_mult
    pub text_dim: usize,     // char-embedding width
    pub conv_layers: usize,  // ConvNeXt-V2 text-refinement blocks
    pub text_vocab: usize,   // char vocab size (from vocab.txt)
    pub mel: MelConfig,
    pub cfm: CfmConfig,
}

#[derive(Debug, Clone)]
pub struct MelConfig {
    pub n_mel: usize,
    pub sample_rate: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_fft: usize,
}

#[derive(Debug, Clone)]
pub struct CfmConfig {
    pub nfe: usize,        // function evaluations (sampling steps)
    pub sway_coef: f64,    // sway sampling coefficient (F5 default -1.0)
    pub cfg_strength: f64, // classifier-free guidance strength
}

impl F5Config {
    /// F5TTS_v1_Base.
    pub fn v1_base() -> Self {
        Self {
            dim: 1024,
            depth: 22,
            heads: 16,
            ff_mult: 2,
            text_dim: 512,
            conv_layers: 4,
            text_vocab: 2546,
            mel: MelConfig { n_mel: 100, sample_rate: 24000, hop_length: 256, win_length: 1024, n_fft: 1024 },
            cfm: CfmConfig { nfe: 32, sway_coef: -1.0, cfg_strength: 2.0 },
        }
    }
    pub fn head_dim(&self) -> usize { self.dim / self.heads }
    pub fn ff_dim(&self) -> usize { self.dim * self.ff_mult }
}
