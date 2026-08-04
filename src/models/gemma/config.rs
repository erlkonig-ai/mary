//! Model configuration for Mistral-family decoders.

/// Configuration for a Mistral-family decoder model.
#[derive(Debug, Clone)]
pub struct MistralConfig {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Hidden dimension (model width).
    pub hidden_dim: usize,
    /// FFN intermediate dimension.
    pub ffn_dim: usize,
    /// Number of query attention heads.
    pub n_heads: usize,
    /// Number of key-value attention heads (for GQA).
    pub n_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RoPE base frequency.
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// Base training sequence length (before YaRN extension).
    pub max_seq_len: usize,
    /// Extended sequence length with YaRN (None = no YaRN, use max_seq_len).
    pub yarn_max_seq_len: Option<usize>,
    /// Whether lm_head shares weights with the embedding layer.
    pub tie_word_embeddings: bool,
    /// HuggingFace model ID for downloading weights.
    pub model_id: &'static str,
    /// Number of safetensors shards.
    pub n_shards: usize,
    /// Whether the model uses per-head QK-norm (Qwen3). False for Mistral.
    pub qk_norm: bool,
}

impl MistralConfig {
    /// Ministral 3B configuration.
    pub fn ministral_3b() -> Self {
        Self {
            n_layers: 26,
            hidden_dim: 3072,
            ffn_dim: 9216, // corrected from 8192
            n_heads: 32,   // corrected from 24
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 131_072,
            rope_theta: 1e6, // corrected from 1e9
            rms_norm_eps: 1e-5,
            max_seq_len: 16384,
            yarn_max_seq_len: Some(262144),
            tie_word_embeddings: true,
            model_id: "mistralai/Ministral-3-3B-Instruct-2512-BF16",
            n_shards: 2,
            qk_norm: false,
        }
    }

    /// Ministral 8B configuration.
    pub fn ministral_8b() -> Self {
        Self {
            n_layers: 34,
            hidden_dim: 4096,
            ffn_dim: 12288,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 131_072,
            rope_theta: 1e9,
            rms_norm_eps: 1e-5,
            max_seq_len: 16384,
            yarn_max_seq_len: Some(262144),
            tie_word_embeddings: true,
            model_id: "mistralai/Ministral-3-8B-Instruct-2506",
            n_shards: 4,
            qk_norm: false,
        }
    }

    /// Ministral 14B configuration.
    pub fn ministral_14b() -> Self {
        Self {
            n_layers: 40,
            hidden_dim: 5120,
            ffn_dim: 16384,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 131_072,
            rope_theta: 1e9,
            rms_norm_eps: 1e-5,
            max_seq_len: 16384,
            yarn_max_seq_len: Some(262144),
            tie_word_embeddings: false,
            model_id: "mistralai/Ministral-3-14B-Instruct-2512",
            n_shards: 4,
            qk_norm: false,
        }
    }

    /// DeepSeek-R1-0528-Qwen3-8B configuration (Qwen3 architecture with QK-norm).
    pub fn deepseek_r1_qwen3_8b() -> Self {
        Self {
            n_layers: 36,
            hidden_dim: 4096,
            ffn_dim: 12288,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 151_936,
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
            max_seq_len: 32768,
            yarn_max_seq_len: Some(131072),
            tie_word_embeddings: false,
            model_id: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            n_shards: 2,
            qk_norm: true,
        }
    }

    /// Short model name for TribleSpace storage (max 32 bytes).
    /// Strips the HuggingFace org prefix and BF16/FP8 suffix.
    pub fn short_name(&self) -> &'static str {
        match self.model_id {
            "mistralai/Ministral-3-3B-Instruct-2512-BF16" => "ministral-3b",
            "mistralai/Ministral-3-8B-Instruct-2506" => "ministral-8b",
            "mistralai/Ministral-3-14B-Instruct-2512" => "ministral-14b",
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" => "deepseek-r1-qwen3-8b",
            other => {
                // Fallback: take after last '/', truncate to 31 bytes
                let s = other.rsplit('/').next().unwrap_or(other);
                if s.len() <= 31 {
                    s
                } else {
                    &s[..31]
                }
            }
        }
    }

    /// Effective maximum sequence length (with YaRN if available).
    pub fn effective_max_seq_len(&self) -> usize {
        self.yarn_max_seq_len.unwrap_or(self.max_seq_len)
    }

    /// Number of query heads per KV head group.
    pub fn n_heads_per_kv(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}
