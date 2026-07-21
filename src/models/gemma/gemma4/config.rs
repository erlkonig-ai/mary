//! Configuration for Gemma 4 text decoder (all variants).

use serde::Deserialize;

/// Layer attention type: sliding window (local) or full (global).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

/// RoPE parameters for one attention type.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: String,
    /// Fraction of head_dim to apply RoPE to (1.0 = full, 0.25 = proportional).
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
}

fn default_partial_rotary() -> f64 { 1.0 }

/// Dual RoPE configuration: separate params for sliding vs full attention.
#[derive(Debug, Clone, Deserialize)]
pub struct DualRopeParams {
    pub sliding_attention: RopeParams,
    pub full_attention: RopeParams,
}

/// Gemma 4 text decoder configuration.
/// Covers E2B, E4B, 26B-A4B (MoE), and 31B.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<LayerType>,
    pub rope_parameters: DualRopeParams,

    // Global attention specifics
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    /// K=V optimization: in global layers, keys = values (halves KV cache).
    #[serde(default)]
    pub attention_k_eq_v: bool,

    // Per-Layer Embeddings (PLE) — E2B, E4B only
    /// PLE embedding dimension per layer (0 = disabled).
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub vocab_size_per_layer_input: usize,

    // Shared KV cache layers — E2B only
    /// Number of layers (from the end) that share KV with earlier layers.
    #[serde(default)]
    pub num_kv_shared_layers: usize,

    // MoE — 26B only
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    // The Python config uses `moe_intermediate_size`. Accept both names.
    #[serde(alias = "moe_intermediate_size")]
    pub expert_intermediate_size: Option<usize>,

    // Misc
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default = "default_softcap")]
    pub final_logit_softcapping: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_activation")]
    pub hidden_activation: String,

    /// When set to `"vision"` (31B+), sliding-attention layers unmask
    /// every (q, kv) pair that sits in the same multimodal span during
    /// prefill — image/audio tokens attend bidirectionally to each other
    /// in local layers while full-attention layers stay causal.
    /// E2B/E4B leave this as `None`.
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,
}

fn default_softcap() -> f64 { 30.0 }
fn default_activation() -> String { "gelu_pytorch_tanh".to_string() }

/// Gemma 4 audio encoder config (Conformer with chunked local attention).
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4AudioConfig {
    #[serde(default = "default_audio_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_audio_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_audio_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_audio_act")]
    pub hidden_act: String,

    #[serde(default = "default_subsampling_channels")]
    pub subsampling_conv_channels: Vec<usize>,

    #[serde(default = "default_conv_kernel")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_residual_weight")]
    pub residual_weight: f32,
    #[serde(default = "default_attention_chunk")]
    pub attention_chunk_size: usize,
    #[serde(default = "default_attention_context_left")]
    pub attention_context_left: usize,
    #[serde(default = "default_attention_context_right")]
    pub attention_context_right: usize,
    #[serde(default = "default_attention_logit_cap")]
    pub attention_logit_cap: f32,
    #[serde(default = "default_invalid_logits")]
    pub attention_invalid_logits_value: f32,

    #[serde(default = "default_true")]
    pub use_clipped_linears: bool,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_gradient_clipping")]
    pub gradient_clipping: f32,
    #[serde(default = "default_output_proj_dims")]
    pub output_proj_dims: usize,
}

fn default_audio_hidden() -> usize { 1024 }
fn default_audio_layers() -> usize { 12 }
fn default_audio_heads() -> usize { 8 }
fn default_audio_act() -> String { "silu".to_string() }
fn default_subsampling_channels() -> Vec<usize> { vec![128, 32] }
fn default_conv_kernel() -> usize { 5 }
fn default_residual_weight() -> f32 { 0.5 }
fn default_attention_chunk() -> usize { 12 }
fn default_attention_context_left() -> usize { 13 }
fn default_attention_context_right() -> usize { 0 }
fn default_attention_logit_cap() -> f32 { 50.0 }
fn default_invalid_logits() -> f32 { -1.0e9 }
fn default_true() -> bool { true }
fn default_rms_eps() -> f64 { 1e-6 }
fn default_gradient_clipping() -> f32 { 1e10 }
fn default_output_proj_dims() -> usize { 1536 }

/// Top-level Gemma 4 config (wraps text + vision + audio configs).
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>, // parsed later when needed
    #[serde(default)]
    pub audio_config: Option<Gemma4AudioConfig>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub audio_token_id: Option<u32>,
    #[serde(default)]
    pub vision_soft_tokens_per_image: Option<usize>,
}

impl Gemma4Config {
    /// Load from a config.json file.
    pub fn load(path: &std::path::Path) -> Self {
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read config: {}", e));
        serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("Failed to parse Gemma4 config: {}", e))
    }
}

impl Gemma4TextConfig {
    /// Effective KV heads for global attention layers.
    pub fn global_kv_heads(&self) -> usize {
        self.num_global_key_value_heads.unwrap_or(self.num_key_value_heads)
    }

    /// Effective head dim for global attention layers.
    pub fn global_head_dim(&self) -> usize {
        self.global_head_dim.unwrap_or(self.head_dim)
    }

    /// Whether this layer uses sliding (local) or full (global) attention.
    pub fn layer_type(&self, layer_idx: usize) -> LayerType {
        self.layer_types[layer_idx]
    }

    /// Number of dimensions that get RoPE for a given layer type.
    pub fn rope_dim(&self, layer_type: LayerType) -> usize {
        let hd = match layer_type {
            LayerType::SlidingAttention => self.head_dim,
            LayerType::FullAttention => self.global_head_dim(),
        };
        let factor = match layer_type {
            LayerType::SlidingAttention => 1.0,
            LayerType::FullAttention => self.rope_parameters.full_attention.partial_rotary_factor,
        };
        (hd as f64 * factor) as usize
    }

    /// Whether PLE is enabled.
    pub fn has_ple(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }

    /// Whether this is an MoE model.
    pub fn has_moe(&self) -> bool {
        self.enable_moe_block
    }

    /// First layer index that shares KV cache with an earlier layer.
    /// Returns num_hidden_layers if no sharing.
    pub fn first_shared_kv_layer(&self) -> usize {
        self.num_hidden_layers - self.num_kv_shared_layers
    }
}
