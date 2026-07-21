//! Configuration structs for Qwen2.5-VL 7B.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2_5VlConfig {
    pub text_config: Qwen2_5VlTextConfig,
    pub vision_config: Qwen2_5VlVisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2_5VlTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<QwenMropeScaling>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_dropout: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QwenMropeScaling {
    #[serde(default)]
    pub rope_type: String,
    #[serde(default, rename = "type")]
    pub kind: String,
    pub mrope_section: [usize; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2_5VlVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub hidden_act: String,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub window_size: usize,
    pub out_hidden_size: usize,
    pub fullatt_block_indexes: Vec<usize>,
}

impl Qwen2_5VlConfig {
    pub fn load(path: &std::path::Path) -> Self {
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read Qwen2.5-VL config: {e}"));
        serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("failed to parse Qwen2.5-VL config: {e}"))
    }
}

impl Qwen2_5VlTextConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// The real Qwen2.5-VL-7B-Instruct text config — the backbone of
    /// `nomic-embed-multimodal-7b`. (M-RoPE collapses to standard 1D RoPE for
    /// pure-text sequences, which is the embedder's text path.)
    pub fn nomic_mm7b() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            head_dim: Some(128),
            hidden_act: "silu".into(),
            max_position_embeddings: 128000,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            rope_scaling: Some(QwenMropeScaling {
                rope_type: "mrope".into(),
                kind: "mrope".into(),
                mrope_section: [16, 24, 24],
            }),
            tie_word_embeddings: false,
            attention_dropout: 0.0,
        }
    }
}

impl Qwen2_5VlVisionConfig {
    /// The real Qwen2.5-VL-7B-Instruct vision-tower config (nomic carries no
    /// vision LoRA, so the base vision weights are final).
    pub fn nomic_mm7b() -> Self {
        Self {
            depth: 32,
            hidden_size: 1280,
            hidden_act: "silu".into(),
            intermediate_size: 3420,
            num_heads: 16,
            in_channels: 3,
            patch_size: 14,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            window_size: 112,
            out_hidden_size: 3584,
            fullatt_block_indexes: vec![7, 15, 23, 31],
        }
    }
}
