use serde::Deserialize;

/// Outer Mistral3ForConditionalGeneration config (contains text_config).
#[derive(Debug, Deserialize)]
struct Mistral3OuterConfig {
    text_config: Mistral3TextConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct Mistral3TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub vocab_size: usize,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    1_000_000_000.0
}

/// Mistral3 text encoder config (extracted from the text_config section).
#[derive(Debug, Clone)]
pub struct Mistral3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
}

impl Mistral3Config {
    /// Load from the Mistral3ForConditionalGeneration config.json,
    /// extracting the text_config section.
    pub fn load(path: &std::path::Path) -> Self {
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read Mistral3 config: {}", e));
        let outer: Mistral3OuterConfig = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("Failed to parse Mistral3 config: {}", e));
        let tc = outer.text_config;
        Self {
            hidden_size: tc.hidden_size,
            num_hidden_layers: tc.num_hidden_layers,
            num_attention_heads: tc.num_attention_heads,
            num_key_value_heads: tc.num_key_value_heads,
            head_dim: tc.head_dim,
            intermediate_size: tc.intermediate_size,
            rms_norm_eps: tc.rms_norm_eps,
            rope_theta: tc.rope_theta,
            vocab_size: tc.vocab_size,
        }
    }
}
