use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Flux2TransformerConfig {
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub joint_attention_dim: usize,
    pub in_channels: usize,
    pub mlp_ratio: f64,
    #[serde(default = "default_guidance_embeds")]
    pub guidance_embeds: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub axes_dims_rope: Vec<usize>,
    #[serde(default = "default_timestep_guidance_channels")]
    pub timestep_guidance_channels: usize,
    #[serde(default = "default_eps")]
    pub eps: f64,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    pub out_channels: Option<usize>,
}

fn default_guidance_embeds() -> bool {
    false
}

fn default_rope_theta() -> f64 {
    2000.0
}

fn default_timestep_guidance_channels() -> usize {
    256
}

fn default_eps() -> f64 {
    1e-6
}

fn default_patch_size() -> usize {
    1
}

impl Flux2TransformerConfig {
    pub fn load(path: &std::path::Path) -> Self {
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read transformer config: {}", e));
        serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("Failed to parse transformer config: {}", e))
    }

    /// inner_dim = num_attention_heads * attention_head_dim
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// mlp_hidden_dim = inner_dim * mlp_ratio
    pub fn mlp_hidden_dim(&self) -> usize {
        (self.inner_dim() as f64 * self.mlp_ratio) as usize
    }

    /// Effective out_channels (defaults to in_channels if not specified)
    pub fn effective_out_channels(&self) -> usize {
        self.out_channels.unwrap_or(self.in_channels)
    }
}
