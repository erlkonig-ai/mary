use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default = "default_layers_per_block")]
    pub layers_per_block: usize,
    #[serde(default = "default_act_fn")]
    pub act_fn: String,
    #[serde(default = "default_norm_num_groups")]
    pub norm_num_groups: usize,
    #[serde(default = "default_sample_size")]
    pub sample_size: usize,
    #[serde(default)]
    pub patch_size: Vec<usize>,
    #[serde(default = "default_true")]
    pub force_upcast: bool,
    #[serde(default = "default_true")]
    pub use_quant_conv: bool,
    #[serde(default = "default_true")]
    pub use_post_quant_conv: bool,
    #[serde(default = "default_true")]
    pub mid_block_add_attention: bool,
    #[serde(default = "default_batch_norm_eps")]
    pub batch_norm_eps: f64,
    #[serde(default = "default_batch_norm_momentum")]
    pub batch_norm_momentum: f64,
}

fn default_block_out_channels() -> Vec<usize> {
    vec![128, 256, 512, 512]
}
fn default_layers_per_block() -> usize {
    2
}
fn default_act_fn() -> String {
    "silu".to_string()
}
fn default_norm_num_groups() -> usize {
    32
}
fn default_sample_size() -> usize {
    1024
}
fn default_true() -> bool {
    true
}
fn default_batch_norm_eps() -> f64 {
    1e-4
}
fn default_batch_norm_momentum() -> f64 {
    0.1
}

impl VaeConfig {
    pub fn load(path: &std::path::Path) -> Self {
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read VAE config: {}", e));
        serde_json::from_str(&json).unwrap_or_else(|e| panic!("Failed to parse VAE config: {}", e))
    }
}
