//! SmolVLA configuration — every dimension here is a *measured fact* from
//! introspecting the `lerobot/smolvla_base` checkpoint (2026-06-14, vla-venv,
//! lerobot 0.5.1), not a guess. Total 450,046,176 params:
//!   VLM 350,165,184 (frozen by default, train_expert_only=True)
//!   + action expert (lm_expert) 98,245,840
//!   + projections.
//!
//! SmolVLA = a SmolVLM2 vision-language backbone (frozen) feeding a smaller
//! flow-matching **action expert** that cross-attends the VLM's cached KV and
//! denoises a chunk of future actions. The expert is the only thing we train;
//! the VLM is the perceptual/language prior.

/// A GQA transformer tower's shape (shared between the VLM and the expert).
#[derive(Debug, Clone, Copy)]
pub struct TowerConfig {
    /// Residual width.
    pub width: usize,
    /// Number of transformer layers (both towers are truncated to 16 — real,
    /// not a config typo: SmolVLA cuts the backbone depth).
    pub n_layers: usize,
    /// Query heads.
    pub n_heads: usize,
    /// Key/value heads (GQA 15:5).
    pub n_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// SwiGLU intermediate dimension.
    pub ffn_dim: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base frequency.
    pub rope_theta: f64,
}

/// Full SmolVLA architecture, resolved layer-by-layer from the checkpoint.
#[derive(Debug, Clone)]
pub struct SmolVlaConfig {
    /// The frozen SmolVLM2 backbone (vision + language), width 960.
    pub vlm: TowerConfig,
    /// The action expert, width 720. Its attention runs in the VLM head-space
    /// (q_proj 720->960, k/v_proj 720->320, o_proj 960->720) so it can
    /// cross-attend the VLM's cached KV, then projects back to 720.
    pub expert: TowerConfig,
    /// Padded action/state dimension. The real expressive vector is 9
    /// (head pose 6 + body yaw 1 + antennas 2), zero-padded up to this width.
    pub action_dim: usize,
    /// The semantically-meaningful slice of `action_dim` we actually drive on
    /// the Reachy: [head x,y,z, roll,pitch,yaw, body_yaw, ant_l, ant_r].
    pub expressive_dim: usize,
    /// Action chunk horizon (waypoints predicted per inference). Streamed to
    /// `body.act` at the demo/record control rate; chunks are interruptible
    /// (the next inference supersedes the current chunk mid-execution).
    pub chunk_size: usize,
    /// Flow-matching denoising steps at inference.
    pub num_steps: usize,
    /// Geometric timestep-embedding period range (sensitivity over t∈[0,1]).
    pub min_period: f64,
    pub max_period: f64,
}

impl SmolVlaConfig {
    /// `lerobot/smolvla_base`, retargeted to the Reachy Mini expressive head.
    pub fn smolvla_base() -> Self {
        Self {
            vlm: TowerConfig {
                width: 960,
                n_layers: 16,
                n_heads: 15,
                n_kv_heads: 5,
                head_dim: 64,
                ffn_dim: 2560,
                rms_norm_eps: 1e-5,
                rope_theta: 1e4,
            },
            expert: TowerConfig {
                width: 720,
                n_layers: 16,
                // Attention is computed in VLM head-space (960 = 15*64 for q,
                // 320 = 5*64 for kv), then o_proj maps 960 back to the 720
                // residual. So the *head counts* match the VLM (15:5), but the
                // residual the heads read from / write to is 720.
                n_heads: 15,
                n_kv_heads: 5,
                head_dim: 64,
                ffn_dim: 2048,
                rms_norm_eps: 1e-5,
                rope_theta: 1e4,
            },
            action_dim: 32,
            expressive_dim: 9,
            chunk_size: 50,
            num_steps: 10,
            min_period: 4e-3,
            max_period: 4.0,
        }
    }
}
