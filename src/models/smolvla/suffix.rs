//! `embed_suffix` — fuse a chunk of noised actions with the diffusion timestep
//! into the expert's input tokens. This is the half of the denoiser that
//! doesn't touch the (frozen, KV-cached) VLM, so it ports cleanly on its own.
//!
//! Reference (`VLAFlowMatching.embed_suffix`):
//!   action_emb      = action_in_proj(noisy_actions)            [B,chunk,720]
//!   time_emb        = sinusoidal(timestep, 720)                [B,720]
//!   action_time     = cat([action_emb, time_emb_expanded], 2)  [B,chunk,1440]
//!   action_time     = action_time_mlp_out(silu(action_time_mlp_in(action_time)))
//!
//! The suffix carries `chunk_size` action tokens; the expert self-attends them
//! and cross-attends the prefix KV to predict the flow velocity `v_t`.

use burn::prelude::*;
use burn::tensor::activation::silu;

use super::config::SmolVlaConfig;
use super::projections::Projections;
use super::time::sinusoidal_time_embedding;

/// `noisy_actions: [B, chunk, action_dim]`, `timestep: [B]` → suffix tokens
/// `[B, chunk, expert_width]`.
pub fn embed_suffix<B: Backend>(
    proj: &Projections<B>,
    cfg: &SmolVlaConfig,
    min_period: f64,
    max_period: f64,
    noisy_actions: Tensor<B, 3>,
    timestep: Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [b, chunk, _] = noisy_actions.dims();
    let width = cfg.expert.width;

    // action_emb: [B, chunk, 720]
    let action_emb = proj.action_in_proj.forward(noisy_actions);

    // time_emb: [B, 720] -> broadcast to [B, chunk, 720]
    let time_emb = sinusoidal_time_embedding(timestep, width, min_period, max_period, device)
        .reshape([b, 1, width]);
    let time_emb = time_emb.expand([b, chunk, width]);

    // cat -> [B, chunk, 1440] -> mlp_in -> silu -> mlp_out -> [B, chunk, 720]
    let action_time = Tensor::cat(vec![action_emb, time_emb], 2);
    let h = proj.action_time_mlp_in.forward(action_time);
    let h = silu(h);
    proj.action_time_mlp_out.forward(h)
}
