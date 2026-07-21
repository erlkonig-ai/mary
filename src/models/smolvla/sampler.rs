//! Flow-matching sampler — the Euler integrator that turns noise into an action
//! chunk. Structurally identical to `f5`'s CFM, but integrating *backwards*
//! from t=1 (noise) to t=0 (clean action):
//!
//!   dt  = -1 / num_steps
//!   x_t = noise                                  (at t = 1)
//!   for step in 0..num_steps:
//!       t   = 1 + step·dt
//!       v_t = denoise(x_t, t)                     (the expert's flow velocity)
//!       x_t = x_t + dt·v_t
//!
//! The `denoise` closure is the VLM-expert denoiser (`denoise_step`): it embeds
//! the suffix, cross-attends the cached prefix KV, and projects to `v_t`.
//! Keeping it a closure lets the integrator be verified independently of the
//! (heavier) expert tower.

use burn::prelude::*;

/// Integrate the flow from `noise` to a clean action chunk. `denoise(x_t, t)`
/// returns the flow velocity `v_t` at the given continuous time `t ∈ [0,1]`.
pub fn sample_actions<B, F>(noise: Tensor<B, 3>, num_steps: usize, mut denoise: F) -> Tensor<B, 3>
where
    B: Backend,
    F: FnMut(Tensor<B, 3>, f64) -> Tensor<B, 3>,
{
    let dt = -1.0 / num_steps as f64;
    let mut x_t = noise;
    for step in 0..num_steps {
        let t = 1.0 + step as f64 * dt;
        let v_t = denoise(x_t.clone(), t);
        x_t = x_t + v_t.mul_scalar(dt);
    }
    x_t
}
