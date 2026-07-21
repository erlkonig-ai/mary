//! Conditional flow-matching sampler: a forward-Euler ODE over the F5 velocity
//! field, from noise (t=0) to a clean mel (t=1), with sway sampling and
//! classifier-free guidance. Matches F5's `CFM.sample` (linspace+sway time grid,
//! `odeint` method="euler", CFG `cond + (cond − uncond)·cfg`).

use super::config::CfmConfig;
use super::model::F5Transformer;
use burn::prelude::*;
use burn::tensor::Distribution;

/// Sway time-warp (F5 default coef −1.0): u → u + c·(cos(π/2·u) − 1 + u).
fn sway(u: f64, coef: f64) -> f64 {
    u + coef * ((std::f64::consts::FRAC_PI_2 * u).cos() - 1.0 + u)
}

/// Generate a mel by integrating from fresh N(0,1) noise.
pub fn sample<B: Backend>(
    model: &F5Transformer<B>,
    cond_mel: Tensor<B, 3>,
    text_ids: Tensor<B, 2, Int>,
    cfg: &CfmConfig,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [b, t, m] = cond_mel.dims();
    let x0 = Tensor::random([b, t, m], Distribution::Normal(0.0, 1.0), device);
    integrate(model, x0, cond_mel, text_ids, cfg, device)
}

/// Integrate the velocity field from a given initial noise `x0` — the
/// deterministic core, so probes can inject a fixed y0 for parity.
pub fn integrate<B: Backend>(
    model: &F5Transformer<B>,
    x0: Tensor<B, 3>,
    cond_mel: Tensor<B, 3>,
    text_ids: Tensor<B, 2, Int>,
    cfg: &CfmConfig,
    device: &B::Device,
) -> Tensor<B, 3> {
    let b = cond_mel.dims()[0];
    let mut x = x0;
    let nfe = cfg.nfe.max(1);
    for i in 0..nfe {
        let ta = sway(i as f64 / nfe as f64, cfg.sway_coef);
        let tb = sway((i + 1) as f64 / nfe as f64, cfg.sway_coef);
        let dt = (tb - ta) as f32;
        let time = Tensor::<B, 1>::full([b], ta as f32, device);

        let v_cond = model.forward(x.clone(), cond_mel.clone(), text_ids.clone(), time.clone());
        // F5's CFG: cond + (cond − uncond)·cfg (cfg=0 → pure cond).
        let v = if cfg.cfg_strength < 1e-5 {
            v_cond
        } else {
            let v_uncond =
                model.forward_cfg(x.clone(), cond_mel.clone(), text_ids.clone(), time, true, true);
            v_cond.clone() + (v_cond - v_uncond) * cfg.cfg_strength
        };
        x = x + v * dt;
    }
    x
}
