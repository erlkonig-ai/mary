//! SmolVLA parity gate — the Flux method across the whole action pipeline:
//! time embedding, `embed_suffix`, both expert layer kernels (self/cross),
//! the full 16-layer denoiser → `v_t`, and the 10-step `sample_actions` flow.
//! Loads the real `smolvla_base` weights + PyTorch goldens (from vla-venv) and
//! checks cosine / max-abs parity. The VLM prefix KV cache is borrowed from a
//! golden until the (frozen) VLM tower is ported.
//!
//!   cargo run --release --features smolvla --bin smolvla_probe

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::models::smolvla::config::SmolVlaConfig;
use mary::models::smolvla::denoiser::ExpertDenoiser;
use mary::models::smolvla::layers::{eager_gqa_attention, ExpertLayer, ROPE_MAX_WAVELENGTH};
use mary::models::smolvla::projections::Projections;
use mary::models::smolvla::rope::apply_rope;
use mary::models::smolvla::sampler::sample_actions;
use mary::models::smolvla::suffix::embed_suffix;
use mary::models::smolvla::time::sinusoidal_time_embedding;
use mary::models::smolvla::vision::VisionEncoder;
use mary::models::smolvla::vlm::VlmTower;
use mary::nn::backend::WgpuDevice;
use mary::nn::backend::B;
use mary::nn::npy;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use std::path::{Path, PathBuf};

const CKPT: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
);
const MIN_PERIOD: f64 = 4e-3;
const MAX_PERIOD: f64 = 4.0;

fn loadt<const D: usize>(p: &Path, dev: &WgpuDevice) -> Tensor<B, D> {
    let (d, s) = npy::load_npy(p).unwrap_or_else(|e| panic!("load {}: {e}", p.display()));
    Tensor::<B, D>::from_data(TensorData::new(d, s), dev)
}

fn golden(p: &Path) -> Vec<f32> {
    npy::load_npy(p)
        .unwrap_or_else(|e| panic!("golden {}: {e}", p.display()))
        .0
}

fn metrics(name: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{name}: len {} vs {}", a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxabs = maxabs.max((x - y).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let flag = if cos > 0.9999 && maxabs < 1e-3 {
        "✓"
    } else {
        "✗"
    };
    println!("  {flag} {name:18} cos={cos:.8}  max|Δ|={maxabs:.3e}");
}

fn main() {
    let dev = Default::default();
    let cfg = SmolVlaConfig::smolvla_base();
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(Path::new(CKPT)));
    let proj = Projections::<B>::load(&loader, "model", &dev);
    let g = PathBuf::from("/tmp/codex_outputs/smolvla_probe");

    let noise = loadt::<3>(&g.join("inputs/noise.npy"), &dev); // [1,50,32]
    let (tsd, _) = npy::load_npy(&g.join("inputs/timestep.npy")).unwrap();
    let n = tsd.len();
    let timestep = Tensor::<B, 1>::from_data(TensorData::new(tsd, vec![n]), &dev);

    println!("SmolVLA suffix-path parity (smolvla_base):");

    // 1. geometric time embedding
    let time_emb = sinusoidal_time_embedding::<B>(
        timestep.clone(),
        cfg.expert.width,
        MIN_PERIOD,
        MAX_PERIOD,
        &dev,
    );
    metrics(
        "time_emb",
        &time_emb.into_data().to_vec::<f32>().unwrap(),
        &golden(&g.join("golden/time_emb.npy")),
    );

    // 2. action_in_proj
    let ae = proj.action_in_proj.forward(noise.clone());
    metrics(
        "action_emb",
        &ae.into_data().to_vec::<f32>().unwrap(),
        &golden(&g.join("golden/action_emb.npy")),
    );

    // 3. full embed_suffix (action ⊕ time → mlp_in → silu → mlp_out)
    let suffix = embed_suffix::<B>(&proj, &cfg, MIN_PERIOD, MAX_PERIOD, noise, timestep, &dev);
    metrics(
        "embed_suffix",
        &suffix.into_data().to_vec::<f32>().unwrap(),
        &golden(&g.join("golden/action_time_emb.npy")),
    );

    // 4. expert decoder layer 0 (self-attn, attends prefix KV cache) — gated
    //    once the layer-0 goldens are present.
    if g.join("golden/prefix_kv0_key.npy").exists() {
        println!("Expert layer 0 (self-attn + prefix cache):");
        let layer = ExpertLayer::<B>::load(
            &loader,
            "model.vlm_with_expert.lm_expert.layers.0",
            cfg.expert,
            &dev,
        );
        let x0 = loadt::<3>(&g.join("golden/expert_layer0_in.npy"), &dev); // [1,50,720]
        let pk = loadt::<4>(&g.join("golden/prefix_kv0_key.npy"), &dev); // [1,Lp,5,64]
        let pv = loadt::<4>(&g.join("golden/prefix_kv0_value.npy"), &dev);
        let pos = loadt::<2>(&g.join("golden/denoise_position_ids.npy"), &dev); // [1,50]
                                                                                // mask npy stored as 0/1; rebuild a Bool [1,50,Lp+50]
        let (md, ms) = npy::load_npy(&g.join("golden/denoise_attn_mask.npy")).unwrap();
        let mask = Tensor::<B, 3>::from_data(TensorData::new(md, ms), &dev).greater_elem(0.5);
        // attention output (pre-o_proj) — isolates attn from the residual tail
        if g.join("golden/expert_layer0_attout.npy").exists() {
            let [b, l, _] = x0.dims();
            let (hq, hkv, dh) = (
                cfg.expert.n_heads,
                cfg.expert.n_kv_heads,
                cfg.expert.head_dim,
            );
            let h = layer.input_layernorm.forward(x0.clone());
            let q = apply_rope(
                layer.q_proj.forward(h.clone()).reshape([b, l, hq, dh]),
                pos.clone(),
                ROPE_MAX_WAVELENGTH,
                &dev,
            );
            let k = apply_rope(
                layer.k_proj.forward(h.clone()).reshape([b, l, hkv, dh]),
                pos.clone(),
                ROPE_MAX_WAVELENGTH,
                &dev,
            );
            let v = layer.v_proj.forward(h).reshape([b, l, hkv, dh]);
            let k = Tensor::cat(vec![pk.clone(), k], 1);
            let v = Tensor::cat(vec![pv.clone(), v], 1);
            let attout = eager_gqa_attention(q, k, v, mask.clone());
            metrics(
                "  ↳ attn(pre-o)",
                &attout.into_data().to_vec::<f32>().unwrap(),
                &golden(&g.join("golden/expert_layer0_attout.npy")),
            );
        }
        let out = layer.forward(x0, pos.clone(), pk.clone(), pv, mask.clone(), &dev);
        metrics(
            "expert_layer0",
            &out.into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("golden/expert_layer0_out.npy")),
        );

        // layer 1 — cross-attention (expert reprojects VLM KV, attends prefix only)
        if g.join("golden/prefix_kv1_key.npy").exists() {
            println!("Expert layer 1 (cross-attn into VLM KV):");
            let layer1 = ExpertLayer::<B>::load(
                &loader,
                "model.vlm_with_expert.lm_expert.layers.1",
                cfg.expert,
                &dev,
            );
            let x1 = loadt::<3>(&g.join("golden/expert_layer1_in.npy"), &dev);
            let vk = loadt::<4>(&g.join("golden/prefix_kv1_key.npy"), &dev); // [1,Lp,5,64]
            let vv = loadt::<4>(&g.join("golden/prefix_kv1_value.npy"), &dev);
            let lp = vk.dims()[1];
            // query positions normalized to start at 0; prefix-only mask = first Lp cols
            let mn = pos.clone().min().into_scalar() as f64;
            let qpos = pos.clone().sub_scalar(mn);
            let cross_mask = mask.clone().float().narrow(2, 0, lp).greater_elem(0.5);
            let out1 = layer1.forward_cross(x1, qpos, vk, vv, cross_mask, &dev);
            metrics(
                "expert_layer1",
                &out1.into_data().to_vec::<f32>().unwrap(),
                &golden(&g.join("golden/expert_layer1_out.npy")),
            );
        }

        // full 16-layer expert denoiser -> v_t (borrows the VLM caches from goldens)
        if g.join("golden/prefix_kv_all_key.npy").exists() {
            println!("Full expert denoiser (16 layers) → v_t:");
            let den = ExpertDenoiser::<B>::load(&loader, &cfg, &dev);
            // suffix from embed_suffix at t=1.0 (the FIRST sampler step that
            // produced v_t_step0 — not the 0.8 used by the suffix-path goldens)
            let noise = loadt::<3>(&g.join("inputs/noise.npy"), &dev);
            let ts = Tensor::<B, 1>::from_data(TensorData::new(vec![1.0f32], vec![1]), &dev);
            let suffix = embed_suffix::<B>(&proj, &cfg, MIN_PERIOD, MAX_PERIOD, noise, ts, &dev);
            let ck = loadt::<5>(&g.join("golden/prefix_kv_all_key.npy"), &dev); // [16,1,70,5,64]
            let cv = loadt::<5>(&g.join("golden/prefix_kv_all_value.npy"), &dev);
            let lp = ck.dims()[2];
            let cross_mask = mask.clone().float().narrow(2, 0, lp).greater_elem(0.5);
            let exp_out = den.forward(
                suffix,
                pos.clone(),
                ck.clone(),
                cv.clone(),
                mask.clone(),
                cross_mask.clone(),
                &dev,
            );
            let v_t = proj.action_out_proj.forward(exp_out);
            metrics(
                "denoise v_t (t=1)",
                &v_t.into_data().to_vec::<f32>().unwrap(),
                &golden(&g.join("golden/v_t_step0.npy")),
            );

            // full sampler: 10 Euler steps (1→0), prefix caches fixed, suffix
            // re-embedded each step. The whole inference path bar the VLM tower.
            println!("Full sample_actions (10-step flow):");
            let noise0 = loadt::<3>(&g.join("inputs/noise.npy"), &dev);
            let denoise = |x_t: Tensor<B, 3>, t: f64| {
                let tt = Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], vec![1]), &dev);
                let s = embed_suffix::<B>(&proj, &cfg, MIN_PERIOD, MAX_PERIOD, x_t, tt, &dev);
                let e = den.forward(
                    s,
                    pos.clone(),
                    ck.clone(),
                    cv.clone(),
                    mask.clone(),
                    cross_mask.clone(),
                    &dev,
                );
                proj.action_out_proj.forward(e)
            };
            let actions = sample_actions(noise0, cfg.num_steps, denoise);
            metrics(
                "actions_final",
                &actions.into_data().to_vec::<f32>().unwrap(),
                &golden(&g.join("golden/actions_final.npy")),
            );
        }
    } else {
        println!("(expert layer-0 goldens not present yet — skipping)");
    }

    // ── VLM perceptual tower, text side (vision borrowed from golden) ──
    if g.join("golden/lang_emb.npy").exists() {
        println!("VLM tower (text decoder; image embeddings borrowed):");
        let vlm = VlmTower::<B>::load(&loader, &cfg, &dev);
        let dim = cfg.vlm.width;
        let s = (dim as f64).sqrt();

        // embed_language_tokens (ids stored as f32 → Int)
        let lang_ids = loadt::<2>(&g.join("inputs/lang_tokens_f32.npy"), &dev).int();
        let lang_emb = vlm.embed_language_tokens(lang_ids);
        metrics(
            "lang_emb",
            &lang_emb.clone().into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("golden/lang_emb.npy")),
        );

        // embed_prefix: [image·√d, lang·√d, state_proj(state)]
        let img = loadt::<3>(&g.join("golden/image_hidden_states.npy"), &dev).mul_scalar(s); // [1,64,960]
        let lang = lang_emb.mul_scalar(s);
        let state = loadt::<3>(&g.join("inputs/state_3d.npy"), &dev); // [1,1,32]
        let state_emb = proj.state_proj.forward(state); // [1,1,960]
        let prefix = Tensor::cat(vec![img, lang, state_emb], 1); // [1,70,960]
        metrics(
            "embed_prefix",
            &prefix.clone().into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("golden/prefix_embs.npy")),
        );

        // VLM decoder → per-layer KV cache
        let pos = loadt::<2>(&g.join("golden/prefix_position_ids.npy"), &dev);
        let (mm, ms) = npy::load_npy(&g.join("golden/prefix_att_2d_mask.npy")).unwrap();
        let pmask = Tensor::<B, 3>::from_data(TensorData::new(mm, ms), &dev).greater_elem(0.5);
        let (ck, cv) = vlm.forward_decoder(prefix, pos, pmask, &dev);
        metrics(
            "vlm cache key",
            &ck.into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("golden/prefix_kv_all_key.npy")),
        );
        metrics(
            "vlm cache value",
            &cv.into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("golden/prefix_kv_all_value.npy")),
        );
    }

    // ── SigLIP vision encoder (embed_image) — the last piece ──
    if g.join("golden/vision_patch_embeds.npy").exists() {
        println!("SigLIP vision encoder (embed_image):");
        let venc = VisionEncoder::<B>::load(&loader, &cfg, &dev);
        // preprocessing: raw [0,1] camera frame → resize-with-pad + normalize
        let raw = loadt::<4>(&g.join("inputs/image_raw_0_1.npy"), &dev); // [1,3,512,512] in [0,1]
        let prepped = mary::models::smolvla::pipeline::preprocess_image::<B>(raw, 512);
        metrics(
            "preprocess_image",
            &prepped.into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("inputs/images.npy")),
        );
        let image = loadt::<4>(&g.join("inputs/images.npy"), &dev); // [1,3,512,512]
        let pe = venc.embeddings(image.clone());
        metrics(
            "patch_embeds",
            &pe.clone().into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("golden/vision_patch_embeds.npy")),
        );
        metrics(
            "vision_layer0",
            &venc.layer0(pe).into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("golden/vision_layer0_out.npy")),
        );
        metrics(
            "vision_last_hid",
            &venc
                .encode(image.clone())
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            &golden(&g.join("golden/vision_last_hidden.npy")),
        );
        metrics(
            "image_hidden",
            &venc.embed_image(image).into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("golden/image_hidden_states.npy")),
        );
    }

    // ── FULL END-TO-END: image + language + state + noise → action chunk,
    //    every model tensor computed in Rust (only raw inputs + structural
    //    position/mask bookkeeping come from disk). ──
    if g.join("golden/image_hidden_states.npy").exists()
        && g.join("golden/prefix_kv_all_key.npy").exists()
    {
        println!("END-TO-END (image → actions, no borrowed model tensors):");
        let venc = VisionEncoder::<B>::load(&loader, &cfg, &dev);
        let vlm = VlmTower::<B>::load(&loader, &cfg, &dev);
        let den = ExpertDenoiser::<B>::load(&loader, &cfg, &dev);
        let s = (cfg.vlm.width as f64).sqrt();

        // perception: image → SigLIP → image tokens; language → embeddings; state
        let image = loadt::<4>(&g.join("inputs/images.npy"), &dev);
        let img_emb = venc.embed_image(image).mul_scalar(s);
        let lang_ids = loadt::<2>(&g.join("inputs/lang_tokens_f32.npy"), &dev).int();
        let lang = vlm.embed_language_tokens(lang_ids).mul_scalar(s);
        let state = proj
            .state_proj
            .forward(loadt::<3>(&g.join("inputs/state_3d.npy"), &dev));
        let prefix = Tensor::cat(vec![img_emb, lang, state], 1);

        // VLM decoder → prefix KV cache (in Rust)
        let ppos = loadt::<2>(&g.join("golden/prefix_position_ids.npy"), &dev);
        let (pm, psh) = npy::load_npy(&g.join("golden/prefix_att_2d_mask.npy")).unwrap();
        let pmask = Tensor::<B, 3>::from_data(TensorData::new(pm, psh), &dev).greater_elem(0.5);
        let (ck, cv) = vlm.forward_decoder(prefix, ppos, pmask, &dev);

        // action: 10-step flow over the Rust-computed caches
        let pos = loadt::<2>(&g.join("golden/denoise_position_ids.npy"), &dev);
        let (dm, dsh) = npy::load_npy(&g.join("golden/denoise_attn_mask.npy")).unwrap();
        let mask = Tensor::<B, 3>::from_data(TensorData::new(dm, dsh), &dev).greater_elem(0.5);
        let cross_mask = mask
            .clone()
            .float()
            .narrow(2, 0, ck.dims()[2])
            .greater_elem(0.5);
        let noise0 = loadt::<3>(&g.join("inputs/noise.npy"), &dev);
        let denoise = |x_t: Tensor<B, 3>, t: f64| {
            let tt = Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], vec![1]), &dev);
            let sfx = embed_suffix::<B>(&proj, &cfg, MIN_PERIOD, MAX_PERIOD, x_t, tt, &dev);
            let e = den.forward(
                sfx,
                pos.clone(),
                ck.clone(),
                cv.clone(),
                mask.clone(),
                cross_mask.clone(),
                &dev,
            );
            proj.action_out_proj.forward(e)
        };
        let actions = sample_actions(noise0, cfg.num_steps, denoise);
        metrics(
            "e2e actions",
            &actions.into_data().to_vec::<f32>().unwrap(),
            &golden(&g.join("golden/actions_final.npy")),
        );
    }
}
