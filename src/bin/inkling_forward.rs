//! A real forward pass of Inkling-Small on one machine, paging experts.
//!
//! Every gate so far ends with the same disclaimer: the checkpoint-name to
//! module mapping is authored on both sides, so a shared misreading would pass.
//! This is the check that can settle it. Coherent continuations cannot come out
//! of a wrong mapping — a transposed projection or a swapped gate/up half
//! produces noise, not English.
//!
//! It fits on one machine because the router is small. The layer's experts are
//! 26 GB at f32 and a short prompt touches a few dozen of 256, but which ones
//! is not known until attention has run — so each layer runs attention, routes,
//! and only then reads the selected expert slabs out of the mapping, applying
//! and dropping each before the next. Peak residency is one layer, not one
//! model.
//!
//! Each distinct expert is decoded ONCE and applied to every token that chose
//! it; decoding per (token, expert) pair would repeat most of the work.
//!
//!   cargo run --release --features inkling --bin inkling_forward -- <ckpt> <ids.bin> <out.bin>

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{attention, causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::block::{rms_norm, route, short_conv, Routing};
use mary::models::inkling::config::{AttnKind, InklingConfig};
use mary::models::inkling::load::{split_gate_up, Checkpoint};
use mary::models::inkling::mlp::{dense_mlp, shared_experts};
use mary::models::inkling::stack::{embed_and_norm, head};

/// `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `y = x W^T`, `W` stored `[out, in]`.
fn linear(x: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            out[r * out_dim + o] = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    }
    out
}

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;
    let ids_path = std::env::args().nth(2).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;
    let out_path = std::env::args().nth(3).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;

    let cfg_text = std::fs::read_to_string(ckpt.join("config.json"))?;
    let cfg = InklingConfig::from_json(&cfg_text).context("parsing config.json")?;
    let t = &cfg.text_config;
    let cp = Checkpoint::open(&ckpt)?;

    let ids: Vec<usize> = std::fs::read(&ids_path)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    let n = ids.len();
    anyhow::ensure!(n > 0, "no tokens — the forward would be vacuous");

    let h = t.hidden_size;
    println!("=== forward ===");
    println!("  checkpoint : {}", ckpt.display());
    println!("  tokens     : {n}  {ids:?}");
    println!("  layers     : {}  hidden {h}  experts {}+{} shared",
             t.num_hidden_layers, t.n_routed_experts, t.n_shared_experts);
    println!("  tensors    : {}", cp.len());

    let started = Instant::now();
    let embed_w = cp.tensor("model.llm.embed.weight")?.data;
    let embed_n = cp.tensor("model.llm.embed_norm.weight")?.data;
    let mut x = embed_and_norm(&ids, &embed_w, &embed_n, t.rms_norm_eps, t.vocab_size, h);
    drop(embed_w);
    println!("  embedded in {:.1}s", started.elapsed().as_secs_f32());

    if let Ok(dir) = std::env::var("INK_DUMP_DIR") {
        std::fs::create_dir_all(&dir)?;
        let mut bytes = Vec::with_capacity(x.len() * 4);
        for v in &x {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(format!("{dir}/h_embed.bin"), &bytes)?;
    }

    let ls = LogScaling {
        n_floor: t.log_scaling_n_floor as f32,
        alpha: t.log_scaling_alpha as f32,
    };
    let mask_local = causal_mask(n, Some(t.sliding_window_size));
    let mask_global = causal_mask(n, None);
    let mut expert_loads = 0usize;

    for layer in 0..t.num_hidden_layers {
        let l0 = Instant::now();
        let kind = t.attn_kind(layer);
        let is_local = kind == AttnKind::Local;
        let (heads, kv_heads, head_dim) = t.heads(kind);
        let p = format!("model.llm.layers.{layer}.");
        let g = |nm: &str| -> Result<Vec<f32>> { Ok(cp.tensor(&format!("{p}{nm}"))?.data) };

        // ---- attention ----------------------------------------------------
        let attn_norm = g("attn_norm.weight")?;
        let hn = rms_norm(&x, &attn_norm, t.rms_norm_eps, n, h);
        let dims = AttnDims {
            hidden: h, heads, kv_heads, head_dim,
            d_rel: t.d_rel,
            rel_extent: t.rel_span(kind),
            kernel: t.sconv_kernel_size,
            rms_eps: t.rms_norm_eps,
            kind,
        };
        let wq = g("attn.wq_du.weight")?;
        let wk = g("attn.wk_dv.weight")?;
        let wv = g("attn.wv_dv.weight")?;
        let wr = g("attn.wr_du.weight")?;
        let wo = g("attn.wo_ud.weight")?;
        let qn = g("attn.q_norm.weight")?;
        let kn = g("attn.k_norm.weight")?;
        let ks = g("attn.k_sconv.weight")?;
        let vs = g("attn.v_sconv.weight")?;
        let rp = g("attn.rel_logits_proj.proj")?;
        let aw = AttnWeights {
            wq: &wq, wk: &wk, wv: &wv, wr: &wr, wo: &wo,
            k_sconv: &ks, v_sconv: &vs, q_norm: &qn, k_norm: &kn, rel_proj: &rp,
        };
        let mask = if is_local { &mask_local } else { &mask_global };
        let a = attention(&hn, &aw, &dims, Some(ls), mask, n);
        drop((wq, wk, wv, wr, wo));
        let a = short_conv(&a, &g("attn_sconv.weight")?, n, h, t.sconv_kernel_size);
        for (xi, ai) in x.iter_mut().zip(&a) {
            *xi += ai;
        }

        // ---- MLP ----------------------------------------------------------
        let mlp_norm = g("mlp_norm.weight")?;
        let hn = rms_norm(&x, &mlp_norm, t.rms_norm_eps, n, h);

        let y = if t.is_dense(layer) {
            let fused = g("mlp.w13_dn.weight")?;
            let (gate, up) = split_gate_up(&fused, h);
            let down = g("mlp.w2_md.weight")?;
            let gs = g("mlp.global_scale")?;
            dense_mlp(&hn, &gate, &up, &down, gs[0], n, h, t.dense_intermediate_size)
        } else {
            let inter = t.intermediate_size;
            let rw = g("mlp.gate.weight")?;
            let rb = g("mlp.gate.bias")?;
            let rg = g("mlp.gate.global_scale")?;
            let routing: Vec<Routing> = route(
                &hn, &rw, &rb, rg[0], t.route_scale as f32,
                n, h, t.n_routed_experts, t.n_shared_experts, t.num_experts_per_tok,
            );

            // Group tokens by expert, so each slab is decoded once.
            let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
            for (ti, r) in routing.iter().enumerate() {
                for (slot, &e) in r.experts.iter().enumerate() {
                    by_expert.entry(e).or_default().push((ti, r.weights[slot]));
                }
            }

            let mut acc = vec![0f32; n * h];
            for (&e, toks) in &by_expert {
                let gu = cp.expert_slice(&format!("{p}mlp.experts.w13_weight"), e)?.data;
                let dn = cp.expert_slice(&format!("{p}mlp.experts.w2_weight"), e)?.data;
                expert_loads += 1;
                for &(ti, wgt) in toks {
                    let xt = &hn[ti * h..(ti + 1) * h];
                    let both = linear(xt, &gu, 1, h, 2 * inter);
                    let act: Vec<f32> =
                        (0..inter).map(|i| silu(both[i]) * both[inter + i]).collect();
                    let contrib = linear(&act, &dn, 1, inter, h);
                    for (o, c) in acc[ti * h..(ti + 1) * h].iter_mut().zip(&contrib) {
                        *o += c * wgt;
                    }
                }
                // Dropped here: one expert resident at a time, not 256.
            }

            let sfused = cp.tensor(&format!("{p}mlp.shared_experts.shared_w13_weight"))?.data;
            let per = sfused.len() / t.n_shared_experts;
            let mut sg = Vec::with_capacity(sfused.len() / 2);
            let mut su = Vec::with_capacity(sfused.len() / 2);
            for s in 0..t.n_shared_experts {
                let blk = &sfused[s * per..(s + 1) * per];
                sg.extend_from_slice(&blk[..per / 2]);
                su.extend_from_slice(&blk[per / 2..]);
            }
            drop(sfused);
            let sd = cp.tensor(&format!("{p}mlp.shared_experts.shared_w2_weight"))?.data;
            let gammas: Vec<f32> = routing.iter().flat_map(|r| r.shared_gammas.clone()).collect();
            let sh = shared_experts(&hn, &sg, &su, &sd, &gammas, t.n_shared_experts, n, h, inter);
            acc.iter().zip(&sh).map(|(a, b)| a + b).collect()
        };

        let y = short_conv(&y, &g("mlp_sconv.weight")?, n, h, t.sconv_kernel_size);
        for (xi, yi) in x.iter_mut().zip(&y) {
            *xi += yi;
        }

        if let Ok(dir) = std::env::var("INK_DUMP_DIR") {
            let mut bytes = Vec::with_capacity(x.len() * 4);
            for v in &x {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(format!("{dir}/h_after_{layer:02}.bin"), &bytes)?;
        }
        let norm: f32 = (x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len() as f64).sqrt() as f32;
        println!("  layer {layer:2} [{}] {:.1}s  rms {norm:.4}",
                 if is_local { "local " } else { "global" }, l0.elapsed().as_secs_f32());
    }

    // ---- head --------------------------------------------------------------
    let fnorm = cp.tensor("model.llm.norm.weight")?.data;
    let unembed = cp.tensor("model.llm.unembed.weight")?.data;
    let logits = head(
        &x, &fnorm, &unembed,
        t.logits_mup_width_multiplier as f32,
        t.vocab_size, t.effective_vocab(), t.rms_norm_eps, n, h,
    );
    let v = t.effective_vocab();

    println!("\n=== predictions ===");
    println!("  expert slabs decoded: {expert_loads}");
    println!("  elapsed: {:.1}s", started.elapsed().as_secs_f32());

    let mut top_all: Vec<i64> = Vec::new();
    for ti in 0..n {
        let row = &logits[ti * v..(ti + 1) * v];
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        let top: Vec<usize> = idx[..5].to_vec();
        println!("  after token {ti} (id {}): top5 {:?}  logits {:?}",
                 ids[ti], top,
                 top.iter().map(|&i| (row[i] * 100.0).round() / 100.0).collect::<Vec<_>>());
        for &i in &top {
            top_all.push(i as i64);
        }
    }

    let mut bytes = Vec::new();
    for i in top_all {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    std::fs::write(&out_path, &bytes)?;
    println!("  wrote top-5 ids per position to {}", out_path.display());
    Ok(())
}
