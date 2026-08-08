//! End-to-end gate: one REAL layer of the released checkpoint, in mary.
//!
//! Everything before this ran on random weights. This loads layer weights from
//! `Inkling-Small-NVFP4` through `layout` and `nvfp4`, runs mary's
//! `decoder_layer`, and compares against `transformers` running the same layer
//! on the same input.
//!
//! **What this proves and what it does not.** It exercises the shard plumbing,
//! the BF16 widening, the layout's names, and the whole layer's arithmetic
//! against an independent implementation. It does NOT independently establish
//! the checkpoint-name to module-parameter MAPPING: that mapping is authored on
//! both sides by the same person, so a shared misreading passes. Two things
//! narrow the gap — `load_state_dict(strict=True)` on the reference side makes
//! the mapping's totality machine-checked, and the per-tensor fingerprints here
//! localise any loading error to one tensor. The residual assumption is the
//! `w13` split ([gate; up] rather than [up; gate]), which no comparison of two
//! same-author lanes can falsify; a full forward producing coherent text would,
//! and that needs more memory than one machine here has.
//!
//! Budget, written down first: worst absolute error over the tensor's own
//! scale, `1e-5`. Looser than the random-weight gates because a real layer is
//! ~20k accumulations deep per output with weights spanning several orders of
//! magnitude, so f32 summation order matters more; the number is still reported
//! rather than asserted away.
//!
//!   cargo run --release --features inkling --bin inkling_real_gate -- <ckpt> <oracle>

use std::path::PathBuf;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::config::AttnKind;
use mary::models::inkling::layer::{decoder_layer, LayerMlp, LayerWeights};
use mary::models::inkling::load::{split_gate_up, Checkpoint};

const BUDGET: f32 = 1e-5;

fn read_f32(p: &std::path::Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn num(text: &str, key: &str) -> Result<f64> {
    let pat = format!("\"{key}\"");
    let at = text.find(&pat).with_context(|| format!("manifest has no {key}"))?;
    let rest = &text[at + pat.len()..];
    let colon = rest.find(':').context("malformed manifest")?;
    let s: String = rest[colon + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
        .collect();
    s.parse().with_context(|| format!("{key} is not a number: {s:?}"))
}

/// The fingerprint block for one tensor, located by its module name.
fn fingerprint(man: &str, key: &str, field: &str) -> Result<f64> {
    let at = man
        .find(&format!("\"{key}\""))
        .with_context(|| format!("no fingerprint for {key}"))?;
    let pat = format!("\"{field}\"");
    let rel = man[at..].find(&pat).with_context(|| format!("{key} has no {field}"))?;
    let rest = &man[at + rel + pat.len()..];
    let colon = rest.find(':').context("malformed fingerprint")?;
    let s: String = rest[colon + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
        .collect();
    s.parse().with_context(|| format!("{key}.{field} is not a number: {s:?}"))
}

fn check_fp(man: &str, key: &str, v: &[f32], checks: &mut usize, fails: &mut usize) {
    let sum: f64 = v.iter().map(|&x| x as f64).sum();
    let want = match fingerprint(man, key, "sum") {
        Ok(w) => w,
        Err(e) => {
            println!("  FAIL  {key}: {e}");
            *fails += 1;
            return;
        }
    };
    *checks += 1;
    // The sums are of millions of terms in a different order; compare relative
    // to the magnitude of the sum itself.
    let denom = want.abs().max(1.0);
    let rel = (sum - want).abs() / denom;
    if rel > 1e-4 {
        println!("  FAIL  {key}: sum {sum:+.6e}, reference {want:+.6e} (rel {rel:e})");
        *fails += 1;
    } else {
        println!("  ok    {:-42} sum {:+.6e}  n={}", key, sum, v.len());
    }
}

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1).map(PathBuf::from).context("usage: <ckpt> <oracle>")?;
    let oracle = std::env::args().nth(2).map(PathBuf::from).context("usage: <ckpt> <oracle>")?;
    let man = String::from_utf8(std::fs::read(oracle.join("real_manifest.json"))?)?;

    let layer = num(&man, "layer")? as usize;
    let t = num(&man, "tokens")? as usize;
    let h = num(&man, "hidden")? as usize;
    let heads = num(&man, "heads")? as usize;
    let kv_heads = num(&man, "kv_heads")? as usize;
    let head_dim = num(&man, "head_dim")? as usize;
    let d_rel = num(&man, "d_rel")? as usize;
    let rel_extent = num(&man, "rel_extent")? as usize;
    let window = num(&man, "sliding_window")? as usize;
    let kernel = num(&man, "kernel")? as usize;
    let eps = num(&man, "rms_norm_eps")?;
    let di = num(&man, "dense_intermediate")? as usize;
    let n_floor = num(&man, "log_scaling_n_floor")? as f32;
    let alpha = num(&man, "log_scaling_alpha")? as f32;
    let is_sliding = man.contains("\"is_sliding\": true");

    let cp = Checkpoint::open(&ckpt)?;
    println!("=== checkpoint ===");
    println!("  dir              : {}", ckpt.display());
    println!("  tensors in index : {}", cp.len());
    println!("  layer {layer}: hidden {h}, heads {heads}/{kv_heads}x{head_dim}, {}",
             if is_sliding { "sliding" } else { "global" });
    anyhow::ensure!(cp.len() > 0, "empty index — the gate would be vacuous");

    let mut fails = 0usize;
    let mut checks = 0usize;

    let p = format!("model.llm.layers.{layer}.");
    let g = |n: &str| -> Result<Vec<f32>> { Ok(cp.tensor(&format!("{p}{n}"))?.data) };

    println!("\n=== 1. loaded tensors against the reference's fingerprints ===");
    let attn_norm = g("attn_norm.weight")?;
    let mlp_norm = g("mlp_norm.weight")?;
    let attn_sconv = g("attn_sconv.weight")?;
    let mlp_sconv = g("mlp_sconv.weight")?;
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
    let fused = g("mlp.w13_dn.weight")?;
    let down = g("mlp.w2_md.weight")?;
    let gscale = g("mlp.global_scale")?;

    check_fp(&man, "input_layernorm.weight", &attn_norm, &mut checks, &mut fails);
    check_fp(&man, "post_attention_layernorm.weight", &mlp_norm, &mut checks, &mut fails);
    check_fp(&man, "attn_sconv.conv1d.weight", &attn_sconv, &mut checks, &mut fails);
    check_fp(&man, "mlp_sconv.conv1d.weight", &mlp_sconv, &mut checks, &mut fails);
    check_fp(&man, "self_attn.q_proj.weight", &wq, &mut checks, &mut fails);
    check_fp(&man, "self_attn.k_proj.weight", &wk, &mut checks, &mut fails);
    check_fp(&man, "self_attn.v_proj.weight", &wv, &mut checks, &mut fails);
    check_fp(&man, "self_attn.r_proj.weight", &wr, &mut checks, &mut fails);
    check_fp(&man, "self_attn.o_proj.weight", &wo, &mut checks, &mut fails);
    check_fp(&man, "self_attn.q_norm.weight", &qn, &mut checks, &mut fails);
    check_fp(&man, "self_attn.k_norm.weight", &kn, &mut checks, &mut fails);
    check_fp(&man, "self_attn.k_sconv.conv1d.weight", &ks, &mut checks, &mut fails);
    check_fp(&man, "self_attn.v_sconv.conv1d.weight", &vs, &mut checks, &mut fails);
    check_fp(&man, "self_attn.rel_logits_proj.proj", &rp, &mut checks, &mut fails);
    check_fp(&man, "mlp.down_proj.weight", &down, &mut checks, &mut fails);
    check_fp(&man, "mlp.global_scale", &gscale, &mut checks, &mut fails);

    println!("\n=== 2. the fused w13 split ===");
    let (gate, up) = split_gate_up(&fused, h);
    println!("  fused {} -> gate {} + up {}", fused.len(), gate.len(), up.len());
    check_fp(&man, "mlp.gate_proj.weight", &gate, &mut checks, &mut fails);
    check_fp(&man, "mlp.up_proj.weight", &up, &mut checks, &mut fails);
    // The halves must be distinguishable, or the split's ORDER is untested here
    // even though its correctness would still be assumed.
    checks += 1;
    let gs: f64 = gate.iter().map(|&x| x as f64).sum();
    let us: f64 = up.iter().map(|&x| x as f64).sum();
    if (gs - us).abs() / gs.abs().max(1.0) < 1e-3 {
        println!("  FAIL  the two halves have indistinguishable sums — swapping them would pass");
        fails += 1;
    } else {
        println!("  halves differ (gate sum {gs:+.6e} vs up sum {us:+.6e}), so a swap is visible");
    }

    println!("\n=== 3. run the layer ===");
    let x = read_f32(&oracle.join("real_x.bin"))?;
    let y_ref = read_f32(&oracle.join("real_y.bin"))?;
    anyhow::ensure!(x.len() == t * h, "input is {} not {}", x.len(), t * h);
    anyhow::ensure!(!y_ref.is_empty(), "no reference output — the gate would be vacuous");

    let dims = AttnDims {
        hidden: h, heads, kv_heads, head_dim, d_rel, rel_extent, kernel,
        rms_eps: eps,
        kind: if is_sliding { AttnKind::Local } else { AttnKind::Global },
    };
    let aw = AttnWeights {
        wq: &wq, wk: &wk, wv: &wv, wr: &wr, wo: &wo,
        k_sconv: &ks, v_sconv: &vs, q_norm: &qn, k_norm: &kn, rel_proj: &rp,
    };
    let lw = LayerWeights {
        attn_norm: &attn_norm, mlp_norm: &mlp_norm,
        attn_sconv: &attn_sconv, mlp_sconv: &mlp_sconv,
    };
    let mlp = LayerMlp::Dense { gate: &gate, up: &up, down: &down, global_scale: gscale[0], inter: di };
    let mask = causal_mask(t, if is_sliding { Some(window) } else { None });

    let (mine, _) = decoder_layer(&x, &lw, &aw, &dims, Some(LogScaling { n_floor, alpha }), &mlp, &mask, t);

    let mut worst_abs = 0f32;
    let mut scale = 0f32;
    let mut worst_rel = 0f32;
    for (&a, &b) in mine.iter().zip(&y_ref) {
        checks += 1;
        worst_abs = worst_abs.max((a - b).abs());
        scale = scale.max(b.abs());
        worst_rel = worst_rel.max((a - b).abs() / b.abs().max(1e-6));
    }
    let scaled = worst_abs / scale.max(f32::MIN_POSITIVE);
    println!("  values compared   : {}", mine.len());
    println!("  worst absolute    : {worst_abs:e}  (scale {scale:e})");
    println!("  worst abs / scale : {scaled:e}   <- the criterion, budget {BUDGET:e}");
    println!("  worst relative    : {worst_rel:e}   (reported, not gated)");
    if scaled > BUDGET {
        println!("  FAIL  over budget");
        fails += 1;
    }

    println!("\n=== what this does not prove ===");
    println!("  the checkpoint-name -> module mapping is authored on BOTH sides here,");
    println!("  so a shared misreading passes. The w13 split order in particular is an");
    println!("  assumption; only a full forward producing coherent text would settle it.");
    println!("  log scaling is inert at {t} tokens with n_floor {n_floor} — untested here.");

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!("GATE PASSED — {checks} checks, a real layer runs in mary and matches transformers");
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
