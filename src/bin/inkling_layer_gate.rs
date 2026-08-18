//! Parity gate for Inkling's DENSE MLP and its DENSE decoder layer.
//!
//! Oracle from `golden/capture_inkling_layer.py`. Budget, written down before
//! any number was read: worst absolute error over the tensor's own scale,
//! `1e-6`, the same criterion the attention gate uses and for the same reason —
//! these outputs cancel, so per-element relative error is reported but not
//! gated on.
//!
//! # What this gate stopped covering, and why that is not a hole
//!
//! It used to run the host f32 MoE — `routed_experts`, `shared_experts`, `moe`,
//! separately then together, plus a transposed-expert-matrix falsifier — against
//! this same capture. Those functions are gone, so the checks that called them
//! are gone with them. What replaced the SUBJECT is not this gate: it is
//! `inkling_fp4_expert_gate` and `inkling_bf16_expert_gate`, each holding the
//! device kernel the forward actually issues to a bundle Python wrote. The
//! reference here was never the problem — a Python capture is a reference to the
//! real model — but a gate needs an implementation to point at, and the only
//! remaining implementation of a routed expert is a tensor-core kernel that this
//! gate's f32-slice interface cannot call.
//!
//! Two of the falsifiers died with it and are named so their loss is not
//! silent:
//!
//! * the shared experts consume the block's INPUT, not the routed output. The
//!   device lane's version of that claim is `inkling_forward::shared_experts_bf16`
//!   taking `hn` — the same normed stream the router read — and nothing else.
//! * the stacked expert matrix's ORIENTATION. `2 * intermediate == hidden` in
//!   both releases, so a transposed reading loads without complaint; the oracle
//!   config made it non-square so a transpose was detectable. The device lane
//!   never reshapes that matrix — it hands the packed bytes to the MMA with `n`
//!   and `k` passed explicitly — so the surviving check on the same hazard is
//!   `inkling_fp4_expert_gate`'s f64 arbiter over one whole real expert.
//!
//! Two checks here exist to be non-vacuous rather than passed:
//!
//! 1. The dense MLP is compared against the reference AND against the same
//!    answer with `global_scale` divided out, and must match the first while
//!    differing from the second — otherwise dropping that scalar would pass.
//! 2. The dense decoder layer is run whole, so the four additive paths
//!    (attention, its short conv, the MLP, its short conv) are composed rather
//!    than each checked alone.
//!
//!   cargo run --release --features inkling-cuda --bin inkling_layer_gate -- [<oracle dir>]

use std::path::Path;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::config::AttnKind;
use mary::models::inkling::layer::{decoder_layer, LayerMlp, LayerWeights};
use mary::models::inkling::mlp::dense_mlp;

const BUDGET: f32 = 1e-6;

fn read_f32(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let p = dir.join(name);
    let b = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
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

struct Diff {
    worst_abs: f32,
    scale: f32,
    worst_rel: f32,
    n: usize,
}
impl Diff {
    fn scaled(&self) -> f32 {
        self.worst_abs / self.scale.max(f32::MIN_POSITIVE)
    }
}
fn compare(mine: &[f32], theirs: &[f32]) -> Diff {
    let mut d = Diff { worst_abs: 0.0, scale: 0.0, worst_rel: 0.0, n: mine.len().min(theirs.len()) };
    for (&a, &b) in mine.iter().zip(theirs) {
        let abs = (a - b).abs();
        d.worst_abs = d.worst_abs.max(abs);
        d.scale = d.scale.max(b.abs());
        d.worst_rel = d.worst_rel.max(abs / b.abs().max(1e-6));
    }
    d
}

fn report(label: &str, d: &Diff, checks: &mut usize, fails: &mut usize) {
    *checks += d.n;
    println!("  {label}: {} values, worst abs {:e} / scale {:e} = {:e}, rel {:e}",
             d.n, d.worst_abs, d.scale, d.scaled(), d.worst_rel);
    if d.scaled() > BUDGET {
        println!("    FAIL  over budget {BUDGET:e}");
        *fails += 1;
    }
}

fn main() -> Result<()> {
    let dir = mary::paths::model(std::env::args().nth(1).as_deref(), "inkling-oracle")?;
    let man = String::from_utf8(std::fs::read(dir.join("lyr_manifest.json"))?)?;

    let t = num(&man, "tokens")? as usize;
    let h = num(&man, "hidden")? as usize;
    let di = num(&man, "dense_intermediate")? as usize;
    let kernel = num(&man, "kernel")? as usize;
    let eps = num(&man, "rms_norm_eps")?;
    let heads = num(&man, "heads")? as usize;
    let kv_heads = num(&man, "kv_heads")? as usize;
    let head_dim = num(&man, "head_dim")? as usize;
    let d_rel = num(&man, "d_rel")? as usize;
    let rel_extent = num(&man, "rel_extent")? as usize;
    let window = num(&man, "sliding_window")? as usize;
    let n_floor = num(&man, "log_scaling_n_floor")? as f32;
    let alpha = num(&man, "log_scaling_alpha")? as f32;
    let mlp_gs = num(&man, "mlp_global_scale")? as f32;
    let dense_gs = num(&man, "dense_global_scale")? as f32;

    println!("=== oracle ===");
    println!("  tokens {t} hidden {h}  dense_inter {di}");
    let mut fails = 0usize;
    let mut checks = 0usize;

    let x = read_f32(&dir, "lyr_x.bin")?;
    anyhow::ensure!(x.len() == t * h, "input is {} not {}", x.len(), t * h);

    // ---- dense MLP ---------------------------------------------------------
    println!("=== 1. dense MLP ===");
    let g = read_f32(&dir, "lyr_mlp_gate_proj_weight.bin")?;
    let u = read_f32(&dir, "lyr_mlp_up_proj_weight.bin")?;
    let dn = read_f32(&dir, "lyr_mlp_down_proj_weight.bin")?;
    let y = read_f32(&dir, "lyr_mlp_y.bin")?;
    let y_noscale = read_f32(&dir, "lyr_mlp_y_noscale.bin")?;
    anyhow::ensure!(!y.is_empty(), "no dense-MLP reference — the gate would be vacuous");
    let mine = dense_mlp(&x, &g, &u, &dn, mlp_gs, t, h, di);
    report("dense mlp", &compare(&mine, &y), &mut checks, &mut fails);
    // global_scale must matter.
    let d_ns = compare(&mine, &y_noscale);
    checks += 1;
    println!("  vs the same answer without global_scale: {:e}", d_ns.scaled());
    if d_ns.scaled() <= BUDGET {
        println!("    FAIL  global_scale changes nothing — dropping it would pass");
        fails += 1;
    } else {
        println!("    global_scale is genuinely under test");
    }

    // ---- decoder layers ----------------------------------------------------
    let mask = causal_mask(t, Some(window));
    let dims = AttnDims {
        hidden: h, heads, kv_heads, head_dim, d_rel, rel_extent, kernel,
        rms_eps: eps, kind: AttnKind::Local,
    };
    let ls = LogScaling { n_floor, alpha };

    // Dense only. The `sparse` pass that ran beside it drove `LayerMlp::Sparse`,
    // which is deleted -- a sparse layer's MLP has no host implementation to
    // point a gate at any more.
    for tag in ["dense"] {
        println!("\n=== 2. decoder layer, {tag} ===");
        let p = |n: &str| format!("lyr_{tag}_{n}.bin");
        let attn_norm = read_f32(&dir, &p("input_layernorm_weight"))?;
        let mlp_norm = read_f32(&dir, &p("post_attention_layernorm_weight"))?;
        let attn_sconv = read_f32(&dir, &p("attn_sconv_conv1d_weight"))?;
        let mlp_sconv = read_f32(&dir, &p("mlp_sconv_conv1d_weight"))?;
        let wq = read_f32(&dir, &p("self_attn_q_proj_weight"))?;
        let wk = read_f32(&dir, &p("self_attn_k_proj_weight"))?;
        let wv = read_f32(&dir, &p("self_attn_v_proj_weight"))?;
        let wr = read_f32(&dir, &p("self_attn_r_proj_weight"))?;
        let wo = read_f32(&dir, &p("self_attn_o_proj_weight"))?;
        let ks = read_f32(&dir, &p("self_attn_k_sconv_conv1d_weight"))?;
        let vs = read_f32(&dir, &p("self_attn_v_sconv_conv1d_weight"))?;
        let qn = read_f32(&dir, &p("self_attn_q_norm_weight"))?;
        let kn = read_f32(&dir, &p("self_attn_k_norm_weight"))?;
        let rp = read_f32(&dir, &p("self_attn_rel_logits_proj_proj"))?;
        let y_ref = read_f32(&dir, &p("y"))?;
        anyhow::ensure!(!y_ref.is_empty(), "{tag}: no reference output");

        let aw = AttnWeights {
            wq: &wq, wk: &wk, wv: &wv, wr: &wr, wo: &wo,
            k_sconv: &ks, v_sconv: &vs, q_norm: &qn, k_norm: &kn, rel_proj: &rp,
        };
        let lw = LayerWeights { attn_norm: &attn_norm, mlp_norm: &mlp_norm, attn_sconv: &attn_sconv, mlp_sconv: &mlp_sconv };

        let lg = read_f32(&dir, &p("mlp_gate_proj_weight"))?;
        let lu = read_f32(&dir, &p("mlp_up_proj_weight"))?;
        let ld = read_f32(&dir, &p("mlp_down_proj_weight"))?;
        let mlp = LayerMlp { gate: &lg, up: &lu, down: &ld, global_scale: dense_gs, inter: di };

        let mine = decoder_layer(&x, &lw, &aw, &dims, Some(ls), &mlp, &mask, t);
        report(&format!("{tag} layer"), &compare(&mine, &y_ref), &mut checks, &mut fails);
    }

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!("GATE PASSED — {checks} checks, the dense MLP and dense layer match transformers");
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
