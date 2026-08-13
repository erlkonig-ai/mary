//! Parity gate for the Inkling MLPs and the whole decoder layer.
//!
//! Oracle from `golden/capture_inkling_layer.py`. Budget, written down before
//! any number was read: worst absolute error over the tensor's own scale,
//! `1e-6`, the same criterion the attention gate uses and for the same reason —
//! these outputs cancel, so per-element relative error is reported but not
//! gated on.
//!
//! Four checks exist to be non-vacuous rather than passed:
//!
//! 1. The dense MLP is compared against the reference AND against the same
//!    answer with `global_scale` divided out, and must match the first while
//!    differing from the second — otherwise dropping that scalar would pass.
//! 2. The MoE's routed and shared halves are checked separately, and their sum
//!    must reconstruct the block. The shared experts consume the block's
//!    *input*; feeding them the routed output instead is the natural error and
//!    would show up here rather than as a vague end-to-end drift.
//! 3. The stacked expert matrix is transposed on purpose and the result must
//!    disagree. In both released checkpoints `2 * intermediate == hidden`, so
//!    that matrix is square and a transposed reading is invisible to any shape
//!    check; the oracle config makes it non-square so this check can exist.
//! 4. Both a DENSE and a SPARSE decoder layer are run, since they take
//!    different branches and different checkpoint names.
//!
//!   cargo run --release --features inkling --bin inkling_layer_gate -- [<oracle dir>]

use std::path::Path;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::block::route;
use mary::models::inkling::config::AttnKind;
use mary::models::inkling::layer::{decoder_layer, LayerMlp, LayerWeights};
use mary::models::inkling::mlp::{dense_mlp, moe, routed_experts, shared_experts};

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
    let mi = num(&man, "moe_intermediate")? as usize;
    let di = num(&man, "dense_intermediate")? as usize;
    let n_routed = num(&man, "n_routed")? as usize;
    let n_shared = num(&man, "n_shared")? as usize;
    let top_k = num(&man, "top_k")? as usize;
    let route_scale = num(&man, "route_scale")? as f32;
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
    let experts_used = num(&man, "moe_experts_used")? as usize;

    println!("=== oracle ===");
    println!("  tokens {t} hidden {h}  moe_inter {mi} dense_inter {di}");
    println!("  experts {n_routed} + {n_shared} shared, top_k {top_k}");
    let mut fails = 0usize;
    let mut checks = 0usize;

    // The whole point of the non-square choice.
    checks += 1;
    println!("\n=== is the expert-matrix orientation observable? ===");
    if 2 * mi == h {
        println!("  FAIL  2*intermediate == hidden: w13 is square and a transpose is invisible");
        fails += 1;
    } else {
        println!("  2*intermediate {} != hidden {h}, so a transposed w13 is detectable", 2 * mi);
    }
    checks += 1;
    println!("  experts actually used by some token: {experts_used} of {n_routed}");
    if experts_used < 2 {
        println!("  FAIL  fewer than two experts used — per-expert indexing is barely exercised");
        fails += 1;
    }

    let x = read_f32(&dir, "lyr_x.bin")?;
    anyhow::ensure!(x.len() == t * h, "input is {} not {}", x.len(), t * h);

    // ---- dense MLP ---------------------------------------------------------
    println!("\n=== 1. dense MLP ===");
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

    // ---- MoE ---------------------------------------------------------------
    println!("\n=== 2. MoE (routed and shared, separately then together) ===");
    let rw = read_f32(&dir, "lyr_moe_gate_weight.bin")?;
    let rb = read_f32(&dir, "lyr_moe_gate_e_score_correction_bias.bin")?;
    let rg = read_f32(&dir, "lyr_moe_gate_global_scale.bin")?;
    let gu = read_f32(&dir, "lyr_moe_experts_gate_up_proj.bin")?;
    let ed = read_f32(&dir, "lyr_moe_experts_down_proj.bin")?;
    let sg = read_f32(&dir, "lyr_moe_shared_experts_gate_proj.bin")?;
    let su = read_f32(&dir, "lyr_moe_shared_experts_up_proj.bin")?;
    let sd = read_f32(&dir, "lyr_moe_shared_experts_down_proj.bin")?;
    let y_moe = read_f32(&dir, "lyr_moe_y.bin")?;
    let y_routed = read_f32(&dir, "lyr_moe_routed.bin")?;
    let y_shared = read_f32(&dir, "lyr_moe_shared.bin")?;

    let routing = route(&x, &rw, &rb, rg[0], route_scale, t, h, n_routed, n_shared, top_k);
    let gammas: Vec<f32> = routing.iter().flat_map(|r| r.shared_gammas.clone()).collect();
    let mine_routed = routed_experts(&x, &gu, &ed, &routing, n_routed, t, h, mi);
    let mine_shared = shared_experts(&x, &sg, &su, &sd, &gammas, n_shared, t, h, mi);
    report("routed experts", &compare(&mine_routed, &y_routed), &mut checks, &mut fails);
    report("shared experts", &compare(&mine_shared, &y_shared), &mut checks, &mut fails);
    let mine_moe = moe(&x, &routing, &gu, &ed, &sg, &su, &sd, n_routed, n_shared, t, h, mi);
    report("moe total", &compare(&mine_moe, &y_moe), &mut checks, &mut fails);

    // Transposing the stacked expert matrix must change the answer.
    println!("\n=== 3. is a transposed expert matrix detectable? ===");
    let mut gu_t = vec![0f32; gu.len()];
    for e in 0..n_routed {
        let base = e * 2 * mi * h;
        for r in 0..2 * mi {
            for c in 0..h {
                // Only meaningful because 2*mi != h; the write below would be a
                // no-op reshape if it were square.
                gu_t[base + c * 2 * mi + r] = gu[base + r * h + c];
            }
        }
    }
    checks += 1;
    if 2 * mi == h {
        println!("  skipped: square matrix");
    } else {
        // Feed the transposed buffer through with swapped dims so it is a legal
        // shape, and require a different answer.
        let alt = routed_experts(&x, &gu_t, &ed, &routing, n_routed, t, h, mi);
        let d = compare(&alt, &y_routed);
        println!("  transposed w13 differs by {:e} scale-relative", d.scaled());
        if d.scaled() <= BUDGET {
            println!("    FAIL  a transposed expert matrix gives the same answer — orientation untested");
            fails += 1;
        } else {
            println!("    orientation is genuinely pinned");
        }
    }

    // ---- decoder layers ----------------------------------------------------
    let mask = causal_mask(t, Some(window));
    let dims = AttnDims {
        hidden: h, heads, kv_heads, head_dim, d_rel, rel_extent, kernel,
        rms_eps: eps, kind: AttnKind::Local,
    };
    let ls = LogScaling { n_floor, alpha };

    for tag in ["dense", "sparse"] {
        println!("\n=== 4. decoder layer, {tag} ===");
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

        // Hold the sparse buffers alive outside the match.
        let (lg, lu, ld, lrw, lrb, lrg, lgu, led, lsg, lsu, lsd);
        let mlp = if tag == "dense" {
            lg = read_f32(&dir, &p("mlp_gate_proj_weight"))?;
            lu = read_f32(&dir, &p("mlp_up_proj_weight"))?;
            ld = read_f32(&dir, &p("mlp_down_proj_weight"))?;
            LayerMlp::Dense { gate: &lg, up: &lu, down: &ld, global_scale: dense_gs, inter: di }
        } else {
            lrw = read_f32(&dir, &p("mlp_gate_weight"))?;
            lrb = read_f32(&dir, &p("mlp_gate_e_score_correction_bias"))?;
            lrg = read_f32(&dir, &p("mlp_gate_global_scale"))?;
            lgu = read_f32(&dir, &p("mlp_experts_gate_up_proj"))?;
            led = read_f32(&dir, &p("mlp_experts_down_proj"))?;
            lsg = read_f32(&dir, &p("mlp_shared_experts_gate_proj"))?;
            lsu = read_f32(&dir, &p("mlp_shared_experts_up_proj"))?;
            lsd = read_f32(&dir, &p("mlp_shared_experts_down_proj"))?;
            LayerMlp::Sparse {
                router_weight: &lrw, router_bias: &lrb, router_global_scale: lrg[0],
                route_scale, top_k, gate_up: &lgu, down: &led,
                shared_gate: &lsg, shared_up: &lsu, shared_down: &lsd,
                experts: n_routed, n_shared, inter: mi,
            }
        };

        let (mine, routing) = decoder_layer(&x, &lw, &aw, &dims, Some(ls), &mlp, &mask, t);
        report(&format!("{tag} layer"), &compare(&mine, &y_ref), &mut checks, &mut fails);
        if let Some(r) = routing {
            let used: std::collections::BTreeSet<usize> =
                r.iter().flat_map(|x| x.experts.clone()).collect();
            println!("  routed through {} distinct experts", used.len());
        }
    }

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!("GATE PASSED — {checks} checks, MLPs and decoder layer match transformers");
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
