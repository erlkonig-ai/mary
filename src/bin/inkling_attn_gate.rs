//! Parity gate for Inkling attention, against `transformers` — LOCAL and GLOBAL.
//!
//! Both layers are gated because they are different functions: different head
//! counts (`swa_*` against the global fields), a different relative-table
//! reach, a window mask on one and not the other, and log scaling on the global
//! layer only. A gate on either alone cannot see the difference, which is the
//! same failure that let a single checkpoint hide the `swa` head config.
//!
//! Budget, written down before any number was read: worst absolute error over
//! the tensor's own scale, `1e-6`. Per-element relative error is reported but
//! not gated on — attention outputs cancel, and a relative error divided by a
//! near-zero reference says nothing, which cost a false failure on the short
//! convolution.
//!
//! Four checks exist to be non-vacuous rather than passed:
//!
//! 1. The mask this build constructs must equal the one `transformers` built.
//!    Mask construction is part of the port, so importing the oracle's mask
//!    would be checking the easy half.
//! 2. Log scaling must actually vary on the global layer: with the real
//!    `n_floor` of 128000 and a short sequence tau is exactly 1, and a gate
//!    would "cover" log scaling while testing nothing.
//! 3. The relative bias must actually hit its out-of-range branch, i.e. the
//!    sequence must be longer than `rel_extent`.
//! 4. Running the global layer's weights through the LOCAL configuration must
//!    give a different answer, or the two configurations are indistinguishable
//!    on this corpus.
//!
//!   cargo run --release --features inkling --bin inkling_attn_gate -- [<oracle dir>]

use std::path::Path;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{attention, causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::config::AttnKind;

const BUDGET: f32 = 1e-6;

fn read_f32(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let p = dir.join(name);
    let b = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Pull `key` out of a flat-ish JSON object, searching from `from`.
fn num_after(text: &str, key: &str, from: usize) -> Result<f64> {
    let pat = format!("\"{key}\"");
    let at = text[from..]
        .find(&pat)
        .with_context(|| format!("manifest has no {key} after {from}"))?
        + from;
    let rest = &text[at + pat.len()..];
    let colon = rest.find(':').context("malformed manifest")?;
    let s: String = rest[colon + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
        .collect();
    s.parse()
        .with_context(|| format!("{key} is not a number: {s:?}"))
}

fn num(text: &str, key: &str) -> Result<f64> {
    num_after(text, key, 0)
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
    let mut d = Diff {
        worst_abs: 0.0,
        scale: 0.0,
        worst_rel: 0.0,
        n: mine.len().min(theirs.len()),
    };
    for (&a, &b) in mine.iter().zip(theirs) {
        let abs = (a - b).abs();
        d.worst_abs = d.worst_abs.max(abs);
        d.scale = d.scale.max(b.abs());
        d.worst_rel = d.worst_rel.max(abs / b.abs().max(1e-6));
    }
    d
}

#[allow(clippy::too_many_arguments)]
fn run_layer(
    dir: &Path,
    tag: &str,
    man: &str,
    tokens: usize,
    hidden: usize,
    x: &[f32],
    fails: &mut usize,
    checks: &mut usize,
) -> Result<()> {
    // The per-layer block starts at the tag, so read its fields from there.
    let at = man
        .find(&format!("\"{tag}\""))
        .context("manifest lacks the layer")?;
    let heads = num_after(man, "num_heads", at)? as usize;
    let kv_heads = num_after(man, "num_kv_heads", at)? as usize;
    let head_dim = num_after(man, "head_dim", at)? as usize;
    let rel_extent = num_after(man, "rel_extent", at)? as usize;
    let ref_scaling = num_after(man, "scaling", at)? as f32;
    let is_local = tag == "local";

    let d_rel = num(man, "d_rel")? as usize;
    let kernel = num(man, "kernel")? as usize;
    let eps = num(man, "rms_norm_eps")?;
    let window = num(man, "sliding_window")? as usize;
    let n_floor = num(man, "log_scaling_n_floor")? as f32;
    let alpha = num(man, "log_scaling_alpha")? as f32;

    println!("\n=== {tag} layer: heads {heads}, kv {kv_heads}, head_dim {head_dim}, rel_extent {rel_extent} ===");

    let p = |n: &str| format!("attn_{tag}_{n}");
    let wq = read_f32(dir, &p("wq.bin"))?;
    let wk = read_f32(dir, &p("wk.bin"))?;
    let wv = read_f32(dir, &p("wv.bin"))?;
    let wr = read_f32(dir, &p("wr.bin"))?;
    let wo = read_f32(dir, &p("wo.bin"))?;
    let ks = read_f32(dir, &p("k_sconv.bin"))?;
    let vs = read_f32(dir, &p("v_sconv.bin"))?;
    let qn = read_f32(dir, &p("q_norm.bin"))?;
    let kn = read_f32(dir, &p("k_norm.bin"))?;
    let rp = read_f32(dir, &p("rel_proj.bin"))?;
    let ref_mask = read_f32(dir, &p("mask.bin"))?;
    let y = read_f32(dir, &p("y.bin"))?;
    anyhow::ensure!(
        !y.is_empty(),
        "{tag}: no reference output — the gate would be vacuous"
    );

    // ---- check 1: our mask must match theirs ------------------------------
    let mask = causal_mask(tokens, if is_local { Some(window) } else { None });
    anyhow::ensure!(
        ref_mask.len() == tokens * tokens,
        "{tag}: mask is {} not {}",
        ref_mask.len(),
        tokens * tokens
    );
    let mut mask_bad = 0usize;
    let mut visible = 0usize;
    for i in 0..tokens * tokens {
        *checks += 1;
        let mine_vis = mask[i] == 0.0;
        // torch uses a large negative rather than -inf; compare visibility.
        let theirs_vis = ref_mask[i] > -1e30;
        if mine_vis {
            visible += 1;
        }
        if mine_vis != theirs_vis {
            if mask_bad < 4 {
                println!(
                    "  FAIL  mask[{}][{}]: mine {}, reference {}",
                    i / tokens,
                    i % tokens,
                    mask[i],
                    ref_mask[i]
                );
            }
            mask_bad += 1;
        }
    }
    println!(
        "  mask cells {} , visible {visible}, disagreements {mask_bad}",
        tokens * tokens
    );
    *fails += mask_bad;
    // Non-vacuity: a window mask must actually hide more than a causal one.
    if is_local {
        let causal_only = causal_mask(tokens, None)
            .iter()
            .filter(|&&v| v == 0.0)
            .count();
        *checks += 1;
        if causal_only == visible {
            println!("  FAIL  the window hides nothing here — the local mask is untested");
            *fails += 1;
        } else {
            println!(
                "  window hides {} cells a causal mask would show",
                causal_only - visible
            );
        }
    }

    let dims = AttnDims {
        hidden,
        heads,
        kv_heads,
        head_dim,
        d_rel,
        rel_extent,
        kernel,
        rms_eps: eps,
        kind: if is_local {
            AttnKind::Local
        } else {
            AttnKind::Global
        },
    };
    *checks += 1;
    if (dims.scaling() - ref_scaling).abs() > 1e-9 {
        println!(
            "  FAIL  scaling {} != reference {ref_scaling}",
            dims.scaling()
        );
        *fails += 1;
    } else {
        println!(
            "  scaling {} matches (1/head_dim, not 1/sqrt(head_dim))",
            dims.scaling()
        );
    }

    let w = AttnWeights {
        wq: &wq,
        wk: &wk,
        wv: &wv,
        wr: &wr,
        wo: &wo,
        k_sconv: &ks,
        v_sconv: &vs,
        q_norm: &qn,
        k_norm: &kn,
        rel_proj: &rp,
    };
    let ls = LogScaling { n_floor, alpha };
    let mine = attention(x, &w, &dims, Some(ls), &mask, tokens);

    let d = compare(&mine, &y);
    *checks += d.n;
    println!("  values compared   : {}", d.n);
    println!(
        "  worst absolute    : {:e}  (scale {:e})",
        d.worst_abs, d.scale
    );
    println!("  worst abs / scale : {:e}   <- the criterion", d.scaled());
    println!(
        "  worst relative    : {:e}   (reported, not gated)",
        d.worst_rel
    );
    if d.scaled() > BUDGET {
        println!("  FAIL  over budget");
        *fails += 1;
    }

    // ---- check 3: is the out-of-range relative branch reached? ------------
    *checks += 1;
    if tokens <= rel_extent {
        println!("  FAIL  tokens {tokens} <= rel_extent {rel_extent}: the out-of-range bias branch never fires");
        *fails += 1;
    } else {
        println!(
            "  {} of {} distances exceed rel_extent, so the zeroing branch fires",
            tokens - rel_extent,
            tokens
        );
    }

    // ---- check 4: the other configuration must disagree -------------------
    let other = AttnDims {
        kind: if is_local {
            AttnKind::Global
        } else {
            AttnKind::Local
        },
        ..dims
    };
    let other_mask = causal_mask(tokens, if is_local { None } else { Some(window) });
    let flipped = attention(x, &w, &other, Some(ls), &other_mask, tokens);
    let fd = compare(&flipped, &y);
    *checks += 1;
    println!(
        "  same weights under the OTHER kind: worst abs/scale {:e}",
        fd.scaled()
    );
    if fd.scaled() <= BUDGET {
        println!("  FAIL  local and global are indistinguishable here — the kind is untested");
        *fails += 1;
    } else {
        println!("  the two kinds genuinely differ on this corpus");
    }
    Ok(())
}

fn main() -> Result<()> {
    let dir = mary::paths::model(std::env::args().nth(1).as_deref(), "inkling-oracle")?;
    let man = String::from_utf8(std::fs::read(dir.join("attn_manifest.json"))?)?;
    let tokens = num(&man, "tokens")? as usize;
    let hidden = num(&man, "hidden")? as usize;
    let x = read_f32(&dir, "attn_x.bin")?;

    println!("=== oracle ===");
    println!("  dir    : {}", dir.display());
    println!("  tokens {tokens}  hidden {hidden}");
    anyhow::ensure!(
        x.len() == tokens * hidden,
        "input is {} not {}",
        x.len(),
        tokens * hidden
    );

    let mut fails = 0usize;
    let mut checks = 0usize;

    // Log scaling must be doing something, or the global check covers nothing.
    let tau_max = num_after(&man, "tau_max", man.find("\"global\"").unwrap_or(0))?;
    checks += 1;
    println!("\n=== is log scaling under test? ===");
    println!("  global tau_max reported by the capture: {tau_max}");
    if tau_max <= 1.0 + 1e-6 {
        println!("  FAIL  tau is 1 everywhere — log scaling is inert and the global gate does not test it");
        fails += 1;
    } else {
        println!("  tau varies, so log scaling is genuinely exercised");
    }

    for tag in ["local", "global"] {
        run_layer(&dir, tag, &man, tokens, hidden, &x, &mut fails, &mut checks)?;
    }

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!(
            "GATE PASSED — {checks} checks, attention matches transformers on both layer kinds"
        );
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
