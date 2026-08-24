//! Parity gate for the Inkling block primitives against `transformers`.
//!
//! The oracle is captured by `golden/capture_inkling_block.py`, which runs the
//! real `transformers.models.inkling` modules with seeded random weights.
//!
//! **Budgets, written down before any number was read.** These are f32
//! reductions in a different order from torch's, and `exp`/`ln` come from a
//! different libm, so bitwise equality is not the right bar — unlike the NVFP4
//! decode, where both sides multiply the same exact binary values and 0 was the
//! right bar. Relative error against the reference's own magnitude:
//!
//! | quantity            | budget | why                                        |
//! |---------------------|--------|--------------------------------------------|
//! | RMSNorm             | 1e-6   | one f32 sum of `width` squares, then rsqrt |
//! | short conv          | 1e-6   | a `kernel`-term f32 dot, plus the residual |
//! | router weights      | 1e-5   | logsigmoid + logsumexp, two libm calls deep|
//! | router expert sets  | exact  | a discrete choice; nothing to round        |
//!
//! Three of the checks exist to be *non-vacuous* rather than to be passed:
//!
//! 1. The short conv is compared against BOTH the module's output and the bare
//!    convolution, and must match the former and *differ* from the latter —
//!    otherwise an implementation missing the internal residual would pass.
//! 2. The router is re-run ignoring the score-correction bias, and that version
//!    must choose a different expert set — otherwise the bias is untested.
//! 3. Every check prints how many values it examined.
//!
//!   cargo run --release --features inkling --bin inkling_block_gate -- [<oracle dir>]

use std::path::Path;

use anyhow::{Context, Result};

use mary::models::inkling::block::{rms_norm, route, short_conv};

const BUDGET_RMS: f32 = 1e-6;
const BUDGET_SCONV: f32 = 1e-6;
const BUDGET_ROUTER_W: f32 = 1e-5;

fn read_f32(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let p = dir.join(name);
    let b = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    anyhow::ensure!(b.len() % 4 == 0, "{name} is not a whole number of f32");
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i64(dir: &Path, name: &str) -> Result<Vec<i64>> {
    let p = dir.join(name);
    let b = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    anyhow::ensure!(b.len() % 8 == 0, "{name} is not a whole number of i64");
    Ok(b.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn manifest_num(text: &str, key: &str) -> Result<f64> {
    let pat = format!("\"{key}\"");
    let at = text
        .find(&pat)
        .with_context(|| format!("manifest has no {key}"))?;
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

/// How two tensors differ.
///
/// Per-element relative error is reported but is NOT the pass criterion: it
/// explodes wherever the reference is near zero, and `x + conv(x)` cancels to
/// near zero often. The criterion is the worst absolute error measured against
/// the tensor's own scale, which is the quantity that stays meaningful when an
/// individual output is the difference of two similar numbers.
struct Diff {
    worst_rel: f32,
    worst_abs: f32,
    ref_scale: f32,
    n: usize,
    at: usize,
    mine_at: f32,
    ref_at: f32,
}

impl Diff {
    /// Worst absolute error as a fraction of the tensor's largest magnitude.
    fn scaled(&self) -> f32 {
        self.worst_abs / self.ref_scale.max(f32::MIN_POSITIVE)
    }
}

fn compare(mine: &[f32], theirs: &[f32]) -> Diff {
    let mut d = Diff {
        worst_rel: 0.0,
        worst_abs: 0.0,
        ref_scale: 0.0,
        n: mine.len().min(theirs.len()),
        at: 0,
        mine_at: 0.0,
        ref_at: 0.0,
    };
    for (i, (&a, &b)) in mine.iter().zip(theirs).enumerate() {
        let abs = (a - b).abs();
        d.worst_abs = d.worst_abs.max(abs);
        d.ref_scale = d.ref_scale.max(b.abs());
        let rel = abs / b.abs().max(1e-6);
        if rel > d.worst_rel {
            d.worst_rel = rel;
            d.at = i;
            d.mine_at = a;
            d.ref_at = b;
        }
    }
    d
}

fn worst_rel(mine: &[f32], theirs: &[f32]) -> (f32, usize) {
    let d = compare(mine, theirs);
    (d.worst_rel, d.n)
}

fn main() -> Result<()> {
    let dir = mary::paths::model(std::env::args().nth(1).as_deref(), "inkling-oracle")?;
    let man = String::from_utf8(std::fs::read(dir.join("blk_manifest.json"))?)?;

    let h = manifest_num(&man, "hidden_size")? as usize;
    let k = manifest_num(&man, "kernel")? as usize;
    let t = manifest_num(&man, "tokens")? as usize;
    let eps = manifest_num(&man, "rms_norm_eps")?;
    let n_routed = manifest_num(&man, "n_routed")? as usize;
    let n_shared = manifest_num(&man, "n_shared")? as usize;
    let top_k = manifest_num(&man, "top_k")? as usize;
    let route_scale = manifest_num(&man, "route_scale")? as f32;
    let bias_changes = manifest_num(&man, "tokens_where_bias_changes_selection")? as usize;

    println!("=== oracle ===");
    println!("  dir     : {}", dir.display());
    println!("  hidden {h}  kernel {k}  tokens {t}  eps {eps:e}");
    println!(
        "  experts {n_routed} routed + {n_shared} shared, top_k {top_k}, route_scale {route_scale}"
    );

    let mut fails = 0usize;
    let mut checks = 0usize;

    // ---- RMSNorm -----------------------------------------------------------
    println!("\n=== 1. RMSNorm (budget {BUDGET_RMS:e} relative) ===");
    let x = read_f32(&dir, "blk_rms_x.bin")?;
    let w = read_f32(&dir, "blk_rms_w.bin")?;
    let y = read_f32(&dir, "blk_rms_y.bin")?;
    anyhow::ensure!(
        !y.is_empty(),
        "no RMSNorm reference values — gate would be vacuous"
    );
    let mine = rms_norm(&x, &w, eps, t, h);
    let (worst, n) = worst_rel(&mine, &y);
    checks += n;
    println!("  values compared : {n}");
    println!("  worst relative  : {worst:e}");
    if worst > BUDGET_RMS {
        println!("  FAIL  over budget");
        fails += 1;
    }

    // ---- short conv --------------------------------------------------------
    println!("\n=== 2. short conv, WITH its internal residual (budget {BUDGET_SCONV:e}) ===");
    let xs = read_f32(&dir, "blk_sconv_x.bin")?;
    let ws = read_f32(&dir, "blk_sconv_w.bin")?;
    let ys = read_f32(&dir, "blk_sconv_y.bin")?;
    let ys_bare = read_f32(&dir, "blk_sconv_y_noresid.bin")?;
    let mine = short_conv(&xs, &ws, t, h, k);
    let d = compare(&mine, &ys);
    checks += d.n;
    println!("  values compared : {}", d.n);
    println!(
        "  worst absolute  : {:e}   (tensor scale {:e})",
        d.worst_abs, d.ref_scale
    );
    println!("  worst abs / scale : {:e}  <- the criterion", d.scaled());
    println!(
        "  worst RELATIVE  : {:e} at [{}], mine {:e} vs ref {:e}",
        d.worst_rel, d.at, d.mine_at, d.ref_at
    );
    println!("    (a near-zero reference makes relative error meaningless there;\n     x + conv(x) cancels, so this is reported, not gated on)");
    if d.scaled() > BUDGET_SCONV {
        println!("  FAIL  over budget");
        fails += 1;
    }
    // Non-vacuity: the bare convolution must be a different function, or this
    // check cannot distinguish an implementation that forgot the residual.
    let (worst_bare, _) = worst_rel(&mine, &ys_bare);
    let spread = ys
        .iter()
        .zip(&ys_bare)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    checks += 1;
    println!("  worst relative vs the BARE conv   : {worst_bare:e}");
    println!("  |module - bare| max               : {spread:e}");
    if spread <= 0.0 {
        println!("  FAIL  residual is worth nothing here — this corpus cannot test it");
        fails += 1;
    } else if worst_bare <= BUDGET_SCONV {
        println!("  FAIL  matches the bare conv too — the residual is not being tested");
        fails += 1;
    } else {
        println!("  the two differ, so the residual is genuinely under test");
    }

    // ---- router ------------------------------------------------------------
    println!("\n=== 3. router (weights budget {BUDGET_ROUTER_W:e}, sets exact) ===");
    let xr = read_f32(&dir, "blk_router_x.bin")?;
    let wr = read_f32(&dir, "blk_router_w.bin")?;
    let br = read_f32(&dir, "blk_router_bias.bin")?;
    let gs = read_f32(&dir, "blk_router_gscale.bin")?;
    let ref_idx = read_i64(&dir, "blk_router_topk_idx.bin")?;
    let ref_w = read_f32(&dir, "blk_router_topk_w.bin")?;
    let ref_g = read_f32(&dir, "blk_router_gammas.bin")?;

    let routing = route(
        &xr,
        &wr,
        &br,
        gs[0],
        route_scale,
        t,
        h,
        n_routed,
        n_shared,
        top_k,
    );
    println!("  tokens routed   : {}", routing.len());
    anyhow::ensure!(
        routing.len() == t,
        "routed {} tokens, expected {t}",
        routing.len()
    );

    let mut set_bad = 0usize;
    let mut worst_w = 0f32;
    let mut worst_g = 0f32;
    for (ti, r) in routing.iter().enumerate() {
        checks += 1;
        // `torch.topk(sorted=False)` gives no order guarantee, so compare the
        // SET, then line the weights up by expert rather than by position.
        let mut mine_set: Vec<usize> = r.experts.clone();
        let mut ref_set: Vec<usize> = ref_idx[ti * top_k..(ti + 1) * top_k]
            .iter()
            .map(|&v| v as usize)
            .collect();
        mine_set.sort_unstable();
        ref_set.sort_unstable();
        if mine_set != ref_set {
            if set_bad < 4 {
                println!("  FAIL  token {ti}: chose {mine_set:?}, reference {ref_set:?}");
            }
            set_bad += 1;
            fails += 1;
            continue;
        }
        for (j, &e) in r.experts.iter().enumerate() {
            let pos = ref_idx[ti * top_k..(ti + 1) * top_k]
                .iter()
                .position(|&v| v as usize == e)
                .expect("set already matched");
            let a = r.weights[j];
            let b = ref_w[ti * top_k + pos];
            checks += 1;
            let e_rel = (a - b).abs() / b.abs().max(1e-6);
            if e_rel > worst_w {
                worst_w = e_rel;
            }
        }
        for s in 0..n_shared {
            checks += 1;
            let a = r.shared_gammas[s];
            let b = ref_g[ti * n_shared + s];
            let e_rel = (a - b).abs() / b.abs().max(1e-6);
            if e_rel > worst_g {
                worst_g = e_rel;
            }
        }
    }
    println!("  expert-set mismatches : {set_bad}");
    println!("  worst weight rel      : {worst_w:e}");
    println!("  worst shared-gamma rel: {worst_g:e}");
    if worst_w > BUDGET_ROUTER_W || worst_g > BUDGET_ROUTER_W {
        println!("  FAIL  over budget");
        fails += 1;
    }

    // The weights are normalized across routed AND shared together, so a token
    // sums to route_scale. Checking that catches a normalization done over the
    // routed experts alone, which is the natural wrong implementation.
    let mut worst_sum = 0f32;
    for r in &routing {
        checks += 1;
        let s: f32 = r.weights.iter().chain(&r.shared_gammas).sum();
        worst_sum = worst_sum.max((s - route_scale).abs());
    }
    println!(
        "  worst |sum - route_scale| : {worst_sum:e}  (shared experts share the normalization)"
    );
    if worst_sum > 1e-4 {
        println!("  FAIL  weights do not sum to route_scale");
        fails += 1;
    }

    // Non-vacuity: the bias must actually change the chosen set somewhere.
    checks += 1;
    println!("\n=== 4. is the score-correction bias under test? ===");
    let zero = vec![0f32; br.len()];
    let unbiased = route(
        &xr,
        &wr,
        &zero,
        gs[0],
        route_scale,
        t,
        h,
        n_routed,
        n_shared,
        top_k,
    );
    let differing = routing
        .iter()
        .zip(&unbiased)
        .filter(|(a, b)| {
            let mut x: Vec<usize> = a.experts.clone();
            let mut y: Vec<usize> = b.experts.clone();
            x.sort_unstable();
            y.sort_unstable();
            x != y
        })
        .count();
    println!("  tokens where dropping the bias changes the set: {differing} of {t}");
    println!("  the capture reported                          : {bias_changes} of {t}");
    if differing == 0 {
        println!("  FAIL  the bias changes nothing here — check 3 does not test it");
        fails += 1;
    } else {
        println!("  the bias is genuinely under test");
    }

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!("GATE PASSED — {checks} checks, block primitives match transformers");
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
