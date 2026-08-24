//! Parity gate for [`mary::models::k3::kda`] against the
//! `flash-linear-attention` oracle vectors.
//!
//! The oracle is **executed third-party code**,
//! not a re-reading of Kimi's modeling file: fla 0.5.2's Triton kernels on the
//! GB10 at Kimi's exact call sites and flags, plus fla's own reference
//! implementations, plus those same reference implementations re-run in float64
//! (fla's arithmetic, `torch.float` rebound to `float64`).
//!
//! ## Why the gate is run in two precisions
//!
//! Every check runs the port's f64 instantiation against the float64 arrays and
//! its f32 instantiation against the Triton kernels. That separation is the
//! whole point. An algebraic mistake — the decay applied after the delta rule
//! instead of before, RMS norm where an L2 norm belongs, the unbounded gate
//! branch — moves the f64 comparison by 1e-3 or more while hiding entirely
//! inside the ~1e-7 rounding noise of an fp32-vs-fp32 comparison. A gate that
//! only ever compared f32 to f32 would be a green check that proves nothing.
//!
//! ## Why per-step state, not just the output
//!
//! `o = qᵀS` is a projection of the state through a *normalized* query, so a
//! state that has drifted by a scale factor or lost a decay term still produces
//! plausible-looking output for many tokens. The state is what carries the
//! error. Every recurrence check therefore compares `S` after **each** token
//! against the oracle's per-step state arrays and reports the relative error
//! step by step, and the final output comparison is a corollary rather than the
//! evidence.
//!
//! ## Tolerances
//!
//! Fixed below from the oracle's own measured error budget, before the port was
//! run. They are not tuned to the result.
//!
//! Usage: `kimi_kda_gate [vectors_dir]`; without the argument, the vectors are
//! read from `$MARY_MODELS/k3-oracle`. No path is baked in.

use std::process::ExitCode;

use mary::models::k3::kda::{
    decay_gate, l2_normalize, rms_norm_gated, sigmoid, Elem, Kda, KdaConfig, KdaParams, KdaScratch,
    KdaState, KdaToken, ShortConv, ShortConvState,
};
use mary::nn::npz::{NpyArray, Npz};

/// f64 port vs the float64 oracle arrays. Two independently-written float64
/// implementations were measured to agree at 2e-17 (outputs) / 2e-16 (states);
/// 1e-12 leaves four decades for a third summation order and is still nine
/// decades tighter than any wrong-algebra defect would be.
const TOL_F64: f64 = 1e-12;

/// f32 port vs the Triton kernels. The kernel's own fp32-vs-float64 error is
/// 1.04e-08 absolute on |o|max 0.056, i.e. ~1.9e-07 relative; the port's
/// summation order differs, so allow ~50×. Still 240× tighter than
/// `chunk_kda`'s own 2.4e-03 relative error.
const TOL_F32: f64 = 1e-5;

/// `chunk_kda` output only. Its ~1.3e-04 absolute error is ~2.4e-03 relative
/// and is a property of the kernel's internal reduced-precision accumulation,
/// not of anything a port controls. Used exclusively for the state-layout trap,
/// where the signal (3.17e-02) is 13× the noise. Explicitly a loose
/// consistency check, never a correctness gate.
const TOL_CHUNK: f64 = 5e-3;

/// A negative control must miss by at least this much, relatively. Every
/// rejected alternative the oracle stored (reversed conv cache, the other
/// output-gate ordering, the wrong initial-state layout) missed by 0.5 or more,
/// so 1e-2 only asserts that the discriminator still discriminates.
const MIN_CONTROL_GAP: f64 = 1e-2;

fn main() -> ExitCode {
    let dir =
        mary::paths::model(std::env::args().nth(1).as_deref(), "k3-oracle").unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
    println!("KDA parity gate — oracle vectors in {}", dir.display());

    let kda = Npz::open(&dir.join("kda_oracle.npz")).expect("kda_oracle.npz");
    let fla64 = Npz::open(&dir.join("kda_oracle_f64_fla.npz")).expect("kda_oracle_f64_fla.npz");
    let onorm = Npz::open(&dir.join("kda_output_gate.npz")).expect("kda_output_gate.npz");
    println!(
        "loaded {} + {} + {} arrays\n",
        kda.len(),
        fla64.len(),
        onorm.len()
    );

    let mut r = Report::new();
    let model_dir = mary::paths::model(std::env::var("K3_MODEL_DIR").ok().as_deref(), "kimi-k3")
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
    check_shipping_config(&mut r, &model_dir.join("config.json"));
    check_premises(&mut r, &kda);
    check_gate_sweep(&mut r, &kda);
    check_stages(&mut r, &kda);
    check_recurrence(&mut r, &kda, &fla64);
    check_state_layout_trap(&mut r, &kda);
    check_short_conv(&mut r, &kda);
    check_output_gate(&mut r, &onorm);
    check_state_is_o1(&mut r);

    r.finish()
}

// ---------------------------------------------------------------------------
// comparison plumbing
// ---------------------------------------------------------------------------

/// The shape of a disagreement: worst absolute difference, the reference's own
/// magnitude, and their ratio. Reporting all three matters — an absolute number
/// alone is unreadable without the tensor's scale, and a ratio alone hides a
/// comparison against an all-but-zero reference.
#[derive(Debug, Clone, Copy)]
struct Cmp {
    maxabs: f64,
    refmax: f64,
    rel: f64,
    at: usize,
}

fn compare<E: Elem>(mine: &[E], reference: &[f64]) -> Cmp {
    assert_eq!(
        mine.len(),
        reference.len(),
        "length mismatch: {} vs {}",
        mine.len(),
        reference.len()
    );
    let mut maxabs = 0.0f64;
    let mut at = 0usize;
    let mut refmax = 0.0f64;
    for (i, (&m, &r)) in mine.iter().zip(reference).enumerate() {
        let d = (m.to_f64() - r).abs();
        // `!(d <= maxabs)`, NOT `d > maxabs`. For d = NaN the latter is FALSE,
        // so a non-finite element never updates the maximum and scores as ZERO
        // error — the metric silently agrees with any garbage output. Injecting
        // a NaN into every token of every case previously yielded 72/72 PASS
        // with rows printing `maxabs 5.551e-17 rel 9.872e-16 PASS`.
        if !(d <= maxabs) {
            maxabs = d;
            at = i;
        }
        refmax = refmax.max(r.abs());
    }
    let rel = if refmax > 0.0 {
        maxabs / refmax
    } else {
        maxabs
    };
    Cmp {
        maxabs,
        refmax,
        rel,
        at,
    }
}

struct Check {
    name: String,
    cmp: Cmp,
    tol: f64,
    /// `true` for a negative control, where the port must *miss* the array.
    inverted: bool,
    pass: bool,
}

struct Report {
    checks: Vec<Check>,
}

impl Report {
    fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// The port must match `reference` to `tol` relative.
    fn expect_match<E: Elem>(
        &mut self,
        name: &str,
        mine: &[E],
        reference: &[f64],
        tol: f64,
    ) -> Cmp {
        let cmp = compare(mine, reference);
        let pass = cmp.rel <= tol && cmp.rel.is_finite();
        println!(
            "  {:<62} maxabs {:9.3e}  |ref|max {:9.3e}  rel {:9.3e}  tol {:7.1e}  {}",
            name,
            cmp.maxabs,
            cmp.refmax,
            cmp.rel,
            tol,
            if pass { "PASS" } else { "FAIL" }
        );
        self.checks.push(Check {
            name: name.to_string(),
            cmp,
            tol,
            inverted: false,
            pass,
        });
        cmp
    }

    /// The port must NOT match `reference` — a rejected alternative the oracle
    /// stored on purpose. A gate that only asserts agreement can be satisfied
    /// by a discriminator that has quietly stopped discriminating.
    fn expect_miss<E: Elem>(&mut self, name: &str, mine: &[E], reference: &[f64]) -> Cmp {
        let cmp = compare(mine, reference);
        let pass = cmp.rel >= MIN_CONTROL_GAP;
        println!(
            "  {:<62} maxabs {:9.3e}  |ref|max {:9.3e}  rel {:9.3e}  min {:7.1e}  {}",
            name,
            cmp.maxabs,
            cmp.refmax,
            cmp.rel,
            MIN_CONTROL_GAP,
            if pass { "PASS" } else { "FAIL" }
        );
        self.checks.push(Check {
            name: name.to_string(),
            cmp,
            tol: MIN_CONTROL_GAP,
            inverted: true,
            pass,
        });
        cmp
    }

    fn expect_true(&mut self, name: &str, cond: bool, detail: &str) {
        println!(
            "  {:<62} {:<52} {}",
            name,
            detail,
            if cond { "PASS" } else { "FAIL" }
        );
        self.checks.push(Check {
            name: name.to_string(),
            cmp: Cmp {
                maxabs: 0.0,
                refmax: 0.0,
                rel: 0.0,
                at: 0,
            },
            tol: 0.0,
            inverted: false,
            pass: cond,
        });
    }

    fn finish(self) -> ExitCode {
        let failed: Vec<&Check> = self.checks.iter().filter(|c| !c.pass).collect();
        println!("\n{}", "=".repeat(120));
        if failed.is_empty() {
            println!("GATE PASS — {} checks, 0 failures", self.checks.len());
            ExitCode::SUCCESS
        } else {
            println!(
                "GATE FAIL — {} checks, {} failures:",
                self.checks.len(),
                failed.len()
            );
            for c in &failed {
                println!(
                    "  {} — rel {:.3e} vs {} {:.1e} (worst element #{})",
                    c.name,
                    c.cmp.rel,
                    if c.inverted { "required min" } else { "tol" },
                    c.tol,
                    c.cmp.at
                );
            }
            println!("\nNO PERFORMANCE NUMBER IS REPORTED: the correctness gate did not pass.");
            ExitCode::FAILURE
        }
    }
}

/// Load an oracle array into the port's element type. `.to_f64()` then a cast
/// reproduces exactly what the generator did (`torch.tensor(x_f64).float()`),
/// and is bit-exact for arrays already stored as f32.
fn arr<E: Elem>(a: &NpyArray) -> Vec<E> {
    a.to_f64().into_iter().map(E::from_f64).collect()
}

// ---------------------------------------------------------------------------
// 1. premises
// ---------------------------------------------------------------------------

/// The constants this port hard-codes, checked against the oracle rather than
/// assumed. Cheap, and a premise error has been the expensive failure mode on
/// this model.
/// Check the SHIPPING config against the checkpoint's own `config.json`.
///
/// `KdaConfig::k3()` had exactly one occurrence in the tree — its own
/// definition. Every check ran a 4-head `case_cfg()`, so `96` appeared nowhere
/// and corrupting `k3()` to `num_heads: 7, head_k_dim: 3, conv_kernel: 9,
/// gate_lower_bound: None` still yielded 72/72 and exit 0. The gate proved the
/// algorithm and never touched the configuration the model will actually run
/// with.
///
/// This reads the checkpoint, not the oracle. The previous "premise check"
/// compared the port's -5.0 against the oracle npz's -5.0 — both descended from
/// one reading of config.json, so it could only ever confirm that reading
/// against itself.
fn check_shipping_config(r: &mut Report, config_json: &std::path::Path) {
    let raw = match std::fs::read_to_string(config_json) {
        Ok(t) => t,
        Err(e) => {
            r.expect_true(
                "shipping config: config.json readable",
                false,
                &format!("{}: {e}", config_json.display()),
            );
            return;
        }
    };
    // Deliberately not a JSON dependency: these are unambiguous scalar fields
    // and a regex keeps the gate free of a parser whose own behaviour would
    // then need gating.
    let field = |name: &str| -> Option<f64> { regex_lite_find(&raw, name) };
    let cfg = mary::models::k3::kda::KdaConfig::k3();

    let checks: [(&str, Option<f64>, f64); 4] = [
        ("num_heads", field("\"num_heads\""), cfg.num_heads as f64),
        ("head_dim", field("\"head_dim\""), cfg.head_k_dim as f64),
        (
            "short_conv_kernel_size",
            field("\"short_conv_kernel_size\""),
            cfg.conv_kernel as f64,
        ),
        (
            "gate_lower_bound",
            field("\"gate_lower_bound\""),
            cfg.gate_lower_bound.unwrap_or(f64::NAN),
        ),
    ];
    for (name, found, mine) in checks {
        match found {
            Some(v) => r.expect_true(
                &format!("shipping config: {name}"),
                (v - mine).abs() < 1e-9,
                &format!("config.json {v} vs KdaConfig::k3() {mine}"),
            ),
            None => r.expect_true(
                &format!("shipping config: {name}"),
                false,
                "field absent from config.json",
            ),
        }
    }
    // head_v_dim is not a separate config field on this checkpoint; assert the
    // identity the port relies on rather than leaving it unchecked.
    r.expect_true(
        "shipping config: head_v_dim == head_k_dim",
        cfg.head_v_dim == cfg.head_k_dim,
        &format!("{} vs {}", cfg.head_v_dim, cfg.head_k_dim),
    );
}

/// Find `"name": <number>` in raw JSON. Returns the FIRST occurrence inside the
/// linear_attn_config block when present, else the first anywhere.
fn regex_lite_find(raw: &str, quoted_name: &str) -> Option<f64> {
    let scope = raw
        .find("\"linear_attn_config\"")
        .map(|i| &raw[i..])
        .unwrap_or(raw);
    for hay in [scope, raw] {
        if let Some(i) = hay.find(quoted_name) {
            let rest = &hay[i + quoted_name.len()..];
            let rest = rest.trim_start().strip_prefix(':')?.trim_start();
            let end = rest
                .find(|c: char| {
                    !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e')
                })
                .unwrap_or(rest.len());
            if let Ok(v) = rest[..end].parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

fn check_premises(r: &mut Report, kda: &Npz) {
    println!("=== 1. Premises ===");
    let lb = kda.get("gate_lower_bound").scalar();
    r.expect_true(
        "gate_lower_bound == -5.0",
        lb == -5.0,
        &format!("oracle says {}", lb),
    );

    // The recurrence cases run K = 128, so the port's K^-0.5 must be the
    // oracle's scale exactly, not merely nearly.
    let cfg = case_cfg();
    let want = kda.get("small_scale").scalar();
    r.expect_true(
        "q scale == K^-0.5 (bit-exact)",
        cfg.q_scale() == want,
        &format!("port {:.17} vs oracle {:.17}", cfg.q_scale(), want),
    );

    // `e^-5` is the structural retention floor the bounded gate implies.
    let floor = (-5.0f64).exp();
    r.expect_true(
        "retention floor e^-5",
        (floor - 0.0067379469990854670).abs() < 1e-18,
        &format!("e^-5 = {:.19}", floor),
    );
}

/// The shape the three recurrence cases share: 4 heads, K = V = 128.
fn case_cfg() -> KdaConfig {
    KdaConfig {
        num_heads: 4,
        head_k_dim: 128,
        head_v_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        l2norm_eps: 1e-6,
    }
}

// ---------------------------------------------------------------------------
// 2. the decay gate, in isolation
// ---------------------------------------------------------------------------

/// The 382-point × 8-`A_log` sweep, both branches. `z = g_raw + dt_bias` runs
/// to ±1e3 and `exp(A_log)` to 32, so the sigmoid's argument reaches ±3.2e4 —
/// the range that breaks a naive `1/(1+exp(-x))`.
fn check_gate_sweep(r: &mut Report, kda: &Npz) {
    println!("\n=== 2. Decay gate (isolated sweep, [3, 8, 128]) ===");
    let a_log64: Vec<f64> = kda.get("gate_A_log_f64").to_f64();
    let dt64: Vec<f64> = kda.get("gate_dt_bias_f64").to_f64();
    let g64: Vec<f64> = kda.get("gate_g_raw_f64").to_f64();
    let (t_n, h_n, k_n) = {
        let s = &kda.get("gate_g_raw_f64").shape;
        (s[0], s[1], s[2])
    };

    let mut bounded64 = vec![0.0f64; t_n * h_n * k_n];
    let mut softplus64 = vec![0.0f64; t_n * h_n * k_n];
    let mut bounded32 = vec![0.0f32; t_n * h_n * k_n];
    let mut softplus32 = vec![0.0f32; t_n * h_n * k_n];
    for t in 0..t_n {
        for h in 0..h_n {
            for k in 0..k_n {
                let i = (t * h_n + h) * k_n + k;
                let (a, gr, db) = (a_log64[h], g64[i], dt64[h * k_n + k]);
                bounded64[i] = decay_gate(a, gr, db, Some(-5.0));
                softplus64[i] = decay_gate(a, gr, db, None);
                bounded32[i] = decay_gate(a as f32, gr as f32, db as f32, Some(-5.0f32));
                softplus32[i] = decay_gate(a as f32, gr as f32, db as f32, None);
            }
        }
    }

    r.expect_match(
        "f64 bounded branch vs gate_out_lowerbound_f64_numpy",
        &bounded64,
        &kda.get("gate_out_lowerbound_f64_numpy").to_f64(),
        TOL_F64,
    );
    // The Triton kernel's own fp32-vs-float64 error is 2.82e-06 on the bounded
    // branch and 5.00e-05 on the softplus branch (whose values reach -3.2e4),
    // so these compare against the kernel at its own precision, not tighter.
    r.expect_match(
        "f32 bounded branch vs fused_kda_gate (Triton, GB10)",
        &bounded32,
        &kda.get("gate_out_lowerbound_f32_fla_triton").to_f64(),
        TOL_F32,
    );
    r.expect_match(
        "f32 softplus branch vs fused_kda_gate (Triton, GB10)",
        &softplus32,
        &kda.get("gate_out_softplus_f32_fla_triton").to_f64(),
        TOL_F32,
    );

    // --- the softplus branch's float64 column, and a disagreement worth naming ---
    //
    // Every implementation that actually RUNS linearizes softplus above 20:
    // torch's `F.softplus` (beta=1, threshold=20), and fla's Triton kernel,
    // whose generated PTX literally reads `setp.gt.f32 p, $in, 20.`
    // (`fla/ops/utils/softplus.py`). This port does the same. The oracle's
    // float64 column was transcribed with a threshold of **30**
    // (`gen_kda.py:74`), so above z = 20 the two definitions differ by
    // exp(A_log)·log1p(exp(−z)) — 5.14e-08 at the sweep's z = 20.25 with
    // exp(A_log) = 32, i.e. 1.6e-12 relative on values of 3.2e4.
    //
    // That is a defect in one oracle array, on the branch Kimi K3 does not take
    // (`gate_lower_bound = -5.0`). It is stated here as an exact identity
    // rather than absorbed into a widened tolerance: the region where the
    // definitions agree is still checked at full f64 precision, and the region
    // where they disagree is explained by reproducing the threshold-30 form and
    // showing it *is* the stored array.
    let z_all: Vec<f64> = kda.get("gate_z_f64").to_f64();
    let sp_ref: Vec<f64> = kda.get("gate_out_softplus_f64_numpy").to_f64();
    let sp_tri: Vec<f64> = kda.get("gate_out_softplus_f32_fla_triton").to_f64();
    let mut t30 = vec![0.0f64; sp_ref.len()];
    let (mut mine_lo, mut ref_lo) = (Vec::new(), Vec::new());
    let (mut mine_hi32, mut tri_hi32) = (Vec::new(), Vec::new());
    let mut worst_hi = 0.0f64;
    for t in 0..t_n {
        for h in 0..h_n {
            for k in 0..k_n {
                let i = (t * h_n + h) * k_n + k;
                let z = z_all[i];
                t30[i] = -a_log64[h].exp() * if z > 30.0 { z } else { z.exp().ln_1p() };
                if z <= 20.0 {
                    mine_lo.push(softplus64[i]);
                    ref_lo.push(sp_ref[i]);
                } else {
                    worst_hi = worst_hi.max((softplus64[i] - sp_ref[i]).abs());
                    mine_hi32.push(softplus32[i]);
                    tri_hi32.push(sp_tri[i]);
                }
            }
        }
    }
    r.expect_match(
        "f64 softplus, z <= 20 (where every definition agrees)",
        &mine_lo,
        &ref_lo,
        TOL_F64,
    );
    r.expect_match(
        "FINDING: gate_out_softplus_f64_numpy IS the threshold-30 form",
        &t30,
        &sp_ref,
        TOL_F64,
    );
    r.expect_match(
        "f32 softplus, z > 20 vs the Triton kernel (threshold 20, PTX-verified)",
        &mine_hi32,
        &tri_hi32,
        TOL_F32,
    );
    println!(
        "    note: {} of {} sweep points have z > 20; there the port (threshold 20,\n\
         \x20         torch + Triton) differs from the oracle's float64 column\n\
         \x20         (threshold 30) by at most {:.3e} absolute on |g|max 3.2e4.",
        mine_hi32.len(),
        z_all.len(),
        worst_hi
    );
    r.expect_match(
        "f32 bounded branch vs naive_kda_lowerbound_gate (fla torch)",
        &bounded32,
        &kda.get("gate_out_lowerbound_f32_fla_naive").to_f64(),
        TOL_F32,
    );

    // The consequence the module docs claim, measured on the oracle's own
    // sweep rather than asserted: retention is floored at e^-5 and never
    // reaches it, and the two branches are nowhere near each other.
    let floor = (-5.0f64).exp();
    let ret_min = bounded64
        .iter()
        .map(|g| g.exp())
        .fold(f64::INFINITY, f64::min);
    let ret_max = bounded64.iter().map(|g| g.exp()).fold(0.0f64, f64::max);
    r.expect_true(
        "bounded retention exp(g) stays in (e^-5, 1)",
        ret_min >= floor && ret_max <= 1.0 && (ret_min - floor).abs() / floor < 1e-9,
        &format!(
            "min {:.17} (e^-5 = {:.17}), max {:.6}",
            ret_min, floor, ret_max
        ),
    );
    let unbounded_min = softplus64
        .iter()
        .map(|g| g.exp())
        .fold(f64::INFINITY, f64::min);
    r.expect_true(
        "softplus branch would erase a channel outright",
        unbounded_min < 1e-30,
        &format!(
            "its min retention is {:.3e}, vs the bounded {:.3e}",
            unbounded_min, ret_min
        ),
    );
    // If the branches were close, taking the wrong one would be forgivable.
    r.expect_miss(
        "NEGATIVE CONTROL: bounded branch must not match softplus oracle",
        &bounded64,
        &kda.get("gate_out_softplus_f64_numpy").to_f64(),
    );

    let beta_raw: Vec<f64> = kda.get("beta_raw_f64").to_f64();
    let beta64: Vec<f64> = beta_raw.iter().map(|&b| sigmoid(b)).collect();
    let beta32: Vec<f32> = beta_raw.iter().map(|&b| sigmoid(b as f32)).collect();
    r.expect_match(
        "f64 sigmoid(beta) vs beta_sigmoid_f64_numpy",
        &beta64,
        &kda.get("beta_sigmoid_f64_numpy").to_f64(),
        TOL_F64,
    );
    r.expect_match(
        "f32 sigmoid(beta) vs beta_sigmoid_f32_torch",
        &beta32,
        &kda.get("beta_sigmoid_f32_torch").to_f64(),
        TOL_F32,
    );
}

// ---------------------------------------------------------------------------
// 3. the stages the kernel fuses
// ---------------------------------------------------------------------------

/// `use_qk_l2norm_in_kernel` / `use_gate_in_kernel` / `use_beta_sigmoid_in_kernel`
/// are all `True`, so the port performs these inside `step` and they are not
/// separately observable there. The oracle exports them as intermediates
/// precisely so a divergence can be localized before it reaches the state.
fn check_stages(r: &mut Report, kda: &Npz) {
    println!("\n=== 3. Fused stages (q/k L2 norm, gate, beta) ===");
    let cfg = case_cfg();
    for p in ["small_", "smallst_", "twochunk_"] {
        let shape = kda.get(&format!("{p}q_raw_f64")).shape.clone();
        let (b_n, t_n, h_n, k_n) = (shape[0], shape[1], shape[2], shape[3]);
        let q: Vec<f64> = kda.get(&format!("{p}q_raw_f64")).to_f64();
        let k: Vec<f64> = kda.get(&format!("{p}k_raw_f64")).to_f64();
        let g_raw: Vec<f64> = kda.get(&format!("{p}g_raw_f64")).to_f64();
        let a_log: Vec<f64> = kda.get(&format!("{p}A_log_f64")).to_f64();
        let dt: Vec<f64> = kda.get(&format!("{p}dt_bias_f64")).to_f64();
        let beta_raw: Vec<f64> = kda.get(&format!("{p}beta_raw_f64")).to_f64();

        let n = b_n * t_n * h_n * k_n;
        let (mut qn, mut kn, mut gg) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let eps = cfg.l2norm_eps;
        for i in 0..(b_n * t_n * h_n) {
            let (s, e) = (i * k_n, (i + 1) * k_n);
            l2_normalize(&q[s..e], eps, &mut qn[s..e]);
            l2_normalize(&k[s..e], eps, &mut kn[s..e]);
            let h = i % h_n;
            for j in 0..k_n {
                gg[s + j] = decay_gate(a_log[h], g_raw[s + j], dt[h * k_n + j], Some(-5.0));
            }
        }
        let bs: Vec<f64> = beta_raw.iter().map(|&b| sigmoid(b)).collect();

        r.expect_match(
            &format!("{p}q L2 norm (sum, eps inside sqrt)"),
            &qn,
            &kda.get(&format!("{p}q_l2norm_f64")).to_f64(),
            TOL_F64,
        );
        r.expect_match(
            &format!("{p}k L2 norm"),
            &kn,
            &kda.get(&format!("{p}k_l2norm_f64")).to_f64(),
            TOL_F64,
        );
        r.expect_match(
            &format!("{p}gate g"),
            &gg,
            &kda.get(&format!("{p}g_f64")).to_f64(),
            TOL_F64,
        );
        r.expect_match(
            &format!("{p}sigmoid(beta)"),
            &bs,
            &kda.get(&format!("{p}beta_sig_f64")).to_f64(),
            TOL_F64,
        );
    }

    // The trap the manifest calls out: RMS norm here would be a silent
    // sqrt(128) = 11.3x error. Show the wrong form actually is that far off,
    // so the check above is known to be able to see it.
    let q: Vec<f64> = kda.get("small_q_raw_f64").to_f64();
    let k_n = 128usize;
    let mut rms = vec![0.0f64; q.len()];
    for i in 0..(q.len() / k_n) {
        let (s, e) = (i * k_n, (i + 1) * k_n);
        let mean: f64 = q[s..e].iter().map(|x| x * x).sum::<f64>() / k_n as f64;
        let inv = 1.0 / (mean + 1e-6).sqrt();
        for j in s..e {
            rms[j] = q[j] * inv;
        }
    }
    r.expect_miss(
        "NEGATIVE CONTROL: RMS-norm form must not match q_l2norm",
        &rms,
        &kda.get("small_q_l2norm_f64").to_f64(),
    );
}

// ---------------------------------------------------------------------------
// 4. the recurrence
// ---------------------------------------------------------------------------

/// Drive one case through [`Kda::step`], collecting the state after every
/// token.
///
/// The `[T, B, HV, K, V]` history returned here is the *harness* collecting
/// evidence; the layer itself holds exactly one `[HV, K, V]` state per sequence
/// and never sees a `T`-shaped buffer (see `check_state_is_o1`).
#[allow(clippy::too_many_arguments)]
fn run_case<E: Elem>(
    cfg: &KdaConfig,
    params: &KdaParams<E>,
    b_n: usize,
    t_n: usize,
    q: &[E],
    k: &[E],
    v: &[E],
    g_raw: &[E],
    beta_raw: &[E],
    init_kv: Option<&[E]>,
) -> (Vec<E>, Vec<E>) {
    let kda = Kda::new(*cfg, params.clone());
    let (h_n, k_d, v_d) = (cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim);
    let (hk, hv) = (h_n * k_d, h_n * v_d);
    let st_elems = cfg.state_elems();

    let mut o = vec![E::ZERO; b_n * t_n * hv];
    let mut states = vec![E::ZERO; t_n * b_n * st_elems];
    let mut scr = KdaScratch::new(cfg);

    for b in 0..b_n {
        let mut st = match init_kv {
            Some(s0) => KdaState::from_kv(cfg, &s0[b * st_elems..(b + 1) * st_elems]),
            None => KdaState::zeros(cfg),
        };
        for t in 0..t_n {
            let tok_k = (b * t_n + t) * hk;
            let tok_v = (b * t_n + t) * hv;
            let tok_h = (b * t_n + t) * h_n;
            kda.step(
                &mut st,
                &mut scr,
                KdaToken {
                    q_raw: &q[tok_k..tok_k + hk],
                    k_raw: &k[tok_k..tok_k + hk],
                    v: &v[tok_v..tok_v + hv],
                    g_raw: &g_raw[tok_k..tok_k + hk],
                    beta_raw: &beta_raw[tok_h..tok_h + h_n],
                },
                &mut o[tok_v..tok_v + hv],
            );
            let dst = (t * b_n + b) * st_elems;
            states[dst..dst + st_elems].copy_from_slice(st.as_kv());
        }
    }
    (o, states)
}

/// Per-step relative error, printed step by step — the check that catches a
/// decay-term error the output alone would hide.
fn report_per_step<E: Elem>(
    r: &mut Report,
    label: &str,
    mine: &[E],
    reference: &[f64],
    t_n: usize,
    stride: usize,
    tol: f64,
    print_every: usize,
) {
    let mut worst = (0usize, 0.0f64);
    println!("    per-step state, {}:", label);
    for t in 0..t_n {
        let (s, e) = (t * stride, (t + 1) * stride);
        let c = compare(&mine[s..e], &reference[s..e]);
        if c.rel > worst.1 {
            worst = (t, c.rel);
        }
        if t % print_every == 0 || t == t_n - 1 {
            println!(
                "      t={:>4}  maxabs {:9.3e}  |S|max {:9.3e}  rel {:9.3e}",
                t, c.maxabs, c.refmax, c.rel
            );
        }
    }
    if print_every > 1 {
        println!(
            "      ({} of {} steps shown; every step was compared)",
            t_n.div_ceil(print_every),
            t_n
        );
    }
    println!("      worst step t={} at rel {:.3e}", worst.0, worst.1);
    r.expect_match(
        &format!("{} (max over all steps)", label),
        mine,
        reference,
        tol,
    );
}

fn check_recurrence(r: &mut Report, kda: &Npz, fla64: &Npz) {
    println!("\n=== 4. Recurrence — per-step state and output ===");
    let cfg = case_cfg();
    let st_elems = cfg.state_elems();

    for p in ["small_", "smallst_", "twochunk_"] {
        let shape = kda.get(&format!("{p}q_raw_f64")).shape.clone();
        let (b_n, t_n) = (shape[0], shape[1]);
        assert_eq!(shape[2], cfg.num_heads);
        assert_eq!(shape[3], cfg.head_k_dim);
        let every = if t_n > 32 { 16 } else { 1 };
        println!(
            "\n  -- {p} (B={b_n}, T={t_n}, HV={}, K=V={}) --",
            cfg.num_heads, cfg.head_k_dim
        );

        let params64 = KdaParams::new(
            &cfg,
            &arr::<f64>(kda.get(&format!("{p}A_log_f64"))),
            &arr::<f64>(kda.get(&format!("{p}dt_bias_f64"))),
        );
        let init64: Option<Vec<f64>> = kda
            .contains(&format!("{p}initial_state_f64"))
            .then(|| arr(kda.get(&format!("{p}initial_state_f64"))));

        // --- f64: does the algebra match, at machine precision? ---
        let (o64, s64) = run_case(
            &cfg,
            &params64,
            b_n,
            t_n,
            &arr::<f64>(kda.get(&format!("{p}q_raw_f64"))),
            &arr::<f64>(kda.get(&format!("{p}k_raw_f64"))),
            &arr::<f64>(kda.get(&format!("{p}v_f64"))),
            &arr::<f64>(kda.get(&format!("{p}g_raw_f64"))),
            &arr::<f64>(kda.get(&format!("{p}beta_raw_f64"))),
            init64.as_deref(),
        );
        report_per_step(
            r,
            &format!("{p}f64 state vs {p}state_per_step_f64_ref"),
            &s64,
            &kda.get(&format!("{p}state_per_step_f64_ref")).to_f64(),
            t_n,
            b_n * st_elems,
            TOL_F64,
            every,
        );
        // fla's OWN code, run in float64 — the reference with no authorship
        // shared with this port at all.
        let fla_key = format!("{p}state_per_step_fla_naive_f64");
        if fla64.contains(&fla_key) {
            report_per_step(
                r,
                &format!("{p}f64 state vs fla naive (float64)"),
                &s64,
                &fla64.get(&fla_key).to_f64(),
                t_n,
                b_n * st_elems,
                TOL_F64,
                every,
            );
        }
        r.expect_match(
            &format!("{p}f64 output vs {p}o_f64_ref"),
            &o64,
            &kda.get(&format!("{p}o_f64_ref")).to_f64(),
            TOL_F64,
        );
        r.expect_match(
            &format!("{p}f64 output vs fla naive_recurrent_kda (float64)"),
            &o64,
            &fla64.get(&format!("{p}o_fla_naive_recurrent_f64")).to_f64(),
            TOL_F64,
        );
        r.expect_match(
            &format!("{p}f64 final state vs fla naive (float64)"),
            &s64[(t_n - 1) * b_n * st_elems..],
            &fla64
                .get(&format!("{p}final_state_fla_naive_recurrent_f64"))
                .to_f64(),
            TOL_F64,
        );

        // --- f32: is single precision enough, against the Triton kernel? ---
        // Fed the exact reduced-precision inputs the kernel itself saw.
        let params32 = KdaParams::new(
            &cfg,
            &arr::<f32>(kda.get(&format!("{p}A_log_f64"))),
            &arr::<f32>(kda.get(&format!("{p}dt_bias_f64"))),
        );
        let init32: Option<Vec<f32>> = init64
            .as_ref()
            .map(|s| s.iter().map(|&x| x as f32).collect());
        let (o32, s32) = run_case(
            &cfg,
            &params32,
            b_n,
            t_n,
            &arr::<f32>(kda.get(&format!("{p}kin_f32_q"))),
            &arr::<f32>(kda.get(&format!("{p}kin_f32_k"))),
            &arr::<f32>(kda.get(&format!("{p}kin_f32_v"))),
            &arr::<f32>(kda.get(&format!("{p}kin_f32_g_raw"))),
            &arr::<f32>(kda.get(&format!("{p}beta_raw_f64"))),
            init32.as_deref(),
        );
        let step_key = format!("{p}state_per_step_kernel_f32");
        if kda.contains(&step_key) {
            report_per_step(
                r,
                &format!("{p}f32 state vs fused_recurrent_kda per-step (Triton)"),
                &s32,
                &kda.get(&step_key).to_f64(),
                t_n,
                b_n * st_elems,
                TOL_F32,
                every,
            );
        }
        r.expect_match(
            &format!("{p}f32 output vs fused_recurrent_kda (Triton)"),
            &o32,
            &kda.get(&format!("{p}o_fused_recurrent_f32")).to_f64(),
            TOL_F32,
        );
        r.expect_match(
            &format!("{p}f32 final state vs fused_recurrent_kda [K,V]"),
            &s32[(t_n - 1) * b_n * st_elems..],
            &kda.get(&format!("{p}final_state_fused_recurrent_f32_kv"))
                .to_f64(),
            TOL_F32,
        );
        // chunk_kda carries ~2.4e-3 of its own relative error; a loose
        // consistency check, labelled as such.
        r.expect_match(
            &format!("{p}f32 output vs chunk_kda (loose, kernel error 2.4e-3)"),
            &o32,
            &kda.get(&format!("{p}o_chunk_f32")).to_f64(),
            TOL_CHUNK,
        );
    }
}

// ---------------------------------------------------------------------------
// 5. the state-layout trap
// ---------------------------------------------------------------------------

/// Kimi passes `transpose_state_layout=True` — in fla 0.5.2 a deprecated alias
/// for `state_v_first`, which governs the layout of the *input* `initial_state`
/// too. fla's docstring says otherwise, and a port that believes it produces
/// output that is 56% wrong and entirely plausible.
///
/// So: the correct path must match the correct oracle, the deliberately-wrong
/// path must reproduce the stored negative control, and the two must be far
/// apart. Asserting only the first would pass even if `from_vk` were the
/// identity.
fn check_state_layout_trap(r: &mut Report, kda: &Npz) {
    println!("\n=== 5. state_v_first initial-state layout (the 56% trap) ===");
    let cfg = case_cfg();
    let st_elems = cfg.state_elems();
    let shape = kda.get("smallst_q_raw_f64").shape.clone();
    let (b_n, t_n) = (shape[0], shape[1]);

    let params32 = KdaParams::new(
        &cfg,
        &arr::<f32>(kda.get("smallst_A_log_f64")),
        &arr::<f32>(kda.get("smallst_dt_bias_f64")),
    );
    let (q, k, v, g) = (
        arr::<f32>(kda.get("smallst_kin_f32_q")),
        arr::<f32>(kda.get("smallst_kin_f32_k")),
        arr::<f32>(kda.get("smallst_kin_f32_v")),
        arr::<f32>(kda.get("smallst_kin_f32_g_raw")),
    );
    let beta = arr::<f32>(kda.get("smallst_beta_raw_f64"));
    let kv: Vec<f32> = arr(kda.get("smallst_initial_state_f64"));
    let vk: Vec<f32> = arr(kda.get("smallst_initial_state_vfirst_f64"));

    // from_vk on the v-first array must reconstruct the [K,V] state exactly.
    let via_vk: Vec<f32> = (0..b_n)
        .flat_map(|b| {
            KdaState::from_vk(&cfg, &vk[b * st_elems..(b + 1) * st_elems])
                .as_kv()
                .to_vec()
        })
        .collect();
    r.expect_match(
        "from_vk(v-first array) == the [K,V] initial state",
        &via_vk,
        &kv.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
        0.0,
    );

    let (o_right, _) = run_case(&cfg, &params32, b_n, t_n, &q, &k, &v, &g, &beta, Some(&kv));
    // The bug, committed on purpose: feed the v-first array as if it were [K,V].
    let (o_wrong, _) = run_case(&cfg, &params32, b_n, t_n, &q, &k, &v, &g, &beta, Some(&vk));

    r.expect_match(
        "correct h0 vs smallst_o_chunk_f32_vfirst (Kimi's actual call)",
        &o_right,
        &kda.get("smallst_o_chunk_f32_vfirst").to_f64(),
        TOL_CHUNK,
    );
    r.expect_miss(
        "NEGATIVE CONTROL: correct h0 must not match the WRONG-layout oracle",
        &o_right,
        &kda.get("smallst_o_chunk_f32_vfirst_WRONG_h0_layout")
            .to_f64(),
    );
    r.expect_match(
        "wrong h0 reproduces smallst_o_chunk_f32_vfirst_WRONG_h0_layout",
        &o_wrong,
        &kda.get("smallst_o_chunk_f32_vfirst_WRONG_h0_layout")
            .to_f64(),
        TOL_CHUNK,
    );
    let gap = compare(
        &o_wrong,
        &o_right.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
    );
    println!(
        "    the trap costs {:.3e} absolute on |o|max {:.3e} — {:.1}% relative",
        gap.maxabs,
        gap.refmax,
        100.0 * gap.rel
    );
}

// ---------------------------------------------------------------------------
// 6. the short convolution
// ---------------------------------------------------------------------------

fn check_short_conv(r: &mut Report, kda: &Npz) {
    println!("\n=== 6. Short convolution (depthwise, W=4, causal, silu) ===");
    let x_shape = kda.get("conv_x_f64").shape.clone();
    let (b_n, t_n, d_n) = (x_shape[0], x_shape[1], x_shape[2]);
    let w_n = kda.get("conv_weight_f64").shape[2];

    let conv64 = ShortConv::new(d_n, w_n, &arr::<f64>(kda.get("conv_weight_f64")));
    let conv32 = ShortConv::new(d_n, w_n, &arr::<f32>(kda.get("conv_weight_f64")));
    let x64: Vec<f64> = arr(kda.get("conv_x_f64"));
    let x32: Vec<f32> = arr(kda.get("conv_x_f64"));

    // Whole sequence, f64.
    let mut y64 = vec![0.0f64; b_n * t_n * d_n];
    let mut fin64 = vec![0.0f64; b_n * d_n * w_n];
    for b in 0..b_n {
        let mut st = ShortConvState::zeros(&conv64);
        conv64.forward(
            &mut st,
            t_n,
            &x64[b * t_n * d_n..(b + 1) * t_n * d_n],
            &mut y64[b * t_n * d_n..(b + 1) * t_n * d_n],
        );
        fin64[b * d_n * w_n..(b + 1) * d_n * w_n].copy_from_slice(st.as_slice());
    }
    r.expect_match(
        "f64 output vs conv_y_f64_ref",
        &y64,
        &kda.get("conv_y_f64_ref").to_f64(),
        TOL_F64,
    );
    r.expect_match(
        "f64 cache vs conv_final_state_f64_ref (most-recent LAST)",
        &fin64,
        &kda.get("conv_final_state_f64_ref").to_f64(),
        TOL_F64,
    );
    r.expect_miss(
        "NEGATIVE CONTROL: cache must not match the reversed ordering",
        &fin64,
        &kda.get("conv_final_state_f64_ref_reversed").to_f64(),
    );

    // Whole sequence, f32, vs fla's ShortConvolution.
    let mut y32 = vec![0.0f32; b_n * t_n * d_n];
    let mut fin32 = vec![0.0f32; b_n * d_n * w_n];
    for b in 0..b_n {
        let mut st = ShortConvState::zeros(&conv32);
        conv32.forward(
            &mut st,
            t_n,
            &x32[b * t_n * d_n..(b + 1) * t_n * d_n],
            &mut y32[b * t_n * d_n..(b + 1) * t_n * d_n],
        );
        fin32[b * d_n * w_n..(b + 1) * d_n * w_n].copy_from_slice(st.as_slice());
    }
    r.expect_match(
        "f32 output vs fla ShortConvolution",
        &y32,
        &kda.get("conv_y_f32_fla_full").to_f64(),
        TOL_F32,
    );
    r.expect_match(
        "f32 cache vs fla's returned cache",
        &fin32,
        &kda.get("conv_final_state_f32_fla_full").to_f64(),
        TOL_F32,
    );

    // Prefill 12 + continue 4, and 4 single-token decode steps: the streaming
    // boundary, which is where a cache-ordering mistake actually bites.
    let (pre, cont) = (12usize, 4usize);
    let mut y_pre = vec![0.0f32; b_n * pre * d_n];
    let mut st_pre = vec![0.0f32; b_n * d_n * w_n];
    let mut y_cont = vec![0.0f32; b_n * cont * d_n];
    let mut st_cont = vec![0.0f32; b_n * d_n * w_n];
    let mut y_step = vec![0.0f32; b_n * cont * d_n];
    let mut st_step = vec![0.0f32; cont * b_n * d_n * w_n];
    for b in 0..b_n {
        let xb = &x32[b * t_n * d_n..(b + 1) * t_n * d_n];
        let mut st = ShortConvState::zeros(&conv32);
        conv32.forward(
            &mut st,
            pre,
            xb,
            &mut y_pre[b * pre * d_n..(b + 1) * pre * d_n],
        );
        st_pre[b * d_n * w_n..(b + 1) * d_n * w_n].copy_from_slice(st.as_slice());
        // Continue from that cache, in one call and then step by step.
        let mut st_c = st.clone();
        conv32.forward(
            &mut st_c,
            cont,
            &xb[pre * d_n..],
            &mut y_cont[b * cont * d_n..(b + 1) * cont * d_n],
        );
        st_cont[b * d_n * w_n..(b + 1) * d_n * w_n].copy_from_slice(st_c.as_slice());
        let mut st_s = st;
        for j in 0..cont {
            let o = (b * cont + j) * d_n;
            conv32.step(
                &mut st_s,
                &xb[(pre + j) * d_n..(pre + j + 1) * d_n],
                &mut y_step[o..o + d_n],
            );
            let dst = (j * b_n + b) * d_n * w_n;
            st_step[dst..dst + d_n * w_n].copy_from_slice(st_s.as_slice());
        }
    }
    r.expect_match(
        "prefill(12) output vs fla",
        &y_pre,
        &kda.get("conv_y_f32_fla_prefill12").to_f64(),
        TOL_F32,
    );
    r.expect_match(
        "prefill(12) cache vs fla",
        &st_pre,
        &kda.get("conv_state_f32_fla_prefill12").to_f64(),
        TOL_F32,
    );
    r.expect_match(
        "continue(4) output vs fla",
        &y_cont,
        &kda.get("conv_y_f32_fla_continue4").to_f64(),
        TOL_F32,
    );
    r.expect_match(
        "continue(4) cache vs fla",
        &st_cont,
        &kda.get("conv_final_state_f32_fla_continue4").to_f64(),
        TOL_F32,
    );
    r.expect_match(
        "stepwise(4) output vs fla decode path",
        &y_step,
        &kda.get("conv_y_f32_fla_stepwise4").to_f64(),
        TOL_F32,
    );
    r.expect_match(
        "stepwise(4) per-step cache vs fla",
        &st_step,
        &kda.get("conv_state_f32_fla_stepwise4").to_f64(),
        TOL_F32,
    );
}

// ---------------------------------------------------------------------------
// 7. the output gate
// ---------------------------------------------------------------------------

fn check_output_gate(r: &mut Report, onorm: &Npz) {
    println!("\n=== 7. Output gate (FusedRMSNormGated, sigmoid) ===");
    let shape = onorm.get("onorm_x_f64").shape.clone();
    let d_n = *shape.last().unwrap();
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let x: Vec<f64> = onorm.get("onorm_x_f64").to_f64();
    let g: Vec<f64> = onorm.get("onorm_g_f64").to_f64();
    let w: Vec<f64> = onorm.get("onorm_weight_f64").to_f64();

    let mut y64 = vec![0.0f64; x.len()];
    let mut y32 = vec![0.0f32; x.len()];
    let (x32, g32, w32): (Vec<f32>, Vec<f32>, Vec<f32>) = (
        x.iter().map(|&v| v as f32).collect(),
        g.iter().map(|&v| v as f32).collect(),
        w.iter().map(|&v| v as f32).collect(),
    );
    for i in 0..rows {
        let (s, e) = (i * d_n, (i + 1) * d_n);
        rms_norm_gated(&x[s..e], &g[s..e], &w, 1e-5, &mut y64[s..e]);
        rms_norm_gated(&x32[s..e], &g32[s..e], &w32, 1e-5, &mut y32[s..e]);
    }
    r.expect_match(
        "f64 vs onorm_y_f64_candidateA (norm over the UNGATED x)",
        &y64,
        &onorm.get("onorm_y_f64_candidateA").to_f64(),
        TOL_F64,
    );
    r.expect_miss(
        "NEGATIVE CONTROL: must not match candidate B (rmsnorm(x*sigmoid(g))*w)",
        &y64,
        &onorm.get("onorm_y_f64_candidateB").to_f64(),
    );
    r.expect_match(
        "f32 vs the executed FusedRMSNormGated kernel",
        &y32,
        &onorm.get("onorm_y_f32_fla").to_f64(),
        TOL_F32,
    );
}

// ---------------------------------------------------------------------------
// 8. the O(1) claim, measured
// ---------------------------------------------------------------------------

/// The state must not know how long the sequence is. `Box<[E]>` makes that
/// true by construction — there is no `push` to call — but a claim carried only
/// by a type is a claim nobody has watched fail, so drive it and weigh the
/// state at 1, 16, 128 and 1024 tokens.
fn check_state_is_o1(r: &mut Report) {
    println!("\n=== 8. State is O(1) in sequence length ===");
    let cfg = case_cfg();
    let params = KdaParams::new(
        &cfg,
        &vec![0.7f64; cfg.num_heads],
        &vec![-0.2f64; cfg.num_heads * cfg.head_k_dim],
    );
    let kda = Kda::new(cfg, params);
    let conv = ShortConv::new(
        cfg.num_heads * cfg.head_k_dim,
        cfg.conv_kernel,
        &vec![0.1f64; cfg.num_heads * cfg.head_k_dim * cfg.conv_kernel],
    );
    let (hk, hv) = (
        cfg.num_heads * cfg.head_k_dim,
        cfg.num_heads * cfg.head_v_dim,
    );

    let mut sizes = Vec::new();
    for &t_n in &[1usize, 16, 128, 1024] {
        let mut st = KdaState::<f64>::zeros(&cfg);
        let mut cst = ShortConvState::zeros(&conv);
        let mut scr = KdaScratch::new(&cfg);
        let (mut out, mut cout) = (vec![0.0f64; hv], vec![0.0f64; hk]);
        for t in 0..t_n {
            let x: Vec<f64> = (0..hk)
                .map(|i| ((i * 31 + t * 7) as f64 * 0.013).sin())
                .collect();
            conv.step(&mut cst, &x, &mut cout);
            kda.step(
                &mut st,
                &mut scr,
                KdaToken {
                    q_raw: &cout,
                    k_raw: &cout,
                    v: &cout[..hv],
                    g_raw: &x,
                    beta_raw: &vec![0.3f64; cfg.num_heads],
                },
                &mut out,
            );
        }
        println!(
            "    T={:>5}  KdaState {:>7} B   ShortConvState {:>5} B   |o|max {:.4}",
            t_n,
            st.byte_len(),
            cst.byte_len(),
            out.iter().fold(0.0f64, |a, b| a.max(b.abs()))
        );
        sizes.push((st.byte_len(), cst.byte_len()));
    }
    let flat = sizes.windows(2).all(|w| w[0] == w[1]);
    r.expect_true(
        "state bytes constant across T = 1 .. 1024",
        flat && sizes[0].0 == cfg.state_elems() * 8,
        &format!(
            "{} B recurrent + {} B conv, unchanged",
            sizes[0].0, sizes[0].1
        ),
    );
}
