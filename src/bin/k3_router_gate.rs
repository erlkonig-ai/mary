//! Parity gate for [`mary::models::k3::router`] against the Kimi K3
//! whole-layer oracle.
//!
//! ## What it is gated against
//!
//! `layer_oracle_prefix13_bf16.npz` — a real 13-layer prefix of Kimi K3 driven
//! through the shipped `KimiLinearModel.forward` on real token ids, with
//! forward hooks on `block_sparse_moe.gate`. Of the arrays used here:
//!
//! * `L*_moe_gate_in_bf16bits`, `L*_moe_gate_out_topk_idx`,
//!   `L*_moe_gate_out_topk_weight` are **module input/output captures** — the
//!   shipped gate's own return values, with none of the oracle author's
//!   arithmetic between the module and the array.
//! * `L*_moe_router_{logits,scores,scores_for_choice,topk_weight_prerenorm}`
//!   are **derived-and-pinned**: the oracle recomputed the shipped expression
//!   to expose intermediates the module does not return, then asserted the
//!   recomputation against what the module *did* return (`topk_idx` bit-equal,
//!   `topk_weight` at 0.0). They are not independent of the shipped source, and
//!   the checks that use them are labelled so.
//! * `L*_moe_router_ALT_*` are stored **negative controls** — the wrong
//!   variants, captured on purpose.
//!
//! The stage checks therefore run in both directions: every stage is checked in
//! isolation against a pinned intermediate, and the end-to-end route is checked
//! against the two genuine captures.
//!
//! ## The router weight is a second artifact, and it is asserted directly
//!
//! The gate projection is not in the oracle npz, so it is exported out of the
//! checkpoint shards by `golden/export_k3_router_weights.py` into
//! `k3router_gateweights_routerport.npz` with a sidecar manifest carrying a
//! SHA-256 per array **computed from the safetensors shard**. This gate
//! recomputes SHA-256 over the bytes it parsed out of the npz and compares. The
//! digest crosses two implementations (Python `hashlib` vs the `sha2` crate)
//! and two parsers (`safetensors` vs `mary::nn::npz`), so it pins the artifact
//! to its source rather than round-tripping a writer against its own reader.
//!
//! Independently: the exported bias, rounded to bf16, must equal the oracle's
//! own captured `e_score_correction_bias` bit-for-bit — two unrelated
//! producers of the same 896 numbers.
//!
//! Usage: `k3_router_gate [vectors_dir] [model_dir]`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mary::models::k3::router::{
    bf16_bits_to_f32, Accum, Router, RouterActivation, RouterConfig, Scores, ScoresForChoice,
};
use mary::nn::npz::{NpyArray, NpyData, Npz};
use sha2::{Digest, Sha256};

/// MoE layers in the oracle prefix. Layer 0 is `first_k_dense_replace = 1`'s
/// dense MLP and has no router at all.
const LAYERS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

/// Tokens per capture (`[2, 16]` flattened).
const TOKENS: usize = 32;

/// f64-accumulated logits vs the oracle's f32 `F.linear`. The oracle's own
/// value is a *rounded f32*, so the floor here is one f32 ulp at |logit| ~ 4,
/// i.e. ~2.4e-07 absolute; measured worst over the three probed layers was
/// 4.4e-06 absolute / 9.7e-07 relative, which is the f32 rounding of a
/// 7168-term reduction. 1e-5 relative leaves an order of magnitude and is still
/// four decades below any algebraic defect (a transposed weight, a wrong layer,
/// a missing cast) which moves this by O(1).
const TOL_LOGITS_F64: f64 = 1e-5;

/// f32-accumulated logits: a strictly sequential 7168-term f32 sum, whose error
/// grows like sqrt(n)·eps·Σ|terms| rather than like the reference's blocked
/// reduction. Reported and bounded, never used to justify a selection.
const TOL_LOGITS_F32: f64 = 1e-3;

/// `sigmoid` on f32 input: within one f32 ulp at 1.0 (5.96e-08); 2 ulp of head
/// room.
const TOL_SCORES: f64 = 1.2e-7;

/// Renormalisation: a 16-term f32 sum whose order need not match torch's.
const TOL_NORM: f64 = 1e-6;

/// Bit-exact. `scores + bias` and `scores.gather(idx)` involve no reduction and
/// no transcendental, so anything but 0.0 is a real disagreement.
const TOL_EXACT: f64 = 0.0;

/// A negative control must miss by at least this much, absolutely. The stored
/// wrong-weight control misses the true weights by 8.93e-03 at its closest
/// layer, so this only asserts the discriminator still discriminates.
const MIN_CONTROL_GAP: f64 = 1e-3;

/// f32-vs-bf16 `e_score_correction_bias`: the number of the 384 token/layer
/// selections that change when the checkpoint's f32 bias is used instead of the
/// bf16 rounding the oracle's model held. Pinned so that a change in either
/// artifact, or in the port's selection, is caught rather than absorbed.
const EXPECTED_F32_BIAS_DIVERGENT_TOKENS: usize = 6;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "./k3-oracle".to_string()),
    );
    let model = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "./kimi-k3".to_string()),
    );
    println!("Kimi K3 MoE router gate");
    println!("  oracle vectors : {}", dir.display());
    println!("  checkpoint     : {}", model.display());

    let oracle = match Npz::open(&dir.join("layer_oracle_prefix13_bf16.npz")) {
        Ok(z) => z,
        Err(e) => {
            eprintln!("GATE FAIL — cannot open layer_oracle_prefix13_bf16.npz: {e}");
            return ExitCode::FAILURE;
        }
    };
    let weights = match Npz::open(&dir.join("k3router_gateweights_routerport.npz")) {
        Ok(z) => z,
        Err(e) => {
            eprintln!("GATE FAIL — cannot open k3router_gateweights_routerport.npz: {e}");
            return ExitCode::FAILURE;
        }
    };
    let manifest_raw =
        match std::fs::read_to_string(dir.join("k3router_gateweights_routerport_manifest.json")) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("GATE FAIL — cannot read the weight manifest: {e}");
                return ExitCode::FAILURE;
            }
        };
    let manifest: serde_json::Value = match serde_json::from_str(&manifest_raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("GATE FAIL — weight manifest is not JSON: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "  loaded {} oracle arrays + {} weight arrays\n",
        oracle.len(),
        weights.len()
    );

    let mut r = Report::new();

    check_shipping_config(&mut r, &model.join("config.json"));
    check_artifact(&mut r, &weights, &manifest, &oracle);
    check_premises(&mut r, &oracle, &weights);

    let mut gated: Vec<usize> = Vec::new();
    let mut f32_bias_divergent = 0usize;
    for &l in LAYERS.iter() {
        println!("--- layer {l} ---");
        let router = build_router(&weights, l);
        check_stages(&mut r, &oracle, &router, l);
        check_end_to_end(&mut r, &oracle, &router, l);
        check_controls(&mut r, &oracle, &router, l);
        f32_bias_divergent += measure_f32_bias(&mut r, &oracle, &weights, &router, l);
        gated.push(l);
        println!();
    }

    // ---- totality: no layer may be quietly skipped -------------------------
    r.expect_true(
        "totality: every MoE layer in the prefix was gated",
        gated == LAYERS.to_vec(),
        &format!("gated {gated:?}, expected {:?}", LAYERS),
    );
    r.expect_true(
        "f32-vs-bf16 bias: the divergence count is the pinned one",
        f32_bias_divergent == EXPECTED_F32_BIAS_DIVERGENT_TOKENS,
        &format!(
            "{f32_bias_divergent} of {} token selections change (pinned {})",
            LAYERS.len() * TOKENS,
            EXPECTED_F32_BIAS_DIVERGENT_TOKENS
        ),
    );

    r.finish()
}

// ---------------------------------------------------------------------------
// plumbing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Cmp {
    maxabs: f64,
    refmax: f64,
    rel: f64,
    at: usize,
    n: usize,
    /// Elements where the port's value, the reference, or their difference is
    /// not finite. Counted separately because a running maximum cannot hold a
    /// NaN — see [`compare`].
    nonfinite: usize,
}

/// Worst absolute difference, the reference's magnitude, and their ratio, plus
/// a count of non-finite elements.
///
/// ## Why the count exists, and why `!(d <= maxabs)` is not enough on its own
///
/// The usual advice is to write `!(d <= maxabs)` rather than `d > maxabs`,
/// because for `d = NaN` the latter is FALSE and a NaN would never update the
/// maximum. True, and necessary — but not sufficient. `!(d <= maxabs)` *does*
/// record the NaN, and then the very next element is compared against a NaN
/// maximum: `!(0.0 <= NaN)` is also TRUE, so a perfectly ordinary finite
/// difference **overwrites** it and the NaN vanishes. The maximum is
/// NaN-aware but not NaN-sticky.
///
/// This is not hypothetical. A mutant that wrote one `f32::NAN` into
/// `Router::normalize`'s output passed this gate 350/350 with the
/// `!(d <= maxabs)` form and no counter (mutation run 1, M21). Both are kept:
/// the counter is what fails the check, the comparator keeps `maxabs`
/// meaningful.
fn compare(mine: &[f32], reference: &[f64]) -> Cmp {
    assert_eq!(
        mine.len(),
        reference.len(),
        "length mismatch: {} vs {}",
        mine.len(),
        reference.len()
    );
    assert!(!mine.is_empty(), "comparing empty arrays proves nothing");
    let mut maxabs = 0.0f64;
    let mut at = 0usize;
    let mut refmax = 0.0f64;
    let mut nonfinite = 0usize;
    let mut first_bad = usize::MAX;
    for (i, (&m, &rf)) in mine.iter().zip(reference).enumerate() {
        let d = (m as f64 - rf).abs();
        if !m.is_finite() || !rf.is_finite() || !d.is_finite() {
            nonfinite += 1;
            if first_bad == usize::MAX {
                first_bad = i;
            }
            continue;
        }
        if !(d <= maxabs) {
            maxabs = d;
            at = i;
        }
        if !(rf.abs() <= refmax) {
            refmax = rf.abs();
        }
    }
    if first_bad != usize::MAX {
        at = first_bad;
    }
    let rel = if refmax > 0.0 { maxabs / refmax } else { maxabs };
    Cmp {
        maxabs,
        refmax,
        rel,
        at,
        n: mine.len(),
        nonfinite,
    }
}

struct Check {
    name: String,
    detail: String,
    pass: bool,
}

struct Report {
    checks: Vec<Check>,
}

impl Report {
    fn new() -> Self {
        Self { checks: Vec::new() }
    }

    fn push(&mut self, name: &str, detail: String, pass: bool) {
        println!(
            "  {:<64} {:<58} {}",
            name,
            detail,
            if pass { "PASS" } else { "FAIL" }
        );
        self.checks.push(Check {
            name: name.to_string(),
            detail,
            pass,
        });
    }

    /// Absolute-error match. Absolute, not relative: several of these compare
    /// probabilities in [0, 1] where a relative bound against a near-zero
    /// reference is meaningless.
    fn expect_match(&mut self, name: &str, mine: &[f32], reference: &[f64], tol: f64) -> Cmp {
        let c = compare(mine, reference);
        let pass = c.nonfinite == 0 && !(c.maxabs > tol) && c.maxabs.is_finite();
        self.push(
            name,
            format!(
                "n {:>7}  maxabs {:9.3e}  |ref|max {:9.3e}  rel {:9.3e}  tol {:7.1e}{}",
                c.n,
                c.maxabs,
                c.refmax,
                c.rel,
                tol,
                nonfinite_note(&c)
            ),
            pass,
        );
        c
    }

    /// The port (or a hand-built wrong variant) must MISS this array — it is a
    /// stored alternative that is wrong on purpose.
    fn expect_miss(&mut self, name: &str, mine: &[f32], reference: &[f64]) -> Cmp {
        let c = compare(mine, reference);
        // a control that "misses" because it went non-finite is not a miss
        let pass = c.nonfinite == 0 && !(c.maxabs <= MIN_CONTROL_GAP);
        self.push(
            name,
            format!(
                "n {:>7}  maxabs {:9.3e}  |ref|max {:9.3e}  required min {:7.1e}{}",
                c.n,
                c.maxabs,
                c.refmax,
                MIN_CONTROL_GAP,
                nonfinite_note(&c)
            ),
            pass,
        );
        c
    }

    fn expect_true(&mut self, name: &str, cond: bool, detail: &str) {
        self.push(name, detail.to_string(), cond);
    }

    fn finish(self) -> ExitCode {
        let failed: Vec<&Check> = self.checks.iter().filter(|c| !c.pass).collect();
        println!("\n{}", "=".repeat(128));
        if self.checks.len() < 300 {
            println!(
                "GATE FAIL — only {} checks ran; the gate is expected to run 300+. \
                 A gate that stopped early passes vacuously.",
                self.checks.len()
            );
            return ExitCode::FAILURE;
        }
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
                println!("  {}  |  {}", c.name, c.detail);
            }
            println!("\nNO PERFORMANCE NUMBER IS REPORTED: the correctness gate did not pass.");
            ExitCode::FAILURE
        }
    }
}

fn nonfinite_note(c: &Cmp) -> String {
    if c.nonfinite == 0 {
        String::new()
    } else {
        format!("  NON-FINITE {} (first at #{})", c.nonfinite, c.at)
    }
}

fn key(l: usize, s: &str) -> String {
    format!("L{l:02}_{s}")
}

/// An oracle array as f64, asserting it is not empty first.
fn f64s(z: &Npz, name: &str) -> Vec<f64> {
    let a = z.get(name);
    assert!(!a.is_empty(), "oracle array '{name}' is EMPTY");
    a.to_f64()
}

/// A bf16-bit-pattern array widened to f32.
fn bf16s(z: &Npz, name: &str) -> Vec<f32> {
    let a = z.get(name);
    assert!(!a.is_empty(), "oracle array '{name}' is EMPTY");
    match &a.data {
        NpyData::U16(v) => bf16_bits_to_f32(v),
        other => panic!("'{name}' is not uint16 bf16 bits: {other:?}"),
    }
}

fn u16s<'a>(z: &'a Npz, name: &str) -> &'a [u16] {
    let a = z.get(name);
    assert!(!a.is_empty(), "array '{name}' is EMPTY");
    match &a.data {
        NpyData::U16(v) => v,
        other => panic!("'{name}' is not uint16: {other:?}"),
    }
}

fn idx_u32(z: &Npz, name: &str) -> Vec<u32> {
    let a = z.get(name);
    assert!(!a.is_empty(), "oracle array '{name}' is EMPTY");
    match &a.data {
        NpyData::I64(v) => v.iter().map(|&x| x as u32).collect(),
        other => panic!("'{name}' is not int64: {other:?}"),
    }
}

fn shape_of<'a>(z: &'a Npz, name: &str) -> &'a [usize] {
    let a: &NpyArray = z.get(name);
    &a.shape
}

// ---------------------------------------------------------------------------
// 0. the shipping config, read from the checkpoint
// ---------------------------------------------------------------------------

/// Every field `RouterConfig::k3()` hard-codes, re-read from the
/// checkpoint's own `config.json`.
///
/// This exists because a port's config constants are the one thing an oracle
/// comparison cannot reach: the gate could be perfect on 32 tokens of 12 layers
/// and the shipping constant still be wrong.
fn check_shipping_config(r: &mut Report, config_json: &Path) {
    println!("--- shipping config (checkpoint config.json) ---");
    let raw = match std::fs::read_to_string(config_json) {
        Ok(t) => t,
        Err(e) => {
            r.expect_true(
                "config.json readable",
                false,
                &format!("{}: {e}", config_json.display()),
            );
            return;
        }
    };
    let v: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            r.expect_true("config.json parses", false, &format!("{e}"));
            return;
        }
    };
    let tc = v.get("text_config").unwrap_or(&v);
    let cfg = RouterConfig::k3();

    let num = |k: &str| -> Option<f64> { tc.get(k).and_then(|x| x.as_f64()) };
    let text = |k: &str| -> Option<&str> { tc.get(k).and_then(|x| x.as_str()) };
    let flag = |k: &str| -> Option<bool> { tc.get(k).and_then(|x| x.as_bool()) };

    let numeric: [(&str, Option<f64>, f64); 6] = [
        ("hidden_size", num("hidden_size"), cfg.hidden_size as f64),
        ("num_experts", num("num_experts"), cfg.num_experts as f64),
        (
            "num_experts_per_token",
            num("num_experts_per_token"),
            cfg.top_k as f64,
        ),
        (
            "routed_scaling_factor",
            num("routed_scaling_factor"),
            cfg.routed_scaling_factor as f64,
        ),
        (
            "num_expert_group",
            num("num_expert_group"),
            cfg.num_expert_group as f64,
        ),
        ("topk_group", num("topk_group"), cfg.topk_group as f64),
    ];
    for (name, got, want) in numeric {
        r.expect_true(
            &format!("config: {name}"),
            got == Some(want),
            &format!("checkpoint {got:?} vs port {want}"),
        );
    }
    r.expect_true(
        "config: moe_router_activation_func",
        text("moe_router_activation_func") == Some("sigmoid")
            && cfg.activation == RouterActivation::Sigmoid,
        &format!("checkpoint {:?} vs port Sigmoid", text("moe_router_activation_func")),
    );
    r.expect_true(
        "config: moe_renormalize",
        flag("moe_renormalize") == Some(cfg.renormalize),
        &format!("checkpoint {:?} vs port {}", flag("moe_renormalize"), cfg.renormalize),
    );
    // The premise this task was handed says "topk_method noaux_tc". It is in
    // the config — and the shipped gate never reads it. Assert both halves.
    r.expect_true(
        "config: topk_method is noaux_tc (and is DEAD on this config)",
        text("topk_method") == Some("noaux_tc")
            && cfg.num_expert_group == 1
            && !(cfg.num_expert_group > 1 && cfg.num_expert_group > cfg.topk_group),
        &format!(
            "topk_method {:?}, num_expert_group {} -> grouped branch not taken",
            text("topk_method"),
            cfg.num_expert_group
        ),
    );
    r.expect_true(
        "config: first_k_dense_replace = 1, so layer 0 has no router",
        num("first_k_dense_replace") == Some(1.0) && LAYERS[0] == 1,
        &format!("first_k_dense_replace {:?}, first gated layer {}", num("first_k_dense_replace"), LAYERS[0]),
    );
    r.expect_true(
        "port: the grouped-routing branch is refused, not faked",
        RouterConfig {
            num_expert_group: 8,
            topk_group: 4,
            ..cfg
        }
        .validate()
        .is_err()
            && cfg.validate().is_ok(),
        "num_expert_group=8/topk_group=4 rejected; shipping config accepted",
    );
    println!();
}

// ---------------------------------------------------------------------------
// 1. the exported weight artifact, asserted directly
// ---------------------------------------------------------------------------

fn check_artifact(r: &mut Report, w: &Npz, manifest: &serde_json::Value, oracle: &Npz) {
    println!("--- weight artifact (SHA-256 against the safetensors shards) ---");
    let arrays = manifest
        .get("arrays")
        .and_then(|a| a.as_object())
        .expect("manifest has no 'arrays' object");
    r.expect_true(
        "manifest: one entry per exported array",
        arrays.len() == w.len() && arrays.len() == 3 * LAYERS.len(),
        &format!(
            "manifest {} entries, npz {} arrays, expected {}",
            arrays.len(),
            w.len(),
            3 * LAYERS.len()
        ),
    );

    for (name, meta) in arrays {
        let want = meta.get("sha256").and_then(|x| x.as_str()).expect("sha256");
        let want_shape: Vec<usize> = meta
            .get("shape")
            .and_then(|x| x.as_array())
            .expect("shape")
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let a = w.get(name);
        assert!(!a.is_empty(), "exported array '{name}' is EMPTY");
        // hash the little-endian bytes of exactly what the npz reader parsed
        let bytes: Vec<u8> = match &a.data {
            NpyData::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            NpyData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            other => panic!("'{name}': unexpected dtype {other:?}"),
        };
        let got = hex(&Sha256::digest(&bytes));
        r.expect_true(
            &format!("artifact {name}"),
            got == want && a.shape == want_shape,
            &format!("sha256 {}… shape {:?}", &got[..16], a.shape),
        );
    }

    println!("  -- the exported bias, independently produced, must equal the oracle's --");
    for &l in LAYERS.iter() {
        let mine = u16s(w, &key(l, "gate_bias_bf16bits"));
        let theirs = u16s(oracle, &key(l, "moe_router_e_score_correction_bias_bf16bits"));
        r.expect_true(
            &format!("L{l:02} exported bias == oracle-captured bias (bit-for-bit)"),
            mine == theirs && mine.len() == 896,
            &format!("{} values, {} differ", mine.len(), mine.iter().zip(theirs).filter(|(a, b)| a != b).count()),
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// 2. premises, checked against three independent sources
// ---------------------------------------------------------------------------

fn check_premises(r: &mut Report, o: &Npz, w: &Npz) {
    println!("--- premises (config / oracle capture / checkpoint shard must agree) ---");
    let cfg = RouterConfig::k3();
    for &l in LAYERS.iter() {
        let lg = shape_of(o, &key(l, "moe_router_logits")).to_vec();
        let ix = shape_of(o, &key(l, "moe_gate_out_topk_idx")).to_vec();
        let gi = shape_of(o, &key(l, "moe_gate_in_bf16bits")).to_vec();
        let gw = shape_of(w, &key(l, "gate_weight_bf16bits")).to_vec();
        let ok = lg == vec![TOKENS, cfg.num_experts]
            && ix == vec![TOKENS, cfg.top_k]
            && gi == vec![2, 16, cfg.hidden_size]
            && gw == vec![cfg.num_experts, cfg.hidden_size]
            && gi[0] * gi[1] == TOKENS;
        r.expect_true(
            &format!("L{l:02} shapes: 896 experts / 16 active / 7168 hidden, from 3 sources"),
            ok,
            &format!("logits {lg:?} idx {ix:?} gate_in {gi:?} W {gw:?}"),
        );
    }
    println!();
}

fn build_router(w: &Npz, l: usize) -> Router {
    let cfg = RouterConfig::k3();
    let weight = bf16s(w, &key(l, "gate_weight_bf16bits"));
    // the bf16 rounding, because that is what the oracle's model held
    let bias = bf16s(w, &key(l, "gate_bias_bf16bits"));
    Router::new(cfg, weight, bias).expect("router")
}

// ---------------------------------------------------------------------------
// 3. stages, each isolated on the oracle's own input
// ---------------------------------------------------------------------------

fn check_stages(r: &mut Report, o: &Npz, router: &Router, l: usize) {
    let cfg = *router.config();

    // --- logits: my matmul on the captured gate input --------------------
    let h = bf16s(o, &key(l, "moe_gate_in_bf16bits"));
    let ref_logits = f64s(o, &key(l, "moe_router_logits"));
    let mine64 = router.logits(&h, TOKENS, Accum::F64);
    r.expect_match(
        &format!("L{l:02} logits (f64 accum) vs oracle [pinned]"),
        &mine64,
        &ref_logits,
        TOL_LOGITS_F64,
    );
    let mine32 = router.logits(&h, TOKENS, Accum::F32);
    r.expect_match(
        &format!("L{l:02} logits (f32 accum) vs oracle [pinned]"),
        &mine32,
        &ref_logits,
        TOL_LOGITS_F32,
    );

    // --- scores: sigmoid of the ORACLE logits, so the matmul is out of it --
    let ref_scores = f64s(o, &key(l, "moe_router_scores"));
    let oracle_logits_f32: Vec<f32> = ref_logits.iter().map(|&x| x as f32).collect();
    let my_scores = router.scores(&oracle_logits_f32, TOKENS);
    r.expect_match(
        &format!("L{l:02} scores = sigmoid(oracle logits) [pinned]"),
        my_scores.as_slice(),
        &ref_scores,
        TOL_SCORES,
    );

    // --- scores_for_choice: ORACLE scores + my bias, bit-exact -------------
    let oracle_scores = Scores::from_raw(
        ref_scores.iter().map(|&x| x as f32).collect(),
        TOKENS,
        cfg.num_experts,
    );
    let my_sfc = router.scores_for_choice(&oracle_scores);
    let ref_sfc = f64s(o, &key(l, "moe_router_scores_for_choice"));
    r.expect_match(
        &format!("L{l:02} scores_for_choice = oracle scores + bias (EXACT)"),
        my_sfc.as_slice(),
        &ref_sfc,
        TOL_EXACT,
    );

    // --- SELECTION, from the oracle's own scores_for_choice ---------------
    let oracle_sfc = ScoresForChoice::from_raw(
        ref_sfc.iter().map(|&x| x as f32).collect(),
        TOKENS,
        cfg.num_experts,
    );
    let my_idx = router.select(&oracle_sfc);
    let ref_idx = idx_u32(o, &key(l, "moe_gate_out_topk_idx"));
    let (same, ntok) = set_equal_rows(&my_idx, &ref_idx, cfg.top_k);
    r.expect_true(
        &format!("L{l:02} SELECTION from oracle sfc == captured topk_idx (as sets)"),
        same == ntok && ntok == TOKENS,
        &format!("{same}/{ntok} tokens agree"),
    );
    // shipped topk uses sorted=False; assert the disagreement is only ORDER
    r.expect_true(
        &format!("L{l:02} shipped topk_idx is NOT in the port's order (sorted=False)"),
        my_idx != ref_idx,
        "sets agree, sequences differ — order is not part of the contract",
    );
    r.expect_true(
        &format!("L{l:02} selection is well-determined (16th > 17th sfc, strictly)"),
        {
            let m = min_tie_margin(&ref_sfc, TOKENS, cfg.num_experts, cfg.top_k);
            m > 0.0
        },
        &format!(
            "min margin {:.6e}",
            min_tie_margin(&ref_sfc, TOKENS, cfg.num_experts, cfg.top_k)
        ),
    );
    r.expect_true(
        &format!("L{l:02} topk indices distinct, in range, exactly {}", cfg.top_k),
        idx_well_formed(&my_idx, cfg.top_k, cfg.num_experts, TOKENS),
        "16 distinct experts < 896 per token",
    );

    // --- WEIGHTING, from the ORACLE's idx: independent of selection --------
    let my_pre = router.combine_weights(&oracle_scores, &ref_idx);
    let ref_pre = f64s(o, &key(l, "moe_router_topk_weight_prerenorm"));
    r.expect_match(
        &format!("L{l:02} prerenorm weight = oracle scores @ ORACLE idx (EXACT)"),
        &my_pre,
        &ref_pre,
        TOL_EXACT,
    );

    // --- normalisation, from the ORACLE's prerenorm ------------------------
    let my_w = router.normalize(
        &ref_pre.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        TOKENS,
    );
    let ref_w = f64s(o, &key(l, "moe_gate_out_topk_weight"));
    r.expect_match(
        &format!("L{l:02} normalize(oracle prerenorm) == captured topk_weight"),
        &my_w,
        &ref_w,
        TOL_NORM,
    );
    r.expect_true(
        &format!("L{l:02} weights positive and sum to the scaling factor"),
        weights_sane(&ref_w, TOKENS, cfg.top_k, cfg.routed_scaling_factor as f64),
        "all > 0, row sums within 1e-6 of 1.0",
    );
}

// ---------------------------------------------------------------------------
// 4. end to end, against the two genuine module captures
// ---------------------------------------------------------------------------

fn check_end_to_end(r: &mut Report, o: &Npz, router: &Router, l: usize) {
    let cfg = *router.config();
    let h = bf16s(o, &key(l, "moe_gate_in_bf16bits"));
    let ref_idx = idx_u32(o, &key(l, "moe_gate_out_topk_idx"));
    let ref_w = f64s(o, &key(l, "moe_gate_out_topk_weight"));

    for (lane, acc, tol) in [
        ("f64 accum", Accum::F64, TOL_NORM),
        ("f32 accum", Accum::F32, TOL_NORM),
    ] {
        let out = router.route(&h, TOKENS, acc);
        let (same, ntok) = set_equal_rows(&out.idx, &ref_idx, cfg.top_k);
        r.expect_true(
            &format!("L{l:02} route [{lane}] selection == captured topk_idx"),
            same == ntok && ntok == TOKENS,
            &format!("{same}/{ntok} tokens agree"),
        );
        r.expect_true(
            &format!("L{l:02} route [{lane}] weights finite, positive, sum to the factor"),
            weights_sane(
                &out.weight.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                TOKENS,
                cfg.top_k,
                cfg.routed_scaling_factor as f64,
            ),
            "the PORT's own weights, not the oracle's",
        );
        // pair the weights THROUGH the index — the orders differ by construction
        match pair_by_index(&out.idx, &out.weight, &ref_idx, &ref_w, cfg.top_k, TOKENS) {
            Some((mine, theirs)) => {
                r.expect_match(
                    &format!("L{l:02} route [{lane}] weight per expert == captured"),
                    &mine,
                    &theirs,
                    tol,
                );
            }
            None => r.expect_true(
                &format!("L{l:02} route [{lane}] weight per expert == captured"),
                false,
                "index sets differ; weights not comparable",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// 5. negative controls — the port must FAIL these
// ---------------------------------------------------------------------------

fn check_controls(r: &mut Report, o: &Npz, router: &Router, l: usize) {
    let cfg = *router.config();
    let ref_scores = f64s(o, &key(l, "moe_router_scores"));
    let ref_sfc = f64s(o, &key(l, "moe_router_scores_for_choice"));
    let ref_idx = idx_u32(o, &key(l, "moe_gate_out_topk_idx"));
    let ref_w = f64s(o, &key(l, "moe_gate_out_topk_weight"));
    let ref_logits = f64s(o, &key(l, "moe_router_logits"));

    // --- C7: combining weight taken from the BIASED score ------------------
    // The port cannot express this — `combine_weights` takes `Scores`, and
    // `ScoresForChoice` is a different type. So the gate builds it by hand,
    // shows it reproduces the stored control, and shows the control is far from
    // the truth. Both halves matter: reproducing it proves the control is the
    // mistake it claims to be, and the gap proves the mistake is detectable.
    let sfc_as_scores = Scores::from_raw(
        ref_sfc.iter().map(|&x| x as f32).collect(),
        TOKENS,
        cfg.num_experts,
    );
    let wrong_pre = router.combine_weights(&sfc_as_scores, &ref_idx);
    let wrong_w = router.normalize(&wrong_pre, TOKENS);
    let ref_alt_w = f64s(
        o,
        &key(l, "moe_router_ALT_topk_weight_from_scores_for_choice"),
    );
    r.expect_match(
        &format!("L{l:02} CONTROL bias-in-weight reproduces the stored ALT array"),
        &wrong_w,
        &ref_alt_w,
        TOL_NORM,
    );
    r.expect_miss(
        &format!("L{l:02} CONTROL bias-in-weight MISSES the true topk_weight"),
        &wrong_w,
        &ref_w,
    );
    // and the port's own answer must be the true one, on the same inputs
    let right_pre = router.combine_weights(
        &Scores::from_raw(
            ref_scores.iter().map(|&x| x as f32).collect(),
            TOKENS,
            cfg.num_experts,
        ),
        &ref_idx,
    );
    let right_w = router.normalize(&right_pre, TOKENS);
    r.expect_miss(
        &format!("L{l:02} CONTROL the port's weight MISSES the ALT array"),
        &right_w,
        &ref_alt_w,
    );

    // --- C8: bias added to the LOGIT instead of the score ------------------
    let logits_plus_bias: Vec<f32> = ref_logits
        .iter()
        .enumerate()
        .map(|(i, &x)| x as f32 + router.bias()[i % cfg.num_experts])
        .collect();
    let alt_scores = router.scores(&logits_plus_bias, TOKENS);
    let alt_sfc = ScoresForChoice::from_raw(
        alt_scores.as_slice().to_vec(),
        TOKENS,
        cfg.num_experts,
    );
    let alt_idx = router.select(&alt_sfc);
    let ref_alt_idx = idx_u32(o, &key(l, "moe_router_ALT_topk_idx_bias_on_logits"));
    let (same_alt, n_alt) = set_equal_rows(&alt_idx, &ref_alt_idx, cfg.top_k);
    r.expect_true(
        &format!("L{l:02} CONTROL bias-on-logits reproduces the stored ALT idx"),
        same_alt == n_alt && n_alt == TOKENS,
        &format!("{same_alt}/{n_alt} tokens agree"),
    );
    let (same_true, _) = set_equal_rows(&alt_idx, &ref_idx, cfg.top_k);
    let pinned = f64s(o, &key(l, "moe_router_ALT_bias_on_logits_n_tokens_same_set"))[0] as usize;
    r.expect_true(
        &format!("L{l:02} CONTROL bias-on-logits reroutes; count matches the oracle's"),
        same_true == pinned && same_true < TOKENS,
        &format!("{same_true}/{TOKENS} tokens keep the same set (oracle counted {pinned})"),
    );

    // --- the required case: the bias CHANGES which experts are chosen ------
    let unbiased_sfc = ScoresForChoice::from_raw(
        ref_scores.iter().map(|&x| x as f32).collect(),
        TOKENS,
        cfg.num_experts,
    );
    let nobias_idx = router.select(&unbiased_sfc);
    let (same_nb, _) = set_equal_rows(&nobias_idx, &ref_idx, cfg.top_k);
    let swaps = swap_count(&nobias_idx, &ref_idx, cfg.top_k, TOKENS);
    r.expect_true(
        &format!("L{l:02} the correction bias CHANGES the chosen experts"),
        same_nb == 0 && swaps > 0,
        &format!(
            "dropping the bias reroutes {}/{TOKENS} tokens, {swaps} expert swaps",
            TOKENS - same_nb
        ),
    );
    // ...and dropping it does NOT change the weights of whatever is chosen:
    // proof that selection and weighting are separate concerns here.
    let nb_pre = router.combine_weights(
        &Scores::from_raw(
            ref_scores.iter().map(|&x| x as f32).collect(),
            TOKENS,
            cfg.num_experts,
        ),
        &ref_idx,
    );
    r.expect_match(
        &format!("L{l:02} the bias does NOT enter the weight (same gather, EXACT)"),
        &nb_pre,
        &f64s(o, &key(l, "moe_router_topk_weight_prerenorm")),
        TOL_EXACT,
    );
}

// ---------------------------------------------------------------------------
// 6. the unresolved convention: f32 bias on disk vs its bf16 rounding
// ---------------------------------------------------------------------------

/// The checkpoint stores `e_score_correction_bias` as **float32**; a
/// `dtype=bfloat16` model load rounds it, and the oracle was captured that way.
/// Returns the number of tokens whose selection changes if the on-disk f32 bias
/// is used instead. Measured, not assumed — and reported, because no available
/// measurement says which one production serves.
fn measure_f32_bias(r: &mut Report, o: &Npz, w: &Npz, router: &Router, l: usize) -> usize {
    let cfg = *router.config();
    let ref_scores = f64s(o, &key(l, "moe_router_scores"));
    let ref_idx = idx_u32(o, &key(l, "moe_gate_out_topk_idx"));

    let bias_f32 = w.get(&key(l, "gate_bias_f32"));
    assert!(!bias_f32.is_empty(), "gate_bias_f32 is EMPTY");
    let bias_f32 = bias_f32.to_f32();
    let bias_bf16 = router.bias().to_vec();
    // same non-sticky-maximum hazard as `compare`: a NaN must abort, not be
    // recorded and then overwritten by the next finite element
    let mut dmax = 0f64;
    for (&a, &b) in bias_f32.iter().zip(&bias_bf16) {
        let d = (a as f64 - b as f64).abs();
        if !d.is_finite() {
            dmax = f64::NAN;
            break;
        }
        if !(d <= dmax) {
            dmax = d;
        }
    }

    let alt = Router::new(cfg, router.weight().to_vec(), bias_f32).expect("f32-bias router");
    let scores = Scores::from_raw(
        ref_scores.iter().map(|&x| x as f32).collect(),
        TOKENS,
        cfg.num_experts,
    );
    let sfc = alt.scores_for_choice(&scores);
    let idx = alt.select(&sfc);
    let (same, _) = set_equal_rows(&idx, &ref_idx, cfg.top_k);
    let diff = TOKENS - same;
    r.expect_true(
        &format!("L{l:02} f32-vs-bf16 bias divergence measured (not asserted equal)"),
        dmax > 0.0,
        &format!("max|f32-bf16| {dmax:.3e}; {diff}/{TOKENS} token selections change"),
    );
    diff
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Rows compared as SETS: `torch.topk(sorted=False)` fixes no order.
fn set_equal_rows(a: &[u32], b: &[u32], k: usize) -> (usize, usize) {
    assert_eq!(a.len(), b.len(), "row length mismatch");
    assert!(!a.is_empty(), "comparing empty index arrays proves nothing");
    let n = a.len() / k;
    let mut same = 0;
    for t in 0..n {
        let mut x: Vec<u32> = a[t * k..(t + 1) * k].to_vec();
        let mut y: Vec<u32> = b[t * k..(t + 1) * k].to_vec();
        x.sort_unstable();
        y.sort_unstable();
        if x == y {
            same += 1;
        }
    }
    (same, n)
}

fn swap_count(a: &[u32], b: &[u32], k: usize, n: usize) -> usize {
    let mut c = 0;
    for t in 0..n {
        let y: std::collections::BTreeSet<u32> = b[t * k..(t + 1) * k].iter().copied().collect();
        c += a[t * k..(t + 1) * k].iter().filter(|e| !y.contains(e)).count();
    }
    c
}

fn idx_well_formed(idx: &[u32], k: usize, n_experts: usize, tokens: usize) -> bool {
    if idx.len() != tokens * k {
        return false;
    }
    (0..tokens).all(|t| {
        let row: std::collections::BTreeSet<u32> =
            idx[t * k..(t + 1) * k].iter().copied().collect();
        row.len() == k && row.iter().all(|&e| (e as usize) < n_experts)
    })
}

/// Align two (idx, weight) routings by expert id. `None` if the index sets
/// differ for any token — the weights are then not comparable and the caller
/// must fail rather than compare a shorter pair.
fn pair_by_index(
    a_idx: &[u32],
    a_w: &[f32],
    b_idx: &[u32],
    b_w: &[f64],
    k: usize,
    tokens: usize,
) -> Option<(Vec<f32>, Vec<f64>)> {
    let mut mine = Vec::with_capacity(tokens * k);
    let mut theirs = Vec::with_capacity(tokens * k);
    for t in 0..tokens {
        let mut m: BTreeMap<u32, f32> = BTreeMap::new();
        for j in 0..k {
            if m.insert(a_idx[t * k + j], a_w[t * k + j]).is_some() {
                return None;
            }
        }
        let mut o: BTreeMap<u32, f64> = BTreeMap::new();
        for j in 0..k {
            if o.insert(b_idx[t * k + j], b_w[t * k + j]).is_some() {
                return None;
            }
        }
        if m.keys().ne(o.keys()) {
            return None;
        }
        mine.extend(m.values().copied());
        theirs.extend(o.values().copied());
    }
    if mine.is_empty() {
        return None;
    }
    Some((mine, theirs))
}

/// Smallest gap between the k-th and (k+1)-th `scores_for_choice` over tokens —
/// how close the selection came to being a coin flip.
fn min_tie_margin(sfc: &[f64], tokens: usize, n: usize, k: usize) -> f64 {
    let mut min = f64::INFINITY;
    for t in 0..tokens {
        let mut row: Vec<f64> = sfc[t * n..(t + 1) * n].to_vec();
        row.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let m = row[k - 1] - row[k];
        // NaN must poison the answer outright — `!(m >= min)` would record it
        // and the next finite margin would overwrite it (see `compare`)
        if !m.is_finite() {
            return f64::NAN;
        }
        if m < min {
            min = m;
        }
    }
    min
}

fn weights_sane(w: &[f64], tokens: usize, k: usize, scale: f64) -> bool {
    if w.len() != tokens * k || w.is_empty() {
        return false;
    }
    (0..tokens).all(|t| {
        let row = &w[t * k..(t + 1) * k];
        let s: f64 = row.iter().sum();
        row.iter().all(|&x| x > 0.0 && x.is_finite()) && (s - scale).abs() < 1e-6
    })
}
