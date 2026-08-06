//! `k3_layer_gate` — a **whole Kimi-K3 decoder layer**, run in mary and held to
//! the shipped `modeling_kimi_linear.py` at every sub-block boundary.
//!
//! Two layers are gated, chosen because between them they contain every
//! structural case a K3 layer has: **layer 3** (MLA full attention + MoE) and
//! **layer 4** (KDA linear attention + MoE). Both run inside a resumed
//! [`DepthMixer`], so the AttnRes entry mixture, the accumulator and the MLP
//! mixture are exercised, not stubbed.
//!
//! # ⚠ The premise this job was given is wrong, and the gate says so first
//!
//! The task brief says *"24 MLA full-attention layers at the positions in
//! `linear_attn_config.full_attn_layers`"*. **The config's lists are
//! one-indexed and the layer indices are not.** `full_attn_layers` begins
//! `[4, 8, 12, …]`; the layers that actually carry `self_attn.kv_a_proj_with_mqa`
//! are `[3, 7, 11, …]`. Layer 4 is a **KDA** layer. A port that trusts the
//! brief puts full attention at 4, 8, 12 … and builds 93 layers with the wrong
//! 24 of them full-attention — and every weight it asks for is missing, so it
//! fails loudly at load; the dangerous variant is a loader that falls back.
//!
//! The `P` lane below establishes this **by measurement, from the checkpoint's
//! tensor names alone**, for all 93 layers, and only then compares the result
//! against what the config parser says. (The correction was first reported by
//! the oracle job; this gate re-derives it rather than citing it.)
//!
//! # What this gate is for
//!
//! The four operators this layer composes each have their own gate, and each
//! passed. Composition is where they can still all be right and the layer
//! wrong: a sublayer wired to the previous one's output instead of the depth
//! mixture, an accumulator folded in at the wrong point, a norm applied to the
//! mixture instead of to what fed it. None of those is visible to an operator
//! gate, and all of them produce a plausible number.
//!
//! So the gate is **two lanes over the same layer**:
//!
//! * the **cascade** lane — one call to `K3DecoderLayer::forward` from the
//!   layer's own input and a resumed depth bank, with every intermediate
//!   compared. This is the claim "a whole layer runs and matches".
//! * the **teacher-forced** lane — each sublayer driven from the *oracle's*
//!   captured input for that boundary. bfloat16 rounding compounds, so a
//!   cascade comparison cannot be tight enough to see a one-ulp mistake;
//!   driven from the reference's own inputs, each step is comparable to an ulp.
//!
//! Neither alone is enough. A cascade-only gate has slack everywhere; a
//! teacher-forced-only gate cannot see a wiring error at all, because
//! teacher-forcing *is* the wiring.
//!
//! # Where the MoE lane's teacher-forcing comes from
//!
//! The MoE block is the expensive part: 172–192 of the 896 routed experts are
//! decoded from packed MXFP4 nibbles per layer, and running it twice would
//! double the gate's cost for nothing. It is run **once**, from the cascade's
//! own `post_attention_layernorm_out` — and check `<L>/wire.moe_in` asserts
//! **bit-exactly** that this equals the oracle's `moe_in`. When that check
//! passes, the cascade's MoE lane *is* the teacher-forced one, as an identity
//! rather than as an assumption; when it fails, the gate fails. The identity is
//! reported, never inferred.
//!
//! # Reading the output
//!
//! Every check prints `PASS`/`FAIL`, a measured error, and the budget it was
//! held to. The per-sub-block table at the end is a re-print of the cascade
//! lane, ordered by the layer's control flow. The verdict line is the only
//! thing a script should parse.
//!
//! **`K3LAYER_SELFTEST=1`** runs only the `S` lane — the gate's checks on its
//! own comparators — and exits. It is what the mutation harness uses to confirm
//! the harness itself is live before it starts breaking things.
//!
//! # Running
//!
//! ```text
//! cargo run --release --features k3 --bin k3_layer_gate
//! ```
//! Oracle dir: argv[1] or `K3_ORACLE_DIR` (default `./k3-oracle`).
//! Model dir:  argv[2] or `K3_MODEL_DIR`  (default `./kimi-k3`).

use burn::backend::NdArray;
use burn::prelude::*;
use mary::models::k3::attn_res::{stack_candidates, AttnResParams, DepthMixer};
use mary::models::k3::ckpt::Ckpt;
use mary::models::k3::kda::{Kda, KdaParams, KdaScratch, KdaState, KdaToken};
use mary::models::k3::kda_attn::{KdaAttnConfig, KdaCache};
use mary::models::k3::layer::{K3Attn, K3DecoderLayer, K3Ffn, K3FfnTrace};
use mary::models::k3::mla::{MlaBlock, MlaConfig, Precision};
use mary::models::k3::moe::{BlockTrace, LatentMoe, MoeDims, Routing as MoeRouting};
use mary::models::k3::ops::{linear, rms_norm, ActRound};
use mary::models::k3::router::{Accum, Router, RouterConfig};
use mary::models::k3::{AttnKind, K3Config, KdaAttention};
use mary::nn::npz::{NpyData, Npz};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

type B = NdArray<f32>;
type Dev = Device<B>;

// ===========================================================================
// Budgets — derived, and written down before any number was read
// ===========================================================================

/// Spacing of bfloat16 relative to a value's magnitude. bf16 has **8 bits of
/// significand** (1 implicit + 7 stored), so consecutive values differ by
/// `2^-7` relative.
const BF16_ULP_REL: f64 = 1.0 / 128.0;

/// Relative error of one f32 reduction over a few thousand terms, generously:
/// `sqrt(K)·2^-24` is 5e-6 at K = 7168; this allows ~4x for cancellation.
const F32_STEP_REL: f64 = 2e-5;

/// Budget for a bf16-lane comparison whose reference passed through `n` bf16
/// roundings the port must reproduce.
///
/// Two things fit in `n`: the port computes in f32 and rounds where the
/// reference rounds, so it can differ by one ulp wherever f32 accumulation
/// noise pushes a value across a rounding boundary; and the shipped bf16 GEMM
/// is itself not fp32-exact, landing up to ~0.9 ulp of the tensor maximum from
/// the true product.
fn bf16_budget(n_roundings: usize, ref_absmax: f64) -> f64 {
    (n_roundings as f64) * BF16_ULP_REL * ref_absmax
}

/// Budget for an f32-lane comparison over `n` chained reductions.
fn f32_budget(n: usize, ref_absmax: f64) -> f64 {
    (n as f64) * F32_STEP_REL * ref_absmax
}

/// Minimum fraction of elements that must be **bit-identical** for an
/// elementwise or narrow operation driven from a captured input.
///
/// Derived: f32 accumulation noise (~`sqrt(K)·2^-24`, 5e-6 at K = 7168)
/// straddles a bf16 rounding boundary (spacing `2^-7`) with probability
/// ~6.5e-4, so 99.5% leaves a factor of ~8.
const EXACT_FRAC_ELEMENTWISE: f64 = 0.995;

/// Minimum fraction of elements that must be **bit-identical** for a wide
/// (32-row, K in the thousands) GEMM against the shipped bf16 tensor.
///
/// This one exists because a max-error budget alone cannot see a port that
/// stops rounding: dropping the bf16 cast on a projection output moves *almost
/// every* element by well under one ulp, so `max|d|` stays inside budget while
/// the bit-exact fraction collapses. Measured on this checkpoint a correct
/// port reproduces 30–56% of a wide projection's bits exactly (the rest is the
/// shipped GEMM's own fp32 accumulation order, which nothing can reproduce);
/// a port that drops the rounding reproduces **0.0009%**. 10% sits a factor of
/// 3 below the worst correct case and four orders of magnitude above the
/// broken one.
const EXACT_FRAC_WIDE_GEMM: f64 = 0.10;

/// Minimum fraction of elements that must be **bit-identical** for an operation
/// whose *reference* is not itself fp32-exact.
///
/// Two kinds land here. A reduction over K elements: this port cannot reproduce
/// torch's f32 summation order, and the resulting few-ulp difference lands on a
/// scale shared by the whole row, so one flipped bit in the scale re-rounds
/// every element of that row. And a transcendental: torch's bf16 `sigmoid` is
/// not correctly rounded. MEASURED here, after the `recip` fix: 99.9%+ for
/// both, against 0.0009% for a port that has dropped its bf16 rounding
/// entirely. 90% keeps the arm that catches a dropped rounding while not
/// failing on summation order.
const EXACT_FRAC_REDUCTION: f64 = 0.90;

/// A negative control must miss the truth by at least this multiple of the
/// budget it would have been allowed. Anything less is a control that passes
/// by luck.
const CONTROL_MARGIN: f64 = 4.0;

// ===========================================================================
// The oracle
// ===========================================================================

/// The oracle bundle, with a record of every key actually read.
///
/// The record is not bookkeeping: `Z2` asserts the gate touched the number of
/// distinct arrays its lane structure implies. An oracle key that is never read
/// is a check that was never written, and a gate cannot tell you about a check
/// it does not have — but it can tell you it read fewer arrays than it should
/// have.
struct Oracle {
    npz: Npz,
    used: RefCell<BTreeSet<String>>,
}

impl Oracle {
    fn open(path: &Path) -> Oracle {
        let npz = Npz::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        assert!(!npz.is_empty(), "oracle npz is empty: {}", path.display());
        Oracle { npz, used: RefCell::new(BTreeSet::new()) }
    }

    fn mark(&self, key: &str) {
        self.used.borrow_mut().insert(key.to_string());
    }

    /// A `uint16` array read as bfloat16 bit patterns, widened to f32.
    fn bf16(&self, key: &str) -> Vec<f32> {
        self.mark(key);
        let a = self.npz.get(key);
        assert!(!a.is_empty(), "oracle array {key} is EMPTY");
        assert!(
            matches!(a.data, NpyData::U16(_)),
            "{key} is not uint16; `_bf16bits` arrays hold raw bfloat16 patterns"
        );
        a.bf16_to_f64().into_iter().map(|x| x as f32).collect()
    }

    /// A float array, at whatever float width it is stored in.
    fn f32(&self, key: &str) -> Vec<f32> {
        self.mark(key);
        let a = self.npz.get(key);
        assert!(!a.is_empty(), "oracle array {key} is EMPTY");
        assert!(
            matches!(a.data, NpyData::F32(_) | NpyData::F64(_)),
            "{key} is not a float array"
        );
        a.to_f32()
    }

    fn i64(&self, key: &str) -> Vec<i64> {
        self.mark(key);
        let a = self.npz.get(key);
        assert!(!a.is_empty(), "oracle array {key} is EMPTY");
        match &a.data {
            NpyData::I64(v) => v.clone(),
            _ => panic!("{key} is not int64"),
        }
    }

    fn scalar(&self, key: &str) -> f64 {
        self.mark(key);
        self.npz.get(key).scalar()
    }

    fn shape(&self, key: &str) -> Vec<usize> {
        self.npz.get(key).shape.clone()
    }

    fn n_used(&self) -> usize {
        self.used.borrow().len()
    }

    fn n_arrays(&self) -> usize {
        self.npz.len()
    }
}

fn sha256_file(p: &Path) -> String {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let mut h = Sha256::new();
    h.update(&b);
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

// ===========================================================================
// Reporting
// ===========================================================================

struct Check {
    id: String,
    what: String,
    ok: bool,
    detail: String,
    /// Set for the checks that make up the per-sub-block error table.
    table: Option<(String, String, f64, f64)>,
}

struct Cmp {
    n: usize,
    max_abs: f64,
    ref_absmax: f64,
    exact_frac: f64,
    nonfinite: usize,
}

/// Compare two arrays, refusing every degenerate case that would make the
/// comparison vacuous.
///
/// Empty is a panic, not a pass: `iter().all()` over nothing is `true`, and a
/// zero-length oracle compared to a zero-length result is a green measurement
/// of nothing. Non-finite values are *counted*, never folded into `max` —
/// `f64::max` returns the other operand on NaN, so a fold would erase a NaN
/// with the next ordinary element.
fn compare(got: &[f32], want: &[f32]) -> Cmp {
    assert!(!got.is_empty(), "comparison against an EMPTY result");
    assert!(!want.is_empty(), "comparison against an EMPTY reference");
    assert_eq!(got.len(), want.len(), "length mismatch {} vs {}", got.len(), want.len());
    let mut max_abs = 0f64;
    let mut ref_absmax = 0f64;
    let mut exact = 0usize;
    let mut nonfinite = 0usize;
    for (&g, &w) in got.iter().zip(want.iter()) {
        if !g.is_finite() || !w.is_finite() {
            nonfinite += 1;
            continue;
        }
        let d = (g as f64 - w as f64).abs();
        if d > max_abs {
            max_abs = d;
        }
        let a = (w as f64).abs();
        if a > ref_absmax {
            ref_absmax = a;
        }
        if g.to_bits() == w.to_bits() {
            exact += 1;
        }
    }
    Cmp { n: got.len(), max_abs, ref_absmax, exact_frac: exact as f64 / got.len() as f64, nonfinite }
}

struct Report {
    checks: Vec<Check>,
    quiet: bool,
}

impl Report {
    fn new() -> Report {
        Report { checks: Vec::new(), quiet: false }
    }

    fn push(&mut self, id: &str, what: &str, ok: bool, detail: String) {
        if !self.quiet {
            println!("  [{}] {id}  {what}\n         {detail}", if ok { "PASS" } else { "FAIL" });
        }
        self.checks.push(Check {
            id: id.to_string(),
            what: what.to_string(),
            ok,
            detail,
            table: None,
        });
    }

    /// `got` must match `want` to `budget`, and at least `exact_min` of the
    /// elements must be bit-identical when one is given.
    ///
    /// The failure test is `!(d <= budget)`, never `d > budget`: the latter is
    /// false for NaN, which would score garbage as zero error. `nonfinite > 0`
    /// fails independently, so a NaN cannot pass by any route.
    fn close(
        &mut self,
        id: &str,
        what: &str,
        got: &[f32],
        want: &[f32],
        budget: f64,
        exact_min: Option<f64>,
    ) -> f64 {
        let c = compare(got, want);
        let within = !(c.max_abs > budget) && c.max_abs.is_finite();
        let exact_ok = match exact_min {
            Some(m) => !(c.exact_frac < m),
            None => true,
        };
        let ok = within && exact_ok && c.nonfinite == 0;
        let extra = match exact_min {
            Some(m) => {
                format!(", bit-exact {:.4}% (min {:.2}%)", c.exact_frac * 100.0, m * 100.0)
            }
            None => format!(", bit-exact {:.4}%", c.exact_frac * 100.0),
        };
        self.push(
            id,
            what,
            ok,
            format!(
                "n={} max|d|={:.5e} budget={:.5e} ref|max|={:.5e}{extra}{}",
                c.n,
                c.max_abs,
                budget,
                c.ref_absmax,
                if c.nonfinite > 0 { format!(", NON-FINITE {}", c.nonfinite) } else { String::new() }
            ),
        );
        c.max_abs
    }

    /// Like [`Self::close`], and additionally recorded in the per-sub-block
    /// error table under `(layer, stage)`.
    #[allow(clippy::too_many_arguments)]
    fn stage(
        &mut self,
        id: &str,
        layer: &str,
        stage: &str,
        what: &str,
        got: &[f32],
        want: &[f32],
        budget: f64,
        exact_min: Option<f64>,
    ) {
        let d = self.close(id, what, got, want, budget, exact_min);
        let last = self.checks.last_mut().expect("just pushed");
        last.table = Some((layer.to_string(), stage.to_string(), d, budget));
    }

    fn exact(&mut self, id: &str, what: &str, got: &[f32], want: &[f32]) {
        let c = compare(got, want);
        let ok = c.exact_frac == 1.0 && c.nonfinite == 0;
        self.push(
            id,
            what,
            ok,
            format!("n={} bit-exact {:.6}% max|d|={:.3e}", c.n, c.exact_frac * 100.0, c.max_abs),
        );
    }

    fn boolean(&mut self, id: &str, what: &str, ok: bool, detail: String) {
        self.push(id, what, ok, detail);
    }

    /// A negative control: `got` must be MEASURABLY different from `want`.
    fn must_differ(&mut self, id: &str, what: &str, got: &[f32], want: &[f32], floor: f64) {
        let c = compare(got, want);
        let ok = !(c.max_abs <= floor) && c.max_abs.is_finite();
        self.push(
            id,
            what,
            ok,
            format!(
                "n={} max|d|={:.5e} must exceed {:.5e} (bit-exact {:.3}%)",
                c.n,
                c.max_abs,
                floor,
                c.exact_frac * 100.0
            ),
        );
    }

    /// A negative control, stated as what it means: the positive check would
    /// REJECT this wrong variant, and the wrong variant is clearly separated
    /// from where the correct implementation actually lands.
    ///
    /// Prefer this to `must_differ` with a hand-scaled floor. A floor derived
    /// from a budget rides on the layer's activation scale, so the same
    /// control can pass on one layer and fail on another for reasons that have
    /// nothing to do with whether it discriminates -- which is precisely what
    /// happened to `neg.a_log_last96` on layer 12.
    fn must_reject(
        &mut self,
        id: &str,
        what: &str,
        got: &[f32],
        want: &[f32],
        budget: f64,
        correct_dev: f64,
    ) {
        let c = compare(got, want);
        let clears_budget = !(c.max_abs <= budget);
        let separated = !(c.max_abs <= CONTROL_MARGIN * correct_dev);
        self.push(
            id,
            what,
            clears_budget && separated && c.max_abs.is_finite(),
            format!(
                "n={} max|d|={:.5e}; positive check's budget {:.5e} ({}); correct \
                 implementation measures {:.5e}, so separation {:.1}x (need {:.0}x) ({}); \
                 bit-exact {:.3}%",
                c.n,
                c.max_abs,
                budget,
                if clears_budget { "exceeded" } else { "NOT EXCEEDED" },
                correct_dev,
                c.max_abs / correct_dev,
                CONTROL_MARGIN,
                if separated { "separated" } else { "NOT SEPARATED" },
                c.exact_frac * 100.0
            ),
        );
    }

    fn failures(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| !c.ok).collect()
    }
}

fn absmax(v: &[f32]) -> f64 {
    v.iter().fold(0f64, |m, &x| m.max((x as f64).abs()))
}

fn t2v(v: Vec<f32>, rows: usize, cols: usize, dev: &Dev) -> Tensor<B, 2> {
    assert_eq!(v.len(), rows * cols, "t2v: {} values into [{rows}, {cols}]", v.len());
    Tensor::from_data(TensorData::new(v, [rows, cols]), dev)
}

fn t1v(v: Vec<f32>, dev: &Dev) -> Tensor<B, 1> {
    let n = v.len();
    assert!(n > 0, "t1v: empty");
    Tensor::from_data(TensorData::new(v, [n]), dev)
}

fn vec_of<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec().expect("f32 tensor")
}

fn bf(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// `x[m, k] @ w[n, k]^T` in float64 — the dumbest possible loop, so it shares
/// no implementation with anything it checks.
fn host_f64_matmul(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f64> {
    assert_eq!(x.len(), m * k);
    assert!(w.len() >= n * k);
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for t in 0..k {
                acc += x[i * k + t] as f64 * w[j * k + t] as f64;
            }
            out[i * n + j] = acc;
        }
    }
    out
}

// ===========================================================================
// S — the gate's checks on its own comparators
// ===========================================================================

/// Every comparator, shown failing on the thing it exists to catch.
///
/// A gate never seen to fail is decoration. These run first, before anything
/// touches the checkpoint, so a comparator that has been quietly broken is
/// loud at second zero rather than never.
fn lane_selftest(r: &mut Report) {
    println!("\n== S: the gate's comparators, shown failing ==");
    let mut probe = Report::new();
    probe.quiet = true;

    let base: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();

    // S1 — a NaN must fail, and must not be swallowed by a max-fold.
    let mut nan = base.clone();
    nan[7] = f32::NAN;
    probe.close("_", "_", &nan, &base, 1e9, None);
    let s1 = !probe.checks.pop().unwrap().ok;
    r.boolean(
        "S1",
        "close() FAILS on a NaN even with an enormous budget — `!(d <= max)` plus a \
         non-finite count, so garbage cannot score as zero error",
        s1,
        "one NaN planted in an otherwise identical array, budget 1e9".into(),
    );

    // S2 — an out-of-budget difference must fail.
    let mut off = base.clone();
    off[13] += 0.5;
    probe.close("_", "_", &off, &base, 1e-3, None);
    let s2 = !probe.checks.pop().unwrap().ok;
    probe.close("_", "_", &off, &base, 1.0, None);
    let s2b = probe.checks.pop().unwrap().ok;
    r.boolean(
        "S2",
        "close() FAILS at budget 1e-3 on a 0.5 perturbation and PASSES at budget 1.0 — \
         the budget is consulted in both directions",
        s2 && s2b,
        "one element moved by 0.5".into(),
    );

    // S3 — a collapsed bit-exact fraction must fail even inside budget.
    let tiny: Vec<f32> = base.iter().map(|&x| x * (1.0 + 1e-7)).collect();
    probe.close("_", "_", &tiny, &base, 1.0, Some(0.5));
    let s3 = !probe.checks.pop().unwrap().ok;
    probe.close("_", "_", &tiny, &base, 1.0, None);
    let s3b = probe.checks.pop().unwrap().ok;
    r.boolean(
        "S3",
        "close() FAILS on an exact-fraction floor that a within-budget perturbation \
         breaks — this is the arm that sees a dropped rounding, which max|d| cannot",
        s3 && s3b,
        "every element scaled by 1+1e-7: inside any budget, 0% bit-exact".into(),
    );

    // S4 — an empty comparison must panic, not pass.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let empty: Vec<f32> = Vec::new();
    let s4 = std::panic::catch_unwind(|| compare(&empty, &empty)).is_err();
    let s4b = std::panic::catch_unwind(|| compare(&[1.0f32], &[])).is_err();
    let s4c = std::panic::catch_unwind(|| compare(&[1.0f32, 2.0], &[1.0])).is_err();
    std::panic::set_hook(hook);
    r.boolean(
        "S4",
        "compare() PANICS on an empty array or a length mismatch rather than reporting \
         zero error — a zero-length oracle against a zero-length result is a green \
         measurement of nothing",
        s4 && s4b && s4c,
        "empty vs empty, non-empty vs empty, and 2 vs 1".into(),
    );

    // S5 — a negative control that coincides with the truth must fail.
    probe.must_differ("_", "_", &base, &base, 1e-9);
    let s5 = !probe.checks.pop().unwrap().ok;
    probe.must_differ("_", "_", &off, &base, 1e-9);
    let s5b = probe.checks.pop().unwrap().ok;
    r.boolean(
        "S5",
        "must_differ() FAILS when the 'wrong' alternative equals the truth — a control \
         that cannot fail is not a control",
        s5 && s5b,
        "identical arrays, then one element moved".into(),
    );

    // S6 — exact() must see a single ulp.
    let mut ulp = base.clone();
    ulp[3] = f32::from_bits(ulp[3].to_bits() ^ 1);
    probe.exact("_", "_", &ulp, &base);
    let s6 = !probe.checks.pop().unwrap().ok;
    probe.exact("_", "_", &base, &base);
    let s6b = probe.checks.pop().unwrap().ok;
    {

        // S5b: must_reject's two arms, each shown failing ALONE. An arm that is

        // never exercised is not a control, and this comparator has two.

        let base: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

        let mut far = base.clone();

        far[7] += 1.0;

        let mut s = Report::new();

        // clears a tiny budget, but the correct implementation is just as far off

        s.must_reject("probe", "budget cleared, separation absent", &far, &base, 1e-6, 1.0);

        // stands clear of a tiny correct error, but never reaches the budget

        s.must_reject("probe", "separated, budget not reached", &far, &base, 1e9, 1e-9);

        // and both together pass

        s.must_reject("probe", "both", &far, &base, 1e-6, 1e-9);

        let v: Vec<bool> = s.checks.iter().map(|c| c.ok).collect();

        r.boolean(

            "S5b",

            "must_reject() FAILS when EITHER arm fails — a wrong variant that clears the \

             budget but sits where the correct implementation does is not a control, and \

             neither is one that is well separated but never reaches the rejection \

             threshold. Both arms shown failing alone, then passing together",

            v == vec![false, false, true],

            format!("[budget-only, separation-only, both] ok = {v:?}"),

        );

        }

        r.boolean(
        "S6",
        "exact() FAILS on a one-ulp difference in one element of 64 and PASSES on an \
         identical array",
        s6 && s6b,
        "one mantissa bit flipped".into(),
    );

    // S7 — the masked comparator must not let the mask sentinel set the budget.
    //
    // This one exists because the first version of this gate got it wrong: the
    // MLA score array carries the additive mask's -3.4e38 at disallowed
    // positions, and a budget scaled to `absmax` of that array is 1e37 — a
    // check that cannot fail. `absmax_where` takes the magnitude of the
    // ALLOWED positions only.
    let scores = vec![1.0f32, 2.0, -3.4e38, -3.4e38];
    let allowed = [true, true, false, false];
    let am = absmax_where(&scores, &allowed);
    r.boolean(
        "S7",
        "absmax_where ignores the mask sentinel — a budget scaled to the -3.4e38 an \
         additive causal mask writes is a budget nothing can exceed",
        (am - 2.0).abs() < 1e-9 && absmax(&scores) > 1e37,
        format!("absmax_where {am}, plain absmax {:.3e}", absmax(&scores)),
    );

    assert!(probe.checks.is_empty(), "self-test probe left checks behind");
}

// ===========================================================================
// P — the premises, measured
// ===========================================================================

/// Re-derive the layer-kind map from the checkpoint's tensor names, for all 93
/// layers, and only then compare it with the config parser.
///
/// This is the check that catches the wrong premise this job was given. The
/// evidence is `self_attn.kv_a_proj_with_mqa` (MLA-only) versus `self_attn.A_log`
/// (KDA-only): a layer must carry exactly one of them.
fn lane_premises(r: &mut Report, ck: &Ckpt, cfg: &K3Config, oracle_dir: &Path) {
    println!("\n== P: the premises, re-derived from the checkpoint ==");
    let t = &cfg.text_config;

    let mut measured: Vec<AttnKind> = Vec::with_capacity(t.num_hidden_layers);
    let mut both = 0usize;
    let mut neither = 0usize;
    for l in 0..t.num_hidden_layers {
        let p = format!("language_model.model.layers.{l}.self_attn");
        let mla = ck.has(&format!("{p}.kv_a_proj_with_mqa.weight"));
        let kda = ck.has(&format!("{p}.A_log"));
        match (mla, kda) {
            (true, false) => measured.push(AttnKind::Mla),
            (false, true) => measured.push(AttnKind::Kda),
            (true, true) => {
                both += 1;
                measured.push(AttnKind::Mla);
            }
            (false, false) => {
                neither += 1;
                measured.push(AttnKind::Kda);
            }
        }
    }
    r.boolean(
        "P1",
        "every one of the 93 layers carries EXACTLY ONE of kv_a_proj_with_mqa (MLA) and \
         A_log (KDA) — the two readings are disjoint and total, so the tensor names alone \
         determine the layer kind",
        both == 0 && neither == 0 && measured.len() == 93,
        format!("{} layers, {both} carry both, {neither} carry neither", measured.len()),
    );

    let mla_idx: Vec<usize> =
        (0..measured.len()).filter(|&l| measured[l] == AttnKind::Mla).collect();
    r.boolean(
        "P2",
        "the MEASURED MLA positions are `full_attn_layers - 1`, NOT `full_attn_layers` — \
         the config's lists are one-indexed and the layer indices are not. THE PREMISE \
         THIS JOB WAS GIVEN IS WRONG",
        mla_idx.len() == 24
            && mla_idx
                .iter()
                .zip(t.linear_attn_config.full_attn_layers.iter())
                .all(|(&m, &c)| m + 1 == c),
        format!(
            "measured MLA at {:?}..{:?}; full_attn_layers begins {:?}",
            &mla_idx[..4.min(mla_idx.len())],
            &mla_idx[mla_idx.len().saturating_sub(2)..],
            &t.linear_attn_config.full_attn_layers[..4]
        ),
    );

    let agree = (0..t.num_hidden_layers).all(|l| t.attn_kind(l) == measured[l]);
    r.boolean(
        "P3",
        "the config parser's attn_kind agrees with the tensor-name evidence on ALL 93 \
         layers — the one-indexing is handled in exactly one place and it is handled right",
        agree,
        format!(
            "layer 3 measured {:?} / parsed {:?}; layer 4 measured {:?} / parsed {:?}",
            measured[3],
            t.attn_kind(3),
            measured[4],
            t.attn_kind(4)
        ),
    );

    r.boolean(
        "P4",
        "the shape constants this gate depends on are the checkpoint's, read from \
         config.json rather than assumed",
        t.hidden_size == 7168
            && t.num_hidden_layers == 93
            && t.attn_res_block_size == Some(12)
            && (t.rms_norm_eps - 1e-5).abs() < 1e-12
            && t.first_k_dense_replace == 1
            && t.num_experts == 896
            && t.num_experts_per_token == 16
            && t.mla_use_nope,
        format!(
            "hidden {} layers {} block_size {:?} eps {:e} dense {} experts {}/{} nope {}",
            t.hidden_size,
            t.num_hidden_layers,
            t.attn_res_block_size,
            t.rms_norm_eps,
            t.first_k_dense_replace,
            t.num_experts_per_token,
            t.num_experts,
            t.mla_use_nope
        ),
    );

    let mut pad_ok = true;
    let mut live_nonzero = 0usize;
    let nh = t.linear_attn_config.num_heads;
    for l in [4usize, 5, 6] {
        let (_, a) = ck.f32(&format!("language_model.model.layers.{l}.self_attn.A_log"));
        pad_ok &= a.len() == 128 && a[nh..].iter().all(|&v| v == 0.0);
        live_nonzero += a[..nh].iter().filter(|&&v| v != 0.0).count();
    }
    r.boolean(
        "P5",
        "A_log is 96 live entries zero-padded to 128, and the padding is at the END — so \
         taking the first 96 is right and taking the last 96 is the bug (exp(0)=1 makes a \
         padded head's decay rate 1)",
        pad_ok && live_nonzero == 3 * nh,
        format!("3 layers checked, {live_nonzero} of {} live entries non-zero", 3 * nh),
    );

    const SHA_PREFIX: &str =
        "fdb3b897f0bb43e8506d27dd283defee87910006dd1038c131687a1b48e61d7c";
    const SHA_INV: &str = "af853091814e9627cd61c20f1c55a4acfab978c928b304715819a4a0f7d067eb";
    let hp = sha256_file(&oracle_dir.join("layer_oracle_prefix13_bf16.npz"));
    let hi = sha256_file(&oracle_dir.join("layer_oracle_prefix13_bf16_inventory.json"));
    r.boolean(
        "P6",
        "the oracle .npz and its inventory are the pinned artifacts, by sha256 of the \
         bytes on disk — not a round trip through this gate's own reader",
        hp == SHA_PREFIX && hi == SHA_INV,
        format!("npz {}.. inventory {}..", &hp[..16], &hi[..16]),
    );
}

// ===========================================================================
// One layer
// ===========================================================================

struct LayerResult {
    out_absmax: f64,
    /// Checks this layer contributed in total.
    checks: usize,
    /// Of those, how many came from the attention sublayer's own lane. Split
    /// out so `Z1` can assert both halves rather than only their sum — a lane
    /// that lost checks while another gained them would otherwise cancel.
    attn_checks: usize,
}

/// Build a [`MoeRouting`] from the oracle's captured `topk_idx`/`topk_weight`,
/// so the MoE can be run with the *shipped* selection instead of its own.
fn pinned_routing(p: &Oracle, tag: &str, tokens: usize, top_k: usize, dev: &Dev) -> MoeRouting<B> {
    let idx: Vec<usize> =
        p.i64(&format!("{tag}_moe_gate_out_topk_idx")).into_iter().map(|v| v as usize).collect();
    let w = p.f32(&format!("{tag}_moe_gate_out_topk_weight"));
    let pre = p.f32(&format!("{tag}_moe_router_topk_weight_prerenorm"));
    assert_eq!(idx.len(), tokens * top_k, "pinned routing idx length");
    assert_eq!(w.len(), tokens * top_k, "pinned routing weight length");
    let e = p.f32(&format!("{tag}_moe_router_scores")).len() / tokens;
    MoeRouting {
        logits: t2v(p.f32(&format!("{tag}_moe_router_logits")), tokens, e, dev),
        scores: t2v(p.f32(&format!("{tag}_moe_router_scores")), tokens, e, dev),
        scores_for_choice: t2v(p.f32(&format!("{tag}_moe_router_scores_for_choice")), tokens, e, dev),
        topk_idx: idx,
        topk_weight_prerenorm: pre,
        topk_weight: w,
        tokens,
        top_k,
        min_topk_margin: f32::NAN,
    }
}

/// Every stage of the MoE block, compared. Shared by the teacher-forced lane
/// (driven from the oracle's `moe_in`) and the cascade lane (driven from the
/// layer's own `post_attention_layernorm` output), so the two are held to the
/// same set of boundaries and differ only in their input and their budgets.
#[allow(clippy::too_many_arguments)]
fn check_moe_stages(
    r: &mut Report,
    p: &Oracle,
    tag: &str,
    lane: &str,
    bt: &BlockTrace<B>,
    dims: &MoeDims,
    n_extra: usize,
    routing_term: f64,
    table: bool,
) {
    let stages: [(&str, &str, Vec<f32>, String, usize); 5] = [
        (
            "latent_down",
            "7a MoE latent down",
            vec_of(bt.latent_down_out.clone()),
            format!("{tag}_moe_latent_down_out_bf16bits"),
            2,
        ),
        (
            "combine",
            "7b MoE top-16 combine",
            vec_of(bt.combined.clone()),
            format!("{tag}_moe_latent_norm_in_bf16bits"),
            6,
        ),
        (
            "latent_norm",
            "7c MoE latent norm",
            vec_of(bt.normed.clone()),
            format!("{tag}_moe_latent_norm_out_bf16bits"),
            8,
        ),
        (
            "latent_up",
            "7d MoE latent up",
            vec_of(bt.latent_up_out.clone()),
            format!("{tag}_moe_latent_up_out_bf16bits"),
            10,
        ),
        (
            "shared",
            "7e MoE shared experts",
            vec_of(bt.shared_out.clone().expect("K3 has shared experts")),
            format!("{tag}_moe_shared_out_bf16bits"),
            4,
        ),
    ];
    for (name, stage, got, key, n) in stages {
        let want = p.bf16(&key);
        let am = absmax(&want);
        // The shared experts never see the router, so no routing term for them.
        let rt = if name == "shared" || name == "latent_down" { 0.0 } else { routing_term };
        let budget = bf16_budget(n + n_extra, am) + rt * am;
        let id = format!("{tag}/{lane}.moe.{name}");
        let what = format!("{lane}: {stage}");
        if table {
            r.stage(&id, tag, stage, &what, &got, &want, budget, None);
        } else {
            r.close(&id, &what, &got, &want, budget, None);
        }
    }
    let out_ref = p.bf16(&format!("{tag}_moe_out_bf16bits"));
    let am = absmax(&out_ref);
    let budget = bf16_budget(12 + n_extra, am) + routing_term * am;
    let id = format!("{tag}/{lane}.moe.out");
    let what = format!("{lane}: the whole MoE block, moe_in -> moe_out");
    if table {
        r.stage(&id, tag, "7 MoE block", &what, &vec_of(bt.out.clone()), &out_ref, budget, None);
    } else {
        r.close(&id, &what, &vec_of(bt.out.clone()), &out_ref, budget, None);
    }
    let _ = dims;
}

/// How many tokens' selected expert SETS differ from the shipped run, and the
/// largest weight discrepancy over the experts both agree on.
fn routing_divergence(
    routing: &MoeRouting<B>,
    ref_idx: &[i64],
    ref_w: &[f32],
    tokens: usize,
    k: usize,
) -> (Vec<usize>, f64) {
    let mut flipped: Vec<usize> = Vec::new();
    let mut wmax = 0f64;
    for tk in 0..tokens {
        let mine: BTreeSet<usize> = routing.topk_idx[tk * k..(tk + 1) * k].iter().copied().collect();
        let theirs: BTreeSet<usize> =
            ref_idx[tk * k..(tk + 1) * k].iter().map(|&x| x as usize).collect();
        assert_eq!(mine.len(), k, "duplicate expert in this port's selection");
        assert_eq!(theirs.len(), k, "duplicate expert in the shipped selection");
        if mine != theirs {
            flipped.push(tk);
        }
        for j in 0..k {
            let id = routing.topk_idx[tk * k + j];
            if let Some(pos) =
                ref_idx[tk * k..(tk + 1) * k].iter().position(|&x| x as usize == id)
            {
                let d = (routing.topk_weight[tk * k + j] as f64 - ref_w[tk * k + pos] as f64).abs();
                if d > wmax {
                    wmax = d;
                }
            }
        }
    }
    (flipped, wmax)
}

#[allow(clippy::too_many_lines)]
fn run_layer(
    r: &mut Report,
    p: &Oracle,
    ck: &Ckpt,
    cfg: &K3Config,
    dims: &MoeDims,
    dev: &Dev,
    layer: usize,
) -> LayerResult {
    let start_checks = r.checks.len();
    let tag = format!("L{layer:02}");
    let t = &cfg.text_config;
    let kind = t.attn_kind(layer);
    println!("\n== {tag}: a whole decoder layer ({kind:?} + MoE) ==");

    let hidden = t.hidden_size;
    let shape = p.shape(&format!("{tag}_layer_in_bf16bits"));
    assert_eq!(shape.len(), 3, "{tag}_layer_in rank");
    let (batch, seq) = (shape[0], shape[1]);
    let tokens = batch * seq;
    assert_eq!(shape[2], hidden, "{tag}_layer_in width");

    // ---- the resumed depth bank ---------------------------------------
    let schedule: Vec<bool> =
        (0..t.num_hidden_layers).map(|l| t.is_attn_res_checkpoint(l)).collect();
    let bank_flat = p.bf16(&format!("{tag}_blockres_in_bf16bits"));
    let bshape = p.shape(&format!("{tag}_blockres_in_bf16bits"));
    let nblocks = bshape[1];
    assert_eq!(bshape, vec![tokens, nblocks, hidden], "{tag}_blockres_in shape");
    let bank: Vec<Tensor<B, 2>> = (0..nblocks)
        .map(|s| {
            let mut v = Vec::with_capacity(tokens * hidden);
            for tok in 0..tokens {
                let base = tok * nblocks * hidden + s * hidden;
                v.extend_from_slice(&bank_flat[base..base + hidden]);
            }
            t2v(v, tokens, hidden, dev)
        })
        .collect();
    let want_depth = schedule[..layer].iter().filter(|&&b| b).count();
    r.boolean(
        &format!("{tag}/bank.depth"),
        "the resumed depth bank has exactly as many snapshots as the schedule takes \
         before this layer — the bank is EVIDENCE FROM THE ORACLE, not something this \
         gate derived, and its depth is the one property a hand-built bank gets wrong",
        nblocks == want_depth && nblocks > 0,
        format!("{nblocks} snapshots; schedule takes {want_depth} before layer {layer}"),
    );
    let mut mixer = DepthMixer::<B>::resume(schedule, bank.clone(), layer);

    // ---- weights --------------------------------------------------------
    let (ln1_w, ln2_w) = ck.layer_norms::<B>(layer, dev);
    let lp = format!("language_model.model.layers.{layer}");
    let sa_res: AttnResParams<B> =
        ck.attn_res_site(&format!("{lp}.self_attention_res"), t.rms_norm_eps, dev);
    let mlp_res: AttnResParams<B> = ck.attn_res_site(&format!("{lp}.mlp_res"), t.rms_norm_eps, dev);
    let moe_w = ck.moe_block_weights::<B>(layer, true, dev);

    let attn = match kind {
        AttnKind::Mla => {
            let mc = MlaConfig::from_text_config(t).expect("MlaConfig");
            let w = ck.mla_weights::<B>(layer, &mc, dev);
            K3Attn::Mla(Box::new(MlaBlock::new(mc, w, Precision::Bf16)))
        }
        AttnKind::Kda => {
            let kc = KdaAttnConfig::from_text_config(t).expect("KdaAttnConfig");
            let w = ck.kda_weights::<B>(layer, &kc, dev);
            K3Attn::Kda(Box::new(KdaAttention::new(kc, w, ActRound::Bf16)))
        }
    };
    assert!(t.is_moe_layer(layer), "layer {layer} must be a MoE layer for this gate");
    let dec = K3DecoderLayer::new(
        layer,
        dims.clone(),
        ActRound::Bf16,
        ln1_w.clone(),
        ln2_w.clone(),
        sa_res.clone(),
        mlp_res.clone(),
        attn,
        K3Ffn::Moe(Box::new(moe_w.clone())),
    );

    r.exact(
        &format!("{tag}/w.sa_res_score_weight"),
        "the self-attention AttnRes score weight (norm.weight * proj.weight) built from \
         the checkpoint bytes equals the oracle's, bit for bit",
        &vec_of(sa_res.score_weight()),
        &p.f32(&format!("{tag}_attnres_sa_score_weight")),
    );
    r.exact(
        &format!("{tag}/w.mlp_res_score_weight"),
        "the MLP AttnRes score weight built from the checkpoint bytes equals the \
         oracle's, bit for bit",
        &vec_of(mlp_res.score_weight()),
        &p.f32(&format!("{tag}_attnres_mlp_score_weight")),
    );

    // =====================================================================
    // TEACHER-FORCED lane
    // =====================================================================
    println!("\n  -- {tag} teacher-forced (each sublayer from the ORACLE's own input) --");
    let layer_in = p.bf16(&format!("{tag}_layer_in_bf16bits"));
    let sa_out_ref = p.bf16(&format!("{tag}_attnres_sa_out_bf16bits"));
    let ln1_out_ref = p.bf16(&format!("{tag}_input_layernorm_out_bf16bits"));
    let mlp_out_ref = p.bf16(&format!("{tag}_attnres_mlp_out_bf16bits"));
    let ln2_out_ref = p.bf16(&format!("{tag}_post_attention_layernorm_out_bf16bits"));
    let moe_in_ref = p.bf16(&format!("{tag}_moe_in_bf16bits"));
    let moe_out_ref = p.bf16(&format!("{tag}_moe_out_bf16bits"));
    let layer_out_ref = p.bf16(&format!("{tag}_layer_out_bf16bits"));
    let mlp_prefix_ref = p.bf16(&format!("{tag}_attnres_mlp_prefix_sum_bf16bits"));

    let v_sa = stack_candidates(&bank, t2v(layer_in.clone(), tokens, hidden, dev));
    r.exact(
        &format!("{tag}/tf.attnres_sa.stack"),
        "the self-attention AttnRes candidate stack is cat(bank, prefix_sum) with the \
         accumulator LAST — bit-exact against the shipped `v`",
        &vec_of(v_sa.clone()),
        &p.bf16(&format!("{tag}_attnres_sa_v_bf16bits")),
    );
    let mix_sa = sa_res.mix(v_sa);
    let sa_scores_ref = p.f32(&format!("{tag}_attnres_sa_scores"));
    r.close(
        &format!("{tag}/tf.attnres_sa.scores"),
        "AttnRes depth scores (normalised candidate . query direction), f32 lane",
        &vec_of(mix_sa.scores.clone()),
        &sa_scores_ref,
        f32_budget(2, absmax(&sa_scores_ref)),
        None,
    );
    let sa_probs_ref = p.f32(&format!("{tag}_attnres_sa_probs"));
    r.close(
        &format!("{tag}/tf.attnres_sa.probs"),
        "AttnRes depth attention (softmax over slots), f32 lane",
        &vec_of(mix_sa.probs.clone()),
        &sa_probs_ref,
        f32_budget(3, absmax(&sa_probs_ref)),
        None,
    );
    r.close(
        &format!("{tag}/tf.attnres_sa.out"),
        "the self-attention AttnRes mixture: a convex combination of the RAW candidates",
        &vec_of(mix_sa.out),
        &sa_out_ref,
        bf16_budget(2, absmax(&sa_out_ref)),
        Some(EXACT_FRAC_ELEMENTWISE),
    );

    // AT A BOUNDARY THE MLP SITE MIXES OVER ONE MORE CANDIDATE THAN THE
    // SELF-ATTENTION SITE. The snapshot is pushed BETWEEN the two mixtures, so
    // the sa site sees `bank ++ [acc]` and the mlp site sees
    // `bank ++ [snapshot] ++ [acc]`. Gating only layers 3 and 4 — neither a
    // boundary — this distinction never arose, and the gate stacked the
    // pre-snapshot bank at both sites. Adding layer 12 surfaced it immediately
    // as a shape mismatch (2 slots offered where the oracle had 3).
    let mlp_bank: Vec<Tensor<B, 2>> = if t.is_attn_res_checkpoint(layer) {
        let mut b = bank.clone();
        b.push(t2v(layer_in.clone(), tokens, hidden, dev));
        b
    } else {
        bank.clone()
    };
    let v_mlp = stack_candidates(&mlp_bank, t2v(mlp_prefix_ref.clone(), tokens, hidden, dev));
    r.exact(
        &format!("{tag}/tf.attnres_mlp.stack"),
        "the MLP AttnRes candidate stack, accumulator LAST — bit-exact against the \
         shipped `v`",
        &vec_of(v_mlp.clone()),
        &p.bf16(&format!("{tag}_attnres_mlp_v_bf16bits")),
    );
    let mix_mlp = mlp_res.mix(v_mlp);
    let mlp_scores_ref = p.f32(&format!("{tag}_attnres_mlp_scores"));
    r.close(
        &format!("{tag}/tf.attnres_mlp.scores"),
        "MLP-side AttnRes depth scores, f32 lane",
        &vec_of(mix_mlp.scores.clone()),
        &mlp_scores_ref,
        f32_budget(2, absmax(&mlp_scores_ref)),
        None,
    );
    r.close(
        &format!("{tag}/tf.attnres_mlp.out"),
        "the MLP-side AttnRes mixture",
        &vec_of(mix_mlp.out),
        &mlp_out_ref,
        bf16_budget(2, absmax(&mlp_out_ref)),
        Some(EXACT_FRAC_ELEMENTWISE),
    );

    let ln1_tf = rms_norm(
        t2v(sa_out_ref.clone(), tokens, hidden, dev),
        &ln1_w,
        t.rms_norm_eps,
        ActRound::Bf16,
    );
    r.close(
        &format!("{tag}/tf.input_layernorm"),
        "input_layernorm from the oracle's own input",
        &vec_of(ln1_tf),
        &ln1_out_ref,
        bf16_budget(2, absmax(&ln1_out_ref)),
        Some(EXACT_FRAC_ELEMENTWISE),
    );
    let ln2_tf = rms_norm(
        t2v(mlp_out_ref.clone(), tokens, hidden, dev),
        &ln2_w,
        t.rms_norm_eps,
        ActRound::Bf16,
    );
    r.close(
        &format!("{tag}/tf.post_attention_layernorm"),
        "post_attention_layernorm from the oracle's own input",
        &vec_of(ln2_tf),
        &ln2_out_ref,
        bf16_budget(2, absmax(&ln2_out_ref)),
        Some(EXACT_FRAC_ELEMENTWISE),
    );
    let last: Vec<f32> =
        mlp_prefix_ref.iter().zip(moe_out_ref.iter()).map(|(&a, &b)| bf(a + b)).collect();
    r.exact(
        &format!("{tag}/tf.layer_out_add"),
        "the layer's final residual add, driven from two captured tensors, is bit-exactly \
         the shipped layer_out — i.e. the add rounds to bf16",
        &last,
        &layer_out_ref,
    );

    let ln1_ref_t = t2v(ln1_out_ref.clone(), tokens, hidden, dev);
    let attn_checks = match &dec.attn {
        K3Attn::Mla(m) => run_mla_tf(r, p, m, &tag, ln1_ref_t, batch, seq, hidden, dev),
        K3Attn::Kda(k) => run_kda_tf(r, p, ck, k, &tag, layer, ln1_ref_t, batch, dev),
    };

    // ---- the MoE block, teacher-forced from the oracle's own moe_in -------
    println!("\n  -- {tag} MoE, teacher-forced from moe_in --");
    let moe = LatentMoe::new(dims.clone());
    let t_moe = Instant::now();
    let tf_bt = moe.forward_traced(
        t2v(moe_in_ref.clone(), tokens, hidden, dev),
        &moe_w,
        |id| ck.expert::<B>(layer, id, dev),
    );
    let ref_idx = p.i64(&format!("{tag}_moe_gate_out_topk_idx"));
    let ref_w = p.f32(&format!("{tag}_moe_gate_out_topk_weight"));
    let (flipped, wmax) = routing_divergence(&tf_bt.routing, &ref_idx, &ref_w, tokens, dims.top_k);
    r.boolean(
        &format!("{tag}/tf.moe.router_set"),
        "driven from the oracle's own moe_in, the routed-expert SET is the shipped one \
         for EVERY token and the combining weights agree through the index pairing",
        flipped.is_empty() && !(wmax > 1e-6),
        format!("{tokens} tokens x top-{}, 0 set flips required, {} seen, max |dw| = {wmax:.3e}", dims.top_k, flipped.len()),
    );
    r.close(
        &format!("{tag}/tf.moe.scores"),
        "the router's sigmoid scores, f32 lane",
        &vec_of(tf_bt.routing.scores.clone()),
        &p.f32(&format!("{tag}_moe_router_scores")),
        f32_budget(2, 1.0),
        None,
    );
    check_moe_stages(r, p, &tag, "tf", &tf_bt, dims, 0, 0.0, false);
    println!("         (teacher-forced MoE: {:.1} s)", t_moe.elapsed().as_secs_f64());

    // =====================================================================
    // CASCADE lane
    // =====================================================================
    println!("\n  -- {tag} cascade (one forward, every boundary compared) --");
    let mut cache = dec.new_cache(batch);
    let mut fetched: Vec<usize> = Vec::new();
    let t0 = Instant::now();
    let tr = dec.forward(
        &mut mixer,
        t2v(layer_in.clone(), tokens, hidden, dev),
        batch,
        &mut cache,
        |id| {
            fetched.push(id);
            ck.expert::<B>(layer, id, dev)
        },
    );
    let elapsed = t0.elapsed();

    let entry = tr.entry_mix.as_ref().expect("this layer's bank is non-empty");
    r.stage(
        &format!("{tag}/casc.attnres_sa"),
        &tag,
        "1 AttnRes(self-attn)",
        "cascade: the entry depth mixture",
        &vec_of(entry.out.clone()),
        &sa_out_ref,
        bf16_budget(2, absmax(&sa_out_ref)),
        Some(EXACT_FRAC_ELEMENTWISE),
    );
    r.stage(
        &format!("{tag}/casc.input_layernorm"),
        &tag,
        "2 input_layernorm",
        "cascade: input_layernorm",
        &vec_of(tr.input_layernorm_out.clone()),
        &ln1_out_ref,
        bf16_budget(3, absmax(&ln1_out_ref)),
        Some(EXACT_FRAC_ELEMENTWISE),
    );
    let attn_out_ref = p.bf16(&format!("{tag}_attn_o_proj_out_bf16bits"));
    let attn_n = if kind == AttnKind::Kda { 12 } else { 12 };
    r.stage(
        &format!("{tag}/casc.attn"),
        &tag,
        &format!("3 self_attn ({kind:?})"),
        "cascade: the whole attention sublayer",
        &vec_of(tr.attn.out(batch)),
        &attn_out_ref,
        bf16_budget(attn_n, absmax(&attn_out_ref)),
        None,
    );
    r.stage(
        &format!("{tag}/casc.prefix_sum"),
        &tag,
        "4 prefix_sum += attn",
        "cascade: the accumulator after the attention output is folded in",
        &vec_of(tr.prefix_sum_after_attn.clone()),
        &mlp_prefix_ref,
        bf16_budget(attn_n + 2, absmax(&mlp_prefix_ref)),
        None,
    );
    r.stage(
        &format!("{tag}/casc.attnres_mlp"),
        &tag,
        "5 AttnRes(mlp)",
        "cascade: the MLP-side depth mixture",
        &vec_of(tr.mlp_mix.out.clone()),
        &mlp_out_ref,
        bf16_budget(attn_n + 4, absmax(&mlp_out_ref)),
        None,
    );
    r.stage(
        &format!("{tag}/casc.post_attention_layernorm"),
        &tag,
        "6 post_attn_layernorm",
        "cascade: post_attention_layernorm",
        &vec_of(tr.post_attention_layernorm_out.clone()),
        &ln2_out_ref,
        bf16_budget(attn_n + 6, absmax(&ln2_out_ref)),
        None,
    );

    let bt = match &tr.ffn {
        K3FfnTrace::Moe(b) => b,
        K3FfnTrace::Dense(_) => panic!("layer {layer} is a MoE layer"),
    };

    // ---- the routing instability, measured ------------------------------
    //
    // This is the finding that decides how tight a whole-layer cascade can be.
    let drift = compare(&vec_of(tr.post_attention_layernorm_out.clone()), &moe_in_ref);
    let sref = p.f32(&format!("{tag}_moe_router_scores"));
    let sc = compare(&vec_of(bt.routing.scores.clone()), &sref);
    let sfc = p.f32(&format!("{tag}_moe_router_scores_for_choice"));
    let n_exp = sfc.len() / tokens;
    let mut margins: Vec<f64> = Vec::with_capacity(tokens);
    for tk in 0..tokens {
        let mut row: Vec<f32> = sfc[tk * n_exp..(tk + 1) * n_exp].to_vec();
        row.sort_by(|a, b| b.partial_cmp(a).expect("finite router scores"));
        margins.push((row[dims.top_k - 1] - row[dims.top_k]) as f64);
    }
    let min_margin = margins.iter().copied().fold(f64::INFINITY, f64::min);
    let (c_flipped, c_wmax) = routing_divergence(&bt.routing, &ref_idx, &ref_w, tokens, dims.top_k);
    let c_flips = c_flipped.len();
    r.boolean(
        &format!("{tag}/R1"),
        "MEASURED: K3's top-16 expert selection is NOT stable under bfloat16-scale input \
         drift. The cascade's router input differs from the shipped one by rounding \
         alone, and that already moves the scores by MORE than the gap between the 16th \
         and 17th expert — so some tokens must select a different set. This is a property \
         of the model, not of the port, and it is the reason the cascade MoE lane below \
         carries a routing term the teacher-forced lane does not",
        sc.max_abs > min_margin,
        format!(
            "router-input drift {:.3e} (|ref|max {:.3e}); score drift {:.3e}; min top-16/17 \
             margin {min_margin:.3e}; {c_flips} of {tokens} tokens flip their set; max |dw| {c_wmax:.3e}",
            drift.max_abs, drift.ref_absmax, sc.max_abs
        ),
    );
    // ...and every one of those flips stays at the BOUNDARY.
    //
    // Two earlier versions of this check were unsound in opposite ways. The
    // first bounded the flip COUNT at a quarter of the tokens, which passed
    // layers 3 and 4 and failed layer 12 -- not because layer 12 is ported
    // wrong (its teacher-forced router is bit-exact for every token) but
    // because its 16/17 margin is 25x tighter, so rounding moves far more
    // tokens across it. The second required each flipped token's margin to be
    // under twice the drift, which layer 12 satisfies for EVERY token by a
    // factor of five -- a check that could not fail on the layer it was
    // written for.
    //
    // What rounding can actually do is bounded in the score axis, not the
    // token axis: a perturbation of `drift` moves an expert across the
    // threshold only if it was already within `drift` of it. So every expert
    // that enters or leaves a token's set must lie in that window. An expert
    // ranked far above the cut cannot be dropped by re-rounding, and a port
    // that drops one is wrong no matter how few tokens it touches.
    let window = 2.0 * sc.max_abs;
    let mut outside: Vec<(usize, usize, f64)> = Vec::new();
    let mut excluded = 0usize;
    for tk in 0..tokens {
        let row = &sfc[tk * n_exp..(tk + 1) * n_exp];
        let mut srt: Vec<f32> = row.to_vec();
        srt.sort_by(|a, b| b.partial_cmp(a).expect("finite router scores"));
        let thr = srt[dims.top_k - 1] as f64;
        // the witness: experts this token's window puts out of reach
        excluded += row.iter().filter(|&&s| ((s as f64) - thr).abs() > window).count();
        if !c_flipped.contains(&tk) {
            continue;
        }
        let mine_t: BTreeSet<usize> =
            bt.routing.topk_idx[tk * dims.top_k..(tk + 1) * dims.top_k].iter().copied().collect();
        let theirs_t: BTreeSet<usize> = ref_idx[tk * dims.top_k..(tk + 1) * dims.top_k]
            .iter()
            .map(|&x| x as usize)
            .collect();
        for &e in mine_t.symmetric_difference(&theirs_t) {
            let d = ((row[e] as f64) - thr).abs();
            if !(d <= window) {
                outside.push((tk, e, d));
            }
        }
    }
    r.boolean(
        &format!("{tag}/R2"),
        "...and every flip stays at the BOUNDARY: a score moves by at most the drift, so \
         an expert can only cross the top-16 threshold if it was already within that of \
         it. An expert entering or leaving a set from FURTHER away did not get there by \
         rounding, and no flip count — high or low — would notice",
        outside.is_empty() && excluded > 0,
        format!(
            "{c_flips} of {tokens} tokens flip; every changed expert within {window:.3e} of \
             its token's threshold ({} outside{}); witness: the window puts {excluded} of \
             {} token-expert pairs out of reach, so this check has something to catch",
            outside.len(),
            outside
                .first()
                .map(|&(tk, e, d)| format!(", worst: token {tk} expert {e} at {d:.3e}"))
                .unwrap_or_default(),
            tokens * n_exp
        ),
    );

    // The routing term: when a token swaps one of its k experts, the block's
    // output changes by the swapped pair's combining weights times the expert
    // outputs' scale. The k renormalised weights sum to 1, so the smallest —
    // and a swap can only ever involve the smallest — is at most 1/k of the
    // total; allowing for both members of the pair and for the expert output
    // exceeding the block's own scale, 4/k relative is the bound.
    let routing_term = if c_flips > 0 { 4.0 / dims.top_k as f64 } else { 0.0 };
    check_moe_stages(r, p, &tag, "casc", bt, dims, 2, routing_term, true);

    let meta: BTreeSet<usize> =
        p.i64(&format!("meta_expert_ids_{tag}")).into_iter().map(|v| v as usize).collect();
    let mine: BTreeSet<usize> = fetched.iter().copied().collect();
    // What is left here is only what a broken port could actually violate.
    // The previous version asserted every "extra" expert traces to a flipped
    // token; that is a theorem (a non-flipped token selects exactly the
    // shipped set, so every extra necessarily comes from a flipped one) and
    // theorems do not test anything. R2 above now carries the real
    // constraint on WHICH experts may differ.
    //
    // Sparsity does have teeth: a port that quietly ran the MoE densely, or
    // fetched an expert twice, produces a fetch list this rejects. So does
    // the count bound -- the port cannot need more experts than the shipped
    // run plus what the flipped tokens' set differences can introduce.
    let mut symdiff_total = 0usize;
    for &tk in &c_flipped {
        let mine_t: BTreeSet<usize> =
            bt.routing.topk_idx[tk * dims.top_k..(tk + 1) * dims.top_k].iter().copied().collect();
        let theirs_t: BTreeSet<usize> = ref_idx[tk * dims.top_k..(tk + 1) * dims.top_k]
            .iter()
            .map(|&x| x as usize)
            .collect();
        symdiff_total += mine_t.difference(&theirs_t).count();
    }
    let extra = mine.difference(&meta).count();
    r.boolean(
        &format!("{tag}/casc.moe.expert_set"),
        "the layer fetched each expert exactly once and touched far fewer than all 896 — a \
         port that silently went dense, or double-fetched, fails here — and needed no more \
         beyond the shipped run's set than the flipped tokens' differences can introduce",
        fetched.len() == mine.len()
            && !mine.is_empty()
            && mine.len() < dims.num_experts
            && extra <= symdiff_total,
        format!(
            "{} distinct in {} calls, of {}; shipped run materialised {}; {extra} beyond it, \
             flipped tokens introduce at most {symdiff_total}",
            mine.len(),
            fetched.len(),
            dims.num_experts,
            meta.len()
        ),
    );
    r.stage(
        &format!("{tag}/casc.layer_out"),
        &tag,
        "8 LAYER OUTPUT",
        "cascade: THE WHOLE LAYER, layer_in -> layer_out",
        &vec_of(tr.out.clone()),
        &layer_out_ref,
        bf16_budget(attn_n + 8, absmax(&layer_out_ref))
            + routing_term * absmax(&moe_out_ref),
        None,
    );
    // A boundary layer PUSHES a snapshot; a non-boundary one must not. Checking
    // only the length would still pass mutant M05, which replaces the snapshot's
    // CONTENT with the mixture output — so the pushed slot is compared against
    // the oracle bit-for-bit. The snapshot is the raw layer input, appended
    // last, and nothing downstream in this layer reads it, which is exactly why
    // a wrong one is invisible until the next boundary twelve layers later.
    let is_boundary = t.is_attn_res_checkpoint(layer);
    let want_len = nblocks + usize::from(is_boundary);
    r.boolean(
        &format!("{tag}/casc.bank"),
        "the depth bank grew by exactly one snapshot at a boundary layer and not at all \
         otherwise, and the mixer advanced exactly one layer",
        tr.bank_len == want_len && mixer.layer() == layer + 1,
        format!(
            "bank {nblocks} -> {} (want {want_len}, boundary {is_boundary}), mixer at layer {}",
            tr.bank_len,
            mixer.layer()
        ),
    );
    if is_boundary {
        // `blockres_out` holds the bank AFTER this layer: oldest first, the new
        // snapshot last.
        let out_flat = p.bf16(&format!("{tag}_blockres_out_bf16bits"));
        let oshape = p.shape(&format!("{tag}_blockres_out_bf16bits"));
        assert_eq!(oshape, vec![tokens, want_len, hidden], "{tag}_blockres_out shape");
        let last = want_len - 1;
        let mut want_snap = Vec::with_capacity(tokens * hidden);
        for tok in 0..tokens {
            let base = tok * want_len * hidden + last * hidden;
            want_snap.extend_from_slice(&out_flat[base..base + hidden]);
        }
        let got_snap = vec_of(mixer.bank()[last].clone());
        let c = compare(&got_snap, &want_snap);
        r.boolean(
            &format!("{tag}/casc.snapshot"),
            "the snapshot this boundary pushed is BIT-EXACT against the oracle's bank — it \
             is the raw layer input and nothing in this layer reads it back, so a wrong \
             snapshot stays invisible until the next boundary twelve layers on",
            c.max_abs == 0.0,
            format!("max|d| {:.4e}, {:.4}% bit-exact over {} elems", c.max_abs, c.exact_frac * 100.0, c.n),
        );
    }
    println!("         (cascade forward: {:.1} s, {} experts)", elapsed.as_secs_f64(), fetched.len());

    // ---- the same cascade with the SHIPPED routing pinned ----------------
    //
    // The decomposition that makes the cascade's residual attributable: run the
    // cascade's own latent through the MoE with the shipped selection instead
    // of the one it computed. What is left is arithmetic, and it is tight.
    println!("\n  -- {tag} cascade with the SHIPPED routing pinned --");
    let pin = pinned_routing(p, &tag, tokens, dims.top_k, dev);
    let t_pin = Instant::now();
    let combined = moe.moe_infer(bt.latent_down_out.clone(), &pin, |id| {
        ck.expert::<B>(layer, id, dev)
    });
    let normed = moe.rms_norm(combined.clone(), moe_w.norm.clone().expect("latent norm"));
    let up = moe.linear(normed, moe_w.up_proj.clone());
    let out = moe.combine_with_shared(up, bt.shared_out.clone().expect("shared"));
    let comb_ref = p.bf16(&format!("{tag}_moe_latent_norm_in_bf16bits"));
    r.stage(
        &format!("{tag}/pin.moe.combine"),
        &tag,
        "7b* MoE combine, routing pinned",
        "the cascade's own latent through the SHIPPED expert selection: with the \
         discontinuity removed, the top-16 combination is back at rounding scale",
        &vec_of(combined),
        &comb_ref,
        bf16_budget(8, absmax(&comb_ref)),
        None,
    );
    r.stage(
        &format!("{tag}/pin.layer_out"),
        &tag,
        "8* LAYER OUTPUT, routing pinned",
        "THE WHOLE LAYER with the shipped expert selection pinned — the composition's \
         own arithmetic, with the model's routing discontinuity taken out",
        &vec_of(t2v(
            vec_of(out)
                .iter()
                .zip(vec_of(tr.prefix_sum_after_attn.clone()).iter())
                .map(|(&m, &ps)| bf(ps + m))
                .collect(),
            tokens,
            hidden,
            dev,
        )),
        &layer_out_ref,
        bf16_budget(attn_n + 10, absmax(&layer_out_ref)),
        None,
    );
    println!("         (pinned-routing MoE: {:.1} s)", t_pin.elapsed().as_secs_f64());

    // =====================================================================
    // NEGATIVE CONTROLS and INVARIANCES
    // =====================================================================
    println!("\n  -- {tag} negative controls --");
    r.must_differ(
        &format!("{tag}/neg.attnres_sa_normalised"),
        "NEGATIVE: combining the NORMALISED candidates instead of the raw ones must not \
         reproduce the mixture",
        &p.bf16(&format!("{tag}_attnres_sa_ALT_out_combine_normalized_bf16bits")),
        &sa_out_ref,
        CONTROL_MARGIN * bf16_budget(2, absmax(&sa_out_ref)),
    );
    r.must_differ(
        &format!("{tag}/neg.attnres_mlp_normalised"),
        "NEGATIVE: same for the MLP-side mixture",
        &p.bf16(&format!("{tag}_attnres_mlp_ALT_out_combine_normalized_bf16bits")),
        &mlp_out_ref,
        CONTROL_MARGIN * bf16_budget(2, absmax(&mlp_out_ref)),
    );
    let alt_first = p.bf16(&format!("{tag}_attnres_sa_ALT_out_prefix_first_bf16bits"));
    r.close(
        &format!("{tag}/inv.attnres_slot_order"),
        "INVARIANCE (not a control): softmax over slots is permutation-equivariant and \
         `probs @ v` sums over them, so putting the accumulator FIRST gives the same \
         vector to float-summation noise. What the order decides is the BANK, and that is \
         pinned bit-exactly by tf.attnres_sa.stack",
        &alt_first,
        &sa_out_ref,
        bf16_budget(2, absmax(&sa_out_ref)),
        None,
    );
    let v_first = stack_candidates(&[t2v(layer_in.clone(), tokens, hidden, dev)], bank[0].clone());
    r.must_differ(
        &format!("{tag}/neg.stack_order"),
        "NEGATIVE: the candidate STACK with the accumulator first is measurably not the \
         shipped `v` — the invariance above is about the mixture, not about the stack",
        &vec_of(v_first),
        &p.bf16(&format!("{tag}_attnres_sa_v_bf16bits")),
        1e-6,
    );
    r.must_differ(
        &format!("{tag}/neg.router_weight_source"),
        "NEGATIVE: taking the combining weight from the BIASED score must not reproduce \
         the shipped topk_weight",
        &p.f32(&format!("{tag}_moe_router_ALT_topk_weight_from_scores_for_choice")),
        &ref_w,
        1e-4,
    );
    let alt_idx = p.i64(&format!("{tag}_moe_router_ALT_topk_idx_bias_on_logits"));
    let k = dims.top_k;
    let differing = (0..tokens)
        .filter(|&tk| {
            let a: BTreeSet<i64> = alt_idx[tk * k..(tk + 1) * k].iter().copied().collect();
            let b: BTreeSet<i64> = ref_idx[tk * k..(tk + 1) * k].iter().copied().collect();
            a != b
        })
        .count();
    r.boolean(
        &format!("{tag}/neg.router_bias_placement"),
        "NEGATIVE: adding e_score_correction_bias to the LOGIT instead of to the sigmoid \
         SCORE selects a different expert set — measurably, on real tokens",
        differing > 0,
        format!("{differing} of {tokens} tokens change their selected set"),
    );
    let f32_norm = rms_norm(
        t2v(sa_out_ref.clone(), tokens, hidden, dev),
        &ln1_w,
        t.rms_norm_eps,
        ActRound::None,
    );
    r.must_differ(
        &format!("{tag}/neg.round_lane_is_real"),
        "NEGATIVE: ActRound::None really is a different lane — the same norm without the \
         bf16 rounding does NOT reproduce the shipped bf16 output. A rounding lane that \
         secretly did nothing would pass every budget in this gate",
        &vec_of(f32_norm),
        &ln1_out_ref,
        1e-9,
    );
    let other = if layer == 3 { 4 } else { 3 };
    let (other_ln1, _) = ck.layer_norms::<B>(other, dev);
    let wrong = rms_norm(
        t2v(sa_out_ref.clone(), tokens, hidden, dev),
        &other_ln1,
        t.rms_norm_eps,
        ActRound::Bf16,
    );
    r.must_differ(
        &format!("{tag}/neg.wrong_layer_gain"),
        "NEGATIVE: input_layernorm with the OTHER gated layer's gain must not reproduce \
         this layer's output — the per-layer weight load is checked, not just the shape",
        &vec_of(wrong),
        &ln1_out_ref,
        CONTROL_MARGIN * bf16_budget(2, absmax(&ln1_out_ref)),
    );
    let wrong_res: AttnResParams<B> = ck.attn_res_site(
        &format!("language_model.model.layers.{other}.self_attention_res"),
        t.rms_norm_eps,
        dev,
    );
    r.must_differ(
        &format!("{tag}/neg.wrong_layer_attnres"),
        "NEGATIVE: the AttnRes site of the OTHER gated layer must not reproduce this \
         layer's mixture — the score weight is a per-layer weight and it is checked as one",
        &vec_of(wrong_res.mix(stack_candidates(&bank, t2v(layer_in, tokens, hidden, dev))).out),
        &sa_out_ref,
        CONTROL_MARGIN * bf16_budget(2, absmax(&sa_out_ref)),
    );

    LayerResult {
        out_absmax: absmax(&layer_out_ref),
        checks: r.checks.len() - start_checks,
        attn_checks,
    }
}

/// `absmax` over the positions a boolean predicate allows.
///
/// Needed because the MLA score array carries the additive causal mask's
/// -3.4e38 at every disallowed position, and a budget scaled to the `absmax` of
/// *that* is 1e37 — a check nothing can fail. This gate's first version made
/// exactly that mistake; `S7` now shows the fix failing on the bug.
fn absmax_where(v: &[f32], allowed: &[bool]) -> f64 {
    assert_eq!(v.len(), allowed.len(), "absmax_where: length mismatch");
    assert!(allowed.iter().any(|&a| a), "absmax_where: nothing is allowed");
    v.iter()
        .zip(allowed)
        .filter(|(_, &a)| a)
        .fold(0f64, |m, (&x, _)| m.max((x as f64).abs()))
}

/// Compare against a bfloat16 oracle array at an `n`-rounding budget.
///
/// A free function and not a closure so it does not hold a borrow of the report
/// across the rest of a lane — which matters here because the negative controls
/// at the end of each lane need the report too, and a lane whose controls had to
/// be moved elsewhere to satisfy the borrow checker is a lane whose controls get
/// forgotten.
fn one(
    r: &mut Report,
    p: &Oracle,
    id: &str,
    what: &str,
    got: Vec<f32>,
    key: &str,
    n: usize,
    ex: Option<f64>,
) {
    let want = p.bf16(key);
    r.close(id, what, &got, &want, bf16_budget(n, absmax(&want)), ex);
}

/// The MLA sublayer.
///
/// Two passes over the same block. First **per operation**, each driven from the
/// oracle's own captured input for that boundary — the tight lane, where a
/// one-ulp mistake is visible. Then one `forward`, which is the composition of
/// exactly those methods and nothing else — the lane that sees a wiring error.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_mla_tf(
    r: &mut Report,
    p: &Oracle,
    m: &MlaBlock<B>,
    tag: &str,
    hidden_in: Tensor<B, 2>,
    batch: usize,
    seq: usize,
    hidden: usize,
    dev: &Dev,
) -> usize {
    let start = r.checks.len();
    let c = &m.cfg;
    let (h, qh, dv) = (c.num_heads, c.q_head_dim(), c.v_head_dim);
    let x3 = hidden_in.reshape([batch, seq, hidden]);
    let t3 = |v: Vec<f32>, w: usize| -> Tensor<B, 3> {
        let n = v.len() / (batch * w);
        assert_eq!(n, seq, "t3 width {w}");
        Tensor::from_data(TensorData::new(v, [batch, seq, w]), dev)
    };
    let t4 = |v: Vec<f32>, d1: usize, d2: usize, d3: usize| -> Tensor<B, 4> {
        Tensor::from_data(TensorData::new(v, [batch, d1, d2, d3]), dev)
    };

    // ---- the mask, and the positions it allows ---------------------------
    let mask = MlaBlock::<B>::causal_mask(batch, seq, seq, 0, dev);
    let mask_ref = p.bf16(&format!("{tag}_mla_attn_mask_bf16bits"));
    r.exact(
        &format!("{tag}/tf.mla.mask"),
        "the additive causal mask this block builds is bit-exactly the shipped one, \
         sentinel included",
        &vec_of(mask.clone()),
        &mask_ref,
    );
    // [B,1,Tq,Tkv] -> [B,H,Tq,Tkv]
    let allowed: Vec<bool> = (0..batch * h * seq * seq)
        .map(|i| {
            let b = i / (h * seq * seq);
            let rest = i % (seq * seq);
            mask_ref[b * seq * seq + rest] == 0.0
        })
        .collect();

    // ---- per operation, from the oracle's own inputs ----------------------
    one(
        r,
        p,
        &format!("{tag}/tf.mla.q_a_proj"),
        "q_a_proj, from input_layernorm's captured output",
        vec_of(m.q_a_proj(x3.clone())),
        &format!("{tag}_attn_q_a_proj_out_bf16bits"),
        2,
        Some(EXACT_FRAC_WIDE_GEMM),
    );
    let q_a_ln_in = t3(p.bf16(&format!("{tag}_attn_q_a_layernorm_in_bf16bits")), c.q_lora_rank.unwrap());
    one(
        r,
        p,
        &format!("{tag}/tf.mla.q_a_layernorm"),
        "q_a_layernorm at MLA's own 1e-6 epsilon (not the model's 1e-5), from its own \
         captured input",
        vec_of(m.q_a_norm(q_a_ln_in)),
        &format!("{tag}_attn_q_a_layernorm_out_bf16bits"),
        2,
        Some(EXACT_FRAC_REDUCTION),
    );
    let q_b_in = t3(p.bf16(&format!("{tag}_attn_q_b_proj_in_bf16bits")), c.q_lora_rank.unwrap());
    let q_b_out_ref = p.bf16(&format!("{tag}_attn_q_b_proj_out_bf16bits"));
    one(
        r,
        p,
        &format!("{tag}/tf.mla.q_b_proj"),
        "q_b_proj, from its own captured input",
        vec_of(m.q_b_proj(q_b_in)),
        &format!("{tag}_attn_q_b_proj_out_bf16bits"),
        2,
        Some(EXACT_FRAC_WIDE_GEMM),
    );
    one(
        r,
        p,
        &format!("{tag}/tf.mla.kv_a_proj"),
        "kv_a_proj_with_mqa, from input_layernorm's captured output",
        vec_of(m.kv_a_proj(x3.clone())),
        &format!("{tag}_attn_kv_a_proj_with_mqa_out_bf16bits"),
        2,
        Some(EXACT_FRAC_WIDE_GEMM),
    );
    let kv_a_out_ref = p.bf16(&format!("{tag}_attn_kv_a_proj_with_mqa_out_bf16bits"));
    let kv_a_out_t = t3(kv_a_out_ref.clone(), c.kv_lora_rank + c.qk_carried_head_dim);
    r.exact(
        &format!("{tag}/tf.mla.kv_latent_slice"),
        "the latent that feeds kv_a_layernorm is the FIRST 512 of the 576-wide projection \
         — bit-exact, because which half goes where is not something a tolerance can see",
        &vec_of(m.kv_latent(kv_a_out_t.clone())),
        &p.bf16(&format!("{tag}_attn_kv_a_layernorm_in_bf16bits")),
    );
    let kv_a_ln_in = t3(p.bf16(&format!("{tag}_attn_kv_a_layernorm_in_bf16bits")), c.kv_lora_rank);
    one(
        r,
        p,
        &format!("{tag}/tf.mla.kv_a_layernorm"),
        "kv_a_layernorm, from its own captured input",
        vec_of(m.kv_a_norm(kv_a_ln_in)),
        &format!("{tag}_attn_kv_a_layernorm_out_bf16bits"),
        2,
        Some(EXACT_FRAC_REDUCTION),
    );
    let kv_b_in = t3(p.bf16(&format!("{tag}_attn_kv_b_proj_in_bf16bits")), c.kv_lora_rank);
    let kv_b_out_ref = p.bf16(&format!("{tag}_attn_kv_b_proj_out_bf16bits"));
    one(
        r,
        p,
        &format!("{tag}/tf.mla.kv_b_proj"),
        "kv_b_proj, from its own captured input",
        vec_of(m.kv_b_proj(kv_b_in)),
        &format!("{tag}_attn_kv_b_proj_out_bf16bits"),
        2,
        Some(EXACT_FRAC_WIDE_GEMM),
    );

    // Assembly, from the captured projections: BIT-EXACT, because assembly is
    // slicing and concatenation and nothing else.
    let (q_states, _) = m.assemble_query(t3(q_b_out_ref.clone(), h * qh));
    r.exact(
        &format!("{tag}/tf.mla.query_states"),
        "the assembled query, from the captured q_b_proj output — bit-exact: assembly is \
         a slice and a concatenation, nothing arithmetic happens to the carried lane",
        &vec_of(q_states),
        &p.bf16(&format!("{tag}_mla_query_states_bf16bits")),
    );
    let (k_states, v_states, _) = m.assemble_kv(
        t3(kv_b_out_ref.clone(), h * c.kv_b_head_dim()),
        m.kv_carried(kv_a_out_t),
    );
    r.exact(
        &format!("{tag}/tf.mla.key_states"),
        "the assembled key, from the captured projections — kv_b_proj's FIRST 128 per \
         head, then kv_a_proj's [512:576] broadcast over all 96 heads. Bit-exact",
        &vec_of(k_states),
        &p.bf16(&format!("{tag}_mla_key_states_bf16bits")),
    );
    r.exact(
        &format!("{tag}/tf.mla.value_states"),
        "the value: kv_b_proj's SECOND 128 per head. Bit-exact",
        &vec_of(v_states),
        &p.bf16(&format!("{tag}_mla_value_states_bf16bits")),
    );

    // Scores, from the captured query/key states. Compared only where the mask
    // allows; the masked positions must be exactly the sentinel.
    let q_ref = t4(p.bf16(&format!("{tag}_mla_query_states_bf16bits")), h, seq, qh);
    let k_ref = t4(p.bf16(&format!("{tag}_mla_key_states_bf16bits")), h, seq, qh);
    let scores = vec_of(m.attn_scores(q_ref, k_ref, Some(mask.clone())));
    let scores_ref = p.bf16(&format!("{tag}_mla_attn_scores_precast_bf16bits"));
    let score_scale = absmax_where(&scores_ref, &allowed);
    let s_allowed: Vec<f32> =
        scores.iter().zip(&allowed).filter(|(_, &a)| a).map(|(&x, _)| x).collect();
    let s_ref_allowed: Vec<f32> =
        scores_ref.iter().zip(&allowed).filter(|(_, &a)| a).map(|(&x, _)| x).collect();
    let score_budget = bf16_budget(3, score_scale);
    r.close(
        &format!("{tag}/tf.mla.scores"),
        "q.k * 192**-0.5 + mask, at the ALLOWED positions — the budget is scaled to the \
         allowed scores' magnitude, never to the mask sentinel",
        &s_allowed,
        &s_ref_allowed,
        score_budget,
        None,
    );
    let s_masked: Vec<f32> =
        scores.iter().zip(&allowed).filter(|(_, &a)| !a).map(|(&x, _)| x).collect();
    let s_ref_masked: Vec<f32> =
        scores_ref.iter().zip(&allowed).filter(|(_, &a)| !a).map(|(&x, _)| x).collect();
    r.exact(
        &format!("{tag}/tf.mla.scores_masked"),
        "at the DISALLOWED positions the score is bit-exactly the shipped sentinel — \
         checked separately so it cannot dominate the budget of the check above",
        &s_masked,
        &s_ref_masked,
    );

    // The softmax, from the captured pre-softmax scores. This is the fp32
    // island, so it is comparable at f32 precision.
    let sc_ref = t4(scores_ref.clone(), h, seq, seq);
    let (pre, cast) = m.attn_probs(sc_ref);
    let pre_ref = p.f32(&format!("{tag}_mla_attn_probs_precast"));
    r.close(
        &format!("{tag}/tf.mla.probs_precast"),
        "the fp32 softmax BEFORE the cast back, from the captured scores — the fp32 \
         island, comparable at f32 precision because nothing bf16 is between",
        &vec_of(pre),
        &pre_ref,
        f32_budget(2, absmax(&pre_ref)),
        None,
    );
    one(
        r,
        p,
        &format!("{tag}/tf.mla.probs_cast"),
        "the softmax CAST BACK to bf16, from the captured scores — the rows sum to one \
         only to bf16 precision after this, which is what the shipped eager path returns",
        vec_of(cast),
        &format!("{tag}_mla_attn_probs_bf16bits"),
        1,
        Some(EXACT_FRAC_ELEMENTWISE),
    );
    let probs_ref = t4(p.bf16(&format!("{tag}_mla_attn_probs_bf16bits")), h, seq, seq);
    let v_ref = t4(p.bf16(&format!("{tag}_mla_value_states_bf16bits")), h, seq, dv);
    one(
        r,
        p,
        &format!("{tag}/tf.mla.attn_apply"),
        "probs . v from the captured probs and values",
        vec_of(m.attn_apply(probs_ref, v_ref)),
        &format!("{tag}_mla_attn_out_heads_bf16bits"),
        2,
        None,
    );
    one(
        r,
        p,
        &format!("{tag}/tf.mla.g_proj"),
        "g_proj, from input_layernorm's captured output",
        vec_of(m.g_proj(x3).expect("K3 MLA has an output gate")),
        &format!("{tag}_attn_g_proj_out_bf16bits"),
        2,
        Some(EXACT_FRAC_WIDE_GEMM),
    );
    let heads_ref = t4(p.bf16(&format!("{tag}_mla_attn_out_heads_bf16bits")), seq, h, dv);
    let g_ref = t3(p.bf16(&format!("{tag}_attn_g_proj_out_bf16bits")), h * dv);
    one(
        r,
        p,
        &format!("{tag}/tf.mla.output_gate"),
        "the sigmoid gate applied in the 96*128 space, BEFORE o_proj, from captured \
         tensors on both sides",
        vec_of(m.apply_output_gate(heads_ref, Some(&g_ref))),
        &format!("{tag}_attn_o_proj_in_bf16bits"),
        2,
        Some(EXACT_FRAC_REDUCTION),
    );
    let o_in = t3(p.bf16(&format!("{tag}_attn_o_proj_in_bf16bits")), h * dv);
    one(
        r,
        p,
        &format!("{tag}/tf.mla.o_proj"),
        "o_proj, from its own captured input",
        vec_of(m.o_proj(o_in)),
        &format!("{tag}_attn_o_proj_out_bf16bits"),
        2,
        Some(EXACT_FRAC_WIDE_GEMM),
    );

    // ---- and the whole block, composed -----------------------------------
    let x3b = t3(p.bf16(&format!("{tag}_input_layernorm_out_bf16bits")), hidden);
    let tr = m.forward(x3b, Some(mask), None);
    one(
        r,
        p,
        &format!("{tag}/tf.mla.block"),
        "the whole MLA sublayer as ONE forward — the composition of exactly the methods \
         above, so this sees a wiring error that none of them can",
        vec_of(tr.out.clone()),
        &format!("{tag}_attn_o_proj_out_bf16bits"),
        7,
        None,
    );
    let n = tr.assert_carried_verbatim();
    r.boolean(
        &format!("{tag}/tf.mla.carried_verbatim"),
        "the carried (never-rotated) 64-wide lanes of q and k are BIT-IDENTICAL to the \
         projection outputs they were sliced from — NoPE stated as an executable claim",
        n > 0,
        format!("{n} elements compared and identical"),
    );

    let alt = p.bf16(&format!("{tag}_mla_ALT_attn_out_swapped_halves_bf16bits"));
    let want = p.bf16(&format!("{tag}_mla_attn_out_heads_bf16bits"));
    r.close(
        &format!("{tag}/inv.mla_split_point"),
        "INVARIANCE (not a control): swapping the nope/carried halves of q AND k leaves \
         the attention output alone, because nothing is rotated. What matters is where \
         each dimension COMES FROM, and that is pinned bit-exactly by key_states and \
         carried_verbatim",
        &alt,
        &want,
        bf16_budget(4, absmax(&want)),
        None,
    );
    r.checks.len() - start
}

/// The KDA sublayer: per operation from the oracle's captured inputs, then the
/// whole block as one forward.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_kda_tf(
    r: &mut Report,
    p: &Oracle,
    ck: &Ckpt,
    k: &KdaAttention<B>,
    tag: &str,
    layer: usize,
    hidden_in: Tensor<B, 2>,
    batch: usize,
    dev: &Dev,
) -> usize {
    let start = r.checks.len();
    let tokens = hidden_in.dims()[0];
    let proj = k.cfg.proj_size();
    let cfg = k.cfg.kda;

    // ---- the projections, from input_layernorm's captured output ---------
    for (name, w, key, ex) in [
        ("q_proj", &k.w.q_proj, "attn_q_proj_out_bf16bits", EXACT_FRAC_WIDE_GEMM),
        ("k_proj", &k.w.k_proj, "attn_k_proj_out_bf16bits", EXACT_FRAC_WIDE_GEMM),
        ("v_proj", &k.w.v_proj, "attn_v_proj_out_bf16bits", EXACT_FRAC_WIDE_GEMM),
        ("f_a_proj", &k.w.f_a_proj, "attn_f_a_proj_out_bf16bits", EXACT_FRAC_WIDE_GEMM),
        ("b_proj", &k.w.b_proj, "attn_b_proj_out_bf16bits", EXACT_FRAC_WIDE_GEMM),
        ("g_proj", &k.w.g_proj, "attn_g_proj_out_bf16bits", EXACT_FRAC_WIDE_GEMM),
    ] {
        one(
            r,
            p,
            &format!("{tag}/tf.kda.{name}"),
            name,
            vec_of(linear(hidden_in.clone(), w, ActRound::Bf16)),
            &format!("{tag}_{key}"),
            2,
            Some(ex),
        );
    }
    let f_a_out = p.bf16(&format!("{tag}_attn_f_a_proj_out_bf16bits"));
    one(
        r,
        p,
        &format!("{tag}/tf.kda.f_b_proj"),
        "f_b_proj — the decay gate, pre-dt_bias, from its own captured input",
        vec_of(linear(
            t2v(f_a_out.clone(), tokens, k.cfg.gate_rank, dev),
            &k.w.f_b_proj,
            ActRound::Bf16,
        )),
        &format!("{tag}_attn_f_b_proj_out_bf16bits"),
        2,
        Some(EXACT_FRAC_ELEMENTWISE),
    );

    // ---- the weights the recurrence needs, from the checkpoint bytes -----
    let (_, a_full) = ck.f32(&format!("language_model.model.layers.{layer}.self_attn.A_log"));
    let a_live = a_full[..cfg.num_heads].to_vec();
    r.exact(
        &format!("{tag}/w.a_log"),
        "A_log[:96] read from the checkpoint bytes equals the A_log the shipped kernel \
         was handed, bit for bit",
        &a_live,
        &p.f32(&format!("{tag}_kda_in_A_log")),
    );
    let (_, dt_bias) = ck.f32(&format!("language_model.model.layers.{layer}.self_attn.dt_bias"));
    r.exact(
        &format!("{tag}/w.dt_bias"),
        "dt_bias read from the checkpoint bytes equals the shipped kernel's, bit for bit",
        &dt_bias,
        &p.f32(&format!("{tag}_kda_in_dt_bias")),
    );

    // ---- the recurrence ALONE, from the oracle's own q/k/v/g/beta --------
    let core = Kda::new(cfg, KdaParams::new(&cfg, &a_live, &dt_bias));
    let (qo, ko, vo, go) = (
        p.bf16(&format!("{tag}_kda_in_q_bf16bits")),
        p.bf16(&format!("{tag}_kda_in_k_bf16bits")),
        p.bf16(&format!("{tag}_kda_in_v_bf16bits")),
        p.bf16(&format!("{tag}_kda_in_g_bf16bits")),
    );
    let beta = p.f32(&format!("{tag}_kda_in_beta"));
    let seq = tokens / batch;
    let mut out = vec![0f32; tokens * proj];
    let mut scratch = KdaScratch::new(&cfg);
    let mut states: Vec<KdaState<f32>> = Vec::new();
    for b in 0..batch {
        let mut st = KdaState::zeros(&cfg);
        for tt in 0..seq {
            let i = b * seq + tt;
            core.step(
                &mut st,
                &mut scratch,
                KdaToken {
                    q_raw: &qo[i * proj..(i + 1) * proj],
                    k_raw: &ko[i * proj..(i + 1) * proj],
                    v: &vo[i * proj..(i + 1) * proj],
                    g_raw: &go[i * proj..(i + 1) * proj],
                    beta_raw: &beta[i * cfg.num_heads..(i + 1) * cfg.num_heads],
                },
                &mut out[i * proj..(i + 1) * proj],
            );
        }
        states.push(st);
    }
    let out_bf: Vec<f32> = out.iter().map(|&x| bf(x)).collect();
    let core_ref = p.bf16(&format!("{tag}_kda_out_o_bf16bits"));
    r.close(
        &format!("{tag}/tf.kda.core_isolated"),
        "the recurrence driven from the ORACLE's own q/k/v/g/beta — this port steps token \
         by token, the reference is fla's chunked Triton kernel, so the residual here is \
         summation order and nothing else",
        &out_bf,
        &core_ref,
        bf16_budget(4, absmax(&core_ref)),
        None,
    );
    let vk: Vec<f32> = states.iter().flat_map(|s| s.to_vk(&cfg)).collect();
    let st_ref = p.f32(&format!("{tag}_kda_out_final_state"));
    r.close(
        &format!("{tag}/tf.kda.final_state"),
        "the final recurrent state in the [HV, V, K] layout `transpose_state_layout=True` \
         returns. The budget is one bfloat16 ulp of the state's own scale, not an f32 \
         one: the state is a sum of rank-1 outer products of bf16 inputs, and a chunked \
         and a sequential accumulation of those differ at that scale",
        &vk,
        &st_ref,
        bf16_budget(1, absmax(&st_ref)),
        None,
    );
    let kv: Vec<f32> = states.iter().flat_map(|s| s.as_kv().to_vec()).collect();
    r.must_differ(
        &format!("{tag}/neg.state_layout"),
        "NEGATIVE: the UNtransposed [HV, K, V] state must not reproduce the shipped one — \
         the transpose is load-bearing, not cosmetic, and feeding a [K,V] state to a \
         v-first consumer changes the output by tens of percent, silently",
        &kv,
        &st_ref,
        CONTROL_MARGIN * bf16_budget(1, absmax(&st_ref)),
    );
    let flags = p.i64(&format!("{tag}_kda_flags"));
    let lb = p.scalar(&format!("{tag}_kda_lower_bound"));
    r.boolean(
        &format!("{tag}/tf.kda.flags"),
        "the shipped chunk_kda call fuses the q/k L2-norm, the gate and the beta sigmoid, \
         with a safe gate at lower_bound -5.0 — which is why this port performs all three \
         inside its own step and why KdaConfig carries gate_lower_bound",
        flags.iter().all(|&f| f == 1) && flags.len() == 6
            && (lb - cfg.gate_lower_bound.expect("K3 bounds its gate")).abs() < 1e-12,
        format!("flags {flags:?}, lower_bound {lb}"),
    );

    // ---- and the whole block, composed -----------------------------------
    let mut cache = KdaCache::zeros(k, batch);
    let tr = k.forward(hidden_in, &mut cache);
    for (name, got, key) in [
        ("q_conv1d", tr.q_conv_out.clone(), "attn_q_conv1d_out0_bf16bits"),
        ("k_conv1d", tr.k_conv_out.clone(), "attn_k_conv1d_out0_bf16bits"),
        ("v_conv1d", tr.v_conv_out.clone(), "attn_v_conv1d_out0_bf16bits"),
    ] {
        one(
            r,
            p,
            &format!("{tag}/tf.kda.{name}"),
            "silu(depthwise causal conv, width 4) with the checkpoint's F32 weight",
            vec_of(got),
            &format!("{tag}_{key}"),
            3,
            Some(EXACT_FRAC_ELEMENTWISE),
        );
    }
    one(
        r,
        p,
        &format!("{tag}/tf.kda.core"),
        "the recurrence, cascaded from this port's own projections and convolutions",
        vec_of(tr.core_out.clone()),
        &format!("{tag}_kda_out_o_bf16bits"),
        8,
        None,
    );
    // Measured against what `tf.kda.core` above would actually do to this
    // variant, and against where that check's own port output lands -- not
    // against a budget, which on layer 12 sits an order of magnitude above the
    // correct implementation and made a 21x-separated control look like a
    // failure.
    let core_dev = compare(&vec_of(tr.core_out.clone()), &core_ref).max_abs;
    r.must_reject(
        &format!("{tag}/neg.a_log_last96"),
        "NEGATIVE: taking A_log as the LAST 96 of the padded 128 must not reproduce the \
         recurrence output — the 32 padding zeros would give those heads a decay rate of \
         exp(0) = 1, i.e. no decay at all",
        &p.bf16(&format!("{tag}_kda_ALT_o_A_log_last96_of_padded_bf16bits")),
        &core_ref,
        bf16_budget(8, absmax(&core_ref)),
        core_dev,
    );
    one(
        r,
        p,
        &format!("{tag}/tf.kda.o_norm"),
        "FusedRMSNormGated: rmsnorm(x)*w*sigmoid(g) — candidate A, normalising the \
         UNGATED x",
        vec_of(tr.o_norm_out.clone()),
        &format!("{tag}_attn_o_norm_out_bf16bits"),
        10,
        None,
    );
    one(
        r,
        p,
        &format!("{tag}/tf.kda.block"),
        "the whole KDA sublayer as ONE forward",
        vec_of(tr.out.clone()),
        &format!("{tag}_attn_o_proj_out_bf16bits"),
        12,
        None,
    );
    r.boolean(
        &format!("{tag}/tf.kda.state_is_o1"),
        "the KDA cache is fixed-size in sequence length — the whole reason 69 of 93 \
         layers are linear-attention ones",
        cache.byte_len() == batch * (cfg.state_elems() + 3 * proj * cfg.conv_kernel) * 4,
        format!(
            "{} bytes for {batch} sequences x {seq} tokens; {} would be the same for 128k tokens",
            cache.byte_len(),
            cache.byte_len()
        ),
    );
    r.checks.len() - start
}

// ===========================================================================
// X — cross-implementation, no oracle involved
// ===========================================================================

/// `k3::router::Router` (slices) and `k3::moe::LatentMoe::route` (burn) are two
/// independently written transcriptions of `KimiMoEGate.forward`, landed by two
/// different jobs against two different oracles. That duplication is a defect —
/// it is reported as one — but while it exists it is also evidence, and the
/// cheapest way to use it is to make the two check each other on real weights
/// and real activations.
fn lane_cross(r: &mut Report, p: &Oracle, ck: &Ckpt, dims: &MoeDims, dev: &Dev, layer: usize) {
    let tag = format!("L{layer:02}");
    let w = ck.moe_block_weights::<B>(layer, true, dev);
    let moe = LatentMoe::new(dims.clone());
    let hin = p.bf16(&format!("{tag}_moe_gate_in_bf16bits"));
    let tokens = hin.len() / dims.hidden_size;
    let burn_routing = moe.route(t2v(hin.clone(), tokens, dims.hidden_size, dev), &w.router);

    let mut rc = RouterConfig::k3();
    rc.hidden_size = dims.hidden_size;
    rc.num_experts = dims.num_experts;
    rc.top_k = dims.top_k;
    rc.renormalize = dims.moe_renormalize;
    rc.routed_scaling_factor = dims.routed_scaling_factor as f32;
    let slice_router =
        Router::new(rc, vec_of(w.router.weight.clone()), vec_of(w.router.bias.clone()))
            .expect("Router::new");
    let slice_routing = slice_router.route(&hin, tokens, Accum::F32);

    let k = dims.top_k;
    let mut same = true;
    let mut wmax = 0f64;
    for tk in 0..tokens {
        let a: BTreeSet<usize> =
            burn_routing.topk_idx[tk * k..(tk + 1) * k].iter().copied().collect();
        let b: BTreeSet<usize> = slice_routing.idx_row(tk).iter().map(|&x| x as usize).collect();
        same &= a == b && a.len() == k;
        for j in 0..k {
            let id = burn_routing.topk_idx[tk * k + j] as u32;
            if let Some(pos) = slice_routing.idx_row(tk).iter().position(|&x| x == id) {
                let d = (burn_routing.topk_weight[tk * k + j] as f64
                    - slice_routing.weight_row(tk)[pos] as f64)
                    .abs();
                if d > wmax {
                    wmax = d;
                }
            } else {
                same = false;
            }
        }
    }
    r.boolean(
        "X1",
        "the two independent router transcriptions in this crate — burn tensors and raw \
         slices, written by different jobs against different oracles — select the SAME \
         expert set for every token and agree on the combining weights",
        same && !(wmax > 1e-6),
        format!("{tokens} tokens x top-{k}, max |dw| = {wmax:.3e}"),
    );

    let (ws, wv) =
        ck.bf16(&format!("language_model.model.layers.{layer}.self_attn.g_proj.weight"));
    let cols = 8usize;
    let x = &hin[..4 * dims.hidden_size];
    let want = host_f64_matmul(x, 4, dims.hidden_size, &wv, cols);
    let sub = t2v(wv[..cols * ws[1]].to_vec(), cols, ws[1], dev);
    let got = vec_of(linear(t2v(x.to_vec(), 4, dims.hidden_size, dev), &sub, ActRound::None));
    let wantf: Vec<f32> = want.iter().map(|&v| v as f32).collect();
    r.close(
        "X2",
        "ops::linear against a float64 host matmul over the same checkpoint bytes — the \
         shared `nn.Linear` is the operation it claims to be, independently of any oracle",
        &got,
        &wantf,
        f32_budget(1, absmax(&wantf)),
        None,
    );

    let gain: Vec<f32> = (0..dims.hidden_size).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
    let mut want_n = vec![0f32; 4 * dims.hidden_size];
    for row in 0..4 {
        let s = &x[row * dims.hidden_size..(row + 1) * dims.hidden_size];
        let ms: f64 =
            s.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / dims.hidden_size as f64;
        let d = (ms + dims.rms_norm_eps).sqrt();
        for (i, &v) in s.iter().enumerate() {
            want_n[row * dims.hidden_size + i] = bf(bf((v as f64 / d) as f32) * gain[i]);
        }
    }
    let got_n = vec_of(rms_norm(
        t2v(x.to_vec(), 4, dims.hidden_size, dev),
        &t1v(gain, dev),
        dims.rms_norm_eps,
        ActRound::Bf16,
    ));
    r.close(
        "X3",
        "ops::rms_norm against a float64 host recompute with the cast in the shipped \
         place — two roundings, `x / sqrt(v)` and never `recip()`",
        &got_n,
        &want_n,
        bf16_budget(1, absmax(&want_n)),
        Some(EXACT_FRAC_ELEMENTWISE),
    );
}

// ===========================================================================
// main
// ===========================================================================

/// Checks the S lane produces. Exact, not a floor: a lane that lost a check is
/// a lane that stopped testing something.
const N_SELFTEST: usize = 8;
/// Checks the P lane produces.
const N_PREMISES: usize = 6;
/// Checks the O lane produces.
const N_ORACLE: usize = 2;
/// Checks the X lane produces.
const N_CROSS: usize = 3;
/// Checks a gated layer produces, excluding its attention sublayer. Identical
/// for both kinds by construction: everything outside `self_attn` is the same
/// module in both layers.
const N_LAYER_COMMON: usize = 49;
/// Checks the MLA sublayer's lane produces.
const N_MLA: usize = 22;
/// Checks the KDA sublayer's lane produces.
const N_KDA: usize = 21;
/// Distinct oracle arrays the gate must read. An exact floor, equal to what a
/// complete run reads: it can only rise when checks are added, and any check
/// that stops running drops it.
const N_ORACLE_ARRAYS_MIN: usize = 117;

/// The M lane: the whole prefix, from token ids to the final hidden state.
///
/// Returns the number of checks it added, so `Z1` can keep accounting for
/// every check rather than trusting a constant.
fn lane_model(
    r: &mut Report,
    p: &Oracle,
    ck: &Ckpt,
    cfg: &K3Config,
    dims: &MoeDims,
    dev: &Dev,
) -> usize {
    let before = r.checks.len();
    let t = &cfg.text_config;
    let n_layers = p.i64("meta_n_layers")[0] as usize;
    let ids = p.i64("model_input_ids");
    let ish = p.shape("model_input_ids");
    assert_eq!(ish.len(), 2, "model_input_ids rank");
    let (batch, seq) = (ish[0], ish[1]);
    let tokens = batch * seq;
    let hidden = t.hidden_size;
    assert_eq!(ids.len(), tokens, "model_input_ids length");

    println!("
== M: the whole {n_layers}-layer prefix, token ids -> hidden ==");

    // ---- embed_tokens ---------------------------------------------------
    //
    // Gathered row by row out of the raw checkpoint bytes. The table is
    // vocab x hidden and only `tokens` of its rows are ever touched, so
    // decoding all of it would cost gigabytes to read 32 rows.
    let (dt, esh, ebytes) = ck.raw("language_model.model.embed_tokens.weight");
    assert_eq!(dt, "BF16", "embed_tokens dtype");
    assert_eq!(esh, vec![t.vocab_size, hidden], "embed_tokens shape");
    let mut emb: Vec<f32> = Vec::with_capacity(tokens * hidden);
    for &id in &ids {
        let row = usize::try_from(id).expect("token id is not negative");
        assert!(row < t.vocab_size, "token id {row} outside the vocabulary");
        let base = row * hidden * 2;
        for j in 0..hidden {
            let o = base + j * 2;
            emb.push(f32::from_bits(
                (u16::from_le_bytes([ebytes[o], ebytes[o + 1]]) as u32) << 16,
            ));
        }
    }
    // The first layer's recorded input IS the embedding output, so this is a
    // real comparison and not a restatement: it catches a wrong row order
    // (the [b, t] -> b*seq + t flattening) and a wrong table.
    r.exact(
        "M/embed",
        "embed_tokens, gathered for this drive's ids, equals the input the oracle \
         recorded for layer 0 — bit for bit. Catches both a wrong table and a wrong \
         row order, which a shape check cannot tell apart",
        &emb,
        &p.bf16("L00_layer_in_bf16bits"),
    );

    // ---- the chain ------------------------------------------------------
    let schedule: Vec<bool> = (0..n_layers).map(|l| t.is_attn_res_checkpoint(l)).collect();
    let mut mixer = DepthMixer::<B>::new(schedule.clone());
    let mut hs = t2v(emb, tokens, hidden, dev);
    let mut rel: Vec<f64> = Vec::with_capacity(n_layers);

    for layer in 0..n_layers {
        let lp = format!("language_model.model.layers.{layer}");
        let (ln1_w, ln2_w) = ck.layer_norms::<B>(layer, dev);
        let sa_res: AttnResParams<B> =
            ck.attn_res_site(&format!("{lp}.self_attention_res"), t.rms_norm_eps, dev);
        let mlp_res: AttnResParams<B> =
            ck.attn_res_site(&format!("{lp}.mlp_res"), t.rms_norm_eps, dev);
        let attn = match t.attn_kind(layer) {
            AttnKind::Mla => {
                let mc = MlaConfig::from_text_config(t).expect("MlaConfig");
                K3Attn::Mla(Box::new(MlaBlock::new(mc.clone(), ck.mla_weights::<B>(layer, &mc, dev), Precision::Bf16)))
            }
            AttnKind::Kda => {
                let kc = KdaAttnConfig::from_text_config(t).expect("KdaAttnConfig");
                K3Attn::Kda(Box::new(KdaAttention::new(kc.clone(), ck.kda_weights::<B>(layer, &kc, dev), ActRound::Bf16)))
            }
        };
        let ffn = if t.is_moe_layer(layer) {
            K3Ffn::Moe(Box::new(ck.moe_block_weights::<B>(layer, true, dev)))
        } else {
            K3Ffn::Dense(Box::new(ck.mlp_weights::<B>(&format!("{lp}.mlp"), dev)))
        };
        let dec = K3DecoderLayer::new(
            layer,
            dims.clone(),
            ActRound::Bf16,
            ln1_w,
            ln2_w,
            sa_res,
            mlp_res,
            attn,
            ffn,
        );
        let mut cache = dec.new_cache(batch);
        let tr = dec.forward(&mut mixer, hs.clone(), batch, &mut cache, |id| {
            ck.expert(layer, id, dev)
        });
        hs = tr.out.clone();

        // The bank depth after this layer is a structural fact the schedule
        // fixes in advance; a mixer that took a snapshot on the wrong layer
        // lands here rather than drifting numerically for five more layers.
        let want_bank = schedule[..=layer].iter().filter(|&&b| b).count();
        r.boolean(
            &format!("M/L{layer:02}.bank"),
            "the depth bank holds exactly the snapshots the schedule takes through this \
             layer — a snapshot on the wrong layer is structural and shows here",
            mixer.bank().len() == want_bank,
            format!("{} snapshots, schedule takes {want_bank} through layer {layer}", mixer.bank().len()),
        );

        let want = p.bf16(&format!("L{layer:02}_layer_out_bf16bits"));
        let got = vec_of(hs.clone());
        let c = compare(&got, &want);
        let scale = absmax(&want);
        rel.push(if scale > 0.0 { c.max_abs / scale } else { f64::INFINITY });
        // 8 ulps per layer, accumulated. Stated up front rather than fitted to
        // what the run produced; if the real accumulation does not fit, that is
        // a measurement worth having and not a number to widen.
        r.close(
            &format!("M/L{layer:02}.out"),
            "the hidden state after this layer, carried from the token ids through every \
             layer before it under its own error — no oracle tensor enters the chain",
            &got,
            &want,
            bf16_budget(8 * (layer + 1), scale),
            None,
        );
    }

    // ---- the model-level AttnRes and the final norm ----------------------
    let out_res: AttnResParams<B> =
        ck.attn_res_site("language_model.model.output_attn_res", t.rms_norm_eps, dev);
    let mix = mixer.finish(hs.clone(), &out_res);
    let (nsh, nw) = ck.bf16("language_model.model.norm.weight");
    assert_eq!(nsh, vec![hidden], "model.norm.weight shape");
    // via t2v + reshape rather than a from_floats overload, so this uses the
    // same tensor construction every other check in this file goes through
    let norm_w = t2v(nw, 1, hidden, dev).reshape([hidden]);
    let normed = rms_norm(mix.out.clone(), &norm_w, t.rms_norm_eps, ActRound::Bf16);
    let want_final = p.bf16("model_last_hidden_state_bf16bits");
    let fscale = absmax(&want_final);
    r.close(
        "M/last_hidden_state",
        "the whole prefix: token ids, embedding, thirteen decoder layers, the \
         model-level depth mixture and model.norm, against the hidden state the \
         shipped model returned",
        &vec_of(normed),
        &want_final,
        bf16_budget(8 * (n_layers + 1), fscale),
        None,
    );

    // ---- how the error GREW ---------------------------------------------
    //
    // The more useful instrument. A bug confined to one layer barely moves the
    // final number when thirteen layers of honest rounding sit on top of it,
    // but it shows as a STEP here. The bound is on the ratio between
    // consecutive layers, so it does not care about the absolute scale.
    let mut worst = (0usize, 1.0f64);
    for l in 1..rel.len() {
        let prev = rel[l - 1];
        let ratio = if prev > 0.0 { rel[l] / prev } else { f64::INFINITY };
        if ratio > worst.1 {
            worst = (l, ratio);
        }
    }
    println!("  relative error by layer:");
    for (l, e) in rel.iter().enumerate() {
        println!("    L{l:02}  {e:.4e}");
    }
    r.boolean(
        "M/growth",
        "the relative error grows smoothly along the chain — no single layer multiplies \
         it by more than 8. A layer that is wrong in a way thirteen layers of rounding \
         would otherwise hide shows up as a step here rather than as a slightly larger \
         final number",
        worst.1 <= 8.0 && rel.iter().all(|e| e.is_finite()),
        format!("worst step at L{:02}, x{:.2}", worst.0, worst.1),
    );

    r.checks.len() - before
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let oracle_dir = PathBuf::from(
        args.get(1)
            .cloned()
            .or_else(|| std::env::var("K3_ORACLE_DIR").ok())
            .unwrap_or_else(|| "./k3-oracle".into()),
    );
    let model_dir = PathBuf::from(
        args.get(2)
            .cloned()
            .or_else(|| std::env::var("K3_MODEL_DIR").ok())
            .unwrap_or_else(|| "./kimi-k3".into()),
    );
    let dev: Dev = Default::default();
    let t_start = Instant::now();

    println!("k3_layer_gate — a whole Kimi-K3 decoder layer against the whole-layer oracle");
    println!("  oracle: {}", oracle_dir.display());
    println!("  model:  {}", model_dir.display());

    let mut r = Report::new();
    lane_selftest(&mut r);
    assert_eq!(r.checks.len(), N_SELFTEST, "the S lane changed size");
    if std::env::var("K3LAYER_SELFTEST").is_ok() {
        let f = r.failures().len();
        println!("\nSELFTEST: {} checks, {f} failed", r.checks.len());
        std::process::exit(if f == 0 { 0 } else { 1 });
    }

    let cfg: K3Config = {
        let p = model_dir.join("config.json");
        let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        serde_json::from_str(&s).expect("parse config.json")
    };
    let ck = Ckpt::open(&model_dir);
    let dims = MoeDims::from_text_config(&cfg.text_config).expect("MoeDims");
    let p = Oracle::open(&oracle_dir.join("layer_oracle_prefix13_bf16.npz"));

    let before = r.checks.len();
    lane_premises(&mut r, &ck, &cfg, &oracle_dir);
    assert_eq!(r.checks.len() - before, N_PREMISES, "the P lane changed size");

    println!("\n== O: the oracle bundle ==");
    r.boolean(
        "O1",
        "the oracle bundle holds the 1171 arrays its manifest declares",
        p.n_arrays() == 1171,
        format!("{} arrays", p.n_arrays()),
    );
    r.boolean(
        "O2",
        "the drive is the manifest's: batch 2, seq 16, seed 20260805, 13 layers",
        p.i64("meta_n_layers")[0] == 13
            && p.i64("meta_seed")[0] == 20260805
            && p.shape("input_ids") == vec![2, 16],
        format!(
            "n_layers {} seed {} input_ids {:?}",
            p.i64("meta_n_layers")[0],
            p.i64("meta_seed")[0],
            p.shape("input_ids")
        ),
    );

    const LAYERS: [usize; 3] = [3, 4, 12];
    let mut per_layer: Vec<(usize, LayerResult)> = Vec::new();
    for l in LAYERS {
        let res = run_layer(&mut r, &p, &ck, &cfg, &dims, &dev, l);
        per_layer.push((l, res));
    }

    let n_model = if std::env::var("K3_MODEL_LANE").is_ok() {
        lane_model(&mut r, &p, &ck, &cfg, &dims, &dev)
    } else {
        println!("
== M: skipped (set K3_MODEL_LANE=1 for the whole-prefix chain) ==");
        0
    };

    println!("
== X: cross-implementation, no oracle involved ==");
    let before = r.checks.len();
    lane_cross(&mut r, &p, &ck, &dims, &dev, 4);
    assert_eq!(r.checks.len() - before, N_CROSS, "the X lane changed size");

    println!("\n== Z: totality ==");
    // Derived from whatever LAYERS actually ran, so adding a layer cannot make
    // the totality check quietly stop covering it.
    let sum_layers: usize = per_layer.iter().map(|(_, res)| res.checks).sum();
    let shapes_ok = per_layer.iter().all(|(_, res)| {
        (res.attn_checks == N_MLA || res.attn_checks == N_KDA)
            && res.checks - res.attn_checks >= N_LAYER_COMMON
    });
    let per_layer_desc: Vec<String> = per_layer
        .iter()
        .map(|(l, res)| format!("L{l:02}:{}({}+{})", res.checks, res.checks - res.attn_checks, res.attn_checks))
        .collect();
    let derived = N_SELFTEST + N_PREMISES + N_ORACLE + N_CROSS + sum_layers + n_model;
    r.boolean(
        "Z1",
        "the check count is the sum of the lane counts, and each gated layer contributed \
         exactly the number of checks its attention kind implies — a lane that silently \
         narrowed itself (skipping later layers, returning early) lands here rather than \
         in the headline. The two totality checks themselves are the only ones outside \
         the sum, so a complete run is derived + 2",
        r.checks.len() == derived && shapes_ok && per_layer.len() == LAYERS.len(),
        format!(
            "{} = S{N_SELFTEST} + P{N_PREMISES} + O{N_ORACLE} + X{N_CROSS} + M{n_model} + {}",
            r.checks.len(),
            per_layer_desc.join(" + ")
        ),
    );
    r.boolean(
        "Z2",
        "the gate read at least as many distinct oracle arrays as a complete run does — \
         an oracle key that is never read is a boundary that is never checked, and the \
         count is the only thing that notices",
        p.n_used() >= N_ORACLE_ARRAYS_MIN,
        format!("{} distinct arrays read of {}", p.n_used(), p.n_arrays()),
    );

    println!("\n== PER-SUB-BLOCK ERROR TABLE ==\n");
    println!(
        "{:<6} {:<32} {:>12} {:>12} {:>7}  {}",
        "layer", "sub-block", "max|d|", "budget", "d/bud", "verdict"
    );
    for c in &r.checks {
        if let Some((layer, stage, d, bud)) = &c.table {
            println!(
                "{:<6} {:<32} {:>12.4e} {:>12.4e} {:>7.3}  {}",
                layer,
                stage,
                d,
                bud,
                if *bud > 0.0 { d / bud } else { 0.0 },
                if c.ok { "PASS" } else { "FAIL" }
            );
        }
    }
    for (l, res) in &per_layer {
        println!("\n  L{l:02}: {} checks, |layer_out|max = {:.4e}", res.checks, res.out_absmax);
    }

    let fails = r.failures();
    println!("\n=======================================================================");
    if fails.is_empty() {
        println!("GATE PASSED — {} checks, 0 failures", r.checks.len());
        println!("wall clock: {:.1} s", t_start.elapsed().as_secs_f64());
    } else {
        println!("GATE FAILED — {} checks, {} FAILURES", r.checks.len(), fails.len());
        for c in &fails {
            println!("  FAIL {}  {}\n       {}", c.id, c.what, c.detail);
        }
        println!("\nNo timing number is reported: the gate did not pass.");
        std::process::exit(1);
    }
}
