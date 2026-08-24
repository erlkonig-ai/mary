//! Correctness gate for [`mary::models::k3::mla`] — the NoPE MLA block —
//! against a whole-layer oracle captured from the shipped `KimiMLAAttention`
//! running on real Kimi K3 checkpoint weights.
//!
//! # What this gate is comparing against
//!
//! The oracle directory (`--vectors`, else `$MARY_MODELS/k3-oracle`) holds two
//! `.npz` bundles produced by driving
//! `modeling_kimi_linear.py`'s own `KimiLinearModel.forward` over a real
//! 13-layer prefix of the checkpoint. Both files' sha256 are pinned below and
//! re-checked on every run: a gate that reads whatever happens to be at a path
//! is gating against a moving target. The arrays used here are forward-hook
//! captures of the MLA sub-modules, so no arithmetic of the oracle author's
//! sits between the shipped module and the numbers.
//!
//! Four lanes, three of them independent references:
//!
//! | lane | reference | shape | what it can see |
//! |---|---|---|---|
//! | `f64` | `mla_L03_f64pure__*` — torch float64, fp32 islands widened | B=1, T=4 | everything; ~1e-13 noise floor |
//! | `bf16` | `L{03,07,11}_*_bf16bits` — torch bfloat16 on CUDA | B=2, T=16 | weight layout at all three MLA layers, one ULP |
//! | `f32` | `mla_L03_f32__*` — torch float32 on CUDA | B=1, T=4 | the f32 path, ~1e-6 |
//! | `cache` | `mla_L03_cache_prefill12_{key,value}` + the prefix run | B=1, T=12+4 | the KV cache and the decode path |
//!
//! The f64 lane is the sharp one. bfloat16 has an 8-bit mantissa, so an error
//! of 3e-3 relative is invisible in a bfloat16 comparison; a wrong RMSNorm
//! epsilon (1e-5 instead of the MLA-internal 1e-6) moves the answer by 4e-6 and
//! *only* the float64 lane can see it. The bfloat16 lane earns its place by
//! covering all three MLA layers, i.e. by being the lane that would catch a
//! weight loaded from the wrong shard.
//!
//! # What is asserted directly rather than through a composition
//!
//! Every projection, both RMSNorms, the assembled q/k/v, the pre-softmax
//! scores, the pre- and post-cast probabilities, the per-head attention output,
//! the gate projection, the gated input to `o_proj` and the block output are
//! each compared to their own captured array. An end-to-end-only comparison
//! would pass with two compensating errors, and could not say which projection
//! is mis-loaded.
//!
//! Three further things are asserted about *artifacts* rather than about the
//! port's round trips:
//!
//! * the on-disk safetensors shapes, against the shapes the parsed `config.json`
//!   predicts, before any tensor data is read;
//! * the oracle's own key/value provenance (`key_states[..., 0:128]` is
//!   `kv_b_proj`'s first half; `value_states` its second; `key_states[...,
//!   128:192]` is `kv_a_proj_with_mqa[..., 512:576]` broadcast over all 96
//!   heads) — re-derived here from the oracle arrays, so the port's structure
//!   is justified by the vectors and not only by reading the source;
//! * the causal mask this port constructs, against the mask the oracle
//!   captured, bit for bit.
//!
//! # The no-rotation controls
//!
//! `mla_use_nope` is true and nothing is rotated. Two controls hold that down:
//!
//! * **positive control** (`rope_would_break_it`): a real rotary embedding is
//!   applied to the carried lanes of the port's own q and k, the rest of the
//!   block is recomputed, and the result is required to be *far* from the
//!   oracle. This is what makes the other checks meaningful — a gate that
//!   cannot tell a rotated block from an unrotated one proves nothing by
//!   matching the unrotated one.
//! * **invariance, not a control** (`swapped_halves_is_an_invariance`): the
//!   oracle ships `mla_ALT_attn_out_swapped_halves`, built by swapping the
//!   128-wide and 64-wide halves of q and k. With nothing rotated that is a
//!   *joint permutation* of both sides of a dot product, so it is the identity
//!   to rounding. The gate asserts it MATCHES, and says so, rather than
//!   pretending it is a discriminating negative control.
//!
//! # Comparator discipline
//!
//! Every tolerance test is written `!(d <= tol)`, never `d > tol`: the latter
//! is false for NaN, so a NaN would score as zero error and pass. Every
//! comparison asserts both arrays are non-empty and the same length before
//! looking at values. The number of checks each lane contributes is asserted
//! against a constant, so a lane that silently does not run fails the gate
//! rather than passing vacuously.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use half::bf16;
use serde::Serialize;
use sha2::{Digest, Sha256};

use mary::models::k3::config::{AttnKind, K3Config};
use mary::models::k3::mla::{MlaBlock, MlaConfig, MlaKvCache, MlaTrace, MlaWeights, Precision};
use mary::nn::npz::Npz;

type F64B = NdArray<f64>;
type F32B = NdArray<f32>;

/// sha256 of `layer_oracle_prefix13_bf16.npz`, from `MANIFEST_layer_oracle.md`.
const SHA_PREFIX13: &str = "fdb3b897f0bb43e8506d27dd283defee87910006dd1038c131687a1b48e61d7c";
/// sha256 of `layer_oracle_ladder.npz`, from the same manifest.
const SHA_LADDER: &str = "83daedc5071e93bcbed3f7bedeaefbc84c309ddc08c43fcaa6346150f958d1e5";

/// The three MLA layers inside the oracle's 13-layer prefix. These are
/// `full_attn_layers - 1`: the config's lists are 1-based and the layer indices
/// are not. The gate re-derives them from the config rather than trusting this
/// constant — see `check_layer_kinds`.
const MLA_LAYERS: [usize; 3] = [3, 7, 11];

/// Per-lane expected check counts. A lane that fails to run contributes zero
/// and trips this, instead of an `all()` over an empty set returning true.
const N_CHECKS_SETUP: usize = 6;
const N_CHECKS_PER_SUBBLOCK_LANE: usize = 27;
/// Checks in one teacher-forced lane: eight projections/norms driven from the
/// oracle's own captured input, three bit-exact assemblies, four attention
/// intermediates, and the output gate.
const N_CHECKS_TEACHER_FORCED: usize = 16;
/// The one `..._weight_shapes_match_config` fact a lane records the first time
/// it loads a layer's weights.
const N_CHECKS_WEIGHT_SHAPE: usize = 1;
const N_CHECKS_CONTROL: usize = 2;
const N_CHECKS_CACHE: usize = 6;

// ---------------------------------------------------------------------------
// checks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct Check {
    lane: String,
    name: String,
    /// The oracle array (or artifact) this check is against.
    oracle: String,
    /// What the check demands: `match` (small difference), `differ` (large
    /// difference — a positive control), `exact` (bitwise), `fact` (a boolean
    /// property with no tolerance).
    kind: &'static str,
    /// Elements compared. Asserted non-zero before any comparison.
    n: usize,
    /// Elements that were masked out (`|oracle| > 1e30`) and compared exactly.
    n_masked: usize,
    max_abs: f64,
    ref_absmax: f64,
    /// `max_abs / ref_absmax`.
    rel: f64,
    tol: f64,
    /// Fraction of elements whose round-to-bfloat16 bit patterns agree exactly.
    bitexact: Option<f64>,
    min_bitexact: Option<f64>,
    ok: bool,
    detail: String,
}

/// `max |a - b|`, **propagating NaN instead of dropping it**.
///
/// `f64::max` returns the other operand when one side is NaN, so the obvious
/// `fold(0.0, |m, d| m.max(d))` silently discards every NaN it meets and
/// reports a clean maximum over the finite elements. An array of garbage then
/// scores exactly zero error and every downstream `!(d <= tol)` passes — the
/// comparator discipline defeated one layer below where it is written. Found by
/// mutation G01, which put a NaN in the port and watched a numeric check stay
/// green. This returns NaN the moment it sees one; every comparator then fails
/// on its `is_nan` test.
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if d.is_nan() {
            return f64::NAN;
        }
        if d > m {
            m = d;
        }
    }
    m
}

/// `max |a|`, propagating NaN. Same reasoning as [`max_abs_diff`].
fn absmax(a: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for x in a {
        let v = x.abs();
        if v.is_nan() {
            return f64::NAN;
        }
        if v > m {
            m = v;
        }
    }
    m
}

#[derive(Default)]
struct Gate {
    checks: Vec<Check>,
}

impl Gate {
    fn n_since(&self, mark: usize) -> usize {
        self.checks.len() - mark
    }

    fn failed(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| !c.ok).collect()
    }

    /// Guard every comparison: an empty array or a length mismatch is a bug in
    /// the gate itself, so it aborts loudly instead of scoring 0.0 error over
    /// nothing.
    fn guard(name: &str, a: &[f64], b: &[f64]) {
        assert!(
            !a.is_empty() && !b.is_empty(),
            "{}: comparing empty arrays (port {}, oracle {}) — a zero-length \
             match is a green measurement of nothing",
            name,
            a.len(),
            b.len()
        );
        assert_eq!(
            a.len(),
            b.len(),
            "{}: length mismatch, port {} vs oracle {}",
            name,
            a.len(),
            b.len()
        );
    }

    /// Compare two arrays, requiring `max|a-b| / max|b| <= tol`.
    fn cmp(&mut self, lane: &str, name: &str, oracle: &str, port: &[f64], gold: &[f64], tol: f64) {
        Self::guard(name, port, gold);
        let ref_absmax = absmax(gold);
        let max_abs = max_abs_diff(port, gold);
        let rel = if ref_absmax > 0.0 {
            max_abs / ref_absmax
        } else {
            max_abs
        };
        // `!(rel <= tol)` and not `rel > tol`: NaN fails the former, passes the
        // latter.
        let ok = !(rel > tol) && !rel.is_nan();
        self.checks.push(Check {
            lane: lane.into(),
            name: name.into(),
            oracle: oracle.into(),
            kind: "match",
            n: port.len(),
            n_masked: 0,
            max_abs,
            ref_absmax,
            rel,
            tol,
            bitexact: None,
            min_bitexact: None,
            ok,
            detail: String::new(),
        });
    }

    /// The teacher-forced bfloat16 criterion: one operation, driven from the
    /// reference's own captured input for that boundary.
    ///
    /// **Default: bit agreement.** With identical inputs and identical weights
    /// the only difference between the port and torch is the order of one fp32
    /// accumulation — relative ~1e-6, far under half a bfloat16 ULP of 2e-3 —
    /// so a correct port lands within one ULP and rounds to the *same bfloat16
    /// bits* for essentially every element. That is the check: `max|port-gold|
    /// <= ref_absmax * 2^-7` **and** at least `min_bitexact` of the elements
    /// bit-identical. Fifteen of the sixteen operations gated here satisfy it
    /// at 0.9978 to 1.0000.
    ///
    /// **The exception, stated as a predicate rather than a name.** cuBLAS does
    /// not promise fp32 accumulation for bfloat16 GEMMs:
    /// `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`
    /// defaults to **true**, which permits a split-K reduction to combine its
    /// partial sums in bfloat16. Split-K is a long-reduction strategy, so the
    /// exposure is a property of `K`: of the six GEMMs in this block, five
    /// reduce over K = 512, 1536 or 7168, and exactly one — `o_proj`, K =
    /// 12288 — is long enough to expect it. Where `reference_may_split_k` is
    /// set, bit agreement is not achievable by *any* fp32-accumulating
    /// implementation, so the gate asks the stronger question instead: which of
    /// the two runs is closer to the exact answer? The float64 arbiter — the
    /// same port code in a width where bfloat16 rounding is invisible, gated to
    /// 1e-15 against torch's own float64 run — decides, and the port must be at
    /// least **twice** as close.
    ///
    /// The 1-for-1 correspondence is the evidence: the one operation that fails
    /// bit agreement (0.377-0.397 bit-identical, a hundredfold outlier against
    /// the other fifteen) is the one operation whose K predicts it, and there
    /// the port is 3.1-4.5x closer to exact than the reference. `o_proj`'s
    /// arithmetic is not left ungated by this — the float32 lane pins it to
    /// 1.7e-7 and the float64 lane to 6e-16. Only its bfloat16 *bit pattern* is
    /// unreproducible.
    ///
    /// Why "at least twice as close" and not "no worse": a port that merely
    /// rounds somewhere else is about as far from exact as the reference, so
    /// "no worse" would accept it. Mutants M03 (RMSNorm rounding order) and M16
    /// (softmax cast-back dropped) are exactly that shape, and both survived an
    /// earlier version of this function that offered "no worse than the
    /// reference" as an alternative to bit agreement for *every* operation.
    fn cmp_bf16_ref(
        &mut self,
        lane: &str,
        name: &str,
        oracle: &str,
        port: &[f64],
        gold: &[f64],
        reference: &[f64],
        min_bitexact: f64,
        reference_may_split_k: bool,
    ) {
        Self::guard(name, port, gold);
        Self::guard(name, port, reference);
        // Masked score positions hold finfo(bf16).min. Left in, they would set
        // `ref_absmax` to 3.4e38 and drive every relative error to ~1e-40 — a
        // comparator defeated by its own outlier, reporting a number that
        // cannot fail. They are excluded from the statistics and required to
        // agree exactly instead.
        const BIG: f64 = 1e30;
        let mut ref_absmax = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut same = 0usize;
        let mut n_open = 0usize;
        let mut n_masked = 0usize;
        let mut mask_mismatch = 0usize;
        let mut saw_nan = false;
        // A bfloat16 lane must EMIT bfloat16. Comparing `round(port)` to
        // `round(gold)` — which is what the bit-agreement fraction does — is
        // blind to a rounding step the port skipped entirely: an unrounded f32
        // value rounds to the same bfloat16 as the reference's already-rounded
        // one, and scores as identical. So the values themselves are checked
        // for representability, which is the property the lane claims. Found by
        // mutation M16 (the softmax's cast back to the activation dtype
        // dropped), which survived a version of this function that only
        // compared rounded values.
        let mut n_not_bf16 = 0usize;
        let mut n_gold_not_bf16 = 0usize;
        for ((p, q), r) in port.iter().zip(gold).zip(reference) {
            if p.is_nan() || q.is_nan() || r.is_nan() {
                saw_nan = true;
                continue;
            }
            if q.abs() > BIG || p.abs() > BIG {
                n_masked += 1;
                if p != q {
                    mask_mismatch += 1;
                }
                continue;
            }
            n_open += 1;
            if bf16::from_f64(*p).to_f64() != *p {
                n_not_bf16 += 1;
            }
            if bf16::from_f64(*q).to_f64() != *q {
                n_gold_not_bf16 += 1;
            }
            ref_absmax = ref_absmax.max(q.abs());
            max_abs = max_abs.max((p - q).abs());
            if bf16::from_f64(*p).to_bits() == bf16::from_f64(*q).to_bits() {
                same += 1;
            }
        }
        assert!(n_open > 0, "{}: every element was masked out or NaN", name);
        let rel = if ref_absmax > 0.0 {
            max_abs / ref_absmax
        } else {
            max_abs
        };
        let frac = same as f64 / n_open as f64;
        let ulp = 2f64.powi(-7);
        let e_port = max_abs_diff(port, reference);
        let e_oracle = max_abs_diff(gold, reference);

        assert_eq!(
            n_gold_not_bf16, 0,
            "{}: {} of the reference's own values are not representable in \
             bfloat16 — this lane is not reading a bfloat16 array",
            name, n_gold_not_bf16
        );
        let ok = !saw_nan
            && mask_mismatch == 0
            && n_not_bf16 == 0
            && if reference_may_split_k {
                assert!(
                    e_oracle > 0.0,
                    "{}: the bfloat16 reference is bit-identical to the float64 \
                     arbiter, so the closer-to-exact test measures nothing",
                    name
                );
                !(e_port > 0.5 * e_oracle) && !e_port.is_nan()
            } else {
                !(rel > ulp) && !rel.is_nan() && !(frac < min_bitexact)
            };
        self.checks.push(Check {
            lane: lane.into(),
            name: name.into(),
            oracle: oracle.into(),
            kind: if reference_may_split_k {
                "split-k"
            } else {
                "bits"
            },
            n: port.len(),
            n_masked,
            max_abs,
            ref_absmax,
            rel,
            tol: ulp,
            bitexact: Some(frac),
            min_bitexact: Some(min_bitexact),
            ok,
            detail: format!(
                "{}{}{}; |port-f64| {:.4e} vs |reference-f64| {:.4e} = {:.3}x",
                if saw_nan { "NaN PRESENT; " } else { "" },
                if n_not_bf16 > 0 {
                    format!("{} PORT VALUES NOT REPRESENTABLE IN BF16; ", n_not_bf16)
                } else {
                    String::new()
                },
                if reference_may_split_k {
                    "K>8192: reference may have reduced in bfloat16, so the port \
                     must be >=2x closer to exact"
                } else {
                    "bit agreement demanded"
                },
                e_port,
                e_oracle,
                e_port / e_oracle
            ),
        });
    }

    /// Compare a bfloat16 cascade against a bfloat16 reference *through* a
    /// float64 reference computed by this gate from the same input and the same
    /// weights.
    ///
    /// A cascade cannot be compared to one ULP and should not pretend to be.
    /// bfloat16 rounding compounds: after six stages the port's run and torch's
    /// run are each within a ULP or two of the exact answer at every step, and
    /// therefore a few ULP from *each other*, with no error anywhere. Demanding
    /// one ULP end to end would be a tolerance that a correct port cannot meet;
    /// widening the tolerance until it passes would be a tolerance chosen to
    /// pass.
    ///
    /// So this compares the two runs' **distances from the exact answer**:
    ///
    /// ```text
    /// e_port   = max |port_bf16   - port_f64|
    /// e_oracle = max |oracle_bf16 - port_f64|
    /// require    e_port <= ratio * e_oracle
    /// ```
    ///
    /// Both are the same rounding process on the same arithmetic, so the ratio
    /// is ~1 for a correct port whatever the stage; `ratio` only has to cover
    /// the spread of a max-over-N of independent rounding realisations. A real
    /// mistake moves `e_port` by orders of magnitude and does not touch
    /// `e_oracle`. `rel` in the report is the ratio itself.
    ///
    /// `e_oracle` must be non-zero, or the reference *is* the oracle and the
    /// check is vacuous.
    fn cmp_ref(
        &mut self,
        lane: &str,
        name: &str,
        oracle: &str,
        port: &[f64],
        gold: &[f64],
        reference: &[f64],
        ratio: f64,
    ) {
        Self::guard(name, port, gold);
        Self::guard(name, port, reference);
        let e_port = max_abs_diff(port, reference);
        let e_oracle = max_abs_diff(gold, reference);
        assert!(
            e_oracle > 0.0,
            "{}: the bfloat16 oracle is bit-identical to the float64 reference, \
             so this comparison measures nothing",
            name
        );
        let r = e_port / e_oracle;
        let ok = !(r > ratio) && !r.is_nan();
        self.checks.push(Check {
            lane: lane.into(),
            name: name.into(),
            oracle: oracle.into(),
            kind: "ratio",
            n: port.len(),
            n_masked: 0,
            max_abs: e_port,
            ref_absmax: e_oracle,
            rel: r,
            tol: ratio,
            bitexact: None,
            min_bitexact: None,
            ok,
            detail: format!(
                "port is {:.3}x the reference bfloat16 run's own distance from float64",
                r
            ),
        });
    }

    /// Compare pre-softmax scores, where the masked positions hold
    /// `finfo(bf16).min` and would otherwise swamp the relative error.
    ///
    /// The masked positions are compared *exactly* and their positions must
    /// agree — an off-by-one causal mask changes which entries are masked, and
    /// that is caught here rather than washed out.
    fn cmp_masked(
        &mut self,
        lane: &str,
        name: &str,
        oracle: &str,
        port: &[f64],
        gold: &[f64],
        tol: f64,
    ) {
        Self::guard(name, port, gold);
        const BIG: f64 = 1e30;
        let mut n_masked = 0usize;
        let mut mask_mismatch = 0usize;
        let mut ref_absmax = 0.0f64;
        let mut max_abs = 0.0f64;
        for (p, g) in port.iter().zip(gold) {
            let gm = g.abs() > BIG;
            let pm = p.abs() > BIG;
            if gm || pm {
                n_masked += 1;
                if !(gm && pm) || p != g {
                    mask_mismatch += 1;
                }
            } else {
                let d = (p - g).abs();
                if d.is_nan() || g.is_nan() {
                    ref_absmax = f64::NAN;
                    max_abs = f64::NAN;
                } else {
                    ref_absmax = ref_absmax.max(g.abs());
                    max_abs = max_abs.max(d);
                }
            }
        }
        let n_open = port.len() - n_masked;
        assert!(
            n_open > 0 && n_masked > 0,
            "{}: expected both masked and unmasked entries, got {} masked of {}",
            name,
            n_masked,
            port.len()
        );
        let rel = if ref_absmax > 0.0 {
            max_abs / ref_absmax
        } else {
            max_abs
        };
        let ok = !(rel > tol) && !rel.is_nan() && !max_abs.is_nan() && mask_mismatch == 0;
        self.checks.push(Check {
            lane: lane.into(),
            name: name.into(),
            oracle: oracle.into(),
            kind: "match",
            n: port.len(),
            n_masked,
            max_abs,
            ref_absmax,
            rel,
            tol,
            bitexact: None,
            min_bitexact: None,
            ok,
            detail: format!(
                "{} masked positions, {} of them mismatched",
                n_masked, mask_mismatch
            ),
        });
    }

    /// Require two arrays to be *bitwise* identical.
    fn cmp_exact(&mut self, lane: &str, name: &str, oracle: &str, a: &[f64], b: &[f64]) {
        Self::guard(name, a, b);
        let bad = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let max_abs = max_abs_diff(a, b);
        let ref_absmax = absmax(b);
        self.checks.push(Check {
            lane: lane.into(),
            name: name.into(),
            oracle: oracle.into(),
            kind: "exact",
            n: a.len(),
            n_masked: 0,
            max_abs,
            ref_absmax,
            rel: 0.0,
            tol: 0.0,
            bitexact: None,
            min_bitexact: None,
            ok: bad == 0,
            detail: format!("{} of {} elements differ bitwise", bad, a.len()),
        });
    }

    /// A positive control: require the two arrays to be *far apart*.
    fn expect_differ(
        &mut self,
        lane: &str,
        name: &str,
        oracle: &str,
        a: &[f64],
        b: &[f64],
        min_rel: f64,
        detail: &str,
    ) {
        Self::guard(name, a, b);
        let ref_absmax = absmax(b);
        let max_abs = max_abs_diff(a, b);
        let rel = if ref_absmax > 0.0 {
            max_abs / ref_absmax
        } else {
            max_abs
        };
        // `!(rel < min_rel)` is false for NaN too, so a NaN control fails.
        let ok = rel >= min_rel;
        self.checks.push(Check {
            lane: lane.into(),
            name: name.into(),
            oracle: oracle.into(),
            kind: "differ",
            n: a.len(),
            n_masked: 0,
            max_abs,
            ref_absmax,
            rel,
            tol: min_rel,
            bitexact: None,
            min_bitexact: None,
            ok,
            detail: detail.into(),
        });
    }

    /// A boolean property with no numbers attached.
    fn fact(&mut self, lane: &str, name: &str, oracle: &str, ok: bool, n: usize, detail: String) {
        self.checks.push(Check {
            lane: lane.into(),
            name: name.into(),
            oracle: oracle.into(),
            kind: "fact",
            n,
            n_masked: 0,
            max_abs: 0.0,
            ref_absmax: 0.0,
            rel: 0.0,
            tol: 0.0,
            bitexact: None,
            min_bitexact: None,
            ok,
            detail,
        });
    }
}

// ---------------------------------------------------------------------------
// checkpoint access
// ---------------------------------------------------------------------------

/// The safetensors shards, read through the index rather than guessed at.
struct Ckpt {
    dir: PathBuf,
    map: HashMap<String, String>,
}

impl Ckpt {
    fn open(dir: &Path) -> Self {
        let idx = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
            .expect("model.safetensors.index.json");
        let v: serde_json::Value = serde_json::from_str(&idx).expect("index json");
        let map = v["weight_map"]
            .as_object()
            .expect("weight_map")
            .iter()
            .map(|(k, s)| (k.clone(), s.as_str().expect("shard name").to_string()))
            .collect();
        Self {
            dir: dir.to_path_buf(),
            map,
        }
    }

    /// Read one BF16 tensor, widened to f64 exactly. Refuses any other dtype
    /// rather than silently converting — every MLA weight in this checkpoint is
    /// BF16 (unlike the routed experts, which are packed MXFP4), and a dtype
    /// surprise means the tensor is not what this gate thinks it is.
    fn tensor_bf16(&self, name: &str) -> (Vec<f64>, Vec<usize>) {
        let shard = self
            .map
            .get(name)
            .unwrap_or_else(|| panic!("tensor '{}' is not in the checkpoint index", name));
        let path = self.dir.join(shard);
        let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap shard");
        let st = safetensors::SafeTensors::deserialize(&mmap).expect("safetensors header");
        let view = st.tensor(name).unwrap_or_else(|e| panic!("{}: {e}", name));
        assert_eq!(
            view.dtype(),
            safetensors::Dtype::BF16,
            "{}: expected BF16, found {:?}",
            name,
            view.dtype()
        );
        let shape = view.shape().to_vec();
        let data = view.data();
        assert!(!data.is_empty(), "{}: zero-length tensor", name);
        let v: Vec<f64> = data
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f64())
            .collect();
        assert_eq!(
            v.len(),
            shape.iter().product::<usize>(),
            "{}: shape/data mismatch",
            name
        );
        (v, shape)
    }
}

fn t2<B: Backend>(v: Vec<f64>, r: usize, c: usize, dev: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(v, [r, c]).convert::<B::FloatElem>(), dev)
}

fn t1<B: Backend>(v: Vec<f64>, n: usize, dev: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(v, [n]).convert::<B::FloatElem>(), dev)
}

/// Load one MLA block's eight weights, asserting every on-disk shape against
/// the shape the parsed config predicts *before* the data is used.
fn load_weights<B: Backend>(
    ck: &Ckpt,
    cfg: &MlaConfig,
    layer: usize,
    dev: &B::Device,
    g: &mut Gate,
    record: bool,
) -> MlaWeights<B> {
    let h = cfg.num_heads;
    let qlora = cfg.q_lora_rank.expect("q_lora_rank");
    let want: Vec<(&str, [usize; 2])> = vec![
        ("q_a_proj", [qlora, cfg.hidden_size]),
        ("q_a_layernorm", [qlora, 0]),
        ("q_b_proj", [h * cfg.q_head_dim(), qlora]),
        (
            "kv_a_proj_with_mqa",
            [cfg.kv_lora_rank + cfg.qk_carried_head_dim, cfg.hidden_size],
        ),
        ("kv_a_layernorm", [cfg.kv_lora_rank, 0]),
        ("kv_b_proj", [h * cfg.kv_b_head_dim(), cfg.kv_lora_rank]),
        ("o_proj", [cfg.hidden_size, h * cfg.v_head_dim]),
        ("g_proj", [h * cfg.v_head_dim, cfg.hidden_size]),
    ];
    let mut got: HashMap<&str, (Vec<f64>, Vec<usize>)> = HashMap::new();
    let mut shape_ok = true;
    let mut detail = String::new();
    for (part, exp) in &want {
        let name = format!(
            "language_model.model.layers.{}.self_attn.{}.weight",
            layer, part
        );
        let (v, shape) = ck.tensor_bf16(&name);
        let expected: Vec<usize> = if exp[1] == 0 {
            vec![exp[0]]
        } else {
            exp.to_vec()
        };
        if shape != expected {
            shape_ok = false;
            detail.push_str(&format!(
                "{} on disk {:?} != config {:?}; ",
                part, shape, expected
            ));
        }
        got.insert(part, (v, shape));
    }
    if record {
        g.fact(
            "setup",
            &format!("L{:02}_weight_shapes_match_config", layer),
            "safetensors headers vs config.json",
            shape_ok,
            want.len(),
            if shape_ok {
                "all 8 shapes as predicted".into()
            } else {
                detail
            },
        );
    }
    let take = |k: &str| got.get(k).unwrap().clone();
    let (qa, qas) = take("q_a_proj");
    let (qan, _) = take("q_a_layernorm");
    let (qb, qbs) = take("q_b_proj");
    let (kva, kvas) = take("kv_a_proj_with_mqa");
    let (kvan, _) = take("kv_a_layernorm");
    let (kvb, kvbs) = take("kv_b_proj");
    let (o, os) = take("o_proj");
    let (gp, gps) = take("g_proj");
    MlaWeights {
        q_a_proj: t2(qa, qas[0], qas[1], dev),
        q_a_layernorm: t1(qan, qlora, dev),
        q_b_proj: t2(qb, qbs[0], qbs[1], dev),
        kv_a_proj_with_mqa: t2(kva, kvas[0], kvas[1], dev),
        kv_a_layernorm: t1(kvan, cfg.kv_lora_rank, dev),
        kv_b_proj: t2(kvb, kvbs[0], kvbs[1], dev),
        o_proj: t2(o, os[0], os[1], dev),
        g_proj: Some(t2(gp, gps[0], gps[1], dev)),
    }
}

// ---------------------------------------------------------------------------
// oracle access
// ---------------------------------------------------------------------------

/// One oracle lane's key naming. `bits` selects the `_bf16bits` (uint16 holding
/// raw bfloat16 bit patterns) variant where one exists.
struct Lane<'a> {
    z: &'a Npz,
    prefix: String,
    bits: bool,
}

impl<'a> Lane<'a> {
    /// Fetch an array by its sub-block name, returning the key it resolved to
    /// so the report names the array it actually compared against.
    fn get(&self, name: &str) -> (String, Vec<f64>) {
        if self.bits {
            let k = format!("{}{}_bf16bits", self.prefix, name);
            if self.z.contains(&k) {
                let a = self.z.get(&k);
                assert!(!a.is_empty(), "oracle array {} is empty", k);
                return (k.clone(), a.bf16_to_f64());
            }
        }
        let k = format!("{}{}", self.prefix, name);
        let a = self.z.get(&k);
        assert!(!a.is_empty(), "oracle array {} is empty", k);
        (k.clone(), a.to_f64())
    }

    fn dims(&self, name: &str) -> Vec<usize> {
        let k = if self.bits {
            let kb = format!("{}{}_bf16bits", self.prefix, name);
            if self.z.contains(&kb) {
                kb
            } else {
                format!("{}{}", self.prefix, name)
            }
        } else {
            format!("{}{}", self.prefix, name)
        };
        self.z.get(&k).shape.clone()
    }
}

fn host<B: Backend, const D: usize>(t: &Tensor<B, D>) -> Vec<f64> {
    t.clone()
        .into_data()
        .convert::<f64>()
        .into_vec()
        .expect("readback")
}

// ---------------------------------------------------------------------------
// one sub-block lane
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Tol {
    /// Absolute-relative tolerance (float64 / float32 lanes).
    Rel(f64),
    /// Compare the port's and the reference's distances from a float64 run of
    /// the same input, and require the ratio to be at most this. See
    /// [`Gate::cmp_ref`].
    RefRatio(f64),
}

#[allow(clippy::too_many_arguments)]
fn run_subblock_lane<B: Backend>(
    g: &mut Gate,
    lane_name: &str,
    layer: usize,
    lane: &Lane,
    blk: &MlaBlock<B>,
    dev: &B::Device,
    tol: Tol,
    reference: Option<&HashMap<&'static str, Vec<f64>>>,
) -> MlaTrace<B> {
    let lp = format!("L{:02}_", layer);
    let k = |s: &str| format!("{}{}", lp, s);

    let (in_key, hidden_v) = lane.get(&k("attn_q_a_proj_in"));
    let in_dims = lane.dims(&k("attn_q_a_proj_in"));
    assert_eq!(
        in_dims.len(),
        3,
        "layer input must be [B,T,H], got {:?}",
        in_dims
    );
    let (b, t, dh) = (in_dims[0], in_dims[1], in_dims[2]);
    assert!(
        b > 0 && t > 0 && dh > 0,
        "degenerate layer input {:?}",
        in_dims
    );

    // Provenance of the input itself: the MLA block's input is the decoder
    // layer's `input_layernorm` output, and all three MLA projections see the
    // same tensor. Checked on the oracle so a lane that fed the port the wrong
    // array cannot pass by being self-consistent.
    let (ln_key, ln_v) = lane.get(&k("input_layernorm_out"));
    g.cmp_exact(
        lane_name,
        "input_is_input_layernorm_out",
        &format!("{} vs {}", in_key, ln_key),
        &hidden_v,
        &ln_v,
    );
    let (gin_key, gin_v) = lane.get(&k("attn_g_proj_in"));
    g.cmp_exact(
        lane_name,
        "g_proj_sees_the_same_hidden",
        &format!("{} vs {}", in_key, gin_key),
        &hidden_v,
        &gin_v,
    );

    let hidden: Tensor<B, 3> = Tensor::from_data(
        TensorData::new(hidden_v, [b, t, dh]).convert::<B::FloatElem>(),
        dev,
    );

    // The mask this port builds, against the mask the oracle captured.
    let mask = MlaBlock::<B>::causal_mask(b, t, t, 0, dev);
    let (mask_key, mask_v) = lane.get(&k("mla_attn_mask"));
    g.cmp_exact(lane_name, "causal_mask", &mask_key, &host(&mask), &mask_v);

    let tr = blk.forward(hidden, Some(mask), None);

    // The two structural no-rotation assertions, run inside `forward` and
    // re-run here for their element counts.
    let n_verbatim = tr.assert_carried_verbatim();
    g.fact(
        lane_name,
        "carried_lane_verbatim",
        "port-internal: assembled q/k carried lane vs projection output",
        true,
        n_verbatim,
        "bit-identical; any rotation between projection and assembly aborts".into(),
    );
    let n_bcast = tr.assert_carried_is_broadcast();
    g.fact(
        lane_name,
        "carried_key_broadcast_over_heads",
        "port-internal: key_states carried lane across 96 heads",
        true,
        n_bcast,
        "identical across heads while the passed lane is not".into(),
    );

    let cmp = |g: &mut Gate, name: &'static str, port: Vec<f64>, oracle_name: &str| {
        let (key, gold) = lane.get(&k(oracle_name));
        match tol {
            Tol::Rel(t) => g.cmp(lane_name, name, &key, &port, &gold, t),
            Tol::RefRatio(r) => {
                let rf = reference
                    .expect("a RefRatio lane needs a float64 reference")
                    .get(name)
                    .unwrap_or_else(|| panic!("no float64 reference recorded for '{}'", name));
                g.cmp_ref(lane_name, name, &key, &port, &gold, rf, r);
            }
        }
    };

    cmp(
        g,
        "q_a_proj_out",
        host(&tr.q_a_proj_out),
        "attn_q_a_proj_out",
    );
    cmp(
        g,
        "q_a_layernorm_out",
        host(&tr.q_a_layernorm_out),
        "attn_q_a_layernorm_out",
    );
    cmp(
        g,
        "q_b_proj_out",
        host(&tr.q_b_proj_out),
        "attn_q_b_proj_out",
    );
    cmp(
        g,
        "kv_a_proj_out",
        host(&tr.kv_a_proj_out),
        "attn_kv_a_proj_with_mqa_out",
    );
    cmp(
        g,
        "kv_a_layernorm_in",
        host(&tr.kv_a_layernorm_in),
        "attn_kv_a_layernorm_in",
    );
    cmp(
        g,
        "kv_a_layernorm_out",
        host(&tr.kv_a_layernorm_out),
        "attn_kv_a_layernorm_out",
    );
    cmp(
        g,
        "kv_b_proj_out",
        host(&tr.kv_b_proj_out),
        "attn_kv_b_proj_out",
    );
    cmp(
        g,
        "query_states",
        host(&tr.query_states),
        "mla_query_states",
    );
    cmp(g, "key_states", host(&tr.key_states), "mla_key_states");
    cmp(
        g,
        "value_states",
        host(&tr.value_states),
        "mla_value_states",
    );
    // `attn_probs_precast` is a float32 array in the bfloat16 lane too — it is
    // the fp32 island's output, before the cast back — so it is compared with a
    // float tolerance in every lane rather than a bfloat16 one.
    {
        let (key, gold) = lane.get(&k("mla_attn_probs_precast"));
        match tol {
            Tol::Rel(t) => g.cmp(
                lane_name,
                "attn_probs_precast",
                &key,
                &host(&tr.probs_precast),
                &gold,
                t,
            ),
            Tol::RefRatio(r) => {
                let rf = reference
                    .expect("reference")
                    .get("attn_probs_precast")
                    .expect("reference");
                g.cmp_ref(
                    lane_name,
                    "attn_probs_precast",
                    &key,
                    &host(&tr.probs_precast),
                    &gold,
                    rf,
                    r,
                )
            }
        }
    }
    cmp(g, "attn_probs", host(&tr.probs), "mla_attn_probs");
    cmp(
        g,
        "attn_out_heads",
        host(&tr.attn_out_heads),
        "mla_attn_out_heads",
    );
    cmp(
        g,
        "g_proj_out",
        host(tr.g_proj_out.as_ref().expect("output gate")),
        "attn_g_proj_out",
    );
    cmp(g, "o_proj_in", host(&tr.o_proj_in), "attn_o_proj_in");
    cmp(g, "block_out", host(&tr.out), "attn_o_proj_out");

    // Scores carry the mask's finfo(bf16).min at the disallowed positions.
    {
        let (key, gold) = lane.get(&k("mla_attn_scores_precast"));
        match tol {
            Tol::Rel(t) => {
                g.cmp_masked(lane_name, "attn_scores", &key, &host(&tr.scores), &gold, t)
            }
            // Masked positions hold finfo(bf16).min in all three runs and so
            // contribute exactly zero to both distances; no special handling.
            Tol::RefRatio(r) => {
                let rf = reference
                    .expect("reference")
                    .get("attn_scores")
                    .expect("reference");
                g.cmp_ref(
                    lane_name,
                    "attn_scores",
                    &key,
                    &host(&tr.scores),
                    &gold,
                    rf,
                    r,
                )
            }
        }
    }

    // The softmax scale is the FULL 192-wide head dim, carried lane included.
    {
        let (key, gold) = lane.get(&k("mla_scaling"));
        let want = blk.cfg.scaling();
        let ok = gold.len() == 1
            && (gold[0] - want).abs() <= 1e-15
            && (gold[0] - 128f64.powf(-0.5)).abs() > 1e-6;
        g.fact(
            lane_name,
            "scaling_is_q_head_dim",
            &key,
            ok,
            1,
            format!(
                "oracle {:.17}, port {:.17} (192^-0.5); 128^-0.5 = {:.17}",
                gold[0],
                want,
                128f64.powf(-0.5)
            ),
        );
    }

    // C13 / C13b re-derived from the oracle arrays: where each half of the key
    // and the value actually comes from. Asserted on the ORACLE, so the port's
    // structure is justified by the vectors rather than by a reading of the
    // source.
    {
        let h = blk.cfg.num_heads;
        let nope = blk.cfg.qk_nope_head_dim;
        let dv = blk.cfg.v_head_dim;
        let qh = blk.cfg.q_head_dim();
        let carried = blk.cfg.qk_carried_head_dim;
        let kvr = blk.cfg.kv_lora_rank;
        let (kvb_key, kvb) = lane.get(&k("attn_kv_b_proj_out")); // [B,T,H*(nope+dv)]
        let (kkey, keys) = lane.get(&k("mla_key_states")); // [B,H,T,qh]
        let (vkey, vals) = lane.get(&k("mla_value_states")); // [B,H,T,dv]
        let (akey, kva) = lane.get(&k("attn_kv_a_proj_with_mqa_out")); // [B,T,kvr+carried]
        let mut k_from_kvb = Vec::with_capacity(b * h * t * nope);
        let mut v_from_kvb = Vec::with_capacity(b * h * t * dv);
        let mut k_carried = Vec::with_capacity(b * h * t * carried);
        let mut k_carried_src = Vec::with_capacity(b * h * t * carried);
        for bi in 0..b {
            for hi in 0..h {
                for ti in 0..t {
                    let base = ((bi * t + ti) * h + hi) * (nope + dv);
                    k_from_kvb.extend_from_slice(&kvb[base..base + nope]);
                    v_from_kvb.extend_from_slice(&kvb[base + nope..base + nope + dv]);
                    let kb = ((bi * h + hi) * t + ti) * qh;
                    k_carried.extend_from_slice(&keys[kb + nope..kb + qh]);
                    let ab = (bi * t + ti) * (kvr + carried) + kvr;
                    k_carried_src.extend_from_slice(&kva[ab..ab + carried]);
                }
            }
        }
        let mut k_pass = Vec::with_capacity(b * h * t * nope);
        for bi in 0..b {
            for hi in 0..h {
                for ti in 0..t {
                    let kb = ((bi * h + hi) * t + ti) * qh;
                    k_pass.extend_from_slice(&keys[kb..kb + nope]);
                }
            }
        }
        g.cmp_exact(
            lane_name,
            "oracle_key_pass_is_kv_b_first_half",
            &format!("{} vs {}", kkey, kvb_key),
            &k_pass,
            &k_from_kvb,
        );
        g.cmp_exact(
            lane_name,
            "oracle_value_is_kv_b_second_half",
            &format!("{} vs {}", vkey, kvb_key),
            &vals,
            &v_from_kvb,
        );
        g.cmp_exact(
            lane_name,
            "oracle_key_carried_is_kv_a_tail_broadcast",
            &format!("{} vs {}", kkey, akey),
            &k_carried,
            &k_carried_src,
        );
    }

    // The swapped-halves array is an INVARIANCE, not a negative control: with
    // nothing rotated it is a joint permutation of both sides of a dot product.
    // Assert it matches, and say what that does and does not prove.
    {
        let (key, gold) = lane.get(&k("mla_ALT_attn_out_swapped_halves"));
        match tol {
            Tol::Rel(t) => g.cmp(
                lane_name,
                "swapped_halves_is_an_invariance",
                &key,
                &host(&tr.attn_out_heads),
                &gold,
                t.max(1e-3),
            ),
            Tol::RefRatio(r) => {
                let rf = reference
                    .expect("reference")
                    .get("attn_out_heads")
                    .expect("reference");
                g.cmp_ref(
                    lane_name,
                    "swapped_halves_is_an_invariance",
                    &key,
                    &host(&tr.attn_out_heads),
                    &gold,
                    rf,
                    r,
                )
            }
        }
    }

    tr
}

/// Read a 3-d oracle array into a tensor of the requested backend.
fn oracle_t3<B: Backend>(lane: &Lane, name: &str, dev: &B::Device) -> (String, Tensor<B, 3>) {
    let (key, v) = lane.get(name);
    let d = lane.dims(name);
    assert_eq!(d.len(), 3, "{}: expected a [B,T,W] array, got {:?}", key, d);
    (
        key,
        Tensor::from_data(
            TensorData::new(v, [d[0], d[1], d[2]]).convert::<B::FloatElem>(),
            dev,
        ),
    )
}

/// Read a 4-d oracle array into a tensor of the requested backend.
fn oracle_t4<B: Backend>(lane: &Lane, name: &str, dev: &B::Device) -> (String, Tensor<B, 4>) {
    let (key, v) = lane.get(name);
    let d = lane.dims(name);
    assert_eq!(d.len(), 4, "{}: expected a 4-d array, got {:?}", key, d);
    (
        key,
        Tensor::from_data(
            TensorData::new(v, [d[0], d[1], d[2], d[3]]).convert::<B::FloatElem>(),
            dev,
        ),
    )
}

/// Every field of a trace, keyed by the name the sub-block checks use, so a
/// float64 run can serve as the reference for a bfloat16 cascade.
fn reference_map<B: Backend>(tr: &MlaTrace<B>) -> HashMap<&'static str, Vec<f64>> {
    let mut m: HashMap<&'static str, Vec<f64>> = HashMap::new();
    m.insert("q_a_proj_out", host(&tr.q_a_proj_out));
    m.insert("q_a_layernorm_out", host(&tr.q_a_layernorm_out));
    m.insert("q_b_proj_out", host(&tr.q_b_proj_out));
    m.insert("kv_a_proj_out", host(&tr.kv_a_proj_out));
    m.insert("kv_a_layernorm_in", host(&tr.kv_a_layernorm_in));
    m.insert("kv_a_layernorm_out", host(&tr.kv_a_layernorm_out));
    m.insert("kv_b_proj_out", host(&tr.kv_b_proj_out));
    m.insert("query_states", host(&tr.query_states));
    m.insert("key_states", host(&tr.key_states));
    m.insert("value_states", host(&tr.value_states));
    m.insert("attn_scores", host(&tr.scores));
    m.insert("attn_probs_precast", host(&tr.probs_precast));
    m.insert("attn_probs", host(&tr.probs));
    m.insert("attn_out_heads", host(&tr.attn_out_heads));
    m.insert(
        "g_proj_out",
        host(tr.g_proj_out.as_ref().expect("output gate")),
    );
    m.insert("o_proj_in", host(&tr.o_proj_in));
    m.insert("block_out", host(&tr.out));
    m
}

/// Drive each shipped operation from the **oracle's own captured input** for
/// that boundary, rather than from the previous operation's output.
///
/// This is the sharp bfloat16 lane. Nothing here compounds: with identical
/// inputs the only difference between the port and torch is the order of one
/// fp32 accumulation, so a correct port lands within one bfloat16 ULP and
/// rounds to the same bits for ~99.9% of elements. Where it does not — see
/// [`Gate::cmp_bf16_ref`] — a float64 computation of the same operation, by the
/// same code, decides which of the two runs is further from the exact answer.
///
/// The three assembly steps are pure slicing, concatenation and a broadcast —
/// no arithmetic at all — so they are required to be **bit-exact**, which is
/// also the check that pins where each half of the key and the value comes from.
fn run_teacher_forced_lane(
    g: &mut Gate,
    layer: usize,
    lane: &Lane,
    b32: &MlaBlock<F32B>,
    b64: &MlaBlock<F64B>,
    dev32: &burn_ndarray::NdArrayDevice,
    dev64: &burn_ndarray::NdArrayDevice,
) {
    let lp = format!("L{:02}_", layer);
    let k = |s: &str| format!("{}{}", lp, s);
    const MIN_BITEXACT: f64 = 0.99;

    // --- the eight projections and norms, each from its own captured input ---
    macro_rules! op3 {
        ($name:literal, $in:literal, $out:literal, $m:ident, $splitk:literal) => {{
            let (_, x32) = oracle_t3::<F32B>(lane, &k($in), dev32);
            let (_, x64) = oracle_t3::<F64B>(lane, &k($in), dev64);
            let (okey, gold) = lane.get(&k($out));
            g.cmp_bf16_ref(
                "tf",
                $name,
                &okey,
                &host(&b32.$m(x32)),
                &gold,
                &host(&b64.$m(x64)),
                MIN_BITEXACT,
                $splitk,
            );
        }};
    }
    op3!(
        "q_a_proj",
        "attn_q_a_proj_in",
        "attn_q_a_proj_out",
        q_a_proj,
        false
    ); // K=7168
    op3!(
        "q_a_layernorm",
        "attn_q_a_layernorm_in",
        "attn_q_a_layernorm_out",
        q_a_norm,
        false
    );
    op3!(
        "q_b_proj",
        "attn_q_b_proj_in",
        "attn_q_b_proj_out",
        q_b_proj,
        false
    ); // K=1536
    op3!(
        "kv_a_proj",
        "attn_kv_a_proj_with_mqa_in",
        "attn_kv_a_proj_with_mqa_out",
        kv_a_proj,
        false
    ); // K=7168
    op3!(
        "kv_a_layernorm",
        "attn_kv_a_layernorm_in",
        "attn_kv_a_layernorm_out",
        kv_a_norm,
        false
    );
    op3!(
        "kv_b_proj",
        "attn_kv_b_proj_in",
        "attn_kv_b_proj_out",
        kv_b_proj,
        false
    ); // K=512
    op3!("o_proj", "attn_o_proj_in", "attn_o_proj_out", o_proj, true); // K=12288 -- the one long reduction
    {
        let (_, x32) = oracle_t3::<F32B>(lane, &k("attn_g_proj_in"), dev32);
        let (_, x64) = oracle_t3::<F64B>(lane, &k("attn_g_proj_in"), dev64);
        let (okey, gold) = lane.get(&k("attn_g_proj_out"));
        g.cmp_bf16_ref(
            "tf",
            "g_proj",
            &okey,
            &host(&b32.g_proj(x32).expect("output gate")),
            &gold,
            &host(&b64.g_proj(x64).expect("output gate")),
            MIN_BITEXACT,
            false, // K=7168
        );
    }

    // --- the three assemblies: no arithmetic, so bit-exact -----------------
    {
        let (qb_key, qb) = oracle_t3::<F32B>(lane, &k("attn_q_b_proj_out"), dev32);
        let (q_states, _) = b32.assemble_query(qb);
        let (okey, gold) = lane.get(&k("mla_query_states"));
        g.cmp_exact(
            "tf",
            "assemble_query",
            &format!("{} -> {}", qb_key, okey),
            &host(&q_states),
            &gold,
        );

        let (kvb_key, kvb) = oracle_t3::<F32B>(lane, &k("attn_kv_b_proj_out"), dev32);
        let (kva_key, kva) = oracle_t3::<F32B>(lane, &k("attn_kv_a_proj_with_mqa_out"), dev32);
        let (k_states, v_states, _) = b32.assemble_kv(kvb, b32.kv_carried(kva));
        let (okey, gold) = lane.get(&k("mla_key_states"));
        g.cmp_exact(
            "tf",
            "assemble_key",
            &format!("{} + {} -> {}", kvb_key, kva_key, okey),
            &host(&k_states),
            &gold,
        );
        let (okey, gold) = lane.get(&k("mla_value_states"));
        g.cmp_exact(
            "tf",
            "assemble_value",
            &format!("{} -> {}", kvb_key, okey),
            &host(&v_states),
            &gold,
        );
    }

    // --- the attention core, one operation at a time -----------------------
    {
        let (_, q32) = oracle_t4::<F32B>(lane, &k("mla_query_states"), dev32);
        let (_, k32) = oracle_t4::<F32B>(lane, &k("mla_key_states"), dev32);
        let (_, m32) = oracle_t4::<F32B>(lane, &k("mla_attn_mask"), dev32);
        let (_, q64) = oracle_t4::<F64B>(lane, &k("mla_query_states"), dev64);
        let (_, k64) = oracle_t4::<F64B>(lane, &k("mla_key_states"), dev64);
        let (_, m64) = oracle_t4::<F64B>(lane, &k("mla_attn_mask"), dev64);
        let (okey, gold) = lane.get(&k("mla_attn_scores_precast"));
        g.cmp_bf16_ref(
            "tf",
            "attend_scores",
            &okey,
            &host(&b32.attn_scores(q32, k32, Some(m32))),
            &gold,
            &host(&b64.attn_scores(q64, k64, Some(m64))),
            MIN_BITEXACT,
            false, // K=192
        );

        // The softmax, from the reference's OWN scores: a one-ULP difference in
        // a score moves a probability by ~p ULP, so driving this from the port's
        // recomputed scores would measure the scores again instead of the
        // softmax.
        let (_, sc32) = oracle_t4::<F32B>(lane, &k("mla_attn_scores_precast"), dev32);
        let (_, sc64) = oracle_t4::<F64B>(lane, &k("mla_attn_scores_precast"), dev64);
        let (p32_pre, p32) = b32.attn_probs(sc32);
        let (p64_pre, p64) = b64.attn_probs(sc64);
        let (okey, gold) = lane.get(&k("mla_attn_probs_precast"));
        g.cmp(
            "tf",
            "attend_probs_precast",
            &okey,
            &host(&p32_pre),
            &gold,
            1e-5,
        );
        let _ = p64_pre;
        let (okey, gold) = lane.get(&k("mla_attn_probs"));
        g.cmp_bf16_ref(
            "tf",
            "attend_probs",
            &okey,
            &host(&p32),
            &gold,
            &host(&p64),
            MIN_BITEXACT,
            false,
        );

        let (_, pr32) = oracle_t4::<F32B>(lane, &k("mla_attn_probs"), dev32);
        let (_, v32) = oracle_t4::<F32B>(lane, &k("mla_value_states"), dev32);
        let (_, pr64) = oracle_t4::<F64B>(lane, &k("mla_attn_probs"), dev64);
        let (_, v64) = oracle_t4::<F64B>(lane, &k("mla_value_states"), dev64);
        let (okey, gold) = lane.get(&k("mla_attn_out_heads"));
        g.cmp_bf16_ref(
            "tf",
            "attend_out_heads",
            &okey,
            &host(&b32.attn_apply(pr32, v32)),
            &gold,
            &host(&b64.attn_apply(pr64, v64)),
            MIN_BITEXACT,
            false, // K=T
        );
    }

    // --- the output gate, from the oracle's attention output and g_proj ----
    {
        let (_, ao32) = oracle_t4::<F32B>(lane, &k("mla_attn_out_heads"), dev32);
        let (_, gp32) = oracle_t3::<F32B>(lane, &k("attn_g_proj_out"), dev32);
        let (_, ao64) = oracle_t4::<F64B>(lane, &k("mla_attn_out_heads"), dev64);
        let (_, gp64) = oracle_t3::<F64B>(lane, &k("attn_g_proj_out"), dev64);
        let (okey, gold) = lane.get(&k("attn_o_proj_in"));
        g.cmp_bf16_ref(
            "tf",
            "output_gate",
            &okey,
            &host(&b32.apply_output_gate(ao32, Some(&gp32))),
            &gold,
            &host(&b64.apply_output_gate(ao64, Some(&gp64))),
            MIN_BITEXACT,
            false, // elementwise, no reduction
        );
    }
}

// ---------------------------------------------------------------------------
// the positive control: what a rotation would do
// ---------------------------------------------------------------------------

/// Apply an HF-style rotary embedding to the last `carried` dims of a
/// `[B, H, T, D]` tensor laid out row-major, in place.
///
/// This lives in the gate and nowhere else. The port has no rotation function;
/// if it did, this control would be measuring the port against itself.
fn apply_rope(x: &mut [f64], b: usize, h: usize, t: usize, d: usize, carried: usize, theta: f64) {
    let half = carried / 2;
    assert!(
        half > 0 && carried <= d,
        "bad carried width {} in {}",
        carried,
        d
    );
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                let base = ((bi * h + hi) * t + ti) * d + (d - carried);
                for i in 0..half {
                    let inv = theta.powf(-(2.0 * i as f64) / carried as f64);
                    let ang = ti as f64 * inv;
                    let (s, c) = ang.sin_cos();
                    let a = x[base + i];
                    let bb = x[base + i + half];
                    x[base + i] = a * c - bb * s;
                    x[base + i + half] = bb * c + a * s;
                }
            }
        }
    }
}

/// Recompute the block from already-assembled q/k/v, in plain f64 — no Burn, no
/// port code. Returns the per-head attention output `[B, T, H, dv]`.
#[allow(clippy::too_many_arguments)]
fn attention_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    mask: &[f64],
    b: usize,
    h: usize,
    t: usize,
    qh: usize,
    dv: usize,
    scaling: f64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; b * t * h * dv];
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..t {
                let mut sc = vec![0.0f64; t];
                for ki in 0..=qi {
                    let qb = ((bi * h + hi) * t + qi) * qh;
                    let kb = ((bi * h + hi) * t + ki) * qh;
                    let mut s = 0.0;
                    for d in 0..qh {
                        s += q[qb + d] * k[kb + d];
                    }
                    sc[ki] = s * scaling + mask[(bi * t + qi) * t + ki];
                }
                for ki in (qi + 1)..t {
                    sc[ki] = mask[(bi * t + qi) * t + ki];
                }
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for ki in 0..t {
                    let p = e[ki] / z;
                    if p == 0.0 {
                        continue;
                    }
                    let vb = ((bi * h + hi) * t + ki) * dv;
                    let ob = ((bi * t + qi) * h + hi) * dv;
                    for d in 0..dv {
                        out[ob + d] += p * v[vb + d];
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------

fn sha256_file(p: &Path) -> String {
    let mut f = std::fs::File::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut hasher = Sha256::new();
    std::io::copy(&mut f, &mut hasher).expect("hash");
    format!("{:x}", hasher.finalize())
}

fn arg(name: &str, default: &str) -> String {
    arg_opt(name).unwrap_or_else(|| default.to_string())
}

/// The optional form of [`arg`]: absent means "not given", which is not the
/// same as a guessed default. Model paths use this one.
fn arg_opt(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let ckpt_dir =
        mary::paths::model(arg_opt("--ckpt").as_deref(), "kimi-k3").unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
    let vec_dir =
        mary::paths::model(arg_opt("--vectors").as_deref(), "k3-oracle").unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
    let json_out = arg("--json", "");
    let t_start = Instant::now();
    let mut g = Gate::default();

    // ---- 1. the oracle is the oracle ------------------------------------
    let p13 = vec_dir.join("layer_oracle_prefix13_bf16.npz");
    let plad = vec_dir.join("layer_oracle_ladder.npz");
    for (p, want) in [(&p13, SHA_PREFIX13), (&plad, SHA_LADDER)] {
        let got = sha256_file(p);
        g.fact(
            "setup",
            &format!("sha256_{}", p.file_name().unwrap().to_string_lossy()),
            "MANIFEST_layer_oracle.md",
            got == want,
            1,
            format!("{} (want {})", got, want),
        );
    }
    println!(
        "[setup] oracle sha256 checked ({:.1}s)",
        t_start.elapsed().as_secs_f64()
    );

    // ---- 2. the config, and the layer-index base trap --------------------
    let cfg_json = std::fs::read_to_string(ckpt_dir.join("config.json")).expect("config.json");
    let k3 = K3Config::from_json(&cfg_json).expect("parse config.json");
    let text = &k3.text_config;

    // The premise handed to this port said the MLA layers are AT
    // full_attn_layers. They are at full_attn_layers - 1. Re-derived here from
    // the checkpoint's own tensor names, not from the config: a layer is MLA
    // iff it carries kv_a_proj_with_mqa.
    let ck = Ckpt::open(&ckpt_dir);
    let mut mla_by_name: Vec<usize> = (0..text.num_hidden_layers)
        .filter(|l| {
            ck.map.contains_key(&format!(
                "language_model.model.layers.{}.self_attn.kv_a_proj_with_mqa.weight",
                l
            ))
        })
        .collect();
    mla_by_name.sort_unstable();
    let mla_by_cfg: Vec<usize> = (0..text.num_hidden_layers)
        .filter(|l| text.attn_kind(*l) == AttnKind::Mla)
        .collect();
    let naive: Vec<usize> = text.linear_attn_config.full_attn_layers.clone();
    g.fact(
        "setup",
        "mla_layers_are_full_attn_layers_minus_one",
        "checkpoint tensor names vs config.json",
        mla_by_name == mla_by_cfg && mla_by_name != naive && mla_by_name.len() == 24,
        mla_by_name.len(),
        format!(
            "by tensor name {:?}..., by config {:?}..., the 1-based list would say {:?}...",
            &mla_by_name[..4.min(mla_by_name.len())],
            &mla_by_cfg[..4.min(mla_by_cfg.len())],
            &naive[..4.min(naive.len())]
        ),
    );
    g.fact(
        "setup",
        "gated_layers_are_mla",
        "config.attn_kind",
        MLA_LAYERS
            .iter()
            .all(|l| text.attn_kind(*l) == AttnKind::Mla)
            && text.attn_kind(4) == AttnKind::Kda,
        MLA_LAYERS.len(),
        "layers 3/7/11 are MLA and layer 4 is KDA".into(),
    );

    let mcfg = MlaConfig::from_text_config(text).expect("MLA config");
    g.fact(
        "setup",
        "config_matches_checkpoint",
        "config.json",
        mcfg.hidden_size == 7168
            && mcfg.num_heads == 96
            && mcfg.q_lora_rank == Some(1536)
            && mcfg.kv_lora_rank == 512
            && mcfg.qk_nope_head_dim == 128
            && mcfg.qk_carried_head_dim == 64
            && mcfg.v_head_dim == 128
            && mcfg.q_head_dim() == 192
            && mcfg.use_output_gate,
        9,
        format!("{:?}", mcfg),
    );

    // The structural half of the no-rotation claim: there is no rotated block
    // for a config edit to select.
    let mut not_nope = text.clone();
    not_nope.mla_use_nope = false;
    g.fact(
        "setup",
        "non_nope_config_is_refused",
        "MlaConfig::from_text_config",
        MlaConfig::from_text_config(&not_nope).is_err()
            && MlaConfig::from_text_config(text).is_ok(),
        1,
        "a config with mla_use_nope=false cannot build this block".into(),
    );

    // ---- 3. load the oracles --------------------------------------------
    let z13 = Npz::open(&p13).expect("prefix13 npz");
    let zlad = Npz::open(&plad).expect("ladder npz");
    println!(
        "[setup] oracles loaded: {} + {} arrays ({:.1}s)",
        z13.len(),
        zlad.len(),
        t_start.elapsed().as_secs_f64()
    );
    assert_eq!(g.checks.len(), N_CHECKS_SETUP, "setup check count drifted");

    let dev64: burn_ndarray::NdArrayDevice = Default::default();
    let dev32: burn_ndarray::NdArrayDevice = Default::default();

    // ---- 4. lane f64 — the sharp one -------------------------------------
    let mark = g.checks.len();
    let w64 = load_weights::<F64B>(&ck, &mcfg, 3, &dev64, &mut g, true);
    let blk64 = MlaBlock::new(mcfg.clone(), w64, Precision::Exact);
    let lane64 = Lane {
        z: &zlad,
        prefix: "mla_L03_f64pure__".into(),
        bits: false,
    };
    let tr64 = run_subblock_lane(
        &mut g,
        "f64",
        3,
        &lane64,
        &blk64,
        &dev64,
        Tol::Rel(1e-11),
        None,
    );
    assert_eq!(
        g.n_since(mark),
        N_CHECKS_PER_SUBBLOCK_LANE + N_CHECKS_WEIGHT_SHAPE,
        "f64 lane produced the wrong number of checks"
    );
    println!("[f64 ] lane done ({:.1}s)", t_start.elapsed().as_secs_f64());

    // ---- 5. the positive control: rotation would break it ----------------
    {
        let mark = g.checks.len();
        let [b, h, t, qh] = tr64.query_states.dims();
        let dv = mcfg.v_head_dim;
        let carried = mcfg.qk_carried_head_dim;
        let mut q = host(&tr64.query_states);
        let mut kk = host(&tr64.key_states);
        let v = host(&tr64.value_states);
        let mask = MlaBlock::<F64B>::causal_mask(b, t, t, 0, &dev64);
        let mask_v: Vec<f64> = host(&mask);

        // Sanity: the unrotated recomputation reproduces the port, so the
        // control's own arithmetic is not the thing that moved.
        let plain = attention_f64(&q, &kk, &v, &mask_v, b, h, t, qh, dv, mcfg.scaling());
        let (okey, gold) = lane64.get("L03_mla_attn_out_heads");
        g.cmp(
            "control",
            "independent_f64_attention_reproduces_oracle",
            &okey,
            &plain,
            &gold,
            1e-11,
        );

        apply_rope(&mut q, b, h, t, qh, carried, 10000.0);
        apply_rope(&mut kk, b, h, t, qh, carried, 10000.0);
        let rotated = attention_f64(&q, &kk, &v, &mask_v, b, h, t, qh, dv, mcfg.scaling());
        g.expect_differ(
            "control",
            "rope_would_break_it",
            &okey,
            &rotated,
            &gold,
            0.05,
            "same code path, RoPE applied to the carried lanes of q and k only",
        );
        assert_eq!(
            g.n_since(mark),
            N_CHECKS_CONTROL,
            "control lane check count drifted"
        );
        println!(
            "[ctrl] positive control done ({:.1}s)",
            t_start.elapsed().as_secs_f64()
        );
    }

    // ---- 6. lane f32 ------------------------------------------------------
    let mark = g.checks.len();
    let w32_l3 = load_weights::<F32B>(&ck, &mcfg, 3, &dev32, &mut g, false);
    let blk32 = MlaBlock::new(mcfg.clone(), w32_l3.clone(), Precision::Exact);
    let lane32 = Lane {
        z: &zlad,
        prefix: "mla_L03_f32__".into(),
        bits: false,
    };
    run_subblock_lane(
        &mut g,
        "f32",
        3,
        &lane32,
        &blk32,
        &dev32,
        Tol::Rel(1e-4),
        None,
    );
    assert_eq!(
        g.n_since(mark),
        N_CHECKS_PER_SUBBLOCK_LANE,
        "f32 lane check count drifted"
    );
    println!("[f32 ] lane done ({:.1}s)", t_start.elapsed().as_secs_f64());

    // ---- 7. teacher-forced bfloat16, all three MLA layers -----------------
    let lane13 = Lane {
        z: &z13,
        prefix: String::new(),
        bits: true,
    };
    let mut blk_bf16_l3: Option<MlaBlock<F32B>> = None;
    for layer in MLA_LAYERS {
        let mark = g.checks.len();
        let w = if layer == 3 {
            w32_l3.clone()
        } else {
            load_weights::<F32B>(&ck, &mcfg, layer, &dev32, &mut g, true)
        };
        let blk = MlaBlock::new(mcfg.clone(), w, Precision::Bf16);
        // The float64 arbiter for this layer: the same code, the same weights,
        // in a width where bfloat16 rounding is invisible.
        let blk64 = MlaBlock::new(
            mcfg.clone(),
            load_weights::<F64B>(&ck, &mcfg, layer, &dev64, &mut g, false),
            Precision::Exact,
        );
        run_teacher_forced_lane(&mut g, layer, &lane13, &blk, &blk64, &dev32, &dev64);
        let extra = if layer == 3 { 0 } else { N_CHECKS_WEIGHT_SHAPE };
        assert_eq!(
            g.n_since(mark),
            N_CHECKS_TEACHER_FORCED + extra,
            "teacher-forced lane L{:02} check count drifted",
            layer
        );
        if layer == 3 {
            blk_bf16_l3 = Some(blk);
        }
        println!(
            "[tf  ] L{:02} done ({:.1}s)",
            layer,
            t_start.elapsed().as_secs_f64()
        );
    }
    let blk_bf16 = blk_bf16_l3.expect("layer 3 bf16 block");

    // ---- 7b. the bfloat16 cascade, against a float64 run of the same input -
    //
    // Layer 3 only: the wiring is layer-independent code, and it is already
    // pinned exactly by the float64 lane. What this adds is that the *bfloat16
    // rounding placement* does not degrade the cascade — measured against a
    // float64 run this gate computes from the same input and the same weights,
    // so the criterion is a ratio of two error magnitudes rather than a
    // tolerance anyone chose.
    let mark = g.checks.len();
    let (_, hv13) = lane13.get("L03_attn_q_a_proj_in");
    let d13 = lane13.dims("L03_attn_q_a_proj_in");
    let blk64_full = MlaBlock::new(
        mcfg.clone(),
        load_weights::<F64B>(&ck, &mcfg, 3, &dev64, &mut g, false),
        Precision::Exact,
    );
    let hid64: Tensor<F64B, 3> = Tensor::from_data(
        TensorData::new(hv13, [d13[0], d13[1], d13[2]]).convert::<f64>(),
        &dev64,
    );
    let tr64_full = blk64_full.forward(
        hid64,
        Some(MlaBlock::<F64B>::causal_mask(
            d13[0], d13[1], d13[1], 0, &dev64,
        )),
        None,
    );
    let reference = reference_map(&tr64_full);
    println!(
        "[ref ] float64 reference for the bf16 cascade ({:.1}s)",
        t_start.elapsed().as_secs_f64()
    );
    run_subblock_lane(
        &mut g,
        "bf16",
        3,
        &lane13,
        &blk_bf16,
        &dev32,
        Tol::RefRatio(4.0),
        Some(&reference),
    );
    assert_eq!(
        g.n_since(mark),
        N_CHECKS_PER_SUBBLOCK_LANE,
        "bf16 cascade check count drifted"
    );
    println!(
        "[bf16] cascade done ({:.1}s)",
        t_start.elapsed().as_secs_f64()
    );

    // ---- 8. lane cache: prefill 12, then continue 4, then decode 4 --------
    {
        let mark = g.checks.len();
        let lane = Lane {
            z: &z13,
            prefix: String::new(),
            bits: true,
        };
        let (in_key, hidden_v) = lane.get("L03_attn_q_a_proj_in");
        let dims = lane.dims("L03_attn_q_a_proj_in");
        let (t, dh) = (dims[1], dims[2]);
        // batch 0 only — the cache lane in the oracle used hs[0:1].
        let b0: Vec<f64> = hidden_v[..t * dh].to_vec();
        let hid = |from: usize, to: usize| -> Tensor<F32B, 3> {
            let v = b0[from * dh..to * dh].to_vec();
            Tensor::from_data(
                TensorData::new(v, [1, to - from, dh]).convert::<f32>(),
                &dev32,
            )
        };
        let (out_key, out_full) = lane.get("L03_attn_o_proj_out");
        let want = |from: usize, to: usize| -> Vec<f64> { out_full[from * dh..to * dh].to_vec() };

        let mut cache = MlaKvCache::<F32B>::new();
        let pre = blk_bf16.forward(
            hid(0, 12),
            Some(MlaBlock::<F32B>::causal_mask(1, 12, 12, 0, &dev32)),
            Some(&mut cache),
        );
        g.fact(
            "cache",
            "prefill_cache_length",
            "port-internal",
            cache.len() == 12,
            1,
            format!(
                "cache holds {} tokens after a 12-token prefill",
                cache.len()
            ),
        );
        // The KV-cache arrays live in the ladder bundle, not the prefix run.
        let clane = Lane {
            z: &zlad,
            prefix: String::new(),
            bits: true,
        };
        let (kkey, kgold) = clane.get("mla_L03_cache_prefill12_key");
        let (vkey, vgold) = clane.get("mla_L03_cache_prefill12_value");
        // Reference: the float64 run of the full 16 tokens, sliced to batch 0.
        // Attention is causal and every other operation is per token, so tokens
        // 0..12 of a 16-token run ARE what a 12-token run computes — that is
        // what the prefill/continue split is testing in the first place, and it
        // is checked, not assumed, by the fact that the continue-4 output is
        // compared against the full run's tokens 12..16.
        let [_, hh, _, qh] = tr64_full.key_states.dims();
        let slice_b0 = |t: &Tensor<F64B, 4>, t0: usize, t1: usize, w: usize| -> Vec<f64> {
            host(&t.clone().slice([0..1, 0..hh, t0..t1, 0..w]))
        };
        let out64 = host(&tr64_full.out);
        g.cmp_ref(
            "cache",
            "prefill12_key_cache",
            &kkey,
            &host(&cache.key().unwrap()),
            &kgold,
            &slice_b0(&tr64_full.key_states, 0, 12, qh),
            4.0,
        );
        g.cmp_ref(
            "cache",
            "prefill12_value_cache",
            &vkey,
            &host(&cache.value().unwrap()),
            &vgold,
            &slice_b0(&tr64_full.value_states, 0, 12, mcfg.v_head_dim),
            4.0,
        );
        g.cmp_ref(
            "cache",
            "prefill12_out",
            &format!("{}[0,0:12]", out_key),
            &host(&pre.out),
            &want(0, 12),
            &out64[..12 * dh],
            4.0,
        );

        let cont = blk_bf16.forward(
            hid(12, 16),
            Some(MlaBlock::<F32B>::causal_mask(1, 4, 16, 12, &dev32)),
            Some(&mut cache),
        );
        g.cmp_ref(
            "cache",
            "continue4_out",
            &format!("{}[0,12:16]", out_key),
            &host(&cont.out),
            &want(12, 16),
            &out64[12 * dh..16 * dh],
            4.0,
        );

        let mut c2 = MlaKvCache::<F32B>::new();
        let _ = blk_bf16.forward(
            hid(0, 12),
            Some(MlaBlock::<F32B>::causal_mask(1, 12, 12, 0, &dev32)),
            Some(&mut c2),
        );
        let mut steps: Vec<f64> = Vec::new();
        for ti in 12..16 {
            let s = blk_bf16.forward(
                hid(ti, ti + 1),
                Some(MlaBlock::<F32B>::causal_mask(1, 1, ti + 1, ti, &dev32)),
                Some(&mut c2),
            );
            steps.extend(host(&s.out));
        }
        g.cmp_ref(
            "cache",
            "stepwise4_out",
            &format!("{}[0,12:16]", out_key),
            &steps,
            &want(12, 16),
            &out64[12 * dh..16 * dh],
            4.0,
        );
        assert_eq!(
            g.n_since(mark),
            N_CHECKS_CACHE,
            "cache lane check count drifted"
        );
        let _ = in_key;
        println!("[cach] lane done ({:.1}s)", t_start.elapsed().as_secs_f64());
    }

    // ---- 9. verdict -------------------------------------------------------
    // Three cascade lanes (f64, f32, bf16), three teacher-forced lanes
    // (L03/L07/L11), one weight-shape fact per distinct layer loaded, the
    // control pair, the cache lane, setup.
    let expected = N_CHECKS_SETUP
        + N_CHECKS_CONTROL
        + N_CHECKS_CACHE
        + N_CHECKS_PER_SUBBLOCK_LANE * 3      // f64, f32, bf16 cascade
        + N_CHECKS_TEACHER_FORCED * 3         // bf16 per-operation, L03/L07/L11
        + N_CHECKS_WEIGHT_SHAPE * 3; // one per distinct layer loaded
    assert_eq!(
        g.checks.len(),
        expected,
        "check-count drift: {} recorded, {} expected — a lane was skipped or duplicated",
        g.checks.len(),
        expected
    );

    println!(
        "\n{:<6} {:<44} {:>10} {:>11} {:>11} {:>9} {:>6}",
        "lane", "check", "n", "max|d|", "rel", "bitexact", "ok"
    );
    for c in &g.checks {
        println!(
            "{:<6} {:<44} {:>10} {:>11.3e} {:>11.3e} {:>9} {:>6}",
            c.lane,
            c.name,
            c.n,
            c.max_abs,
            c.rel,
            c.bitexact
                .map(|f| format!("{:.5}", f))
                .unwrap_or_else(|| "-".into()),
            if c.ok { "PASS" } else { "FAIL" }
        );
    }

    let failed = g.failed();
    println!(
        "\n{} checks, {} failed, {:.1}s",
        g.checks.len(),
        failed.len(),
        t_start.elapsed().as_secs_f64()
    );
    if !json_out.is_empty() {
        std::fs::write(&json_out, serde_json::to_string_pretty(&g.checks).unwrap())
            .expect("write json");
        println!("wrote {}", json_out);
    }
    if !failed.is_empty() {
        for c in &failed {
            println!(
                "FAIL {}/{} vs {}: max|d| {:.6e} rel {:.6e} tol {:.6e} bitexact {:?} {}",
                c.lane, c.name, c.oracle, c.max_abs, c.rel, c.tol, c.bitexact, c.detail
            );
        }
        println!("\nGATE FAILED — no performance number is reported.");
        std::process::exit(1);
    }

    // Only now, after every check has passed, is a timing number meaningful.
    let lane = Lane {
        z: &z13,
        prefix: String::new(),
        bits: true,
    };
    let (_, hv) = lane.get("L03_attn_q_a_proj_in");
    let d = lane.dims("L03_attn_q_a_proj_in");
    let hidden: Tensor<F32B, 3> = Tensor::from_data(
        TensorData::new(hv, [d[0], d[1], d[2]]).convert::<f32>(),
        &dev32,
    );
    let mask = MlaBlock::<F32B>::causal_mask(d[0], d[1], d[1], 0, &dev32);
    let blk_plain = MlaBlock::new(mcfg.clone(), blk_bf16.w.clone(), Precision::Exact);
    let t0 = Instant::now();
    let out = blk_plain.forward(hidden, Some(mask), None);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "\nGATE PASSED: {} checks.\nforward [{},{},{}] on NdArray/f32 CPU: {:.0} ms \
         (a CPU reference lane, not an inference claim); out absmax {:.4}",
        g.checks.len(),
        d[0],
        d[1],
        d[2],
        ms,
        absmax(&host(&out.out))
    );
}
