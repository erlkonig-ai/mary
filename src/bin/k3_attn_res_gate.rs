//! k3_attn_res_gate — the correctness gate for `mary::models::k3::attn_res`.
//!
//! Three independent artifacts are read, and the gate's power comes from
//! making them agree with each other rather than from any one of them:
//!
//! 1. **The checkpoint**, unmodified and read-only: `config.json` for the
//!    boundary period and `model.safetensors.index.json` + the shard headers
//!    for the AttnRes parameter tensors. This is where the *weights* come from
//!    — not from the oracle — so a port that reads the wrong tensor name, or
//!    the right names at the wrong ranks, dies here rather than being carried
//!    by a pre-multiplied array someone else computed.
//! 2. **The whole-layer oracle** `layer_oracle_prefix13_bf16.npz`: 13 real
//!    decoder layers of Kimi K3 driven through the checkpoint's own
//!    `KimiLinearModel.forward` on real token ids. Every `*_in`/`*_out` array
//!    is a forward-hook capture taken while the shipped forward ran.
//! 3. **This port**, run on every backend compiled in.
//!
//! ## What each section proves
//!
//! * `boundary` — the reset schedule, from three directions that can disagree:
//!   the config predicate, the oracle's own per-layer bank-size deltas, and the
//!   checkpoint's tensor inventory. An off-by-one in which layers snapshot is
//!   the failure this primitive exists to prevent: the model still runs and
//!   still emits fluent-looking garbage. The 13-layer window contains **two**
//!   boundaries (0 and 12) with eleven non-boundaries between them, which is
//!   exactly what a ±1 shift disturbs — and the gate measures that the shifted
//!   schedules really do differ inside the window, so the check is known to be
//!   able to fail before it is trusted for passing.
//! * `rounding` — `round_bf16` against an independent bit-level
//!   round-to-nearest-even, on the backend, plus where its domain ends at both
//!   ends and how far that is from the data.
//! * `weights` — `AttnResParams` built from the checkpoint's own bytes against
//!   the oracle's captured `score_weight`, **bit-exactly**, at all 26 sites.
//! * `sites` — every mixture call site teacher-forced from the oracle's own
//!   `block_residual` / `prefix_sum`: the candidate stack, the scores, the
//!   probabilities and the output, each asserted separately. A gate that only
//!   compared outputs could not tell a wrong normalisation from a wrong
//!   mixture; these are different arrays and are compared as such.
//! * `controls` — the stored negative controls, each first shown to be
//!   genuinely different from the truth (a control that coincides with the
//!   right answer is not evidence), then shown not to match this port. Plus
//!   the two *invariances* the oracle's manifest reports, which are
//!   deliberately **not** used as discriminators and are labelled as such.
//! * `chain` — the depth state machine run across all 13 layers, chained: the
//!   bank and the accumulator carry from layer to layer through this port's
//!   own code, and layer *n+1*'s input is this port's layer-*n* output. Only
//!   the sublayer outputs (attention, MLP/MoE) come from the oracle, because
//!   this gate ports AttnRes and not MLA/KDA/MoE. Said plainly: **the depth
//!   axis is chained, the sublayers are teacher-forced.**
//!
//! Run:
//!   `cargo run --release --features k3-attn-res --bin k3_attn_res_gate`
//! adding `--features k3-attn-res-cuda` for the GB10 lane. Paths default to
//! the Spark's; override with `K3_CHECKPOINT` and `K3_LAYER_ORACLE`.

use std::collections::{BTreeSet, HashMap};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use memmap2::Mmap;
use safetensors::SafeTensors;

use mary::models::k3::attn_res::{AttnResParams, DepthMixer, round_bf16, stack_candidates};
use mary::models::k3::layout::LayerPart;
use mary::models::k3::{Dtype, K3Config, Slot, describe};
use mary::nn::npz::Npz;

// ---------------------------------------------------------------------------
// Shape of the oracle run
// ---------------------------------------------------------------------------

/// `batch 2 x seq 16`, flattened — every AttnRes array is per token.
const TOKENS: usize = 32;
/// `hidden_size`. Asserted against the config and against every array's own
/// length, never assumed.
const HIDDEN: usize = 7168;
/// Decoder layers the oracle captured.
const LAYERS: usize = 13;

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// bfloat16 outputs are compared as `max |d| / max |reference|`, in units of
/// the array's own top-of-range bfloat16 step (`2^-8`, 0.39%).
///
/// A per-element ulp budget looks stricter and is in fact meaningless here: a
/// mixture output has elements arbitrarily close to zero, where any absolute
/// disagreement is thousands of ulp. Measured, torch's own f32 pass sits
/// **17 bfloat16 ulp** from a float64 transcription of its own formula on this
/// data — so a 1-ulp element-wise budget would reject the reference
/// implementation. The element-wise distance is still reported, as a
/// diagnostic; it is not a criterion.
///
/// The budget is `2^-8` — exactly one step at the top of the array, the finest
/// distinction the stored dtype can make there. Measured, this port and the
/// shipped module sit at **1.04e-3**, a quarter of it, and the pre-rounding
/// quantities they are made of (`scores`, `probs`) agree four decades tighter
/// under their own budgets below. Widening past one step would stop the
/// criterion meaning anything, since a whole step is a different bfloat16.
const OUT_REL: f64 = 0.003_906_25;

/// How much closer a reproduction of a stored negative control must be to that
/// control than the truth is, before the gate accepts that the control is the
/// specific mis-port it is named after. A ratio, so it does not depend on which
/// array's scale the distances are taken against.
const CONTROL_REPRODUCE_RATIO: f64 = 10.0;

/// f32 intermediates (`scores`) — relative to the largest magnitude in the
/// reference array. The score path reduces 7168 f32 products; torch does that
/// on the GPU with a tree reduction and this does it however the backend
/// pleases, so a handful of ulp of disagreement is the arithmetic, not the
/// port. 1e-5 is ~84 f32 eps.
const SCORE_RTOL: f64 = 1e-5;

/// Probabilities are in `[0, 1]` and are compared absolutely; a softmax over
/// ≤ 9 slots has no accumulation to speak of.
const PROB_ATOL: f64 = 1e-6;

/// How far a *negative control* must sit from the truth before the gate will
/// accept it as a control at all, relative to the array's own scale. Below
/// this, "the port does not match the control" is a statement about float
/// noise rather than about the port.
const CONTROL_MIN_REL: f64 = 1e-2;

/// The slot-order **invariance** budget, relative. The oracle's manifest
/// reports ≤ 5e-3 relative across its call sites for reordering the candidate
/// stack; this port must exhibit the same insensitivity, because it is a
/// property of the mathematics (softmax is permutation-equivariant and the
/// matmul sums over the slots) rather than of either implementation.
const INVARIANCE_REL: f64 = 5e-3;

// ---------------------------------------------------------------------------
// Metrics — NaN-poisoned rather than NaN-blind, and empty-refusing
// ---------------------------------------------------------------------------

/// Keep the worse of two error figures, so that a NaN once seen is never
/// overwritten by a later finite number. `a.max(b)` does the opposite.
fn worse(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b > a {
        b
    } else {
        a
    }
}

/// The smaller of two "gap" figures, NaN-poisoned the same way.
fn smaller(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b < a {
        b
    } else {
        a
    }
}

/// Max `|a - b|`. Empty inputs are a hard error: a zero-length comparison is a
/// green measurement of nothing. A NaN anywhere poisons the result to NaN, and
/// every threshold is written `!(d <= max)` so NaN fails.
fn max_abs_diff(what: &str, a: &[f32], b: &[f32]) -> f64 {
    assert!(!a.is_empty(), "{what}: comparison over an EMPTY array");
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: length {} vs {}",
        a.len(),
        b.len()
    );
    let mut m = 0.0f64;
    let mut nan = false;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64) - (*y as f64);
        if d.is_nan() {
            nan = true;
        } else if d.abs() > m {
            m = d.abs();
        }
    }
    if nan { f64::NAN } else { m }
}

/// Largest magnitude in an array — the scale a relative budget is taken against.
fn max_abs(what: &str, a: &[f32]) -> f64 {
    assert!(!a.is_empty(), "{what}: max over an EMPTY array");
    let mut m = 0.0f64;
    let mut nan = false;
    for x in a {
        let v = (*x as f64).abs();
        if v.is_nan() {
            nan = true;
        } else if v > m {
            m = v;
        }
    }
    if nan { f64::NAN } else { m }
}

/// Smallest non-zero magnitude — the low end of the rounding domain check.
fn min_nonzero_abs(a: &[f32]) -> f64 {
    let mut m = f64::INFINITY;
    for x in a {
        let v = (*x as f64).abs();
        if v > 0.0 && v < m {
            m = v;
        }
    }
    m
}

/// Relative max difference, against the reference's own scale.
fn max_rel_diff(what: &str, got: &[f32], want: &[f32]) -> f64 {
    let scale = max_abs(what, want);
    let d = max_abs_diff(what, got, want);
    if scale == 0.0 {
        // An all-zero reference has no relative scale; report the absolute
        // difference rather than dividing by zero into a silent infinity.
        return d;
    }
    d / scale
}

/// bfloat16 bit pattern of an f32, round-to-nearest-even.
///
/// Written from the IEEE rules — integer bit manipulation, no float tricks —
/// so that it is genuinely a second opinion on `attn_res::round_bf16`, which
/// is a float-arithmetic identity. If both were the same idea, agreeing would
/// prove nothing.
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x0040;
    }
    let lsb = (b >> 16) & 1;
    let rounded = b.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Map a bfloat16 bit pattern to a monotone integer, so that "one ulp apart"
/// is "adjacent integers" across the sign boundary as well as within a sign.
fn bf16_key(b: u16) -> i32 {
    let mag = (b & 0x7FFF) as i32;
    if b & 0x8000 != 0 { -mag } else { mag }
}

/// Max distance in bfloat16 ulp, and how many elements are not bit-identical.
/// NaN in either array poisons the distance to NaN.
fn max_ulp_bf16(what: &str, got: &[f32], want: &[f32]) -> (f64, usize) {
    assert!(
        !got.is_empty(),
        "{what}: ulp comparison over an EMPTY array"
    );
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut top = 0i32;
    let mut differing = 0usize;
    let mut nan = false;
    for (g, w) in got.iter().zip(want) {
        if g.is_nan() || w.is_nan() {
            nan = true;
            continue;
        }
        let (gb, wb) = (bf16_bits(*g), bf16_bits(*w));
        if gb != wb {
            differing += 1;
            let d = (bf16_key(gb) - bf16_key(wb)).abs();
            if d > top {
                top = d;
            }
        }
    }
    (if nan { f64::NAN } else { top as f64 }, differing)
}

/// Whether every element is exactly a bfloat16 value — i.e. rounding it again
/// changes nothing. Used to assert that the port rounded at all.
fn all_exact_bf16(a: &[f32]) -> bool {
    assert!(!a.is_empty(), "bf16-exactness over an EMPTY array");
    a.iter()
        .all(|x| bf16_to_f32(bf16_bits(*x)).to_bits() == x.to_bits())
}

/// Bit-for-bit equality of two f32 arrays.
fn bits_equal(what: &str, a: &[f32], b: &[f32]) -> bool {
    assert!(!a.is_empty(), "{what}: bit comparison over an EMPTY array");
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: length {} vs {}",
        a.len(),
        b.len()
    );
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The mixture, transcribed once more in float64 on the host.
///
/// This is not a second opinion on whether the port is *right* — it is the same
/// formula and would agree with a wrong reading of the source just as happily.
/// Its job is to split the port-vs-oracle difference into the part that is
/// arithmetic and the part that is not: both sides are f32 (torch's on the GPU,
/// the backend's here), and neither is the true value. Without it, a lane whose
/// reduction sums 7168 terms less carefully than torch's looks exactly like a
/// port that computes the wrong function.
fn mix_f64(
    v: &[f32],
    sw: &[f32],
    eps: f64,
    tokens: usize,
    slots: usize,
    hidden: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f32>) {
    assert_eq!(v.len(), tokens * slots * hidden, "mix_f64: v size");
    assert_eq!(sw.len(), hidden, "mix_f64: score weight size");
    let mut scores = vec![0.0f64; tokens * slots];
    let mut probs = vec![0.0f64; tokens * slots];
    let mut out = vec![0.0f32; tokens * hidden];
    for t in 0..tokens {
        for s in 0..slots {
            let row = &v[(t * slots + s) * hidden..(t * slots + s + 1) * hidden];
            let mut sq = 0.0f64;
            for x in row {
                sq += (*x as f64) * (*x as f64);
            }
            let scale = 1.0 / (sq / hidden as f64 + eps).sqrt();
            let mut acc = 0.0f64;
            for (x, w) in row.iter().zip(sw) {
                acc += (*x as f64) * scale * (*w as f64);
            }
            scores[t * slots + s] = acc;
        }
        let m = (0..slots)
            .map(|s| scores[t * slots + s])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut z = 0.0f64;
        for s in 0..slots {
            let e = (scores[t * slots + s] - m).exp();
            probs[t * slots + s] = e;
            z += e;
        }
        for s in 0..slots {
            probs[t * slots + s] /= z;
        }
        for h in 0..hidden {
            let mut acc = 0.0f64;
            for s in 0..slots {
                acc += probs[t * slots + s] * (v[(t * slots + s) * hidden + h] as f64);
            }
            out[t * hidden + h] = bf16_to_f32(bf16_bits(acc as f32));
        }
    }
    (scores, probs, out)
}

/// Run a closure with panic reporting suppressed — for the places the gate
/// deliberately trips an assertion and catches it.
fn expect_panic(f: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    expect_panic_saying(f, "").0
}

/// The same, but requiring the panic to say a specific thing.
///
/// Needed because "it panicked" is a weak claim about a guard: a malformed
/// tensor usually trips *something* eventually. A `[2, hidden]` projection is
/// rejected by `reshape` even with the row assertion deleted — same exit code,
/// unrecognisable message. Requiring the message is what makes the guard's
/// diagnostic, which is the only thing it contributes, observable at all.
fn expect_panic_saying(f: impl FnOnce() + std::panic::UnwindSafe, needle: &str) -> (bool, String) {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let r = panic::catch_unwind(f);
    panic::set_hook(prev);
    match r {
        Ok(()) => (false, "did not panic".to_string()),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic>".to_string());
            let first = msg.lines().next().unwrap_or("").to_string();
            (msg.contains(needle), first)
        }
    }
}

// ---------------------------------------------------------------------------
// The recorder
// ---------------------------------------------------------------------------

struct Gate {
    lane: String,
    passed: usize,
    failures: Vec<String>,
    verbose: bool,
}

impl Gate {
    fn new(lane: &str, verbose: bool) -> Self {
        Self {
            lane: lane.to_string(),
            passed: 0,
            failures: Vec::new(),
            verbose,
        }
    }

    fn record(&mut self, id: &str, ok: bool, detail: String) {
        if ok {
            self.passed += 1;
            if self.verbose {
                println!("    ok   {id:<46} {detail}");
            }
        } else {
            println!("    FAIL {id:<46} {detail}");
            self.failures.push(format!("{id}: {detail}"));
        }
    }

    /// `got <= max`, written so that NaN fails.
    fn le(&mut self, id: &str, got: f64, max: f64, unit: &str) {
        let ok = !(got > max) && !got.is_nan();
        self.record(id, ok, format!("{got:.4e} {unit} (budget {max:.4e})"));
    }

    /// `got > min` — for non-vacuity claims, where a small number is the
    /// failure. NaN fails too.
    fn gt(&mut self, id: &str, got: f64, min: f64, unit: &str) {
        let ok = got > min;
        self.record(id, ok, format!("{got:.4e} {unit} (must exceed {min:.4e})"));
    }

    fn truth(&mut self, id: &str, ok: bool, detail: impl Into<String>) {
        self.record(id, ok, detail.into());
    }

    fn passed_all(&self) -> bool {
        self.failures.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Loading: the checkpoint
// ---------------------------------------------------------------------------

/// The shard index plus a cache of mmaps, so 54 small reads do not re-map 14
/// shards of 17 GB each.
struct Checkpoint {
    dir: PathBuf,
    index: HashMap<String, String>,
    maps: HashMap<String, Mmap>,
    cfg: K3Config,
}

impl Checkpoint {
    fn open(dir: &Path) -> Result<Self> {
        let cfg_txt = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("reading {}/config.json", dir.display()))?;
        let cfg = K3Config::from_json(&cfg_txt).map_err(|e| anyhow::anyhow!(e))?;
        let idx_txt = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
            .context("reading model.safetensors.index.json")?;
        let idx: serde_json::Value = serde_json::from_str(&idx_txt)?;
        let map = idx
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("index.json has no weight_map")?;
        let index: HashMap<String, String> = map
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
            .collect();
        anyhow::ensure!(!index.is_empty(), "index.json weight_map is empty");
        Ok(Self {
            dir: dir.to_path_buf(),
            index,
            maps: HashMap::new(),
            cfg,
        })
    }

    fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Read one BF16 tensor, widened to f32, with its on-disk shape.
    fn read_bf16(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let shard = self
            .index
            .get(name)
            .with_context(|| format!("{name}: not in the checkpoint index"))?
            .clone();
        if !self.maps.contains_key(&shard) {
            let f = std::fs::File::open(self.dir.join(&shard))
                .with_context(|| format!("opening shard {shard}"))?;
            let m = unsafe { Mmap::map(&f) }.with_context(|| format!("mmapping {shard}"))?;
            self.maps.insert(shard.clone(), m);
        }
        let mmap = &self.maps[&shard];
        let st = SafeTensors::deserialize(mmap)
            .with_context(|| format!("parsing the header of {shard}"))?;
        let t = st
            .tensor(name)
            .with_context(|| format!("{name} absent from {shard}"))?;
        if t.dtype() != safetensors::Dtype::BF16 {
            bail!("{name}: on-disk dtype is {:?}, expected BF16", t.dtype());
        }
        let raw = t.data();
        anyhow::ensure!(raw.len() % 2 == 0, "{name}: odd byte count");
        let vals: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        anyhow::ensure!(!vals.is_empty(), "{name}: EMPTY tensor");
        Ok((vals, t.shape().to_vec()))
    }
}

/// One call site's two parameter tensors, straight from the checkpoint.
struct ParamPair {
    norm: Vec<f32>,
    norm_shape: Vec<usize>,
    proj: Vec<f32>,
    proj_shape: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Loading: the oracle
// ---------------------------------------------------------------------------

fn bf(z: &Npz, key: &str) -> Vec<f32> {
    let v: Vec<f32> = z
        .get(key)
        .bf16_to_f64()
        .into_iter()
        .map(|x| x as f32)
        .collect();
    assert!(!v.is_empty(), "{key}: EMPTY oracle array");
    v
}

fn f32s(z: &Npz, key: &str) -> Vec<f32> {
    let v = z.get(key).to_f32();
    assert!(!v.is_empty(), "{key}: EMPTY oracle array");
    v
}

fn scalar_usize(z: &Npz, key: &str) -> usize {
    let s = z.get(key).scalar();
    assert!(s >= 0.0 && s.fract() == 0.0, "{key}: {s} is not a count");
    s as usize
}

/// `[tokens, slots, hidden]` -> one `[tokens, hidden]` vector per slot.
fn split_slots(data: &[f32], tokens: usize, slots: usize, hidden: usize) -> Vec<Vec<f32>> {
    assert_eq!(
        data.len(),
        tokens * slots * hidden,
        "slot split: size mismatch"
    );
    (0..slots)
        .map(|s| {
            let mut out = Vec::with_capacity(tokens * hidden);
            for t in 0..tokens {
                let off = t * slots * hidden + s * hidden;
                out.extend_from_slice(&data[off..off + hidden]);
            }
            out
        })
        .collect()
}

/// Everything one AttnRes call site needs, from the oracle.
struct Site {
    label: String,
    slots: usize,
    bank: Vec<Vec<f32>>,
    prefix_sum: Vec<f32>,
    v: Vec<f32>,
    score_weight: Vec<f32>,
    scores: Vec<f32>,
    probs: Vec<f32>,
    out: Vec<f32>,
    alt_normalized: Vec<f32>,
    alt_prefix_first: Vec<f32>,
    /// The label of the sibling site whose norm gain is the cross control.
    cross_label: String,
}

/// One decoder layer's chained-lane inputs and expected state.
struct LayerTape {
    layer_in: Vec<f32>,
    attn_out: Vec<f32>,
    mlp_out: Vec<f32>,
    layer_out: Vec<f32>,
    input_layernorm_in: Vec<f32>,
    post_attention_layernorm_in: Vec<f32>,
    mlp_prefix_sum: Vec<f32>,
    nb_in: usize,
    nb_out: usize,
    bank_out: Vec<Vec<f32>>,
    has_sa_site: bool,
}

struct Oracle {
    sites: Vec<Site>,
    params: HashMap<String, ParamPair>,
    tape: Vec<LayerTape>,
    final_norm_in: Vec<f32>,
    /// The MEASURED checkpoint layers: those whose bank grew.
    measured_checkpoints: BTreeSet<usize>,
}

impl Oracle {
    fn site(&self, label: &str) -> &Site {
        self.sites
            .iter()
            .find(|s| s.label == label)
            .unwrap_or_else(|| panic!("no site {label}"))
    }
    fn param(&self, label: &str) -> &ParamPair {
        self.params
            .get(label)
            .unwrap_or_else(|| panic!("no params for {label}"))
    }
}

fn load_oracle(z: &Npz, ck: &mut Checkpoint) -> Result<Oracle> {
    let mut sites = Vec::new();
    let mut params: HashMap<String, ParamPair> = HashMap::new();
    let mut tape = Vec::new();
    let mut measured_checkpoints = BTreeSet::new();

    // Parameters for EVERY site the architecture has in these 13 layers,
    // including layer 0's self-attention pair which the forward never uses.
    // Reading it anyway is the point: the port is handed it and must decline.
    for l in 0..LAYERS {
        for (kind, np, pp) in [
            (
                "sa",
                LayerPart::SelfAttentionResNorm,
                LayerPart::SelfAttentionResProj,
            ),
            ("mlp", LayerPart::MlpResNorm, LayerPart::MlpResProj),
        ] {
            let (norm, norm_shape) =
                ck.read_bf16(&Slot::Layer { layer: l, part: np }.tensor_name())?;
            let (proj, proj_shape) =
                ck.read_bf16(&Slot::Layer { layer: l, part: pp }.tensor_name())?;
            params.insert(
                format!("L{l:02}.{kind}"),
                ParamPair {
                    norm,
                    norm_shape,
                    proj,
                    proj_shape,
                },
            );
        }
    }
    {
        let (norm, norm_shape) = ck.read_bf16(&Slot::OutputAttnResNorm.tensor_name())?;
        let (proj, proj_shape) = ck.read_bf16(&Slot::OutputAttnResProj.tensor_name())?;
        params.insert(
            "MODEL.output".to_string(),
            ParamPair {
                norm,
                norm_shape,
                proj,
                proj_shape,
            },
        );
    }

    for l in 0..LAYERS {
        let lp = format!("L{l:02}");
        let nb_in = scalar_usize(z, &format!("{lp}_blockres_in_nblocks"));
        let nb_out = scalar_usize(z, &format!("{lp}_blockres_out_nblocks"));
        if nb_out > nb_in {
            measured_checkpoints.insert(l);
        }
        anyhow::ensure!(nb_out > 0, "{lp}: bank is empty on exit");
        let bank_out = split_slots(
            &bf(z, &format!("{lp}_blockres_out_bf16bits")),
            TOKENS,
            nb_out,
            HIDDEN,
        );
        let has_sa_site = z.contains(&format!("{lp}_attnres_sa_out_bf16bits"));

        for kind in ["sa", "mlp"] {
            let base = format!("{lp}_attnres_{kind}");
            if !z.contains(&format!("{base}_out_bf16bits")) {
                continue;
            }
            let br = bf(z, &format!("{base}_block_residual_bf16bits"));
            anyhow::ensure!(br.len() % (TOKENS * HIDDEN) == 0, "{base}: ragged bank");
            let nb = br.len() / (TOKENS * HIDDEN);
            sites.push(Site {
                label: format!("{lp}.{kind}"),
                slots: nb + 1,
                bank: split_slots(&br, TOKENS, nb, HIDDEN),
                prefix_sum: bf(z, &format!("{base}_prefix_sum_bf16bits")),
                v: bf(z, &format!("{base}_v_bf16bits")),
                score_weight: f32s(z, &format!("{base}_score_weight")),
                scores: f32s(z, &format!("{base}_scores")),
                probs: f32s(z, &format!("{base}_probs")),
                out: bf(z, &format!("{base}_out_bf16bits")),
                alt_normalized: bf(z, &format!("{base}_ALT_out_combine_normalized_bf16bits")),
                alt_prefix_first: bf(z, &format!("{base}_ALT_out_prefix_first_bf16bits")),
                cross_label: format!("{lp}.{}", if kind == "sa" { "mlp" } else { "sa" }),
            });
        }

        let t = LayerTape {
            layer_in: bf(z, &format!("{lp}_layer_in_bf16bits")),
            attn_out: bf(z, &format!("{lp}_attn_o_proj_out_bf16bits")),
            mlp_out: bf(
                z,
                &format!(
                    "{lp}_{}_bf16bits",
                    if l == 0 { "mlp_out" } else { "moe_out" }
                ),
            ),
            layer_out: bf(z, &format!("{lp}_layer_out_bf16bits")),
            input_layernorm_in: bf(z, &format!("{lp}_input_layernorm_in_bf16bits")),
            post_attention_layernorm_in: bf(
                z,
                &format!("{lp}_post_attention_layernorm_in_bf16bits"),
            ),
            mlp_prefix_sum: bf(z, &format!("{lp}_attnres_mlp_prefix_sum_bf16bits")),
            nb_in,
            nb_out,
            bank_out,
            has_sa_site,
        };
        for (name, a) in [
            ("layer_in", &t.layer_in),
            ("attn_out", &t.attn_out),
            ("mlp_out", &t.mlp_out),
            ("layer_out", &t.layer_out),
            ("mlp_prefix_sum", &t.mlp_prefix_sum),
        ] {
            anyhow::ensure!(
                a.len() == TOKENS * HIDDEN,
                "{lp}.{name}: {} elements, expected {}",
                a.len(),
                TOKENS * HIDDEN
            );
        }
        tape.push(t);
    }

    {
        let base = "MODEL_attnres_output";
        let br = bf(z, &format!("{base}_block_residual_bf16bits"));
        let nb = br.len() / (TOKENS * HIDDEN);
        sites.push(Site {
            label: "MODEL.output".to_string(),
            slots: nb + 1,
            bank: split_slots(&br, TOKENS, nb, HIDDEN),
            prefix_sum: bf(z, &format!("{base}_prefix_sum_bf16bits")),
            v: bf(z, &format!("{base}_v_bf16bits")),
            score_weight: f32s(z, &format!("{base}_score_weight")),
            scores: f32s(z, &format!("{base}_scores")),
            probs: f32s(z, &format!("{base}_probs")),
            out: bf(z, &format!("{base}_out_bf16bits")),
            alt_normalized: bf(z, &format!("{base}_ALT_out_combine_normalized_bf16bits")),
            alt_prefix_first: bf(z, &format!("{base}_ALT_out_prefix_first_bf16bits")),
            // No sibling sublayer; the last layer's MLP gain is just as wrong
            // and just as available.
            cross_label: format!("L{:02}.mlp", LAYERS - 1),
        });
    }

    Ok(Oracle {
        sites,
        params,
        tape,
        final_norm_in: bf(z, "final_norm_in_bf16bits"),
        measured_checkpoints,
    })
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

fn t1<B: Backend>(v: &[f32], dev: &Device<B>) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(v.to_vec(), [v.len()]), dev)
}

fn t2<B: Backend>(v: &[f32], r: usize, c: usize, dev: &Device<B>) -> Tensor<B, 2> {
    assert_eq!(v.len(), r * c, "t2: {} elements for [{r}, {c}]", v.len());
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), dev)
}

fn host<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("host readback")
}

fn params_of<B: Backend>(p: &ParamPair, eps: f64, dev: &Device<B>) -> AttnResParams<B> {
    AttnResParams::new(
        t1::<B>(&p.norm, dev),
        t2::<B>(&p.proj, 1, p.norm.len(), dev),
        eps,
    )
}

/// One f32 ulp, relative — the unit the primitive measurements are quoted in.
const F32_EPS: f64 = f32::EPSILON as f64;

/// What this backend's own arithmetic does, measured **before** any of the
/// port's numbers are looked at.
///
/// This section exists because of a real bug it would have caught in one step
/// and did not, since it was written afterwards. The shipped module normalises
/// with `rsqrt(variance + eps)`, and transcribing that as `.sqrt().recip()` is
/// the obvious thing to write. On burn-ndarray, `recip` is an **approximate**
/// reciprocal: 2.85e-3 relative, ~24000 f32 ulp. Because the RMSNorm scale is
/// one scalar per row, that error does not average out over the 7168-term
/// reduction — it multiplies the whole score. The port therefore divides, and
/// these numbers are what says whether that is still the right choice on
/// whatever backend is running.
///
/// The probe range is the variance range the real data actually produces
/// (2.3e-5 .. 1.8), so the figures describe this workload rather than a
/// generic sweep.
fn section_primitives<B: Backend>(g: &mut Gate, dev: &Device<B>) {
    println!("  -- primitives --");
    let n = 20000;
    let probe: Vec<f32> = (1..=n).map(|i| 2.0e-5 + (i as f32) * 1.0e-4).collect();
    let ones: Vec<f32> = probe.iter().map(|_| 1.0f32).collect();
    let pt = t1::<B>(&probe, dev);

    let r_sqrt = host(pt.clone().sqrt());
    let r_div = host(t1::<B>(&ones, dev) / pt.clone().sqrt());
    let r_recip = host(pt.clone().sqrt().recip());

    let (mut e_sqrt, mut e_div, mut e_recip) = (0.0f64, 0.0f64, 0.0f64);
    for (i, x) in probe.iter().enumerate() {
        let root = (*x as f64).sqrt();
        e_sqrt = worse(e_sqrt, ((r_sqrt[i] as f64) / root - 1.0).abs());
        e_div = worse(e_div, (r_div[i] as f64) * root - 1.0);
        e_recip = worse(e_recip, ((r_recip[i] as f64) * root - 1.0).abs());
    }
    // The two the port uses must be ulp-accurate...
    g.le("primitives.sqrt", e_sqrt / F32_EPS, 4.0, "f32 ulp");
    g.le(
        "primitives.rsqrt_by_division",
        e_div / F32_EPS,
        4.0,
        "f32 ulp",
    );
    // ...and the one it deliberately does not use is reported, not budgeted:
    // holding a backend to an accuracy it never promised would make this gate
    // fail on a backend change that is none of the port's business.
    println!(
        "     recip (NOT used by the port): {:.1} f32 ulp = {e_recip:.3e} rel — \
         the reason the normalisation divides",
        e_recip / F32_EPS
    );

    // The reduction, on the real width, flat vs pairwise, against float64.
    let rows = 64usize;
    let mut data: Vec<f32> = Vec::with_capacity(rows * HIDDEN);
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    for _ in 0..rows * HIDDEN {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((seed >> 40) as f32) / (1u32 << 24) as f32 - 0.5;
        data.push(u);
    }
    let flat = host(t2::<B>(&data, rows, HIDDEN, dev).sum_dim(1));
    let mut e_flat = 0.0f64;
    for r in 0..rows {
        let row = &data[r * HIDDEN..(r + 1) * HIDDEN];
        let mut acc = 0.0f64;
        let mut mag = 0.0f64;
        for x in row {
            acc += *x as f64;
            mag += (*x as f64).abs();
        }
        e_flat = worse(e_flat, ((flat[r] as f64 - acc) / mag).abs());
    }
    println!(
        "     backend sum_dim over {HIDDEN} terms: {:.1} f32 ulp of sum|terms| \
         (the port reduces pairwise instead)",
        e_flat / F32_EPS
    );

    // The mixture's own shape, both ways. The contracted axis is the slot
    // axis — three long here — so `matmul` and a broadcast-multiply-and-sum
    // are the same arithmetic on paper. They are not the same arithmetic on a
    // backend that routes GEMM through tensor cores, and this is where that
    // shows.
    let (n, sl, m) = (256usize, 3usize, 512usize);
    let mut w: Vec<f32> = Vec::with_capacity(n * sl);
    let mut x: Vec<f32> = Vec::with_capacity(n * sl * m);
    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 40) as f32) / (1u32 << 24) as f32
    };
    for _ in 0..n {
        let mut row: Vec<f32> = (0..sl).map(|_| next()).collect();
        let z: f32 = row.iter().sum();
        for r in row.iter_mut() {
            *r /= z;
        }
        w.extend(row);
    }
    for _ in 0..n * sl * m {
        x.push((next() - 0.5) * 20.0);
    }
    let wt = t2::<B>(&w, n, sl, dev);
    let xt: Tensor<B, 3> = Tensor::from_data(TensorData::new(x.clone(), [n, sl, m]), dev);
    let by_matmul = host(
        wt.clone()
            .reshape([n, 1, sl])
            .matmul(xt.clone())
            .reshape([n, m]),
    );
    let by_sum = host((wt.reshape([n, sl, 1]) * xt).sum_dim(1).reshape([n, m]));
    let (mut e_mm, mut e_sum) = (0.0f64, 0.0f64);
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0f64;
            for s in 0..sl {
                acc += (w[i * sl + s] as f64) * (x[(i * sl + s) * m + j] as f64);
            }
            let scale = acc.abs().max(1.0);
            e_mm = worse(e_mm, ((by_matmul[i * m + j] as f64) - acc).abs() / scale);
            e_sum = worse(e_sum, ((by_sum[i * m + j] as f64) - acc).abs() / scale);
        }
    }
    println!(
        "     {n}x[1,{sl}]@[{sl},{m}] mixture: matmul {:.1} f32 ulp | broadcast-multiply-and-sum \
         {:.1} f32 ulp   (the port does the latter)",
        e_mm / F32_EPS,
        e_sum / F32_EPS
    );
    // Only the one the port uses is held to a budget. The other is reported so
    // that the reason for the choice is a measurement on every run, not a
    // comment that ages.
    g.le("primitives.mixture_by_sum", e_sum / F32_EPS, 8.0, "f32 ulp");
}

// ---------------------------------------------------------------------------
// Section 1 — the boundary
// ---------------------------------------------------------------------------

fn section_boundary(g: &mut Gate, o: &Oracle, ck: &Checkpoint) {
    println!("  -- boundary --");
    let cfg = &ck.cfg.text_config;

    g.truth(
        "cfg.attn_res_block_size",
        cfg.attn_res_block_size == Some(12),
        format!("{:?}", cfg.attn_res_block_size),
    );
    g.truth(
        "cfg.num_hidden_layers",
        cfg.num_hidden_layers == 93,
        format!("{}", cfg.num_hidden_layers),
    );
    g.truth(
        "cfg.hidden_size",
        cfg.hidden_size == HIDDEN,
        format!("{}", cfg.hidden_size),
    );
    g.truth(
        "cfg.rms_norm_eps",
        (cfg.rms_norm_eps - 1e-5).abs() < 1e-12,
        format!("{:e}", cfg.rms_norm_eps),
    );

    // The schedule the port runs on, straight from the config predicate.
    let schedule: Vec<bool> = (0..cfg.num_hidden_layers)
        .map(|l| cfg.is_attn_res_checkpoint(l))
        .collect();
    let checkpoints: Vec<usize> = schedule
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    g.truth(
        "schedule.length",
        schedule.len() == cfg.num_hidden_layers && !schedule.is_empty(),
        format!("{} layers", schedule.len()),
    );
    g.truth(
        "schedule.checkpoints",
        checkpoints == vec![0, 12, 24, 36, 48, 60, 72, 84],
        format!("{checkpoints:?}"),
    );
    g.truth(
        "schedule.bank_size",
        checkpoints.len() == cfg.attn_res_bank_size() && cfg.attn_res_bank_size() == 8,
        format!("{} snapshots", cfg.attn_res_bank_size()),
    );
    g.truth(
        "schedule.last_layer_not_a_checkpoint",
        !schedule[cfg.num_hidden_layers - 1],
        format!(
            "layer {} snapshots: {}",
            cfg.num_hidden_layers - 1,
            schedule[cfg.num_hidden_layers - 1]
        ),
    );
    let mut longest_run = 0usize;
    let mut run = 0usize;
    for &is_cp in &schedule {
        run = if is_cp { 1 } else { run + 1 };
        longest_run = longest_run.max(run);
    }
    g.truth(
        "schedule.longest_accumulation_run",
        longest_run == 12,
        format!(
            "{longest_run} layers (block size {:?})",
            cfg.attn_res_block_size
        ),
    );

    // MEASURED: which layers the shipped forward actually snapshotted, read off
    // the oracle's own bank-size deltas, not off any config or source file.
    g.truth(
        "oracle.layers_present",
        o.tape.len() == LAYERS,
        format!("{} layers of tape", o.tape.len()),
    );
    let measured: Vec<usize> = o.measured_checkpoints.iter().copied().collect();
    let from_cfg: Vec<usize> = checkpoints
        .iter()
        .copied()
        .filter(|&l| l < LAYERS)
        .collect();
    g.truth(
        "boundary.config_vs_measured",
        measured == from_cfg && !measured.is_empty(),
        format!("measured {measured:?} vs config {from_cfg:?}"),
    );
    g.truth(
        "boundary.two_crossings_in_window",
        measured.len() == 2,
        format!("{} crossings in layers 0..{LAYERS}", measured.len()),
    );
    let deltas_sane = o
        .tape
        .iter()
        .all(|t| t.nb_out == t.nb_in || t.nb_out == t.nb_in + 1);
    g.truth("boundary.bank_grows_by_at_most_one", deltas_sane, "");
    g.truth(
        "boundary.bank_starts_empty",
        o.tape[0].nb_in == 0,
        format!("{} entries at layer 0 entry", o.tape[0].nb_in),
    );
    let chained = (1..LAYERS).all(|l| o.tape[l].nb_in == o.tape[l - 1].nb_out);
    g.truth("boundary.oracle_bank_chains", chained, "");

    // The self-attention mixture is absent exactly where the bank is empty —
    // and present everywhere else, or "absent at layer 0" would be a claim
    // about an oracle that simply never captured the array.
    let sa_sites = o.tape.iter().filter(|t| t.has_sa_site).count();
    g.truth(
        "boundary.sa_mix_present_except_layer0",
        !o.tape[0].has_sa_site && sa_sites == LAYERS - 1,
        format!("{sa_sites} of {LAYERS} layers carry an sa mixture"),
    );

    // NON-VACUITY. A boundary check is worthless if the window cannot see the
    // difference between the right schedule and a shifted one. Measure that it
    // can, inside 0..13, before believing the agreement above.
    for shift in [-1i64, 1] {
        let shifted: BTreeSet<usize> = (0..LAYERS as i64)
            .filter(|&l| {
                let src = l - shift;
                src >= 0 && (src as usize) % 12 == 0
            })
            .map(|l| l as usize)
            .collect();
        g.truth(
            &format!("boundary.offby{shift:+}_is_visible"),
            shifted != o.measured_checkpoints,
            format!(
                "shifted {shifted:?} vs measured {:?}",
                o.measured_checkpoints
            ),
        );
    }

    // The checkpoint's own inventory: AttnRes parameters exist at EVERY layer,
    // so the schedule is the only thing selecting where snapshots happen.
    let mut missing = Vec::new();
    for l in 0..cfg.num_hidden_layers {
        for part in [
            LayerPart::SelfAttentionResNorm,
            LayerPart::SelfAttentionResProj,
            LayerPart::MlpResNorm,
            LayerPart::MlpResProj,
        ] {
            let n = Slot::Layer { layer: l, part }.tensor_name();
            if !ck.has(&n) {
                missing.push(n);
            }
        }
    }
    for s in [Slot::OutputAttnResNorm, Slot::OutputAttnResProj] {
        if !ck.has(&s.tensor_name()) {
            missing.push(s.tensor_name());
        }
    }
    g.truth(
        "checkpoint.attn_res_tensors_complete",
        missing.is_empty(),
        format!(
            "{} names expected, {} missing",
            4 * cfg.num_hidden_layers + 2,
            missing.len()
        ),
    );
}

// ---------------------------------------------------------------------------
// Section 2 — bfloat16 rounding
// ---------------------------------------------------------------------------

fn section_rounding<B: Backend>(g: &mut Gate, o: &Oracle, dev: &Device<B>) {
    println!("  -- rounding --");

    // A stress set across the documented domain: every decade of normal f32,
    // both signs, exact ties, and zero — plus real activations.
    let mut probe: Vec<f32> = Vec::new();
    for e in -37i32..=30 {
        let m = 10f32.powi(e);
        for f in [1.0f32, 1.5, 1.9999999, 3.0, 7.0, 1.00390625, 1.00390624] {
            probe.push(m * f);
            probe.push(-m * f);
        }
    }
    probe.extend([
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ]);
    // Exact halfway cases between adjacent bfloat16 values — the ONLY inputs
    // where the round-half-to-even rule decides anything, and therefore the
    // only ones that can tell two rounding schemes apart.
    //
    // Built from bit patterns, because arithmetic cannot express them by
    // accident: the midpoint between bfloat16 `b` and `b + 1` is the f32 whose
    // top 16 bits are `b` and whose low 16 bits are `0x8000`. The earlier
    // version of this probe stepped by 2^-8 and offset by 2^-9 and called the
    // result a tie; bfloat16's step at 1.0 is 2^-7, so every one of those
    // points sat at a quarter step and the tie rule was never exercised. A
    // mutant that changed the splitting constant from 2^16+1 to 2^16 passed
    // the whole gate because of it.
    let mut ties = 0usize;
    for exp in 0x38u16..0x48 {
        for man in 0u16..128 {
            let b = (exp << 7) | man;
            // Odd and even last bits both appear, so the tie breaks in both
            // directions somewhere in this sweep.
            probe.push(f32::from_bits(((b as u32) << 16) | 0x0000_8000));
            probe.push(f32::from_bits((((b | 0x8000) as u32) << 16) | 0x0000_8000));
            ties += 2;
        }
    }
    assert!(ties >= 2000, "tie probe is too small: {ties}");
    let real: Vec<f32> = o.sites.iter().flat_map(|s| s.out.iter().copied()).collect();
    probe.extend(real.iter().take(200_000).copied());
    assert!(
        probe.len() > 1000,
        "rounding probe set is too small to mean anything"
    );

    let got = host(round_bf16(t1::<B>(&probe, dev)));
    let want: Vec<f32> = probe.iter().map(|x| bf16_to_f32(bf16_bits(*x))).collect();
    g.truth(
        "round_bf16.matches_bitlevel_rne",
        bits_equal("round_bf16", &got, &want),
        format!(
            "{} probe values, of which {ties} are exact bfloat16 ties",
            probe.len()
        ),
    );

    // EXHAUSTIVE over one binade. Every f32 in [1, 2) is 2^23 consecutive bit
    // patterns, i.e. every significand the format has; and the identity is
    // exactly scale-invariant across binades, because a multiply and two
    // subtractions by powers of two only move the exponent. So this is not a
    // sample of the normal range — it *is* the normal range, once. A sampled
    // probe is what let a mutated splitting constant through.
    {
        let mut agree = true;
        let mut checked = 0usize;
        for chunk in 0..8u32 {
            let n = 1u32 << 20;
            let vals: Vec<f32> = (0..n)
                .map(|i| f32::from_bits(0x3F80_0000 + chunk * n + i))
                .collect();
            let got = host(round_bf16(t1::<B>(&vals, dev)));
            for (i, x) in vals.iter().enumerate() {
                if got[i].to_bits() != bf16_to_f32(bf16_bits(*x)).to_bits() {
                    agree = false;
                }
            }
            checked += vals.len();
        }
        g.truth(
            "round_bf16.exhaustive_over_one_binade",
            agree && checked == (1 << 23),
            format!("{checked} consecutive f32 in [1, 2), every significand"),
        );
    }

    // Idempotence on real bfloat16 data: the oracle's arrays are already
    // bfloat16, so rounding them must be the identity.
    let already: Vec<f32> = o
        .tape
        .iter()
        .flat_map(|t| t.layer_out.iter().copied())
        .collect();
    let re = host(round_bf16(t1::<B>(&already, dev)));
    g.truth(
        "round_bf16.identity_on_bf16_data",
        bits_equal("idempotence", &re, &already),
        format!("{} oracle values", already.len()),
    );

    // The domain, and the margin to the real data.
    //
    // Stated as a BAND: `round_bf16` must be exact everywhere within three
    // decades of the activations on either side. The tempting version of this
    // check — scan outwards until it breaks, then divide — is worse than
    // useless: `10f32.powi(-40)` evaluates to exactly 0.0 (the intermediate
    // 10^40 overflows), 0.0 rounds correctly, no breakdown is ever found, and
    // the margin comes out as `x / 0 = inf`. That version passed. It could not
    // have done anything else.
    let peak = o
        .tape
        .iter()
        .map(|t| max_abs("layer_out", &t.layer_out))
        .fold(0.0f64, f64::max);
    let floor = o
        .sites
        .iter()
        .map(|s| min_nonzero_abs(&s.out))
        .fold(f64::INFINITY, f64::min);
    assert!(
        peak.is_finite() && peak > 0.0,
        "no activation scale to test the band against"
    );
    assert!(
        floor.is_finite() && floor > 0.0,
        "no activation floor to test the band against"
    );
    let (lo, hi) = ((floor / 1e3) as f32, (peak * 1e3) as f32);
    let steps = 4000;
    let mut band: Vec<f32> = Vec::with_capacity(2 * steps + 4);
    let ratio = ((hi as f64) / (lo as f64)).powf(1.0 / steps as f64);
    let mut x = lo as f64;
    for _ in 0..=steps {
        band.push(x as f32);
        band.push(-(x as f32));
        x *= ratio;
    }
    band.push(lo);
    band.push(hi);
    assert!(band.len() > 100, "band probe is too small to mean anything");
    let got_band = host(round_bf16(t1::<B>(&band, dev)));
    let want_band: Vec<f32> = band.iter().map(|v| bf16_to_f32(bf16_bits(*v))).collect();
    g.truth(
        "round_bf16.exact_on_the_data_band",
        bits_equal("band", &got_band, &want_band) && (lo as f64) < floor && (hi as f64) > peak,
        format!(
            "{} values over [{lo:.3e}, {hi:.3e}]; activations span [{floor:.3e}, {peak:.3}] — \
             three decades of headroom each side",
            band.len()
        ),
    );

    // Where it actually breaks, MEASURED by bit pattern (not by `powi`, which
    // cannot express a subnormal). Reported, not budgeted: the numbers below
    // are a property of the identity, and the band check above is what the
    // port depends on.
    let mut worst_small = 0.0f64;
    let mut smalls: Vec<f32> = Vec::new();
    for b in 1u32..=0x0080_0000 {
        if b % 4099 == 1 || b < 64 {
            smalls.push(f32::from_bits(b));
        }
    }
    let got_s = host(round_bf16(t1::<B>(&smalls, dev)));
    for (i, v) in smalls.iter().enumerate() {
        if got_s[i].to_bits() != bf16_to_f32(bf16_bits(*v)).to_bits() {
            worst_small = worst_small.max(*v as f64);
        }
    }
    let mut smallest_big = f64::INFINITY;
    let bigs: Vec<f32> = (0..2000).map(|i| 1e30f32 * 1.01f32.powi(i)).collect();
    let got_b = host(round_bf16(t1::<B>(&bigs, dev)));
    for (i, v) in bigs.iter().enumerate() {
        if v.is_finite() && got_b[i].to_bits() != bf16_to_f32(bf16_bits(*v)).to_bits() {
            smallest_big = smallest_big.min(*v as f64);
        }
    }
    g.truth(
        "round_bf16.domain_is_bounded_on_both_sides",
        worst_small > 0.0 && smallest_big.is_finite() && worst_small < floor && smallest_big > peak,
        format!(
            "subnormals up to {worst_small:.4e} are returned unrounded; breaks above \
             {smallest_big:.4e} — activations [{floor:.3e}, {peak:.3}] sit between, by \
             {:.0e}x and {:.0e}x",
            floor / worst_small,
            smallest_big / peak
        ),
    );
}

// ---------------------------------------------------------------------------
// Section 3 — the weights, from the checkpoint
// ---------------------------------------------------------------------------

fn section_weights<B: Backend>(g: &mut Gate, o: &Oracle, ck: &Checkpoint, dev: &Device<B>) {
    println!("  -- weights --");
    let eps = ck.cfg.text_config.rms_norm_eps;

    // The shapes the layout derives from the config, against the shard headers.
    // A norm/proj swap is invisible in their product (multiplication commutes)
    // and visible only in the ranks.
    let mut shape_ok = true;
    let mut detail = String::new();
    for (label, slots) in [
        (
            "MODEL.output",
            (Slot::OutputAttnResNorm, Slot::OutputAttnResProj),
        ),
        (
            "L00.sa",
            (
                Slot::Layer {
                    layer: 0,
                    part: LayerPart::SelfAttentionResNorm,
                },
                Slot::Layer {
                    layer: 0,
                    part: LayerPart::SelfAttentionResProj,
                },
            ),
        ),
        (
            "L00.mlp",
            (
                Slot::Layer {
                    layer: 0,
                    part: LayerPart::MlpResNorm,
                },
                Slot::Layer {
                    layer: 0,
                    part: LayerPart::MlpResProj,
                },
            ),
        ),
    ] {
        let p = o.param(label);
        let (norm_want, proj_want) = (describe(&ck.cfg, slots.0), describe(&ck.cfg, slots.1));
        let ok = matches!(&norm_want, Some((s, d)) if s.dims() == [HIDDEN] && matches!(d, Dtype::Bf16))
            && matches!(&proj_want, Some((s, d)) if s.dims() == [1, HIDDEN] && matches!(d, Dtype::Bf16))
            && p.norm_shape == vec![HIDDEN]
            && p.proj_shape == vec![1, HIDDEN];
        if !ok {
            shape_ok = false;
            detail = format!("{label}: header {:?}/{:?}", p.norm_shape, p.proj_shape);
        }
    }
    g.truth(
        "layout.attn_res_shapes",
        shape_ok,
        if detail.is_empty() {
            "norm [7168], proj [1,7168], BF16 — layout and headers agree".to_string()
        } else {
            detail
        },
    );

    let mut exact_sites = 0usize;
    for s in &o.sites {
        let p = params_of::<B>(o.param(&s.label), eps, dev);
        let sw = host(p.score_weight());
        if bits_equal(&format!("{}.score_weight", s.label), &sw, &s.score_weight) {
            exact_sites += 1;
        } else {
            g.truth(
                &format!("weights.{}.score_weight", s.label),
                false,
                format!("max |d| = {:e}", max_abs_diff("sw", &sw, &s.score_weight)),
            );
        }
    }
    g.truth(
        "weights.score_weight_bitexact_all_sites",
        exact_sites == o.sites.len() && !o.sites.is_empty(),
        format!(
            "{}/{} sites bit-exact from the checkpoint's own bytes",
            exact_sites,
            o.sites.len()
        ),
    );

    // The cross-site control: the *other* sublayer's norm gain, same layer, in
    // the same product. If this also matched, the identity above would say
    // nothing about which tensor was read.
    let mut min_cross = f64::INFINITY;
    for s in &o.sites {
        let cross = o.param(&s.cross_label);
        let p = AttnResParams::<B>::new(
            t1::<B>(&cross.norm, dev),
            t2::<B>(&o.param(&s.label).proj, 1, HIDDEN, dev),
            eps,
        );
        min_cross = smaller(
            min_cross,
            max_rel_diff("cross", &host(p.score_weight()), &s.score_weight),
        );
    }
    g.gt(
        "weights.cross_site_control_differs",
        min_cross,
        CONTROL_MIN_REL,
        "rel (min over sites)",
    );

    // The rank assertions must actually fire — BOTH of them. `[H, 1]` is
    // rejected by the width check alone, so testing only that shape leaves the
    // row check untested: deleting it changes nothing observable. `[2, H]` is
    // the shape only the row check can reject.
    let p0 = o.param("L00.mlp");
    let wide: Vec<f32> = p0.proj.iter().chain(p0.proj.iter()).copied().collect();
    let (transposed, m1) = expect_panic_saying(
        AssertUnwindSafe(|| {
            AttnResParams::<B>::new(
                t1::<B>(&p0.norm, dev),
                t2::<B>(&p0.proj, HIDDEN, 1, dev),
                eps,
            );
        }),
        "stores exactly one",
    );
    let (two_rows, m2) = expect_panic_saying(
        AssertUnwindSafe(|| {
            AttnResParams::<B>::new(t1::<B>(&p0.norm, dev), t2::<B>(&wide, 2, HIDDEN, dev), eps);
        }),
        "stores exactly one",
    );
    // ...and one that only the WIDTH check can reject, so both assertions are
    // exercised rather than one of them shadowing the other.
    let (narrow, m3) = expect_panic_saying(
        AssertUnwindSafe(|| {
            AttnResParams::<B>::new(
                t1::<B>(&p0.norm, dev),
                t2::<B>(&p0.proj[..HIDDEN / 2], 1, HIDDEN / 2, dev),
                eps,
            );
        }),
        "must both be the hidden size",
    );
    g.truth(
        "weights.rank_assertion_fires",
        transposed && two_rows && narrow,
        format!(
            "[7168,1] -> {transposed} | [2,7168] -> {two_rows} | [1,3584] -> {narrow}   ({m1:.40} / {m2:.40} / {m3:.40})"
        ),
    );
}

// ---------------------------------------------------------------------------
// Section 4 — every mixture call site
// ---------------------------------------------------------------------------

fn section_sites<B: Backend>(g: &mut Gate, o: &Oracle, eps: f64, dev: &Device<B>) {
    println!("  -- sites --");
    assert!(
        !o.sites.is_empty(),
        "no AttnRes call sites found in the oracle"
    );

    let mut stack_ok = true;
    let mut worst_scores = 0.0f64;
    let mut worst_probs = 0.0f64;
    let mut worst_prob_sum = 0.0f64;
    let mut worst_out_ulp = 0.0f64;
    let mut worst_out_abs = 0.0f64;
    let mut differing_total = 0usize;
    let mut all_rounded = true;
    let mut worst_port_f64_scores = 0.0f64;
    let mut worst_oracle_f64_scores = 0.0f64;
    let mut worst_port_f64_probs = 0.0f64;
    let mut worst_oracle_f64_probs = 0.0f64;
    let mut worst_port_f64_ulp = 0.0f64;
    let mut worst_oracle_f64_ulp = 0.0f64;

    for s in &o.sites {
        let p = params_of::<B>(o.param(&s.label), eps, dev);
        let bank: Vec<Tensor<B, 2>> = s
            .bank
            .iter()
            .map(|b| t2::<B>(b, TOKENS, HIDDEN, dev))
            .collect();
        let acc = t2::<B>(&s.prefix_sum, TOKENS, HIDDEN, dev);

        // The candidate stack itself — accumulator LAST — against the array the
        // shipped module actually built.
        let v = stack_candidates(&bank, acc);
        assert_eq!(
            v.dims(),
            [TOKENS, s.slots, HIDDEN],
            "{}: stack shape",
            s.label
        );
        if !bits_equal(&format!("{}.v", s.label), &host(v.clone()), &s.v) {
            stack_ok = false;
            g.truth(
                &format!("sites.{}.stack", s.label),
                false,
                "candidate stack differs",
            );
        }

        let m = p.mix(v);
        let scores = host(m.scores);
        let probs = host(m.probs);
        let out = host(m.out);

        let (ref_scores, ref_probs, ref_out) =
            mix_f64(&s.v, &s.score_weight, eps, TOKENS, s.slots, HIDDEN);
        let ref_scores32: Vec<f32> = ref_scores.iter().map(|x| *x as f32).collect();
        let ref_probs32: Vec<f32> = ref_probs.iter().map(|x| *x as f32).collect();
        worst_port_f64_scores = worse(
            worst_port_f64_scores,
            max_rel_diff("scores/f64", &scores, &ref_scores32),
        );
        worst_oracle_f64_scores = worse(
            worst_oracle_f64_scores,
            max_rel_diff("oracle/f64", &s.scores, &ref_scores32),
        );
        worst_port_f64_probs = worse(
            worst_port_f64_probs,
            max_abs_diff("probs/f64", &probs, &ref_probs32),
        );
        worst_oracle_f64_probs = worse(
            worst_oracle_f64_probs,
            max_abs_diff("oracle probs/f64", &s.probs, &ref_probs32),
        );
        worst_port_f64_ulp = worse(worst_port_f64_ulp, max_rel_diff("out/f64", &out, &ref_out));
        worst_oracle_f64_ulp = worse(
            worst_oracle_f64_ulp,
            max_rel_diff("oracle out/f64", &s.out, &ref_out),
        );

        worst_scores = worse(worst_scores, max_rel_diff("scores", &scores, &s.scores));
        worst_probs = worse(worst_probs, max_abs_diff("probs", &probs, &s.probs));
        for t in 0..TOKENS {
            let sum: f64 = probs[t * s.slots..(t + 1) * s.slots]
                .iter()
                .map(|x| *x as f64)
                .sum();
            worst_prob_sum = worse(worst_prob_sum, (sum - 1.0).abs());
        }
        let (ulp, differing) = max_ulp_bf16("out", &out, &s.out);
        worst_out_ulp = worse(worst_out_ulp, ulp);
        differing_total += differing;
        worst_out_abs = worse(worst_out_abs, max_rel_diff("out", &out, &s.out));
        if !all_exact_bf16(&out) {
            all_rounded = false;
        }
    }

    println!(
        "     ATTRIBUTION vs a float64 host transcription of the same formula:\n\
         \x20      scores  port {worst_port_f64_scores:.3e} rel   oracle(torch f32) {worst_oracle_f64_scores:.3e} rel\n\
         \x20      probs   port {worst_port_f64_probs:.3e} abs   oracle {worst_oracle_f64_probs:.3e} abs\n\
         \x20      out     port {worst_port_f64_ulp:.3e} rel   oracle {worst_oracle_f64_ulp:.3e} rel"
    );
    g.truth(
        "sites.count",
        o.sites.len() == 26,
        format!("{} call sites (12 sa + 13 mlp + 1 model)", o.sites.len()),
    );
    g.truth(
        "sites.candidate_stack_bitexact",
        stack_ok,
        "accumulator LAST, all sites",
    );
    g.le("sites.scores", worst_scores, SCORE_RTOL, "rel");
    // ...and, separately, the port's arithmetic must be no worse than the
    // arithmetic it is a port of. A fixed budget cannot say that: this data is
    // benign enough that a flat 7168-term reduction also fits inside 1e-5, so
    // holding only the fixed budget left the pairwise reduction untestable —
    // a mutant that reverted it passed. Both sides here are measured against
    // the same float64 transcription, so the ratio is scale-free and moves
    // with the backend rather than with a number chosen in advance.
    g.le(
        "sites.scores_no_worse_than_torch",
        worst_port_f64_scores / worst_oracle_f64_scores,
        2.0,
        "x the shipped module's own f32 error",
    );
    g.le(
        "sites.probs_no_worse_than_torch",
        worst_port_f64_probs / worst_oracle_f64_probs,
        2.0,
        "x the shipped module's own f32 error",
    );
    g.le("sites.probs", worst_probs, PROB_ATOL, "abs");
    g.le("sites.probs_sum_to_one", worst_prob_sum, PROB_ATOL, "abs");
    g.le("sites.out", worst_out_abs, OUT_REL, "rel-to-max");
    g.truth(
        "sites.out_rounded_to_bf16",
        all_rounded,
        format!(
            "{differing_total} of {} elements differ at all; worst element {worst_out_ulp:.0} bf16 ulp \
             [diagnostic, not a criterion]",
            o.sites.len() * TOKENS * HIDDEN
        ),
    );
}

// ---------------------------------------------------------------------------
// Section 5 — controls, and the two invariances
// ---------------------------------------------------------------------------

fn section_controls<B: Backend>(g: &mut Gate, o: &Oracle, eps: f64, dev: &Device<B>) {
    println!("  -- controls --");

    let mut min_control_gap = f64::INFINITY;
    let mut min_port_gap = f64::INFINITY;
    let mut worst_reproduce = f64::INFINITY;
    let mut worst_inv_oracle = 0.0f64;
    let mut worst_inv_port = 0.0f64;

    for s in &o.sites {
        let p = params_of::<B>(o.param(&s.label), eps, dev);
        let bank: Vec<Tensor<B, 2>> = s
            .bank
            .iter()
            .map(|b| t2::<B>(b, TOKENS, HIDDEN, dev))
            .collect();
        let acc = t2::<B>(&s.prefix_sum, TOKENS, HIDDEN, dev);
        let v = stack_candidates(&bank, acc.clone());
        let m = p.mix(v.clone());
        let out = host(m.out);

        // (a) The control is genuinely a different answer from the truth.
        min_control_gap = smaller(
            min_control_gap,
            max_rel_diff("control gap", &s.alt_normalized, &s.out),
        );
        // (b) This port does not produce it.
        min_port_gap = smaller(
            min_port_gap,
            max_rel_diff("port gap", &out, &s.alt_normalized),
        );

        // (c) The control IS the mis-port it is named after — combine over the
        //     NORMALISED candidates instead of the raw ones. Reproducing it
        //     pins its meaning; without this step "we do not match that array"
        //     is a claim about an array of unknown provenance.
        let variance = v.clone().powf_scalar(2.0).mean_dim(2);
        let k = v.clone() * variance.add_scalar(eps).sqrt().recip();
        let wrong = round_bf16(
            m.probs
                .clone()
                .reshape([TOKENS, 1, s.slots])
                .matmul(k)
                .reshape([TOKENS, HIDDEN]),
        );
        // Both distances are taken against the SAME array's scale — the
        // control's — because a ratio of two differently-normalised relative
        // errors is not a ratio of anything.
        let d_repro = max_rel_diff("reproduce", &host(wrong), &s.alt_normalized);
        let d_truth = max_rel_diff("truth vs control", &s.out, &s.alt_normalized);
        worst_reproduce = smaller(worst_reproduce, d_truth / d_repro.max(1e-12));

        // The slot-order INVARIANCE. Not a discriminator — the oracle's own
        // manifest demotes it, and the mathematics says why. Checked as a
        // property this port must share, labelled so no reader mistakes it for
        // a control that could fail.
        worst_inv_oracle = worse(
            worst_inv_oracle,
            max_rel_diff("inv oracle", &s.alt_prefix_first, &s.out),
        );
        let mut reordered: Vec<Tensor<B, 3>> = vec![acc.reshape([TOKENS, 1, HIDDEN])];
        for b in &bank {
            reordered.push(b.clone().reshape([TOKENS, 1, HIDDEN]));
        }
        let out_first = host(p.mix(Tensor::cat(reordered, 1)).out);
        worst_inv_port = worse(worst_inv_port, max_rel_diff("inv port", &out_first, &out));
    }

    // The normalisation the port does NOT do: `sqrt().recip()`, the literal
    // transcription of the shipped `rsqrt`. Run through the identical mixture
    // so the only difference is that one op, and required to be *worse*. This
    // is what stops the division from being reverted to a reciprocal by
    // someone reading the source for fidelity.
    {
        let s = o.site("L01.sa");
        let p = params_of::<B>(o.param("L01.sa"), eps, dev);
        let bank: Vec<Tensor<B, 2>> = s
            .bank
            .iter()
            .map(|b| t2::<B>(b, TOKENS, HIDDEN, dev))
            .collect();
        let v = stack_candidates(&bank, t2::<B>(&s.prefix_sum, TOKENS, HIDDEN, dev));
        let (ref_scores, _, _) = mix_f64(&s.v, &s.score_weight, eps, TOKENS, s.slots, HIDDEN);
        let ref32: Vec<f32> = ref_scores.iter().map(|x| *x as f32).collect();
        let variance = v.clone().powf_scalar(2.0).mean_dim(2);
        let recip_scores = host(
            (v.clone()
                * variance.add_scalar(eps).sqrt().recip()
                * t1::<B>(&s.score_weight, dev).reshape([1, 1, HIDDEN]))
            .sum_dim(2)
            .reshape([TOKENS, s.slots]),
        );
        let port_scores = host(p.mix(v).scores);
        let e_recip = max_rel_diff("recip scores", &recip_scores, &ref32);
        let e_port = max_rel_diff("port scores", &port_scores, &ref32);
        g.gt(
            "controls.rsqrt_via_recip_is_worse",
            e_recip / e_port.max(1e-12),
            10.0,
            "x worse",
        );
        println!("     rsqrt via recip {e_recip:.3e} rel vs the port's division {e_port:.3e} rel");
    }

    g.gt(
        "controls.combine_normalized_is_wrong",
        min_control_gap,
        CONTROL_MIN_REL,
        "rel (min)",
    );
    g.gt(
        "controls.port_rejects_combine_normalized",
        min_port_gap,
        CONTROL_MIN_REL,
        "rel (min)",
    );
    g.gt(
        "controls.combine_normalized_reproduced",
        worst_reproduce,
        CONTROL_REPRODUCE_RATIO,
        "x closer than the truth (min over sites)",
    );
    g.le(
        "invariance.slot_order_oracle",
        worst_inv_oracle,
        INVARIANCE_REL,
        "rel [NOT a control]",
    );
    g.le(
        "invariance.slot_order_port",
        worst_inv_port,
        INVARIANCE_REL,
        "rel [NOT a control]",
    );

    // The slot order DOES matter for the bank, and that is measurable: at layer
    // 12 the snapshot sits last, and the first entry is a different tensor.
    let l12 = &o.tape[12];
    g.truth(
        "bank.snapshot_is_appended_last",
        bits_equal("snap last", &l12.bank_out[l12.nb_out - 1], &l12.layer_in),
        "L12 bank[-1] == layer_in, bit-exactly",
    );
    g.truth(
        "bank.snapshot_is_not_first",
        !bits_equal("snap first", &l12.bank_out[0], &l12.layer_in),
        "L12 bank[0] != layer_in — the ordering is discriminating here",
    );
}

// ---------------------------------------------------------------------------
// Section 6 — the chained depth state machine
// ---------------------------------------------------------------------------

fn section_chain<B: Backend>(g: &mut Gate, o: &Oracle, ck: &Checkpoint, dev: &Device<B>) {
    println!("  -- chain --");
    let cfg = &ck.cfg.text_config;
    let eps = cfg.rms_norm_eps;

    // The schedule comes from the config, via the model's own constructor. The
    // oracle ran a 13-layer prefix, so the mixer runs the first 13 entries of
    // that same schedule — and the gate asserts the truncation is a prefix of
    // what `from_config` produces rather than a second, hand-written rule.
    let full = DepthMixer::<B>::from_config(cfg).expect("config has attn_res_block_size");
    g.truth(
        "chain.schedule_from_config",
        full.schedule().len() == cfg.num_hidden_layers,
        format!("{} layers", full.schedule().len()),
    );
    let prefix: Vec<bool> = full.schedule()[..LAYERS].to_vec();
    let mut mixer = DepthMixer::<B>::new(prefix.clone());
    g.truth(
        "chain.schedule_is_config_prefix",
        mixer.schedule() == &full.schedule()[..LAYERS],
        format!(
            "crossings at {:?}",
            prefix
                .iter()
                .enumerate()
                .filter(|(_, b)| **b)
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        ),
    );

    let mut worst_entry_ulp = 0.0f64;
    let mut worst_mlp_ulp = 0.0f64;
    let mut bank_exact = true;
    let mut acc_exact = true;
    let mut layer_out_exact = true;
    let mut mix_presence_ok = true;
    let mut reset_nonvacuous = true;
    let mut min_reset_gap = f64::INFINITY;

    let mut hidden = t2::<B>(&o.tape[0].layer_in, TOKENS, HIDDEN, dev);

    for l in 0..LAYERS {
        let tape = &o.tape[l];
        let sa_params = params_of::<B>(o.param(&format!("L{l:02}.sa")), eps, dev);
        let mlp_params = params_of::<B>(o.param(&format!("L{l:02}.mlp")), eps, dev);

        let entry = mixer.enter_layer(hidden.clone(), &sa_params);

        // The mixture happens exactly when the bank is non-empty — measured
        // against whether the shipped forward emitted sa arrays at all.
        if entry.mix.is_some() != tape.has_sa_site {
            mix_presence_ok = false;
            g.truth(
                &format!("chain.L{l:02}.sa_mix_presence"),
                false,
                format!("port {} oracle {}", entry.mix.is_some(), tape.has_sa_site),
            );
        }
        worst_entry_ulp = worse(
            worst_entry_ulp,
            max_rel_diff(
                &format!("L{l:02} entry"),
                &host(entry.to_attention),
                &tape.input_layernorm_in,
            ),
        );

        // THE BANK, at the crossing — not merely the layer output.
        if mixer.bank_len() != tape.nb_out {
            bank_exact = false;
            g.truth(
                &format!("chain.L{l:02}.bank_len"),
                false,
                format!("port {} oracle {}", mixer.bank_len(), tape.nb_out),
            );
        } else {
            for (i, e) in mixer.bank().iter().enumerate() {
                if !bits_equal(
                    &format!("L{l:02} bank[{i}]"),
                    &host(e.clone()),
                    &tape.bank_out[i],
                ) {
                    bank_exact = false;
                    g.truth(&format!("chain.L{l:02}.bank[{i}]"), false, "differs");
                }
            }
        }

        let attn = t2::<B>(&tape.attn_out, TOKENS, HIDDEN, dev);
        let mix = mixer.after_attention(attn.clone(), &mlp_params);

        // THE RESET. The accumulator after attention is `layer_in + attn_out`
        // on an ordinary layer and `attn_out` alone on a checkpoint layer.
        let acc = host(
            mixer
                .accumulator()
                .expect("accumulator after attention")
                .clone(),
        );
        if !bits_equal(&format!("L{l:02} acc"), &acc, &tape.mlp_prefix_sum) {
            acc_exact = false;
            g.truth(
                &format!("chain.L{l:02}.accumulator"),
                false,
                format!(
                    "max |d| {:e}",
                    max_abs_diff("acc", &acc, &tape.mlp_prefix_sum)
                ),
            );
        }
        // ...and on a checkpoint layer the un-reset alternative must be a
        // materially different tensor, or "the reset happened" is vacuous.
        if prefix[l] {
            let unreset = host(round_bf16(hidden.clone() + attn));
            let rel = max_rel_diff("unreset", &unreset, &tape.mlp_prefix_sum);
            min_reset_gap = smaller(min_reset_gap, rel);
            if !(rel > CONTROL_MIN_REL) {
                reset_nonvacuous = false;
            }
        }

        worst_mlp_ulp = worse(
            worst_mlp_ulp,
            max_rel_diff(
                &format!("L{l:02} mlp"),
                &host(mix.out),
                &tape.post_attention_layernorm_in,
            ),
        );

        hidden = mixer.after_mlp(t2::<B>(&tape.mlp_out, TOKENS, HIDDEN, dev));
        let got = host(hidden.clone());
        if !bits_equal(&format!("L{l:02} out"), &got, &tape.layer_out) {
            layer_out_exact = false;
            g.truth(
                &format!("chain.L{l:02}.layer_out"),
                false,
                format!("max |d| {:e}", max_abs_diff("out", &got, &tape.layer_out)),
            );
        }
    }

    g.truth(
        "chain.sa_mix_presence",
        mix_presence_ok,
        "mixture iff bank non-empty, all 13 layers",
    );
    g.truth(
        "chain.bank_bitexact_every_layer",
        bank_exact,
        "13 layers, incl. both crossings",
    );
    g.truth(
        "chain.accumulator_bitexact_every_layer",
        acc_exact,
        "the reset, at every layer",
    );
    g.truth(
        "chain.reset_is_not_vacuous",
        reset_nonvacuous,
        "un-reset alternative differs",
    );
    g.gt(
        "chain.reset_gap",
        min_reset_gap,
        CONTROL_MIN_REL,
        "rel (min over crossings)",
    );
    g.truth(
        "chain.layer_out_bitexact",
        layer_out_exact,
        "13 layers chained",
    );
    g.le("chain.entry_mix", worst_entry_ulp, OUT_REL, "rel-to-max");
    g.le("chain.mlp_mix", worst_mlp_ulp, OUT_REL, "rel-to-max");

    // The model-level AttnRes, over the bank this run actually built.
    let model_site = o.site("MODEL.output");
    let out_params = params_of::<B>(o.param("MODEL.output"), eps, dev);
    g.truth(
        "chain.final_hidden_is_model_prefix_sum",
        bits_equal(
            "final hidden",
            &host(hidden.clone()),
            &model_site.prefix_sum,
        ),
        "the chained layer-12 output is what the output mixture accumulates",
    );
    let v = host(stack_candidates(mixer.bank(), hidden.clone()));
    g.truth(
        "chain.output_stack_bitexact",
        bits_equal("model v", &v, &model_site.v),
        format!(
            "{} slots (bank {} + accumulator)",
            model_site.slots,
            mixer.bank_len()
        ),
    );
    let fin = mixer.finish(hidden, &out_params);
    g.le(
        "chain.output_attn_res",
        max_rel_diff("model out", &host(fin.out), &o.final_norm_in),
        OUT_REL,
        "rel-to-max",
    );

    // The call-order guard must actually guard.
    let caught = expect_panic(AssertUnwindSafe(|| {
        let mut m = DepthMixer::<B>::new(vec![true, false]);
        let x = t2::<B>(&o.tape[0].layer_in, TOKENS, HIDDEN, dev);
        m.enter_layer(x.clone(), &out_params);
        m.enter_layer(x, &out_params); // out of order
    }));
    g.truth(
        "chain.call_order_guard_fires",
        caught,
        "two enter_layer in a row rejected",
    );
}

// ---------------------------------------------------------------------------
// Lanes
// ---------------------------------------------------------------------------

fn run_lane<B: Backend>(
    lane: &str,
    dev: &Device<B>,
    o: &Oracle,
    ck: &Checkpoint,
    verbose: bool,
) -> bool {
    println!("\n=== lane: {lane} ===");
    let mut g = Gate::new(lane, verbose);
    let eps = ck.cfg.text_config.rms_norm_eps;
    section_primitives::<B>(&mut g, dev);
    section_boundary(&mut g, o, ck);
    section_rounding::<B>(&mut g, o, dev);
    section_weights::<B>(&mut g, o, ck, dev);
    section_sites::<B>(&mut g, o, eps, dev);
    section_controls::<B>(&mut g, o, eps, dev);
    section_chain::<B>(&mut g, o, ck, dev);
    println!(
        "  lane {}: {} checks passed, {} failed",
        g.lane,
        g.passed,
        g.failures.len()
    );
    for f in &g.failures {
        println!("    - {f}");
    }
    g.passed_all()
}

fn main() -> Result<()> {
    let ck_dir = mary::paths::model(std::env::var("K3_CHECKPOINT").ok().as_deref(), "kimi-k3")?;
    let oracle_path = mary::paths::model(
        std::env::var("K3_LAYER_ORACLE").ok().as_deref(),
        "k3-oracle/layer_oracle_prefix13_bf16.npz",
    )?;
    let verbose = std::env::var("K3_GATE_VERBOSE").is_ok();

    println!("k3_attn_res_gate — mary::models::k3::attn_res vs the shipped AttnRes");
    println!("checkpoint: {}", ck_dir.display());
    println!("oracle:     {}", oracle_path.display());

    let mut ck = Checkpoint::open(&ck_dir)?;
    let z = Npz::open(&oracle_path).context("opening the layer oracle")?;
    println!("oracle arrays: {}", z.len());
    let o = load_oracle(&z, &mut ck)?;
    println!(
        "sites: {}  layers: {}  MEASURED checkpoint layers: {:?}",
        o.sites.len(),
        o.tape.len(),
        o.measured_checkpoints
    );

    let mut lanes: Vec<(&str, bool)> = Vec::new();
    {
        type Cpu = burn::backend::NdArray;
        let dev = Device::<Cpu>::default();
        lanes.push((
            "ndarray-cpu",
            run_lane::<Cpu>("ndarray-cpu", &dev, &o, &ck, verbose),
        ));
    }
    #[cfg(feature = "k3-attn-res-cuda")]
    {
        type Gpu = burn::backend::Cuda;
        let dev = Device::<Gpu>::default();
        lanes.push(("cuda", run_lane::<Gpu>("cuda", &dev, &o, &ck, verbose)));
    }
    #[cfg(not(feature = "k3-attn-res-cuda"))]
    println!(
        "\n=== lane: cuda === NOT COMPILED IN (build with --features k3-attn-res-cuda). \
         Nothing in this run is evidence about the CUDA backend."
    );

    println!("\n=== summary ===");
    for (n, r) in &lanes {
        println!("  {:<14} {}", n, if *r { "PASS" } else { "FAIL" });
    }
    // `all()` over an empty set is TRUE, so no lane at all must fail explicitly.
    let pass = !lanes.is_empty() && lanes.iter().all(|(_, r)| *r);
    if lanes.is_empty() {
        println!("  NO LANE RAN — nothing was verified.");
    }
    println!(
        "\nGATE: {}  ({} lane(s))",
        if pass { "PASS" } else { "FAIL" },
        lanes.len()
    );
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
