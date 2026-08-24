//! PersonaPlex-7B **depth transformer on cubecl** (Metal *and* CUDA) — the
//! realtime depformer, moved off the host.
//!
//! Same math as [`super::depth`] (the burn parity reference) and
//! [`super::depth_fast`] (the hand-written CPU predictor this replaces),
//! rebuilt as hand-launched cubecl kernels on the raw (non-fused) device, in
//! the same shape as the temporal stack's [`super::temporal_metal`]: the seven
//! per-layer matvecs go through that module's [`QLinear`] format dispatch
//! (q4 / q8 / f16), the norms ride as explicit f32 alpha arrays into fused
//! add+rms kernels, and every buffer is preallocated once.
//!
//! ## Why the GPU at all — the frame is pure weight bandwidth
//!
//! The depformer generates `dep_q = 16` codebooks per temporal frame
//! autoregressively, and every one of the 16 in-frame steps has its OWN weight
//! set (moshi's `multi_linear` row-slicing of `in_proj_weight` / `out_proj`
//! plus the `gating.{0..15}` ModuleList). So a frame touches **1.334 G weights
//! exactly once**, through 16 strictly sequential steps: 5.34 GB at f32,
//! 2.67 GB at f16, **1.42 GB at q8**, 0.75 GB at q4. Nothing is reused, no
//! arithmetic intensity to exploit — the only lever is bytes per weight and
//! the bandwidth of the engine that streams them. That is exactly the lever
//! `nn::q4` was built for and exactly the engine the CPU is worst at.
//!
//! ## Dispatch budget (the design constraint, measured)
//!
//! `depth_gpu_probe dispatch` measures the per-dispatch floor at **~6.3 µs**
//! on M4 Max (of which ~4.8 µs is host-side wgpu encode) and **~3.8 µs** on
//! GB10/CUDA, and this stack's matvecs are SMALL — at q8 a whole layer-step is
//! ~13.6 MB, i.e. ~43 µs of actual streaming. **That 13.6 MB / 43 µs is PER
//! LAYER-STEP, not per frame**: a frame is 96 layer-steps and streams the
//! 1.42 GB quoted above, so the frame is ~1.7x its own bandwidth floor, not
//! ~150x. (Reading those two numbers as one cost a whole diagnosis on
//! 2026-08-21.) Dispatch count is still a first-class cost, not bookkeeping —
//! it is 14-17% of a measured frame: the layout below is 7 dispatches per
//! layer-step (qkv, attn, o, add+rms, gate‖up+SwiGLU, down, add[+rms]) = 42,
//! plus 3 per step (input build + norm, logit head, argmax/select) and ONE
//! `depformer_in` matvec for the whole frame: `16 · 45 + 1 = 721`. Every
//! fusion in that list exists to delete a floor-priced dispatch, not to save
//! arithmetic. The 721 is confirmed by `nsys`: 115,360 `cuLaunchKernel` over
//! 160 frames, as 401 matvec + 192 add+rms + 96 attn + 16 input + 16 argmax.
//!
//! ## The autoregressive chain never leaves the device
//!
//! Step `s`'s input embedding is indexed by step `s-1`'s *chosen token*. Doing
//! that on the host would mean 16 blocking readbacks per frame — at Metal
//! sync cost that is the whole budget. So the chosen token lives in a device
//! slot: [`dep_argmax_kernel`] writes both the emitted token and the `prev`
//! slot the next [`dep_input_kernel`] gathers its embedding row with, and the
//! host reads back exactly once per frame (the 16 tokens; the 16×2048 logit
//! rows only when a caller asks). The host-side forcing rule (moshi's
//! `LMGen` cache providing the target, or a teacher trajectory) collapses to
//! one `[16]` u32 array uploaded at frame start, `NO_FORCE` where the step is
//! free.
//!
//! ## Codebook depth is a parameter, not a constant
//!
//! [`DepthGpu::frame`] takes `n_q` and runs exactly that many steps. Because
//! the weight sets are per-step, running fewer steps *skips their weights
//! entirely* — cost is proportional, including the stacked `depformer_in`
//! matvec, whose row-major layout makes a step prefix a byte prefix
//! ([`QLinear::forward_rows`]). Adapting depth per frame is therefore a
//! scheduling decision on top of this port, with no kernel work left to do.
//!
//! ## What is exact and what is not
//!
//! The attention scale 1/√64 = 2⁻³ is a power of two and rides on q inside
//! [`dep_attn_kernel`] (exact in f32). The window is moshi's effective 15
//! (see the `RingKVCache` analysis in [`super::depth`]) expressed as a
//! visible range `max(0, s-14)..=s`, softmax-identical to a mask. The KV
//! cache is **f32** here — unlike the temporal stack's f16 cache, because the
//! depth cache is 16 slots (1.5 MB total), so there is no bandwidth argument
//! for shrinking it. Quantization of the matvec weights is a real numerics
//! change and this module does not claim token-exactness against the CPU
//! path; `depth_gpu_probe gate` reports per-codebook logit cosine and
//! codebook-token agreement against [`super::depth_fast`] on identical
//! inputs.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::config as cfg;
use super::temporal_metal::{as_bytes, encode, QLinear, WeightFmt};
use crate::nn::q4::{self, Rt};
use crate::nn::weight_loader::WeightLoader;

const D: usize = cfg::DEP_DIM; // 1024
const FH: usize = cfg::DEP_FFN_HIDDEN; // 2816
const HEADS: u32 = cfg::DEP_HEADS as u32; // 16
const HD: u32 = cfg::DEP_HEAD_DIM as u32; // 64
const STEPS: usize = cfg::DEP_Q; // 16
const LAYERS: usize = cfg::DEP_LAYERS; // 6
const CARD: usize = cfg::CARD; // 2048
/// Effective attention window — moshi's dead `depformer_context` knob plus the
/// `RingKVCache` wrap off-by-one ≡ a sliding window of capacity − 1 = 15.
const WINDOW: usize = cfg::WEIGHTS_PER_STEP - 1;
const EPS: f32 = 1e-8; // cfg::RMS_EPS
/// `forced[s]` sentinel: this step's prev-token is the step's own choice.
pub const NO_FORCE: u32 = u32::MAX;

const NORM_THREADS: u32 = 256;
const ARGMAX_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

/// The in-frame step's input, fused with layer 0's `norm1`:
/// `x = depformer_in.{s}(transformer_out) + emb(prev)`, then
/// `y = x · alpha · rsqrt(mean(x²) + eps)`.
///
/// The conditioning half is already computed — all 16 `depformer_in`
/// projections batch into one frame-start matvec because `transformer_out` is
/// constant across the in-frame steps — so this kernel only adds the
/// embedding row. `prev` is a DEVICE slot (see module docs): `emb` is bound to
/// `depformer_text_emb` at step 0 and `depformer_emb.{s-1}` after, so the
/// table choice stays a host-side binding and the row index stays on device.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn dep_input_kernel(
    cond: &Array<f32>,
    emb: &Array<f32>,
    prev: &Array<u32>,
    alpha: &Array<f32>,
    x: &mut Array<f32>,
    y: &mut Array<f32>,
    step: u32,
    eps: f32,
    #[comptime] hidden: u32,
    #[comptime] cube_dim: u32,
) {
    let i = UNIT_POS_X;
    let cbase = step * hidden;
    let ebase = prev[0] * hidden;
    let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    let mut acc = f32::new(0.0);
    let mut k = i;
    while k < hidden {
        let v = cond[(cbase + k) as usize] + emb[(ebase + k) as usize];
        x[k as usize] = v;
        acc += v * v;
        k += cube_dim;
    }
    red[i as usize] = acc;
    sync_cube();
    let mut stride = u32::new((cube_dim / 2) as i64);
    while stride > 0 {
        if i < stride {
            red[i as usize] = red[i as usize] + red[(i + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let s = 1.0 / (red[0] / (hidden as f32) + eps).sqrt();
    let mut k = i;
    while k < hidden {
        y[k as usize] = x[k as usize] * s * alpha[k as usize];
        k += cube_dim;
    }
}

/// Residual add fused with the CONSUMING weighted RMS: `x += delta;
/// y = x · alpha · rsqrt(mean(x²) + eps)`. One cube — each thread owns its
/// element subset for both the add and the reduction, so there is no
/// cross-thread hazard.
///
/// `norm = false` is the last layer's MLP add: the depformer has **no final
/// norm** (`linears.{s}` applies straight to the residual stream), so that
/// dispatch is a bare add and the whole reduction compiles out.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn add_rms_kernel(
    x: &mut Array<f32>,
    delta: &Array<f32>,
    alpha: &Array<f32>,
    y: &mut Array<f32>,
    eps: f32,
    #[comptime] hidden: u32,
    #[comptime] cube_dim: u32,
    #[comptime] norm: bool,
) {
    let i = UNIT_POS_X;
    let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    let mut acc = f32::new(0.0);
    let mut k = i;
    while k < hidden {
        let v = x[k as usize] + delta[k as usize];
        x[k as usize] = v;
        acc += v * v;
        k += cube_dim;
    }
    if comptime![norm] {
        red[i as usize] = acc;
        sync_cube();
        let mut stride = u32::new((cube_dim / 2) as i64);
        while stride > 0 {
            if i < stride {
                red[i as usize] = red[i as usize] + red[(i + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        let s = 1.0 / (red[0] / (hidden as f32) + eps).sqrt();
        let mut k = i;
        while k < hidden {
            y[k as usize] = x[k as usize] * s * alpha[k as usize];
            k += cube_dim;
        }
    }
}

/// In-frame attention for one layer-step: cache write, scores, softmax and
/// weighted-V in ONE dispatch, one cube per head, `HD` threads.
///
/// The whole in-frame sequence is at most 16 positions of 64 dims, so the
/// temporal stack's split-K flash-decoding shape would be pure overhead here;
/// the entire head fits one cube and the scores fit shared memory. The
/// visible range is `lo..=step` (the effective window 15, see module docs) and
/// `q_scale` is the exact 2⁻³ attention scale.
///
/// The current slot is read directly from `qkv`, not back through `kc`/`vc`.
/// `sync_cube` lowers to a workgroup-memory barrier on WGSL and does not make
/// one thread's storage write visible to another. Cache writes are therefore
/// pure output here and are consumed only by later dispatches.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn dep_attn_kernel(
    qkv: &Array<f32>,
    kc: &mut Array<f32>,
    vc: &mut Array<f32>,
    out: &mut Array<f32>,
    layer: u32,
    step: u32,
    lo: u32,
    q_scale: f32,
    #[comptime] hd: u32,
    #[comptime] hidden: u32,
    #[comptime] steps: u32,
) {
    let i = UNIT_POS_X;
    let h = CUBE_POS_X;
    let dim = h * hd + i;
    let slot = (layer * steps + step) * hidden;
    kc[(slot + dim) as usize] = qkv[(hidden + dim) as usize];
    vc[(slot + dim) as usize] = qkv[(2 * hidden + dim) as usize];

    let mut qsh = SharedMemory::<f32>::new(comptime!(hd as usize));
    let mut sc = SharedMemory::<f32>::new(comptime!(steps as usize));
    qsh[i as usize] = qkv[dim as usize] * q_scale;
    sync_cube();

    let n = step + 1 - lo;
    if i < n {
        let key_step = lo + i;
        let kb = (layer * steps + key_step) * hidden + h * hd;
        let live_kb = hidden + h * hd;
        let mut s = f32::new(0.0);
        if key_step < step {
            for j in 0..hd {
                s += qsh[j as usize] * kc[(kb + j) as usize];
            }
        } else {
            for j in 0..hd {
                s += qsh[j as usize] * qkv[(live_kb + j) as usize];
            }
        }
        sc[i as usize] = s;
    }
    sync_cube();

    // ≤16 scores: one thread's serial softmax beats any reduction here.
    if i == 0 {
        let mut m = f32::new(-3.40282e38);
        let mut t = u32::new(0);
        while t < n {
            if sc[t as usize] > m {
                m = sc[t as usize];
            }
            t += 1;
        }
        let mut sum = f32::new(0.0);
        let mut t = u32::new(0);
        while t < n {
            let p = (sc[t as usize] - m).exp();
            sc[t as usize] = p;
            sum += p;
            t += 1;
        }
        let mut t = u32::new(0);
        while t < n {
            sc[t as usize] = sc[t as usize] / sum;
            t += 1;
        }
    }
    sync_cube();

    let mut acc = f32::new(0.0);
    let mut t = u32::new(0);
    while t + 1 < n {
        acc += sc[t as usize] * vc[((layer * steps + lo + t) * hidden + h * hd + i) as usize];
        t += 1;
    }
    acc += sc[(n - 1) as usize] * qkv[(2 * hidden + dim) as usize];
    out[dim as usize] = acc;
}

/// Greedy select + the prev-token chain, in one cube.
///
/// Reads the step's logit row, copies it into the frame's `[16, CARD]` slab
/// (free — the row is already in registers), argmaxes it FIRST-INDEX-WINS
/// (torch's CPU tie behavior, which [`super::depth::argmax`] also implements:
/// the reduction breaks ties by lower index, not by lower thread id), writes
/// the emitted token, and advances `prev` — `forced[step]` when the host
/// pinned this step's prev-token, else the step's own choice.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn dep_argmax_kernel(
    lrow: &Array<f32>,
    logits: &mut Array<f32>,
    forced: &Array<u32>,
    tok: &mut Array<u32>,
    prev: &mut Array<u32>,
    step: u32,
    no_force: u32,
    #[comptime] card: u32,
    #[comptime] cube_dim: u32,
) {
    let i = UNIT_POS_X;
    let base = step * card;
    let mut bv = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    let mut bi = SharedMemory::<u32>::new(comptime!(cube_dim as usize));
    let mut best_v = f32::new(-3.40282e38);
    let mut best_i = u32::new(card as i64);
    let mut k = i;
    while k < card {
        let v = lrow[k as usize];
        logits[(base + k) as usize] = v;
        if v > best_v {
            best_v = v;
            best_i = k;
        }
        k += cube_dim;
    }
    bv[i as usize] = best_v;
    bi[i as usize] = best_i;
    sync_cube();
    let mut stride = u32::new((cube_dim / 2) as i64);
    while stride > 0 {
        if i < stride {
            let ov = bv[(i + stride) as usize];
            let oi = bi[(i + stride) as usize];
            if ov > bv[i as usize] {
                bv[i as usize] = ov;
                bi[i as usize] = oi;
            } else if ov == bv[i as usize] {
                if oi < bi[i as usize] {
                    bi[i as usize] = oi;
                }
            }
        }
        sync_cube();
        stride /= 2;
    }
    if i == 0 {
        let t = bi[0];
        tok[step as usize] = t;
        let f = forced[step as usize];
        if f == no_force {
            prev[0] = t;
        } else {
            prev[0] = f;
        }
    }
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------

/// One layer's four matvec weights for ONE in-frame step. `norm1`/`norm2` are
/// SHARED across the 16 steps (only the projections and the FFN are per-step),
/// so the alphas live once per layer in [`DepthGpu::norms`].
struct LayerW {
    /// `[3·1024, 1024]` rows q‖k‖v — moshi's `in_proj_weight` row-block for
    /// this step, unfolded (alphas ride into the norm kernels).
    qkv: QLinear,
    /// `[1024, 1024]` `out_proj` row-block for this step.
    o: QLinear,
    /// `[2·2816, 1024]` gate‖up with rows INTERLEAVED (even `2j` = gate_j,
    /// odd `2j+1` = up_j) — the layout the fused SwiGLU epilogue reads.
    gate_up: QLinear,
    /// `[1024, 2816]`.
    down: QLinear,
}

struct StepW {
    layers: Vec<LayerW>,
    /// `linears.{s}` `[2048, 1024]` — reads the raw residual (no final norm).
    head: QLinear,
}

/// Per-matvec weight format for the depth stack.
///
/// A depth frame is ~73% pure weight streaming (measured: 7.12 ms of a 9.77 ms
/// q8 frame on GB10), so format is expressible per tensor, not just per model.
/// The MLP pair carries most of the bytes: at q8 a layer-step streams 13.6 MB,
/// of which `gate_up` is 6.13 MB (45%) and `down` 3.06 MB (22%), against
/// `qkv`'s 3.34 MB (25%) and `o`'s 1.11 MB (8%).
///
/// ## The obvious use of this knob is MEASURED AND REFUTED — default to q8
///
/// The reason to want per-tensor formats is the hypothesis that some block is
/// quantization-tolerant, so its bytes can go to q4 while the sensitive blocks
/// stay q8. For this stack that hypothesis is FALSE. `gate --frames 6` against
/// `depth_fast` on the real weights, free-running codebook agreement (the
/// metric that decides — teacher-forced flatters, because forcing resets the
/// chain every step):
///
/// | format          | frame bytes | free-running | GB10 ms | M4 Max ms |
/// |-----------------|-------------|--------------|---------|-----------|
/// | `q8`            | 1.42 GB     | **94.79%**   | 9.77    | 7.10      |
/// | `q8:gate_up=q4` | 1.14 GB     | 77.08%       | 9.31    | 6.90      |
/// | `q8:mlp=q4`     | 1.00 GB     | 63.54%       | 8.05    | 7.02      |
/// | `q4`            | 0.75 GB     | 58.33%       | 6.95    | 6.94      |
///
/// The damage tracks the FRACTION OF WEIGHT MASS moved to q4 and barely
/// notices which tensor moved: -17.7 points for 20% of frame bytes, -31.3 for
/// 30%, -36.5 for 47%. There is no tolerant sub-block to exploit, and the best
/// trade on offer (`q8:mlp=q4`, 1.21x on CUDA) costs 31 points of free-running
/// agreement — a different model, not a faster one. On Metal it buys nothing
/// at all, because that frame is fixed-cost-dominated (~4.9 ms of its 7.10 ms
/// is dispatch, not bytes). **Uniform q8 is the only format that gates.**
///
/// The knob stays because it is the instrument that produced that table in one
/// command each, and because the next reader will have the same idea.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthFmt {
    /// `self_attn.in_proj_weight` row-block, `[3·1024, 1024]`.
    pub qkv: WeightFmt,
    /// `self_attn.out_proj.weight` row-block, `[1024, 1024]`.
    pub o: WeightFmt,
    /// `gating.{s}.linear_in.weight` interleaved, `[2·2816, 1024]`.
    pub gate_up: WeightFmt,
    /// `gating.{s}.linear_out.weight`, `[1024, 2816]`.
    pub down: WeightFmt,
    /// `linears.{s}.weight`, `[2048, 1024]`.
    pub head: WeightFmt,
    /// All 16 `depformer_in.{s}` stacked, `[16·1024, 4096]`.
    pub dep_in: WeightFmt,
}

impl DepthFmt {
    /// Every matvec in the same format — the historical whole-model choice.
    pub const fn uniform(f: WeightFmt) -> Self {
        Self {
            qkv: f,
            o: f,
            gate_up: f,
            down: f,
            head: f,
            dep_in: f,
        }
    }

    /// `"<base>"` or `"<base>:<field>=<fmt>[,<field>=<fmt>…]"`, where `base`
    /// and `fmt` are `q4` / `q8` / `f16` and `field` is one of `qkv`, `o`,
    /// `gate_up`, `down`, `head`, `dep_in`, or the group alias `mlp`
    /// (= `gate_up` + `down`). So `q8` is uniform q8 and `q8:mlp=q4` is the
    /// q4-MLP variant.
    pub fn parse(spec: &str) -> Option<Self> {
        let (base, rest) = match spec.split_once(':') {
            Some((b, r)) => (b, Some(r)),
            None => (spec, None),
        };
        let mut out = Self::uniform(WeightFmt::parse(base)?);
        for item in rest.into_iter().flat_map(|r| r.split(',')) {
            let (field, f) = item.split_once('=')?;
            let f = WeightFmt::parse(f)?;
            match field {
                "qkv" => out.qkv = f,
                "o" => out.o = f,
                "gate_up" => out.gate_up = f,
                "down" => out.down = f,
                "mlp" => {
                    out.gate_up = f;
                    out.down = f;
                }
                "head" => out.head = f,
                "dep_in" => out.dep_in = f,
                _ => return None,
            }
        }
        Some(out)
    }

    /// The shortest spec that round-trips through [`Self::parse`]: the
    /// majority format as the base, then the fields that differ from it.
    pub fn label(&self) -> String {
        let fields = self.fields();
        let base = *fields
            .iter()
            .map(|(_, f)| f)
            .max_by_key(|f| fields.iter().filter(|(_, g)| g == *f).count())
            .expect("six fields");
        let diff: Vec<String> = fields
            .iter()
            .filter(|(_, f)| *f != base)
            .map(|(n, f)| format!("{n}={}", f.name()))
            .collect();
        if diff.is_empty() {
            base.name().to_string()
        } else {
            format!("{}:{}", base.name(), diff.join(","))
        }
    }

    fn fields(&self) -> [(&'static str, WeightFmt); 6] {
        [
            ("qkv", self.qkv),
            ("o", self.o),
            ("gate_up", self.gate_up),
            ("down", self.down),
            ("head", self.head),
            ("dep_in", self.dep_in),
        ]
    }
}

/// The depth transformer resident on the GPU. Load once ([`Self::load`]), then
/// [`Self::frame`] per temporal frame — no per-frame device allocation beyond
/// the two tiny host→device uploads (`prev` seed and the `[16]` forcing array).
pub struct DepthGpu {
    client: q4::Client,
    fmt: DepthFmt,
    steps: Vec<StepW>,
    /// Per layer: (`norm1.alpha`, `norm2.alpha`), `[1024]` f32 each.
    norms: Vec<(Handle, Handle)>,
    /// All 16 `depformer_in.{s}` stacked `[16·1024, 4096]` — ONE matvec per
    /// frame (`transformer_out` is constant across the in-frame steps).
    dep_in: QLinear,
    /// `depformer_text_emb [32001, 1024]` f32 — step 0's prev-token table.
    text_emb: Handle,
    /// `depformer_emb.{0..14} [2049, 1024]` f32 — step `s` embeds with `s-1`.
    audio_emb: Vec<Handle>,

    // fixed device scratch
    cond: Handle,   // [16·1024] f32
    x: Handle,      // [1024] residual stream
    xn: Handle,     // [1024] normed
    qkvb: Handle,   // [3·1024]
    attn: Handle,   // [1024]
    delta: Handle,  // [1024] o_proj / down_proj output feeding the residual add
    act: Handle,    // [2816] post-SwiGLU
    kc: Handle,     // [6·16·1024] f32 key slots
    vc: Handle,     // [6·16·1024] f32 value slots
    lrow: Handle,   // [2048] the step's logit row
    logits: Handle, // [16·2048] the frame's logit slab
    tok: Handle,    // [16] u32 emitted tokens
}

/// Interleave a `[2·FH, D]` gate‖up block (moshi ships gate rows then up rows)
/// into the fused-SwiGLU row order: even row `2j` = gate_j, odd `2j+1` = up_j.
fn interleave_gate_up(gu: &[f32]) -> Vec<f32> {
    assert_eq!(gu.len(), 2 * FH * D);
    let mut out = vec![0f32; 2 * FH * D];
    for j in 0..FH {
        out[(2 * j) * D..(2 * j + 1) * D].copy_from_slice(&gu[j * D..(j + 1) * D]);
        out[(2 * j + 1) * D..(2 * j + 2) * D].copy_from_slice(&gu[(FH + j) * D..(FH + j + 1) * D]);
    }
    out
}

impl DepthGpu {
    /// Load the depformer from `loader`, encode each matvec weight in its
    /// [`DepthFmt`] slot and upload to the default device. Weight residency at
    /// a uniform format: ~1.42 GB (q8), ~0.75 GB (q4), ~2.67 GB (f16), plus
    /// 257 MB of f32 embedding tables; a mixed format lands in between.
    ///
    /// Weights are read, encoded and uploaded one layer-step at a time, so
    /// peak host memory is one layer's f32 working set (~800 MB) rather than
    /// the whole 5.3 GB. Measured end-to-end from the pile: ~26 s at q8.
    pub fn load(loader: &WeightLoader, fmt: DepthFmt) -> Self {
        let client = q4::client_for_default_device();
        let n = cfg::WEIGHTS_PER_STEP;
        let mut steps: Vec<StepW> = (0..n)
            .map(|_| StepW {
                layers: Vec::with_capacity(LAYERS),
                head: QLinear::F16 {
                    w: client.empty(4),
                    out_dim: 0,
                    in_dim: 0,
                },
            })
            .collect();
        let mut norms: Vec<(Handle, Handle)> = Vec::with_capacity(LAYERS);

        for i in 0..LAYERS {
            let src = format!("depformer.layers.{i}");
            let (in_proj, s) = loader.load_f32(&format!("{src}.self_attn.in_proj_weight"));
            assert_eq!(s, vec![n * 3 * D, D], "{src}: in_proj_weight shape");
            let (out_proj, s) = loader.load_f32(&format!("{src}.self_attn.out_proj.weight"));
            assert_eq!(s, vec![n * D, D], "{src}: out_proj shape");
            let (a1, s) = loader.load_f32(&format!("{src}.norm1.alpha"));
            assert_eq!(s, vec![1, 1, D], "{src}: norm1.alpha shape");
            let (a2, s) = loader.load_f32(&format!("{src}.norm2.alpha"));
            assert_eq!(s, vec![1, 1, D], "{src}: norm2.alpha shape");
            norms.push((
                client.create_from_slice(as_bytes(&a1)),
                client.create_from_slice(as_bytes(&a2)),
            ));

            for (t, step) in steps.iter_mut().enumerate() {
                let qkv = in_proj[t * 3 * D * D..(t + 1) * 3 * D * D].to_vec();
                let o = out_proj[t * D * D..(t + 1) * D * D].to_vec();
                let (gu, s) = loader.load_f32(&format!("{src}.gating.{t}.linear_in.weight"));
                assert_eq!(s, vec![2 * FH, D], "{src}: gating.{t}.linear_in shape");
                let gu = interleave_gate_up(&gu);
                let (down, s) = loader.load_f32(&format!("{src}.gating.{t}.linear_out.weight"));
                assert_eq!(s, vec![D, FH], "{src}: gating.{t}.linear_out shape");

                step.layers.push(LayerW {
                    qkv: upload(&client, &qkv, 3 * D, D, fmt.qkv),
                    o: upload(&client, &o, D, D, fmt.o),
                    gate_up: upload(&client, &gu, 2 * FH, D, fmt.gate_up),
                    down: upload(&client, &down, D, FH, fmt.down),
                });
            }
        }

        let mut dep_in_all: Vec<f32> = Vec::with_capacity(n * D * cfg::DIM);
        for (t, step) in steps.iter_mut().enumerate() {
            let (w, s) = loader.load_f32(&format!("depformer_in.{t}.weight"));
            assert_eq!(s, vec![D, cfg::DIM], "depformer_in.{t} shape");
            dep_in_all.extend_from_slice(&w);
            let (h, s) = loader.load_f32(&format!("linears.{t}.weight"));
            assert_eq!(s, vec![CARD, D], "linears.{t} shape");
            step.head = upload(&client, &h, CARD, D, fmt.head);
        }
        let dep_in = upload(&client, &dep_in_all, n * D, cfg::DIM, fmt.dep_in);
        drop(dep_in_all);

        let (text_emb, s) = loader.load_f32("depformer_text_emb.weight");
        assert_eq!(s, vec![cfg::TEXT_VOCAB, D], "depformer_text_emb shape");
        let text_emb = client.create_from_slice(as_bytes(&text_emb));
        let audio_emb: Vec<Handle> = (0..n - 1)
            .map(|t| {
                let (w, s) = loader.load_f32(&format!("depformer_emb.{t}.weight"));
                assert_eq!(s, vec![cfg::AUDIO_VOCAB, D], "depformer_emb.{t} shape");
                client.create_from_slice(as_bytes(&w))
            })
            .collect();

        Self::assemble(client, fmt, steps, norms, dep_in, text_emb, audio_emb)
    }

    /// Deterministic synthetic weights at the real shapes — pure-timing
    /// construction for format sweeps without the 5.3 GB pile read. The
    /// bandwidth and dispatch behaviour is value-independent; the numerics
    /// are meaningless.
    pub fn synthetic(fmt: DepthFmt) -> Self {
        let client = q4::client_for_default_device();
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut fill = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((state >> 40) as f32) / (1u32 << 24) as f32 - 0.5) * 0.04
                })
                .collect()
        };
        let steps: Vec<StepW> = (0..STEPS)
            .map(|_| StepW {
                layers: (0..LAYERS)
                    .map(|_| LayerW {
                        qkv: upload(&client, &fill(3 * D * D), 3 * D, D, fmt.qkv),
                        o: upload(&client, &fill(D * D), D, D, fmt.o),
                        gate_up: upload(&client, &fill(2 * FH * D), 2 * FH, D, fmt.gate_up),
                        down: upload(&client, &fill(D * FH), D, FH, fmt.down),
                    })
                    .collect(),
                head: upload(&client, &fill(CARD * D), CARD, D, fmt.head),
            })
            .collect();
        let norms = (0..LAYERS)
            .map(|_| {
                (
                    client.create_from_slice(as_bytes(&fill(D))),
                    client.create_from_slice(as_bytes(&fill(D))),
                )
            })
            .collect();
        let dep_in = upload(
            &client,
            &fill(STEPS * D * cfg::DIM),
            STEPS * D,
            cfg::DIM,
            fmt.dep_in,
        );
        let text_emb = client.create_from_slice(as_bytes(&fill(cfg::TEXT_VOCAB * D)));
        let audio_emb = (0..STEPS - 1)
            .map(|_| client.create_from_slice(as_bytes(&fill(cfg::AUDIO_VOCAB * D))))
            .collect();
        Self::assemble(client, fmt, steps, norms, dep_in, text_emb, audio_emb)
    }

    fn assemble(
        client: q4::Client,
        fmt: DepthFmt,
        steps: Vec<StepW>,
        norms: Vec<(Handle, Handle)>,
        dep_in: QLinear,
        text_emb: Handle,
        audio_emb: Vec<Handle>,
    ) -> Self {
        let f32s = |n: usize| client.empty(n * 4);
        Self {
            cond: f32s(STEPS * D),
            x: f32s(D),
            xn: f32s(D),
            qkvb: f32s(3 * D),
            attn: f32s(D),
            delta: f32s(D),
            act: f32s(FH),
            kc: f32s(LAYERS * STEPS * D),
            vc: f32s(LAYERS * STEPS * D),
            lrow: f32s(CARD),
            logits: f32s(STEPS * CARD),
            tok: client.empty(STEPS * 4),
            client,
            fmt,
            steps,
            norms,
            dep_in,
            text_emb,
            audio_emb,
        }
    }

    /// Weight bytes a full 16-codebook frame streams (every per-step matvec
    /// exactly once, plus the stacked conditioning projection).
    pub fn frame_weight_bytes(&self) -> usize {
        let per_step: usize = self.steps[0]
            .layers
            .iter()
            .map(|l| l.qkv.bytes() + l.o.bytes() + l.gate_up.bytes() + l.down.bytes())
            .sum::<usize>()
            + self.steps[0].head.bytes();
        STEPS * per_step + self.dep_in.bytes()
    }

    pub fn fmt(&self) -> DepthFmt {
        self.fmt
    }

    /// Submit one temporal frame's depformer pass — same contract as
    /// [`super::depth::DepthTransformer::frame`] / [`super::depth_fast::DepthFast::frame`]:
    /// `transformer_out` is the post-`out_norm` temporal hidden `[4096]`,
    /// `text_token` the frame's `next_text_token`, and `forced[s]` pins step
    /// `s`'s contribution to the prev-token chain ([`NO_FORCE`] = the step's
    /// own choice). A teacher trajectory is expressed the same way — the
    /// forcing rule is `forced[s]` first, so the host collapses forcing and
    /// teacher-forcing into this one array.
    ///
    /// `n_q` in-frame steps are generated (`n_q <= 16`); the weight sets are
    /// per-step, so the cost is proportional. NON-BLOCKING — read the result
    /// with [`Self::tokens`] / [`Self::logits`].
    pub fn frame_submit(
        &mut self,
        transformer_out: &[f32],
        text_token: u32,
        forced: &[u32],
        n_q: usize,
    ) {
        assert_eq!(transformer_out.len(), cfg::DIM);
        assert!(n_q <= STEPS && n_q > 0, "n_q {n_q}");
        assert_eq!(forced.len(), STEPS, "forced length");
        assert!(
            (text_token as usize) < cfg::TEXT_VOCAB,
            "text token {text_token}"
        );
        let c = &self.client;
        let x_in = c.create_from_slice(as_bytes(transformer_out));
        let forced_d = c.create_from_slice(as_bytes(forced));
        let prev = c.create_from_slice(as_bytes(&[text_token]));
        let arr = |h: &Handle, n: usize| unsafe { ArrayArg::from_raw_parts(h.clone(), n) };

        // All n_q conditioning projections in ONE matvec: `transformer_out` is
        // constant across the in-frame steps, and the stacked weight is
        // step-major so n_q steps are a row prefix.
        self.dep_in.forward_rows(c, &x_in, &self.cond, n_q * D);

        for s in 0..n_q {
            let emb = if s == 0 {
                &self.text_emb
            } else {
                &self.audio_emb[s - 1]
            };
            let evocab = if s == 0 {
                cfg::TEXT_VOCAB
            } else {
                cfg::AUDIO_VOCAB
            };
            unsafe {
                dep_input_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_single(),
                    CubeDim::new_1d(NORM_THREADS),
                    arr(&self.cond, STEPS * D),
                    arr(emb, evocab * D),
                    arr(&prev, 1),
                    arr(&self.norms[0].0, D),
                    arr(&self.x, D),
                    arr(&self.xn, D),
                    s as u32,
                    EPS,
                    D as u32,
                    NORM_THREADS,
                );
            }

            let lo = (s + 1).saturating_sub(WINDOW) as u32;
            let step_w = &self.steps[s];
            for (li, l) in step_w.layers.iter().enumerate() {
                l.qkv.forward(c, &self.xn, &self.qkvb);
                unsafe {
                    dep_attn_kernel::launch_unchecked::<Rt>(
                        c,
                        CubeCount::new_1d(HEADS),
                        CubeDim::new_1d(HD),
                        arr(&self.qkvb, 3 * D),
                        arr(&self.kc, LAYERS * STEPS * D),
                        arr(&self.vc, LAYERS * STEPS * D),
                        arr(&self.attn, D),
                        li as u32,
                        s as u32,
                        lo,
                        (cfg::DEP_HEAD_DIM as f32).powf(-0.5),
                        HD,
                        D as u32,
                        STEPS as u32,
                    );
                }
                l.o.forward(c, &self.attn, &self.delta);
                self.add_rms(&self.norms[li].1, true);
                l.gate_up.forward_swiglu(c, &self.xn, &self.act);
                l.down.forward(c, &self.act, &self.delta);
                // The last layer's MLP add is a BARE add: `linears.{s}` reads
                // the raw residual stream (the depformer has no final norm).
                let last = li + 1 == LAYERS;
                let alpha = if last {
                    &self.norms[li].1
                } else {
                    &self.norms[li + 1].0
                };
                self.add_rms(alpha, !last);
            }

            step_w.head.forward(c, &self.x, &self.lrow);
            unsafe {
                dep_argmax_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_single(),
                    CubeDim::new_1d(ARGMAX_THREADS),
                    arr(&self.lrow, CARD),
                    arr(&self.logits, STEPS * CARD),
                    arr(&forced_d, STEPS),
                    arr(&self.tok, STEPS),
                    arr(&prev, 1),
                    s as u32,
                    NO_FORCE,
                    CARD as u32,
                    ARGMAX_THREADS,
                );
            }
        }
    }

    fn add_rms(&self, alpha: &Handle, norm: bool) {
        let c = &self.client;
        let arr = |h: &Handle, n: usize| unsafe { ArrayArg::from_raw_parts(h.clone(), n) };
        unsafe {
            add_rms_kernel::launch_unchecked::<Rt>(
                c,
                CubeCount::new_single(),
                CubeDim::new_1d(NORM_THREADS),
                arr(&self.x, D),
                arr(&self.delta, D),
                arr(alpha, D),
                arr(&self.xn, D),
                EPS,
                D as u32,
                NORM_THREADS,
                norm,
            );
        }
    }

    /// Blocking readback of the frame's emitted tokens (the ONLY sync in a
    /// greedy frame).
    pub fn tokens(&self, n_q: usize) -> Vec<i64> {
        use cubecl::CubeElement;
        let bytes = self
            .client
            .read_one(self.tok.clone())
            .expect("tok readback");
        u32::from_bytes(&bytes)[..n_q]
            .iter()
            .map(|&t| t as i64)
            .collect()
    }

    /// Blocking readback of the frame's `[n_q, 2048]` logit rows — the gate's
    /// view, not the realtime path's.
    pub fn logits(&self, n_q: usize) -> Vec<f32> {
        use cubecl::CubeElement;
        let bytes = self
            .client
            .read_one(self.logits.clone())
            .expect("logits readback");
        f32::from_bytes(&bytes)[..n_q * CARD].to_vec()
    }

    /// Submit + read the emitted tokens.
    pub fn frame(
        &mut self,
        transformer_out: &[f32],
        text_token: u32,
        forced: &[u32],
        n_q: usize,
    ) -> Vec<i64> {
        self.frame_submit(transformer_out, text_token, forced, n_q);
        self.tokens(n_q)
    }
}

/// Encode + upload one row-major `[out, in]` f32 weight in `fmt`.
fn upload(
    client: &q4::Client,
    w: &[f32],
    out_dim: usize,
    in_dim: usize,
    fmt: WeightFmt,
) -> QLinear {
    assert_eq!(w.len(), out_dim * in_dim);
    let enc = encode(w, out_dim, in_dim, fmt);
    QLinear::upload(client, &enc, out_dim, in_dim, fmt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every spec [`DepthFmt::label`] emits parses back to the same value, and
    /// a uniform format labels as the bare format name.
    #[test]
    fn depth_fmt_specs_round_trip() {
        for spec in [
            "q4",
            "q8",
            "f16",
            "q8:gate_up=q4",
            "q8:gate_up=q4,down=q4",
            "q8:qkv=f16,gate_up=q4,down=q4",
        ] {
            let f = DepthFmt::parse(spec).expect(spec);
            assert_eq!(f.label(), spec);
            assert_eq!(DepthFmt::parse(&f.label()), Some(f));
        }
        // `mlp` is a group alias, so it canonicalizes to its two members.
        assert_eq!(
            DepthFmt::parse("q8:mlp=q4"),
            DepthFmt::parse("q8:gate_up=q4,down=q4")
        );
        assert_eq!(DepthFmt::parse("q8:mlp=q4").unwrap().qkv, WeightFmt::Q8);
        assert_eq!(DepthFmt::parse("q9"), None);
        assert_eq!(DepthFmt::parse("q8:nope=q4"), None);
        assert_eq!(DepthFmt::parse("q8:gate_up"), None);
    }

    /// The gate‖up interleave is the identity on content: even rows are the
    /// SiLU branch in source order, odd rows the up branch.
    #[test]
    fn interleave_gate_up_is_a_permutation() {
        let gu: Vec<f32> = (0..2 * FH * D).map(|i| i as f32).collect();
        let out = interleave_gate_up(&gu);
        for j in [0usize, 1, 7, FH / 2, FH - 1] {
            assert_eq!(&out[(2 * j) * D..(2 * j + 1) * D], &gu[j * D..(j + 1) * D]);
            assert_eq!(
                &out[(2 * j + 1) * D..(2 * j + 2) * D],
                &gu[(FH + j) * D..(FH + j + 1) * D]
            );
        }
    }
}
