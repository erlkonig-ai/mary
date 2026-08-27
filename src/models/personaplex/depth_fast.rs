//! PersonaPlex-7B depth transformer — the **fast CPU predictor** for the
//! realtime decode build (Lane B). Same math as [`super::depth`] (the burn
//! parity reference), rebuilt for the 80 ms/frame budget:
//!
//! - **All 16 per-step weight sets are pre-sliced at construction** (the
//!   per-frame `adapt_layer_steps` slicing + burn tensor building of depth.rs
//!   disappears): step-major flat row-major matrices with the folds applied
//!   once at load — `norm1.alpha` into the qkv rows, `norm2.alpha` into the
//!   gate‖up rows, the 1/√64 = 2⁻³ attention scale into the q rows (a power
//!   of two, exact in f32; depth.rs applies it to the activations instead,
//!   which is bit-identical modulo underflow).
//! - **Fixed preallocated buffers** for the whole in-frame loop: KV as
//!   `[6 layers × 16 slots × 1024]` flat slabs (window-15 mask as a visible
//!   range `max(0, s-14)..=s`, softmax-identical to depth.rs's mask — see the
//!   RingKVCache analysis in depth.rs), scratch activations, and a
//!   `[16 × 2048]` logits slab the gate reads back. Zero per-frame allocation.
//! - **The 16 `depformer_in` conditioning gemvs collapse to ONE**
//!   `[16·1024, 4096]` gemv per frame: `transformer_out` is constant across
//!   the in-frame steps, so all 16 projections batch at frame start.
//! - **Two weight-storage modes**, chosen at load:
//!   - `f32`: Accelerate `sgemv` through the row-parallel work-stealing pool
//!     (`qwen3tts::cpu::sgemv_mt`, 2 AMX streams — the measured ceiling).
//!   - `f16` storage with **f32 accumulate**: a hand NEON kernel
//!     (`fcvtl`+`fmla`: weights convert f16→f32 in-register, activations stay
//!     full f32, products accumulate in f32) on its own work-stealing
//!     row-chunk pool across P-cores (`MARY_DEPTH_THREADS`, default 6 —
//!     NEON has no AMX two-stream ceiling; each core adds bandwidth).
//!     The checkpoint is bf16 (8 mantissa bits), so f16 storage (10 mantissa
//!     bits) is EXACT for every weight with |w| ≥ 2⁻¹⁴ ≈ 6.1e-5 and only
//!     degrades gradually below (f16 denormals); the numerics stay in the
//!     depth.rs family — the gate demands ~1.0 cos + full argmax agreement,
//!     not a q4-style relaxed bar.
//!
//! The frame is pure weight bandwidth: 1.334 G weights touched exactly once
//! per frame (5.34 GB f32 / 2.67 GB f16) through 16 strictly sequential
//! steps — the same predictor shape that won on CPU for qwen3tts.
//!
//! Gate + bench: `moshi_depth_probe` (teacher-forced parity vs the oracle
//! goldens AND exact-token agreement vs `depth.rs`, then min-of-medians
//! frame timing).

use std::time::Instant;

use super::config as cfg;
use super::depth::argmax;
use super::sampling::Sampler;
use crate::models::qwen3tts::cpu::{self, sgemv_mt, softmax};
use crate::nn::weight_loader::{HostF32, WeightLoader};

const D: usize = cfg::DEP_DIM; // 1024
const FH: usize = cfg::DEP_FFN_HIDDEN; // 2816
const HEADS: usize = cfg::DEP_HEADS; // 16
const HD: usize = cfg::DEP_HEAD_DIM; // 64
const STEPS: usize = cfg::DEP_Q; // 16
const LAYERS: usize = cfg::DEP_LAYERS; // 6
/// Effective attention window (see depth.rs: dead `depformer_context` knob +
/// the RingKVCache wrap off-by-one ≡ sliding window of capacity − 1 = 15).
const WINDOW: usize = cfg::WEIGHTS_PER_STEP - 1;

// ────────────────────────── f16 gemv kernel + pool ──────────────────────────

/// `Σ f32(w16[i]) · x[i]` over `n` (multiple of 32): f16 weights converted
/// in-register (`fcvtl`), f32 activations, f32 accumulate — 32-way split
/// accumulation (4 vector accumulators × 4 lanes × 2 interleaves).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn hdot(w: *const u16, x: *const f32, n: usize) -> f32 {
    unsafe {
        debug_assert!(n >= 32 && n % 32 == 0);
        let out: f32;
        core::arch::asm!(
            "movi v0.16b, #0",
            "movi v1.16b, #0",
            "movi v2.16b, #0",
            "movi v3.16b, #0",
            "2:",
            "ldp  q4, q5, [{w}]",
            "ldp  q6, q7, [{w}, #32]",
            "add  {w}, {w}, #64",
            "ldp  q16, q17, [{x}]",
            "ldp  q18, q19, [{x}, #32]",
            "ldp  q20, q21, [{x}, #64]",
            "ldp  q22, q23, [{x}, #96]",
            "add  {x}, {x}, #128",
            "fcvtl  v24.4s, v4.4h",
            "fcvtl2 v25.4s, v4.8h",
            "fcvtl  v26.4s, v5.4h",
            "fcvtl2 v27.4s, v5.8h",
            "fmla v0.4s, v24.4s, v16.4s",
            "fmla v1.4s, v25.4s, v17.4s",
            "fmla v2.4s, v26.4s, v18.4s",
            "fmla v3.4s, v27.4s, v19.4s",
            "fcvtl  v24.4s, v6.4h",
            "fcvtl2 v25.4s, v6.8h",
            "fcvtl  v26.4s, v7.4h",
            "fcvtl2 v27.4s, v7.8h",
            "fmla v0.4s, v24.4s, v20.4s",
            "fmla v1.4s, v25.4s, v21.4s",
            "fmla v2.4s, v26.4s, v22.4s",
            "fmla v3.4s, v27.4s, v23.4s",
            "subs {n}, {n}, #32",
            "b.ne 2b",
            "fadd v0.4s, v0.4s, v1.4s",
            "fadd v2.4s, v2.4s, v3.4s",
            "fadd v0.4s, v0.4s, v2.4s",
            "faddp v0.4s, v0.4s, v0.4s",
            "faddp s0, v0.2s",
            w = inout(reg) w => _,
            x = inout(reg) x => _,
            n = inout(reg) n => _,
            out("v0") out,
            out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            options(nostack, readonly),
        );
        out
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn hdot(w: *const u16, x: *const f32, n: usize) -> f32 {
    let mut acc = 0f32;
    for i in 0..n {
        acc += half::f16::from_bits(*w.add(i)).to_f32() * *x.add(i);
    }
    acc
}

/// Serial `y = W·x` for row-major f16 `W: [m, n]`.
fn hgemv(w: &[u16], m: usize, n: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), m * n);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    for (r, yr) in y.iter_mut().enumerate() {
        *yr = unsafe { hdot(w.as_ptr().add(r * n), x.as_ptr(), n) };
    }
}

// Row-parallel work-stealing pool for the f16 kernel, mirroring the design
// findings of `qwen3tts::cpu` (fixed slices die by straggler on a shared
// machine; a generation-tagged CAS chunk grid repairs that). Unlike the
// Accelerate sgemv pool there is no AMX two-stream ceiling — the NEON kernel
// runs per-core, so width buys bandwidth until DRAM saturates.
// `MARY_DEPTH_THREADS` ways total (0/1 = serial), default 6.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

#[derive(Clone, Copy)]
struct HJob {
    w: *const u16,
    x: *const f32,
    y: *mut f32,
    m: usize,
    n: usize,
    chunk: usize,
}

struct HPool {
    epoch: AtomicUsize,
    /// Generation-tagged steal counter: `epoch << CHUNK_BITS | next_chunk`
    /// (claimed by CAS so a stale worker never eats a new job's chunk).
    next: AtomicUsize,
    done: AtomicUsize,
    stop: AtomicBool,
    job: std::cell::UnsafeCell<HJob>,
    lock: Mutex<()>,
    cv: Condvar,
    ways: usize,
}

// Job pointers are published strictly before the epoch bump (Release) and
// read strictly after (Acquire); they never outlive the dispatch call.
unsafe impl Sync for HPool {}
unsafe impl Send for HPool {}

const CHUNK_BITS: u32 = 20;

fn hsteal(pool: &HPool, job: &HJob, r#gen: usize) {
    let n_chunks = job.m.div_ceil(job.chunk);
    loop {
        let v = pool.next.load(Ordering::Acquire);
        let (g, i) = (v >> CHUNK_BITS, v & ((1 << CHUNK_BITS) - 1));
        if g != r#gen || i >= n_chunks {
            return;
        }
        if pool
            .next
            .compare_exchange_weak(v, v + 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            continue;
        }
        let start = i * job.chunk;
        let len = job.chunk.min(job.m - start);
        unsafe {
            let w = std::slice::from_raw_parts(job.w.add(start * job.n), len * job.n);
            let x = std::slice::from_raw_parts(job.x, job.n);
            let y = std::slice::from_raw_parts_mut(job.y.add(start), len);
            hgemv(w, len, job.n, x, y);
        }
        pool.done.fetch_add(1, Ordering::AcqRel);
    }
}

fn hworker(pool: &'static HPool) {
    cpu::set_interactive_qos();
    let mut seen = 0usize;
    loop {
        // spin briefly for the next job (in-frame gaps are µs), then park
        let mut spins = 0u32;
        loop {
            if pool.stop.load(Ordering::Acquire) {
                return;
            }
            let e = pool.epoch.load(Ordering::Acquire);
            if e != seen {
                seen = e;
                break;
            }
            spins += 1;
            if spins > 50_000 {
                let mut g = pool.lock.lock().unwrap();
                loop {
                    if pool.stop.load(Ordering::Acquire) {
                        return;
                    }
                    let e = pool.epoch.load(Ordering::Acquire);
                    if e != seen {
                        seen = e;
                        break;
                    }
                    g = pool.cv.wait(g).unwrap();
                }
                break;
            }
            std::hint::spin_loop();
        }
        let job = unsafe { *pool.job.get() };
        hsteal(pool, &job, seen);
    }
}

fn hpool() -> Option<&'static HPool> {
    static POOL: OnceLock<Option<&'static HPool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let ways = std::env::var("MARY_DEPTH_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(6)
            .min(12);
        if ways < 2 {
            return None;
        }
        let p: &'static HPool = Box::leak(Box::new(HPool {
            epoch: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job: std::cell::UnsafeCell::new(HJob {
                w: std::ptr::null(),
                x: std::ptr::null(),
                y: std::ptr::null_mut(),
                m: 0,
                n: 0,
                chunk: 1,
            }),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            ways,
        }));
        for i in 1..ways {
            std::thread::Builder::new()
                .name(format!("hgemv-pool-{i}"))
                .spawn(move || hworker(p))
                .expect("spawn hgemv pool worker");
        }
        Some(p)
    })
}

/// `y = W·x` like [`hgemv`], work-stealing 32-row-aligned chunks across the
/// pool. Deterministic: every row is one `hdot` over the same data regardless
/// of which thread claims its chunk.
fn hgemv_mt(w: &[u16], m: usize, n: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), m * n);
    let Some(pool) = hpool() else {
        return hgemv(w, m, n, x, y);
    };
    let chunk = (m.div_ceil(4 * pool.ways)).next_multiple_of(32).max(32);
    let n_chunks = m.div_ceil(chunk);
    static DISPATCH: Mutex<()> = Mutex::new(());
    let _d = DISPATCH.lock().unwrap();
    unsafe {
        *pool.job.get() = HJob {
            w: w.as_ptr(),
            x: x.as_ptr(),
            y: y.as_mut_ptr(),
            m,
            n,
            chunk,
        };
    }
    let r#gen = pool.epoch.load(Ordering::Relaxed) + 1;
    pool.done.store(0, Ordering::Release);
    pool.next.store(r#gen << CHUNK_BITS, Ordering::Release);
    pool.epoch.store(r#gen, Ordering::Release);
    {
        let _g = pool.lock.lock().unwrap();
        pool.cv.notify_all();
    }
    let job = unsafe { *pool.job.get() };
    hsteal(pool, &job, r#gen);
    while pool.done.load(Ordering::Acquire) < n_chunks {
        std::hint::spin_loop();
    }
}

// ─────────────────────────── weights + scalar math ──────────────────────────

/// One weight matrix in the selected in-memory storage width.
enum Mat {
    F32(Vec<f32>),
    /// f16 bit patterns (`half::f16::to_bits`), f32-accumulate NEON kernel.
    F16(Vec<u16>),
}

impl Mat {
    fn new(v: Vec<f32>, f16: bool) -> Self {
        if f16 {
            Mat::F16(
                v.into_iter()
                    .map(|x| half::f16::from_f32(x).to_bits())
                    .collect(),
            )
        } else {
            Mat::F32(v)
        }
    }

    /// `y = W·x`, row-major `[m, n]`, threaded.
    ///
    /// f32 goes through the Accelerate pool. Note `n` here reaches 2816
    /// (down) and 4096 (the fused dep_in) — beyond the pool's
    /// byte-transparency-verified n ≤ 2048 set. That verification concerned
    /// pool-on == pool-off BIT identity, which this predictor does not gate
    /// on (the bar is cos + argmax vs the goldens and depth.rs); within a
    /// fixed `MARY_PRED_THREADS` the chunk grid is fixed, so results stay
    /// deterministic run-to-run.
    fn gemv(&self, m: usize, n: usize, x: &[f32], y: &mut [f32]) {
        match self {
            Mat::F32(w) => sgemv_mt(w, m, n, x, y),
            Mat::F16(w) => hgemv_mt(w, m, n, x, y),
        }
    }

    fn bytes_per_elem(&self) -> usize {
        match self {
            Mat::F32(_) => 4,
            Mat::F16(_) => 2,
        }
    }
}

/// The per-step qkv row-block of a depformer layer with the load-time folds
/// applied: rows `[t·3D, (t+1)·3D)` of `in_proj [16·3D, D]`, `norm1.alpha`
/// folded into the columns, the exact 2⁻³ attention scale onto the q rows.
/// Kept as one named transform so future derived formats can reuse exactly the
/// same computation as [`DepthFast::load`].
pub(crate) fn fold_qkv_step(in_proj: &[f32], a1: &[f32], t: usize) -> Vec<f32> {
    let mut qkv = in_proj[t * 3 * D * D..(t + 1) * 3 * D * D].to_vec();
    for (r, row) in qkv.chunks_exact_mut(D).enumerate() {
        let qs = if r < D { 0.125f32 } else { 1.0 };
        for (w, &al) in row.iter_mut().zip(a1) {
            *w = *w * al * qs;
        }
    }
    qkv
}

/// One step's gate‖up rows with `norm2.alpha` folded into the columns — the
/// gate_up twin of [`fold_qkv_step`].
pub(crate) fn fold_gate_up(mut gu: Vec<f32>, a2: &[f32]) -> Vec<f32> {
    for row in gu.chunks_exact_mut(D) {
        for (w, &al) in row.iter_mut().zip(a2) {
            *w *= al;
        }
    }
    gu
}

/// Weightless RMS norm (`x · rsqrt(mean(x²) + eps)`) — the alpha weights are
/// folded into the consuming matmul rows, exactly like `layers::rms` +
/// `fold_in` in the burn path. f64 mean accumulation (matches `cpu::rms_norm`).
fn rms_ip(x: &[f32], eps: f64, out: &mut [f32]) {
    let mean: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let s = ((mean + eps).sqrt().recip()) as f32;
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v * s;
    }
}

/// f32 dot with 4-lane split accumulation (vectorizes; not bit-order-gated).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0f32; 4];
    for (ca, cb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for j in 0..4 {
            s[j] += ca[j] * cb[j];
        }
    }
    (s[0] + s[1]) + (s[2] + s[3])
}

struct LayerW {
    /// `[3·1024, 1024]` rows q‖k‖v — norm1.alpha folded into the columns,
    /// q rows scaled by 2⁻³ (the folded attention scale).
    qkv: Mat,
    /// `[1024, 1024]` out_proj row-block for this step.
    o: Mat,
    /// `[2·2816, 1024]` rows gate‖up — norm2.alpha folded into the columns.
    gate_up: Mat,
    /// `[1024, 2816]`.
    down: Mat,
}

struct StepW {
    layers: Vec<LayerW>,
    /// `linears.{s}` `[2048, 1024]` — reads the raw residual (no final norm).
    head: Mat,
}

/// The fast depformer predictor. Construct once ([`Self::load`]), then call
/// [`Self::frame`] per temporal frame — no per-frame allocation.
pub struct DepthFast {
    steps: Vec<StepW>,
    /// All 16 `depformer_in.{s}` stacked: `[16·1024, 4096]` — one gemv/frame.
    dep_in: Mat,
    /// `depformer_text_emb [32001, 1024]` — row lookups, kept f32 (16 rows of
    /// 4 KB per frame; storage width is irrelevant here). Owned on the
    /// materialize path.
    text_emb: HostF32,
    /// `depformer_emb.{0..14} [2049, 1024]`.
    audio_emb: Vec<HostF32>,

    // fixed scratch
    cond: Vec<f32>,       // [16·1024] per-step conditioning
    kc: Vec<f32>,         // [6·16·1024] key slots
    vc: Vec<f32>,         // [6·16·1024] value slots
    x: Vec<f32>,          // [1024] residual stream
    xin: Vec<f32>,        // [1024] normed input
    qkv_buf: Vec<f32>,    // [3·1024]
    attn: Vec<f32>,       // [1024]
    proj: Vec<f32>,       // [1024]
    gu: Vec<f32>,         // [2·2816]
    act: Vec<f32>,        // [2816]
    logits_buf: Vec<f32>, // [16·2048], valid after frame()

    // always-on timing decomposition (µs-scale overhead per frame)
    t_cond: f64,
    t_gemv: f64,
    t_head: f64,
    t_frame: f64,
    frames: u64,
}

impl DepthFast {
    /// Load + pre-slice all 16 per-step weight sets. `f16`: store matrix
    /// weights as f16 (f32 accumulate at compute).
    pub fn load(loader: &WeightLoader, f16: bool) -> Self {
        let n = cfg::WEIGHTS_PER_STEP;
        let mut steps: Vec<StepW> = Vec::with_capacity(n);
        for _ in 0..n {
            steps.push(StepW {
                layers: Vec::with_capacity(LAYERS),
                head: Mat::F32(Vec::new()),
            });
        }

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

            for (t, step) in steps.iter_mut().enumerate() {
                // qkv block: norm1 fold + exact 2⁻³ on the q rows
                let qkv = fold_qkv_step(&in_proj, &a1, t);
                let o = out_proj[t * D * D..(t + 1) * D * D].to_vec();

                let (gu, s) = loader.load_f32(&format!("{src}.gating.{t}.linear_in.weight"));
                assert_eq!(s, vec![2 * FH, D], "{src}: gating.{t}.linear_in shape");
                let gu = fold_gate_up(gu, &a2);
                let (down, s) = loader.load_f32(&format!("{src}.gating.{t}.linear_out.weight"));
                assert_eq!(s, vec![D, FH], "{src}: gating.{t}.linear_out shape");

                step.layers.push(LayerW {
                    qkv: Mat::new(qkv, f16),
                    o: Mat::new(o, f16),
                    gate_up: Mat::new(gu, f16),
                    down: Mat::new(down, f16),
                });
            }
        }

        let mut dep_in_all: Vec<f32> = Vec::with_capacity(n * D * cfg::DIM);
        for (t, step) in steps.iter_mut().enumerate() {
            let (w, s) = loader.load_f32(&format!("depformer_in.{t}.weight"));
            assert_eq!(s, vec![D, cfg::DIM], "depformer_in.{t} shape");
            dep_in_all.extend_from_slice(&w);
            let (h, s) = loader.load_f32(&format!("linears.{t}.weight"));
            assert_eq!(s, vec![cfg::CARD, D], "linears.{t} shape");
            step.head = Mat::new(h, f16);
        }
        let (text_emb, s) = loader.load_f32("depformer_text_emb.weight");
        assert_eq!(s, vec![cfg::TEXT_VOCAB, D], "depformer_text_emb shape");
        let audio_emb: Vec<HostF32> = (0..n - 1)
            .map(|t| {
                let (w, s) = loader.load_f32(&format!("depformer_emb.{t}.weight"));
                assert_eq!(s, vec![cfg::AUDIO_VOCAB, D], "depformer_emb.{t} shape");
                HostF32::Owned(w)
            })
            .collect();

        Self::assemble(
            steps,
            Mat::new(dep_in_all, f16),
            HostF32::Owned(text_emb),
            audio_emb,
        )
    }

    /// Deterministic synthetic weights at the real shapes — pure-timing
    /// construction for thread/storage sweeps without the 5.3 GiB pile read.
    pub fn synthetic(f16: bool) -> Self {
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
        let steps = (0..STEPS)
            .map(|_| StepW {
                layers: (0..LAYERS)
                    .map(|_| LayerW {
                        qkv: Mat::new(fill(3 * D * D), f16),
                        o: Mat::new(fill(D * D), f16),
                        gate_up: Mat::new(fill(2 * FH * D), f16),
                        down: Mat::new(fill(D * FH), f16),
                    })
                    .collect(),
                head: Mat::new(fill(cfg::CARD * D), f16),
            })
            .collect();
        let dep_in = Mat::new(fill(STEPS * D * cfg::DIM), f16);
        let text_emb = HostF32::Owned(fill(cfg::TEXT_VOCAB * D));
        let audio_emb = (0..STEPS - 1)
            .map(|_| HostF32::Owned(fill(cfg::AUDIO_VOCAB * D)))
            .collect();
        Self::assemble(steps, dep_in, text_emb, audio_emb)
    }

    fn assemble(
        steps: Vec<StepW>,
        dep_in: Mat,
        text_emb: HostF32,
        audio_emb: Vec<HostF32>,
    ) -> Self {
        Self {
            steps,
            dep_in,
            text_emb,
            audio_emb,
            cond: vec![0f32; STEPS * D],
            kc: vec![0f32; LAYERS * STEPS * D],
            vc: vec![0f32; LAYERS * STEPS * D],
            x: vec![0f32; D],
            xin: vec![0f32; D],
            qkv_buf: vec![0f32; 3 * D],
            attn: vec![0f32; D],
            proj: vec![0f32; D],
            gu: vec![0f32; 2 * FH],
            act: vec![0f32; FH],
            logits_buf: vec![0f32; STEPS * cfg::CARD],
            t_cond: 0.0,
            t_gemv: 0.0,
            t_head: 0.0,
            t_frame: 0.0,
            frames: 0,
        }
    }

    /// Weight bytes touched per frame (every per-step matrix exactly once).
    pub fn frame_weight_bytes(&self) -> usize {
        let bpe = self.steps[0].layers[0].qkv.bytes_per_elem();
        let per_layer = 3 * D * D + D * D + 2 * FH * D + D * FH;
        (STEPS * (LAYERS * per_layer + cfg::CARD * D) + STEPS * D * cfg::DIM) * bpe
    }

    /// Whether the matrix weights are stored f16 (the `f16` flag this was
    /// loaded with) — read off the storage rather than kept as a second copy
    /// of the same fact.
    pub fn is_f16(&self) -> bool {
        self.steps[0].layers[0].qkv.bytes_per_elem() == 2
    }

    /// The 16 × 2048 logit rows of the last [`Self::frame`], row-major.
    pub fn logits(&self) -> &[f32] {
        &self.logits_buf
    }

    /// One temporal frame's depformer pass — same contract as
    /// [`super::depth::DepthTransformer::frame`]: `transformer_out` is the
    /// post-`out_norm` temporal hidden `[4096]`, `text_token` the frame's
    /// `next_text_token`; `forced[s]` teacher-forces the prev-token chain
    /// where the LMGen cache provided the target, `teacher` pins every step's
    /// input to an oracle trajectory (gate mode). Greedy tokens returned;
    /// logits land in [`Self::logits`].
    pub fn frame(
        &mut self,
        transformer_out: &[f32],
        text_token: i64,
        forced: &[Option<i64>; cfg::DEP_Q],
        teacher: Option<&[i64]>,
        mut sampler: Option<&mut Sampler>,
    ) -> [i64; cfg::DEP_Q] {
        assert_eq!(transformer_out.len(), cfg::DIM);
        let t_frame = Instant::now();

        // all 16 conditioning projections in one gemv
        let t0 = Instant::now();
        self.dep_in
            .gemv(STEPS * D, cfg::DIM, transformer_out, &mut self.cond);
        self.t_cond += t0.elapsed().as_secs_f64();

        let mut tokens = [0i64; cfg::DEP_Q];
        let mut prev = text_token;
        for s in 0..STEPS {
            // x = dep_in_s(transformer_out) + emb(prev)
            let emb = if s == 0 {
                assert!(
                    (0..cfg::TEXT_VOCAB as i64).contains(&prev),
                    "text token {prev}"
                );
                &self.text_emb[prev as usize * D..(prev as usize + 1) * D]
            } else {
                assert!(
                    (0..cfg::AUDIO_VOCAB as i64).contains(&prev),
                    "audio token {prev}"
                );
                &self.audio_emb[s - 1][prev as usize * D..(prev as usize + 1) * D]
            };
            for i in 0..D {
                self.x[i] = self.cond[s * D + i] + emb[i];
            }

            let step = &self.steps[s];
            let lo = (s + 1).saturating_sub(WINDOW); // visible keys lo..=s
            for (li, l) in step.layers.iter().enumerate() {
                // ── attention ──
                rms_ip(&self.x, cfg::RMS_EPS, &mut self.xin);
                let t0 = Instant::now();
                l.qkv.gemv(3 * D, D, &self.xin, &mut self.qkv_buf);
                self.t_gemv += t0.elapsed().as_secs_f64();
                let (q, rest) = self.qkv_buf.split_at(D);
                let (k, v) = rest.split_at(D);
                let base = (li * STEPS + s) * D;
                self.kc[base..base + D].copy_from_slice(k);
                self.vc[base..base + D].copy_from_slice(v);

                let mut scores = [0f32; STEPS];
                for h in 0..HEADS {
                    let qh = &q[h * HD..(h + 1) * HD];
                    for t in lo..=s {
                        let kh = &self.kc[(li * STEPS + t) * D + h * HD..][..HD];
                        scores[t] = dot(qh, kh); // 2⁻³ scale folded in q rows
                    }
                    softmax(&mut scores[lo..=s]);
                    let out = &mut self.attn[h * HD..(h + 1) * HD];
                    out.fill(0.0);
                    for t in lo..=s {
                        let vh = &self.vc[(li * STEPS + t) * D + h * HD..][..HD];
                        let p = scores[t];
                        for (o, &vv) in out.iter_mut().zip(vh) {
                            *o += p * vv;
                        }
                    }
                }
                let t0 = Instant::now();
                l.o.gemv(D, D, &self.attn, &mut self.proj);
                self.t_gemv += t0.elapsed().as_secs_f64();
                for (xi, &p) in self.x.iter_mut().zip(&self.proj) {
                    *xi += p;
                }

                // ── SwiGLU FFN ──
                rms_ip(&self.x, cfg::RMS_EPS, &mut self.xin);
                let t0 = Instant::now();
                l.gate_up.gemv(2 * FH, D, &self.xin, &mut self.gu);
                self.t_gemv += t0.elapsed().as_secs_f64();
                for i in 0..FH {
                    let g = self.gu[i];
                    self.act[i] = g / (1.0 + (-g).exp()) * self.gu[FH + i];
                }
                let t0 = Instant::now();
                l.down.gemv(D, FH, &self.act, &mut self.proj);
                self.t_gemv += t0.elapsed().as_secs_f64();
                for (xi, &p) in self.x.iter_mut().zip(&self.proj) {
                    *xi += p;
                }
            }

            // ── logit head (no final norm) + token (sampled or greedy) ──
            let t0 = Instant::now();
            let row = &mut self.logits_buf[s * cfg::CARD..(s + 1) * cfg::CARD];
            step.head.gemv(cfg::CARD, D, &self.x, row);
            tokens[s] = match sampler.as_deref_mut() {
                Some(smp) => smp.token(row) as i64,
                None => argmax(row) as i64,
            };
            self.t_head += t0.elapsed().as_secs_f64();

            prev = forced[s].unwrap_or_else(|| teacher.map_or(tokens[s], |t| t[s]));
        }

        self.t_frame += t_frame.elapsed().as_secs_f64();
        self.frames += 1;
        tokens
    }

    /// Drain the per-frame timing decomposition accumulated since the last
    /// call: (frames, total, cond, stack gemv, head, scalar-rest) in ms/frame.
    pub fn take_bench(&mut self) -> (u64, f64, f64, f64, f64, f64) {
        let n = self.frames.max(1) as f64;
        let r = (
            self.frames,
            self.t_frame / n * 1e3,
            self.t_cond / n * 1e3,
            self.t_gemv / n * 1e3,
            self.t_head / n * 1e3,
            (self.t_frame - self.t_cond - self.t_gemv - self.t_head) / n * 1e3,
        );
        self.t_cond = 0.0;
        self.t_gemv = 0.0;
        self.t_head = 0.0;
        self.t_frame = 0.0;
        self.frames = 0;
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The NEON f16 kernel vs a scalar f64 reference: same weights, same x.
    #[test]
    fn hdot_matches_scalar() {
        let n = 2816;
        let mut state = 12345u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / (1u32 << 24) as f32 - 0.5
        };
        let w16: Vec<u16> = (0..n)
            .map(|_| half::f16::from_f32(rnd()).to_bits())
            .collect();
        let x: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let got = unsafe { hdot(w16.as_ptr(), x.as_ptr(), n) };
        let want: f64 = w16
            .iter()
            .zip(&x)
            .map(|(&w, &xv)| half::f16::from_bits(w).to_f32() as f64 * xv as f64)
            .sum();
        assert!(
            (got as f64 - want).abs() < 1e-3 * want.abs().max(1.0),
            "hdot {got} vs scalar {want}"
        );
    }

    /// Threaded f16 gemv == serial f16 gemv (row decomposition is exact).
    #[test]
    fn hgemv_mt_matches_serial() {
        let (m, n) = (321 * 4, 1024); // not chunk-aligned on purpose... m mult of 1 row
        let mut state = 999u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / (1u32 << 24) as f32 - 0.5
        };
        let w: Vec<u16> = (0..m * n)
            .map(|_| half::f16::from_f32(rnd()).to_bits())
            .collect();
        let x: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let mut y_serial = vec![0f32; m];
        let mut y_mt = vec![0f32; m];
        hgemv(&w, m, n, &x, &mut y_serial);
        hgemv_mt(&w, m, n, &x, &mut y_mt);
        assert_eq!(y_serial, y_mt, "row decomposition must be bit-exact");
    }
}
