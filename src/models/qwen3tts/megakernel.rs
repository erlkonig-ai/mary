//! Hand-fused cubecl kernels for the talker's single-token decode step —
//! research prototype for cutting per-frame **host submissions** (the loop's
//! actual bottleneck, see PORT_NOTES.md).
//!
//! The Burn path costs ~18 dispatches per decoder layer (reduce+elementwise
//! pairs for each weightless rms, a matmul per projection, a 4-kernel softmax,
//! two cache `cat` copies). This module folds each layer into **5 dispatches**:
//!
//!   1. [`qkv_rope_cache_kernel`] — rms(x) → wide qkv matvec → per-head q/k
//!      RMS-norm → RoPE chain → write q + append k/v into a **preallocated**
//!      ring of cache buffers (no `cat`, no realloc, no copy of history).
//!   2. [`attn_decode_kernel`]    — scores → softmax → weighted-V, one cube
//!      per query head, shared-memory softmax (GQA folded by indexing).
//!   3. [`matvec_kernel`]         — o_proj + residual add.
//!   4. [`mlp_gateup_kernel`]     — rms(x) → gate‖up matvec → silu·mul.
//!   5. [`matvec_kernel`]         — down_proj + residual add.
//!
//! Whole frame: 28×5 + 1 (final weighted rms) = **141 dispatches** vs ~500+.
//! The engine shares the *same GPU weight buffers* as the Burn `Talker` (the
//! `CubeTensor` handles are extracted from its tensors, zero copy) and mirrors
//! its math exactly — the fold conventions (norm weights folded into matmul
//! rows, 1/√d and q/k-norm weights in the `w`/`w_rot` chain, rotate_half
//! pre-applied to weight rows) are inherited, not reimplemented.
//!
//! Backend: the **raw** (non-fusion) f32 backend the voice runs on —
//! `nn::backend::speak::Raw`: wgpu/Metal on the Mac, CUDA on the Linux (Spark)
//! build, the same selection `nn::q4::Rt` makes. The fusion wrapper adds
//! nothing here — this module *is* the fusion, done once at compile time
//! instead of per-frame at graph-capture time. On CUDA this is also what
//! removes the Burn loop's per-frame recompiles: every buffer here is
//! preallocated, so no kernel ever sees a new shape.

use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{DType, FloatDType, Tensor as BurnTensor, TensorPrimitive};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

use super::config::*;
use super::layers::KvCache;
use super::pipeline::FrameStepper;
use super::talker::Talker;

/// The raw (non-fusion) f32 and f16 backends whose tensors the engine aliases.
pub type Raw = crate::nn::backend::speak::Raw;
pub type RawHalf = crate::nn::backend::speak::RawHalf;

/// The cubecl runtime behind [`Raw`] — mirrors the `nn::q4::Rt` selection so
/// the engine's kernels launch on the same device the talker's tensors live on.
#[cfg(any(feature = "cuda-backend", all(target_os = "linux", feature = "qwen3tts")))]
pub use cubecl::cuda::CudaRuntime as Rt;
#[cfg(not(any(feature = "cuda-backend", all(target_os = "linux", feature = "qwen3tts"))))]
pub use cubecl::wgpu::WgpuRuntime as Rt;

type Client = cubecl::client::ComputeClient<Rt>;

const HIDDEN: u32 = TALKER_HIDDEN as u32; // 2048
const HEADS: u32 = TALKER_HEADS as u32; // 16
const KV_HEADS: u32 = TALKER_KV_HEADS as u32; // 8
const D: u32 = TALKER_HEAD_DIM as u32; // 128
const HH: u32 = HEADS + KV_HEADS; // 24
const WIDE_OUT: u32 = (2 * HH + KV_HEADS) * D; // 7168
const KV_DIM: u32 = KV_HEADS * D; // 1024
const INTER: u32 = 6144;
const EPS: f32 = 1e-6; // TALKER_EPS

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

/// Stage 1: weightless rms(x) + wide fused matvec `[qk | R(qk) | v]` + per-head
/// q/k RMS-norm + RoPE chain + KV-cache append, one dispatch.
///
/// Cube `h < heads+kv_heads` owns qk head `h`: 128 threads compute its `qk`
/// and `R(qk)` columns (one output element each), reduce the head variance in
/// shared memory, apply the `(qk·w·cos + qkR·w_rot·sin)·s` chain and write to
/// `q_out` (query heads) or `kcache[pos]` (key heads). Cubes beyond that copy
/// v heads into `vcache[pos]`. The rms(x) reduction runs redundantly in every
/// cube (2048 reads — noise), keeping the barrier structure cube-uniform.
///
/// `wn` = `[w ‖ w_rot]` (the RoPE chain weights, q/k-norm × 1/√d folds
/// included); `rope` = per-position `[cos(d) ‖ sin(d)]` in full-width
/// rotate_half form.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn qkv_rope_cache_kernel<W: Float>(
    x: &Array<f32>,
    wide: &Array<W>,
    wn: &Array<f32>,
    rope: &Array<f32>,
    q_out: &mut Array<f32>,
    kcache: &mut Array<f32>,
    vcache: &mut Array<f32>,
    pos: u32,
    eps: f32,
    #[comptime] hidden: u32,
    #[comptime] heads: u32,
    #[comptime] kv_heads: u32,
    #[comptime] d: u32,
) {
    let i = UNIT_POS_X;
    let head = CUBE_POS_X;
    let hh = heads + kv_heads;
    let wide_out = (2 * hh + kv_heads) * d;

    let mut red = SharedMemory::<f32>::new(comptime!(d as usize));

    // cooperative Σx² (redundant per cube; x is 8 KB)
    let mut acc = f32::new(0.0);
    let mut k = i;
    while k < hidden {
        let v = x[k as usize];
        acc += v * v;
        k += d;
    }
    red[i as usize] = acc;
    sync_cube();
    let mut stride = u32::new((d / 2) as i64);
    while stride > 0 {
        if i < stride {
            red[i as usize] = red[i as usize] + red[(i + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let rms_s = 1.0 / (red[0] / (hidden as f32) + eps).sqrt();
    sync_cube(); // red is reused for the head variance below

    // this cube's two columns (v cubes read the same column twice — cached)
    let qk_cube = head < hh;
    let c0 = if qk_cube {
        head * d + i
    } else {
        2 * hh * d + (head - hh) * d + i
    };
    let c1 = if qk_cube { hh * d + head * d + i } else { c0 };
    let mut y0 = f32::new(0.0);
    let mut y1 = f32::new(0.0);
    for k in 0..hidden {
        let xv = x[k as usize];
        y0 += xv * f32::cast_from(wide[(k * wide_out + c0) as usize]);
        y1 += xv * f32::cast_from(wide[(k * wide_out + c1) as usize]);
    }
    y0 *= rms_s;
    y1 *= rms_s;

    // per-head variance of the qk block (v cubes compute-and-discard so the
    // barriers stay cube-uniform)
    red[i as usize] = y0 * y0;
    sync_cube();
    let mut stride = u32::new((d / 2) as i64);
    while stride > 0 {
        if i < stride {
            red[i as usize] = red[i as usize] + red[(i + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let s = 1.0 / (red[0] / (d as f32) + eps).sqrt();

    let kv_dim = kv_heads * d;
    if qk_cube {
        let idx = head * d + i;
        let tc = pos * 2 * d + i; // cos
        let ts = pos * 2 * d + d + i; // sin
        let out = (y0 * wn[idx as usize] * rope[tc as usize]
            + y1 * wn[(hh * d + idx) as usize] * rope[ts as usize])
            * s;
        if head < heads {
            q_out[idx as usize] = out;
        } else {
            kcache[(pos * kv_dim + (head - heads) * d + i) as usize] = out;
        }
    } else {
        vcache[(pos * kv_dim + (head - hh) * d + i) as usize] = y0;
    }
}

/// Stage 2: single-token causal attention over the cache, one cube per query
/// head. Scores + shared-memory softmax + weighted-V in one dispatch; GQA is
/// pure indexing (`kvh = h / groups`), 1/√d is already folded into the q chain
/// weights upstream. `len` counts the cache *including* the current position.
#[cube(launch_unchecked)]
fn attn_decode_kernel(
    q: &Array<f32>,
    kcache: &Array<f32>,
    vcache: &Array<f32>,
    out: &mut Array<f32>,
    len: u32,
    #[comptime] kv_heads: u32,
    #[comptime] groups: u32,
    #[comptime] d: u32,
    #[comptime] max_scores: u32,
) {
    let i = UNIT_POS_X;
    let h = CUBE_POS_X;
    let kvh = h / groups;
    let kv_dim = kv_heads * d;

    let mut qsh = SharedMemory::<f32>::new(comptime!(d as usize));
    let mut scores = SharedMemory::<f32>::new(comptime!(max_scores as usize));

    let qv = q[(h * d + i) as usize];
    qsh[i as usize] = qv;
    sync_cube();

    let mut t = i;
    while t < len {
        let base = (t * kv_dim + kvh * d) as usize;
        let mut s = f32::new(0.0);
        for dd in 0..d {
            s += qsh[dd as usize] * kcache[base + dd as usize];
        }
        scores[t as usize] = s;
        t += d;
    }
    sync_cube();

    // max
    let mut m = f32::new(-3.40282e38);
    let mut t = i;
    while t < len {
        let sv = scores[t as usize];
        if sv > m {
            m = sv;
        }
        t += d;
    }
    qsh[i as usize] = m;
    sync_cube();
    let mut stride = u32::new((d / 2) as i64);
    while stride > 0 {
        if i < stride {
            let other = qsh[(i + stride) as usize];
            if other > qsh[i as usize] {
                qsh[i as usize] = other;
            }
        }
        sync_cube();
        stride /= 2;
    }
    let mx = qsh[0];
    sync_cube(); // qsh is reused for the sum

    // exp + partial sums
    let mut sum = f32::new(0.0);
    let mut t = i;
    while t < len {
        let p = (scores[t as usize] - mx).exp();
        scores[t as usize] = p;
        sum += p;
        t += d;
    }
    qsh[i as usize] = sum;
    sync_cube();
    let mut stride = u32::new((d / 2) as i64);
    while stride > 0 {
        if i < stride {
            qsh[i as usize] = qsh[i as usize] + qsh[(i + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let total = qsh[0];

    // weighted V — thread i owns output dim i
    let mut acc = f32::new(0.0);
    for t in 0..len {
        acc += scores[t as usize] * vcache[(t * kv_dim + kvh * d + i) as usize];
    }
    out[(h * d + i) as usize] = acc / total;
}

/// Stages 3/5: plain matvec `dst (+)= srcᵀ·W` against a pre-transposed weight
/// (`[in, out]` row-major), optionally accumulating into `dst` (residual add).
#[cube(launch_unchecked)]
fn matvec_kernel<W: Float>(
    src: &Array<f32>,
    w: &Array<W>,
    dst: &mut Array<f32>,
    #[comptime] in_dim: u32,
    #[comptime] out_dim: u32,
    #[comptime] residual: bool,
) {
    let j = ABSOLUTE_POS as u32;
    if j < out_dim {
        let mut acc = f32::new(0.0);
        for k in 0..in_dim {
            acc += src[k as usize] * f32::cast_from(w[(k * out_dim + j) as usize]);
        }
        if residual {
            dst[j as usize] = dst[j as usize] + acc;
        } else {
            dst[j as usize] = acc;
        }
    }
}

/// Stage 4: weightless rms(x) + fused gate‖up matvec + SwiGLU (`silu(g)·u`),
/// one dispatch. Cube-cooperative rms like stage 1; thread j owns gate col j
/// and up col j+inter of the pre-transposed `[hidden, 2·inter]` weight.
#[cube(launch_unchecked)]
fn mlp_gateup_kernel<W: Float>(
    x: &Array<f32>,
    gate_up: &Array<W>,
    act: &mut Array<f32>,
    eps: f32,
    #[comptime] hidden: u32,
    #[comptime] inter: u32,
    #[comptime] cube_dim: u32,
) {
    let i = UNIT_POS_X;
    let j = CUBE_POS_X * cube_dim + i;
    let two_i = 2 * inter;

    let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    let mut acc = f32::new(0.0);
    let mut k = i;
    while k < hidden {
        let v = x[k as usize];
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
    let rms_s = 1.0 / (red[0] / (hidden as f32) + eps).sqrt();

    let mut g = f32::new(0.0);
    let mut u = f32::new(0.0);
    for k in 0..hidden {
        let xv = x[k as usize];
        g += xv * f32::cast_from(gate_up[(k * two_i + j) as usize]);
        u += xv * f32::cast_from(gate_up[(k * two_i + inter + j) as usize]);
    }
    g *= rms_s;
    u *= rms_s;
    act[j as usize] = g / (1.0 + (-g).exp()) * u;
}

/// Final stack norm: weighted RMSNorm in one single-cube dispatch.
#[cube(launch_unchecked)]
fn rmsnorm_kernel(
    x: &Array<f32>,
    weight: &Array<f32>,
    out: &mut Array<f32>,
    eps: f32,
    #[comptime] hidden: u32,
    #[comptime] cube_dim: u32,
) {
    let i = UNIT_POS_X;
    let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));
    let mut acc = f32::new(0.0);
    let mut k = i;
    while k < hidden {
        let v = x[k as usize];
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
    let rms_s = 1.0 / (red[0] / (hidden as f32) + eps).sqrt();
    let mut k = i;
    while k < hidden {
        out[k as usize] = x[k as usize] * rms_s * weight[k as usize];
        k += cube_dim;
    }
}

/// One-time cache import: `dst[offset + j] = src[j]`.
#[cube(launch_unchecked)]
fn copy_offset_kernel(src: &Array<f32>, dst: &mut Array<f32>, n: u32, offset: u32) {
    let j = ABSOLUTE_POS as u32;
    if j < n {
        let v = src[j as usize];
        dst[(offset + j) as usize] = v;
    }
}

// ---------------------------------------------------------------------------
// host engine
// ---------------------------------------------------------------------------

/// A Burn backend whose float tensors are `CubeTensor`s on [`Rt`] — the raw
/// f32 and f16 lanes the voice runs on, on either platform.
pub trait EngineBackend:
    Backend<FloatTensorPrimitive = CubeTensor<Rt>> + BackendTypes + 'static
{
}
impl<B: Backend<FloatTensorPrimitive = CubeTensor<Rt>> + BackendTypes + 'static> EngineBackend for B {}

/// Extract the contiguous `CubeTensor` behind a Burn tensor: its buffer and
/// element type.
fn cube_handle<B: EngineBackend, const DIM: usize>(t: &BurnTensor<B, DIM>) -> (Handle, DType) {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(c) => {
            let mut c: CubeTensor<Rt> = c;
            if !c.is_contiguous() {
                // e.g. stride bookkeeping on size-1 dims after transpose+mul
                c = burn_cubecl::kernel::into_contiguous(c);
            }
            (c.handle, c.dtype)
        }
        _ => panic!("expected a plain float tensor"),
    }
}

/// The same, for buffers the kernels read as f32 (chain weights, the final
/// norm): cast first, so an f16 talker's small tensors arrive widened.
fn cube_handle_f32<B: EngineBackend, const DIM: usize>(t: &BurnTensor<B, DIM>) -> Handle {
    let (h, dt) = cube_handle(&t.clone().cast(FloatDType::F32));
    assert_eq!(dt, DType::F32);
    h
}

struct LayerBufs {
    wide: Handle,    // [hidden × wide_out]
    wn: Handle,      // [2 · hh · d] = w ‖ w_rot (f32)
    o: Handle,       // [hidden × hidden] (pre-transposed)
    gate_up: Handle, // [hidden × 2·inter]
    down: Handle,    // [inter × hidden]
    kcache: Handle,  // [max_seq × kv_dim], position-major, f32
    vcache: Handle,
}

/// Fused decode-step engine for the talker. Aliases the Burn [`Talker`]'s GPU
/// weight buffers (zero copy, f16 or f32 as loaded) and owns preallocated f32
/// activation + KV-cache buffers. One [`step`](Self::step) = 141 dispatches +
/// one blocking readback.
pub struct TalkerEngine {
    client: Client,
    layers: Vec<LayerBufs>,
    norm_w: Handle,
    rope: Handle,
    q: Handle,
    attn: Handle,
    act: Handle,
    out: Handle,
    len: usize,
    max_seq: usize,
    /// The element the four weight-streaming kernels read: f16 on the half
    /// lane, f32 otherwise. Activations, caches and chain weights stay f32.
    w16: bool,
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

impl TalkerEngine {
    /// Build the engine over a loaded talker. `max_seq` bounds prefill+frames
    /// (cache memory: `28 layers × 2 × max_seq × 4 KiB`).
    pub fn new<B: EngineBackend>(talker: &Talker<B>, max_seq: usize) -> Self {
        assert!(
            max_seq <= MAX_SCORES as usize,
            "max_seq exceeds kernel shared-memory cap"
        );
        let probe = talker.norm.weight.clone();
        let client = match probe.into_primitive() {
            TensorPrimitive::Float(c) => c.client,
            _ => unreachable!(),
        };

        let mut dtype: Option<DType> = None;
        let layers: Vec<LayerBufs> = {
            // every streamed weight must share one element type
            let mut weight = |t: (Handle, DType)| -> Handle {
                match dtype {
                    None => dtype = Some(t.1),
                    Some(d) => assert_eq!(d, t.1, "talker weight element types differ"),
                }
                t.0
            };
            talker
                .layers
                .iter()
                .map(|l| {
                    let wn_t = BurnTensor::cat(
                        vec![
                            l.attn.w.clone().reshape([(HH * D) as usize]),
                            l.attn.w_rot.clone().reshape([(HH * D) as usize]),
                        ],
                        0,
                    );
                    LayerBufs {
                        wide: weight(cube_handle(&l.attn.wide_t)),
                        wn: cube_handle_f32(&wn_t),
                        o: weight(cube_handle(&l.attn.o_proj.weight_t)),
                        gate_up: weight(cube_handle(&l.gate_up_t)),
                        down: weight(cube_handle(&l.down_proj.weight_t)),
                        kcache: client.empty(max_seq * KV_DIM as usize * 4),
                        vcache: client.empty(max_seq * KV_DIM as usize * 4),
                    }
                })
                .collect()
        };
        let w16 = match dtype.expect("a talker has layers") {
            DType::F16 => true,
            DType::F32 => false,
            other => panic!("the engine reads f16 or f32 talker weights, not {other:?}"),
        };

        // full-width rotate_half RoPE table, [pos][cos(d) ‖ sin(d)]
        let mut rope = vec![0f32; max_seq * 2 * D as usize];
        let half = (D / 2) as usize;
        for p in 0..max_seq {
            for i in 0..half {
                let r = p as f64 * TALKER_ROPE_THETA.powf(-2.0 * i as f64 / D as f64);
                let base = p * 2 * D as usize;
                rope[base + i] = r.cos() as f32;
                rope[base + half + i] = r.cos() as f32;
                rope[base + D as usize + i] = r.sin() as f32;
                rope[base + D as usize + half + i] = r.sin() as f32;
            }
        }

        Self {
            rope: client.create_from_slice(as_bytes(&rope)),
            norm_w: cube_handle_f32(&talker.norm.weight),
            q: client.empty(TALKER_HIDDEN * 4),
            attn: client.empty(TALKER_HIDDEN * 4),
            act: client.empty(INTER as usize * 4),
            out: client.empty(TALKER_HIDDEN * 4),
            layers,
            client,
            len: 0,
            max_seq,
            w16,
        }
    }

    /// Import the KV caches produced by a Burn-path prefill (one-time,
    /// host-roundtripped: `[1, hkv, L, d] → [L, hkv·d]` position-major).
    pub fn import_caches<B: EngineBackend>(&mut self, caches: &[KvCache<B>]) {
        assert_eq!(caches.len(), self.layers.len());
        let mut len = 0;
        for (l, c) in self.layers.iter().zip(caches) {
            for (src, dst) in [(&c.k, &l.kcache), (&c.v, &l.vcache)] {
                let t = src.as_ref().expect("prefilled cache").clone();
                let [_, hkv, seq, d] = t.dims();
                assert!(seq <= self.max_seq, "prefill longer than max_seq");
                len = seq;
                let data = t.into_data().convert::<f32>().to_vec::<f32>().unwrap();
                let mut host = vec![0f32; seq * KV_DIM as usize];
                for h in 0..hkv {
                    for p in 0..seq {
                        let s = (h * seq + p) * d;
                        let o = p * KV_DIM as usize + h * d;
                        host[o..o + d].copy_from_slice(&data[s..s + d]);
                    }
                }
                let n = (seq * KV_DIM as usize) as u32;
                let src_h = self.client.create_from_slice(as_bytes(&host));
                unsafe {
                    copy_offset_kernel::launch_unchecked::<Rt>(
                        &self.client,
                        CubeCount::new_1d(n.div_ceil(256)),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(src_h, n as usize),
                        ArrayArg::from_raw_parts(dst.clone(), self.max_seq * KV_DIM as usize),
                        n,
                        0,
                    );
                }
            }
        }
        self.len = len;
    }

    /// Current sequence length (prefill + generated frames).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The cache ring's capacity: prefill + frames a pass can hold.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Whether the streamed weights are f16 (the half lane) or f32.
    pub fn half_weights(&self) -> bool {
        self.w16
    }

    /// Encode + submit one decode step (141 dispatches), without reading back.
    pub fn step_submit(&mut self, x_host: &[f32]) {
        if self.w16 {
            self.step_submit_w::<f16>(x_host)
        } else {
            self.step_submit_w::<f32>(x_host)
        }
    }

    fn step_submit_w<W: Float>(&mut self, x_host: &[f32]) {
        assert_eq!(x_host.len(), TALKER_HIDDEN);
        assert!(self.len < self.max_seq, "cache full");
        let pos = self.len as u32;
        let x = self.client.create_from_slice(as_bytes(x_host));
        let arr = |h: &Handle, n: u32| unsafe { ArrayArg::from_raw_parts(h.clone(), n as usize) };

        for l in &self.layers {
            unsafe {
                qkv_rope_cache_kernel::launch_unchecked::<W, Rt>(
                    &self.client,
                    CubeCount::new_1d(HH + KV_HEADS),
                    CubeDim::new_1d(D),
                    arr(&x, HIDDEN),
                    arr(&l.wide, HIDDEN * WIDE_OUT),
                    arr(&l.wn, 2 * HH * D),
                    arr(&self.rope, self.max_seq as u32 * 2 * D),
                    arr(&self.q, HIDDEN),
                    arr(&l.kcache, self.max_seq as u32 * KV_DIM),
                    arr(&l.vcache, self.max_seq as u32 * KV_DIM),
                    pos,
                    EPS,
                    HIDDEN,
                    HEADS,
                    KV_HEADS,
                    D,
                );
                attn_decode_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d(HEADS),
                    CubeDim::new_1d(D),
                    arr(&self.q, HIDDEN),
                    arr(&l.kcache, self.max_seq as u32 * KV_DIM),
                    arr(&l.vcache, self.max_seq as u32 * KV_DIM),
                    arr(&self.attn, HIDDEN),
                    pos + 1,
                    KV_HEADS,
                    HEADS / KV_HEADS,
                    D,
                    MAX_SCORES,
                );
                matvec_kernel::launch_unchecked::<W, Rt>(
                    &self.client,
                    CubeCount::new_1d(HIDDEN / 128),
                    CubeDim::new_1d(128),
                    arr(&self.attn, HIDDEN),
                    arr(&l.o, HIDDEN * HIDDEN),
                    arr(&x, HIDDEN),
                    HIDDEN,
                    HIDDEN,
                    true,
                );
                mlp_gateup_kernel::launch_unchecked::<W, Rt>(
                    &self.client,
                    CubeCount::new_1d(INTER / 128),
                    CubeDim::new_1d(128),
                    arr(&x, HIDDEN),
                    arr(&l.gate_up, HIDDEN * 2 * INTER),
                    arr(&self.act, INTER),
                    EPS,
                    HIDDEN,
                    INTER,
                    128,
                );
                matvec_kernel::launch_unchecked::<W, Rt>(
                    &self.client,
                    CubeCount::new_1d(HIDDEN / 128),
                    CubeDim::new_1d(128),
                    arr(&self.act, INTER),
                    arr(&l.down, INTER * HIDDEN),
                    arr(&x, HIDDEN),
                    INTER,
                    HIDDEN,
                    true,
                );
            }
        }
        unsafe {
            rmsnorm_kernel::launch_unchecked::<Rt>(
                &self.client,
                CubeCount::new_single(),
                CubeDim::new_1d(256),
                arr(&x, HIDDEN),
                arr(&self.norm_w, HIDDEN),
                arr(&self.out, HIDDEN),
                EPS,
                HIDDEN,
                256,
            );
        }
        self.len += 1;
    }

    /// Blocking readback of the last step's normed hidden state `[2048]` —
    /// the frame's one GPU sync.
    pub fn read_hidden(&self) -> Vec<f32> {
        let bytes = self.client.read_one(self.out.clone()).expect("readback");
        let mut v = vec![0f32; TALKER_HIDDEN];
        v.copy_from_slice(f32::from_bytes(&bytes));
        v
    }

    /// One full decode step: submit + sync. Mirrors
    /// `talker.forward(embeds, caches) → last_hidden` on the Burn path.
    pub fn step(&mut self, x_host: &[f32]) -> Vec<f32> {
        self.step_submit(x_host);
        self.read_hidden()
    }

    /// Dispatches encoded per [`step_submit`] — the number the whole module
    /// exists to shrink.
    pub const DISPATCHES_PER_STEP: usize = TALKER_LAYERS * 5 + 1;
}

/// [`FrameStepper`] over the engine: the Burn talker runs each pass's prefill
/// (its shapes are one-off anyway), then the engine takes the frames from the
/// imported caches — one host row in, one normed hidden state out per frame.
pub struct EngineStepper<'a, B: EngineBackend> {
    talker: &'a Talker<B>,
    engine: &'a mut TalkerEngine,
    pending: Option<BurnTensor<B, 3>>,
}

impl<'a, B: EngineBackend> EngineStepper<'a, B> {
    pub fn new(
        talker: &'a Talker<B>,
        engine: &'a mut TalkerEngine,
        prefill: BurnTensor<B, 3>,
        device: &<B as BackendTypes>::Device,
    ) -> Self {
        let mut caches = talker.new_caches();
        let h = talker.forward(prefill, &mut caches, device);
        engine.import_caches(&caches);
        Self {
            talker,
            engine,
            pending: Some(h),
        }
    }
}

impl<B: EngineBackend> FrameStepper for EngineStepper<'_, B> {
    fn hidden(&mut self) -> Vec<f32> {
        match self.pending.take() {
            Some(h) => self.talker.last_hidden(h),
            None => self.engine.read_hidden(),
        }
    }

    fn submit(&mut self, x: &[f32]) -> bool {
        if self.engine.len() >= self.engine.max_seq() {
            return false;
        }
        self.engine.step_submit(x);
        true
    }
}

/// Shared-memory cap of [`attn_decode_kernel`]'s score buffer (f32 count).
/// 2048 × 4 B = 8 KiB, comfortably under Metal's 32 KiB threadgroup limit.
pub const MAX_SCORES: u32 = 2048;

// ---------------------------------------------------------------------------
// microbenchmark kernels (persistent-kernel experiment, see the probe)
// ---------------------------------------------------------------------------

/// One matvec step of a dependent chain, multi-cube (`x_{t+1} = W·x_t`, scaled
/// to stay bounded). The chain is driven by K separate dispatches.
#[cube(launch_unchecked)]
pub fn chain_matvec_kernel(
    buf: &mut Array<f32>,
    w: &Array<f32>,
    src_off: u32,
    dst_off: u32,
    #[comptime] n: u32,
) {
    let j = ABSOLUTE_POS as u32;
    if j < n {
        let mut acc = f32::new(0.0);
        for k in 0..n {
            acc += buf[(src_off + k) as usize] * w[(k * n + j) as usize];
        }
        buf[(dst_off + j) as usize] = acc * 0.03;
    }
}

/// The same K-step dependent chain inside ONE dispatch: a single cube loops
/// device-side with `sync_cube()` between steps. This is the only legal form
/// of a persistent AR loop on wgpu/Metal — there is no grid-wide barrier, so
/// a persistent kernel is capped at one workgroup (= one GPU core). The probe
/// measures what that costs in bandwidth vs. the multi-dispatch chain.
#[cube(launch_unchecked)]
pub fn persistent_chain_kernel(
    buf: &mut Array<f32>,
    w: &Array<f32>,
    steps: u32,
    #[comptime] n: u32,
    #[comptime] cube_dim: u32,
) {
    let i = UNIT_POS_X;
    for step in 0..steps {
        let src = (step % 2) * n;
        let dst = n - src;
        let mut j = i;
        while j < n {
            let mut acc = f32::new(0.0);
            for k in 0..n {
                acc += buf[(src + k) as usize] * w[(k * n + j) as usize];
            }
            buf[(dst + j) as usize] = acc * 0.03;
            j += cube_dim;
        }
        sync_cube();
    }
}

/// Minimal kernel for measuring pure per-dispatch host overhead.
#[cube(launch_unchecked)]
pub fn touch_kernel(buf: &mut Array<f32>) {
    if ABSOLUTE_POS == 0 {
        buf[0] = buf[0] + 1.0;
    }
}

/// Handle + client access for the probe's microbenchmarks.
pub fn client_for_device() -> Client {
    use cubecl::Runtime;
    Rt::client(&Default::default())
}
