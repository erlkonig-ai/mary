//! q4 — 4-bit grouped weight quantization + dequant-in-kernel matvec, the
//! **bandwidth lever** for the PersonaPlex-7B (Moshi) realtime lane.
//!
//! Why this exists (see `moshi_realtime_probe`, commit 2e6ae2e): the Moshi
//! temporal transformer's single-token decode step must stream every weight
//! byte once per step. At f16 that is 17.2 GB/step = a 32–42 ms physics floor
//! on M4 Max — the 80 ms/frame budget dies once depth + mimi + submission are
//! added. At ~4.5 bits/weight the floor drops to ~9–12 ms. Kyutai's MLX Moshi
//! ships 4-bit on identical silicon; this module is mary's equivalent primitive.
//!
//! ## Format (GGUF-Q4_0-style, chosen for the M=1 matvec access pattern)
//!
//! Weights are logically `[out, in]` row-major (the Linear convention used by
//! every loader in mary). Quantization groups are **32 consecutive weights
//! along the input dim** of one row:
//!
//! - per group: `d = w[argmax |w|] / -8` stored as **f16**; each weight
//!   `q = clamp(round(w/d) + 8, 0, 15)` stored as a **nibble**.
//! - packed words: `wq: [out, in/8]` row-major `u32`, nibble `k%8` of word
//!   `k/8` at bits `4*(k%8)` (little-nibble-endian, GGUF order).
//! - scales: `scales: [out, in/32]` row-major f16.
//!
//! Cost: 4 bits + 16/32 bits scale = **4.5 bits/weight** (0.5625 B), a 3.56×
//! reduction vs f16. Row-major (not the megakernel's pre-transposed layout)
//! because the kernel assigns **one 32-thread group per output row**: threads
//! sweep the row's packed words together, so consecutive threads read
//! consecutive words — coalesced *within* the row, llama.cpp-mul_mv-style.
//! A pre-transposed `[in, out]` layout would instead splay each group's scale
//! and words across `out`-strided cachelines.
//!
//! ## Kernel
//!
//! [`q4_matvec_kernel`]: cube = 256 threads = 8 row-groups × 32 threads.
//! Thread `lane` of row-group `r` handles scale-groups `lane, lane+32, …` of
//! row `CUBE_POS*8 + r`: loads the f16 scale, then 4 packed u32 words (16 B =
//! one full quant group), dequantizes nibbles in registers, accumulates
//! `Σ (q-8)·x[k]` in f32, multiplies by the scale per group, and finally
//! tree-reduces the 32 partial sums in shared memory. Accumulation is f32
//! throughout; nibble→float via `Cast`. [`f16_matvec_kernel`] is the same
//! skeleton reading raw f16 rows — the controlled bandwidth baseline that
//! isolates "4× fewer bytes" from kernel-shape effects.
//!
//! Feature-gated (`q4`) like `qwen3tts::megakernel`: needs the `cubecl` dep at
//! the version burn 0.21 embeds. Additive — no existing model path changes.

use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;


// Backend selection. `cuda-backend` swaps the whole q4 lane onto CUDA; the
// default stays wgpu/Metal. burn re-exports the same WgpuRuntime type
// (burn-wgpu lib.rs:17), so naming cubecl directly drops burn out of this
// module entirely.
#[cfg(feature = "cuda-backend")]
pub use cubecl::cuda::CudaRuntime as Rt;
#[cfg(not(feature = "cuda-backend"))]
pub use cubecl::wgpu::WgpuRuntime as Rt;
pub type Client = cubecl::client::ComputeClient<Rt>;

/// Quantization group size (weights per f16 scale), along the input dim.
pub const GROUP: usize = 32;
/// Output rows computed per cube (8 × 32 threads = 256-thread cubes).
pub const ROWS_PER_CUBE: u32 = 8;
/// Threads cooperating on one output row (= one Apple simdgroup).
pub const THREADS_PER_ROW: u32 = 32;

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

/// `y[row] = Σ_k dequant(wq[row,k]) · x[k]` — dequant-in-kernel q4 matvec.
///
/// One 32-thread group per output row; thread `lane` owns scale-groups
/// `lane, lane+32, …`. Per group: 1 f16 scale load + 4 u32 word loads
/// (= 32 nibbles), all register-dequantized, f32 accumulation, shared-memory
/// tree reduce across the 32 lanes. Requires `in_dim % 32 == 0` and
/// `out rows % ROWS_PER_CUBE == 0` (all Moshi dims qualify).
/// `swiglu_pairs` (comptime): the fused gate‖up epilogue — rows are
/// interleaved (even row `2j` = gate_j, odd row `2j+1` = up_j) and lane 0 of
/// each even row writes `y[j] = silu(g)·u` (`y` has `out/2` elements) instead
/// of the raw row sums. Same arithmetic as a separate SwiGLU dispatch over
/// the two row sums — the dot products and the silu expression are
/// unchanged — it only deletes the intermediate buffer round-trip and the
/// extra dispatch.
#[cube(launch_unchecked)]
#[allow(clippy::manual_is_multiple_of)] // `%` is the cube-kernel primitive
fn q4_matvec_kernel(
    x: &Array<Vector<f32, Const<4>>>,
    wq: &Array<u32>,
    scales: &Array<f16>,
    y: &mut Array<f32>,
    #[comptime] in_dim: u32,
    #[comptime] rows_per_cube: u32,
    #[comptime] threads_per_row: u32,
    #[comptime] swiglu_pairs: bool,
) {
    let lane = UNIT_POS_X % threads_per_row;
    let row = CUBE_POS_X * rows_per_cube + UNIT_POS_X / threads_per_row;
    let groups = in_dim / 32;
    let words_per_row = in_dim / 8;

    // Two packed words (16 nibbles) per iteration, lane-strided so consecutive
    // threads read consecutive word-pairs (coalesced). The nibble unpack is
    // hand-unrolled with CONSTANT shift amounts and split across two
    // independent partial sums. Evolution, measured on M4 Max at the Moshi
    // step (all 224 temporal matvecs, effective = q4 bytes / time; runs on an
    // active desktop vary ~±25% with WindowServer/WebKit GPU contention —
    // numbers below are same-session comparisons, best observed for this
    // kernel is 310 GB/s effective on the quietest run):
    //   runtime `>> (n*4)` loop:   67 GB/s — ALU-issue-bound, the entire
    //                              bandwidth win thrown away (slower than f16)
    //   1-word constant unroll:   156 GB/s
    //   2-word / 2 accumulators:  219 GB/s  <- this kernel
    //   4-word / 4 accumulators:  182 GB/s  (register pressure regression)
    // The independent s0/s1 chains break the serial FMA dependency on the
    // accumulator; the scale is re-fetched per pair (adjacent lanes share it —
    // L1 broadcast, not extra traffic), and pairs never straddle a scale
    // group (4 words/group), so one fetch per pair is exact.
    // The 16 activations per iteration arrive as four vec4<f32> loads
    // (was 16 scalar loads — this kernel is ALU-issue-bound, so load-issue
    // width is the cheap win); components are consumed in the original
    // order, so the FMA sequence — and every result bit — is unchanged.
    let mut acc0 = f32::new(0.0);
    let mut acc1 = f32::new(0.0);
    let pairs = words_per_row / 2;
    let mut p = lane;
    while p < pairs {
        let w = p * 2;
        let word0 = wq[(row * words_per_row + w) as usize];
        let word1 = wq[(row * words_per_row + w + 1) as usize];
        let d = f32::cast_from(scales[(row * groups + w / 4) as usize]);
        let x0 = x[(w * 2) as usize];
        let x1 = x[(w * 2 + 1) as usize];
        let x2 = x[(w * 2 + 2) as usize];
        let x3 = x[(w * 2 + 3) as usize];
        let mut s0 = (f32::cast_from(word0 & 15) - 8.0) * x0[0];
        let mut s1 = (f32::cast_from(word1 & 15) - 8.0) * x2[0];
        s0 += (f32::cast_from((word0 >> 4) & 15) - 8.0) * x0[1];
        s1 += (f32::cast_from((word1 >> 4) & 15) - 8.0) * x2[1];
        s0 += (f32::cast_from((word0 >> 8) & 15) - 8.0) * x0[2];
        s1 += (f32::cast_from((word1 >> 8) & 15) - 8.0) * x2[2];
        s0 += (f32::cast_from((word0 >> 12) & 15) - 8.0) * x0[3];
        s1 += (f32::cast_from((word1 >> 12) & 15) - 8.0) * x2[3];
        s0 += (f32::cast_from((word0 >> 16) & 15) - 8.0) * x1[0];
        s1 += (f32::cast_from((word1 >> 16) & 15) - 8.0) * x3[0];
        s0 += (f32::cast_from((word0 >> 20) & 15) - 8.0) * x1[1];
        s1 += (f32::cast_from((word1 >> 20) & 15) - 8.0) * x3[1];
        s0 += (f32::cast_from((word0 >> 24) & 15) - 8.0) * x1[2];
        s1 += (f32::cast_from((word1 >> 24) & 15) - 8.0) * x3[2];
        s0 += (f32::cast_from(word0 >> 28) - 8.0) * x1[3];
        s1 += (f32::cast_from(word1 >> 28) - 8.0) * x3[3];
        acc0 += d * s0;
        acc1 += d * s1;
        p += threads_per_row;
    }
    let acc = acc0 + acc1;

    let mut red = SharedMemory::<f32>::new(comptime!((rows_per_cube * threads_per_row) as usize));
    red[UNIT_POS_X as usize] = acc;
    sync_cube();
    let mut stride = u32::new((threads_per_row / 2) as i64);
    while stride > 0 {
        if lane < stride {
            red[UNIT_POS_X as usize] =
                red[UNIT_POS_X as usize] + red[(UNIT_POS_X + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if comptime![swiglu_pairs] {
        let local_row = UNIT_POS_X / threads_per_row;
        if lane == 0 && local_row % 2 == 0 {
            let g = red[UNIT_POS_X as usize];
            let u = red[(UNIT_POS_X + threads_per_row) as usize];
            y[(row / 2) as usize] = g / (1.0 + (-g).exp()) * u;
        }
    } else if lane == 0 {
        y[row as usize] = red[UNIT_POS_X as usize];
    }
}

/// f16-weight twin of [`q4_matvec_kernel`] — identical thread/reduce shape,
/// raw `[out, in]` f16 rows, with VECTORIZED loads: each lane-iteration's 8
/// weights arrive as two `vec4<f16>` loads (8 B) and the 8 activations as
/// two `vec4<f32>` loads, replacing eight scalar 2-byte + eight scalar
/// 4-byte loads. The f16 stack is pure weight bandwidth and the scalar
/// build left load-issue width on the table (~400 GB/s of the M4 Max's
/// ~546). The per-row FMA ORDER is unchanged — the same eight sequential
/// scalar FMAs per iteration, components extracted in order — so results
/// stay bit-identical to the scalar build (gated).
/// `swiglu_pairs`: same fused gate‖up epilogue as [`q4_matvec_kernel`].
#[cube(launch_unchecked)]
#[allow(clippy::manual_is_multiple_of)] // `%` is the cube-kernel primitive
fn f16_matvec_kernel(
    x: &Array<Vector<f32, Const<4>>>,
    w: &Array<Vector<f16, Const<4>>>,
    y: &mut Array<f32>,
    #[comptime] in_dim: u32,
    #[comptime] rows_per_cube: u32,
    #[comptime] threads_per_row: u32,
    #[comptime] swiglu_pairs: bool,
) {
    let lane = UNIT_POS_X % threads_per_row;
    let row = CUBE_POS_X * rows_per_cube + UNIT_POS_X / threads_per_row;
    let octs = in_dim / 8;

    // 8 weights per lane-strided iteration — mirrors the q4 kernel's
    // word-per-iteration shape so the two differ only in bytes fetched.
    let mut acc = f32::new(0.0);
    let mut o = lane;
    while o < octs {
        let wb = row * (in_dim / 4) + o * 2;
        let xb = o * 2;
        let w0 = w[wb as usize];
        let w1 = w[(wb + 1) as usize];
        let x0 = x[xb as usize];
        let x1 = x[(xb + 1) as usize];
        acc += f32::cast_from(w0[0]) * x0[0];
        acc += f32::cast_from(w0[1]) * x0[1];
        acc += f32::cast_from(w0[2]) * x0[2];
        acc += f32::cast_from(w0[3]) * x0[3];
        acc += f32::cast_from(w1[0]) * x1[0];
        acc += f32::cast_from(w1[1]) * x1[1];
        acc += f32::cast_from(w1[2]) * x1[2];
        acc += f32::cast_from(w1[3]) * x1[3];
        o += threads_per_row;
    }

    let mut red = SharedMemory::<f32>::new(comptime!((rows_per_cube * threads_per_row) as usize));
    red[UNIT_POS_X as usize] = acc;
    sync_cube();
    let mut stride = u32::new((threads_per_row / 2) as i64);
    while stride > 0 {
        if lane < stride {
            red[UNIT_POS_X as usize] =
                red[UNIT_POS_X as usize] + red[(UNIT_POS_X + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if comptime![swiglu_pairs] {
        let local_row = UNIT_POS_X / threads_per_row;
        if lane == 0 && local_row % 2 == 0 {
            let g = red[UNIT_POS_X as usize];
            let u = red[(UNIT_POS_X + threads_per_row) as usize];
            y[(row / 2) as usize] = g / (1.0 + (-g).exp()) * u;
        }
    } else if lane == 0 {
        y[row as usize] = red[UNIT_POS_X as usize];
    }
}

// ---------------------------------------------------------------------------
// host-side quantization (CPU, done once at weight load)
// ---------------------------------------------------------------------------

/// Quantize a row-major `[out, in]` f32 weight to (packed nibbles, f16 scales).
///
/// GGUF-Q4_0 convention per 32-weight group: `d = w[argmax |w|] / -8` (so the
/// max-magnitude weight lands exactly on code 0), `q = round(w/d) + 8` clamped
/// to `0..=15`. The scale is rounded through f16 *before* quantizing so the
/// stored nibbles are optimal for the scale that will actually be used.
pub fn quantize_q4(w: &[f32], out_dim: usize, in_dim: usize) -> (Vec<u32>, Vec<f16>) {
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(in_dim % GROUP, 0, "in_dim must be a multiple of {GROUP}");
    let words_per_row = in_dim / 8;
    let groups_per_row = in_dim / GROUP;
    let mut wq = vec![0u32; out_dim * words_per_row];
    let mut scales = vec![f16::ZERO; out_dim * groups_per_row];
    for j in 0..out_dim {
        for g in 0..groups_per_row {
            let base = j * in_dim + g * GROUP;
            let grp = &w[base..base + GROUP];
            let mut amax = 0f32;
            let mut mx = 0f32;
            for &v in grp {
                if v.abs() > amax {
                    amax = v.abs();
                    mx = v;
                }
            }
            let ds = f16::from_f32(mx / -8.0);
            let d = ds.to_f32();
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            scales[j * groups_per_row + g] = ds;
            for (i, &v) in grp.iter().enumerate() {
                let q = ((v * id + 8.5).floor() as i32).clamp(0, 15) as u32;
                let k = g * GROUP + i;
                wq[j * words_per_row + k / 8] |= q << (4 * (k % 8));
            }
        }
    }
    (wq, scales)
}

/// Reconstruct the f32 weight the kernel effectively sees — the CPU oracle for
/// the parity gate.
pub fn dequantize_q4(wq: &[u32], scales: &[f16], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let words_per_row = in_dim / 8;
    let groups_per_row = in_dim / GROUP;
    let mut w = vec![0f32; out_dim * in_dim];
    for j in 0..out_dim {
        for k in 0..in_dim {
            let word = wq[j * words_per_row + k / 8];
            let q = (word >> (4 * (k % 8))) & 15;
            let d = scales[j * groups_per_row + k / GROUP].to_f32();
            w[j * in_dim + k] = (q as f32 - 8.0) * d;
        }
    }
    w
}

// ---------------------------------------------------------------------------
// host-side wrapper
// ---------------------------------------------------------------------------

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A q4-quantized Linear weight resident on the GPU: packed nibbles + f16
/// scales, launched via [`q4_matvec_kernel`]. Quantize once at load
/// ([`Q4Linear::from_f32`]), then [`forward`](Q4Linear::forward) per step.
pub struct Q4Linear {
    pub wq: Handle,
    pub scales: Handle,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Q4Linear {
    /// Quantize a row-major `[out, in]` f32 weight and upload it.
    pub fn from_f32(client: &Client, w: &[f32], out_dim: usize, in_dim: usize) -> Self {
        let (wq, scales) = quantize_q4(w, out_dim, in_dim);
        Self::from_packed(client, &wq, &scales, out_dim, in_dim)
    }

    /// Upload pre-packed data (e.g. quantized once and persisted).
    pub fn from_packed(
        client: &Client,
        wq: &[u32],
        scales: &[f16],
        out_dim: usize,
        in_dim: usize,
    ) -> Self {
        assert_eq!(wq.len(), out_dim * in_dim / 8);
        assert_eq!(scales.len(), out_dim * in_dim / GROUP);
        Self {
            wq: client.create_from_slice(as_bytes(wq)),
            scales: client.create_from_slice(as_bytes(scales)),
            out_dim,
            in_dim,
        }
    }

    /// Allocate zeroed packed buffers at the right sizes — **benchmark-only**
    /// (GPU memory traffic is value-independent; timing needs shapes, not
    /// weights).
    pub fn empty(client: &Client, out_dim: usize, in_dim: usize) -> Self {
        Self {
            wq: client.empty(out_dim * in_dim / 2),
            scales: client.empty(out_dim * in_dim / GROUP * 2),
            out_dim,
            in_dim,
        }
    }

    /// Bytes this weight streams per matvec (packed words + scales).
    pub fn bytes(&self) -> usize {
        self.out_dim * self.in_dim / 2 + self.out_dim * self.in_dim / GROUP * 2
    }

    /// Submit `y = Wq · x` (`x`: `in_dim` f32, `y`: `out_dim` f32, both device
    /// buffers). Non-blocking; sync via a readback on `y` (or later).
    pub fn forward(&self, client: &Client, x: &Handle, y: &Handle) {
        self.launch(client, x, y, self.out_dim, false);
    }

    /// Like [`Self::forward`] but computing only the first `rows` output rows
    /// (`rows % ROWS_PER_CUBE == 0`). The weights are row-major, so a row
    /// prefix is a byte prefix: a shorter launch streams strictly fewer bytes.
    pub fn forward_rows(&self, client: &Client, x: &Handle, y: &Handle, rows: usize) {
        self.launch(client, x, y, rows, false);
    }

    /// Fused gate‖up + SwiGLU: the weight rows are interleaved
    /// (even = gate_j, odd = up_j) and `y` receives `silu(g)·u` per pair
    /// (`out_dim/2` f32). One dispatch replaces gate + up + swiglu.
    pub fn forward_swiglu(&self, client: &Client, x: &Handle, y: &Handle) {
        self.launch(client, x, y, self.out_dim, true);
    }

    fn launch(&self, client: &Client, x: &Handle, y: &Handle, rows: usize, swiglu_pairs: bool) {
        assert_eq!(rows as u32 % ROWS_PER_CUBE, 0);
        assert!(rows <= self.out_dim);
        assert_eq!(self.in_dim % GROUP, 0);
        let y_len = if swiglu_pairs { self.out_dim / 2 } else { rows };
        unsafe {
            q4_matvec_kernel::launch_unchecked::<Rt>(
                client,
                CubeCount::new_1d(rows as u32 / ROWS_PER_CUBE),
                CubeDim::new_1d(ROWS_PER_CUBE * THREADS_PER_ROW),
                ArrayArg::from_raw_parts(x.clone(), self.in_dim / 4),
                ArrayArg::from_raw_parts(self.wq.clone(), self.out_dim * self.in_dim / 8),
                ArrayArg::from_raw_parts(self.scales.clone(), self.out_dim * self.in_dim / GROUP),
                ArrayArg::from_raw_parts(y.clone(), y_len),
                self.in_dim as u32,
                ROWS_PER_CUBE,
                THREADS_PER_ROW,
                swiglu_pairs,
            );
        }
    }
}

/// Submit `y = W · x` for a raw row-major `[out, in]` **f16** weight buffer —
/// the controlled baseline sharing [`q4_matvec_kernel`]'s thread shape.
pub fn f16_matvec(
    client: &Client,
    x: &Handle,
    w: &Handle,
    y: &Handle,
    out_dim: usize,
    in_dim: usize,
) {
    f16_launch(client, x, w, y, out_dim, in_dim, false);
}

/// Fused gate‖up + SwiGLU over interleaved f16 rows (see
/// [`Q4Linear::forward_swiglu`]): `y[j] = silu(row_2j·x)·(row_2j+1·x)`,
/// `out_dim/2` outputs.
pub fn f16_matvec_swiglu(
    client: &Client,
    x: &Handle,
    w: &Handle,
    y: &Handle,
    out_dim: usize,
    in_dim: usize,
) {
    f16_launch(client, x, w, y, out_dim, in_dim, true);
}

fn f16_launch(
    client: &Client,
    x: &Handle,
    w: &Handle,
    y: &Handle,
    out_dim: usize,
    in_dim: usize,
    swiglu_pairs: bool,
) {
    assert_eq!(out_dim as u32 % ROWS_PER_CUBE, 0);
    assert_eq!(in_dim % GROUP, 0); // implies the vec4 loads' in_dim % 8 == 0
    let y_len = if swiglu_pairs { out_dim / 2 } else { out_dim };
    unsafe {
        f16_matvec_kernel::launch_unchecked::<Rt>(
            client,
            CubeCount::new_1d(out_dim as u32 / ROWS_PER_CUBE),
            CubeDim::new_1d(ROWS_PER_CUBE * THREADS_PER_ROW),
            ArrayArg::from_raw_parts(x.clone(), in_dim / 4),
            ArrayArg::from_raw_parts(w.clone(), out_dim * in_dim / 4),
            ArrayArg::from_raw_parts(y.clone(), y_len),
            in_dim as u32,
            ROWS_PER_CUBE,
            THREADS_PER_ROW,
            swiglu_pairs,
        );
    }
}

/// The compute client for the default wgpu/Metal device — same client (and
/// queue) burn's `Metal` backends use, so buffers and timings interleave.
pub fn client_for_default_device() -> Client {
    use cubecl::Runtime;
    Rt::client(&Default::default())
}

/// Alias a pile blob's mmap'd bytes straight onto the Metal device as a
/// cubecl buffer — ZERO-COPY (the `register_external_aliased` seam of the
/// cubecl fork; unified memory makes the mmap'd pages directly
/// GPU-addressable). Returns `None` when the bytes aren't mmap-backed (the
/// caller falls back to an upload copy). The registered buffer pins the mmap
/// for its whole life via the keepalive, so the pile/reader may be dropped
/// afterwards. Mirrors the per-tensor body of
/// [`crate::persist::load_gemma4_aliased_from_pile`]; requires the V3 pile's
/// 256-byte record alignment (asserted) so the in-page offset satisfies
/// Metal's buffer-offset alignment.
#[cfg(target_os = "macos")]
pub fn alias_pile_blob(client: &Client, bytes: &anybytes::Bytes) -> Option<Handle> {
    use memmap2::MmapRaw;
    use std::sync::Arc;
    const PAGE: u64 = 16384;
    let blob_ptr = bytes.as_ptr() as u64;
    let nbytes = bytes.len() as u64;
    assert_eq!(
        blob_ptr % 256,
        0,
        "pile blob not 256-aligned — V3 invariant violated"
    );
    // The owner downcast = capability check (mmap?) + region bounds + keepalive.
    let mmap = bytes.clone().downcast_to_owner::<MmapRaw>().ok()?;
    let region_end = mmap.as_ptr() as u64 + mmap.len() as u64;
    let page_start = blob_ptr & !(PAGE - 1);
    let off_in_page = blob_ptr - page_start;
    let page_len = ((blob_ptr + nbytes + PAGE - 1) & !(PAGE - 1)).min(region_end) - page_start;
    let keepalive: Arc<dyn std::any::Any + Send + Sync> = mmap.clone();
    // SAFETY: page_start/page_len is a page-aligned superset of the blob,
    // inside the (page-aligned) mmap which `keepalive` pins for the buffer's
    // life.
    Some(unsafe {
        client.register_external_aliased(
            page_start as *mut core::ffi::c_void,
            page_len,
            off_in_page,
            nbytes,
            keepalive,
        )
    })
}

// ---------------------------------------------------------------------------
// tests (pure CPU: format round-trip)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift_fill(n: usize, mut s: u64, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32;
                (u * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[test]
    fn q4_roundtrip_error_is_q4_class() {
        let (out, inn) = (64, 256);
        let w = xorshift_fill(out * inn, 7, 0.05);
        let (wq, scales) = quantize_q4(&w, out, inn);
        let wd = dequantize_q4(&wq, &scales, out, inn);
        let mut num = 0f64;
        let mut den = 0f64;
        for (a, b) in w.iter().zip(&wd) {
            num += ((a - b) as f64).powi(2);
            den += (*a as f64).powi(2);
        }
        let rel = (num / den).sqrt();
        // analytic q4_0 error for uniform[-a,a]: step d ≈ a/8, RMS error
        // d/√12, signal RMS a/√3  =>  rel ≈ (1/8)·(√3/√12) ≈ 0.063
        assert!(
            rel < 0.08,
            "q4 round-trip rel RMS {rel} out of class (expect ~0.063)"
        );
    }

    #[test]
    fn q4_max_magnitude_weight_is_near_exact() {
        // the argmax-|w| element maps to code 0 => reconstructs to -8·d = mx
        // exactly up to the f16 rounding of d itself
        let mut w = vec![0.01f32; GROUP];
        w[13] = -0.5;
        let (wq, scales) = quantize_q4(&w, 1, GROUP);
        let wd = dequantize_q4(&wq, &scales, 1, GROUP);
        assert!(
            (wd[13] - w[13]).abs() < 3e-4,
            "amax element should be f16-exact, got {} vs {}",
            wd[13],
            w[13]
        );
    }

    #[test]
    fn q4_zero_group_roundtrips_to_zero() {
        let w = vec![0f32; GROUP * 2];
        let (wq, scales) = quantize_q4(&w, 1, GROUP * 2);
        let wd = dequantize_q4(&wq, &scales, 1, GROUP * 2);
        assert!(wd.iter().all(|v| *v == 0.0));
    }
}
