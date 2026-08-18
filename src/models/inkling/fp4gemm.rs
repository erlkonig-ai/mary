//! Routed-expert FFN on the native NVFP4 tensor-core path.
//!
//! The existing device lane decodes each expert's packed E2M1/E4M3 blocks into
//! a full f32 matrix on the device and then multiplies that in f32. For one
//! expert that materialises 67.1 MB (w13) + 33.6 MB (w2) of f32 that is read
//! back once and thrown away. This module skips the decode: the packed bytes go
//! straight into `mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X…ue4m3`,
//! the instruction `nvfp4_mma_probe` proved CubeCL reaches on sm_121a.
//!
//! ## Why the activations are also 4-bit
//!
//! The MMA takes E2M1 for BOTH operands — there is no mixed f32xE2M1 form at
//! `kind::mxf4nvf4` — so the activation has to be quantised too. That is not a
//! liberty taken for speed: the checkpoint's own `hf_quant_config.json`
//! specifies
//!
//!     "*input_quantizer": num_bits [2,1], block_sizes {-1: 16,
//!                         type: "dynamic", scale_bits: [4,3]}, enable: true
//!
//! i.e. E2M1 activations in dynamic per-16 blocks with E4M3 scales — exactly
//! what this path feeds the instruction. The f32-activation lane it replaces is
//! the one deviating from the checkpoint's intended numerics, not this one.
//!
//! ## What this is NOT
//!
//! It is not a fast GEMM. Each plane owns one 16x8 output tile and streams its
//! own eight weight rows from global memory, so the weights are read exactly
//! once and the activation (a few KB) is re-read per plane out of L2. For the
//! shape this lane actually runs — M is the handful of tokens that routed to
//! one expert, against N=4096, K=4096 — the arithmetic intensity is so low that
//! the kernel is bound by streaming the weights in, and a fancier tiling would
//! not change that. `inkling_forward`'s own per-pass report breaks the lane
//! into slice / bind+enqueue+sync / remainder, which is where that number comes
//! from now that the lane-comparison bench is gone with the lanes it compared.

use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::{e2m1x2, e4m3};

/// Rows of one MMA tile — the M granularity everything here is padded to.
pub const MTILE: usize = 16;
/// Columns of one MMA tile.
pub const NTILE: usize = 8;
/// K covered by one `m16n8k64` instruction.
pub const KTILE: usize = 64;
/// Logical elements per E4M3 block scale.
pub const GROUP: usize = 16;

/// `out = (a @ b^T) * scale`, with `a` and `b` both NVFP4.
///
/// `a` is `[m_pad, k/2]` packed bytes and `a_sc` `[m_pad, k/16]` E4M3 scales;
/// `b` is `[n, k/2]` / `[n, k/16]`, i.e. the checkpoint's own `[out, in]`
/// orientation, which is already the column-major B the instruction wants.
/// `out` is `[m_pad, n]` f32.
///
/// One plane per `(m_tile, n_tile)`; the K loop accumulates in the MMA's own
/// f32 accumulator, which measured closer to an f64 sum than a sequential f32
/// host lane over the same products.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear<AB: Scalar, S: Scalar, NA: Size, NC: Size>(
    a: &Tensor<Vector<AB, NA>>,
    a_sc: &Tensor<S>,
    b: &Tensor<Vector<AB, NA>>,
    b_sc: &Tensor<S>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(MTILE, NTILE, KTILE, 4usize);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let n_tile = CUBE_POS_X as usize;
    let m_tile = CUBE_POS_Y as usize;
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();
    let sia = def.scales_index(lane, MatrixIdent::A) as usize;
    let sib = def.scales_index(lane, MatrixIdent::B) as usize;
    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE);

    for t in 0..k_tiles {
        let kbase = t * KTILE;
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
            let gr = row as usize + m_base;
            let gc = col as usize + kbase;
            reg_a[i] = a[(gr * size_k / 2 + gc / 2) / a.vector_size()];
        }
        #[unroll]
        for i in 0..vc_b {
            let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
            let gr = col as usize + n_base;
            let gc = row as usize + kbase;
            reg_b[i] = b[(gr * size_k / 2 + gc / 2) / b.vector_size()];
        }

        let mut sa = Vector::<S, NS>::empty();
        let mut sb = Vector::<S, NS>::empty();
        #[unroll]
        for i in 0..scales_count {
            sa[i] = a_sc[(sia + m_base) * spr + t * 4 + i];
            sb[i] = b_sc[(sib + n_base) * spr + t * 4 + i];
        }

        let d = def.execute_scaled(&reg_a, &reg_b, &acc, sa, sb);
        #[unroll]
        for i in 0..vc_c {
            acc[i] = d[i];
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let gr = row as usize + m_base;
        let gc = col as usize + n_base;
        out[(gr * size_n + gc) / out.vector_size()] =
            acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// De-interleave the fused gate/up result and apply the gate, in one pass.
///
/// The checkpoint stores w13's output rows alternating `g0, u0, g1, u1, …`, so
/// after `out = x @ w13^T` column `2i` is the gate and `2i + 1` the up. Doing
/// the de-interleave here, on the `[m, 2*inter]` result, moves it off the
/// `[2*inter, hidden]` weight — 16x2048 elements touched instead of 4096x4096.
#[cube(launch)]
pub fn gate_up_silu<O: Scalar + Cast>(
    both: &Tensor<f32>,
    act: &mut Tensor<O>,
    #[comptime] inter: usize,
    #[comptime] halved: bool,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx < act.len() {
        let r = idx / inter;
        let i = idx % inter;
        // Two readings of w13's output axis are live in this tree: INTERLEAVED
        // (g0,u0,g1,u1,...) and HALVED (all gates, then all ups). They are
        // shape-identical and numerically different, which is exactly the kind
        // of thing that passes every parity gate built on the same assumption.
        // `halved` exists so the question can be settled by running it.
        let (g, u) = if comptime!(halved) {
            (both[r * 2 * inter + i], both[r * 2 * inter + inter + i])
        } else {
            (both[r * 2 * inter + 2 * i], both[r * 2 * inter + 2 * i + 1])
        };
        // The output type is the NEXT matmul's operand type: f32 for the NVFP4
        // lane, whose second GEMM re-quantises from f32 anyway, and bf16 for
        // the layer-2 lane, whose second GEMM takes bf16 directly. ONE
        // implementation of the interleave, because the INTERLEAVED/HALVED
        // question above is exactly the kind that a second transcription gets
        // silently wrong. `O::cast_from` is the identity when `O` is f32.
        act[idx] = O::cast_from((g / (1.0f32 + Exp::exp(-g))) * u);
    }
}

/// Launch [`fp4_linear`] for a `[m_pad, k] x [n, k]^T` product.
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    a_sc: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Handle {
    assert_eq!(m_pad % MTILE, 0, "m_pad {m_pad} is not a multiple of {MTILE}");
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let vs = 32 / e2m1x2::cube_type().size_bits();
    let spr = k / GROUP;

    unsafe {
        fp4_linear::launch::<e2m1x2, e4m3, R>(
            client,
            CubeCount::Static((n / NTILE) as u32, (m_pad / MTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            TensorArg::from_raw_parts(a.clone(), [k / 2, 1].into(), [m_pad, k / 2].into()),
            TensorArg::from_raw_parts(a_sc.clone(), [spr, 1].into(), [m_pad, spr].into()),
            TensorArg::from_raw_parts(b.clone(), [k / 2, 1].into(), [n, k / 2].into()),
            TensorArg::from_raw_parts(b_sc.clone(), [spr, 1].into(), [n, spr].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_pad, n].into()),
            k,
            n,
            scale,
        )
    };
    out
}

/// Launch [`gate_up_silu`] over an `[m_pad, 2 * inter]` fused result, f32 out.
pub fn gate_up_silu_launch<R: Runtime>(
    client: &ComputeClient<R>,
    both: &Handle,
    m_pad: usize,
    inter: usize,
) -> Handle {
    gate_up_silu_launch_as::<f32, R>(client, both, m_pad, inter)
}

/// The same, BF16 out — what the layer-2 lane feeds straight back into the MMA.
///
/// A separate entry point rather than a turbofish at the call site so the two
/// lanes read the same, and so nothing but the element type differs between
/// them.
pub fn gate_up_silu_bf16_launch<R: Runtime>(
    client: &ComputeClient<R>,
    both: &Handle,
    m_pad: usize,
    inter: usize,
) -> Handle {
    gate_up_silu_launch_as::<half::bf16, R>(client, both, m_pad, inter)
}

fn gate_up_silu_launch_as<O: Scalar + Cast + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    both: &Handle,
    m_pad: usize,
    inter: usize,
) -> Handle {
    // INK_W13_HALVED=1 selects the contiguous reading, for the A/B.
    let halved = std::env::var("INK_W13_HALVED").map(|v| v == "1").unwrap_or(false);
    let n = m_pad * inter;
    let act = client.empty(n * core::mem::size_of::<O>());
    let threads = 256u32;
    let blocks = n.div_ceil(threads as usize) as u32;
    unsafe {
        gate_up_silu::launch::<O, R>(
            client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(threads),
            TensorArg::from_raw_parts(both.clone(), [2 * inter, 1].into(), [m_pad, 2 * inter].into()),
            TensorArg::from_raw_parts(act.clone(), [inter, 1].into(), [m_pad, inter].into()),
            inter,
            halved,
        )
    };
    act
}

// ---------------------------------------------------------------------------
// Activation quantisation
// ---------------------------------------------------------------------------

/// Quantise activations to NVFP4: E2M1 codes, one E4M3 scale per 16.
///
/// One unit per 16-element block. `x` is `[rows, k]` f32 flattened; `codes` is
/// `[rows, k/8]` u32 with element `i` of a block at bits `4*(i%8)` of word
/// `i/8` (low nibble first, so the bytes match the checkpoint's own packing and
/// the same buffer can be bound as `e2m1x2`); `scales` is `[rows, k/16]` E4M3.
///
/// The recipe is the checkpoint's: `scale = amax/6` rounded to E4M3, then each
/// element rounded to the nearest E2M1 code of `x/scale`. Rounding is
/// round-to-nearest with exact midpoints going AWAY from zero (a midpoint lands
/// on the `<` boundary and falls through to the larger code). An all-zero block
/// yields a zero scale byte and zero codes.
#[cube(launch)]
pub fn quantize_act(x: &Tensor<f32>, codes: &mut Tensor<u32>, scales: &mut Tensor<e4m3>) {
    let blk = ABSOLUTE_POS as usize;
    if blk < scales.len() {
        let base = blk * GROUP;

        let mut amax = 0.0f32;
        #[unroll]
        for i in 0..GROUP {
            let v = Abs::abs(x[base + i]);
            if v > amax {
                amax = v;
            }
        }

        // Round the block scale through E4M3 and read back what it became: the
        // codes have to be computed against the scale the MMA will actually
        // apply, not the exact amax/6 the host imagined.
        let sq = e4m3::cast_from(amax / 6.0f32);
        let s = f32::cast_from(sq);
        scales[blk] = sq;

        let mut w0 = 0u32;
        let mut w1 = 0u32;
        if s > 0.0f32 {
            let inv = 1.0f32 / s;
            #[unroll]
            for i in 0..GROUP {
                let q = x[base + i] * inv;
                let a = Abs::abs(q);
                // magnitude grid 0, .5, 1, 1.5, 2, 3, 4, 6 -> midpoints below
                let mut m = 7u32;
                if a < 0.25f32 {
                    m = 0u32;
                } else if a < 0.75f32 {
                    m = 1u32;
                } else if a < 1.25f32 {
                    m = 2u32;
                } else if a < 1.75f32 {
                    m = 3u32;
                } else if a < 2.5f32 {
                    m = 4u32;
                } else if a < 3.5f32 {
                    m = 5u32;
                } else if a < 5.0f32 {
                    m = 6u32;
                }
                let c = if q < 0.0f32 { m + 8u32 } else { m };
                if i < 8 {
                    w0 |= c << (4 * i as u32);
                } else {
                    w1 |= c << (4 * (i - 8) as u32);
                }
            }
        }
        codes[2 * blk] = w0;
        codes[2 * blk + 1] = w1;
    }
}

/// Host-side twin of [`quantize_act`], for gates and for the CPU lane.
///
/// Returns `(packed_bytes, scale_bytes)` in exactly the layout the device
/// kernel writes, so a gate can compare them bitwise.
pub fn quantize_act_host(x: &[f32], k: usize) -> (Vec<u8>, Vec<u8>) {
    use crate::models::inkling::nvfp4::e4m3_to_f32;
    assert_eq!(x.len() % k, 0, "x is not a whole number of rows of {k}");
    assert_eq!(k % GROUP, 0, "{k} is not a multiple of {GROUP}");
    let nblocks = x.len() / GROUP;
    let mut codes = vec![0u8; x.len() / 2];
    let mut scales = vec![0u8; nblocks];
    for b in 0..nblocks {
        let base = b * GROUP;
        let amax = (0..GROUP).map(|i| x[base + i].abs()).fold(0.0f32, f32::max);
        let sb = f32_to_e4m3(amax / 6.0);
        scales[b] = sb;
        let s = e4m3_to_f32(sb);
        if !(s > 0.0) {
            continue;
        }
        for i in 0..GROUP {
            let q = x[base + i] / s;
            let a = q.abs();
            let m: u8 = if a < 0.25 {
                0
            } else if a < 0.75 {
                1
            } else if a < 1.25 {
                2
            } else if a < 1.75 {
                3
            } else if a < 2.5 {
                4
            } else if a < 3.5 {
                5
            } else if a < 5.0 {
                6
            } else {
                7
            };
            let c = if q < 0.0 { m + 8 } else { m };
            let j = base + i;
            if j % 2 == 0 {
                codes[j / 2] |= c;
            } else {
                codes[j / 2] |= c << 4;
            }
        }
    }
    (codes, scales)
}

/// Round a non-negative f32 to the nearest E4M3 (bias 7, 3 mantissa bits) byte.
///
/// Exhaustive rather than clever: E4M3FN has 256 patterns and the finite
/// non-negative ones are 128, so scanning them is both obviously correct and
/// fast enough for the few thousand block scales an expert needs. A hand-rolled
/// bit twiddle is what got the subnormal branch of `e4m3_to_f32` wrong once
/// already.
pub fn f32_to_e4m3(v: f32) -> u8 {
    use crate::models::inkling::nvfp4::e4m3_to_f32;
    if !(v > 0.0) {
        return 0;
    }
    let mut best = 0u8;
    let mut bestd = f32::INFINITY;
    for b in 0u16..128 {
        let d = e4m3_to_f32(b as u8);
        if !d.is_finite() {
            continue;
        }
        let e = (d - v).abs();
        if e < bestd {
            bestd = e;
            best = b as u8;
        }
    }
    best
}

/// Quantise an `[rows, k]` f32 host slice and upload it, ready for
/// [`fp4_linear_launch`]. `rows` is padded up to a multiple of [`MTILE`].
pub fn upload_quantized_act<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    rows: usize,
    k: usize,
) -> (Handle, Handle, usize) {
    let m_pad = rows.div_ceil(MTILE) * MTILE;
    let mut padded = vec![0f32; m_pad * k];
    padded[..rows * k].copy_from_slice(&x[..rows * k]);
    let (codes, scales) = quantize_act_host(&padded, k);
    (
        client.create_from_slice(&codes),
        client.create_from_slice(&scales),
        m_pad,
    )
}

// ---------------------------------------------------------------------------
// Zero copy
// ---------------------------------------------------------------------------

/// Every host mapping a weight source reads through, registered with the GPU
/// **once**.
///
/// The obvious way to alias — call
/// [`ComputeClient::register_external_aliased`] per expert slab — is a trap on
/// this backend, and measurably so: it cost 2.6 s more per forward than simply
/// copying. `create_from_slice` posts its work with `submit`, which returns
/// immediately, but `register_external_aliased` has to hand back a `Handle` the
/// server constructs, so it uses `submit_blocking` — a synchronous round trip
/// to the device thread. At four slabs per expert and ~9950 expert-loads that
/// is ~40 000 blocking hops, and they cost far more than the copies they save.
///
/// Registering is per *mapping*: nine round trips for a sharded checkpoint, ONE
/// for a pile. Everything after that is [`Handle::offset_start`], pure
/// arithmetic on the client side.
///
/// A slab is located by POINTER CONTAINMENT rather than by re-deriving its file
/// offset from a tensor name, an expert index and a shape. That is what lets one
/// implementation serve both sources — the caller already holds the borrowed
/// bytes, and where they live is a fact about the pointer, not something to
/// recompute and get subtly wrong.
pub struct Aliases {
    /// `(base address, length, registered handle)` per mapping.
    maps: Vec<(usize, usize, Handle)>,
    /// What the binds actually did. See [`BindStats`].
    stats: BindCounters,
}

/// Why one bind did or did not become a zero-copy alias.
///
/// The distinction between the two copy causes is the whole value of counting:
/// an unaligned copy is a fact about how the SOURCE lays its bytes out and is
/// fixable by changing the source, while an unmapped one means the registration
/// never happened and no amount of alignment will help. A single "copied"
/// counter conflates a data-layout problem with a setup problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    /// Aliased in place — the GPU reads the source's own pages.
    Alias,
    /// Copied because the slab's ADDRESS is not 4-byte aligned. Carries the
    /// residue, because WHICH residue says where the misalignment came from: a
    /// safetensors shard packs tensors back to back with no padding, so a
    /// residue of 2 is an odd number of BF16 elements sitting upstream.
    CopyUnaligned(usize),
    /// Copied because the slab lives in no registered mapping — a `Vec` the
    /// caller built, or a source whose mappings were never registered.
    CopyUnmapped,
    /// Nothing to bind.
    Empty,
}

/// Interior-mutable counters. `slice_or_copy` takes `&self` because every
/// caller holds a shared reference, so the accounting has to be atomic rather
/// than `&mut`.
#[derive(Default)]
struct BindCounters {
    alias_calls: core::sync::atomic::AtomicU64,
    alias_bytes: core::sync::atomic::AtomicU64,
    copy_calls: core::sync::atomic::AtomicU64,
    copy_bytes: core::sync::atomic::AtomicU64,
    copy_nanos: core::sync::atomic::AtomicU64,
    /// Unaligned copies by residue mod 4; index 0 is unused.
    unaligned: [core::sync::atomic::AtomicU64; 4],
    unmapped: core::sync::atomic::AtomicU64,
}

/// What the binds of one run cost, split by whether they aliased.
///
/// Not a profiler. It answers the one question the zero-copy seam exists to
/// answer — how much of the weight traffic actually avoided a copy — and, when
/// the answer is "not all of it", which of the two reasons was to blame.
#[derive(Default, Clone, Copy, Debug)]
pub struct BindStats {
    pub alias_calls: u64,
    pub alias_bytes: u64,
    pub copy_calls: u64,
    pub copy_bytes: u64,
    /// HOST time inside `create_from_slice`. The copy itself is posted
    /// asynchronously, so this is the staging and enqueue, not the DMA.
    pub copy_nanos: u64,
    /// Unaligned copies by residue mod 4; index 0 is unused.
    pub unaligned: [u64; 4],
    pub unmapped: u64,
}

impl BindStats {
    pub fn calls(&self) -> u64 {
        self.alias_calls + self.copy_calls
    }

    /// Fraction of BINDS that aliased. `None` when nothing was bound, which is
    /// the honest answer — a rate over zero calls is not 0% or 100%.
    pub fn alias_fraction(&self) -> Option<f64> {
        match self.calls() {
            0 => None,
            n => Some(self.alias_calls as f64 / n as f64),
        }
    }

    /// Fraction of BYTES that aliased. Different from the call fraction
    /// whenever the two classes are different sizes, which they are here: a
    /// code plane is eight times a scale plane.
    pub fn alias_byte_fraction(&self) -> Option<f64> {
        match self.alias_bytes + self.copy_bytes {
            0 => None,
            n => Some(self.alias_bytes as f64 / n as f64),
        }
    }

    pub fn report(&self) -> String {
        let mb = |b: u64| b as f64 / (1u64 << 20) as f64;
        let mut s = String::new();
        s.push_str(&format!(
            "    bind ALIAS  {:8} calls  {:10.0} MiB   0.000 s\n",
            self.alias_calls,
            mb(self.alias_bytes)
        ));
        s.push_str(&format!(
            "    bind COPY   {:8} calls  {:10.0} MiB  {:6.3} s\n",
            self.copy_calls,
            mb(self.copy_bytes),
            self.copy_nanos as f64 / 1e9
        ));
        match (self.alias_fraction(), self.alias_byte_fraction()) {
            (Some(c), Some(b)) => s.push_str(&format!(
                "    aliased     {:8.1}% of binds, {:.1}% of bytes\n",
                c * 100.0,
                b * 100.0
            )),
            _ => s.push_str("    aliased     (nothing was bound)\n"),
        }
        // Only printed when there is something to explain. A line of zeroes
        // reads as a finding.
        if self.copy_calls > 0 {
            let residues: Vec<String> = (1..4)
                .filter(|i| self.unaligned[*i] > 0)
                .map(|i| format!("{} at addr%4=={i}", self.unaligned[i]))
                .collect();
            if !residues.is_empty() {
                s.push_str(&format!("    copied because UNALIGNED: {}\n", residues.join(", ")));
            }
            if self.unmapped > 0 {
                s.push_str(&format!(
                    "    copied because UNMAPPED : {} (outside every registered mapping)\n",
                    self.unmapped
                ));
            }
        }
        s
    }
}

impl Aliases {
    /// Register every mapping of a source. `None` if the device cannot address
    /// host memory directly.
    pub fn register<R: Runtime>(
        client: &ComputeClient<R>,
        mappings: Vec<(usize, usize, std::sync::Arc<dyn core::any::Any + Send + Sync>)>,
    ) -> Option<Self> {
        if !cubecl::cuda::supports_zero_copy_host(0) {
            return None;
        }
        let mut maps = Vec::with_capacity(mappings.len());
        for (base, len, keep) in mappings {
            // SAFETY: the mapping is read-only and `keep` holds it for as long
            // as the handle lives; cubecl pins external handles immutable.
            let h = unsafe {
                client.register_external_aliased(
                    base as *mut core::ffi::c_void,
                    len as u64,
                    0,
                    len as u64,
                    keep,
                )
            };
            maps.push((base, len, h));
        }
        Some(Aliases { maps, stats: BindCounters::default() })
    }

    /// An `Aliases` that aliases NOTHING, so the copying lane is still counted.
    ///
    /// Without this, `INK_ZEROCOPY=0` and "this device cannot alias" are both
    /// spelled `None` at the call site and neither reports what it moved — the
    /// A/B has a measured side and an unmeasured one, which is not an A/B.
    pub fn disabled() -> Self {
        Aliases { maps: Vec::new(), stats: BindCounters::default() }
    }

    pub fn len(&self) -> usize {
        self.maps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.maps.is_empty()
    }

    /// The binds so far.
    pub fn stats(&self) -> BindStats {
        use core::sync::atomic::Ordering::Relaxed;
        BindStats {
            alias_calls: self.stats.alias_calls.load(Relaxed),
            alias_bytes: self.stats.alias_bytes.load(Relaxed),
            copy_calls: self.stats.copy_calls.load(Relaxed),
            copy_bytes: self.stats.copy_bytes.load(Relaxed),
            copy_nanos: self.stats.copy_nanos.load(Relaxed),
            unaligned: [
                0,
                self.stats.unaligned[1].load(Relaxed),
                self.stats.unaligned[2].load(Relaxed),
                self.stats.unaligned[3].load(Relaxed),
            ],
            unmapped: self.stats.unmapped.load(Relaxed),
        }
    }

    /// Zero the counters, so a per-token figure is a per-token figure.
    pub fn stats_reset(&self) {
        use core::sync::atomic::Ordering::Relaxed;
        self.stats.alias_calls.store(0, Relaxed);
        self.stats.alias_bytes.store(0, Relaxed);
        self.stats.copy_calls.store(0, Relaxed);
        self.stats.copy_bytes.store(0, Relaxed);
        self.stats.copy_nanos.store(0, Relaxed);
        for u in &self.stats.unaligned {
            u.store(0, Relaxed);
        }
        self.stats.unmapped.store(0, Relaxed);
    }

    /// What [`Aliases::slice`] would decide, and WHY — without binding anything.
    ///
    /// Split out from `slice` so the decision can be audited over a whole model
    /// without a GPU and without a `Handle` per leaf. Both `slice` and the audit
    /// call this, so a check that says "every leaf aliases" is reading the same
    /// predicate the runtime does rather than a second transcription of it.
    pub fn classify(&self, data: &[u8]) -> Bind {
        if data.is_empty() {
            return Bind::Empty;
        }
        let p = data.as_ptr() as usize;
        if p % 4 != 0 {
            return Bind::CopyUnaligned(p % 4);
        }
        match self
            .maps
            .iter()
            .find(|(b, l, _)| p >= *b && p + data.len() <= b + l)
        {
            Some(_) => Bind::Alias,
            None => Bind::CopyUnmapped,
        }
    }

    /// A borrowed slice as a zero-copy offset view of the mapping it lives in.
    ///
    /// `None` when the slice is not 4-byte aligned — the expert GEMM issues
    /// 4-byte vector loads, so that is the real bound and 16 would be a
    /// superstition. It matters that this is the true bound and not a guess:
    /// safetensors packs tensors back to back with no padding, and the
    /// checkpoint puts `w13_weight` and `w2_weight` at offsets congruent to 4
    /// mod 16, so a 16-byte test would refuse every weight slab in the model and
    /// fall back to copying forever while looking like it worked.
    ///
    /// `None` also when the slice belongs to no registered mapping, which is the
    /// honest answer for a `Vec` the caller built.
    pub fn slice(&self, data: &[u8]) -> Option<Handle> {
        if !matches!(self.classify(data), Bind::Alias) {
            return None;
        }
        let p = data.as_ptr() as usize;
        let (base, len, h) = self
            .maps
            .iter()
            .find(|(b, l, _)| p >= *b && p + data.len() <= b + l)?;
        let off = (p - base) as u64;
        Some(
            h.clone()
                .offset_start(off)
                .offset_end(*len as u64 - off - data.len() as u64),
        )
    }

    /// WHERE a borrowed slice lives: `(mapping index, byte offset)`.
    ///
    /// The same pointer-containment lookup [`Aliases::slice`] does, stopping
    /// one step short of building a `Handle`. The grouped routed-expert lane
    /// ([`super::moegroup`]) needs the offsets and not the handles: it binds
    /// the mapping ONCE for the whole layer and lets the kernel pick an
    /// expert's planes out of it, so a per-expert `Handle` would be twenty-odd
    /// clones of the same pointer for the privilege of throwing them away.
    ///
    /// `None` for the same two reasons `slice` returns `None` — unaligned, or
    /// in no registered mapping — and deliberately WITHOUT counting a bind,
    /// because the caller is still deciding whether it can take this lane at
    /// all. It counts with [`Aliases::note_alias`] once it has committed.
    pub fn locate(&self, data: &[u8]) -> Option<(usize, u64)> {
        if !matches!(self.classify(data), Bind::Alias) {
            return None;
        }
        let p = data.as_ptr() as usize;
        let (i, (base, _, _)) = self
            .maps
            .iter()
            .enumerate()
            .find(|(_, (b, l, _))| p >= *b && p + data.len() <= b + l)?;
        Some((i, (p - base) as u64))
    }

    /// The registered handle for a whole mapping, and its length in bytes.
    ///
    /// Not a slice of it: this is the buffer the grouped GEMM binds, with the
    /// per-expert offsets travelling separately as device data.
    pub fn map(&self, i: usize) -> Option<(Handle, usize)> {
        self.maps.get(i).map(|(_, l, h)| (h.clone(), *l))
    }

    /// Charge `bytes` to the alias counters for a bind that went through
    /// [`Aliases::locate`] rather than [`Aliases::slice_or_copy`].
    ///
    /// The seam moved but the accounting must not: the report's "100% of binds
    /// aliased" line is only worth reading if every weight the device sees is
    /// still counted somewhere, and a lane that quietly stopped reporting would
    /// look like a lane that stopped moving bytes.
    pub fn note_alias(&self, bytes: usize) {
        use core::sync::atomic::Ordering::Relaxed;
        self.stats.alias_calls.fetch_add(1, Relaxed);
        self.stats.alias_bytes.fetch_add(bytes as u64, Relaxed);
    }

    /// [`Aliases::slice`], falling back to an ordinary copy — and COUNTING
    /// which of the two happened.
    ///
    /// The counting is here rather than at the call sites because this is the
    /// seam the question is about: every weight the expert lane hands the GPU
    /// passes through exactly this function, so a total taken here is a total
    /// over the whole lane by construction and cannot miss a path someone
    /// added later.
    pub fn slice_or_copy<R: Runtime>(&self, client: &ComputeClient<R>, data: &[u8]) -> Handle {
        use core::sync::atomic::Ordering::Relaxed;
        let kind = self.classify(data);
        match kind {
            Bind::Alias => {
                self.stats.alias_calls.fetch_add(1, Relaxed);
                self.stats.alias_bytes.fetch_add(data.len() as u64, Relaxed);
                self.slice(data).expect("classified as aliasable")
            }
            _ => {
                match kind {
                    Bind::CopyUnaligned(r) => {
                        self.stats.unaligned[r].fetch_add(1, Relaxed);
                    }
                    Bind::CopyUnmapped => {
                        self.stats.unmapped.fetch_add(1, Relaxed);
                    }
                    _ => {}
                }
                let t = std::time::Instant::now();
                let h = client.create_from_slice(data);
                self.stats.copy_calls.fetch_add(1, Relaxed);
                self.stats.copy_bytes.fetch_add(data.len() as u64, Relaxed);
                self.stats
                    .copy_nanos
                    .fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                h
            }
        }
    }
}

#[cfg(test)]
mod bind_tests {
    use super::*;

    /// The predicate the runtime binds on, checked without a GPU.
    ///
    /// `classify` is the only place the 4-byte rule lives now, so this covers
    /// `slice`, `slice_or_copy` and the offline audit at once.
    #[test]
    fn classify_names_the_reason_not_just_the_verdict() {
        let al = Aliases::disabled();
        // Nothing is registered, so an aligned slice is UNMAPPED, not aliasable
        // — and saying so is the point: it is a different repair.
        let v = vec![0u8; 64];
        let base = v.as_ptr() as usize;
        let pad = (4 - base % 4) % 4;
        assert_eq!(al.classify(&v[pad..pad + 16]), Bind::CopyUnmapped);
        // and a deliberately odd offset reports its residue
        assert_eq!(al.classify(&v[pad + 1..pad + 17]), Bind::CopyUnaligned(1));
        assert_eq!(al.classify(&v[pad + 2..pad + 18]), Bind::CopyUnaligned(2));
        assert_eq!(al.classify(&[]), Bind::Empty);
    }

    /// A rate over zero calls is neither 0% nor 100%, and reporting either
    /// would be a green check over an empty measurement.
    #[test]
    fn an_empty_run_has_no_alias_rate() {
        assert_eq!(BindStats::default().alias_fraction(), None);
        let s = BindStats { alias_calls: 3, copy_calls: 1, ..Default::default() };
        assert_eq!(s.alias_fraction(), Some(0.75));
    }
}
