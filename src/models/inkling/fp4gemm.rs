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
//! not change that. See `inkling_expert_lane_bench` for where the lane's time
//! really goes.

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
pub fn gate_up_silu(
    both: &Tensor<f32>,
    act: &mut Tensor<f32>,
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
        act[idx] = (g / (1.0f32 + Exp::exp(-g))) * u;
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

/// Launch [`gate_up_silu`] over an `[m_pad, 2 * inter]` fused result.
pub fn gate_up_silu_launch<R: Runtime>(
    client: &ComputeClient<R>,
    both: &Handle,
    m_pad: usize,
    inter: usize,
) -> Handle {
    // INK_W13_HALVED=1 selects the contiguous reading, for the A/B.
    let halved = std::env::var("INK_W13_HALVED").map(|v| v == "1").unwrap_or(false);
    let n = m_pad * inter;
    let act = client.empty(n * core::mem::size_of::<f32>());
    let threads = 256u32;
    let blocks = n.div_ceil(threads as usize) as u32;
    unsafe {
        gate_up_silu::launch::<R>(
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
// Where the expert bytes come from
// ---------------------------------------------------------------------------

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;

/// One expert's packed NVFP4 weight, **borrowed** out of the checkpoint mapping.
///
/// The point of the lifetime is that there is no copy: `codes` and `scales`
/// point into the mmap'd shard. `Checkpoint::expert_slice_packed` returns the
/// same data as two owned `Vec<u8>`, which measured at 2.74 ms per expert
/// against 0.0000 ms here, because it re-runs `SafeTensors::deserialize` four
/// times per slab and then copies 12.6 MB out.
pub struct PackedRef<'a> {
    pub codes: &'a [u8],
    pub scales: &'a [u8],
    pub scale2: f32,
    /// Output rows of this expert's matrix.
    pub rows: usize,
    /// Packed bytes per row; the logical width is `2 * cols`.
    pub cols: usize,
    /// The mapping `codes` points into. Cloning this `Arc` is what a zero-copy
    /// handle holds so the pages cannot be unmapped under a running kernel;
    /// codes and scales live in DIFFERENT shards, hence two of them.
    pub codes_keep: std::sync::Arc<Mmap>,
    pub scales_keep: std::sync::Arc<Mmap>,
}

/// The checkpoint's shards, mapped once, with every tensor's extent resolved up
/// front.
///
/// [`crate::models::inkling::load::Checkpoint`] re-opens, re-maps and re-parses
/// a shard on every single accessor call — `shape_of`, `tensor`, and each
/// `with_bytes` — so reading one expert slab costs four full header
/// deserializations of a multi-gigabyte file. Here the headers are parsed once
/// at construction (9 shards, ~0.2 ms each) and every later lookup is pointer
/// arithmetic.
///
/// The header is parsed directly rather than through `SafeTensors` because a
/// `SafeTensors` borrows its mapping, which would make this struct
/// self-referential; extents are plain numbers and outlive nothing.
pub struct ExpertSource {
    maps: Vec<std::sync::Arc<Mmap>>,
    /// tensor name -> (map index, start, end) relative to the file
    extents: HashMap<String, (usize, u64, u64)>,
    shapes: HashMap<String, Vec<usize>>,
    dtypes: HashMap<String, String>,
}

impl ExpertSource {
    /// Map every shard named by the index and resolve all tensor extents.
    pub fn open(dir: &Path) -> Result<Self> {
        let idx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("model.safetensors.index.json")).context("reading index")?,
        )?;
        let wm = idx["weight_map"].as_object().context("index has no weight_map")?;
        let mut shard_names: Vec<String> =
            wm.values().filter_map(|v| v.as_str().map(str::to_string)).collect();
        shard_names.sort();
        shard_names.dedup();

        let mut maps = Vec::with_capacity(shard_names.len());
        let mut extents = HashMap::new();
        let mut shapes = HashMap::new();
        let mut dtypes = HashMap::new();

        for (mi, shard) in shard_names.iter().enumerate() {
            let file = std::fs::File::open(dir.join(shard))
                .with_context(|| format!("opening shard {shard}"))?;
            // SAFETY: the checkpoint is read-only and nothing else writes it.
            let map = unsafe { Mmap::map(&file) }?;
            let hlen = u64::from_le_bytes(map[0..8].try_into().unwrap());
            let header: serde_json::Value = serde_json::from_slice(&map[8..8 + hlen as usize])
                .with_context(|| format!("parsing header of {shard}"))?;
            let data_start = 8 + hlen;
            for (name, meta) in header.as_object().context("header is not an object")? {
                if name == "__metadata__" {
                    continue;
                }
                let Some(off) = meta.get("data_offsets").and_then(|v| v.as_array()) else {
                    continue;
                };
                let (s, e) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
                extents.insert(name.clone(), (mi, data_start + s, data_start + e));
                if let Some(sh) = meta.get("shape").and_then(|v| v.as_array()) {
                    shapes.insert(
                        name.clone(),
                        sh.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect(),
                    );
                }
                if let Some(dt) = meta.get("dtype").and_then(|v| v.as_str()) {
                    dtypes.insert(name.clone(), dt.to_string());
                }
            }
            maps.push(std::sync::Arc::new(map));
        }
        Ok(ExpertSource { maps, extents, shapes, dtypes })
    }

    /// The whole tensor, borrowed.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let &(mi, s, e) = self
            .extents
            .get(name)
            .with_context(|| format!("{name} is not in the checkpoint"))?;
        Ok(&self.maps[mi][s as usize..e as usize])
    }

    /// The mapping a tensor's bytes live in, for use as a zero-copy keepalive.
    pub fn keepalive(&self, name: &str) -> Result<std::sync::Arc<Mmap>> {
        let &(mi, _, _) = self
            .extents
            .get(name)
            .with_context(|| format!("{name} is not in the checkpoint"))?;
        Ok(self.maps[mi].clone())
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(self.shapes.get(name).with_context(|| format!("{name} has no shape"))?)
    }

    pub fn has(&self, name: &str) -> bool {
        self.extents.contains_key(name)
    }

    /// Is this expert stack packed NVFP4 (as opposed to the one BF16 layer)?
    pub fn is_nvfp4(&self, base: &str) -> bool {
        self.has(&format!("{base}.scale"))
            && self.dtypes.get(base).map(|d| d == "U8").unwrap_or(false)
    }

    /// Expert `e` of a `[experts, rows, cols]` packed stack, without copying.
    pub fn expert(&self, base: &str, e: usize) -> Result<PackedRef<'_>> {
        let shape = self.shape(base)?;
        if shape.len() != 3 {
            bail!("{base} is rank {}", shape.len());
        }
        let (experts, rows, cols) = (shape[0], shape[1], shape[2]);
        if e >= experts {
            bail!("expert {e} of {experts}");
        }
        let logical = cols * 2;
        if logical % GROUP != 0 {
            bail!("{logical} logical is not a multiple of {GROUP}");
        }
        let spr = logical / GROUP;

        let all = self.bytes(base)?;
        if all.len() != experts * rows * cols {
            bail!("{base} is {} bytes, want {}", all.len(), experts * rows * cols);
        }
        let codes = &all[e * rows * cols..(e + 1) * rows * cols];

        let sname = format!("{base}.scale");
        let sall = self.bytes(&sname)?;
        let s0 = e * rows * spr;
        if sall.len() < s0 + rows * spr {
            bail!("{sname} is short");
        }
        let scales = &sall[s0..s0 + rows * spr];

        let s2 = self.bytes(&format!("{base}.scale2"))?;
        if s2.len() < 4 * (e + 1) {
            bail!("{base}.scale2 is short");
        }
        let scale2 = f32::from_le_bytes(s2[4 * e..4 * e + 4].try_into().unwrap());

        Ok(PackedRef {
            codes,
            scales,
            scale2,
            rows,
            cols,
            codes_keep: self.keepalive(base)?,
            scales_keep: self.keepalive(&sname)?,
        })
    }
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

/// Expose `data` to the GPU **without copying it**, keeping `keep` alive for as
/// long as the returned handle (and anything derived from it) exists.
///
/// Returns `None` when the device cannot address host memory directly, or when
/// the region is not aligned well enough for the vectorised loads the expert
/// GEMM issues — callers fall back to a copy rather than get a wrong answer.
///
/// ## What keeps the mapping alive
///
/// `keep` is an `Arc<Mmap>` cloned from [`ExpertSource`], handed to cubecl as
/// the storage entry's keepalive. cubecl drops it only when it deallocates that
/// entry, which it cannot do while a handle referencing it is alive. So the
/// chain is: kernel -> binding -> Handle -> storage entry -> Arc<Mmap> -> pages.
/// Nothing in that chain is a bare pointer with an implicit lifetime, which
/// matters because a kernel reading unmapped pages does NOT reliably fault —
/// it reads whatever the address holds now, and that is a wrong answer rather
/// than a crash.
pub fn alias_bytes<R: Runtime>(
    client: &ComputeClient<R>,
    data: &[u8],
    keep: std::sync::Arc<Mmap>,
) -> Option<Handle> {
    if !cubecl::cuda::supports_zero_copy_host(0) {
        return None;
    }
    // The expert GEMM binds these bytes as `Vector<e2m1x2, 4>` -- four 1-byte
    // e2m1x2 elements, i.e. a 4-byte load -- so 4-byte alignment is the real
    // requirement and 16 would be a superstition. It matters that this is the
    // true bound and not a guess: safetensors packs tensors back to back with
    // no padding, and this checkpoint puts w13_weight and w2_weight at offsets
    // congruent to 4 (mod 16), so a 16-byte test would refuse every weight
    // slab in the model and fall back to copying forever while looking like it
    // worked. `inkling_zerocopy_gate` checks the aliased result byte-for-byte
    // against the copied one, which is what actually settles it.
    if (data.as_ptr() as usize) % 4 != 0 || data.is_empty() {
        return None;
    }
    let len = data.len() as u64;
    // SAFETY: `data` borrows pages of `keep`'s mapping, which is read-only
    // (`PROT_READ`, `MAP_PRIVATE`) and lives at least as long as the `Arc` we
    // hand over. cubecl pins external handles immutable, so no kernel will try
    // to write them.
    Some(unsafe {
        client.register_external_aliased(
            data.as_ptr() as *mut core::ffi::c_void,
            len,
            0,
            len,
            keep as std::sync::Arc<dyn core::any::Any + Send + Sync>,
        )
    })
}

/// [`alias_bytes`], falling back to an ordinary copy when aliasing is refused.
pub fn alias_or_copy<R: Runtime>(
    client: &ComputeClient<R>,
    data: &[u8],
    keep: std::sync::Arc<Mmap>,
) -> Handle {
    match alias_bytes(client, data, keep) {
        Some(h) => h,
        None => client.create_from_slice(data),
    }
}

/// Every shard mapping, registered with the GPU **once**.
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
/// Registering is per *mapping*, though, not per tensor: nine shards, nine
/// round trips for the whole run. Everything after that is
/// [`Handle::offset_start`], which is pure arithmetic on the client side.
pub struct AliasedShards {
    shards: Vec<Handle>,
    lens: Vec<u64>,
}

impl ExpertSource {
    /// Register all mapped shards for zero-copy access. `None` if the device
    /// cannot address host memory directly.
    pub fn alias_shards<R: Runtime>(&self, client: &ComputeClient<R>) -> Option<AliasedShards> {
        if !cubecl::cuda::supports_zero_copy_host(0) {
            return None;
        }
        let mut shards = Vec::with_capacity(self.maps.len());
        let mut lens = Vec::with_capacity(self.maps.len());
        for m in &self.maps {
            let len = m.len() as u64;
            let keep: std::sync::Arc<dyn core::any::Any + Send + Sync> = m.clone();
            // SAFETY: the mapping is read-only and `keep` holds it for as long
            // as the handle lives; cubecl pins external handles immutable.
            let h = unsafe {
                client.register_external_aliased(
                    m.as_ptr() as *mut core::ffi::c_void,
                    len,
                    0,
                    len,
                    keep,
                )
            };
            shards.push(h);
            lens.push(len);
        }
        Some(AliasedShards { shards, lens })
    }

    /// An expert's packed codes and scales as zero-copy offset views.
    ///
    /// Returns `None` when any slab is not 4-byte aligned — the expert GEMM
    /// issues 4-byte vector loads, so that is the real bound (this checkpoint's
    /// weight tensors sit at offsets congruent to 4 mod 16, which is fine for a
    /// 4-byte load and would fail a naive 16-byte test).
    pub fn expert_aliased(
        &self,
        al: &AliasedShards,
        base: &str,
        e: usize,
    ) -> Result<Option<(Handle, Handle)>> {
        let shape = self.shape(base)?;
        let (rows, cols) = (shape[1], shape[2]);
        let spr = cols * 2 / GROUP;
        let sname = format!("{base}.scale");

        let &(mi_c, cs, _) = self.extents.get(base).context("codes extent")?;
        let &(mi_s, ss, _) = self.extents.get(&sname).context("scales extent")?;

        let c_off = cs + (e * rows * cols) as u64;
        let c_len = (rows * cols) as u64;
        let s_off = ss + (e * rows * spr) as u64;
        let s_len = (rows * spr) as u64;

        let c_ptr = self.maps[mi_c].as_ptr() as usize + c_off as usize;
        let s_ptr = self.maps[mi_s].as_ptr() as usize + s_off as usize;
        if c_ptr % 4 != 0 || s_ptr % 4 != 0 {
            return Ok(None);
        }

        let ch = al.shards[mi_c]
            .clone()
            .offset_start(c_off)
            .offset_end(al.lens[mi_c] - c_off - c_len);
        let sh = al.shards[mi_s]
            .clone()
            .offset_start(s_off)
            .offset_end(al.lens[mi_s] - s_off - s_len);
        Ok(Some((ch, sh)))
    }
}
