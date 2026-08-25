//! W4A16: an NVFP4 **weight** against a BF16 **activation**.
//!
//! [`super::fp4gemm::fp4_linear`] runs the hardware block-scaled MMA
//! (`kind::mxf4nvf4`), and that instruction takes E2M1 for BOTH operands — so
//! it quantises the activation too. For the ROUTED experts that is not a
//! liberty: the checkpoint's `hf_quant_config.json` sets `"*input_quantizer"
//! … enable: true`, i.e. the publisher calibrated for a 4-bit activation and
//! W4A4 is what they meant.
//!
//! For everything else in the model they did not. The sink experts, the
//! attention projections and the unembedding carry no `.scale` sidecar and no
//! `input_amax`; the publisher left them BF16 and never calibrated an
//! activation quantiser for them. Quantising their activations is a numerics
//! decision nobody made.
//!
//! But the *reason* to want four bits there is untouched by that. Decode is
//! bound on the DRAM read, not on the MMA: `lm_head` alone is
//! `[201024, 4096]` BF16 = 1.65 GB, read once per step and once per draft
//! depth. The bandwidth win of FP4 lives entirely in that read. So this lane
//! takes the half of NVFP4 that is free — the stored bytes — and declines the
//! half that costs calibration.
//!
//! ## Where the dequantisation happens
//!
//! In registers, during the B-fragment load, and nowhere else. Decoding the
//! weight into a scratch buffer first and multiplying that would spend exactly
//! the bandwidth this exists to save (and then some: the scratch is written as
//! well as read). Each lane reads the packed `u32` word holding its own two
//! codes plus the one E4M3 scale covering them, expands them to BF16 in two
//! registers, and hands those straight to
//! [`cmma::MmaDefinition::execute`]. The weight is never wider than four bits
//! anywhere outside a register file.
//!
//! Structurally this is [`super::bf16gemm::bf16_linear`]: same `m16n8k16`
//! instruction, same one-plane-per-`(m_tile, n_tile)` grid, same
//! `position_of_nth` fragment addressing, same f32 accumulator. The only edit
//! is what sits between the global load and `reg_b`.
//!
//! ## Two facts that make the pair-at-a-time decode exact
//!
//! * `vector_size(B)` is 2 for BF16 and `position_of_nth` returns an EVEN `k`
//!   for the first of each pair, so a lane's two elements are `k` and `k + 1`
//!   with `k` even — one packed byte, low nibble first ([`super::nvfp4`]
//!   settled that ordering against `compressed_tensors`).
//! * A 16-element scale block starts at a multiple of 16, so an even-aligned
//!   pair never straddles one. Both elements share a scale.
//!
//! The loads below are written per-element anyway, indexing `(gc + j)` rather
//! than assuming `gc`. On the layout above both `j` resolve to the same word
//! and the same scale byte, so it is one DRAM line either way and the kernel
//! does not silently depend on a fragment layout it cannot check.

use cubecl::e4m3;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::bf16;

/// Rows of one MMA tile — the M granularity everything here is padded to.
pub const MTILE: usize = 16;
/// Columns of one MMA tile.
pub const NTILE: usize = 8;
/// K covered by one `m16n8k16` instruction — the BF16 operand's k, not the
/// 4-bit operand's. The weight being narrow does not widen the instruction.
pub const KTILE: usize = 16;
/// Logical elements per E4M3 block scale (NVFP4's `group_size`).
pub const GROUP: usize = 16;
/// E2M1 codes packed into one `u32` word (eight nibbles).
pub const CODES_PER_WORD: usize = 8;

/// The value of one E2M1 code, as [`super::nvfp4::FP4_E2M1`] tabulates it.
///
/// Written as seven comparisons rather than an array index because a
/// runtime-indexed local array spills to local memory, and rather than the
/// exponent/mantissa arithmetic because THIS is the host table transcribed —
/// the values on the right are `FP4_E2M1[1..8]` in order, so the two can be
/// diffed by eye. The subnormal (`0x1` = 0.5) is the entry the arithmetic form
/// gets wrong, and it is the one that is hardest to notice.
#[cube]
fn e2m1_value(code: u32) -> f32 {
    let mag = code & 7u32;
    let mut v = f32::new(0.0f32);
    if mag == 1u32 {
        v = f32::new(0.5f32);
    }
    if mag == 2u32 {
        v = f32::new(1.0f32);
    }
    if mag == 3u32 {
        v = f32::new(1.5f32);
    }
    if mag == 4u32 {
        v = f32::new(2.0f32);
    }
    if mag == 5u32 {
        v = f32::new(3.0f32);
    }
    if mag == 6u32 {
        v = f32::new(4.0f32);
    }
    if mag == 7u32 {
        v = f32::new(6.0f32);
    }
    // Bit 3 is the sign. `-0.0` for code `0x8` is what the table says and what
    // the packer emits for a negative value that rounds to zero magnitude.
    if code >= 8u32 {
        v = -v;
    }
    v
}

/// `out = (a @ b^T) * scale`, with `a` BF16 and `b` NVFP4.
///
/// `a` is `[m_pad, k]` BF16; `b` is `[n, k/8]` `u32` (element `i` of word `w`
/// at bits `4*(i%8)`, low nibble = lowest index) and `b_sc` is `[n, k/16]`
/// E4M3 — the layout [`super::fp4quant`] writes and the checkpoint stores.
/// `out` is `[m_pad, n]` f32, the accumulator's own type.
///
/// `scale` is the tensor-wide `scale2`; it is applied once to the accumulator
/// rather than per element, which is where [`super::fp4gemm::fp4_linear`] puts
/// it too. That is a deliberate deviation from
/// [`super::nvfp4::decode_row`]'s block-scale-then-`scale2` order — folding it
/// into the accumulator is not associativity-equivalent — and it is the same
/// deviation the NVFP4 lane already makes, so the two lanes agree with each
/// other.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear<AB: Scalar + Cast, S: Scalar, NA: Size, NC: Size>(
    a: &Tensor<Vector<AB, NA>>,
    b: &Tensor<u32>,
    b_sc: &Tensor<S>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    // 1 for BF16. Kept in the A index arithmetic so this reads as the same
    // kernel as its two parents rather than a third dialect of it.
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

    // Packed words and scale bytes per weight row.
    let wpr = comptime!(size_k / CODES_PER_WORD);
    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE);

    for t in 0..k_tiles {
        let kbase = t * KTILE;
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
            let gr = row as usize + m_base;
            let gc = col as usize + kbase;
            reg_a[i] = a[(gr * size_k + gc) / a.vector_size()];
        }
        #[unroll]
        for i in 0..vc_b {
            // B is column-major w.r.t. the tile: `col` indexes n, `row`
            // indexes k, and the checkpoint's `[out, in]` rows are exactly
            // that. Same addressing as the BF16 lane; only the fetch differs.
            let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
            let gr = col as usize + n_base;
            let gc = row as usize + kbase;

            // ONE word and ONE scale for the whole fragment element group,
            // not one of each per element. The two facts in the module header
            // say they are the same word and the same scale byte -- `gc` is
            // even and a 16-element scale block starts at a multiple of 16 --
            // so the per-element form was issuing four memory instructions
            // where the BF16 lane issues one, for a quarter of the bytes. That
            // is why the four-bit head measured 69 GB/s against the BF16
            // head's 163: at 2 KB a row the lane is instruction-bound in this
            // load, not bandwidth-bound.
            //
            // This DOES now depend on the fragment layout the header describes.
            // The detector is `linear_w4a16_tracks_linear_bf16_on_the_same_weight`:
            // a wrong nibble or a wrong scale is an order-one error and blows
            // straight past that test's bound, which is what it is for.
            let word = b[gr * wpr + gc / CODES_PER_WORD];
            let s = f32::cast_from(b_sc[gr * spr + gc / GROUP]);
            let mut v = Vector::<AB, NA>::empty();
            #[unroll]
            for j in 0..vs_b {
                let kk = gc + j;
                let code = (word >> (4 * (kk % CODES_PER_WORD)) as u32) & 15u32;
                // The one widening in the lane, and it stops at the register.
                v[j] = AB::cast_from(e2m1_value(code) * s);
            }
            reg_b[i] = v;
        }

        let d = def.execute(&reg_a, &reg_b, &acc);
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
        out[(gr * size_n + gc) / out.vector_size()] = acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// Launch [`w4a16_linear`] for a `[m_pad, k] x [n, k]^T` product.
///
/// Mirrors [`super::fp4gemm::fp4_linear_launch`] minus the activation's codes
/// and scales: `a` is a BF16 handle, not a quantised pair.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Handle {
    assert_eq!(
        m_pad % MTILE,
        0,
        "m_pad {m_pad} is not a multiple of {MTILE}"
    );
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    // Two BF16 per `.b32`, which is what `contiguous_elements` reports for A
    // and B and what the fragment layout actually is.
    let vs = 32 / bf16::cube_type().size_bits();
    let wpr = k / CODES_PER_WORD;
    let spr = k / GROUP;

    unsafe {
        w4a16_linear::launch::<bf16, e4m3, R>(
            client,
            CubeCount::Static((n / NTILE) as u32, (m_pad / MTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            TensorArg::from_raw_parts(a.clone(), [k, 1].into(), [m_pad, k].into()),
            TensorArg::from_raw_parts(b.clone(), [wpr, 1].into(), [n, wpr].into()),
            TensorArg::from_raw_parts(b_sc.clone(), [spr, 1].into(), [n, spr].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_pad, n].into()),
            k,
            n,
            scale,
        )
    };
    out
}

// ---------------------------------------------------------------------------
// The same product, with the three things the SASS said were wrong.
//
// `w4a16_linear` above is one 32-thread cube per 16x8 output tile. On GB10
// `cudaDevAttrMaxBlocksPerMultiprocessor` is 24 and a warp slot count of 48, so
// a one-warp cube caps residency at 24 warps of 48 -- 50% occupancy that no
// register budget can recover (it uses 40 registers; 51 blocks' worth would
// fit). And its K loop is not unrolled, so a warp has exactly one k-tile of
// weight fetch outstanding at a time, behind which sit 154 SASS instructions of
// address arithmetic, bounds clamping and nibble expansion.
//
// The evidence that the 154 is NOT the binding constraint is `fp4gemm`'s
// `fp4_linear`: it runs the hardware block-scaled MMA, needs no software
// dequantisation at all, and executes 12 416 instructions per warp over the
// head's K against this kernel's 39 424 -- 3.2x fewer -- for 4.31 ms against
// 4.53. Instruction count moved by 3.2x and time moved by 5%. So the fix is
// not fewer instructions; it is more warps and more fetches in flight.
//
// Three changes, each independently checkable:
//
// * **Planes per cube.** `CubeDim::new_2d(32, PLANES)` with a plane per n-tile.
//   At `PLANES = 4` the block limit stops binding and the register budget takes
//   over: 40 x 128 = 5120 registers a cube, 12 cubes an SM, 48 warps -- full
//   residency.
// * **Two k-tiles per iteration, one 16-byte load.** A lane's two B words for
//   k-tile `t` are at packed indices `kbase/8` and `kbase/8 + 1`, and the next
//   k-tile's are `+2` and `+3`. When `kbase` is a multiple of 32 those four
//   words are 16 aligned contiguous bytes, so ONE `LDG.E.128` replaces four
//   `LDG.E.32` and the second k-tile's fetch is already in a register when the
//   first MMA issues.
// * **One scale byte per k-tile, not two.** Both fragment halves of a k-tile
//   resolve to the same `gc / GROUP`, which the original loads twice.
//
// A is left alone deliberately: at `m_pad = 16` it is 128 KiB, L2-resident, and
// re-read by every cube. It is 4 of the 8 loads and the reason to fix it is
// tile reuse, not width -- a separate change with a separate measurement.

/// Planes per cube. 4 x 32 = 128 threads, which at 40 registers a thread is 12
/// cubes and all 48 warp slots on an SM.
pub const PLANES: u32 = 4;
/// K covered by one loop iteration: two `m16n8k16` steps, one 16-byte B load.
pub const KSTEP: usize = 2 * KTILE;
/// Packed `u32` words one lane fetches per iteration.
pub const WORDS_PER_STEP: usize = 4;

/// `out = (a @ b^T) * scale`, `a` BF16 and `b` NVFP4 — same contract as
/// [`w4a16_linear`], same numerics, different residency.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_wide<AB: Scalar + Cast, S: Scalar, NA: Size, NC: Size, NB: Size>(
    a: &Tensor<Vector<AB, NA>>,
    b: &Tensor<Vector<u32, NB>>,
    b_sc: &Tensor<S>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    // One plane per n-tile; the cube covers `PLANES` of them.
    let n_tile = CUBE_POS_X as usize * comptime!(PLANES as usize) + UNIT_POS_Y as usize;
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

    // `b` is indexed in 16-byte vectors, so a row is `size_k / 8 / 4` of them.
    let vpr = comptime!(size_k / CODES_PER_WORD / WORDS_PER_STEP);
    let spr = comptime!(size_k / GROUP);
    let steps = comptime!(size_k / KSTEP);

    // The lane's own k offset inside a tile, hoisted: it does not depend on the
    // step, and the original recomputed it from `UNIT_POS_PLANE` six times an
    // iteration.
    let (b_row0, b_col0) = def.position_of_nth(lane, 0u32, MatrixIdent::B);
    let gr = b_col0 as usize + n_base;
    let klane = b_row0 as usize;

    for s in 0..steps {
        let kbase = s * KSTEP;
        // ONE 16-byte load: the four packed words covering both k-tiles.
        let quad = b[gr * vpr + kbase / (CODES_PER_WORD * WORDS_PER_STEP)];

        #[unroll]
        for half in 0..2usize {
            let khalf = kbase + half * KTILE;
            #[unroll]
            for i in 0..vc_a {
                let (row, col) =
                    def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
                let ga = row as usize + m_base;
                let gc = col as usize + khalf;
                reg_a[i] = a[(ga * size_k + gc) / a.vector_size()];
            }
            // Both fragment halves of a k-tile share this scale byte, because
            // `klane` is at most 6 and a 16-element block starts at a multiple
            // of 16. The original read it twice.
            let sc = f32::cast_from(b_sc[gr * spr + khalf / GROUP]);
            #[unroll]
            for i in 0..vc_b {
                // Fragment element `i` sits `i * 8` further along k, which is
                // exactly one packed word: word `2 * half + i` of the quad.
                let word = quad[2 * half + i];
                let mut v = Vector::<AB, NA>::empty();
                #[unroll]
                for j in 0..vs_b {
                    let code = (word >> (4 * (klane + j)) as u32) & 15u32;
                    v[j] = AB::cast_from(e2m1_value(code) * sc);
                }
                reg_b[i] = v;
            }
            let d = def.execute(&reg_a, &reg_b, &acc);
            #[unroll]
            for i in 0..vc_c {
                acc[i] = d[i];
            }
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let go = row as usize + m_base;
        let gc = col as usize + n_base;
        out[(go * size_n + gc) / out.vector_size()] = acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

/// Launch [`w4a16_linear_wide`]; same signature as [`w4a16_linear_launch`].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_wide_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Handle {
    assert_eq!(
        m_pad % MTILE,
        0,
        "m_pad {m_pad} is not a multiple of {MTILE}"
    );
    let ntiles = n / NTILE;
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(
        ntiles % PLANES as usize,
        0,
        "{ntiles} n-tiles do not divide into cubes of {PLANES} planes"
    );
    assert_eq!(k % KSTEP, 0, "k {k} is not a multiple of {KSTEP}");

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let vs = 32 / bf16::cube_type().size_bits();
    let vpr = k / CODES_PER_WORD / WORDS_PER_STEP;
    let spr = k / GROUP;

    unsafe {
        w4a16_linear_wide::launch::<bf16, e4m3, R>(
            client,
            CubeCount::Static((ntiles / PLANES as usize) as u32, (m_pad / MTILE) as u32, 1),
            CubeDim::new_2d(32, PLANES),
            vs,
            2,
            WORDS_PER_STEP,
            TensorArg::from_raw_parts(a.clone(), [k, 1].into(), [m_pad, k].into()),
            TensorArg::from_raw_parts(b.clone(), [vpr, 1].into(), [n, vpr].into()),
            TensorArg::from_raw_parts(b_sc.clone(), [spr, 1].into(), [n, spr].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_pad, n].into()),
            k,
            n,
            scale,
        )
    };
    out
}
