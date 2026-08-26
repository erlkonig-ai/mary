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

    // M in x, N in y. Grid x varies fastest, so the cubes that share a weight
    // row — the whole n-tile's `[8, k]` of codes and scales — are the ones
    // launched adjacently, and one DRAM read of that row serves all
    // `m_pad / 16` of them out of L2. The other order puts consumers of the
    // same row `n / 8` cubes apart, which at the head shape is the whole
    // weight table between them, so every m-tile re-reads from DRAM.
    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize;
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
    // N rides grid y, which CUDA caps at 65535 (x is 2^31-1). The largest N in
    // the model is the unembedding's 201024 = 25128 tiles, well inside it, but
    // the cap is silent if it is ever exceeded so it is checked here.
    assert!(
        n / NTILE <= 65535,
        "{} n-tiles exceed the 65535 grid-y limit",
        n / NTILE
    );

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    // Two BF16 per `.b32`, which is what `contiguous_elements` reports for A
    // and B and what the fragment layout actually is.
    let vs = 32 / bf16::cube_type().size_bits();
    let wpr = k / CODES_PER_WORD;
    let spr = k / GROUP;

    unsafe {
        w4a16_linear::launch::<bf16, e4m3, R>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE) as u32, 1),
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
// The same product with more warps and wider loads — kept because it MEASURED
// SLOWER, which is the useful part.
//
// The starting suspicion was residency and memory-level parallelism.
// `w4a16_linear` above is one 32-thread cube per 16x8 output tile; on GB10
// `cudaDevAttrMaxBlocksPerMultiprocessor` is 24 against 48 warp slots, so a
// one-warp cube caps residency at 50% and no register budget can recover it (it
// uses 40 registers; 51 blocks' worth would fit). Its K loop is not unrolled
// either, so a warp has one k-tile of weight fetch outstanding at a time behind
// 154 SASS instructions of address arithmetic, bounds clamping and nibble
// expansion.
//
// This variant changes three things: `PLANES` planes per cube instead of one, a
// single 16-byte `LDG.E.128` covering both k-tiles of a step (a lane's two B
// words for k-tile `t` are packed indices `kbase/8` and `kbase/8 + 1`, and the
// next k-tile's are `+2` and `+3`, which at a 32-aligned `kbase` are four
// contiguous aligned words), and one scale byte per k-tile where the original
// loads the same byte twice. It is 269 SASS instructions per two k-tiles
// against 154 per one — 13% fewer per k-tile — at the same 40 registers.
//
// Measured at the head's own shape (m_pad 16, k 4096, n 201024, 0.431 GiB of
// codes + scales, min of four warm launches, launch + sync, GB10, GPU
// otherwise idle), against `w4a16_linear` in the same process:
//
// ```text
//   PLANES   wide         original    verdict
//   1        4.95 ms      4.97 ms     wider loads and 13% fewer instructions: nothing
//   4        5.27-5.35    4.66-4.74   more warps: 13% WORSE
//   8        5.32-6.76    4.81-4.91   more warps still: 10-38% WORSE
// ```
//
// So this lane is not instruction-bound and not occupancy-bound. The
// corroborating measurement is `fp4gemm::fp4_linear`, which runs the hardware
// block-scaled MMA and needs no software dequantisation at all: 12 416
// instructions per warp over the head's K against this kernel's 39 424, 3.2x
// fewer, for 4.53 ms against 4.66. Instruction count moved 3.2x and time moved
// 3%.
//
// What is left is the access pattern itself. Every warp owns eight weight rows
// and consumes 32 bytes from each before advancing, so the memory system sees
// 32-byte reads 2048 bytes apart, thousands of warps deep — and adding warps
// adds streams, which is why more planes made it worse. A fully coalesced read
// of the SAME 0.431 GiB in the same process runs at 158-172 GB/s against these
// kernels' 98-106. The gap is coalescing, and coalescing needs a cooperative
// stage through shared memory, which is exactly what one warp per output tile
// forecloses. `fp4gemm`'s module header says "a fancier tiling would not change
// that"; at the ROUTED-EXPERT shape it measures right (`fp4_linear_grouped`
// reaches 171 GB/s, at the ceiling), and at THIS shape it does not.
//
// A is left alone throughout: at `m_pad = 16` it is 128 KiB, L2-resident, and
// re-read by every cube.
//
// One thing above IS occupancy-shaped after all, and it is not in the kernel:
// the GRID ORDER. See `fp4gemm`'s module header for the full sweep. The short
// version, same harness and same framing (head shape `k = 4096`, `n = 201024`,
// min of four warm launches, launch + sync, GB10, GPU idle, kernel time):
//
// ```text
//   m_pad   N in x     M in x     ratio        wide, N in x -> M in x
//      16   4.98 ms    4.88 ms    1.02         4.83 -> 4.87   (unchanged)
//      32   9.87       6.99       1.41         9.83 -> 5.90
//      64  18.92      16.43       1.15        19.27 -> 14.38
//     128  38.59      33.33       1.16        40.96 -> 29.82
// ```
//
// Both lanes gain, and BOTH gain less than `fp4_linear` does at the same shapes
// (1.97-2.41x). The obvious difference is A: this lane's activation is BF16, so
// its A traffic is four times the NVFP4 lane's over the same `m_pad`, and A is
// the operand the other launch order was serving out of L2. Obvious is not
// measured, though — that is a hypothesis, and this timer cannot separate it
// from the instruction count the block above already showed this lane is
// carrying.

/// Planes per cube. At 4 the block limit stops binding and the register budget
/// takes over — 40 x 128 = 5120 registers a cube, 12 cubes an SM, all 48 warp
/// slots — and the lane gets 13% SLOWER for it. Left at 1 because that is what
/// measured best; the constant stays so the experiment can be re-run by
/// changing one character.
pub const PLANES: u32 = 1;
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

    // One plane per n-tile; the cube covers `PLANES` of them. M in x for the
    // same reason as `w4a16_linear` above: weight-row sharers launch adjacent.
    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize * comptime!(PLANES as usize) + UNIT_POS_Y as usize;
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
    assert!(
        ntiles / PLANES as usize <= 65535,
        "{} n-cubes exceed the 65535 grid-y limit",
        ntiles / PLANES as usize
    );

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let vs = 32 / bf16::cube_type().size_bits();
    let vpr = k / CODES_PER_WORD / WORDS_PER_STEP;
    let spr = k / GROUP;

    unsafe {
        w4a16_linear_wide::launch::<bf16, e4m3, R>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (ntiles / PLANES as usize) as u32, 1),
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

// ---------------------------------------------------------------------------
// Pre-permuted ("swizzled") B layout for m16n8k16
// ---------------------------------------------------------------------------

/// Bytes one `(n_tile, k_tile)` block of codes occupies: `NTILE` rows x
/// `KTILE / 2` packed bytes.
pub const SWZ_BLOCK_CODES: usize = NTILE * KTILE / 2;
/// Bytes one `(n_tile, k_tile)` block of scales occupies: one E4M3 per row.
pub const SWZ_BLOCK_SCALES: usize = NTILE;

/// Where lane `l`'s `i`-th B load lands in a swizzled block.
///
/// `mma16_frag_map` dumps the map off the device and it holds the closed form
///
/// ```text
///   col = lane >> 2                  the n column, 0..8
///   row = 2 * (lane & 3) + 8 * i     the k element, 0..16
/// ```
///
/// exactly, on sm_121a, for all 32 lanes. So a lane's `i`-th load wants the
/// packed word covering k elements `[8i, 8i+8)` of weight row `col`, and the
/// warp's `i`-th load wants eight such words — one per row.
///
/// Row-major that is eight addresses `k/2` bytes apart: eight 32-byte sector
/// requests for 32 useful bytes. Written down as
///
/// ```text
///   dst_word(col, w) = w * NTILE + col        w = row / 8, in 0..2
/// ```
///
/// load `i` is words `[8i, 8i+8)` of the block, i.e. **32 contiguous bytes** —
/// one sector where the row-major form takes eight.
///
/// ## What this does and does not buy
///
/// It does not save DRAM bytes. Over a whole k loop the row-major form already
/// reaches 100% sector and 100% line utilisation, because the k loop walks each
/// of the eight rows FORWARD and a half-used sector is finished by the next few
/// k tiles out of L1 — `mma16_lane_dump` prints both counts side by side. What
/// it saves is REQUESTS: 4096 sector requests per warp k loop against 512
/// distinct sectors, an 8x amplification that the permutation removes.
#[inline]
fn swz_word_k16(col: usize, w: usize) -> usize {
    w * NTILE + col
}

/// Permute `[n, k/8]` packed E2M1 codes into `m16n8k16` B-fragment order.
///
/// Block `(n_tile, k_tile)` occupies [`SWZ_BLOCK_CODES`] consecutive bytes at
/// `((n_tile * k/KTILE) + k_tile) * SWZ_BLOCK_CODES`. Same length, same bytes,
/// different order. `src` and `dst` must not overlap.
pub fn swizzle_w4a16_codes_into(src: &[u8], dst: &mut [u8], n: usize, k: usize) {
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");
    assert_eq!(src.len(), n * k / 2, "codes are not [n, k/2] bytes");
    assert_eq!(
        dst.len(),
        src.len(),
        "destination is not the source's length"
    );
    let kt = k / KTILE;
    let row_w = k / CODES_PER_WORD;
    for nt in 0..n / NTILE {
        for t in 0..kt {
            let blk = (nt * kt + t) * SWZ_BLOCK_CODES;
            for col in 0..NTILE {
                for w in 0..KTILE / CODES_PER_WORD {
                    let s = ((nt * NTILE + col) * row_w + t * (KTILE / CODES_PER_WORD) + w) * 4;
                    let d = blk + swz_word_k16(col, w) * 4;
                    dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
                }
            }
        }
    }
}

/// [`swizzle_w4a16_codes_into`] allocating its own destination.
pub fn swizzle_w4a16_codes(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    swizzle_w4a16_codes_into(src, &mut dst, n, k);
    dst
}

/// Permute `[n, k/16]` E4M3 block scales to match [`swizzle_w4a16_codes_into`].
///
/// One `m16n8k16` covers exactly [`GROUP`] k elements, so a fragment consumes
/// ONE scale byte per weight row: eight bytes, row-major eight separate
/// sectors. Blocked as `[n_tile][k_tile][8]` they are eight contiguous bytes.
pub fn swizzle_w4a16_scales_into(src: &[u8], dst: &mut [u8], n: usize, k: usize) {
    assert_eq!(n % NTILE, 0);
    assert_eq!(k % KTILE, 0);
    assert_eq!(src.len(), n * (k / GROUP), "scales are not [n, k/16]");
    assert_eq!(
        dst.len(),
        src.len(),
        "destination is not the source's length"
    );
    let kt = k / KTILE;
    let spr = k / GROUP;
    for nt in 0..n / NTILE {
        for t in 0..kt {
            let blk = (nt * kt + t) * SWZ_BLOCK_SCALES;
            for col in 0..NTILE {
                dst[blk + col] = src[(nt * NTILE + col) * spr + t];
            }
        }
    }
}

/// [`swizzle_w4a16_scales_into`] allocating its own destination.
pub fn swizzle_w4a16_scales(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    swizzle_w4a16_scales_into(src, &mut dst, n, k);
    dst
}

/// Whether a `[n, k]` weight can be written in the swizzled layout at all.
pub fn swizzleable(n: usize, k: usize) -> bool {
    n % NTILE == 0 && k % KTILE == 0
}

/// Permute a `[n, k/8]` packed-code plane on the DEVICE, into a fresh handle.
///
/// The head's codes are produced on the device by
/// [`super::fp4quant::quantize_nvfp4_bf16`], not copied out of the pile, so
/// there is no host memcpy for the permutation to ride inside the way the
/// routed experts' has. A device pass is the cheap alternative: one linear
/// write of the destination with a gathered read, 0.43 GiB at the head's shape,
/// once per process at startup.
///
/// Written destination-linear on purpose. The gather side is the scattered one,
/// and a scattered READ of a resident table is what this whole permutation
/// exists to make the GEMM stop doing per step — paying it once is the trade.
#[cube(launch)]
pub fn swizzle_codes_dev(src: &Tensor<u32>, dst: &mut Tensor<u32>, #[comptime] k_tiles: usize) {
    let d = ABSOLUTE_POS as usize;
    if d < dst.len() {
        let wpb = comptime!(SWZ_BLOCK_CODES / 4);
        let per_row = comptime!(KTILE / CODES_PER_WORD);
        let blk = d / wpb;
        let j = d % wpb;
        let w = j / NTILE;
        let col = j % NTILE;
        let nt = blk / k_tiles;
        let t = blk % k_tiles;
        dst[d] = src[(nt * NTILE + col) * (k_tiles * per_row) + t * per_row + w];
    }
}

/// Permute a `[n, k/16]` E4M3 scale plane on the DEVICE. See
/// [`swizzle_codes_dev`].
#[cube(launch)]
pub fn swizzle_scales_dev<S: Scalar>(
    src: &Tensor<S>,
    dst: &mut Tensor<S>,
    #[comptime] k_tiles: usize,
) {
    let d = ABSOLUTE_POS as usize;
    if d < dst.len() {
        let blk = d / NTILE;
        let col = d % NTILE;
        let nt = blk / k_tiles;
        let t = blk % k_tiles;
        dst[d] = src[(nt * NTILE + col) * k_tiles + t];
    }
}

/// Permute both planes of an already-quantised `[n, k]` weight on the device.
///
/// Returns fresh handles; the caller drops the row-major ones. Panics rather
/// than silently declining on a shape the layout cannot express — a weight that
/// half-permuted would be a kernel reading the wrong bytes and producing
/// NUMBERS, which is the one failure mode this must not have.
pub fn swizzle_w4a16_device<R: Runtime>(
    client: &ComputeClient<R>,
    codes: &Handle,
    scales: &Handle,
    n: usize,
    k: usize,
) -> (Handle, Handle) {
    assert!(swizzleable(n, k), "[{n}, {k}] is not swizzleable");
    let k_tiles = k / KTILE;
    let words = n * k / CODES_PER_WORD;
    let sc = n * (k / GROUP);
    let dc = client.empty(words * 4);
    let ds = client.empty(sc);
    let threads = 256u32;
    unsafe {
        swizzle_codes_dev::launch::<R>(
            client,
            CubeCount::Static(words.div_ceil(threads as usize) as u32, 1, 1),
            CubeDim::new_1d(threads),
            TensorArg::from_raw_parts(codes.clone(), [1].into(), [words].into()),
            TensorArg::from_raw_parts(dc.clone(), [1].into(), [words].into()),
            k_tiles,
        )
    };
    unsafe {
        swizzle_scales_dev::launch::<e4m3, R>(
            client,
            CubeCount::Static(sc.div_ceil(threads as usize) as u32, 1, 1),
            CubeDim::new_1d(threads),
            TensorArg::from_raw_parts(scales.clone(), [1].into(), [sc].into()),
            TensorArg::from_raw_parts(ds.clone(), [1].into(), [sc].into()),
            k_tiles,
        )
    };
    (dc, ds)
}

/// Whether the head/sink W4A16 weights are written in MMA-fragment order.
///
/// On by default. `INK_W4A16_SWZ=0` is the A/B arm.
///
/// ## What it measured
///
/// **The kernel, alone.** `w4a16_swz_probe`, one process, four interleaved
/// arms, first two reps discarded, per LAUNCH of one `[16, 4096] x
/// [201024, 4096]^T` product (the unembedding's own shape at decode, `m = 1`
/// padded to one m-tile), one GB10 box, GB/s over the weight planes only
/// (0.431 GiB) and not over a step:
///
/// | arm              | p50      | GB/s  |
/// |------------------|---------:|------:|
/// | row-major        | 4.832 ms |  95.9 |
/// | both planes swz  | 3.984 ms | 116.3 |
/// | codes only swz   | 4.069 ms | 113.8 |
/// | coalesced ceiling| 2.172 ms | 213.2 |
///
/// 12 warm reps of 14. The two swizzled arms are indistinguishable within
/// their spread, so this does not establish that the scale plane's permutation
/// is worth anything on its own; both are permuted because keeping one plane
/// row-major would be a second layout for no measured gain.
///
/// **End to end.** `bench-decode.sh -n 4 --gen 12 --layers 21:42`, `INK_KV=1`,
/// a 3720-token prompt, ctx 3732, ONE GB10 box holding layers 21..42 and the
/// head (not the two-node pipe), arms interleaved, per DECODE STEP, p50 of 11
/// warm passes a rep:
///
/// | arm       | reps (p50 each)        | p50      |
/// |-----------|------------------------|---------:|
/// | swz       | 55.9, 56.1, 56.1 ms    | 56.1 ms  |
/// | row-major | 57.1, 57.2, 57.3 ms    | 57.2 ms  |
///
/// 1.1 ms a step, 1.9%, and the two arms' rep bands do not overlap. Eight reps
/// were run and TWO were discarded as contended -- `swz` rep 1 and `row-major`
/// rep 4, identified by their PREFILL, 165.5 s and 95.5 s against 14.2-15.3 s
/// for the other six; another agent's job arrived on the box mid-run. One
/// discard fell on each arm.
///
/// The gap between 17.5% on the kernel and 1.9% on the step is not a
/// discrepancy: the head is one term of a step that also streams 5.36 GiB of
/// attention, dense-MLP and routed-expert weight, and 1.1 ms of a 57.2 ms step
/// is most of what a 0.85 ms kernel saving can be worth once it is enqueued
/// against a host that is also doing everything else.
///
/// **Numerics.** Bit-identical, as a permutation must be: max deviation
/// 0.000e0 over 1024 outputs of a `[16, 256] x [64, 256]^T` product against the
/// row-major lane. Reported as an observation; there is no gate on it.
///
/// # RETRACTED FOR THE SINKS, 2026-08-26. Do not quote the table above at any
/// # shape but the head's.
///
/// SUPERSEDED IN SCOPE by the section below it, which measured the range this
/// one inferred from two points. The +25% is real and the lesson is right; what
/// is wrong here is the UNIT. "The sinks" is not a shape -- it is two shapes,
/// and only one of them loses. Read both sections, and take the rule from the
/// second: `swizzle_pays`.
///
/// The end-to-end row (`swz` 56.1 against `row-major` 57.2 ms/step) was taken
/// with `ANN_BUDGET_DEFAULT` already at 8192, so the head went through
/// `linear_ann` in BOTH arms and read identical bytes -- meaning the 1.1 ms was
/// attributed to the SINKS. Two things were wrong with that. It was confounded:
/// the `swz` arm of that A/B was the one carrying the `linear_ann` seam bug, so
/// the two arms generated different token streams and routed to different
/// experts. And when the unconfounded A/B was finally run -- a109514 against
/// e30f22a, ctx 3732, split 21, 7 reps, 441 warm passes a side -- it came back
/// the other way:
///
/// ```text
///   w4a16_linear, both nodes   6.84 -> 8.55 ms   (+1.71, +25% SLOWER permuted)
///     head 3.30 -> 4.05, tail 3.54 -> 4.50
/// ```
///
/// The kernel name changed `w4a16_linear_ab` -> `w4a16_linear_swz_ab`, so the
/// permuted path is unambiguously what ran.
///
/// The KERNEL table at the top stands -- it is the head's shape, 201024 x 4096,
/// ~25128 cubes, and 95.9 -> 116.3 GB/s there is real. The sinks are 8192x4096
/// and 4096x2048, i.e. 1024 and 512 cubes. `swz_grid_scaling` had already shown
/// the multiplier shrinking as the grid starves; at the sink shapes it inverts.
///
/// So `w4a16_bind` now permutes the HEAD only, and only when the approximate
/// lane is off. The sinks are correct in either layout -- they never touch
/// `ann_logits` -- so leaving them row-major costs nothing but the speed it
/// buys back. THE LESSON, which is the reason this retraction is written out
/// rather than deleted: a bandwidth figure is a property of a SHAPE, and this
/// one was carried across a 25x difference in cube count without anybody
/// noticing that the framing rule had been dropped.
///
/// # WHY the multiplier is shape-dependent, and what the shape actually is
/// # (2026-08-26)
///
/// The retraction above is right that it lost and right about the lesson. It is
/// wrong about the UNIT: "the sinks" is not a shape, it is two shapes, and they
/// fall on opposite sides of the line. `w4a16_swz_grid` is the A/B neither
/// earlier measurement had -- rotating buffers so neither arm is the L2-warm
/// one, 20 pipelined launches to ONE sync so the host round trip is not the
/// floor, arms round-robined, min of 6 rounds after 2 discarded, `m_pad` 16,
/// one GB10 box, GB/s over the weight planes only, `ratio` = row-major time /
/// swizzled time:
///
/// ```text
///   k=2048     512 cubes   70.7 ->  62.0 GB/s   0.88   <- sink `down`, ACTUAL
///             1024         92.6 -> 100.5        1.09
///            25128        116.7 -> 133.0        1.14
///   k=4096     256         49.0 ->  34.3        0.70
///              512         77.9 ->  65.7        0.84
///              768         90.7 ->  93.0        1.03   <- crossover
///             1024         94.3 -> 106.6        1.13   <- sink `gate_up`, ACTUAL
///             2048         90.9 -> 113.5        1.25   <- dense `g`/`u`
///            25128        107.3 -> 132.6        1.24   <- the head
///   k=16384    512         46.9 ->  58.4        1.25   <- dense `down`, ACTUAL
///             1024         66.4 ->  95.1        1.43
///             2048         67.8 ->  98.3        1.45
/// ```
///
/// TWO knobs, not one. The multiplier rises monotonically with CUBE COUNT and
/// crosses 1.0 near 750 cubes -- 0.65 of a wave, a wave being 48 SMs x 24 blocks
/// = 1152 single-warp cubes -- and the value it saturates to is set by K:
/// ~1.10 at k=2048, ~1.24 at k=4096, ~1.45 at k=16384. Cube count alone does not
/// decide it, which is why `down` at 512 cubes LOSES for the sinks (k=2048) and
/// WINS for the dense MLP (k=16384) at the identical cube count.
///
/// That cube count is the cause and not a correlate is the one thing a sweep
/// over `n` cannot show, so it was isolated: same `[4096, 4096]` table, same
/// bytes, only the m-tile count varied, so only the warps moved.
///
/// ```text
///   m_pad  16 ->  512 cubes  0.86       m_pad  64 -> 2048 cubes  1.15
///   m_pad  32 -> 1024 cubes  1.09       m_pad 128 -> 4096 cubes  1.20
/// ```
///
/// ## The mechanism, off `ncu`
///
/// Load sectors and requests per warp per k-tile, at both sink shapes, from
/// `l1tex__t_{requests,sectors}_pipe_lsu_mem_global_op_ld.sum` over
/// `cubes * k_tiles`. The decomposition closes to the integer on all three
/// arms and both shapes, which is why it can be read as a mechanism and not as
/// a correlation:
///
/// ```text
///                             requests   sectors   =    A  + codes + scales
///   row-major                     8         64        4x8=32   2x8    2x8
///   swizzled, scales permuted     7         35        4x8=32   2x1    1x1
///   swizzled, scales row-major    7         42        4x8=32   2x1    1x8
/// ```
///
/// So the permutation is worth 8x on each B stream -- 32 sectors of B become 3
/// -- and only **1.83x on the kernel**, because A is 32 of the 64 sectors and
/// the permutation does not touch it. After permuting B, A is 32 of the
/// remaining 35: **91% of the swizzled kernel's sector traffic is now the
/// activation.** PARKED, not a task, and it wants this rule shipped and measured
/// in-model first: A's floor is 16 sectors per warp-k-tile (one k-tile of A is
/// 16 rows x 16 k x 2 B = 512 B), so a fragment-ordered A would cut 35 -> 19, a
/// further 1.84x, on top of B's 1.83x. At decode there is ONE m-tile, so every
/// cube reads the identical A -- 16 x k x 2 bytes, 128 KiB at k = 4096 -- and a
/// per-step permutation of that is ~0.5 us against a 0.2 ms kernel. It is the
/// largest single number here and it is deliberately not being acted on yet.
///
/// And the row-major form's "wasted" seven-eighths is not waste, which is the
/// part that decides the sign. A 32-byte sector holds eight packed words = FOUR
/// k-tiles of one weight row, so row-major's burst of 8 parallel sector requests
/// is an incidental 4-deep PREFETCH: it misses once and then hits L1 for three
/// k-tiles (`l1tex__t_sector_hit_rate` 93.9-94.7% against the swizzled form's
/// 89.0-91.3%). The swizzled 64-byte block is consumed by the k-tile that
/// fetched it, so it exposes a fresh 2-sector miss EVERY k-tile instead of an
/// 8-sector burst every fourth -- four times as many exposed latencies at a
/// quarter of the per-warp memory-level parallelism, on a k loop that is not
/// unrolled and so has one k-tile of fetch outstanding at a time.
///
/// With enough resident warps the machine hides that with thread-level
/// parallelism and the 8x request saving is all that is left, so the permutation
/// wins. Below ~0.65 waves there is nothing to hide it with and the lost
/// prefetch dominates. `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active`
/// says exactly that: at 1024 cubes the swizzled kernel stalls LESS on memory
/// (13.2-13.8 against 16.7-17.1), and at 512 cubes that advantage is gone
/// (16.6-18.7 against 16.7-17.0).
///
/// THREE candidates die here, and all three had been stated confidently.
///
/// * It is NOT occupancy. Both lanes report the identical achieved occupancy
///   (44.4% at 1024 cubes, 22.1% at 512), both are capped by
///   `launch__occupancy_limit_blocks` = 24 -- one warp per cube against 48 warp
///   slots -- and below that by the grid itself.
/// * It is NOT register pressure. The swizzled kernel uses FEWER registers, 36
///   against 44 (38 with the scales left row-major).
/// * It is NOT "a separate kernel tuned for the head's regime", which is what
///   `20a0b06`'s commit message says and what this doc has to outlive, because a
///   commit message cannot be corrected. Strip the comments and the two kernels
///   differ by SIX LINES, all of them index arithmetic: same `CubeCount`, same
///   `CubeDim::new_1d(32)`, same tile, same unrolls, same accumulator, same
///   output store, 74 code lines against 79 (the permuted one is the LONGER).
///   There is no tuning knob that differs, so there is nothing to re-tune; the
///   whole difference in behaviour is the access pattern above.
///
/// One inference to retire with them, because it is the intuitive one and it is
/// backwards: "at low occupancy there is less to hide latency with, so halving
/// the request count matters MORE". Fewer requests is not more latency hiding.
/// At low occupancy what is scarce is outstanding requests PER WARP, and the
/// permutation removes them -- eight sectors in flight from one instruction
/// become one. That is why the sign inverts downward and not upward.
///
/// A measurement caveat that belongs with the numbers: `ncu`'s CYCLE counts are
/// inflated 2-7x for this kernel under kernel replay and reverse the ranking, so
/// only its COUNTERS above are usable. Every time figure here comes from
/// `w4a16_swz_grid`, not from `ncu`.
///
/// ## What it says about the four consumers
///
/// * sink `gate_up` `[8192, 4096]`, 1024 cubes: the permutation WINS 1.13-1.22x.
///   `w4a16_bind` currently declines it. That is the one place this file is
///   leaving measured speed on the table.
/// * sink `down` `[4096, 2048]`, 512 cubes: LOSES 0.88x. Declining it is right.
/// * the head `[201024, 4096]`, 25128 cubes: WINS 1.24x, and is declined for a
///   CORRECTNESS reason (the `ann_logits` seam), not this one.
/// * the dense MLP, which is the next candidate and the reason this was chased:
///   `g`/`u` `[16384, 4096]` at 2048 cubes WINS 1.25x, and `down`
///   `[4096, 16384]` at 512 cubes WINS 1.25x. Reasoning from the sinks as a
///   proxy would have got the second one backwards -- it is at the same starved
///   cube count that loses for the sinks, and it wins anyway because k is 8x
///   longer.
///
/// A NOTE ON THE +25% ABOVE, which this does not reproduce for `gate_up`:
/// `a109514 -> e30f22a` spans `de482cb`, which the comment in `w4a16_bind`
/// already flags as bundling the sink permutation with moving the pool
/// `memory_usage` barrier off the decode path; and at `a109514`
/// `ann_owns_m1` was `ann_budget() > 0` with no `for_ann` gate, so NOTHING was
/// permuted in that arm. The kernel-level A/B above varies the layout and
/// nothing else.
///
/// Finally, the reason `w4a16_swz_probe` must not be run at a sink shape: its
/// four arms share buffers -- the `stream ceiling` arm reads the same `b_row` /
/// `bs_row` the row-major arm reads, immediately before it -- and the two plane
/// sets are 36 MiB against a 24 MiB L2, so the row-major arm is L2-warm every
/// rep and the swizzled arms are L2-cold every rep. It reports 0.176 against
/// 0.306 ms there, a 74% "loss" that is an artifact of the arm ORDER. At the
/// head's 463 MiB nothing is resident and the confound does not exist, which is
/// why the table at the top of this doc stands.
pub fn swizzle_w4a16() -> bool {
    std::env::var("INK_W4A16_SWZ")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Does the `m16n8k16` permutation PAY at a `[n, k]` weight's decode grid?
///
/// The shipped policy used to be a WEIGHT-KIND rule -- `!for_ann`, i.e. never a
/// sink -- derived from one shape. It is a CUBE-COUNT-AND-K rule, because that
/// is what the crossover is a function of. Measured by `w4a16_swz_grid`
/// (rotating buffers so neither arm is the L2-warm one, 20 pipelined launches to
/// one sync so the host round trip is not the floor, arms round-robined, min of
/// 6 rounds after 2 discarded, `m_pad` 16, one GB10 box, GB/s over the weight
/// planes only). `ratio` is row-major time / swizzled time; above 1 it pays:
///
/// ```text
///                 cubes    row -> swz GB/s   ratio      this predicate
///   k=2048          512     70.7 ->  62.0     0.88   no   <- sink `down`
///                  1024     92.6 -> 100.5     1.09   YES
///                 25128    116.7 -> 133.0     1.14   YES
///   k=4096          256     49.0 ->  34.3     0.70   no
///                   512     77.9 ->  65.7     0.84   no
///                   768     90.7 ->  93.0     1.03   no   (crossover, declined)
///                  1024     94.3 -> 106.6     1.13   YES  <- sink `gate_up`
///                  2048     90.9 -> 113.5     1.25   YES  <- dense `g`/`u`
///                 25128    107.3 -> 132.6     1.24   YES  <- the head
///   k=16384         512     46.9 ->  58.4     1.25   YES  <- dense `down`
///                  1024     66.4 ->  95.1     1.43   YES
///                  2048     67.8 ->  98.3     1.45   YES
/// ```
///
/// MEASURED vs INTERPOLATED, because a threshold whose provenance is lost is a
/// threshold nobody can revise:
///
/// * `cubes >= 1024` is measured to pay at every k measured (1.09, 1.13, 1.43).
///   The bound is placed at 1024 and not at the ~750-cube crossover on purpose:
///   1.03 is not worth a layout, and the crossover moves with k.
/// * `cubes >= 512 && k >= 16384` is the single measured point at 1.25. It is
///   NOT extrapolated downward -- k = 8192 at 512 cubes has never been measured,
///   and the predicate declines it.
/// * `k < 2048` is below the measured floor entirely and is declined.
/// * 512..1024 cubes below k = 16384 is measured to LOSE (0.84-0.88).
///
/// WHY it is a function of these two things, since a threshold without its
/// mechanism becomes folklore -- which is how the +25% came to be applied 25x
/// outside its range in the first place:
///
/// A 32-byte sector holds eight packed words = FOUR k-tiles of one weight row,
/// so the row-major form's burst of eight sector requests is an incidental
/// 4-deep L1 PREFETCH: it misses once and then hits L1 for three k-tiles
/// (`l1tex__t_sector_hit_rate` 94.7% against the permuted form's 89.0-91.3%).
/// Its "wasted seven-eighths" was never waste, which is why every framing built
/// on sector efficiency alone was going to mislead. The permuted form's 64-byte
/// block is consumed by the k-tile that fetched it, so it exposes a fresh
/// 2-sector miss EVERY k-tile instead of an 8-sector burst every fourth -- four
/// times the exposed latencies at a quarter of the per-warp memory-level
/// parallelism, on a k loop that is not unrolled.
///
/// So the trade is 8x fewer requests against a lost prefetch, and which side
/// pays depends on whether there are enough resident warps to hide the latency
/// the prefetch was covering. Cube count is that warp count: each cube is ONE
/// warp, `launch__occupancy_limit_blocks` is 24, and 48 SMs x 24 = a 1152-cube
/// wave, so the ~750-cube crossover is 0.65 of a wave. K is the second knob
/// because a longer k loop spreads the L1 working set and makes the row-major
/// form's reuse less viable, so the saturated multiplier grows with it: ~1.10 at
/// k=2048, ~1.24 at k=4096, ~1.45 at k=16384. That is why dense `down` at 512
/// cubes WINS 1.25x where sink `down` at 512 cubes LOSES 0.88x -- same starved
/// geometry, opposite sign, because k is 8x longer.
///
/// ## Judged at the DECODE grid, deliberately
///
/// The layout is chosen once per weight at bind time and cannot vary per call,
/// so it must be judged at one `m_pad`, and that one is decode's `MTILE`. This
/// is safe in the direction that matters: cubes are `(m_pad / MTILE) * (n /
/// NTILE)`, so a prefill has strictly MORE cubes than the decode this was
/// judged at, and more cubes only moves further into the region where the
/// permutation pays. A layout that is right for decode cannot be wrong for
/// prefill on this axis.
pub fn swizzle_pays(n: usize, k: usize) -> bool {
    let cubes = n / NTILE;
    if k < 2048 {
        return false;
    }
    if cubes >= 1024 {
        return true;
    }
    cubes >= 512 && k >= 16384
}

/// Does `INK_W4A16_SWZ=1` explicitly ask for the permutation where the shipped
/// policy declines it?
///
/// [`swizzle_w4a16`] answers "is it allowed"; this answers "is it demanded". The
/// distinction exists because the shipped policy never permutes a sink, on a
/// measurement (25% slower at 512-1024 cubes) that has no counterpart at any
/// other shape -- we know 25128 cubes wins and 512-1024 loses and NOTHING in
/// between. Without a way to turn it back on, that crossover cannot be measured
/// without a rebuild, and an unmeasurable policy hardens into folklore, which is
/// exactly how the original figure came to be applied 25x outside its range.
///
/// This never overrides the CORRECTNESS guard. `ann_owns_m1` stays absolute:
/// `linear_ann` reads row-major and permuting the head while the approximate
/// lane owns m=1 makes every exact rescore read permuted bytes, which produced
/// plausible logits rather than an error. Forcing is a speed knob only.
pub fn swizzle_w4a16_forced() -> bool {
    std::env::var("INK_W4A16_SWZ")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// [`w4a16_linear`] reading a B operand written by [`swizzle_w4a16_codes_into`].
///
/// The ONLY difference is the two global indices. Everything else — the
/// dequantise, the scale application, the accumulator, the output store — is
/// the same code, because the permutation moves bytes and changes nothing about
/// what they mean.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_swz<AB: Scalar + Cast, S: Scalar, NA: Size, NC: Size>(
    a: &Tensor<Vector<AB, NA>>,
    b: &Tensor<u32>,
    b_sc: &Tensor<S>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] swz_sc: bool,
    scale: f32,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let m_tile = CUBE_POS_X as usize;
    let n_tile = CUBE_POS_Y as usize;
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

    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE);
    let wpb = comptime!(SWZ_BLOCK_CODES / 4);

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
            // Written out of `position_of_nth` rather than assumed, so it
            // tracks the target's own fragment layout: `row` is the k element,
            // `col` the n column, and `swz_word_k16`'s `w` is `row / 8`.
            let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
            let w = row as usize / CODES_PER_WORD;
            let blk = (n_tile * k_tiles + t) * wpb;
            let word = b[blk + w * NTILE + col as usize];
            let s = if swz_sc {
                f32::cast_from(b_sc[(n_tile * k_tiles + t) * NTILE + col as usize])
            } else {
                f32::cast_from(b_sc[(col as usize + n_base) * spr + (row as usize + kbase) / GROUP])
            };
            let mut v = Vector::<AB, NA>::empty();
            #[unroll]
            for j in 0..vs_b {
                let kk = row as usize + kbase + j;
                let code = (word >> (4 * (kk % CODES_PER_WORD)) as u32) & 15u32;
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

/// Launch [`w4a16_linear_swz`]. `swz_sc` says whether `b_sc` is permuted too.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_linear_swz_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    swz_sc: bool,
    scale: f32,
) -> Handle {
    assert_eq!(
        m_pad % MTILE,
        0,
        "m_pad {m_pad} is not a multiple of {MTILE}"
    );
    assert!(swizzleable(n, k), "[{n}, {k}] is not swizzleable");
    assert!(
        n / NTILE <= 65535,
        "{} n-tiles exceed the 65535 grid-y limit",
        n / NTILE
    );

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let vs = 32 / bf16::cube_type().size_bits();
    let wpr = k / CODES_PER_WORD;
    let spr = k / GROUP;

    unsafe {
        w4a16_linear_swz::launch::<bf16, e4m3, R>(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            TensorArg::from_raw_parts(a.clone(), [k, 1].into(), [m_pad, k].into()),
            TensorArg::from_raw_parts(b.clone(), [wpr, 1].into(), [n, wpr].into()),
            TensorArg::from_raw_parts(b_sc.clone(), [spr, 1].into(), [n, spr].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_pad, n].into()),
            k,
            n,
            swz_sc,
            scale,
        )
    };
    out
}

#[cfg(test)]
mod swizzle_k16_tests {
    use super::*;

    /// Every source byte lands somewhere, exactly once, on both planes.
    ///
    /// The failure this excludes is a colliding destination formula, which
    /// loses one byte and duplicates another and which no same-shaped output
    /// check would call an error. A non-square shape, so a transposed n/k
    /// cannot pass.
    /// The predicate agrees with the measured table at the four real consumers.
    ///
    /// The pair that matters is the last two: the SAME 512 cubes, opposite
    /// answers, because k differs 8x. A predicate that keyed on cube count alone
    /// would get one of them wrong, and the sink is the one that was used as the
    /// dense MLP's proxy.
    #[test]
    fn swizzle_pays_matches_the_measured_shapes() {
        // head [201024, 4096], 25128 cubes, measured 1.24x.
        assert!(swizzle_pays(201024, 4096));
        // sink gate_up [8192, 4096], 1024 cubes, measured 1.13-1.22x.
        assert!(swizzle_pays(8192, 4096));
        // dense g/u [16384, 4096], 2048 cubes, measured 1.25x.
        assert!(swizzle_pays(16384, 4096));

        // sink down [4096, 2048], 512 cubes, measured 0.88x -- LOSES.
        assert!(!swizzle_pays(4096, 2048));
        // dense down [4096, 16384], the SAME 512 cubes, measured 1.25x -- WINS.
        assert!(swizzle_pays(4096, 16384));

        // Declined below the measured floor rather than extrapolated into it.
        assert!(
            !swizzle_pays(4096, 8192),
            "512 cubes at k=8192 is unmeasured"
        );
        assert!(!swizzle_pays(2048, 4096), "256 cubes measured 0.70x");
        assert!(!swizzle_pays(201024, 1024), "k below the measured floor");
    }

    #[test]
    fn the_k16_permutation_is_a_bijection_on_both_planes() {
        let (n, k) = (16usize, 128usize);
        let codes: Vec<u8> = (0..n * k / 2).map(|i| (i % 251) as u8).collect();
        let mut a = swizzle_w4a16_codes(&codes, n, k);
        let mut b = codes.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);

        let mut seen = vec![false; codes.len()];
        let kt = k / KTILE;
        for nt in 0..n / NTILE {
            for t in 0..kt {
                for col in 0..NTILE {
                    for w in 0..KTILE / CODES_PER_WORD {
                        let d = (nt * kt + t) * SWZ_BLOCK_CODES + swz_word_k16(col, w) * 4;
                        for x in 0..4 {
                            assert!(!seen[d + x], "destination {} written twice", d + x);
                            seen[d + x] = true;
                        }
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|v| *v),
            "some destination was never written"
        );

        let scales: Vec<u8> = (0..n * (k / GROUP)).map(|i| (i % 241) as u8).collect();
        let mut a = swizzle_w4a16_scales(&scales, n, k);
        let mut b = scales.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}

// ---------------------------------------------------------------------------
// The m16n8k16 B fragment map, off the device
// ---------------------------------------------------------------------------

/// Dump `position_of_nth` for the `m16n8k16` fragment, as the DEVICE answers it.
///
/// [`super::fp4gemm::fp4_frag_b_map`] does this for `m16n8k64`
/// (`new_scaled::<e4m3>`); this is its twin for the shape BOTH four-bit-head and
/// BF16 lanes actually issue. It is a separate dump and not a re-read of the
/// other because the two are different instructions: a different `k`, a
/// different operand width, a different constructor. The FP4 permutation was
/// derived from the first map, and nothing about that derivation transfers.
///
/// One `MmaDefinition` serves both consumers here:
/// [`w4a16_linear`] and [`super::bf16gemm::bf16_linear`] both construct
/// `MmaDefinition::<bf16, bf16, f32>::new(16, 8, 16)` — same types, same
/// constructor, same shape — so one dump settles the layout for 3.70 GiB/step
/// of BF16 traffic and the 0.43 GiB/step head at once.
///
/// Layout of `out` (u32 words):
///
/// ```text
///   0   .. 256   B:   [lane * 4 + i] -> (row, col)      i < 4
///   256 .. 512   A:   same indexing
///   512 .. 768   Acc: same indexing
///   768 ..       counts: see the writes below
/// ```
#[cube(launch)]
pub fn mma16_frag_map<AB: Scalar>(out: &mut Tensor<u32>) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    // B is indexed EXACTLY as `w4a16_linear` indexes it -- `i * vs_b`, no
    // packing factor -- so this dumps the addresses that kernel computes and
    // not a neighbouring convention.
    #[unroll]
    for i in 0..vc_b {
        let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
        out[(lane as usize * 4 + i) * 2] = row;
        out[(lane as usize * 4 + i) * 2 + 1] = col;
    }
    #[unroll]
    for i in 0..vc_a {
        let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
        out[256 + (lane as usize * 4 + i) * 2] = row;
        out[256 + (lane as usize * 4 + i) * 2 + 1] = col;
    }
    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        out[512 + (lane as usize * 4 + i) * 2] = row;
        out[512 + (lane as usize * 4 + i) * 2 + 1] = col;
    }
    if lane == 0 {
        out[768] = vc_b as u32;
        out[769] = vs_b as u32;
        out[770] = ec_b as u32;
        out[771] = vc_a as u32;
        out[772] = vs_a as u32;
        out[773] = ec_a as u32;
        out[774] = vc_c as u32;
        out[775] = vs_c as u32;
        out[776] = ec_c as u32;
        out[777] = pack as u32;
    }
}

/// Launch [`mma16_frag_map`] and return the raw `u32` dump.
pub fn mma16_frag_map_launch<R: Runtime>(client: &ComputeClient<R>) -> Handle {
    let out = client.empty(1024 * 4);
    unsafe {
        mma16_frag_map::launch::<bf16, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            TensorArg::from_raw_parts(out.clone(), [1].into(), [1024].into()),
        )
    };
    out
}
