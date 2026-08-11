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
) {
    let idx = ABSOLUTE_POS as usize;
    if idx < act.len() {
        let r = idx / inter;
        let i = idx % inter;
        let g = both[r * 2 * inter + 2 * i];
        let u = both[r * 2 * inter + 2 * i + 1];
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
        )
    };
    act
}
