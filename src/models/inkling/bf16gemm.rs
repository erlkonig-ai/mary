//! Routed-expert FFN on the native BF16 tensor-core path.
//!
//! Inkling-Small is mixed precision: 41 of its 42 layers carry NVFP4 experts and
//! layer 2 carries plain BF16 ones — `w13_weight` `[256, 4096, 4096]` and
//! `w2_weight` `[256, 4096, 2048]`, 12.9 GiB, with no `.scale` sidecar because
//! nothing quantised them. The authors left an early, sensitive layer
//! unquantised; the runtime's job is to run it AS BF16, not to re-quantise it
//! and not to widen it.
//!
//! This is [`super::fp4gemm::fp4_linear`] minus the block-scale plumbing.
//! Everything structural is the same — one plane per `(m_tile, n_tile)`, the
//! weights streamed once out of global memory, the activation re-read per plane
//! out of L2, the accumulation in the MMA's own f32 accumulator — and the two
//! differences are both consequences of the format:
//!
//! * [`cmma::MmaDefinition::new`] instead of `new_scaled`, because BF16 carries
//!   no scales. Same type, same `execute` protocol, different constructor.
//! * `m16n8k16` instead of `m16n8k64`: the k of one instruction is a property of
//!   the operand width, and 16 bits per element buys a quarter of the k that 4
//!   bits did.
//!
//! ## f32 accumulation is not widening
//!
//! `mma.sync…bf16` has no BF16-accumulator form: the instruction multiplies
//! BF16 by BF16 and accumulates into f32, and f32 is its OWN output type. The
//! weight is never materialised wider than it is stored — the bytes go from the
//! checkpoint's mapping into A/B registers as `bf16`. What the f32 lane this
//! replaces did, and what the standing spec forbids, is something else: it read
//! the stored BF16 and wrote 604 MB of f32 per token to hold weights the
//! checkpoint keeps in 151 MB.
//!
//! ## The activation is BF16 too
//!
//! For the same reason the NVFP4 lane's activation is E2M1: the instruction
//! takes the same type for both operands. Here that costs nothing to justify —
//! the reference implementation runs the whole model in BF16, so a BF16
//! activation is what `transformers` multiplies as well. The cast is
//! [`to_bf16`], on the DEVICE, so its rounding is the hardware's
//! round-to-nearest-even and matches `torch.Tensor.to(torch.bfloat16)` rather
//! than a hand-rolled host twin of it.

use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::bf16;

/// Rows of one MMA tile — the M granularity everything here is padded to.
pub const MTILE: usize = 16;
/// Columns of one MMA tile.
pub const NTILE: usize = 8;
/// K covered by one `m16n8k16` instruction.
///
/// A quarter of the NVFP4 lane's, and for the same reason its operands are a
/// quarter the width: one `.b32` register holds two BF16 where it held eight
/// E2M1 codes.
pub const KTILE: usize = 16;

/// `out = a @ b^T`, with `a` and `b` both BF16.
///
/// `a` is `[m_pad, k]` and `b` is `[n, k]`, i.e. the checkpoint's own
/// `[out, in]` orientation, which is already the column-major B the instruction
/// wants. `out` is `[m_pad, n]` f32 — the accumulator's type, read out as it
/// stands.
#[cube(launch)]
pub fn bf16_linear<AB: Scalar, NA: Size, NC: Size>(
    a: &Tensor<Vector<AB, NA>>,
    b: &Tensor<Vector<AB, NA>>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    // 1 for BF16. Kept in the index arithmetic anyway so this reads as the same
    // kernel as the packed one rather than a second dialect of it.
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
            // B is column-major w.r.t. the tile: `col` indexes n, `row` indexes
            // k, and the checkpoint's `[out, in]` rows are exactly that.
            let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
            let gr = col as usize + n_base;
            let gc = row as usize + kbase;
            reg_b[i] = b[(gr * size_k + gc) / b.vector_size()];
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
        out[(gr * size_n + gc) / out.vector_size()] = acc[i];
    }
}

/// Round an f32 buffer to BF16, elementwise, on the device.
///
/// The residual stream is f32 on the host, the MMA takes BF16 on both operands,
/// and this is the one place that difference is paid. Deliberately a device
/// kernel and not a host loop: `bf16::cast_from` lowers to the hardware's
/// `cvt.rn.bf16.f32`, which is round-to-nearest-even — the same rounding
/// `torch.Tensor.to(torch.bfloat16)` performs, so the oracle and the lane agree
/// on the operand BITS and every remaining difference is the accumulation's.
/// `n_in` is how much of `y` `x` actually covers; the rest is the MMA's M
/// padding and is written as zero. The lane this replaces built the padded f32
/// buffer first — `Tensor::cat(x, Tensor::zeros(…))`, an allocation and two
/// scatters — and then cast the whole thing. A decode step feeds one token
/// against a sixteen-row tile, so that was fifteen rows of zeros materialised
/// in f32, cast to BF16, and multiplied, 127 times a token. The cast was
/// already visiting every element; it may as well decide which ones exist.
#[cube(launch)]
pub fn to_bf16(x: &Tensor<f32>, y: &mut Tensor<bf16>, n_in: usize) {
    let idx = ABSOLUTE_POS as usize;
    if idx < y.len() {
        let mut v = f32::new(0.0f32);
        if idx < n_in {
            v = x[idx];
        }
        y[idx] = bf16::cast_from(v);
    }
}

/// Launch [`bf16_linear`] for a `[m_pad, k] x [n, k]^T` product.
pub fn bf16_linear_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
) -> Handle {
    assert_eq!(m_pad % MTILE, 0, "m_pad {m_pad} is not a multiple of {MTILE}");
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    // One 32-bit register's worth of operand per vector, which is what
    // `contiguous_elements` reports for A and B and what the fragment layout
    // actually is: two BF16 per `.b32`.
    let vs = 32 / bf16::cube_type().size_bits();

    unsafe {
        bf16_linear::launch::<bf16, R>(
            client,
            CubeCount::Static((n / NTILE) as u32, (m_pad / MTILE) as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            TensorArg::from_raw_parts(a.clone(), [k, 1].into(), [m_pad, k].into()),
            TensorArg::from_raw_parts(b.clone(), [k, 1].into(), [n, k].into()),
            TensorArg::from_raw_parts(out.clone(), [n, 1].into(), [m_pad, n].into()),
            k,
            n,
        )
    };
    out
}

/// Launch [`to_bf16`] over `n_in` f32 elements into an `n_out`-element BF16
/// buffer, zeroing the tail.
///
/// `n_out == n_in` is the unpadded case and costs the same branch.
pub fn to_bf16_launch<R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    n_in: usize,
    n_out: usize,
) -> Handle {
    assert!(n_in <= n_out, "{n_in} f32 do not fit in {n_out} BF16");
    let out = client.empty(n_out * core::mem::size_of::<bf16>());
    let threads = 256u32;
    let blocks = n_out.div_ceil(threads as usize) as u32;
    unsafe {
        to_bf16::launch::<R>(
            client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(threads),
            TensorArg::from_raw_parts(x.clone(), [1].into(), [n_in].into()),
            TensorArg::from_raw_parts(out.clone(), [1].into(), [n_out].into()),
            n_in,
        )
    };
    out
}

/// Upload an `[rows, k]` f32 host slice, padded to [`MTILE`], as BF16.
///
/// Returns `(bf16 handle, m_pad)`. The padding rows are zero and stay zero: a
/// zero BF16 times anything is a zero product, so the padded output rows are
/// exactly zero and a gate can assert it.
pub fn upload_bf16_act<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    rows: usize,
    k: usize,
) -> (Handle, usize) {
    use cubecl::prelude::CubeElement;
    let m_pad = rows.div_ceil(MTILE) * MTILE;
    let h = client.create_from_slice(f32::as_bytes(&x[..rows * k]));
    (to_bf16_launch(client, &h, rows * k, m_pad * k), m_pad)
}

/// A dense `[out, in]` weight on the device, as the BF16 the pile stores.
///
/// Not a Burn tensor, because Burn's f32 backend cannot hold one without
/// widening it — which is the whole thing this type exists to refuse. It is a
/// handle over BF16 bytes, and where those bytes came from is usually the
/// pile's own mapping with nothing copied at all.
pub struct Bf16W {
    pub h: Handle,
    /// Output rows.
    pub n: usize,
    /// Input columns.
    pub k: usize,
}

impl Bf16W {
    /// Stored bytes this weight spans.
    pub fn bytes(&self) -> usize {
        self.n * self.k * 2
    }

    /// The shapes the `m16n8k16` instruction can tile without a remainder.
    ///
    /// Checked at construction rather than at launch: a weight that cannot be
    /// tiled is a fact about the model, and finding it out on the first token
    /// of a two-node run is finding it out in the worst place.
    pub fn tileable(n: usize, k: usize) -> bool {
        n % NTILE == 0 && k % KTILE == 0
    }
}

/// `x @ Wᵀ` with `x` f32 on the device and `W` the BF16 it is stored as.
///
/// `x_h` is `[m, k]` f32, `m_pad` is `m` rounded up to [`MTILE`], and the
/// return is `[m_pad, n]` f32 — the accumulator's own type. The activation is
/// cast to BF16 on the device by [`to_bf16`], whose rounding is the hardware's
/// round-to-nearest-even and therefore the same one
/// `torch.Tensor.to(torch.bfloat16)` performs, and which now writes the M
/// padding as it goes rather than being handed a buffer somebody else padded.
pub fn linear_bf16<R: Runtime>(
    client: &ComputeClient<R>,
    x_h: &Handle,
    w: &Bf16W,
    m: usize,
    m_pad: usize,
) -> Handle {
    let a = to_bf16_launch(client, x_h, m * w.k, m_pad * w.k);
    bf16_linear_launch(client, &a, &w.h, m_pad, w.k, w.n)
}
