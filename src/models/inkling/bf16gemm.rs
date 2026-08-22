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

/// A census of the hand lane, so "how much still goes through it" is a number.
///
/// Slots: 0 = launches, 1 = summed `m*k*n` MACs, 2 = launches reached via the
/// `align < MIN_TUNED_ALIGN` gate in [`bf16_gemm`], 3 = launches reached
/// because `INK_GEMM=hand mma` forced it.
pub static HAND: [core::sync::atomic::AtomicU64; 4] =
    [const { core::sync::atomic::AtomicU64::new(0) }; 4];

/// Print the hand-lane census. Prints nothing when the lane never ran.
pub fn report_hand() {
    use core::sync::atomic::Ordering::Relaxed;
    let a: Vec<u64> = HAND.iter().map(|c| c.load(Relaxed)).collect();
    if a[0] == 0 {
        println!("  hand BF16 lane: 0 launches -- every plain-BF16 GEMM went to a tuned lane");
        return;
    }
    println!(
        "  hand BF16 lane: {} launches, {:.3} GMAC, {} via the alignment gate, {} forced",
        a[0],
        a[1] as f64 / 1e9,
        a[2],
        a[3]
    );
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
    {
        use core::sync::atomic::Ordering::Relaxed;
        HAND[0].fetch_add(1, Relaxed);
        HAND[1].fetch_add((m_pad * k * n) as u64, Relaxed);
    }
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
    /// Byte alignment of the bound buffer, capped at 16.
    ///
    /// Load-bearing, not bookkeeping. The tuned lane picks its load width from
    /// the SHAPE and never from the pointer, so a `[4096, 4096]` operand gets
    /// 16-byte loads at whatever address it sits at — and this runtime's
    /// weights are ALIASED out of an arena that only promises 4. A 4-aligned
    /// weight on a 16-byte
    /// load is `CUDA_ERROR_MISALIGNED_ADDRESS`, an async fault that takes the
    /// server down with it, so the alignment has to be known BEFORE the launch
    /// and it is only knowable at the bind.
    ///
    /// **Where the 4 comes from.** Not the pile: the startup weight copy packs
    /// its views back to back with `cursor = end + (4 - end % 4) % 4`, twice,
    /// in its layout pass. Sixteen there instead of four costs at most 15 bytes
    /// per view — about 14 KB across an 8-layer share — and it makes every
    /// weight reachable by the tuned lanes. Measured with the startup copy ON,
    /// layers 0:8, 512-token prefill, p50 over 24 warm passes: 438.3 ms at 4,
    /// 332.3 ms at 16, against the 330.1 ms `INK_ALIGN_COPY=1` buys by
    /// duplicating 908 MiB of weight. The change belongs in the startup copy,
    /// not here.
    pub align: usize,
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
) -> Handle {
    let rows = rows_for(w.align, m);
    let a = to_bf16_launch(client, x_h, m * w.k, rows * w.k);
    bf16_gemm(client, &a, &w.h, rows, w.k, w.n, w.align)
}

/// A BF16-to-BF16 copy into an `n_out`-element buffer, zeroing the tail.
///
/// [`to_bf16`] is this kernel with a cast in the middle; this is the same
/// padding for an activation that is BF16 already. It exists only for the M
/// padding — when a lane needs none, [`linear_bf16_narrow`] does not call it at
/// all and the activation is handed to the MMA where it lies.
#[cube(launch)]
pub fn pad_bf16(x: &Tensor<bf16>, y: &mut Tensor<bf16>, n_in: usize) {
    let idx = ABSOLUTE_POS as usize;
    if idx < y.len() {
        let mut v = bf16::cast_from(0.0f32);
        if idx < n_in {
            v = x[idx];
        }
        y[idx] = v;
    }
}

/// Launch [`pad_bf16`].
pub fn pad_bf16_launch<R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    n_in: usize,
    n_out: usize,
) -> Handle {
    assert!(n_in <= n_out, "{n_in} BF16 do not fit in {n_out}");
    let out = client.empty(n_out * core::mem::size_of::<bf16>());
    let threads = 256u32;
    let blocks = n_out.div_ceil(threads as usize) as u32;
    unsafe {
        pad_bf16::launch::<R>(
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

/// [`linear_bf16`] for an activation that is **already** BF16.
///
/// The narrow lane's normed residual stream is BF16 when it reaches here, and
/// [`linear_bf16`] would then be handed an f32 buffer that only exists to be
/// cast straight back — `[n, 4096]`, 16 KiB a token, allocated, written, read
/// once and freed, several times a layer.
///
/// So this skips the cast. And when the weight's alignment lets the tuned lane
/// take `m` rows unpadded — which [`rows_for`] reports and which is the common
/// case for a prefill — it skips the COPY as well and hands the MMA the
/// residual stream's own buffer. Nothing writes through it: `bf16_gemm` reads A
/// and writes a fresh accumulator.
pub fn linear_bf16_narrow<R: Runtime>(
    client: &ComputeClient<R>,
    x_h: &Handle,
    w: &Bf16W,
    m: usize,
) -> Handle {
    let rows = rows_for(w.align, m);
    if rows == m {
        return bf16_gemm(client, x_h, &w.h, rows, w.k, w.n, w.align);
    }
    let a = pad_bf16_launch(client, x_h, m * w.k, rows * w.k);
    bf16_gemm(client, &a, &w.h, rows, w.k, w.n, w.align)
}

// ---------------------------------------------------------------------------
// The tuned lane.
//
// [`bf16_linear`] above is one 32-thread warp per 16x8 output tile, reading A
// and B straight out of global memory before every `mma`. There is no shared
// memory in it, no `cp.async`, no double buffering and no m-tile reuse, so the
// tile it computes is 16x8 no matter how much of the machine is idle and every
// cube re-reads the whole activation. That is why the forward measured 2.9
// TFLOP/s on a part that does 95.9.
//
// None of that has to be written here. `cubek::matmul` is the matmul burn
// dispatches to for exactly these shapes — staged through shared memory, plane
// (multi-warp) cubes, double-buffered and TMA variants, a gemv lane for skinny
// m — and it takes BINDINGS, not Burn tensors. A raw `Handle` is a binding. So
// the whole of the port is describing the three operands and picking a
// strategy.
//
// The transpose is free and is the reason `b` needs no touching: the weight is
// `[n, k]` row-major, and `[k, n]` with strides `[1, k]` is the same bytes read
// as the column-major B this product wants.

/// Which `cubek` matmul routine to launch.
///
/// Not an implementation detail: the right answer is different for a 512-row
/// prefill GEMM and a 16-row decode one, and the bench binary walks this enum
/// rather than a comment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// The hand-written `mma.sync…bf16` kernel above.
    Hand,
    /// `cubek`'s own dispatch: simple cyclic cmma, falling back to units.
    Auto,
    SimpleCyclicCmma,
    SimpleCyclicMma,
    SimpleTmaCmma,
    SimpleTmaMma,
    DoubleCyclicCmma,
    DoubleCyclicMma,
    DoubleTmaCmma,
    DoubleTmaMma,
    DoubleHybridCmma,
    DoubleHybridMma,
    OrderedDoubleCmma,
    OrderedDoubleMma,
    SpecializedCyclicCmma,
    SpecializedTmaCmma,
    SimpleUnit,
    DoubleUnit,
    SimpleVecMat,
    DoubleVecMat,
    GemvUnitPerpendicular,
    GemvPlaneParallel,
    /// `double tma mma` with the specialized (producer/consumer) schedule.
    DoubleTmaMmaSpec,
    /// `simple tma mma` with the multi-row selection, i.e. m-tile reuse.
    SimpleTmaMmaMulti,
    /// `simple cyclic mma` with the multi-row selection.
    SimpleCyclicMmaMulti,
    /// `ordered double mma`, k partitioned two ways inside a stage.
    OrderedDoubleMmaPk2,
    /// ...and four.
    OrderedDoubleMmaPk4,
}

impl Lane {
    /// Every lane, in the order the bench should report them.
    pub const ALL: &'static [Lane] = &[
        Lane::Hand,
        Lane::Auto,
        Lane::SimpleCyclicCmma,
        Lane::SimpleCyclicMma,
        Lane::SimpleTmaCmma,
        Lane::SimpleTmaMma,
        Lane::DoubleCyclicCmma,
        Lane::DoubleCyclicMma,
        Lane::DoubleTmaCmma,
        Lane::DoubleTmaMma,
        Lane::DoubleHybridCmma,
        Lane::DoubleHybridMma,
        Lane::OrderedDoubleCmma,
        Lane::OrderedDoubleMma,
        Lane::SpecializedCyclicCmma,
        Lane::SpecializedTmaCmma,
        Lane::SimpleUnit,
        Lane::DoubleUnit,
        Lane::SimpleVecMat,
        Lane::DoubleVecMat,
        Lane::GemvUnitPerpendicular,
        Lane::GemvPlaneParallel,
        Lane::DoubleTmaMmaSpec,
        Lane::SimpleTmaMmaMulti,
        Lane::SimpleCyclicMmaMulti,
        Lane::OrderedDoubleMmaPk2,
        Lane::OrderedDoubleMmaPk4,
    ];

    /// The name the bench prints.
    pub fn name(&self) -> &'static str {
        match self {
            Lane::Hand => "hand mma",
            Lane::Auto => "cubek auto",
            Lane::SimpleCyclicCmma => "simple cyclic cmma",
            Lane::SimpleCyclicMma => "simple cyclic mma",
            Lane::SimpleTmaCmma => "simple tma cmma",
            Lane::SimpleTmaMma => "simple tma mma",
            Lane::DoubleCyclicCmma => "double cyclic cmma",
            Lane::DoubleCyclicMma => "double cyclic mma",
            Lane::DoubleTmaCmma => "double tma cmma",
            Lane::DoubleTmaMma => "double tma mma",
            Lane::DoubleHybridCmma => "double hybrid cmma",
            Lane::DoubleHybridMma => "double hybrid mma",
            Lane::OrderedDoubleCmma => "ordered double cmma",
            Lane::OrderedDoubleMma => "ordered double mma",
            Lane::SpecializedCyclicCmma => "specialized cyclic cmma",
            Lane::SpecializedTmaCmma => "specialized tma cmma",
            Lane::SimpleUnit => "simple unit",
            Lane::DoubleUnit => "double unit",
            Lane::SimpleVecMat => "simple vecmat",
            Lane::DoubleVecMat => "double vecmat",
            Lane::GemvUnitPerpendicular => "gemv unit perp",
            Lane::GemvPlaneParallel => "gemv plane par",
            Lane::DoubleTmaMmaSpec => "double tma mma spec",
            Lane::SimpleTmaMmaMulti => "simple tma mma multi",
            Lane::SimpleCyclicMmaMulti => "simple cyclic mma multi",
            Lane::OrderedDoubleMmaPk2 => "ordered double mma pk2",
            Lane::OrderedDoubleMmaPk4 => "ordered double mma pk4",
        }
    }

    fn strategy(&self) -> cubek::matmul::launch::Strategy {
        use cubek::matmul::launch::Strategy;
        match self {
            Lane::Hand => unreachable!("the hand lane is not a cubek strategy"),
            Lane::Auto => Strategy::Auto,
            Lane::SimpleCyclicCmma => Strategy::SimpleCyclicCmma(Default::default()),
            Lane::SimpleCyclicMma => Strategy::SimpleCyclicMma(Default::default()),
            Lane::SimpleTmaCmma => Strategy::SimpleTmaCmma(Default::default()),
            Lane::SimpleTmaMma => Strategy::SimpleTmaMma(Default::default()),
            Lane::DoubleCyclicCmma => Strategy::DoubleCyclicCmma(Default::default()),
            Lane::DoubleCyclicMma => Strategy::DoubleCyclicMma(Default::default()),
            Lane::DoubleTmaCmma => Strategy::DoubleTmaCmma(Default::default()),
            Lane::DoubleTmaMma => Strategy::DoubleTmaMma(Default::default()),
            Lane::DoubleHybridCmma => Strategy::DoubleHybridCmma(Default::default()),
            Lane::DoubleHybridMma => Strategy::DoubleHybridMma(Default::default()),
            Lane::OrderedDoubleCmma => Strategy::OrderedDoubleCmma(Default::default()),
            Lane::OrderedDoubleMma => Strategy::OrderedDoubleMma(Default::default()),
            Lane::SpecializedCyclicCmma => Strategy::SpecializedCyclicCmma(Default::default()),
            Lane::SpecializedTmaCmma => Strategy::SpecializedTmaCmma(Default::default()),
            Lane::SimpleUnit => Strategy::SimpleUnit(Default::default()),
            Lane::DoubleUnit => Strategy::DoubleUnit(Default::default()),
            Lane::SimpleVecMat => Strategy::SimpleVecMat(Default::default()),
            Lane::DoubleVecMat => Strategy::DoubleVecMat(Default::default()),
            Lane::GemvUnitPerpendicular => Strategy::GemvUnitPerpendicular(Default::default()),
            Lane::GemvPlaneParallel => Strategy::GemvPlaneParallel(Default::default()),
            Lane::DoubleTmaMmaSpec => {
                use cubek::matmul::routines::{double_buffering::DoubleBufferingArgs, BlueprintStrategy};
                use cubek::matmul::components::tile::TileMatmulKind;
                Strategy::DoubleTmaMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    tile_matmul: TileMatmulKind::Mma,
                    specialized: true,
                }))
            }
            Lane::SimpleTmaMmaMulti => {
                use cubek::matmul::routines::{simple::SimpleArgs, BlueprintStrategy};
                use cubek::matmul::components::tile::TileMatmulKind;
                Strategy::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
                    tile_matmul: TileMatmulKind::Mma,
                    multi_rows: true,
                }))
            }
            Lane::SimpleCyclicMmaMulti => {
                use cubek::matmul::routines::{simple::SimpleArgs, BlueprintStrategy};
                use cubek::matmul::components::tile::TileMatmulKind;
                Strategy::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
                    tile_matmul: TileMatmulKind::Mma,
                    multi_rows: true,
                }))
            }
            Lane::OrderedDoubleMmaPk2 | Lane::OrderedDoubleMmaPk4 => {
                use cubek::matmul::routines::{ordered_double_buffering::OrderedSelectionArgs, BlueprintStrategy};
                use cubek::matmul::components::tile::TileMatmulKind;
                Strategy::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    tile_matmul: TileMatmulKind::Mma,
                    partition_k: Some(if *self == Lane::OrderedDoubleMmaPk2 { 2 } else { 4 }),
                    row_count: None,
                    rows_per_plane: None,
                }))
            }
        }
    }
}

/// A `[shape]`-shaped, `[strides]`-strided view of `h`, as the launcher wants it.
fn binding<R: Runtime>(h: &Handle, shape: [usize; 2], strides: [usize; 2]) -> TensorBinding<R> {
    TensorBinding {
        handle: h.clone().binding(),
        strides: strides.into(),
        shape: shape.into(),
        runtime: core::marker::PhantomData,
    }
}

/// `out = a @ b^T` through `cubek`'s tuned matmul, with `a` and `b` BF16 and
/// `out` f32.
///
/// Same contract as [`bf16_linear_launch`] and deliberately the same signature,
/// so the two are interchangeable at every call site and the bench can A/B them
/// by swapping a function pointer.
///
/// Nothing here is padded. The hand kernel needs `m` a multiple of 16 because
/// its grid IS the tiling; this one bounds-checks its own tiles, so `m_pad` may
/// be the true `m`. It is still accepted as `m_pad` because the caller has one.
pub fn bf16_linear_cubek_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    lane: Lane,
) -> Handle {
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    use cubek::matmul::definition::{MatmulElems, MatmulGlobalElems};
    use cubek::std::InputBinding;

    let bf = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
    let f32s = StorageType::Scalar(ElemType::Float(FloatKind::F32));

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());

    // `[k, n]` strided `[1, k]` over a `[n, k]` row-major buffer: the same
    // bytes, read as the column-major B the product wants. No copy, no kernel.
    let lhs = InputBinding::new(binding::<R>(a, [m_pad, k], [k, 1]), bf);
    let rhs = InputBinding::new(binding::<R>(b, [k, n], [1, k]), bf);
    let outb = binding::<R>(&out, [m_pad, n], [n, 1]);

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: bf,
        rhs: bf,
        out: f32s,
    });

    cubek::matmul::launch::launch_ref::<R>(&lane.strategy(), client, lhs, rhs, outb, &mut dtypes)
        .unwrap_or_else(|e| panic!("cubek matmul [{m_pad},{k}]x[{n},{k}]^T on {lane:?}: {e:?}"));
    out
}

/// The same, returning the setup error instead of panicking on it.
///
/// Every `cubek` strategy declines some shapes -- a TMA lane wants alignments a
/// 258-wide router projection does not have, a gemv lane wants a small m -- and
/// "declines" is an answer, not a crash. The bench walks the whole enum and
/// needs to be told which ones answered.
#[allow(clippy::result_large_err)]
pub fn try_bf16_linear_cubek_launch<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    lane: Lane,
) -> Result<Handle, cubek::matmul::definition::MatmulSetupError> {
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    use cubek::matmul::definition::{MatmulElems, MatmulGlobalElems};
    use cubek::std::InputBinding;

    let bf = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
    let f32s = StorageType::Scalar(ElemType::Float(FloatKind::F32));
    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let lhs = InputBinding::new(binding::<R>(a, [m_pad, k], [k, 1]), bf);
    let rhs = InputBinding::new(binding::<R>(b, [k, n], [1, k]), bf);
    let outb = binding::<R>(&out, [m_pad, n], [n, 1]);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: bf,
        rhs: bf,
        out: f32s,
    });
    cubek::matmul::launch::launch_ref::<R>(&lane.strategy(), client, lhs, rhs, outb, &mut dtypes)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Which lane the forward actually runs.
//
// Measured on this box with `inkling_bf16_gemm_bench`, over the nine plain-BF16
// shapes a 20-layer node issues, summed per pass (synchronised, 30 iters at
// m=512 and 20 at m=16):
//
//   m = 512 (prefill)             m = 16 (decode)
//     double tma mma      84.8      double tma mma      36.1
//     simple tma mma      86.0      double cyclic mma   38.4
//     double cyclic mma  119.0      simple tma mma      38.6
//     hand mma          1176.3      hand mma            50.0
//
// `double tma mma` wins both ends, so it heads the preference list; the rest is
// a FALLBACK CHAIN, not a tuning surface. A strategy declines a shape it cannot
// tile or align, and declining is an answer -- the walk stops at the first one
// that takes the shape and the hand kernel, which takes every shape the forward
// issues, is the floor.
//
// The choice is cached per `(m, k, n)`: a probe costs a failed setup, and the
// forward issues the same handful of shapes tens of thousands of times.

/// The order [`bf16_gemm`] tries lanes in, unless `INK_GEMM` names one.
///
/// The TMA lanes head it, and they are the reason the alignment gate below is
/// not optional. A `cuTensorMap` requires a 16-byte-aligned global address; the
/// first attempt at this list died on "Tensor pointer must be 16 byte aligned"
/// during the layer uploads, because the weights are ALIASED out of the pile
/// mapping at whatever offset the leaf lies at and that seam only promises 4.
/// It is an ASYNC launch fault that poisons the server, so it cannot be caught
/// and retried -- the alignment has to be decided before the launch. With
/// `Bf16W::align` deciding it, a weight the TMA lane would fault on never
/// reaches this list at all, and the lane is safe on both bind arms.
/// It opens with a GEMV, and that is the decode step's whole story: a decode
/// step feeds ONE row, `gemv plane par` is the only routine here that reaches
/// this part's memory roofline on one row (219 GB/s on `attn wq` against 121
/// for the best tiled lane, 252 GB/s on the 134 MB `dense w13` against 120),
/// and it DECLINES the shape as soon as m grows. That decline used to be
/// described here as costing "a prefill pass one failed setup per shape and
/// nothing else", which is true at m = 512 and badly false at m = 2: the lane
/// below the gemv is the PREFILL lane, so a two-row pass fell all the way past
/// the widths in between and paid a fixed ~23 ms a node for it. That is what
/// [`PREFERENCE_NARROW`] is, and this list is now the m = 1 and m >= MTILE list
/// rather than the only one.
///
/// `simple vecmat` is deliberately NOT on the list: it is a hair slower than the
/// gemv at m=1 and it ACCEPTS m=16, where it is four times slower than the tiled
/// lanes -- a lane that takes a shape it should refuse is worse than one that
/// refuses a shape it could take. It is off [`PREFERENCE_NARROW`] for the same
/// reason, measured there.
///
/// It ends with `simple unit`, not the hand kernel. The hand kernel needs m
/// padded to 16 and the tuned lanes do not, so once [`rows_for`] stops padding
/// there is no unpadded fallback to the hand lane -- `simple unit` is the
/// routine `cubek`'s own `Auto` falls back to and it takes every shape.
const PREFERENCE: &[Lane] = &[
    Lane::GemvPlaneParallel,
    Lane::DoubleTmaMmaSpec,
    Lane::DoubleTmaMma,
    Lane::SimpleTmaMma,
    Lane::DoubleCyclicMma,
    Lane::SimpleCyclicMma,
    Lane::SpecializedCyclicCmma,
    Lane::SimpleCyclicCmma,
    Lane::SimpleUnit,
];

/// The order at `2 <= m < MTILE`, which is the width a speculative verify feeds.
///
/// [`PREFERENCE`] is an ORDER, and an order can only be right for one m. It was
/// written for a decode step, where `gemv plane par` is the answer, and the lane
/// below it -- `double tma mma spec` -- is the answer at a prefill. Between them
/// sits the width a verify pass actually feeds, and neither end of the list is
/// right there: `gemv plane par` DECLINES m > 1 outright
/// (`GemvKind::from_problem` requires m == 1 or n == 1), so the walk fell
/// straight through to the prefill lane and every BF16 weight in the model was
/// streamed at 80-120 GB/s instead of 220-250.
///
/// **That is what capped speculation at break-even.** The cost of the downgrade
/// is proportional to the model's WEIGHT BYTES, which is a constant of the model
/// and not a function of w, so it appeared as a fixed penalty the moment a pass
/// grew a second row -- and a fixed cost that does not scale with the work looks
/// like physics rather than like a list in the wrong order. Measured end to end
/// on this box, `INK_LAYERS=0:21 INK_REPEAT=1`, p50 over 12 warm passes:
///
///     w = 1, gemv lane reachable                73.2 ms
///     w = 1, INK_GEMM=double tma mma spec       98.2 ms   <- the lane, alone
///     w = 2, the order below unheaded          109.2 ms
///
/// 22.8 ms of the 36.0 ms step from w = 1 to w = 2 is that one lane. It is not a
/// property of the uncached lane either: the same A/B on the CACHED decode
/// (`INK_KV=1`, m = 1) costs 70.1 / 73.0 ms with the gemv lane and 96.2 / 93.3
/// without it, so a cached multi-row verify walks into the same wall.
///
/// ## Why this head, measured end to end and not in the harness
///
/// p50 over 12 warm passes, same node, same prompts, one column per head of
/// this list (`INK_GEMM_NARROW`), against the order as written and against a
/// per-shape auto-tuner:
///
///     w   as written   pk4 head   vecmat head   dcm head   per-shape tuner
///     1        73.2      (n/a)         (n/a)      (n/a)             83.4
///     2       109.2      101.0          98.1      104.3             97.2
///     3       115.8      102.0         140.7      110.9            109.7
///     5       125.8      116.3         200.3      123.3            117.3
///     8       134.4      133.6         195.0      131.8            133.8
///
/// Two recorded negatives in that table, and both are the reason this is a list
/// and not something cleverer.
///
/// `simple vecmat` is the best single choice at w = 2 and a disaster from w = 3
/// on, because its cost is linear in m where the tiled lanes' is flat to
/// [`MTILE`]. It is deliberately absent below: a lane that is right for one
/// width of a band and 1.5x wrong for the rest of it is worse than one that is
/// second-best everywhere.
///
/// A per-shape AUTO-TUNER -- time all thirteen candidate lanes on the real
/// operands at first touch, keep the fastest per `(m, k, n)` -- was built and
/// measured worse. It costs 10.2 ms a pass at m = 1 (83.4 against 73.2), where
/// it abandons `gemv plane par` for seven of the nine shapes because in
/// isolation the lanes come out within a percent of each other and the pass says
/// otherwise; and it loses to this list at w = 3 and w = 5. A GEMM timed alone
/// is a GEMM that had the whole device, and four of a layer's projections are
/// independent and overlap. An isolated timer cannot see that, so the selection
/// stays a measured order and the measurement stays end to end.
///
/// The band stops at [`MTILE`] because that is where [`PREFERENCE`]'s own
/// measurement starts (its comment reports m = 512 and m = 16), and starts at 2
/// because m = 1 is the one width [`PREFERENCE`] was written for.
const PREFERENCE_NARROW: &[Lane] = &[
    Lane::OrderedDoubleMmaPk4,
    Lane::DoubleCyclicMma,
    Lane::DoubleHybridMma,
    Lane::OrderedDoubleMma,
    Lane::SimpleTmaMma,
    Lane::DoubleTmaMmaSpec,
    Lane::DoubleTmaMma,
    Lane::SimpleCyclicMma,
    Lane::SpecializedCyclicCmma,
    Lane::SimpleCyclicCmma,
    Lane::SimpleUnit,
];

/// `INK_GEMM_NARROW=<lane name>`, parsed once: a lane to put at the head of
/// [`PREFERENCE_NARROW`]. It is how the table in that doc comment was produced
/// without a rebuild per arm, and it is the A/B handle for the next person who
/// wants to move the head.
fn narrow_head() -> Option<Lane> {
    use std::sync::OnceLock;
    static HEAD: OnceLock<Option<Lane>> = OnceLock::new();
    *HEAD.get_or_init(|| {
        let name = std::env::var("INK_GEMM_NARROW").ok()?;
        Some(Lane::from_name(&name).unwrap_or_else(|| {
            panic!(
                "INK_GEMM_NARROW={name} names no lane; the lanes are: {}",
                Lane::ALL.iter().map(|l| l.name()).collect::<Vec<_>>().join(", ")
            )
        }))
    })
}

/// The order to walk for an `m`-row activation.
///
/// Three regimes and two lists: m = 1 is a decode step and wants the gemv,
/// m >= MTILE is a prefill and wants the tiled lane, and between them is the
/// verify width, which wants neither. Returned by value because the walk only
/// runs on a cache MISS -- once per distinct `(m, k, n)` in a process.
fn preference(m: usize) -> Vec<Lane> {
    if !(2..MTILE).contains(&m) {
        return PREFERENCE.to_vec();
    }
    let mut v = PREFERENCE_NARROW.to_vec();
    if let Some(head) = narrow_head() {
        v.retain(|&l| l != head);
        v.insert(0, head);
    }
    v
}

/// The alignment a weight needs before the tuned lanes may touch it.
pub const MIN_TUNED_ALIGN: usize = 16;

/// How many rows the lane will compute for an `m`-row activation.
///
/// The hand kernel's grid IS its tiling, so it needs `m` rounded up to
/// [`MTILE`] and the caller slices the padding off afterwards. The tuned lanes
/// bounds-check their own tiles and take `m` as it stands — which is not a
/// tidiness point. A decode step feeds ONE row, and padding it to sixteen both
/// multiplied fifteen rows of zeros by every weight in the model and, far worse,
/// HID `m = 1` from the gemv routines: they decline m=16, so the padding was
/// the reason the fastest decode kernel on this box was never reachable.
pub fn rows_for(align: usize, m: usize) -> usize {
    // `INK_GEMM=hand mma` is an A/B arm and it has to get its padding, or the
    // one thing it exists to be compared against cannot run. Asking
    // [`forced_lane`] here rather than assuming is the same discipline the
    // alignment check is: the padding is the LANE's requirement, so the lane
    // decides it.
    if align >= MIN_TUNED_ALIGN && forced_lane() != Some(Lane::Hand) {
        m
    } else {
        m.div_ceil(MTILE) * MTILE
    }
}

impl Lane {
    /// The inverse of [`Lane::name`], for `INK_GEMM`.
    pub fn from_name(s: &str) -> Option<Lane> {
        Lane::ALL.iter().copied().find(|l| l.name() == s)
    }
}

/// `INK_GEMM`, parsed once. `None` means "walk [`PREFERENCE`]".
fn forced_lane() -> Option<Lane> {
    use std::sync::OnceLock;
    static FORCED: OnceLock<Option<Lane>> = OnceLock::new();
    *FORCED.get_or_init(|| {
        let name = std::env::var("INK_GEMM").ok()?;
        Some(Lane::from_name(&name).unwrap_or_else(|| {
            panic!(
                "INK_GEMM={name} names no lane; the lanes are: {}",
                Lane::ALL.iter().map(|l| l.name()).collect::<Vec<_>>().join(", ")
            )
        }))
    })
}

/// The per-shape decision, read or written through the one map that holds it.
fn lane_cache(
    shape: (usize, usize, usize),
    set: Option<Lane>,
) -> Option<Lane> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), Lane>>> = OnceLock::new();
    let mut map = CACHE.get_or_init(Default::default).lock().expect("lane cache");
    match set {
        Some(lane) => {
            map.insert(shape, lane);
            Some(lane)
        }
        None => map.get(&shape).copied(),
    }
}

/// Launch one lane, which must be known to accept the shape.
fn launch_lane<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    m: usize,
    k: usize,
    n: usize,
    lane: Lane,
) -> Handle {
    match lane {
        Lane::Hand => bf16_linear_launch(client, a, b, m, k, n),
        _ => bf16_linear_cubek_launch(client, a, b, m, k, n, lane),
    }
}

/// `out = a @ b^T` for BF16 operands and an f32 accumulator, on the fastest
/// lane this box has for the shape.
///
/// `a` is `[m, k]`, `b` is `[n, k]` (the checkpoint's own orientation) and the
/// result is `[m, n]` f32.
pub fn bf16_gemm<R: Runtime>(
    client: &ComputeClient<R>,
    a: &Handle,
    b: &Handle,
    m: usize,
    k: usize,
    n: usize,
    align: usize,
) -> Handle {
    // A weight the tuned lane would fault on goes to the hand kernel, which
    // loads 4 bytes at a time and takes any 4-aligned address. This is a
    // per-WEIGHT decision and not a per-shape one: two `[4096, 4096]`
    // projections in the same layer land at different offsets in the mapping.
    if align < MIN_TUNED_ALIGN {
        HAND[2].fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        return bf16_linear_launch(client, a, b, m, k, n);
    }
    if let Some(lane) = forced_lane() {
        if lane == Lane::Hand {
            HAND[3].fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }
        return launch_lane(client, a, b, m, k, n, lane);
    }
    let shape = (m, k, n);
    if let Some(lane) = lane_cache(shape, None) {
        return launch_lane(client, a, b, m, k, n, lane);
    }
    for lane in preference(m) {
        if let Ok(h) = try_bf16_linear_cubek_launch(client, a, b, m, k, n, lane) {
            lane_cache(shape, Some(lane));
            return h;
        }
    }
    unreachable!("every order ends with `simple unit`, which takes every shape")
}
