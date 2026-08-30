//! [`super::fp4gemm::fp4_linear_swz`] written out as PTX, by hand.
//!
//! The first hand-written kernel in this tree that is not a smoke test.
//! [`super::rawptx`] proved the path — a `.version`-led source goes to
//! `cuModuleLoadData` instead of NVRTC and rides the same stream, arena,
//! capture and by-value blob — on a two-buffer `alpha * x`. This is the same
//! path carrying the routed lane's actual GEMM: five buffers, one f32 scalar,
//! the NVFP4 block-scaled tensor-core MMA, and a K loop unrolled into the
//! text.
//!
//! # Why generate the text instead of writing it
//!
//! JP, 2026-08-30: *the kernel knows its dims at compile time*. So
//! [`fp4_linear_swz_ptx`] is a **function of the shape** — `size_k`, `size_n`
//! and whether the scale plane was permuted — and everything derived from
//! those (`k_tiles = size_k / 64`, `spr = size_k / 16`, the row pitch
//! `size_k / 2`, the 256-byte swizzled block stride) is folded at generation
//! time and appears in the text as a literal. The K loop is not a loop: it is
//! `k_tiles` blocks of straight-line code, each addressing its operands as
//! **one base register plus one immediate**. There is no induction variable,
//! no compare, no branch and no address arithmetic inside the kernel at all
//! past the prologue.
//!
//! That is also what makes the kernel id correct without any extra machinery.
//! [`super::rawcuda::RawCudaKernel`] hashes the source bytes, so a
//! specialisation for `(4096, 4096, swz)` and one for `(2048, 4096, swz)` are
//! *different kernels* with different ids, different modules and different
//! cache entries — exactly as two comptime instantiations of the cubecl
//! kernel are. [`fp4_linear_swz_ptx_kernel`] memoises them so that a launch
//! does not re-`format!` fifty kilobytes and re-FNV it; at the shapes this
//! runs, that cost would otherwise land inside the measured launch.
//!
//! # The lane's job, and where every address comes from
//!
//! One 32-lane plane per cube, one `(m_tile, n_tile)` per cube, grid `x` = M
//! tiles and grid `y` = N tiles — the launch geometry of
//! [`super::fp4gemm::fp4_linear_swz_launch`], unchanged, because the grid
//! order is a measured property of that lane (M in x so the `m_pad / 16`
//! consumers of one weight row are launched adjacently) and this arm is meant
//! to be swapped in against it, not to differ from it.
//!
//! With `g = lane >> 2` and `tig = lane & 3`, the fragment maps for
//! `m16n8k64` / `scale_vec::4X` on `sm_121a` — `cubecl-cpp`'s closed forms in
//! `cuda/processors.rs`, confirmed against the one-hot device measurement
//! recorded in `nvfp4_mma_probe`'s header — are:
//!
//! ```text
//!   A reg i (i in 0..4)   row = g + 8*(i & 1)     col = tig*8 + 32*(i >> 1)
//!   B reg i (i in 0..2)   col = g                 row = tig*8 + 32*i
//!   C/D pair i (0..2)     row = g + 8*i           col = tig*2  (+0, +1)
//!   A scales              row = g + (lane & 1)*8, four bytes, one per k-block
//!   B scales              row = g,                four bytes, one per k-block
//! ```
//!
//! Feed those through the cubecl kernel's own index expressions and every
//! operand collapses to base + immediate:
//!
//! ```text
//!   A codes    pa0 = a + (m_base + g)*(K/2) + tig*4     [pa0 + t*32]      i=0
//!              pa1 = pa0 + 8*(K/2)                      [pa1 + t*32]      i=1
//!                                                       [pa0 + t*32 + 16] i=2
//!                                                       [pa1 + t*32 + 16] i=3
//!   B codes    pb  = b + n_tile*k_tiles*256 + lane*4    [pb + t*256]      i=0
//!                                                       [pb + t*256+128]  i=1
//!   A scales   psa = a_sc + (m_base + g + (lane&1)*8)*spr   [psa + t*4]
//!   B scales   psb = b_sc + n_tile*k_tiles*32 + g*4         [psb + t*32]   (swizzled)
//!              psb = b_sc + (n_base + g)*spr                [psb + t*4]    (row-major)
//!   out        pout = out + ((m_base+g)*N + n_base + 2*tig)*4
//!                                            [pout], [pout + 32*N]
//! ```
//!
//! The B line is the whole point of the swizzled layout stated in one
//! expression: the byte offset inside the `(n_tile, k_tile)` block is
//! `(32*i + lane) * 4`, so load `i` across the warp is 128 **consecutive**
//! bytes in lane order. [`super::fp4gemm::swz_word`]'s derivation says this;
//! here it is not a derivation but the immediate in the instruction.
//!
//! # Bit-identity with the cubecl arm
//!
//! **Identical, and not merely close, everywhere it matters:**
//!
//! * The MMA is the same instruction with the same operands in the same
//!   order. The mnemonic below is a verbatim copy of what
//!   `cubecl-cpp`'s `mma_scaled_template` renders for
//!   `(e2m1x2, e2m1x2, f32, k=64, ue4m3, scales_factor 4)` — including the
//!   braces around the single-register scale operands and the two 16-bit
//!   selectors held in a register rather than written as immediates, which is
//!   what NVRTC produces from that template's `"h"` constraints and therefore
//!   the exact form already proven to assemble on this part.
//! * The **accumulate order is the same**: `k_tiles` MMAs in ascending `t`,
//!   each taking the previous D as its C, starting from `+0.0f`. Nothing is
//!   re-associated, split or reordered — the accumulator chain is a true
//!   dependency chain in both arms.
//! * The **scales are applied by the instruction**, not by this code, so
//!   there is no scale-application order to change.
//! * The four scale bytes arrive as one 32-bit register in both arms. cubecl
//!   assigns `sa[i]` for `i in 0..4` into a four-byte vector and passes it as
//!   a `uint32`; this loads the same four bytes with one `ld.global.nc.u32`.
//!   The addresses are 4-byte aligned (`spr = K/16` is a multiple of 4
//!   whenever `K % 64 == 0`, which the launcher asserts) and the device is
//!   little-endian, so byte `i` is scale-block `i` on both sides.
//! * The epilogue is one `mul` per accumulator element by the runtime `scale`
//!   and an 8-byte vector store, as in the cubecl kernel. `mul.rn.f32` is one
//!   IEEE round-to-nearest multiply with no contraction — there is no add to
//!   contract with — which is what NVRTC emits for `acc[i] * scale`.
//!
//! **Where it cannot be identical, and this is not a numeric difference:**
//!
//! * *Instruction scheduling and register allocation.* The K loop is a
//!   runtime loop in the cubecl arm (only the inner fragment loops carry
//!   `#[unroll]`), so NVRTC unrolls it by whatever factor it likes; here it is
//!   unrolled in full. Both arms then hand ptxas a different scheduling
//!   problem. That moves *when* loads issue, never what they load.
//! * *`ld.global.nc`.* The read-only operands are loaded through the
//!   non-coherent path, which is what NVRTC emits for the
//!   `const T* __restrict__` parameters `cubecl-cpp`'s `compile_bindings`
//!   generates for read-only buffers. Same bytes, same cache correctness
//!   (nothing in the grid writes them), so this matches the cubecl arm rather
//!   than departing from it.
//! * *No bounds checks in either arm.* `ExecutionMode` is not consulted.
//!
//! # Not executed here
//!
//! Written on a machine with no CUDA device. `ptxas` is not installed on it
//! either, so the text has not been assembled. On a Spark:
//!
//! ```text
//! ptxas -arch=sm_121a -v -o /dev/null fp4_linear_swz.ptx
//! ptx_fp4_probe                       # both arms, one process, identity + p50
//! ```

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{LazyLock, Mutex};

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::server::Handle;

use super::fp4gemm::{GROUP, KTILE, MTILE, NTILE};
use super::rawcuda::RawArgs;
use super::rawptx::{RawPtxKernel, ptx_header};

/// Threads per cube: one plane, the geometry
/// [`super::fp4gemm::fp4_linear_swz_launch`] launches with.
pub const UNITS: u32 = 32;

/// Bytes of one swizzled `(n_tile, k_tile)` B block: `NTILE` weight rows x
/// `KTILE` codes, four bits each.
const BLOCK_BYTES: usize = NTILE * KTILE / 2;

/// Bytes of one swizzled `(n_tile, k_tile)` B **scale** block: `NTILE` rows x
/// `KTILE / GROUP` E4M3 bytes.
const SCALE_BLOCK_BYTES: usize = NTILE * (KTILE / GROUP);

/// The block-scaled MMA, mnemonic for mnemonic as `cubecl-cpp` renders it.
///
/// `cuda/ptx/mma.rs`'s `mma_scaled_template` with `a = b = e2m1`,
/// `cd = f32`, `k = 64`, `stype = ue4m3`, `scales_factor = 4` produces
/// `kind::mxf4nvf4` (factor 2 or 4) and this operand shape: D and C four
/// `.f32` registers each — the scaled form requires float C/D, unlike the
/// unscaled one — A four `.b32`, B two `.b32`, then per operand a `.b32` of
/// four scale bytes and a `{byte-id, thread-id}` selector pair.
const MMA: &str = "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X\
.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3";

// ---------------------------------------------------------------------------
// The address arithmetic, once
// ---------------------------------------------------------------------------
//
// Every function below is used BY the emitter to produce a literal, and by
// the tests to check that literal against the cubecl kernel's own index
// expressions re-derived from `cubecl-cpp`'s fragment maps. They are the only
// place the layout is written down.

/// Byte immediate of A register `i` at k tile `t`, off whichever of the two A
/// row bases `i & 1` selects.
fn a_imm(t: usize, i: usize) -> usize {
    // `t * 32`: one k tile is `KTILE` codes = 32 bytes of a weight row.
    // `16 * (i >> 1)`: A registers 2 and 3 sit 32 codes further along k.
    t * (KTILE / 2) + (KTILE / 4) * (i >> 1)
}

/// Byte distance between the two A row bases: the fragment's rows `g` and
/// `g + 8`, i.e. eight rows of `size_k / 2` packed bytes.
fn a_row_gap(size_k: usize) -> usize {
    (MTILE / 2) * (size_k / 2)
}

/// Byte immediate of B register `i` at k tile `t`, off the single B base.
///
/// Inside the `(n_tile, t)` block, lane `l`'s load `i` is word `32 * i + l`
/// ([`super::fp4gemm::swz_word`]), so the lane part is in the base and the
/// immediate is the block and the load index.
fn b_imm(t: usize, i: usize) -> usize {
    t * BLOCK_BYTES + i * (BLOCK_BYTES / 2)
}

/// Byte immediate of the four A scale bytes at k tile `t`.
fn a_scale_imm(t: usize) -> usize {
    t * (KTILE / GROUP)
}

/// Byte immediate of the four B scale bytes at k tile `t`, in whichever
/// layout `swz_sc` names.
fn b_scale_imm(t: usize, swz_sc: bool) -> usize {
    if swz_sc {
        t * SCALE_BLOCK_BYTES
    } else {
        t * (KTILE / GROUP)
    }
}

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------

/// The entry-point symbol for a specialisation: the shape is in the name so
/// that an `nsys`/`ncu` trace or a `ptxas -v` report says which kernel it is
/// looking at without being told.
pub fn fp4_linear_swz_ptx_name(size_k: usize, size_n: usize, swz_sc: bool) -> String {
    format!(
        "fp4_linear_swz_k{size_k}_n{size_n}{}",
        if swz_sc { "_s" } else { "" }
    )
}

/// The complete PTX module for one shape: [`ptx_header`] then the entry.
///
/// `size_k` and `size_n` are the K and N of the `[m_pad, K] x [N, K]^T`
/// product; `swz_sc` says whether the E4M3 scale plane was permuted alongside
/// the codes ([`super::fp4gemm::swizzle_b_scales`]). `m_pad` is deliberately
/// NOT a parameter: it is the grid's x extent and nothing in the kernel
/// depends on it, exactly as in the cubecl arm where it is a launch argument
/// and not a comptime one.
///
/// # Panics
///
/// If the shape does not tile: `size_k` must be a multiple of [`KTILE`] and
/// `size_n` of [`NTILE`], which is [`super::fp4gemm::swizzleable`]'s rule and
/// the launcher's assertion.
pub fn fp4_linear_swz_ptx(size_k: usize, size_n: usize, swz_sc: bool) -> String {
    assert_eq!(
        size_k % KTILE,
        0,
        "size_k {size_k} is not a multiple of {KTILE}"
    );
    assert_eq!(
        size_n % NTILE,
        0,
        "size_n {size_n} is not a multiple of {NTILE}"
    );

    let name = fp4_linear_swz_ptx_name(size_k, size_n, swz_sc);
    let k_tiles = size_k / KTILE;
    let spr = size_k / GROUP;
    let khalf = size_k / 2;
    let b_block_stride = k_tiles * BLOCK_BYTES;
    let bsc_block_stride = k_tiles * SCALE_BLOCK_BYTES;
    let row_gap = a_row_gap(size_k);
    let out_row_gap = (MTILE / 2) * size_n * 4;

    let mut s = ptx_header();
    let _ = write!(
        s,
        r#"
// {name}
//
// out[m_pad, {size_n}] = (a[m_pad, {size_k}] @ b[{size_n}, {size_k}]^T) * scale,
// both operands NVFP4 (E2M1 codes, one E4M3 scale per {GROUP}), B{} written in
// {MTILE}x{NTILE}x{KTILE} fragment order. One plane per (m_tile, n_tile);
// {k_tiles} k tiles, unrolled, every operand base + immediate.
//
// info: {{ float scale @0 }}
.visible .entry {name}(
    .param .u64 a_ptr,
    .param .u64 asc_ptr,
    .param .u64 b_ptr,
    .param .u64 bsc_ptr,
    .param .u64 out_ptr,
    .param .align 8 .b8 info[8]
)
.maxntid {UNITS}, 1, 1
{{
    .reg .b32  %r<16>;
    .reg .b64  %rd<14>;
    .reg .b16  %hz<1>;
    .reg .f32  %fc<4>;
    .reg .f32  %fs<1>;
    // One fresh name per unrolled load, so that ptxas is free to hoist a
    // tile's operands ahead of the MMAs still consuming the previous ones.
    // Reusing four A names across {k_tiles} tiles would write a WAR edge into
    // the text between every load and the MMA before it.
    .reg .b32  %ra<{n_ra}>;
    .reg .b32  %rb<{n_rb}>;
    .reg .b32  %sa<{k_tiles}>;
    .reg .b32  %sb<{k_tiles}>;

    // lane decomposition: g = lane >> 2 (the fragment's row/column group),
    // tig = lane & 3 (the position inside the group).
    mov.u32     %r0, %tid.x;
    mov.u32     %r1, %ctaid.x;                  // m_tile
    mov.u32     %r2, %ctaid.y;                  // n_tile
    shr.u32     %r3, %r0, 2;                    // g
    and.b32     %r4, %r0, 3;                    // tig

    // A codes: pa0 = a + (m_tile*{MTILE} + g)*{khalf} + tig*4, pa1 = pa0 + {row_gap}
    ld.param.u64 %rd0, [a_ptr];
    cvta.to.global.u64 %rd0, %rd0;
    mad.lo.u32  %r5, %r1, {MTILE}, %r3;
    mul.wide.u32 %rd2, %r5, {khalf};
    add.s64     %rd1, %rd0, %rd2;
    shl.b32     %r6, %r4, 2;
    cvt.u64.u32 %rd2, %r6;
    add.s64     %rd1, %rd1, %rd2;
    add.s64     %rd3, %rd1, {row_gap};

    // B codes: pb = b + n_tile*{b_block_stride} + lane*4. Load i of k tile t is
    // word 32*i + lane of the 256-byte block, i.e. 128 contiguous bytes across
    // the warp in lane order.
    ld.param.u64 %rd4, [b_ptr];
    cvta.to.global.u64 %rd4, %rd4;
    mul.wide.u32 %rd6, %r2, {b_block_stride};
    add.s64     %rd5, %rd4, %rd6;
    shl.b32     %r7, %r0, 2;
    cvt.u64.u32 %rd6, %r7;
    add.s64     %rd5, %rd5, %rd6;

    // A scales: psa = a_sc + (m_tile*{MTILE} + g + (lane & 1)*8)*{spr}
    ld.param.u64 %rd7, [asc_ptr];
    cvta.to.global.u64 %rd7, %rd7;
    and.b32     %r8, %r0, 1;
    shl.b32     %r9, %r8, 3;
    add.s32     %r10, %r3, %r9;
    mad.lo.u32  %r11, %r1, {MTILE}, %r10;
    mul.wide.u32 %rd13, %r11, {spr};
    add.s64     %rd8, %rd7, %rd13;

    // B scales
    ld.param.u64 %rd9, [bsc_ptr];
    cvta.to.global.u64 %rd9, %rd9;
"#,
        if swz_sc { " and its scale plane" } else { "" },
        n_ra = 4 * k_tiles,
        n_rb = 2 * k_tiles,
    );

    if swz_sc {
        let _ = write!(
            s,
            "    mul.wide.u32 %rd13, %r2, {bsc_block_stride};\n\
             \x20   add.s64     %rd10, %rd9, %rd13;\n\
             \x20   shl.b32     %r12, %r3, 2;\n\
             \x20   cvt.u64.u32 %rd13, %r12;\n\
             \x20   add.s64     %rd10, %rd10, %rd13;\n",
        );
    } else {
        let _ = write!(
            s,
            "    mad.lo.u32  %r12, %r2, {NTILE}, %r3;\n\
             \x20   mul.wide.u32 %rd13, %r12, {spr};\n\
             \x20   add.s64     %rd10, %rd9, %rd13;\n",
        );
    }

    s.push_str(
        "\n    // accumulator, and the {byte-id, thread-id} selector pair the\n\
         \x20   // instruction takes for each operand's scales.\n\
         \x20   mov.f32     %fc0, 0f00000000;\n\
         \x20   mov.f32     %fc1, 0f00000000;\n\
         \x20   mov.f32     %fc2, 0f00000000;\n\
         \x20   mov.f32     %fc3, 0f00000000;\n\
         \x20   mov.u16     %hz0, 0;\n",
    );

    for t in 0..k_tiles {
        let _ = write!(s, "\n    // k tile {t}\n");
        for i in 0..4 {
            let base = if i & 1 == 0 { "%rd1" } else { "%rd3" };
            let _ = writeln!(
                s,
                "    ld.global.nc.u32 %ra{}, [{base}+{}];",
                4 * t + i,
                a_imm(t, i)
            );
        }
        for i in 0..2 {
            let _ = writeln!(
                s,
                "    ld.global.nc.u32 %rb{}, [%rd5+{}];",
                2 * t + i,
                b_imm(t, i)
            );
        }
        let _ = writeln!(s, "    ld.global.nc.u32 %sa{t}, [%rd8+{}];", a_scale_imm(t));
        let _ = writeln!(
            s,
            "    ld.global.nc.u32 %sb{t}, [%rd10+{}];",
            b_scale_imm(t, swz_sc)
        );
        let _ = write!(
            s,
            "    {MMA}\n\
             \x20       {{%fc0, %fc1, %fc2, %fc3}},\n\
             \x20       {{%ra{}, %ra{}, %ra{}, %ra{}}},\n\
             \x20       {{%rb{}, %rb{}}},\n\
             \x20       {{%fc0, %fc1, %fc2, %fc3}},\n\
             \x20       {{%sa{t}}}, {{%hz0, %hz0}},\n\
             \x20       {{%sb{t}}}, {{%hz0, %hz0}};\n",
            4 * t,
            4 * t + 1,
            4 * t + 2,
            4 * t + 3,
            2 * t,
            2 * t + 1,
        );
    }

    let _ = write!(
        s,
        r#"
    // out[(m_base + g + 8*i)*{size_n} + n_base + 2*tig] = acc * scale, as two
    // 8-byte vector stores -- the accumulator pair (fc0,fc1) is row g and
    // (fc2,fc3) is row g+8, both at columns 2*tig and 2*tig+1.
    ld.param.f32 %fs0, [info];
    mul.rn.f32  %fc0, %fc0, %fs0;
    mul.rn.f32  %fc1, %fc1, %fs0;
    mul.rn.f32  %fc2, %fc2, %fs0;
    mul.rn.f32  %fc3, %fc3, %fs0;

    ld.param.u64 %rd11, [out_ptr];
    cvta.to.global.u64 %rd11, %rd11;
    shl.b32     %r13, %r4, 1;
    mad.lo.u32  %r13, %r2, {NTILE}, %r13;
    mul.wide.u32 %rd13, %r5, {size_n};
    cvt.u64.u32 %rd12, %r13;
    add.s64     %rd13, %rd13, %rd12;
    shl.b64     %rd13, %rd13, 2;
    add.s64     %rd12, %rd11, %rd13;
    st.global.v2.f32 [%rd12+0], {{%fc0, %fc1}};
    st.global.v2.f32 [%rd12+{out_row_gap}], {{%fc2, %fc3}};

    ret;
}}
"#
    );

    s
}

/// One [`RawPtxKernel`] per shape, built once.
///
/// The text is tens of kilobytes and its FNV-1a is part of the kernel id, so
/// regenerating it per launch would put a measurable slice of host time
/// *inside* the launch this module exists to time. Cloning is two `Arc`s and
/// a `String`.
pub fn fp4_linear_swz_ptx_kernel(size_k: usize, size_n: usize, swz_sc: bool) -> RawPtxKernel {
    static CACHE: LazyLock<Mutex<HashMap<(usize, usize, bool), RawPtxKernel>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    CACHE
        .lock()
        .expect("ptx kernel cache")
        .entry((size_k, size_n, swz_sc))
        .or_insert_with(|| {
            RawPtxKernel::new(
                fp4_linear_swz_ptx_name(size_k, size_n, swz_sc),
                fp4_linear_swz_ptx(size_k, size_n, swz_sc),
                CubeDim::new_1d(UNITS),
            )
        })
        .clone()
}

/// [`super::fp4gemm::fp4_linear_swz_launch`]'s arguments, its grid, its
/// output — the hand-written arm of the same product, so a caller can run
/// either on the same handles.
///
/// Not generic over `Runtime`: PTX is a CUDA artefact and the routing that
/// gets it past NVRTC lives in the CUDA server.
///
/// # Preconditions
///
/// As unchecked as the cubecl arm and for the same reason — nothing bounds
/// checks a fragment load on either side. `a` must hold `m_pad * (k / 2)`
/// bytes and `a_sc` `m_pad * (k / 16)`; `b` must hold `n * (k / 2)` bytes in
/// [`super::fp4gemm::swizzle_b_codes`] order and `b_sc` `n * (k / 16)`, in
/// [`super::fp4gemm::swizzle_b_scales`] order iff `swz_sc`.
#[allow(clippy::too_many_arguments)]
pub fn fp4_linear_swz_ptx_launch(
    client: &ComputeClient<CudaRuntime>,
    a: &Handle,
    a_sc: &Handle,
    b: &Handle,
    b_sc: &Handle,
    m_pad: usize,
    k: usize,
    n: usize,
    scale: f32,
    swz_sc: bool,
) -> Handle {
    assert_eq!(
        m_pad % MTILE,
        0,
        "m_pad {m_pad} is not a multiple of {MTILE}"
    );
    assert_eq!(n % NTILE, 0, "n {n} is not a multiple of {NTILE}");
    assert_eq!(k % KTILE, 0, "k {k} is not a multiple of {KTILE}");
    assert!(
        n / NTILE <= 65535,
        "{} n-tiles exceed the 65535 grid-y limit",
        n / NTILE
    );

    let out = client.empty(m_pad * n * core::mem::size_of::<f32>());
    let kernel = fp4_linear_swz_ptx_kernel(k, n, swz_sc);
    // SAFETY: the shape assertions above are the kernel's whole contract on
    // the grid; the buffer lengths are the caller's, as documented.
    unsafe {
        kernel.launch(
            client,
            CubeCount::Static((m_pad / MTILE) as u32, (n / NTILE) as u32, 1),
            RawArgs::new()
                .buffer(a)
                .buffer(a_sc)
                .buffer(b)
                .buffer(b_sc)
                .buffer(&out)
                .f32(scale),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------------
    // `cubecl-cpp`'s fragment maps, transcribed. These are the SOURCE of the
    // closed forms the emitter uses, so the tests below re-derive the cubecl
    // kernel's addresses from them and compare -- rather than restating the
    // emitter's own arithmetic, which would agree with itself by definition.
    // ----------------------------------------------------------------------

    /// `cubecl-cpp/src/cuda/processors.rs` `row_index`, for `MatrixIdent::A`
    /// and `MatrixIdent::B`, with `elems_per_reg = 8` (eight E2M1 codes in a
    /// 32-bit register).
    fn row_index(lane: usize, n: usize, ident: char) -> usize {
        const EPR: usize = 8;
        match ident {
            'a' => lane / 4 + ((n / EPR) & 1) * 8,
            'b' => (lane % 4) * EPR + (n % EPR) + EPR * 4 * (n / EPR),
            _ => lane / 4 + ((n << 2) & 8),
        }
    }

    /// `col_index`, same source.
    fn col_index(lane: usize, n: usize, ident: char) -> usize {
        const EPR: usize = 8;
        match ident {
            'a' => (lane % 4) * EPR + (n % EPR) + 4 * EPR * ((n / (2 * EPR)) & 1),
            'b' => lane / 4,
            _ => (lane % 4) * 2 + (n % 2),
        }
    }

    /// `cubecl-core`'s `MmaDefinition::scales_index`.
    fn scales_index(lane: usize, ident: char) -> usize {
        match ident {
            'a' => lane / 4 + (lane % 4 % 2) * 8,
            _ => lane / 4,
        }
    }

    /// Every operand address the emitter writes as `base + immediate`, against
    /// the same address re-derived from the cubecl kernel's index expressions
    /// and the fragment maps above.
    ///
    /// This is the gate that a machine with no GPU can hold: a wrong lane
    /// decomposition or a wrong immediate is a wrong ANSWER, not a crash, and
    /// nothing else here would catch it.
    #[test]
    fn every_operand_address_matches_the_cubecl_kernel() {
        for &(size_k, size_n) in &[(4096usize, 4096usize), (2048, 4096), (256, 8), (1024, 24)] {
            let k_tiles = size_k / KTILE;
            let spr = size_k / GROUP;
            for &swz_sc in &[true, false] {
                for m_tile in 0..3 {
                    for n_tile in 0..3 {
                        if n_tile * NTILE >= size_n {
                            continue;
                        }
                        let m_base = m_tile * MTILE;
                        let n_base = n_tile * NTILE;
                        for lane in 0..32usize {
                            let g = lane >> 2;
                            let tig = lane & 3;

                            // The prologue's bases, as the PTX computes them.
                            let pa = [
                                (m_base + g) * (size_k / 2) + tig * 4,
                                (m_base + g) * (size_k / 2) + tig * 4 + a_row_gap(size_k),
                            ];
                            let pb = n_tile * k_tiles * BLOCK_BYTES + lane * 4;
                            let psa = (m_base + g + (lane & 1) * 8) * spr;
                            let psb = if swz_sc {
                                n_tile * k_tiles * SCALE_BLOCK_BYTES + g * 4
                            } else {
                                (n_base + g) * spr
                            };

                            for t in 0..k_tiles {
                                let kbase = t * KTILE;

                                // A: `a[(gr * size_k / 2 + gc / 2) / 4]`, an
                                // index in 4-byte words, times 4 = the byte.
                                for i in 0..4 {
                                    // `position_of_nth(lane, i * vs_a * pack)`
                                    // with `vs_a * pack` = 8 codes.
                                    let row = row_index(lane, i * 8, 'a');
                                    let col = col_index(lane, i * 8, 'a');
                                    let want =
                                        ((m_base + row) * (size_k / 2) + (col + kbase) / 2) / 4 * 4;
                                    assert_eq!(
                                        pa[i & 1] + a_imm(t, i),
                                        want,
                                        "A k={size_k} n={size_n} t={t} i={i} lane={lane}"
                                    );
                                }

                                // B, in swizzled order: block
                                // `(n_tile * k_tiles + t)`, word `32*i + lane`.
                                for i in 0..2 {
                                    let row = row_index(lane, i * 8, 'b');
                                    let col = col_index(lane, i * 8, 'b');
                                    let w = row / 8;
                                    let blk = (n_tile * k_tiles + t) * BLOCK_BYTES;
                                    let off = (w / 4) * 32 + col * 4 + (w % 4);
                                    assert_eq!(
                                        pb + b_imm(t, i),
                                        blk + off * 4,
                                        "B k={size_k} t={t} i={i} lane={lane}"
                                    );
                                    // and the property that makes it coalesced
                                    assert_eq!(off, 32 * i + lane, "B word order");
                                }

                                // Scales: four contiguous bytes each, one
                                // 32-bit load, 4-byte aligned.
                                let sia = scales_index(lane, 'a');
                                assert_eq!(
                                    psa + a_scale_imm(t),
                                    (sia + m_base) * spr + t * 4,
                                    "A scales t={t} lane={lane}"
                                );
                                assert_eq!((psa + a_scale_imm(t)) % 4, 0);

                                let sib = scales_index(lane, 'b');
                                let want_b = if swz_sc {
                                    ((n_tile * k_tiles + t) * NTILE + sib) * 4
                                } else {
                                    (sib + n_base) * spr + t * 4
                                };
                                assert_eq!(
                                    psb + b_scale_imm(t, swz_sc),
                                    want_b,
                                    "B scales t={t} lane={lane} swz={swz_sc}"
                                );
                                assert_eq!((psb + b_scale_imm(t, swz_sc)) % 4, 0);
                            }

                            // The two output stores.
                            let pout = ((m_base + g) * size_n + n_base + 2 * tig) * 4;
                            for i in 0..2 {
                                let row = row_index(lane, i * 2, 'c');
                                let col = col_index(lane, i * 2, 'c');
                                let want = ((m_base + row) * size_n + n_base + col) / 2 * 8;
                                let got = pout + i * (MTILE / 2) * size_n * 4;
                                assert_eq!(got, want, "out i={i} lane={lane}");
                                assert_eq!(got % 8, 0, "v2.f32 store must be 8-byte aligned");
                            }
                        }
                    }
                }
            }
        }
    }

    /// The module's shape: it is PTX, it opens with the pinned header, it
    /// names its entry, its block size agrees with the launch, and the
    /// `info[N]` it declares is exactly the blob the launcher packs.
    #[test]
    fn the_module_is_ptx_and_declares_what_the_launch_binds() {
        let ptx = fp4_linear_swz_ptx(4096, 4096, true);
        assert!(super::super::rawptx::is_ptx(&ptx));
        assert!(ptx.contains(".visible .entry fp4_linear_swz_k4096_n4096_s("));
        assert!(ptx.contains(&format!(".maxntid {UNITS}, 1, 1")));
        assert_eq!(ptx.matches(".param .u64").count(), 5, "five buffers");

        let (_, rest) = ptx.split_once(".b8 info[").expect("info[N]");
        let declared: usize = rest.split_once(']').unwrap().0.parse().unwrap();
        assert_eq!(declared, RawArgs::new().f32(1.0).words().len() * 8);

        let k = fp4_linear_swz_ptx_kernel(4096, 4096, true);
        assert_eq!(k.name(), "fp4_linear_swz_k4096_n4096_s");
        assert_eq!(k.cube_dim(), CubeDim::new_1d(UNITS));
        assert_eq!(k.ptx(), ptx);
    }

    /// One MMA per k tile, in ascending order, each consuming the previous
    /// accumulator — the property that makes the accumulate order identical to
    /// the cubecl arm's, checked on the text because that is where it lives.
    #[test]
    fn the_k_loop_is_unrolled_once_per_tile() {
        for size_k in [64usize, 256, 2048, 4096] {
            let ptx = fp4_linear_swz_ptx(size_k, 4096, true);
            assert_eq!(ptx.matches(MMA).count(), size_k / KTILE);
            assert_eq!(
                ptx.matches("ld.global.nc.u32").count(),
                8 * (size_k / KTILE)
            );
            // No loop: nothing branches, so there is no induction variable to
            // get wrong and no k-tile address computed at runtime.
            assert!(!ptx.contains("bra"), "the K loop must be straight-line");
            assert!(!ptx.contains("setp"));
            // Every MMA reads and writes the one accumulator quad.
            assert_eq!(
                ptx.matches("{%fc0, %fc1, %fc2, %fc3}").count(),
                2 * (size_k / KTILE)
            );
        }
    }

    /// A specialisation is its own kernel: the id follows the text, so two
    /// shapes never share a module or a cache entry.
    #[test]
    fn each_shape_is_its_own_kernel() {
        let a = fp4_linear_swz_ptx_kernel(4096, 4096, true);
        let same = fp4_linear_swz_ptx_kernel(4096, 4096, true);
        let k2 = fp4_linear_swz_ptx_kernel(2048, 4096, true);
        let n2 = fp4_linear_swz_ptx_kernel(4096, 8192, true);
        let rowmajor = fp4_linear_swz_ptx_kernel(4096, 4096, false);
        assert_eq!(a.ptx(), same.ptx());
        assert_ne!(a.ptx(), k2.ptx());
        assert_ne!(a.ptx(), n2.ptx());
        assert_ne!(a.ptx(), rowmajor.ptx());
        assert_ne!(a.name(), rowmajor.name());
    }

    /// Shapes that do not tile are refused where the launcher would refuse
    /// them, not silently mis-addressed.
    #[test]
    #[should_panic(expected = "is not a multiple of 64")]
    fn a_k_that_does_not_tile_is_refused() {
        let _ = fp4_linear_swz_ptx(32, 4096, true);
    }
}
