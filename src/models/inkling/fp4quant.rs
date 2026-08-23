//! `fp4quant` — dynamic **NVFP4 activation quantization** as a CubeCL kernel.
//!
//! The weight side of Inkling arrives already quantized and only has to be
//! *decoded* ([`super::nvfp4`]). The activation side does not: `hf_quant_config`
//! asks for `"*input_quantizer": { num_bits: [2,1], block_sizes: {-1: 16,
//! type: "dynamic", scale_bits: [4,3]}, algorithm: "max" }`, i.e. the
//! activations must be quantized **at run time, per forward pass**, into the
//! same NVFP4 shape the weights are already in. That is this module.
//!
//! ## The recipe (one 16-element block along the contiguous axis)
//!
//! ```text
//! amax   = max |x_i|                      over the 16 elements
//! scale  = amax / 6.0                     6.0 = largest E2M1 magnitude
//! byte   = E4M3(scale)                    round-to-nearest-even
//! s      = decode(byte)                   the value actually stored
//! code_i = argmin_c |E2M1[c] - x_i / s|   round-to-nearest
//! ```
//!
//! There is **no second (`scale2`) level** here. The weights have one because
//! they are quantized offline against a whole tensor; a dynamic per-block
//! activation quantizer has only the block, so `scale` must land in E4M3's own
//! range (`2^-9 .. 448`) unaided. `scale` is clamped to 448 before the cast so
//! that overflow behaviour is this kernel's decision and not the conversion
//! instruction's — CubeCL emits `__nv_cvt_float_to_fp8(..., __NV_NOSAT, ...)`,
//! which would otherwise turn an overflowing scale into the E4M3 NaN pattern.
//!
//! ## Tie-breaking, stated rather than left to chance
//!
//! * **The E4M3 scale** rounds to nearest, ties to **even** significand — that
//!   is what `__nv_cvt_float_to_fp8` does, and `fp4quant_gate` reproduces it on
//!   the host by bracketing the 127 non-negative E4M3 patterns.
//! * **The E2M1 code** is chosen by seven `>=` comparisons against the
//!   midpoints between the eight representable magnitudes
//!   (`0.25 0.75 1.25 1.75 2.5 3.5 5.0`). `>=` means an exact tie rounds
//!   **away from zero** (up in magnitude). All seven midpoints are exactly
//!   representable in f32 *and* f64, so a host reference in either precision
//!   applies the identical test with no rounding step of its own.
//! * A negative input that rounds to magnitude zero keeps its sign and becomes
//!   code `0x8` (`-0.0`), not code `0x0`. `-0.0` itself is *not* negative under
//!   `< 0.0` and becomes code `0x0`.
//! * An all-zero block emits a zero scale byte and sixteen zero codes; the
//!   `s > 0` guard makes that explicit rather than relying on `0/0` producing a
//!   NaN that fails every comparison.
//!
//! ## Output layout
//!
//! `codes` is `[rows, k/8]` `u32`: word `w` holds elements `8w..8w+7`, element
//! `i` at bits `4*(i%8)` — lowest index in the **low** nibble, the ordering
//! [`super::nvfp4`] settled against `compressed_tensors`. On a little-endian
//! host that is byte-identical to the checkpoint's packed `e2m1x2` rows, so the
//! same buffer can be bound straight into the NVFP4 tensor-core MMA (see
//! `nvfp4_mma_probe`) without a repack. `scales` is `[rows, k/16]` `e4m3`, one
//! byte per block, in the same row-major order.
//!
//! Behind the `q4` feature, which is what pulls the `cubecl` dependency in
//! (`cuda-backend` enables it). Gated by `fp4quant_gate`.

use cubecl::client::ComputeClient;
use cubecl::e4m3;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Logical elements covered by one E4M3 block scale (NVFP4's `group_size`).
pub const GROUP: usize = 16;
/// Largest representable E2M1 magnitude — the divisor in `scale = amax / 6`.
pub const FP4_MAX: f32 = 6.0;
/// Largest finite E4M3 magnitude; `scale` is clamped here before the cast.
pub const E4M3_MAX: f32 = 448.0;
/// Elements per packed `u32` word (eight nibbles).
pub const CODES_PER_WORD: usize = 8;

/// One thread per 16-element block, this many threads per cube.
const CUBE_SIZE: u32 = 256;

/// Nearest E2M1 code to `v / s`, as a 4-bit value in a `u32`.
///
/// `s == 0` (an all-zero block) yields code 0 without dividing.
#[cube]
fn e2m1_code(v: f32, s: f32) -> u32 {
    let mut m = u32::new(0);
    if s > 0.0 {
        let t = v / s;
        let mut a = t;
        if t < 0.0 {
            a = -t;
        }
        // Seven midpoints, ascending; `>=` sends an exact tie away from zero.
        if a >= 0.25 {
            m = u32::new(1);
        }
        if a >= 0.75 {
            m = u32::new(2);
        }
        if a >= 1.25 {
            m = u32::new(3);
        }
        if a >= 1.75 {
            m = u32::new(4);
        }
        if a >= 2.5 {
            m = u32::new(5);
        }
        if a >= 3.5 {
            m = u32::new(6);
        }
        if a >= 5.0 {
            m = u32::new(7);
        }
        if t < 0.0 {
            m += u32::new(8);
        }
    }
    m
}

/// One thread per 16-element block: reduce `amax`, emit the E4M3 scale byte and
/// the block's two packed `u32` code words.
///
/// A block is exactly two output words, so no two threads ever touch the same
/// word — the pack needs no atomics and no shared memory.
#[cube(launch_unchecked)]
fn quantize_nvfp4_kernel<E: Scalar + Cast>(
    x: &Array<E>,
    codes: &mut Array<u32>,
    scales: &mut Array<e4m3>,
    blocks: usize,
) {
    let blk = ABSOLUTE_POS;
    if blk < blocks {
        let base = blk * 16;

        let mut amax = f32::new(0.0f32);
        #[unroll]
        for i in 0..16usize {
            let v = f32::cast_from(x[base + i]);
            let mut a = v;
            if v < 0.0 {
                a = -v;
            }
            if a > amax {
                amax = a;
            }
        }

        let mut sf = amax / 6.0;
        if sf > 448.0 {
            sf = f32::new(448.0f32);
        }
        let se = e4m3::cast_from(sf);
        // Round-trip through the stored byte: the codes must be chosen against
        // the scale the consumer will actually read back, not the f32 that was
        // rounded to produce it.
        let s = f32::cast_from(se);
        scales[blk] = se;

        let mut w0 = u32::new(0);
        #[unroll]
        for i in 0..8usize {
            w0 |= e2m1_code(f32::cast_from(x[base + i]), s) << (4 * i) as u32;
        }
        let mut w1 = u32::new(0);
        #[unroll]
        for i in 0..8usize {
            w1 |= e2m1_code(f32::cast_from(x[base + 8 + i]), s) << (4 * i) as u32;
        }
        codes[blk * 2] = w0;
        codes[blk * 2 + 1] = w1;
    }
}

/// Quantize `x` (`[rows, k]`, f32, row-major, contiguous) to NVFP4 on device.
///
/// Returns `(codes, scales)`: `codes` is `[rows, k/8]` `u32` (element `i` of
/// word `w` at bits `4*(i%8)`, low nibble = lowest index), `scales` is
/// `[rows, k/16]` [`e4m3`] bytes. Both buffers are freshly allocated.
///
/// `k` must be a positive multiple of 64 (every Inkling activation width is);
/// `rows` is unconstrained. Blocks never straddle a row because `k % 16 == 0`
/// follows from `k % 64 == 0`.
pub fn quantize_nvfp4<R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    rows: usize,
    k: usize,
) -> (Handle, Handle) {
    quantize_nvfp4_as::<f32, R>(client, x, rows, k)
}

/// The same, reading a BF16 activation.
///
/// The quantizer's own arithmetic does not move: every element is widened to
/// f32 on the way in and the block amax, the E4M3 scale and the seven midpoint
/// comparisons are the f32 ones they were. What changes is the buffer the
/// producer had to write, and on a prefill that is the whole point -- the
/// stacked `[k * n, hidden]` activation this reads is the largest thing a
/// routed-expert layer holds.
///
/// It costs nothing in accuracy that the lane was not already paying: the
/// destination is FOUR BITS with one E4M3 scale per sixteen elements, so the
/// question is whether BF16's eight mantissa bits can miss an E2M1 code the f32
/// would have hit, and the codes are seven comparisons against midpoints two
/// orders of magnitude coarser than that.
pub fn quantize_nvfp4_bf16<R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    rows: usize,
    k: usize,
) -> (Handle, Handle) {
    quantize_nvfp4_as::<half::bf16, R>(client, x, rows, k)
}

/// [`quantize_nvfp4`] at a named input element type.
fn quantize_nvfp4_as<E: Scalar + Cast, R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    rows: usize,
    k: usize,
) -> (Handle, Handle) {
    assert!(
        k > 0 && k % 64 == 0,
        "k must be a positive multiple of 64, got {k}"
    );
    assert!(rows > 0, "rows must be non-zero");

    let n = rows * k;
    let blocks = n / GROUP;
    let words = n / CODES_PER_WORD;

    let codes = client.empty(words * core::mem::size_of::<u32>());
    let scales = client.empty(blocks);

    let cubes = blocks.div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        quantize_nvfp4_kernel::launch_unchecked::<E, R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(x.clone(), n),
            ArrayArg::from_raw_parts(codes.clone(), words),
            ArrayArg::from_raw_parts(scales.clone(), blocks),
            blocks,
        );
    }

    (codes, scales)
}
