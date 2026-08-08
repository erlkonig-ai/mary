//! NVFP4 decode for Inkling's expert weights.
//!
//! This is the one place the K3 port does not carry over. K3 is MXFP4: one
//! E8M0 exponent per 32 elements, no second level. Inkling is NVFP4:
//!
//! * one **E4M3** scale per **16** logical elements (`group_size` in
//!   `hf_quant_config.json`),
//! * a per-expert **F32** `scale2` on top of that,
//! * an `input_amax` for activation calibration, which the weight decode does
//!   not use.
//!
//! Two 4-bit codes live in each byte, **low nibble first**. That ordering
//! cannot be established from a checkpoint: swapping the pair permutes each
//! block but leaves its min, max, mean and code multiset identical, so every
//! self-consistency check passes either way. It was settled against
//! `compressed_tensors.compressors.unpack_fp4_from_uint8`, and
//! `inkling_nvfp4_gate` re-checks it against that authority's output rather
//! than trusting this comment.

/// E2M1 values by 4-bit code: sign(1), exponent(2), mantissa(1).
pub const FP4_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Logical elements covered by one E4M3 block scale.
pub const GROUP: usize = 16;

/// Largest finite E4M3 magnitude, and the largest FP4 magnitude. Their product
/// with `scale2` bounds anything this decode can produce, which is what makes
/// the bound in the gate checkable rather than decorative.
pub const E4M3_MAX: f32 = 448.0;
pub const FP4_MAX: f32 = 6.0;

/// Decode one `float8_e4m3fn` byte.
///
/// `fn` is the finite variant: no infinities, and the only NaNs are `0x7F` and
/// `0xFF`. Exponent bias is 7; a zero exponent is subnormal.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (b >> 3) & 0x0F;
    let mant = b & 0x07;
    if exp == 0 {
        // Subnormal: mant * 2^-6 / 8 == mant * 2^-9. Reuse exp2i rather than a
        // hand-written bit pattern -- the literal here was 0x3600_0000 (2^-19)
        // and every one of the 14 subnormal patterns was wrong. The slice's 25
        // distinct scales contain no subnormals, so only the full 256-entry
        // domain check caught it.
        sign * (mant as f32) * exp2i(-9)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + (mant as f32) / 8.0) * exp2i(exp as i32 - 7)
    }
}

/// `2^n` for the small exponents E4M3 can hold, without a libm call.
fn exp2i(n: i32) -> f32 {
    // E4M3's exponent range after bias is -6..=8, so the bits are always valid.
    f32::from_bits((((n + 127) as u32) & 0xFF) << 23)
}

/// The two codes packed in one byte, in checkpoint order.
#[inline]
pub fn split_byte(b: u8) -> (u8, u8) {
    (b & 0x0F, (b >> 4) & 0x0F)
}

/// Decode one packed row into `out`.
///
/// `codes` holds `n` bytes, `scales` holds `n * 2 / GROUP` E4M3 bytes, and
/// `out` must be `n * 2` long. Returns the number of values written so a caller
/// can check it rather than assume it.
pub fn decode_row(codes: &[u8], scales: &[u8], scale2: f32, out: &mut [f32]) -> usize {
    let logical = codes.len() * 2;
    assert_eq!(out.len(), logical, "output must hold two values per code byte");
    assert_eq!(
        scales.len() * GROUP,
        logical,
        "expected one E4M3 scale per {GROUP} logical elements"
    );

    for (block, &s) in scales.iter().enumerate() {
        // Apply the block scale BEFORE scale2, matching the reference. Folding
        // the two scales together first is one multiply cheaper and disagrees
        // in the last bit on 7% of values, because float multiply does not
        // associate.
        let block_scale = e4m3_to_f32(s);
        let lo = block * GROUP;
        for i in 0..GROUP / 2 {
            let byte = codes[lo / 2 + i];
            let (first, second) = split_byte(byte);
            out[lo + 2 * i] = FP4_E2M1[first as usize] * block_scale * scale2;
            out[lo + 2 * i + 1] = FP4_E2M1[second as usize] * block_scale * scale2;
        }
    }
    logical
}

/// Decode a whole stacked expert matrix.
///
/// `codes` is `[experts, rows, bytes]` flattened, `scales` is
/// `[experts, rows, bytes * 2 / GROUP]`, `scale2` is one per expert. Returns
/// how many values were written.
pub fn decode_stacked(
    codes: &[u8],
    scales: &[u8],
    scale2: &[f32],
    experts: usize,
    rows: usize,
    bytes_per_row: usize,
    out: &mut [f32],
) -> usize {
    let logical = bytes_per_row * 2;
    let scales_per_row = logical / GROUP;
    assert_eq!(codes.len(), experts * rows * bytes_per_row);
    assert_eq!(scales.len(), experts * rows * scales_per_row);
    assert_eq!(scale2.len(), experts);
    assert_eq!(out.len(), experts * rows * logical);

    let mut written = 0;
    for e in 0..experts {
        for r in 0..rows {
            let ci = (e * rows + r) * bytes_per_row;
            let si = (e * rows + r) * scales_per_row;
            let oi = (e * rows + r) * logical;
            written += decode_row(
                &codes[ci..ci + bytes_per_row],
                &scales[si..si + scales_per_row],
                scale2[e],
                &mut out[oi..oi + logical],
            );
        }
    }
    written
}
