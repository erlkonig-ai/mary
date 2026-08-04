//! MXFP4 (OCP microscaling FP4) decode + a **bit-exact** MXFP4 → NVFP4
//! transcode — the bridge from a 4-bit checkpoint stored in the microscaling
//! format to the NVFP4 tensor-core path.
//!
//! Why this exists: Kimi-K3 ships 92.67% of its 2.78 T parameters as MXFP4
//! (packed E2M1 nibbles + one E8M0 exponent per 32 weights, `4.25 bits/param`),
//! but the tensor cores this project can already drive exactly (sm_121, via
//! CubeCL) speak **NVFP4** — the same E2M1 codes with a per-**16** E4M3 block
//! scale and one f32 per-tensor global scale. Dequantizing to f16 and
//! requantizing would throw away the one property that makes the checkpoint
//! usable at all: that the stored values are already the model. This module
//! instead performs a **relabelling**. The 4-bit codes are never touched (the
//! `packed` slice is borrowed, not copied), and the scales are re-expressed —
//! not re-derived — so every dequantized weight comes out bit-identical.
//!
//! ## Why the relabelling is exact
//!
//! An E8M0 scale is *exactly* a power of two (`2^(byte-127)`, no mantissa at
//! all). E4M3FN represents exact powers of two over `2^-9 … 2^8` — **18
//! octaves** (`2^-9,2^-8,2^-7` as subnormals, `2^-6 … 2^8` as mantissa-zero
//! normals). So for a tensor whose E8M0 exponents span `e_min..e_max`, picking
//! `global = 2^(e_max - 8)` puts every block scale at `2^(e - e_max + 8)`,
//! inside the window whenever `e_max - e_min <= 17`. Splitting a 32-block into
//! two 16-blocks that inherit the same scale is free — NVFP4 uses the same
//! E2M1 code table and the same low-nibble-first packing, so no element moves.
//! The dequantized product `code · 2^a · 2^b` is exact in f32 because `code`
//! carries at most 3 significant bits and both factors are powers of two.
//!
//! Measured on this checkpoint (`k3oracle`, 9 real expert tensors, 9.3 M
//! blocks): worst single-tensor span **11 octaves**, union across three
//! experts 12 — comfortably inside 18. `mxfp4_gate` is the proof, at full
//! scale, against the oracle's sha256 of the complete decode.
//!
//! ## What this module does NOT claim
//!
//! - Nothing here says an arbitrary NVFP4 pipeline is lossless. The exactness
//!   depends on leaving the global scale a **power of two**; the usual
//!   `amax/(6·448)` recipe gives a non-power-of-two and the block scales stop
//!   being exactly representable. [`Nvfp4::global_scale`] is always `2^g`.
//! - The block-scale array is produced in **logical row-major** order. A
//!   tensor-core kernel generally wants a swizzled scale-factor layout; that
//!   permutation is the consumer's business and is not applied here.
//! - Three experts out of 82,432 were measured. [`transcode_to_nvfp4`] checks
//!   the span of the tensor in front of it and returns an error rather than
//!   assuming, so a wider expert fails loudly instead of quietly rounding.

use half::f16;

/// Weights per E8M0 scale in MXFP4 (OCP `MXFP4` group size).
pub const MX_BLOCK: usize = 32;
/// Weights per E4M3 scale in NVFP4 — exactly half an MXFP4 block.
pub const NV_BLOCK: usize = 16;

/// The 16 E2M1 codes. Bit layout MSB..LSB is `s | e1 e0 | m0` with exponent
/// bias 1: `e == 0` is subnormal `(-1)^s · m · 0.5`, `e >= 1` is normal
/// `(-1)^s · 2^(e-1) · (1 + m/2)`. E2M1 has **no Inf and no NaN** — `0xF` is
/// −6.0 — and two zeros, both of which the quantizer emits, so the sign of
/// zero has to survive the decode (`E2M1[8]` is −0.0, not +0.0).
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Smallest exponent `e` with `2^e` exactly representable in E4M3FN
/// (subnormal, mantissa 1).
pub const E4M3_POW2_MIN: i32 = -9;
/// Largest exponent `e` with `2^e` exactly representable in E4M3FN
/// (`0x78` = 256; the max *finite* E4M3 value is 448, but that is not a power
/// of two).
pub const E4M3_POW2_MAX: i32 = 8;
/// Below this exponent the E4M3 encoding of `2^e` is subnormal. Still exact,
/// but hardware that flushes subnormal scale operands to zero would silently
/// zero whole blocks — [`Nvfp4::subnormal_block_scales`] flags it.
pub const E4M3_POW2_MIN_NORMAL: i32 = -6;

/// `2^e` as an f32, or `None` outside f32's normal exponent range.
fn pow2_f32(e: i32) -> Option<f32> {
    if !(-126..=127).contains(&e) {
        return None;
    }
    Some(f32::from_bits(((e + 127) as u32) << 23))
}

/// Decode one E8M0 scale byte. E8M0 is precisely f32's exponent field — no
/// sign, no mantissa, bias 127 — so `byte << 23` *is* the answer for every
/// normal case. `0x00` is `2^-127`, one octave below f32's smallest normal
/// (an exact subnormal); `0xFF` is NaN by definition of the format.
pub fn e8m0_to_f32(byte: u8) -> f32 {
    match byte {
        0xFF => f32::NAN,
        0x00 => f32::from_bits(1 << 22), // 2^-127 = 2^22 · 2^-149
        _ => f32::from_bits((byte as u32) << 23),
    }
}

/// Decode one E4M3FN byte (1 sign, 4 exponent bits biased by 7, 3 mantissa
/// bits; `0x7F`/`0xFF` are the only NaNs and there is no Inf).
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((byte >> 3) & 0x0F) as i32;
    let m = byte & 0x07;
    if e == 0x0F && m == 7 {
        f32::NAN
    } else if e == 0 {
        sign * (m as f32) * (1.0 / 512.0) // subnormal: m · 2^-9
    } else {
        sign * (1.0 + m as f32 * 0.125) * pow2_f32(e - 7).expect("E4M3 exponent is in f32 range")
    }
}

/// Encode `2^e` as an E4M3FN byte, or `None` when `2^e` is not an E4M3 value.
///
/// Deliberately narrow: the transcode only ever needs to re-express exact
/// powers of two, so there is no general (rounding) f32 → E4M3 path here to be
/// wrong in an untested corner.
pub fn e4m3_from_pow2(e: i32) -> Option<u8> {
    match e {
        -9 => Some(0x01), // subnormal, mantissa 1
        -8 => Some(0x02), // subnormal, mantissa 2
        -7 => Some(0x04), // subnormal, mantissa 4
        E4M3_POW2_MIN_NORMAL..=E4M3_POW2_MAX => Some(((e + 7) as u8) << 3),
        _ => None,
    }
}

/// Why a tensor could not be transcoded. Every variant is a refusal, never a
/// fallback to an approximate encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeError {
    /// An E8M0 scale byte was `0xFF` (NaN). Nothing sensible to relabel it to.
    ScaleNan { index: usize },
    /// The tensor's exponents span more than E4M3's 18-octave power-of-two
    /// window, so no single power-of-two global scale makes every block scale
    /// exact. Requantizing would be the only way out — this module will not.
    ExponentSpan { e_min: i32, e_max: i32, octaves: usize },
    /// `2^(e_max - 8)` is outside f32's normal range; the global scale itself
    /// could not be held exactly.
    GlobalScaleRange { g: i32 },
}

impl std::fmt::Display for TranscodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScaleNan { index } => write!(f, "E8M0 scale[{index}] is NaN (0xFF)"),
            Self::ExponentSpan { e_min, e_max, octaves } => write!(
                f,
                "E8M0 exponents span {octaves} octaves ({e_min}..{e_max}); E4M3 holds {} exactly",
                E4M3_POW2_MAX - E4M3_POW2_MIN + 1
            ),
            Self::GlobalScaleRange { g } => write!(f, "global scale 2^{g} is not a normal f32"),
        }
    }
}

impl std::error::Error for TranscodeError {}

/// The NVFP4 side of a transcode.
///
/// Dequantization is `w = E2M1[code] · e4m3_to_f32(block_scale) ·
/// global_scale`. Consumers whose convention divides by the global scale want
/// `1.0 / global_scale`, which — being a power of two — is also exact.
#[derive(Debug)]
pub struct Nvfp4<'a> {
    /// The original packed nibbles, **borrowed unchanged**. NVFP4 and MXFP4
    /// agree on the E2M1 table and on low-nibble-first packing, so a transcode
    /// rewrites no weight bytes at all; this field is the proof in the type.
    pub packed: &'a [u8],
    /// E4M3FN block scales, `[rows, cols/16]` row-major. Each MXFP4 scale
    /// appears twice, for the two halves of its 32-element block.
    pub block_scale: Vec<u8>,
    /// The per-tensor global scale, always `2^g`.
    pub global_scale: f32,
    /// `g` itself, so a caller can reason in exponents without a log.
    pub global_exp: i32,
    /// True when some block scale landed in E4M3's subnormal range. Exact, but
    /// a warning sign for kernels that flush subnormal scales.
    pub subnormal_block_scales: bool,
    pub rows: usize,
    pub cols: usize,
}

/// The exponent budget of an MXFP4 scale plane: `(e_min, e_max)` unbiased.
///
/// This is the whole question of whether an exact transcode exists, isolated
/// so it can be swept over a checkpoint without decoding a single weight.
pub fn scale_exponent_range(scale: &[u8]) -> Result<(i32, i32), TranscodeError> {
    assert!(!scale.is_empty(), "empty scale plane");
    let mut e_min = i32::MAX;
    let mut e_max = i32::MIN;
    for (index, &b) in scale.iter().enumerate() {
        if b == 0xFF {
            return Err(TranscodeError::ScaleNan { index });
        }
        let e = b as i32 - 127;
        e_min = e_min.min(e);
        e_max = e_max.max(e);
    }
    Ok((e_min, e_max))
}

/// Decode MXFP4 to f32. `cols` is the **logical** element count per row, so
/// `packed` is `rows · cols/2` bytes and `scale` is `rows · cols/32` bytes,
/// both row-major.
pub fn decode_mxfp4(packed: &[u8], scale: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(cols % MX_BLOCK, 0, "cols {cols} must be a multiple of {MX_BLOCK}");
    assert_eq!(packed.len(), rows * cols / 2, "packed plane is not [{rows}, {cols}/2]");
    assert_eq!(scale.len(), rows * cols / MX_BLOCK, "scale plane is not [{rows}, {cols}/{MX_BLOCK}]");
    let blocks_per_row = cols / MX_BLOCK;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            // Multiplying by a positive power of two carries the sign of zero
            // through, which is what keeps code 0x8 decoding to -0.0.
            let s = e8m0_to_f32(scale[r * blocks_per_row + b]);
            let pbase = r * (cols / 2) + b * (MX_BLOCK / 2);
            let obase = r * cols + b * MX_BLOCK;
            for k in 0..MX_BLOCK / 2 {
                let byte = packed[pbase + k];
                out[obase + 2 * k] = E2M1[(byte & 0x0F) as usize] * s;
                out[obase + 2 * k + 1] = E2M1[(byte >> 4) as usize] * s;
            }
        }
    }
    out
}

/// Decode MXFP4 straight to f16.
///
/// Exact for this checkpoint (every stored value is `c · 2^e` with `c` at most
/// 3 significant bits, well inside f16's range) but **not exact in general**:
/// an E8M0 exponent below −24 or above 15 would round or flush. Use
/// [`decode_mxfp4`] when exactness is the point.
pub fn decode_mxfp4_f16(packed: &[u8], scale: &[u8], rows: usize, cols: usize) -> Vec<f16> {
    decode_mxfp4(packed, scale, rows, cols).into_iter().map(f16::from_f32).collect()
}

/// Relabel an MXFP4 tensor as NVFP4 without touching a weight byte.
///
/// Picks `g = e_max - 8`, seating the tensor's largest block scale at E4M3's
/// largest exact power of two and giving the *small* end all the remaining
/// headroom (which is where the span actually lives — quantizer scales cluster
/// at the top and tail downward). Errors rather than approximating.
pub fn transcode_to_nvfp4<'a>(
    packed: &'a [u8],
    scale: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Nvfp4<'a>, TranscodeError> {
    assert_eq!(cols % MX_BLOCK, 0, "cols {cols} must be a multiple of {MX_BLOCK}");
    assert_eq!(packed.len(), rows * cols / 2, "packed plane is not [{rows}, {cols}/2]");
    assert_eq!(scale.len(), rows * cols / MX_BLOCK, "scale plane is not [{rows}, {cols}/{MX_BLOCK}]");

    let (e_min, e_max) = scale_exponent_range(scale)?;
    let g = e_max - E4M3_POW2_MAX;
    if e_min - g < E4M3_POW2_MIN {
        return Err(TranscodeError::ExponentSpan {
            e_min,
            e_max,
            octaves: (e_max - e_min + 1) as usize,
        });
    }
    let global_scale = pow2_f32(g).ok_or(TranscodeError::GlobalScaleRange { g })?;

    // One MXFP4 scale -> two NVFP4 scales, adjacent along the row. The halves
    // inherit the same value, so no block is renormalized and no code moves.
    let mut block_scale = vec![0u8; rows * cols / NV_BLOCK];
    let mut subnormal_block_scales = false;
    for (i, &b) in scale.iter().enumerate() {
        let e = b as i32 - 127 - g;
        subnormal_block_scales |= e < E4M3_POW2_MIN_NORMAL;
        let byte = e4m3_from_pow2(e).expect("span was checked against the E4M3 window");
        block_scale[2 * i] = byte;
        block_scale[2 * i + 1] = byte;
    }

    Ok(Nvfp4 {
        packed,
        block_scale,
        global_scale,
        global_exp: g,
        subnormal_block_scales,
        rows,
        cols,
    })
}

/// Decode an NVFP4 tensor to f32 — the inverse of [`transcode_to_nvfp4`], and
/// the thing that has to come out bit-identical to [`decode_mxfp4`].
pub fn decode_nvfp4(nv: &Nvfp4<'_>) -> Vec<f32> {
    let blocks_per_row = nv.cols / NV_BLOCK;
    let mut out = vec![0f32; nv.rows * nv.cols];
    for r in 0..nv.rows {
        for b in 0..blocks_per_row {
            // Both factors are powers of two and the code has <= 3 significant
            // bits, so the two multiplies are exact in f32 (no rounding is
            // even possible short of over/underflow).
            let s = e4m3_to_f32(nv.block_scale[r * blocks_per_row + b]) * nv.global_scale;
            let pbase = r * (nv.cols / 2) + b * (NV_BLOCK / 2);
            let obase = r * nv.cols + b * NV_BLOCK;
            for k in 0..NV_BLOCK / 2 {
                let byte = nv.packed[pbase + k];
                out[obase + 2 * k] = E2M1[(byte & 0x0F) as usize] * s;
                out[obase + 2 * k + 1] = E2M1[(byte >> 4) as usize] * s;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// tests (pure CPU: the two code tables from their bit fields, and a synthetic
// transcode round-trip; `mxfp4_gate` is the real-checkpoint gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_table_is_its_bit_fields() {
        for code in 0u8..16 {
            let s = if code & 8 != 0 { -1.0f32 } else { 1.0 };
            let e = ((code >> 1) & 3) as i32;
            let m = (code & 1) as f32;
            let v = if e == 0 { s * m * 0.5 } else { s * (1.0 + m * 0.5) * pow2_f32(e - 1).unwrap() };
            // to_bits, not ==, so the two zeros stay distinguishable.
            assert_eq!(E2M1[code as usize].to_bits(), v.to_bits(), "code {code:#x}");
        }
    }

    #[test]
    fn e8m0_is_the_f32_exponent_field() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(122), 0.03125); // 2^-5, the checkpoint's usual scale
        assert_eq!(e8m0_to_f32(112), 3.0517578125e-05); // 2^-15
        // 0x01 is 2^-126 = f32's smallest normal, so 0x00 is one exact halving
        // below it — the subnormal branch, checked without a decimal literal.
        assert_eq!(e8m0_to_f32(1), f32::MIN_POSITIVE);
        assert_eq!(e8m0_to_f32(0) * 2.0, f32::MIN_POSITIVE);
        assert_eq!(e8m0_to_f32(254), pow2_f32(127).unwrap());
        assert!(e8m0_to_f32(0xFF).is_nan());
    }

    #[test]
    fn e4m3_pow2_window_is_18_octaves() {
        let exact: Vec<i32> = (-30..=30)
            .filter(|&e| {
                e4m3_from_pow2(e).is_some_and(|b| e4m3_to_f32(b) == pow2_f32(e).unwrap())
            })
            .collect();
        assert_eq!(exact.first(), Some(&E4M3_POW2_MIN));
        assert_eq!(exact.last(), Some(&E4M3_POW2_MAX));
        assert_eq!(exact.len(), 18);
    }

    #[test]
    fn e4m3_known_encodings() {
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x78), 256.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // max finite, not a power of two
        assert_eq!(e4m3_to_f32(0x01), 1.0 / 512.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert!(e4m3_to_f32(0xFF).is_nan());
    }

    /// One 32-element block per row, scales walking the full 18-octave window:
    /// the transcode has to stay bit-exact right up to the edge and refuse the
    /// step past it.
    fn synthetic(exps: &[i32]) -> (Vec<u8>, Vec<u8>) {
        let packed: Vec<u8> = (0..exps.len() * MX_BLOCK / 2).map(|i| (i % 256) as u8).collect();
        let scale: Vec<u8> = exps.iter().map(|&e| (e + 127) as u8).collect();
        (packed, scale)
    }

    #[test]
    fn transcode_is_bit_exact_across_the_whole_window() {
        let exps: Vec<i32> = (-9..=8).collect(); // 18 octaves, the maximum
        let (packed, scale) = synthetic(&exps);
        let (rows, cols) = (exps.len(), MX_BLOCK);
        let mx = decode_mxfp4(&packed, &scale, rows, cols);
        let nv = transcode_to_nvfp4(&packed, &scale, rows, cols).unwrap();
        assert_eq!(nv.global_exp, 0);
        assert!(nv.subnormal_block_scales);
        assert_eq!(nv.packed.as_ptr(), packed.as_ptr(), "nibbles must be borrowed, not rebuilt");
        let back = decode_nvfp4(&nv);
        for (i, (a, b)) in mx.iter().zip(&back).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} vs {b}");
        }
    }

    #[test]
    fn transcode_refuses_a_19_octave_span() {
        let exps: Vec<i32> = (-9..=9).collect();
        let (packed, scale) = synthetic(&exps);
        let err = transcode_to_nvfp4(&packed, &scale, exps.len(), MX_BLOCK).unwrap_err();
        assert_eq!(err, TranscodeError::ExponentSpan { e_min: -9, e_max: 9, octaves: 19 });
    }

    #[test]
    fn a_kimi_shaped_span_stays_in_e4m3_normals() {
        // L01_E000 w1's measured range: bytes 112..122 = exponents -15..-5.
        let exps: Vec<i32> = (-15..=-5).collect();
        let (packed, scale) = synthetic(&exps);
        let nv = transcode_to_nvfp4(&packed, &scale, exps.len(), MX_BLOCK).unwrap();
        assert_eq!(nv.global_exp, -13);
        assert!(!nv.subnormal_block_scales);
    }

    #[test]
    fn negative_zero_survives_the_decode() {
        let packed = vec![0x80u8; MX_BLOCK / 2]; // low nibble 0x0 (+0), high 0x8 (-0)
        let scale = vec![122u8];
        let out = decode_mxfp4(&packed, &scale, 1, MX_BLOCK);
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[1].to_bits(), (-0.0f32).to_bits());
    }
}
