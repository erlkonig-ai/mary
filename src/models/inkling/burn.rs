//! Inkling's Burn lane — the same arithmetic as the f32 slice lane, on a backend.
//!
//! Mirrors how `k3` keeps `kda.rs` beside its Burn path: the slice lane in
//! [`crate::models::inkling::mlp`] is the reference, gated against
//! `transformers`, and this is checked against *it*. So the Burn lane gets a
//! real oracle without needing torch in the loop.
//!
//! Scope is the cost-dominant part first. A 5-token forward decoded 929 expert
//! slabs; each one is a `[2 * intermediate, hidden]` and a
//! `[hidden, intermediate]` matmul, which is where the time goes. RMSNorm is
//! here too because every block runs two of them.
//!
//! Everything stays f32. The slice lane is f32 and the checkpoint's dense
//! weights are BF16 widened to f32, so a rounding policy like K3's `ActRound`
//! would be describing a lane that does not exist yet; when a bf16 lane is
//! added it should get that treatment explicitly rather than by default.

use burn::prelude::*;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

/// `x * sigmoid(x)`, elementwise.
pub fn silu<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let s = burn::tensor::activation::sigmoid(x.clone());
    x * s
}

/// `nn.Linear(bias=False)`: `x @ Wᵀ` for a `[out, in]` weight.
///
/// The weight keeps its checkpoint orientation, so a transposition mistake is
/// a shape error rather than a plausible wrong answer.
pub fn linear<B: Backend>(x: Tensor<B, 2>, w: Tensor<B, 2>) -> Tensor<B, 2> {
    let [_, k] = x.dims();
    let [_, kw] = w.dims();
    assert_eq!(k, kw, "linear: x is [_, {k}] but the weight is [_, {kw}]");
    x.matmul(w.transpose())
}

/// RMS normalization with a per-feature gain.
///
/// Divides by `sqrt(var + eps)` rather than multiplying by its reciprocal: on
/// some backends `recip` dispatches to an approximate SIMD reciprocal, which
/// cost K3 about fourteen bits of accuracy before it was caught. Same hazard
/// here, same avoidance.
pub fn rms_norm<B: Backend>(x: Tensor<B, 2>, gain: Tensor<B, 1>, eps: f64) -> Tensor<B, 2> {
    let [_, w] = x.dims();
    assert_eq!(gain.dims()[0], w, "rms_norm: gain is {} wide, input {w}", gain.dims()[0]);
    let mean_sq = x.clone().powf_scalar(2.0).mean_dim(1);
    let denom = mean_sq.add_scalar(eps).sqrt();
    let normed = x / denom;
    normed * gain.unsqueeze::<2>()
}

/// One expert's feed-forward: `down(silu(gate) * up)`.
///
/// `gate_up` is `[2 * intermediate, hidden]` with the gate rows FIRST — the
/// checkpoint stores them interleaved and
/// [`crate::models::inkling::load::deinterleave_fused`] puts them in this order
/// at load. Passing a raw checkpoint tensor here is shape-identical and wrong,
/// which is exactly the bug that made the whole model emit noise while every
/// parity gate passed.
pub fn expert_ffn<B: Backend>(
    x: Tensor<B, 2>,
    gate_up: Tensor<B, 2>,
    down: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [two_inter, _] = gate_up.dims();
    assert!(two_inter % 2 == 0, "gate_up must have an even row count");
    let inter = two_inter / 2;
    let both = linear(x, gate_up);
    let [rows, _] = both.dims();
    let gate = both.clone().slice([0..rows, 0..inter]);
    let up = both.slice([0..rows, inter..2 * inter]);
    linear(silu(gate) * up, down)
}

/// The dense MLP: `down(silu(gate(x)) * up(x)) * global_scale`.
pub fn dense_mlp<B: Backend>(
    x: Tensor<B, 2>,
    gate: Tensor<B, 2>,
    up: Tensor<B, 2>,
    down: Tensor<B, 2>,
    global_scale: f32,
) -> Tensor<B, 2> {
    let g = linear(x.clone(), gate);
    let u = linear(x, up);
    linear(silu(g) * u, down).mul_scalar(global_scale)
}


/// The SHARED experts, on the device, from weights that are ALREADY on it.
///
/// The host lane in [`crate::models::inkling::mlp::shared_experts`] is the
/// reference and this has to agree with it. Two details are easy to lose in
/// translation and each one changes every token:
///
/// * the gamma multiplies the ACTIVATION — between the SwiGLU and the down
///   projection — not the block's output. Applying it after `down` is a
///   different function whenever `down` is not the identity, which is always.
/// * gammas arrive token-major, `[token, shared]`, so shared expert `s` takes a
///   stride-`n_shared` column and not a contiguous block. A contiguous read is
///   correct for `n_shared == 1` and silently wrong for 2, which is what this
///   checkpoint has.
///
/// `gate` and `up` are `[inter, hidden]` per shared expert and `down` is
/// `[hidden, inter]` — one tensor each rather than a stacked rank 3, because
/// these are uploaded once for the whole run and never sliced again. That is
/// the entire point: the caller holds the handles, and a token costs a matmul
/// against memory the device already owns rather than a fresh upload.
pub fn shared_experts_dev<B: Backend>(
    x: Tensor<B, 2>,
    gate: &[Tensor<B, 2>],
    up: &[Tensor<B, 2>],
    down: &[Tensor<B, 2>],
    gammas: &[f32],
    n_shared: usize,
) -> Tensor<B, 2> {
    assert_eq!(gate.len(), n_shared, "{} gate weights for {n_shared} shared experts", gate.len());
    assert_eq!(up.len(), n_shared, "{} up weights for {n_shared} shared experts", up.len());
    assert_eq!(down.len(), n_shared, "{} down weights for {n_shared} shared experts", down.len());
    let [n, _] = x.dims();
    assert_eq!(gammas.len(), n * n_shared, "{} gammas for {n} tokens", gammas.len());

    let dev = x.device();
    let mut out: Option<Tensor<B, 2>> = None;
    for s in 0..n_shared {
        let g = linear(x.clone(), gate[s].clone());
        let u = linear(x.clone(), up[s].clone());
        let col: Vec<f32> = (0..n).map(|t| gammas[t * n_shared + s]).collect();
        let gam = Tensor::<B, 2>::from_data(TensorData::new(col, [n, 1]), &dev);
        let c = linear(silu(g) * u * gam, down[s].clone());
        out = Some(match out {
            Some(o) => o + c,
            None => c,
        });
    }
    out.expect("a MoE layer with no shared experts")
}

/// FP4 (E2M1) values by 4-bit code, and the E4M3 table, as device tensors.
///
/// Built on the host from the scalar decoders in
/// [`crate::models::inkling::nvfp4`], which are gated bit-exactly against
/// `compressed_tensors` and against torch over all 256 E4M3 patterns. A gather
/// through those tables cannot drift from the CPU lane; a reimplemented
/// bit-twiddle could.
fn luts<B: Backend>(dev: &B::Device) -> (Tensor<B, 1>, Tensor<B, 1>) {
    use crate::models::inkling::nvfp4::{e4m3_to_f32, FP4_E2M1};
    let fp4 = Tensor::from_data(TensorData::new(FP4_E2M1.to_vec(), [16]), dev);
    // NaN would poison a gather, and only 0x7F/0xFF are NaN in E4M3-fn; they
    // never appear as a block scale, so map them to zero rather than carrying
    // NaN into every product in the row.
    let e4m3: Vec<f32> = (0..256u16)
        .map(|b| {
            let v = e4m3_to_f32(b as u8);
            if v.is_nan() { 0.0 } else { v }
        })
        .collect();
    let e4m3 = Tensor::from_data(TensorData::new(e4m3, [256]), dev);
    (fp4, e4m3)
}

/// Look up `idx` in a 1-D table, preserving the index tensor's shape.
fn gather2<B: Backend>(table: Tensor<B, 1>, idx: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    let [r, c] = idx.dims();
    table.select(0, idx.reshape([r * c])).reshape([r, c])
}

/// Dequantise NVFP4 on device.
///
/// `codes` is `[rows, bytes]` holding the packed byte values, `scales` is
/// `[rows, bytes * 2 / GROUP]` holding raw E4M3 byte values, and `scale2` is one
/// factor per row. Returns `[rows, bytes * 2]`.
///
/// Nibble order is low-first, settled against
/// `compressed_tensors.compressors.unpack_fp4_from_uint8`; the association is
/// `(fp4 * block_scale) * scale2`, matching the reference, because float
/// multiplication does not associate and the CPU lane was gated on that order.
pub fn dequant_nvfp4<B: Backend>(
    codes: Tensor<B, 2, Int>,
    scales: Tensor<B, 2, Int>,
    scale2: Tensor<B, 1>,
) -> Tensor<B, 2> {
    use crate::models::inkling::nvfp4::GROUP;
    let dev = codes.device();
    let (fp4_lut, e4m3_lut) = luts::<B>(&dev);
    let [rows, bytes] = codes.dims();
    let logical = bytes * 2;

    // Two 4-bit codes per byte, low nibble FIRST.
    let lo = codes.clone().bitwise_and_scalar(0x0Fi32.elem());
    let hi = codes
        .bitwise_right_shift_scalar(4i32.elem())
        .bitwise_and_scalar(0x0Fi32.elem());
    // Interleave: [rows, bytes, 2] -> [rows, 2 * bytes] gives lo, hi, lo, hi...
    let pairs = Tensor::cat(
        vec![lo.reshape([rows, bytes, 1]), hi.reshape([rows, bytes, 1])],
        2,
    )
    .reshape([rows, logical]);
    let vals = gather2(fp4_lut, pairs);

    // One E4M3 scale per GROUP logical elements, widened to match.
    let n_scales = logical / GROUP;
    let s = gather2(e4m3_lut, scales)
        .reshape([rows, n_scales, 1])
        .repeat_dim(2, GROUP)
        .reshape([rows, logical]);

    // Block scale first, then the per-row factor -- the reference's order.
    vals.mul(s).mul(scale2.reshape([rows, 1]).repeat_dim(1, logical))
}


/// Reorder a fused `[2 * intermediate, hidden]` from the checkpoint's
/// interleave to gate-rows-first, on device.
///
/// The checkpoint stores gate and up **alternating by row**: gate is the even
/// output rows, up the odd ones. It does NOT store them as contiguous halves.
/// Splitting down the middle loads without complaint and scrambles every
/// SwiGLU in every layer -- shape-identical, catastrophically wrong, and
/// invisible to any check that compares two lanes which share the assumption.
/// Authority is `transformers`' `conversion_mapping.py`, key `inkling_mm_model`.
///
/// This is the device twin of
/// [`crate::models::inkling::load::deinterleave_fused`]; the two must agree.
pub fn deinterleave_rows_device<B: Backend>(fused: Tensor<B, 2>) -> Tensor<B, 2> {
    let [rows, _] = fused.dims();
    assert!(rows % 2 == 0, "fused row count {rows} is odd; gate/up cannot interleave");
    let half = rows / 2;
    let mut order: Vec<i32> = Vec::with_capacity(rows);
    order.extend((0..half).map(|r| (2 * r) as i32)); // gate: even rows
    order.extend((0..half).map(|r| (2 * r + 1) as i32)); // up: odd rows
    let dev = fused.device();
    let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(order, [rows]), &dev);
    fused.select(0, idx)
}

/// Upload one expert's packed NVFP4 bytes and dequantise them on the device.
///
/// Takes exactly what [`crate::models::inkling::load::Checkpoint::expert_slice_packed`]
/// returns, so the host never materialises the f32 weight. Returns
/// `[rows, cols * 2]`.
///
/// `scale2` is one factor for the whole expert; it is broadcast to every row
/// because [`dequant_nvfp4`] takes a per-row vector, which is the shape the
/// stacked layout would need if scale2 ever became per-row.
pub fn expert_weight_from_packed<B: Backend>(
    codes: &[u8],
    scales: &[u8],
    scale2: f32,
    rows: usize,
    cols: usize,
    dev: &B::Device,
) -> Tensor<B, 2> {
    assert_eq!(codes.len(), rows * cols, "codes is {} bytes, want {rows}x{cols}", codes.len());
    assert_eq!(scales.len() % rows, 0, "{} scales do not divide {rows} rows", scales.len());
    let n_scales = scales.len() / rows;

    // Bitcast four bytes into a word rather than widening each to an i32:
    // same bytes, a quarter of the elements, and no host-side expansion of the
    // very data the packed path exists to keep packed.
    assert_eq!(cols % 4, 0, "{cols} bytes per row does not pack into i32 words");
    assert_eq!(n_scales % 4, 0, "{n_scales} scales per row does not pack into i32 words");
    let word = |b: &[u8]| -> Vec<i32> {
        b.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let codes_t =
        Tensor::<B, 2, Int>::from_data(TensorData::new(word(codes), [rows, cols / 4]), dev);
    let scales_t =
        Tensor::<B, 2, Int>::from_data(TensorData::new(word(scales), [rows, n_scales / 4]), dev);
    let s2 = Tensor::<B, 1>::from_data(TensorData::new(vec![scale2; rows], [rows]), dev);
    dequant_nvfp4_words(codes_t, scales_t, s2)
}


/// Dequantise NVFP4 from **word-packed** codes, so the host never widens them.
///
/// `code_words` is `[rows, cols / 4]`: four consecutive packed bytes bitcast
/// into one little-endian `i32`. `scale_words` is `[rows, n_scales / 4]`, the
/// same treatment for the raw E4M3 scale bytes. Returns `[rows, cols * 2]`.
///
/// Why words: uploading one `i32` per byte expands the packed weight 4x on the
/// host, which measured at 27.6s of a 38.8s expert lane against 4.5s of actual
/// device work. A bitcast moves the same bytes and a quarter as many elements.
///
/// The nibble arithmetic is simpler here than in the byte form, not more
/// complex. Little-endian byte j occupies bits `8j..8j+7`, and NVFP4 stores the
/// low nibble first, so logical element k of a word is exactly `(w >> 4k) & 0xF`
/// for k in 0..8 — the byte stage has no reason to exist. Sign extension from
/// the arithmetic shift is discarded by the mask.
///
/// Gated against [`dequant_nvfp4`], which is itself bit-exact against
/// `compressed_tensors`, so this inherits that oracle rather than asserting its
/// own correctness.
pub fn dequant_nvfp4_words<B: Backend>(
    code_words: Tensor<B, 2, Int>,
    scale_words: Tensor<B, 2, Int>,
    scale2: Tensor<B, 1>,
) -> Tensor<B, 2> {
    use crate::models::inkling::nvfp4::GROUP;
    let dev = code_words.device();
    let (fp4_lut, e4m3_lut) = luts::<B>(&dev);
    let [rows, cwords] = code_words.dims();
    let [rows_s, swords] = scale_words.dims();
    assert_eq!(rows, rows_s, "codes have {rows} rows, scales {rows_s}");
    let logical = cwords * 8;

    // Eight 4-bit codes per word, low nibble first, in logical order.
    let mut nib = Vec::with_capacity(8);
    for k in 0..8u32 {
        nib.push(
            code_words
                .clone()
                .bitwise_right_shift_scalar(((4 * k) as i32).elem())
                .bitwise_and_scalar(0x0Fi32.elem())
                .reshape([rows, cwords, 1]),
        );
    }
    let codes = Tensor::cat(nib, 2).reshape([rows, logical]);
    let vals = gather2(fp4_lut, codes);

    // Four E4M3 scale bytes per word, likewise in order.
    let mut by = Vec::with_capacity(4);
    for j in 0..4u32 {
        by.push(
            scale_words
                .clone()
                .bitwise_right_shift_scalar(((8 * j) as i32).elem())
                .bitwise_and_scalar(0xFFi32.elem())
                .reshape([rows, swords, 1]),
        );
    }
    let n_scales = swords * 4;
    assert_eq!(n_scales * GROUP, logical, "{n_scales} scales cannot cover {logical} values");
    let s = gather2(e4m3_lut, Tensor::cat(by, 2).reshape([rows, n_scales]))
        .reshape([rows, n_scales, 1])
        .repeat_dim(2, GROUP)
        .reshape([rows, logical]);

    // Block scale first, then the per-row factor -- the reference's order,
    // because float multiplication does not associate.
    vals.mul(s).mul(scale2.reshape([rows, 1]).repeat_dim(1, logical))
}
