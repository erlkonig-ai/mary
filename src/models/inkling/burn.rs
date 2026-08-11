//! Inkling's Burn lane — the same arithmetic as the f32 slice lane, on a backend.
//!
//! Mirrors how `k3` keeps `kda.rs` beside its Burn path: the slice lane in
//! [`crate::models::inkling::mlp`] is the reference, gated against
//! `transformers`, and this is checked against *it*. So the Burn lane gets a
//! real oracle without needing torch in the loop.
//!
//! Scope was the cost-dominant part first — the routed experts — and is now the
//! whole arithmetic of a decoder layer: attention with its two short
//! convolutions, the shared experts, the dense MLP, RMSNorm, and the NVFP4
//! decode. What drove the second half was a measured 401 s forward of 109
//! tokens in which attention was 108 s and the shared plus dense MLPs 145 s, all
//! of it scalar host code left behind as a correctness reference. Moving both
//! took them to 8.9 s and 15.0 s.
//!
//! Every one of these is gated against `transformers` by `inkling_burn_gate`,
//! not against the slice lane: the two lanes were written by the same hand and
//! agreeing with each other proves only that.
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

/// The shared experts, on device — every token visits all of them.
///
/// `gate` and `up` are `[n_shared * intermediate, hidden]`, `down` is
/// `[n_shared * hidden, intermediate]`, and `gammas` is `[tokens, n_shared]`.
/// The gamma multiplies the **activation**, before the down projection — not
/// the block's output, which is algebraically the same only because `down` is
/// linear and is a different function the moment anything else is inserted.
///
/// Gate and up arrive already split rather than fused, so this can be gated
/// straight against `transformers`' own `shared_experts.gate_proj` /
/// `up_proj` with nothing transcribed in between; the checkpoint's interleaved
/// `shared_w13_weight` is turned into these by [`split_shared_fused`].
pub fn shared_experts<B: Backend>(
    x: Tensor<B, 2>,
    gate: Tensor<B, 2>,
    up: Tensor<B, 2>,
    down: Tensor<B, 2>,
    gammas: Tensor<B, 2>,
    n_shared: usize,
) -> Tensor<B, 2> {
    let [tokens, hidden] = x.dims();
    let [drows, inter] = down.dims();
    assert_eq!(drows, n_shared * hidden, "shared w2 has {drows} rows, want {}", n_shared * hidden);
    assert_eq!(gate.dims(), [n_shared * inter, hidden], "shared gate is {:?}", gate.dims());
    assert_eq!(up.dims(), [n_shared * inter, hidden], "shared up is {:?}", up.dims());
    assert_eq!(gammas.dims(), [tokens, n_shared], "gammas must be [tokens, n_shared]");

    let mut acc: Option<Tensor<B, 2>> = None;
    for s in 0..n_shared {
        let g = gate.clone().slice([s * inter..(s + 1) * inter, 0..hidden]);
        let u = up.clone().slice([s * inter..(s + 1) * inter, 0..hidden]);
        let dn = down.clone().slice([s * hidden..(s + 1) * hidden, 0..inter]);
        let gamma = gammas.clone().slice([0..tokens, s..s + 1]);
        let act = silu(linear(x.clone(), g)) * linear(x.clone(), u) * gamma;
        let contrib = linear(act, dn);
        acc = Some(match acc {
            None => contrib,
            Some(a) => a + contrib,
        });
    }
    acc.expect("a MoE layer has at least one shared expert")
}

/// Split the checkpoint's fused `shared_w13_weight` into gate and up blocks.
///
/// `fused` is `[n_shared * 2 * intermediate, hidden]` in **checkpoint
/// interleave** — gate on the even rows, up on the odd ones, per shared expert.
/// Returns `[n_shared * intermediate, hidden]` twice, which is what
/// [`shared_experts`] wants.
///
/// Splitting on device keeps the host copy in raw checkpoint order, so the
/// 33 M-element shuffle per layer never happens on a scalar loop.
pub fn split_shared_fused<B: Backend>(
    fused: Tensor<B, 2>,
    n_shared: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [frows, hidden] = fused.dims();
    assert_eq!(frows % (2 * n_shared), 0, "{frows} rows do not split into {n_shared} experts");
    let inter = frows / (2 * n_shared);
    let mut gates = Vec::with_capacity(n_shared);
    let mut ups = Vec::with_capacity(n_shared);
    for s in 0..n_shared {
        let gu = deinterleave_rows_device(
            fused.clone().slice([s * 2 * inter..(s + 1) * 2 * inter, 0..hidden]),
        );
        gates.push(gu.clone().slice([0..inter, 0..hidden]));
        ups.push(gu.slice([inter..2 * inter, 0..hidden]));
    }
    (Tensor::cat(gates, 0), Tensor::cat(ups, 0))
}

/// Depthwise causal short convolution **plus its internal residual**, on device.
///
/// The device twin of [`crate::models::inkling::block::short_conv`]:
///
/// ```text
/// conv[t] = sum_{j=0}^{k-1} w[j] * x[t + j - (k - 1)]        x[<0] = 0
/// out[t]  = x[t] + conv[t]
/// ```
///
/// Written as `k` shifted slices of a front-zero-padded input rather than as a
/// convolution kernel, because `k` is 4 and the shift is exactly what the
/// formula says. Returning only `conv` — dropping the module's own residual —
/// is the mistake this shape makes hard to hide.
pub fn short_conv<B: Backend>(x: Tensor<B, 2>, weight: Tensor<B, 2>) -> Tensor<B, 2> {
    let [tokens, dim] = x.dims();
    let [wdim, kernel] = weight.dims();
    assert_eq!(dim, wdim, "short_conv: x is [_, {dim}] but the weight is [{wdim}, _]");
    assert!(kernel > 0, "a short convolution needs a kernel");
    let dev = x.device();
    let pad: Tensor<B, 2> = Tensor::zeros([kernel - 1, dim], &dev);
    let padded = Tensor::cat(vec![pad, x.clone()], 0);

    let mut conv: Option<Tensor<B, 2>> = None;
    for j in 0..kernel {
        // t + j - (kernel - 1) in x is t + j in the padded tensor.
        let seg = padded.clone().slice([j..j + tokens, 0..dim]);
        let wj = weight.clone().slice([0..dim, j..j + 1]).reshape([1, dim]);
        let term = seg * wj;
        conv = Some(match conv {
            None => term,
            Some(c) => c + term,
        });
    }
    x + conv.expect("kernel > 0")
}

/// RMS-normalize each head slice of `[tokens, heads * head_dim]`.
fn head_rms_norm<B: Backend>(
    v: Tensor<B, 2>,
    gain: Tensor<B, 1>,
    heads: usize,
    head_dim: usize,
    eps: f64,
) -> Tensor<B, 2> {
    let [tokens, width] = v.dims();
    assert_eq!(width, heads * head_dim, "{width} is not {heads} x {head_dim}");
    rms_norm(v.reshape([tokens * heads, head_dim]), gain, eps).reshape([tokens, width])
}

/// Every weight one attention layer needs, already on the device.
///
/// Orientations are the checkpoint's: `w*` are `[out, in]` the way `nn.Linear`
/// stores them, the short convolutions are `[dim, kernel]`, and `rel_proj` is
/// `[d_rel, rel_extent]`.
pub struct AttnWeightsDev<B: Backend> {
    pub wq: Tensor<B, 2>,
    pub wk: Tensor<B, 2>,
    pub wv: Tensor<B, 2>,
    pub wr: Tensor<B, 2>,
    pub wo: Tensor<B, 2>,
    pub k_sconv: Tensor<B, 2>,
    pub v_sconv: Tensor<B, 2>,
    pub q_norm: Tensor<B, 1>,
    pub k_norm: Tensor<B, 1>,
    pub rel_proj: Tensor<B, 2>,
}

/// One attention layer over a whole sequence, on the device, no cache.
///
/// The device twin of [`crate::models::inkling::attn::attention`] and gated
/// against the same `transformers` capture, not against it: matching the slice
/// lane would only prove the two agree, and they were written by the same hand.
///
/// `mask` is the additive `[tokens, tokens]` mask — zero where a key is visible
/// and `-inf` where it is not — because a local layer's mask carries the sliding
/// window and a global layer's does not.
///
/// Two things are folded together here that a careless reading separates:
/// log scaling multiplies the query **and** the relative-position bias, and only
/// on global layers; and the bias is zero outside `0 <= q - k < rel_extent`,
/// while causality lives in the mask.
pub fn attention<B: Backend>(
    x: Tensor<B, 2>,
    w: &AttnWeightsDev<B>,
    d: &crate::models::inkling::attn::AttnDims,
    log_scaling: Option<crate::models::inkling::attn::LogScaling>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 2> {
    use crate::models::inkling::config::AttnKind;

    let [tokens, hidden] = x.dims();
    assert_eq!(hidden, d.hidden, "x is [_, {hidden}] but the config says {}", d.hidden);
    assert_eq!(mask.dims(), [tokens, tokens], "the mask must be [tokens, tokens]");
    let dev = x.device();
    let (heads, kv_heads, head_dim) = (d.heads, d.kv_heads, d.head_dim);
    let groups = d.groups();
    assert_eq!(groups * kv_heads, heads, "{heads} heads do not divide into {kv_heads} kv heads");

    // K and V pass through their short convolutions; Q does not.
    let q = linear(x.clone(), w.wq.clone());
    let k = short_conv(linear(x.clone(), w.wk.clone()), w.k_sconv.clone());
    let v = short_conv(linear(x.clone(), w.wv.clone()), w.v_sconv.clone());
    let r = linear(x, w.wr.clone());

    let q = head_rms_norm(q, w.q_norm.clone(), heads, head_dim, d.rms_eps);
    let k = head_rms_norm(k, w.k_norm.clone(), kv_heads, head_dim, d.rms_eps);

    // Log scaling: the same vector the slice lane builds, from the same method.
    let taus: Vec<f32> = (0..tokens)
        .map(|t| match (d.kind, log_scaling) {
            (AttnKind::Global, Some(ls)) => ls.tau(t),
            _ => 1.0,
        })
        .collect();
    let tau: Tensor<B, 1> = Tensor::from_data(TensorData::new(taus, [tokens]), &dev);
    let q = q * tau.clone().reshape([tokens, 1]);

    // Only distances that can occur are worth projecting: a distance is at most
    // `tokens - 1` and the table stops at `rel_extent`.
    let eff = d.rel_extent.min(tokens);
    let mut idx = vec![0i32; tokens * tokens];
    let mut valid = vec![0f32; tokens * tokens];
    for qi in 0..tokens {
        for ki in 0..tokens {
            let dist = qi as isize - ki as isize;
            if dist >= 0 && (dist as usize) < d.rel_extent {
                idx[qi * tokens + ki] = dist as i32;
                valid[qi * tokens + ki] = 1.0;
            }
        }
    }
    let idx: Tensor<B, 3, Int> =
        Tensor::from_data(TensorData::new(idx, [1, tokens, tokens]), &dev).repeat_dim(0, heads);
    let valid: Tensor<B, 3> = Tensor::from_data(TensorData::new(valid, [1, tokens, tokens]), &dev);

    let rel = r
        .reshape([tokens * heads, d.d_rel])
        .matmul(w.rel_proj.clone().slice([0..d.d_rel, 0..eff]))
        .reshape([tokens, heads, eff])
        .swap_dims(0, 1)
        * tau.reshape([1, tokens, 1]);
    let bias = rel.gather(2, idx) * valid;

    // [heads, tokens, head_dim]; the KV heads are repeated in place, so head h
    // reads kv head h / groups exactly as the slice lane indexes it.
    let qh = q.reshape([tokens, heads, head_dim]).swap_dims(0, 1);
    let expand = |t: Tensor<B, 2>| -> Tensor<B, 3> {
        t.reshape([tokens, kv_heads, head_dim])
            .swap_dims(0, 1)
            .reshape([kv_heads, 1, tokens, head_dim])
            .repeat_dim(1, groups)
            .reshape([heads, tokens, head_dim])
    };
    let kh = expand(k);
    let vh = expand(v);

    let scores = qh.matmul(kh.swap_dims(1, 2)).mul_scalar(d.scaling()) + bias
        + mask.reshape([1, tokens, tokens]);
    let probs = burn::tensor::activation::softmax(scores, 2);
    let out = probs
        .matmul(vh)
        .swap_dims(0, 1)
        .reshape([tokens, heads * head_dim]);
    linear(out, w.wo.clone())
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
