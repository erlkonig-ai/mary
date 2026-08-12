//! NVFP4 weight decode as ONE CUDA kernel, and the upload that feeds it.
//!
//! The Burn lane in [`crate::models::inkling::burn::dequant_nvfp4_words`]
//! expresses the decode as tensor algebra: eight shifts, eight masks, a
//! `cat`, a gather, the same again for the scales, a `repeat_dim` and two
//! multiplies. That is 46 kernels per expert weight and — because `cat` is
//! `slice_assign`, which is out of place — about 1.4 GB of device traffic to
//! produce 100 MB of f32. Measured on a five-token forward: 19 368
//! `slice_assign` launches, 4.6 s, 48% of all GPU time in the model.
//!
//! Here it is one kernel per weight: one thread per packed word, eight
//! outputs each, the gate/up de-interleave folded into the destination row so
//! the permutation costs nothing, and no intermediate at all. The arithmetic
//! is the same in the same order — `(fp4 * block_scale) * scale2`, f32 — so
//! `inkling_expert_probe` gates it BITWISE against the Burn chain rather than
//! on a tolerance.
//!
//! Uploads go through `create_from_slice` on a borrowed mmap span, so the host
//! never widens or copies the packed bytes. The Burn path went through
//! `TensorData` → a 2-D `cuMemcpy2DAsync` per tensor, which measured 1.26 ms
//! per call — 6.3 s of a 45.7 s forward in the memcpy API alone.

use std::cell::RefCell;

use burn::backend::wgpu::CubeTensor; // the generic CubeTensor, re-exported
use burn::tensor::{DType, Tensor, TensorPrimitive};
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::Runtime;

use crate::models::inkling::nvfp4::{e4m3_to_f32, FP4_E2M1, GROUP};

/// The backend this module hands tensors back on.
pub type Bk = burn::backend::Cuda<f32>;
type Client = cubecl::client::ComputeClient<CudaRuntime>;

/// Threads per cube. The launch is exact — every expert dimension in the
/// checkpoint divides it — so the kernel needs no bounds check.
const CUBE: usize = 256;

/// Decode one NVFP4 weight, `[rows, cols]` packed bytes to `[rows, cols * 2]`
/// f32, optionally de-interleaving gate/up rows on the way out.
///
/// One thread per 4-byte code word: eight logical values, one E4M3 block
/// scale (`GROUP` is 16, so a word spans exactly half a scale group), one
/// per-expert `scale2`. `permute` sends source row `2j` to `j` and `2j + 1`
/// to `half + j` — the checkpoint's gate/up interleave, undone by writing
/// somewhere else rather than by a second pass over 67 MB.
#[cube(launch_unchecked)]
#[allow(clippy::manual_is_multiple_of)] // `%` is the cube-kernel primitive
fn nvfp4_dequant_kernel(
    code_words: &Array<u32>,
    scale_words: &Array<u32>,
    fp4_lut: &Array<f32>,
    e4m3_lut: &Array<f32>,
    scale2: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] cwords: usize,
    #[comptime] swords: usize,
    #[comptime] half: usize,
    #[comptime] permute: bool,
) {
    let idx = ABSOLUTE_POS;
    let r = idx / cwords;
    let w = idx % cwords;

    let mut dst = r;
    if comptime![permute] {
        dst = (r % 2) * half + r / 2;
    }

    let word = code_words[idx];
    // Two words per E4M3 scale byte: 8 values a word, one scale per 16.
    let sb = w / 2;
    let sword = scale_words[r * swords + sb / 4];
    let sh = (8 * (sb % 4)) as u32;
    let s = e4m3_lut[((sword >> sh) & 255) as usize];
    let s2 = scale2[0];
    let base = dst * cwords * 8 + w * 8;

    out[base] = fp4_lut[(word & 15) as usize] * s * s2;
    out[base + 1] = fp4_lut[((word >> 4) & 15) as usize] * s * s2;
    out[base + 2] = fp4_lut[((word >> 8) & 15) as usize] * s * s2;
    out[base + 3] = fp4_lut[((word >> 12) & 15) as usize] * s * s2;
    out[base + 4] = fp4_lut[((word >> 16) & 15) as usize] * s * s2;
    out[base + 5] = fp4_lut[((word >> 20) & 15) as usize] * s * s2;
    out[base + 6] = fp4_lut[((word >> 24) & 15) as usize] * s * s2;
    out[base + 7] = fp4_lut[((word >> 28) & 15) as usize] * s * s2;
}

thread_local! {
    /// The two decode tables, uploaded once. 1 KB, but a per-expert upload of
    /// it is 1 666 more host-to-device calls in a forward.
    static LUTS: RefCell<Option<(Handle, Handle)>> = const { RefCell::new(None) };
}

/// Little-endian bytes of an f32 slice, for the two small table uploads.
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

/// The FP4 and E4M3 tables on device, built from the same scalar decoders the
/// CPU lane is gated on.
fn luts(client: &Client) -> (Handle, Handle) {
    LUTS.with(|c| {
        let mut c = c.borrow_mut();
        if let Some((a, b)) = c.as_ref() {
            return (a.clone(), b.clone());
        }
        let fp4: Vec<f32> = FP4_E2M1.to_vec();
        // NaN would poison a decode, and only 0x7F/0xFF are NaN in E4M3-fn;
        // they never appear as a block scale. Same substitution as the Burn
        // lane's `luts`, so the two tables are the same table.
        let e4m3: Vec<f32> = (0..256u16)
            .map(|b| {
                let v = e4m3_to_f32(b as u8);
                if v.is_nan() { 0.0 } else { v }
            })
            .collect();
        let h = (
            client.create_from_slice(&f32_bytes(&fp4)),
            client.create_from_slice(&f32_bytes(&e4m3)),
        );
        *c = Some(h.clone());
        h
    })
}

/// Widen BF16 to f32 on the device: one thread per packed pair.
///
/// BF16 -> f32 is the sixteen stored bits in the HIGH half of an f32 and zeros
/// below, so this is a shift and a mask, exact in both directions. The host
/// twin in [`crate::models::inkling::load::Checkpoint::expert_slice`] computes
/// the same thing one scalar at a time behind a 100 MB allocation.
#[cube(launch_unchecked)]
fn bf16_widen_kernel(words: &Array<u32>, out: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    let w = words[i];
    // Little-endian: the low half-word is the EARLIER element.
    out[2 * i] = f32::reinterpret(w << u32::new(16));
    out[2 * i + 1] = f32::reinterpret(w & u32::new(0xffff_0000i64));
}

/// Upload one expert's BF16 bytes and widen them on the device.
///
/// `raw` is `[rows, cols]` little-endian BF16 — exactly what
/// [`crate::models::inkling::load::Bf16ExpertRef`] borrows out of the mapping.
/// Returns `[rows, cols]` f32, bit-identical to widening on the host.
pub fn expert_weight_bf16(
    raw: &[u8],
    rows: usize,
    cols: usize,
    device: &CudaDevice,
) -> Tensor<Bk, 2> {
    let client = CudaRuntime::client(device);
    let n = rows * cols;
    assert_eq!(raw.len(), n * 2, "raw is {} bytes, want {rows}x{cols} BF16", raw.len());
    assert_eq!(n % 2, 0, "{n} elements do not pack into 4-byte words");
    let words = n / 2;
    assert_eq!(words % CUBE, 0, "{words} words do not divide into cubes of {CUBE}");

    let src_h = client.create_from_slice(raw);
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    unsafe {
        bf16_widen_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            CubeCount::new_1d((words / CUBE) as u32),
            CubeDim::new_1d(CUBE as u32),
            ArrayArg::from_raw_parts(src_h, words),
            ArrayArg::from_raw_parts(out_h.clone(), n),
        );
    }
    let cube = CubeTensor::<CudaRuntime>::new_contiguous(
        client.clone(),
        device.clone(),
        [rows, cols].into(),
        out_h,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(cube))
}

/// Upload the packed bytes and nothing else — the floor every decode path
/// shares, and the only part of an expert load that is irreducibly a copy.
pub fn upload_only(codes: &[u8], scales: &[u8], device: &CudaDevice) {
    drop(upload_held(codes, scales, device));
}

/// The same upload, handing the handles BACK.
///
/// Dropping a handle returns its memory to the pool, which is real work and
/// is not part of an upload. A probe that drops inside its own timer measures
/// the two together and reports an upload slower than the whole lane that
/// contains it — which is how this function came to exist.
pub fn upload_held(codes: &[u8], scales: &[u8], device: &CudaDevice) -> (Handle, Handle) {
    let client = CudaRuntime::client(device);
    (client.create_from_slice(codes), client.create_from_slice(scales))
}

/// Upload one expert's packed weight and decode it, in two kernels' worth of
/// nothing: a host-to-device copy of the checkpoint's own bytes and one
/// dequant launch.
///
/// `codes` is `[rows, cols]` bytes and `scales` is `[rows, cols * 2 / GROUP]`
/// raw E4M3 bytes — exactly what
/// [`crate::models::inkling::load::PackedExpertRef`] borrows out of the
/// mapping. Returns `[rows, cols * 2]`, gate rows first when `permute`.
pub fn expert_weight_fused(
    codes: &[u8],
    scales: &[u8],
    scale2: f32,
    rows: usize,
    cols: usize,
    permute: bool,
    device: &CudaDevice,
) -> Tensor<Bk, 2> {
    let client = CudaRuntime::client(device);
    assert_eq!(codes.len(), rows * cols, "codes is {} bytes, want {rows}x{cols}", codes.len());
    assert_eq!(cols % 4, 0, "{cols} bytes per row does not pack into words");
    let logical = cols * 2;
    assert_eq!(logical % GROUP, 0, "{logical} values do not group by {GROUP}");
    let sbytes = logical / GROUP;
    assert_eq!(scales.len(), rows * sbytes, "scales is {} bytes, want {rows}x{sbytes}", scales.len());
    assert_eq!(sbytes % 4, 0, "{sbytes} scale bytes per row do not pack into words");
    assert!(!permute || rows % 2 == 0, "cannot de-interleave {rows} rows");

    let cwords = cols / 4;
    let swords = sbytes / 4;
    let threads = rows * cwords;
    assert_eq!(threads % CUBE, 0, "{threads} words do not divide into cubes of {CUBE}");

    let (fp4_h, e4m3_h) = luts(&client);
    let codes_h = client.create_from_slice(codes);
    let scales_h = client.create_from_slice(scales);
    let s2_h = client.create_from_slice(&f32_bytes(&[scale2]));
    let out_h = client.empty(rows * logical * core::mem::size_of::<f32>());

    unsafe {
        nvfp4_dequant_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            CubeCount::new_1d((threads / CUBE) as u32),
            CubeDim::new_1d(CUBE as u32),
            ArrayArg::from_raw_parts(codes_h, threads),
            ArrayArg::from_raw_parts(scales_h, rows * swords),
            ArrayArg::from_raw_parts(fp4_h, 16),
            ArrayArg::from_raw_parts(e4m3_h, 256),
            ArrayArg::from_raw_parts(s2_h, 1),
            ArrayArg::from_raw_parts(out_h.clone(), rows * logical),
            cwords,
            swords,
            rows / 2,
            permute,
        );
    }

    let cube = CubeTensor::<CudaRuntime>::new_contiguous(
        client.clone(),
        device.clone(),
        [rows, logical].into(),
        out_h,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(cube))
}
