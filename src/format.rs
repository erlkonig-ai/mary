//! mary — neural-network models as content-addressed graphs in TribleSpace.
//!
//! The format, in three primitives:
//!   - a **tensor** is a self-describing leaf: a content-addressed `Array<F32>`
//!     data blob + an `Array<U64>` shape blob. Identical tensors dedup.
//!   - a **module** is an entity. Its parameters are tensor leaves reached by
//!     role-edges; `weight` and `bias` are the universal ones. A Linear is
//!     `{weight, bias?}`; a LayerNorm, Conv, Embedding are the same shape of
//!     thing — bias is just a parameter a module may or may not have.
//!   - **composition** is role-edges (`GenId`) from a module to its children,
//!     ordered (where order matters) by an `index`.
//!
//! Loading is config-free: a tensor carries its own shape. The franken-stitch is
//! rewiring role-edges between module subgraphs.
//!
//! Named for Mary — Mary Wollstonecraft Shelley, who wrote the creature into
//! being, and her mother Mary Wollstonecraft. She gives the assembled parts life.

use burn::prelude::*;
use triblespace::prelude::blobencodings::{elements, Array};
use triblespace::prelude::inlineencodings::{GenId, Handle, ShortString, U256BE};
use triblespace::prelude::*;

/// Flat f32 tensor data.
pub type F32Array = Array<elements::F32>;
/// Tensor shape (row-major dims).
pub type U64Array = Array<elements::U64>;
/// Packed quantized weight words (q4_0 nibbles / q8_0 biased bytes, 4 or 8
/// weights per `u32` — see `mary::nn::q4` and
/// `models::personaplex::temporal_metal` for the exact bit layouts).
pub type U32Array = Array<elements::U32>;

pub mod attrs {
    use super::{F32Array, GenId, Handle, ShortString, U32Array, U64Array, U256BE};
    use crate::f16enc::F16Array;
    use triblespace::prelude::*;

    attributes! {
        // ── tensor leaf ──
        /// Flat f32 data of a tensor leaf.
        "572B45D52A47608F283D0F778597137A" as data: Handle<F32Array>;
        /// Flat f16 (half) data of a tensor leaf — the half-width alternative to
        /// `data`, used for weights whose native dtype is 16-bit (halves the pile
        /// and matches the GPU dtype for zero-copy load). A leaf carries `data`
        /// XOR `data_f16`, plus `shape`.
        "467CCF3FDCCCCE599F6C1B933EACD933" as data_f16: Handle<F16Array>;
        /// Row-major shape (u64 dims) of a tensor leaf.
        "D09A91FC3F04C40AE4A42CD6628A9E38" as shape: Handle<U64Array>;

        // ── quantized tensor leaf (derived runtime formats) ──
        // A quantized leaf carries `data_q4` XOR `data_q8`, plus `q_scales`
        // and `shape` (the logical row-major `[out, in]`). The packed layout
        // is the KERNEL ABI — documented at the quantizers
        // (`nn::q4::quantize_q4`, `temporal_metal::quantize_q8`) and pinned
        // by the `format_marker` on the derived model entity.
        /// GGUF-Q4_0-style packed nibbles of a row-major `[out, in]` weight:
        /// word `k/8` of a row holds weights `k..k+8`, nibble `k%8` at bits
        /// `4·(k%8)`; dequant `w = (nibble − 8) · scale`. Minted 2026-07-12.
        "2ADC6462A7F70E230558C5D681E38768" as data_q4: Handle<U32Array>;
        /// q8_0-style packed BIASED bytes of a row-major `[out, in]` weight:
        /// word `k/4` holds weights `k..k+4`, byte `k%4` at bits `8·(k%4)`,
        /// stored `q + 128`; dequant `w = (byte − 128) · scale`. Minted
        /// 2026-07-12.
        "23178058559C762BB4B1FEAA36B3566D" as data_q8: Handle<U32Array>;
        /// Per-group f16 scales of a quantized leaf — one scale per 32
        /// consecutive weights along the input dim, row-major `[out, in/32]`.
        /// Minted 2026-07-12.
        "F9EA2FB90DC094D42A4845B013950032" as q_scales: Handle<F16Array>;
        /// Layout/ABI version of a DERIVED model entity (quantized sibling
        /// piles). The value is a minted version id; a loader accepts the
        /// pile only when the marker equals its compiled-in constant — the
        /// file format IS the kernel ABI, so any layout change mints a new
        /// version id and re-derives. Minted 2026-07-12.
        "2CC4D16369C4980BCB512937DA204FF5" as format_marker: GenId;

        // ── universal module params (role-edges → tensor leaves) ──
        /// A module's weight tensor.
        "4629D277AD6B52B50DA78DEF63440AF1" as weight: GenId;
        /// A module's bias tensor (optional).
        "18E898172078C843A0351C3D880CC238" as bias: GenId;

        // ── module metadata ──
        /// Module kind tag ("linear", "layernorm", "conv", "embedding", …).
        "52C4A211D2A08BA25C27FFD79FF24C93" as kind: ShortString;
        /// Provenance: the tensor's original safetensors key (e.g. "…grn.gamma").
        "09EA2F7BCF9B0C9714EE39CF269DF2D5" as safetensor_path: Handle<blobencodings::LongString>;
        /// Ordered position among siblings (for layer sequences).
        "33CE12B1B940B13E48D8E5B0ADFD2421" as index: U256BE;

        // ── model root ──
        /// Reference to a model's root entity.
        "3F46CDE630964D78D62DA32F4A8558C1" as model_root: GenId;
        /// A model's module (repeated edge model → module entities).
        "B4B6EC08A0CD70DE63A690168EE78F0F" as member: GenId;
        /// Model name / HuggingFace id (shared id with avatar/gaze).
        "4C1CD1611863E7854C59C7DC706DF77A" as model_name: Handle<blobencodings::LongString>;
        /// Canonical model identity on a shared `mary` branch (e.g. "clip_vit_base",
        /// "qwen3tts", "flux_klein") — lets a loader select ONE model out of a pile
        /// that holds many. The genealogy discriminator: a root (original) and its
        /// derivations (naive-fp4, tuned-fp4-vN) share a `model_id`, distinguished
        /// by `format_marker`; a child links to its root via `model_root`. Minted
        /// 2026-07-24.
        "C8A11B350180DC49007393D5E0AB7100" as model_id: ShortString;
    }
}

type BlobErr = Box<dyn std::error::Error>;

fn tensor_f32<B: Backend, const D: usize>(t: &Tensor<B, D>) -> Vec<f32> {
    t.to_data().to_vec().unwrap()
}

/// Store a flat f32 buffer + its shape as a self-describing leaf. Content-addressed.
pub fn put_raw(
    blobs: &mut impl BlobStorePut,
    data: &[f32],
    shape: &[u64],
) -> Result<Fragment, BlobErr> {
    let d = blobs.put::<F32Array, _>(data.to_vec())?;
    let s = blobs.put::<U64Array, _>(shape.to_vec())?;
    Ok(entity! { _ @ attrs::data: d, attrs::shape: s })
}

/// Store a flat f32 buffer DOWN-CAST to f16 + its shape as a half-width leaf
/// (`data_f16`). Halves the pile for 16-bit-native weights and stores them in the
/// GPU's dtype, so the load needs no conversion. f32→f16 is lossless for weights
/// that originated as bf16 (f16's 10-bit mantissa covers bf16's 7).
pub fn put_raw_f16(
    blobs: &mut impl BlobStorePut,
    data: &[f32],
    shape: &[u64],
) -> Result<Fragment, BlobErr> {
    let halves: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
    // Plain `put`: under the V3 pile every record is already 256-aligned, so the
    // f16 data lands GPU-ready for zero-copy aliasing with no special path (the
    // old `put_aligned` was the V2-era shim for exactly this and is now redundant).
    let d = blobs.put::<crate::f16enc::F16Array, _>(halves)?;
    let s = blobs.put::<U64Array, _>(shape.to_vec())?;
    Ok(entity! { _ @ attrs::data_f16: d, attrs::shape: s })
}

/// Store a PACKED-Q4 weight (`nn::q4::quantize_q4` output: nibble words +
/// f16 group scales) as a quantized leaf (`{data_q4, q_scales, shape}`).
/// `shape` is the logical row-major `[out, in]`. The packed words are the
/// exact bytes the q4 matvec kernel streams — persisting them once is what
/// makes the load a pure mmap (no quantization pass).
pub fn put_raw_q4(
    blobs: &mut impl BlobStorePut,
    wq: &[u32],
    scales: &[half::f16],
    shape: &[u64],
) -> Result<Fragment, BlobErr> {
    let d = blobs.put::<U32Array, _>(wq.to_vec())?;
    let sc = blobs.put::<crate::f16enc::F16Array, _>(scales.to_vec())?;
    let s = blobs.put::<U64Array, _>(shape.to_vec())?;
    Ok(entity! { _ @ attrs::data_q4: d, attrs::q_scales: sc, attrs::shape: s })
}

/// Store a PACKED-Q8 weight (`temporal_metal::quantize_q8` output: biased
/// byte words + f16 group scales) as a quantized leaf
/// (`{data_q8, q_scales, shape}`) — the q8 twin of [`put_raw_q4`].
pub fn put_raw_q8(
    blobs: &mut impl BlobStorePut,
    wq: &[u32],
    scales: &[half::f16],
    shape: &[u64],
) -> Result<Fragment, BlobErr> {
    let d = blobs.put::<U32Array, _>(wq.to_vec())?;
    let sc = blobs.put::<crate::f16enc::F16Array, _>(scales.to_vec())?;
    let s = blobs.put::<U64Array, _>(shape.to_vec())?;
    Ok(entity! { _ @ attrs::data_q8: d, attrs::q_scales: sc, attrs::shape: s })
}

/// Store a tensor as a self-describing leaf (`{data, shape}`). Content-addressed:
/// identical tensors collapse to one entity. Returns the leaf Fragment.
pub fn put_tensor<B: Backend, const D: usize>(
    blobs: &mut impl BlobStorePut,
    t: &Tensor<B, D>,
) -> Result<Fragment, BlobErr> {
    let dims: Vec<u64> = t.dims().iter().map(|&d| d as u64).collect();
    put_raw(blobs, &tensor_f32(t), &dims)
}

/// Load a tensor leaf by id into a `Tensor<B, D>` using its stored shape.
pub fn load_tensor<B: Backend, const D: usize>(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    id: Id,
    device: &B::Device,
) -> Tensor<B, D> {
    let (dh, sh) = find!(
        (d, s),
        pattern!(tribles, [{ id @ attrs::data: ?d, attrs::shape: ?s }])
    )
    .next()
    .expect("tensor leaf not found");
    let data: anybytes::Bytes = blobs.get(dh).expect("data blob");
    let data: anybytes::View<[f32]> = data.view().expect("data view");
    let shp: anybytes::Bytes = blobs.get(sh).expect("shape blob");
    let shp: anybytes::View<[u64]> = shp.view().expect("shape view");
    assert_eq!(shp.len(), D, "tensor rank mismatch");
    let mut dims = [0usize; D];
    for (i, d) in shp.iter().enumerate() {
        dims[i] = *d as usize;
    }
    Tensor::<B, 1>::from_floats(&data[..], device).reshape(dims)
}

/// Store a Linear (`weight` [out,in], optional `bias` [out]) as a module entity.
/// Returns a fragment rooted at the module, carrying the module + leaf facts.
pub fn put_linear<B: Backend>(
    blobs: &mut impl BlobStorePut,
    weight: &Tensor<B, 2>,
    bias: Option<&Tensor<B, 1>>,
) -> Result<Fragment, BlobErr> {
    let w = put_tensor(blobs, weight)?;
    let w_id = w.root().expect("weight leaf root");
    let mut facts = w.into_facts();

    let bias_id = match bias {
        Some(b) => {
            let bf = put_tensor(blobs, b)?;
            let bid = bf.root().expect("bias leaf root");
            facts += bf.into_facts();
            Some(bid)
        }
        None => None,
    };

    let module = entity! { _ @ attrs::kind: "linear", attrs::weight: w_id, attrs::bias?: bias_id };
    let mid = module.root().expect("module root");
    facts += module.into_facts();
    Ok(Fragment::rooted(mid, facts))
}

/// Load a Linear module by id → (weight [out,in], optional bias [out]).
pub fn load_linear<B: Backend>(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    id: Id,
    device: &B::Device,
) -> (Tensor<B, 2>, Option<Tensor<B, 1>>) {
    let (w_id,) = find!((w: Id), pattern!(tribles, [{ id @ attrs::weight: ?w }]))
        .next()
        .expect("linear missing weight");
    let weight = load_tensor::<B, 2>(tribles, blobs, w_id, device);
    let bias = find!((b: Id), pattern!(tribles, [{ id @ attrs::bias: ?b }]))
        .next()
        .map(|(b_id,)| load_tensor::<B, 1>(tribles, blobs, b_id, device));
    (weight, bias)
}

/// Store a named module (`kind`, `name`) with a rank-`DW` weight and optional
/// 1-D bias — the general case behind `put_linear` (DW=2), convs (DW=3), and
/// norms/embeddings (DW=1, bias-free). Returns a fragment rooted at the module.
pub fn put_module<B: Backend, const DW: usize>(
    blobs: &mut impl BlobStorePut,
    kind: &str,
    name: &str,
    weight: &Tensor<B, DW>,
    bias: Option<&Tensor<B, 1>>,
) -> Result<Fragment, BlobErr> {
    let w = put_tensor(blobs, weight)?;
    let w_id = w.root().expect("weight leaf root");
    let mut facts = w.into_facts();
    let bias_id = match bias {
        Some(b) => {
            let bf = put_tensor(blobs, b)?;
            let bid = bf.root().expect("bias leaf root");
            facts += bf.into_facts();
            Some(bid)
        }
        None => None,
    };
    let name_h = blobs.put::<blobencodings::LongString, _>(name.to_string())?;
    let m = entity! { _ @ attrs::kind: kind, attrs::safetensor_path: name_h, attrs::weight: w_id, attrs::bias?: bias_id };
    let mid = m.root().expect("module root");
    facts += m.into_facts();
    Ok(Fragment::rooted(mid, facts))
}

/// Load a rank-`DW` module by id → (weight, optional 1-D bias).
pub fn load_module<B: Backend, const DW: usize>(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    id: Id,
    device: &B::Device,
) -> (Tensor<B, DW>, Option<Tensor<B, 1>>) {
    let (w_id,) = find!((w: Id), pattern!(tribles, [{ id @ attrs::weight: ?w }]))
        .next()
        .expect("module missing weight");
    let weight = load_tensor::<B, DW>(tribles, blobs, w_id, device);
    let bias = find!((b: Id), pattern!(tribles, [{ id @ attrs::bias: ?b }]))
        .next()
        .map(|(b_id,)| load_tensor::<B, 1>(tribles, blobs, b_id, device));
    (weight, bias)
}
