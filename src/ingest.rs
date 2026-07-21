//! Bridge between safetensors files and the `mary::format` graph. Ingest *any*
//! safetensors into the pile — each tensor a content-addressed leaf inside a
//! named module, gathered under a model entity — and materialize it back into a
//! `key → (data, shape)` map for the `WeightLoader::Pile` reconstruction path.
//! Generic over model: F5/voice, flux, gaze all ingest the same way.

use crate::format::attrs;
#[cfg(feature = "import")]
use crate::format::{put_raw, put_raw_f16};
#[cfg(feature = "import")]
use crate::nn::weight_loader::get_tensor_f32;
#[cfg(feature = "import")]
use safetensors::SafeTensors;
use std::collections::HashMap;
use triblespace::prelude::blobencodings::LongString;
use triblespace::prelude::*;

#[cfg(feature = "import")]
type Err = Box<dyn std::error::Error>;

/// Width at which tensor leaves are stored in the pile. `F16` halves the pile and
/// matches the GPU dtype (the prerequisite for zero-copy load); use it for models
/// whose native weights are 16-bit (e.g. bf16 Gemma). `F32` is the lossless
/// default for precision-sensitive small models (the embedders).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LeafDtype {
    F32,
    F16,
}

/// Ingest a single-file safetensors blob into the pile as a model graph. Returns
/// the model Fragment (rooted at the model entity), carrying all module + leaf facts.
/// Import-only: this is the safetensors → pile direction.
#[cfg(feature = "import")]
pub fn save_safetensors(
    bytes: &[u8],
    model_name: &str,
    blobs: &mut impl BlobStorePut,
    dtype: LeafDtype,
) -> Result<Fragment, Err> {
    save_safetensors_filtered(bytes, model_name, blobs, dtype, |_| true)
}

/// [`save_safetensors`] restricted to the tensors whose name passes `keep` —
/// the way to persist ONE component of a multi-component checkpoint under its own
/// model entity (e.g. the qwen3tts talker as a half-width `talker_f16` variant
/// next to the exact f32 leaves).
#[cfg(feature = "import")]
pub fn save_safetensors_filtered(
    bytes: &[u8],
    model_name: &str,
    blobs: &mut impl BlobStorePut,
    dtype: LeafDtype,
    keep: impl Fn(&str) -> bool,
) -> Result<Fragment, Err> {
    let st = SafeTensors::deserialize(bytes)?;
    let mut members: Vec<Id> = Vec::new();
    let mut facts = TribleSet::new();
    for key in st.names().into_iter().filter(|k| keep(k)) {
        // skip non-float tensors (int buffers etc.) — the model forward never loads them
        let view = st.tensor(key)?;
        use safetensors::Dtype;
        if !matches!(view.dtype(), Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16) {
            continue;
        }
        let (data, shape) = get_tensor_f32(&st, key);
        let shp: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let leaf = match dtype {
            LeafDtype::F32 => put_raw(blobs, &data, &shp)?,
            LeafDtype::F16 => put_raw_f16(blobs, &data, &shp)?,
        };
        let leaf_id = leaf.root().expect("leaf root");
        facts += leaf.into_facts();
        let kind = match shape.len() {
            1 => "vector",
            2 => "matrix",
            3 => "conv",
            _ => "tensor",
        };
        let name_h = blobs.put::<LongString, _>(key.to_string())?;
        let m = entity! { _ @ attrs::kind: kind, attrs::safetensor_path: name_h, attrs::weight: leaf_id };
        members.push(m.root().expect("module root"));
        facts += m.into_facts();
    }
    let mn = blobs.put::<LongString, _>(model_name.to_string())?;
    let model = entity! { _ @ attrs::model_name: mn, attrs::member*: members.iter() };
    let model_id = model.root().expect("model root");
    facts += model.into_facts();
    Ok(Fragment::rooted(model_id, facts))
}

fn read_string(blobs: &impl BlobStoreGet, h: Inline<inlineencodings::Handle<LongString>>) -> String {
    let v: anybytes::View<str> = blobs.get(h).expect("string blob");
    v.to_string()
}

/// A pile-resident tensor leaf addressed by its content handles (cheap to hold:
/// two ~32-byte handles, no data). [`read_leaf`] fetches the data on demand.
/// A leaf is stored either as f32 (`data`) or half-width f16 (`data_f16`).
#[derive(Clone, Copy)]
pub enum LeafHandles {
    F32(
        Inline<inlineencodings::Handle<crate::format::F32Array>>,
        Inline<inlineencodings::Handle<crate::format::U64Array>>,
    ),
    F16(
        Inline<inlineencodings::Handle<crate::f16enc::F16Array>>,
        Inline<inlineencodings::Handle<crate::format::U64Array>>,
    ),
}

/// Read one tensor leaf's `(f32 data, shape)` from its blob handles — the
/// expensive part (the data blob) happens here, lazily, one tensor at a time.
/// f16 leaves are up-cast to f32 here for the f32-centric model loaders; the
/// fast pile load instead uploads the f16 bytes to the GPU at native width.
pub fn read_leaf(blobs: &impl BlobStoreGet, handles: LeafHandles) -> (Vec<f32>, Vec<usize>) {
    match handles {
        LeafHandles::F32(dh, sh) => {
            let db: anybytes::Bytes = blobs.get(dh).expect("data blob");
            let data: Vec<f32> = db.view::<[f32]>().expect("data view")[..].to_vec();
            (data, read_shape(blobs, sh))
        }
        LeafHandles::F16(dh, sh) => {
            let db: anybytes::Bytes = blobs.get(dh).expect("data_f16 blob");
            let data: Vec<f32> = db
                .view::<[half::f16]>()
                .expect("f16 data view")
                .iter()
                .map(|h| h.to_f32())
                .collect();
            (data, read_shape(blobs, sh))
        }
    }
}

/// Read just a leaf's shape blob — the cheap half of [`read_leaf`], for gates
/// and loaders that want dims without materializing the data.
pub fn read_shape(
    blobs: &impl BlobStoreGet,
    sh: Inline<inlineencodings::Handle<crate::format::U64Array>>,
) -> Vec<usize> {
    let sbb: anybytes::Bytes = blobs.get(sh).expect("shape blob");
    sbb.view::<[u64]>().expect("shape view").iter().map(|&d| d as usize).collect()
}

/// Build a `name → handle-pair` INDEX of a model graph — reads only the small
/// tensor-NAME blobs, never the data. The index is tiny (two handles per
/// tensor); [`read_leaf`] fetches each tensor's data lazily. This is what lets a
/// large model (the 31B) load from a pile without materializing its whole f32
/// keymap in RAM at once — peak CPU is one tensor, not the full set. Each leaf is
/// resolved as f16 (`data_f16`) if present, else f32 (`data`).
pub fn index_keymap(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    model_id: Id,
) -> HashMap<String, LeafHandles> {
    let mut map = HashMap::new();
    let members: Vec<_> = find!((m: Id), pattern!(tribles, [{ model_id @ attrs::member: ?m }])).collect();
    for (mid,) in members {
        let (name_h, w_id) = find!(
            (n, w: Id),
            pattern!(tribles, [{ mid @ attrs::safetensor_path: ?n, attrs::weight: ?w }])
        )
        .next()
        .expect("module name/weight");
        let key = read_string(blobs, name_h);
        let f16 = find!(
            (d, s),
            pattern!(tribles, [{ w_id @ attrs::data_f16: ?d, attrs::shape: ?s }])
        )
        .next();
        if let Some((dh, sh)) = f16 {
            map.insert(key, LeafHandles::F16(dh, sh));
        } else {
            let (dh, sh) = find!(
                (d, s),
                pattern!(tribles, [{ w_id @ attrs::data: ?d, attrs::shape: ?s }])
            )
            .next()
            .expect("leaf data/shape");
            map.insert(key, LeafHandles::F32(dh, sh));
        }
    }
    map
}

/// Materialize a model graph back into a `key → (data, shape)` map by walking its
/// member modules. Feeds `WeightLoader::Pile`. Reads every tensor's f32 data into
/// RAM — fine for small models; large models stream via [`index_keymap`] +
/// [`read_leaf`] instead.
pub fn load_keymap(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    model_id: Id,
) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    index_keymap(tribles, blobs, model_id)
        .into_iter()
        .map(|(k, handles)| (k, read_leaf(blobs, handles)))
        .collect()
}
