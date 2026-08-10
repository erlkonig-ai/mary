//! Typed tensor leaves.
//!
//! A leaf is one stored tensor of a model. Today [`LeafHandles`] carries TWO
//! handles — the data and, separately, the shape — and reading one materialises
//! it:
//!
//! ```ignore
//! let data: Vec<f32> = db.view::<[f32]>().expect("data view")[..].to_vec();
//! ```
//!
//! That line obtains a zero-copy view and discards it on the same expression.
//!
//! A [`Tensor`] leaf fixes both. The shape lives in the blob's header, so one
//! handle replaces two and a leaf cannot be paired with the wrong shape. And
//! the payload is read as a view over the pile's mapping, because for a model a
//! `Vec` is not merely wasteful — it decides whether the thing loads at all.
//!
//! # The type IS the query
//!
//! One anchor yields a distinct attribute id per `(element, rank)`, so
//! `Handle<Tensor<F32, 2>>` and `Handle<Tensor<F16, 2>>` are different
//! attributes. A reader asking for f32 matrices cannot receive f16 ones, and a
//! rank-2 query cannot receive a rank-4 tensor.
//!
//! The cost, stated plainly: a reader cannot iterate "every leaf" in one query.
//! It queries per (element, rank). That is the guarantee working rather than a
//! limitation to route around — an untyped iteration is exactly what lets a q4
//! leaf be read as f16 — but it does change the shape of a keymap builder.
//!
//! # Migration
//!
//! Additive. Existing piles keep their `LeafHandles` leaves and keep loading;
//! nothing here rewrites them. A model moves over when its persist binary does,
//! one at a time, and the two forms can coexist in one pile because they are
//! different attributes rather than different values under one.

use anyhow::Result;
use triblespace::core::attribute::Attribute;
use triblespace::core::blob::encodings::tensor::elements::{F16, F32};
use triblespace::core::blob::encodings::tensor::{
    tensor_blob, Tensor, TensorElement, TensorView,
};
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::core::id_hex;
use triblespace::core::inline::encodings::hash::Handle;
use triblespace::prelude::*;

/// Anchor every leaf attribute derives from. Minted 2026-08-10.
///
/// One anchor, not one per dtype: `Attribute::anchored` derives the id from
/// (anchor, value encoding), and the encoding already carries the element type
/// and the rank. Eight combinations cost one minted id.
pub const LEAF_ANCHOR: Id = id_hex!("743E98D23794CA9BEFE727D07482D8D5");

/// The leaf attribute for one element format and rank.
pub fn leaf<T: TensorElement, const RANK: usize>() -> Attribute<Handle<Tensor<T, RANK>>> {
    Attribute::anchored(LEAF_ANCHOR)
}

/// Store a tensor as a typed leaf blob.
///
/// Takes the dims and the raw payload the caller already has. Fallible because
/// the payload is checked against what the dims and element format imply —
/// here, once, rather than as a misread tensor later that produces plausible
/// numbers instead of an error.
pub fn leaf_blob<T: TensorElement, const RANK: usize>(
    dims: [u64; RANK],
    payload: anybytes::Bytes,
) -> Result<Blob<Tensor<T, RANK>>> {
    tensor_blob::<T, RANK>(dims, payload).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Read a leaf WITHOUT materialising it.
///
/// Returns the view. The caller decides whether it ever needs a copy, and for a
/// model the answer is usually no — the bytes are handed to a kernel or aliased
/// onto a device.
pub fn read_leaf<T: TensorElement, const RANK: usize>(
    blob: Blob<Tensor<T, RANK>>,
) -> Result<TensorView> {
    TensorView::try_from_blob(blob).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize) -> anybytes::Bytes {
        anybytes::Bytes::from_source(vec![0u8; n])
    }

    /// One handle, not two, and the shape travels with the data so they cannot
    /// be paired wrongly.
    #[test]
    fn a_leaf_carries_its_own_shape() {
        let blob = leaf_blob::<F32, 2>([8, 16], bytes(8 * 16 * 4)).expect("well formed");
        let view = read_leaf::<F32, 2>(blob).expect("reads");
        assert_eq!(view.dims(), &[8, 16]);
        assert_eq!(view.payload().len(), 512);
    }

    /// THE property JP asked for. Element type and rank are both in the
    /// attribute id, so a reader cannot be handed the wrong kind of leaf.
    #[test]
    fn element_and_rank_are_both_in_the_attribute_id() {
        let f32_2 = leaf::<F32, 2>().id();
        let f16_2 = leaf::<F16, 2>().id();
        let f32_4 = leaf::<F32, 4>().id();
        assert_ne!(f32_2, f16_2, "an f16 leaf is not an f32 leaf");
        assert_ne!(f32_2, f32_4, "a rank-4 leaf is not a rank-2 leaf");
        assert_eq!(f32_2, leaf::<F32, 2>().id(), "and the id is stable");
    }

    /// A payload that contradicts the dims is refused where refusing is cheap.
    /// The same bytes read as the wrong shape produce numbers, not an error.
    #[test]
    fn a_payload_that_does_not_match_the_dims_is_refused() {
        assert!(leaf_blob::<F32, 2>([8, 16], bytes(500)).is_err());
        // and an f16 payload offered as f32 is caught by its length alone
        assert!(leaf_blob::<F32, 2>([8, 16], bytes(8 * 16 * 2)).is_err());
        assert!(leaf_blob::<F16, 2>([8, 16], bytes(8 * 16 * 2)).is_ok());
    }
}

/// Which element format a typed leaf turned out to hold.
///
/// The type parameter is gone by the time a leaf is in a by-name index — a
/// keymap spans every dtype in the model — so the fact travels as data instead.
/// This is erasure done ONCE, at the index boundary, from a read that was
/// typed: the leaf was fetched as `Tensor<F32, 2>` or not at all. It is not the
/// untyped iteration the module docs warn about, where a q4 leaf can be read as
/// f16, because no read here is performed without its type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F32,
    F16,
}

/// One typed leaf, resolved.
///
/// Holds a [`TensorView`], which holds `Bytes` — a view over the pile's mapping,
/// not a copy. Building an index of a whole model therefore costs handles and
/// headers, not weights.
pub struct TypedLeaf {
    pub elem: Elem,
    pub view: TensorView,
}

impl TypedLeaf {
    /// Logical dims.
    pub fn dims(&self) -> &[u64] {
        self.view.dims()
    }

    /// Shape as the `Vec<usize>` the loaders speak.
    pub fn shape(&self) -> Vec<usize> {
        self.view.dims().iter().map(|&d| d as usize).collect()
    }

    /// ZERO-COPY view of an f32 leaf. `None` for f16 — the caller wants
    /// [`Self::to_f32`] there, which must convert and therefore must allocate.
    ///
    /// Works on every platform and for every model, which the previous
    /// zero-copy path did not: it existed only behind
    /// `WeightLoader::Aliased`, gated to macOS and two model features. The
    /// payload is a slice of the blob starting at a 256-byte boundary, so it is
    /// aligned for `[f32]` by construction.
    pub fn view_f32(&self) -> Option<anybytes::View<[f32]>> {
        match self.elem {
            Elem::F32 => self.view.payload().clone().view::<[f32]>().ok(),
            Elem::F16 => None,
        }
    }

    /// Materialise as f32. Allocates — call [`Self::view_f32`] first and only
    /// fall back to this when it returns `None`.
    pub fn to_f32(&self) -> Vec<f32> {
        match self.elem {
            Elem::F32 => self
                .view
                .payload()
                .clone()
                .view::<[f32]>()
                .expect("f32 payload view")[..]
                .to_vec(),
            Elem::F16 => self
                .view
                .payload()
                .clone()
                .view::<[half::f16]>()
                .expect("f16 payload view")
                .iter()
                .map(|h| h.to_f32())
                .collect(),
        }
    }
}

/// Index every typed leaf in a pile by its `safetensor_path` name.
///
/// One query per `(element, rank)` — fourteen, not one — because that is what
/// typing the attribute means. Each hit is read AS its type; nothing is
/// interpreted without one.
///
/// Joins module to leaf directly rather than walking `member` edges from a
/// model root, so a pile holding several models (or one whose root layout
/// differs) indexes the same way. Names are unique within a pile by
/// construction — they are the safetensors keys.
///
/// Costs headers and handles, not weights: the views alias the pile's mapping.
pub fn index_typed_by_name(
    tribles: &TribleSet,
    blobs: &impl triblespace::prelude::BlobStoreGet,
) -> std::collections::HashMap<String, TypedLeaf> {
    use crate::format::attrs;
    let mut map = std::collections::HashMap::new();

    macro_rules! sweep {
        ($elem:ty, $rank:literal, $tag:expr) => {{
            for (n, h) in triblespace::macros::find!(
                (n: Inline<Handle<blobencodings::LongString>>,
                 h: Inline<Handle<Tensor<$elem, $rank>>>),
                triblespace::macros::pattern!(tribles, [
                    { _?m @ attrs::safetensor_path: ?n, attrs::weight: _?w },
                    { _?w @ leaf::<$elem, $rank>(): ?h },
                ])
            ) {
                let name = crate::ingest::read_string(blobs, n);
                let blob: Blob<Tensor<$elem, $rank>> =
                    blobs.get(h).expect("typed leaf blob");
                let view = TensorView::try_from_blob(blob).expect("typed leaf decodes");
                map.insert(name, TypedLeaf { elem: $tag, view });
            }
        }};
    }

    sweep!(F32, 0, Elem::F32);
    sweep!(F32, 1, Elem::F32);
    sweep!(F32, 2, Elem::F32);
    sweep!(F32, 3, Elem::F32);
    sweep!(F32, 4, Elem::F32);
    sweep!(F32, 5, Elem::F32);
    sweep!(F32, 6, Elem::F32);
    sweep!(F16, 0, Elem::F16);
    sweep!(F16, 1, Elem::F16);
    sweep!(F16, 2, Elem::F16);
    sweep!(F16, 3, Elem::F16);
    sweep!(F16, 4, Elem::F16);
    sweep!(F16, 5, Elem::F16);
    sweep!(F16, 6, Elem::F16);

    map
}

/// Index EVERY typed leaf in a pile by its entity id, regardless of which model
/// root (if any) reaches it.
///
/// The by-name index walks `member` edges and so only sees leaves hanging off a
/// model. This sees all of them, which is what a verifier wants: it must not
/// take the graph's word for what exists.
pub fn index_typed_all(
    tribles: &TribleSet,
    blobs: &impl triblespace::prelude::BlobStoreGet,
) -> std::collections::HashMap<Id, TypedLeaf> {
    let mut map = std::collections::HashMap::new();

    macro_rules! sweep_all {
        ($elem:ty, $rank:literal, $tag:expr) => {{
            for (e, h) in triblespace::macros::find!(
                (e: Id, h: Inline<Handle<Tensor<$elem, $rank>>>),
                triblespace::macros::pattern!(tribles, [
                    { ?e @ leaf::<$elem, $rank>(): ?h },
                ])
            ) {
                let blob: Blob<Tensor<$elem, $rank>> =
                    blobs.get(h).expect("typed leaf blob");
                let view = TensorView::try_from_blob(blob).expect("typed leaf decodes");
                map.insert(e, TypedLeaf { elem: $tag, view });
            }
        }};
    }

    sweep_all!(F32, 0, Elem::F32);
    sweep_all!(F32, 1, Elem::F32);
    sweep_all!(F32, 2, Elem::F32);
    sweep_all!(F32, 3, Elem::F32);
    sweep_all!(F32, 4, Elem::F32);
    sweep_all!(F32, 5, Elem::F32);
    sweep_all!(F32, 6, Elem::F32);
    sweep_all!(F16, 0, Elem::F16);
    sweep_all!(F16, 1, Elem::F16);
    sweep_all!(F16, 2, Elem::F16);
    sweep_all!(F16, 3, Elem::F16);
    sweep_all!(F16, 4, Elem::F16);
    sweep_all!(F16, 5, Elem::F16);
    sweep_all!(F16, 6, Elem::F16);

    map
}
