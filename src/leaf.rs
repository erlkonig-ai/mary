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
    use triblespace::core::blob::encodings::tensor::elements::{F16, F32};

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
