//! Inkling experts as pile blobs.
//!
//! The checkpoint stores one expert matrix as five separate safetensors entries
//! bound only by a naming convention — `w13_weight`, `.scale`, `.scale2`,
//! `.input_amax`, `.original_shape`. Nothing makes that bundle atomic, and a
//! reader holding only the first has bytes it cannot interpret: the packed
//! shape says 2048 where the tensor is 4096 wide, and the truth lives in a
//! sixth place the reader has to know to consult.
//!
//! A [`Tensor<NVFP4, 2>`] blob is the bundle made atomic. One content-addressed
//! handle carries the codes, the block scales and the global scale together,
//! with the LOGICAL dimensions in its header — so `original_shape` has nothing
//! left to correct.
//!
//! # Why per expert
//!
//! The stacked form is 256 independent expert matrices in one array: data,
//! block scales and `scale2` all slice cleanly on the outermost dimension.
//! Storing them separately is what makes a checkpoint shareable — a node
//! fetches the experts it holds rather than a 12 GiB slab it either has or does
//! not, deduplication works per expert, and a layer split across two machines
//! becomes a partition of blob handles rather than a file format problem.
//!
//! # Payload layout
//!
//! `[codes][block scales][global scale]`, each contiguous, in that order.
//! Lengths are implied by the header's dims and the element format, so nothing
//! records the boundaries: `codes` is `elems / 2`, `scales` is `elems / 16`,
//! and the global scale is the trailing four bytes.

use anyhow::Result;
use anybytes::Bytes;
use triblespace::core::blob::encodings::tensor::{
    elements::{NVFP4, NVFP4_BLOCK},
    tensor_blob, Tensor, TensorElement,
};
use triblespace::core::blob::Blob;

use super::load::PackedExpert;

/// One expert, packed, as a single self-contained blob.
///
/// The dims are LOGICAL — `[rows, cols * 2]` — because `cols` counts packed
/// bytes and two E2M1 values live in each. Writing the packed width here is
/// what makes a checkpoint need an `original_shape` field; writing the logical
/// width is what makes it unnecessary.
pub fn expert_blob(q: &PackedExpert) -> Result<Blob<Tensor<NVFP4, 2>>> {
    let logical = q.cols * 2;
    let elems = q.rows * logical;

    // Checked here rather than trusted, because the failure is silent: a
    // scales plane of the wrong length still decodes, just against the wrong
    // blocks, and produces numbers rather than an error.
    anyhow::ensure!(
        q.codes.len() == elems / 2,
        "codes are {} bytes, {} logical elements imply {}",
        q.codes.len(),
        elems,
        elems / 2
    );
    anyhow::ensure!(
        q.scales.len() == elems / NVFP4_BLOCK,
        "scales are {} bytes, {} logical elements in blocks of {NVFP4_BLOCK} imply {}",
        q.scales.len(),
        elems,
        elems / NVFP4_BLOCK
    );

    let mut payload = Vec::with_capacity(NVFP4::payload_len(elems));
    payload.extend_from_slice(&q.codes);
    payload.extend_from_slice(&q.scales);
    payload.extend_from_slice(&q.scale2.to_le_bytes());

    tensor_blob::<NVFP4, 2>(
        [q.rows as u64, logical as u64],
        Bytes::from_source(payload),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Split a blob's payload back into its three planes.
///
/// The inverse of [`expert_blob`]'s layout, offered so a consumer does not have
/// to re-derive the offsets and get them subtly wrong. Both sides compute the
/// boundaries from the same two facts — the element count and the block size —
/// so they cannot disagree.
pub fn split_payload(payload: &[u8], elems: usize) -> Result<(&[u8], &[u8], f32)> {
    let want = NVFP4::payload_len(elems);
    anyhow::ensure!(payload.len() == want, "payload is {} bytes, expected {want}", payload.len());
    let codes_len = elems / 2;
    let scales_len = elems / NVFP4_BLOCK;
    let codes = &payload[..codes_len];
    let scales = &payload[codes_len..codes_len + scales_len];
    let tail = &payload[codes_len + scales_len..];
    let scale2 = f32::from_le_bytes(tail.try_into().expect("four trailing bytes"));
    Ok((codes, scales, scale2))
}

/// Facts naming an expert blob.
///
/// The weight attribute is DERIVED per (element, rank) from one anchor, so
/// `Handle<Tensor<NVFP4, 2>>` and `Handle<Tensor<BF16, 3>>` are different
/// attributes with different ids. A query for packed rank-2 experts cannot
/// return a dense rank-3 tensor: the type is the query, not a convention the
/// caller has to remember.
pub mod attrs {
    use super::*;
    use triblespace::core::attribute::Attribute;
    use triblespace::core::id_hex;
    use triblespace::core::inline::encodings::hash::Handle;
    use triblespace::prelude::*;

    /// Anchor the weight attribute family derives from. Minted 2026-08-10.
    pub const WEIGHT_ANCHOR: Id = id_hex!("0B51DA3E67216213871743E045590DBC");

    /// The weight attribute for any element format and rank.
    ///
    /// One anchor yields a distinct id per `(element, rank)`, which is what
    /// makes the type the query. `weight_nvfp4_2` below is this same attribute
    /// spelled concretely — `attributes!` derives from `(anchor, encoding)`
    /// exactly as `Attribute::anchored` does, so the ids are identical — and it
    /// exists because `entity!` takes an attribute PATH rather than an
    /// expression.
    pub fn weight<T: TensorElement, const RANK: usize>(
    ) -> Attribute<Handle<Tensor<T, RANK>>> {
        Attribute::anchored(WEIGHT_ANCHOR)
    }

    attributes! {
        /// A packed rank-2 expert. Same anchor as [`weight`], so this is that
        /// attribute at `(NVFP4, 2)` and not a second one beside it.
        "0B51DA3E67216213871743E045590DBC" as weight_nvfp4_2:
            inlineencodings::Handle<Tensor<NVFP4, 2>>;
        // The checkpoint tensor name lives in `metadata::name` as a LongString
        // handle, not here. It was a ShortString attribute until a real name —
        // `model.llm.layers.10.mlp.experts.w13_weight`, 42 characters — panicked
        // the encoder, which answers a too-long value with unwrap() rather than
        // an error. Two copies of one string, and the redundant one was the copy
        // that could not hold it.
        /// Which expert of the stacked matrix.
        "A6ED6DBA4BE63E4E34F2787DA84AD860" as expert_index: inlineencodings::I256BE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace::core::blob::TryFromBlob;
    use triblespace::core::blob::encodings::tensor::TensorView;

    /// A synthetic expert with the real checkpoint's proportions, scaled down.
    /// `PackedExpert` is a plain struct, so this needs no checkpoint.
    fn expert(rows: usize, logical: usize) -> PackedExpert {
        let elems = rows * logical;
        PackedExpert {
            codes: (0..elems / 2).map(|i| i as u8).collect(),
            scales: (0..elems / NVFP4_BLOCK).map(|i| (i % 251) as u8).collect(),
            scale2: 0.125,
            rows,
            cols: logical / 2,
        }
    }

    /// The blob states the LOGICAL width, so nothing downstream needs an
    /// `original_shape` field to correct it.
    #[test]
    fn the_blob_states_logical_dims_not_packed_ones() {
        let q = expert(64, 128);
        assert_eq!(q.cols, 64, "the checkpoint's packed width");
        let blob = expert_blob(&q).expect("well formed");
        let view: TensorView = blob.try_from_blob().expect("decodes");
        assert_eq!(view.dims(), &[64, 128], "logical, twice the packed width");
        assert_eq!(view.elems(), 64 * 128);
    }

    /// Codes, scales and the global scale survive as one atomic artifact —
    /// which is the point, since the checkpoint binds them only by name.
    #[test]
    fn all_three_planes_round_trip_through_one_handle() {
        let q = expert(64, 128);
        let blob = expert_blob(&q).expect("well formed");
        let view: TensorView = blob.try_from_blob().expect("decodes");
        let (codes, scales, scale2) =
            split_payload(view.payload(), view.elems()).expect("splits");
        assert_eq!(codes, &q.codes[..], "codes");
        assert_eq!(scales, &q.scales[..], "block scales");
        assert_eq!(scale2, q.scale2, "global scale");
    }

    /// A scales plane of the wrong length still decodes — against the wrong
    /// blocks — so it has to be refused rather than discovered later as numbers
    /// that look plausible.
    #[test]
    fn a_mis_sized_scales_plane_is_refused() {
        let mut q = expert(64, 128);
        q.scales.truncate(q.scales.len() - 1);
        let err = expert_blob(&q).expect_err("must refuse");
        assert!(format!("{err}").contains("scales are"), "{err}");
    }

    #[test]
    fn mis_sized_codes_are_refused_too() {
        let mut q = expert(64, 128);
        q.codes.push(0);
        let err = expert_blob(&q).expect_err("must refuse");
        assert!(format!("{err}").contains("codes are"), "{err}");
    }

    /// Inkling's real proportions: a 4096x4096 expert packs to 4096x2048 bytes
    /// with 4096x256 block scales, which is what the checkpoint stores.
    #[test]
    fn the_real_expert_proportions_line_up() {
        let (rows, logical) = (4096usize, 4096usize);
        let elems = rows * logical;
        assert_eq!(elems / 2, 4096 * 2048, "packed width matches the checkpoint");
        assert_eq!(elems / NVFP4_BLOCK, 4096 * 256, "scale width matches");
        assert_eq!(NVFP4::payload_len(elems), elems / 2 + elems / NVFP4_BLOCK + 4);
    }
}
