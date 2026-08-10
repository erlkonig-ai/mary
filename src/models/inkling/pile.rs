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
        /// Which transformer layer it belongs to.
        ///
        /// Stored as a fact rather than parsed out of the tensor name at read
        /// time, because splitting a model across machines is a QUERY — "give
        /// me layers 0..21" — and a query over a string you have to parse is
        /// not one.
        "BCDDFBCFF89F67EE0B1E527C4872CED7" as layer: inlineencodings::I256BE;
    }
}

/// One expert in a pile, named but NOT loaded.
///
/// A handle, not bytes. Selecting which experts a machine holds must not depend
/// on reading them: a 21/21 layer split is a decision about ~5,000 handles, and
/// materialising even one of them to make it would defeat the split.
#[derive(Debug, Clone, Copy)]
pub struct ExpertRef {
    pub layer: i64,
    pub expert: i64,
    pub handle: triblespace::prelude::Inline<
        triblespace::core::inline::encodings::hash::Handle<Tensor<NVFP4, 2>>,
    >,
}

/// Every expert whose layer falls in `range`, as handles.
///
/// This is what makes splitting a model across machines a QUERY. A node asks
/// for the layers it holds and gets references; nothing is read until something
/// is actually computed. The weight attribute is typed per (element, rank), so
/// this cannot return a dense tensor by accident — a BF16 rank-3 weight is a
/// different attribute and simply does not match.
pub fn experts_in_layers(
    space: &triblespace::prelude::TribleSet,
    range: std::ops::RangeInclusive<i64>,
) -> Vec<ExpertRef> {
    use triblespace::macros::pattern;
    let mut out: Vec<ExpertRef> = triblespace::macros::find!(
        (layer: i64, expert: i64, handle: triblespace::prelude::Inline<
            triblespace::core::inline::encodings::hash::Handle<Tensor<NVFP4, 2>>>),
        pattern!(space, [{ _?e @
            attrs::layer: ?layer,
            attrs::expert_index: ?expert,
            attrs::weight_nvfp4_2: ?handle
        }])
    )
    .filter(|(layer, _, _)| range.contains(layer))
    .map(|(layer, expert, handle)| ExpertRef { layer, expert, handle })
    .collect();
    out.sort_by_key(|r| (r.layer, r.expert));
    out
}

/// The layer a checkpoint tensor name belongs to.
///
/// `model.llm.layers.10.mlp.experts.w13_weight` is layer 10. Returns None for
/// names carrying no layer — the embedding, the final norm — rather than
/// guessing, so a tensor with no layer is visibly absent from a layer query
/// instead of silently landing in layer 0.
pub fn layer_of(tensor_name: &str) -> Option<i64> {
    let rest = tensor_name.split("layers.").nth(1)?;
    rest.split('.').next()?.parse().ok()
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

    #[test]
    fn a_layer_is_read_from_the_name_and_absent_when_there_is_none() {
        assert_eq!(layer_of("model.llm.layers.10.mlp.experts.w13_weight"), Some(10));
        assert_eq!(layer_of("model.llm.layers.0.mlp.w13_dn"), Some(0));
        assert_eq!(layer_of("model.mtp.layers.3.attn.wq_du"), Some(3));
        // No layer at all: absent, not zero. A tensor that silently joined
        // layer 0 would ship to the wrong machine in a 21/21 split.
        assert_eq!(layer_of("model.llm.embed"), None);
        assert_eq!(layer_of("model.audio.encoder.weight"), None);
    }

    /// A layer query returns only the layers asked for. The negative half
    /// matters more than the positive one: a node that received experts from a
    /// layer it does not hold would compute with weights it has no business
    /// having, and the arithmetic would look fine.
    #[test]
    fn a_layer_query_excludes_the_layers_it_did_not_ask_for() {
        use triblespace::macros::entity;
        use triblespace::prelude::*;

        let mut space = TribleSet::new();
        for (layer, idx) in [(3i64, 0i64), (3, 1), (30, 0)] {
            // A distinct handle per row, so nothing collapses by accident.
            let mut q = expert(16, 32);
            q.scale2 = layer as f32 + idx as f32 / 100.0;
            let handle = expert_blob(&q).expect("well formed").get_handle();
            space += entity! { &ufoid() @
                attrs::layer: layer,
                attrs::expert_index: idx,
                attrs::weight_nvfp4_2: handle,
            }
            .into_facts();
        }

        let held = experts_in_layers(&space, 0..=20);
        assert_eq!(held.len(), 2, "two experts in layer 3");
        assert!(held.iter().all(|r| r.layer == 3), "layer 30 must not appear");
        assert_eq!(held[0].expert, 0, "and they come back ordered");
        assert_eq!(held[1].expert, 1);

        assert_eq!(experts_in_layers(&space, 21..=41).len(), 1, "the other half");
        assert_eq!(experts_in_layers(&space, 100..=200).len(), 0, "and an empty range is empty");
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
