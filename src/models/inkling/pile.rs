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
    elements::{BF16, F32, NVFP4, NVFP4_BLOCK},
    tensor_blob, Tensor, TensorElement, TensorView,
};
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::prelude::BlobStoreGet;

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
        ///
        /// DELIBERATELY the ANCHORED arm (`as`, not `unsafe as`). Every other
        /// minted id in this crate is pinned, because a pinned id is a promise
        /// that data on disk stays reachable. This one is the exception: its
        /// entire purpose is to COINCIDE with `weight::<NVFP4, 2>()`, which is
        /// `Attribute::anchored` and therefore derives. Pin it and the two stop
        /// being the same attribute — the importer writes experts under the
        /// literal while every generic reader looks under the derived id and
        /// finds nothing.
        ///
        /// That is not hypothetical. A bulk pass on 2026-08-11 converted all 52
        /// minted ids to `unsafe as` to repair genuine drift, and swept this one
        /// up with them. Caught before 144 GiB of experts were written under an
        /// id no reader would have asked for. The invariant is asserted below.
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

/// Which element format an expert leaf holds, and its handle.
///
/// Two variants because the weight attribute is DERIVED per (element, rank):
/// `Handle<Tensor<NVFP4, 2>>` and `Handle<Tensor<BF16, 2>>` are different
/// attributes with different ids, so "every expert" is two queries and the
/// answer has to be able to say which one it came from. That is the type being
/// the query, paid for at the one place it costs anything.
#[derive(Debug, Clone, Copy)]
pub enum ExpertHandle {
    Nvfp4(
        triblespace::prelude::Inline<
            triblespace::core::inline::encodings::hash::Handle<Tensor<NVFP4, 2>>,
        >,
    ),
    Bf16(
        triblespace::prelude::Inline<
            triblespace::core::inline::encodings::hash::Handle<Tensor<BF16, 2>>,
        >,
    ),
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
    pub handle: ExpertHandle,
}

/// Every expert whose layer falls in `range`, as handles — BOTH element formats.
///
/// This is what makes splitting a model across machines a QUERY. A node asks
/// for the layers it holds and gets references; nothing is read until something
/// is actually computed.
///
/// It sweeps NVFP4 **and** BF16 because Inkling-Small's layer 2 is the odd one
/// out: its experts have no `.scale` sidecar in the checkpoint and land in the
/// pile as `Tensor<BF16, 2>`. A packed-only query is not a filter over that
/// layer, it is a hole — a node told it holds layers 0..=20 would receive every
/// expert of nineteen layers and none of layer 2, compute anyway, and be wrong
/// in exactly one fortieth of the model. Which of the two a leaf is stays in the
/// answer (see [`ExpertHandle`]) rather than being re-derived from a name.
pub fn experts_in_layers(
    space: &triblespace::prelude::TribleSet,
    range: std::ops::RangeInclusive<i64>,
) -> Vec<ExpertRef> {
    use triblespace::core::inline::encodings::hash::Handle;
    use triblespace::macros::pattern;
    use triblespace::prelude::Inline;

    let mut out: Vec<ExpertRef> = Vec::new();
    for (layer, expert, handle) in triblespace::macros::find!(
        (layer: i64, expert: i64, handle: Inline<Handle<Tensor<NVFP4, 2>>>),
        pattern!(space, [{ _?e @
            attrs::layer: ?layer,
            attrs::expert_index: ?expert,
            attrs::weight_nvfp4_2: ?handle
        }])
    ) {
        if range.contains(&layer) {
            out.push(ExpertRef { layer, expert, handle: ExpertHandle::Nvfp4(handle) });
        }
    }
    for (layer, expert, handle) in triblespace::macros::find!(
        (layer: i64, expert: i64, handle: Inline<Handle<Tensor<BF16, 2>>>),
        pattern!(space, [{ _?e @
            attrs::layer: ?layer,
            attrs::expert_index: ?expert,
            attrs::weight::<BF16, 2>(): ?handle
        }])
    ) {
        if range.contains(&layer) {
            out.push(ExpertRef { layer, expert, handle: ExpertHandle::Bf16(handle) });
        }
    }
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

// ---------------------------------------------------------------------------
// Reading the model back OUT
// ---------------------------------------------------------------------------

/// Which element format a leaf turned out to hold.
///
/// The type parameter is gone by the time a leaf sits in a by-name index — the
/// model spans two dtypes and five ranks — so the fact travels as data instead.
/// This is erasure done ONCE, at the index boundary, from reads that were each
/// typed: a leaf was fetched as `Tensor<BF16, 2>` or not at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    Bf16,
    F32,
}

impl Elem {
    /// Bytes one element occupies on disk.
    pub fn width(self) -> usize {
        match self {
            Elem::Bf16 => 2,
            Elem::F32 => 4,
        }
    }
}

/// One tensor of the model, resolved: its dims from the blob header, its bytes
/// as a VIEW over the pile's mapping.
///
/// Not a `Vec`. `Bytes` here is `Bytes::from_raw_parts(slice, mmap.clone())` —
/// the pile's own mapping with an `Arc` keeping it alive — so an index over the
/// whole model costs handles and headers, never weights, and a lane that hands
/// the bytes to a GPU hands over the mapping itself.
#[derive(Clone)]
pub struct Leaf {
    pub elem: Elem,
    pub dims: Vec<u64>,
    pub bytes: anybytes::Bytes,
    /// Which transformer layer this tensor belongs to, when it belongs to one.
    ///
    /// A FACT the importer wrote, not a substring of the name — which is what
    /// makes "give me layers 0..=19" a query. `None` for the embedding, the
    /// final norm and the unembedding: absent rather than zero, because a
    /// tensor that silently joined layer 0 would ship to the wrong machine.
    pub layer: Option<i64>,
}

impl Leaf {
    /// Shape as the `Vec<usize>` the loaders speak.
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().map(|&d| d as usize).collect()
    }

    /// How many elements. From the dims, not from the byte length.
    pub fn elems(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    /// Widen to f32 — the ONE conversion, made explicit and made the caller's.
    ///
    /// [`crate::models::inkling::load::Checkpoint::tensor`] does this on every
    /// read because a safetensors reader has nothing else to hand back; here the
    /// stored form is reachable, so widening is a thing a caller ASKS for when
    /// it is about to compute in f32, and a device lane that wants the bytes
    /// takes [`Leaf::bytes`] instead.
    pub fn to_f32(&self) -> Vec<f32> {
        match self.elem {
            Elem::Bf16 => self
                .bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            Elem::F32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }
}

/// One expert's packed NVFP4 weight, read out of the pile.
///
/// The three planes are `Bytes` slices of ONE blob, which is what the pile
/// format made atomic: the checkpoint binds `w13_weight`, `.scale` and
/// `.scale2` by naming convention across three different shards, and a reader
/// holding only the first has bytes it cannot interpret. Here there is one
/// handle, and the planes are offsets inside it that both sides compute from the
/// same two facts (see [`split_payload`]).
pub struct PackedSlab {
    pub codes: anybytes::Bytes,
    pub scales: anybytes::Bytes,
    pub scale2: f32,
    /// Output rows of this expert's matrix.
    pub rows: usize,
    /// Packed bytes per row; the logical width is `2 * cols`.
    pub cols: usize,
}

/// A model located in a pile: every tensor found, nothing widened, nothing
/// copied.
///
/// The whole reader is two hash maps built once at open. There is no shard
/// index, no header cache, no mapping cache and no span table, because the
/// questions those answer — which file is this tensor in, where in it, is the
/// header parsed yet — do not exist for a content-addressed store. A handle IS
/// the location.
pub struct PileSource {
    reader: triblespace::core::repo::pile::PileReader,
    /// Everything the branch asserts, kept rather than dropped after the index
    /// is built.
    ///
    /// It costs a few MB against a 159 GiB model and it is what makes the pile
    /// AUTHORITATIVE rather than merely sufficient for weights: the checkpoint's
    /// `config.json` and its siblings live here as facts (see
    /// [`crate::jsonfacts`]), and a runtime that had to reopen the pile to read
    /// them would pay the 18-second index build twice to answer a question the
    /// first open already had in hand.
    facts: triblespace::prelude::TribleSet,
    /// Dense tensors, read as their type at index time. `Leaf` is a view, so
    /// holding all 968 of them costs kilobytes.
    dense: std::collections::HashMap<String, Leaf>,
    /// Experts, as HANDLES: 20 480 of them, and reading even one to build the
    /// index would be 7 MiB of BLAKE3 for a lookup table.
    experts: std::collections::HashMap<(String, i64), ExpertRef>,
    /// How many experts each stacked matrix name has — from the facts, so a
    /// caller never infers a count from an error.
    stacked: std::collections::HashMap<String, usize>,
}

impl PileSource {
    /// Open a pile and resolve every tensor of the model on `branch`.
    ///
    /// Reads the dense leaves (their headers and their content hashes; the
    /// payloads stay in the mapping) and takes the experts as handles.
    pub fn open(path: &std::path::Path, branch: &str) -> Result<Self> {
        use triblespace::core::inline::encodings::hash::Handle;
        use triblespace::core::metadata;
        use triblespace::core::repo::{ancestors, Repository};
        use triblespace::macros::{find, pattern};
        use triblespace::prelude::*;

        let mut pile = Pile::open(path).map_err(|e| anyhow::anyhow!("open {path:?}: {e:?}"))?;
        // Read path: never amputate. A torn tail is an operator decision.
        pile.refresh()
            .map_err(|e| anyhow::anyhow!("load {path:?}: {e:?}"))?;
        let mut repo = Repository::new(
            pile,
            ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng),
            TribleSet::new(),
        )
        .map_err(|e| anyhow::anyhow!("repo: {e:?}"))?;
        let branch_id = repo
            .lookup_branch(branch)
            .map_err(|e| anyhow::anyhow!("lookup {branch}: {e:?}"))?
            .ok_or_else(|| anyhow::anyhow!("no {branch:?} branch in {path:?}"))?;
        let mut ws = repo
            .pull(branch_id)
            .map_err(|e| anyhow::anyhow!("pull: {e:?}"))?;
        let head = ws
            .head()
            .ok_or_else(|| anyhow::anyhow!("{branch:?} has no commits"))?;
        let facts: TribleSet = ws
            .checkout(ancestors(head))
            .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
            .facts()
            .clone();
        let reader = repo
            .storage_mut()
            .reader()
            .map_err(|e| anyhow::anyhow!("reader: {e:?}"))?;
        repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;

        // ── the experts, as handles ─────────────────────────────────────────
        // First, because what it produces is also what tells the dense sweep
        // which entities are NOT dense. An expert entity carries an
        // `expert_index`; a dense one does not. That is the distinction, and it
        // is a FACT rather than a substring test on the name — which matters,
        // because all 256 experts of one matrix share one name and a dense map
        // built without the distinction would hold whichever of them the query
        // happened to yield last.
        let mut experts = std::collections::HashMap::new();
        let mut stacked: std::collections::HashMap<String, usize> = Default::default();
        let mut expert_ids: std::collections::HashSet<Id> = Default::default();
        macro_rules! sweep_experts {
            ($ty:ty, $attr:expr, $wrap:expr) => {{
                for (e, n, i, l, h) in find!(
                    (e: Id,
                     n: Inline<Handle<blobencodings::LongString>>,
                     i: i64,
                     l: i64,
                     h: Inline<Handle<Tensor<$ty, 2>>>),
                    pattern!(&facts, [
                        { ?e @ metadata::name: ?n, attrs::expert_index: ?i,
                          attrs::layer: ?l, $attr: ?h },
                    ])
                ) {
                    let name: anybytes::View<str> = reader
                        .get(n)
                        .map_err(|err| anyhow::anyhow!("expert name blob: {err:?}"))?;
                    let name = name.to_string();
                    let c = stacked.entry(name.clone()).or_insert(0);
                    *c = (*c).max(i as usize + 1);
                    expert_ids.insert(e);
                    experts.insert(
                        (name, i),
                        ExpertRef { layer: l, expert: i, handle: $wrap(h) },
                    );
                }
            }};
        }
        sweep_experts!(NVFP4, attrs::weight_nvfp4_2, ExpertHandle::Nvfp4);
        sweep_experts!(BF16, attrs::weight::<BF16, 2>(), ExpertHandle::Bf16);

        // ── the dense tensors, by name ──────────────────────────────────────
        // One query per (element, rank) — ten, not one — because that is what
        // typing the attribute means, and each hit is read AS its type. Nothing
        // is interpreted without one, so a BF16 matrix cannot arrive where f32
        // was asked for.
        let mut dense = std::collections::HashMap::new();
        macro_rules! sweep_dense {
            ($ty:ty, $rank:literal, $tag:expr) => {{
                for (e, n, h) in find!(
                    (e: Id,
                     n: Inline<Handle<blobencodings::LongString>>,
                     h: Inline<Handle<Tensor<$ty, $rank>>>),
                    pattern!(&facts, [
                        { ?e @ metadata::name: ?n, attrs::weight::<$ty, $rank>(): ?h },
                    ])
                ) {
                    if expert_ids.contains(&e) {
                        continue;
                    }
                    let name: anybytes::View<str> = reader
                        .get(n)
                        .map_err(|err| anyhow::anyhow!("name blob: {err:?}"))?;
                    let blob: Blob<Tensor<$ty, $rank>> = reader
                        .get(h)
                        .map_err(|err| anyhow::anyhow!("{}: leaf blob: {err:?}", &*name))?;
                    let view: TensorView = TensorView::try_from_blob(blob)
                        .map_err(|err| anyhow::anyhow!("{}: decode: {err}", &*name))?;
                    // The layer is optional in the graph, so it is optional
                    // here: an `exists!` rather than a second required clause,
                    // which would silently drop the embedding and the head.
                    let layer = find!(
                        (l: i64),
                        pattern!(&facts, [{ (e) @ attrs::layer: ?l }])
                    )
                    .next()
                    .map(|(l,)| l);
                    dense.insert(
                        name.to_string(),
                        Leaf {
                            elem: $tag,
                            dims: view.dims().to_vec(),
                            bytes: view.payload().clone(),
                            layer,
                        },
                    );
                }
            }};
        }
        sweep_dense!(BF16, 0, Elem::Bf16);
        sweep_dense!(BF16, 1, Elem::Bf16);
        sweep_dense!(BF16, 2, Elem::Bf16);
        sweep_dense!(BF16, 3, Elem::Bf16);
        sweep_dense!(BF16, 4, Elem::Bf16);
        sweep_dense!(F32, 0, Elem::F32);
        sweep_dense!(F32, 1, Elem::F32);
        sweep_dense!(F32, 2, Elem::F32);
        sweep_dense!(F32, 3, Elem::F32);
        sweep_dense!(F32, 4, Elem::F32);

        anyhow::ensure!(!dense.is_empty(), "{path:?}: no dense leaves on {branch:?}");
        Ok(PileSource { reader, facts, dense, experts, stacked })
    }

    /// One dense tensor by checkpoint name, as a view.
    pub fn leaf(&self, name: &str) -> Result<&Leaf> {
        self.dense
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{name} is not in the pile"))
    }

    /// Every dense tensor name, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.dense.keys().cloned().collect();
        v.sort();
        v
    }

    /// How many tensors this source located — dense leaves plus experts.
    pub fn len(&self) -> usize {
        self.dense.len() + self.experts.len()
    }

    /// Dense leaves in the index.
    pub fn dense_len(&self) -> usize {
        self.dense.len()
    }

    /// Expert leaves in the index — each expert of each stack, individually.
    pub fn expert_len(&self) -> usize {
        self.experts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many experts a stacked matrix holds.
    pub fn expert_count(&self, base: &str) -> Result<usize> {
        self.stacked
            .get(base)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("{base} is not a stacked expert matrix in this pile"))
    }

    /// Whether a stacked matrix's experts are packed NVFP4 rather than BF16.
    ///
    /// Answered by the ATTRIBUTE the leaves were written under, not by probing
    /// for a `.scale` sidecar's existence. The checkpoint has to ask "is there a
    /// tensor with this name plus `.scale`?", which is a question about a naming
    /// convention; here the element format is part of the leaf's identity.
    pub fn is_nvfp4(&self, base: &str) -> bool {
        matches!(
            self.experts.get(&(base.to_string(), 0)),
            Some(ExpertRef { handle: ExpertHandle::Nvfp4(_), .. })
        )
    }

    /// Fault a LAYER RANGE's whole share into memory, and say what it cost.
    ///
    /// The deployment question a split has to answer is not how fast a pass is
    /// but whether a node's share STAYS resident, and that needs the share
    /// named without reading the model to find it. Here it is a query over the
    /// layer facts — dense leaves and experts alike, both element formats — so
    /// the thing warmed is exactly the thing
    /// [`experts_in_layers`] would hand a node.
    ///
    /// Returns `(bytes touched, leaves touched)`.
    pub fn warm(&self, lo: i64, hi: i64) -> Result<(u64, usize)> {
        let (mut bytes, mut leaves) = (0u64, 0usize);
        let mut sum = 0u64;
        for leaf in self
            .dense
            .values()
            .filter(|l| matches!(l.layer, Some(x) if x >= lo && x <= hi))
        {
            // One byte per 4 KiB page: the kernel's readahead turns a
            // forward-sequential fault pattern into large sequential reads.
            let b: &[u8] = &leaf.bytes;
            let mut i = 0usize;
            while i < b.len() {
                sum = sum.wrapping_add(b[i] as u64);
                i += 4096;
            }
            bytes += b.len() as u64;
            leaves += 1;
        }
        // One byte per 4 KiB, exactly as the dense arm above, and for the same
        // reason: these pages have to be FAULTED, not merely named.
        //
        // This used to resolve the handle and add its length, on the belief that
        // reading a blob validates it and so touches every page. It does not —
        // the reader hands back a VIEW over the pile's mapping, and a view costs
        // nothing to make. The counter therefore reported bytes nobody had read,
        // and the giveaway was in the number it produced: a second pass claiming
        // 155.81 GiB "touched" in 3.3 s is 47 GB/s, which is not a memory
        // bandwidth this part has and certainly not a disk one. A residency
        // instrument that reports a working set it never faulted answers the one
        // question it exists to answer with a number that cannot be wrong,
        // which is worse than not asking.
        for ((name, idx), r) in self.experts.iter() {
            if r.layer < lo || r.layer > hi {
                continue;
            }
            let mut touch = |b: &[u8]| {
                let mut i = 0usize;
                while i < b.len() {
                    sum = sum.wrapping_add(b[i] as u64);
                    i += 4096;
                }
                b.len() as u64
            };
            let n = match r.handle {
                ExpertHandle::Nvfp4(_) => {
                    let q = self.expert_packed(name, *idx as usize)?;
                    touch(&q.codes) + touch(&q.scales)
                }
                ExpertHandle::Bf16(_) => touch(&self.expert_bf16(name, *idx as usize)?.bytes),
            };
            bytes += n;
            leaves += 1;
        }
        std::hint::black_box(sum);
        Ok((bytes, leaves))
    }

    /// One expert's NVFP4 planes, read out of the pile and **not decoded**.
    pub fn expert_packed(&self, base: &str, e: usize) -> Result<PackedSlab> {
        let h = match self.experts.get(&(base.to_string(), e as i64)).map(|r| r.handle) {
            Some(ExpertHandle::Nvfp4(h)) => h,
            Some(ExpertHandle::Bf16(_)) => {
                anyhow::bail!("{base}[{e}] is BF16, not packed NVFP4")
            }
            None => anyhow::bail!("{base}[{e}] is not in the pile"),
        };
        let blob: Blob<Tensor<NVFP4, 2>> = self
            .reader
            .get(h)
            .map_err(|err| anyhow::anyhow!("{base}[{e}]: {err:?}"))?;
        let view: TensorView = TensorView::try_from_blob(blob)
            .map_err(|err| anyhow::anyhow!("{base}[{e}]: decode: {err}"))?;
        let dims = view.dims();
        anyhow::ensure!(dims.len() == 2, "{base}[{e}] is rank {}", dims.len());
        let (rows, logical) = (dims[0] as usize, dims[1] as usize);
        let elems = rows * logical;
        let payload = view.payload();
        // The boundaries are derived, here, from the element count and the
        // block size — the same two facts the writer used. Nothing on disk
        // records them, so the two sides cannot disagree about them.
        let codes_len = elems / 2;
        let scales_len = elems / NVFP4_BLOCK;
        let (_, _, scale2) = split_payload(payload, elems)?;
        Ok(PackedSlab {
            codes: payload.slice(..codes_len),
            scales: payload.slice(codes_len..codes_len + scales_len),
            scale2,
            rows,
            cols: logical / 2,
        })
    }

    /// The pile's mapping, as `(base, len, keepalive)` — a list of ONE.
    ///
    /// A pile is one file, so a zero-copy lane registers it once and every slab
    /// afterwards is offset arithmetic. The checkpoint's answer is nine shards,
    /// and the only reason that number is not one is that safetensors has a
    /// 2 GiB-ish practical shard ceiling and a 159 GiB model does not fit in it.
    ///
    /// Recovered from a leaf rather than stored: the pile hands out
    /// `Bytes::from_raw_parts(slice, mmap.clone())`, so the mapping IS the
    /// owner of every payload and asking a payload for its owner is exact. A
    /// second `mmap` of the same file would be a different address range and
    /// every offset computed against it would be silently wrong.
    pub fn mappings(&self) -> Result<Vec<(usize, usize, std::sync::Arc<dyn std::any::Any + Send + Sync>)>> {
        let any = self
            .dense
            .values()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no leaves to recover the mapping from"))?;
        let map: std::sync::Arc<memmap2::MmapRaw> = any
            .bytes
            .clone()
            .downcast_to_owner()
            .map_err(|_| anyhow::anyhow!("a pile leaf is not backed by the pile's mapping"))?;
        Ok(vec![(
            map.as_ptr() as usize,
            map.len(),
            map as std::sync::Arc<dyn std::any::Any + Send + Sync>,
        )])
    }

    /// Everything the branch asserts.
    pub fn facts(&self) -> &triblespace::prelude::TribleSet {
        &self.facts
    }

    /// The blob reader, so a caller can resolve handles the facts name.
    pub fn reader(&self) -> &triblespace::core::repo::pile::PileReader {
        &self.reader
    }

    /// Every `(stacked matrix name, expert index)` this pile holds, sorted.
    ///
    /// The index is already built at open, so this is a rename of what is in
    /// memory rather than a query. It exists because an AUDIT has to be driven
    /// by what the pile actually contains — asking it for the experts a layer
    /// range implies would make a leaf nobody indexed invisible to the audit by
    /// construction.
    pub fn expert_keys(&self) -> Vec<(String, i64)> {
        let mut v: Vec<(String, i64)> = self.experts.keys().cloned().collect();
        v.sort();
        v
    }

    /// Whether one expert leaf is packed NVFP4 (rather than BF16).
    pub fn expert_is_nvfp4(&self, base: &str, e: i64) -> Option<bool> {
        self.experts
            .get(&(base.to_string(), e))
            .map(|r| matches!(r.handle, ExpertHandle::Nvfp4(_)))
    }

    /// One expert of a BF16 stack, as a view.
    pub fn expert_bf16(&self, base: &str, e: usize) -> Result<Leaf> {
        let h = match self.experts.get(&(base.to_string(), e as i64)).map(|r| r.handle) {
            Some(ExpertHandle::Bf16(h)) => h,
            Some(ExpertHandle::Nvfp4(_)) => {
                anyhow::bail!("{base}[{e}] is packed NVFP4, not BF16")
            }
            None => anyhow::bail!("{base}[{e}] is not in the pile"),
        };
        let blob: Blob<Tensor<BF16, 2>> = self
            .reader
            .get(h)
            .map_err(|err| anyhow::anyhow!("{base}[{e}]: {err:?}"))?;
        let view: TensorView = TensorView::try_from_blob(blob)
            .map_err(|err| anyhow::anyhow!("{base}[{e}]: decode: {err}"))?;
        Ok(Leaf {
            elem: Elem::Bf16,
            dims: view.dims().to_vec(),
            bytes: view.payload().clone(),
            layer: self
                .experts
                .get(&(base.to_string(), e as i64))
                .map(|r| r.layer),
        })
    }
}

#[cfg(test)]
mod anchor_tests {
    use super::attrs;
    use triblespace::core::blob::encodings::tensor::elements::NVFP4;

    /// `weight_nvfp4_2` and `weight::<NVFP4, 2>()` must be ONE attribute.
    ///
    /// They are declared two different ways — one through `attributes!`, one
    /// through `Attribute::anchored` — and nothing but this test keeps them
    /// equal. A single `unsafe` keyword on the declaration silently separates
    /// them, and the symptom is an importer and a reader that disagree about
    /// where the weights are, with no error from either side.
    #[test]
    fn nvfp4_expert_attribute_matches_the_generic() {
        assert_eq!(
            attrs::weight_nvfp4_2.id(),
            attrs::weight::<NVFP4, 2>().id(),
            "weight_nvfp4_2 must be weight::<NVFP4,2>(); if this fails, check \
             whether the declaration was changed to `unsafe as`"
        );
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
