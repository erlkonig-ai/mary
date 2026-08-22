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

use anyhow::{Context, Result};
use anybytes::Bytes;
use triblespace::core::blob::encodings::tensor::{
    elements::{BF16, F32, NVFP4, NVFP4_BLOCK},
    tensor_blob, Tensor, TensorElement, TensorView,
};
use triblespace::core::blob::{Blob, TryFromBlob};
use triblespace::prelude::BlobStoreGet;

use super::load::PackedExpert;

/// One line of the kernel's memory accounting, for a startup that competes with
/// the GPU for ONE pool.
///
/// On a unified-memory part the anonymous startup copy, the pile's page cache
/// and everything CUDA reserves are the same 121 GiB. A timer cannot say which
/// of them ran out, and `MemAvailable` alone cannot either: clean page cache
/// counts as available right up to the moment something needs it and the
/// reclaim has to happen synchronously. So print all four.
pub fn mem_line(label: &str) -> String {
    let mut free = 0u64;
    let mut avail = 0u64;
    let mut cached = 0u64;
    let mut anon = 0u64;
    let mut swapfree = 0u64;
    let mut swaptotal = 0u64;
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        let kb = |l: &str| -> u64 {
            l.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0)
        };
        for l in s.lines() {
            if l.starts_with("MemFree:") {
                free = kb(l);
            } else if l.starts_with("MemAvailable:") {
                avail = kb(l);
            } else if l.starts_with("Cached:") {
                cached = kb(l);
            } else if l.starts_with("AnonPages:") {
                anon = kb(l);
            } else if l.starts_with("SwapFree:") {
                swapfree = kb(l);
            } else if l.starts_with("SwapTotal:") {
                swaptotal = kb(l);
            }
        }
    }
    let mut rss = 0u64;
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for l in s.lines() {
            if l.starts_with("VmRSS:") {
                rss = l.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
            }
        }
    }
    let g = |kb: u64| kb as f64 / (1u64 << 20) as f64;
    format!(
        "    mem[{label}]: rss {:.1} GiB, anon {:.1}, free {:.1}, available {:.1}, cached {:.1}, swap used {:.1}",
        g(rss),
        g(anon),
        g(free),
        g(avail),
        g(cached),
        g(swaptotal.saturating_sub(swapfree)),
    )
}

/// Hands each leaf's source pages back to the kernel once it has been copied
/// into the anonymous arena.
///
/// Two calls, and both are needed, because they undo different things.
/// `madvise(MADV_DONTNEED)` zaps this process's page-table entries, which takes
/// the pages out of RSS but leaves them in the page cache; `posix_fadvise(...,
/// POSIX_FADV_DONTNEED)` drops the cache pages themselves, and it only works on
/// pages nobody has mapped -- which is why the madvise has to come first. The
/// pile is mapped shared and read-only, so neither is destructive: a later read
/// of the same bytes re-faults them from the file.
///
/// What this buys is the thing the startup copy is actually fighting for. On a
/// unified-memory part the source page cache, the anonymous arena and
/// everything CUDA reserves are ONE 121 GiB pool, and reclaim gets to choose
/// between them. At `INK_LAYERS=0:20` it chose wrong twice over: without the
/// madvise it held 36 GiB of source pages in RSS and paged 16 GiB of freshly
/// written arena out to swap instead; with the madvise but without the fadvise
/// it still kept the unmapped cache and started swapping again as soon as the
/// share passed ~85 GiB. Handing the pages back as each leaf lands takes the
/// choice away.
///
/// `INK_RELEASE_SOURCE=0` keeps the pages, as the A/B arm.
struct SourceRelease {
    /// The pile file, reopened read-only for `posix_fadvise` alone. The mapping
    /// does not keep an fd and the advice needs one.
    file: Option<std::fs::File>,
    /// Base address of the pile's mapping, so a payload pointer becomes a file
    /// offset by subtraction. `None` when the mapping is not the whole file, in
    /// which case that subtraction would be a guess.
    map_base: Option<usize>,
    /// The file's length, so the advice never names a range past its end.
    file_len: usize,
    page: usize,
    on: bool,
}

impl SourceRelease {
    #[cfg(target_os = "linux")]
    fn new(path: &std::path::Path, map_base: usize, map_len: usize) -> Self {
        let on = !std::env::var("INK_RELEASE_SOURCE").map(|v| v == "0").unwrap_or(false);
        // SAFETY: `sysconf` reads a static system parameter.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = if page > 0 { page as usize } else { 4096 };
        let file = std::fs::File::open(path).ok();
        // The mapping starts at file offset 0, so a payload pointer minus the
        // base IS the file offset -- but the pile RESERVES address space it has
        // not filled (256 GiB of mapping over a 171 GiB file, so appends need no
        // remap), and the reservation is exactly why this cannot be an equality
        // test. What has to hold is that the file is no LONGER than the map,
        // and that the advice never runs past the file's end.
        let file_len = file
            .as_ref()
            .and_then(|f| f.metadata().ok())
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let whole = file_len > 0 && file_len <= map_len;
        println!(
            "    source release: {}, map {} bytes, file {} bytes, offsets {}",
            if on { "on" } else { "OFF" },
            map_len,
            file_len,
            if whole { "usable" } else { "UNUSABLE (page cache will not be dropped)" },
        );
        Self { file, map_base: whole.then_some(map_base), file_len, page, on }
    }

    #[cfg(not(target_os = "linux"))]
    fn new(_path: &std::path::Path, _map_base: usize, _map_len: usize) -> Self {
        Self { file: None, map_base: None, file_len: 0, page: 4096, on: false }
    }

    /// Rounded INWARD to whole pages, so a leaf never releases a page its
    /// neighbour is still reading.
    #[cfg(target_os = "linux")]
    fn release(&self, src: &Bytes) {
        use std::os::fd::AsRawFd;
        if !self.on {
            return;
        }
        let lo = src.as_ptr() as usize;
        let hi = (lo + src.len()) / self.page * self.page;
        let lo = lo.next_multiple_of(self.page);
        if hi <= lo {
            return;
        }
        // SAFETY: the range lies inside the pile's live mapping, which outlives
        // this call; `MADV_DONTNEED` on a shared file mapping only zaps
        // page-table entries, so the bytes are still readable and unchanged.
        unsafe {
            libc::madvise(lo as *mut libc::c_void, hi - lo, libc::MADV_DONTNEED);
        }
        if let (Some(base), Some(f)) = (self.map_base, self.file.as_ref()) {
            if hi - base > self.file_len {
                return;
            }
            // SAFETY: `f` is open for the duration of the call; the advice is
            // a hint and cannot invalidate anything.
            unsafe {
                libc::posix_fadvise(
                    f.as_raw_fd(),
                    (lo - base) as libc::off_t,
                    (hi - lo) as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn release(&self, _src: &Bytes) {}
}

/// One GiB.
const GIB: u64 = 1 << 30;

/// Total RAM this machine has, capped by a cgroup limit if one is set.
///
/// `MemTotal`, not `MemAvailable`. The two answer different questions and the
/// admission gate needs both: available says whether the copy can be MADE right
/// now, total says whether the finished process can LIVE. A box whose page
/// cache is warm reports most of it as available -- clean file pages are
/// reclaimable, so they count -- which is exactly how a share that cannot fit
/// gets admitted and then thrashes.
fn mem_total_bytes() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    let kb = status
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())
        .context("/proc/meminfo has no numeric MemTotal")?;
    let host = kb.checked_mul(1024).context("MemTotal overflow")?;
    let cgroup = match std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        Ok(max) if max.trim() != "max" => {
            max.trim().parse::<u64>().context("parsing cgroup memory.max")?
        }
        _ => u64::MAX,
    };
    Ok(host.min(cgroup))
}

/// What the run needs BESIDES the weight share.
///
/// On a unified-memory part the CUDA context, the device-resident activations
/// and the anonymous arena are one pool, so the share is not the footprint --
/// it is the footprint minus everything that has not been allocated yet. The
/// gate that only compared the share against `MemAvailable` was measuring the
/// first of those against a number that already counts the page cache the copy
/// is about to want back, which is why `INK_LAYERS=0:30` (116.24 GiB of
/// weights on a 121.6 GiB box) was ADMITTED and then thrashed.
///
/// The three terms, measured on this part:
///
/// * the CUDA context is 0.2 GiB;
/// * the per-layer resident activations are 4.1 GiB for a 20-layer share --
///   the KV-adjacent and MLP intermediates, which scale with the RANGE and
///   barely with the sequence;
/// * 4 GiB is left for the kernel, the shell and the page-cache working window,
///   and that number is not a guess either. It is where the measured cliff is;
/// * cubecl's two largest pool PAGES, from
///   [`super::budget::pool_page_floor`]. The pool reserves 41.74 GiB to hold
///   1.14 GiB of live tensors at 16,384 tokens, because a page is allocated
///   whole and returned only when every slice of it is free -- so the space
///   between what the tensors are and what the device has handed out is the
///   largest single term in this function, and it was missing;
/// * `attention_bytes` is everything that scales with the SEQUENCE, and it is
///   the term this function did not have. See
///   [`super::budget::prefill_activation_bytes`]. It was briefly the score
///   matrices alone, which was not enough by itself: once the dense lane
///   blocked its queries that term stopped growing -- 13.84 GiB at 16,384
///   tokens, 13.34 at 81,920, 13.52 at 100,623 -- and the gate went back to
///   being flat in the one variable it was added to track. What actually
///   scales is the routed-expert lane, whose every stage is
///   `num_experts_per_tok` rows a token: 372 KiB a token against the score
///   blocks' bounded 8 GiB.
///
/// The ladder, on a 121.63 GiB box, caches dropped and swap reset before each
/// row, `INK_GEN=1`, at 01211be plus this change. "free" and "swap" are read
/// straight after the startup copy, "peak" is `/usr/bin/time -v`:
///
/// | `INK_LAYERS` | share | peak RSS | free | swap | outcome |
/// |---|---|---|---|---|---|
/// | 0:20 | 80.72 GiB | 87.16 | 30.9 | 0.0 | forward 5.9 s |
/// | 0:28 | 109.14 | 113.9 | 4.1 | 0.0 | forward 12.5 s |
/// | 0:29 | 112.69 | 116.3 | 1.0 | 0.0 | **`CUDA_ERROR_OUT_OF_MEMORY`** |
/// | 0:30 | 116.24 | 117.5 | 1.0 | 1.4 | forward 45.0 s -- 3.6x 0:28 |
///
/// 0:29 dies outright and 0:30 survives only by swapping, so 0:28 is the
/// honest ceiling and the floor is set to put the refusal between them: this
/// function predicts 119.08 GiB at 0:28 against a machine of 121.63 (admitted,
/// 2.55 GiB of nominal headroom) and 122.84 at 0:29 (refused by 1.21). The
/// 0:30 row was taken with the OLD gate, which admitted it.
///
/// It derives from the machine (`MemTotal`), never from a constant: the two
/// nodes this runs on differ by 2 GiB, and a gate hard-coded to either figure
/// is wrong on the other.
fn run_overhead_bytes(layers: usize, attention_bytes: u64, machine: u64) -> u64 {
    const CUDA_CONTEXT: u64 = GIB / 5;
    const ACTIVATIONS_PER_LAYER: u64 = 41 * GIB / 200;
    const OS_FLOOR: u64 = 4 * GIB;
    CUDA_CONTEXT
        + ACTIVATIONS_PER_LAYER * layers as u64
        + OS_FLOOR
        + super::budget::pool_page_floor(machine)
        + attention_bytes
}

fn mem_available_bytes() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    let kb = status
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())
        .context("/proc/meminfo has no numeric MemAvailable")?;
    let host = kb.checked_mul(1024).context("MemAvailable overflow")?;
    let cgroup = match (
        std::fs::read_to_string("/sys/fs/cgroup/memory.max"),
        std::fs::read_to_string("/sys/fs/cgroup/memory.current"),
    ) {
        (Ok(max), Ok(current)) if max.trim() != "max" => {
            let max = max
                .trim()
                .parse::<u64>()
                .context("parsing cgroup memory.max")?;
            let current = current
                .trim()
                .parse::<u64>()
                .context("parsing cgroup memory.current")?;
            max.saturating_sub(current)
        }
        _ => u64::MAX,
    };
    Ok(host.min(cgroup))
}

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
        // The checkpoint tensor name lives in `metadata::name` as a UTF8String
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

/// One expert's BF16 weight, read out of the pile as a VIEW.
///
/// The unquantised sibling of [`PackedSlab`], and simpler for the reason layer
/// 2 exists at all: nothing was quantised, so there are no planes to split and
/// the payload IS the matrix.
pub struct Bf16Slab {
    pub bytes: Bytes,
    /// Output rows of this expert's matrix.
    pub rows: usize,
    /// Input columns — logical, and here also stored.
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
    /// Where the pile is, kept only so the startup copy can reopen it: the
    /// mapping does not carry an fd and `posix_fadvise` needs one.
    path: std::path::PathBuf,
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
    /// Anonymous startup copy of the share this process owns. Once present,
    /// every byte a device handle can alias is a view into this allocation,
    /// never into the reclaimable pile mapping.
    copied: Option<anybytes::Bytes>,
    copied_experts: std::collections::HashMap<(String, i64), CopiedExpert>,
}

#[derive(Clone)]
struct CopiedExpert {
    payload: anybytes::Bytes,
    rows: usize,
    logical: usize,
    nvfp4: bool,
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

        let t_open = std::time::Instant::now();
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
        let open_secs = t_open.elapsed().as_secs_f64();
        let t_experts = std::time::Instant::now();

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
                     n: Inline<Handle<blobencodings::UTF8String>>,
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

        let experts_secs = t_experts.elapsed().as_secs_f64();
        let t_dense = std::time::Instant::now();
        // ── the dense tensors, by name ──────────────────────────────────────
        // One query per (element, rank) — ten, not one — because that is what
        // typing the attribute means, and each hit is read AS its type. Nothing
        // is interpreted without one, so a BF16 matrix cannot arrive where f32
        // was asked for.
        let mut dense = std::collections::HashMap::new();
        // How many threads read the dense leaves. Each `get` is a BLAKE3 over
        // the payload and the payloads are the big ones -- the embedding table
        // is 2.40 GiB and the unembedding 1.61 -- so this sweep was 15.0 s of a
        // 23.6 s index build, on one core, for 968 leaves the queries had
        // already located. The reads are independent: `PileReader::get` takes
        // `&self` and its validation cache hashes outside its lock.
        let index_threads: usize = std::env::var("INK_INDEX_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| *n >= 1)
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
            });
        macro_rules! sweep_dense {
            ($ty:ty, $rank:literal, $tag:expr) => {{
                // Located first, read second. The `find!` iterator borrows
                // `facts` and is not something to hand to a thread; the handles
                // it yields are plain `Copy` values that are.
                let hits: Vec<_> = find!(
                    (e: Id,
                     n: Inline<Handle<blobencodings::UTF8String>>,
                     h: Inline<Handle<Tensor<$ty, $rank>>>),
                    pattern!(&facts, [
                        { ?e @ metadata::name: ?n, attrs::weight::<$ty, $rank>(): ?h },
                    ])
                )
                .filter(|(e, _, _)| !expert_ids.contains(e))
                .collect();
                if !hits.is_empty() {
                    let chunk = hits.len().div_ceil(index_threads).max(1);
                    let reader = &reader;
                    let facts = &facts;
                    let parts: Vec<Result<Vec<(String, Leaf)>>> = std::thread::scope(|sc| {
                        let handles: Vec<_> = hits
                            .chunks(chunk)
                            .map(|c| {
                                sc.spawn(move || -> Result<Vec<(String, Leaf)>> {
                                    let mut out = Vec::with_capacity(c.len());
                                    for (e, n, h) in c {
                                        let name: anybytes::View<str> = reader
                                            .get(*n)
                                            .map_err(|err| anyhow::anyhow!("name blob: {err:?}"))?;
                                        let blob: Blob<Tensor<$ty, $rank>> =
                                            reader.get(*h).map_err(|err| {
                                                anyhow::anyhow!("{}: leaf blob: {err:?}", &*name)
                                            })?;
                                        let view: TensorView = TensorView::try_from_blob(blob)
                                            .map_err(|err| {
                                                anyhow::anyhow!("{}: decode: {err}", &*name)
                                            })?;
                                        // The layer is optional in the graph, so it is
                                        // optional here: an `exists!` rather than a second
                                        // required clause, which would silently drop the
                                        // embedding and the head.
                                        let layer = find!(
                                            (l: i64),
                                            pattern!(facts, [{ (*e) @ attrs::layer: ?l }])
                                        )
                                        .next()
                                        .map(|(l,)| l);
                                        out.push((
                                            name.to_string(),
                                            Leaf {
                                                elem: $tag,
                                                dims: view.dims().to_vec(),
                                                bytes: view.payload().clone(),
                                                layer,
                                            },
                                        ));
                                    }
                                    Ok(out)
                                })
                            })
                            .collect();
                        handles.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    for part in parts {
                        for (name, leaf) in part? {
                            dense.insert(name, leaf);
                        }
                    }
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
        println!(
            "    index build: pile open + checkout {open_secs:.1}s, {} expert handles {experts_secs:.1}s, {} dense leaves {:.1}s",
            experts.len(),
            dense.len(),
            t_dense.elapsed().as_secs_f64(),
        );
        Ok(PileSource {
            path: path.to_path_buf(),
            reader,
            facts,
            dense,
            experts,
            stacked,
            copied: None,
            copied_experts: std::collections::HashMap::new(),
        })
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

    /// One expert's NVFP4 planes, read out of the pile and **not decoded**.
    pub fn expert_packed(&self, base: &str, e: usize) -> Result<PackedSlab> {
        if let Some(c) = self.copied_experts.get(&(base.to_string(), e as i64)) {
            anyhow::ensure!(c.nvfp4, "{base}[{e}] is BF16, not packed NVFP4");
            let elems = c.rows * c.logical;
            let codes_len = elems / 2;
            let scales_len = elems / NVFP4_BLOCK;
            let (_, _, scale2) = split_payload(&c.payload, elems)?;
            return Ok(PackedSlab {
                codes: c.payload.slice(..codes_len),
                scales: c.payload.slice(codes_len..codes_len + scales_len),
                scale2,
                rows: c.rows,
                cols: c.logical / 2,
            });
        }
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

    /// One expert's BF16 bytes, as a view over the pile's mapping.
    ///
    /// The dual of [`PileSource::expert_packed`], and it refuses the other
    /// format for the same reason that one does: which of the two a leaf holds
    /// is part of its identity here, so asking for the wrong one is an error
    /// and never a reinterpretation.
    pub fn expert_bf16(&self, base: &str, e: usize) -> Result<Bf16Slab> {
        if let Some(c) = self.copied_experts.get(&(base.to_string(), e as i64)) {
            anyhow::ensure!(!c.nvfp4, "{base}[{e}] is packed NVFP4, not BF16");
            anyhow::ensure!(
                c.payload.len() == c.rows * c.logical * 2,
                "{base}[{e}]: {} bytes for {}x{} BF16",
                c.payload.len(),
                c.rows,
                c.logical
            );
            return Ok(Bf16Slab {
                bytes: c.payload.clone(),
                rows: c.rows,
                cols: c.logical,
            });
        }
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
        let dims = view.dims();
        anyhow::ensure!(dims.len() == 2, "{base}[{e}] is rank {}", dims.len());
        let (rows, cols) = (dims[0] as usize, dims[1] as usize);
        let payload = view.payload();
        anyhow::ensure!(
            payload.len() == rows * cols * 2,
            "{base}[{e}]: {} bytes for {rows}x{cols} BF16",
            payload.len()
        );
        Ok(Bf16Slab { bytes: payload.clone(), rows, cols })
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
        if let Some(bytes) = &self.copied {
            let view: anybytes::View<[u8]> = bytes
                .clone()
                .view()
                .map_err(|e| anyhow::anyhow!("viewing the anonymous weight allocation: {e}"))?;
            let owner: std::sync::Arc<Vec<u8>> = view
                .downcast_to_owner()
                .map_err(|_| anyhow::anyhow!("anonymous weight allocation lost its Vec owner"))?;
            return Ok(vec![(
                bytes.as_ptr() as usize,
                bytes.len(),
                owner as std::sync::Arc<dyn std::any::Any + Send + Sync>,
            )]);
        }
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

    /// The same, restricted to a half-open LAYER range.
    ///
    /// The layer is a FACT the importer wrote and this index kept
    /// ([`ExpertRef::layer`]), not a substring of the name, which is what makes
    /// "the experts this node holds" a lookup rather than a parse. A node that
    /// warmed the whole pile would read 159 GiB to prepare for the 85 it runs.
    pub fn expert_keys_in(&self, range: std::ops::Range<usize>) -> Vec<(String, i64)> {
        let mut v: Vec<(String, i64)> = self
            .experts
            .iter()
            .filter(|((_, _), r)| {
                r.layer >= range.start as i64 && r.layer < range.end as i64
            })
            .map(|((n, e), _)| (n.clone(), *e))
            .collect();
        v.sort();
        v
    }

    /// Copy exactly one node's share out of the file-backed pile mapping into
    /// one anonymous allocation. The GPU may safely alias this allocation:
    /// anonymous pages have no backing store the kernel can silently re-read
    /// them from, so they cannot be reclaimed while this process owns them.
    ///
    /// # One pass, not two, and why that is a memory question
    ///
    /// This used to be a sequential `fetch+verify` loop that read and BLAKE3'd
    /// every leaf, followed by a threaded `memcpy` loop that copied what the
    /// first loop had faulted in. On a discrete-memory box that is merely two
    /// loops; on a unified-memory one it is the whole problem. Between the two
    /// loops the process holds the share TWICE — once as the pile's mapped page
    /// cache, once as the anonymous arena — and this node's share is 80.72 GiB
    /// against 119-121 GiB of RAM that the GPU also lives in. Measured at
    /// `INK_LAYERS=0:20`: 117.4 GiB resident, 0.7 GiB free, and the entire
    /// 16 GiB swap consumed, because the kernel chose to page out the arena we
    /// had just written rather than evict the mapped file pages we were done
    /// with. The second loop then ran at 3.0 GiB/s instead of 50 — it was
    /// re-reading its own source off the SSD — and on the node with 2 GiB less
    /// of the two this runs on, the CUDA context that comes afterwards could
    /// not be created at all.
    ///
    /// So: the shapes are probed ONCE PER STACKED MATRIX rather than once per
    /// expert (every expert of one stack is the same matrix, and the copy
    /// asserts that), the layout is computed from those, the arena is allocated
    /// zeroed-by-the-kernel rather than written with 80 GiB of zeros nobody
    /// reads, and then one threaded pass fetches, verifies, copies and releases
    /// each leaf's source pages. Peak residency becomes the arena plus a
    /// working window instead of the arena plus the whole share.
    ///
    /// `attention_bytes` is what prefill will hold in ACTIVATIONS at this
    /// sequence length, from [`super::budget::prefill_activation_bytes`]: the
    /// residual stream, every layer's kept keys and values, and the widest
    /// single layer's working set -- which on this model is the routed-expert
    /// lane, six rows a token through buffers as wide as the hidden size.
    /// It is a parameter rather than something this module derives because the
    /// sequence length is a fact about the RUN and the weight share is a fact
    /// about the checkpoint, and folding one into the other is how the gate
    /// came to charge a constant for something linear in the sequence.
    pub fn copy_share(
        &mut self,
        layers: std::ops::Range<usize>,
        global_dense: &[&str],
        attention_bytes: u64,
    ) -> Result<(usize, usize, u64)> {
        anyhow::ensure!(self.copied.is_none(), "the weight share was already copied");

        /// The shape every expert of one stacked matrix has.
        #[derive(Clone, Copy)]
        struct Shape {
            rows: usize,
            logical: usize,
            nvfp4: bool,
            payload: usize,
        }

        let keys = self.expert_keys_in(layers.clone());

        // One probe per stacked matrix, not per expert. A stack is 256 slices
        // of ONE matrix, so its element format and its dimensions are a
        // property of the stack; reading all 9 216 headers to discover 40
        // identical answers is 80 GiB of BLAKE3 paid to learn a layout. The
        // assumption is not silent: the copy below refuses any leaf whose
        // payload is not the length this shape implies.
        let t_probe = std::time::Instant::now();
        let mut shapes: std::collections::HashMap<String, Shape> = Default::default();
        for (name, e) in &keys {
            if shapes.contains_key(name) {
                continue;
            }
            let r = &self.experts[&(name.clone(), *e)];
            let shape = match r.handle {
                ExpertHandle::Nvfp4(h) => {
                    let blob: Blob<Tensor<NVFP4, 2>> = self
                        .reader
                        .get(h)
                        .map_err(|err| anyhow::anyhow!("{name}[{e}]: {err:?}"))?;
                    let view = TensorView::try_from_blob(blob)
                        .map_err(|err| anyhow::anyhow!("{name}[{e}]: decode: {err}"))?;
                    Shape {
                        rows: view.dims()[0] as usize,
                        logical: view.dims()[1] as usize,
                        nvfp4: true,
                        payload: view.payload().len(),
                    }
                }
                ExpertHandle::Bf16(h) => {
                    let blob: Blob<Tensor<BF16, 2>> = self
                        .reader
                        .get(h)
                        .map_err(|err| anyhow::anyhow!("{name}[{e}]: {err:?}"))?;
                    let view = TensorView::try_from_blob(blob)
                        .map_err(|err| anyhow::anyhow!("{name}[{e}]: decode: {err}"))?;
                    Shape {
                        rows: view.dims()[0] as usize,
                        logical: view.dims()[1] as usize,
                        nvfp4: false,
                        payload: view.payload().len(),
                    }
                }
            };
            shapes.insert(name.clone(), shape);
        }
        let probe_secs = t_probe.elapsed().as_secs_f64();

        let globals: std::collections::HashSet<&str> = global_dense.iter().copied().collect();
        for name in &globals {
            anyhow::ensure!(
                self.dense.contains_key(*name),
                "startup-copy table {name} is not in the pile"
            );
        }
        let mut dense_names: Vec<String> = self
            .dense
            .iter()
            .filter(|(name, leaf)| {
                leaf.layer.map(|l| layers.contains(&(l as usize))).unwrap_or(false)
                    || globals.contains(name.as_str())
            })
            .map(|(name, _)| name.clone())
            .collect();
        dense_names.sort();

        // The DESTINATION of every leaf, computed before anything is read.
        // Disjoint destinations are what let the pass below be threaded:
        // experts in `expert_keys_in` order, then dense leaves by name, each
        // padded to a SIXTEEN-byte boundary.
        //
        // Sixteen, not the four this used to write. The tuned BF16 GEMM lanes
        // pick their load width from the tensor SHAPE and never from the
        // pointer, so a weight aliased at 4 mod 16 raises
        // `CUDA_ERROR_MISALIGNED_ADDRESS` -- an async fault that poisons the
        // CUDA context -- and the dense lane has to route it to the slower hand
        // kernel instead. Sixteen costs at most 15 bytes per view, about 14 KB
        // on a 20-layer share, and it is the whole benefit `INK_ALIGN_COPY=1`
        // was buying by DUPLICATING 908 MiB of weight into a fresh device
        // allocation; that arm is moot now.
        const VIEW_ALIGN: usize = 16;
        // What each layer costs, so a refusal can name the range that WOULD
        // fit instead of leaving the operator to bisect for it. Accumulated
        // here because this loop is the only place the byte counts exist.
        let mut per_layer: std::collections::BTreeMap<i64, usize> = Default::default();
        let mut fixed_bytes = 0usize;
        let mut cursor = 0usize;
        let mut expert_offsets = Vec::with_capacity(keys.len());
        for key in &keys {
            let start = cursor;
            let end = start
                .checked_add(shapes[&key.0].payload)
                .context("weight share byte count overflow")?;
            cursor = end.next_multiple_of(VIEW_ALIGN);
            *per_layer.entry(self.experts[key].layer).or_default() += cursor - start;
            expert_offsets.push((start, end));
        }
        let mut dense_offsets = Vec::with_capacity(dense_names.len());
        for name in &dense_names {
            let start = cursor;
            let end = start
                .checked_add(self.dense[name].bytes.len())
                .context("weight share byte count overflow")?;
            cursor = end.next_multiple_of(VIEW_ALIGN);
            match self.dense[name].layer.filter(|l| layers.contains(&(*l as usize))) {
                Some(l) => *per_layer.entry(l).or_default() += cursor - start,
                // The embedding and unembedding tables belong to no layer, so
                // they are what a shorter range still has to pay.
                None => fixed_bytes += cursor - start,
            }
            dense_offsets.push((start, end));
        }
        let total = cursor;

        // The admission gate. Two questions, both of which have to be yes:
        // can the copy be MADE now (`MemAvailable`), and can the finished
        // process LIVE (`MemTotal` against the share plus everything the run
        // allocates after it -- see `run_overhead_bytes`).
        let available = mem_available_bytes()?;
        let machine = mem_total_bytes()?;
        let n_layers = layers.len();
        let overhead = run_overhead_bytes(n_layers, attention_bytes, machine);
        let need = total as u64 + overhead;
        let gib = |b: u64| b as f64 / GIB as f64;
        if need > machine || total as u64 > available {
            // The largest range starting at `layers.start` that WOULD fit. Both
            // sides of the test grow with the layer count, so the predicate is
            // monotone and the last k that passes is the answer; the byte
            // counts are the exact ones this layout just computed, not an
            // average, because a refusal that makes the operator bisect for the
            // answer is half a bug.
            let mut acc = fixed_bytes as u64;
            let mut fits: Option<(usize, u64)> = None;
            for (k, bytes) in per_layer.values().enumerate() {
                acc += *bytes as u64;
                let k = k + 1;
                let fits_here = acc + run_overhead_bytes(k, attention_bytes, machine);
                if fits_here <= machine && acc <= available {
                    fits = Some((k, acc));
                }
            }
            let advice = match fits {
                Some((k, share)) => format!(
                    "The largest range that fits here is INK_LAYERS={}:{} -- {:.2} GiB of weights, \
                     {:.2} GiB with the context and activations. Give the rest to another node.",
                    layers.start,
                    layers.start + k,
                    gib(share),
                    gib(share + run_overhead_bytes(k, attention_bytes, machine)),
                ),
                None => format!(
                    "Not even one layer fits: layer {} alone is {:.2} GiB on top of {:.2} GiB of \
                     tables this range cannot drop. This model needs more nodes, not a smaller \
                     range.",
                    layers.start,
                    gib(*per_layer.values().next().unwrap_or(&0) as u64),
                    gib(fixed_bytes as u64),
                ),
            };
            // Which half of the sum is the problem. A refusal that only ever
            // says "use fewer layers" sends the operator to buy nodes when the
            // fix is a shorter input: the activation working set is linear in
            // the sequence and very nearly FLAT in the range, because layers
            // run one at a time and each frees its own before the next
            // allocates. No layer split touches it.
            let cause = if attention_bytes > total as u64 {
                format!(
                    "\n  THE SEQUENCE, NOT THE SPLIT: {:.2} GiB of that is activations at this \
                     input length, against {:.2} GiB of weights. Splitting the stack across more \
                     nodes does not help -- the widest layer is as wide on every node, and every \
                     node attends over the whole sequence. Shorten the input.",
                    gib(attention_bytes),
                    gib(total as u64),
                )
            } else {
                String::new()
            };
            anyhow::bail!(
                "INK_LAYERS={}:{} is {n_layers} layers = {:.2} GiB of weights; with the CUDA \
                 context and this node's activations that is {:.2} GiB, and this machine has \
                 {:.2} GiB ({:.2} GiB available right now). Refusing.{cause}\n  \
                 A share this size fits only by taking memory the GPU still needs. Measured on \
                 a 121.63 GiB box: 0:29 (112.69 GiB) was admitted by the old gate and died with \
                 CUDA_ERROR_OUT_OF_MEMORY, and 0:30 (116.24 GiB) survived only by taking 1.4 GiB \
                 of swap, which cost it 45.0 s of forward against 12.5 s at 0:28. Nothing here \
                 is reclaimable -- the arena is anonymous on purpose, so the GPU can alias it -- \
                 so the kernel pages the weights themselves.\n  \
                 {advice}",
                layers.start,
                layers.end,
                gib(total as u64),
                gib(need),
                gib(machine),
                gib(available),
            );
        }
        println!(
            "    admission: {:.2} GiB of weights + {:.2} GiB context/activations = {:.2} GiB of \
             {:.2} GiB ({:.2} GiB of headroom, {:.2} GiB available now)",
            gib(total as u64),
            gib(overhead),
            gib(need),
            gib(machine),
            gib(machine - need),
            gib(available),
        );

        // Zeroed by the KERNEL, not by us. `Vec::resize(total, 0)` wrote 80.72
        // GiB of zeros that the copy immediately overwrote — 25 s of pure waste
        // whose real cost was not the time but the residency: it faulted the
        // whole arena in before a single weight had been copied, which is what
        // forced the source page cache out to make room. `alloc_zeroed` is
        // `calloc`, and for an allocation this size glibc hands back a fresh
        // anonymous mapping whose pages are already zero and not yet resident,
        // so each page arrives exactly once, when the copy writes it.
        //
        // `VIEW_ALIGN - 1` bytes longer than the layout, because the offsets
        // are only 16-aligned RELATIVE to the base and every view's real
        // address is base + offset. glibc mmaps an allocation this size and
        // hands back a page-aligned pointer, so `skew` is measured to be zero
        // -- but measured, not assumed, because the whole point of the
        // alignment is that a weight at 4 mod 16 faults the CUDA context
        // asynchronously and there is no failure to see at the bind.
        let (mut arena, skew) = {
            let bytes = total + VIEW_ALIGN - 1;
            let layout = std::alloc::Layout::from_size_align(bytes, 1)
                .map_err(|e| anyhow::anyhow!("startup-copy layout for {bytes} bytes: {e}"))?;
            // SAFETY: `bytes > 0` (the share always has leaves), the layout is
            // the one `Vec<u8>` itself uses (align 1), and the pointer is
            // handed to exactly one `Vec` which owns and frees it.
            let p = unsafe { std::alloc::alloc_zeroed(layout) };
            anyhow::ensure!(
                !p.is_null(),
                "cannot allocate {:.2} GiB for this node's startup weight copy",
                bytes as f64 / (1u64 << 30) as f64,
            );
            let skew = (VIEW_ALIGN - (p as usize) % VIEW_ALIGN) % VIEW_ALIGN;
            (unsafe { Vec::from_raw_parts(p, bytes, bytes) }, skew)
        };

        // How many threads fetch, verify and copy. Unset means one per core.
        // `INK_COPY_THREADS=1` is the sequential lane this replaced, kept
        // selectable so the two can be run back to back out of ONE binary.
        let threads: usize = std::env::var("INK_COPY_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n >= 1)
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
            });

        enum Job<'a> {
            Expert(&'a (String, i64), Shape),
            Dense(&'a str),
        }
        let mut jobs: Vec<(usize, usize, Job)> = Vec::with_capacity(keys.len() + dense_names.len());
        for (k, &(start, end)) in keys.iter().zip(expert_offsets.iter()) {
            jobs.push((start, end, Job::Expert(k, shapes[&k.0])));
        }
        for (name, &(start, end)) in dense_names.iter().zip(dense_offsets.iter()) {
            jobs.push((start, end, Job::Dense(name.as_str())));
        }

        // Split by BYTES, not by leaf count: the leaves are not the same size
        // (a dense leaf is 537 MB and an expert plane is 14 MB), so an even
        // split of the list is an uneven split of the work.
        let per = total.div_ceil(threads);
        let mut bounds: Vec<usize> = Vec::with_capacity(threads + 1);
        let mut j = 0usize;
        bounds.push(0);
        for t in 1..threads {
            let target = per * t;
            while j < jobs.len() && jobs[j].0 < target {
                j += 1;
            }
            bounds.push(j);
        }
        bounds.push(jobs.len());

        let mut rest: &mut [u8] = &mut arena.as_mut_slice()[skew..skew + total];
        let mut shards: Vec<(&mut [u8], usize, &[(usize, usize, Job)])> = Vec::new();
        let mut base = 0usize;
        for w in bounds.windows(2) {
            let (a, b) = (w[0], w[1]);
            let span_end = if b == jobs.len() { total } else { jobs[b].0 };
            let (head, tail) = rest.split_at_mut(span_end - base);
            shards.push((head, base, &jobs[a..b]));
            rest = tail;
            base = span_end;
        }

        let (map_base, map_len, _keep) = {
            let m = self.mappings()?;
            let (b, l, k) = m.into_iter().next().context("the pile has no mapping")?;
            (b, l, k)
        };
        let release = SourceRelease::new(&self.path, map_base, map_len);
        let release = &release;
        let reader = &self.reader;
        let experts = &self.experts;
        let dense = &self.dense;
        let t_copy = std::time::Instant::now();
        let results: Vec<Result<()>> = std::thread::scope(|sc| {
            let handles: Vec<_> = shards
                .into_iter()
                .map(|(buf, base, mine)| {
                    sc.spawn(move || -> Result<()> {
                        for (start, end, job) in mine {
                            let src: Bytes = match job {
                                Job::Expert(k, shape) => {
                                    let r = &experts[*k];
                                    let payload = match r.handle {
                                        ExpertHandle::Nvfp4(h) => {
                                            let blob: Blob<Tensor<NVFP4, 2>> =
                                                reader.get(h).map_err(|err| {
                                                    anyhow::anyhow!("{}[{}]: {err:?}", k.0, k.1)
                                                })?;
                                            let view = TensorView::try_from_blob(blob).map_err(
                                                |err| {
                                                    anyhow::anyhow!(
                                                        "{}[{}]: decode: {err}",
                                                        k.0,
                                                        k.1
                                                    )
                                                },
                                            )?;
                                            anyhow::ensure!(
                                                shape.nvfp4
                                                    && view.dims()[0] as usize == shape.rows
                                                    && view.dims()[1] as usize == shape.logical,
                                                "{}[{}] is {:?}, but its stack is {}x{}",
                                                k.0,
                                                k.1,
                                                view.dims(),
                                                shape.rows,
                                                shape.logical,
                                            );
                                            view.payload().clone()
                                        }
                                        ExpertHandle::Bf16(h) => {
                                            let blob: Blob<Tensor<BF16, 2>> =
                                                reader.get(h).map_err(|err| {
                                                    anyhow::anyhow!("{}[{}]: {err:?}", k.0, k.1)
                                                })?;
                                            let view = TensorView::try_from_blob(blob).map_err(
                                                |err| {
                                                    anyhow::anyhow!(
                                                        "{}[{}]: decode: {err}",
                                                        k.0,
                                                        k.1
                                                    )
                                                },
                                            )?;
                                            anyhow::ensure!(
                                                !shape.nvfp4
                                                    && view.dims()[0] as usize == shape.rows
                                                    && view.dims()[1] as usize == shape.logical,
                                                "{}[{}] is {:?}, but its stack is {}x{}",
                                                k.0,
                                                k.1,
                                                view.dims(),
                                                shape.rows,
                                                shape.logical,
                                            );
                                            view.payload().clone()
                                        }
                                    };
                                    anyhow::ensure!(
                                        payload.len() == end - start,
                                        "{}[{}] is {} bytes where its stack implies {}",
                                        k.0,
                                        k.1,
                                        payload.len(),
                                        end - start,
                                    );
                                    payload
                                }
                                Job::Dense(name) => dense[*name].bytes.clone(),
                            };
                            buf[start - base..end - base].copy_from_slice(&src);
                            release.release(&src);
                        }
                        Ok(())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for r in results {
            r?;
        }
        let copy_secs = t_copy.elapsed().as_secs_f64();
        println!(
            "    startup copy: {} shape probe{} {:.1}s, fetch+verify+copy {:.1}s ({:.2} GiB/s, {threads} thread{})",
            shapes.len(),
            if shapes.len() == 1 { "" } else { "s" },
            probe_secs,
            copy_secs,
            total as f64 / (1u64 << 30) as f64 / copy_secs.max(1e-9),
            if threads == 1 { "" } else { "s" },
        );
        println!("{}", mem_line("after fetch+verify+copy"));

        // `Bytes` owns the allocator's Vec; `View` proves and retains the new
        // anonymous backing before subviews replace every mmap-backed payload.
        let bytes = anybytes::Bytes::from_source(arena);
        let view: anybytes::View<[u8]> = bytes
            .clone()
            .view()
            .map_err(|e| anyhow::anyhow!("viewing the anonymous weight allocation: {e}"))?;
        let bytes = view.bytes();
        for ((key, shape), (start, end)) in keys
            .iter()
            .map(|k| (k, shapes[&k.0]))
            .zip(expert_offsets)
        {
            self.copied_experts.insert(key.clone(), CopiedExpert {
                payload: bytes.slice(skew + start..skew + end),
                rows: shape.rows,
                logical: shape.logical,
                nvfp4: shape.nvfp4,
            });
        }
        for (name, (start, end)) in dense_names.iter().zip(dense_offsets) {
            self.dense
                .get_mut(name)
                .expect("selected dense leaf")
                .bytes = bytes.slice(skew + start..skew + end);
        }
        self.copied = Some(bytes);
        Ok((self.copied_experts.len(), dense_names.len(), total as u64))
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
