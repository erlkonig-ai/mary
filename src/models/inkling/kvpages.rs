//! A paged store for one attention layer's keys or values.
//!
//! ## Why pages, when a contiguous tensor already works
//!
//! Not for speed at the read: [`PageStore::materialize`] concatenates, so a
//! decode step pays what the contiguous cache paid. Pages buy three things the
//! contiguous form cannot express.
//!
//! * *A prefix becomes shareable.* Burn tensors clone by handle, so
//!   [`PageStore::share_prefix`] hands a second cache the same device buffers
//!   at no copy and no recomputation. A context laid out as
//!   `[stable memory cover][recent detail][live turn]` then pays for the tail
//!   only — which is the shape the memory cover already has, coarse to fine,
//!   the slow-moving part first. That is the whole argument for this file.
//! * *Growth stops being quadratic in copies.* `Tensor::cat` on append rewrites
//!   the whole cache every step; appending into a partial page rewrites one
//!   page.
//! * *FP4 KV becomes expressible at all.* The reference implementation asserts
//!   it: `fp4 KV requires paged attention` (`inklingdeus`
//!   `third_party/inkling_sm120_fa4/interface.py`). Payload at half width plus
//!   per-block scales is a page layout, not a tensor layout.
//!
//! ## The two ends move for different reasons
//!
//! A sliding-window layer drops from the FRONT as the window advances; a
//! speculative batch truncates from the BACK when drafts are rejected. They are
//! not the same operation and neither is a page boundary, so this store keeps
//! an explicit `head` offset into page 0 and a logical `len`, and lets both
//! ends land mid-page. Rounding either to a page would either forget rows the
//! window still needs or keep rows a rejection discarded.
//!
//! ## One geometry, two element types
//!
//! The paragraph above promised that "a later FP4 store is a change of element
//! type rather than of geometry", and [`Pages`] is that promise kept: the head
//! offset, the page-boundary arithmetic, the two trims and the shared prefix
//! are written ONCE, over anything that is fixed-width rows which can be cut
//! and rejoined ([`PageRows`]). [`PageStore`] is that core over dense float
//! rows; [`Fp4PageStore`] is the same core over NVFP4 rows. A paging bug
//! therefore cannot exist in one of them and not the other, which matters more
//! than the lines saved — the failure mode here is silently reordered rows, and
//! two transcriptions of `drop_front` are two chances at it.

use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Rows per page. 128 matches the reference implementation's `page_size`, which
/// is what its FP4 payload/scale shapes are cut to; keeping the same number
/// means a later FP4 store is a change of element type rather than of geometry.
pub const PAGE: usize = 128;

/// What a page is made of: fixed-width rows that can be cut and rejoined.
///
/// Three operations and no arithmetic. That is deliberate — [`Pages`] must not
/// be able to do anything to a row's CONTENT, only to decide which rows are
/// where, so that "the store reordered my keys" and "the store corrupted my
/// keys" stay different bugs with different homes.
pub trait PageRows: Clone {
    /// How many rows this holds.
    fn rows(&self) -> usize;
    /// The LOGICAL row width — what a caller thinks a row is, which for an FP4
    /// row is not the width of any buffer backing it.
    fn width(&self) -> usize;
    /// Rows `from..to`, as a new value.
    fn slice_rows(&self, from: usize, to: usize) -> Self;
    /// Two or more of these, end to end, in order.
    fn concat(parts: Vec<Self>) -> Self
    where
        Self: Sized;
}

/// The page-boundary arithmetic, over any [`PageRows`].
///
/// Invariants, checked by [`Pages::assert_sound`]:
/// * every page but the last holds exactly [`PAGE`] rows;
/// * `head < PAGE`, and `head` counts rows already dropped from page 0;
/// * `len` is the LOGICAL row count, excluding `head`;
/// * `head + len` equals the total rows actually stored.
#[derive(Clone, Debug)]
pub struct Pages<R: PageRows> {
    pages: Vec<R>,
    head: usize,
    len: usize,
}

impl<R: PageRows> Default for Pages<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: PageRows> Pages<R> {
    /// An empty set of pages.
    pub fn new() -> Self {
        Self {
            pages: Vec::new(),
            head: 0,
            len: 0,
        }
    }

    /// Logical rows currently held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Rows stored, including the ones `head` has dropped from page 0.
    fn stored(&self) -> usize {
        self.head + self.len
    }

    /// The first page, or `None` while empty — how a wrapper asks the pages
    /// where they live without this type knowing what a device is.
    pub fn first(&self) -> Option<&R> {
        self.pages.first()
    }

    /// Append `rows`, filling the last partial page before starting a new one.
    ///
    /// Takes the whole batch at once rather than row by row: a speculative
    /// verify appends `k + 1` rows and splitting them into `k + 1` slice
    /// assignments would copy the tail page that many times.
    pub fn append(&mut self, rows: R) {
        let n = rows.rows();
        if n == 0 {
            return;
        }
        let mut written = 0usize;
        // Fill the tail page first, if it has room.
        let tail = self.stored() % PAGE;
        if tail != 0 {
            let room = PAGE - tail;
            let take = room.min(n);
            let last = self.pages.len() - 1;
            let part = rows.slice_rows(0, take);
            self.pages[last] = R::concat(vec![self.pages[last].clone(), part]);
            written = take;
        }
        while written < n {
            let take = PAGE.min(n - written);
            self.pages.push(rows.slice_rows(written, written + take));
            written += take;
        }
        self.len += n;
    }

    /// Drop `n` rows from the FRONT — the sliding window advancing.
    ///
    /// Whole pages are released; the remainder becomes `head`. The partial page
    /// is NOT rewritten, so a window that advances one row a step does no work
    /// until it crosses a page boundary.
    pub fn drop_front(&mut self, n: usize) {
        assert!(n <= self.len, "dropping {n} of {} rows", self.len);
        let head = self.head + n;
        let whole = head / PAGE;
        if whole > 0 {
            self.pages.drain(0..whole.min(self.pages.len()));
        }
        self.head = head % PAGE;
        self.len -= n;
        if self.len == 0 {
            self.pages.clear();
            self.head = 0;
        }
    }

    /// Truncate to `keep` logical rows — a speculative batch being rejected.
    pub fn truncate(&mut self, keep: usize) {
        assert!(keep <= self.len, "keeping {keep} of {} rows", self.len);
        if keep == self.len {
            return;
        }
        self.len = keep;
        let stored = self.stored();
        let full = stored.div_ceil(PAGE);
        self.pages.truncate(full);
        let tail = stored % PAGE;
        if tail != 0 {
            let last = self.pages.len() - 1;
            self.pages[last] = self.pages[last].slice_rows(0, tail);
        }
        if self.len == 0 {
            self.pages.clear();
            self.head = 0;
        }
    }

    /// The rows as one value, in order, or `None` while empty.
    ///
    /// Empty is a real state — a fresh cache, or one a rejection emptied — and
    /// there is no zero-row [`PageRows`] to invent here without knowing what a
    /// device is, so the wrapper says what an empty read means.
    pub fn gather(&self) -> Option<R> {
        if self.len == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(self.pages.len());
        let mut skip = self.head;
        let mut left = self.len;
        for p in &self.pages {
            let rows = p.rows();
            let from = skip.min(rows);
            skip -= from;
            if from >= rows || left == 0 {
                continue;
            }
            let take = (rows - from).min(left);
            out.push(p.slice_rows(from, from + take));
            left -= take;
        }
        debug_assert_eq!(left, 0, "gather lost rows");
        Some(if out.len() == 1 {
            out.pop().expect("one part")
        } else {
            R::concat(out)
        })
    }

    /// Share the first `rows` logical rows with a new set of pages, without
    /// copying.
    ///
    /// The point of the file. Burn clones a tensor by handle, so the returned
    /// pages reference the same device buffers; recomputing a prefix's KV is
    /// replaced by cloning `rows / PAGE` handles. Refuses a prefix that does not
    /// land on a page boundary, because a shared partial page would be written
    /// through by whichever store appended next.
    pub fn share_prefix(&self, rows: usize) -> Option<Self> {
        if self.head != 0 || rows > self.len || rows % PAGE != 0 {
            return None;
        }
        Some(Self {
            pages: self.pages[..rows / PAGE].to_vec(),
            head: 0,
            len: rows,
        })
    }

    /// Panics unless every documented invariant holds.
    pub fn assert_sound(&self, width: usize) {
        assert!(self.head < PAGE, "head {} is a whole page", self.head);
        let stored = self.stored();
        assert_eq!(
            self.pages.len(),
            stored.div_ceil(PAGE),
            "{} pages for {stored} stored rows",
            self.pages.len()
        );
        for (i, p) in self.pages.iter().enumerate() {
            let (n, w) = (p.rows(), p.width());
            assert_eq!(w, width, "page {i} is {w} wide");
            let want = if i + 1 == self.pages.len() {
                let t = stored % PAGE;
                if t == 0 { PAGE } else { t }
            } else {
                PAGE
            };
            assert_eq!(n, want, "page {i} holds {n} rows, wanted {want}");
        }
    }
}

impl<B: Backend> PageRows for Tensor<B, 2> {
    fn rows(&self) -> usize {
        self.dims()[0]
    }

    fn width(&self) -> usize {
        self.dims()[1]
    }

    fn slice_rows(&self, from: usize, to: usize) -> Self {
        let w = self.dims()[1];
        self.clone().slice([from..to, 0..w])
    }

    fn concat(parts: Vec<Self>) -> Self {
        Tensor::cat(parts, 0)
    }
}

/// One layer's keys or values, stored dense as pages of at most [`PAGE`] rows.
#[derive(Clone, Debug)]
pub struct PageStore<B: Backend> {
    pages: Pages<Tensor<B, 2>>,
    width: usize,
}

impl<B: Backend> PageStore<B> {
    /// An empty store for rows of `width` columns.
    pub fn new(width: usize) -> Self {
        Self {
            pages: Pages::new(),
            width,
        }
    }

    /// Logical rows currently held.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Columns per row, fixed at construction.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The device the pages live on, or `None` while the store is empty.
    ///
    /// Empty is a real state — a fresh cache, or one a rejection emptied — and
    /// a store with no pages has no device to report, so the caller says what
    /// to do rather than being handed a default that might be the wrong GPU.
    pub fn device(&self) -> Option<B::Device> {
        self.pages.first().map(|p| p.device())
    }

    /// Append `rows`, filling the last partial page before starting a new one.
    pub fn append(&mut self, rows: Tensor<B, 2>) {
        let [_, w] = rows.dims();
        assert_eq!(
            w, self.width,
            "a {w}-wide row into a {}-wide store",
            self.width
        );
        self.pages.append(rows);
    }

    /// Drop `n` rows from the FRONT — the sliding window advancing.
    pub fn drop_front(&mut self, n: usize) {
        self.pages.drop_front(n);
    }

    /// Truncate to `keep` logical rows — a speculative batch being rejected.
    pub fn truncate(&mut self, keep: usize) {
        self.pages.truncate(keep);
    }

    /// The rows as one contiguous tensor, in order.
    ///
    /// This is what the present attention read wants. It concatenates, so it is
    /// no cheaper than the contiguous cache was — the win is elsewhere, see the
    /// module docs.
    pub fn materialize(&self, dev: &B::Device) -> Tensor<B, 2> {
        self.pages
            .gather()
            .unwrap_or_else(|| Tensor::zeros([0, self.width], dev))
    }

    /// Share the first `rows` logical rows with a new store, without copying.
    pub fn share_prefix(&self, rows: usize) -> Option<Self> {
        Some(Self {
            pages: self.pages.share_prefix(rows)?,
            width: self.width,
        })
    }

    /// Panics unless every documented invariant holds.
    pub fn assert_sound(&self) {
        self.pages.assert_sound(self.width);
    }
}

// ---------------------------------------------------------------------------
// NVFP4 pages
// ---------------------------------------------------------------------------

use super::fp4quant::{
    dequantize_nvfp4, dequantize_nvfp4_bf16, quantize_nvfp4, quantize_nvfp4_bf16,
};
use super::seam::{self, Bk};
use burn::tensor::{DType, Int};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::ComputeClient;

/// Whether the KV cache holds its pages as NVFP4 rather than as the dtype it
/// was handed. **Off by default; `INK_FP4_KV=1` turns it on.**
///
/// ## What it is for, said plainly
///
/// Capacity, not speed. The reference implementation reaches 1,048,576 tokens
/// of context with an `fp4_mx_block16` KV pool against 262,144 without one, and
/// that ratio is the whole feature: four bits plus one E4M3 scale per sixteen
/// is 4.5 bits a value against BF16's sixteen, so a token's KV costs 3.56x
/// less and the context that fits grows by the same factor.
///
/// It will not make a short generation faster and it is not supposed to.
/// Decode `pass_ms` on this part measured FLAT from 512 to 8192 tokens of
/// context (59.88 / 59.46 / 57.64 ms, one decode step, GB10) because a decode
/// step reads 275.8 MB of weights per layer-step and the KV beside it is
/// noise; the crossover where the KV read equals the weight read is around 67k
/// tokens. Below that this trades a quantize on write and a dequantize on read
/// for memory it did not need. Above it, and for the contexts that do not fit
/// at all today, it is the only thing that helps.
///
/// ## What it costs, measured
///
/// A switch rather than an unconditional change for the reason
/// [`super::burn::attn_bf16`] is one: what it trades is precision, and the
/// honest way to price precision is to run both arms of the same binary
/// against the same harness. Unlike `attn_bf16`, this one has not won that
/// comparison, which is why it is off by default.
///
/// The codec itself is tight: a real-width BF16 row round-trips to within
/// `amax / 6` of every 16-element block, which IS the theoretical worst case
/// for NVFP4 — half the gap between the two widest E2M1 magnitudes. There is
/// no slack left in this file to recover.
///
/// What that costs downstream is another question, and on the one probe that
/// exists here it is not small. `the_fp4_cache_engages_at_real_width` runs a
/// single local layer — window 512, 5-token prefill, 16 decode steps,
/// synthetic sinusoidal input and weights, `[1, 4096]` output per step — and
/// reports, against the BF16 dense cache and worst over the sixteen steps:
///
/// ```text
/// NVFP4 cache : max-abs 4.9e-1 of the dense max-abs, RMS 9.1e-1 of the dense RMS
/// f32   cache : max-abs 6.2e-3                     , RMS 1.0e-2
/// ```
///
/// The second row is the control and it is the load-bearing part: it is the
/// trade `attn_bf16` already ships, measured the identical way on the
/// identical input. So the probe is NOT merely hypersensitive — it moves 1% for
/// BF16 and 91% for NVFP4, an ~88x larger perturbation. Both figures are one
/// synthetic layer and neither is a statement about the model; `golden/paired/`
/// is where that question is asked. But nobody should read "3.56x more context"
/// without also reading this.
///
/// ## That RMS figure is the WRONG GATE, and this is what should replace it
///
/// The 91% above is why this switch is off, and it should not be. It measures
/// PERTURBATION of a dense RMS on one synthetic layer, and nobody wants an
/// unperturbed RMS — what a long-context KV cache is FOR is retrieving
/// something from far back in the prompt. Those are different properties, and
/// the second does not follow from the first in either direction.
///
/// The direct evidence that it does not: the reference implementation SHIPS an
/// FP4 KV pool (`fp4_mx_block16`, the same 0.5625 bytes an element this file
/// stores) and retrieves a needle EXACTLY from a 307,581-token prompt, with a
/// natural stop. So the capability survives the numerics this probe rejects,
/// and the probe is rejecting a lane on a criterion its own users do not hold.
///
/// The replacement gate is a retrieval-on-long-context probe -- the reference's
/// `bench/probe_longctx.py` lifts directly -- run on both arms of one binary,
/// the same pairing every other comparison in this file uses. Pass is
/// "retrieves the needle at a context BF16 cannot hold at all", which is the
/// claim, rather than "perturbs an RMS by less than X", which is not.
///
/// ## Why it is still off TODAY, which is a third reason again
///
/// Neither the RMS figure nor the retrieval question is currently the binding
/// constraint. [`KvStore::materialize`] copies a page per layer, so this arm's
/// read path is "read 4.5 bits, write 16, read 16" -- MORE bandwidth than the
/// BF16 arm it replaces, not less. Until that copy goes, an on-by-default FP4
/// KV would be slower for the contexts that already fit and would prove nothing
/// about the ones that do not. Flip it when the copy is gone AND the retrieval
/// probe has run; not on the strength of either alone.
pub fn fp4_kv() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_FP4_KV")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

thread_local! {
    /// A per-thread override of [`fp4_kv`], for tests only.
    ///
    /// Exactly [`super::burn::CacheLane`]'s problem and exactly its answer. The
    /// env switch is a process-global `OnceLock`, so a test binary gets ONE arm
    /// and the other is never exercised — and for a change whose whole subject
    /// is a comparison between two arms, that means the interesting one can sit
    /// there unrun while everything passes. It nearly did: the cached-attention
    /// tests build 8-wide KV rows, which [`KvStore::new`] sends to the dense arm
    /// whatever the env var says, so `INK_FP4_KV=1` moved not one of them.
    static FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether THIS thread's new stores are NVFP4: the override if one is set, the
/// process default otherwise.
fn fp4_kv_now() -> bool {
    FORCED.with(|c| c.get()).unwrap_or_else(fp4_kv)
}

/// Force the KV element type for as long as this value lives. Tests only.
///
/// A guard rather than a closure for the reason [`super::burn::CacheLane`] is
/// one: it is a one-line addition at the top of a function body, where wrapping
/// the body would reindent it and hide the change in the diff.
#[cfg(test)]
pub(crate) struct Fp4Lane(Option<bool>);

#[cfg(test)]
impl Fp4Lane {
    /// NVFP4 pages, whatever the environment says.
    pub(crate) fn on() -> Self {
        Fp4Lane(FORCED.with(|c| c.replace(Some(true))))
    }

    /// Dense pages, whatever the environment says.
    pub(crate) fn off() -> Self {
        Fp4Lane(FORCED.with(|c| c.replace(Some(false))))
    }
}

#[cfg(test)]
impl Drop for Fp4Lane {
    fn drop(&mut self) {
        FORCED.with(|c| c.set(self.0));
    }
}

/// The narrowest logical row an NVFP4 page can hold.
///
/// Not a property of this file: [`quantize_nvfp4`] requires `k % 64 == 0`, and
/// that constraint is also exactly what makes the scale bytes pack into whole
/// `u32` words (`k / 16` scales, four to a word). Inkling's KV row is
/// `kv_heads * head_dim` = 8 x 128 = 1024, so it clears this by sixteen.
pub const FP4_ROW_ALIGN: usize = 64;

/// A page's worth of rows, in the reference's NVFP4 KV layout.
///
/// ## The layout, and the part that is easy to get wrong
///
/// The reference stores a page as payload `[page_size, heads, dim/2]` bytes
/// (two E2M1 codes to a byte) beside scales `[page_size, heads, dim/16]` E4M3,
/// with the quantization blocks running along the **FEATURE** dimension. A KV
/// row here is already head-major — head `h`'s `dim` features are contiguous at
/// `h * dim` — so a row-major NVFP4 quantization of the `[rows, heads * dim]`
/// page with block 16 along the row IS that layout, element for element, and
/// no block ever straddles two heads because `dim` is a multiple of 16.
///
/// Quantizing along the feature axis rather than across tokens is what makes
/// the store paged-safe at all: every row's scales are its own, so appending a
/// row cannot change any earlier row's numbers, and `drop_front`/`truncate` cut
/// at a row without re-quantizing anything.
///
/// ## Why the buffers wear `Int`
///
/// `codes` is `[rows, width / 8]` — one `u32` per eight codes — and `scales` is
/// `[rows, width / 64]` — one `u32` per four E4M3 bytes. Neither is a dtype
/// Burn names, and neither needs to be: what the paging core asks of a row is
/// `slice` and `cat`, which move bytes and do no arithmetic. Wearing `Int` buys
/// those two for free and byte-identically. Nothing ever reads them as
/// integers; the only consumer is the dequant kernel, which takes the raw
/// handle back.
#[derive(Clone, Debug)]
pub struct Fp4Rows {
    codes: Tensor<Bk, 2, Int>,
    scales: Tensor<Bk, 2, Int>,
    width: usize,
}

impl PageRows for Fp4Rows {
    fn rows(&self) -> usize {
        self.codes.dims()[0]
    }

    fn width(&self) -> usize {
        self.width
    }

    fn slice_rows(&self, from: usize, to: usize) -> Self {
        let cw = self.codes.dims()[1];
        let sw = self.scales.dims()[1];
        Self {
            codes: self.codes.clone().slice([from..to, 0..cw]),
            scales: self.scales.clone().slice([from..to, 0..sw]),
            width: self.width,
        }
    }

    fn concat(parts: Vec<Self>) -> Self {
        let width = parts[0].width;
        assert!(
            parts.iter().all(|p| p.width == width),
            "concatenating NVFP4 rows of different widths"
        );
        Self {
            codes: Tensor::cat(parts.iter().map(|p| p.codes.clone()).collect(), 0),
            scales: Tensor::cat(parts.iter().map(|p| p.scales.clone()).collect(), 0),
            width,
        }
    }
}

/// One layer's keys or values, stored as NVFP4 pages.
///
/// The same [`Pages`] core [`PageStore`] uses, over [`Fp4Rows`] instead of over
/// dense tensors. Quantization happens on APPEND and dequantization on
/// [`Fp4PageStore::materialize`], which is the shape that costs the attention
/// kernels nothing: they go on receiving the tensor they always received, at
/// the dtype they always received it, and the four-bit form exists only between
/// the two calls — which is where the context lives, and therefore where the
/// bytes are.
#[derive(Clone)]
pub struct Fp4PageStore {
    pages: Pages<Fp4Rows>,
    width: usize,
    /// The dtype a caller appends and expects back. Recorded rather than fixed
    /// because the KV lane has two arms (see `attn_bf16`) and a store that
    /// silently returned f32 to a BF16 lane would insert a widening the lane
    /// spent a whole change removing.
    dtype: DType,
    /// The client the pages were allocated on, captured at the first append.
    ///
    /// Taken from a tensor that is already on the device rather than from
    /// `CudaRuntime::client(&Default::default())`, for the reason
    /// [`super::seam::client_of`] gives: two calls to that are *meant* to
    /// return the same client, and "meant to" is not a thing to bet a pointer
    /// on.
    client: Option<ComputeClient<CudaRuntime>>,
}

impl Fp4PageStore {
    /// An empty store for rows of `width` logical columns, holding and
    /// returning `dtype`.
    ///
    /// `width` must be a positive multiple of [`FP4_ROW_ALIGN`]; see there for
    /// why that is not this file's rule to bend.
    pub fn new(width: usize, dtype: DType) -> Self {
        assert!(
            width > 0 && width.is_multiple_of(FP4_ROW_ALIGN),
            "an NVFP4 KV row must be a positive multiple of {FP4_ROW_ALIGN}, got {width}"
        );
        assert!(
            matches!(dtype, DType::F32 | DType::BF16),
            "an NVFP4 KV store quantizes from f32 or bf16, not {dtype:?}"
        );
        Self {
            pages: Pages::new(),
            width,
            dtype,
            client: None,
        }
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// LOGICAL columns per row — what the caller appends and gets back, not the
    /// width of anything stored.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Quantize `rows` to NVFP4 and append them.
    ///
    /// The quantization is per row and per 16 features, so this touches no row
    /// already in the store — including the one in a half-filled tail page,
    /// which is joined to as packed codes and never decoded and re-encoded. A
    /// row therefore carries exactly one rounding, no matter how many appends
    /// happen after it.
    pub fn append(&mut self, rows: Tensor<Bk, 2>) {
        let [n, w] = rows.dims();
        assert_eq!(
            w, self.width,
            "a {w}-wide row into a {}-wide store",
            self.width
        );
        if n == 0 {
            return;
        }
        let client = seam::client_of(&rows);
        let device = rows.device();
        let (handle, dt) = seam::handle_of_any(rows);
        assert_eq!(
            dt, self.dtype,
            "a {dt:?} row into a store that promised {:?}",
            self.dtype
        );
        let (codes, scales) = match dt {
            DType::BF16 => quantize_nvfp4_bf16(&client, &handle, n, w),
            DType::F32 => quantize_nvfp4(&client, &handle, n, w),
            other => panic!("an NVFP4 KV store cannot quantize {other:?}"),
        };
        let page = Fp4Rows {
            codes: seam::int_tensor_of(client.clone(), device.clone(), codes, n, w / 8),
            scales: seam::int_tensor_of(client.clone(), device, scales, n, w / 64),
            width: w,
        };
        self.client = Some(client);
        self.pages.append(page);
    }

    /// Drop `n` rows from the FRONT — the sliding window advancing.
    pub fn drop_front(&mut self, n: usize) {
        self.pages.drop_front(n);
    }

    /// Truncate to `keep` logical rows — a speculative batch being rejected.
    pub fn truncate(&mut self, keep: usize) {
        self.pages.truncate(keep);
    }

    /// The rows as one contiguous dense tensor, in order, at the dtype they
    /// were appended in.
    ///
    /// One dequant launch over the whole retained context, after the pages have
    /// been joined — not one per page. The join is `cat` on the PACKED buffers,
    /// so it moves 4.5 bits a value where the dense store moved sixteen.
    pub fn materialize(&self, dev: &burn::backend::cuda::CudaDevice) -> Tensor<Bk, 2> {
        let Some(all) = self.pages.gather() else {
            let empty = Tensor::<Bk, 2>::zeros([0, self.width], dev);
            return match self.dtype {
                DType::BF16 => empty.cast(burn::tensor::FloatDType::BF16),
                _ => empty,
            };
        };
        let n = all.rows();
        let client = self
            .client
            .clone()
            .expect("a non-empty NVFP4 store was filled, so it has a client");
        let codes = seam::int_handle_of(all.codes);
        let scales = seam::int_handle_of(all.scales);
        let out = match self.dtype {
            DType::BF16 => dequantize_nvfp4_bf16(&client, &codes, &scales, n, self.width),
            _ => dequantize_nvfp4(&client, &codes, &scales, n, self.width),
        };
        seam::tensor_of_dt(client, dev.clone(), out, n, self.width, self.dtype)
    }

    /// Share the first `rows` logical rows with a new store, without copying.
    pub fn share_prefix(&self, rows: usize) -> Option<Self> {
        Some(Self {
            pages: self.pages.share_prefix(rows)?,
            width: self.width,
            dtype: self.dtype,
            client: self.client.clone(),
        })
    }

    /// Panics unless every documented invariant holds.
    pub fn assert_sound(&self) {
        self.pages.assert_sound(self.width);
    }
}

/// Which of the two stores a KV cache is holding.
///
/// The arm is chosen once, at construction, from [`fp4_kv`] — not per append,
/// because a store that changed its element type halfway would have two
/// roundings on some rows and one on others, and no way for a reader to know
/// which. Everything the cache does to a store other than filling and reading
/// it (`len`, both trims) is arithmetic on row counts and is written once here.
#[derive(Clone)]
pub enum KvStore<B: Backend> {
    /// Dense, at the dtype the lane hands it. What the cache has always done.
    Wide(PageStore<B>),
    /// NVFP4: 4.5 bits a value, dequantized on read.
    Fp4(Fp4PageStore),
}

impl<B: Backend> KvStore<B> {
    /// Logical rows currently held.
    pub fn len(&self) -> usize {
        match self {
            Self::Wide(s) => s.len(),
            Self::Fp4(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// LOGICAL columns per row.
    pub fn width(&self) -> usize {
        match self {
            Self::Wide(s) => s.width(),
            Self::Fp4(s) => s.width(),
        }
    }

    /// Whether this store is holding NVFP4 pages — the one place the choice is
    /// observable, so that a test can assert the switch did something rather
    /// than assert that the numbers still look plausible.
    pub fn is_fp4(&self) -> bool {
        matches!(self, Self::Fp4(_))
    }

    /// Drop `n` rows from the FRONT — the sliding window advancing.
    pub fn drop_front(&mut self, n: usize) {
        match self {
            Self::Wide(s) => s.drop_front(n),
            Self::Fp4(s) => s.drop_front(n),
        }
    }

    /// Truncate to `keep` logical rows — a speculative batch being rejected.
    pub fn truncate(&mut self, keep: usize) {
        match self {
            Self::Wide(s) => s.truncate(keep),
            Self::Fp4(s) => s.truncate(keep),
        }
    }

    /// Share the first `rows` logical rows with a new store, without copying.
    pub fn share_prefix(&self, rows: usize) -> Option<Self> {
        match self {
            Self::Wide(s) => s.share_prefix(rows).map(Self::Wide),
            Self::Fp4(s) => s.share_prefix(rows).map(Self::Fp4),
        }
    }

    /// Panics unless every documented invariant holds.
    pub fn assert_sound(&self) {
        match self {
            Self::Wide(s) => s.assert_sound(),
            Self::Fp4(s) => s.assert_sound(),
        }
    }
}

impl KvStore<Bk> {
    /// An empty store for `width`-column rows of `dtype`, on whichever arm
    /// [`fp4_kv`] selects.
    ///
    /// A row that is not a multiple of [`FP4_ROW_ALIGN`] falls back to the
    /// dense arm rather than panicking: the switch is a global, this
    /// constructor is reached from more than the one 1024-wide KV row, and a
    /// process-wide env var should not be able to turn an unrelated width into
    /// a crash.
    pub fn new(width: usize, dtype: DType) -> Self {
        if fp4_kv_now() && width > 0 && width.is_multiple_of(FP4_ROW_ALIGN) {
            Self::Fp4(Fp4PageStore::new(width, dtype))
        } else {
            Self::Wide(PageStore::new(width))
        }
    }

    /// An empty dense store — the arm that ignores [`fp4_kv`].
    pub fn wide(width: usize) -> Self {
        Self::Wide(PageStore::new(width))
    }

    /// An empty NVFP4 store — the arm that ignores [`fp4_kv`].
    pub fn fp4(width: usize, dtype: DType) -> Self {
        Self::Fp4(Fp4PageStore::new(width, dtype))
    }

    /// Append `rows`, quantizing first on the NVFP4 arm.
    pub fn append(&mut self, rows: Tensor<Bk, 2>) {
        match self {
            Self::Wide(s) => s.append(rows),
            Self::Fp4(s) => s.append(rows),
        }
    }

    /// The rows as one contiguous dense tensor, in order — what the attention
    /// read wants, on either arm.
    pub fn materialize(&self, dev: &burn::backend::cuda::CudaDevice) -> Tensor<Bk, 2> {
        match self {
            Self::Wide(s) => s.materialize(dev),
            Self::Fp4(s) => s.materialize(dev),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The same backend the cache tests use: these are page-arithmetic properties,
    // but they are properties OF burn tensors, so they are checked on the device
    // the cache actually lives on rather than on a convenient stand-in.
    type B = burn::backend::Cuda<f32>;

    const W: usize = 4;

    /// Rows whose every element is the row's absolute index, so `materialize`
    /// can be checked for CONTENT and ORDER rather than only for shape — a
    /// store that returns the right number of wrong rows passes a shape test.
    fn rows(from: usize, n: usize) -> Tensor<B, 2> {
        let data: Vec<f32> = (from..from + n)
            .flat_map(|i| std::iter::repeat_n(i as f32, W))
            .collect();
        Tensor::<B, 1>::from_floats(data.as_slice(), &Default::default()).reshape([n, W])
    }

    fn contents(s: &PageStore<B>) -> Vec<usize> {
        if s.is_empty() {
            return Vec::new();
        }
        let flat: Vec<f32> = s
            .materialize(&Default::default())
            .into_data()
            .to_vec()
            .unwrap();
        flat.chunks(W)
            .map(|c| {
                assert!(c.iter().all(|x| *x == c[0]), "a row was torn: {c:?}");
                c[0] as usize
            })
            .collect()
    }

    #[test]
    fn append_spans_page_boundaries_and_keeps_order() {
        let mut s = PageStore::<B>::new(W);
        // deliberately unaligned batches, crossing PAGE more than once
        for (from, n) in [(0, 5), (5, PAGE), (5 + PAGE, 1), (6 + PAGE, 2 * PAGE)] {
            s.append(rows(from, n));
            s.assert_sound();
        }
        let want: Vec<usize> = (0..6 + 3 * PAGE).collect();
        assert_eq!(contents(&s), want);
    }

    #[test]
    fn front_drop_is_the_sliding_window_and_may_land_mid_page() {
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, 3 * PAGE));
        s.drop_front(1); // mid-page, releases nothing
        s.assert_sound();
        assert_eq!(contents(&s).first().copied(), Some(1));
        s.drop_front(PAGE); // now crosses a boundary
        s.assert_sound();
        assert_eq!(contents(&s).first().copied(), Some(1 + PAGE));
        assert_eq!(s.len(), 3 * PAGE - 1 - PAGE);
        assert_eq!(contents(&s).last().copied(), Some(3 * PAGE - 1));
    }

    #[test]
    fn truncate_is_a_rejected_draft_and_survives_a_later_append() {
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, PAGE + 10));
        s.truncate(PAGE + 4); // reject 6 drafted rows
        s.assert_sound();
        assert_eq!(s.len(), PAGE + 4);
        // the accepted token then continues from where the kept rows end
        s.append(rows(PAGE + 4, 3));
        s.assert_sound();
        let want: Vec<usize> = (0..PAGE + 7).collect();
        assert_eq!(contents(&s), want);
    }

    #[test]
    fn both_ends_compose() {
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, 2 * PAGE + 7));
        s.drop_front(PAGE + 3);
        s.truncate(s.len() - 5);
        s.assert_sound();
        let want: Vec<usize> = (PAGE + 3..2 * PAGE + 2).collect();
        assert_eq!(contents(&s), want);
    }

    #[test]
    fn a_shared_prefix_is_the_same_rows_and_does_not_move_when_the_parent_grows() {
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, 2 * PAGE + 40));
        let shared = s.share_prefix(2 * PAGE).expect("page-aligned prefix");
        shared.assert_sound();
        assert_eq!(contents(&shared), (0..2 * PAGE).collect::<Vec<_>>());

        // The parent keeps generating. The share must not follow it — that is
        // the whole promise, and a Vec of handles could easily alias.
        s.append(rows(2 * PAGE + 40, 200));
        assert_eq!(shared.len(), 2 * PAGE);
        assert_eq!(contents(&shared), (0..2 * PAGE).collect::<Vec<_>>());
    }

    #[test]
    fn an_unaligned_or_offset_prefix_is_refused_rather_than_silently_copied() {
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, 3 * PAGE));
        // not a page boundary: the last page would be written through by
        // whichever store appended next
        assert!(s.share_prefix(PAGE + 1).is_none());
        assert!(s.share_prefix(4 * PAGE).is_none(), "longer than the store");
        // once the front has moved, page 0 is not the prefix's first row
        s.drop_front(1);
        assert!(s.share_prefix(PAGE).is_none(), "offset store cannot share");
    }

    #[test]
    fn emptying_by_either_end_leaves_a_reusable_store() {
        for by_front in [true, false] {
            let mut s = PageStore::<B>::new(W);
            s.append(rows(0, PAGE + 3));
            if by_front {
                s.drop_front(PAGE + 3)
            } else {
                s.truncate(0)
            }
            s.assert_sound();
            assert!(s.is_empty());
            s.append(rows(0, 2));
            s.assert_sound();
            assert_eq!(contents(&s), vec![0, 1]);
        }
    }

    // -----------------------------------------------------------------------
    // NVFP4 pages
    //
    // Same question as above and a different arithmetic: does the STORE keep
    // the rows it was given, in the order it was given them. So these reuse the
    // index-carrying-row trick — each row is a constant equal to its own
    // absolute index + 1 — because a store that returns the right number of
    // wrong rows passes a shape test, and NVFP4 does not change that.
    //
    // What it DOES change is the comparison. Four bits with one E4M3 scale per
    // sixteen recovers a value to within half an E4M3 ulp, which is 1/16, so
    // every assertion here is relative and none is exact. Asserting equality
    // would be asserting a theorem about lossless quantization that is false.
    // -----------------------------------------------------------------------

    /// Widest thing NVFP4 can be off by, as a fraction, for a value that is not
    /// tiny inside its own block.
    ///
    /// Not a fudge factor picked by running the test: a block's scale is
    /// `E4M3(amax / 6)` and a value at the amax lands on code 7, so what it
    /// recovers is `6 * E4M3(amax / 6)`. E4M3 carries three mantissa bits, so
    /// round-to-nearest is off by at most half of the 1/8 relative spacing.
    const FP4_TOL: f32 = 1.0 / 16.0;

    /// Logical row width for the FP4 tests. `quantize_nvfp4` requires a multiple
    /// of 64; 128 is Inkling's `head_dim`, i.e. one head's worth, which is the
    /// smallest slice of a real KV row that has the same block geometry.
    const FW: usize = 128;

    fn fp4_dev() -> burn::backend::cuda::CudaDevice {
        Default::default()
    }

    /// Rows carrying their own absolute index — in the SIGN BITS, not in the
    /// magnitude.
    ///
    /// The dense tests write the index into every element and read it straight
    /// back. That trick does not survive four bits and it should not be made to:
    /// a constant row recovers as `6 * E4M3(v / 6)`, whose granularity is 1/16
    /// relative, so consecutive indices above ~16 land on the same value. A test
    /// that "fixed" this with a tolerance would stop being able to tell rows
    /// apart at all, which is exactly the property it exists to check.
    ///
    /// So the index goes where NVFP4 is exact. Every element is `±1`, so every
    /// 16-element block has `amax = 1`, one scale, and code 7 with a sign; the
    /// SIGN survives quantization bit for bit while the magnitude does not.
    /// Feature `j` carries bit `j % 16` of the index, repeated across all eight
    /// blocks of the row — so a block that was dropped, duplicated or swapped
    /// with a neighbour's shows up as a disagreement WITHIN the row, on top of
    /// the row order the caller is checking.
    fn frows(from: usize, n: usize) -> Tensor<B, 2> {
        let data: Vec<f32> = (from..from + n)
            .flat_map(|i| (0..FW).map(move |j| if (i >> (j % 16)) & 1 == 1 { 1.0 } else { -1.0 }))
            .collect();
        Tensor::<B, 1>::from_floats(data.as_slice(), &fp4_dev()).reshape([n, FW])
    }

    /// The index each stored row is carrying, read back out of the sign bits.
    ///
    /// Exact, with no tolerance anywhere: the claim being made is about which
    /// rows are where, and that claim is not approximate just because their
    /// contents are.
    fn fcontents(s: &Fp4PageStore) -> Vec<usize> {
        if s.is_empty() {
            return Vec::new();
        }
        let flat: Vec<f32> = s.materialize(&fp4_dev()).into_data().to_vec().unwrap();
        flat.chunks(FW)
            .map(|row| {
                let bits = |o: usize| -> usize {
                    (0..16).filter(|j| row[o + j] > 0.0).map(|j| 1 << j).sum()
                };
                let idx = bits(0);
                for b in 1..FW / 16 {
                    assert_eq!(
                        bits(b * 16),
                        idx,
                        "block {b} of a row disagrees with block 0 — the row was torn"
                    );
                }
                // All sixteen magnitudes in a block are equal by construction,
                // so the reconstruction is one value repeated. Say so, since a
                // scale applied to the wrong block would show up here first.
                let m = row[0].abs();
                assert!(
                    row.iter().all(|x| x.abs() == m),
                    "a ±1 row came back with unequal magnitudes"
                );
                assert!(
                    (m - 1.0).abs() <= FP4_TOL,
                    "a ±1 row recovered magnitude {m}"
                );
                idx
            })
            .collect()
    }

    #[test]
    fn fp4_append_spans_page_boundaries_and_keeps_order() {
        let mut s = Fp4PageStore::new(FW, DType::F32);
        for (from, n) in [(0, 5), (5, PAGE), (5 + PAGE, 1), (6 + PAGE, 2 * PAGE)] {
            s.append(frows(from, n));
            s.assert_sound();
        }
        let want: Vec<usize> = (0..6 + 3 * PAGE).collect();
        assert_eq!(fcontents(&s), want);
    }

    #[test]
    fn fp4_both_ends_compose_and_a_prefix_still_shares() {
        let mut s = Fp4PageStore::new(FW, DType::F32);
        s.append(frows(0, 2 * PAGE + 7));
        let shared = s.share_prefix(2 * PAGE).expect("page-aligned prefix");
        s.drop_front(PAGE + 3);
        s.truncate(s.len() - 5);
        s.assert_sound();
        assert_eq!(fcontents(&s), (PAGE + 3..2 * PAGE + 2).collect::<Vec<_>>());
        // The share must not have followed either trim — the same promise the
        // dense store makes, and the one a Vec of handles could easily break.
        shared.assert_sound();
        assert_eq!(fcontents(&shared), (0..2 * PAGE).collect::<Vec<_>>());
    }

    #[test]
    fn fp4_pages_carry_the_reference_payload_and_scale_geometry() {
        // The reference stores a page as payload `[page_size, heads, dim / 2]`
        // bytes and scales `[page_size, heads, dim / 16]` E4M3. For one head's
        // width that is `dim / 2` payload bytes and `dim / 16` scale bytes per
        // row; here they are packed four and one to a `u32` word respectively.
        let mut s = Fp4PageStore::new(FW, DType::F32);
        s.append(frows(0, 3));
        let page = s.pages.gather().expect("three rows");
        assert_eq!(
            page.codes.dims(),
            [3, FW / 8],
            "payload is two codes a byte"
        );
        assert_eq!(page.scales.dims(), [3, FW / 64], "one E4M3 per 16 features");
        // ...which is the 3.56x the whole feature is for: 4 bits a value plus one
        // E4M3 byte per 16, against BF16's 16.
        let packed = (FW / 8 + FW / 64) * 4;
        assert_eq!(packed * 32, FW * 2 * 9, "4.5 bits a value, not 16");
    }

    #[test]
    fn fp4_blocks_run_along_the_feature_dimension() {
        // Alternating 16-wide blocks two and a half orders of magnitude apart.
        // With one scale per 16 FEATURES both recover; with one scale per row —
        // or per anything wider — the small blocks divide by the large blocks'
        // scale, `200 / 6 = 33.3`, and `0.5 / 33.3` rounds to code 0.
        let vals: Vec<f32> = (0..FW)
            .map(|j| if (j / 16) % 2 == 0 { 0.5 } else { 200.0 })
            .collect();
        let mut s = Fp4PageStore::new(FW, DType::F32);
        s.append(Tensor::<B, 1>::from_floats(vals.as_slice(), &fp4_dev()).reshape([1, FW]));
        let got: Vec<f32> = s.materialize(&fp4_dev()).into_data().to_vec().unwrap();
        for (j, (g, w)) in got.iter().zip(vals.iter()).enumerate() {
            assert!(
                (g - w).abs() <= FP4_TOL * w,
                "feature {j}: {g} for {w} — blocks are not on the feature axis"
            );
        }
    }

    /// The store's own contract, at the geometry and the dtype the runtime
    /// actually uses: `kv_heads * head_dim` = 1024-wide rows, held BF16.
    ///
    /// The tests above are 128 wide and f32, which is one head and the wide
    /// lane — neither is what a decode step holds. This one is, and it is the
    /// test that says whether a drift measured further downstream belongs to
    /// this file or to what reads it.
    ///
    /// The bound is per 16-element BLOCK and scaled by that block's amax, which
    /// is the only yardstick NVFP4 supports: a block's scale is `amax / 6`, so
    /// the absolute error an element can carry is set by the block it is in and
    /// not by its own size. An element near zero inside a block with a large
    /// amax is allowed to be wrong by a lot in relative terms, and a test that
    /// demanded otherwise would be demanding something four bits cannot do.
    #[test]
    fn a_real_width_bf16_row_round_trips_within_the_nvfp4_block_bound() {
        const KW: usize = 1024;
        let n = 40usize;
        // Structured rather than constant, and spanning a wide dynamic range
        // within each block, so the per-block scale is actually doing work.
        let data: Vec<f32> = (0..n * KW)
            .map(|i| {
                let t = ((i * 2_654_435_761usize) % 2003) as f32 / 1001.5 - 1.0;
                t * t * t * 8.0
            })
            .collect();
        let src = Tensor::<B, 1>::from_floats(data.as_slice(), &fp4_dev())
            .reshape([n, KW])
            .cast(burn::tensor::FloatDType::BF16);

        let mut s = Fp4PageStore::new(KW, DType::BF16);
        s.append(src.clone());
        s.assert_sound();
        assert_eq!(s.len(), n);

        let a: Vec<f32> = src
            .cast(burn::tensor::FloatDType::F32)
            .into_data()
            .to_vec()
            .unwrap();
        let b: Vec<f32> = s
            .materialize(&fp4_dev())
            .cast(burn::tensor::FloatDType::F32)
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(a.len(), b.len());

        let mut worst = 0f32;
        for blk in 0..a.len() / 16 {
            let lo = blk * 16;
            let amax = a[lo..lo + 16].iter().fold(0.0f32, |m, x| m.max(x.abs()));
            for i in lo..lo + 16 {
                worst = worst.max((a[i] - b[i]).abs() / amax.max(1e-6));
            }
        }
        println!("real-width bf16 KV round trip: worst {worst:e} of block amax");
        // Half the widest gap between adjacent E2M1 magnitudes is `2 / 6` of
        // the amax (between 4 and 6), plus the E4M3 scale's own 1/16. Anything
        // under that is the codec; anything over it is a bug in this file.
        assert!(
            worst < 1.0 / 3.0 + 1.0 / 16.0,
            "a real-width BF16 row lost {worst} of its block amax"
        );
    }

    #[test]
    fn the_switch_picks_an_arm_and_both_arms_hold_the_same_rows() {
        // `fp4_kv()` is a process-global `OnceLock`, so a test binary gets ONE
        // reading of the env var and cannot exercise both lanes through it. The
        // two explicit constructors are what makes the comparison runnable at
        // all, the same way `CacheLane` does for the BF16 cache.
        assert!(KvStore::<Bk>::fp4(FW, DType::F32).is_fp4());
        assert!(!KvStore::<Bk>::wide(FW).is_fp4());

        // Something with structure, so the agreement is about the quantizer and
        // not about a constant that happens to survive.
        let n = PAGE + 9;
        let data: Vec<f32> = (0..n * FW)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.25 + 1.0)
            .collect();
        let t = Tensor::<B, 1>::from_floats(data.as_slice(), &fp4_dev()).reshape([n, FW]);

        let mut wide = KvStore::<Bk>::wide(FW);
        let mut narrow = KvStore::<Bk>::fp4(FW, DType::F32);
        wide.append(t.clone());
        narrow.append(t);
        wide.drop_front(3);
        narrow.drop_front(3);
        wide.truncate(wide.len() - 4);
        narrow.truncate(narrow.len() - 4);
        assert_eq!(wide.len(), narrow.len());

        let a: Vec<f32> = wide.materialize(&fp4_dev()).into_data().to_vec().unwrap();
        let b: Vec<f32> = narrow.materialize(&fp4_dev()).into_data().to_vec().unwrap();
        assert_eq!(a.len(), b.len());
        // Per 16-element block the scale is `amax / 6`, so the absolute error is
        // bounded by half the gap between the two largest E2M1 magnitudes times
        // that scale — `amax / 6` is the right yardstick, not each element's own
        // value, and an element near zero inside a block with a large amax is
        // allowed to be wrong by a lot in relative terms.
        for blk in 0..a.len() / 16 {
            let lo = blk * 16;
            let amax = a[lo..lo + 16].iter().fold(0.0f32, |m, x| m.max(x.abs()));
            for i in lo..lo + 16 {
                assert!(
                    (a[i] - b[i]).abs() <= 0.2 * amax + 1e-6,
                    "element {i}: wide {} vs fp4 {} (block amax {amax})",
                    a[i],
                    b[i]
                );
            }
        }
    }
}
