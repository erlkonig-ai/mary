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

use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Rows per page. 128 matches the reference implementation's `page_size`, which
/// is what its FP4 payload/scale shapes are cut to; keeping the same number
/// means a later FP4 store is a change of element type rather than of geometry.
pub const PAGE: usize = 128;

/// One layer's keys or values, stored as pages of at most [`PAGE`] rows.
///
/// Invariants, checked by [`PageStore::assert_sound`]:
/// * every page but the last holds exactly `PAGE` rows;
/// * `head < PAGE`, and `head` counts rows already dropped from page 0;
/// * `len` is the LOGICAL row count, excluding `head`;
/// * `head + len` equals the total rows actually stored.
#[derive(Clone, Debug)]
pub struct PageStore<B: Backend> {
    pages: Vec<Tensor<B, 2>>,
    head: usize,
    len: usize,
    width: usize,
}

impl<B: Backend> PageStore<B> {
    /// An empty store for rows of `width` columns.
    pub fn new(width: usize) -> Self {
        Self {
            pages: Vec::new(),
            head: 0,
            len: 0,
            width,
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

    /// Append `rows`, filling the last partial page before starting a new one.
    ///
    /// Takes the whole batch at once rather than row by row: a speculative
    /// verify appends `k + 1` rows and splitting them into `k + 1` slice
    /// assignments would copy the tail page that many times.
    pub fn append(&mut self, rows: Tensor<B, 2>) {
        let [n, w] = rows.dims();
        assert_eq!(
            w, self.width,
            "a {w}-wide row into a {}-wide store",
            self.width
        );
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
            let part = rows.clone().slice([0..take, 0..w]);
            self.pages[last] = Tensor::cat(vec![self.pages[last].clone(), part], 0);
            written = take;
        }
        while written < n {
            let take = PAGE.min(n - written);
            self.pages
                .push(rows.clone().slice([written..written + take, 0..w]));
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
            let w = self.width;
            self.pages[last] = self.pages[last].clone().slice([0..tail, 0..w]);
        }
        if self.len == 0 {
            self.pages.clear();
            self.head = 0;
        }
    }

    /// The rows as one contiguous tensor, in order.
    ///
    /// This is what the present attention read wants. It concatenates, so it is
    /// no cheaper than the contiguous cache was — the win is elsewhere, see the
    /// module docs.
    pub fn materialize(&self, dev: &B::Device) -> Tensor<B, 2> {
        if self.len == 0 {
            return Tensor::zeros([0, self.width], dev);
        }
        let mut out = Vec::with_capacity(self.pages.len());
        let mut skip = self.head;
        let mut left = self.len;
        for p in &self.pages {
            let rows = p.dims()[0];
            let from = skip.min(rows);
            skip -= from;
            if from >= rows || left == 0 {
                continue;
            }
            let take = (rows - from).min(left);
            out.push(p.clone().slice([from..from + take, 0..self.width]));
            left -= take;
        }
        debug_assert_eq!(left, 0, "materialize lost rows");
        if out.len() == 1 {
            out.pop().unwrap()
        } else {
            Tensor::cat(out, 0)
        }
    }

    /// Share the first `rows` logical rows with a new store, without copying.
    ///
    /// The point of the file. Burn clones a tensor by handle, so the returned
    /// store references the same device buffers; recomputing a prefix's KV is
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
            width: self.width,
        })
    }

    /// Panics unless every documented invariant holds.
    pub fn assert_sound(&self) {
        assert!(self.head < PAGE, "head {} is a whole page", self.head);
        let stored = self.stored();
        assert_eq!(
            self.pages.len(),
            stored.div_ceil(PAGE),
            "{} pages for {stored} stored rows",
            self.pages.len()
        );
        for (i, p) in self.pages.iter().enumerate() {
            let [n, w] = p.dims();
            assert_eq!(w, self.width, "page {i} is {w} wide");
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
    fn rows(from: usize, n: usize, dev: &B::Device) -> Tensor<B, 2> {
        let data: Vec<f32> = (from..from + n)
            .flat_map(|i| std::iter::repeat_n(i as f32, W))
            .collect();
        Tensor::<B, 1>::from_floats(data.as_slice(), dev).reshape([n, W])
    }

    fn contents(s: &PageStore<B>, dev: &B::Device) -> Vec<usize> {
        if s.is_empty() {
            return Vec::new();
        }
        let flat: Vec<f32> = s.materialize(dev).into_data().to_vec().unwrap();
        flat.chunks(W)
            .map(|c| {
                assert!(c.iter().all(|x| *x == c[0]), "a row was torn: {c:?}");
                c[0] as usize
            })
            .collect()
    }

    #[test]
    fn append_spans_page_boundaries_and_keeps_order() {
        let dev = Default::default();
        let mut s = PageStore::<B>::new(W);
        // deliberately unaligned batches, crossing PAGE more than once
        for (from, n) in [(0, 5), (5, PAGE), (5 + PAGE, 1), (6 + PAGE, 2 * PAGE)] {
            s.append(rows(from, n, &dev));
            s.assert_sound();
        }
        let want: Vec<usize> = (0..6 + 3 * PAGE).collect();
        assert_eq!(contents(&s, &dev), want);
    }

    #[test]
    fn front_drop_is_the_sliding_window_and_may_land_mid_page() {
        let dev = Default::default();
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, 3 * PAGE, &dev));
        s.drop_front(1); // mid-page, releases nothing
        s.assert_sound();
        assert_eq!(contents(&s, &dev).first().copied(), Some(1));
        s.drop_front(PAGE); // now crosses a boundary
        s.assert_sound();
        assert_eq!(contents(&s, &dev).first().copied(), Some(1 + PAGE));
        assert_eq!(s.len(), 3 * PAGE - 1 - PAGE);
        assert_eq!(contents(&s, &dev).last().copied(), Some(3 * PAGE - 1));
    }

    #[test]
    fn truncate_is_a_rejected_draft_and_survives_a_later_append() {
        let dev = Default::default();
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, PAGE + 10, &dev));
        s.truncate(PAGE + 4); // reject 6 drafted rows
        s.assert_sound();
        assert_eq!(s.len(), PAGE + 4);
        // the accepted token then continues from where the kept rows end
        s.append(rows(PAGE + 4, 3, &dev));
        s.assert_sound();
        let want: Vec<usize> = (0..PAGE + 7).collect();
        assert_eq!(contents(&s, &dev), want);
    }

    #[test]
    fn both_ends_compose() {
        let dev = Default::default();
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, 2 * PAGE + 7, &dev));
        s.drop_front(PAGE + 3);
        s.truncate(s.len() - 5);
        s.assert_sound();
        let want: Vec<usize> = (PAGE + 3..2 * PAGE + 2).collect();
        assert_eq!(contents(&s, &dev), want);
    }

    #[test]
    fn a_shared_prefix_is_the_same_rows_and_does_not_move_when_the_parent_grows() {
        let dev = Default::default();
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, 2 * PAGE + 40, &dev));
        let shared = s.share_prefix(2 * PAGE).expect("page-aligned prefix");
        shared.assert_sound();
        assert_eq!(contents(&shared, &dev), (0..2 * PAGE).collect::<Vec<_>>());

        // The parent keeps generating. The share must not follow it — that is
        // the whole promise, and a Vec of handles could easily alias.
        s.append(rows(2 * PAGE + 40, 200, &dev));
        assert_eq!(shared.len(), 2 * PAGE);
        assert_eq!(contents(&shared, &dev), (0..2 * PAGE).collect::<Vec<_>>());
    }

    #[test]
    fn an_unaligned_or_offset_prefix_is_refused_rather_than_silently_copied() {
        let dev = Default::default();
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, 3 * PAGE, &dev));
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
        let dev = Default::default();
        for by_front in [true, false] {
            let mut s = PageStore::<B>::new(W);
            s.append(rows(0, PAGE + 3, &dev));
            if by_front {
                s.drop_front(PAGE + 3)
            } else {
                s.truncate(0)
            }
            s.assert_sound();
            assert!(s.is_empty());
            s.append(rows(0, 2, &dev));
            s.assert_sound();
            assert_eq!(contents(&s, &dev), vec![0, 1]);
        }
    }
}
