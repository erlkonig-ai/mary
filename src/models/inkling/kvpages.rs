//! A paged store for one attention layer's keys or values.
//!
//! ## Why pages, when a contiguous tensor already works
//!
//! [`PageStore::materialize`] concatenates, so a decode step that reads through
//! it pays what the contiguous cache paid — which is why the attention read
//! does not: [`KvStore::parts`] hands the pages over as they are and the score
//! and output products decompose over them. Beyond that, pages buy three things
//! the contiguous form cannot express.
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

/// How many pages a store keeps before merging the settled ones into one.
///
/// The read costs launches per PAGE and the merge costs a copy per MERGE, so
/// this is the one knob between them. A merge leaves two pages — the merged run
/// and the tail being filled — so the count oscillates between 2 and this, and
/// the read is never more than [`MAX_PAGES`] chunks HOWEVER LONG the context
/// is. Getting back from 2 to 9 takes `(MAX_PAGES - 1) * PAGE = 896` rows, so a
/// merge copies the retained context once every 896 appends, against the
/// once-per-append copy `gather` used to do.
///
/// Both ends of that trade get cheaper as the context grows — the launch count
/// stops growing and the copy is amortized over more rows — which is why it is
/// a constant and not a tuning parameter. Below `MAX_PAGES * PAGE = 1024` rows
/// nothing merges at all and the read is simply the pages, one chunk each.
pub const MAX_PAGES: usize = 8;

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
    /// `rows` rows of zeros, of this value's width and element type.
    ///
    /// Zeros and not uninitialised memory, and the reason is the same one
    /// `PagedKv`'s tail pad gives: a page is read WHOLE, so the rows past
    /// [`Pages::fill`] reach the attention products. They are masked to `-inf`
    /// in the scores, which makes their probability exactly zero, and `0 * NaN`
    /// is NaN. A dead row has to be FINITE; zero is the cheapest finite thing.
    fn zeroed(&self, rows: usize) -> Self;
    /// Overwrite rows `at .. at + src.rows()` with `src`, IN PLACE.
    ///
    /// In place is the whole point of the capacity page: the alternative is the
    /// `concat` that used to grow the tail by a row a step, which allocates a
    /// buffer of a size never seen before on every single decode step. An
    /// implementation must take the buffer out of `self` before writing it
    /// (`std::mem::replace`), because a clone leaves two live references and
    /// Burn answers that by copying the whole page to write one row of it.
    fn write_rows(&mut self, at: usize, src: Self);
}

/// The page-boundary arithmetic, over any [`PageRows`].
///
/// Invariants, checked by [`Pages::assert_sound`]:
/// * a page is never empty, and every page but the last holds a whole number
///   of [`PAGE`]s — except page 0, which [`Pages::drop_front`] may have cut;
/// * `head < PAGE`, and `head` counts rows already dropped from page 0;
/// * `len` is the LOGICAL row count, excluding `head`;
/// * `head + len` equals the total rows actually stored.
///
/// ## Why a page may be BIGGER than [`PAGE`]
///
/// Because the read is per page, and one launch per page is the cost that
/// grows. `Tensor::cat` on this backend is one `slice_assign` kernel per input
/// (`burn-backend`'s `cat_with_slice_assign`), so a 64-page cache paid 64
/// launches AND a full copy on every [`Pages::gather`] — that is, on every
/// layer of every decode step. Reading the pages directly removes the copy but
/// leaves the launch count, which at 8k of context is worse than the copy it
/// saved.
///
/// So [`Pages::append`] merges the settled pages into one whenever the count
/// passes [`MAX_PAGES`]. The merge is the same `cat` the read used to do, but
/// it happens once per `(MAX_PAGES - 1) * PAGE` appends instead of once per
/// append — at 8k of context that is a copy every 896 steps rather than every
/// step, and the read is then at most [`MAX_PAGES`] chunks whatever the context
/// length is. Nothing else in this file cared about a page's size: every
/// operation below walks ROWS, not page indices.
///
/// ## The last page is a CAPACITY, and `fill` is what is real in it
///
/// [`Pages::append`] used to `concat` the page being written with the row
/// arriving, which allocates a buffer one row bigger than any that has existed
/// before on EVERY decode step. That is shape drift at its narrowest, and it is
/// what stops a captured CUDA graph being replayed on a later step: measured at
/// 21 Inkling layers, 589 allocations escaped a capture and exactly one of them
/// was on the GPU -- this `concat`.
///
/// So the page being written is allocated at its full [`PAGE`] rows once and
/// written into, and `fill` counts the rows of it that are real. The shape of
/// every page is then constant for as long as the page lives. It is the same
/// layout `dev_lane::SlotCache` already uses for the slot batch -- `cap`
/// against `len`, a `slice_assign` of one row instead of a `cat` of everything
/// -- and this is that layout brought to the single-sequence cache.
///
/// The rows between `fill` and the page's capacity are DEAD, and every reader
/// already carries them, because a page has always been read whole: the
/// `PagedKv` mask marks every slot outside `head .. head + len` as `-inf`, and
/// the fused lane clamps each run to `hi = head + len`. That is why this
/// change is confined to this file.
///
/// ## RESERVED pages: one buffer, one address, for the life of the process
///
/// Everything above grows on demand, and growing is the one thing a captured
/// CUDA graph cannot survive: pushing a page allocates a buffer no graph node
/// points at, and releasing one frees a buffer several of them do. That is the
/// whole of the 128-step replay epoch -- a page holds [`PAGE`] rows, a decode
/// step appends one, and on the 128th the structure moves.
///
/// [`Pages::reserved`] is the other arm. The store is handed ONE page at
/// construction, sized for the longest context the run is admitted to, and
/// after that:
///
/// * `append` only ever writes in place into it -- it never pushes, and
///   [`Pages::append_is_in_place`] therefore keeps answering yes for the whole
///   reservation rather than for the rest of a 128-row page;
/// * `drop_front` only ever moves `head` -- it never releases a page and never
///   cuts page 0, so [`Pages::drop_is_bookkeeping_only`] keeps answering yes
///   for a windowed layer too;
/// * nothing is ever freed, merged or re-allocated, so every device address
///   this store hands out is the address it handed out on step one.
///
/// The `head < PAGE` bound is what the two predicates give up, and it is
/// affordable here for a reason that is specific to the reserved arm: the dead
/// prefix is bounded by [`Pages::compact`] instead, which copies the live rows
/// back to row 0 once `head` reaches an epoch. That copy is `len` rows once per
/// epoch against a page cut's `stored - head` rows once per [`PAGE`].
///
/// What a reservation does NOT buy on its own is an unbounded replay epoch,
/// and the honest reason is one layer up: the FP4 arm hands the attention
/// kernel DEQUANTIZED rows, and that buffer is allocated per step at a size
/// [`Pages::read_rows`] chooses. Fixing the pages fixes the addresses and the
/// allocation; the epoch is then that read window's granularity, which is a
/// tunable rather than a page size. See [`kv_epoch`].
///
/// ## Why a reservation is affordable at all, which is a fact about the KERNEL
///
/// A reserved page is mostly dead rows -- a global store holds 1,048,576 of
/// them and a decode step at 3732 tokens of context reads past 3732 -- and the
/// obvious objection is that the read then costs the reservation rather than
/// the context. It does not, and the reason is in [`super::flash`]: a
/// `KeyRun` carries `rows` (the buffer) beside `lo .. hi` (the live keys), the
/// launcher sizes its grid from `hi - lo`, and the kernel's own key loop runs
/// `s_lo .. s_hi` inside that range. `rows` reaches the kernel ONLY as the
/// binding length of an array argument. So a dead row costs nothing to score
/// and nothing to accumulate, and the whole of what the reservation adds is
/// the DEQUANT, which is why [`Pages::read_rows`] and not the reservation is
/// what the epoch bounds.
///
/// The same paragraph is why a replay stays correct when the launcher's split
/// count goes stale. `splits` is a grid dimension baked into a capture, but
/// each split's range is `per = ceil((khi - klo) / splits)` computed IN the
/// kernel from the patched bounds -- so a captured 30-way split still covers
/// the whole live range as that range grows, at slightly larger slices. Stale
/// there is suboptimal, never wrong.
///
/// The dense [`PageStore`] arm has neither property: [`PagedKv`] reads pages
/// whole and would carry every dead row into a score matrix, and cutting the
/// page down first is a `slice`, which allocates. That is why
/// [`super::burn::AttnCache::reserve_kv`] refuses anything but an FP4 cache.
///
/// [`PagedKv`]: super::burn
#[derive(Clone, Debug)]
pub struct Pages<R: PageRows> {
    pages: Vec<R>,
    head: usize,
    len: usize,
    /// Real rows in the LAST page. Every earlier page is full.
    fill: usize,
    /// Rows the single page was RESERVED at, or 0 on the grow-on-demand arm.
    ///
    /// Non-zero implies `pages.len() == 1` for the life of the store, and every
    /// branch below that would have changed the page structure is disabled.
    reserved: usize,
    /// How far [`Pages::read_rows`] rounds the read window up, in the reserved
    /// arm. Zero elsewhere.
    epoch: usize,
    /// Rows a reader takes from the reserved page. MONOTONE: it grows to the
    /// next epoch when the rows do and it never shrinks again.
    ///
    /// Never shrinking is what makes a windowed store settle. Its stored rows
    /// oscillate -- `head` walks out to an epoch and [`Pages::compact`] pulls it
    /// back -- so a window derived from them afresh each step would cross an
    /// epoch boundary TWICE a cycle, once up and once down, and each crossing
    /// is a step no replay can stand in for. Monotone, it crosses once ever:
    /// the store reaches `window + epoch` rows, and from then on the only thing
    /// that ends a replay run is the compaction itself.
    read: usize,
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
            fill: 0,
            reserved: 0,
            epoch: 0,
            read: 0,
        }
    }

    /// An empty set of pages backed by ONE page that is already allocated.
    ///
    /// `page` must hold `rows` rows and is the only buffer this store will ever
    /// have. `epoch` is the granularity [`Pages::read_rows`] rounds the read
    /// window up to, and also how far `head` is allowed to run before
    /// [`Pages::compact`] pulls the live rows back to row 0.
    pub fn reserved(page: R, rows: usize, epoch: usize) -> Self {
        assert!(rows > 0, "a reservation of no rows is not a reservation");
        assert_eq!(
            page.rows(),
            rows,
            "the reserved page holds {} rows, not the {rows} it was reserved at",
            page.rows()
        );
        assert!(epoch > 0 && epoch <= rows, "an epoch of {epoch} in {rows}");
        Self {
            pages: vec![page],
            head: 0,
            len: 0,
            fill: 0,
            reserved: rows,
            epoch,
            read: epoch.min(rows),
        }
    }

    /// Rows this store reserved, or `None` on the grow-on-demand arm.
    pub fn reservation(&self) -> Option<usize> {
        (self.reserved > 0).then_some(self.reserved)
    }

    /// Seat `rows` at the front of an EMPTY reservation.
    ///
    /// The one way rows enter a reserved store other than by appending, and it
    /// exists for exactly one caller: the prefill-to-decode handover, which
    /// moves already-packed rows rather than re-encoding them.
    pub fn write_reserved(&mut self, rows: R) {
        assert!(
            self.reserved > 0,
            "write_reserved on a grow-on-demand store"
        );
        assert_eq!(self.len, 0, "write_reserved into a store holding rows");
        let n = rows.rows();
        assert!(
            n <= self.reserved,
            "seating {n} rows in a {}-row reservation",
            self.reserved
        );
        if n == 0 {
            return;
        }
        self.pages[0].write_rows(0, rows);
        self.head = 0;
        self.fill = n;
        self.len = n;
        self.grow_read();
    }

    /// How many rows of the reserved page a reader should take, or `None` when
    /// the pages are handed over whole.
    ///
    /// `head + len` rounded up to [`Pages::reserved`]'s epoch. Rounding is what
    /// keeps the number STILL: a reader that took exactly `head + len` would
    /// change its shape every decode step, and a shape that moves every step is
    /// a fresh kernel compilation and a dead graph capture. Rounding up to an
    /// epoch changes it once per epoch instead, at the cost of carrying at most
    /// `epoch - 1` dead rows -- which the reader masks, exactly as it already
    /// masks the dead half of a capacity page.
    pub fn read_rows(&self) -> Option<usize> {
        (self.reserved > 0).then_some(self.read)
    }

    /// Pull the live rows back to row 0 of the reserved page.
    ///
    /// The reserved arm's answer to a dead prefix: `drop_front` never cuts, so
    /// a windowed layer's `head` would otherwise walk the whole reservation and
    /// take the read window with it. One copy of `len` rows resets it. The
    /// buffer does not move and no allocation happens -- the rows are read out
    /// of the page and written back into the front of the same page.
    ///
    /// Nothing about the CONTENT changes, so the caller's `base` (the absolute
    /// position of logical row 0) is untouched. What does change is every
    /// offset a captured region baked in, which is why the step that compacts
    /// is not a replayable one -- see [`Pages::append_is_in_place`].
    fn compact(&mut self) {
        debug_assert!(self.reserved > 0 && self.pages.len() == 1);
        if self.head == 0 {
            return;
        }
        if self.len > 0 {
            let live = self.pages[0].slice_rows(self.head, self.head + self.len);
            self.pages[0].write_rows(0, live);
        }
        self.fill = self.len;
        self.head = 0;
    }

    /// Real rows in page `i`: [`Pages::fill`] for the page being written, the
    /// whole page for every settled one.
    ///
    /// Every row walk below goes through this rather than through
    /// `pages[i].rows()`, because those two agree for every page but the one
    /// that matters.
    fn rows_at(&self, i: usize) -> usize {
        if i + 1 == self.pages.len() {
            self.fill
        } else {
            self.pages[i].rows()
        }
    }

    /// Logical rows currently held.
    pub fn len(&self) -> usize {
        self.len
    }

    /// WOULD an append of `n` rows change the page STRUCTURE?
    ///
    /// A replayed CUDA graph re-executes the append the capture recorded, into
    /// the address the capture recorded, at the row the capture baked in. That
    /// is the whole write on 127 of every 128 steps -- and on the 128th the
    /// eager path pushes a NEW page, which allocates a buffer no graph node
    /// points at and shifts every later row. A replay cannot do that, so the
    /// step that would do it has to run eagerly.
    ///
    /// Asked BEFORE the step, and pure, so the caller can choose the lane
    /// rather than discover mid-write that it chose wrong.
    pub fn append_is_in_place(&self, n: usize) -> bool {
        if self.reserved > 0 {
            // Two conditions, and the second one is the reserved arm's own.
            // The write stays inside the page for the whole reservation, which
            // is the point -- but a reader's row count is `stored` rounded up
            // to an epoch, and a captured region baked THAT in as a shape. So
            // the step that grows the read window is not replayable either,
            // even though nothing was allocated and nothing moved.
            return self.pages.len() == 1
                && self.fill + n <= self.reserved
                && self.read_rows() == Some(self.window_rows(self.stored() + n));
        }
        match self.pages.last().map(|p| p.rows()) {
            Some(cap) => self.fill + n <= cap && self.pages.len() <= MAX_PAGES,
            None => false,
        }
    }

    /// What [`Pages::read_rows`] would be if the store held `stored` rows.
    ///
    /// Monotone in the CURRENT window, which is the whole trick -- see the
    /// `read` field. Never zero either: an empty store still has to hand its
    /// reader a buffer with a shape, and a zero-row dequant is not a shape any
    /// kernel here is written for.
    fn window_rows(&self, stored: usize) -> usize {
        self.read
            .max(stored.next_multiple_of(self.epoch))
            .max(self.epoch)
            .min(self.reserved)
    }

    /// Let the read window catch up with the rows, after an append.
    ///
    /// A no-op on the grow-on-demand arm, and that is a guard rather than a
    /// nicety: `epoch` is zero there and `next_multiple_of(0)` divides by zero.
    fn grow_read(&mut self) {
        if self.reserved == 0 {
            return;
        }
        self.read = self.window_rows(self.stored());
    }

    /// The same question for the sliding window's advance.
    ///
    /// `drop_front` releases whole pages and cuts page 0 once `head` reaches
    /// [`PAGE`]; both move a POINTER a graph node holds. Neither happens while
    /// the head stays inside page 0 and below [`PAGE`], which is the ordinary
    /// step.
    pub fn drop_is_bookkeeping_only(&self, n: usize) -> bool {
        if n == 0 {
            return true;
        }
        if n > self.len || self.len == n {
            return false;
        }
        let head = self.head + n;
        if self.reserved > 0 {
            // Neither a release nor a cut is possible here -- there is one page
            // and it is never given up. What bounds the answer instead is
            // `compact`, which pulls the live rows back to row 0 once the dead
            // prefix reaches an epoch, and which is a device write rather than
            // bookkeeping.
            return head < self.epoch && head < self.rows_at(0);
        }
        head < PAGE && !self.pages.is_empty() && head < self.rows_at(0)
    }

    /// Record an append whose device write a REPLAY already performed.
    ///
    /// The bytes are in the page; only the bookkeeping is behind. This is the
    /// half of [`Pages::append`] that is not a device write, and it exists
    /// because a replayed step runs no host code at all inside the region --
    /// so without it `fill`, `len` and every scalar derived from them stop
    /// advancing and the NEXT eager step writes over the row this one wrote.
    ///
    /// Panics rather than silently doing something else if the append would
    /// not have been in place: the caller is supposed to have asked
    /// [`Pages::append_is_in_place`] first, and a wrong answer here is a wrong
    /// answer one step later with nothing to see.
    pub fn note_appended(&mut self, n: usize) {
        assert!(
            self.append_is_in_place(n),
            "note_appended({n}) on a store whose eager append would have changed its page \
             structure (fill {}, pages {}) -- a replay cannot have done that",
            self.fill,
            self.pages.len()
        );
        self.fill += n;
        self.len += n;
        // A no-op by construction -- `append_is_in_place` refused the step
        // where the window would move -- and here anyway, so the replayed
        // path and the eager one cannot drift apart in a way only a long run
        // would show.
        self.grow_read();
    }

    /// Record the window advance that goes with it.
    pub fn note_dropped(&mut self, n: usize) {
        assert!(
            self.drop_is_bookkeeping_only(n),
            "note_dropped({n}) would have released or cut a page (head {}, len {})",
            self.head,
            self.len
        );
        self.head += n;
        self.len -= n;
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

    /// Rows already dropped from the front of page 0.
    ///
    /// Part of the paged read's contract: the pages are handed over WHOLE, so
    /// the first `head` rows of chunk 0 are keys the sliding window has already
    /// forgotten and the reader must mask them. Slicing them off instead would
    /// cost a copy and — worse — make chunk 0's shape walk `1..=PAGE` as the
    /// window advances, which is a fresh kernel compilation per step.
    pub fn head(&self) -> usize {
        self.head
    }

    /// The pages themselves, whole and in order — the read that does NOT
    /// concatenate.
    ///
    /// Covers `head + len` rows: `head` dead ones at the front, then the
    /// retained context. Clones are handle clones, so this moves no bytes.
    /// [`Pages::gather`] is this followed by a `cat`; the point of having both
    /// is that the `cat` is what a decode step should not be paying.
    pub fn parts(&self) -> Vec<R> {
        self.pages.clone()
    }

    /// Append `rows`, filling the last partial page before starting a new one.
    ///
    /// Takes the whole batch at once rather than row by row: a speculative
    /// verify appends `k + 1` rows and splitting them into `k + 1` slice
    /// assignments would copy the tail page that many times.
    /// Three paths, and the FIRST one is the decode step.
    ///
    /// * Room in the page being written: a `write_rows` into it, in place, at
    ///   a shape that does not move. A decode step is this and nothing else,
    ///   127 times out of 128.
    /// * Whole pages: pushed as they are. A prefill is mostly these, and an
    ///   exact-size page needs no capacity and no copy.
    /// * A remainder: opens a fresh [`PAGE`]-row page and writes into it.
    pub fn append(&mut self, rows: R) {
        let n = rows.rows();
        if n == 0 {
            return;
        }
        if self.reserved > 0 {
            // The reserved arm has exactly one path: write in place. Reclaim
            // the dead prefix first if the tail has reached the end of the
            // reservation -- that is the only thing standing between a
            // windowed layer and a page it has walked off the end of.
            if self.fill + n > self.reserved {
                self.compact();
            }
            assert!(
                self.fill + n <= self.reserved,
                "an append of {n} rows past a KV reservation of {} (head {}, live {}) -- the \
                 sequence is longer than the context this run reserved for",
                self.reserved,
                self.head,
                self.len
            );
            let at = self.fill;
            self.pages[0].write_rows(at, rows);
            self.fill += n;
            self.len += n;
            self.grow_read();
            return;
        }
        let mut written = 0usize;
        // Room in the page being written?
        //
        // Asked of the LAST PAGE's capacity against `fill`, rather than of
        // `stored() % PAGE`: those agreed while every page was exactly `PAGE`
        // rows, and they stop agreeing the moment `drop_front` cuts page 0 or
        // `truncate` re-opens a merged one.
        if let Some(cap) = self.pages.last().map(|p| p.rows()) {
            if self.fill < cap {
                let take = (cap - self.fill).min(n);
                let last = self.pages.len() - 1;
                let at = self.fill;
                self.pages[last].write_rows(at, rows.slice_rows(0, take));
                self.fill += take;
                written = take;
            }
        }
        while n - written >= PAGE {
            self.pages.push(rows.slice_rows(written, written + PAGE));
            self.fill = PAGE;
            written += PAGE;
        }
        if written < n {
            let mut page = rows.zeroed(PAGE);
            page.write_rows(0, rows.slice_rows(written, n));
            self.pages.push(page);
            self.fill = n - written;
        }
        self.len += n;
        if self.pages.len() > MAX_PAGES {
            self.merge_settled();
        }
    }

    /// Join every page but the last into one, so the read stays a handful of
    /// chunks however long the context gets.
    ///
    /// Row content and row order are untouched — this is `cat` of adjacent runs
    /// — so `head` still counts from the same first row and every offset below
    /// still resolves by walking rows. The last page is left alone because it is
    /// the one [`Pages::append`] is still filling.
    fn merge_settled(&mut self) {
        if self.pages.len() < 3 {
            return;
        }
        let last = self.pages.pop().expect("at least three pages");
        let settled = R::concat(std::mem::take(&mut self.pages));
        self.pages = vec![settled, last];
    }

    /// Drop `n` rows from the FRONT — the sliding window advancing.
    ///
    /// Whole pages are released; the remainder becomes `head`. The partial page
    /// is NOT rewritten, so a window that advances one row a step does no work
    /// until it crosses a page boundary.
    pub fn drop_front(&mut self, n: usize) {
        assert!(n <= self.len, "dropping {n} of {} rows", self.len);
        self.len -= n;
        if self.reserved > 0 {
            // No release and no cut: the page stays, and the dead prefix is
            // bounded by a compaction rather than by cutting the buffer up.
            // Emptying is the same -- the reservation outlives the rows in it.
            self.head += n;
            if self.len == 0 || self.head >= self.epoch {
                self.compact();
            }
            return;
        }
        if self.len == 0 {
            self.pages.clear();
            self.head = 0;
            self.fill = 0;
            return;
        }
        // Release whole pages by REAL rows, not by dividing through `PAGE`: a
        // merged page is many pages' worth, page 0 may already have been cut,
        // and the page being written holds `fill` of its capacity.
        let mut head = self.head + n;
        let mut release = 0usize;
        while release < self.pages.len() && head >= self.rows_at(release) {
            head -= self.rows_at(release);
            release += 1;
        }
        if release > 0 {
            self.pages.drain(0..release);
        }
        self.head = head;
        // Keep `head < PAGE`, which bounds the dead rows the paged read carries
        // to less than one page whatever page 0's size is. Cutting page 0 costs
        // a copy of what survives it, and it happens at most once per `PAGE`
        // rows dropped.
        if self.head >= PAGE {
            let cut = self.head;
            let rows = self.pages[0].rows();
            self.pages[0] = self.pages[0].slice_rows(cut, rows);
            // If page 0 is also the page being written, the cut took `cut` real
            // rows off the front of it as well as `cut` of its capacity.
            if self.pages.len() == 1 {
                self.fill -= cut;
            }
            self.head = 0;
        }
    }

    /// Truncate to `keep` logical rows — a speculative batch being rejected.
    pub fn truncate(&mut self, keep: usize) {
        assert!(keep <= self.len, "keeping {keep} of {} rows", self.len);
        if keep == self.len {
            return;
        }
        if self.reserved > 0 {
            // The rejected rows stay in the page as dead ones, exactly as they
            // do on the grow-on-demand arm -- `fill` simply stops ahead of
            // them. There is no capacity to re-open, because the capacity is
            // the reservation and it does not change.
            self.len = keep;
            self.fill = self.head + keep;
            if keep == 0 {
                self.compact();
            }
            return;
        }
        self.len = keep;
        if self.len == 0 {
            self.pages.clear();
            self.head = 0;
            self.fill = 0;
            return;
        }
        // By REAL rows, for the same reason `drop_front` is: pages are not all
        // the same size once the settled ones have been merged.
        let stored = self.stored();
        let mut before = 0usize;
        let mut last = 0usize;
        while last < self.pages.len() && before + self.rows_at(last) < stored {
            before += self.rows_at(last);
            last += 1;
        }
        self.pages.truncate(last + 1);
        let tail = stored - before;
        // The rejected rows are not cut out of the page; `fill` simply stops
        // ahead of them and they become dead. They hold real keys rather than
        // zeros, which is sound for the same reason the zero rows are: a dead
        // row is masked to `-inf`, so it is multiplied by a probability of
        // exactly zero, and the only thing that turns that into a NaN is an
        // infinity, which a key is not.
        //
        // The capacity is re-opened to a whole number of `PAGE`s, because when
        // this page settles it must satisfy the invariant that every page but
        // the first and the last is one. That is also what keeps a MERGED page
        // from being carried as thousands of dead columns after a rollback into
        // it: `next_multiple_of` cuts it back to just past the kept rows.
        let cap = self.pages[last].rows();
        let want = tail.next_multiple_of(PAGE).max(PAGE).min(cap);
        if want < cap {
            self.pages[last] = self.pages[last].slice_rows(0, want);
        }
        self.fill = tail;
    }

    /// Remove `n` logical rows starting at `from` -- an EVICTION from the middle
    /// of the sequence, the rows a folded span of the moment held.
    ///
    /// Every row after the range moves down by `n`. Nothing about the surviving
    /// rows' content changes and the caller's `base` stays what it was: the
    /// positions the survivors stand at are the caller's to keep (see
    /// [`super::burn::AttnCache::evict`], which carries them as a gap table).
    /// On the reserved arm this is one copy of the tail back into the same
    /// page -- the buffer does not move, which is what a captured graph wants
    /// -- and on the grow-on-demand arm the pages the range touches are rebuilt
    /// from their kept rows and the pages inside it are dropped.
    pub fn remove(&mut self, from: usize, n: usize) {
        assert!(
            from + n <= self.len,
            "removing rows {from}..{} of {} rows",
            from + n,
            self.len
        );
        if n == 0 {
            return;
        }
        if from + n == self.len {
            self.truncate(from);
            return;
        }
        if self.reserved > 0 {
            let at = self.head + from;
            let tail = self.pages[0].slice_rows(at + n, self.head + self.len);
            self.pages[0].write_rows(at, tail);
            self.len -= n;
            self.fill = self.head + self.len;
            return;
        }
        let lo = self.head + from;
        let hi = lo + n;
        let mut kept: Vec<R> = Vec::with_capacity(self.pages.len());
        let mut start = 0usize;
        let mut first_dropped = false;
        let mut last_touched = false;
        let count = self.pages.len();
        for i in 0..count {
            let rows = self.rows_at(i);
            let end = start + rows;
            let page = &self.pages[i];
            if end <= lo || start >= hi {
                // By REAL rows: the last page carries capacity past `fill`.
                kept.push(page.slice_rows(0, rows));
            } else {
                let cut_lo = lo.max(start) - start;
                let cut_hi = hi.min(end) - start;
                let mut parts = Vec::with_capacity(2);
                if cut_lo > 0 {
                    parts.push(page.slice_rows(0, cut_lo));
                }
                if cut_hi < rows {
                    parts.push(page.slice_rows(cut_hi, rows));
                }
                if parts.is_empty() {
                    if i == 0 {
                        first_dropped = true;
                    }
                } else {
                    kept.push(R::concat(parts));
                }
                if i + 1 == count {
                    last_touched = true;
                }
            }
            start = end;
        }
        // One page again. A settled page must be a whole number of `PAGE`s,
        // and a cut page is not, so the survivors become one page that is
        // also the last page -- whose size is free -- and `append` opens a
        // fresh `PAGE` after it. An eviction is a rare, whole-store event; the
        // copy is the price of keeping every other step's invariants intact.
        let _ = last_touched;
        self.len -= n;
        self.pages = vec![R::concat(kept)];
        self.fill = self.pages[0].rows();
        if first_dropped {
            // The dead prefix went with its page.
            self.head = 0;
        }
        debug_assert_eq!(self.fill, self.head + self.len);
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
        // A reserved store cannot share: its one page is the buffer it goes on
        // writing into, so a second store holding a handle to it would watch
        // its "shared prefix" be overwritten by the next append.
        if self.reserved > 0 {
            return None;
        }
        if self.head != 0 || rows > self.len || rows % PAGE != 0 {
            return None;
        }
        // Walk rows: a merged page spans several page boundaries, so a prefix
        // that is a whole number of PAGEs may still land inside one. The pages
        // it covers WHOLE are still shared by handle, which is the promise; only
        // the one it splits is copied, and only that one.
        let mut pages = Vec::new();
        let mut left = rows;
        for (i, p) in self.pages.iter().enumerate() {
            if left == 0 {
                break;
            }
            // REAL rows: the page being written is a capacity, and sharing its
            // dead half would hand the other store rows that are not keys.
            let take = self.rows_at(i).min(left);
            pages.push(if take == p.rows() {
                p.clone()
            } else {
                p.slice_rows(0, take)
            });
            left -= take;
        }
        // Every page the share holds is exactly its real rows, so the share
        // starts with no capacity: its first append opens a fresh page rather
        // than writing into a buffer the parent may also be writing.
        let fill = pages.last().map(|p| p.rows()).unwrap_or(0);
        Some(Self {
            pages,
            head: 0,
            len: rows,
            fill,
            reserved: 0,
            epoch: 0,
            read: 0,
        })
    }

    /// Panics unless every documented invariant holds.
    pub fn assert_sound(&self, width: usize) {
        if self.reserved > 0 {
            assert_eq!(
                self.pages.len(),
                1,
                "a reserved store grew to {} pages",
                self.pages.len()
            );
            assert!(
                self.head < self.epoch,
                "head {} has reached the {}-row epoch without compacting",
                self.head,
                self.epoch
            );
            assert_eq!(
                self.pages[0].rows(),
                self.reserved,
                "the reserved page is {} rows against a {}-row reservation",
                self.pages[0].rows(),
                self.reserved
            );
            assert_eq!(
                self.fill,
                self.stored(),
                "fill {} against head {} + len {} -- the one page holds every row",
                self.fill,
                self.head,
                self.len
            );
            assert_eq!(
                self.pages[0].width(),
                width,
                "the reserved page is not {width} wide"
            );
            assert!(
                self.read >= self.stored() && self.read <= self.reserved,
                "a read window of {} against {} stored rows in a {}-row reservation",
                self.read,
                self.stored(),
                self.reserved
            );
            return;
        }
        assert!(self.head < PAGE, "head {} is a whole page", self.head);
        let stored = self.stored();
        assert!(
            self.pages.len() <= MAX_PAGES,
            "{} pages, over the {MAX_PAGES} the read is sized for",
            self.pages.len()
        );
        assert!(
            self.pages.is_empty() || self.fill <= self.pages[self.pages.len() - 1].rows(),
            "fill {} is past the page being written",
            self.fill
        );
        let mut total = 0usize;
        for (i, p) in self.pages.iter().enumerate() {
            let (n, w) = (self.rows_at(i), p.width());
            assert_eq!(w, width, "page {i} is {w} wide");
            assert!(n > 0, "page {i} is empty");
            // The appends fill before they grow, so only the page being filled
            // may be a partial one — plus page 0, which `drop_front` cuts to
            // keep the read's dead prefix under one page.
            if i + 1 != self.pages.len() && i != 0 {
                assert!(
                    n.is_multiple_of(PAGE),
                    "settled page {i} holds {n} rows, not a whole number of pages"
                );
            }
            total += n;
        }
        assert_eq!(
            total,
            stored,
            "{} pages hold {total} REAL rows against {stored} stored",
            self.pages.len()
        );
        assert!(
            self.pages.is_empty() || self.head < self.rows_at(0),
            "head {} is past page 0's {} real rows",
            self.head,
            self.rows_at(0)
        );
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

    fn zeroed(&self, rows: usize) -> Self {
        // Built from a row of THIS tensor rather than from `Tensor::zeros`, so
        // the page keeps the dtype the lane holds its cache at. A BF16 cache
        // that grew an f32 page would be a widening the narrow lane spent a
        // whole change removing, and it would not show up as a type error.
        let w = self.dims()[1];
        self.clone()
            .slice([0..1, 0..w])
            .zeros_like()
            .repeat_dim(0, rows)
    }

    fn write_rows(&mut self, at: usize, src: Self) {
        let [n, w] = src.dims();
        // Out of the field first: `self.clone().slice_assign(..)` leaves two
        // live references to the same buffer and Burn answers that by copying
        // the whole page to write one row of it -- which is the allocation this
        // whole layout exists to remove.
        //
        // The placeholder is a HANDLE CLONE of `src` and not `Tensor::empty`,
        // and that is not a style choice. `Tensor::empty` is an allocation, and
        // an allocation is precisely what a graph capture cannot contain: the
        // one GPU page that escaped a 21-layer capture came out of exactly this
        // call, through `ComputeClient::empty` -> `Command::reserve`, at a size
        // that was not even stable between runs (32768 bytes, then 16384 --
        // it is a POOL page, so its size says nothing about the tensor that
        // asked for it). Cloning a tensor that is already alive moves no bytes
        // and takes no page. It is overwritten on the next line.
        let dst = std::mem::replace(self, src.clone());
        *self = dst.slice_assign([at..at + n, 0..w], src);
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

    /// An empty store whose one page is allocated NOW, at `rows` rows, and
    /// never re-allocated. See [`Pages::reserved`].
    pub fn reserved(width: usize, rows: usize, epoch: usize, dev: &B::Device) -> Self {
        let page = Tensor::<B, 2>::zeros([rows, width], dev);
        Self {
            pages: Pages::reserved(page, rows, epoch),
            width,
        }
    }

    /// Rows this store reserved, or `None` on the grow-on-demand arm.
    pub fn reservation(&self) -> Option<usize> {
        self.pages.reservation()
    }

    /// Move this store's RETAINED rows into a fresh reservation of `rows`.
    /// See [`Fp4PageStore::into_reserved`].
    pub fn into_reserved(self, rows: usize, epoch: usize, dev: &B::Device) -> Self {
        let live = self.pages.len();
        assert!(
            live <= rows,
            "{live} retained rows do not fit a {rows}-row reservation"
        );
        let mut out = Self::reserved(self.width, rows, epoch, dev);
        if let Some(held) = self.pages.gather() {
            out.pages.write_reserved(held);
        }
        out
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

    /// Remove `n` logical rows starting at `from` — an eviction. See
    /// [`Pages::remove`].
    pub fn remove(&mut self, from: usize, n: usize) {
        self.pages.remove(from, n);
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

    /// Rows already dropped from the front of chunk 0 — see [`Pages::head`].
    pub fn head(&self) -> usize {
        self.pages.head()
    }

    /// See [`Pages::append_is_in_place`].
    pub fn append_is_in_place(&self, n: usize) -> bool {
        self.pages.append_is_in_place(n)
    }

    /// See [`Pages::drop_is_bookkeeping_only`].
    pub fn drop_is_bookkeeping_only(&self, n: usize) -> bool {
        self.pages.drop_is_bookkeeping_only(n)
    }

    /// See [`Pages::note_appended`].
    pub fn note_appended(&mut self, n: usize) {
        self.pages.note_appended(n)
    }

    /// See [`Pages::note_dropped`].
    pub fn note_dropped(&mut self, n: usize) {
        self.pages.note_dropped(n)
    }

    /// The pages whole and in order, covering `head() + len()` rows.
    ///
    /// The read that does not concatenate; see [`Pages::parts`].
    ///
    /// A reserved store cuts its one page down to [`Pages::read_rows`] first,
    /// because handing over a whole reservation would make every reader walk
    /// rows nothing has ever written. On the dense arm that cut is a COPY --
    /// Burn's `slice` allocates -- which is one more reason the reserved arm is
    /// scoped to the NVFP4 decode path, where the same cut is free.
    pub fn parts(&self) -> Vec<Tensor<B, 2>> {
        match self.pages.read_rows() {
            Some(rows) => self
                .pages
                .first()
                .map(|p| vec![p.slice_rows(0, rows)])
                .unwrap_or_default(),
            None => self.pages.parts(),
        }
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
/// was handed. **Always on; there is no switch.**
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
///
/// **There is no switch.** It was `INK_FP4_KV`, default off, on two
/// conditions: that `materialize`'s per-layer copy go, and that a retrieval
/// probe run. The copy is GONE -- the read is paged now, and the FP4 arm
/// dequantises a page at a time instead of writing tens of MB to DRAM and
/// reading them straight back. The other condition was a numerical one (NVFP4
/// perturbs 91% of dense RMS against BF16's 1%) and it was the wrong
/// criterion: the reference implementation ships the same `fp4_mx_block16` and
/// retrieves a needle EXACTLY from a 307,581-token prompt. Nobody wants an
/// unperturbed RMS; they want retrieval.
pub fn fp4_kv() -> bool {
    true
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
    /// whatever the lane says, so the arm moved not one of them.
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

    fn zeroed(&self, rows: usize) -> Self {
        // Zero codes and zero scales dequantize to zero: E2M1 code 0 is +0.0
        // and an E4M3 scale byte of 0 is 0.0, so a dead row reaches the
        // attention products as zeros whichever half of the pair is read. That
        // is exactly what `PagedKv`'s tail pad used to build with a `cat`.
        let cw = self.codes.dims()[1];
        let sw = self.scales.dims()[1];
        Self {
            codes: self
                .codes
                .clone()
                .slice([0..1, 0..cw])
                .zeros_like()
                .repeat_dim(0, rows),
            scales: self
                .scales
                .clone()
                .slice([0..1, 0..sw])
                .zeros_like()
                .repeat_dim(0, rows),
            width: self.width,
        }
    }

    fn write_rows(&mut self, at: usize, src: Self) {
        assert_eq!(
            src.width, self.width,
            "a {}-wide NVFP4 row into a {}-wide page",
            src.width, self.width
        );
        let n = src.codes.dims()[0];
        let cw = self.codes.dims()[1];
        let sw = self.scales.dims()[1];
        // Both halves, and both taken out of the struct first -- see the dense
        // implementation for why a clone would copy the page, and for why the
        // placeholder is a handle clone of the incoming rows rather than
        // `Tensor::empty`, which allocates.
        let codes = std::mem::replace(&mut self.codes, src.codes.clone());
        self.codes = codes.slice_assign([at..at + n, 0..cw], src.codes);
        let scales = std::mem::replace(&mut self.scales, src.scales.clone());
        self.scales = scales.slice_assign([at..at + n, 0..sw], src.scales);
    }
}

/// One run of NVFP4 rows as the device holds them: code words, block scales,
/// and how many of the buffer's rows are being read.
///
/// The two handles are what [`Fp4Rows`] wears as `Int` tensors, handed over
/// raw. Nothing here is a Burn tensor because nothing downstream wants one —
/// the only consumer is a kernel that indexes both buffers itself, and going
/// through a tensor would only reintroduce the dtype fiction the doc on
/// [`Fp4Rows`] explains.
#[derive(Clone, Debug)]
pub struct PackedRun {
    pub codes: cubecl::server::Handle,
    pub scales: cubecl::server::Handle,
    /// Rows read, which may be fewer than the buffers hold.
    pub rows: usize,
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

    /// An empty store whose one NVFP4 page is allocated NOW, at `rows` rows,
    /// and never re-allocated.
    ///
    /// Zeroed rather than uninitialised, for the reason [`PageRows::zeroed`]
    /// gives: the read hands over rows past `fill`, they reach the attention
    /// products, and a dead row has to be finite. Zero codes and zero scales
    /// dequantize to zero.
    ///
    /// The two buffers are `[rows, width / 8]` and `[rows, width / 64]` `I32`,
    /// which is 4.5 bits a value: 576 bytes for a 1024-wide row, against the
    /// 2048 a BF16 row of the same width would take.
    pub fn reserved(
        width: usize,
        dtype: DType,
        rows: usize,
        epoch: usize,
        dev: &burn::backend::cuda::CudaDevice,
    ) -> Self {
        assert!(
            width > 0 && width.is_multiple_of(FP4_ROW_ALIGN),
            "an NVFP4 KV row must be a positive multiple of {FP4_ROW_ALIGN}, got {width}"
        );
        assert!(
            matches!(dtype, DType::F32 | DType::BF16),
            "an NVFP4 KV store quantizes from f32 or bf16, not {dtype:?}"
        );
        let codes = Tensor::<Bk, 2, Int>::zeros([rows, width / 8], dev);
        let scales = Tensor::<Bk, 2, Int>::zeros([rows, width / 64], dev);
        // The client comes from a tensor that is already on the device, for the
        // reason `client` documents: two `CudaRuntime::client(&Default)` calls
        // are MEANT to return the same one.
        let client = seam::client_of(&Tensor::<Bk, 2>::zeros([1, 1], dev));
        let page = Fp4Rows {
            codes,
            scales,
            width,
        };
        Self {
            pages: Pages::reserved(page, rows, epoch),
            width,
            dtype,
            client: Some(client),
        }
    }

    /// Rows this store reserved, or `None` on the grow-on-demand arm.
    pub fn reservation(&self) -> Option<usize> {
        self.pages.reservation()
    }

    /// Move this store's RETAINED rows into a fresh reservation of `rows`.
    ///
    /// The seam between a prefill and a reserved decode. A prefill appends the
    /// whole prompt in one call and a windowed layer then throws most of it
    /// away, so reserving before the prefill would size a local layer's
    /// reservation by the PROMPT rather than by its window -- 3732 rows a store
    /// instead of 1024, on thirty-five of forty-two layers. Reserving after the
    /// trim sizes it by what is actually kept.
    ///
    /// The rows move PACKED. `gather` joins codes and scales as bytes and
    /// `write_rows` writes them as bytes, so a row that was quantized once at
    /// the prefill is still carrying exactly that one rounding. Nothing here
    /// dequantizes and re-quantizes, which would be a second rounding and would
    /// make the reserved arm numerically different from the arm it is supposed
    /// to be a relocation of.
    pub fn into_reserved(
        self,
        rows: usize,
        epoch: usize,
        dev: &burn::backend::cuda::CudaDevice,
    ) -> Self {
        let live = self.pages.len();
        assert!(
            live <= rows,
            "{live} retained rows do not fit a {rows}-row reservation"
        );
        let mut out = Self::reserved(self.width, self.dtype, rows, epoch, dev);
        if let Some(packed) = self.pages.gather() {
            out.pages.write_reserved(packed);
        }
        // Keep the client the rows were actually allocated on, if there was
        // one: `reserved` had to invent one from the device.
        if let Some(c) = self.client {
            out.client = Some(c);
        }
        out
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

    /// Remove `n` logical rows starting at `from` — an eviction. See
    /// [`Pages::remove`].
    pub fn remove(&mut self, from: usize, n: usize) {
        self.pages.remove(from, n);
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
        self.dequantize(all, dev)
    }

    /// Rows already dropped from the front of chunk 0 — see [`Pages::head`].
    pub fn head(&self) -> usize {
        self.pages.head()
    }

    /// See [`Pages::append_is_in_place`].
    pub fn append_is_in_place(&self, n: usize) -> bool {
        self.pages.append_is_in_place(n)
    }

    /// See [`Pages::drop_is_bookkeeping_only`].
    pub fn drop_is_bookkeeping_only(&self, n: usize) -> bool {
        self.pages.drop_is_bookkeeping_only(n)
    }

    /// See [`Pages::note_appended`].
    pub fn note_appended(&mut self, n: usize) {
        self.pages.note_appended(n)
    }

    /// See [`Pages::note_dropped`].
    pub fn note_dropped(&mut self, n: usize) {
        self.pages.note_dropped(n)
    }

    /// The pages whole and in order, each dequantized on its own.
    ///
    /// One dequant launch per page instead of one over the whole context, and
    /// the packed `cat` [`Fp4PageStore::materialize`] does first goes away
    /// entirely. It also changes what the dequant's OUTPUT costs: a page is
    /// `PAGE * width` values — 256 KB at Inkling's 1024-wide BF16 KV row —
    /// which the score product consumes immediately, where the whole-context
    /// form wrote tens of megabytes to DRAM and read them straight back. That
    /// is the read path this arm needs to be worth its rounding: as
    /// [`fp4_kv`] says, materializing turns 4.5 bits a value into a 16-bit
    /// round trip, i.e. MORE traffic than the BF16 cache it replaces.
    ///
    /// A RESERVED store hands over one chunk: the first [`Pages::read_rows`]
    /// rows of its one page. That cut costs nothing at all, and the reason is
    /// worth saying plainly -- the dequant kernel is told its row count as a
    /// SCALAR and indexes row-major from the base of the buffer, so reading a
    /// prefix is a smaller `rows` argument against the same handle. No slice,
    /// no copy, and the address it reads from is the address it read from on
    /// step one.
    pub fn parts(&self, dev: &burn::backend::cuda::CudaDevice) -> Vec<Tensor<Bk, 2>> {
        if let Some(rows) = self.pages.read_rows() {
            let page = self.pages.first().expect("a reserved store has its page");
            return vec![self.dequantize_rows(page.clone(), rows, dev)];
        }
        self.pages
            .parts()
            .into_iter()
            .map(|p| self.dequantize(p, dev))
            .collect()
    }

    /// The same retained context [`Fp4PageStore::parts`] returns, PACKED — the
    /// stored code words and their E4M3 block scales, with no dequant launch
    /// and no expanded page.
    ///
    /// This is the read for a consumer that dequantises in registers, and it
    /// is the whole of what such a consumer needs: the runs cover the same
    /// rows, in the same order, cut at the same page boundaries, so the reader
    /// swaps and nothing about the geometry moves. What it removes is the
    /// round trip — the dequant writes a BF16 page and the consumer reads it
    /// straight back, and at Inkling's 1024-wide row those two halves are
    /// 2 B a value each against the 0.5625 B a value this hands over instead.
    ///
    /// `rows` may be FEWER than the buffers hold, for [`Fp4PageStore::parts`]'s
    /// reason: a reserved store's read is a prefix of one page, and a prefix is
    /// a smaller count against the same handle rather than a slice.
    pub fn packed_parts(&self) -> Vec<PackedRun> {
        if let Some(rows) = self.pages.read_rows() {
            let page = self.pages.first().expect("a reserved store has its page");
            return vec![Self::packed_run(page.clone(), rows)];
        }
        self.pages
            .parts()
            .into_iter()
            .map(|p| {
                let n = p.rows();
                Self::packed_run(p, n)
            })
            .collect()
    }

    /// One [`Fp4Rows`] as raw handles, over a prefix of `n` rows.
    fn packed_run(rows: Fp4Rows, n: usize) -> PackedRun {
        assert!(
            n <= rows.rows(),
            "a {n}-row packed read of a {}-row page",
            rows.rows()
        );
        PackedRun {
            codes: seam::int_handle_of(rows.codes),
            scales: seam::int_handle_of(rows.scales),
            rows: n,
        }
    }

    /// One run of packed rows back to the dtype it was appended in.
    fn dequantize(&self, rows: Fp4Rows, dev: &burn::backend::cuda::CudaDevice) -> Tensor<Bk, 2> {
        let n = rows.rows();
        self.dequantize_rows(rows, n, dev)
    }

    /// [`Fp4PageStore::dequantize`] over a PREFIX of `rows`.
    ///
    /// `n` may be fewer rows than the buffers hold. That is not a slice: the
    /// codes and scales are row-major and the kernel takes its row count as a
    /// scalar, so a shorter count reads a shorter prefix of the same bytes.
    fn dequantize_rows(
        &self,
        rows: Fp4Rows,
        n: usize,
        dev: &burn::backend::cuda::CudaDevice,
    ) -> Tensor<Bk, 2> {
        assert!(
            n <= rows.rows(),
            "a {n}-row read of a {}-row page",
            rows.rows()
        );
        let client = self
            .client
            .clone()
            .expect("a non-empty NVFP4 store was filled, so it has a client");
        let codes = seam::int_handle_of(rows.codes);
        let scales = seam::int_handle_of(rows.scales);
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

    /// Rows this store reserved, or `None` on the grow-on-demand arm.
    pub fn reservation(&self) -> Option<usize> {
        match self {
            Self::Wide(s) => s.reservation(),
            Self::Fp4(s) => s.reservation(),
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

    /// Remove `n` logical rows starting at `from` — an eviction. See
    /// [`Pages::remove`].
    pub fn remove(&mut self, from: usize, n: usize) {
        match self {
            Self::Wide(s) => s.remove(from, n),
            Self::Fp4(s) => s.remove(from, n),
        }
    }

    /// See [`Pages::append_is_in_place`].
    pub fn append_is_in_place(&self, n: usize) -> bool {
        match self {
            Self::Wide(s) => s.append_is_in_place(n),
            Self::Fp4(s) => s.append_is_in_place(n),
        }
    }

    /// See [`Pages::drop_is_bookkeeping_only`].
    pub fn drop_is_bookkeeping_only(&self, n: usize) -> bool {
        match self {
            Self::Wide(s) => s.drop_is_bookkeeping_only(n),
            Self::Fp4(s) => s.drop_is_bookkeeping_only(n),
        }
    }

    /// See [`Pages::note_appended`].
    pub fn note_appended(&mut self, n: usize) {
        match self {
            Self::Wide(s) => s.note_appended(n),
            Self::Fp4(s) => s.note_appended(n),
        }
    }

    /// See [`Pages::note_dropped`].
    pub fn note_dropped(&mut self, n: usize) {
        match self {
            Self::Wide(s) => s.note_dropped(n),
            Self::Fp4(s) => s.note_dropped(n),
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

    /// An empty store whose pages are allocated NOW and never re-allocated.
    ///
    /// Same arm choice as [`KvStore::new`], because the reservation is about
    /// WHERE the rows live and not about what they are.
    pub fn reserved(
        width: usize,
        dtype: DType,
        rows: usize,
        epoch: usize,
        dev: &burn::backend::cuda::CudaDevice,
    ) -> Self {
        if fp4_kv_now() && width > 0 && width.is_multiple_of(FP4_ROW_ALIGN) {
            Self::Fp4(Fp4PageStore::reserved(width, dtype, rows, epoch, dev))
        } else {
            Self::Wide(PageStore::reserved(width, rows, epoch, dev))
        }
    }

    /// Move this store's RETAINED rows into a fresh reservation of `rows`.
    /// See [`Fp4PageStore::into_reserved`].
    pub fn into_reserved(
        self,
        rows: usize,
        epoch: usize,
        dev: &burn::backend::cuda::CudaDevice,
    ) -> Self {
        match self {
            Self::Wide(s) => Self::Wide(s.into_reserved(rows, epoch, dev)),
            Self::Fp4(s) => Self::Fp4(s.into_reserved(rows, epoch, dev)),
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

    /// The retained context as CHUNKS the attention products consume directly,
    /// in order, covering `head() + len()` rows.
    ///
    /// The same rows [`KvStore::materialize`] returns, minus the copy that
    /// joined them and plus `head()` dead rows at the front — the reader masks
    /// those, and [`Pages::head`] says why they are not sliced off here.
    /// Attention is a sum over key positions, so both products decompose:
    /// `q @ k^T` per chunk concatenated on the key axis, and `p @ v` per chunk
    /// summed. The softmax between them still sees the whole key axis, so this
    /// is not an approximation of the materialized read — it is the same
    /// arithmetic with the intermediate copy removed.
    pub fn parts(&self, dev: &burn::backend::cuda::CudaDevice) -> Vec<Tensor<Bk, 2>> {
        match self {
            Self::Wide(s) => s.parts(),
            Self::Fp4(s) => s.parts(dev),
        }
    }

    /// The same runs [`KvStore::parts`] covers, PACKED — or `None` on the dense
    /// arm, which has no packed form to hand over.
    ///
    /// `None` is the honest answer rather than a dense fallback: a caller
    /// asking for this is choosing a reader, and a store that quietly returned
    /// something the packed reader would misread is the failure this cannot
    /// have. See [`Fp4PageStore::packed_parts`].
    pub fn packed_parts(&self) -> Option<Vec<PackedRun>> {
        match self {
            Self::Wide(_) => None,
            Self::Fp4(s) => Some(s.packed_parts()),
        }
    }

    /// Rows already dropped from the front of chunk 0 — see [`Pages::head`].
    pub fn head(&self) -> usize {
        match self {
            Self::Wide(s) => s.head(),
            Self::Fp4(s) => s.head(),
        }
    }
}

// ---------------------------------------------------------------------------
// The reservation policy
// ---------------------------------------------------------------------------

/// Rows a windowed store keeps beyond its window, and how far the read window
/// is rounded up. **`INK_KV_EPOCH`, default 512 rows.**
///
/// This is the reserved arm's ONE knob, and it is the replay epoch measured in
/// decode steps. Two things happen once per epoch and not otherwise:
/// [`Pages::read_rows`] grows by one epoch, and a windowed store's
/// [`Pages::compact`] pulls its live rows back to row 0. Both change something
/// a captured region baked in, so both end a replay run -- and between them
/// nothing does, which is the whole point of reserving.
///
/// ## What a larger epoch costs, per store per layer per decode step
///
/// Dead rows in the read window, and on the NVFP4 arm a dead row is a row the
/// dequant kernel writes for nothing. At Inkling's 1024-wide KV row that is
/// `2 KiB` of BF16 output per dead row, and the window carries at most
/// `epoch - 1` of them.
///
/// * A GLOBAL store (7 of 42 layers) reads `fill` rounded up, so it carries a
///   half-epoch on average: at 512 that is ~0.5 MiB written per store per
///   layer-step, 7 MiB a step over both stores and all seven layers, against
///   the ~11.6 GB of weights a 42-layer decode step reads. Under 0.1%.
/// * A LOCAL store (35 of 42) reads `head + 512` rounded up, and `head` runs to
///   a full epoch before compacting -- so its window is up to `window + epoch`
///   rows against the 512 it needs. At 512 that is 1024 rows against 512, i.e.
///   ~1 MiB extra per store per layer-step and ~70 MiB a step over both stores
///   and all thirty-five layers. Around 0.6% of the same 11.6 GB.
///
/// So 512 buys a 4x longer replay epoch than the 128-row page it replaces for
/// well under 1% of the step's traffic, and 4096 would buy 32x for something
/// closer to 5%. It is a knob and not a constant because that trade is a
/// property of the deployment, not of this file.
pub fn kv_epoch() -> usize {
    static EPOCH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *EPOCH.get_or_init(|| {
        std::env::var("INK_KV_EPOCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| *n >= 1)
            .unwrap_or(512)
    })
}

/// The configured KV reservation, in TOKENS of context, or `None` when the
/// decode path grows its pages on demand. **`INK_KV_PREALLOC`, default off.**
///
/// `INK_KV_PREALLOC=<n>` reserves for `n` tokens; `INK_KV_PREALLOC=max` reserves
/// for the model's declared `model_max_length`, which reaches this module
/// through [`note_model_max_length`] because this file does not read a
/// checkpoint.
///
/// Off by default because it is a memory reservation with a real number on it
/// and nobody should discover it by upgrading. See [`KvPlan::report`] for what
/// that number is on this model.
pub fn kv_prealloc() -> Option<usize> {
    static WANT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = WANT.get_or_init(|| std::env::var("INK_KV_PREALLOC").ok());
    match raw.as_deref() {
        None | Some("") | Some("0") | Some("off") => None,
        Some("max") | Some("1") | Some("on") => Some(model_max_length()),
        Some(n) => match n.parse::<usize>() {
            Ok(0) => None,
            Ok(n) => Some(n),
            Err(_) => {
                panic!("INK_KV_PREALLOC={n:?} is not a token count, \"max\", \"on\" or \"off\"")
            }
        },
    }
}

/// Whether this run decodes a BATCH of sequences rather than one.
///
/// Read here and not passed in because the alternative is threading a flag
/// through the prefill for one guard; see [`KvPlan::from_env`] for why the
/// guard exists at all. `INK_SLOTS` is parsed exactly as the binary parses it,
/// so the two cannot disagree about what "a batch" means.
fn slot_lane() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INK_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            > 1
    })
}

static MAX_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Tell this module the checkpoint's declared `model_max_length`.
///
/// Called from the admission gate, which is the one place in the library that
/// holds an `InklingTextConfig` and runs once before the first token. It is
/// what makes `INK_KV_PREALLOC=max` mean anything.
pub fn note_model_max_length(n: usize) {
    MAX_LEN.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// What [`note_model_max_length`] was told, panicking if nothing was.
fn model_max_length() -> usize {
    match MAX_LEN.load(std::sync::atomic::Ordering::Relaxed) {
        0 => panic!(
            "INK_KV_PREALLOC=max wants the checkpoint's model_max_length and nothing has told \
             this process what it is -- pass an explicit token count instead"
        ),
        n => n,
    }
}

/// How many rows one KV store reserves, and how much that is in bytes.
///
/// One value rather than a scatter of arithmetic, because the store, the
/// admission gate and the startup report must not be able to disagree about it
/// -- the disagreement they would have is "the gate says 1.3 GiB and the run
/// takes 7.9", which is the failure this type exists to prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPlan {
    /// Tokens of context the run is reserving for.
    pub context: usize,
    /// Rows one GLOBAL store reserves: the whole context.
    pub global_rows: usize,
    /// Rows one LOCAL store reserves: its window plus one epoch, which is all a
    /// compacting store can ever hold. NOT the context -- multiplying the
    /// context by all 42 layers over-counts this model by 5.98x.
    pub local_rows: usize,
    /// See [`kv_epoch`].
    pub epoch: usize,
}

impl KvPlan {
    /// The plan for a `context`-token run of a model with this window.
    pub fn new(context: usize, window: usize) -> Self {
        let epoch = kv_epoch();
        Self {
            context,
            global_rows: context.next_multiple_of(epoch).max(epoch),
            local_rows: (window + epoch).next_multiple_of(epoch),
            epoch,
        }
    }

    /// The configured plan, or `None` when preallocation is off -- or when the
    /// run is on a lane this is not scoped to.
    ///
    /// `INK_SLOTS=b` is the one such lane. Its batch lives in
    /// [`super::burn::SlotCache`], which is built by MATERIALISING each
    /// prefilled [`super::burn::AttnCache`] and then dropping it -- so a
    /// reservation taken at the end of a prefill would be allocated, copied out
    /// of, and thrown away, once per slot per layer. That is not a correctness
    /// problem and it IS a memory-behaviour change to a lane this was not asked
    /// to touch, which is the same thing as a silent one. So the slot lane is
    /// refused here rather than left to discover it.
    pub fn from_env(window: usize) -> Option<Self> {
        if slot_lane() {
            return None;
        }
        kv_prealloc().map(|context| Self::new(context, window))
    }

    /// Rows a store reserves for a layer with this window.
    pub fn rows_for(&self, window: Option<usize>) -> usize {
        match window {
            Some(w) => self
                .local_rows
                .max((w + self.epoch).next_multiple_of(self.epoch)),
            None => self.global_rows,
        }
    }

    /// Bytes one NVFP4 row of `width` logical columns occupies.
    ///
    /// `width / 8` code words and `width / 64` scale words, both `u32`: 576
    /// bytes for a 1024-wide row. That is 4.5 bits a value, not 16 -- writing
    /// it as `width * 2` would over-charge this reservation by 3.56x.
    pub const fn row_bytes(width: usize) -> u64 {
        ((width / 8) + (width / 64)) as u64 * 4
    }

    /// Bytes this plan reserves, over `globals` global layers and `locals`
    /// local ones, counting BOTH stores (keys and values) of each.
    pub fn bytes(&self, globals: usize, locals: usize, width: usize) -> u64 {
        let per = Self::row_bytes(width);
        2 * (globals as u64 * self.global_rows as u64 + locals as u64 * self.local_rows as u64)
            * per
    }

    /// One line saying what was reserved, with its framing rule attached.
    ///
    /// The framing is the load-bearing part: the number is per RUN and per
    /// NODE, it is the reservation and not a peak, and it counts retained rows
    /// rather than context times layers. That last distinction is worth 5.98x
    /// on this model and is exactly the arithmetic error this reports against.
    pub fn report(&self, globals: usize, locals: usize, width: usize) -> String {
        const GIB: f64 = (1u64 << 30) as f64;
        let bytes = self.bytes(globals, locals, width);
        let rows = globals * self.global_rows + locals * self.local_rows;
        format!(
            "  KV preallocated: {:.3} GiB reserved once for this NODE's {} attention layers \
             ({globals} global x {} rows + {locals} local x {} rows = {rows} retained rows, K and \
             V, NVFP4 at {} bytes a {width}-wide row), sized for {} tokens of context, epoch {} \
             rows. Held for the life of the process: never freed, never moved.",
            bytes as f64 / GIB,
            globals + locals,
            self.global_rows,
            self.local_rows,
            Self::row_bytes(width),
            self.context,
            self.epoch,
        )
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
    fn remove_from_the_middle_keeps_order_across_pages() {
        let mut s = PageStore::<B>::new(W);
        s.append(rows(0, 3 * PAGE));
        // A span that starts mid-page and ends mid-page, crossing a boundary.
        let (from, n) = (PAGE / 2, PAGE);
        s.remove(from, n);
        s.assert_sound();
        let want: Vec<usize> = (0..from).chain(from + n..3 * PAGE).collect();
        assert_eq!(contents(&s), want);
        assert_eq!(s.len(), 3 * PAGE - n);
        // The store keeps working afterwards: appends land after the survivors.
        s.append(rows(3 * PAGE, 3));
        s.assert_sound();
        let want: Vec<usize> = (0..from).chain(from + n..3 * PAGE + 3).collect();
        assert_eq!(contents(&s), want);
        // Removing a whole page in the middle, and a range that ends at the end.
        s.remove(PAGE, PAGE);
        s.assert_sound();
        assert_eq!(s.len(), 3 * PAGE + 3 - n - PAGE);
        let tail = s.len() - 2;
        s.remove(tail, 2);
        s.assert_sound();
        assert_eq!(s.len(), 3 * PAGE + 1 - n - PAGE);
    }

    #[test]
    fn remove_on_the_reserved_arm_moves_the_tail_in_place() {
        let dev = Default::default();
        let mut s = PageStore::<B>::reserved(W, 64, 16, &dev);
        s.append(rows(0, 40));
        s.remove(10, 5);
        s.assert_sound();
        let want: Vec<usize> = (0..10).chain(15..40).collect();
        assert_eq!(contents(&s), want);
        assert_eq!(s.len(), 35);
        s.append(rows(40, 3));
        s.assert_sound();
        let want: Vec<usize> = (0..10).chain(15..43).collect();
        assert_eq!(contents(&s), want);
        // Two evictions compose.
        s.remove(0, 10);
        s.assert_sound();
        let want: Vec<usize> = (15..43).collect();
        assert_eq!(contents(&s), want);
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

    /// The rows a PAGED read hands over, as indices, and where the live ones
    /// start.
    ///
    /// Deliberately not routed through `materialize`: this is the read the
    /// attention actually does, and if it agreed with `contents` only because
    /// both went through the same `gather` the agreement would be worth
    /// nothing.
    fn parts_contents(s: &PageStore<B>) -> (usize, Vec<usize>) {
        let flat: Vec<f32> = Tensor::cat(s.parts(), 0).into_data().to_vec().unwrap();
        let rows = flat
            .chunks(W)
            .map(|c| {
                assert!(c.iter().all(|x| *x == c[0]), "a row was torn: {c:?}");
                c[0] as usize
            })
            .collect();
        (s.head(), rows)
    }

    /// A cache long enough to merge, read the paged way, is the materialized
    /// read exactly — same rows, same order, once the dead prefix is dropped.
    ///
    /// The failure this is aimed at is not "the numbers moved": it is a page
    /// dropped, duplicated or swapped with its neighbour, which a shape check
    /// cannot see and a tolerance would hide. So every row carries its own
    /// index and the comparison is `==`.
    #[test]
    fn the_paged_read_is_the_materialized_read() {
        let mut s = PageStore::<B>::new(W);
        // Past MAX_PAGES twice over, in batches that do not divide PAGE, so
        // merges happen with a partial tail page outstanding.
        let mut at = 0usize;
        while at < 3 * MAX_PAGES * PAGE {
            let n = 37.min(3 * MAX_PAGES * PAGE - at);
            s.append(rows(at, n));
            at += n;
        }
        // Crossing PAGE cuts page 0 and takes the head back to zero...
        s.drop_front(PAGE + 5);
        s.assert_sound();
        assert_eq!(s.head(), 0, "a drop past PAGE should have cut page 0");
        // ...and the next drop leaves a head that is neither zero nor a page
        // boundary, which is the case the paged read carries as dead columns
        // instead of slicing off.
        s.drop_front(7);
        s.assert_sound();
        assert_eq!(s.head(), 7);

        let live = PAGE + 12;
        let (head, paged) = parts_contents(&s);
        assert_eq!(
            &paged[head..head + s.len()],
            contents(&s).as_slice(),
            "the paged read and the materialized read disagree"
        );
        assert_eq!(
            &paged[head..head + s.len()],
            (live..at).collect::<Vec<_>>().as_slice(),
            "and neither of them is the sequence that went in"
        );
    }

    /// Merging the settled pages is a `cat` of adjacent runs and nothing else:
    /// the rows, their order, and both trims all survive it.
    #[test]
    fn merging_settled_pages_changes_nothing_a_reader_can_see() {
        let mut s = PageStore::<B>::new(W);
        for i in 0..(MAX_PAGES + 4) {
            s.append(rows(i * PAGE, PAGE));
            s.assert_sound();
        }
        let total = (MAX_PAGES + 4) * PAGE;
        assert!(
            s.parts().len() <= MAX_PAGES,
            "{} chunks after {} pages' worth",
            s.parts().len(),
            MAX_PAGES + 4
        );
        assert_eq!(contents(&s), (0..total).collect::<Vec<_>>());

        // Both ends, on a store whose page 0 is now many pages wide. The front
        // drop crosses PAGE, which is what cuts page 0 rather than releasing it.
        s.drop_front(2 * PAGE + 9);
        s.assert_sound();
        assert!(s.head() < PAGE, "head {} is a whole page", s.head());
        s.truncate(s.len() - 11);
        s.assert_sound();
        assert_eq!(
            contents(&s),
            (2 * PAGE + 9..total - 11).collect::<Vec<_>>(),
            "a merged store lost or reordered rows under a trim"
        );

        // And it is still a store: the next token appends onto the kept rows.
        s.append(rows(total - 11, 3));
        s.assert_sound();
        assert_eq!(
            contents(&s).last().copied(),
            Some(total - 9),
            "an append after a merge and a trim did not land at the end"
        );
    }

    /// A prefix that is a whole number of PAGEs but lands INSIDE a merged page
    /// is still shared, and still does not follow the parent.
    #[test]
    fn a_shared_prefix_survives_a_merge() {
        let mut s = PageStore::<B>::new(W);
        for i in 0..(MAX_PAGES + 2) {
            s.append(rows(i * PAGE, PAGE));
        }
        let shared = s.share_prefix(3 * PAGE).expect("page-aligned prefix");
        shared.assert_sound();
        assert_eq!(contents(&shared), (0..3 * PAGE).collect::<Vec<_>>());
        s.append(rows((MAX_PAGES + 2) * PAGE, 5));
        assert_eq!(shared.len(), 3 * PAGE);
        assert_eq!(contents(&shared), (0..3 * PAGE).collect::<Vec<_>>());
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

    /// Per-page dequantization is BIT-identical to dequantizing the joined
    /// context, and still in order, on a store long enough to have merged.
    ///
    /// Exact, with no tolerance: NVFP4 quantizes per row and per 16 features,
    /// so how the rows are GROUPED when they are decoded cannot change one
    /// output element. If it ever does, the grouping is reaching across a block
    /// boundary, which is the bug worth failing for — and a tolerance sized for
    /// four bits would swallow it whole.
    #[test]
    fn fp4_paged_read_is_the_materialized_read_bit_for_bit() {
        let mut s = Fp4PageStore::new(FW, DType::F32);
        let mut at = 0usize;
        while at < 2 * MAX_PAGES * PAGE {
            let n = 37.min(2 * MAX_PAGES * PAGE - at);
            s.append(frows(at, n));
            at += n;
        }
        s.drop_front(PAGE + 5);
        s.drop_front(7);
        s.assert_sound();
        let head = s.head();
        assert_eq!(head, 7, "the head the read has to mask");

        let want: Vec<f32> = s.materialize(&fp4_dev()).into_data().to_vec().unwrap();
        let paged: Vec<f32> = Tensor::cat(s.parts(&fp4_dev()), 0)
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(paged.len(), (head + s.len()) * FW);
        assert_eq!(
            &paged[head * FW..],
            want.as_slice(),
            "per-page dequantization disagrees with the whole-context one"
        );
        assert_eq!(fcontents(&s), (PAGE + 12..at).collect::<Vec<_>>());
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

    // -----------------------------------------------------------------------
    // The RESERVED arm
    //
    // What these check is not "the reservation works" -- it is the one claim
    // the reservation is FOR: that the page structure stops moving. Every
    // assertion below is either "this predicate keeps answering yes" or "these
    // are the same rows the grow-on-demand arm holds", because those are the
    // two ways preallocation can be wrong and the ways it can be wrong
    // silently.
    // -----------------------------------------------------------------------

    /// How many decode steps a store refuses to replay across a run.
    ///
    /// The number the whole change is about. On the grow-on-demand arm it is
    /// once per [`PAGE`] rows, because that is when a page is pushed; on the
    /// reserved arm it is once per epoch, because nothing is pushed and the
    /// only thing left that moves is the read window's row count.
    fn refusals(reserved: Option<(usize, usize)>, steps: usize, window: Option<usize>) -> usize {
        let mut s = match reserved {
            Some((rows, epoch)) => PageStore::<B>::reserved(W, rows, epoch, &Default::default()),
            None => PageStore::<B>::new(W),
        };
        let mut refused = 0usize;
        for i in 0..steps {
            let drop = match window {
                Some(w) if s.len() + 1 > w => s.len() + 1 - w,
                _ => 0,
            };
            if !s.append_is_in_place(1) || !s.drop_is_bookkeeping_only(drop) {
                refused += 1;
            }
            s.append(rows(i, 1));
            if drop > 0 {
                s.drop_front(drop);
            }
            s.assert_sound();
        }
        refused
    }

    #[test]
    fn a_reservation_moves_the_replay_boundary_from_a_page_to_an_epoch() {
        const STEPS: usize = 2048;
        const EPOCH: usize = 512;
        // Grow-on-demand: a page is pushed every PAGE rows and every push is a
        // buffer no graph node points at.
        // ...plus the very first step, which has no page at all to write into.
        let demand = refusals(None, STEPS, None);
        assert_eq!(
            demand,
            STEPS / PAGE,
            "the on-demand arm should refuse once per {PAGE}-row page"
        );
        // Reserved: nothing is ever pushed -- not even on step one, because the
        // page is already there -- so the only refusal left is the step per
        // epoch on which the READ window grows a row count.
        let held = refusals(Some((STEPS + EPOCH, EPOCH)), STEPS, None);
        assert_eq!(
            held,
            STEPS / EPOCH - 1,
            "the reserved arm should refuse once per {EPOCH}-row epoch and never at the head"
        );
        assert!(
            demand >= 5 * held,
            "an epoch of {EPOCH} against a page of {PAGE} should be a 4x longer replay run \
             at worst, got {demand} refusals against {held}"
        );
    }

    #[test]
    fn a_reserved_window_compacts_rather_than_walking_off_its_page() {
        // A local layer: window 512, one row a step, for many epochs. The
        // reservation is `window + epoch` and nothing more -- which is the
        // arithmetic that keeps 35 of 42 layers from being charged the whole
        // context.
        const WINDOW: usize = 64;
        const EPOCH: usize = 32;
        let mut s = PageStore::<B>::reserved(W, WINDOW + EPOCH, EPOCH, &Default::default());
        s.append(rows(0, WINDOW));
        s.assert_sound();
        for i in 0..8 * EPOCH {
            s.append(rows(WINDOW + i, 1));
            s.drop_front(1);
            s.assert_sound();
            assert_eq!(s.len(), WINDOW, "the window changed size at step {i}");
            assert_eq!(
                s.reservation(),
                Some(WINDOW + EPOCH),
                "the reservation moved at step {i}"
            );
        }
        // The rows are the last WINDOW appended, in order, after eight
        // compactions -- which is the thing a ring buffer gets wrong.
        let n = WINDOW + 8 * EPOCH;
        let want: Vec<usize> = (n - WINDOW..n).collect();
        assert_eq!(contents(&s), want);
    }

    #[test]
    fn the_reserved_arm_holds_exactly_the_rows_the_on_demand_arm_does() {
        // Both ends and a rollback, on both arms, from the same script. A
        // reservation is supposed to be a RELOCATION of the cache and nothing
        // else, so any disagreement here is the change being wrong.
        let script = |s: &mut PageStore<B>| {
            s.append(rows(0, 3 * PAGE + 7));
            s.drop_front(PAGE + 3);
            s.append(rows(3 * PAGE + 7, 40));
            s.truncate(s.len() - 5);
            s.append(rows(9000, 2));
            s.drop_front(11);
        };
        let mut demand = PageStore::<B>::new(W);
        script(&mut demand);
        demand.assert_sound();
        let mut held = PageStore::<B>::reserved(W, 8 * PAGE, PAGE, &Default::default());
        script(&mut held);
        held.assert_sound();
        assert_eq!(held.len(), demand.len());
        assert_eq!(contents(&held), contents(&demand));
        assert_eq!(held.reservation(), Some(8 * PAGE));
        assert_eq!(demand.reservation(), None);
    }

    #[test]
    fn the_prefill_handover_moves_packed_rows_and_does_not_round_twice() {
        // The seam `AttnCache::reserve_kv` uses. It must be a byte move: a
        // dequantize-and-requantize would be a SECOND rounding, and the
        // reserved arm would then differ numerically from the arm it is
        // supposed to be a relocation of. Exact equality is the whole test.
        let mut s = Fp4PageStore::new(FW, DType::F32);
        s.append(frows(0, 2 * PAGE + 17));
        s.drop_front(5);
        let before: Vec<f32> = s.materialize(&fp4_dev()).into_data().to_vec().unwrap();
        let held = s.into_reserved(4 * PAGE, PAGE, &fp4_dev());
        held.assert_sound();
        assert_eq!(held.reservation(), Some(4 * PAGE));
        let after: Vec<f32> = held.materialize(&fp4_dev()).into_data().to_vec().unwrap();
        assert_eq!(before, after, "the handover re-rounded the rows");
        let want: Vec<usize> = (5..2 * PAGE + 17).collect();
        assert_eq!(fcontents(&held), want);
    }

    #[test]
    fn a_reserved_fp4_read_is_one_chunk_at_a_still_row_count() {
        // The property the graph capture actually depends on: one run, at an
        // address and a row count that do not move between steps.
        let mut s = Fp4PageStore::new(FW, DType::F32);
        s.append(frows(0, 300));
        let mut s = s.into_reserved(2048, 512, &fp4_dev());
        let shape = |s: &Fp4PageStore| {
            let p = s.parts(&fp4_dev());
            assert_eq!(p.len(), 1, "a reserved store reads as one chunk");
            p[0].dims()[0]
        };
        let first = shape(&s);
        assert_eq!(first, 512, "300 rows read at the 512-row epoch");
        for i in 0..100 {
            s.append(frows(300 + i, 1));
            assert_eq!(shape(&s), first, "the read row count moved at step {i}");
        }
        // ...and it moves exactly once, when the epoch is crossed.
        for i in 0..120 {
            s.append(frows(400 + i, 1));
        }
        assert_eq!(shape(&s), 1024, "521 rows should read at two epochs");
    }

    #[test]
    fn a_plan_charges_retained_rows_and_not_context_times_layers() {
        // The 5.98x this exists to avoid. Inkling: 42 layers, 7 of them global,
        // a 512-token window, a 1024-wide KV row.
        const CTX: usize = 1 << 20;
        const WIDTH: usize = 1024;
        let plan = KvPlan::new(CTX, 512);
        assert_eq!(plan.global_rows, CTX, "a global store reserves the context");
        assert!(
            plan.local_rows <= 512 + plan.epoch,
            "a local store reserves its window and one epoch, got {}",
            plan.local_rows
        );
        // 4.5 bits a value, not 16.
        assert_eq!(KvPlan::row_bytes(WIDTH), 576);
        let held = plan.bytes(7, 35, WIDTH);
        // What the naive arithmetic would have charged.
        let naive = 2u64 * 42 * CTX as u64 * KvPlan::row_bytes(WIDTH);
        let ratio = naive as f64 / held as f64;
        assert!(
            (5.5..6.5).contains(&ratio),
            "42 x context over-counts the retained rows by {ratio:.2}x, want ~5.98"
        );
        // The headline number, on a 119 GiB part.
        let gib = held as f64 / (1u64 << 30) as f64;
        assert!(
            (7.5..8.5).contains(&gib),
            "a 1M-token NVFP4 KV pool for 42 layers is {gib:.3} GiB, want ~7.9"
        );
    }

    // -----------------------------------------------------------------------
    // The paging core, on the HOST
    //
    // Everything above needs a GPU, which means the page arithmetic can only be
    // checked on a machine that has one -- and the page arithmetic is the half
    // of this file that has nothing to do with a device. `HostRows` is
    // `PageRows` over a `Vec<usize>` of row labels, so the boundary cases run
    // in a second on any machine and a mistake in `drop_front` stops being
    // something you find out about after a forty-minute queue for a box.
    // -----------------------------------------------------------------------

    /// Rows that are just their own labels. Fixed-width, cuttable, rejoinable.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct HostRows {
        rows: Vec<usize>,
        width: usize,
    }

    impl HostRows {
        fn of(from: usize, n: usize, width: usize) -> Self {
            Self {
                rows: (from..from + n).collect(),
                width,
            }
        }
    }

    impl PageRows for HostRows {
        fn rows(&self) -> usize {
            self.rows.len()
        }
        fn width(&self) -> usize {
            self.width
        }
        fn slice_rows(&self, from: usize, to: usize) -> Self {
            Self {
                rows: self.rows[from..to].to_vec(),
                width: self.width,
            }
        }
        fn concat(parts: Vec<Self>) -> Self {
            let width = parts[0].width;
            Self {
                rows: parts.iter().flat_map(|p| p.rows.clone()).collect(),
                width,
            }
        }
        fn zeroed(&self, rows: usize) -> Self {
            Self {
                rows: vec![usize::MAX; rows],
                width: self.width,
            }
        }
        fn write_rows(&mut self, at: usize, src: Self) {
            self.rows[at..at + src.rows.len()].clone_from_slice(&src.rows);
        }
    }

    const HW: usize = 3;

    fn host_reserved(rows: usize, epoch: usize) -> Pages<HostRows> {
        Pages::reserved(
            HostRows {
                rows: vec![usize::MAX; rows],
                width: HW,
            },
            rows,
            epoch,
        )
    }

    fn live(p: &Pages<HostRows>) -> Vec<usize> {
        p.gather().map(|r| r.rows).unwrap_or_default()
    }

    /// The two arms, driven by the same script, must hold the same rows.
    ///
    /// A reservation is a RELOCATION of the cache and nothing else. Any
    /// disagreement here is the change being wrong, and this is the cheapest
    /// place in the repo to find that out.
    fn both_arms(script: impl Fn(&mut Pages<HostRows>), rows: usize, epoch: usize) {
        let mut demand = Pages::<HostRows>::new();
        script(&mut demand);
        demand.assert_sound(HW);
        let mut held = host_reserved(rows, epoch);
        script(&mut held);
        held.assert_sound(HW);
        assert_eq!(
            held.len(),
            demand.len(),
            "the two arms hold different counts"
        );
        assert_eq!(
            live(&held),
            live(&demand),
            "the two arms hold different rows"
        );
    }

    #[test]
    fn host_reserved_and_on_demand_agree_over_both_ends_and_a_rollback() {
        both_arms(
            |p| {
                p.append(HostRows::of(0, 3 * PAGE + 7, HW));
                p.drop_front(PAGE + 3);
                p.append(HostRows::of(3 * PAGE + 7, 40, HW));
                p.truncate(p.len() - 5);
                p.append(HostRows::of(9000, 2, HW));
                p.drop_front(11);
            },
            8 * PAGE,
            PAGE,
        );
    }

    #[test]
    fn host_reserved_and_on_demand_agree_over_a_long_sliding_window() {
        both_arms(
            |p| {
                p.append(HostRows::of(0, 64, HW));
                for i in 0..600usize {
                    p.append(HostRows::of(64 + i, 1, HW));
                    if p.len() > 64 {
                        p.drop_front(p.len() - 64);
                    }
                }
            },
            64 + 32,
            32,
        );
    }

    #[test]
    fn host_reserved_and_on_demand_agree_when_a_batch_is_rejected_whole() {
        both_arms(
            |p| {
                p.append(HostRows::of(0, 200, HW));
                p.append(HostRows::of(200, 5, HW));
                p.truncate(200);
                p.append(HostRows::of(300, 5, HW));
                p.truncate(0);
                p.append(HostRows::of(400, 9, HW));
            },
            4 * PAGE,
            PAGE,
        );
    }

    #[test]
    fn host_a_reservation_moves_the_replay_boundary_from_a_page_to_an_epoch() {
        const STEPS: usize = 2048;
        const EPOCH: usize = 512;
        let count = |mut p: Pages<HostRows>| {
            let mut refused = 0usize;
            for i in 0..STEPS {
                if !p.append_is_in_place(1) {
                    refused += 1;
                }
                p.append(HostRows::of(i, 1, HW));
                p.assert_sound(HW);
            }
            refused
        };
        let demand = count(Pages::<HostRows>::new());
        assert_eq!(demand, STEPS / PAGE);
        let held = count(host_reserved(STEPS + EPOCH, EPOCH));
        assert_eq!(held, STEPS / EPOCH - 1);
        assert!(demand >= 5 * held, "{demand} against {held}");
    }

    #[test]
    fn host_a_reserved_window_never_refuses_more_than_once_an_epoch() {
        // The claim `step_is_replayable` is made of, on the layer kind that
        // used to bound it: a windowed store, where the on-demand arm refuses
        // because `head` reaches a page and the reserved arm refuses only when
        // it reaches an epoch.
        const WINDOW: usize = 512;
        const EPOCH: usize = 512;
        const STEPS: usize = 4096;
        let count = |mut p: Pages<HostRows>| {
            p.append(HostRows::of(0, WINDOW, HW));
            let mut refused = 0usize;
            for i in 0..STEPS {
                let drop = (p.len() + 1).saturating_sub(WINDOW);
                if !p.append_is_in_place(1) || !p.drop_is_bookkeeping_only(drop) {
                    refused += 1;
                }
                p.append(HostRows::of(WINDOW + i, 1, HW));
                if drop > 0 {
                    p.drop_front(drop);
                }
                p.assert_sound(HW);
            }
            refused
        };
        let demand = count(Pages::<HostRows>::new());
        let held = count(host_reserved(WINDOW + EPOCH, EPOCH));
        // One per epoch -- the compaction -- plus exactly one at the head,
        // where the monotone read window settles from `window` to
        // `window + epoch` and never moves again.
        assert_eq!(
            held,
            STEPS / EPOCH + 1,
            "a reserved window should refuse once an epoch plus once at the head"
        );
        assert!(
            demand >= 4 * held,
            "the on-demand window refused {demand} times and the reserved one {held}"
        );
    }

    #[test]
    fn host_a_replayed_reserved_run_keeps_the_same_bookkeeping_as_an_eager_one() {
        // The path the graph lane actually drives. A replayed step runs no host
        // code inside the region, so `note_appended` / `note_dropped` are the
        // ONLY things that move the counters -- and if they drift from what the
        // eager path would have done, the next eager step writes over the row
        // this one wrote and nothing anywhere errors.
        const WINDOW: usize = 512;
        const EPOCH: usize = 256;
        const STEPS: usize = 2000;
        let mut eager = host_reserved(WINDOW + EPOCH, EPOCH);
        let mut lane = host_reserved(WINDOW + EPOCH, EPOCH);
        eager.append(HostRows::of(0, WINDOW, HW));
        lane.append(HostRows::of(0, WINDOW, HW));
        let mut replayed = 0usize;
        for i in 0..STEPS {
            let row = HostRows::of(WINDOW + i, 1, HW);
            let drop = (eager.len() + 1).saturating_sub(WINDOW);
            eager.append(row.clone());
            if drop > 0 {
                eager.drop_front(drop);
            }
            // The lane asks FIRST and pure, exactly as `step_is_replayable`
            // does, and only then chooses which half of the step to run.
            let can = lane.append_is_in_place(1) && lane.drop_is_bookkeeping_only(drop);
            if can {
                // The device write is pretended -- what is being checked is the
                // counters, which is all a replay leaves to the host.
                lane.pages[0].write_rows(lane.fill, row);
                lane.note_appended(1);
                if drop > 0 {
                    lane.note_dropped(drop);
                }
                replayed += 1;
            } else {
                lane.append(row);
                if drop > 0 {
                    lane.drop_front(drop);
                }
            }
            lane.assert_sound(HW);
            assert_eq!(
                (lane.len(), lane.head(), lane.read_rows()),
                (eager.len(), eager.head(), eager.read_rows()),
                "the replayed bookkeeping drifted at step {i}"
            );
        }
        assert_eq!(
            live(&lane),
            live(&eager),
            "the two runs hold different rows"
        );
        assert!(
            replayed > STEPS - 2 * (STEPS / EPOCH + 2),
            "only {replayed} of {STEPS} steps replayed"
        );
    }
}
