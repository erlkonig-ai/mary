//! Turning [`super::tp`]'s index ranges into BYTES, and being honest about
//! which of them still alias the pile and which have to be copied.
//!
//! [`super::tp`] says which rows a rank owns. That is not yet enough to bind
//! anything, because the binder takes a byte slice and the answer to "which
//! bytes" depends on which AXIS was cut:
//!
//! * A cut on the OUTPUT axis of a `[out, in]` weight is a run of whole rows,
//!   so it is a contiguous span of the mapping and [`rows`] returns a
//!   subslice. The zero-copy bind still aliases; the rank pays no copy, no
//!   extra residency, and no load time. `wq`, `wk`, `wv`, `wr` and the
//!   unembedding are all this case, and between them they are most of the
//!   bytes.
//! * A cut on the INPUT axis is a run of COLUMNS, which is a stride and not a
//!   span. [`cols`] has to gather it, so the result is a fresh allocation the
//!   pile does not back. `wo` and the dense `w2` are this case.
//!
//! The distinction is the whole reason this module is separate from `tp`: the
//! arithmetic is symmetric between the two, and the COST is not. Anything that
//! goes through [`cols`] costs a copy at load and device memory that is not
//! shared with the mapping, so the design should reach for it only where the
//! alternative is worse. It is used twice per layer's worth of weights and
//! nowhere in the hot path.
//!
//! # The failure this module exists to make loud
//!
//! Every function here can be got wrong in a way that produces FINITE NUMBERS
//! AND FLUENT TEXT, which is why each has a test pinning the exact wrong answer
//! rather than only the right one:
//!
//! * a contiguous half of `w13`'s `[2 * inter, hidden]` is all of the gate and
//!   none of the up ([`w13_rows`]);
//! * a column range read as a row range returns real weights from the wrong
//!   place ([`cols`]);
//! * a share that overlaps its peer double-counts under an all-reduce sum and
//!   one that gaps drops weight, and neither raises anything.
//!
//! None of these can be caught downstream. A wrong shard is not a crash and not
//! a NaN; it is a slightly different model, and the only place it is cheap to
//! notice is here.

use std::ops::Range;

/// A row-major `[rows, cols]` weight, as the bytes the pile stores.
///
/// `elem` is the stored element width in bytes — 2 for the BF16 the projections
/// are stored as. Kept as a number rather than an enum because this module does
/// no arithmetic ON the elements, only around them, and the caller has already
/// checked the dtype at the point it read the leaf.
#[derive(Clone, Copy, Debug)]
pub struct Slab<'a> {
    pub bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub elem: usize,
}

impl<'a> Slab<'a> {
    pub fn new(bytes: &'a [u8], rows: usize, cols: usize, elem: usize) -> anyhow::Result<Self> {
        let want = rows * cols * elem;
        anyhow::ensure!(
            bytes.len() == want,
            "a [{rows}, {cols}] weight of {elem}-byte elements is {want} bytes, got {}",
            bytes.len()
        );
        Ok(Self {
            bytes,
            rows,
            cols,
            elem,
        })
    }

    fn row_span(&self, r: &Range<usize>) -> anyhow::Result<Range<usize>> {
        anyhow::ensure!(
            r.end <= self.rows,
            "rows {}..{} run past a [{}, {}] weight",
            r.start,
            r.end,
            self.rows,
            self.cols
        );
        let w = self.cols * self.elem;
        Ok(r.start * w..r.end * w)
    }
}

/// A run of whole rows, as a SUBSLICE — the aliasing case.
///
/// Returns a borrow rather than a `Vec` on purpose: the caller must be able to
/// tell, from the type, that binding this costs nothing. If this ever has to
/// become owned, the cost model of the split changes and the change should be
/// visible at every call site rather than hidden behind a signature that did
/// not move.
///
/// Alignment survives: a row of a `[*, hidden]` BF16 weight is `hidden * 2`
/// bytes, and every offset this produces is a multiple of that, so a slice that
/// began 16-byte aligned stays 16-byte aligned for any `hidden` divisible by 8.
/// That matters because the tuned lane picks its load width from the SHAPE and
/// a 4-aligned pointer under a 16-byte load is an async fault that takes the
/// server down.
pub fn rows<'a>(s: &Slab<'a>, r: Range<usize>) -> anyhow::Result<&'a [u8]> {
    let span = s.row_span(&r)?;
    Ok(&s.bytes[span])
}

/// A run of COLUMNS, gathered — the copying case.
///
/// `[rows, cols]` keeping `c` of the columns becomes `[rows, c.len()]`, which
/// is what a row-parallel operand needs: `wo` is `[hidden, heads * head_dim]`
/// and a rank owns the columns matching the heads it computed, so that its
/// partial product covers exactly its own heads and the all-reduce sums the
/// partials into the whole.
///
/// This allocates, and there is no version of it that does not. A column range
/// of a row-major matrix is `rows` separate runs; the mapping cannot express
/// it, and neither can a bind.
pub fn cols(s: &Slab<'_>, c: Range<usize>) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        c.end <= s.cols,
        "columns {}..{} run past a [{}, {}] weight",
        c.start,
        c.end,
        s.rows,
        s.cols
    );
    let stride = s.cols * s.elem;
    let lo = c.start * s.elem;
    let hi = c.end * s.elem;
    let mut out = Vec::with_capacity(s.rows * (hi - lo));
    for r in 0..s.rows {
        out.extend_from_slice(&s.bytes[r * stride + lo..r * stride + hi]);
    }
    Ok(out)
}

/// This rank's half of a `w13`, which is TWO disjoint row ranges.
///
/// `w13` is `[2 * inter, hidden]` with the whole gate block followed by the
/// whole up block, so a rank's share of the intermediate axis appears twice —
/// once in each block — and the two are `inter` rows apart. The result is the
/// gate share followed by the up share, i.e. the same gate-before-up layout at
/// half the width, so the consumer's slicing is unchanged.
///
/// The reason this is a named function with a test rather than two calls to
/// [`rows`] at the call site: the WRONG version is one call to [`rows`] with
/// `0..inter`, it is shorter, it looks obviously right, and it silently
/// computes with all of the gate and none of the up.
pub fn w13_rows(s: &Slab<'_>, gate: Range<usize>, up: Range<usize>) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        gate.len() == up.len(),
        "a w13 half wants equal gate and up shares, got {} and {}",
        gate.len(),
        up.len()
    );
    anyhow::ensure!(
        gate.end <= up.start,
        "the gate block ends at {} and the up block starts at {}; these overlap, which \
         means the [2 * inter, hidden] layout is not gate-before-up any more",
        gate.end,
        up.start
    );
    let mut out = rows(s, gate)?.to_vec();
    out.extend_from_slice(rows(s, up)?);
    Ok(out)
}

/// What a rank's shares cost in bytes, aliased against copied.
///
/// Exists so the load path can REPORT the split's residency rather than have it
/// inferred. The aliased term is shared with the mapping and the copied term is
/// not, so a box's actual footprint is the copied term plus whatever of the
/// mapping it touches — and only one of those two numbers is under this code's
/// control.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShardCost {
    pub aliased: usize,
    pub copied: usize,
}

impl ShardCost {
    pub fn alias(&mut self, n: usize) {
        self.aliased += n;
    }
    pub fn copy(&mut self, n: usize) {
        self.copied += n;
    }
}

impl std::fmt::Display for ShardCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        write!(
            f,
            "{:.1} MiB aliased + {:.1} MiB copied",
            mib(self.aliased),
            mib(self.copied)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inkling::tp::Tp;

    /// A `[rows, cols]` slab whose every element encodes its own address, so a
    /// wrong slice is identifiable rather than merely different.
    fn slab(rows: usize, cols: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(rows * cols * 2);
        for r in 0..rows {
            for c in 0..cols {
                v.extend_from_slice(&((r * 1000 + c) as u16).to_le_bytes());
            }
        }
        v
    }

    fn at(bytes: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])
    }

    #[test]
    fn a_row_range_is_a_subslice_and_starts_where_it_should() {
        let b = slab(8, 4);
        let s = Slab::new(&b, 8, 4, 2).unwrap();
        let half = rows(&s, 4..8).unwrap();
        assert_eq!(half.len(), 4 * 4 * 2);
        // First element of row 4, not of row 0.
        assert_eq!(at(half, 0), 4000);
        assert_eq!(at(half, 3), 4003);
        assert_eq!(at(half, 4), 5000);
    }

    #[test]
    fn the_two_ranks_row_shares_tile_the_weight() {
        let b = slab(32, 4);
        let s = Slab::new(&b, 32, 4, 2).unwrap();
        let (a, c) = (Tp::new(0, 2).unwrap(), Tp::new(1, 2).unwrap());
        let ra = rows(&s, a.shard("rows", 32).unwrap()).unwrap();
        let rb = rows(&s, c.shard("rows", 32).unwrap()).unwrap();
        assert_eq!(ra.len() + rb.len(), b.len());
        let mut joined = ra.to_vec();
        joined.extend_from_slice(rb);
        assert_eq!(joined, b, "the two shares do not reassemble the original");
    }

    #[test]
    fn a_column_range_gathers_a_stride_not_a_span() {
        let b = slab(4, 8);
        let s = Slab::new(&b, 4, 8, 2).unwrap();
        let right = cols(&s, 4..8).unwrap();
        assert_eq!(right.len(), 4 * 4 * 2);
        // Row 0's columns 4..8, then row 1's -- NOT rows 2..4, which is what a
        // row-range reading of the same numbers would have returned.
        assert_eq!(at(&right, 0), 4);
        assert_eq!(at(&right, 3), 7);
        assert_eq!(at(&right, 4), 1004);
        let as_rows = rows(&s, 2..4).unwrap();
        assert_ne!(
            at(&right, 0),
            at(as_rows, 0),
            "columns 4..8 and rows 2..4 must not agree, or this test proves nothing"
        );
    }

    #[test]
    fn column_shares_also_tile() {
        let b = slab(4, 8);
        let s = Slab::new(&b, 4, 8, 2).unwrap();
        let l = cols(&s, 0..4).unwrap();
        let r = cols(&s, 4..8).unwrap();
        // Interleaved back together row by row, they are the original.
        let mut back = Vec::new();
        for row in 0..4 {
            back.extend_from_slice(&l[row * 8..(row + 1) * 8]);
            back.extend_from_slice(&r[row * 8..(row + 1) * 8]);
        }
        assert_eq!(back, b);
    }

    /// The one the module header calls out: a contiguous half is all gate.
    #[test]
    fn w13_takes_a_share_from_each_block_not_the_first_block_whole() {
        const INTER: usize = 8;
        let b = slab(2 * INTER, 4);
        let s = Slab::new(&b, 2 * INTER, 4, 2).unwrap();
        let t = Tp::new(0, 2).unwrap();
        let (g, u) = t.w13_halves(INTER).unwrap();
        assert_eq!((g.clone(), u.clone()), (0..4, 8..12));
        let mine = w13_rows(&s, g, u).unwrap();
        assert_eq!(mine.len(), 8 * 4 * 2);
        // Gate rows 0..4 then UP rows 8..12.
        assert_eq!(at(&mine, 0), 0);
        assert_eq!(
            at(&mine, 4 * 4),
            8000,
            "the second block must be the UP rows"
        );
        // The wrong answer, pinned: rows 0..8 is the whole gate block and no up.
        let wrong = rows(&s, 0..8).unwrap();
        assert_eq!(at(wrong, 4 * 4), 4000);
        assert_ne!(at(&mine, 4 * 4), at(wrong, 4 * 4));
    }

    #[test]
    fn both_ranks_w13_halves_reassemble_the_whole_weight() {
        const INTER: usize = 8;
        let b = slab(2 * INTER, 4);
        let s = Slab::new(&b, 2 * INTER, 4, 2).unwrap();
        let mut seen: Vec<u16> = Vec::new();
        for rank in 0..2 {
            let t = Tp::new(rank, 2).unwrap();
            let (g, u) = t.w13_halves(INTER).unwrap();
            let mine = w13_rows(&s, g, u).unwrap();
            for i in 0..mine.len() / 2 {
                seen.push(at(&mine, i));
            }
        }
        seen.sort();
        let mut want: Vec<u16> = (0..b.len() / 2).map(|i| at(&b, i)).collect();
        want.sort();
        assert_eq!(seen, want, "the four w13 ranges do not tile the weight");
    }

    #[test]
    fn a_slab_that_is_not_its_declared_shape_is_refused() {
        let b = slab(4, 4);
        assert!(Slab::new(&b, 4, 4, 2).is_ok());
        assert!(Slab::new(&b, 4, 8, 2).is_err());
        let s = Slab::new(&b, 4, 4, 2).unwrap();
        assert!(rows(&s, 2..6).is_err());
        assert!(cols(&s, 2..6).is_err());
    }

    #[test]
    fn an_overlapping_w13_split_is_refused_rather_than_silently_double_counted() {
        let b = slab(16, 4);
        let s = Slab::new(&b, 16, 4, 2).unwrap();
        // gate 0..8 and up 4..12 overlap: under an all-reduce sum this would
        // double-count four rows and drop four others, and produce text.
        assert!(w13_rows(&s, 0..8, 4..12).is_err());
        // Unequal shares are refused too.
        assert!(w13_rows(&s, 0..4, 8..14).is_err());
    }

    #[test]
    fn the_cost_report_separates_what_the_mapping_backs_from_what_it_does_not() {
        let mut c = ShardCost::default();
        c.alias(2 * 1024 * 1024);
        c.copy(1024 * 1024);
        assert_eq!(
            c,
            ShardCost {
                aliased: 2 * 1024 * 1024,
                copied: 1024 * 1024
            }
        );
        assert_eq!(c.to_string(), "2.0 MiB aliased + 1.0 MiB copied");
    }
}
