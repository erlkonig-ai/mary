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

// ---------------------------------------------------------------------------
// The cut the ROUTED EXPERTS take, which happens inside the startup copy.
// ---------------------------------------------------------------------------

/// Which axis of one stored `[rows, cols]` matrix a rank keeps.
///
/// Named for the AXIS and not for the weight, because the two matrices of one
/// expert are cut on different axes by the same [`super::tp::Tp`] range and
/// swapping them is the silent failure this module exists for: `w13` cut on
/// columns and `w2` cut on rows is a real matrix of real weights, of the right
/// shape, computing a different model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cut {
    /// A run of whole ROWS -- the OUTPUT axis. Contiguous in a row-major
    /// store, so the copy is one `memcpy` per plane.
    Rows(Range<usize>),
    /// A run of COLUMNS -- the INPUT axis. `rows` separate runs; the copy
    /// gathers.
    Cols(Range<usize>),
}

impl Cut {
    /// The dims of the cut matrix, given the whole one's.
    pub fn dims(&self, rows: usize, cols: usize) -> (usize, usize) {
        match self {
            Cut::Rows(r) => (r.len(), cols),
            Cut::Cols(c) => (rows, c.len()),
        }
    }

    /// Whether this is the identity -- the whole matrix, uncut.
    pub fn is_whole(&self, rows: usize, cols: usize) -> bool {
        match self {
            Cut::Rows(r) => *r == (0..rows),
            Cut::Cols(c) => *c == (0..cols),
        }
    }
}

/// How many bytes one row-major plane spends per column: `num` bytes per `den`
/// columns.
///
/// NVFP4 codes are one nibble a column (`1/2`), its E4M3 block scales are one
/// byte per sixteen (`1/16`), and BF16 is two bytes a column (`2/1`). Stating
/// the RATIO rather than a byte width is what lets one cutter serve all three,
/// and it is what makes a column range that does not land on a byte boundary an
/// error HERE rather than a half-byte shift that reads as slightly different
/// weights.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub num: usize,
    pub den: usize,
}

impl Plane {
    /// NVFP4 4-bit codes: two columns to the byte.
    pub const NVFP4_CODES: Plane = Plane { num: 1, den: 2 };
    /// NVFP4 E4M3 block scales: one byte per 16 columns.
    pub const NVFP4_SCALES: Plane = Plane { num: 1, den: 16 };
    /// BF16: two bytes a column.
    pub const BF16: Plane = Plane { num: 2, den: 1 };

    fn bytes(&self, cols: usize) -> anyhow::Result<usize> {
        anyhow::ensure!(
            cols % self.den == 0,
            "a plane of {}/{} bytes per column cannot address {cols} columns: the boundary \
             falls inside a byte, and a half-byte shift is a different weight, not a smaller one",
            self.num,
            self.den
        );
        Ok(cols / self.den * self.num)
    }
}

/// This rank's share of one row-major plane.
///
/// Returns a BORROW for [`Cut::Rows`] -- a row range is a span, so the arena
/// copy that follows is the only copy -- and an owned gather for [`Cut::Cols`].
/// The type says which, at every call site, for the same reason [`rows`] and
/// [`cols`] have different return types.
pub fn cut_plane<'a>(
    src: &'a [u8],
    rows: usize,
    cols: usize,
    plane: Plane,
    cut: &Cut,
) -> anyhow::Result<std::borrow::Cow<'a, [u8]>> {
    let stride = plane.bytes(cols)?;
    anyhow::ensure!(
        src.len() == rows * stride,
        "a [{rows}, {cols}] plane of {}/{} bytes per column is {} bytes, got {}",
        plane.num,
        plane.den,
        rows * stride,
        src.len()
    );
    match cut {
        Cut::Rows(r) => {
            anyhow::ensure!(
                r.end <= rows,
                "rows {}..{} run past a [{rows}, {cols}] plane",
                r.start,
                r.end
            );
            Ok(std::borrow::Cow::Borrowed(
                &src[r.start * stride..r.end * stride],
            ))
        }
        Cut::Cols(c) => {
            anyhow::ensure!(
                c.end <= cols,
                "columns {}..{} run past a [{rows}, {cols}] plane",
                c.start,
                c.end
            );
            let lo = plane.bytes(c.start)?;
            let hi = plane.bytes(c.end)?;
            let mut out = Vec::with_capacity(rows * (hi - lo));
            for r in 0..rows {
                out.extend_from_slice(&src[r * stride + lo..r * stride + hi]);
            }
            Ok(std::borrow::Cow::Owned(out))
        }
    }
}

/// This rank's cut of one routed-expert stacked matrix, chosen by NAME.
///
/// The two matrices are cut on DIFFERENT axes and by the SAME intermediate
/// range, and which is which is a fact about the checkpoint rather than about
/// the caller -- so it is decided here, once, from the name the pile stores:
///
/// * `mlp.experts.w13_weight` is `[2 * inter, hidden]` with its output rows
///   INTERLEAVED `g0, u0, g1, u1, ...` (`Interleave(dim=1)` in the conversion,
///   and what [`super::fp4gemm::gate_up_silu`] reads). A rank's share of the
///   intermediate axis is therefore ONE contiguous run of `2 * inter / world`
///   rows -- see [`super::tp::Tp::w13_interleaved_rows`].
/// * `mlp.experts.w2_weight` is `[hidden, inter]`, so the same intermediate
///   range is a COLUMN range.
///
/// A name this does not know is an ERROR and never "leave it whole": an
/// unsharded operand under an all-reduce is not less parallel, it is summed
/// twice. If a new expert stack appears, it has to be cut here or the run has
/// to refuse.
pub fn routed_cut(
    tp: crate::models::inkling::tp::Tp,
    name: &str,
    rows: usize,
    cols: usize,
) -> anyhow::Result<Cut> {
    let base = name.rsplit('.').take(3).collect::<Vec<_>>();
    // `model.llm.layers.7.mlp.experts.w13_weight` -> ["w13_weight", "experts", "mlp"]
    match (base.first().copied(), base.get(1).copied()) {
        (Some("w13_weight"), Some("experts")) => {
            anyhow::ensure!(
                rows % 2 == 0,
                "{name} is [{rows}, {cols}]; a fused gate/up matrix has an even row count"
            );
            Ok(Cut::Rows(
                tp.w13_interleaved_rows(rows / 2)
                    .map_err(|e| anyhow::anyhow!("{name}: {e}"))?,
            ))
        }
        (Some("w2_weight"), Some("experts")) => Ok(Cut::Cols(
            tp.routed_inter(cols)
                .map_err(|e| anyhow::anyhow!("{name}: {e}"))?,
        )),
        _ => anyhow::bail!(
            "{name} is an expert stack this split does not know how to cut. Every routed \
             operand sits under the MoE all-reduce, so leaving one whole makes both ranks \
             compute it and the reduce sum two copies -- 2x, finite, and fluent. Add its cut \
             to `tpshard::routed_cut` or run without INK_TP."
        ),
    }
}

// ---------------------------------------------------------------------------
// The cuts the DENSE leaves take, which ALSO happen inside the startup copy.
// ---------------------------------------------------------------------------

/// What one dense leaf KEEPS under the split, with the leaf read as a
/// row-major `[rows, cols]` matrix.
///
/// Row runs are a LIST because the shared experts' fused `w13` is
/// `[n_shared, 2 * inter, hidden]` and a rank's share of the intermediate axis
/// appears once per expert block (twice, when the blocks are stored halved) --
/// several spans of one leaf, in order. Every other row cut is a list of one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Keep {
    /// Whole-row runs, concatenated in order: spans, one `memcpy` each.
    Rows(Vec<Range<usize>>),
    /// One column range of every row: a gather.
    Cols(Range<usize>),
}

/// This rank's cut of one DENSE leaf, as the startup copy applies it.
///
/// # Why the dense cuts moved into the copy
///
/// Until this existed the binders in `assembly` cut these leaves at BIND time
/// -- a row range of the arena for `wq`, a gather for `wo`, the de-interleave
/// and then a row range for the fused `w13`s -- and every one of those cuts
/// was applied to a leaf the startup copy had already written into the arena
/// WHOLE. The device read half either way, so it was never a bandwidth term;
/// it was ~5.2 GiB of residency on a 119.63 GiB box that admitted 87.07 GiB of
/// arena plus 5.21 GiB of pool with 1.95 GiB to spare, and a decode step on
/// that box cost 56.5 ms when the allocator was not fighting for pages and
/// 200-670 ms when it was (`tp.rs`, "What is left REPLICATED in the arena").
///
/// Here the cut is a change of SOURCE index inside the copy that already
/// runs, exactly as [`routed_cut`] is for the experts: the arena is allocated
/// at the cut size, the leaf's `dims` become the cut dims, and the binder is
/// handed this rank's share as if it were the whole leaf. The bytes that
/// reach `bind_bf16` are the SAME bytes the bind-time cut produced, in the
/// same order -- the tests below pin that per tensor -- so this is a
/// copy-order change and not an arithmetic one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseCut {
    /// The leaf as a matrix: the leading dim, then the product of the rest.
    pub rows: usize,
    pub cols: usize,
    pub keep: Keep,
    /// The leaf's dims AFTER the cut, in the checkpoint's own rank -- so a
    /// `[n_shared, 2 * inter, hidden]` leaf stays rank 3 and the binder that
    /// reads `dims[1]` reads this rank's width.
    pub dims: Vec<u64>,
}

impl DenseCut {
    fn rows_kept(&self) -> usize {
        match &self.keep {
            Keep::Rows(runs) => runs.iter().map(|r| r.len()).sum(),
            Keep::Cols(_) => self.rows,
        }
    }

    fn cols_kept(&self) -> usize {
        match &self.keep {
            Keep::Rows(_) => self.cols,
            Keep::Cols(c) => c.len(),
        }
    }

    /// Bytes the cut leaf occupies at `width` bytes an element -- what the
    /// arena layout reserves for it.
    pub fn bytes(&self, width: usize) -> usize {
        self.rows_kept() * self.cols_kept() * width
    }

    /// Copy what is kept out of `src`, the WHOLE leaf, into `dst`, which must
    /// be exactly [`DenseCut::bytes`] long. Spans are `memcpy`s; a column cut
    /// gathers row by row straight into the arena, with no transient buffer.
    pub fn copy_into(&self, src: &[u8], width: usize, dst: &mut [u8]) -> anyhow::Result<()> {
        let stride = self.cols * width;
        anyhow::ensure!(
            src.len() == self.rows * stride,
            "a [{}, {}] leaf of {width}-byte elements is {} bytes, got {}",
            self.rows,
            self.cols,
            self.rows * stride,
            src.len()
        );
        anyhow::ensure!(
            dst.len() == self.bytes(width),
            "the cut of a [{}, {}] leaf is {} bytes, but its arena slot is {}",
            self.rows,
            self.cols,
            self.bytes(width),
            dst.len()
        );
        match &self.keep {
            Keep::Rows(runs) => {
                let mut at = 0usize;
                for r in runs {
                    anyhow::ensure!(
                        r.end <= self.rows,
                        "rows {}..{} run past a [{}, {}] leaf",
                        r.start,
                        r.end,
                        self.rows,
                        self.cols
                    );
                    let n = r.len() * stride;
                    dst[at..at + n].copy_from_slice(&src[r.start * stride..r.end * stride]);
                    at += n;
                }
            }
            Keep::Cols(c) => {
                anyhow::ensure!(
                    c.end <= self.cols,
                    "columns {}..{} run past a [{}, {}] leaf",
                    c.start,
                    c.end,
                    self.rows,
                    self.cols
                );
                let (lo, hi) = (c.start * width, c.end * width);
                for (r, out) in dst.chunks_exact_mut(hi - lo).enumerate() {
                    out.copy_from_slice(&src[r * stride + lo..r * stride + hi]);
                }
            }
        }
        Ok(())
    }
}

/// This rank's cut of one dense leaf, chosen by NAME and stored dims -- or
/// `None` for a leaf the split REPLICATES.
///
/// The rule per tensor is the binder's own, restated once here so the two
/// cannot drift: `assembly::bind_layer` cuts the four output-parallel
/// projections and both attention short convolutions by row and `wo` by
/// column; `DeviceDense::dense_for` cuts the interleaved `w13_dn` by row (one
/// contiguous run, because interleaving keeps each unit's gate and up
/// adjacent -- [`super::tp::Tp::w13_interleaved_rows`]) and `w2_md` by column;
/// `DeviceDense::shared_for` cuts the shared `w13` per expert block in
/// whichever of its two readings `shared_w13_halved` selects, and the shared
/// `w2` per expert block by column, which over the stacked
/// `[n_shared * hidden, inter]` matrix is ONE column range.
///
/// Only `model.llm.layers.*`: the MTP heads wear the same suffixes and are
/// bound whole by a binder that knows no split. Everything else -- the norm
/// gains, the router projection, the MLP short convolution, the embedding and
/// unembedding tables -- is replicated, so `None` is the correct answer and
/// not a pass-through: none of it sits under a reduce.
///
/// A leaf this DOES know, at dims it does not expect, is an ERROR rather than
/// left whole, for the reason [`routed_cut`] gives: the binder will trust that
/// the copy cut it, and an uncut operand under the all-reduce is summed twice.
pub fn dense_cut(
    tp: crate::models::inkling::tp::Tp,
    name: &str,
    dims: &[u64],
    shared_w13_halved: bool,
) -> anyhow::Result<Option<DenseCut>> {
    let Some(rest) = name
        .strip_prefix("model.llm.layers.")
        .and_then(|s| s.split_once('.'))
        .map(|(_, rest)| rest)
    else {
        return Ok(None);
    };
    let d: Vec<usize> = dims.iter().map(|&x| x as usize).collect();
    let lift = |e: crate::models::inkling::tp::Indivisible| anyhow::anyhow!("{name}: {e}");
    let as_u64 = |v: &[usize]| v.iter().map(|&x| x as u64).collect::<Vec<u64>>();
    // The leaf as a matrix: the leading dim by the product of the rest.
    let matrix = || -> anyhow::Result<(usize, usize)> {
        anyhow::ensure!(
            d.len() >= 2,
            "{name} is {dims:?}; this split cuts it as a matrix"
        );
        Ok((d[0], d[1..].iter().product()))
    };
    // A run of whole rows, on whichever head axis `axis` names.
    let rows_on = |axis: &'static str| -> anyhow::Result<DenseCut> {
        let (rows, cols) = matrix()?;
        let r = tp.shard(axis, rows).map_err(lift)?;
        let mut dims = d.clone();
        dims[0] = r.len();
        Ok(DenseCut {
            rows,
            cols,
            keep: Keep::Rows(vec![r]),
            dims: as_u64(&dims),
        })
    };
    let cut = match rest {
        // Output-parallel: whole rows of `[out, hidden]`, and the per-KV-head
        // convolution state that follows the same head cut.
        "attn.wq_du.weight" | "attn.wr_du.weight" => rows_on("num_attention_heads * head_dim")?,
        "attn.wk_dv.weight"
        | "attn.wv_dv.weight"
        | "attn.k_sconv.weight"
        | "attn.v_sconv.weight" => rows_on("num_key_value_heads * head_dim")?,
        // Input-parallel: the columns matching the heads this rank computed.
        "attn.wo_ud.weight" => {
            anyhow::ensure!(
                d.len() == 2,
                "{name} is {dims:?}, not [hidden, heads * head_dim]"
            );
            let (rows, cols) = (d[0], d[1]);
            let c = tp
                .shard("num_attention_heads * head_dim", cols)
                .map_err(lift)?;
            DenseCut {
                rows,
                cols,
                dims: vec![rows as u64, c.len() as u64],
                keep: Keep::Cols(c),
            }
        }
        // The dense MLP, stored INTERLEAVED `g0, u0, g1, u1, ...`: a rank's
        // units are one contiguous run of `2 * inter / world` rows.
        "mlp.w13_dn.weight" => {
            anyhow::ensure!(d.len() == 2, "{name} is {dims:?}, not [2 * inter, hidden]");
            let (rows, cols) = (d[0], d[1]);
            anyhow::ensure!(
                rows % 2 == 0,
                "{name} is [{rows}, {cols}]; a fused gate/up matrix has an even row count"
            );
            let r = tp.w13_interleaved_rows(rows / 2).map_err(lift)?;
            DenseCut {
                rows,
                cols,
                dims: vec![r.len() as u64, cols as u64],
                keep: Keep::Rows(vec![r]),
            }
        }
        "mlp.w2_md.weight" => {
            anyhow::ensure!(d.len() == 2, "{name} is {dims:?}, not [hidden, inter]");
            let (rows, cols) = (d[0], d[1]);
            let c = tp.dense_inter(cols).map_err(lift)?;
            DenseCut {
                rows,
                cols,
                dims: vec![rows as u64, c.len() as u64],
                keep: Keep::Cols(c),
            }
        }
        // The shared experts' fused `w13`, `[n_shared, 2 * inter, hidden]`:
        // the same share of the intermediate axis out of EVERY expert block,
        // laid out the way `split_shared_w13_bytes` will read the result.
        "mlp.shared_experts.shared_w13_weight" => {
            anyhow::ensure!(
                d.len() == 3,
                "{name} is {dims:?}, not [n_shared, 2 * inter, hidden]"
            );
            let (n_shared, two_inter, h) = (d[0], d[1], d[2]);
            anyhow::ensure!(
                two_inter % 2 == 0,
                "{name} is {dims:?}; a fused gate/up block has an even row count"
            );
            let inter = two_inter / 2;
            let mut runs = Vec::with_capacity(2 * n_shared);
            for s in 0..n_shared {
                let base = s * two_inter;
                if shared_w13_halved {
                    let (g, u) = tp.w13_halved_rows(inter).map_err(lift)?;
                    runs.push(base + g.start..base + g.end);
                    runs.push(base + u.start..base + u.end);
                } else {
                    let r = tp.w13_interleaved_rows(inter).map_err(lift)?;
                    runs.push(base + r.start..base + r.end);
                }
            }
            let kept = 2 * tp.share("intermediate_size", inter).map_err(lift)?;
            DenseCut {
                rows: n_shared * two_inter,
                cols: h,
                keep: Keep::Rows(runs),
                dims: vec![n_shared as u64, kept as u64, h as u64],
            }
        }
        // `[n_shared, hidden, inter]`: the blocks stack row-wise, so one column
        // range of the `[n_shared * hidden, inter]` matrix is every block's
        // column range, in block order -- what the bind-time gather produced.
        "mlp.shared_experts.shared_w2_weight" => {
            anyhow::ensure!(
                d.len() == 3,
                "{name} is {dims:?}, not [n_shared, hidden, inter]"
            );
            let (n_shared, h, inter) = (d[0], d[1], d[2]);
            let c = tp.shared_inter(inter).map_err(lift)?;
            DenseCut {
                rows: n_shared * h,
                cols: inter,
                dims: vec![n_shared as u64, h as u64, c.len() as u64],
                keep: Keep::Cols(c),
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(cut))
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
        let (g, u) = t.w13_halved_rows(INTER).unwrap();
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
    fn both_ranks_w13_halved_rows_reassemble_the_whole_weight() {
        const INTER: usize = 8;
        let b = slab(2 * INTER, 4);
        let s = Slab::new(&b, 2 * INTER, 4, 2).unwrap();
        let mut seen: Vec<u16> = Vec::new();
        for rank in 0..2 {
            let t = Tp::new(rank, 2).unwrap();
            let (g, u) = t.w13_halved_rows(INTER).unwrap();
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

    /// A row cut is a span and a column cut is a gather, on a plane whose
    /// element is HALF A BYTE -- which is the case `rows`/`cols` above cannot
    /// express and the routed experts are entirely made of.
    #[test]
    fn a_nibble_plane_cuts_on_both_axes() {
        // [4, 8] NVFP4 codes: 4 bytes a row, byte `r * 4 + c/2` holds columns
        // `2c, 2c+1`.
        let src: Vec<u8> = (0..16u8).collect();
        let s = &src[..];
        let top = cut_plane(s, 4, 8, Plane::NVFP4_CODES, &Cut::Rows(2..4)).unwrap();
        assert_eq!(&*top, &[8, 9, 10, 11, 12, 13, 14, 15]);
        let right = cut_plane(s, 4, 8, Plane::NVFP4_CODES, &Cut::Cols(4..8)).unwrap();
        // Columns 4..8 are bytes 2..4 of every row.
        assert_eq!(&*right, &[2, 3, 6, 7, 10, 11, 14, 15]);
        // and the two are not the same bytes, or this proves nothing
        assert_ne!(&*top, &*right);
    }

    /// The half-byte boundary, refused rather than shifted.
    #[test]
    fn a_column_cut_that_lands_inside_a_byte_is_refused() {
        // 4-bit codes cannot start at column 3.
        let codes = vec![0u8; 4 * 4];
        assert!(cut_plane(&codes, 4, 8, Plane::NVFP4_CODES, &Cut::Cols(3..7)).is_err());
        assert!(cut_plane(&codes, 4, 8, Plane::NVFP4_CODES, &Cut::Cols(2..6)).is_ok());
        // and E4M3 scales cannot start at column 8 of a 16-per-byte grouping.
        // `[2, 32]` scales are one byte per 16 columns: 2 rows x 2 bytes.
        let sc = vec![0u8; 2 * 2];
        assert!(cut_plane(&sc, 2, 32, Plane::NVFP4_SCALES, &Cut::Cols(8..24)).is_err());
        assert!(cut_plane(&sc, 2, 32, Plane::NVFP4_SCALES, &Cut::Cols(0..16)).is_ok());
    }

    /// The two ranks' cuts tile the plane, on both axes and at both widths.
    #[test]
    fn the_two_ranks_plane_cuts_tile_the_plane() {
        let (a, b) = (Tp::new(0, 2).unwrap(), Tp::new(1, 2).unwrap());
        let src: Vec<u8> = (0..64u8).collect();
        // Rows: concatenation is the original.
        let mut joined = cut_plane(
            &src,
            8,
            16,
            Plane::NVFP4_CODES,
            &Cut::Rows(a.shard("r", 8).unwrap()),
        )
        .unwrap()
        .into_owned();
        joined.extend_from_slice(
            &cut_plane(
                &src,
                8,
                16,
                Plane::NVFP4_CODES,
                &Cut::Rows(b.shard("r", 8).unwrap()),
            )
            .unwrap(),
        );
        assert_eq!(joined, src);
        // Columns: interleaving row by row is the original.
        let l = cut_plane(
            &src,
            8,
            16,
            Plane::NVFP4_CODES,
            &Cut::Cols(a.shard("c", 16).unwrap()),
        )
        .unwrap()
        .into_owned();
        let r = cut_plane(
            &src,
            8,
            16,
            Plane::NVFP4_CODES,
            &Cut::Cols(b.shard("c", 16).unwrap()),
        )
        .unwrap()
        .into_owned();
        let mut back = Vec::new();
        for row in 0..8 {
            back.extend_from_slice(&l[row * 4..(row + 1) * 4]);
            back.extend_from_slice(&r[row * 4..(row + 1) * 4]);
        }
        assert_eq!(back, src);
    }

    /// The routed cut names the axis from the WEIGHT, and gets both right.
    #[test]
    fn the_routed_cut_reads_w13_by_row_and_w2_by_column() {
        const INTER: usize = 2048;
        const H: usize = 4096;
        let (a, b) = (Tp::new(0, 2).unwrap(), Tp::new(1, 2).unwrap());
        let n13 = "model.llm.layers.7.mlp.experts.w13_weight";
        let n2 = "model.llm.layers.7.mlp.experts.w2_weight";
        assert_eq!(
            routed_cut(a, n13, 2 * INTER, H).unwrap(),
            Cut::Rows(0..2048)
        );
        assert_eq!(
            routed_cut(b, n13, 2 * INTER, H).unwrap(),
            Cut::Rows(2048..4096)
        );
        assert_eq!(routed_cut(a, n2, H, INTER).unwrap(), Cut::Cols(0..1024));
        assert_eq!(routed_cut(b, n2, H, INTER).unwrap(), Cut::Cols(1024..2048));
        // The shapes agree: rank r's w13 produces `inter/2` intermediate units
        // and its w2 consumes exactly `inter/2` of them.
        let (r13, c13) = routed_cut(b, n13, 2 * INTER, H).unwrap().dims(2 * INTER, H);
        let (r2, c2) = routed_cut(b, n2, H, INTER).unwrap().dims(H, INTER);
        assert_eq!(
            r13 / 2,
            c2,
            "w13's rows and w2's columns must name the same units"
        );
        assert_eq!((c13, r2), (H, H));
    }

    /// A stack nobody taught this about is an ERROR, not a pass-through.
    #[test]
    fn an_unknown_expert_stack_is_refused_rather_than_left_whole() {
        let a = Tp::new(0, 2).unwrap();
        let e = routed_cut(a, "model.llm.layers.7.mlp.experts.w4_weight", 16, 16).unwrap_err();
        assert!(e.to_string().contains("does not know how to cut"));
        // and the identity world still cuts nothing
        let one = Tp::default();
        assert_eq!(
            routed_cut(one, "model.llm.layers.7.mlp.experts.w2_weight", 16, 16).unwrap(),
            Cut::Cols(0..16)
        );
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

    // ---- the dense cuts, byte for byte against the bind-time path ---------
    //
    // Each of these runs the cut the binder USED to make on a synthetic leaf,
    // runs `dense_cut` + `copy_into` on the same leaf, and demands the same
    // bytes. Small shapes, every element its own address, both ranks.

    /// Apply a `DenseCut` the way `copy_share` does: into a slot of exactly
    /// its size.
    fn apply(cut: &DenseCut, src: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; cut.bytes(2)];
        cut.copy_into(src, 2, &mut out).unwrap();
        out
    }

    fn l(name: &str) -> String {
        format!("model.llm.layers.3.{name}")
    }

    #[test]
    fn the_projections_cut_by_row_exactly_as_the_binder_did() {
        // wq [heads * head_dim, hidden] = [8, 4]; wo [hidden, heads * head_dim].
        let b = slab(8, 4);
        let s = Slab::new(&b, 8, 4, 2).unwrap();
        for rank in 0..2 {
            let t = Tp::new(rank, 2).unwrap();
            let r = t.shard("rows", 8).unwrap();
            let want = rows(&s, r.clone()).unwrap();
            for nm in [
                "attn.wq_du.weight",
                "attn.wk_dv.weight",
                "attn.wr_du.weight",
            ] {
                let c = dense_cut(t, &l(nm), &[8, 4], false).unwrap().unwrap();
                assert_eq!(c.dims, vec![4, 4]);
                assert_eq!(apply(&c, &b), want, "{nm} rank {rank}");
            }
            // The convolution state is rank 3 in the pile and cuts on its
            // leading axis with the rest flattened.
            let c = dense_cut(t, &l("attn.k_sconv.weight"), &[8, 1, 4], false)
                .unwrap()
                .unwrap();
            assert_eq!(c.dims, vec![4, 1, 4]);
            assert_eq!(apply(&c, &b), want);
        }
    }

    #[test]
    fn wo_cuts_by_column_exactly_as_the_binder_did() {
        let b = slab(4, 8);
        let s = Slab::new(&b, 4, 8, 2).unwrap();
        for rank in 0..2 {
            let t = Tp::new(rank, 2).unwrap();
            let want = cols(&s, t.shard("cols", 8).unwrap()).unwrap();
            let c = dense_cut(t, &l("attn.wo_ud.weight"), &[4, 8], false)
                .unwrap()
                .unwrap();
            assert_eq!(c.dims, vec![4, 4]);
            assert_eq!(apply(&c, &b), want, "rank {rank}");
            // and NOT the row reading of the same numbers
            assert_ne!(apply(&c, &b), rows(&s, t.shard("r", 4).unwrap()).unwrap());
        }
    }

    /// The dense MLP: the binder de-interleaved the fused weight and THEN took
    /// a row range of each half. The copy takes one contiguous run of the
    /// interleaved weight, and de-interleaving THAT has to give the same two
    /// halves.
    #[test]
    fn the_dense_w13_cut_deinterleaves_to_the_binders_halves() {
        use crate::models::inkling::load::split_gate_up_bytes;
        const INTER: usize = 8;
        const H: usize = 4;
        let fused = slab(2 * INTER, H);
        for rank in 0..2 {
            let t = Tp::new(rank, 2).unwrap();
            // bind time
            let (g, u) = split_gate_up_bytes(&fused, H, 2);
            let r = t.dense_inter(INTER).unwrap();
            let gs = Slab::new(&g, INTER, H, 2).unwrap();
            let us = Slab::new(&u, INTER, H, 2).unwrap();
            let want_g = rows(&gs, r.clone()).unwrap().to_vec();
            let want_u = rows(&us, r.clone()).unwrap().to_vec();
            // copy time
            let c = dense_cut(
                t,
                &l("mlp.w13_dn.weight"),
                &[2 * INTER as u64, H as u64],
                false,
            )
            .unwrap()
            .unwrap();
            assert_eq!(c.dims, vec![INTER as u64, H as u64]);
            let cut = apply(&c, &fused);
            let (got_g, got_u) = split_gate_up_bytes(&cut, H, 2);
            assert_eq!(got_g, want_g, "gate, rank {rank}");
            assert_eq!(got_u, want_u, "up, rank {rank}");
            // and w2 by column
            let w2 = slab(H, INTER);
            let ds = Slab::new(&w2, H, INTER, 2).unwrap();
            let c2 = dense_cut(t, &l("mlp.w2_md.weight"), &[H as u64, INTER as u64], false)
                .unwrap()
                .unwrap();
            assert_eq!(apply(&c2, &w2), cols(&ds, r).unwrap());
        }
    }

    /// The shared experts, in BOTH readings of the fused block: the binder
    /// split the whole weight and then took rows `e * inter + r` of each half
    /// for each expert `e`; the copy takes the runs and the split of the cut
    /// weight at the LOCAL width has to agree.
    #[test]
    fn the_shared_w13_cut_splits_to_the_binders_per_expert_rows() {
        use crate::models::inkling::load::split_shared_w13_bytes;
        const NS: usize = 2;
        const INTER: usize = 8;
        const H: usize = 4;
        let fused = slab(NS * 2 * INTER, H);
        for halved in [false, true] {
            for rank in 0..2 {
                let t = Tp::new(rank, 2).unwrap();
                let r = t.shared_inter(INTER).unwrap();
                // bind time: `DeviceDense::shared_for`'s `per_expert_rows`
                let (g, u) = split_shared_w13_bytes(&fused, NS, INTER, H, halved, 2);
                let per = |src: &[u8]| -> Vec<u8> {
                    let s = Slab::new(src, NS * INTER, H, 2).unwrap();
                    let mut out = Vec::new();
                    for e in 0..NS {
                        out.extend_from_slice(
                            rows(&s, e * INTER + r.start..e * INTER + r.end).unwrap(),
                        );
                    }
                    out
                };
                let (want_g, want_u) = (per(&g), per(&u));
                // copy time
                let c = dense_cut(
                    t,
                    &l("mlp.shared_experts.shared_w13_weight"),
                    &[NS as u64, 2 * INTER as u64, H as u64],
                    halved,
                )
                .unwrap()
                .unwrap();
                assert_eq!(c.dims, vec![NS as u64, INTER as u64, H as u64]);
                let cut = apply(&c, &fused);
                let (got_g, got_u) = split_shared_w13_bytes(&cut, NS, INTER / 2, H, halved, 2);
                assert_eq!(got_g, want_g, "gate, halved={halved} rank {rank}");
                assert_eq!(got_u, want_u, "up, halved={halved} rank {rank}");
                // The two readings cut DIFFERENT rows of the same leaf, or
                // the flag is not doing anything here.
                let other = dense_cut(
                    t,
                    &l("mlp.shared_experts.shared_w13_weight"),
                    &[NS as u64, 2 * INTER as u64, H as u64],
                    !halved,
                )
                .unwrap()
                .unwrap();
                assert_ne!(apply(&other, &fused), cut);
            }
        }
    }

    #[test]
    fn the_shared_w2_cut_is_the_binders_per_expert_column_gather() {
        const NS: usize = 2;
        const INTER: usize = 8;
        const H: usize = 4;
        let w2 = slab(NS * H, INTER);
        for rank in 0..2 {
            let t = Tp::new(rank, 2).unwrap();
            let r = t.shared_inter(INTER).unwrap();
            // bind time: per expert block, a column gather
            let mut want = Vec::new();
            for e in 0..NS {
                let blk = &w2[e * H * INTER * 2..(e + 1) * H * INTER * 2];
                let s = Slab::new(blk, H, INTER, 2).unwrap();
                want.extend_from_slice(&cols(&s, r.clone()).unwrap());
            }
            let c = dense_cut(
                t,
                &l("mlp.shared_experts.shared_w2_weight"),
                &[NS as u64, H as u64, INTER as u64],
                false,
            )
            .unwrap()
            .unwrap();
            assert_eq!(c.dims, vec![NS as u64, H as u64, INTER as u64 / 2]);
            assert_eq!(apply(&c, &w2), want, "rank {rank}");
        }
    }

    /// Both ranks' dense cuts TILE every leaf they cut: no byte twice, none
    /// dropped. The row cuts concatenate back; the column cuts interleave.
    #[test]
    fn both_ranks_dense_cuts_tile_every_leaf() {
        let cases: [(&str, Vec<u64>); 6] = [
            ("attn.wq_du.weight", vec![8, 4]),
            ("attn.wo_ud.weight", vec![4, 8]),
            ("mlp.w13_dn.weight", vec![16, 4]),
            ("mlp.w2_md.weight", vec![4, 8]),
            ("mlp.shared_experts.shared_w13_weight", vec![2, 16, 4]),
            ("mlp.shared_experts.shared_w2_weight", vec![2, 4, 8]),
        ];
        for (nm, dims) in cases {
            let n: usize = dims.iter().product::<u64>() as usize;
            let src: Vec<u8> = (0..n * 2).map(|i| (i % 251) as u8).collect();
            let (a, b) = (Tp::new(0, 2).unwrap(), Tp::new(1, 2).unwrap());
            let ca = dense_cut(a, &l(nm), &dims, false).unwrap().unwrap();
            let cb = dense_cut(b, &l(nm), &dims, false).unwrap().unwrap();
            let (pa, pb) = (apply(&ca, &src), apply(&cb, &src));
            assert_eq!(pa.len() + pb.len(), src.len(), "{nm}");
            let mut seen: Vec<u8> = pa.iter().chain(pb.iter()).copied().collect();
            let mut all = src.clone();
            seen.sort();
            all.sort();
            assert_eq!(seen, all, "{nm}: the two shares are not the whole leaf");
        }
    }

    /// What is NOT cut: the MTP heads (same suffixes, a binder that knows no
    /// split), and every replicated leaf. And a known leaf at the wrong rank
    /// is refused rather than left whole.
    #[test]
    fn replicated_leaves_and_the_mtp_heads_are_left_whole_and_wrong_dims_refuse() {
        let t = Tp::new(0, 2).unwrap();
        assert_eq!(
            dense_cut(
                t,
                "model.mtp.layers.0.transformer_block.attn.wq_du.weight",
                &[8, 4],
                false
            )
            .unwrap(),
            None
        );
        for nm in [
            "attn.q_norm.weight",
            "attn.rel_logits_proj.proj",
            "attn_norm.weight",
            "mlp.gate.weight",
            "mlp.sconv.weight",
            "mlp.global_scale",
        ] {
            assert_eq!(dense_cut(t, &l(nm), &[8, 4], false).unwrap(), None, "{nm}");
        }
        assert_eq!(
            dense_cut(t, "model.llm.unembed.weight", &[16, 4], false).unwrap(),
            None
        );
        // A shared w13 that is not rank 3 is not a leaf this knows how to
        // cut, and the binder would trust that it had been.
        assert!(
            dense_cut(
                t,
                &l("mlp.shared_experts.shared_w13_weight"),
                &[32, 4],
                false
            )
            .is_err()
        );
        assert!(dense_cut(t, &l("attn.wo_ud.weight"), &[4, 2, 4], false).is_err());
        // The identity world cuts everything to itself.
        let one = Tp::default();
        let c = dense_cut(one, &l("attn.wq_du.weight"), &[8, 4], false)
            .unwrap()
            .unwrap();
        assert_eq!(c.dims, vec![8, 4]);
        let b = slab(8, 4);
        assert_eq!(apply(&c, &b), b);
    }
}
