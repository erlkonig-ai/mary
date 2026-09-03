//! Learned experts on their way back to the pile.
//!
//! The online learner ([`super::learn`]) moves the NVFP4 codes of the last
//! layer's routed experts IN PLACE, inside the copied arena the serving path
//! reads through. That arena is this process's memory and dies with it, so a
//! resident that learned from a day of turns and restarted would wake as the
//! checkpoint again. What survives has to be written back as what the pile
//! stores: one whole expert per leaf, row-major, `[codes][scales][scale2]`.
//!
//! The arena is not that shape. It holds this RANK's cut of each expert (a
//! tensor-parallel pair splits every routed expert down the intermediate axis,
//! [`super::tpshard::routed_cut`]), permuted into MMA-fragment order if the
//! kernels asked for it ([`super::fp4gemm::swizzle_b_codes`]). So getting an
//! expert out is three inversions and one join:
//!
//! 1. un-permute the cut ([`super::fp4gemm::unswizzle_b_codes_into`]),
//! 2. compare it with the same cut of the pile's expert -- BYTES, not values;
//!    an expert is learned iff its codes moved -- and keep the ones that did,
//! 3. carry every other rank's cut to rank 0 over the rank link
//!    ([`super::tpcomm::Pass::Export`]),
//! 4. concatenate the cuts back into the whole ([`assemble`]).
//!
//! What this module deliberately does NOT do is decide the identity of the
//! model that results. A learned expert is a new leaf (its id derives from its
//! bytes), and a root over the parent's members with those leaves substituted
//! is a new model root; but the loader today resolves experts by NAME and index
//! over the whole collection ([`super::pile::PileSource::open`]), so two roots
//! whose leaves share names would be read as one model with whichever leaf the
//! sweep yielded last. Until the loader selects by root, this module stops at
//! the assembled [`PackedExpert`]s, and nothing here publishes.

use anyhow::{Context, Result};
use triblespace::prelude::Id;
use std::io::Read;

use super::fp4gemm::{unswizzle_b_codes_into, unswizzle_b_scales_into};
use super::load::PackedExpert;
use super::source::Weights;
use super::tp::Tp;
use super::tpshard::{Cut, Plane, cut_plane, routed_cut};

/// One rank's cut of one expert the learner has moved.
///
/// Row-major and un-permuted -- the pile's byte order -- and cut exactly as
/// the arena copy cut the pile's whole expert for this rank, so that `world`
/// of these, one per rank, join back into that whole under [`assemble`].
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedCut {
    /// The stacked matrix's name, e.g. `model.llm.layers.41.mlp.experts.w2_weight`.
    pub name: String,
    pub layer: i64,
    pub expert: i64,
    pub rank: u32,
    pub world: u32,
    /// Which rows or columns of the whole expert this is.
    pub cut: Cut,
    /// Rows and logical columns OF THIS CUT.
    pub rows: u32,
    pub logical: u32,
    /// `[rows, logical / 2]` E2M1 code bytes.
    pub codes: Vec<u8>,
    /// `[rows, logical / 16]` E4M3 block-scale bytes.
    pub scales: Vec<u8>,
    pub scale2: f32,
}

/// One whole expert, every rank's cut joined, as the pile would store it.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedExpert {
    pub name: String,
    pub layer: i64,
    pub expert: i64,
    pub packed: PackedExpert,
}

/// The routed experts of `layer` whose codes differ from the pile's, as this
/// rank's cut of them.
///
/// The caller has synced the device: the arena is written by kernels, and a
/// read that races the last update would export a mixture of two steps.
///
/// Byte comparison against the pile's own cut of the same expert, because that
/// is the question -- "did the learner move this" -- asked of the thing that
/// answers it. No decoding, no tolerance, no reference.
pub fn export_learned(
    src: &Weights,
    tp: Option<Tp>,
    layer: usize,
    n_routed: usize,
) -> Result<Vec<LearnedCut>> {
    let swizzled = src.experts_swizzled();
    let (rank, world) = match tp {
        Some(t) if t.is_split() => (t.rank() as u32, t.world() as u32),
        _ => (0, 1),
    };
    let mut out = Vec::new();
    for suffix in ["w13_weight", "w2_weight"] {
        let name = format!("model.llm.layers.{layer}.mlp.experts.{suffix}");
        for e in 0..n_routed {
            let stored = src
                .expert_packed_stored(&name, e)
                .with_context(|| format!("{name}[{e}]: the pile's expert"))?;
            let (rows, logical) = (stored.rows, stored.cols * 2);
            let cut = match tp {
                Some(t) if t.is_split() => routed_cut(t, &name, rows, logical)?,
                _ => Cut::Rows(0..rows),
            };
            let (cut_rows, cut_logical) = cut.dims(rows, logical);
            let stored_codes = cut_plane(&stored.codes, rows, logical, Plane::NVFP4_CODES, &cut)?;
            let stored_scales =
                cut_plane(&stored.scales, rows, logical, Plane::NVFP4_SCALES, &cut)?;

            let live = src
                .expert_packed(&name, e)
                .with_context(|| format!("{name}[{e}]: the live expert"))?;
            anyhow::ensure!(
                live.rows == cut_rows && live.cols * 2 == cut_logical,
                "{name}[{e}]: the arena holds [{}, {}] where this rank's cut of the pile's \
                 [{rows}, {logical}] is [{cut_rows}, {cut_logical}]",
                live.rows,
                live.cols * 2
            );
            let (codes, scales) = if swizzled {
                let mut codes = vec![0u8; live.codes.len()];
                let mut scales = vec![0u8; live.scales.len()];
                unswizzle_b_codes_into(&live.codes, &mut codes, cut_rows, cut_logical);
                unswizzle_b_scales_into(&live.scales, &mut scales, cut_rows, cut_logical);
                (codes, scales)
            } else {
                (live.codes.to_vec(), live.scales.to_vec())
            };
            // The learner never touches the scales, so a moved scale plane is
            // a layout error, not a learned expert -- and it must not pass as
            // one.
            anyhow::ensure!(
                scales[..] == stored_scales[..] && live.scale2 == stored.scale2,
                "{name}[{e}]: the block scales differ from the pile's, which the learner \
                 does not write: the arena's layout and the export's inverse of it disagree"
            );
            if codes[..] == stored_codes[..] {
                continue;
            }
            out.push(LearnedCut {
                name: name.clone(),
                layer: layer as i64,
                expert: e as i64,
                rank,
                world,
                cut: cut.clone(),
                rows: cut_rows as u32,
                logical: cut_logical as u32,
                codes,
                scales,
                scale2: live.scale2,
            });
        }
    }
    Ok(out)
}

/// Every rank's cut of every learned expert, joined back into whole experts.
///
/// Refuses an expert some rank did not export: a whole expert with one
/// rank's half from the checkpoint and the other's from today is a model
/// nobody ran.
pub fn assemble(cuts: Vec<LearnedCut>) -> Result<Vec<LearnedExpert>> {
    use std::collections::BTreeMap;

    let mut by: BTreeMap<(String, i64), Vec<LearnedCut>> = BTreeMap::new();
    for cut in cuts {
        by.entry((cut.name.clone(), cut.expert))
            .or_default()
            .push(cut);
    }
    let mut out = Vec::with_capacity(by.len());
    for ((name, expert), mut parts) in by {
        let world = parts[0].world as usize;
        anyhow::ensure!(
            parts.iter().all(|p| p.world as usize == world),
            "{name}[{expert}]: the ranks disagree about the world size"
        );
        anyhow::ensure!(
            parts.len() == world,
            "{name}[{expert}]: {} of {world} ranks exported their cut",
            parts.len()
        );
        parts.sort_by_key(|p| p.rank);
        for (i, p) in parts.iter().enumerate() {
            anyhow::ensure!(
                p.rank as usize == i,
                "{name}[{expert}]: rank {} exported twice or rank {i} not at all",
                p.rank
            );
        }
        let layer = parts[0].layer;
        let scale2 = parts[0].scale2;
        anyhow::ensure!(
            parts.iter().all(|p| p.layer == layer && p.scale2 == scale2),
            "{name}[{expert}]: the ranks disagree about the layer or the global scale"
        );
        let packed = match &parts[0].cut {
            Cut::Rows(_) => {
                let logical = parts[0].logical as usize;
                let mut codes = Vec::new();
                let mut scales = Vec::new();
                let mut next = 0usize;
                for p in &parts {
                    let Cut::Rows(r) = &p.cut else {
                        anyhow::bail!("{name}[{expert}]: rank {} cut columns where rank 0 cut rows", p.rank)
                    };
                    anyhow::ensure!(
                        r.start == next && r.len() == p.rows as usize && p.logical as usize == logical,
                        "{name}[{expert}]: rank {}'s rows {}..{} do not continue at {next}",
                        p.rank,
                        r.start,
                        r.end
                    );
                    codes.extend_from_slice(&p.codes);
                    scales.extend_from_slice(&p.scales);
                    next = r.end;
                }
                PackedExpert {
                    codes,
                    scales,
                    scale2,
                    rows: next,
                    cols: logical / 2,
                }
            }
            Cut::Cols(_) => {
                let rows = parts[0].rows as usize;
                let mut next = 0usize;
                for p in &parts {
                    let Cut::Cols(c) = &p.cut else {
                        anyhow::bail!("{name}[{expert}]: rank {} cut rows where rank 0 cut columns", p.rank)
                    };
                    anyhow::ensure!(
                        c.start == next && c.len() == p.logical as usize && p.rows as usize == rows,
                        "{name}[{expert}]: rank {}'s columns {}..{} do not continue at {next}",
                        p.rank,
                        c.start,
                        c.end
                    );
                    next = c.end;
                }
                let logical = next;
                let mut codes = Vec::with_capacity(rows * logical / 2);
                let mut scales = Vec::with_capacity(rows * logical / 16);
                for r in 0..rows {
                    for p in &parts {
                        let cs = p.logical as usize / 2;
                        let ss = p.logical as usize / 16;
                        codes.extend_from_slice(&p.codes[r * cs..(r + 1) * cs]);
                        scales.extend_from_slice(&p.scales[r * ss..(r + 1) * ss]);
                    }
                }
                PackedExpert {
                    codes,
                    scales,
                    scale2,
                    rows,
                    cols: logical / 2,
                }
            }
        };
        out.push(LearnedExpert {
            name,
            layer,
            expert,
            packed,
        });
    }
    Ok(out)
}

// ── the wire form, for the rank link ────────────────────────────────────────
//
// Length-prefixed and fixed-width, like the pass command: a short read is an
// error, never a resynchronisation. Big-endian throughout, matching
// `Pass::encode`.

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    out.extend_from_slice(bytes);
}

fn take<const N: usize>(reader: &mut impl Read, what: &str) -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("read a learned cut's {what}"))?;
    Ok(buf)
}

fn take_u32(reader: &mut impl Read, what: &str) -> Result<u32> {
    Ok(u32::from_be_bytes(take::<4>(reader, what)?))
}

fn take_bytes(reader: &mut impl Read, what: &str) -> Result<Vec<u8>> {
    let len = take_u32(reader, what)? as usize;
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("read a learned cut's {what} ({len} bytes)"))?;
    Ok(buf)
}

impl LearnedCut {
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        put_bytes(out, self.name.as_bytes());
        out.extend_from_slice(&self.layer.to_be_bytes());
        out.extend_from_slice(&self.expert.to_be_bytes());
        out.extend_from_slice(&self.rank.to_be_bytes());
        out.extend_from_slice(&self.world.to_be_bytes());
        let (kind, range) = match &self.cut {
            Cut::Rows(r) => (0u8, r),
            Cut::Cols(c) => (1u8, c),
        };
        out.push(kind);
        out.extend_from_slice(&(range.start as u32).to_be_bytes());
        out.extend_from_slice(&(range.end as u32).to_be_bytes());
        out.extend_from_slice(&self.rows.to_be_bytes());
        out.extend_from_slice(&self.logical.to_be_bytes());
        put_bytes(out, &self.codes);
        put_bytes(out, &self.scales);
        out.extend_from_slice(&self.scale2.to_be_bytes());
    }

    pub fn decode(reader: &mut impl Read) -> Result<Self> {
        let name = String::from_utf8(take_bytes(reader, "name")?)
            .context("a learned cut's name is not UTF-8")?;
        let layer = i64::from_be_bytes(take::<8>(reader, "layer")?);
        let expert = i64::from_be_bytes(take::<8>(reader, "expert")?);
        let rank = take_u32(reader, "rank")?;
        let world = take_u32(reader, "world")?;
        let kind = take::<1>(reader, "cut kind")?[0];
        let start = take_u32(reader, "cut start")? as usize;
        let end = take_u32(reader, "cut end")? as usize;
        let cut = match kind {
            0 => Cut::Rows(start..end),
            1 => Cut::Cols(start..end),
            other => anyhow::bail!("a learned cut's kind byte is {other:#04x}, not rows or columns"),
        };
        let rows = take_u32(reader, "rows")?;
        let logical = take_u32(reader, "logical columns")?;
        let codes = take_bytes(reader, "codes")?;
        let scales = take_bytes(reader, "scales")?;
        let scale2 = f32::from_be_bytes(take::<4>(reader, "global scale")?);
        anyhow::ensure!(
            codes.len() == rows as usize * logical as usize / 2
                && scales.len() == rows as usize * logical as usize / 16,
            "{name}[{expert}] rank {rank}: {} code bytes and {} scale bytes do not fit \
             [{rows}, {logical}]",
            codes.len(),
            scales.len()
        );
        Ok(LearnedCut {
            name,
            layer,
            expert,
            rank,
            world,
            cut,
            rows,
            logical,
            codes,
            scales,
            scale2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(rank: u32, cut: Cut, rows: usize, logical: usize, seed: u8) -> LearnedCut {
        let codes: Vec<u8> = (0..rows * logical / 2).map(|i| (i as u8) ^ seed).collect();
        let scales: Vec<u8> = (0..rows * logical / 16).map(|i| (i as u8).wrapping_mul(seed)).collect();
        LearnedCut {
            name: "model.llm.layers.41.mlp.experts.x".into(),
            layer: 41,
            expert: 7,
            rank,
            world: 2,
            cut,
            rows: rows as u32,
            logical: logical as u32,
            codes,
            scales,
            scale2: 0.5,
        }
    }

    #[test]
    fn the_wire_form_round_trips() {
        let c = cut(1, Cut::Cols(32..64), 4, 32, 0x5a);
        let mut wire = Vec::new();
        c.encode_into(&mut wire);
        let back = LearnedCut::decode(&mut wire.as_slice()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn row_cuts_join_by_rank_order() {
        // rows 0..2 on rank 0, 2..4 on rank 1, delivered in the wrong order.
        let r1 = cut(1, Cut::Rows(2..4), 2, 32, 2);
        let r0 = cut(0, Cut::Rows(0..2), 2, 32, 1);
        let whole = assemble(vec![r1.clone(), r0.clone()]).unwrap();
        assert_eq!(whole.len(), 1);
        let p = &whole[0].packed;
        assert_eq!((p.rows, p.cols), (4, 16));
        assert_eq!(&p.codes[..32], &r0.codes[..]);
        assert_eq!(&p.codes[32..], &r1.codes[..]);
        assert_eq!(&p.scales[..4], &r0.scales[..]);
        assert_eq!(&p.scales[4..], &r1.scales[..]);
    }

    #[test]
    fn column_cuts_interleave_per_row() {
        // 2 rows; columns 0..32 on rank 0, 32..64 on rank 1.
        let r0 = cut(0, Cut::Cols(0..32), 2, 32, 3);
        let r1 = cut(1, Cut::Cols(32..64), 2, 32, 4);
        let whole = assemble(vec![r0.clone(), r1.clone()]).unwrap();
        let p = &whole[0].packed;
        assert_eq!((p.rows, p.cols), (2, 32));
        // row 0: rank 0's 16 code bytes then rank 1's; row 1 likewise.
        assert_eq!(&p.codes[..16], &r0.codes[..16]);
        assert_eq!(&p.codes[16..32], &r1.codes[..16]);
        assert_eq!(&p.codes[32..48], &r0.codes[16..]);
        assert_eq!(&p.codes[48..], &r1.codes[16..]);
        assert_eq!(&p.scales[..2], &r0.scales[..2]);
        assert_eq!(&p.scales[2..4], &r1.scales[..2]);
    }

    #[test]
    fn a_missing_rank_is_refused() {
        let r0 = cut(0, Cut::Rows(0..2), 2, 32, 1);
        let err = assemble(vec![r0]).unwrap_err().to_string();
        assert!(err.contains("1 of 2 ranks"), "{err}");
    }
}

// ── the learned VERSION: a root in the model graph, with a parent ──────────
//
// A learned model is a new ROOT in the same collection as the model it grew
// from: the parent's `member` edges, except that the experts that moved are
// new leaves, plus a `parent` edge to the root it was learned from and the
// recipe that took it there. Nothing about the parent changes, and the leaves
// that did not move are the same entities, so a version costs only the
// experts it moved. Two versions with one parent are a branch. The versions no
// other version names as `parent` are the heads, and the loader takes the
// one head as the model when nothing names a root. A root's id is its member
// set, so a version that learned nothing IS its parent.
//
// The checkpoint was imported in pieces (41 partial roots; experts that belong
// to no root), so the first version's parent is minted here once: the GENESIS
// root, whose members are every leaf the loader takes from the whole
// collection. It is intrinsic, so a second run mints the same id.
//
// JP, 2026-09-03 17:35Z: not a collection per snapshot -- a DAG in one graph,
// so a model can branch off a model, and the experiment's metadata and its
// explanation sit on the graph where a later analysis can query them.

pub mod attrs {
    use triblespace::prelude::blobencodings::UTF8String;
    use triblespace::prelude::inlineencodings::{F64, GenId, Handle, ShortString, U256BE};
    use triblespace::prelude::*;

    attributes! {
        /// The root this version was learned from. Minted 2026-09-03.
        "914320431BD23350DEE18D3D54FA84F1" as parent: GenId;
        /// The learner's step size on the routed experts' codes.
        "EA48C51D05FC180A5855106F0C0CCCAE" as learn_lr: F64;
        /// How hard her own rows were held to the distribution that said
        /// them (the anchor's weight). Absent when unanchored.
        "8BF7CAB60E71E7186955ACDAF6EF3867" as learn_anchor: F64;
        /// Where the stochastic rounding's step counter started.
        "E8C7ED409F235DA6A787B0B094269DD8" as learn_seed: U256BE;
        /// Scored passes learned from, parent to this version.
        "361889279FA26F7AD22187DF37DC7EC7" as learn_steps: U256BE;
        /// What was learned from, in words: a corpus and its lines, or a
        /// span of the archive.
        "9F65DE719752C3AB83F709B55567B247" as learned_span: Handle<UTF8String>;
        /// Why this version exists, in its author's words.
        "365C41C24AB9FF7DF2C78105DB1FDE97" as explanation: Handle<UTF8String>;
        /// The code that learned it: a git revision.
        "6CEEC82A5A1A1EC65A4609C025E42767" as code_revision: ShortString;
    }
}

/// How a version was learned. Facts on the version root, for analysis later.
#[derive(Clone, Debug, Default)]
pub struct VersionRecipe {
    pub lr: f64,
    pub anchor: Option<f64>,
    pub seed: u64,
    pub steps: u64,
    pub span: String,
    pub explanation: String,
    pub code_revision: String,
}

/// What a persisted version is, for the record that says it happened.
#[derive(Clone, Debug)]
pub struct Persisted {
    /// The version root, committed -- or equal to `parent` when nothing had
    /// moved and nothing was written.
    pub root: Id,
    pub parent: Id,
    pub name: String,
    /// Experts whose bytes moved.
    pub replaced: usize,
    /// Whether the parent was minted as the genesis root in the same commit.
    pub genesis: bool,
}

/// A version assembled from learned experts, before or after it is committed.
pub struct LearnedVersion {
    /// The version root.
    pub root: Id,
    /// The root it was learned from.
    pub parent: Id,
    /// Whether `parent` is the genesis root, minted by this assembly.
    pub genesis: bool,
    /// A label: the model's name and the moment.
    pub name: String,
    /// The facts to ADD: new leaves, the version root (and the genesis root
    /// when minted), their annotations. Never the parent's facts, which the
    /// collection already holds.
    pub facts: triblespace::core::trible::TribleSet,
    /// Leaves whose bytes moved and were replaced by new leaf entities.
    pub replaced: usize,
    /// Members of the version root.
    pub members: usize,
}

/// The heads of the version DAG in `facts`: every root that carries `parent`
/// or is named as one, minus those some version names as its parent. Empty
/// when no version exists yet.
pub fn version_heads(facts: &triblespace::core::trible::TribleSet) -> Vec<Id> {
    use std::collections::BTreeSet;
    use triblespace::macros::{find, pattern};
    let mut nodes: BTreeSet<Id> = BTreeSet::new();
    let mut parents: BTreeSet<Id> = BTreeSet::new();
    for (v, p) in find!((v: Id, p: Id), pattern!(facts, [{ ?v @ attrs::parent: ?p }])) {
        nodes.insert(v);
        nodes.insert(p);
        parents.insert(p);
    }
    nodes.difference(&parents).copied().collect()
}

/// Every leaf the loader takes from the whole collection: the packed experts,
/// and the dense tensors of every element type and rank it sweeps.
fn all_leaves(facts: &triblespace::core::trible::TribleSet) -> std::collections::BTreeSet<Id> {
    use super::pile::attrs as ink;
    use triblespace::core::blob::encodings::tensor::elements::{BF16, F32};
    use triblespace::core::metadata;
    use triblespace::macros::{find, pattern};
    let mut out = std::collections::BTreeSet::new();
    for (e,) in find!(
        (e: Id),
        pattern!(facts, [{ ?e @ metadata::name: _?n, ink::expert_index: _?i, ink::weight_nvfp4_2: _?h }])
    ) {
        out.insert(e);
    }
    macro_rules! dense {
        ($ty:ty, $rank:literal) => {
            for (e,) in find!(
                (e: Id),
                pattern!(facts, [{ ?e @ metadata::name: _?n, ink::weight::<$ty, $rank>(): _?h }])
            ) {
                out.insert(e);
            }
        };
    }
    dense!(BF16, 0);
    dense!(BF16, 1);
    dense!(BF16, 2);
    dense!(BF16, 3);
    dense!(BF16, 4);
    dense!(F32, 0);
    dense!(F32, 1);
    dense!(F32, 2);
    dense!(F32, 3);
    out
}

/// The moment, as `YYYYMMDDTHHMMSSZ`, from the system clock with no crate.
pub fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, m, s) = (rem / 3_600, rem % 3_600 / 60, rem % 60);
    // Civil date from days since the epoch (Hinnant).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(mo <= 2);
    format!("{y:04}{mo:02}{d:02}T{h:02}{m:02}{s:02}Z")
}

/// Assemble a version from the learned experts, as a child of `parent` -- or
/// of the graph's one head when `parent` is `None`, or of the genesis root
/// minted here when the graph has no version yet. Writes the new leaves'
/// blobs into the pile (content addressed, so a repeat is a no-op) and
/// nothing else: no record is committed here.
pub fn learned_version(
    pile: &mut triblespace::prelude::Pile,
    learned: &[LearnedExpert],
    parent: Option<Id>,
    recipe: &VersionRecipe,
) -> Result<LearnedVersion> {
    use super::pile::attrs as ink;
    use crate::format::attrs as model;
    use std::collections::{BTreeSet, HashMap};
    use triblespace::core::inline::encodings::hash::Handle;
    use triblespace::core::metadata;
    use triblespace::core::trible::TribleSet;
    use triblespace::macros::{entity, find, pattern};
    use triblespace::prelude::*;

    let graph = crate::model_collection::mary_model_graph_name();
    let snapshot = crate::model_collection::snapshot_model_collection_named_local_latest(pile, graph)
        .with_context(|| format!("the model collection '{graph}'"))?;
    let facts = crate::model_collection::project_legacy_model_attributes(snapshot.facts()).facts;
    let (_, _, reader) = snapshot.into_parts();
    let mut added = TribleSet::new();

    // The model's name, functional per root: the one name the checkpoint's
    // roots agree on, carried onto every version.
    let names: BTreeSet<Inline<Handle<blobencodings::UTF8String>>> = find!(
        (mn: Inline<Handle<blobencodings::UTF8String>>),
        pattern!(&facts, [{ _?r @ model::model_name: ?mn }])
    )
    .map(|(mn,)| mn)
    .collect();
    let model_name = match names.len() {
        1 => names.into_iter().next(),
        _ => None,
    };
    let label_of = |what: &str| -> String {
        match &model_name {
            Some(h) => {
                let name: Result<anybytes::View<str>, _> = reader.get(*h);
                name.map(|s| format!("{} {what}", &*s))
                    .unwrap_or_else(|_| what.to_string())
            }
            None => what.to_string(),
        }
    };

    // 1. The parent, and its members.
    let heads = version_heads(&facts);
    let (parent, genesis, parent_members): (Id, bool, Vec<Id>) = match parent {
        Some(p) => (p, false, Vec::new()),
        None if heads.len() == 1 => (heads[0], false, Vec::new()),
        None if heads.len() > 1 => anyhow::bail!(
            "the model graph has {} heads ({}); name the parent",
            heads.len(),
            heads.iter().map(|i| format!("{i:X}")).collect::<Vec<_>>().join(", ")
        ),
        None => {
            let leaves = all_leaves(&facts);
            anyhow::ensure!(!leaves.is_empty(), "'{graph}' holds no tensor leaves");
            let root = entity! { _ @ model::member*: leaves.iter() };
            let id = root.root().expect("a root has a root");
            added += root;
            let name_h = pile
                .put::<blobencodings::UTF8String, _>(label_of("checkpoint"))
                .map_err(|e| anyhow::anyhow!("store the genesis label: {e:?}"))?;
            added += entity! { ExclusiveId::force_ref(&id) @ metadata::name: name_h };
            if let Some(mn) = model_name {
                added += entity! { ExclusiveId::force_ref(&id) @ model::model_name: mn };
            }
            (id, true, leaves.into_iter().collect())
        }
    };
    let parent_members: Vec<Id> = match parent_members.is_empty() {
        false => parent_members,
        true => find!((m: Id), pattern!(&facts, [{ (parent) @ model::member: ?m }]))
            .map(|(m,)| m)
            .collect(),
    };
    anyhow::ensure!(!parent_members.is_empty(), "parent {parent:X} has no members");
    let parent_set: BTreeSet<Id> = parent_members.iter().copied().collect();

    // 2. New leaves, and the parent's leaves they replace.
    let mut subst: HashMap<Id, Id> = HashMap::new();
    for x in learned {
        let blob = super::pile::expert_blob(&x.packed)
            .with_context(|| format!("{}[{}] as a pile leaf", x.name, x.expert))?;
        let handle = pile
            .put(blob)
            .map_err(|e| anyhow::anyhow!("store {}[{}]: {e:?}", x.name, x.expert))?;
        let name_h = pile
            .put::<blobencodings::UTF8String, _>(x.name.clone())
            .map_err(|e| anyhow::anyhow!("store the name of {}: {e:?}", x.name))?;
        let old: Vec<Id> = find!(
            (e: Id),
            pattern!(&facts, [{ ?e @ metadata::name: (name_h), ink::expert_index: (x.expert), ink::layer: (x.layer) }])
        )
        .map(|(e,)| e)
        .filter(|e| parent_set.contains(e))
        .collect();
        anyhow::ensure!(
            old.len() == 1,
            "{}[{}] (layer {}) names {} leaves among the parent's members, not one",
            x.name,
            x.expert,
            x.layer,
            old.len()
        );
        let leaf = entity! { _ @
            ink::weight_nvfp4_2: handle,
            ink::expert_index: x.expert,
            metadata::name: name_h,
            ink::layer: x.layer,
        };
        let new_id = leaf.root().expect("a leaf has a root");
        added += leaf;
        subst.insert(old[0], new_id);
    }

    // 3. The version root: the parent's members with the moved leaves
    //    replaced. Its id is that set, so a version that moved nothing is
    //    its parent and is not annotated as a child of itself.
    let members: Vec<Id> = parent_members
        .iter()
        .map(|m| subst.get(m).copied().unwrap_or(*m))
        .collect();
    let root_e = entity! { _ @ model::member*: members.iter() };
    let root = root_e.root().expect("a root has a root");
    let name = label_of(&format!("learned {}", utc_stamp()));
    if root != parent {
        added += root_e;
        let name_h = pile
            .put::<blobencodings::UTF8String, _>(name.clone())
            .map_err(|e| anyhow::anyhow!("store the version label: {e:?}"))?;
        let span_h = pile
            .put::<blobencodings::UTF8String, _>(recipe.span.clone())
            .map_err(|e| anyhow::anyhow!("store the learned span: {e:?}"))?;
        let why_h = pile
            .put::<blobencodings::UTF8String, _>(recipe.explanation.clone())
            .map_err(|e| anyhow::anyhow!("store the explanation: {e:?}"))?;
        added += entity! { ExclusiveId::force_ref(&root) @
            attrs::parent: parent,
            metadata::name: name_h,
            attrs::learn_lr: recipe.lr,
            attrs::learn_seed: recipe.seed,
            attrs::learn_steps: recipe.steps,
            attrs::learned_span: span_h,
            attrs::explanation: why_h,
            attrs::code_revision: recipe.code_revision.as_str(),
        };
        if let Some(w) = recipe.anchor {
            added += entity! { ExclusiveId::force_ref(&root) @ attrs::learn_anchor: w };
        }
        if let Some(mn) = model_name {
            added += entity! { ExclusiveId::force_ref(&root) @ model::model_name: mn };
        }
    }
    Ok(LearnedVersion {
        root,
        parent,
        genesis,
        name,
        facts: added,
        replaced: subst.len(),
        members: members.len(),
    })
}

/// Commit the version's facts into the model graph, signed with `key`. The
/// collection's WRITE policy decides whether that signature counts.
pub fn publish_version(
    pile: &mut triblespace::prelude::Pile,
    key: &ed25519_dalek::SigningKey,
    version: LearnedVersion,
) -> Result<()> {
    use triblespace::prelude::*;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("refresh before publishing '{}': {e:?}", version.name))?;
    let graph = crate::model_collection::mary_model_graph_name();
    let collection = crate::model_collection::collection_or_create(pile, key, graph)?;
    let fragment: Fragment = version.facts.into();
    pile.commit(collection, key, fragment)
        .map_err(|e| anyhow::anyhow!("publish '{}': {e}", version.name))?;
    Ok(())
}
