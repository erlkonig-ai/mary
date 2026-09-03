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

pub use super::version::LearnedExpert;

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
