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
//! 3. collect an explicitly successful changed-cut response from EVERY rank
//!    over the rank link ([`super::tpcomm::Pass::Export`]),
//! 4. restore absent unchanged cuts from their agreed immutable checkpoint,
//!    then concatenate the cuts back into the whole
//!    ([`assemble_completed_exports`], [`assemble`]).
//!
//! What this module deliberately does NOT do is decide the identity of the
//! model that results. A learned expert is a new leaf (its id derives from its
//! bytes), and a root over the parent's members with those leaves substituted
//! is a new model root. This module returns assembled [`PackedExpert`]s; the
//! engine owns publishing a version. Reconstruction uses the source identity
//! admitted at startup, not a newly discovered root in the mutable pile file.

use anyhow::{Context, Result};
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

/// One explicitly SUCCESSFUL rank response to the same export operation.
///
/// Empty `cuts` means this rank completed its synced comparison and no codes
/// differed from its immutable source. A failed, missing, or unread response
/// must NEVER be represented by an empty envelope. The collector attributes
/// rank/world to the admitted peer socket, not to a cut's self-reported fields.
/// `model_identity` is the identity already agreed for that socket's startup
/// cohort (or explicitly returned by it), never a substitute for agreement.
#[derive(Clone, Debug, PartialEq)]
pub struct RankExport {
    pub rank: u32,
    pub world: u32,
    pub model_identity: [u8; 32],
    pub cuts: Vec<LearnedCut>,
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

/// Assemble the actual live student bank after EVERY rank exported successfully.
///
/// A locally absent expert cut is then known byte-identical to that rank's
/// checkpoint cut. Reconstruct only such missing cuts for the union of changed
/// experts; wholly untouched experts are neither loaded here nor transferred.
/// The collector must hold the synchronous export boundary across all ranks:
/// no learner update may interleave the per-rank exports.
///
/// Before source reads this checks complete unique rank responses, startup
/// identity, and the declared learned layer/expert domain. Every emitted cut
/// is then checked against its checkpoint's exact TP geometry and immutable
/// scales. Generic [`assemble`] retains its strict completeness contract.
pub fn assemble_completed_exports(
    src: &Weights,
    world: usize,
    layer: usize,
    n_routed: usize,
    exports: Vec<RankExport>,
) -> Result<Vec<LearnedExpert>> {
    assemble_completed_with(src.model_identity(), world, layer, n_routed, exports, |name, expert| {
        let stored = src.expert_packed_stored(name, expert)?;
        Ok(PackedExpert {
            codes: stored.codes.to_vec(), scales: stored.scales.to_vec(),
            scale2: stored.scale2, rows: stored.rows, cols: stored.cols,
        })
    })
}

fn assemble_completed_with(
    identity: [u8; 32],
    world: usize,
    layer: usize,
    n_routed: usize,
    exports: Vec<RankExport>,
    mut stored_expert: impl FnMut(&str, usize) -> Result<PackedExpert>,
) -> Result<Vec<LearnedExpert>> {
    use std::collections::BTreeMap;

    let world_u32 = u32::try_from(world).context("export world does not fit its wire field")?;
    let layer_i64 = i64::try_from(layer).context("export layer does not fit its identity field")?;
    anyhow::ensure!(world > 0 && n_routed > 0, "completed export needs a nonempty world and expert domain");
    anyhow::ensure!(exports.len() == world,
        "completed export has {} successful rank responses, expected {world}", exports.len());
    let mut seen = vec![false; world];
    for export in &exports {
        let rank = export.rank as usize;
        anyhow::ensure!(rank < world && export.world == world_u32, "export response rank/world is outside the admitted cohort");
        anyhow::ensure!(!seen[rank], "export received rank {rank} twice");
        anyhow::ensure!(export.model_identity == identity,
            "export rank {rank} did not use the agreed immutable checkpoint");
        seen[rank] = true;
    }
    let w13 = format!("model.llm.layers.{layer}.mlp.experts.w13_weight");
    let w2 = format!("model.llm.layers.{layer}.mlp.experts.w2_weight");
    let mut changed: BTreeMap<(String, i64), BTreeMap<u32, LearnedCut>> = BTreeMap::new();
    for export in exports {
        for cut in export.cuts {
            anyhow::ensure!(cut.rank == export.rank && cut.world == export.world,
                "a learned cut claims a different rank/world than its successful response");
            anyhow::ensure!(cut.layer == layer_i64 && (cut.name == w13 || cut.name == w2),
                "a learned cut does not name the configured layer's routed W13/W2 bank");
            let expert = usize::try_from(cut.expert).context("a learned expert index is negative or too large")?;
            anyhow::ensure!(expert < n_routed, "learned expert {expert} exceeds the configured expert domain");
            let rank = cut.rank;
            let parts = changed.entry((cut.name.clone(), cut.expert)).or_default();
            anyhow::ensure!(parts.insert(rank, cut).is_none(), "a successful export repeats one expert cut on rank {rank}");
        }
    }

    let mut complete = Vec::new();
    for ((name, expert), mut parts) in changed {
        let stored = stored_expert(&name, expert as usize)
            .with_context(|| format!("{name}[{expert}]: read the agreed checkpoint expert"))?;
        let logical = stored.cols.checked_mul(2).context("checkpoint expert width overflow")?;
        let elements = stored.rows.checked_mul(logical).context("checkpoint expert size overflow")?;
        anyhow::ensure!(stored.rows > 0 && logical > 0 && logical % 16 == 0
            && stored.rows <= u32::MAX as usize && logical <= u32::MAX as usize,
            "{name}[{expert}]: invalid checkpoint expert dimensions");
        anyhow::ensure!(stored.codes.len() == elements / 2 && stored.scales.len() == elements / 16,
            "{name}[{expert}]: checkpoint planes do not match their dimensions");
        anyhow::ensure!(stored.scale2.is_finite() && stored.scale2 >= 0.0,
            "{name}[{expert}]: checkpoint global scale is invalid");
        for rank in 0..world {
            let geometry = match world {
                1 => Cut::Rows(0..stored.rows),
                _ => routed_cut(Tp::new(rank, world)?, &name, stored.rows, logical)?,
            };
            let (rows, columns) = geometry.dims(stored.rows, logical);
            // This also checks packed block boundaries on a W2 column cut.
            let scales = cut_plane(&stored.scales, stored.rows, logical, Plane::NVFP4_SCALES, &geometry)?;
            let cut = if let Some(cut) = parts.remove(&(rank as u32)) {
                anyhow::ensure!(cut.cut == geometry && cut.rows as usize == rows && cut.logical as usize == columns,
                    "{name}[{expert}] rank {rank}: exported cut differs from checkpoint TP geometry");
                let cut_elements = rows.checked_mul(columns).context("learned cut size overflow")?;
                anyhow::ensure!(cut.codes.len() == cut_elements / 2 && cut.scales.len() == scales.len(),
                    "{name}[{expert}] rank {rank}: exported planes differ from the admitted cut size");
                anyhow::ensure!(cut.scales.as_slice() == scales.as_ref() && cut.scale2.to_bits() == stored.scale2.to_bits(),
                    "{name}[{expert}] rank {rank}: the learner changed immutable checkpoint scales");
                cut
            } else {
                LearnedCut {
                    name: name.clone(), layer: layer_i64, expert, rank: rank as u32,
                    world: world_u32, cut: geometry.clone(), rows: rows as u32, logical: columns as u32,
                    codes: cut_plane(&stored.codes, stored.rows, logical, Plane::NVFP4_CODES, &geometry)?.into_owned(),
                    scales: scales.into_owned(), scale2: stored.scale2,
                }
            };
            complete.push(cut);
        }
    }
    assemble(complete)
}

/// Every rank's cut of every learned expert, joined back into whole experts.
///
/// Refuses an expert with a missing rank: a flat cut list alone cannot prove
/// that an absent cut was unchanged. [`assemble_completed_exports`] supplies
/// that proof from explicit all-rank success and fills it before calling here.
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

    const CHECKPOINT: [u8; 32] = [0x42; 32];

    fn checkpoint(rows: usize, logical: usize) -> PackedExpert {
        PackedExpert {
            codes: (0..rows * logical / 2).map(|i| (i as u8).wrapping_mul(11)).collect(),
            scales: (0..rows * logical / 16).map(|i| 0x30 + (i % 8) as u8).collect(),
            scale2: 0.37, rows, cols: logical / 2,
        }
    }

    fn response(rank: u32, cuts: Vec<LearnedCut>) -> RankExport {
        RankExport { rank, world: 2, model_identity: CHECKPOINT, cuts }
    }

    fn checkpoint_cut(name: &str, stored: &PackedExpert, rank: u32) -> LearnedCut {
        let logical = stored.cols * 2;
        let geometry = routed_cut(Tp::new(rank as usize, 2).unwrap(), name, stored.rows, logical).unwrap();
        let (rows, columns) = geometry.dims(stored.rows, logical);
        LearnedCut {
            name: name.into(), layer: 41, expert: 7, rank, world: 2,
            rows: rows as u32, logical: columns as u32,
            codes: cut_plane(&stored.codes, stored.rows, logical, Plane::NVFP4_CODES, &geometry).unwrap().into_owned(),
            scales: cut_plane(&stored.scales, stored.rows, logical, Plane::NVFP4_SCALES, &geometry).unwrap().into_owned(),
            scale2: stored.scale2, cut: geometry,
        }
    }

    fn complete(name: &str, stored: &PackedExpert, responses: Vec<RankExport>) -> Result<Vec<LearnedExpert>> {
        assemble_completed_with(CHECKPOINT, 2, 41, 8, responses, |requested, expert| {
            anyhow::ensure!(requested == name && expert == 7, "unexpected checkpoint request");
            Ok(stored.clone())
        })
    }

    #[test]
    fn completed_export_reconstructs_unchanged_w13_row_rank() {
        let name = "model.llm.layers.41.mlp.experts.w13_weight";
        let stored = checkpoint(128, 64);
        let mut changed = checkpoint_cut(name, &stored, 1);
        assert_eq!(changed.cut, Cut::Rows(64..128));
        changed.codes[0] ^= 3;
        let mut expected = stored.clone();
        expected.codes[64 * 32] ^= 3;
        let output = complete(name, &stored, vec![response(1, vec![changed]), response(0, vec![])]).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].name, name);
        assert_eq!(output[0].expert, 7);
        assert_eq!(output[0].packed, expected);
    }

    #[test]
    fn completed_export_reconstructs_unchanged_w2_column_rank() {
        let name = "model.llm.layers.41.mlp.experts.w2_weight";
        let stored = checkpoint(64, 64);
        let mut changed = checkpoint_cut(name, &stored, 0);
        assert_eq!(changed.cut, Cut::Cols(0..32));
        // Byte one of row one in the rank's sixteen-byte code stride.
        changed.codes[17] ^= 0x10;
        let mut expected = stored.clone();
        expected.codes[33] ^= 0x10;
        let output = complete(name, &stored, vec![response(0, vec![changed]), response(1, vec![])]).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].packed, expected);
    }

    #[test]
    fn completed_empty_exports_do_not_load_or_transfer_untouched_experts() {
        let output = assemble_completed_with(CHECKPOINT, 2, 41, 8,
            vec![response(0, vec![]), response(1, vec![])], |_, _| {
                anyhow::bail!("an unchanged expert should not be read")
            }).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn completed_export_requires_every_successful_rank_and_the_same_checkpoint() {
        let mut wrong_world = response(1, vec![]);
        wrong_world.world = 3;
        let mut wrong_identity = response(1, vec![]);
        wrong_identity.model_identity[0] ^= 1;
        for responses in [
            vec![response(0, vec![])],
            vec![response(0, vec![]), response(0, vec![])],
            vec![response(0, vec![]), response(2, vec![])],
            vec![response(0, vec![]), wrong_world],
            vec![response(0, vec![]), wrong_identity],
        ] {
            let reads = std::cell::Cell::new(0);
            let result = assemble_completed_with(CHECKPOINT, 2, 41, 8, responses, |_, _| {
                reads.set(reads.get() + 1);
                Ok(checkpoint(64, 64))
            });
            assert!(result.is_err());
            assert_eq!(reads.get(), 0, "checkpoint was read before all-rank success was established");
        }
    }

    #[test]
    fn completed_export_refuses_forged_identity_geometry_and_scale_changes() {
        let name = "model.llm.layers.41.mlp.experts.w2_weight";
        let stored = checkpoint(64, 64);
        let base = checkpoint_cut(name, &stored, 0);
        let corruptions: [fn(&mut LearnedCut); 12] = [
            |c| c.rank = 1,
            |c| c.world = 3,
            |c| c.layer = 40,
            |c| c.name = "model.llm.layers.40.mlp.experts.w2_weight".into(),
            |c| c.expert = -1,
            |c| c.expert = 8,
            |c| c.cut = Cut::Rows(0..64),
            |c| c.rows = 63,
            |c| { c.codes.pop(); },
            |c| { c.scales.pop(); },
            |c| c.scales[0] ^= 1,
            |c| c.scale2 = f32::NAN,
        ];
        for corrupt in corruptions {
            let mut cut = base.clone();
            corrupt(&mut cut);
            assert!(complete(name, &stored, vec![response(0, vec![cut]), response(1, vec![])]).is_err());
        }
        assert!(complete(name, &stored, vec![response(0, vec![base.clone(), base]), response(1, vec![])]).is_err());
    }

    #[test]
    fn completed_export_preserves_single_rank_whole_row_geometry() {
        let name = "model.llm.layers.41.mlp.experts.w2_weight";
        let stored = checkpoint(64, 64);
        let mut codes = stored.codes.clone();
        codes[9] ^= 1;
        let cut = LearnedCut {
            name: name.into(), layer: 41, expert: 7, rank: 0, world: 1,
            cut: Cut::Rows(0..64), rows: 64, logical: 64,
            codes: codes.clone(), scales: stored.scales.clone(), scale2: stored.scale2,
        };
        let response = RankExport { rank: 0, world: 1, model_identity: CHECKPOINT, cuts: vec![cut] };
        let output = assemble_completed_with(CHECKPOINT, 1, 41, 8, vec![response], |_, _| Ok(stored.clone())).unwrap();
        assert_eq!(output[0].packed.codes, codes);
        assert_eq!(output[0].packed.scales, stored.scales);
    }

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
