//! Verify a converted pile against its source, tensor by tensor.
//!
//! `pile_leaf_migrate` checks each payload as it encodes it, which proves the
//! ENCODE step. This proves the round trip: it opens both piles independently,
//! reads the source through the old two-handle path and the destination through
//! the typed path, and compares name sets, shapes, and bytes.
//!
//! Worth doing separately rather than trusting the converter's own report. The
//! converter checks a blob it holds in memory against bytes it just read; this
//! reads both piles cold, through the code paths a real loader uses, and so it
//! also answers the question the converter cannot: is the output actually
//! LOADABLE? A converter that produces unreadable output while reporting
//! success is exactly the failure this crate just spent an afternoon on.
//!
//!   pile_leaf_verify <src.pile> <dst.pile>

use anyhow::{Context, Result};
use mary::format::attrs;
use mary::ingest::LeafHandles;
use mary::leaf;
use std::path::Path;
use triblespace::macros::{find, pattern};
use triblespace::prelude::*;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let src = args.next().context("usage: pile_leaf_verify <src.pile> <dst.pile>")?;
    let dst = args.next().context("usage: pile_leaf_verify <src.pile> <dst.pile>")?;
    let (src, dst) = (Path::new(&src), Path::new(&dst));

    // ── source: the old two-handle form ─────────────────────────────────────
    let (_, src_index, src_reader) = mary::persist::load_split_index_from_pile(src, "")?;
    anyhow::ensure!(!src_index.is_empty(), "no leaves in source {src:?}");
    println!("source      {} leaves", src_index.len());

    // ── destination: the typed form ─────────────────────────────────────────
    let (dst_tribles, dst_reader) = mary::persist::checkout_mary_branch(dst)?;
    let roots: Vec<Id> = find!(
        (m: Id, s: Inline<inlineencodings::Handle<blobencodings::LongString>>),
        pattern!(&dst_tribles, [{ ?m @ attrs::source: ?s, attrs::quantization: "native" }])
    )
    .map(|(m, _)| m)
    .collect();
    anyhow::ensure!(!roots.is_empty(), "no native model root in {dst:?}");

    let mut dst_index = std::collections::HashMap::new();
    for root in &roots {
        dst_index.extend(leaf::index_typed(&dst_tribles, &dst_reader, *root));
    }
    println!("destination {} typed leaves", dst_index.len());

    // ── compare ─────────────────────────────────────────────────────────────
    let mut missing = Vec::new();
    for name in src_index.keys() {
        if !dst_index.contains_key(name) {
            missing.push(name.clone());
        }
    }
    let extra: Vec<&String> = dst_index
        .keys()
        .filter(|n| !src_index.contains_key(*n))
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "{} tensors missing from the converted pile, e.g. {:?}",
        missing.len(),
        &missing[..missing.len().min(5)]
    );
    anyhow::ensure!(extra.is_empty(), "{} unexpected tensors in the converted pile", extra.len());

    let mut names: Vec<&String> = src_index.keys().collect();
    names.sort();
    let (mut checked, mut bytes_checked, mut zero_copy) = (0usize, 0usize, 0usize);

    for name in names {
        let typed = &dst_index[name];

        // Source: raw payload bytes + shape, straight from the old handles.
        let handles = src_index[name];
        let (src_bytes, src_shape) = match handles {
            LeafHandles::F32(dh, sh) => {
                let b: anybytes::Bytes = src_reader
                    .get(dh)
                    .map_err(|e| anyhow::anyhow!("{name}: data: {e:?}"))?;
                (b, mary::ingest::read_shape(&src_reader, sh))
            }
            LeafHandles::F16(dh, sh) => {
                let b: anybytes::Bytes = src_reader
                    .get(dh)
                    .map_err(|e| anyhow::anyhow!("{name}: data_f16: {e:?}"))?;
                (b, mary::ingest::read_shape(&src_reader, sh))
            }
        };

        anyhow::ensure!(
            typed.shape() == src_shape,
            "{name}: shape differs: {:?} (source) vs {:?} (converted)",
            src_shape,
            typed.shape()
        );
        anyhow::ensure!(
            &typed.view.payload()[..] == &src_bytes[..],
            "{name}: payload differs ({} vs {} bytes)",
            src_bytes.len(),
            typed.view.payload().len()
        );

        // And confirm the point of the exercise: an f32 leaf serves a ZERO-COPY
        // view of the pile's mapping, on this platform, without a model-specific
        // feature gate.
        if typed.elem == leaf::Elem::F32 {
            let v = typed
                .view_f32()
                .ok_or_else(|| anyhow::anyhow!("{name}: f32 leaf served no zero-copy view"))?;
            anyhow::ensure!(
                v.len() == src_shape.iter().product::<usize>(),
                "{name}: view length {} != {:?}",
                v.len(),
                src_shape
            );
            zero_copy += 1;
        }

        checked += 1;
        bytes_checked += src_bytes.len();
    }

    println!(
        "verified    {checked} tensors, {:.2} GiB, byte-identical",
        bytes_checked as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("zero-copy   {zero_copy} f32 leaves served a view over the mapping");
    Ok(())
}
