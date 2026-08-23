//! Prove a converted pile is equivalent through the PUBLIC loader API.
//!
//! The other two checks work at the blob level: the converter compares payload
//! bytes as it encodes, and `pile_leaf_verify` re-reads both piles cold. Both
//! answer "are the bytes the same". Neither answers the question a model
//! actually asks, which is "does `WeightLoader` hand me the same numbers".
//!
//! That gap is where a conversion would most plausibly go wrong without any of
//! the byte checks noticing — a shape read in the wrong order, a name that
//! resolves to a different leaf, a loader that silently picks the unconverted
//! fallback path. So this compares the two piles through `WeightLoader` itself:
//! same constructor, same `load_f32`, same names, for every tensor in the model.
//!
//! It also asserts the converted pile actually selects the TYPED path. A test
//! that passes because both sides quietly fell back to materializing would be
//! the emptiest kind of green check.
//!
//!   pile_loader_equiv <original.pile> <converted.pile>
//!
//! Materializes the original model, so run it on models that fit in RAM.

use anyhow::{Context, Result};
use mary::nn::weight_loader::WeightLoader;
use std::path::Path;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let src = args
        .next()
        .context("usage: pile_loader_equiv <original.pile> <converted.pile>")?;
    let dst = args
        .next()
        .context("usage: pile_loader_equiv <original.pile> <converted.pile>")?;

    let original = WeightLoader::from_pile(Path::new(&src))?;
    let converted = WeightLoader::from_pile(Path::new(&dst))?;

    // The check that keeps the rest honest.
    anyhow::ensure!(
        matches!(converted, WeightLoader::Typed(_)),
        "the converted pile did NOT select the typed path — this test would \
         otherwise pass by comparing two materialized loaders"
    );
    anyhow::ensure!(
        matches!(original, WeightLoader::Pile(_)),
        "the original pile selected the typed path; it should be unconverted"
    );

    let (WeightLoader::Pile(orig_map), WeightLoader::Typed(_)) = (&original, &converted) else {
        unreachable!("variants checked above")
    };
    println!("original    materialized, {} tensors", orig_map.len());

    let mut names: Vec<&String> = orig_map.keys().collect();
    names.sort();

    let (mut checked, mut elems, mut viewed) = (0usize, 0usize, 0usize);
    for name in names {
        anyhow::ensure!(
            converted.has_weight(name),
            "{name}: present in the original, missing from the converted pile"
        );

        let (a, a_shape) = original.load_f32(name);
        let (b, b_shape) = converted.load_f32(name);
        anyhow::ensure!(
            a_shape == b_shape,
            "{name}: shape differs: {a_shape:?} vs {b_shape:?}"
        );
        anyhow::ensure!(
            a.len() == b.len(),
            "{name}: length differs: {} vs {}",
            a.len(),
            b.len()
        );
        // Exact equality, not a tolerance. The conversion re-frames bytes; it
        // does not compute, so anything but bit-identical is a bug and a
        // tolerance would only hide it.
        if a != b {
            let i = a
                .iter()
                .zip(&b)
                .position(|(x, y)| x != y)
                .expect("lengths equal and slices differ");
            anyhow::bail!("{name}: value differs at index {i}: {} vs {}", a[i], b[i]);
        }

        // The typed side should also serve this without a copy.
        if let Some((v, v_shape)) = converted.view_f32(name) {
            anyhow::ensure!(v_shape == a_shape, "{name}: view shape differs");
            anyhow::ensure!(&v[..] == &a[..], "{name}: view values differ");
            viewed += 1;
        }

        checked += 1;
        elems += a.len();
        if checked % 200 == 0 {
            println!("            {checked} tensors ...");
        }
    }

    println!("verified    {checked} tensors, {elems} elements, exactly equal");
    println!("zero-copy   {viewed} served through view_f32 without materializing");
    Ok(())
}
