//! Import Inkling's packed experts for a RANGE OF LAYERS into a pile.
//!
//! The probe (`inkling_pile_import`) takes one tensor name and a count. This is
//! the deployment shape: a node imports the layers it holds. Splitting a model
//! across machines is a query — `experts_in_layers` — and this is the writing
//! half of that same idea, so the range is the argument rather than a tensor
//! name the caller has to know.
//!
//! It is also what fits. The full NVFP4 checkpoint is ~552 GB, of which the
//! experts are ~500 GB, so "import everything" is not an option on a single
//! machine and a layer range is not a convenience — it is the unit of work.
//!
//! One layer's experts are plain BF16 rather than NVFP4 (the checkpoint has 64
//! `w13_weight` but 63 `.scale`). That layer is REPORTED and skipped here
//! rather than guessed at: the packed path would read sidecars that do not
//! exist, and inventing them is how a model comes back as plausible noise.
//!
//!   inkling_expert_import <ckpt-dir> <pile> --layers A-B [--experts N]

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use mary::models::inkling::load::Checkpoint;
use mary::models::inkling::pile::{attrs, expert_blob, layer_of, split_payload};
use triblespace::core::blob::encodings::tensor::TensorView;
use triblespace::core::blob::TryFromBlob;
use triblespace::core::id::ExclusiveId;
use triblespace::core::metadata;
use triblespace::core::repo::Repository;
use triblespace::macros::entity;
use triblespace::prelude::*;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .context("usage: inkling_expert_import <ckpt-dir> <pile> --layers A-B [--experts N]")?;
    let pile_path = args
        .next()
        .context("usage: inkling_expert_import <ckpt-dir> <pile> --layers A-B [--experts N]")?;
    let mut layers: Option<(i64, i64)> = None;
    let mut experts_cap = usize::MAX;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--layers" => {
                let v = args.next().context("--layers needs A-B")?;
                let (a, b) = v.split_once('-').context("--layers wants A-B")?;
                layers = Some((a.parse()?, b.parse()?));
            }
            "--experts" => experts_cap = args.next().context("--experts needs N")?.parse()?,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let (lo, hi) = layers.context("--layers A-B is required — this imports a node's share")?;

    let ck = Checkpoint::open(&dir).with_context(|| format!("opening {dir}"))?;

    // The stacked expert matrices in range. Sidecars are reached through the
    // index by the reader, so only the weights are enumerated here.
    let mut bases: Vec<String> = ck
        .names()
        .into_iter()
        .filter(|n| n.ends_with(".experts.w13_weight") || n.ends_with(".experts.w2_weight"))
        .filter(|n| matches!(layer_of(n), Some(l) if l >= lo && l <= hi))
        .collect();
    bases.sort();
    anyhow::ensure!(!bases.is_empty(), "no expert matrices in layers {lo}-{hi}");
    println!("checkpoint {dir}");
    println!("layers     {lo}..={hi}");
    println!("matrices   {}", bases.len());

    // Skip-and-report, never skip-silently.
    let (packed, dense): (Vec<&String>, Vec<&String>) =
        bases.iter().partition(|b| ck.is_nvfp4(b));
    if !dense.is_empty() {
        println!(
            "           {} matrix/matrices are BF16, not NVFP4 — SKIPPED (no sidecars): {:?}",
            dense.len(),
            dense
        );
    }
    println!("           {} NVFP4 matrices to import", packed.len());

    let path = std::path::Path::new(&pile_path);
    if !path.exists() {
        println!("creating a new pile at {pile_path}");
        std::fs::File::create(path)?;
    }
    let store = Pile::open(path).map_err(|e| anyhow::anyhow!("open pile: {e:?}"))?;
    let mut repo = Repository::new(
        store,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repository: {e:?}"))?;
    let branch = repo
        .ensure_branch("inkling", None)
        .map_err(|e| anyhow::anyhow!("branch: {e:?}"))?;
    let mut ws = repo
        .pull(branch)
        .map_err(|e| anyhow::anyhow!("pull: {e:?}"))?;
    let mut change = TribleSet::new();

    let (mut n, mut total) = (0usize, 0usize);
    for base in packed {
        let layer = layer_of(base);
        // How many experts this matrix stacks — asked of the checkpoint rather
        // than assumed, so a model with a different expert count imports
        // correctly instead of silently importing a prefix.
        let mut e = 0usize;
        while e < experts_cap {
            let q = match ck.expert_slice_packed(base, e) {
                Ok(q) => q,
                Err(_) => break, // past the last expert in this stack
            };
            let blob = expert_blob(&q).with_context(|| format!("{base}[{e}] to blob"))?;
            let blob2 = expert_blob(&q)?;

            // Same checks the probe makes, kept at scale: the round trip is
            // where a packing bug shows up, and it costs a decode.
            let view: TensorView = TryFromBlob::try_from_blob(blob)
                .map_err(|err| anyhow::anyhow!("{base}[{e}]: decode: {err}"))?;
            anyhow::ensure!(
                view.dims() == [q.rows as u64, (q.cols * 2) as u64],
                "{base}[{e}]: dims {:?}",
                view.dims()
            );
            let (codes, scales, scale2) = split_payload(view.payload(), view.elems())?;
            anyhow::ensure!(codes == &q.codes[..], "{base}[{e}]: codes differ after a round trip");
            anyhow::ensure!(scales == &q.scales[..], "{base}[{e}]: scales differ");
            anyhow::ensure!(scale2 == q.scale2, "{base}[{e}]: global scale differs");

            let bytes = blob2.bytes.len();
            let handle = ws.put(blob2);
            let name_h = ws.put::<blobencodings::LongString, _>(base.to_string());
            let mut facts = entity! { &ufoid() @
                attrs::weight_nvfp4_2: handle,
                attrs::expert_index: e as i64,
                metadata::name: name_h,
            };
            if let Some(l) = layer {
                let root = facts.root().expect("rooted");
                facts += entity! { ExclusiveId::force_ref(&root) @ attrs::layer: l };
            }
            change += facts;

            total += bytes;
            n += 1;
            e += 1;
            if n % 200 == 0 {
                println!(
                    "  {n} experts, {:.1} GiB ...",
                    total as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            }
        }
        println!("  {base}: {e} experts");
    }

    ws.commit(change, &format!("inkling experts, layers {lo}..={hi}"));
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;

    println!(
        "imported   {n} experts, {:.2} GiB, each verified to round-trip",
        total as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("wrote      {pile_path}");
    Ok(())
}
