//! Convert Inkling's stacked NVFP4 experts into per-expert tensor blobs.
//!
//! Reads real checkpoint bytes and asserts the conversion against them, because
//! the arithmetic that matters — packed width, block count, where the global
//! scale sits — is exactly the arithmetic a synthetic fixture can agree with
//! while both are wrong.
//!
//! Read-only. It opens the checkpoint, converts, verifies, and reports. Writing
//! a pile is a separate step and deliberately not reachable from here.
//!
//!   inkling_pile_import <checkpoint-dir> [tensor-base] [experts]

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use mary::models::inkling::load::Checkpoint;
use mary::models::inkling::pile::{attrs, expert_blob, experts_in_layers, layer_of, split_payload};
use triblespace::core::blob::encodings::tensor::TensorView;
use triblespace::core::blob::TryFromBlob;
use triblespace::core::metadata;
use triblespace::core::repo::Repository;
use triblespace::macros::entity;
use triblespace::prelude::*;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().context("usage: inkling_pile_import <dir> [base] [experts]")?;
    let base = args
        .next()
        .unwrap_or_else(|| "model.llm.layers.10.mlp.experts.w13_weight".to_string());
    let count: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(3);
    // Writing is opt-in and last: a run with no pile path converts, verifies
    // and reports, touching nothing.
    let pile_path = args.next();

    let ck = Checkpoint::open(&dir).with_context(|| format!("opening {dir}"))?;
    println!("checkpoint {dir}");
    println!("tensor     {base}");

    // Held open across the whole run rather than per expert: opening a pile is
    // O(records), and this writes hundreds of them.
    let mut writing = match &pile_path {
        None => None,
        Some(p) => {
            let path = std::path::Path::new(p);
            if !path.exists() {
                println!("creating a new pile at {p}");
                std::fs::File::create(path)?;
            }
            let store = Pile::open(path).map_err(|e| anyhow::anyhow!("open pile: {e:?}"))?;
            let mut repo =
                Repository::new(store, SigningKey::generate(&mut rand::rngs::OsRng), TribleSet::new())
                    .map_err(|e| anyhow::anyhow!("repository: {e:?}"))?;
            let branch = repo
                .ensure_branch("inkling", None)
                .map_err(|e| anyhow::anyhow!("branch: {e:?}"))?;
            let ws = repo.pull(branch).map_err(|e| anyhow::anyhow!("pull: {e:?}"))?;
            Some((repo, ws, TribleSet::new()))
        }
    };

    let mut total = 0usize;
    for e in 0..count {
        let q = ck
            .expert_slice_packed(&base, e)
            .with_context(|| format!("slicing expert {e}"))?;
        let logical = q.cols * 2;

        let blob = expert_blob(&q).with_context(|| format!("expert {e} to blob"))?;
        let blob2 = expert_blob(&q)?;
        let bytes = blob.bytes.len();
        let handle = blob.get_handle();
        let view: TensorView = blob.try_from_blob().context("decoding the blob just built")?;

        // The claims worth checking against real bytes rather than a fixture.
        anyhow::ensure!(
            view.dims() == [q.rows as u64, logical as u64],
            "expert {e}: dims {:?}, expected [{}, {}]",
            view.dims(),
            q.rows,
            logical
        );
        let (codes, scales, scale2) = split_payload(view.payload(), view.elems())?;
        anyhow::ensure!(codes == &q.codes[..], "expert {e}: codes differ after a round trip");
        anyhow::ensure!(scales == &q.scales[..], "expert {e}: scales differ");
        anyhow::ensure!(scale2 == q.scale2, "expert {e}: global scale differs");

        if let Some((_, ws, change)) = writing.as_mut() {
            // put() returns the handle the facts then name, so the blob and the
            // fact about it cannot refer to different bytes.
            let handle = ws.put(blob2);
            let mut facts = entity! { &ufoid() @
                attrs::weight_nvfp4_2: handle,
                attrs::expert_index: e as i64,
                metadata::name: base.to_string().to_blob().get_handle(),
            };
            // Absent rather than zero when the name carries no layer: a tensor
            // that silently joined layer 0 would ship to the wrong machine.
            if let Some(l) = layer_of(&base) {
                let root = facts.root().expect("rooted");
                facts += entity! { ExclusiveId::force_ref(&root) @ attrs::layer: l };
            }
            *change += facts;
        }

        total += bytes;
        println!(
            "  expert {e:>3}  dims [{}, {}]  blob {:>10} B  scale2 {:+.6}  {:?}",
            q.rows, logical, bytes, scale2, handle
        );
    }

    // Read back what was written, as a QUERY over layers rather than a read of
    // bytes. This is the split: a node asks for the layers it holds and gets
    // handles, and nothing is materialised to answer.
    if let Some((_, ws, _)) = writing.as_mut() {
        let space = ws.checkout(..).map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
        let _ = &space;
    }

    if let Some((mut repo, mut ws, change)) = writing {
        ws.commit(change, "inkling experts");
        repo.push(&mut ws).map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
        repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;
        println!("wrote {count} expert(s) to {}", pile_path.clone().unwrap());

        // Query the pile we just wrote, by layer.
        let space = ws.checkout(..).map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?;
        let held = experts_in_layers(&space, 0..=20);
        let all = experts_in_layers(&space, i64::MIN..=i64::MAX);
        println!(
            "query layers 0..=20 -> {} expert handle(s) of {} total, nothing materialised",
            held.len(),
            all.len()
        );
        for r in all.iter().take(3) {
            println!("  layer {:>3}  expert {:>3}  {:?}", r.layer, r.expert, r.handle);
        }
    }

    println!(
        "{count} expert(s), {total} B total, {} B each on average",
        total / count.max(1)
    );
    Ok(())
}
