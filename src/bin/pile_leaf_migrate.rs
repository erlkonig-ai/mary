//! Convert a model pile's leaves from `{data|data_f16, shape}` to typed tensors.
//!
//! Pile to pile, into a NEW file. The source is only ever read.
//!
//! # Substitution, not reconstruction
//!
//! The converter does not rebuild the model graph. It COPIES the pile and
//! substitutes one representation for another:
//!
//!   - every fact is carried over verbatim except the three being replaced
//!     (`data`, `data_f16`, `shape`);
//!   - each leaf keeps its OWN entity id and gains one typed leaf fact;
//!   - every blob is copied except the superseded data and shape blobs.
//!
//! Substitution applies ONLY to the leaves actually converted. Everything else
//! — including leaves in encodings this tool has no path for, such as the q4/q8
//! quantized ones — passes through whole, `shape` included.
//!
//! This matters more than it sounds. A converter that rebuilds only what it
//! understands silently drops what it does not — and these piles carry more
//! than weights. `nomic_text` holds a 30,000-piece tokenizer graph hanging off
//! its own root; the rebuild-the-members version of this tool would have
//! reported a confident "112 leaves converted" and produced a pile with no
//! tokenizer in it. Copy-and-substitute cannot lose data it has no schema for,
//! which is exactly the property worth having when the pile knows things the
//! tool does not.
//!
//! Preserving ids also means nothing dangles and nothing downstream has to be
//! told a new address: module edges, model roots, `member` lists and tokenizer
//! references all still resolve. The weights are bit-identical, so the model an
//! id names is the same model; only how its bytes are framed has changed.
//!
//! It does mean a converted pile and a FRESH import of the same checkpoint hold
//! different ids for the same weights, because a leaf id is the content address
//! of its facts and the two forms are different facts. That is content
//! addressing behaving correctly rather than a defect, and preservation is the
//! right side to take: an id already written down in another pile, a commit, or
//! a note has to keep resolving, whereas an id nobody has yet is free. Two
//! piles converge again once both are on the typed form and reimported.
//!
//! # What is verified
//!
//! Byte-identity is established twice, and neither half is assumed: each
//! payload is decoded back out of the blob just built and compared to the
//! source bytes (here), and the handle IS the hash of those bytes, which the
//! store checks on every read. `pile_leaf_verify` then re-checks the finished
//! pile cold, through the code path a real loader uses.
//!
//! The conversion never materialises a tensor — no `view::<[f32]>()`, no `Vec`,
//! no cast. The source blob's bytes go into the tensor payload as they are.
//!
//! Writing goes through `mary::leaf::put_leaf_as`, the same writer every
//! importer uses — so the rank dispatch, the length check and the read-back
//! comparison are shared with the write path rather than restated here, and a
//! converted leaf is byte-for-byte what a fresh import would have written.
//!
//!   pile_leaf_migrate <src.pile> <dst.pile>

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use mary::format::attrs;
use mary::leaf;
use std::collections::HashSet;
use std::path::Path;
use triblespace::core::blob::encodings::UnknownBlob;
use triblespace::core::blob::Blob;
use triblespace::core::id::ExclusiveId;
use triblespace::core::repo::{ancestors, BlobStoreList, Repository};
use triblespace::macros::{find, pattern};
use triblespace::prelude::*;

/// Read a pile's facts from whichever branch holds them.
///
/// Two layouts exist in the wild: the current `mary` branch and the older
/// `main`. Detected, not assumed — and the branch NAME is carried out so the
/// destination lands on the same one.
fn checkout_any(
    pile_path: &Path,
) -> Result<(
    &'static str,
    TribleSet,
    triblespace::core::repo::pile::PileReader,
)> {
    for branch in ["mary", "main"] {
        let mut pile =
            Pile::open(pile_path).map_err(|e| anyhow::anyhow!("open {pile_path:?}: {e:?}"))?;
        pile.refresh().map_err(|e| {
            anyhow::anyhow!("{pile_path:?} failed to load ({e:?}); refusing to auto-truncate")
        })?;
        let mut repo = Repository::new(
            pile,
            SigningKey::generate(&mut rand::rngs::OsRng),
            TribleSet::new(),
        )
        .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
        let found = repo
            .lookup_branch(branch)
            .map_err(|e| anyhow::anyhow!("lookup {branch}: {e:?}"))?;
        let Some(branch_id) = found else {
            repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;
            continue;
        };
        let mut ws = repo
            .pull(branch_id)
            .map_err(|e| anyhow::anyhow!("pull {branch}: {e:?}"))?;
        let Some(head) = ws.head() else {
            repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;
            continue;
        };
        let tribles: TribleSet = ws
            .checkout(ancestors(head))
            .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
            .facts()
            .clone();
        let reader = repo
            .storage_mut()
            .reader()
            .map_err(|e| anyhow::anyhow!("reader: {e:?}"))?;
        repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;
        return Ok((branch, tribles, reader));
    }
    anyhow::bail!("no 'mary' or 'main' branch with commits in {pile_path:?}")
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let src = args
        .next()
        .context("usage: pile_leaf_migrate <src.pile> <dst.pile>")?;
    let dst = args
        .next()
        .context("usage: pile_leaf_migrate <src.pile> <dst.pile>")?;
    let (src, dst) = (Path::new(&src), Path::new(&dst));
    anyhow::ensure!(
        src.canonicalize()? != dst.canonicalize().unwrap_or_else(|_| dst.to_path_buf()),
        "src and dst are the same pile file — refusing to write into the source"
    );

    let (branch, tribles, reader) = checkout_any(src)?;
    eprintln!(
        "[migrate] {src:?}: branch '{branch}', {} facts",
        tribles.len()
    );

    // Every leaf, by its own entity id. `data` XOR `data_f16`, plus `shape`.
    let mut leaves: Vec<(Id, bool)> = Vec::new();
    for (e,) in find!(
        (e: Id),
        pattern!(&tribles, [{ ?e @ attrs::data: _?d, attrs::shape: _?s }])
    ) {
        leaves.push((e, false));
    }
    for (e,) in find!(
        (e: Id),
        pattern!(&tribles, [{ ?e @ attrs::data_f16: _?d, attrs::shape: _?s }])
    ) {
        leaves.push((e, true));
    }
    anyhow::ensure!(!leaves.is_empty(), "no tensor leaves in {src:?}");
    leaves.sort_by_key(|(e, _)| format!("{e}"));
    eprintln!("[migrate] {} leaves", leaves.len());

    if !dst.exists() {
        std::fs::File::create(dst).map_err(|e| anyhow::anyhow!("create {dst:?}: {e}"))?;
    }
    let mut pile = Pile::open(dst).map_err(|e| anyhow::anyhow!("open {dst:?}: {e:?}"))?;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("load {dst:?}: {e:?}; refusing to auto-truncate"))?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = match repo
        .lookup_branch(branch)
        .map_err(|e| anyhow::anyhow!("lookup {branch}: {e:?}"))?
    {
        Some(id) => id,
        None => *repo
            .create_branch(branch, None)
            .map_err(|e| anyhow::anyhow!("create {branch}: {e:?}"))?,
    };
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull {branch}: {e:?}"))?;

    // ── the substitution ────────────────────────────────────────────────────
    // Carry every fact EXCEPT the three being replaced ON THE LEAVES ACTUALLY
    // CONVERTED. The entity check is not a refinement, it is the difference
    // between substitution and quiet mutilation: a quantized leaf carries
    // `data_q4`/`data_q8` + `q_scales` + `shape`, so it is NOT in `leaves`
    // (no `data`/`data_f16`) and gets no typed leaf — but a blanket filter on
    // the attribute would still strip its `shape`, leaving a leaf whose
    // dimensions are simply gone.
    //
    // personaplex_q4.pile: 195 shape facts, 66 convertible leaves, 129 q4. A
    // blanket filter converts 66 and silently unshapes 129, and reports
    // success. Found by auditing this tool against the very failure it was
    // written to avoid — the docs above claimed copy-and-substitute "cannot
    // lose data it has no schema for", and for `shape` that was false.
    let converted: HashSet<Id> = leaves.iter().map(|(e, _)| *e).collect();
    let replaced = [attrs::data.id(), attrs::data_f16.id(), attrs::shape.id()];
    let mut facts: TribleSet = tribles
        .iter()
        .filter(|t| !(replaced.contains(t.a()) && converted.contains(t.e())))
        .cloned()
        .collect();
    eprintln!(
        "[migrate] carrying {} facts verbatim, replacing {}",
        facts.len(),
        tribles.len() - facts.len()
    );

    // The blobs the old representation used, which the new one supersedes.
    let mut superseded: HashSet<[u8; 32]> = HashSet::new();
    let (mut f32n, mut f16n, mut total) = (0usize, 0usize, 0usize);

    for (leaf_id, is_f16) in &leaves {
        let leaf_id = *leaf_id;
        let (dh_raw, sh) = if *is_f16 {
            find!(
                (d: Inline<inlineencodings::Handle<mary::f16enc::F16Array>>,
                 s: Inline<inlineencodings::Handle<mary::format::U64Array>>),
                pattern!(&tribles, [{ leaf_id @ attrs::data_f16: ?d, attrs::shape: ?s }])
            )
            .next()
            .map(|(d, s)| (d.raw, s))
            .context("f16 leaf handles")?
        } else {
            find!(
                (d: Inline<inlineencodings::Handle<mary::format::F32Array>>,
                 s: Inline<inlineencodings::Handle<mary::format::U64Array>>),
                pattern!(&tribles, [{ leaf_id @ attrs::data: ?d, attrs::shape: ?s }])
            )
            .next()
            .map(|(d, s)| (d.raw, s))
            .context("f32 leaf handles")?
        };
        superseded.insert(dh_raw);
        superseded.insert(sh.raw);

        let dims: Vec<u64> = mary::ingest::read_shape(&reader, sh)
            .iter()
            .map(|&d| d as u64)
            .collect();
        let handle: Inline<inlineencodings::Handle<UnknownBlob>> = Inline::new(dh_raw);
        let bytes: anybytes::Bytes = reader
            .get(handle)
            .map_err(|e| anyhow::anyhow!("{leaf_id}: data blob: {e:?}"))?;
        total += bytes.len();

        let name = format!("{leaf_id}");
        let elem = if *is_f16 {
            f16n += 1;
            leaf::Elem::F16
        } else {
            f32n += 1;
            leaf::Elem::F32
        };
        // `put_leaf_as` is the same writer every importer uses, so the rank
        // dispatch and the round-trip check exist once rather than twice.
        let e = leaf::put_leaf_as(
            repo.storage_mut(),
            &ExclusiveId::force_ref(&leaf_id),
            elem,
            &dims,
            bytes,
            &name,
        )?;
        facts += e.into_facts();

        if (f32n + f16n) % 200 == 0 {
            eprintln!("[migrate] {} leaves converted ...", f32n + f16n);
        }
    }

    // ── carry the remaining blobs ───────────────────────────────────────────
    // Everything the pile holds that the substitution did not supersede:
    // tokenizer pieces, names, anything this tool has no schema for. Copied by
    // bytes, so the handles come out identical and every carried fact still
    // resolves.
    let mut copied = 0usize;
    let mut copied_bytes = 0usize;
    for info in reader.blobs() {
        let info = info.map_err(|e| anyhow::anyhow!("list blobs: {e:?}"))?;
        if superseded.contains(&info.handle.raw) {
            continue;
        }
        let bytes: anybytes::Bytes = reader
            .get(info.handle)
            .map_err(|e| anyhow::anyhow!("copy blob: {e:?}"))?;
        copied_bytes += bytes.len();
        let out: Inline<inlineencodings::Handle<blobencodings::RawBytes>> =
            ws.put(Blob::<blobencodings::RawBytes>::new(bytes));
        anyhow::ensure!(
            out.raw == info.handle.raw,
            "blob handle changed on copy — content addressing violated"
        );
        copied += 1;
    }
    eprintln!(
        "[migrate] carried {copied} other blobs ({:.2} GiB)",
        copied_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    ws.commit(facts, "typed tensor leaves");
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close().map_err(|e| anyhow::anyhow!("close: {e:?}"))?;

    eprintln!(
        "[migrate] {} leaves ({f32n} f32, {f16n} f16), {:.2} GiB of payload, all verified \
         byte-identical",
        f32n + f16n,
        total as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}
