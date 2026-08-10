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
//!   pile_leaf_migrate <src.pile> <dst.pile>

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use mary::format::attrs;
use mary::leaf;
use std::collections::HashSet;
use std::path::Path;
use triblespace::core::blob::encodings::tensor::elements::{F16, F32};
use triblespace::core::blob::encodings::UnknownBlob;
use triblespace::core::blob::Blob;
use triblespace::core::id::ExclusiveId;
use triblespace::core::repo::{ancestors, BlobStoreList, Repository};
use triblespace::macros::{entity, find, pattern};
use triblespace::prelude::*;

/// Build the typed blob, verify it against the source bytes, and attach it to
/// the leaf's OWN entity id.
///
/// The rank dispatch is the price of rank-in-the-type, paid once here rather
/// than by every reader — and worth being precise about, because it replaces a
/// check rather than adding one: `format::load_tensor` already carried
/// `assert_eq!(shp.len(), D)`, so consumers were already passing the rank
/// statically and paying for it at runtime.
macro_rules! typed_leaf {
    ($elem:ty, $rank:literal, $ws:expr, $id:expr, $dims:expr, $payload:expr, $name:expr) => {{
        let dims: [u64; $rank] = $dims.as_slice().try_into().expect("rank checked by caller");
        let src_bytes = $payload;
        let blob = leaf::leaf_blob::<$elem, $rank>(dims, src_bytes.clone())?;

        // Verified where both halves are in hand: decode what was just built
        // and compare. A view, not a copy.
        let view = leaf::read_leaf::<$elem, $rank>(blob.clone())?;
        anyhow::ensure!(
            view.dims() == dims,
            "{}: dims changed in conversion: {:?} -> {:?}",
            $name,
            dims,
            view.dims()
        );
        anyhow::ensure!(
            &view.payload()[..] == &src_bytes[..],
            "{}: payload changed in conversion ({} src bytes vs {} stored)",
            $name,
            src_bytes.len(),
            view.payload().len()
        );

        let handle = $ws.put(blob);
        entity! { $id @ leaf::leaf::<$elem, $rank>(): handle }
    }};
}

/// Dispatch over the ranks models actually use.
///
/// Rank 0 is a real case, not an edge case: `clip`'s `logit_scale` is a scalar,
/// and a rank-0 tensor is one element with no dims. Rank 5+ is REFUSED rather
/// than flattened — a pile holding one should say so, not come back as
/// plausible numbers.
macro_rules! by_rank {
    ($elem:ty, $ws:expr, $id:expr, $dims:expr, $payload:expr, $name:expr) => {
        match $dims.len() {
            0 => typed_leaf!($elem, 0, $ws, $id, $dims, $payload, $name),
            1 => typed_leaf!($elem, 1, $ws, $id, $dims, $payload, $name),
            2 => typed_leaf!($elem, 2, $ws, $id, $dims, $payload, $name),
            3 => typed_leaf!($elem, 3, $ws, $id, $dims, $payload, $name),
            4 => typed_leaf!($elem, 4, $ws, $id, $dims, $payload, $name),
            r => anyhow::bail!(
                "{}: rank {r} exceeds the ranks this converter dispatches (0..=4); \
                 add an arm rather than flattening",
                $name
            ),
        }
    };
}

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
    // Carry every fact EXCEPT the three being replaced. Values are copied
    // verbatim, so nothing needs to be understood in order to survive.
    let replaced = [attrs::data.id(), attrs::data_f16.id(), attrs::shape.id()];
    let mut facts: TribleSet = tribles
        .iter()
        .filter(|t| !replaced.contains(t.a()))
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
        let e = if *is_f16 {
            f16n += 1;
            by_rank!(F16, ws, ExclusiveId::force_ref(&leaf_id), dims, bytes, name)
        } else {
            f32n += 1;
            by_rank!(F32, ws, ExclusiveId::force_ref(&leaf_id), dims, bytes, name)
        };
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
