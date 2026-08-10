//! Convert a model pile's leaves from `{data|data_f16, shape}` to typed tensors.
//!
//! Pile to pile, into a NEW file. The source is only ever read — these are the
//! curated model piles, so a conversion writes elsewhere and is then COMPARED
//! against the original rather than trusted.
//!
//! What changes per leaf: two handles become one, the shape moves from a
//! separate blob into the tensor's header, and the element type and rank move
//! into the attribute id. What does not change: a single byte of tensor data.
//! This binary verifies that rather than asserting it: each leaf is decoded back
//! out of the blob just built and its payload compared, byte for byte, to the
//! source blob's bytes.
//!
//! The conversion never materialises a tensor. It moves the source blob's bytes
//! into the tensor payload verbatim — no `view::<[f32]>()`, no `Vec`, no cast —
//! which is both why it is fast and why bit-identity is cheap to establish:
//! there is no decode/encode roundtrip to be lossy in. (Contrast
//! `derive_f16_pile`, which goes through `Vec<f32>` because it is genuinely
//! computing new values.)
//!
//! Byte-identity all the way to disk is then two facts, neither assumed: the
//! payload equals the source bytes (checked here, per leaf), and the handle is
//! the hash of the blob, which the store checks on every read.
//!
//! # What is preserved, and what necessarily is not
//!
//! Entity ids here are CONTENT-DERIVED (`entity! { _ @ … }`). Changing how a
//! leaf is stored changes the leaf's content, hence its id, hence the module
//! that points at it, hence the model root. So the converted pile's ids differ
//! from the source's, unavoidably — that is content addressing working, not a
//! defect. What survives is the `(source, quantization)` label pair, which is
//! how every mary-branch loader actually resolves a model. A caller that
//! recorded a raw root id needs the new one; this binary prints the mapping.
//!
//!   pile_leaf_migrate <src.pile> <dst.pile>

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use mary::format::attrs;
use mary::ingest::LeafHandles;
use mary::leaf;
use std::path::Path;
use triblespace::core::blob::encodings::tensor::elements::{F16, F32};
use triblespace::core::repo::{ancestors, Repository};
use triblespace::macros::{entity, find, pattern};
use triblespace::prelude::*;

/// Build the typed blob, put it, and return the leaf entity — all inside the
/// arm that knows the rank.
///
/// This is where rank-in-the-type is paid for: `shape.len()` is a runtime value
/// and `RANK` is not, so SOMETHING has to dispatch. Worth noting what this
/// replaces rather than adds — `format::load_tensor` already carried
/// `assert_eq!(shp.len(), D)`, so every consumer was already passing the rank
/// statically and eating a runtime check for it. The dispatch moves that check
/// here, once, at conversion time.
macro_rules! typed_leaf {
    ($elem:ty, $rank:literal, $ws:expr, $dims:expr, $payload:expr, $name:expr) => {{
        let dims: [u64; $rank] = $dims.as_slice().try_into().expect("rank checked by caller");
        let src_bytes = $payload;
        let blob = leaf::leaf_blob::<$elem, $rank>(dims, src_bytes.clone())?;

        // VERIFY, per leaf, before it is stored: decode the blob we just built
        // and check that the payload is the source bytes and the dims are the
        // source dims. Cheap — a view, not a copy — and it closes the encode
        // step. Disk integrity is then content-addressing's job: the handle IS
        // the hash of these bytes and the store checks it on read, so a blob
        // that verifies here and reads back under that handle later is the same
        // blob. That is the whole byte-identity argument, and neither half is
        // assumed.
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
        entity! { _ @ leaf::leaf::<$elem, $rank>(): handle }
    }};
}

/// Dispatch over the ranks a model actually uses.
///
/// Rank 5+ is REFUSED, not flattened. A pile containing a rank-5 leaf should
/// say so loudly; silently reshaping it is how a tensor comes back as plausible
/// numbers instead of an error.
macro_rules! by_rank {
    ($elem:ty, $ws:expr, $dims:expr, $payload:expr, $name:expr) => {
        match $dims.len() {
            1 => typed_leaf!($elem, 1, $ws, $dims, $payload, $name),
            2 => typed_leaf!($elem, 2, $ws, $dims, $payload, $name),
            3 => typed_leaf!($elem, 3, $ws, $dims, $payload, $name),
            4 => typed_leaf!($elem, 4, $ws, $dims, $payload, $name),
            r => anyhow::bail!(
                "{}: rank {r} exceeds the ranks this converter dispatches (1..=4); \
                 add an arm rather than flattening",
                $name
            ),
        }
    };
}


/// Resolve a pile's model roots under either layout.
///
/// Returns `(facts, blob reader, [(root id, source label, quantization)])`.
/// A `main`-layout pile has no quantization coordinate, so its models are
/// reported as `native` — which is what they are: faithful imports, no derived
/// format. That is a statement about the old layout, not a guess about content.
fn read_roots(
    src: &Path,
) -> Result<(
    TribleSet,
    triblespace::core::repo::pile::PileReader,
    Vec<(Id, String, String)>,
)> {
    if let Ok((tribles, reader)) = mary::persist::checkout_mary_branch(src) {
        let mut roots: Vec<(Id, String, String)> = Vec::new();
        for (m, s_h, q) in find!(
            (m: Id,
             s: Inline<inlineencodings::Handle<blobencodings::LongString>>,
             q: String),
            pattern!(&tribles, [{ ?m @
                attrs::source: ?s,
                attrs::quantization: ?q,
            }])
        ) {
            let sv: anybytes::View<str> = reader
                .get(s_h)
                .map_err(|e| anyhow::anyhow!("source blob: {e:?}"))?;
            roots.push((m, sv.to_string(), q));
        }
        if !roots.is_empty() {
            eprintln!("[migrate] layout: 'mary' branch (source + quantization)");
            roots.sort_by(|a, b| (&a.1, &a.2).cmp(&(&b.1, &b.2)));
            return Ok((tribles, reader, roots));
        }
    }

    let (tribles, reader) = checkout_main_branch(src)?;
    let mut roots: Vec<(Id, String, String)> = Vec::new();
    for (m, n_h) in find!(
        (m: Id, n: Inline<inlineencodings::Handle<blobencodings::LongString>>),
        pattern!(&tribles, [{ ?m @ attrs::model_name: ?n }])
    ) {
        let nv: anybytes::View<str> = reader
            .get(n_h)
            .map_err(|e| anyhow::anyhow!("model name blob: {e:?}"))?;
        roots.push((m, nv.to_string(), mary::persist::QUANTIZATION_NATIVE.to_string()));
    }
    if !roots.is_empty() {
        eprintln!("[migrate] layout: 'main' branch (model_name)");
    }
    roots.sort_by(|a, b| (&a.1, &a.2).cmp(&(&b.1, &b.2)));
    Ok((tribles, reader, roots))
}

/// The `main`-branch twin of `checkout_mary_branch`. NEVER amputates: a corrupt
/// tail fails loud rather than being silently truncated on a read path.
fn checkout_main_branch(
    pile_path: &Path,
) -> Result<(TribleSet, triblespace::core::repo::pile::PileReader)> {
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
    let branch_id = repo
        .lookup_branch("main")
        .map_err(|e| anyhow::anyhow!("lookup main: {e:?}"))?
        .ok_or_else(|| anyhow::anyhow!("no 'main' branch in {pile_path:?}"))?;
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;
    let head = ws
        .head()
        .ok_or_else(|| anyhow::anyhow!("'main' has no commits"))?;
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
    Ok((tribles, reader))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let src = args.next().context("usage: pile_leaf_migrate <src.pile> <dst.pile>")?;
    let dst = args.next().context("usage: pile_leaf_migrate <src.pile> <dst.pile>")?;
    if let Some(extra) = args.next() {
        anyhow::bail!("unexpected argument {extra}");
    }
    let (src, dst) = (Path::new(&src), Path::new(&dst));
    anyhow::ensure!(
        src.canonicalize()? != dst.canonicalize().unwrap_or_else(|_| dst.to_path_buf()),
        "src and dst are the same pile file — refusing to write into the source"
    );

    // Two layouts exist in the wild and a converter has to read both:
    //
    //   `mary` branch  — a CONTENT-ADDRESSED root labelled `source` +
    //                    `quantization` (the current form).
    //   `main` branch  — one entity per persisted shard carrying `model_name`
    //                    (the older form; most of ~/models is still this).
    //
    // Detected, not assumed: try the mary branch, fall back to main. The
    // destination always gets the current form, so conversion doubles as the
    // layout upgrade.
    let (tribles, reader, roots) = read_roots(src)?;
    anyhow::ensure!(
        !roots.is_empty(),
        "no model roots found in {src:?} (neither a 'mary' branch with \
         source+quantization, nor 'main' with model_name)"
    );
    eprintln!("[migrate] {} model root(s) in {src:?}:", roots.len());
    for (id, source, quant) in &roots {
        eprintln!("[migrate]   {id}  {source}  ({quant})");
    }

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
        .lookup_branch("mary")
        .map_err(|e| anyhow::anyhow!("lookup mary: {e:?}"))?
    {
        Some(id) => id,
        None => *repo
            .create_branch("mary", None)
            .map_err(|e| anyhow::anyhow!("create mary: {e:?}"))?,
    };
    let mut ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull mary: {e:?}"))?;

    let mut facts = TribleSet::new();
    let (mut f32n, mut f16n, mut total) = (0usize, 0usize, 0usize);
    let mut mapping: Vec<(String, String, Id, Id)> = Vec::new();

    for (root, source, quant) in &roots {
        let index = mary::ingest::index_keymap(&tribles, &reader, *root);
        anyhow::ensure!(!index.is_empty(), "{source}: root {root} has no members");
        let mut names: Vec<&String> = index.keys().collect();
        names.sort();
        eprintln!("[migrate] {source} ({quant}): {} leaves", names.len());

        let mut members: Vec<Id> = Vec::new();
        for name in names {
            let handles = index[name];
            let dims_usize = match handles {
                LeafHandles::F32(_, sh) | LeafHandles::F16(_, sh) => {
                    mary::ingest::read_shape(&reader, sh)
                }
            };
            let dims: Vec<u64> = dims_usize.iter().map(|&d| d as u64).collect();

            // The payload, as RAW BYTES. No view::<[f32]>(), no Vec, no cast —
            // the bytes in the source blob are the bytes in the tensor payload.
            let (payload_len, leaf_entity) = match handles {
                LeafHandles::F32(dh, _) => {
                    let bytes: anybytes::Bytes = reader
                        .get(dh)
                        .map_err(|e| anyhow::anyhow!("{name}: data blob: {e:?}"))?;
                    f32n += 1;
                    let n = bytes.len();
                    (n, by_rank!(F32, ws, dims, bytes, name))
                }
                LeafHandles::F16(dh, _) => {
                    let bytes: anybytes::Bytes = reader
                        .get(dh)
                        .map_err(|e| anyhow::anyhow!("{name}: data_f16 blob: {e:?}"))?;
                    f16n += 1;
                    let n = bytes.len();
                    (n, by_rank!(F16, ws, dims, bytes, name))
                }
            };
            let leaf_id = leaf_entity.root().expect("leaf root");
            total += payload_len;
            facts += leaf_entity.into_facts();

            // Module structure carried over unchanged: same kind vocabulary,
            // same name edge, same `weight` role-edge. Only the leaf's storage
            // differs, so only the leaf's attribute should.
            let kind = match dims.len() {
                1 => "vector",
                2 => "matrix",
                3 => "conv",
                _ => "tensor",
            };
            let name_h = ws.put::<blobencodings::LongString, _>(name.clone());
            let m = entity! { _ @
                attrs::kind: kind,
                attrs::safetensor_path: name_h,
                attrs::weight: leaf_id,
            };
            members.push(m.root().expect("module root"));
            facts += m.into_facts();
            if (f32n + f16n) % 100 == 0 {
                eprintln!("[migrate] {} leaves converted ...", f32n + f16n);
            }
        }

        // The labels are what loaders resolve by, so they carry over verbatim.
        // The root ID changes because its members did — see the module docs.
        let src_h = ws.put::<blobencodings::LongString, _>(source.clone());
        let new_root = entity! { _ @
            attrs::source: src_h,
            attrs::quantization: quant.as_str(),
            attrs::member*: members.iter(),
        };
        let new_root_id = new_root.root().expect("model root");
        facts += new_root.into_facts();
        mapping.push((source.clone(), quant.clone(), *root, new_root_id));
    }

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
    eprintln!("[migrate] root id mapping (old -> new):");
    for (source, quant, old, new) in &mapping {
        eprintln!("[migrate]   {source} ({quant}): {old} -> {new}");
    }
    Ok(())
}
