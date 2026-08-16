//! Persist the PersonaPlex-7B checkpoint into Mary's native append-only model
//! collection. The LM and Mimi codec become one exact f32 content-addressed
//! root under the canonical `nvidia/personaplex-7b-v1` source coordinate:
//!
//!   - `model.safetensors` — the 7B LM (temporal transformer + depth
//!     transformer + embeddings), 475 bf16 tensors → exact f32 leaves
//!     (bf16→f32 widening is lossless).
//!   - `tokenizer-e351c8d8-checkpoint125.safetensors` — the Mimi codec
//!     (byte-identical to the ungated kyutai checkpoint the mimi port was
//!     gated against; persisted here so the pile is the SELF-CONTAINED voice
//!     stack: codec encoder + decoder + LM).
//!
//! After this runs, the pile file and its signed native collection commit are
//! the durable weight authority; the HF download is no longer needed. Tensor
//! names don't collide across the two checkpoints
//! (`transformer.*`/`depformer.*`/`emb.*`… vs
//! `encoder.*`/`decoder.*`/`quantizer.*`…), so one strict root serves both
//! components.
//!
//!   cargo run --release --features personaplex,q4,import --bin personaplex_persist -- \
//!     <ckpt-dir> <pile-path> <signing-key>
//!
//! `<ckpt-dir>` holds `model.safetensors` and
//! `tokenizer-e351c8d8-checkpoint125.safetensors` (the HF snapshot layout).
//!
//! ── derive modes (canonical pile → runtime-format sibling; src READ-ONLY) ──
//!
//!   personaplex_persist --derive-fmt <q4|q8|f16> <src-pile> [dst-pile]
//!   personaplex_persist --derive-depth <src-pile> [dst-pile]
//!
//! Runs the load-time transform pass ONCE (quantize/convert the temporal
//! stack, fold+slice the depformer operands) and persists the exact runtime
//! bytes as a derived sibling pile (`<stem>_<fmt>.pile` / `<stem>_depth.pile`,
//! auto-discovered by the realtime loaders — see
//! `mary::models::personaplex::qpile`). SAFETY GATE: the source pile's byte
//! length + sha256 and every other `*.pile` in its directory's byte length
//! are recorded before and verified unchanged after — the canonical piles
//! are never written, only the new sibling file is created.
//!
//! ── gate (always runs): pile round-trip bit-exactness ──
//! Every float tensor of both source files is re-read from the pile and
//! compared BIT-EXACT (f32 to_bits) against the bf16→f32 widening of the
//! safetensors bytes; shapes must match; every data blob must sit 256-aligned
//! in the pile mmap (the V3 payload-alignment invariant); and the LM leaf
//! count must equal `config::CHECKPOINT_TENSORS`. On top of the round-trip,
//! the load-bearing architecture constants in
//! `mary::models::personaplex::config` are asserted against the REAL tensor
//! shapes — the config file stays mechanically verified, not just documented.

use mary::ingest::LeafDtype;
use mary::models::personaplex::{config as cfg, PersonaPlexWeights, SOURCE};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;
use triblespace::prelude::BlobStoreGet;

const LM_FILE: &str = "model.safetensors";
const MIMI_FILE: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";

/// Validate one tensor's shape against the architecture config.
fn expect_shape(st: &SafeTensors, name: &str, want: &[usize]) -> anyhow::Result<()> {
    let view = st
        .tensor(name)
        .map_err(|error| anyhow::anyhow!("missing {name}: {error}"))?;
    anyhow::ensure!(
        view.shape() == want,
        "{name}: shape {:?} != config {want:?}",
        view.shape()
    );
    Ok(())
}

/// Fallibly widen one source tensor to f32 while retaining its exact shape.
fn tensor_f32(view: &TensorView<'_>) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    let shape = view.shape().to_vec();
    let elements = shape.iter().try_fold(1_usize, |product, &dimension| {
        product.checked_mul(dimension)
    });
    let elements = elements.ok_or_else(|| anyhow::anyhow!("tensor shape product overflow"))?;
    let width = match view.dtype() {
        Dtype::F64 => 8,
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        dtype => anyhow::bail!("unsupported source tensor dtype {dtype:?}"),
    };
    let expected_bytes = elements
        .checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("tensor payload byte length overflow"))?;
    anyhow::ensure!(
        view.data().len() == expected_bytes,
        "tensor payload has {} bytes, expected {} for shape {shape:?} and {:?}",
        view.data().len(),
        expected_bytes,
        view.dtype()
    );
    // The checked total length makes every `chunks_exact(width)` item exactly
    // wide enough for direct indexing, without per-scalar fallibility overhead.
    let data = match view.dtype() {
        Dtype::F64 => view
            .data()
            .chunks_exact(8)
            .map(|bytes| {
                f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]) as f32
            })
            .collect(),
        Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect(),
        Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
            .collect(),
        Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
            .collect(),
        dtype => anyhow::bail!("unsupported source tensor dtype {dtype:?}"),
    };
    Ok((data, shape))
}

/// sha256 of a file via the system `shasum` (streamed — no crate dep for a
/// one-shot integrity check in a persist tool).
fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let out = std::process::Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .map_err(|e| anyhow::anyhow!("spawn shasum: {e}"))?;
    anyhow::ensure!(out.status.success(), "shasum failed on {path:?}");
    let s = String::from_utf8_lossy(&out.stdout);
    Ok(s.split_whitespace().next().unwrap_or_default().to_string())
}

/// Byte lengths of every `*.pile` in `dir` EXCEPT `skip` (the destination
/// sibling being created) — the "nothing else changed" half of the safety
/// gate.
fn pile_lengths(dir: &Path, skip: &Path) -> anyhow::Result<Vec<(std::path::PathBuf, u64)>> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().map(|x| x == "pile").unwrap_or(false)
            && p.canonicalize().ok() != skip.canonicalize().ok()
        {
            v.push((p.clone(), std::fs::metadata(&p)?.len()));
        }
    }
    v.sort();
    Ok(v)
}

/// The derive entry: transform-once → sibling pile, with the read-only
/// safety gate around the source directory.
#[cfg(all(feature = "q4", target_os = "macos"))]
fn run_derive(mode: &str, args: &[String]) -> anyhow::Result<()> {
    use mary::models::personaplex::qpile;
    use mary::models::personaplex::temporal_metal::WeightFmt;

    let (fmt, src_i) = match mode {
        "--derive-fmt" => {
            let f = args
                .first()
                .and_then(|s| WeightFmt::parse(s))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "usage: personaplex_persist --derive-fmt <q4|q8|f16> <src-pile> [dst-pile]"
                    )
                })?;
            (Some(f), 1)
        }
        "--derive-depth" => (None, 0),
        _ => unreachable!(),
    };
    let src = Path::new(
        args.get(src_i)
            .ok_or_else(|| anyhow::anyhow!("missing <src-pile>"))?,
    );
    anyhow::ensure!(src.exists(), "source pile missing: {src:?}");
    let tag = fmt.map(qpile::fmt_tag).unwrap_or("depth");
    let dst = args
        .get(src_i + 1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| qpile::derived_sibling_path(src, tag));

    // ── safety gate: record the source's identity + every sibling's length ──
    let src_len = std::fs::metadata(src)?.len();
    eprintln!("[derive] hashing source pile {src:?} ({src_len} bytes) ...");
    let src_sha = sha256_file(src)?;
    let dir = src.parent().unwrap_or(Path::new("."));
    let before = pile_lengths(dir, &dst)?;
    eprintln!(
        "[derive] source sha256 {src_sha}; {} sibling pile(s) length-recorded",
        before.len()
    );

    let t = Instant::now();
    let (count, bytes) = match fmt {
        Some(f) => qpile::derive_temporal_pile(src, &dst, f)?,
        None => qpile::derive_depth_pile(src, &dst)?,
    };
    let secs = t.elapsed().as_secs_f64();

    // ── verify: source (and every other pile) byte-identical/unchanged ──
    anyhow::ensure!(
        std::fs::metadata(src)?.len() == src_len,
        "SOURCE PILE LENGTH CHANGED — investigate immediately"
    );
    let src_sha_after = sha256_file(src)?;
    anyhow::ensure!(
        src_sha_after == src_sha,
        "SOURCE PILE HASH CHANGED ({src_sha} → {src_sha_after}) — investigate immediately"
    );
    let after = pile_lengths(dir, &dst)?;
    anyhow::ensure!(
        before == after,
        "a sibling pile changed length during derive — before {before:?} after {after:?}"
    );

    let dst_len = std::fs::metadata(&dst)?.len();
    println!(
        "derive {tag} DONE in {secs:.1}s: {count} leaves / {bytes} payload bytes → {dst:?} \
         ({dst_len} bytes, {:.2} GiB). Source pile verified unchanged (len {src_len}, sha256 {src_sha}).",
        dst_len as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

#[cfg(not(all(feature = "q4", target_os = "macos")))]
fn run_derive(_mode: &str, _args: &[String]) -> anyhow::Result<()> {
    anyhow::bail!("--derive-* requires the q4 feature on macOS (the realtime lane's target)")
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && (args[1] == "--derive-fmt" || args[1] == "--derive-depth") {
        return run_derive(&args[1], &args[2..]);
    }
    if args.len() != 4 {
        eprintln!(
            "usage: personaplex_persist <ckpt-dir> <pile-path> <signing-key>\n       \
             personaplex_persist --derive-fmt <q4|q8|f16> <src-pile> [dst-pile]\n       \
             personaplex_persist --derive-depth <src-pile> [dst-pile]"
        );
        std::process::exit(2);
    }
    let ckpt_dir = Path::new(&args[1]);
    let pile_path = Path::new(&args[2]);
    let signing_key = signing_key_file::load_existing(Path::new(&args[3]))?;

    let files = [
        (ckpt_dir.join(LM_FILE), LM_FILE),
        (ckpt_dir.join(MIMI_FILE), MIMI_FILE),
    ];
    for (path, _) in &files {
        anyhow::ensure!(path.is_file(), "checkpoint file missing: {path:?}");
    }

    // The generic importer intentionally ingests every safetensors file in a
    // directory. PersonaPlex's authority is narrower: exactly these two files
    // compose the one LM + Mimi root. Reject stale copies and surprise shards
    // before writing anything.
    let (format, detected) = mary::formats::detect_format(ckpt_dir)?;
    anyhow::ensure!(
        format == mary::formats::WeightFormat::Safetensors,
        "PersonaPlex checkpoint must use safetensors, found {format:?}"
    );
    let detected_names: BTreeSet<_> = detected
        .iter()
        .map(|path| {
            path.file_name()
                .map(std::ffi::OsStr::to_owned)
                .ok_or_else(|| anyhow::anyhow!("weight path has no file name: {path:?}"))
        })
        .collect::<anyhow::Result<_>>()?;
    let expected_files: BTreeSet<_> = [LM_FILE, MIMI_FILE]
        .into_iter()
        .map(std::ffi::OsString::from)
        .collect();
    anyhow::ensure!(
        detected_names == expected_files,
        "PersonaPlex checkpoint must contain exactly {expected_files:?}, found {detected_names:?}"
    );

    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(pile_path)
    {
        Ok(_) => eprintln!("personaplex_persist: created new empty pile {pile_path:?}"),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }

    let started = Instant::now();
    let mut pile = Pile::open(pile_path)
        .map_err(|error| anyhow::anyhow!("open model pile {pile_path:?}: {error}"))?;
    let imported = (|| -> anyhow::Result<_> {
        eprintln!("Persisting PersonaPlex checkpoint {SOURCE} from {ckpt_dir:?} ...");
        let candidate = mary::persist::ingest_model_fragment(
            &mut pile,
            ckpt_dir,
            LeafDtype::F32,
            SOURCE,
            mary::persist::QUANTIZATION_NATIVE,
        )?;
        let root = candidate
            .root()
            .ok_or_else(|| anyhow::anyhow!("PersonaPlex candidate has no unique model root"))?;

        // Freeze the preexisting authority only after staged blobs exist, so
        // its reader covers both the authorized graph and candidate handles.
        // Candidate facts are visible solely to validation: no signed record
        // is published until every gate below succeeds.
        let snapshot = mary::model_collection::snapshot_model_collection_local_latest(&mut pile)?;
        let (mut candidate_view, _, reader) = snapshot.into_parts();
        candidate_view += candidate.facts().clone();
        let weights = PersonaPlexWeights::from_graph(&candidate_view, reader)?;
        anyhow::ensure!(
            weights.root() == root,
            "staged PersonaPlex root differs from the uniquely admitted Source/native root"
        );
        eprintln!(
            "Native root holds {} tensors; verifying both safetensors sources ...",
            weights.count()
        );

        let mut expected_names = BTreeSet::new();
        let mut lm_count = 0usize;
        let (mut checked, mut elems) = (0usize, 0usize);
        for (path, entity) in &files {
            let bytes = std::fs::read(path)
                .map_err(|error| anyhow::anyhow!("read source checkpoint {path:?}: {error}"))?;
            let st = SafeTensors::deserialize(&bytes)?;

            if *entity == LM_FILE {
                // Config truth: the load-bearing dims, asserted against real shapes.
                expect_shape(
                    &st,
                    "transformer.layers.0.self_attn.in_proj_weight",
                    &[3 * cfg::DIM, cfg::DIM],
                )?;
                expect_shape(
                    &st,
                    "transformer.layers.0.self_attn.out_proj.weight",
                    &[cfg::DIM, cfg::DIM],
                )?;
                expect_shape(
                    &st,
                    "transformer.layers.0.gating.linear_in.weight",
                    &[cfg::FFN_FUSED_IN, cfg::DIM],
                )?;
                expect_shape(
                    &st,
                    "transformer.layers.0.gating.linear_out.weight",
                    &[cfg::DIM, cfg::FFN_HIDDEN],
                )?;
                expect_shape(
                    &st,
                    &format!("transformer.layers.{}.norm2.alpha", cfg::NUM_LAYERS - 1),
                    &[1, 1, cfg::DIM],
                )?;
                expect_shape(&st, "out_norm.alpha", &[1, 1, cfg::DIM])?;
                expect_shape(
                    &st,
                    "depformer.layers.0.self_attn.in_proj_weight",
                    &[cfg::WEIGHTS_PER_STEP * 3 * cfg::DEP_DIM, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    "depformer.layers.0.self_attn.out_proj.weight",
                    &[cfg::WEIGHTS_PER_STEP * cfg::DEP_DIM, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    &format!(
                        "depformer.layers.{}.gating.{}.linear_in.weight",
                        cfg::DEP_LAYERS - 1,
                        cfg::WEIGHTS_PER_STEP - 1
                    ),
                    &[2 * cfg::DEP_FFN_HIDDEN, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    &format!("emb.{}.weight", cfg::N_Q - 1),
                    &[cfg::AUDIO_VOCAB, cfg::DIM],
                )?;
                expect_shape(&st, "text_emb.weight", &[cfg::TEXT_VOCAB, cfg::DIM])?;
                expect_shape(&st, "text_linear.weight", &[cfg::TEXT_LOGITS, cfg::DIM])?;
                expect_shape(
                    &st,
                    &format!("depformer_in.{}.weight", cfg::DEP_Q - 1),
                    &[cfg::DEP_DIM, cfg::DIM],
                )?;
                expect_shape(
                    &st,
                    &format!("depformer_emb.{}.weight", cfg::DEP_Q - 2),
                    &[cfg::AUDIO_VOCAB, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    "depformer_text_emb.weight",
                    &[cfg::TEXT_VOCAB, cfg::DEP_DIM],
                )?;
                expect_shape(
                    &st,
                    &format!("linears.{}.weight", cfg::DEP_Q - 1),
                    &[cfg::CARD, cfg::DEP_DIM],
                )?;
                eprintln!(
                    "config-truth shape assertions PASSED (temporal ffn {} fused {}, depth ffn {}, {} per-step)",
                    cfg::FFN_HIDDEN,
                    cfg::FFN_FUSED_IN,
                    cfg::DEP_FFN_HIDDEN,
                    cfg::WEIGHTS_PER_STEP
                );
            }

            let mut misaligned = 0usize;
            for name in st.names() {
                let view = st.tensor(name)?;
                if !matches!(
                    view.dtype(),
                    Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16
                ) {
                    continue; // importer skips non-float buffers
                }
                anyhow::ensure!(
                    expected_names.insert(name.to_owned()),
                    "tensor name {name:?} occurs in both PersonaPlex source files"
                );
                if *entity == LM_FILE {
                    lm_count += 1;
                }
                let (want, want_shape) = tensor_f32(&view)?;
                let handles = weights
                    .exact()
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("{entity}/{name}: no native root leaf"))?;
                // Alignment: the raw data blob must sit 256-aligned in the mmap.
                if let mary::ingest::LeafHandles::F32(data, _) = handles {
                    let bytes: anybytes::Bytes = weights
                        .reader()
                        .get(*data)
                        .map_err(|error| anyhow::anyhow!("{name}: {error}"))?;
                    if !(bytes.as_ptr() as usize).is_multiple_of(256) {
                        eprintln!("  MISALIGNED (ptr % 256 != 0): {name}");
                        misaligned += 1;
                    }
                } else {
                    anyhow::bail!("{entity}/{name}: expected an f32 leaf");
                }
                let (got, got_shape) =
                    mary::selection::materialize_leaf(weights.reader(), *handles)
                        .map_err(|error| anyhow::anyhow!("{entity}/{name}: {error}"))?;
                anyhow::ensure!(
                    got_shape == want_shape,
                    "{name}: shape {got_shape:?} != {want_shape:?}"
                );
                anyhow::ensure!(
                    got.len() == want.len(),
                    "{name}: len {} != {}",
                    got.len(),
                    want.len()
                );
                for (index, (&got, &want)) in got.iter().zip(want.iter()).enumerate() {
                    anyhow::ensure!(
                        got.to_bits() == want.to_bits(),
                        "{name}[{index}]: pile {got} != source {want} (bit mismatch)"
                    );
                }
                checked += 1;
                elems += got.len();
            }
            anyhow::ensure!(
                misaligned == 0,
                "{entity}: {misaligned} leaves misaligned — V3 alignment invariant violated"
            );
            eprintln!("{entity}: round-trip verified");
        }

        let actual_names: BTreeSet<_> = weights.exact().keys().cloned().collect();
        anyhow::ensure!(
            actual_names == expected_names,
            "native root name set differs from the exact LM + Mimi source union"
        );
        anyhow::ensure!(
            lm_count == cfg::CHECKPOINT_TENSORS,
            "LM leaf count {lm_count} != expected {} (config::CHECKPOINT_TENSORS)",
            cfg::CHECKPOINT_TENSORS
        );
        drop(weights);
        mary::model_collection::publish_model_fragment(&mut pile, &signing_key, candidate)
            .map_err(|error| anyhow::anyhow!("publish validated PersonaPlex root: {error}"))?;
        Ok((root, checked, elems, lm_count))
    })();

    // `close` is the sole durability boundary, even when a source gate fails.
    let close = pile
        .close()
        .map_err(|error| anyhow::anyhow!("close model pile {pile_path:?}: {error}"));
    let (root, checked, elems, lm_count) = match (imported, close) {
        (Ok(result), Ok(())) => result,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(close_error)) => {
            return Err(error.context(format!("import also failed to close pile: {close_error}")));
        }
    };

    let size = std::fs::metadata(pile_path)?.len();
    println!(
        "PersonaPlex native root {root} valid: {checked} tensors / {elems} elements \
         bit-identical (LM {lm_count} leaves), all payloads 256-aligned; \
         pile {:.2} GiB in {:.1}s",
        size as f64 / (1_u64 << 30) as f64,
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}
