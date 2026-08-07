//! Persist the PersonaPlex-7B checkpoint into a REAL on-disk TribleSpace
//! pile — the model shelf's `personaplex.pile`. Two entities:
//!
//!   - `model.safetensors` — the 7B LM (temporal transformer + depth
//!     transformer + embeddings), 475 bf16 tensors → exact f32 leaves
//!     (bf16→f32 widening is lossless).
//!   - `tokenizer-e351c8d8-checkpoint125.safetensors` — the Mimi codec
//!     (byte-identical to the ungated kyutai checkpoint the mimi port was
//!     gated against; persisted here so the pile is the SELF-CONTAINED voice
//!     stack: codec encoder + decoder + LM).
//!
//! After this runs, the pile file is the durable weight store the LM port
//! loads from — the HF download is no longer needed. Tensor names don't
//! collide across the two checkpoints (`transformer.*`/`depformer.*`/`emb.*`…
//! vs `encoder.*`/`decoder.*`/`quantizer.*`…), so the union keymap serves
//! both components.
//!
//!   cargo run --release --features personaplex,q4,import --bin personaplex_persist -- \
//!     <ckpt-dir> <pile-path>
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

use mary::ingest::read_leaf;
use mary::models::personaplex::config as cfg;
use mary::nn::weight_loader::{get_tensor_f32, read_safetensors_file};
use mary::persist::{load_split_index_from_pile, persist_safetensors_files_to_pile};
use safetensors::SafeTensors;
use std::path::Path;
use std::time::Instant;
use triblespace::prelude::BlobStoreGet;

const LM_FILE: &str = "model.safetensors";
const MIMI_FILE: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";

/// Assert one tensor's shape, by name, against the expectation from config.
fn expect_shape(st: &SafeTensors, name: &str, want: &[usize]) {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("missing {name}: {e}"));
    assert_eq!(view.shape(), want, "{name}: shape mismatch vs config");
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
    if args.len() < 3 {
        eprintln!(
            "usage: personaplex_persist <ckpt-dir> <pile-path>\n       \
             personaplex_persist --derive-fmt <q4|q8|f16> <src-pile> [dst-pile]\n       \
             personaplex_persist --derive-depth <src-pile> [dst-pile]"
        );
        std::process::exit(2);
    }
    let ckpt_dir = Path::new(&args[1]);
    let pile_path = Path::new(&args[2]);

    let files = vec![
        (ckpt_dir.join(LM_FILE), LM_FILE.to_string()),
        (ckpt_dir.join(MIMI_FILE), MIMI_FILE.to_string()),
    ];
    for (p, _) in &files {
        anyhow::ensure!(p.exists(), "checkpoint file missing: {p:?}");
    }

    let t = Instant::now();
    eprintln!("Persisting PersonaPlex checkpoint from {ckpt_dir:?} → {pile_path:?} ...");
    persist_safetensors_files_to_pile(&files, pile_path, mary::ingest::LeafDtype::F32)?;
    let persist_secs = t.elapsed().as_secs_f64();

    // ── gate: round-trip bit-exactness + config-truth shape assertions ──
    let (_f16, leaves, reader) = load_split_index_from_pile(pile_path, "")?;
    eprintln!(
        "Pile holds {} leaves; verifying against the safetensors sources ...",
        leaves.len()
    );

    let mut lm_count = 0usize;
    let (mut checked, mut elems) = (0usize, 0usize);
    for (path, entity) in &files {
        let bytes = read_safetensors_file(path);
        let st = SafeTensors::deserialize(&bytes)?;

        if *entity == LM_FILE {
            // Config truth: the load-bearing dims, asserted against real shapes.
            expect_shape(
                &st,
                "transformer.layers.0.self_attn.in_proj_weight",
                &[3 * cfg::DIM, cfg::DIM],
            );
            expect_shape(
                &st,
                "transformer.layers.0.self_attn.out_proj.weight",
                &[cfg::DIM, cfg::DIM],
            );
            expect_shape(
                &st,
                "transformer.layers.0.gating.linear_in.weight",
                &[cfg::FFN_FUSED_IN, cfg::DIM],
            );
            expect_shape(
                &st,
                "transformer.layers.0.gating.linear_out.weight",
                &[cfg::DIM, cfg::FFN_HIDDEN],
            );
            expect_shape(
                &st,
                &format!("transformer.layers.{}.norm2.alpha", cfg::NUM_LAYERS - 1),
                &[1, 1, cfg::DIM],
            );
            expect_shape(&st, "out_norm.alpha", &[1, 1, cfg::DIM]);
            expect_shape(
                &st,
                "depformer.layers.0.self_attn.in_proj_weight",
                &[cfg::WEIGHTS_PER_STEP * 3 * cfg::DEP_DIM, cfg::DEP_DIM],
            );
            expect_shape(
                &st,
                "depformer.layers.0.self_attn.out_proj.weight",
                &[cfg::WEIGHTS_PER_STEP * cfg::DEP_DIM, cfg::DEP_DIM],
            );
            expect_shape(
                &st,
                &format!(
                    "depformer.layers.{}.gating.{}.linear_in.weight",
                    cfg::DEP_LAYERS - 1,
                    cfg::WEIGHTS_PER_STEP - 1
                ),
                &[2 * cfg::DEP_FFN_HIDDEN, cfg::DEP_DIM],
            );
            expect_shape(
                &st,
                &format!("emb.{}.weight", cfg::N_Q - 1),
                &[cfg::AUDIO_VOCAB, cfg::DIM],
            );
            expect_shape(&st, "text_emb.weight", &[cfg::TEXT_VOCAB, cfg::DIM]);
            expect_shape(&st, "text_linear.weight", &[cfg::TEXT_LOGITS, cfg::DIM]);
            expect_shape(
                &st,
                &format!("depformer_in.{}.weight", cfg::DEP_Q - 1),
                &[cfg::DEP_DIM, cfg::DIM],
            );
            expect_shape(
                &st,
                &format!("depformer_emb.{}.weight", cfg::DEP_Q - 2),
                &[cfg::AUDIO_VOCAB, cfg::DEP_DIM],
            );
            expect_shape(
                &st,
                "depformer_text_emb.weight",
                &[cfg::TEXT_VOCAB, cfg::DEP_DIM],
            );
            expect_shape(
                &st,
                &format!("linears.{}.weight", cfg::DEP_Q - 1),
                &[cfg::CARD, cfg::DEP_DIM],
            );
            eprintln!("config-truth shape assertions PASSED (temporal ffn {} fused {}, depth ffn {}, {} per-step)",
                cfg::FFN_HIDDEN, cfg::FFN_FUSED_IN, cfg::DEP_FFN_HIDDEN, cfg::WEIGHTS_PER_STEP);
        }

        let mut misaligned = 0usize;
        for name in st.names() {
            use safetensors::Dtype;
            let view = st.tensor(name)?;
            if !matches!(
                view.dtype(),
                Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16
            ) {
                continue; // ingest skips non-float buffers
            }
            if *entity == LM_FILE {
                lm_count += 1;
            }
            let (want, want_shape) = get_tensor_f32(&st, name);
            let handles = leaves
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("{entity}/{name}: no pile leaf"))?;
            // Alignment: the raw data blob must sit 256-aligned in the mmap.
            if let mary::ingest::LeafHandles::F32(dh, _) = handles {
                let b: anybytes::Bytes = reader
                    .get(*dh)
                    .map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
                if b.as_ptr() as usize % 256 != 0 {
                    eprintln!("  MISALIGNED (ptr % 256 != 0): {name}");
                    misaligned += 1;
                }
            } else {
                anyhow::bail!("{entity}/{name}: expected an f32 leaf");
            }
            let (got, got_shape) = read_leaf(&reader, *handles);
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
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                anyhow::ensure!(
                    g.to_bits() == w.to_bits(),
                    "{name}[{i}]: pile {g} != source {w} (bit mismatch)"
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
    anyhow::ensure!(
        lm_count == cfg::CHECKPOINT_TENSORS,
        "LM leaf count {lm_count} != expected {} (config::CHECKPOINT_TENSORS)",
        cfg::CHECKPOINT_TENSORS
    );
    println!(
        "personaplex pile gate PASSED: {checked} tensors / {elems} elements bit-identical \
         (LM {lm_count} leaves), all 256-aligned."
    );

    let size = std::fs::metadata(pile_path)?.len();
    println!(
        "Persisted in {persist_secs:.1}s. Pile file {pile_path:?} is {} bytes ({:.2} GiB).",
        size,
        size as f64 / (1u64 << 30) as f64
    );
    Ok(())
}
