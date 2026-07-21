//! Persist the Voxtral-Mini-4B-Realtime-2602 checkpoint (HF `model.safetensors`,
//! 711 tensors, bf16) into a REAL on-disk TribleSpace pile as exact f32 leaves
//! (bf16 → f32 is lossless). After this runs, the pile file is the durable,
//! self-contained weight store the Burn transcriber loads from — the HF cache is no
//! longer needed at runtime.
//!
//!   cargo run --release --features import --bin voxtral_persist -- \
//!     <model-dir> <pile-path>
//!
//! `<model-dir>` is the HF snapshot dir holding `model.safetensors` (e.g.
//! `~/.cache/huggingface/hub/models--mistralai--Voxtral-Mini-4B-Realtime-2602/
//! snapshots/<sha>/`). The pile gains one model entity named
//! `model.safetensors` — the tensor names inside are the HF names
//! (`audio_tower.*`, `language_model.model.*`, `multi_modal_projector.*`).
//!
//! Gate (always runs): every persisted leaf is re-read from the pile and
//! compared BIT-EXACT against the bf16→f32 cast of the source safetensors,
//! and every leaf's payload must sit 256-aligned in the mmap (the V3
//! payload-alignment invariant fast mmap views rely on).
//!
//! ── `--f16-derive` mode ──
//!
//!   cargo run --release --features import --bin voxtral_persist -- \
//!     --f16-derive <f32-pile> [<f16-pile>]
//!
//! Derives the HALF-WIDTH sibling pile (default `<stem>_f16.pile` next to the
//! source — the path `load_loader_with_f16_sibling` auto-discovers): every
//! f32 leaf is read from the source pile, cast host-side to f16
//! (`f16::from_f32`, the same double-rounding the materializing loader
//! applies at load time on the f16 backend, so the fast-loaded transcriber is
//! bit-identical to the old cast-on-load path) and persisted under the
//! `ears_f16` entity. The SOURCE pile is strictly read-only in this mode —
//! its byte length is recorded before and re-checked after as a hard gate.
//! The qwen3tts `talker_f16` precedent, in the separate-pile layout: piles
//! union by `cat` + consolidate if we ever want them merged, and the ~8 GiB
//! f16 pile can deploy without the 16.5 GiB f32.
//!
//! Gate (always runs): every f16 leaf is re-read from the sibling pile and
//! compared BIT-EXACT against `f16::from_f32` of the source's f32 leaf, with
//! shape equality, full 711-tensor coverage, and 256-alignment.

use mary::ingest::{read_leaf, read_shape, LeafDtype, LeafHandles};
use mary::persist::{
    derive_f16_pile, f16_sibling_path, load_split_index_from_pile, persist_safetensors_to_pile,
};
use std::path::Path;
use std::time::Instant;
use triblespace::prelude::BlobStoreGet;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let f16_derive = args.iter().any(|a| a == "--f16-derive");
    let pos: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    if f16_derive {
        if pos.is_empty() {
            eprintln!("usage: voxtral_persist --f16-derive <f32-pile> [<f16-pile>]");
            std::process::exit(2);
        }
        let src = Path::new(pos[0]);
        let dst = pos
            .get(1)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| f16_sibling_path(src));
        return f16_derive_mode(src, &dst);
    }
    if pos.len() < 2 {
        eprintln!("usage: voxtral_persist <model-dir> <pile-path>");
        eprintln!("       voxtral_persist --f16-derive <f32-pile> [<f16-pile>]");
        std::process::exit(2);
    }
    let model_dir = Path::new(pos[0]);
    let pile_path = Path::new(pos[1]);

    let t = Instant::now();
    eprintln!("Persisting Voxtral checkpoint from {model_dir:?} → {pile_path:?} ...");
    persist_safetensors_to_pile(model_dir, pile_path, LeafDtype::F32)?;
    let secs = t.elapsed().as_secs_f64();

    // ── gate: pile round-trip bit-exactness vs the source safetensors ──
    eprintln!("Verifying pile leaves against the safetensors (bf16→f32) ...");
    let (_, index, reader) = load_split_index_from_pile(pile_path, "")?;
    anyhow::ensure!(!index.is_empty(), "no leaves found after persist");

    use mary::nn::weight_loader::{get_tensor_f32, read_safetensors_file};
    use safetensors::SafeTensors;
    let bytes = read_safetensors_file(&model_dir.join("model.safetensors"));
    let st = SafeTensors::deserialize(&bytes)?;
    let expected: usize = st
        .names()
        .iter()
        .filter(|k| {
            use safetensors::Dtype;
            matches!(
                st.tensor(k).map(|v| v.dtype()),
                Ok(Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16)
            )
        })
        .count();
    anyhow::ensure!(
        index.len() == expected,
        "pile holds {} leaves, safetensors has {expected} float tensors",
        index.len()
    );

    let (mut checked, mut elems, mut misaligned) = (0usize, 0usize, 0usize);
    for (name, handles) in &index {
        let LeafHandles::F32(dh, _) = handles else {
            anyhow::bail!("{name}: expected an f32 leaf");
        };
        let raw: anybytes::Bytes = reader.get(*dh).map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
        if !(raw.as_ptr() as usize).is_multiple_of(256) {
            eprintln!("  MISALIGNED (ptr % 256 != 0): {name}");
            misaligned += 1;
        }
        let (stored, shape) = read_leaf(&reader, *handles);
        let (want, want_shape) = get_tensor_f32(&st, name);
        anyhow::ensure!(shape == want_shape, "{name}: shape {shape:?} != {want_shape:?}");
        anyhow::ensure!(stored.len() == want.len(), "{name}: len mismatch");
        for (i, (&a, &b)) in stored.iter().zip(want.iter()).enumerate() {
            anyhow::ensure!(
                a.to_bits() == b.to_bits(),
                "{name}[{i}]: pile {a} != source {b}"
            );
        }
        checked += 1;
        elems += stored.len();
    }
    anyhow::ensure!(misaligned == 0, "{misaligned} leaves misaligned — V3 alignment invariant violated");

    let size = std::fs::metadata(pile_path)?.len();
    println!(
        "voxtral persist gate PASSED: {checked} tensors / {elems} elements bit-identical \
         to bf16→f32(source); all 256-aligned. pile {:.2} GiB, persisted in {secs:.1}s.",
        size as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

/// `--f16-derive`: f32 pile → half-width sibling pile, with the source
/// strictly read-only (hard-gated on its byte length) and a full bit-exact
/// re-read gate on the result.
fn f16_derive_mode(src: &Path, dst: &Path) -> anyhow::Result<()> {
    const ENTITY: &str = "ears_f16";
    let src_len_before = std::fs::metadata(src)?.len();
    eprintln!(
        "Deriving f16 sibling {dst:?} from {src:?} ({src_len_before} bytes, READ-ONLY) ..."
    );
    let t = Instant::now();
    let (count, elems) = derive_f16_pile(src, dst, ENTITY)?;
    let secs = t.elapsed().as_secs_f64();
    eprintln!("Derived {count} tensors / {elems} elements in {secs:.1}s.");

    // ── hard gate: the source pile may not have changed by a single byte ──
    let src_len_after = std::fs::metadata(src)?.len();
    anyhow::ensure!(
        src_len_after == src_len_before,
        "SOURCE PILE LENGTH CHANGED ({src_len_before} → {src_len_after} bytes) — \
         the f32 pile is read-only in --f16-derive mode; STOP and investigate"
    );

    // ── gate: sibling round-trip parity for the f16 entity ──
    // Every f16 leaf must equal the runtime cast of the exact f32 leaf
    // (f16::from_f32 — the same rounding the materializing loader applies),
    // cover every source tensor, and sit 256-aligned in the mmap (the V3
    // payload-alignment invariant the fast loader's mmap views rely on).
    eprintln!("Verifying {ENTITY} against the f32 leaves ...");
    let (f16, rest, f16_reader) = load_split_index_from_pile(dst, ENTITY)?;
    anyhow::ensure!(!f16.is_empty(), "no {ENTITY} leaves found after derive");
    anyhow::ensure!(
        rest.is_empty(),
        "sibling pile holds {} non-{ENTITY} leaves — unexpected",
        rest.len()
    );
    let (_, f32_, src_reader) = load_split_index_from_pile(src, "")?;
    anyhow::ensure!(
        f16.len() == f32_.len(),
        "{} f16 leaves vs {} f32 leaves — incomplete coverage",
        f16.len(),
        f32_.len()
    );
    let (mut checked, mut elems, mut misaligned) = (0usize, 0usize, 0usize);
    for (name, handles) in &f16 {
        let (dh, sh) = match handles {
            LeafHandles::F16(d, s) => (*d, *s),
            LeafHandles::F32(..) => anyhow::bail!("{name}: {ENTITY} entity holds an f32 leaf"),
        };
        let bytes: anybytes::Bytes =
            f16_reader.get(dh).map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
        if !(bytes.as_ptr() as usize).is_multiple_of(256) {
            eprintln!("  MISALIGNED (ptr % 256 != 0): {name}");
            misaligned += 1;
        }
        let stored = bytes.view::<[half::f16]>().map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
        let shape = read_shape(&f16_reader, sh);
        let f32_handles = f32_
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{name}: no matching f32 leaf in source"))?;
        let (exact, exact_shape) = read_leaf(&src_reader, *f32_handles);
        anyhow::ensure!(
            shape == exact_shape,
            "{name}: shape mismatch {shape:?} vs {exact_shape:?}"
        );
        anyhow::ensure!(
            stored.len() == exact.len(),
            "{name}: length mismatch {} vs {}",
            stored.len(),
            exact.len()
        );
        for (i, (&h, &x)) in stored.iter().zip(exact.iter()).enumerate() {
            let want = half::f16::from_f32(x);
            anyhow::ensure!(
                h.to_bits() == want.to_bits(),
                "{name}[{i}]: stored {h:?} != cast {want:?} (from f32 {x})"
            );
        }
        checked += 1;
        elems += stored.len();
    }
    anyhow::ensure!(
        misaligned == 0,
        "{misaligned} f16 leaves misaligned — V3 alignment invariant violated"
    );

    let dst_size = std::fs::metadata(dst)?.len();
    println!(
        "{ENTITY} derive gate PASSED: {checked} tensors / {elems} elements bit-identical to \
         f16(f32-leaf); all 256-aligned; source pile untouched ({src_len_before} bytes). \
         Sibling {dst:?} is {dst_size} bytes ({:.2} GiB).",
        dst_size as f64 / (1u64 << 30) as f64
    );
    Ok(())
}
