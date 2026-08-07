//! Persist BOTH Qwen3-TTS checkpoints into a REAL on-disk TribleSpace pile:
//! the base model (talker + code predictor + speaker encoder,
//! `model.safetensors`, bf16 → f32 leaves — exact) and the codec
//! (`speech_tokenizer/model.safetensors`, f32) — PLUS the half-width
//! `talker_f16` entity: the talker's GPU tensors re-persisted as f16 leaves
//! (`data_f16`), the dtype the production f16 talker runs at, which the fast
//! load in `mary::speak` uploads to the Metal GPU at native width.
//! f32→f16 is the exact same double-rounding the materializing loader performs
//! at load time (bf16 → f32 → f16), so the fast-loaded talker is bit-identical
//! to the old cast-on-load path — gated below by re-reading both entities.
//!
//! After this runs, the pile file is the durable, self-contained weight store
//! `mary::speak` loads from — the `/tmp` safetensors are no longer needed.
//! Tensor names don't collide across the two f32 checkpoints
//! (`talker.*`/`speaker_encoder.*` vs `decoder.*`/`encoder.*`), so the union
//! keymap serves all four components; the f16 variant lives under its OWN entity
//! name and is excluded from f32 loads. The f32 round-trip is bit-identical
//! (see `qwen3tts_pile_test`).
//!
//!   cargo run --release --features speak,import --bin qwen3tts_persist -- \
//!     <model-dir> <pile-path> [--f16-talker-only]
//!
//! `<model-dir>` is the checkpoint dir holding `model.safetensors` with the
//! codec under `speech_tokenizer/`. `--f16-talker-only` skips the (already
//! persisted) f32 checkpoints and just APPENDS the `talker_f16` entity — the
//! pile is append-only, so upgrading an existing pile is exactly this.

#[cfg(target_os = "macos")]
mod imp {

    use mary::ingest::{read_leaf, LeafDtype, LeafHandles};
    use mary::persist::{
        load_split_index_from_pile, persist_safetensors_file_filtered_to_pile,
        persist_safetensors_to_pile,
    };
    use std::path::Path;
    use std::time::Instant;
    use triblespace::prelude::BlobStoreGet;

    /// The talker tensors the GPU loads (everything under `talker.` EXCEPT the
    /// code predictor, which runs on the CPU from the exact f32 leaves).
    fn is_gpu_talker_tensor(name: &str) -> bool {
        name.starts_with("talker.") && !name.starts_with("talker.code_predictor.")
    }

    pub fn run() -> anyhow::Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let f16_only = args.iter().any(|a| a == "--f16-talker-only");
        let fold_derive = args.iter().any(|a| a == "--fold-derive");
        let pos: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();

        // ── `--fold-derive <src-pile> [<dst-pile>]`: derive the FOLDED zero-copy
        // sibling (`<stem>_folded_f16.pile`) for the raw talker lane. Loads the
        // production fused-f16 talker, reads back its fold-transformed GPU
        // tensors, writes them to a NEW sibling pile, and gates the result
        // bit-for-bit through the zero-copy alias. The source pile is read-only.
        #[cfg(target_os = "macos")]
        if fold_derive {
            if pos.is_empty() {
                eprintln!("usage: qwen3tts_persist --fold-derive <src-pile> [<dst-pile>]");
                std::process::exit(2);
            }
            let src = Path::new(pos[0]);
            let dst = pos
                .get(1)
                .map(|p| std::path::PathBuf::from(p.as_str()))
                .unwrap_or_else(|| mary::persist::qwen3tts_folded_sibling_path(src));
            let t = Instant::now();
            let (count, bytes) = mary::persist::derive_qwen3tts_folded_pile(src, &dst)?;
            eprintln!(
                "fold-derive done: {count} tensors / {:.2} GiB → {dst:?} in {:.1}s",
                bytes as f64 / (1u64 << 30) as f64,
                t.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        #[cfg(not(target_os = "macos"))]
        if fold_derive {
            anyhow::bail!("--fold-derive requires macOS (Metal zero-copy alias)");
        }

        if pos.len() < 2 {
            eprintln!(
                "usage: qwen3tts_persist <model-dir> <pile-path> [--f16-talker-only]\n       \
             qwen3tts_persist --fold-derive <src-pile> [<dst-pile>]"
            );
            std::process::exit(2);
        }
        let model_dir = Path::new(pos[0]);
        let pile_path = Path::new(pos[1]);

        let t = Instant::now();
        if !f16_only {
            eprintln!("Persisting base checkpoint from {model_dir:?} → {pile_path:?} ...");
            persist_safetensors_to_pile(model_dir, pile_path, LeafDtype::F32)?;
            let codec_dir = model_dir.join("speech_tokenizer");
            eprintln!("Persisting codec checkpoint from {codec_dir:?} → {pile_path:?} ...");
            persist_safetensors_to_pile(&codec_dir, pile_path, LeafDtype::F32)?;
        }
        eprintln!("Persisting f16 talker variant (entity 'talker_f16') → {pile_path:?} ...");
        persist_safetensors_file_filtered_to_pile(
            &model_dir.join("model.safetensors"),
            "talker_f16",
            pile_path,
            LeafDtype::F16,
            is_gpu_talker_tensor,
        )?;
        let secs = t.elapsed().as_secs_f64();

        // ── gate: pile round-trip parity for the f16 entity ──
        // Every f16 leaf must equal the runtime cast of the exact f32 leaf
        // (f16::from_f32 — the same rounding the materializing loader applies),
        // and must sit 256-aligned in the mmap (the V3 payload-alignment
        // invariant the fast loader's mmap views rely on).
        eprintln!("Verifying talker_f16 against the f32 leaves ...");
        let (f16, f32_, reader) = load_split_index_from_pile(pile_path, "talker_f16")?;
        anyhow::ensure!(!f16.is_empty(), "no talker_f16 leaves found after persist");
        let (mut checked, mut elems, mut misaligned) = (0usize, 0usize, 0usize);
        for (name, handles) in &f16 {
            let (dh, sh) = match handles {
                LeafHandles::F16(d, s) => (*d, *s),
                LeafHandles::F32(..) => {
                    anyhow::bail!("{name}: talker_f16 entity holds an f32 leaf")
                }
            };
            let bytes: anybytes::Bytes = reader
                .get(dh)
                .map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
            if bytes.as_ptr() as usize % 256 != 0 {
                eprintln!("  MISALIGNED (ptr % 256 != 0): {name}");
                misaligned += 1;
            }
            let stored = bytes
                .view::<[half::f16]>()
                .map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
            let f32_handles = f32_
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("{name}: no matching f32 leaf"))?;
            let (exact, shape) = read_leaf(&reader, *f32_handles);
            let (_, f16_shape) = read_leaf(&reader, LeafHandles::F16(dh, sh));
            anyhow::ensure!(
                shape == f16_shape,
                "{name}: shape mismatch {shape:?} vs {f16_shape:?}"
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
        println!(
            "talker_f16 parity gate PASSED: {checked} tensors / {elems} elements bit-identical to \
         f16(f32-leaf); all 256-aligned."
        );

        let size = std::fs::metadata(pile_path)?.len();
        println!(
            "Persisted in {:.1}s. Pile file {pile_path:?} is {} bytes ({:.2} GiB).",
            secs,
            size,
            size as f64 / (1u64 << 30) as f64
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    imp::run()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen3tts_persist: macOS-only lane (folded-sibling derivation).");
    std::process::exit(2);
}
