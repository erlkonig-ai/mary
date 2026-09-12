//! `nomic_pack` — pack a nomic embedding model into a NEW pile of calibrated
//! NVFP4 leaves.
//!
//! Text: `nomic_pack text --model P --corpus F --out O --key K [--calibrate N]
//! [--quantization TAG]`. Vision: `nomic_pack vision --model P --images DIR
//! --out O --key K [--calibrate N] [--quantization TAG] [--source ID]`.
//!
//! Both do the same thing with a different sense: read the f32 model from its
//! pile, run `N` calibration inputs through it with input capture on, pack
//! every linear with `mary::calibrate` (AWQ channel scales, GPTQ feedback,
//! one global scale per tensor), and write `O` as a model pile whose root is
//! labelled `nvfp4-calibrated`. The text pile carries its tokenizer. No score
//! is measured here: `nomic_fp4_probe --model O --quantization
//! nvfp4-calibrated --keep-weights` scores the text artifact against cached
//! f32 vectors, and the vision artifact is validated by its consumer.
//!
//! Recipe and numbers: wiki 88b76d5f, goal dcf9dbaf.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use mary::calibrate::{self, InputStats, Options};
use mary::embed::LocalEmbedder;
use mary::selection::{ModelSelector, TokenizerSelector};

const NOMIC_TEXT_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";
const NOMIC_VISION_MODEL: &str = "nomic-ai/nomic-embed-vision-v1.5";
const PACKED: &str = "nvfp4-calibrated";

/// One text per line: a JSON object with a string `text` field, or the line
/// itself. Empty lines are skipped.
fn read_texts(path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut texts = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('{') {
            let v: serde_json::Value = serde_json::from_str(line)?;
            if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
                if !t.trim().is_empty() {
                    texts.push(t.to_string());
                }
                continue;
            }
        }
        texts.push(line.to_string());
    }
    Ok(texts)
}

/// Image files under `dir`, sorted by path.
fn image_files(dir: &Path) -> Result<Vec<PathBuf>> {
    // Every regular file: the decoder decides what is an image (a pile export
    // names files by content hash, without an extension), and a file that does
    // not decode is skipped and named below.
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    Ok(files)
}

/// `n` items spread evenly over `total`.
fn spread(total: usize, n: usize) -> Vec<usize> {
    let n = n.min(total).max(1);
    let stride = (total / n).max(1);
    (0..n).map(|i| i * stride).collect()
}

fn stats_from_capture(stats: HashMap<String, mary::embed::NomicActStats>) -> HashMap<String, InputStats> {
    stats
        .into_iter()
        .map(|(k, s)| {
            let mean_abs = s.mean_abs();
            (k, InputStats { rows: s.rows, mean_abs })
        })
        .collect()
}

fn pack_and_write(
    mut keymap: calibrate::Keymap,
    inputs: HashMap<String, InputStats>,
    out: &Path,
    key: &Path,
    source: &str,
    quantization: &str,
    tokenizer_json: Option<&[u8]>,
    embeddings: bool,
    append: bool,
) -> Result<()> {
    let key = triblespace::core::signing_key_file::load_existing(key)
        .with_context(|| format!("load signing key {}", key.display()))?;
    let stats_for = |name: &str| inputs.get(&calibrate::nomic_capture_key(name));
    let started = Instant::now();
    let opts = Options { embeddings, ..Options::default() };
    let report = calibrate::pack_keymap(&mut keymap, &[], &stats_for, &opts, &mut |line| {
        eprintln!("  {line}")
    })?;
    eprintln!(
        "packed {} tensors, {} elements, {:.1} MB of packed leaves, in {:.1} s",
        report.tensors,
        report.elements,
        report.packed_bytes as f64 / 1e6,
        started.elapsed().as_secs_f64()
    );
    let root = calibrate::write_packed_pile(out, &key, &keymap, &report.packed, source, quantization, tokenizer_json, append)?;
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    println!("{}: root {root}, {} bytes, {} packed tensors, quantization {quantization}", out.display(), bytes, report.tensors);
    Ok(())
}

fn text(model: &Path, quantization: &str, corpus: &Path, calibrate: usize, out: &Path, key: &Path, embeddings: bool, append: bool) -> Result<()> {
    let snapshot = mary::model_collection::load_model_collection_local_latest(model)
        .with_context(|| format!("open model pile {}", model.display()))?;
    let keymap = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Source { source: NOMIC_TEXT_MODEL, quantization },
    )
    .context("select nomic text weights")?;
    let tokenizer = mary::selection::load_tokenizer_from_graph(
        snapshot.facts(),
        snapshot.store(),
        TokenizerSelector::Name(NOMIC_TEXT_MODEL),
    )
    .context("select nomic tokenizer")?;
    let texts = read_texts(corpus)?;
    anyhow::ensure!(!texts.is_empty(), "no texts in {}", corpus.display());
    let device = mary::embed::default_device();
    let embedder = mary::embed::nomic_text_from_parts(keymap.clone(), tokenizer.clone(), device)?;
    let picks = spread(texts.len(), calibrate);
    let started = Instant::now();
    mary::embed::nomic_activation_capture_start(8);
    for &i in &picks {
        let _ = embedder.embed_document(&texts[i])?;
    }
    let stats = mary::embed::nomic_activation_capture_take();
    drop(embedder);
    eprintln!(
        "captured inputs of {} linears over {} of {} texts in {:.1} s",
        stats.len(),
        picks.len(),
        texts.len(),
        started.elapsed().as_secs_f64()
    );
    let json = tokenizer.to_string(false).map_err(|e| anyhow!("serialise tokenizer: {e}"))?;
    pack_and_write(keymap, stats_from_capture(stats), out, key, NOMIC_TEXT_MODEL, PACKED, Some(json.as_bytes()), embeddings, append)
}

fn vision(model: &Path, source: &str, quantization: &str, images: &Path, calibrate: usize, out: &Path, key: &Path, embeddings: bool, append: bool) -> Result<()> {
    let snapshot = mary::model_collection::load_model_collection_local_latest(model)
        .with_context(|| format!("open model pile {}", model.display()))?;
    let keymap = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Source { source, quantization },
    )
    .context("select nomic vision weights")?;
    let files = image_files(images)?;
    anyhow::ensure!(!files.is_empty(), "no images under {}", images.display());
    let device = mary::embed::default_device();
    let embedder = mary::embed::load_nomic_vision_from_keymap(keymap.clone(), device)?;
    let picks = spread(files.len(), calibrate);
    let started = Instant::now();
    mary::embed::nomic_activation_capture_start(8);
    let mut used = 0usize;
    for &i in &picks {
        let bytes = std::fs::read(&files[i]).with_context(|| format!("read {}", files[i].display()))?;
        match embedder.embed_image(&bytes) {
            Ok(_) => used += 1,
            Err(e) => eprintln!("  skip {}: {e}", files[i].display()),
        }
    }
    let stats = mary::embed::nomic_activation_capture_take();
    drop(embedder);
    anyhow::ensure!(used > 0, "no image could be embedded");
    eprintln!(
        "captured inputs of {} linears over {used} of {} images in {:.1} s",
        stats.len(),
        files.len(),
        started.elapsed().as_secs_f64()
    );
    pack_and_write(keymap, stats_from_capture(stats), out, key, source, PACKED, None, embeddings, append)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
    };
    let usage = "usage: nomic_pack text --model P --corpus F --out O --key K [--calibrate N] [--quantization TAG] [--embeddings] [--append]
       nomic_pack vision --model P --images DIR --out O --key K [--calibrate N] [--quantization TAG] [--source ID] [--embeddings] [--append]";
    let model = || flag("--model").map(PathBuf::from).ok_or_else(|| anyhow!("--model\n{usage}"));
    let out = || flag("--out").map(PathBuf::from).ok_or_else(|| anyhow!("--out\n{usage}"));
    let key = || flag("--key").map(PathBuf::from).ok_or_else(|| anyhow!("--key\n{usage}"));
    let calibrate: usize = flag("--calibrate").map(|s| s.parse()).transpose()?.unwrap_or(1024);
    let quantization = flag("--quantization").unwrap_or_else(|| mary::persist::QUANTIZATION_NATIVE.to_string());
    let embeddings = args.iter().any(|a| a == "--embeddings");
    let append = args.iter().any(|a| a == "--append");
    match args.first().map(String::as_str) {
        Some("text") => {
            let corpus = flag("--corpus").map(PathBuf::from).ok_or_else(|| anyhow!("--corpus\n{usage}"))?;
            text(&model()?, &quantization, &corpus, calibrate, &out()?, &key()?, embeddings, append)
        }
        Some("vision") => {
            let images = flag("--images").map(PathBuf::from).ok_or_else(|| anyhow!("--images\n{usage}"))?;
            let source = flag("--source").unwrap_or_else(|| NOMIC_VISION_MODEL.to_string());
            vision(&model()?, &source, &quantization, &images, calibrate, &out()?, &key()?, embeddings, append)
        }
        _ => Err(anyhow!("{usage}")),
    }
}
