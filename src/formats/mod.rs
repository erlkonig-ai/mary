//! Non-safetensors weight-file importers. Each submodule decodes a container
//! format to the SAME `(name, f32-data, shape)` tuples that
//! [`crate::ingest::ingest_tensors`] consumes, so a model imported from GGUF or
//! a pickled PyTorch `state_dict` lands in the identical content-addressed graph
//! as its safetensors twin (the model-root id is the pure hash of the f32
//! members, independent of the source format). Import-only: an inference/serve
//! build never compiles a weight-file reader.

pub mod gguf;
pub mod pickle;

use std::path::Path;

/// The weight-file container a model directory ships in. Detected from the file
/// extensions actually present (with a magic-number confirmation for GGUF), so
/// `mary import` picks the right decoder without a caller flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// `*.safetensors` (the existing path, unchanged).
    Safetensors,
    /// A single-file `*.gguf` (llama.cpp container).
    Gguf,
    /// Pickled PyTorch `state_dict` (`pytorch_model*.bin` / `*.pth`).
    Pickle,
}

/// GGUF's 4-byte little-endian magic (`"GGUF"`).
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// True iff `path` starts with the GGUF magic — the confirmation that a
/// `.gguf`-extensioned file really is one before we hand it to the parser.
pub fn is_gguf_file(path: &Path) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut buf).map(|_| buf))
        .map(|b| b == GGUF_MAGIC)
        .unwrap_or(false)
}

/// The weight files a model directory ships, tagged with the detected format.
/// Returns `(format, sorted files)`. Priority when a dir mixes formats:
/// safetensors > gguf > pickle (safetensors is the faithful native path; pickle
/// last since HF repos that also ship safetensors keep a stale `.bin` around).
pub fn detect_format(dir: &Path) -> anyhow::Result<(WeightFormat, Vec<std::path::PathBuf>)> {
    let entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    let has_ext = |ext: &str| -> Vec<std::path::PathBuf> {
        let mut v: Vec<_> = entries
            .iter()
            .filter(|p| p.extension().map(|x| x == ext).unwrap_or(false))
            .cloned()
            .collect();
        v.sort();
        v
    };

    let safet = has_ext("safetensors");
    if !safet.is_empty() {
        return Ok((WeightFormat::Safetensors, safet));
    }
    let gguf: Vec<_> = has_ext("gguf").into_iter().filter(|p| is_gguf_file(p)).collect();
    if !gguf.is_empty() {
        return Ok((WeightFormat::Gguf, gguf));
    }
    // Pickle: HF ships `pytorch_model.bin` or sharded `pytorch_model-0000N-of-...bin`
    // (+ a `.index.json`); a bare `.pth` also counts.
    let mut pickle: Vec<_> = entries
        .iter()
        .filter(|p| {
            let is_bin = p.extension().map(|x| x == "bin").unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("pytorch_model") || n == "model.bin")
                    .unwrap_or(false);
            let is_pth = p.extension().map(|x| x == "pth" || x == "pt").unwrap_or(false);
            is_bin || is_pth
        })
        .cloned()
        .collect();
    pickle.sort();
    if !pickle.is_empty() {
        return Ok((WeightFormat::Pickle, pickle));
    }

    anyhow::bail!(
        "no importable weight files in {dir:?} — expected *.safetensors, *.gguf, \
         or pytorch_model*.bin"
    )
}

/// Extract every float tensor from one weight file of the given format as
/// `(name, f32-data, shape)`. The format-dispatch seam feeding
/// [`crate::ingest::ingest_tensors`].
pub fn extract_tensors(
    fmt: WeightFormat,
    file: &Path,
) -> anyhow::Result<Vec<(String, Vec<f32>, Vec<usize>)>> {
    match fmt {
        WeightFormat::Gguf => gguf::extract_tensors(file),
        WeightFormat::Pickle => pickle::extract_tensors(file),
        WeightFormat::Safetensors => {
            anyhow::bail!("safetensors are ingested via ingest::ingest_members, not formats::extract_tensors")
        }
    }
}
