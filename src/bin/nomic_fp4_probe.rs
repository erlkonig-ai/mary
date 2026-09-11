//! How much does FP4 cost an embedding index? A night probe, 2026-09-11.
//!
//! Two questions JP asked: could the nomic text model run in NVFP4, and could
//! the output vectors be NVFP4 too, plain or as the sixteen-lane fractional
//! code the Inkling carrier pilot used. This binary measures both on our own
//! prose, the wiki fragments and journal summaries of the self pile, against
//! the f32 model and f32 vectors, by top-10 neighbour recall and cosine error.
//!
//! ```text
//! nomic_fp4_probe extract --pile <self.pile> --wiki <handle> --journal <handle> --out corpus.jsonl
//! nomic_fp4_probe probe --model <nomic_text.pile> --corpus corpus.jsonl [--queries 200]
//! ```
//!
//! Everything numerical about NVFP4 here is a CPU reference: E2M1 codes
//! {0, 0.5, 1, 1.5, 2, 3, 4, 6}, E4M3 block scales over sixteen elements, one
//! f32 scale per vector or per weight row. It is a measurement of rounding,
//! not of any kernel.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};

// ── NVFP4 reference arithmetic ───────────────────────────────────────────

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const BLOCK: usize = 16;
const E4M3_MAX: f32 = 448.0;

/// Nearest E2M1 magnitude code for `m >= 0`, ties to the even code.
fn e2m1_code(m: f32) -> usize {
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for (code, value) in E2M1.iter().enumerate() {
        let err = (m - value).abs();
        if err < best_err || (err == best_err && code % 2 == 0) {
            best = code;
            best_err = err;
        }
    }
    best
}

/// Round a positive scale to E4M3: three mantissa bits, exponents 2^-6..2^8,
/// subnormals in steps of 2^-9, saturating at 448.
fn e4m3(v: f32) -> f32 {
    if !(v > 0.0) {
        return 0.0;
    }
    if v >= E4M3_MAX {
        return E4M3_MAX;
    }
    let e = v.log2().floor().max(-6.0);
    let step = 2f32.powf(e - 3.0);
    let rounded = (v / step).round() * step;
    rounded.min(E4M3_MAX)
}

/// One NVFP4 vector: per-vector f32 scale, E4M3 block scales, E2M1 codes.
/// Returns the dequantized values and, per element, the lower and upper E2M1
/// neighbours in the same block scaling (for the sixteen-lane code).
fn nvfp4_quantize(x: &[f32]) -> (Vec<f32>, Vec<(f32, f32)>) {
    let absmax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let tensor_scale = if absmax > 0.0 { absmax / (6.0 * E4M3_MAX) } else { 1.0 };
    let mut out = Vec::with_capacity(x.len());
    let mut neighbours = Vec::with_capacity(x.len());
    for block in x.chunks(BLOCK) {
        let block_max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = e4m3(block_max / (6.0 * tensor_scale));
        let unit = scale * tensor_scale;
        for &v in block {
            if unit == 0.0 {
                out.push(0.0);
                neighbours.push((0.0, 0.0));
                continue;
            }
            let m = (v.abs() / unit).min(6.0);
            let code = e2m1_code(m);
            let q = E2M1[code] * unit * v.signum();
            out.push(q);
            // Neighbours bracketing m on the E2M1 ladder.
            let lo = E2M1.iter().rev().find(|&&l| l <= m).copied().unwrap_or(0.0);
            let hi = E2M1.iter().find(|&&h| h >= m).copied().unwrap_or(6.0);
            neighbours.push((lo * unit * v.signum(), hi * unit * v.signum()));
        }
    }
    (out, neighbours)
}

/// Sixteen FP4 lanes: for each element, k of sixteen lanes take the upper
/// E2M1 neighbour and the rest the lower one, so the lane mean lands within
/// one sixteenth of the step. Storage is 16 x 4 bits per element.
fn fp4_lanes16(x: &[f32]) -> Vec<f32> {
    let (_, neighbours) = nvfp4_quantize(x);
    x.iter()
        .zip(neighbours)
        .map(|(&v, (lo, hi))| {
            if hi == lo {
                return lo;
            }
            let f = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            let k = (f * 16.0).round();
            lo + (hi - lo) * k / 16.0
        })
        .collect()
}

/// Two-stage residual NVFP4: quantize, then quantize the residual with its
/// own scales, and add. This is the shape of mary's QuantizedRow, done here
/// in the same reference arithmetic so the comparison is like for like.
fn nvfp4_two_stage(x: &[f32]) -> Vec<f32> {
    let (first, _) = nvfp4_quantize(x);
    let residual: Vec<f32> = x.iter().zip(&first).map(|(v, q)| v - q).collect();
    let (second, _) = nvfp4_quantize(&residual);
    first.iter().zip(second).map(|(a, b)| a + b).collect()
}

fn int8(x: &[f32]) -> Vec<f32> {
    let absmax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if absmax == 0.0 {
        return x.to_vec();
    }
    let unit = absmax / 127.0;
    x.iter().map(|v| (v / unit).round() * unit).collect()
}

fn binary(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect()
}

fn l2_normalize(x: &mut [f32]) {
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 0.0 {
        for v in x.iter_mut() {
            *v /= n;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ── corpus ───────────────────────────────────────────────────────────────

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn json_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    out.push(ch);
                }
            }
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

/// Read `{"id": "...", "source": "...", "text": "..."}` lines written by `extract`.
fn read_corpus(path: &Path) -> Result<Vec<(String, String, String)>> {
    let mut rows = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let field = |name: &str| -> Option<String> {
            let key = format!("\"{name}\": \"");
            let start = line.find(&key)? + key.len();
            let rest = &line[start..];
            // The value ends at the first unescaped quote.
            let mut end = 0;
            let bytes = rest.as_bytes();
            while end < bytes.len() {
                if bytes[end] == b'\\' {
                    end += 2;
                    continue;
                }
                if bytes[end] == b'"' {
                    break;
                }
                end += 1;
            }
            Some(json_unescape(&rest[..end]))
        };
        if let (Some(id), Some(source), Some(text)) = (field("id"), field("source"), field("text")) {
            rows.push((id, source, text));
        }
    }
    Ok(rows)
}

fn extract(pile: &Path, sources: &[(String, [u8; 16], String)], out: &Path, max_chars: usize) -> Result<()> {
    use anybytes::View;
    use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
    use triblespace::core::blob::encodings::utf8string::UTF8String;
    use triblespace::core::blob::{Blob, IntoBlob, TryFromBlob};
    use triblespace::core::collection::records::CollectionHandle;
    use triblespace::core::collection::{Collection, CollectionSnapshotExt};
    use triblespace::core::inline::encodings::hash::Handle;
    use triblespace::core::inline::Inline;
    use triblespace::core::repo::pile::Pile;
    use triblespace::core::repo::{BlobStoreGet, SnapshotSource};
    use triblespace::core::trible::{TribleSet, TRIBLE_LEN};

    let mut pile = Pile::open(pile).map_err(|e| anyhow!("open pile: {e:?}"))?;
    let snapshot = pile.snapshot().map_err(|e| anyhow!("snapshot: {e:?}"))?;
    let mut lines = Vec::new();
    for (name, attribute, handle_hex) in sources {
        let mut raw = [0u8; 32];
        hex_decode(handle_hex, &mut raw)?;
        let handle = CollectionHandle::new(raw);
        let collection: Collection<SimpleArchive> = Collection::open(&snapshot, handle)
            .map_err(|e| anyhow!("open collection {name}: {e}"))?;
        let support = collection
            .admitted(&snapshot)
            .map_err(|e| anyhow!("admitted support of {name}: {e:?}"))?;
        let facts: TribleSet = snapshot
            .collection_exact(collection, &support)
            .map_err(|e| anyhow!("attach {name}: {e:?}"))?
            .view::<TribleSet>()
            .map_err(|e| anyhow!("view {name}: {e:?}"))?;
        let blob: Blob<SimpleArchive> = facts.to_blob();
        let mut seen = std::collections::BTreeSet::new();
        let mut count = 0usize;
        for trible in blob.bytes.as_ref().chunks_exact(TRIBLE_LEN) {
            if trible[16..32] != attribute[..] {
                continue;
            }
            let value: [u8; 32] = trible[32..].try_into().unwrap();
            if !seen.insert(value) {
                continue;
            }
            let text_blob: Blob<UTF8String> = match snapshot.get(Inline::<Handle<UTF8String>>::new(value)) {
                Ok(blob) => blob,
                Err(_) => continue,
            };
            let text: View<str> = match View::try_from_blob(text_blob) {
                Ok(view) => view,
                Err(_) => continue,
            };
            let mut text: String = text.to_string();
            if text.trim().len() < 40 {
                continue;
            }
            if text.len() > max_chars {
                let mut cut = max_chars;
                while !text.is_char_boundary(cut) {
                    cut -= 1;
                }
                text.truncate(cut);
            }
            let id: String = value.iter().map(|b| format!("{b:02x}")).collect();
            lines.push(format!(
                "{{\"id\": \"{id}\", \"source\": \"{name}\", \"text\": \"{}\"}}",
                json_escape(&text)
            ));
            count += 1;
        }
        eprintln!("{name}: {count} texts");
    }
    fs::write(out, lines.join("\n") + "\n")?;
    eprintln!("wrote {} texts to {}", lines.len(), out.display());
    Ok(())
}

fn hex_decode(hex: &str, out: &mut [u8]) -> Result<()> {
    let hex = hex.trim().trim_start_matches("blake3:");
    if hex.len() != out.len() * 2 {
        return Err(anyhow!("expected {} hex digits, got {}", out.len() * 2, hex.len()));
    }
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)?;
    }
    Ok(())
}

// ── model ────────────────────────────────────────────────────────────────

const NOMIC_TEXT_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";

type Keymap = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn load_parts(model_pile: &Path) -> Result<(Keymap, tokenizers::Tokenizer)> {
    use mary::selection::{ModelSelector, TokenizerSelector};
    let snapshot = mary::model_collection::load_model_collection_local_latest(model_pile)
        .with_context(|| format!("open model pile {}", model_pile.display()))?;
    let keymap = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Source {
            source: NOMIC_TEXT_MODEL,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .context("select native nomic text weights")?;
    let tokenizer = mary::selection::load_tokenizer_from_graph(
        snapshot.facts(),
        snapshot.store(),
        TokenizerSelector::Name(NOMIC_TEXT_MODEL),
    )
    .context("select nomic tokenizer")?;
    Ok((keymap, tokenizer))
}

/// Fake-quantize every two-dimensional weight that is not an embedding table
/// or a norm: rows are output features, blocks of sixteen run along the input
/// features, one f32 scale per row. Returns how many tensors were touched.
fn fake_nvfp4_weights(keymap: &mut Keymap) -> (usize, usize) {
    let mut tensors = 0usize;
    let mut elements = 0usize;
    for (name, (data, shape)) in keymap.iter_mut() {
        let lower = name.to_ascii_lowercase();
        if shape.len() != 2 || !lower.contains("weight") {
            continue;
        }
        if lower.contains("embed") || lower.contains("norm") || lower.contains("ln") {
            continue;
        }
        let cols = shape[1];
        if cols % BLOCK != 0 {
            continue;
        }
        for row in data.chunks_mut(cols) {
            let (q, _) = nvfp4_quantize(row);
            row.copy_from_slice(&q);
        }
        tensors += 1;
        elements += data.len();
    }
    (tensors, elements)
}

fn recall_at_k(baseline: &[usize], candidate: &[usize]) -> f32 {
    let hits = candidate.iter().filter(|c| baseline.contains(c)).count();
    hits as f32 / baseline.len().max(1) as f32
}

fn top_k(query: &[f32], docs: &[Vec<f32>], exclude: Option<usize>, k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != exclude)
        .map(|(i, d)| (i, dot(query, d)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn probe(model_pile: &Path, corpus: &Path, queries: usize) -> Result<()> {
    let rows = read_corpus(corpus)?;
    if rows.len() < 20 {
        return Err(anyhow!("corpus has only {} texts", rows.len()));
    }
    eprintln!("corpus: {} texts", rows.len());

    let (keymap, tokenizer) = load_parts(model_pile)?;
    let device = mary::embed::default_device();
    let started = Instant::now();
    let f32_model = mary::embed::nomic_text_from_parts(keymap.clone(), tokenizer.clone(), device.clone())?;
    eprintln!("f32 model built in {:.1} s", started.elapsed().as_secs_f64());

    let started = Instant::now();
    let mut docs: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
    for (_, _, text) in &rows {
        let mut v = f32_model.embed_document(text)?;
        l2_normalize(&mut v);
        docs.push(v);
    }
    let dim = docs[0].len();
    eprintln!("embedded {} documents ({dim}-d) in {:.1} s", docs.len(), started.elapsed().as_secs_f64());

    let queries = queries.min(rows.len());
    let step = rows.len() / queries;
    let query_ids: Vec<usize> = (0..queries).map(|i| i * step).collect();
    let mut qvecs: Vec<Vec<f32>> = Vec::with_capacity(queries);
    for &i in &query_ids {
        let mut v = f32_model.embed_query(&rows[i].2)?;
        l2_normalize(&mut v);
        qvecs.push(v);
    }
    let baseline: Vec<Vec<usize>> = query_ids
        .iter()
        .zip(&qvecs)
        .map(|(&i, q)| top_k(q, &docs, Some(i), 10))
        .collect();

    println!("nomic-embed-text-v1.5 over {} texts, {} queries, {dim}-d, recall@10 against f32 model and f32 vectors", rows.len(), queries);
    println!("{:<34} {:>10} {:>12} {:>14}", "variant", "bits/dim", "recall@10", "mean |dcos|");

    let report = |name: &str, bits: f32, stored: &[Vec<f32>], qs: &[Vec<f32>]| {
        let mut recall = 0f32;
        let mut cos_err = 0f64;
        let mut pairs = 0usize;
        for ((&i, q), base) in query_ids.iter().zip(qs).zip(&baseline) {
            let got = top_k(q, stored, Some(i), 10);
            recall += recall_at_k(base, &got);
            for &j in base {
                let exact = dot(&qvecs[query_ids.iter().position(|&x| x == i).unwrap()], &docs[j]);
                let approx = dot(q, &stored[j]);
                cos_err += (exact - approx).abs() as f64;
                pairs += 1;
            }
        }
        println!(
            "{:<34} {:>10} {:>11.1}% {:>14.5}",
            name,
            format!("{bits:.1}"),
            recall / queries as f32 * 100.0,
            cos_err / pairs.max(1) as f64
        );
    };

    report("f32 vectors", 32.0, &docs, &qvecs);
    let v: Vec<Vec<f32>> = docs.iter().map(|d| { let mut q = nvfp4_quantize(d).0; l2_normalize(&mut q); q }).collect();
    report("NVFP4, one stage", 4.5, &v, &qvecs);
    let v: Vec<Vec<f32>> = docs.iter().map(|d| { let mut q = nvfp4_two_stage(d); l2_normalize(&mut q); q }).collect();
    report("NVFP4, two-stage residual", 9.0, &v, &qvecs);
    let v: Vec<Vec<f32>> = docs.iter().map(|d| { let mut q = fp4_lanes16(d); l2_normalize(&mut q); q }).collect();
    report("FP4, sixteen lanes (fractional)", 64.5, &v, &qvecs);
    let v: Vec<Vec<f32>> = docs.iter().map(|d| { let mut q = int8(d); l2_normalize(&mut q); q }).collect();
    report("int8, per-vector scale", 8.0, &v, &qvecs);
    let v: Vec<Vec<f32>> = docs.iter().map(|d| { let mut q = binary(d); l2_normalize(&mut q); q }).collect();
    let qb: Vec<Vec<f32>> = qvecs.iter().map(|q| { let mut b = binary(q); l2_normalize(&mut b); b }).collect();
    report("binary, both sides", 1.0, &v, &qb);
    // Query side quantized too, for the symmetric NVFP4 case an index would use.
    let v: Vec<Vec<f32>> = docs.iter().map(|d| { let mut q = nvfp4_two_stage(d); l2_normalize(&mut q); q }).collect();
    let q2: Vec<Vec<f32>> = qvecs.iter().map(|q| { let mut b = nvfp4_two_stage(q); l2_normalize(&mut b); b }).collect();
    report("NVFP4 two-stage, both sides", 9.0, &v, &q2);

    // The model itself with NVFP4 weights.
    let mut quantized = keymap;
    let (tensors, elements) = fake_nvfp4_weights(&mut quantized);
    eprintln!("fake-quantized {tensors} weight tensors, {elements} elements, to NVFP4");
    let started = Instant::now();
    let q_model = mary::embed::nomic_text_from_parts(quantized, tokenizer, device)?;
    let mut qdocs: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
    for (_, _, text) in &rows {
        let mut v = q_model.embed_document(text)?;
        l2_normalize(&mut v);
        qdocs.push(v);
    }
    let mut qqueries: Vec<Vec<f32>> = Vec::with_capacity(queries);
    for &i in &query_ids {
        let mut v = q_model.embed_query(&rows[i].2)?;
        l2_normalize(&mut v);
        qqueries.push(v);
    }
    eprintln!("NVFP4-weight model embedded everything in {:.1} s", started.elapsed().as_secs_f64());
    let same_text_cos: f64 = docs.iter().zip(&qdocs).map(|(a, b)| dot(a, b) as f64).sum::<f64>() / docs.len() as f64;
    println!();
    println!("model with NVFP4 linear weights ({tensors} tensors): mean cosine to the f32 model's vector of the same text {same_text_cos:.5}");
    report("NVFP4 weights, f32 vectors", 32.0, &qdocs, &qqueries);
    let v: Vec<Vec<f32>> = qdocs.iter().map(|d| { let mut q = nvfp4_two_stage(d); l2_normalize(&mut q); q }).collect();
    let q2: Vec<Vec<f32>> = qqueries.iter().map(|q| { let mut b = nvfp4_two_stage(q); l2_normalize(&mut b); b }).collect();
    report("NVFP4 weights + two-stage vectors", 9.0, &v, &q2);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
    };
    match args.first().map(String::as_str) {
        Some("extract") => {
            let pile = PathBuf::from(flag("--pile").ok_or_else(|| anyhow!("--pile"))?);
            let out = PathBuf::from(flag("--out").unwrap_or_else(|| "corpus.jsonl".into()));
            let max_chars: usize = flag("--max-chars").map(|s| s.parse()).transpose()?.unwrap_or(4000);
            let mut sources = Vec::new();
            let mut attr = [0u8; 16];
            if let Some(h) = flag("--wiki") {
                hex_decode("6DBBE746B7DD7A4793CA098AB882F553", &mut attr)?;
                sources.push(("wiki".to_string(), attr, h));
            }
            if let Some(h) = flag("--journal") {
                hex_decode("3292CF0B3B6077991D8ECE6E2973D4B6", &mut attr)?;
                sources.push(("journal".to_string(), attr, h));
            }
            extract(&pile, &sources, &out, max_chars)
        }
        Some("probe") => {
            let model = PathBuf::from(flag("--model").ok_or_else(|| anyhow!("--model"))?);
            let corpus = PathBuf::from(flag("--corpus").unwrap_or_else(|| "corpus.jsonl".into()));
            let queries: usize = flag("--queries").map(|s| s.parse()).transpose()?.unwrap_or(200);
            probe(&model, &corpus, queries)
        }
        _ => Err(anyhow!("usage: nomic_fp4_probe extract --pile P --wiki H --journal H --out F | probe --model P --corpus F [--queries N]")),
    }
}
