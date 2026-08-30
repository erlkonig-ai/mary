//! Gate: Inkling's BPE tokenizer survives the trip through the trible graph
//! unchanged, in memory AND through a real pile on disk.
//!
//! `save_tokenizer_json` has been in the tree since 2026-07-16 with unit tests
//! and no on-disk caller. This is the caller, and it is deliberately shaped
//! like `spm_graph_roundtrip`: gate the schema in memory first, then gate the
//! same battery through commit / push / reopen / blob-resolution, because those
//! are four things the in-memory pass cannot see.
//!
//! # What is checked, and why it is ENCODING rather than field equality
//!
//! A tokenizer that is 99% right silently corrupts every prompt. So the
//! comparison is behavioural: the same strings through the tokenizer built from
//! `tokenizer.json` and the tokenizer built from the graph must produce the same
//! ids, and the same ids must decode to the same text. Field-by-field equality
//! would pass on a reconstruction whose `use_regex` differs, and `use_regex` is
//! precisely what was wrong.
//!
//! Two of Inkling's settings are not defaults and both change the token stream:
//!
//!   * `model.ignore_merges: true` — a piece already in the vocab is emitted
//!     whole rather than rebuilt from merges;
//!   * `pre_tokenizer[1].use_regex: false` — the ByteLevel step must NOT
//!     re-apply GPT-2's split on top of the Split that precedes it.
//!
//! Neither was persisted before today. `--mutate` puts each back the way it
//! was, so the gate can be watched failing on exactly the properties it claims
//! to cover — a check that has never failed is not evidence.
//!
//!   inkling_tokenizer_gate <tokenizer.json> [pile --signing-key <existing-key>]
//!                          [--mutate ignore-merges|use-regex|drop-merge]
//!
//! Build: `--features tokenizer` (or any build that pulls it in).

use std::path::Path;

use anyhow::{Context, Result};
use triblespace::core::signing_key_file;
use triblespace::prelude::*;

/// Strings chosen to exercise every lane byte-level BPE has: ASCII, contractions
/// (the `(?i:'s|'t|…)` alternation in the pre-tokenizer's regex), digit runs
/// (`\p{N}{1,3}`, which splits 1234567890 into groups of three), leading and
/// repeated whitespace (the `\s+(?!\S)` / `\s+` tail), newlines, punctuation
/// clusters, CJK and emoji (multi-byte, so the byte-level alphabet is
/// exercised), and code-shaped text.
const PROBES: &[&str] = &[
    "Hello, how are you?",
    "The capital of France is",
    "You are a helpful assistant.",
    "  leading and trailing  ",
    "1234567890",
    "3.14159 and 0.0001 and 1e-06",
    "café naïve résumé",
    "日本語のテキスト",
    "emoji 🙂 and more 🎉",
    "punctuation!?;:'\"()[]{}",
    "it's, they're, we've, I'd, don't, he'll",
    "a",
    "",
    "\n\n\n",
    "tabs\tand\nnewlines\r\nmixed",
    "fn main() { let x: Vec<u8> = vec![1, 2, 3]; }",
    "The quick brown fox jumps over the lazy dog.",
    // Special tokens, which are added tokens rather than vocab entries here —
    // the case `build_tokenizer`'s comment assumed away ("all our added tokens
    // are also vocab entries"). All 60 of Inkling's are above the vocab.
    "<|endoftext|>",
    "<|start|>user<|message|>hi<|end|>",
    // The string behind /tmp/prompt.ids, so the gate covers what the forward-pass
    // verification actually runs on -- and the leading-space variant beside it,
    // because they are DIFFERENT first tokens (976 against 623). This comment
    // named the wrong one of the two until `inkling_encode` re-encoded both and
    // `cmp`d them against the file, which is the general hazard: a label that
    // describes an artefact nobody re-derives drifts and stays plausible.
    "The capital of France is",
    " The capital of France is",
    // Pre-tokens that are NOT vocab entries, which is the only way to reach the
    // merge table at all on this tokenizer.
    //
    // `ignore_merges: true` returns a pre-token whole the moment the vocab
    // contains it, so every sweep over vocab entries — however exhaustive —
    // short-circuits before a single merge is applied. That is why
    // `--mutate drop-merge` passed 199 998 vocab tokens and 199 998 decoded ids
    // without a murmur. A run long enough that no single token covers it has to
    // be built up by merges, and rank 0 (`Ġ` + `Ġ`) is where a space run starts.
    "                                        ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",
    "----------------------------------------",
];

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let json_path = args.next().context(
        "usage: inkling_tokenizer_gate <tokenizer.json> \
             [pile --signing-key <existing-key>] [--mutate WHAT]",
    )?;
    let mut pile: Option<String> = None;
    let mut mutate: Option<String> = None;
    let mut signing_key_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--mutate" => mutate = Some(args.next().context("--mutate needs a name")?),
            "--signing-key" => {
                signing_key_path = Some(args.next().context("--signing-key needs a path")?)
            }
            other => {
                if pile.is_none() {
                    pile = Some(other.to_string());
                } else {
                    anyhow::bail!("unexpected argument {other:?}");
                }
            }
        }
    }
    anyhow::ensure!(
        pile.is_some() || signing_key_path.is_none(),
        "--signing-key requires a destination pile"
    );

    println!("tokenizer: {json_path}");
    let raw = std::fs::read(&json_path)?;
    let v: serde_json::Value = serde_json::from_slice(&raw)?;
    println!(
        "parsed   : {} model, {} vocab, {} merges, {} added tokens",
        v["model"]["type"].as_str().unwrap_or("?"),
        v["model"]["vocab"]
            .as_object()
            .map(|o| o.len())
            .unwrap_or(0),
        v["model"]["merges"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        v["added_tokens"].as_array().map(|a| a.len()).unwrap_or(0),
    );

    // The REFERENCE is always the untouched file. A mutation is applied only to
    // what gets ingested, so the two sides really are "what the checkpoint says"
    // against "what the pile reconstructs".
    let from_file = tokenizers::Tokenizer::from_file(Path::new(&json_path))
        .map_err(|e| anyhow::anyhow!("load {json_path}: {e}"))?;

    let ingest_bytes = match mutate.as_deref() {
        None => raw.clone(),
        Some(what) => {
            let mut m = v.clone();
            match what {
                "ignore-merges" => {
                    m["model"]["ignore_merges"] = serde_json::Value::Bool(false);
                }
                "use-regex" => {
                    let pts = m["pre_tokenizer"]["pretokenizers"]
                        .as_array_mut()
                        .context("--mutate use-regex wants a Sequence pre-tokenizer")?;
                    for p in pts.iter_mut() {
                        if p["type"] == "ByteLevel" {
                            p["use_regex"] = serde_json::Value::Bool(true);
                        }
                    }
                }
                "drop-merge" => {
                    let ms = m["model"]["merges"].as_array_mut().context("no merges")?;
                    // The FIRST merge, not the last: BPE applies merges in rank
                    // order, so dropping a low rank changes far more strings
                    // than dropping the rarest one — a mutation nobody notices
                    // is not a demonstration.
                    ms.remove(0);
                }
                // Half the merge table. The point is diagnostic rather than
                // adversarial: `drop-merge` passing every check could mean rank
                // 0 is inert, or it could mean the merges never reach the
                // rebuilt model at all — and those are a curiosity and a
                // catastrophe respectively. Losing 223 094 of them cannot be
                // inert, so if this passes too, the reconstruction is not using
                // the merge table and every PASS above is worthless.
                "drop-merges-half" => {
                    let ms = m["model"]["merges"].as_array_mut().context("no merges")?;
                    let keep = ms.len() / 2;
                    ms.truncate(keep);
                }
                other => anyhow::bail!(
                    "unknown --mutate {other:?} (ignore-merges | use-regex | drop-merge | drop-merges-half)"
                ),
            }
            println!("MUTATION : {what} — the graph is built from a deliberately wrong config");
            serde_json::to_vec(&m)?
        }
    };

    // ── 1. in-memory graph ──────────────────────────────────────────────────
    let mut blobs = MemoryBlobStore::new();
    let mut counting = mary::tokenizer::CountingBlobs::new(&mut blobs);
    let t0 = std::time::Instant::now();
    let frag = mary::tokenizer::save_tokenizer_json(&ingest_bytes, "inkling-small", &mut counting)
        .map_err(|e| anyhow::anyhow!("ingest: {e}"))?;
    let (mem_puts, mem_nanos) = (counting.puts, counting.nanos);
    let root_id = frag.root().context("tokenizer root")?;
    let tribles = frag.into_facts();
    println!(
        "graph    : {} facts, {mem_puts} blob puts -- {:.2}s inside the puts \
         ({:.0} puts/s, MEMORY store), {:.2}s for the whole ingest",
        tribles.len(),
        mem_nanos as f64 / 1e9,
        mem_puts as f64 * 1e9 / mem_nanos.max(1) as f64,
        t0.elapsed().as_secs_f64(),
    );

    // Discover the node the way a real loader does rather than trusting the
    // fragment root — these disagreed once for SPM, and the failure was silent.
    let tok_id = mary::tokenizer::find_tokenizer(&tribles)
        .context("find_tokenizer must discover the node the loader looks for")?;
    anyhow::ensure!(
        tok_id == root_id,
        "find_tokenizer found a different node than the root"
    );

    let reader = blobs
        .snapshot()
        .map_err(|e| anyhow::anyhow!("snapshot: {e:?}"))?;
    let from_graph = mary::tokenizer::build_tokenizer(&tribles, &reader, tok_id)
        .map_err(|e| anyhow::anyhow!("build from graph: {e}"))?;

    let bad = compare("memory", &from_file, &from_graph, &v)?;
    if bad == 0 {
        println!("\nPASS (in memory) — the graph reconstructs the tokenizer exactly.");
    }

    // ── 2. the same battery through a REAL pile ─────────────────────────────
    let Some(pile) = pile else {
        if bad > 0 {
            println!("\nFAIL (in memory) — {bad} check(s) differ.");
            std::process::exit(1);
        }
        println!("(no pile argument — nothing was written. Pass one to ingest for real.)");
        return Ok(());
    };
    let signing_key_path = signing_key_path
        .as_deref()
        .context("--signing-key <existing-key> is required when writing a pile")?;
    let signing_key = signing_key_file::load_existing(Path::new(signing_key_path))
        .with_context(|| format!("load existing signing key {signing_key_path:?}"))?;

    println!("\n=== ingesting into {pile} ===");
    // A mutation must not be written into a shared pile: the ingest refuses a
    // second tokenizer, so a mutated one would poison the collection
    // permanently.
    if mutate.is_some() {
        let tmp = std::env::temp_dir().join("inkling_tokenizer_mutated.json");
        std::fs::write(&tmp, &ingest_bytes)?;
        println!("(mutated config written to {tmp:?}; ingest it into a SCRATCH pile only)");
    }
    let src = if mutate.is_some() {
        std::env::temp_dir().join("inkling_tokenizer_mutated.json")
    } else {
        Path::new(&json_path).to_path_buf()
    };

    match mary::persist::ingest_hf_tokenizer(Path::new(&pile), &src, "inkling-small", &signing_key)
    {
        Ok(r) => {
            println!("committed: {} facts", r.facts);
            println!(
                "ingest   : {} blob puts in {:.2}s of put time ({:.0} puts/s, ON-DISK pile)",
                r.puts,
                r.put_nanos as f64 / 1e9,
                r.put_rate().unwrap_or(f64::NAN)
            );
            println!(
                "           {:.2}s total, pile grew {:.1} MiB",
                r.total_nanos as f64 / 1e9,
                r.file_growth as f64 / (1u64 << 20) as f64
            );
        }
        Err(e) => {
            println!("ingest declined: {e}");
            println!("(continuing to the read-back gate — an existing graph still has to pass)");
        }
    }

    let from_pile = mary::persist::load_tokenizer_from_pile(Path::new(&pile))
        .context("read the tokenizer back out of the pile")?;
    let bad_pile = compare("pile", &from_file, &from_pile, &v)?;

    if bad > 0 || bad_pile > 0 {
        println!("\nFAIL — {bad} in-memory and {bad_pile} on-disk check(s) differ.");
        std::process::exit(1);
    }
    println!("\nPASS (on disk) — the checkpoint's tokenizer.json is no longer needed.");
    Ok(())
}

/// Compare two tokenizers by BEHAVIOUR. Returns how many checks differed.
fn compare(
    label: &str,
    a: &tokenizers::Tokenizer,
    b: &tokenizers::Tokenizer,
    src: &serde_json::Value,
) -> Result<usize> {
    let mut bad = 0usize;

    // Vocab width, both without and with the added tokens: the added tokens of
    // this tokenizer are NOT vocab entries, so the two numbers differ by 60 and
    // a reconstruction that folded them into the vocab would pass the first
    // check and fail the second.
    for with_added in [false, true] {
        let (x, y) = (a.get_vocab_size(with_added), b.get_vocab_size(with_added));
        if x != y {
            println!("  [{label}] vocab size (added={with_added}) {x} vs {y}");
            bad += 1;
        }
    }

    // Every added token must land on the SAME id. `build_tokenizer` assigns
    // added-token ids by insertion order into `add_special_tokens`, which is a
    // different mechanism from the explicit `"id"` the file carries — so this
    // is two faces of one interface and the only way to know they agree is to
    // ask both.
    let mut id_bad = 0usize;
    if let Some(added) = src["added_tokens"].as_array() {
        for t in added {
            let (Some(c), Some(want)) = (t["content"].as_str(), t["id"].as_u64()) else {
                continue;
            };
            let got = b.token_to_id(c);
            if got != Some(want as u32) {
                if id_bad < 5 {
                    println!("  [{label}] added token {c:?} id {got:?}, file says {want}");
                }
                id_bad += 1;
            }
        }
    }
    if id_bad > 0 {
        println!("  [{label}] {id_bad} added token(s) landed on the wrong id");
        bad += 1;
    }

    // The behavioural check. Both directions: encode the same text to the same
    // ids, and decode the same ids to the same text.
    let mut enc_bad = 0usize;
    for s in PROBES {
        let ea = a
            .encode(*s, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let eb = b
            .encode(*s, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        if ea.get_ids() != eb.get_ids() {
            if enc_bad < 4 {
                println!(
                    "  [{label}] MISMATCH {s:?}\n      file  {:?}\n      graph {:?}",
                    ea.get_ids(),
                    eb.get_ids()
                );
            }
            enc_bad += 1;
        }
        let (da, db) = (
            a.decode(ea.get_ids(), false)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?,
            b.decode(eb.get_ids(), false)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?,
        );
        if da != db {
            println!("  [{label}] DECODE MISMATCH {s:?}: {da:?} vs {db:?}");
            enc_bad += 1;
        }
    }
    println!(
        "  [{label}] encode+decode: {}/{} probe strings identical",
        PROBES.len() - enc_bad.min(PROBES.len()),
        PROBES.len()
    );
    if enc_bad > 0 {
        bad += 1;
    }

    // EVERY vocab entry, through both tokenizers.
    //
    // Twenty hand-picked probe strings are not a check of a 200 000-token BPE
    // model, and this is not a theoretical objection: `--mutate ignore-merges`
    // and `--mutate drop-merge` both PASSED the probe list. A model knob that
    // changes how the merge table is applied shows up on the tokens whose merge
    // path the knob changes, and which those are is not something a person
    // guesses — so ask about all of them.
    let mut vbad = 0usize;
    let mut vchecked = 0usize;
    if let Some(vocab) = src["model"]["vocab"].as_object() {
        for tok in vocab.keys() {
            vchecked += 1;
            let ea = a
                .encode(tok.as_str(), false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            let eb = b
                .encode(tok.as_str(), false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            if ea.get_ids() != eb.get_ids() {
                if vbad < 5 {
                    println!(
                        "  [{label}] VOCAB MISMATCH {tok:?}\n      file  {:?}\n      graph {:?}",
                        ea.get_ids(),
                        eb.get_ids()
                    );
                }
                vbad += 1;
            }
        }
    }
    println!(
        "  [{label}] vocab-key sweep: {}/{vchecked} tokens encode identically",
        vchecked - vbad
    );
    if vbad > 0 {
        bad += 1;
    }

    // The same sweep over the TEXT each id stands for, rather than over the
    // vocab's byte-level spelling of it.
    //
    // These are different questions and the difference is the point. A vocab key
    // is already byte-level-encoded — `Ġthe`, where `Ġ` is U+0120 — so encoding
    // that literal string encodes U+0120 again and never presents `Ġthe` to the
    // BPE model as a pre-token. Decoding the id first gives the string that
    // really produces this token, which is the only input that exercises whether
    // the merge table can rebuild it.
    let mut dbad = 0usize;
    let n_ids = a.get_vocab_size(false) as u32;
    for id in 0..n_ids {
        let Ok(text) = a.decode(&[id], false) else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        let ea = a
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let eb = b
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        if ea.get_ids() != eb.get_ids() {
            if dbad < 5 {
                println!(
                    "  [{label}] ID-{id} MISMATCH {text:?}\n      file  {:?}\n      graph {:?}",
                    ea.get_ids(),
                    eb.get_ids()
                );
            }
            dbad += 1;
        }
    }
    println!(
        "  [{label}] decoded-id sweep: {}/{n_ids} ids re-encode identically",
        n_ids as usize - dbad
    );
    if dbad > 0 {
        bad += 1;
    }

    // The ids the forward-pass verification actually runs on, decoded by both.
    // Not a re-encode: BPE is not injective on retokenisation, so demanding
    // encode(decode(ids)) == ids would be a check about BPE rather than about
    // this graph.
    for f in ["/tmp/prompt.ids", "/tmp/chat_ids.bin"] {
        let Ok(bytes) = std::fs::read(f) else {
            continue;
        };
        let ids: Vec<u32> = bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8")) as u32)
            .collect();
        let (da, db) = (
            a.decode(&ids, false)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?,
            b.decode(&ids, false)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?,
        );
        if da == db {
            println!("  [{label}] {f}: decodes identically -> {:?}", trunc(&da));
        } else {
            println!("  [{label}] {f}: DIFFERS\n      file  {da:?}\n      graph {db:?}");
            bad += 1;
        }
    }
    Ok(bad)
}

fn trunc(s: &str) -> String {
    if s.chars().count() <= 60 {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(60).collect::<String>())
    }
}
