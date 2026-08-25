//! The Kimi-K3 tokenizer gate: mary's pure-Rust tiktoken port vs the SHIPPED
//! Python (`tokenization_kimi.TikTokenTokenizer`), compared as full id
//! SEQUENCES over an adversarial battery plus a 3,000-string fuzz corpus.
//!
//! Goldens come from `golden/capture_k3_tokenizer.py` (run under the oracle
//! venv, which has the real `tiktoken` package). Text crosses the language
//! boundary as hex of its UTF-8 bytes, so no JSON or shell escaping can make
//! the two sides disagree about what was tokenized.
//!
//! The tokenizer under test is built FROM THE GRAPH, not from the file: the
//! run ingests `tiktoken.model` + `tokenizer_config.json` into a tokenizer
//! entity ([`mary::tokenizer::save_tiktoken`]), then reads the rank table, the
//! special tokens and the pre-tokenizer pattern back out with queries. So every
//! id in every case has travelled through the trible schema.
//!
//! What is checked, in order — any failure is fatal and nothing prints "PASS":
//!
//!   1. **vocab fingerprint** — FNV-1a/64 over all 163,584 `(len|bytes|rank)`
//!      triples must equal Python's, proving the base64 parse is byte-identical
//!      to `load_tiktoken_bpe` rather than merely plausible.
//!   2. **pattern** — the pattern stored in the graph must equal the `pat_str`
//!      the shipped class reports.
//!   3. **special tokens** — all 256 `(name, id)` pairs, exactly.
//!   4. **encode** — full id sequences, per case, both `allow_special` modes.
//!   5. **decode** — the decoded text of the reference ids, per case.
//!
//! Every comparison asserts its own non-emptiness where non-emptiness is
//! expected: a battery with no cases, or a case whose ids are empty for
//! non-empty text, fails rather than passing vacuously.
//!
//! `--mutate <name>` deliberately breaks one thing to prove the gate can fail.
//! `--mutate list` names them.
//!
//!   cargo run --release --features k3tok --bin k3_tokenizer_gate
//!   cargo run --release --features k3tok --bin k3_tokenizer_gate -- --mutate swap_ranks

use std::path::PathBuf;

use mary::tiktoken::{KIMI_K3_PAT_STR, Rank, Tiktoken};
use mary::tokenizer;
use serde_json::Value;
use triblespace::prelude::*;

/// Checkpoint directory: `K3_MODEL_DIR`, else `$MARY_MODELS/kimi-k3`, as in the
/// other gates. There is no guessed default.
fn model_dir() -> String {
    mary::paths::model(std::env::var("K3_MODEL_DIR").ok().as_deref(), "kimi-k3")
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        })
        .to_string_lossy()
        .into_owned()
}
const GOLDEN: &str = "/tmp/mary-k3tok/golden/k3_tokenizer_battery.json";

/// The deliberate breakages. Each one is a real, plausible porting mistake; the
/// gate is only worth anything if it catches every one of them.
const MUTANTS: &[(&str, &str)] = &[
    (
        "swap_ranks",
        "swap two COLD tokens' ranks in tiktoken.model before ingest",
    ),
    (
        "swap_ranks_hot",
        "swap two HOT tokens' ranks after the fingerprint check",
    ),
    (
        "drop_han_branch",
        "drop the leading [\\p{Han}]+ branch from pat_str",
    ),
    (
        "drop_lookahead_branch",
        "drop the \\s+(?!\\S) branch from pat_str",
    ),
    (
        "digits_unbounded",
        "\\p{N}{1,3} -> \\p{N}+ (numbers as one token)",
    ),
    (
        "merge_highest_first",
        "merge the HIGHEST-rank pair instead of the lowest",
    ),
    ("special_off_by_one", "shift every special token id by +1"),
    ("no_specials", "encode with allow_special always false"),
    ("skip_chunking", "drop the shipped 25k-char chunking"),
    ("empty_battery", "compare against an empty case list"),
];

fn hex_to_bytes(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd-length hex");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// Rewrite a `tiktoken.model` with two tokens' ranks exchanged — a file-level
/// corruption standing in for a parser that mis-pairs bytes with ranks.
fn swap_ranks_in_file(src: &[u8], a: u32, b: u32) -> Vec<u8> {
    let mut lines: Vec<Vec<u8>> = src.split(|&c| c == b'\n').map(|l| l.to_vec()).collect();
    let find = |lines: &Vec<Vec<u8>>, r: u32| -> usize {
        lines
            .iter()
            .position(|l| l.rsplit(|&c| c == b' ').next() == Some(r.to_string().as_bytes()))
            .unwrap_or_else(|| panic!("no line with rank {r}"))
    };
    let (ia, ib) = (find(&lines, a), find(&lines, b));
    let retag = |l: &[u8], r: u32| -> Vec<u8> {
        let sp = l.iter().position(|&c| c == b' ').unwrap();
        let mut out = l[..=sp].to_vec();
        out.extend_from_slice(r.to_string().as_bytes());
        out
    };
    let (na, nb) = (retag(&lines[ia], b), retag(&lines[ib], a));
    lines[ia] = na;
    lines[ib] = nb;
    lines.join(&b'\n')
}

fn fnv1a64(chunks: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in chunks {
        for &b in *c {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut mutate: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mutate" => {
                let m = args.get(i + 1).expect("--mutate needs a name").clone();
                if m == "list" {
                    println!("mutants:");
                    for (n, d) in MUTANTS {
                        println!("  {n:<22} {d}");
                    }
                    return;
                }
                assert!(
                    MUTANTS.iter().any(|(n, _)| *n == m),
                    "unknown mutant {m:?} (try --mutate list)"
                );
                mutate = Some(m);
                i += 1;
            }
            other => panic!("unknown argument {other:?}"),
        }
        i += 1;
    }
    if let Some(m) = &mutate {
        println!("!! MUTANT ACTIVE: {m} — the gate is EXPECTED to fail below\n");
    }
    let mutate = mutate.as_deref();

    // ── goldens ──────────────────────────────────────────────────────────────
    let gold: Value =
        serde_json::from_slice(&std::fs::read(GOLDEN).unwrap_or_else(|e| {
            panic!("{GOLDEN}: {e} — run golden/capture_k3_tokenizer.py first")
        }))
        .expect("golden json");

    // ── ingest into the graph, then read the tokenizer back OUT of it ───────
    let dir = PathBuf::from(model_dir());
    let mut model_file = std::fs::read(dir.join("tiktoken.model")).expect("tiktoken.model");
    let config_json =
        std::fs::read(dir.join("tokenizer_config.json")).expect("tokenizer_config.json");

    // Corrupt the SOURCE FILE, which is where a real "ranks got mis-paired with
    // bytes" bug would live — not the in-memory table after it has been
    // checked. 5000/5001 are COLD (absent from the whole 93k-token corpus), so
    // this mutant is caught by the whole-vocab fingerprint or not at all.
    if mutate == Some("swap_ranks") {
        model_file = swap_ranks_in_file(&model_file, 5000, 5001);
    }

    let mut pat_str = KIMI_K3_PAT_STR.to_string();
    match mutate {
        Some("drop_han_branch") => pat_str = pat_str.replacen(r"[\p{Han}]+|", "", 1),
        Some("drop_lookahead_branch") => pat_str = pat_str.replacen(r"|\s+(?!\S)", "", 1),
        Some("digits_unbounded") => pat_str = pat_str.replacen(r"\p{N}{1,3}", r"\p{N}+", 1),
        _ => {}
    }

    let t0 = std::time::Instant::now();
    let mut blobs = MemoryBlobStore::new();
    let frag = tokenizer::save_tiktoken(
        &model_file,
        &config_json,
        &pat_str,
        "moonshotai/Kimi-K3",
        &mut blobs,
    )
    .expect("save_tiktoken");
    let tok_id = frag.root().expect("tokenizer root");
    let tribles: TribleSet = frag.into();
    let blobs = BlobStore::reader(&mut blobs).expect("blob reader");
    println!(
        "ingest: {} tribles in {:.1}s",
        tribles.len(),
        t0.elapsed().as_secs_f64()
    );

    let found = tokenizer::find_tokenizer(&tribles).expect("find_tokenizer found no tokenizer");
    assert_eq!(found, tok_id, "find_tokenizer returned a different entity");

    let t0 = std::time::Instant::now();
    let mut ranks: Vec<(Vec<u8>, Rank)> = tokenizer::load_tiktoken_ranks(&tribles, &blobs, tok_id)
        .into_iter()
        .map(|(b, r)| (b, r as Rank))
        .collect();
    let specials_graph: Vec<(String, u64)> = tokenizer::load_added(&tribles, &blobs, tok_id);
    let graph_pat = tokenizer::load_pre_tokenizer_pattern(&tribles, &blobs, tok_id)
        .expect("no pre-tokenizer pattern in the graph");
    println!(
        "graph load: {} ranks + {} specials in {:.1}s",
        ranks.len(),
        specials_graph.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut fail = 0usize;
    macro_rules! check {
        ($ok:expr_2021, $($arg:tt)*) => {
            if $ok { println!("  PASS  {}", format!($($arg)*)); }
            else   { println!("  FAIL  {}", format!($($arg)*)); fail += 1; }
        };
    }

    // ── 1. vocab fingerprint vs Python's base64 parse ───────────────────────
    println!("\n── vocab ──");
    assert!(!ranks.is_empty(), "rank table is empty");
    let mut by_rank = ranks.clone();
    by_rank.sort_by_key(|(_, r)| *r);
    let mut buf: Vec<Vec<u8>> = Vec::with_capacity(by_rank.len() * 3);
    for (b, r) in &by_rank {
        buf.push((b.len() as u32).to_le_bytes().to_vec());
        buf.push(b.clone());
        buf.push(r.to_le_bytes().to_vec());
    }
    let refs: Vec<&[u8]> = buf.iter().map(|v| v.as_slice()).collect();
    let fp = format!("{:016x}", fnv1a64(&refs));
    let want_fp = gold["vocab_fnv1a64"].as_str().unwrap();
    let want_n = gold["n_base"].as_u64().unwrap() as usize;
    check!(
        ranks.len() == want_n,
        "base vocab size {} (want {want_n})",
        ranks.len()
    );
    check!(fp == want_fp, "vocab fingerprint {fp} (want {want_fp})");

    // ── 2. pre-tokenizer pattern ────────────────────────────────────────────
    let want_pat = gold["pat_str"].as_str().unwrap();
    check!(
        KIMI_K3_PAT_STR == want_pat,
        "KIMI_K3_PAT_STR matches the shipped pat_str"
    );
    check!(
        graph_pat == pat_str,
        "pattern survives the graph round trip"
    );

    // ── 3. special tokens ───────────────────────────────────────────────────
    let want_specials: Vec<(String, u64)> = gold["special_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v["name"].as_str().unwrap().to_string(),
                v["id"].as_u64().unwrap(),
            )
        })
        .collect();
    assert!(!want_specials.is_empty(), "golden has no special tokens");
    let mut specials: Vec<(String, u64)> = specials_graph.clone();
    if mutate == Some("special_off_by_one") {
        for s in specials.iter_mut() {
            s.1 += 1;
        }
    }
    let mut got = specials.clone();
    let mut want = want_specials.clone();
    got.sort();
    want.sort();
    check!(
        got.len() == want.len(),
        "special-token count {} (want {})",
        got.len(),
        want.len()
    );
    let bad: Vec<_> = got
        .iter()
        .zip(want.iter())
        .filter(|(a, b)| a != b)
        .take(4)
        .collect();
    check!(
        got == want,
        "all {} special (name, id) pairs; first diffs: {bad:?}",
        want.len()
    );

    // ── build the tokenizer from what the graph gave back ───────────────────
    if mutate == Some("swap_ranks_hot") {
        // 6268 (" ") and 37484 are the two most-used base ids in the corpus.
        // Injected AFTER the fingerprint check, so only the id comparison can
        // catch it — the complement of `swap_ranks`.
        let (a, b) = (6268 as Rank, 37484 as Rank);
        let ia = ranks.iter().position(|(_, r)| *r == a).unwrap();
        let ib = ranks.iter().position(|(_, r)| *r == b).unwrap();
        ranks[ia].1 = b;
        ranks[ib].1 = a;
    }
    let tk = Tiktoken::new(
        ranks.clone(),
        specials.iter().map(|(n, i)| (n.clone(), *i as Rank)),
        &pat_str,
    )
    .expect("Tiktoken::new");
    check!(
        tk.n_vocab() == gold["n_vocab"].as_u64().unwrap() as usize,
        "n_vocab {} (want {})",
        tk.n_vocab(),
        gold["n_vocab"].as_u64().unwrap()
    );

    let encode = |text: &str, allow: bool| -> Vec<Rank> {
        match mutate {
            Some("no_specials") => tk.encode(text, false),
            Some("merge_highest_first") => {
                mary::tiktoken::mutants::encode_highest_rank_first(&tk, text, allow)
            }
            Some("skip_chunking") => mary::tiktoken::mutants::encode_unchunked(&tk, text, allow),
            _ => tk.encode(text, allow),
        }
    };

    // ── 4/5. the named battery: full id sequences, then decode ──────────────
    println!(
        "\n── battery ({} cases) ──",
        gold["cases"].as_array().unwrap().len()
    );
    let empty: Vec<Value> = Vec::new();
    let cases = if mutate == Some("empty_battery") {
        &empty
    } else {
        gold["cases"].as_array().unwrap()
    };
    assert!(
        !cases.is_empty(),
        "the battery is EMPTY — a green run here would measure nothing"
    );
    let mut enc_fail: Vec<String> = Vec::new();
    let mut dec_fail: Vec<String> = Vec::new();
    let mut tokens_compared = 0usize;
    // Which ids the id comparison actually exercises. Reported, not asserted:
    // no battery reaches a 163,840-entry vocab, and pretending otherwise is
    // how a gate starts measuring less than it appears to. The whole-vocab
    // claim rests on the fingerprint check above, which covers all 163,584.
    let mut used: std::collections::HashSet<Rank> = std::collections::HashSet::new();
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let text = String::from_utf8(hex_to_bytes(c["text_hex"].as_str().unwrap())).expect("utf8");
        let allow = c["allow_special"].as_bool().unwrap();
        let want: Vec<Rank> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as Rank)
            .collect();
        used.extend(want.iter().copied());
        assert!(
            !(want.is_empty() && !text.is_empty()),
            "golden case {name:?} has non-empty text but no ids"
        );
        let got = encode(&text, allow);
        tokens_compared += want.len();
        let ok = got == want;
        if !ok {
            enc_fail.push(name.to_string());
            let at = got.iter().zip(want.iter()).position(|(a, b)| a != b);
            println!(
                "  FAIL  encode {name:<26} got {} ids, want {} ids, first diff at {at:?}\n           got  {:?}\n           want {:?}",
                got.len(),
                want.len(),
                &got[..got.len().min(24)],
                &want[..want.len().min(24)],
            );
        } else {
            println!("  PASS  encode {name:<26} {} ids", want.len());
        }
        // decode the REFERENCE ids (not ours) — so this measures decode alone.
        let want_dec = String::from_utf8(hex_to_bytes(c["decoded_hex"].as_str().unwrap())).unwrap();
        let got_dec = tk.decode(&want);
        if got_dec != want_dec {
            dec_fail.push(name.to_string());
            println!(
                "  FAIL  decode {name:<26} got {:?}… want {:?}…",
                &got_dec.chars().take(40).collect::<String>(),
                &want_dec.chars().take(40).collect::<String>()
            );
        }
    }
    check!(
        tokens_compared > 0,
        "battery compared {tokens_compared} tokens (must be > 0)"
    );
    check!(
        enc_fail.is_empty(),
        "battery encode: {} failing case(s) {enc_fail:?}",
        enc_fail.len()
    );
    check!(
        dec_fail.is_empty(),
        "battery decode: {} failing case(s) {dec_fail:?}",
        dec_fail.len()
    );

    // ── the fuzz corpus ─────────────────────────────────────────────────────
    let fuzz = gold["fuzz"].as_array().unwrap();
    assert!(!fuzz.is_empty(), "the fuzz corpus is EMPTY");
    let mut fuzz_fail: Vec<(String, Vec<Rank>, Vec<Rank>)> = Vec::new();
    let mut fuzz_tokens = 0usize;
    for c in fuzz {
        let text = String::from_utf8(hex_to_bytes(c["text_hex"].as_str().unwrap())).expect("utf8");
        let allow = c["allow_special"].as_bool().unwrap();
        let want: Vec<Rank> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as Rank)
            .collect();
        used.extend(want.iter().copied());
        fuzz_tokens += want.len();
        let got = encode(&text, allow);
        if got != want {
            fuzz_fail.push((text, got, want));
        }
    }
    println!(
        "\n── fuzz ({} strings, {fuzz_tokens} reference tokens) ──",
        fuzz.len()
    );
    for (t, got, want) in fuzz_fail.iter().take(5) {
        println!(
            "  FAIL  {:?}\n           got  {:?}\n           want {:?}",
            t.chars().take(60).collect::<String>(),
            &got[..got.len().min(20)],
            &want[..want.len().min(20)]
        );
    }
    check!(
        fuzz_tokens > 0,
        "fuzz compared {fuzz_tokens} tokens (must be > 0)"
    );
    check!(
        fuzz_fail.is_empty(),
        "fuzz: {}/{} strings differ",
        fuzz_fail.len(),
        fuzz.len()
    );

    // ── a small self-check that decode inverts encode on the ASCII cases ────
    // (weak on its own — a property of the pair — so it is reported separately
    //  and never stands in for the sequence comparison above.)
    let rt: Vec<&str> = vec!["hello world", "你好世界", "don't", "1234"];
    let rt_bad: Vec<&str> = rt
        .iter()
        .copied()
        .filter(|s| tk.decode(&tk.encode(s, true)) != **s)
        .collect();
    check!(rt_bad.is_empty(), "round trip (informational): {rt_bad:?}");

    println!(
        "\nid-comparison coverage: {} distinct ids of {} ({:.1}%). The other {} are\n\
         covered by the whole-vocab fingerprint, not by any encode comparison.",
        used.len(),
        tk.n_vocab(),
        100.0 * used.len() as f64 / tk.n_vocab() as f64,
        tk.n_vocab() - used.len(),
    );

    println!();
    if fail == 0 {
        println!(
            "GATE PASS — {} named cases + {} fuzz strings, {} reference tokens, all ids identical to the shipped Python.",
            cases.len(),
            fuzz.len(),
            tokens_compared + fuzz_tokens
        );
    } else {
        println!("GATE FAIL — {fail} check(s) failed.");
        std::process::exit(1);
    }
}
