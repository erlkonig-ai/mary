//! Gate: a SentencePiece UNIGRAM tokenizer survives the trip through the
//! trible graph unchanged.
//!
//! The point of the schema extension is that the tokenizer stops being a loose
//! `.model` file and becomes queryable facts. That is only worth anything if
//! the reconstruction is EXACT — a tokenizer that is 99% right silently
//! corrupts every prompt.
//!
//! So: parse the file, ingest to an in-memory graph, read it back, and compare
//! the two tokenizers by ENCODING BEHAVIOUR, not by field equality. Runs before
//! anything is written to a real pile.
use std::path::Path;

use mary::models::personaplex::spm::SpmTokenizer;
use triblespace::core::signing_key_file;
use triblespace::prelude::*;

/// Strings chosen to exercise every lane the port cares about: plain ASCII,
/// the `▁` word-boundary escape, punctuation, digits, unicode that needs
/// byte-fallback, and the empty string.
const PROBES: &[&str] = &[
    "Hello, how are you?",
    "You are a helpful assistant.",
    "  leading and trailing  ",
    "1234567890",
    "café naïve résumé",
    "日本語のテキスト",
    "emoji 🙂 and more 🎉",
    "punctuation!?;:'\"()[]{}",
    "a",
    "",
    "The quick brown fox jumps over the lazy dog.",
    // The exact system prompt `personaplex_listen` encodes, so the gate covers
    // the string that is actually in production use and not just its neighbours.
    "You are a helpful assistant. You speak with a warm, curious, and direct \
     voice. Answer clearly.",
];

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next();
    let mut pile = None;
    let mut signing_key_path = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--signing-key" => {
                signing_key_path = Some(args.next().unwrap_or_else(|| {
                    eprintln!("--signing-key needs a path");
                    std::process::exit(2)
                }))
            }
            flag if flag.starts_with('-') => {
                eprintln!("unknown argument {flag}");
                std::process::exit(2)
            }
            path if pile.is_none() => pile = Some(path.to_string()),
            other => {
                eprintln!("unexpected argument {other:?}");
                std::process::exit(2)
            }
        }
    }
    if pile.is_none() && signing_key_path.is_some() {
        eprintln!("--signing-key requires a destination pile");
        std::process::exit(2);
    }

    // argv[1], else `$MARY_MODELS/tokenizer_spm_32k_3.model`; never a guess.
    let path = mary::paths::model(model_path.as_deref(), "tokenizer_spm_32k_3.model")
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        })
        .to_string_lossy()
        .into_owned();
    println!("spm model: {path}");

    // ── 1. the file path (the proven one) ──
    let (pieces, add_dummy_prefix, byte_fallback) = SpmTokenizer::parse_model(Path::new(&path));
    println!(
        "parsed   : {} pieces, add_dummy_prefix {add_dummy_prefix}, byte_fallback {byte_fallback}",
        pieces.len()
    );
    let from_file = SpmTokenizer::from_pieces(&pieces, add_dummy_prefix);

    // ── 2. ingest into an in-memory graph ──
    let mut blobs = MemoryBlobStore::new();
    let frag = mary::tokenizer::save_spm_unigram(
        &pieces,
        add_dummy_prefix,
        byte_fallback,
        "kyutai/moshiko-pytorch-bf16 tokenizer_spm_32k_3.model",
        &mut blobs,
    )
    .expect("ingest");
    let root_id = frag.root().expect("tokenizer root");
    let tribles = frag.into_facts();
    println!("graph    : {} facts", tribles.len());

    // DISCOVER the node the way the real loader does, rather than trusting the
    // fragment root. These disagreed once: the graph was written correctly but
    // `find_tokenizer` filtered to BPE/WordPiece and did not know UNIGRAM, so a
    // root-based check passed while every real read said "no tokenizer graph".
    let tok_id = mary::tokenizer::find_tokenizer(&tribles)
        .expect("find_tokenizer must discover the node the loader will look for");
    assert_eq!(
        tok_id, root_id,
        "find_tokenizer found a different node than the fragment root"
    );

    // ── 3. read it back out ──
    let reader = blobs.snapshot().expect("snapshot");
    let back = mary::tokenizer::load_spm_pieces(&tribles, &reader, tok_id);
    println!("read back: {} pieces", back.len());

    // piece table must be identical, in order
    assert_eq!(back.len(), pieces.len(), "piece count changed");
    let mut score_max = 0f32;
    for (i, (a, b)) in pieces.iter().zip(&back).enumerate() {
        assert_eq!(a.0, b.0, "piece {i} bytes differ");
        assert_eq!(a.2, b.2, "piece {i} type differs ({} vs {})", a.2, b.2);
        score_max = score_max.max((a.1 - b.1).abs());
    }
    println!("pieces   : identical bytes+types, max |score delta| {score_max:.3e}");
    assert_eq!(score_max, 0.0, "scores are not bit-exact through F64");

    let adp_back = mary::tokenizer::has_add_prefix_space(&tribles, tok_id);
    assert_eq!(adp_back, add_dummy_prefix, "add_prefix_space flag lost");

    // ── 4. the gate that matters: same ENCODING ──
    let from_graph = SpmTokenizer::from_pieces(&back, adp_back);
    let mut worst = 0usize;
    for s in PROBES {
        let a = from_file.encode(s);
        let b = from_graph.encode(s);
        if a != b {
            println!("  MISMATCH on {s:?}\n    file  {a:?}\n    graph {b:?}");
            worst += 1;
        }
    }
    println!(
        "encode   : {}/{} probe strings identical",
        PROBES.len() - worst,
        PROBES.len()
    );

    // round-trip a decode too, so the id->piece surface is covered
    let ids = from_file.encode("Hello, how are you?");
    let da = from_file.decode(&ids);
    let db = from_graph.decode(&ids);
    println!("decode   : file {da:?} | graph {db:?}");
    assert_eq!(da, db, "decode surface differs");

    assert_eq!(worst, 0, "{worst} probe strings encode differently");
    println!("\nPASS (in memory) — the graph reconstructs the tokenizer exactly.");

    // ── 5. optionally do it for real, and re-run the SAME battery ──
    // The in-memory pass proves the schema. It does NOT prove the pile: commit,
    // push, reopen and blob-resolution are all untested by it. So when a pile is
    // named, ingest and gate again through the on-disk path.
    let Some(pile) = pile else {
        println!("\n(no pile argument — nothing was written. Pass one to ingest for real.)");
        return;
    };
    let signing_key_path = signing_key_path.unwrap_or_else(|| {
        eprintln!("--signing-key <existing-key> is required when writing a pile");
        std::process::exit(2)
    });
    let signing_key =
        signing_key_file::load_existing(Path::new(&signing_key_path)).unwrap_or_else(|error| {
            eprintln!("load existing signing key {signing_key_path:?}: {error}");
            std::process::exit(2)
        });
    println!("\n=== ingesting into {pile} ===");
    match mary::persist::ingest_spm_tokenizer(
        Path::new(&pile),
        Path::new(&path),
        "kyutai/moshiko-pytorch-bf16 tokenizer_spm_32k_3.model",
        &signing_key,
    ) {
        Ok(n) => println!("committed: {n} facts"),
        Err(e) => {
            println!("ingest declined: {e}");
            println!("(continuing to the read-back gate — an existing graph still has to pass)");
        }
    }

    let from_pile = mary::persist::load_spm_tokenizer_from_pile(Path::new(&pile))
        .unwrap_or_else(|e| panic!("read back from pile: {e}"));
    println!("from pile: vocab {}", from_pile.vocab_size());
    assert_eq!(
        from_pile.vocab_size(),
        from_file.vocab_size(),
        "pile tokenizer has a different vocab size"
    );
    let mut bad = 0usize;
    for s in PROBES {
        let a = from_file.encode(s);
        let b = from_pile.encode(s);
        if a != b {
            println!("  MISMATCH on {s:?}\n    file {a:?}\n    pile {b:?}");
            bad += 1;
        }
    }
    println!(
        "encode   : {}/{} probe strings identical through the PILE",
        PROBES.len() - bad,
        PROBES.len()
    );
    assert_eq!(
        bad, 0,
        "{bad} probe strings encode differently from the pile"
    );
    println!("\nPASS (on disk) — PersonaPlex no longer needs the .model file.");
}
