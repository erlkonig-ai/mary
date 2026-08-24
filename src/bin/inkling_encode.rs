//! Text on stdin, an `inkling_forward` ids file out.
//!
//! `inkling_forward` reads ids and never text, which is right — tokenising is
//! not part of a forward pass. What it left behind is ids files as undocumented
//! artefacts: `/tmp/prompt.ids` has been THE prompt for weeks and nothing in the
//! tree says which string it is. That matters the moment a measurement wants
//! more than one prompt, because the alternative is synthesising ids, and random
//! ids measure nothing about a language model — they measure how a model behaves
//! off its own manifold, which is not the question anyone is asking.
//!
//! The tokenizer is the checkpoint's own `tokenizer.json`, loaded by the same
//! `tokenizers::Tokenizer::from_file` that `inkling_tokenizer_gate` uses as the
//! REFERENCE side of its comparison — and that gate is what establishes the
//! pile's reconstruction encodes identically. So there is one tokenizer here,
//! not a second transcription of one.
//!
//! Stdin is taken VERBATIM, including a leading space and including whatever
//! trailing bytes the shell hands over. `printf '%s'` rather than `echo`:
//!
//!   printf '%s' ' The capital of France is' \
//!       | inkling_encode <tokenizer.json> /tmp/prompt.ids
//!
//! Ids go out as i64 little-endian, which is what the forward reads. The decode
//! round trip is printed so a file can be checked against the string it claims
//! to be rather than trusted — and the self-check that this tool is the same
//! tokenizer the existing prompts came from is to re-encode them and `cmp`.
//!
//! Build: `--features tokenizer`.

use std::io::Read as _;
use std::path::Path;

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let json_path = std::env::args()
        .nth(1)
        .context("usage: inkling_encode <tokenizer.json> <out.bin>  (text on stdin)")?;
    let out_path = std::env::args()
        .nth(2)
        .context("usage: inkling_encode <tokenizer.json> <out.bin>  (text on stdin)")?;

    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .context("reading the prompt from stdin")?;
    anyhow::ensure!(
        !text.is_empty(),
        "empty prompt — the forward would be vacuous"
    );

    let tok = tokenizers::Tokenizer::from_file(Path::new(&json_path))
        .map_err(|e| anyhow::anyhow!("load {json_path}: {e}"))?;

    // `false`: no special tokens. The existing prompt files carry none, and a
    // BOS silently prepended here would make a new prompt incomparable with them
    // while every shape check still passed.
    let enc = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();

    let mut bytes = Vec::with_capacity(ids.len() * 8);
    for &i in &ids {
        bytes.extend_from_slice(&(i as i64).to_le_bytes());
    }
    std::fs::write(&out_path, &bytes).with_context(|| format!("writing {out_path}"))?;

    let back = tok
        .decode(&ids, false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!("  text   : {text:?}");
    println!("  ids    : {} -> {out_path}", ids.len());
    println!("           {ids:?}");
    println!(
        "  decode : {back:?}  {}",
        if back == text {
            "== the input"
        } else {
            "DIFFERS from the input"
        }
    );
    Ok(())
}
