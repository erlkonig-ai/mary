//! READ-ONLY: exercise EXACTLY the two calls `faculties/src/bin/duplex.rs`
//! makes to open its weight pile, against whatever pile is named on argv.
//!
//!   mary::persist::personaplex_bundle(&weights)      (duplex.rs:1590)
//!   mary::persist::load_spm_tokenizer_from_pile(..)  (duplex.rs:1606)
//!
//! plus duplex's own vocabulary check. It deliberately stops before
//! `RealtimePipeline::load_auto`, which materialises ~32 GiB onto the GPU.

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: personaplex_load_check <pile>")?;
    let path = std::path::PathBuf::from(path);
    println!("weights {}", path.display());

    let bundle = mary::persist::personaplex_bundle(&path)
        .with_context(|| format!("load the model from {}", path.display()))?;
    println!("bundle authority: {:?}", bundle.authority());
    let source = bundle.into_runtime_source();
    let loader = mary::persist::personaplex_loader(&path)
        .context("project the bundle into a weight loader")?;
    let _ = &source;
    for probe in [
        "transformer.layers.0.self_attn.in_proj_weight",
        "depformer.layers.0.gating.0.linear_in.weight",
        "encoder.model.0.conv.conv.weight",
        "quantizer.acoustic_residual_vector_quantizer.layers.0._codebook.embedding_sum",
    ] {
        println!("  {:<70} {}", probe, loader.has_weight(probe));
    }

    let spm = mary::persist::load_spm_tokenizer_from_pile(&path)
        .context("load the text tokenizer from the weight pile")?;
    println!("spm vocab_size {}", spm.vocab_size());
    let want = mary::models::personaplex::config::TEXT_CARD;
    println!("model TEXT_CARD {want}");
    anyhow::ensure!(
        spm.vocab_size() == want,
        "tokenizer vocabulary {} does not match the model's {want}",
        spm.vocab_size()
    );

    println!("OK: duplex's load path succeeds on this pile");
    Ok(())
}
