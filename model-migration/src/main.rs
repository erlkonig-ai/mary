//! Lean one-shot migration of a legacy Mary model pile.
//!
//! This binary deliberately has no checkpoint reader, HuggingFace client, or
//! model-family policy. Its inputs are the coordinates to add to one exact
//! legacy `main` snapshot; its stdout is the full signed native collection
//! ticket needed for an exact read.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use mary_model_migration::{migrate_legacy_model_main, LegacyModelMigration};
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;

#[derive(Debug, Parser)]
#[command(
    name = "mary-model-migrate",
    about = "Publish a frozen legacy Mary 'main' branch as one native model-collection commit",
    version
)]
struct Args {
    /// Existing legacy model pile. It is never created, repaired, or amputated.
    #[arg(long)]
    pile: PathBuf,
    /// Existing private signing-key file (strict 64-hex, owner-private mode).
    #[arg(long)]
    key: PathBuf,
    /// Existing model_name on the unique legacy weight root (name + member).
    #[arg(long)]
    model_name: String,
    /// Canonical source coordinate for native model selection.
    #[arg(long)]
    source: String,
    /// Canonical quantization/weight-format coordinate.
    #[arg(long)]
    quantization: String,
    /// Optional existing tokenizer name to validate exactly after projection.
    #[arg(long)]
    tokenizer_name: Option<String>,
}

fn lowercase_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String is infallible");
    }
    output
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // No path inference, environment fallback, initialization, or generated
    // key: the durable path is a required CLI argument.
    let signing_key = signing_key_file::load_existing(&args.key)
        .with_context(|| format!("load existing signing key {:?}", args.key))?;
    let mut pile = Pile::open(&args.pile)
        .with_context(|| format!("open existing legacy model pile {:?}", args.pile))?;

    let migration = migrate_legacy_model_main(
        &mut pile,
        &signing_key,
        LegacyModelMigration {
            legacy_model_name: &args.model_name,
            source: &args.source,
            quantization: &args.quantization,
            tokenizer_name: args.tokenizer_name.as_deref(),
        },
    );

    // The core API has no durability policy. This one-shot command owns one
    // explicit close boundary on every post-open path, including validation
    // failure (where Repository's content-addressed reads normally left the
    // pile clean).
    let close = pile.close();
    let result = match (migration, close) {
        (Ok(result), Ok(())) => result,
        (Ok(_), Err(error)) => return Err(anyhow::anyhow!("close migrated pile: {error}")),
        (Err(error), Ok(())) => return Err(error),
        (Err(error), Err(close_error)) => {
            return Err(error.context(format!(
                "migration also failed to close the pile: {close_error}"
            )))
        }
    };

    eprintln!(
        "migrated legacy main branch {} head {:?}: model {}, tokenizer {:?}, {} legacy facts + {} aliases + {} selector facts; native commit {}",
        result.legacy_branch,
        result.legacy_head,
        result.model_root,
        result.tokenizer_root,
        result.legacy_facts,
        result.aliases_added,
        result.selector_facts_added,
        result.commit.id(),
    );

    // Machine-readable stdout: exactly the complete 192-byte signed ticket,
    // lowercase hex plus the conventional trailing newline. An exact reader
    // can reconstruct CollectionCommit::from_bytes from these 384 digits.
    println!("{}", lowercase_hex(&result.commit.to_bytes()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticket_hex_is_exact_and_lowercase() {
        let bytes: Vec<u8> = (0..192).map(|index| index as u8).collect();
        let encoded = lowercase_hex(&bytes);
        assert_eq!(encoded.len(), 384);
        assert!(encoded
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        assert_eq!(&encoded[..8], "00010203");
        assert_eq!(&encoded[encoded.len() - 8..], "bcbdbebf");
    }
}
