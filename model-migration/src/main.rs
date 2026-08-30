//! Lean one-shot migration of a legacy Mary model pile.
//!
//! This binary deliberately has no checkpoint reader or HuggingFace client.
//! Each subcommand names an existing pile, key, and complete legacy authority;
//! stdout is the full signed native collection claim produced by the migration.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use mary::selection::ModelSelector;
use mary_model_migration::{
    adopt_legacy_personaplex_bundle, migrate_legacy_model_main, LegacyModelMigration,
};
use triblespace::core::inline::encodings::hash::{Blake3, Hash};
use triblespace::core::repo::pile::Pile;
use triblespace::core::repo::CommitHandle;
use triblespace::core::signing_key_file;
use triblespace::prelude::blobencodings::SimpleArchive;
use triblespace::prelude::inlineencodings::Handle;
use triblespace::prelude::{Id, Inline, TryToInline};

#[derive(Debug, Parser)]
#[command(
    name = "mary-model-migrate",
    about = "Adopt exact legacy Mary model graphs into native collections",
    version
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Publish one generic legacy model root into `mary-model-graph`.
    Model {
        /// Existing legacy model pile. It is never created, repaired, or amputated.
        #[arg(long)]
        pile: PathBuf,
        /// Existing private signing-key file (strict 64-hex, owner-private mode).
        #[arg(long)]
        key: PathBuf,
        /// Select the unique existing root carrying this model_name and members.
        #[arg(long, conflicts_with = "root")]
        model_name: Option<String>,
        /// Select this exact existing model-root entity id (32 hex digits).
        #[arg(long, conflicts_with = "model_name")]
        root: Option<String>,
        /// Canonical source coordinate for native model selection.
        #[arg(long)]
        source: String,
        /// Canonical quantization/weight-format coordinate.
        #[arg(long)]
        quantization: String,
        /// Optional existing tokenizer name to validate exactly after projection.
        #[arg(long)]
        tokenizer_name: Option<String>,
    },
    /// Adopt an exact legacy PersonaPlex weight commit into `mary-model-bundles`.
    Personaplex {
        /// Existing legacy PersonaPlex pile. It is never created or repaired.
        #[arg(long)]
        pile: PathBuf,
        /// Existing private signing-key file (strict 64-hex, owner-private mode).
        #[arg(long)]
        key: PathBuf,
        /// Exact legacy weight COMMIT (`blake3:<64 hex>` or bare 64 hex).
        #[arg(long, value_parser = parse_commit_handle)]
        legacy_commit: CommitHandle,
    },
}

enum MigrationOutput {
    Model(mary_model_migration::LegacyModelMigrationResult),
    PersonaPlex(mary_model_migration::PersonaPlexLegacyAdoptionResult),
}

fn parse_commit_handle(value: &str) -> Result<CommitHandle, String> {
    let trimmed = value.trim();
    let owned;
    let normalized = if trimmed.contains(':') {
        trimmed
    } else {
        owned = format!("blake3:{trimmed}");
        &owned
    };
    let hash: Inline<Hash<Blake3>> = normalized.try_to_inline().map_err(|error| {
        format!(
            "invalid legacy commit {value:?}: {error:?} (expected `blake3:<64 hex>` or bare hex)"
        )
    })?;
    Ok(Handle::<SimpleArchive>::from_hash(hash))
}

fn parse_model_selector<'a>(
    model_name: Option<&'a str>,
    root: Option<&str>,
) -> anyhow::Result<ModelSelector<'a>> {
    match (model_name, root) {
        (Some(name), None) => Ok(ModelSelector::Name(name)),
        (None, Some(hex)) => Id::from_hex(hex)
            .map(ModelSelector::Root)
            .ok_or_else(|| anyhow::anyhow!("--root {hex:?} is not a valid 32-hex entity id")),
        _ => anyhow::bail!("pass exactly one of --model-name or --root"),
    }
}

fn run(command: Command) -> anyhow::Result<MigrationOutput> {
    // Resolve semantic selectors before opening anything. Invalid or missing
    // coordinates therefore cannot even observe, much less mutate, a pile.
    let model = match &command {
        Command::Model {
            model_name, root, ..
        } => Some(parse_model_selector(
            model_name.as_deref(),
            root.as_deref(),
        )?),
        Command::Personaplex { .. } => None,
    };
    let (pile_path, key_path) = match &command {
        Command::Model { pile, key, .. } | Command::Personaplex { pile, key, .. } => (pile, key),
    };

    // No path inference, environment fallback, initialization, or generated
    // key: the durable path is a required CLI argument.
    let signing_key = signing_key_file::load_existing(key_path)
        .with_context(|| format!("load existing signing key {key_path:?}"))?;
    let mut pile = Pile::open(pile_path)
        .with_context(|| format!("open existing legacy model pile {pile_path:?}"))?;

    let migration = match &command {
        Command::Model {
            source,
            quantization,
            tokenizer_name,
            ..
        } => migrate_legacy_model_main(
            &mut pile,
            &signing_key,
            LegacyModelMigration {
                model: model.expect("the Model arm resolved one selector"),
                source,
                quantization,
                tokenizer_name: tokenizer_name.as_deref(),
            },
        )
        .map(MigrationOutput::Model),
        Command::Personaplex { legacy_commit, .. } => {
            adopt_legacy_personaplex_bundle(&mut pile, &signing_key, *legacy_commit)
                .map(MigrationOutput::PersonaPlex)
        }
    };

    // This command, not the collection API, owns the explicit durability
    // boundary on both success and failure.
    let close = pile.close();
    match (migration, close) {
        (Ok(output), Ok(())) => Ok(output),
        (Ok(_), Err(error)) => Err(anyhow::anyhow!("close migrated pile: {error}")),
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(close_error)) => Err(error.context(format!(
            "migration also failed to close the pile: {close_error}"
        ))),
    }
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
    let migration = run(args.command)?;

    let commit = match migration {
        MigrationOutput::Model(result) => {
            eprintln!(
                "migrated legacy main branch {} head {:?}: model {}, tokenizer {:?}, {} legacy facts + {} aliases + {} selector facts; team {}, native commit {}",
                result.legacy_branch,
                result.legacy_head,
                result.model_root,
                result.tokenizer_root,
                result.legacy_facts,
                result.aliases_added,
                result.selector_facts_added,
                lowercase_hex(&result.team.to_bytes()),
                result.commit.id(),
            );
            result.commit
        }
        MigrationOutput::PersonaPlex(result) => {
            eprintln!(
                "{} exact legacy PersonaPlex commit {:?}: LM {}, Mimi {}, unified root {}, H {:?}, {} legacy facts + {} aliases; team {}, bundle commit {}",
                if result.published {
                    "adopted"
                } else {
                    "already adopted"
                },
                result.legacy_commit,
                result.legacy_lm_root,
                result.legacy_mimi_root,
                result.model_root,
                result.model_archive_data,
                result.legacy_facts,
                result.aliases_added,
                lowercase_hex(&result.team.to_bytes()),
                result.commit.id(),
            );
            result.commit
        }
    };

    // Machine-readable stdout: exactly the complete 192-byte signed claim,
    // lowercase hex plus the conventional trailing newline.
    println!("{}", lowercase_hex(&commit.to_bytes()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_hex_is_exact_and_lowercase() {
        let bytes: Vec<u8> = (0..192).map(|index| index as u8).collect();
        let encoded = lowercase_hex(&bytes);
        assert_eq!(encoded.len(), 384);
        assert!(encoded
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        assert_eq!(&encoded[..8], "00010203");
        assert_eq!(&encoded[encoded.len() - 8..], "bcbdbebf");
    }

    #[test]
    fn model_selector_requires_exactly_one_name_or_root() {
        assert_eq!(
            parse_model_selector(Some("weights.safetensors"), None).unwrap(),
            ModelSelector::Name("weights.safetensors")
        );
        let root_hex = "01".repeat(16);
        let root = Id::from_hex(&root_hex).unwrap();
        assert_eq!(
            parse_model_selector(None, Some(&root_hex)).unwrap(),
            ModelSelector::Root(root)
        );
        assert!(parse_model_selector(None, None).is_err());
        assert!(parse_model_selector(Some("weights.safetensors"), Some(&root_hex)).is_err());
    }

    #[test]
    fn personaplex_cli_accepts_only_an_explicit_pile_key_and_commit() {
        let hash = "01".repeat(32);
        let args = Args::try_parse_from([
            "mary-model-migrate",
            "personaplex",
            "--pile",
            "/models/personaplex.pile",
            "--key",
            "/keys/migration.key",
            "--legacy-commit",
            hash.as_str(),
        ])
        .expect("parse explicit PersonaPlex migration coordinates");
        let Command::Personaplex {
            pile,
            key,
            legacy_commit,
        } = args.command
        else {
            panic!("wrong subcommand")
        };
        assert_eq!(pile, PathBuf::from("/models/personaplex.pile"));
        assert_eq!(key, PathBuf::from("/keys/migration.key"));
        assert_eq!(
            legacy_commit,
            parse_commit_handle(&format!("blake3:{hash}")).unwrap()
        );

        assert!(Args::try_parse_from([
            "mary-model-migrate",
            "personaplex",
            "--pile",
            "/models/personaplex.pile",
            "--key",
            "/keys/migration.key",
        ])
        .is_err());
    }
}
