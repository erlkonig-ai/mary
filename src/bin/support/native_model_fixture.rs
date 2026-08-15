use anyhow::Context;
use ed25519_dalek::SigningKey;
use std::path::Path;
use triblespace::prelude::{Id, Pile};

/// Import one model into a fresh native model-collection pile for a bin fixture.
///
/// The signer is deterministic so the fixture has no ambient key-file or RNG
/// dependency. It is intentionally ephemeral and must never be used as a
/// durable model authority.
pub fn import_native_model_fixture(
    model_dir: &Path,
    pile_path: &Path,
    dtype: mary::ingest::LeafDtype,
    source: &str,
) -> anyhow::Result<Id> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(pile_path)
        .with_context(|| format!("create fresh fixture pile {pile_path:?}"))?;

    let signing_key = SigningKey::from_bytes(&[0x6d; 32]);
    let mut pile =
        Pile::open(pile_path).with_context(|| format!("open fresh fixture pile {pile_path:?}"))?;
    let imported = mary::persist::import_model_to_collection(
        &mut pile,
        &signing_key,
        model_dir,
        dtype,
        source,
        mary::persist::QUANTIZATION_NATIVE,
    );
    let close = pile.close();

    match (imported, close) {
        (Ok((root, _commit)), Ok(())) => Ok(root),
        (Ok(_), Err(error)) => Err(anyhow::anyhow!("close fixture pile: {error}")),
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(close_error)) => Err(error.context(format!(
            "import also failed to close the pile: {close_error}"
        ))),
    }
}
