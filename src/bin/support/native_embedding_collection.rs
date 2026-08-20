use anyhow::Context;
use ed25519_dalek::SigningKey;
use mary::ingest::{LeafDtype, LeafHandles};
use mary::selection::{ModelSelector, SelectedModelIndex, TokenizerSelector};
use std::collections::BTreeMap;
use std::path::Path;
use triblespace::core::collection::CollectionCommit;
use triblespace::core::repo::pile::Pile;
use triblespace::prelude::*;

fn validate_f32_model<R: BlobStoreGet>(
    selected: &SelectedModelIndex<R>,
    contract: &BTreeMap<String, Vec<usize>>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !selected.handles().is_empty(),
        "embedding model root contains no tensors"
    );
    let mut actual_shapes = BTreeMap::new();
    for (name, handles) in selected.handles() {
        let LeafHandles::F32(data, shape) = handles else {
            anyhow::bail!("embedding tensor {name:?} is not an exact f32 leaf");
        };
        let data: anybytes::Bytes = selected
            .reader()
            .get(*data)
            .map_err(|error| anyhow::anyhow!("read embedding tensor {name:?}: {error}"))?;
        anyhow::ensure!(
            (data.as_ptr() as usize).is_multiple_of(256),
            "embedding tensor {name:?} is not 256-byte aligned"
        );
        let values = data
            .view::<[f32]>()
            .with_context(|| format!("decode embedding tensor {name:?}"))?;
        let shape: anybytes::Bytes = selected
            .reader()
            .get(*shape)
            .map_err(|error| anyhow::anyhow!("read shape for {name:?}: {error}"))?;
        let shape = shape
            .view::<[u64]>()
            .with_context(|| format!("decode shape for {name:?}"))?;
        let shape = shape
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension)
                    .with_context(|| format!("shape dimension for {name:?} exceeds usize"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let elements = shape.iter().try_fold(1_usize, |product, &dimension| {
            product
                .checked_mul(dimension)
                .with_context(|| format!("shape element count for {name:?} overflows usize"))
        })?;
        anyhow::ensure!(
            values.len() == elements,
            "embedding tensor {name:?} has {} values but shape describes {elements}",
            values.len()
        );
        actual_shapes.insert(name.clone(), shape);
    }
    for (name, expected) in contract {
        let actual = actual_shapes
            .get(name)
            .with_context(|| format!("embedding model is missing required tensor {name:?}"))?;
        anyhow::ensure!(
            actual == expected,
            "embedding tensor {name:?} has shape {actual:?}, expected {expected:?}"
        );
    }
    if actual_shapes.len() != contract.len() {
        let unexpected = actual_shapes
            .keys()
            .find(|name| !contract.contains_key(*name))
            .expect("different exact-map lengths imply an unexpected key");
        anyhow::bail!("embedding model contains unexpected tensor {unexpected:?}");
    }
    Ok(())
}

fn publish_embedding_candidate_with_contract_impl(
    pile: &mut Pile,
    signing_key: &SigningKey,
    weights: &Path,
    format: mary::formats::WeightFormat,
    tokenizer_json: Option<&Path>,
    source: &str,
    architecture: mary::embed::EmbeddingArchitecture,
    contract: &BTreeMap<String, Vec<usize>>,
) -> anyhow::Result<(Id, CollectionCommit, usize)> {
    match architecture {
        mary::embed::EmbeddingArchitecture::ClipVitBasePatch32
        | mary::embed::EmbeddingArchitecture::NomicTextV15 => anyhow::ensure!(
            tokenizer_json.is_some(),
            "{architecture:?} requires its tokenizer in the same signed cohort"
        ),
        mary::embed::EmbeddingArchitecture::NomicVisionV15 => anyhow::ensure!(
            tokenizer_json.is_none(),
            "{architecture:?} has no tokenizer and rejects a tokenizer attachment"
        ),
    }

    let mut candidate = mary::persist::ingest_weight_file_filtered_fragment(
        pile,
        weights,
        format,
        LeafDtype::F32,
        source,
        mary::persist::QUANTIZATION_NATIVE,
        |name| contract.contains_key(name),
    )?;
    let model_root = candidate
        .root()
        .context("embedding candidate has no unique model root")?;

    if let Some(path) = tokenizer_json {
        let json =
            std::fs::read(path).with_context(|| format!("read tokenizer graph source {path:?}"))?;
        let tokenizer = mary::tokenizer::save_tokenizer_json(&json, source, candidate.blobs_mut())
            .map_err(|error| anyhow::anyhow!("ingest tokenizer graph for {source}: {error}"))?;
        candidate += tokenizer;
    }

    // Stage every dependency before exposing authority. Validate through the
    // destination's own reader, then append the signed commit as the final
    // publication step.
    let candidate_facts = candidate.facts().clone();
    let team = mary::model_collection::model_graph_team_or_own(pile, signing_key)?;
    let prepared = mary::model_collection::prepare_model_fragment(team, candidate)
        .map_err(|error| anyhow::anyhow!("prepare embedding collection commit: {error}"))?;
    let mut staged = prepared
        .stage(pile, signing_key)
        .map_err(|error| anyhow::anyhow!("stage embedding commit dependencies: {error}"))?;

    let snapshot = mary::model_collection::snapshot_model_collection_local_latest(
        staged.store_mut(),
        team,
    )?;
    let (mut facts, _, reader) = snapshot.into_parts();
    facts += candidate_facts;
    let selected = SelectedModelIndex::from_graph(
        &facts,
        reader,
        ModelSelector::Source {
            source,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )?;
    anyhow::ensure!(
        selected.root() == model_root,
        "staged embedding root differs from the unique Source/native root"
    );
    validate_f32_model(&selected, contract)?;

    if tokenizer_json.is_some() {
        let tokenizer = mary::selection::load_tokenizer_from_graph(
            &facts,
            selected.reader(),
            TokenizerSelector::Name(source),
        )?;
        architecture.validate_tokenizer(&tokenizer)?;
    }
    let tensor_count = selected.handles().len();
    drop(selected);

    let commit = staged
        .finalize()
        .map_err(|error| anyhow::anyhow!("publish validated embedding commit: {error}"))?;
    Ok((model_root, commit, tensor_count))
}

/// Publish one architecture-exact embedding model and its optional tokenizer
/// as a single signed native model-collection commit.
pub(super) fn publish_embedding_candidate(
    pile: &mut Pile,
    signing_key: &SigningKey,
    weights: &Path,
    format: mary::formats::WeightFormat,
    tokenizer_json: Option<&Path>,
    source: &str,
    architecture: mary::embed::EmbeddingArchitecture,
) -> anyhow::Result<(Id, CollectionCommit, usize)> {
    let contract = architecture.tensor_shapes();
    publish_embedding_candidate_with_contract_impl(
        pile,
        signing_key,
        weights,
        format,
        tokenizer_json,
        source,
        architecture,
        &contract,
    )
}

/// Tiny synthetic bin tests need to exercise publication without constructing
/// hundreds of real architecture tensors. Production callers cannot supply a
/// contract independently of the architecture.
#[cfg(test)]
#[allow(dead_code)] // This support module is also test-compiled by the two parity bins.
pub(super) fn publish_embedding_candidate_with_contract(
    pile: &mut Pile,
    signing_key: &SigningKey,
    weights: &Path,
    format: mary::formats::WeightFormat,
    tokenizer_json: Option<&Path>,
    source: &str,
    architecture: mary::embed::EmbeddingArchitecture,
    contract: &BTreeMap<String, Vec<usize>>,
) -> anyhow::Result<(Id, CollectionCommit, usize)> {
    publish_embedding_candidate_with_contract_impl(
        pile,
        signing_key,
        weights,
        format,
        tokenizer_json,
        source,
        architecture,
        contract,
    )
}
