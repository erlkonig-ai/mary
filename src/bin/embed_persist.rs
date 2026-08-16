//! Publish the live embedding models from the local Hugging Face cache as
//! signed native Mary collection commits.
//!
//! Each model keeps its own pile because CLIP and Nomic reuse tensor names.
//! The text-capable piles carry their tokenizer graph in the same signed commit
//! as the weights, so ordinary inference needs neither a Repository branch nor
//! a `tokenizer.json` side file.
//!
//! ```text
//! cargo run --release --features embed,import --bin embed_persist -- \
//!   <out-dir> <signing-key>
//! ```

use anyhow::Context;
use ed25519_dalek::SigningKey;
use mary::ingest::{LeafDtype, LeafHandles};
use mary::selection::{ModelSelector, SelectedModelIndex, TokenizerSelector};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;
use triblespace::core::collection::CollectionCommit;
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;
use triblespace::prelude::*;

const CLIP_MODEL: &str = "openai/clip-vit-base-patch32";
const NOMIC_TEXT_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";
const NOMIC_VISION_MODEL: &str = "nomic-ai/nomic-embed-vision-v1.5";

#[derive(Clone, Copy)]
struct ModelSpec {
    source: &'static str,
    stem: &'static str,
    weights: &'static str,
    format: mary::formats::WeightFormat,
    architecture: mary::embed::EmbeddingArchitecture,
    tokenizer: bool,
}

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        source: CLIP_MODEL,
        stem: "clip",
        weights: "pytorch_model.bin",
        format: mary::formats::WeightFormat::Pickle,
        architecture: mary::embed::EmbeddingArchitecture::ClipVitBasePatch32,
        tokenizer: true,
    },
    ModelSpec {
        source: NOMIC_TEXT_MODEL,
        stem: "nomic_text",
        weights: "model.safetensors",
        format: mary::formats::WeightFormat::Safetensors,
        architecture: mary::embed::EmbeddingArchitecture::NomicTextV15,
        tokenizer: true,
    },
    ModelSpec {
        source: NOMIC_VISION_MODEL,
        stem: "nomic_vision",
        weights: "model.safetensors",
        format: mary::formats::WeightFormat::Safetensors,
        architecture: mary::embed::EmbeddingArchitecture::NomicVisionV15,
        tokenizer: false,
    },
];

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

fn publish_embedding_candidate(
    pile: &mut Pile,
    signing_key: &SigningKey,
    weights: &Path,
    format: mary::formats::WeightFormat,
    tokenizer_json: Option<&Path>,
    source: &str,
    architecture: mary::embed::EmbeddingArchitecture,
    contract: &BTreeMap<String, Vec<usize>>,
) -> anyhow::Result<(Id, CollectionCommit, usize)> {
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

    // The prepared commit owns every in-memory tokenizer attachment. Stage all
    // dependencies first, then validate the candidate together with the exact
    // pre-existing authority set through the destination pile's own reader.
    // Only `finalize` below appends the signed membership record.
    let candidate_facts = candidate.facts().clone();
    let prepared = mary::model_collection::prepare_model_fragment(signing_key, candidate)
        .map_err(|error| anyhow::anyhow!("prepare embedding collection commit: {error}"))?;
    let mut staged = prepared
        .stage(pile)
        .map_err(|error| anyhow::anyhow!("stage embedding commit dependencies: {error}"))?;

    let snapshot =
        mary::model_collection::snapshot_model_collection_local_latest(staged.store_mut())?;
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

fn create_pile_if_missing(path: &Path) -> anyhow::Result<()> {
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(_) => eprintln!("embed_persist: created new empty pile {path:?}"),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn import_one(out_dir: &Path, signing_key: &SigningKey, spec: ModelSpec) -> anyhow::Result<()> {
    let snapshot = mary::embed::hf_cache_main_snapshot(spec.source)
        .with_context(|| format!("resolve one cached main revision for {}", spec.source))?;
    let weights = snapshot.join(spec.weights);
    anyhow::ensure!(
        weights.is_file(),
        "{} has no {} in cached main revision {}",
        spec.source,
        spec.weights,
        snapshot.display()
    );
    let tokenizer = spec
        .tokenizer
        .then(|| {
            let path = snapshot.join("tokenizer.json");
            anyhow::ensure!(
                path.is_file(),
                "{} has no tokenizer.json in cached main revision {}",
                spec.source,
                snapshot.display()
            );
            Ok::<_, anyhow::Error>(path)
        })
        .transpose()?;
    let pile_path = out_dir.join(format!("{}.pile", spec.stem));
    create_pile_if_missing(&pile_path)?;

    let started = Instant::now();
    let contract = spec.architecture.tensor_shapes();
    let mut pile = Pile::open(&pile_path)
        .map_err(|error| anyhow::anyhow!("open model pile {pile_path:?}: {error}"))?;
    let imported = publish_embedding_candidate(
        &mut pile,
        signing_key,
        &weights,
        spec.format,
        tokenizer.as_deref(),
        spec.source,
        spec.architecture,
        &contract,
    );
    let close = pile
        .close()
        .map_err(|error| anyhow::anyhow!("close model pile {pile_path:?}: {error}"));
    let (root, commit, tensors) = match (imported, close) {
        (Ok(result), Ok(())) => result,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(close_error)) => {
            return Err(error.context(format!("import also failed to close pile: {close_error}")));
        }
    };

    let size = std::fs::metadata(&pile_path)?.len();
    println!(
        "{}: root {root}, commit {}, {tensors} f32 tensors, {:.2} GiB in {:.1}s",
        spec.stem,
        commit.id(),
        size as f64 / (1_u64 << 30) as f64,
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: embed_persist <out-dir> <signing-key>");
        std::process::exit(2);
    }
    let out_dir = Path::new(&args[1]);
    let signing_key = signing_key_file::load_existing(Path::new(&args[2]))?;
    std::fs::create_dir_all(out_dir)?;
    for &spec in MODELS {
        import_one(out_dir, &signing_key, spec)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: std::path::PathBuf,
    }

    impl TempDir {
        fn new() -> std::io::Result<Self> {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "mary-embed-native-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path)?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    const WORDPIECE: &str = r###"{
      "added_tokens": [],
      "normalizer": {"type": "BertNormalizer", "clean_text": true,
                     "handle_chinese_chars": true, "strip_accents": null,
                     "lowercase": true},
      "pre_tokenizer": {"type": "BertPreTokenizer"},
      "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3}}
    }"###;

    fn write_tensor(path: &Path, name: &str, shape: Vec<usize>, values: &[f32]) {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        serialize_to_file(
            [(name, TensorView::new(Dtype::F32, shape, &bytes).unwrap())],
            &None,
            path,
        )
        .unwrap();
    }

    fn write_weights(path: &Path, values: &[f32]) {
        write_tensor(path, "encoder.weight", vec![2, 2], values);
    }

    fn toy_contract() -> BTreeMap<String, Vec<usize>> {
        BTreeMap::from([("encoder.weight".to_owned(), vec![2, 2])])
    }

    #[test]
    fn text_weights_and_tokenizer_publish_atomically_and_replay_without_growth() {
        let dir = TempDir::new().unwrap();
        let weights = dir.path().join("model.safetensors");
        let tokenizer = dir.path().join("tokenizer.json");
        let pile_path = dir.path().join("nomic_text.pile");
        write_weights(&weights, &[1.0, 2.0, 3.0, 4.0]);
        std::fs::write(&tokenizer, WORDPIECE).unwrap();
        std::fs::File::create(&pile_path).unwrap();
        let key = SigningKey::from_bytes(&[0x42; 32]);
        let mut pile = Pile::open(&pile_path).unwrap();
        let contract = toy_contract();

        let (_, first, count) = publish_embedding_candidate(
            &mut pile,
            &key,
            &weights,
            mary::formats::WeightFormat::Safetensors,
            Some(&tokenizer),
            NOMIC_TEXT_MODEL,
            mary::embed::EmbeddingArchitecture::NomicTextV15,
            &contract,
        )
        .unwrap();
        assert_eq!(count, 1);
        let first_len = std::fs::metadata(&pile_path).unwrap().len();
        let (_, repeated, _) = publish_embedding_candidate(
            &mut pile,
            &key,
            &weights,
            mary::formats::WeightFormat::Safetensors,
            Some(&tokenizer),
            NOMIC_TEXT_MODEL,
            mary::embed::EmbeddingArchitecture::NomicTextV15,
            &contract,
        )
        .unwrap();
        assert_eq!(repeated, first);
        assert_eq!(std::fs::metadata(&pile_path).unwrap().len(), first_len);

        write_weights(&weights, &[9.0, 8.0, 7.0, 6.0]);
        let conflict = publish_embedding_candidate(
            &mut pile,
            &key,
            &weights,
            mary::formats::WeightFormat::Safetensors,
            Some(&tokenizer),
            NOMIC_TEXT_MODEL,
            mary::embed::EmbeddingArchitecture::NomicTextV15,
            &contract,
        )
        .unwrap_err();
        let diagnostic = format!("{conflict:#}");
        assert!(
            diagnostic.contains("ambiguous model root"),
            "unexpected conflict diagnostic: {diagnostic}"
        );

        let snapshot =
            mary::model_collection::snapshot_model_collection_local_latest(&mut pile).unwrap();
        assert_eq!(snapshot.commits(), &[first]);
        mary::selection::load_keymap_from_graph(
            snapshot.facts(),
            snapshot.reader(),
            ModelSelector::Source {
                source: NOMIC_TEXT_MODEL,
                quantization: mary::persist::QUANTIZATION_NATIVE,
            },
        )
        .unwrap();
        let tok = mary::selection::load_tokenizer_from_graph(
            snapshot.facts(),
            snapshot.reader(),
            TokenizerSelector::Name(NOMIC_TEXT_MODEL),
        )
        .unwrap();
        assert_eq!(tok.token_to_id("hello"), Some(3));
        pile.close().unwrap();
    }

    #[test]
    fn failed_preflight_never_publishes_a_collection_commit() {
        let dir = TempDir::new().unwrap();
        let weights = dir.path().join("model.safetensors");
        let tokenizer = dir.path().join("tokenizer.json");
        let pile_path = dir.path().join("bad.pile");
        write_weights(&weights, &[1.0, 2.0, 3.0, 4.0]);
        std::fs::write(&tokenizer, b"not json").unwrap();
        std::fs::File::create(&pile_path).unwrap();
        let key = SigningKey::from_bytes(&[0x43; 32]);
        let mut pile = Pile::open(&pile_path).unwrap();
        let contract = toy_contract();

        publish_embedding_candidate(
            &mut pile,
            &key,
            &weights,
            mary::formats::WeightFormat::Safetensors,
            Some(&tokenizer),
            NOMIC_TEXT_MODEL,
            mary::embed::EmbeddingArchitecture::NomicTextV15,
            &contract,
        )
        .unwrap_err();
        let snapshot =
            mary::model_collection::snapshot_model_collection_local_latest(&mut pile).unwrap();
        assert!(snapshot.commits().is_empty());
        pile.close().unwrap();
    }

    #[test]
    fn incomplete_or_wrong_shaped_architecture_never_publishes() {
        let dir = TempDir::new().unwrap();
        let weights = dir.path().join("model.safetensors");
        let pile_path = dir.path().join("bad-architecture.pile");
        write_weights(&weights, &[1.0, 2.0, 3.0, 4.0]);
        std::fs::File::create(&pile_path).unwrap();
        let key = SigningKey::from_bytes(&[0x44; 32]);
        let mut pile = Pile::open(&pile_path).unwrap();

        let incomplete = BTreeMap::from([
            ("encoder.weight".to_owned(), vec![2, 2]),
            ("encoder.bias".to_owned(), vec![2]),
        ]);
        let error = publish_embedding_candidate(
            &mut pile,
            &key,
            &weights,
            mary::formats::WeightFormat::Safetensors,
            None,
            NOMIC_VISION_MODEL,
            mary::embed::EmbeddingArchitecture::NomicVisionV15,
            &incomplete,
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("missing required tensor \"encoder.bias\""));

        write_tensor(&weights, "encoder.weight", vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let error = publish_embedding_candidate(
            &mut pile,
            &key,
            &weights,
            mary::formats::WeightFormat::Safetensors,
            None,
            NOMIC_VISION_MODEL,
            mary::embed::EmbeddingArchitecture::NomicVisionV15,
            &toy_contract(),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("has shape [4], expected [2, 2]"));

        let snapshot =
            mary::model_collection::snapshot_model_collection_local_latest(&mut pile).unwrap();
        assert!(snapshot.commits().is_empty());
        pile.close().unwrap();
    }
}
