//! Shared root identity and ancestry across import and derivation paths.

#![cfg(feature = "import")]

use mary::format::attrs;
use mary::ingest::{LeafDtype, build_model_root, ingest_tensors, save_safetensors};
use mary::selection::{ModelSelector, SelectedModelIndex};
use safetensors::tensor::{Dtype, TensorView, serialize};
use triblespace::prelude::*;

#[test]
fn per_file_and_combined_imports_share_the_member_only_identity_core() {
    let values = [1.0f32, -2.0, 3.0, 4.0];
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let container = serialize(
        [(
            "linear.weight",
            TensorView::new(Dtype::F32, vec![2, 2], &bytes).unwrap(),
        )],
        &None,
    )
    .unwrap();
    let mut blobs = MemoryBlobStore::new();
    let first = save_safetensors(
        &container,
        "one/model.safetensors",
        &mut blobs,
        LeafDtype::F32,
    )
    .unwrap();
    let renamed = save_safetensors(&container, "another-name", &mut blobs, LeafDtype::F32).unwrap();
    let (members, facts) = ingest_tensors(
        std::iter::once(("linear.weight".into(), values.to_vec(), vec![2, 2])),
        &mut blobs,
        LeafDtype::F32,
    )
    .unwrap();
    let combined = build_model_root(
        &mut blobs,
        "fixture/shared",
        "native",
        members,
        facts,
        &["different-file".into()],
    )
    .unwrap();
    assert_eq!(first.root(), renamed.root());
    assert_eq!(first.root(), combined.root());
    assert_ne!(
        first.facts(),
        renamed.facts(),
        "the name is still retained as an annotation"
    );
}

#[test]
fn f16_derivation_records_the_selected_opaque_base() {
    let dir = std::env::temp_dir().join(format!("mary-root-lineage-{}", fucid().id));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("models.pile");
    std::fs::File::create(&path).unwrap();
    let mut pile = Pile::open(&path).unwrap();
    pile.refresh().unwrap();
    let key = ed25519_dalek::SigningKey::from_bytes(&[0x52; 32]);
    let (members, facts) = ingest_tensors(
        std::iter::once((
            "linear.weight".into(),
            vec![1.0, -2.0, 3.0, 4.0],
            vec![2, 2],
        )),
        &mut pile,
        LeafDtype::F32,
    )
    .unwrap();

    // No hash-derived root: an explicitly minted ID is equally addressable.
    let base = fucid();
    let mut fragment = entity! { &base @ attrs::member*: members.iter() };
    *fragment.facts_mut() += facts;
    mary::model_collection::publish_model_fragment(&mut pile, &key, fragment).unwrap();
    let snapshot =
        mary::model_collection::snapshot_model_collection_local_latest(&mut pile).unwrap();
    let selected =
        SelectedModelIndex::from_snapshot(snapshot, ModelSelector::Root(base.id)).unwrap();
    let (derived, _, count, elements) = mary::persist::derive_selected_f16_to_collection(
        &mut pile,
        &key,
        selected,
        "not-the-base-name",
        "f16",
    )
    .unwrap();
    assert_eq!((count, elements), (1, 4));
    assert_ne!(derived, base.id);
    pile.close().unwrap();

    let snapshot = mary::model_collection::load_model_collection_local_latest(&path).unwrap();
    assert!(exists!(
        (),
        pattern!(snapshot.facts(), [{
            derived @ attrs::parent: &base,
        }])
    ));
    let loaded = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Root(derived),
    )
    .unwrap();
    assert_eq!(
        loaded["linear.weight"],
        (vec![1.0, -2.0, 3.0, 4.0], vec![2, 2])
    );
    assert_eq!(
        mary::selection::select_model_root(
            snapshot.facts(),
            snapshot.store(),
            ModelSelector::Root(base.id),
        )
        .unwrap(),
        base.id,
        "adding a descendant does not replace or re-identify its base",
    );
    drop(snapshot);
    std::fs::remove_dir_all(dir).unwrap();
}
