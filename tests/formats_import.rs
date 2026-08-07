//! Non-safetensors import validation: import a pytorch `.bin` (and, if cached, a
//! GGUF) model through `mary import`'s persist path and assert the tensors load
//! back with correct shapes and values. Gated on the model being present in the
//! local HF cache (these are `#[ignore]` by default — run with
//! `cargo test --features import --test formats_import -- --ignored`), since CI
//! has no network.

#![cfg(feature = "import")]

use std::path::PathBuf;

/// Locate a cached HF snapshot dir for `id`, or `None` if not downloaded.
fn hf_snapshot(id: &str) -> Option<PathBuf> {
    let hf_home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
            PathBuf::from(home).join(".cache/huggingface")
        });
    let repo = format!("models--{}", id.replace('/', "--"));
    let snaps = hf_home.join("hub").join(repo).join("snapshots");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .find(|d| mary::formats::detect_format(d).is_ok())
}

#[test]
#[ignore = "needs hf-internal-testing/tiny-random-MistralForCausalLM in the HF cache"]
fn pytorch_bin_import_roundtrip() {
    let dir = match hf_snapshot("hf-internal-testing/tiny-random-MistralForCausalLM") {
        Some(d) => d,
        None => {
            eprintln!("skip: tiny-random-MistralForCausalLM not cached");
            return;
        }
    };
    let (fmt, files) = mary::formats::detect_format(&dir).unwrap();
    assert_eq!(
        fmt,
        mary::formats::WeightFormat::Pickle,
        "should detect pickle"
    );
    assert_eq!(files.len(), 1);

    let tmp = std::env::temp_dir().join(format!("mary_pickle_test_{}.pile", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let root = mary::persist::persist_model_to_pile(
        &dir,
        &tmp,
        mary::ingest::LeafDtype::F32,
        "mistral-tiny",
        "native",
    )
    .unwrap();
    eprintln!("imported root {root:X}");

    let km = mary::persist::load_keymap_from_mary_branch_quantized(&tmp, "mistral-tiny", "native")
        .unwrap();
    // The tiny Mistral has these tensors with these exact shapes.
    let (embed, eshape) = &km["model.embed_tokens.weight"];
    assert_eq!(eshape, &[32000, 32], "embed shape");
    assert_eq!(embed.len(), 32000 * 32);
    let (ln, lshape) = &km["model.layers.0.input_layernorm.weight"];
    assert_eq!(lshape, &[32]);
    // input_layernorm initializes to all-ones in this fixture.
    for &v in ln.iter() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "layernorm weight should be 1.0, got {v}"
        );
    }
    // Finite, non-degenerate weights everywhere.
    assert!(embed.iter().all(|v| v.is_finite()));
    assert!(embed.iter().any(|&v| v != 0.0));

    let _ = std::fs::remove_file(&tmp);
}
