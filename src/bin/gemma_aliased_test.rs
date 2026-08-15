//! The payoff: explicitly select Gemma 4 from a native model collection and
//! load it with ZERO-COPY weights (each tensor's mmap'd f16 blob aliased
//! straight onto the Metal GPU — no copy, no f32 materialization), then prove
//! it generates token-for-token identically to the selected streaming load.
//! Both run f16 (`BHalf`).
//!
//!   cargo run --release --features gemma,import --bin gemma_aliased_test
//! macOS / Metal only.

#[cfg(target_os = "macos")]
#[path = "support/native_model_fixture.rs"]
mod native_model_fixture;

#[cfg(target_os = "macos")]
mod imp {

    use crate::native_model_fixture::import_native_model_fixture;
    use mary::models::gemma::gemma4::config::Gemma4Config;
    use mary::models::gemma::gemma4::lm::GemmaLM;
    use mary::nn::backend::BHalf;
    use mary::selection::{ModelSelector, SelectedModelIndex};
    use std::path::Path;
    use std::process::Command;

    fn hf(model_id: &str, file: &str) -> String {
        let o = Command::new("python3")
        .args(["-c", &format!(
            "from huggingface_hub import hf_hub_download; print(hf_hub_download('{model_id}', '{file}'))"
        )])
        .output()
        .unwrap();
        String::from_utf8(o.stdout).unwrap().trim().to_string()
    }

    pub fn run() {
        let model_id = "google/gemma-4-E2B-it";
        let config_path = hf(model_id, "config.json");
        let _ = hf(model_id, "model.safetensors");
        let tok = hf(model_id, "tokenizer.json");
        let snapshot = Path::new(&config_path).parent().unwrap();
        let device = mary::models::gemma::metal_device::init_metal_device_16gb();
        let config = Gemma4Config::load(Path::new(&config_path));

        let pile =
            std::env::temp_dir().join(format!("mary_gemma_aliased_{}.pile", std::process::id()));
        let _ = std::fs::remove_file(&pile);
        println!("[persist] {snapshot:?} -> {pile:?} (native collection, f16)");
        let imported_root =
            import_native_model_fixture(snapshot, &pile, mary::ingest::LeafDtype::F16, model_id)
                .expect("import native model collection");

        let chat = "<bos><|turn>user\nWhat is 17 times 23? Answer with just the number.<turn|>\n<|turn>model\n";

        println!("[stream f16] selecting the native root and loading...");
        let stream_snapshot = mary::model_collection::load_model_collection_local_latest(&pile)
            .expect("load native model collection snapshot for streaming");
        let stream_selected = SelectedModelIndex::from_snapshot(
            stream_snapshot,
            ModelSelector::Source {
                source: model_id,
                quantization: mary::persist::QUANTIZATION_NATIVE,
            },
        )
        .expect("select imported model for streaming");
        assert_eq!(stream_selected.root(), imported_root);
        let (streamed_model, _vision) = mary::persist::load_gemma4_streaming_from_index::<BHalf, _>(
            stream_selected,
            config.clone(),
            &device,
        );
        let streamed = GemmaLM::<BHalf>::from_model(
            config.clone(),
            streamed_model,
            Path::new(&tok),
            device.clone(),
        );
        let ids_stream = streamed.complete_ids(chat, 10);
        drop(streamed);

        println!("[aliased] selecting and loading ZERO-COPY from the pile mmap...");
        let alias_snapshot = mary::model_collection::load_model_collection_local_latest(&pile)
            .expect("load native model collection snapshot for aliasing");
        let alias_selected = SelectedModelIndex::from_snapshot(
            alias_snapshot,
            ModelSelector::Source {
                source: model_id,
                quantization: mary::persist::QUANTIZATION_NATIVE,
            },
        )
        .expect("select imported model for aliasing");
        assert_eq!(alias_selected.root(), imported_root);
        let aliased_model = mary::persist::load_gemma4_aliased_from_index(
            alias_selected,
            config.clone(),
            device.clone(),
        )
        .expect("load selected model with aliased f16 weights");
        let aliased = GemmaLM::<BHalf>::from_model(config, aliased_model, Path::new(&tok), device);
        let ids_alias = aliased.complete_ids(chat, 10);

        let _ = std::fs::remove_file(&pile);

        println!(
            "streamed ids {ids_stream:?} -> {:?}",
            aliased.decode(&ids_stream)
        );
        println!(
            "aliased  ids {ids_alias:?} -> {:?}",
            aliased.decode(&ids_alias)
        );
        if ids_stream == ids_alias {
            println!("=== PASS — zero-copy aliased load is token-identical to streamed (f16) ===");
        } else {
            println!("=== FAIL — aliased diverged from streamed ===");
            std::process::exit(1);
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    imp::run()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemma_aliased_test: macOS/Metal-only lane (zero-copy GPU aliasing).");
    std::process::exit(2);
}
