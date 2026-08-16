//! Import a FLUX.2 model directory into Mary's native model collection.
//!
//! The text encoder, transformer, and VAE become three ordinary model roots
//! under stable component source coordinates. Runtime freezes one collection
//! snapshot, indexes all three roots, and still materializes only one component
//! per inference phase.
//!
//! ```text
//! cargo run --release --features flux,import --bin flux_persist -- \
//!   <model-dir> <pile-path> <signing-key>
//! ```

use mary::ingest::LeafDtype;
use mary::models::flux::pipeline::ModelVariant;
use std::path::Path;
use triblespace::core::repo::pile::Pile;
use triblespace::core::signing_key_file;

const COMPONENTS: &[&str] = &["text_encoder", "transformer", "vae"];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: flux_persist <model-dir> <pile-path> <signing-key>");
        std::process::exit(2);
    }
    let model_dir = Path::new(&args[1]);
    let pile_path = Path::new(&args[2]);
    let key_path = Path::new(&args[3]);
    let variant = ModelVariant::detect(model_dir)?;
    let signing_key = signing_key_file::load_existing(key_path)?;

    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(pile_path)
    {
        Ok(_) => eprintln!("flux_persist: created new empty pile {pile_path:?}"),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }

    let mut pile = Pile::open(pile_path)
        .map_err(|error| anyhow::anyhow!("open model pile {pile_path:?}: {error}"))?;
    let imported = (|| -> anyhow::Result<()> {
        for component in COMPONENTS {
            let directory = model_dir.join(component);
            anyhow::ensure!(directory.is_dir(), "missing FLUX component {directory:?}");
            let source = variant.component_source(component);
            eprintln!("flux_persist: importing {source} from {directory:?}");
            let (root, commit) = mary::persist::import_model_to_collection(
                &mut pile,
                &signing_key,
                &directory,
                LeafDtype::F16,
                &source,
                mary::persist::QUANTIZATION_NATIVE,
            )?;
            eprintln!(
                "flux_persist: {component} root {root:X}, native commit {}",
                commit.id()
            );
        }
        Ok(())
    })();
    let close = pile
        .close()
        .map_err(|error| anyhow::anyhow!("close model pile {pile_path:?}: {error}"));
    match (imported, close) {
        (Ok(()), Ok(())) => {}
        (Err(error), Ok(())) => return Err(error),
        (Ok(()), Err(error)) => return Err(error),
        (Err(error), Err(close_error)) => {
            return Err(error.context(format!("import also failed to close pile: {close_error}")))
        }
    }

    let size = std::fs::metadata(pile_path)?.len();
    println!(
        "Pile file {pile_path:?} is {size} bytes ({:.2} GiB).",
        size as f64 / (1u64 << 30) as f64
    );
    Ok(())
}
