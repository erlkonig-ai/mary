//! Real-data proof of the zero-copy weight bridge (Phase 2): import a tiny
//! model to an on-disk native collection as f16, explicitly select its only
//! root, then load ONE weight by ALIASING its mmap'd f16 blob straight onto the
//! GPU — `Bytes::downcast_to_owner::<MmapRaw>` → page-aligned superset →
//! `register_external_aliased` → `CubeTensor` → burn `Tensor<BHalf>` — and
//! assert the tensor's values equal the pile's f16 bytes. Proves the recipe the
//! Gemma assembly will run per weight.
//!
//!   cargo run --release --features gemma,import --bin gemma_alias_probe
//! macOS / Metal only.

#[path = "support/native_model_fixture.rs"]
mod native_model_fixture;

use crate::native_model_fixture::import_native_model_fixture;
use burn::backend::wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
use burn::tensor::{DType, Tensor, TensorPrimitive};
use cubecl::Runtime;
use half::f16;
use mary::ingest::LeafHandles;
use mary::nn::backend::BHalf;
use mary::selection::{ModelSelector, SelectedModelIndex};
use memmap2::MmapRaw;
use std::process::Command;
use std::sync::Arc;
use triblespace::prelude::BlobStoreGet;

const PAGE: u64 = 16384;
const SOURCE: &str = "fixture/gemma-alias-probe";

fn main() {
    // 1. A tiny safetensors with one known tensor.
    let dir = std::env::temp_dir().join(format!("mary_alias_tiny_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let st = dir.join("model.safetensors");
    let py = format!(
        "import numpy as np, safetensors.numpy as st\n\
         w = (np.arange(8*16, dtype=np.float32)*0.25 - 13.0).reshape(8,16)\n\
         st.save_file({{'w': w}}, {st:?})\n",
        st = st.to_str().unwrap()
    );
    assert!(
        Command::new("python3")
            .arg("-c")
            .arg(&py)
            .status()
            .unwrap()
            .success()
    );

    // 2. Import as f16 leaves under one signed native collection root.
    let pile = dir.join("tiny.pile");
    let imported_root =
        import_native_model_fixture(&dir, &pile, mary::ingest::LeafDtype::F16, SOURCE)
            .expect("import tiny native model collection");

    // 3. Freeze the local native collection and select its only model root.
    let snapshot = mary::model_collection::load_model_collection_local_latest(&pile)
        .expect("load tiny native model collection snapshot");
    let selected = SelectedModelIndex::from_snapshot(snapshot, ModelSelector::Only)
        .expect("select the only tiny model root");
    assert_eq!(selected.root(), imported_root);
    let handles = selected.handles().get("w").expect("weight 'w' in pile");
    let (dh, _sh) = match handles {
        LeafHandles::F16(d, s) => (*d, *s),
        LeafHandles::F32(..) => panic!("expected f16 leaf"),
    };

    // 4. The blob's Bytes (a slice into the pile mmap).
    let bytes: anybytes::Bytes = selected.reader().get(dh).expect("data_f16 blob");
    let blob_ptr = bytes.as_ptr() as usize;
    let expected: Vec<f16> = bytes.clone().view::<[f16]>().expect("f16 view")[..].to_vec();
    let n = expected.len();
    let blob_size = (n * 2) as u64;

    // 5. Recover the WHOLE mmap region via the owner downcast (capability +
    //    bounds + keepalive, all in one).
    let mmap: Arc<MmapRaw> = bytes
        .downcast_to_owner::<MmapRaw>()
        .unwrap_or_else(|_| panic!("blob is not mmap-backed — would need the copy fallback"));
    let region_base = mmap.as_ptr() as usize;
    let region_end = region_base + mmap.len();

    // 6. Page-aligned superset of the blob, clamped to the real mapping.
    let page_start = blob_ptr & !((PAGE - 1) as usize);
    assert!(
        page_start >= region_base,
        "page rounding underflowed the mapping"
    );
    let off_in_page = (blob_ptr - page_start) as u64;
    let want_end = ((blob_ptr + n * 2) as u64 + PAGE - 1) & !(PAGE - 1);
    let page_len = want_end.min(region_end as u64) - page_start as u64;

    // 7. Alias it onto the GPU (keepalive = the mmap Arc → buffer owns the map).
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
    let keepalive: Arc<dyn std::any::Any + Send + Sync> = mmap.clone();
    let handle = unsafe {
        client.register_external_aliased(
            page_start as *mut core::ffi::c_void,
            page_len,
            off_in_page,
            blob_size,
            keepalive,
        )
    };

    // 8. Wrap as a burn Tensor<BHalf> over the aliased handle and read it back.
    let cube: CubeTensor<WgpuRuntime> =
        CubeTensor::new_contiguous(client, device, [n].into(), handle, DType::F16);
    let tensor: Tensor<BHalf, 1> = Tensor::from_primitive(TensorPrimitive::Float(cube));
    let got: Vec<f16> = tensor.into_data().to_vec::<f16>().expect("to f16 vec");

    let _ = std::fs::remove_dir_all(&dir);

    let ok = got.len() == n && got == expected;
    println!("aliased tensor[0..4] = {:?}", &got[..4.min(n)]);
    println!("pile f16    [0..4] = {:?}", &expected[..4.min(n)]);
    if ok {
        println!(
            "=== PASS — aliased burn Tensor equals the pile's f16 weight (real-data zero-copy) ==="
        );
    } else {
        println!("=== FAIL ===");
        std::process::exit(1);
    }
}
