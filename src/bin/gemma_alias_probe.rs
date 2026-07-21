//! Real-data proof of the zero-copy weight bridge (Phase 2): persist a tiny
//! model to an on-disk pile as f16, then load ONE weight by ALIASING its mmap'd
//! f16 blob straight onto the GPU — `Bytes::downcast_to_owner::<MmapRaw>` →
//! page-aligned superset → `register_external_aliased` → `CubeTensor` → burn
//! `Tensor<BHalf>` — and assert the tensor's values equal the pile's f16 bytes.
//! Proves the recipe the gemma assembly will run per weight.
//!
//!   cargo run --release --features gemma --bin gemma_alias_probe
//! macOS / Metal only.

use burn::backend::wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
use burn::tensor::{DType, Tensor, TensorPrimitive};
use cubecl::Runtime;
use half::f16;
use mary::ingest::{index_keymap, LeafHandles};
use mary::nn::backend::BHalf;
use memmap2::MmapRaw;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use triblespace::prelude::*;

const PAGE: u64 = 16384;

fn main() {
    // 1. A tiny safetensors with one known tensor.
    let dir = std::env::temp_dir().join(format!("mary_alias_tiny_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let st = dir.join("model.safetensors");
    let py = format!(
        "import numpy as np, safetensors.numpy as st\n\
         w = (np.arange(8*16, dtype=np.float32)*0.25 - 13.0).reshape(8,16)\n\
         st.save_file({{'w': w}}, {st:?})\n",
        st = st.to_str().unwrap()
    );
    assert!(Command::new("python3").arg("-c").arg(&py).status().unwrap().success());

    // 2. Persist as f16 leaves.
    let pile = dir.join("tiny.pile");
    mary::persist::persist_safetensors_to_pile(&dir, &pile, mary::ingest::LeafDtype::F16).unwrap();

    // 3. Open the pile read-only: tribles + reader + model id.
    let (tribles, reader, model_id) = open_pile(&pile);
    let index = index_keymap(&tribles, &reader, model_id);
    let handles = index.get("w").expect("weight 'w' in pile");
    let (dh, _sh) = match handles {
        LeafHandles::F16(d, s) => (*d, *s),
        LeafHandles::F32(..) => panic!("expected f16 leaf"),
    };

    // 4. The blob's Bytes (a slice into the pile mmap).
    let bytes: anybytes::Bytes = reader.get(dh).expect("data_f16 blob");
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
    assert!(page_start >= region_base, "page rounding underflowed the mapping");
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
        println!("=== PASS — aliased burn Tensor equals the pile's f16 weight (real-data zero-copy) ===");
    } else {
        println!("=== FAIL ===");
        std::process::exit(1);
    }
}

fn open_pile(pile_path: &Path) -> (TribleSet, impl BlobStoreGet, Id) {
    use ed25519_dalek::SigningKey;
    let mut pile = Pile::open(pile_path).unwrap();
    pile.refresh().unwrap();
    let mut repo = Repository::new(pile, SigningKey::generate(&mut rand::rngs::OsRng), TribleSet::new()).unwrap();
    let branch_id = repo.lookup_branch("main").unwrap().unwrap();
    let mut ws = repo.pull(branch_id).unwrap();
    let head = ws.head().unwrap();
    let tribles: TribleSet = ws.checkout(ancestors(head)).unwrap().facts().clone();
    let reader = repo.storage_mut().reader().unwrap();
    let model_id = find!(
        (m: Id, n: Inline<inlineencodings::Handle<blobencodings::LongString>>),
        pattern!(&tribles, [{ ?m @ mary::format::attrs::model_name: ?n }])
    )
    .map(|(m, _)| m)
    .next()
    .expect("model entity");
    (tribles, reader, model_id)
}
